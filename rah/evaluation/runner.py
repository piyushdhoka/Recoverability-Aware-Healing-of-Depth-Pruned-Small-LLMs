"""Evaluate a model on the capability suite. Returns per-item records plus a summary.

Generation capabilities (fmt, tool, inst, math, safe) use greedy batched decoding; knowledge (MMLU)
uses next-token probabilities over the answer letters, which also gives calibration (ECE).
"""
import time

import numpy as np
import torch

from ..modeling import chat_prompt, generate
from .scorers import SCORERS, expected_calibration_error

LETTERS = "ABCD"


def letter_token_ids(tok) -> list[list[int]]:
    """Token ids that spell each answer letter, with and without a leading space."""
    ids = []
    for L in LETTERS:
        cands = {tok.encode(L, add_special_tokens=False)[-1], tok.encode(" " + L, add_special_tokens=False)[-1]}
        ids.append(sorted(cands))
    return ids


@torch.no_grad()
def score_know(model, tok, items: list[dict], batch_size: int) -> list[dict]:
    letter_ids = letter_token_ids(tok)
    prompts = [chat_prompt(tok, it["messages"]) for it in items]
    recs = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc, use_cache=False).logits[:, -1].float()   # left padding: last position is real
        probs_all = torch.softmax(logits, dim=-1)
        for j, it in enumerate(items[s:s + batch_size]):
            p = torch.stack([probs_all[j, ids].sum() for ids in letter_ids])
            p = (p / p.sum()).cpu().numpy()
            pred = int(p.argmax())
            recs.append({"id": it["id"], "cap": "know", "pred": pred, "conf": float(p.max()),
                         "score": float(pred == it["meta"]["answer"])})
    return recs


def evaluate(model, tok, items_by_cap: dict, cfg: dict) -> dict:
    """items_by_cap: {cap: [item, ...]} -> {"records": [...], "summary": {...}, "seconds": float}."""
    t0 = time.time()
    was_training = model.training
    model.eval()
    ecfg = cfg["eval"]
    records = []
    for cap, items in items_by_cap.items():
        if not items:
            continue
        if cap == "know":
            records += score_know(model, tok, items, ecfg["batch_size"])
            continue
        prompts = [chat_prompt(tok, it["messages"]) for it in items]
        outs = generate(model, tok, prompts, ecfg["max_new_tokens"][cap], ecfg["batch_size"])
        for it, resp in zip(items, outs):
            rec = {"id": it["id"], "cap": cap, "response": resp}
            rec.update(SCORERS[cap](resp, it["meta"]))
            records.append(rec)
    if was_training:
        model.train()
    return {"records": records, "summary": summarize(records, ecfg.get("ece_bins", 10)),
            "seconds": time.time() - t0}


def summarize(records: list[dict], ece_bins: int = 10) -> dict:
    out = {}
    caps = sorted({r["cap"] for r in records})
    for cap in caps:
        rs = [r for r in records if r["cap"] == cap]
        out[cap] = float(np.mean([r["score"] for r in rs]))
        if cap == "safe":
            unsafe = [r for r in rs if r["unsafe"]]
            benign = [r for r in rs if not r["unsafe"]]
            out["safe_harmful_refusal"] = float(np.mean([r["refused"] for r in unsafe])) if unsafe else float("nan")
            out["safe_over_refusal"] = float(np.mean([r["refused"] for r in benign])) if benign else float("nan")
        if cap == "know":
            out["know_ece"] = expected_calibration_error([r["conf"] for r in rs], [r["score"] for r in rs], ece_bins)
        if cap in ("fmt", "tool"):
            out[f"{cap}_parse_rate"] = float(np.mean([r.get("parsed", False) for r in rs]))
    return out


def retention(summary: dict, base_summary: dict, caps: list[str]) -> dict:
    """r_c = score_c / score_c(M0); guarded for a near-zero baseline."""
    return {c: (summary[c] / base_summary[c]) if base_summary.get(c, 0) > 1e-9 else float("nan") for c in caps}
