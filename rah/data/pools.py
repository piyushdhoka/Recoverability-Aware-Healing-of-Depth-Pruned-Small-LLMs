"""Healing-pool prompts (one pool per capability + a general pool) and teacher-output quality checks.

Pool item: {"id", "pool", "messages", "check"}; targets are generated later by the unpruned teacher.
Sources are disjoint from the evaluation sets: glaive function schemas, Alpaca-cleaned, the GSM8K train split
and PKU-SafeRLHF, versus JSONSchemaBench, BFCL, IFEval, the GSM8K test split, XSTest and MMLU for evaluation.
"""
import json
import random
import re

from . import prompts as P
from ..evaluation.scorers import extract_json, extract_number, parse_call

# IFEval-registry constraints used to build verifiable instruction-following prompts.
# (instruction id, kwargs, natural-language text) — checked with the same lm_eval checkers as IFEval.
INST_CONSTRAINTS = [
    ("change_case:english_lowercase", {}, "Your entire response must be in lowercase letters; no capital letters are allowed."),
    ("change_case:english_capital", {}, "Your entire response must be in English and in all capital letters."),
    ("punctuation:no_comma", {}, "Do not use any commas in your response."),
    ("detectable_format:title", {}, "Include a title wrapped in double angular brackets, such as <<my title>>."),
    ("detectable_format:json_format", {}, "Wrap your entire output in JSON format."),
    ("startend:quotation", {}, "Wrap your entire response with double quotation marks."),
    ("detectable_format:number_bullet_lists", {"num_bullets": 3},
     "Your answer must contain exactly 3 bullet points in markdown format, like:\n* point one\n* point two"),
    ("startend:end_checker", {"end_phrase": "Is there anything else I can help with?"},
     "Finish your response with this exact phrase: Is there anything else I can help with?"),
    ("length_constraints:number_words", {"relation": "less than", "num_words": 60},
     "Answer with less than 60 words."),
    ("detectable_content:postscript", {"postscript_marker": "P.S."},
     "At the end of your response, please explicitly add a postscript starting with P.S."),
]


def _glaive_rows(n_needed: int, seed: int):
    """(functions, first user question) pairs from glaive-function-calling-v2."""
    from datasets import load_dataset

    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    dec = json.JSONDecoder()
    out = []
    for i in idx:
        r = ds[i]
        sysmsg, chat = r.get("system", ""), r.get("chat", "")
        if "<functioncall>" not in chat:
            continue
        start = sysmsg.find("{")
        fns, pos = [], start
        while 0 <= pos < len(sysmsg):
            try:
                obj, end = dec.raw_decode(sysmsg[pos:])
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict) and "name" in obj:
                fns.append(obj)
            nxt = sysmsg.find("{", pos + end)
            pos = nxt
        m = re.search(r"USER:\s*(.*?)\s*(?:ASSISTANT:|$)", chat, flags=re.S)
        if fns and m and m.group(1).strip():
            out.append((i, fns, m.group(1).strip()))
        if len(out) >= n_needed:
            break
    return out


def _alpaca(seed: int):
    from datasets import load_dataset

    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = [(i, (r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")).strip()) for i, r in enumerate(ds)]
    random.Random(seed).shuffle(rows)
    return rows


def build_pools(n: int, seed: int) -> dict:
    from datasets import load_dataset

    pools = {}
    glaive = _glaive_rows(2 * n, seed)
    pools["tool"] = [{"id": f"tool-{i}", "pool": "tool", "messages": P.tool_messages(fns, q), "check": {"functions": fns}}
                     for i, fns, q in glaive[:n]]
    fmt = []
    for i, fns, _ in glaive[n:]:
        schema = fns[0].get("parameters")
        if isinstance(schema, dict) and schema.get("properties"):
            fmt.append({"id": f"fmt-{i}", "pool": "fmt", "messages": P.fmt_messages(schema), "check": {"schema": schema}})
    pools["fmt"] = fmt[:n]

    alpaca = _alpaca(seed)
    rng = random.Random(seed + 1)
    inst = []
    for i, text in alpaca[:n]:
        cid, kw, desc = INST_CONSTRAINTS[rng.randrange(len(INST_CONSTRAINTS))]
        inst.append({"id": f"inst-{i}", "pool": "inst", "messages": P.plain_messages(f"{text}\n\n{desc}"),
                     "check": {"instruction_id_list": [cid], "kwargs": [kw], "prompt": f"{text}\n\n{desc}"}})
    pools["inst"] = inst
    pools["general"] = [{"id": f"gen-{i}", "pool": "general", "messages": P.plain_messages(t), "check": {}}
                        for i, t in alpaca[n:2 * n]]

    gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n))
    pools["math"] = [{"id": f"math-{i}", "pool": "math", "messages": P.math_messages(r["question"]),
                      "check": {"answer": r["answer"].split("####")[-1].strip().replace(",", "")}} for i, r in enumerate(gsm)]

    pku = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train").shuffle(seed=seed)
    harmful, seen = [], set()
    for r in pku:
        if not r["is_response_0_safe"] and not r["is_response_1_safe"] and r["prompt"] not in seen:
            seen.add(r["prompt"])
            harmful.append(r["prompt"])
        if len(harmful) >= n // 2:
            break
    benign = [t for _, t in alpaca[2 * n:2 * n + (n - len(harmful))]]
    safe = [{"id": f"safe-h{i}", "pool": "safe", "messages": P.plain_messages(t), "check": {"unsafe": True}}
            for i, t in enumerate(harmful)]
    safe += [{"id": f"safe-b{i}", "pool": "safe", "messages": P.plain_messages(t), "check": {"unsafe": False}}
             for i, t in enumerate(benign)]
    random.Random(seed).shuffle(safe)
    pools["safe"] = safe
    return pools


def fixture_pools(n: int) -> dict:
    """Tiny synthetic pools for the smoke test."""
    from .eval_sets import fixture_items
    cap_of = {"fmt": "fmt", "tool": "tool", "inst": "inst", "math": "math", "safe": "safe", "general": "math"}
    pools = {}
    for pool, cap in cap_of.items():
        items = fixture_items(cap, n, offset=1000)
        pools[pool] = [{"id": f"{pool}-{k}", "pool": pool, "messages": it["messages"],
                        "check": {} if pool == "general" else _check_from_meta(cap, it["meta"])}
                       for k, it in enumerate(items)]
    return pools


def _check_from_meta(cap, meta):
    if cap == "fmt":
        return {"schema": meta["schema"]}
    if cap == "tool":
        return {"functions": meta["functions"]}
    if cap == "inst":
        return {k: meta[k] for k in ("instruction_id_list", "kwargs", "prompt")}
    if cap == "math":
        return {"answer": meta["answer"]}
    return {"unsafe": meta["unsafe"]}


# --------------------------------------------------------------------------- teacher QC
def passes_qc(pool: str, response: str, check: dict) -> bool:
    """Keep only teacher outputs that actually exhibit the skill we want to restore."""
    if not response.strip():
        return False
    if pool == "fmt":
        import jsonschema
        obj = extract_json(response)
        return obj is not None and jsonschema.validators.validator_for(check["schema"])(check["schema"]).is_valid(obj)
    if pool == "tool":
        name, args = parse_call(extract_json(response))
        return name is not None and args is not None and name in {f["name"] for f in check["functions"]}
    if pool == "inst":
        from ..evaluation.scorers import score_inst
        return score_inst(response, {"key": 0, **check})["score"] == 1.0
    if pool == "math":
        pred = extract_number(response)
        return pred is not None and abs(pred - float(check["answer"])) < 1e-6
    return True
