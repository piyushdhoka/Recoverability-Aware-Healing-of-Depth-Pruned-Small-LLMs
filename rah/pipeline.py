"""Glue shared by the stage scripts: loading data, building pruned models, and one heal+evaluate run."""
import argparse
import gc

import torch

from .config import load_config, paths
from .evaluation.runner import evaluate
from .healing import apply_scope, build_mixture, finalize, mixture_tokens, tokenize_pool, train
from .modeling import PruneSpec, load_model, load_tokenizer, prune_model
from .utils import get_logger, read_json, read_jsonl, set_seed, torch_dtype, write_json

log = get_logger()


def cli(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", required=True, help="config name: qwen | llama | smoke")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.model)
    set_seed(cfg["seed"])
    return args, cfg, paths(cfg)


def load_eval(p, cfg, split: str) -> dict:
    return {c: read_jsonl(p["eval"] / f"{c}_{split}.jsonl") for c in cfg["capabilities"]}


def load_teacher(p, cfg) -> dict:
    return {pool: read_jsonl(p["teacher"] / f"{pool}.jsonl") for pool in cfg["pools"]}


def calib_batches(tok, teacher: dict, cfg: dict, batch_size: int = 8):
    """Equal-length token chunks (no padding) from teacher texts across all pools, BOS-prefixed if the
    tokenizer has one. Used for Block Influence and the linear-patch fit."""
    n_per_pool = max(1, cfg["pruning"]["calib_examples"] // len(teacher))
    L = cfg["pruning"]["calib_max_len"]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    stream = []
    for pool in sorted(teacher):
        for ex in teacher[pool][:n_per_pool]:
            text = ex["messages"][-1]["content"] + "\n" + ex["response"]
            stream += tok(text, add_special_tokens=False)["input_ids"]
    body = L - len(bos)
    chunks = [bos + stream[i:i + body] for i in range(0, len(stream) - body + 1, body)]
    if not chunks:
        raise ValueError("not enough calibration text")
    out = []
    for s in range(0, len(chunks), batch_size):
        ids = torch.tensor(chunks[s:s + batch_size])
        out.append((ids, torch.ones_like(ids)))
    return out


def prune_spec(p, level: str) -> PruneSpec:
    d = read_json(p["results"] / "diagnosis" / "prune_specs.json")[level]
    return PruneSpec(n_layers=d["n_layers"], removed=d["removed"])


def fresh_pruned(cfg, spec: PruneSpec):
    model = load_model(cfg["model_id"], cfg["dtype"])
    return prune_model(model, spec)


def heal_and_evaluate(cfg, p, tok, spec: PruneSpec, tokens_per_pool: dict, scope: str, seed: int,
                      split: str, out_path, tokenized: dict, tag: dict) -> dict:
    """Load a fresh pruned model, heal it with the given mixture/scope, evaluate, save. Resumable."""
    if out_path.exists():
        log.info(f"skip (done): {out_path.relative_to(p['results'].parent.parent)}")
        return read_json(out_path)
    set_seed(seed)
    model = fresh_pruned(cfg, spec)
    mix = build_mixture(tokenized, tokens_per_pool, seed)
    model, lr, desc = apply_scope(model, scope, cfg, spec.cut_adjacent())
    log.info(f"heal {tag} scope={desc} examples={len(mix)} tokens={mixture_tokens(mix)}")
    train_info = train(model, tok, mix, lr, cfg, seed, log)
    model = finalize(model, scope, torch_dtype(cfg["dtype"]))
    ev = evaluate(model, tok, load_eval(p, cfg, split), cfg)
    result = {**tag, "scope": scope, "seed": seed, "split": split, "tokens_per_pool": tokens_per_pool,
              "train": train_info, "summary": ev["summary"], "eval_seconds": ev["seconds"],
              "records": ev["records"]}
    write_json(out_path, result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def tokenized_pools(cfg, p, tok) -> dict:
    return {pool: tokenize_pool(tok, exs, cfg["healing"]["seq_len"]) for pool, exs in load_teacher(p, cfg).items()}


def tokenizer(cfg):
    return load_tokenizer(cfg["model_id"])
