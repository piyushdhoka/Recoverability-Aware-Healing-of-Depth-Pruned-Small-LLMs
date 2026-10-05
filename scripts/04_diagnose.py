"""Stage 4: baseline evaluation, Block-Influence pruning, damage diagnosis, proxy signal, linear-patch baseline.

One model load does everything that needs the UNPRUNED weights:
  M0 test/dev eval -> BI ranking -> span states for the linear patch -> teacher NLL (proxy)
  -> prune (main) -> pruned test/dev eval -> pruned NLL -> linear-patch eval -> reload, prune (light) -> eval
"""
import gc

import _bootstrap  # noqa: F401
import torch

from rah.evaluation.runner import evaluate, retention
from rah.influence import block_influence, lowest_k
from rah.linear_patch import attach_patches, collect_span_states, fit_patches
from rah.modeling import PruneSpec, get_blocks, load_model, prune_model
from rah.pipeline import calib_batches, cli, load_eval, load_teacher, log, tokenizer
from rah.proxy import mean_nll
from rah.utils import read_json, timed, write_json

CHANCE = {"fmt": 0.0, "tool": 0.0, "inst": 0.0, "math": 0.0, "safe": 0.5, "know": 0.25}


def main():
    args, cfg, p = cli(__doc__)
    out = p["results"] / "diagnosis"
    if (out / "damage.json").exists():
        log.info("diagnosis exists, skipping")
        return
    tok = tokenizer(cfg)
    teacher = load_teacher(p, cfg)
    caps, own = cfg["capabilities"], cfg["own_pool"]
    ev = {}

    model = load_model(cfg["model_id"], cfg["dtype"])
    n_layers = len(get_blocks(model)[0])
    for split in ("test", "dev"):
        cached = out / f"eval_base_{split}.json"   # unpruned scores don't depend on the pruning level
        if cached.exists():
            ev[f"base_{split}"] = read_json(cached)
            log.info(f"M0 {split} eval: reusing {cached.name}")
            continue
        with timed(log, f"M0 {split} eval"):
            ev[f"base_{split}"] = evaluate(model, tok, load_eval(p, cfg, split), cfg)

    batches = calib_batches(tok, teacher, cfg)
    bi = block_influence(model, batches)
    specs = {lvl: PruneSpec(n_layers, lowest_k(bi, cfg[f"prune_{lvl}"])) for lvl in ("main", "light")}
    write_json(out / "prune_specs.json", {k: v.to_dict() for k, v in specs.items()})
    write_json(out / "block_influence.json", {"bi": bi.tolist()})
    log.info(f"BI pruning sets: { {k: v.removed for k, v in specs.items()} }")

    span_states = collect_span_states(model, batches, specs["main"])
    n_proxy = cfg["proxy"]["examples_per_pool"]
    sl = cfg["healing"]["seq_len"]
    nll0 = {pool: mean_nll(model, tok, teacher[pool][:n_proxy], sl) for pool in cfg["pools"]}

    prune_model(model, specs["main"])
    for split in ("test", "dev"):
        with timed(log, f"pruned(main) {split} eval"):
            ev[f"main_{split}"] = evaluate(model, tok, load_eval(p, cfg, split), cfg)
    nllp = {pool: mean_nll(model, tok, teacher[pool][:n_proxy], sl) for pool in cfg["pools"]}
    write_json(out / "proxy_dnll.json", {c: nllp[own[c]] - nll0[own[c]] for c in caps})

    handles = attach_patches(model, specs["main"], fit_patches(span_states))
    with timed(log, "linear patch test eval"):
        lp = evaluate(model, tok, load_eval(p, cfg, "test"), cfg)
    for h in handles:
        h.remove()
    write_json(p["results"] / "main" / "linear_patch_s0.json",
               {"method": "linear_patch", "seed": 0, "split": "test", "summary": lp["summary"],
                "records": lp["records"], "tokens_per_pool": {}, "scope": "none"})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = prune_model(load_model(cfg["model_id"], cfg["dtype"]), specs["light"])
    ev["light_test"] = evaluate(model, tok, load_eval(p, cfg, "test"), cfg)

    for k, v in ev.items():
        write_json(out / f"eval_{k}.json", {"summary": v["summary"], "records": v["records"], "seconds": v["seconds"]})
    damage = {}
    for lvl, split in (("main", "test"), ("main", "dev"), ("light", "test")):
        r = retention(ev[f"{lvl}_{split}"]["summary"], ev[f"base_{split}"]["summary"], caps)
        damage[f"{lvl}_{split}"] = {"retention": r, "damage": {c: 1 - r[c] for c in caps}}
    pruned = ev["main_test"]["summary"]
    damage["near_floor"] = [c for c in caps if pruned[c] <= CHANCE[c] + 0.05]
    damage["eval_seconds"] = {k: v["seconds"] for k, v in ev.items()}
    write_json(out / "damage.json", damage)
    if damage["near_floor"]:
        log.warning(f"capabilities near chance after pruning: {damage['near_floor']} -> consider fewer removed blocks")
    log.info(f"retention (main, test): { {c: round(x, 3) for c, x in damage['main_test']['retention'].items()} }")


if __name__ == "__main__":
    main()
