"""Stage 7: main comparison on the TEST split (core methods x seeds, extras/ablations/budget sweep x 1 seed).
Every run is resumable: finished runs are skipped, so the script can be re-launched after a crash.

Extra seeds for the compute-matched heal, e.g.:
    python scripts/07_main.py --model qwen --only uniform --budgets 4480000 --sweep-seeds 1 2"""
import _bootstrap  # noqa: F401

from rah.pipeline import cli, heal_and_evaluate, log, prune_spec, tokenized_pools, tokenizer
from rah.utils import read_json


def main():
    def extra(ap):
        ap.add_argument("--only", nargs="*", default=None, help="restrict to these methods")
        ap.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="with --only: run these seeds for the selected methods (e.g. --seeds 0 1 2)")
        ap.add_argument("--budgets", nargs="*", type=int, default=None,
                        help="restrict budget-sweep jobs to these budgets (e.g. --budgets 4480000)")
        ap.add_argument("--sweep-seeds", nargs="*", type=int, default=None,
                        help="seeds for budget-sweep jobs (default: the first configured seed only)")
    args, cfg, p = cli(__doc__, extra)
    tok = tokenizer(cfg)
    spec = prune_spec(p, "main")
    tokenized = tokenized_pools(cfg, p, tok)
    alloc = read_json(p["results"] / "fit" / "allocations.json")
    mc, out = cfg["main"], p["results"] / "main"
    s0 = mc["seeds"][0]

    jobs = [(m, s) for m in mc["core_methods"] for s in mc["seeds"]]
    jobs += [(m, s0) for m in mc["extra_methods"] + mc["ablations"] if m != "linear_patch"]  # patch: stage 4
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
        if args.seeds:
            jobs = [(m, s) for m in dict.fromkeys(m for m, _ in jobs) for s in args.seeds]
    for m, s in jobs:
        if m not in alloc["methods"]:
            log.warning(f"{m}: no allocation (see stage 6 log), skipping")
            continue
        a = alloc["methods"][m]
        heal_and_evaluate(cfg, p, tok, spec, a["tokens"], a["scope"], s, "test", out / f"{m}_s{s}.json",
                          tokenized, {"method": m})
    sweep_seeds = args.sweep_seeds if args.sweep_seeds else [s0]
    for b, methods in alloc["budget_sweep"].items():
        if args.budgets and int(b) not in args.budgets:
            continue
        for m, a in methods.items():
            if args.only and m not in args.only:
                continue
            for s in sweep_seeds:
                heal_and_evaluate(cfg, p, tok, spec, a["tokens"], a["scope"], s, "test",
                                  p["results"] / "budget" / f"{m}_b{b}_s{s}.json", tokenized,
                                  {"method": m, "budget": int(b)})
    log.info("main stage complete")


if __name__ == "__main__":
    main()
