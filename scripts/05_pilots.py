"""Stage 5: recovery pilots (each pool alone, 2 budgets, fixed scope) and scope pilots (uniform mixture,
3 scopes). All pilots are evaluated on the DEV split only, so the test split never informs RAH's choices."""
import _bootstrap  # noqa: F401

from rah.allocate import uniform
from rah.pipeline import cli, heal_and_evaluate, log, prune_spec, tokenized_pools, tokenizer


def main():
    args, cfg, p = cli(__doc__)
    tok = tokenizer(cfg)
    spec = prune_spec(p, "main")
    tokenized = tokenized_pools(cfg, p, tok)
    out = p["results"] / "pilots"
    pc = cfg["pilots"]
    for pool in cfg["pools"]:
        for b in pc["budgets"]:
            heal_and_evaluate(cfg, p, tok, spec, {pool: b}, pc["scope"], cfg["seed"], "dev",
                              out / f"recovery_{pool}_{b}.json", tokenized,
                              {"kind": "recovery", "pool": pool, "budget": b})
    for scope in pc["scopes"]:
        heal_and_evaluate(cfg, p, tok, spec, uniform(cfg["pools"], pc["scope_budget"]), scope, cfg["seed"], "dev",
                          out / f"scope_{scope}.json", tokenized, {"kind": "scope", "scope_name": scope})
    log.info("pilots complete")


if __name__ == "__main__":
    main()
