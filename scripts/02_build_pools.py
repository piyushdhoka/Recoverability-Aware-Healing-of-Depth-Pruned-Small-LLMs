"""Stage 2: build healing-pool prompts (model-independent; run once, commit data/pools)."""
import _bootstrap  # noqa: F401

from rah.data.pools import build_pools, fixture_pools
from rah.pipeline import cli, log
from rah.utils import write_jsonl


def main():
    args, cfg, p = cli(__doc__)
    if all((p["pools"] / f"{pool}.jsonl").exists() for pool in cfg["pools"]):
        log.info("pools exist, skipping")
        return
    n = cfg["data"]["pool_prompts"]
    pools = fixture_pools(n) if cfg["fixture_data"] else build_pools(n, cfg["seed"])
    for pool in cfg["pools"]:
        write_jsonl(p["pools"] / f"{pool}.jsonl", pools[pool])
        log.info(f"{pool}: {len(pools[pool])} prompts")


if __name__ == "__main__":
    main()
