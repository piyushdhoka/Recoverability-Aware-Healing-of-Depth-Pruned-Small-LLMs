"""Stage 1: build fixed test/dev evaluation items (model-independent; run once, commit data/eval)."""
import _bootstrap  # noqa: F401

from rah.data.eval_sets import BUILDERS, fixture_items
from rah.pipeline import cli, log
from rah.utils import write_jsonl


def main():
    args, cfg, p = cli(__doc__)
    d = cfg["data"]
    for cap in cfg["capabilities"]:
        tpath, dpath = p["eval"] / f"{cap}_test.jsonl", p["eval"] / f"{cap}_dev.jsonl"
        if tpath.exists() and dpath.exists():
            log.info(f"{cap}: exists, skipping")
            continue
        n_t, n_d = d["eval_test"][cap], d["eval_dev"][cap]
        if cfg["fixture_data"]:
            test, dev = fixture_items(cap, n_t), fixture_items(cap, n_d, offset=n_t)
        elif cap == "fmt":
            test, dev = BUILDERS[cap](n_t, n_d, cfg["seed"], d["fmt_subset"], d["max_schema_chars"])
        else:
            test, dev = BUILDERS[cap](n_t, n_d, cfg["seed"])
        assert not ({x["id"] for x in test} & {x["id"] for x in dev}), f"{cap}: test/dev overlap"
        write_jsonl(tpath, test)
        write_jsonl(dpath, dev)
        log.info(f"{cap}: test={len(test)} dev={len(dev)}")


if __name__ == "__main__":
    main()
