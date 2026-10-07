"""Stage 9: paired bootstrap on the HEADLINE metrics (pre-registered capped mean retention and worst-case
retention, plus the uncapped mean) for RAH vs the two allocation baselines. Items are resampled within each
capability, paired across methods; seeds are pooled through per-item means (same convention as stage 8)."""
import _bootstrap  # noqa: F401
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from rah.config import REPO, load_config

N_BOOT, OURS, BASELINES = 2000, ("rah_sum", "rah_capped"), ("uniform", "damage_prop")


def item_means(runs, cap):
    acc = defaultdict(list)
    for r in runs:
        for rec in r["records"]:
            if rec["cap"] == cap:
                acc[rec["id"]].append(rec["score"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    rng, rows = np.random.default_rng(0), []
    for name in ["llama", "qwen", "smollm", "olmo"]:
        res = REPO / "results" / name
        if not (res / "main").exists():
            continue
        caps = load_config(name)["capabilities"]
        base = json.loads((res / "diagnosis" / "eval_base_test.json").read_text(encoding="utf-8"))["summary"]
        runs = defaultdict(list)
        for f in sorted((res / "main").glob("*_s*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            runs[r["method"]].append(r)
        for ours in OURS:
            for other in BASELINES:
                if ours not in runs or other not in runs:
                    continue
                A, B = {}, {}
                for c in caps:
                    a, b = item_means(runs[ours], c), item_means(runs[other], c)
                    ids = sorted(set(a) & set(b))
                    A[c], B[c] = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])

                def metrics(idx):
                    ra = np.array([A[c][idx[c]].mean() / base[c] for c in caps])
                    rb = np.array([B[c][idx[c]].mean() / base[c] for c in caps])
                    return (np.minimum(ra, 1).mean() - np.minimum(rb, 1).mean(), ra.min() - rb.min(),
                            ra.mean() - rb.mean())

                d0 = metrics({c: np.arange(len(A[c])) for c in caps})
                bs = np.array([metrics({c: rng.integers(0, len(A[c]), len(A[c])) for c in caps})
                               for _ in range(N_BOOT)])
                for k, lab in enumerate(["capped_mean", "worst", "uncapped_mean"]):
                    lo, hi = np.quantile(bs[:, k], [0.025, 0.975])
                    p = 2 * min((bs[:, k] <= 0).mean(), (bs[:, k] >= 0).mean())
                    rows.append({"model": name, "ours": ours, "vs": other, "metric": lab, "diff": d0[k],
                                 "ci_low": lo, "ci_high": hi, "p": min(1.0, p)})
    df = pd.DataFrame(rows)
    out = REPO / "results" / "analysis"
    df.to_csv(out / "headline_tests.csv", index=False)
    (out / "headline_tests.md").write_text(df.round(3).to_markdown(index=False), encoding="utf-8")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
