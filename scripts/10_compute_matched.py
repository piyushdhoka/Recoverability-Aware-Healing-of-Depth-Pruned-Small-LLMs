"""Stage 10: compute-matched comparison. RAH and the baselines at the 1M-token heal (3 seeds) against uniform healing that
also gets the pilot tokens (4.48M = 1M + 3.48M; every available seed of it). Paired bootstrap over test items within each
capability on the primary metrics (capped mean retention, worst-case retention), seeds pooled through per-item means."""
import _bootstrap  # noqa: F401
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from rah.config import REPO, load_config

N_BOOT, BUDGET = 2000, 4480000
METHODS_1M = ("rah_capped", "rah_sum", "damage_prop", "uniform")


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


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
        cm_files = sorted((res / "budget").glob(f"uniform_b{BUDGET}_s*.json"))
        if not cm_files:
            print(f"{name}: no uniform_b{BUDGET}_s*.json yet, skipping")
            continue
        caps = load_config(name)["capabilities"]
        base = J(res / "diagnosis" / "eval_base_test.json")["summary"]
        cm = [J(f) for f in cm_files]
        im = {c: item_means(cm, c) for c in caps}                 # seeds pooled per item, as for the 1M methods
        r_cm = np.array([np.mean(list(im[c].values())) / base[c] for c in caps])
        per_seed = [np.minimum([r["summary"][c] / base[c] for c in caps], 1).mean() for r in cm]
        rows.append({"model": name, "method": f"uniform@{BUDGET // 1000}k", "seeds": len(cm),
                     "capped_mean": float(np.minimum(r_cm, 1).mean()), "worst": float(r_cm.min()),
                     "capped_mean_seed_sd": float(np.std(per_seed, ddof=1)) if len(cm) > 1 else float("nan"),
                     **{c: float(v) for c, v in zip(caps, r_cm)},
                     "over_refusal": float(np.mean([r["summary"]["safe_over_refusal"] for r in cm])),
                     "harmful_refusal": float(np.mean([r["summary"]["safe_harmful_refusal"] for r in cm]))})
        for m in METHODS_1M:
            files = sorted((res / "main").glob(f"{m}_s*.json"))
            if not files:
                continue
            runs = [J(f) for f in files]
            A, B = {}, {}
            for c in caps:
                a, b = item_means(runs, c), item_means(cm, c)
                ids = sorted(set(a) & set(b))
                A[c], B[c] = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])

            def metrics(idx):
                ra = np.array([A[c][idx[c]].mean() / base[c] for c in caps])
                rb = np.array([B[c][idx[c]].mean() / base[c] for c in caps])
                return (np.minimum(ra, 1).mean(), ra.min(), np.minimum(ra, 1).mean() - np.minimum(rb, 1).mean(),
                        ra.min() - rb.min())

            d0 = metrics({c: np.arange(len(A[c])) for c in caps})
            bs = np.array([metrics({c: rng.integers(0, len(A[c]), len(A[c])) for c in caps}) for _ in range(N_BOOT)])
            row = {"model": name, "method": f"{m}@1M", "seeds": len(runs), "capped_mean": d0[0], "worst": d0[1]}
            for k, lab in ((2, "capped"), (3, "worst")):
                lo, hi = np.quantile(bs[:, k], [0.025, 0.975])
                p = 2 * min((bs[:, k] <= 0).mean(), (bs[:, k] >= 0).mean())
                row.update({f"d_{lab}_vs_cm": d0[k], f"d_{lab}_lo": lo, f"d_{lab}_hi": hi, f"p_{lab}": min(1.0, p)})
            rows.append(row)
    df = pd.DataFrame(rows)
    out = REPO / "results" / "analysis"
    df.to_csv(out / "compute_matched.csv", index=False)
    (out / "compute_matched.md").write_text(df.round(3).to_markdown(index=False), encoding="utf-8")
    cols = ["model", "method", "seeds", "capped_mean", "worst", "d_capped_vs_cm", "p_capped", "d_worst_vs_cm", "p_worst"]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
