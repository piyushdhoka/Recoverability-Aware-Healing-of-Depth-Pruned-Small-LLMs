"""Stage 8: aggregate every available model into paper tables (CSV + Markdown) and figures.

Run on either laptop after syncing results/ via git. Uses whatever models have finished.
"""
import _bootstrap  # noqa: F401
import argparse
import math
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rah.config import REPO, load_config
from rah.recovery import RecoveryModel, curve
from rah.stats import kendalls_w, paired_bootstrap, retention_ci
from rah.utils import get_logger, read_json, write_json

log = get_logger()
COL = {"uniform": "#8a8984", "damage_prop": "#eb6834", "rah_sum": "#2a78d6", "rah_maxmin": "#1baf7a",
       "rah_proxy": "#4a3aa7", "last_k_uniform": "#eda100", "linear_patch": "#e87ba4"}


def item_means(runs: list[dict], cap: str) -> dict:
    """Per-item score averaged over seeds -> {item_id: score}."""
    acc = defaultdict(list)
    for r in runs:
        for rec in r["records"]:
            if rec["cap"] == cap:
                acc[rec["id"]].append(rec["score"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def analyze_model(name: str, out_dir, n_boot: int):
    cfg = load_config(name)
    res = REPO / "results" / name
    caps = cfg["capabilities"]
    base = read_json(res / "diagnosis" / "eval_base_test.json")["summary"]
    runs = defaultdict(list)
    for f in sorted((res / "main").glob("*_s*.json")):
        r = read_json(f)
        runs[r["method"]].append(r)
    rows, tables = [], {}
    for m, rs in runs.items():
        row = {"model": name, "method": m, "seeds": len(rs)}
        for c in caps:
            im = item_means(rs, c)
            if base[c] > 0 and im:
                rt, lo, hi = retention_ci(list(im.values()), base[c], n_boot)
                row[c], row[f"{c}_lo"], row[f"{c}_hi"] = rt, lo, hi
        vals = [row[c] for c in caps if c in row]
        # Primary (pre-registered 2026-10-05, before RAH test results): mean of min(r_c, 1) + worst case.
        # Exceeding the unpruned model is not "recovery", and one noisy skill (fmt) must not dominate the mean.
        row["mean_ret_capped"] = float(np.mean(np.minimum(vals, 1.0)))
        row["mean_ret"], row["worst_ret"] = float(np.mean(vals)), float(np.min(vals))
        side = [r["summary"] for r in rs]
        for k in ("safe_over_refusal", "know_ece"):
            row[k] = float(np.nanmean([s.get(k, float("nan")) for s in side]))
        rows.append(row)
    main_df = pd.DataFrame(rows).sort_values("mean_ret_capped", ascending=False)

    # paired tests: rah_sum vs each other method, per capability (items averaged over seeds)
    tests = []
    if "rah_sum" in runs:
        for m in runs:
            if m == "rah_sum":
                continue
            for c in caps:
                a, b = item_means(runs["rah_sum"], c), item_means(runs[m], c)
                ids = sorted(set(a) & set(b))
                if ids and base[c] > 0:
                    t = paired_bootstrap([a[i] for i in ids], [b[i] for i in ids], n_boot)
                    tests.append({"model": name, "vs": m, "cap": c, **{k: v / base[c] if k != "p" else v
                                                                       for k, v in t.items()}})
    tables["main"], tables["tests"] = main_df, pd.DataFrame(tests)

    dmg = read_json(res / "diagnosis" / "damage.json")
    fit_path = res / "fit" / "recovery.json"
    fit = RecoveryModel.from_dict(read_json(fit_path)[cfg["fit"]["curve"]]) if fit_path.exists() else None
    h1 = read_json(res / "fit" / "h1.json") if (res / "fit" / "h1.json").exists() else {}
    tables["damage"] = pd.DataFrame([{"model": name, "cap": c, "retention_main": dmg["main_test"]["retention"][c],
                                      "retention_light": dmg["light_test"]["retention"][c],
                                      "ceiling_a": fit.a[c] if fit else math.nan,
                                      "tau": fit.tau[c] if fit else math.nan} for c in caps])

    budget = []
    for f in sorted((res / "budget").glob("*.json")):
        r = read_json(f)
        vals = [r["summary"][c] / base[c] for c in caps if base[c] > 0]
        budget.append({"model": name, "method": r["method"], "budget": r["budget"], "mean_ret": float(np.mean(vals))})
    tables["budget"] = pd.DataFrame(budget)

    figures(name, out_dir, caps, dmg, fit, main_df, res)
    return tables, h1


def figures(name, out_dir, caps, dmg, fit, main_df, res):
    if fit is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        d = [dmg["main_dev"]["damage"][c] for c in caps]
        a = [fit.a[c] for c in caps]
        ax.scatter(d, a, color="#2a78d6")
        for c, x, y in zip(caps, d, a):
            ax.annotate(c, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        lim = max(d + a + [0.1]) * 1.1
        ax.plot([0, lim], [0, lim], ls=":", color="#8a8984", lw=1)
        ax.set_xlabel("Damage 1 - r_c"); ax.set_ylabel("Recovery ceiling a_c")
        ax.set_title(f"{name}: damage vs recoverability", fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / f"{name}_damage_vs_ceiling.pdf"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        x = np.linspace(0, 2e6, 200)
        for c in caps:
            ax.plot(x / 1e6, fit.r0[c] + curve(x, fit.a[c], fit.tau[c], fit.kind), label=c)
        ax.set_xlabel("Own-pool healing tokens (M)"); ax.set_ylabel("Predicted retention")
        ax.legend(fontsize=6, frameon=False); fig.tight_layout()
        fig.savefig(out_dir / f"{name}_recovery_curves.pdf"); plt.close(fig)

        T = np.array([[fit.T[c][p] for p in fit.pools] for c in caps])
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        im = ax.imshow(T, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(fit.pools))); ax.set_xticklabels(fit.pools, rotation=45, fontsize=7)
        ax.set_yticks(range(len(caps))); ax.set_yticklabels(caps, fontsize=7)
        ax.set_xlabel("Healing pool"); ax.set_ylabel("Capability"); fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(out_dir / f"{name}_transfer.pdf"); plt.close(fig)

    if not main_df.empty:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        methods = list(main_df["method"])
        w = 0.8 / len(methods)
        for k, m in enumerate(methods):
            row = main_df[main_df.method == m].iloc[0]
            ys = [row.get(c, np.nan) for c in caps]
            ax.bar(np.arange(len(caps)) + k * w, ys, width=w, label=m, color=COL.get(m, None))
        ax.set_xticks(np.arange(len(caps)) + 0.4 - w / 2); ax.set_xticklabels(caps)
        ax.axhline(1.0, color="#0b0b0b", lw=0.7); ax.set_ylabel("Retention")
        ax.legend(fontsize=6, ncol=4, frameon=False); fig.tight_layout()
        fig.savefig(out_dir / f"{name}_retention_by_method.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=["qwen", "llama", "smollm", "olmo"])
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    out = REPO / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    all_tables, h1s, orders = defaultdict(list), {}, {}
    for name in args.models:
        if not (REPO / "results" / name / "diagnosis" / "damage.json").exists():
            log.warning(f"{name}: no diagnosis yet, skipping")
            continue
        tables, h1 = analyze_model(name, out, args.n_boot)
        for k, v in tables.items():
            all_tables[k].append(v)
        h1s[name] = h1
        dmg = tables["damage"].set_index("cap")["retention_main"]
        orders[name] = dmg.rank().tolist()
    for k, frames in all_tables.items():
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / f"{k}.csv", index=False)
        (out / f"{k}.md").write_text(df.round(4).to_markdown(index=False), encoding="utf-8")
    summary = {"h1": h1s}
    if len(orders) >= 2:
        summary["kendalls_w_breakage_order"] = kendalls_w(list(orders.values()))
    write_json(out / "summary.json", summary)
    log.info(f"analysis written to {out}")


if __name__ == "__main__":
    main()
