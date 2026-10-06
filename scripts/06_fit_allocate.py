"""Stage 6: fit recovery curves + transfer matrix from pilots, test H1, choose scope, and compute every
method's healing allocation (main budget, ablations, budget sweep). Cheap: no GPU needed."""
import _bootstrap  # noqa: F401

from rah import allocate as A
from rah.config import REPO
from rah.evaluation.runner import retention
from rah.proxy import calibrate_mapping, proxy_model
from rah.pipeline import cli, log
from rah.recovery import RecoveryModel, fit_recovery, pilot_gain_table
from rah.stats import spearman
from rah.utils import read_json, write_json


def finite(r: dict) -> dict:
    """A capability the unpruned model scores 0 on has undefined retention; treat it as 0 (and warn)."""
    bad = [c for c, v in r.items() if v != v]
    if bad:
        log.warning(f"undefined retention for {bad} (baseline score 0) -> set to 0")
    return {c: (0.0 if v != v else v) for c, v in r.items()}


def main():
    args, cfg, p = cli(__doc__)
    caps, pools, own = cfg["capabilities"], cfg["pools"], cfg["own_pool"]
    diag = p["results"] / "diagnosis"
    base_dev = read_json(diag / "eval_base_dev.json")["summary"]
    r0 = finite(read_json(diag / "damage.json")["main_dev"]["retention"])

    pilots, scope_pilots = [], {}
    for pool in pools:
        for b in cfg["pilots"]["budgets"]:
            res = read_json(p["results"] / "pilots" / f"recovery_{pool}_{b}.json")
            pilots.append({"pool": pool, "budget": b, "retention": finite(retention(res["summary"], base_dev, caps))})
    for scope in cfg["pilots"]["scopes"]:
        res = read_json(p["results"] / "pilots" / f"scope_{scope}.json")
        scope_pilots[scope] = finite(retention(res["summary"], base_dev, caps))

    fc = cfg["fit"]
    tmf = fc["tau_min_frac"]
    fits = {
        "exp": fit_recovery(r0, pilots, caps, pools, own, "exp", fc["transfer_ridge"], fc["max_ceiling"],
                            tau_min_frac=tmf),
        "hyp": fit_recovery(r0, pilots, caps, pools, own, "hyp", fc["transfer_ridge"], fc["max_ceiling"],
                            tau_min_frac=tmf),
        "no_transfer": fit_recovery(r0, pilots, caps, pools, own, fc["curve"], fc["transfer_ridge"],
                                    fc["max_ceiling"], use_transfer=False, tau_min_frac=tmf),
    }
    # trust region: never give a pool more than trust_factor x the largest budget it was piloted at
    cap = fc["trust_factor"] * max(cfg["pilots"]["budgets"])

    def R(model, budget, objective="sum"):
        return A.rah(model, budget, objective, max_per_pool=cap)
    main_fit = fits[fc["curve"]]
    write_json(p["results"] / "fit" / "recovery.json", {k: v.to_dict() for k, v in fits.items()})
    write_json(p["results"] / "fit" / "pilot_gains.json", pilot_gain_table(r0, pilots, caps))

    damage = [1 - r0[c] for c in caps]
    rho_a, p_a = spearman(damage, [main_fit.a[c] for c in caps])
    rho_t, p_t = spearman(damage, [main_fit.tau[c] for c in caps])
    h1 = {"spearman_damage_vs_ceiling": rho_a, "p": p_a, "spearman_damage_vs_tau": rho_t, "p_tau": p_t,
          "gate_G1_pass": bool(rho_a < 0.8)}
    write_json(p["results"] / "fit" / "h1.json", h1)
    log.info(f"H1: rho(damage, ceiling) = {rho_a:.3f}  -> gate G1 {'PASS' if h1['gate_G1_pass'] else 'PIVOT'}")

    scope_sum = A.choose_scope(scope_pilots, caps, "sum")
    scope_mm = A.choose_scope(scope_pilots, caps, "maxmin")
    std = cfg["pilots"]["scope"]                       # standard scope used by the baselines
    B = cfg["healing"]["budget_tokens"]
    alloc = {
        "uniform": {"tokens": A.uniform(pools, B), "scope": std},
        "damage_prop": {"tokens": A.damage_proportional(r0, own, pools, B), "scope": std},
        "rah_sum": {"tokens": R(main_fit, B, "sum"), "scope": scope_sum},
        "rah_maxmin": {"tokens": R(main_fit, B, "maxmin"), "scope": scope_mm},
        "last_k_uniform": {"tokens": A.uniform(pools, B), "scope": "last_k"},
        "rah_no_transfer": {"tokens": R(fits["no_transfer"], B, "sum"), "scope": scope_sum},
        "rah_no_scope": {"tokens": R(main_fit, B, "sum"), "scope": std},
        "rah_hyp": {"tokens": R(fits["hyp"], B, "sum"), "scope": scope_sum},
        # pre-registered 2026-10-06 19:40 IST (before OLMo stage-7 results): objective capped at retention 1.0
        "rah_capped": {"tokens": A.rah(main_fit, B, "sum", max_per_pool=cap, pred_cap=fc["objective_cap"]),
                       "scope": scope_sum},
    }

    # RAH-proxy: needs the OTHER model family's fitted curves and proxy signal (synced via git).
    others = [d for d in (REPO / "results").iterdir() if d.is_dir() and d.name not in (cfg["name"], "analysis", "smoke")]
    for od in others:
        rec, dn = od / "fit" / "recovery.json", od / "diagnosis" / "proxy_dnll.json"
        if rec.exists() and dn.exists():
            other_fit = RecoveryModel.from_dict(read_json(rec)[fc["curve"]])
            mapping = calibrate_mapping(other_fit, read_json(dn))
            pm = proxy_model(r0, read_json(diag / "proxy_dnll.json"), mapping, caps, pools, own, fc["curve"], fc["max_ceiling"])
            alloc["rah_proxy"] = {"tokens": R(pm, B, "sum"), "scope": std, "calibrated_on": od.name,
                                  "mapping": mapping}
            break
    if "rah_proxy" not in alloc:
        log.warning("rah_proxy skipped: other model's fit not available yet (rerun this stage after syncing)")

    sweep = {}
    for b in cfg["main"]["budget_sweep"]:
        sweep[str(b)] = {"uniform": {"tokens": A.uniform(pools, b), "scope": std},
                         "rah_sum": {"tokens": R(main_fit, b, "sum"), "scope": scope_sum}}
    write_json(p["results"] / "fit" / "allocations.json",
               {"budget": B, "methods": alloc, "budget_sweep": sweep, "scope_pilots": scope_pilots,
                "best_scope_per_capability": A.best_scope_per_capability(scope_pilots, caps),
                "predicted": {m: main_fit.predict(v["tokens"]) for m, v in alloc.items()}})
    for m, v in alloc.items():
        share = {k: round(t / B, 3) for k, t in v["tokens"].items()}
        log.info(f"{m:16s} scope={v['scope']:9s} mix={share}")


if __name__ == "__main__":
    main()
