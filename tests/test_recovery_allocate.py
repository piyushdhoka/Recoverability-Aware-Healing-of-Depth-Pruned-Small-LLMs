import numpy as np

from rah import allocate as A
from rah.recovery import RecoveryModel, curve, fit_recovery

CAPS = ["fmt", "math"]
POOLS = ["fmt", "math"]
OWN = {"fmt": "fmt", "math": "math"}


def truth():
    """math is the MOST damaged but barely recoverable; fmt is less damaged and cheap to restore."""
    m = RecoveryModel(caps=CAPS, pools=POOLS, own=OWN, r0={"fmt": 0.6, "math": 0.2})
    m.a = {"fmt": 0.4, "math": 0.05}
    m.tau = {"fmt": 2e5, "math": 5e5}
    m.T = {"fmt": {"fmt": 1.0, "math": 0.1}, "math": {"fmt": 0.0, "math": 1.0}}
    return m


def pilots_from(m, budgets=(1e5, 3e5)):
    out = []
    for p in POOLS:
        for b in budgets:
            out.append({"pool": p, "budget": b, "retention": m.predict({p: b})})
    return out


def test_fit_recovers_ground_truth():
    m = truth()
    fit = fit_recovery(m.r0, pilots_from(m), CAPS, POOLS, OWN, ridge=1e-6)
    assert abs(fit.a["fmt"] - 0.4) < 0.03
    assert abs(fit.tau["fmt"] - 2e5) / 2e5 < 0.1
    assert abs(fit.T["fmt"]["math"] - 0.1) < 0.05
    assert all(v >= 0 for row in fit.T.values() for v in row.values())


def test_rah_beats_damage_proportional_when_damage_is_not_recoverability():
    m = truth()
    B = 1e6
    val = lambda t: sum(m.predict(t).values())
    rah_t = A.rah(m, B, "sum")
    dmg_t = A.damage_proportional(m.r0, OWN, POOLS, B)
    uni_t = A.uniform(POOLS, B)
    assert abs(sum(rah_t.values()) - B) < 1e-3 * B
    assert val(rah_t) >= val(dmg_t) - 1e-9 and val(rah_t) >= val(uni_t) - 1e-9
    assert dmg_t["math"] > dmg_t["fmt"]          # PASER-style puts most data on the most damaged skill
    assert rah_t["fmt"] > rah_t["math"]          # RAH puts it where it actually recovers
    assert val(rah_t) > val(dmg_t) + 0.02


def test_maxmin_improves_worst_case():
    m = truth()
    B = 1e6
    mm = A.rah(m, B, "maxmin")
    s = A.rah(m, B, "sum")
    assert min(m.predict(mm).values()) >= min(m.predict(s).values()) - 1e-6


def test_curves_and_scope_choice():
    assert curve(0, 1, 1) == 0 and abs(curve(1e9, 0.3, 1e5) - 0.3) < 1e-9
    assert abs(curve(1e5, 0.3, 1e5, "hyp") - 0.15) < 1e-12
    # sum: last_k 1.05 > all_lora 0.90;  worst case: all_lora 0.30 > last_k 0.10
    pilots = {"all_lora": {"fmt": 0.6, "math": 0.3}, "last_k": {"fmt": 0.95, "math": 0.1}}
    assert A.choose_scope(pilots, CAPS, "sum") == "last_k"
    assert A.choose_scope(pilots, CAPS, "maxmin") == "all_lora"
    assert A.best_scope_per_capability(pilots, CAPS) == {"fmt": "last_k", "math": "all_lora"}


def test_trust_region_caps_each_pool_and_keeps_budget():
    m = truth()
    B = 1e6
    t = A.rah(m, B, "sum", max_per_pool=6e5)
    assert abs(sum(t.values()) - B) < 1e-6 * B
    assert max(t.values()) <= 6e5 + 1e-3
    t2 = A.rah(m, B, "sum", max_per_pool=1e5)          # infeasible cap -> relaxed to B/n
    assert abs(sum(t2.values()) - B) < 1e-6 * B and max(t2.values()) <= B / 2 + 1e-3


def test_project_onto_capped_simplex():
    f = A._project(np.array([0.9, 0.1, 0.0]), ub=0.5)
    assert abs(f.sum() - 1) < 1e-9 and f.max() <= 0.5 + 1e-12 and f.min() >= 0


def test_tau_lower_bound_follows_smallest_pilot():
    m = truth()
    m.tau["fmt"] = 1e3                                   # truly faster than any pilot can resolve
    fit = fit_recovery(m.r0, pilots_from(m, budgets=(3e4, 1e5, 3e5)), CAPS, POOLS, OWN, tau_min_frac=0.1)
    assert fit.tau["fmt"] >= 3e3 - 1e-6


def test_pred_cap_redirects_budget_from_saturated_skill():
    # fmt already at 1.0 but its curve "promises" 1.3; math is far below 1.0 and recovers slowly
    m = RecoveryModel(caps=CAPS, pools=POOLS, own=OWN, r0={"fmt": 1.0, "math": 0.2}, max_ceiling=1.5)
    m.a = {"fmt": 0.3, "math": 0.5}
    m.tau = {"fmt": 3e5, "math": 5e5}
    m.T = {"fmt": {"fmt": 1.0, "math": 0.0}, "math": {"fmt": 0.0, "math": 1.0}}
    B = 1e6
    uncapped = A.rah(m, B, "sum")
    capped = A.rah(m, B, "sum", pred_cap=1.0)
    assert uncapped["fmt"] > 0.3 * B                      # chases the >1.0 gain
    assert capped["fmt"] < 0.05 * B and capped["math"] > 0.95 * B   # nothing to gain above 1.0 -> math
    assert abs(sum(capped.values()) - B) < 1e-6 * B


def test_damage_proportional_degenerates_to_uniform():
    t = A.damage_proportional({"fmt": 1.0, "math": 1.0}, OWN, POOLS, 100)
    assert t == {"fmt": 50, "math": 50}
