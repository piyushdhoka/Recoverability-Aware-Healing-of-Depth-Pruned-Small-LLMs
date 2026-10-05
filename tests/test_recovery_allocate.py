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


def test_damage_proportional_degenerates_to_uniform():
    t = A.damage_proportional({"fmt": 1.0, "math": 1.0}, OWN, POOLS, 100)
    assert t == {"fmt": 50, "math": 50}
