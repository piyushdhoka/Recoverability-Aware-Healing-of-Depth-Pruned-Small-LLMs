import numpy as np

from rah.proxy import calibrate_mapping, proxy_model
from rah.recovery import RecoveryModel
from rah.stats import bootstrap_mean_ci, kendalls_w, paired_bootstrap, retention_ci


def test_kendalls_w():
    assert abs(kendalls_w([[1, 2, 3, 4], [1, 2, 3, 4]]) - 1.0) < 1e-12
    assert kendalls_w([[1, 2, 3, 4], [4, 3, 2, 1]]) == 0.0


def test_bootstrap_and_paired():
    m, lo, hi = bootstrap_mean_ci([1, 0, 1, 1, 0, 1, 1, 1], n_boot=500)
    assert lo <= m <= hi
    r = paired_bootstrap([1] * 50, [0] * 50, n_boot=200)
    assert r["diff"] == 1.0 and r["p"] == 0.0
    rt, lo, hi = retention_ci([1, 1, 0, 0], 0.5, n_boot=200)
    assert rt == 1.0 and lo <= rt <= hi


def test_proxy_mapping():
    other = RecoveryModel(caps=["a", "b", "c"], pools=["a", "b", "c"], own={"a": "a", "b": "b", "c": "c"},
                          r0={"a": 0.5, "b": 0.5, "c": 0.5}, a={"a": 0.1, "b": 0.2, "c": 0.3},
                          tau={"a": 1e5, "b": 2e5, "c": 3e5})
    alpha, beta, tau = calibrate_mapping(other, {"a": 1.0, "b": 2.0, "c": 3.0})
    assert abs(alpha) < 1e-9 and abs(beta - 0.1) < 1e-9 and tau == 2e5
    pm = proxy_model({"a": 0.4, "b": 0.9, "c": 0.1}, {"a": 2.0, "b": 2.0, "c": 50.0}, (alpha, beta, tau),
                     ["a", "b", "c"], ["a", "b", "c"], other.own)
    assert abs(pm.a["a"] - 0.2) < 1e-9
    assert pm.a["c"] <= 1.2 - 0.1 + 1e-9          # clipped at the retention ceiling
    assert pm.T["a"] == {"a": 1.0, "b": 0.0, "c": 0.0}
