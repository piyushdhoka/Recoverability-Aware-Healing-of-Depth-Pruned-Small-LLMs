"""Healing-budget allocation rules.

  uniform        equal tokens per pool
  damage_prop    tokens proportional to the damage of the capabilities a pool owns (PASER-style)
  rah_sum        maximise sum_c w_c * predicted retention        (concave program)
  rah_maxmin     maximise min_c predicted retention              (concave program, epigraph form)
"""
import numpy as np
from scipy.optimize import minimize

from .recovery import RecoveryModel


def uniform(pools: list, budget: float) -> dict:
    return {p: budget / len(pools) for p in pools}


def damage_proportional(r0: dict, own: dict, pools: list, budget: float) -> dict:
    dmg = {p: 0.0 for p in pools}
    for c, p in own.items():
        dmg[p] += max(0.0, 1.0 - r0[c])
    total = sum(dmg.values())
    if total <= 1e-12:
        return uniform(pools, budget)
    return {p: budget * dmg[p] / total for p in pools}


def _starts(n: int, seed: int, extra: int = 8):
    rng = np.random.default_rng(seed)
    yield np.full(n, 1.0 / n)
    for i in range(n):                      # one start concentrated on each pool
        f = np.full(n, 0.2 / max(n - 1, 1))
        f[i] = 0.8
        yield f / f.sum()
    for _ in range(extra):
        yield rng.dirichlet(np.ones(n))


def rah(model: RecoveryModel, budget: float, objective: str = "sum", weights: dict | None = None,
        seed: int = 0) -> dict:
    pools, caps = model.pools, model.caps
    w = np.array([(weights or {}).get(c, 1.0) for c in caps])
    n = len(pools)

    def pred(f):
        r = model.predict({p: budget * fi for p, fi in zip(pools, f)})
        return np.array([r[c] for c in caps])

    simplex = {"type": "eq", "fun": lambda z: np.sum(z[:n]) - 1.0}
    best = None
    for f0 in _starts(n, seed):
        if objective == "sum":
            res = minimize(lambda f: -float(w @ pred(f)), f0, method="SLSQP",
                           bounds=[(0.0, 1.0)] * n, constraints=[simplex])
            val, f = -res.fun, res.x
        elif objective == "maxmin":
            z0 = np.concatenate([f0, [pred(f0).min()]])
            cons = [simplex, {"type": "ineq", "fun": lambda z: pred(z[:n]) - z[n]}]
            res = minimize(lambda z: -z[n], z0, method="SLSQP", bounds=[(0.0, 1.0)] * n + [(None, None)],
                           constraints=cons)
            f = res.x[:n]
            val = float(pred(np.clip(f, 0, 1)).min())
        else:
            raise ValueError(objective)
        f = np.clip(f, 0.0, None)
        f = f / f.sum() if f.sum() > 0 else np.full(n, 1.0 / n)
        if best is None or val > best[0] + 1e-12:
            best = (val, f)
    return {p: float(budget * fi) for p, fi in zip(pools, best[1])}


def choose_scope(scope_pilots: dict, caps: list, objective: str = "sum", weights: dict | None = None) -> str:
    """scope_pilots: {scope: {cap: retention}} from equal-budget uniform-mixture pilots."""
    def value(r):
        vals = [r[c] * (weights or {}).get(c, 1.0) for c in caps]
        return min(vals) if objective == "maxmin" else sum(vals)
    return max(scope_pilots, key=lambda s: value(scope_pilots[s]))


def best_scope_per_capability(scope_pilots: dict, caps: list) -> dict:
    """For H3: which scope recovers each capability best."""
    return {c: max(scope_pilots, key=lambda s: scope_pilots[s][c]) for c in caps}
