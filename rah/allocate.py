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
        seed: int = 0, max_per_pool: float | None = None, pred_cap: float | None = None) -> dict:
    """max_per_pool: trust region, at most this many tokens per pool (relaxed to budget/n if infeasible),
    so the optimiser never relies on curve extrapolation far beyond the piloted budgets.
    pred_cap: if set, predicted retention above this value earns nothing in the objective (aligns the
    optimiser with the capped-mean metric: exceeding the unpruned model is not "recovery")."""
    pools, caps = model.pools, model.caps
    w = np.array([(weights or {}).get(c, 1.0) for c in caps])
    n = len(pools)
    ub = 1.0 if max_per_pool is None else min(1.0, max(max_per_pool, budget / n) / budget)

    def pred(f):
        r = model.predict({p: budget * fi for p, fi in zip(pools, f)})
        return np.array([r[c] for c in caps])

    def value_sum(f):
        p = pred(f)
        if pred_cap is not None:
            p = np.minimum(p, pred_cap)      # min(concave, const) is concave: problem stays well-behaved
        return float(w @ p)

    simplex = {"type": "eq", "fun": lambda z: np.sum(z[:n]) - 1.0}
    best = None
    for f0 in _starts(n, seed):
        f0 = _project(f0, ub)
        if objective == "sum":
            res = minimize(lambda f: -value_sum(f), f0, method="SLSQP",
                           bounds=[(0.0, ub)] * n, constraints=[simplex])
            f = res.x
        elif objective == "maxmin":
            z0 = np.concatenate([f0, [pred(f0).min()]])
            cons = [simplex, {"type": "ineq", "fun": lambda z: pred(z[:n]) - z[n]}]
            res = minimize(lambda z: -z[n], z0, method="SLSQP", bounds=[(0.0, ub)] * n + [(None, None)],
                           constraints=cons)
            f = res.x[:n]
        else:
            raise ValueError(objective)
        f = _project(f, ub)
        val = value_sum(f) if objective == "sum" else float(pred(f).min())
        if best is None or val > best[0] + 1e-12:
            best = (val, f)
    return {p: float(budget * fi) for p, fi in zip(pools, best[1])}


def _project(f, ub: float, iters: int = 50):
    """Project onto {f >= 0, f <= ub, sum f = 1} by alternating clip and rescale of the free mass."""
    f = np.clip(np.asarray(f, dtype=float), 0.0, ub)
    for _ in range(iters):
        gap = 1.0 - f.sum()
        if abs(gap) < 1e-12:
            break
        free = (f < ub - 1e-12) if gap > 0 else (f > 1e-12)
        if not free.any():
            break
        f[free] += gap / free.sum()
        f = np.clip(f, 0.0, ub)
    return f


def choose_scope(scope_pilots: dict, caps: list, objective: str = "sum", weights: dict | None = None) -> str:
    """scope_pilots: {scope: {cap: retention}} from equal-budget uniform-mixture pilots."""
    def value(r):
        vals = [r[c] * (weights or {}).get(c, 1.0) for c in caps]
        return min(vals) if objective == "maxmin" else sum(vals)
    return max(scope_pilots, key=lambda s: value(scope_pilots[s]))


def best_scope_per_capability(scope_pilots: dict, caps: list) -> dict:
    """For H3: which scope recovers each capability best."""
    return {c: max(scope_pilots, key=lambda s: scope_pilots[s][c]) for c in caps}
