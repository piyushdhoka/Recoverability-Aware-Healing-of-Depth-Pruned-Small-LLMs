"""Recovery model: per-capability saturating curves + a non-negative pool->capability transfer matrix.

    gain_c(t) = g_c( x_c ),   x_c = sum_p T[c, p] * t_p,   T[c, own(c)] = 1,  T >= 0
    exp:  g(x) = a (1 - exp(-x / tau))        hyp:  g(x) = a x / (x + tau)

T is constrained non-negative so the allocation problem stays concave; negative interference between
pools is still visible in the raw pilot table and is reported descriptively.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


def curve(x, a, tau, kind: str = "exp"):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if kind == "exp":
        return a * (1.0 - np.exp(-x / tau))
    if kind == "hyp":
        return a * x / (x + tau)
    raise ValueError(kind)


@dataclass
class RecoveryModel:
    caps: list
    pools: list
    own: dict
    r0: dict                      # retention of the pruned, un-healed model (dev set)
    a: dict = field(default_factory=dict)
    tau: dict = field(default_factory=dict)
    T: dict = field(default_factory=dict)  # T[c][p]
    kind: str = "exp"
    max_ceiling: float = 1.2
    fit_rmse: dict = field(default_factory=dict)

    def effective_tokens(self, t: dict) -> dict:
        return {c: sum(self.T[c].get(p, 0.0) * t.get(p, 0.0) for p in self.pools) for c in self.caps}

    def predict(self, t: dict) -> dict:
        """Predicted retention per capability after healing with tokens t[pool]."""
        x = self.effective_tokens(t)
        return {c: float(min(self.max_ceiling, self.r0[c] + curve(x[c], self.a[c], self.tau[c], self.kind)))
                for c in self.caps}

    def to_dict(self):
        return {k: getattr(self, k) for k in ("caps", "pools", "own", "r0", "a", "tau", "T", "kind",
                                               "max_ceiling", "fit_rmse")}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def fit_recovery(r0: dict, pilots: list[dict], caps: list, pools: list, own: dict, kind: str = "exp",
                 ridge: float = 0.05, max_ceiling: float = 1.2, use_transfer: bool = True) -> RecoveryModel:
    """pilots: [{"pool": p, "budget": tokens, "retention": {cap: r}}, ...] (evaluated on dev items)."""
    budgets = np.array([pl["budget"] for pl in pilots], dtype=float)
    lo_tau, hi_tau = budgets.min() / 50.0, budgets.max() * 200.0
    model = RecoveryModel(caps=caps, pools=pools, own=own, r0=r0, kind=kind, max_ceiling=max_ceiling)
    for c in caps:
        others = [p for p in pools if p != own[c]] if use_transfer else []
        pts = [(pl["pool"], pl["budget"], pl["retention"][c] - r0[c]) for pl in pilots
               if use_transfer or pl["pool"] == own[c]]
        a_max = max(1e-3, max_ceiling - r0[c])
        own_gain = max([d for p, _, d in pts if p == own[c]] + [1e-3])

        def unpack(z):
            a, log_tau = z[0], z[1]
            T = {own[c]: 1.0, **{p: z[2 + i] for i, p in enumerate(others)}}
            return a, float(np.exp(log_tau)), T

        def resid(z):
            a, tau, T = unpack(z)
            r = [curve(T.get(p, 0.0) * b, a, tau, kind) - d for p, b, d in pts]
            return np.array(r + [np.sqrt(ridge) * z[2 + i] for i in range(len(others))])

        z0 = np.concatenate([[min(a_max, max(own_gain * 1.5, 1e-3)), np.log(np.median(budgets))],
                             np.full(len(others), 0.1)])
        lb = np.concatenate([[0.0, np.log(lo_tau)], np.zeros(len(others))])
        ub = np.concatenate([[a_max, np.log(hi_tau)], np.full(len(others), 3.0)])
        z0 = np.clip(z0, lb + 1e-9, ub - 1e-9)
        sol = least_squares(resid, z0, bounds=(lb, ub))
        a, tau, T = unpack(sol.x)
        model.a[c], model.tau[c] = float(a), tau
        model.T[c] = {p: float(T.get(p, 0.0)) for p in pools}
        n = len(pts)
        model.fit_rmse[c] = float(np.sqrt(np.mean(resid(sol.x)[:n] ** 2))) if n else float("nan")
    return model


def pilot_gain_table(r0: dict, pilots: list[dict], caps: list) -> dict:
    """Raw observed gain table {pool: {budget: {cap: delta}}} (includes negative transfer)."""
    out = {}
    for pl in pilots:
        out.setdefault(pl["pool"], {})[str(pl["budget"])] = {c: pl["retention"][c] - r0[c] for c in caps}
    return out
