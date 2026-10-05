"""Zero-cost proxy (RAH-proxy): predict recovery ceilings without pilot runs.

Signal: per-capability increase in negative log-likelihood of the teacher's own responses after pruning,
dNLL_c = NLL_pruned - NLL_teacher, a Monte-Carlo estimate of KL(teacher || pruned) on that capability's
data. Only scalars are stored, so the teacher and pruned model never need to be in memory together.
A linear map dNLL -> ceiling a_c is calibrated on the *other* model family's fitted curves.
"""
import numpy as np
import torch

from .modeling import chat_ids
from .recovery import RecoveryModel


@torch.no_grad()
def mean_nll(model, tok, examples: list[dict], seq_len: int) -> float:
    total, n = 0.0, 0
    for ex in examples:
        ids, labels = chat_ids(tok, ex["messages"], ex["response"], max_len=seq_len)
        ids_t = torch.tensor([ids], device=model.device)
        lab_t = torch.tensor([labels], device=model.device)
        k = int((lab_t[:, 1:] != -100).sum())
        if k == 0:
            continue
        loss = model(input_ids=ids_t, labels=lab_t, use_cache=False).loss
        total += float(loss) * k
        n += k
    return total / max(n, 1)


def calibrate_mapping(other: RecoveryModel, other_dnll: dict) -> tuple[float, float, float]:
    """Least-squares a_c ~ alpha + beta * dNLL_c on the other family; returns (alpha, beta, median tau)."""
    caps = [c for c in other.caps if c in other_dnll]
    x = np.array([other_dnll[c] for c in caps])
    y = np.array([other.a[c] for c in caps])
    A = np.vstack([np.ones_like(x), x]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(alpha), float(beta), float(np.median([other.tau[c] for c in caps]))


def proxy_model(r0: dict, dnll: dict, mapping: tuple, caps: list, pools: list, own: dict,
                kind: str = "exp", max_ceiling: float = 1.2) -> RecoveryModel:
    alpha, beta, tau = mapping
    m = RecoveryModel(caps=caps, pools=pools, own=own, r0=r0, kind=kind, max_ceiling=max_ceiling)
    for c in caps:
        m.a[c] = float(np.clip(alpha + beta * dnll[c], 0.0, max(1e-3, max_ceiling - r0[c])))
        m.tau[c] = tau
        m.T[c] = {p: (1.0 if p == own[c] else 0.0) for p in pools}   # no transfer information without pilots
    return m
