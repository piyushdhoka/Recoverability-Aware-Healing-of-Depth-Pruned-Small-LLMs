"""Statistics: bootstrap CIs on retention, paired bootstrap between methods, rank agreement."""
import numpy as np
from scipy.stats import spearmanr


def bootstrap_mean_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def retention_ci(item_scores, base_score: float, n_boot: int = 2000, seed: int = 0):
    """Retention = mean(item_scores)/base_score with a bootstrap CI over items (baseline held fixed)."""
    m, lo, hi = bootstrap_mean_ci(item_scores, n_boot, seed)
    return m / base_score, lo / base_score, hi / base_score


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) over paired items, 95% CI, and two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(bs, 0.025)),
            "ci_high": float(np.quantile(bs, 0.975)), "p": float(min(1.0, p))}


def spearman(x, y):
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def kendalls_w(rank_matrix) -> float:
    """Kendall's coefficient of concordance for m raters (rows) ranking n items (cols), no ties correction."""
    R = np.asarray(rank_matrix, dtype=float)
    m, n = R.shape
    if n < 2 or m < 2:
        return float("nan")
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))
