"""Decision intervals and decision rules of the plan (Phase 5, "Statistics"; Phase 1, "Decision rule").

Decision interval = the union of
  (a) a paired cluster bootstrap over site groups: 2,000 resamples of the G
      groups with replacement; the same draws for both encoders, because the
      statistic is computed from one weight vector for both; percentile 2.5-97.5;
  (b) a delete-one-group jackknife t-interval with G - 1 degrees of freedom,
      centred on the full-sample estimate.
The statistic is any function of per-study weights (multiplicities), for
example Delta1 = macro AUC(DINOv2) - macro AUC(ConvNeXt) on S2 predictions.

Decision rule, applied in this order, for a difference A - B:
  interval inside [-0.02, +0.02]: "equivalent"; else lower bound above 0:
  "A better"; else upper bound below 0: "B better"; else "inconclusive".
"""
from __future__ import annotations

from typing import Callable

import numpy as np

MARGIN = 0.02
N_BOOT = 2000


def _codes(groups) -> tuple[np.ndarray, int]:
    import pandas as pd

    g = pd.Series(np.asarray(groups)).astype("category")
    codes = g.cat.codes.to_numpy().astype(int)
    if (codes < 0).any():
        raise ValueError("a study without a site group reached an interval")
    return codes, len(g.cat.categories)


def cluster_bootstrap(stat: Callable[[np.ndarray], float], groups, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    codes, G = _codes(groups)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, G, size=G)
        vals[b] = stat(np.bincount(draw, minlength=G)[codes].astype(float))
    ok = np.isfinite(vals)
    if not ok.any():
        return {"lo": None, "hi": None, "valid_share": 0.0}
    lo, hi = np.percentile(vals[ok], [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "valid_share": float(ok.mean())}


def jackknife_t(stat: Callable[[np.ndarray], float], groups, alpha: float = 0.05) -> dict:
    from scipy.stats import t as student_t

    codes, G = _codes(groups)
    full = stat(np.ones(len(codes)))
    if G < 3 or not np.isfinite(full):
        return {"lo": None, "hi": None, "se": None}
    reps = np.array([stat((codes != g).astype(float)) for g in range(G)])
    if not np.isfinite(reps).all():
        return {"lo": None, "hi": None, "se": None}
    se = float(np.sqrt((G - 1) / G * ((reps - reps.mean()) ** 2).sum()))
    q = float(student_t.ppf(1 - alpha / 2, G - 1))
    return {"lo": float(full - q * se), "hi": float(full + q * se), "se": se}


def decision_interval(stat: Callable[[np.ndarray], float], groups, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimate, both component intervals and their union."""
    codes, G = _codes(groups)
    est = stat(np.ones(len(codes)))
    b = cluster_bootstrap(stat, groups, n_boot, seed)
    j = jackknife_t(stat, groups)
    parts = [(x["lo"], x["hi"]) for x in (b, j) if x["lo"] is not None]
    lo = min(p[0] for p in parts) if parts else None
    hi = max(p[1] for p in parts) if parts else None
    return {"estimate": float(est) if np.isfinite(est) else None, "lo": lo, "hi": hi,
            "half_width": (hi - lo) / 2 if parts else None, "bootstrap": b, "jackknife": j,
            "n_groups": int(G), "approximate": G < 10}


def decide(lo, hi, margin: float = MARGIN, a: str = "A better", b: str = "B better") -> str:
    if lo is None or hi is None:
        return "not testable"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if lo > 0:
        return a
    if hi < 0:
        return b
    return "inconclusive"


def combine_groupings(lang: str, vendor: str) -> str:
    """H1: the same category under G-lang and G-vendor is the decision; else grouping-sensitive."""
    return lang if lang == vendor else "inconclusive (grouping-sensitive)"
