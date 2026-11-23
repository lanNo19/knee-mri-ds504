"""AUC metrics of the plan (Phase 5, "Metrics"), all tie-aware and weightable.

Weights are per-study multiplicities, so the cluster bootstrap (a group drawn
twice counts twice) and the jackknife (a group left out counts zero) reuse
the same code.

- auc: one finding, one set of studies.
- stratified_auc: AUC over positive-negative pairs inside the same stratum
  (test fold, or fold x site group), pair-weighted across strata. Pairs never
  cross fold models, so calibration offsets between folds cannot move it.
- macro: the mean over a fixed list of findings (NaN if any is undefined, so
  the set never changes silently inside a resample).
Labels: 1 present, 0 absent, NaN uncertain or masked (left out).
"""
from __future__ import annotations

import numpy as np


def auc(scores, y, w=None) -> tuple[float, float]:
    """(AUC, pair weight) of scores for binary labels y; ties count one half."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(s) if w is None else np.asarray(w, dtype=float)
    m = np.isfinite(s) & ~np.isnan(y) & (w > 0)
    s, y, w = s[m], y[m], w[m]
    wp, wn = w * (y == 1), w * (y == 0)
    P, N = wp.sum(), wn.sum()
    if P <= 0 or N <= 0:
        return float("nan"), 0.0
    order = np.argsort(s, kind="mergesort")
    s, wp, wn = s[order], wp[order], wn[order]
    _, start = np.unique(s, return_index=True)
    cp, cn = np.add.reduceat(wp, start), np.add.reduceat(wn, start)
    below = np.cumsum(cn) - cn  # negative weight strictly below each distinct score
    return float((cp * (below + 0.5 * cn)).sum() / (P * N)), float(P * N)


def stratified_auc(scores, y, strata, w=None) -> tuple[float, float]:
    """Pair-weighted AUC over strata (pairs only inside a stratum); strata < 0 are left out."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    st = np.asarray(strata)
    w = np.ones_like(s) if w is None else np.asarray(w, dtype=float)
    num = den = 0.0
    for k in np.unique(st):
        if isinstance(k, (int, np.integer)) and k < 0:
            continue
        m = st == k
        a, pw = auc(s[m], y[m], w[m])
        if pw > 0:
            num += a * pw
            den += pw
    return (num / den if den > 0 else float("nan")), den


def macro(scores, y, strata, findings, w=None) -> float:
    """Mean over the fixed findings (column indices) of the stratified AUC; NaN if one is undefined."""
    vals = []
    for j in findings:
        a, pw = stratified_auc(scores[:, j], y[:, j], strata, w)
        if not pw > 0:
            return float("nan")
        vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def macro_available(scores, y, findings) -> float:
    """Mean AUC over the findings that have both classes here (inner validation scoring only)."""
    vals = [auc(scores[:, j], y[:, j])[0] for j in findings]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def per_fold_macro_mean(scores, y, folds, findings) -> float:
    """Mean over folds of each fold's macro AUC (the participants' target metric)."""
    vals = []
    for k in np.unique(folds):
        if k < 0:
            continue
        m = folds == k
        v = macro(scores[m], y[m], np.zeros(int(m.sum()), int), findings)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def strata_codes(*cols) -> np.ndarray:
    """Integer stratum per study from several columns (any value < 0 or NaN in a column -> -1)."""
    import pandas as pd

    df = pd.DataFrame({i: np.asarray(c) for i, c in enumerate(cols)})
    bad = np.zeros(len(df), bool)
    for c in df.columns:
        col = df[c]
        bad |= col.isna().to_numpy()
        if pd.api.types.is_numeric_dtype(col):
            bad |= (col.fillna(-1) < 0).to_numpy()
    codes = df.astype(str).agg("|".join, axis=1).astype("category").cat.codes.to_numpy().astype(int)
    codes[bad] = -1
    return codes


def eligible(y, strata, min_pos: int = 5, min_neg: int = 5) -> list[int]:
    """Findings with at least min_pos positives and min_neg negatives in every stratum >= 0."""
    st = np.asarray(strata)
    keep = []
    for j in range(y.shape[1]):
        ok = True
        for k in np.unique(st[st >= 0]):
            col = y[st == k, j]
            if (col == 1).sum() < min_pos or (col == 0).sum() < min_neg:
                ok = False
                break
        if ok and (st >= 0).any():
            keep.append(j)
    return keep


# ---------------------------------------------------------------- secondary metrics (descriptive)

def average_precision(scores, y, w=None) -> tuple[float, float]:
    """(AP, positive weight). AP = the sum over distinct score thresholds, from the highest down, of
    precision times the step in recall (tied scores share one threshold): scikit-learn's definition."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(s) if w is None else np.asarray(w, dtype=float)
    m = np.isfinite(s) & ~np.isnan(y) & (w > 0)
    s, y, w = s[m], y[m], w[m]
    wp = w * (y == 1)
    P = wp.sum()
    if P <= 0:
        return float("nan"), 0.0
    order = np.argsort(-s, kind="mergesort")
    s, wp, w = s[order], wp[order], w[order]
    _, start = np.unique(-s, return_index=True)
    tp, flagged = np.add.reduceat(wp, start), np.add.reduceat(w, start)
    precision = np.cumsum(tp) / np.cumsum(flagged)
    return float((tp * precision).sum() / P), float(P)


def random_ap(P: float, n: float) -> float:
    """The expected AP of a random ranking (no ties) with P positives among n studies:
    (P-1)/(n-1) + (n-P)/(n-1) * H_n/n, H_n the n-th harmonic number. It is above the prevalence P/n,
    clearly so for rare findings (5 in 700: 0.016 against 0.007), so "times chance" must use this."""
    from scipy.special import digamma

    if n <= 1:
        return 1.0 if P > 0 else float("nan")
    H = float(digamma(n + 1.0) + np.euler_gamma)
    return float((P - 1) / (n - 1) + (n - P) / (n - 1) * H / n)


def _strata(strata):
    st = np.asarray(strata)
    return st, [k for k in np.unique(st) if not (isinstance(k, (int, np.integer)) and k < 0)]


def stratified_ap(scores, y, strata, w=None) -> tuple[float, float]:
    """(AP, positive weight) with scores ranked only inside a stratum (test fold).

    Each stratum's AP is weighted by its positives, so the result is the mean over all positives of
    the precision at that positive's rank inside its own fold; scores of different fold models are
    never compared. Strata < 0 are left out; a stratum without positives adds nothing."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    st, keys = _strata(strata)
    w = np.ones_like(s) if w is None else np.asarray(w, dtype=float)
    num = den = 0.0
    for k in keys:
        m = st == k
        a, P = average_precision(s[m], y[m], w[m])
        if P > 0:
            num += a * P
            den += P
    return (num / den if den > 0 else float("nan")), den


def stratified_ap_chance(y, strata) -> tuple[float, float]:
    """(chance, prevalence_weighted) for stratified_ap, from labels and strata only.

    chance: each stratum's random_ap, weighted by its positives exactly as the AP is.
    prevalence_weighted: the same weighting of each stratum's plain prevalence (a lower, cruder level)."""
    y = np.asarray(y, dtype=float)
    st, keys = _strata(strata)
    rand = prev = den = 0.0
    for k in keys:
        yk = y[(st == k) & ~np.isnan(y)]
        P, n = float((yk == 1).sum()), float(len(yk))
        if P > 0:
            rand += P * random_ap(P, n)
            prev += P * P / n
            den += P
    if den <= 0:
        return float("nan"), float("nan")
    return rand / den, prev / den


def macro_ap(scores, y, strata, findings, w=None) -> float:
    """Mean over the fixed findings of stratified_ap; NaN if one is undefined."""
    vals = [stratified_ap(scores[:, j], y[:, j], strata, w) for j in findings]
    if not vals or any(not v[1] > 0 for v in vals):
        return float("nan")
    return float(np.mean([v[0] for v in vals]))


def macro_ap_chance(y, strata, findings) -> tuple[float, float]:
    """(chance, prevalence_weighted): means over the fixed findings of stratified_ap_chance."""
    vals = [stratified_ap_chance(y[:, j], strata) for j in findings]
    if not vals or any(not np.isfinite(v[0]) for v in vals):
        return float("nan"), float("nan")
    return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))


def threshold_at_sensitivity(scores, positive, target: float = 0.95) -> float:
    """The highest threshold t for which at least `target` of the positives score >= t (NaN without positives)."""
    p = np.sort(np.asarray(scores, dtype=float)[np.asarray(positive, dtype=bool)])
    if not len(p):
        return float("nan")
    return float(p[int(np.floor((1 - target) * len(p) + 1e-9))])


def confusion(flagged, positive) -> dict:
    """Exact counts of a binary decision against binary labels (callers round them before reporting)."""
    f, p = np.asarray(flagged, dtype=bool), np.asarray(positive, dtype=bool)
    return {"tp": int((f & p).sum()), "fp": int((f & ~p).sum()), "fn": int((~f & p).sum()), "tn": int((~f & ~p).sum())}


def rates(c: dict, suppress_small: bool = False) -> dict:
    """Sensitivity, specificity, precision and F1 from confusion counts (None where undefined).

    suppress_small (for summaries): a rate is None when a cell it uses holds 1 to 4 studies, because a
    rate beside a cell shown as "<5" would give that small count back."""
    def div(a, b, cells):
        if suppress_small and any(1 <= c[k] <= 4 for k in cells):
            return None
        return a / b if b > 0 else None

    return {"sensitivity": div(c["tp"], c["tp"] + c["fn"], ("tp", "fn")),
            "specificity": div(c["tn"], c["tn"] + c["fp"], ("tn", "fp")),
            "precision": div(c["tp"], c["tp"] + c["fp"], ("tp", "fp")),
            "f1": div(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"], ("tp", "fp", "fn"))}
