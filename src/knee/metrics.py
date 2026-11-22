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
