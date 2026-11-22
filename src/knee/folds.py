"""Folds (plan, "Splits").

S2 (main): GroupKFold on G-lang, balanced by study count. 6 or more groups:
5 folds; 4-5 groups: leave one group out; fewer than 4: H1 and H2 are "not
testable" (folds are still written, flagged).
G-vendor: the same procedure on G-vendor groups.
S1 (seen sites): random folds with the same sizes as the S2 folds; flagged
near-duplicate pairs stay together.
Inner folds: GroupKFold with k = min(4, groups in the outer training set).
Gold studies are never trained on; each is scored by the fold whose test
fold holds its group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 20261122


def outer_folds(groups: pd.Series) -> tuple[pd.Series, dict]:
    """groups: index UID -> group label (training studies only)."""
    g = groups.dropna()
    n_groups = g.nunique()
    if n_groups >= 6:
        k, scheme = 5, "group_5fold"
    elif n_groups >= 4:
        k, scheme = n_groups, "leave_one_group_out"
    else:
        k, scheme = max(n_groups, 2), "not_testable"
    fold = pd.Series(-1, index=g.index, dtype=int)
    if n_groups >= 2:
        gkf = GroupKFold(n_splits=min(k, n_groups))
        for j, (_, te) in enumerate(gkf.split(np.zeros(len(g)), groups=g.to_numpy())):
            fold.iloc[te] = j
    info = {"scheme": scheme, "n_groups": int(n_groups), "n_folds": int(fold.max() + 1)}
    return fold, info


def seen_site_folds(s2_fold: pd.Series, pairs: list[tuple[str, str]] | None = None, seed: int = SEED) -> pd.Series:
    """Random folds with the S2 fold sizes; listed pairs share a fold.

    Pairs are placed first, then single studies in random order, each into the
    fold with the most room left, so the sizes match S2 exactly unless pairs
    alone overflow a fold.
    """
    rng = np.random.default_rng(seed)
    uids = list(s2_fold.index)
    parent = {u: u for u in uids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs or []:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    units: dict[str, list[str]] = {}
    for u in uids:
        units.setdefault(find(u), []).append(u)
    multi = [v for v in units.values() if len(v) > 1]
    single = [v for v in units.values() if len(v) == 1]
    rng.shuffle(multi)
    rng.shuffle(single)
    remaining = s2_fold.value_counts().sort_index().to_dict()
    out = {}
    for unit in multi + single:
        j = max(sorted(remaining), key=lambda f: remaining[f])
        for u in unit:
            out[u] = j
        remaining[j] -= len(unit)
    return pd.Series(out).reindex(s2_fold.index).astype(int)


def inner_folds(groups: pd.Series, train_mask: pd.Series, max_k: int = 4) -> pd.Series:
    """Inner GroupKFold labels for the training part of one outer fold (-1 elsewhere)."""
    g = groups[train_mask].dropna()
    k = min(max_k, g.nunique())
    out = pd.Series(-1, index=groups.index, dtype=int)
    if k >= 2:
        for j, (_, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(g)), groups=g.to_numpy())):
            out.loc[g.index[te]] = j
    return out


def fold_summary(fold: pd.Series, groups: pd.Series) -> list[dict]:
    from . import agg

    rows = []
    for j in sorted(fold.unique()):
        if j < 0:
            continue
        m = fold == j
        rows.append({"fold": int(j), "studies": agg.count(int(m.sum())), "n_groups": int(groups[m].nunique())})
    return rows
