"""N3: folds (CPU). Input: N0's output (private/n0/studies.csv), attached as a notebook output.

Writes private/n3/folds.csv with, per study:
  s2_fold      outer fold on G-lang (main)
  vendor_fold  outer fold on G-vendor (sensitivity)
  s1_fold      random fold with the same sizes as S2 (seen sites)
  s2_inner_k / vendor_inner_k   inner fold within outer fold k, -1 when not in that training set
  train, scored, is_gold
and agg_n3.json with fold sizes and group counts, what happened to the flagged pairs
(pairs.*: applied, together, across folds) and s2_sha16 (the file's hash without s1_fold).

After the pilot: flagged near-duplicate pairs come from the pilot's private
pairs.csv (pair_id, uid_a, uid_b). Ablay confirms pairs by looking at their
images in his own browser and passes only their pair_ids in cell 2
(confirmed_pair_ids), never UIDs. All flagged pairs stay together in S1;
confirmed pairs merge their G-lang groups before the folds are re-frozen.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import agg, folds, paths
from . import common

UID = "StudyInstanceUID"


def _merge_groups(st: pd.DataFrame, confirmed: list[tuple[str, str]]) -> int:
    """Merge the G-lang (and G-vendor) groups of confirmed same-knee pairs. Returns merges made."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    made = 0
    for a, b in confirmed:
        if a in st.index and b in st.index:
            ga, gb = st.at[a, "g_lang"], st.at[b, "g_lang"]
            if pd.notna(ga) and pd.notna(gb) and find(ga) != find(gb):
                ra, rb = sorted((find(ga), find(gb)))
                parent[rb] = ra
                made += 1
    if made:
        st["g_lang"] = st["g_lang"].map(lambda g: find(g) if pd.notna(g) else g)
        st["g_vendor"] = [
            (f"{g}|{str(v).split('|', 1)[1]}" if pd.notna(g) and pd.notna(v) and "|" in str(v) else v)
            for g, v in zip(st["g_lang"], st["g_vendor"])
        ]
    return made


def pair_facts(out: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict:
    """What the flagged pairs did to the folds (counts rounded as usual).

    applied: both studies are training studies with an S2 fold, so the pair shaped the S1 folds
        (folds.seen_site_folds silently skips any other pair; those are counted as not_applied).
    together: every applied pair shares an S1 fold (None when no pair was applied).
    across_vendor_folds / across_s2_folds: pairs, with both studies in the table, whose folds differ
        (such a pair sits on both sides of a train-test split under that grouping)."""
    known = [(a, b) for a, b in pairs if a in out.index and b in out.index]
    ok = out["train"] & (out["s2_fold"] >= 0)
    applied = [(a, b) for a, b in known if ok[a] and ok[b]]
    return {"pairs_applied": agg.count(len(applied)),
            "pairs_not_applied": agg.count(len(pairs) - len(applied)),
            "pairs_together": all(out.at[a, "s1_fold"] == out.at[b, "s1_fold"] for a, b in applied) if applied else None,
            "pairs_across_vendor_folds": agg.count(sum(out.at[a, "vendor_fold"] != out.at[b, "vendor_fold"] for a, b in known)),
            "pairs_across_s2_folds": agg.count(sum(out.at[a, "s2_fold"] != out.at[b, "s2_fold"] for a, b in known))}


def run(studies_csv: str | Path | None = None, expect_n0: str | None = None,
        confirmed_pair_ids: list[int] | None = None, quiet: bool = True) -> dict:
    """expect_n0: the first characters of studies.csv's sha16 from N0's summary (optional check)."""
    s: dict = {}
    with common.notebook("n3", {"confirmed_pairs": len(confirmed_pair_ids or [])}, quiet=quiet):
        path, up = common.upstream("studies.csv", studies_csv)
        common.expect(up, expect_n0)
        s["upstream_n0"] = up
        st = pd.read_csv(path).set_index(UID)
        pairs: list[tuple[str, str]] = []
        confirmed: list[tuple[str, str]] = []
        pp = paths.find_file("pairs.csv")
        if pp is not None:
            pdf = pd.read_csv(pp)
            pairs = list(zip(pdf["uid_a"].astype(str), pdf["uid_b"].astype(str)))
            ids = set(confirmed_pair_ids or [])
            confirmed = [(a, b) for pid, a, b in zip(pdf["pair_id"], pdf["uid_a"].astype(str), pdf["uid_b"].astype(str)) if int(pid) in ids]
        s["pairs"] = {"flagged": agg.count(len(pairs)), "confirmed": agg.count(len(confirmed)),
                      "group_merges": agg.count(_merge_groups(st, confirmed))}
        od = paths.out_dir("private", "n3")
        tr = st[st["in_training"].astype(bool)]
        out = pd.DataFrame(index=st.index)
        out["train"] = st["in_training"].astype(bool)
        out["scored"] = st["scored"].astype(bool) & st["has_report"].astype(bool)
        out["is_gold"] = st["is_gold"].astype(bool)
        gold_idx = st.index[st["is_gold"].astype(bool)]
        for name, col in (("s2", "g_lang"), ("vendor", "g_vendor")):
            f, info = folds.outer_folds(tr[col])
            out[f"{name}_fold"] = -1
            out.loc[f.index, f"{name}_fold"] = f
            g2f = pd.Series(f.values, index=tr.loc[f.index, col].values).groupby(level=0).first()
            out.loc[gold_idx, f"{name}_fold"] = st.loc[gold_idx, col].map(g2f).fillna(-1).astype(int).values
            s[name] = {**info, "folds": folds.fold_summary(f, tr.loc[f.index, col]),
                       "gold_unassigned": agg.count(int((out.loc[gold_idx, f"{name}_fold"] < 0).sum()))}
            k_out = int(f.max() + 1)
            inner_k = []
            for k in range(k_out):
                mask = pd.Series(False, index=tr.index)
                mask[f.index[f != k]] = True
                inner = folds.inner_folds(tr[col], mask)
                out[f"{name}_inner_{k}"] = -1
                out.loc[inner.index, f"{name}_inner_{k}"] = inner
                inner_k.append(int(inner.max() + 1))
            s[name]["inner_k"] = inner_k
        s2 = out.loc[out["train"] & (out["s2_fold"] >= 0), "s2_fold"]
        s1 = folds.seen_site_folds(s2, pairs)
        out["s1_fold"] = -1
        out.loc[s1.index, "s1_fold"] = s1
        s["s1"] = {"sizes_match_s2": bool((s1.value_counts().sort_index() == s2.value_counts().sort_index()).all())}
        s["train_studies"] = agg.count(int(out["train"].sum()))
        s["scored_train_studies"] = agg.count(int((out["train"] & out["scored"]).sum()))
        s["pairs"].update(pair_facts(out, pairs))
        out.reset_index().to_csv(od / "folds.csv", index=False)
        s["folds_sha16"] = common.sha16(od / "folds.csv")
        s["s2_sha16"] = folds.s2_sha16(od / "folds.csv")  # unchanged by a rerun that only regroups S1
        agg.write("n3", s)
    common.summary({"s2": {k: s["s2"][k] for k in ("scheme", "n_groups", "n_folds")},
                    "vendor": {k: s["vendor"][k] for k in ("scheme", "n_groups", "n_folds")},
                    "upstream_n0": s["upstream_n0"], "folds_sha16": s["folds_sha16"],
                    "s2_sha16": s["s2_sha16"], "pairs": s["pairs"]})
    return s
