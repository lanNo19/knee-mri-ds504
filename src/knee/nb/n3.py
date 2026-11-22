"""N3: folds (CPU). Input: N0's output (n0/studies.csv), attached as a notebook output.

Writes n3/folds.csv (private) with, per study:
  s2_fold      outer fold on G-lang (main)
  vendor_fold  outer fold on G-vendor (sensitivity)
  s1_fold      random fold with the same sizes as S2 (seen sites)
  s2_inner_k   inner fold within outer fold k (k = 0..K-1), -1 when not in that training set
  train        True if the study may be trained on (not gold, has labels)
  scored       True if the study is scored (sure language ID)
and agg_n3.json with fold sizes and group counts.
"""
from __future__ import annotations

import pandas as pd

from .. import agg, folds, paths, safe_log

UID = "StudyInstanceUID"


def _load_studies() -> pd.DataFrame:
    p = paths.find_file("studies.csv")
    if p is None:
        raise FileNotFoundError("N0 output not attached")
    return pd.read_csv(p).set_index(UID)


def run(pairs: list[tuple[str, str]] | None = None, quiet: bool = True) -> dict:
    """pairs: confirmed same-knee pairs from the pilot (empty at first)."""
    summary: dict = {}
    with safe_log.step("n3", quiet=quiet):
        st = _load_studies()
        od = paths.out_dir("n3")
        tr = st[st["in_training"].astype(bool)]
        out = pd.DataFrame(index=st.index)
        out["train"] = st["in_training"].astype(bool)
        out["scored"] = st["scored"].astype(bool) & st["has_report"].astype(bool)
        out["is_gold"] = st["is_gold"].astype(bool)
        for name, col in (("s2", "g_lang"), ("vendor", "g_vendor")):
            f, info = folds.outer_folds(tr[col])
            out[f"{name}_fold"] = -1
            out.loc[f.index, f"{name}_fold"] = f
            # Gold studies: the fold whose test fold holds their group
            g2f = pd.Series(f.values, index=tr.loc[f.index, col].values).groupby(level=0).first()
            gold_idx = st.index[st["is_gold"].astype(bool)]
            out.loc[gold_idx, f"{name}_fold"] = st.loc[gold_idx, col].map(g2f).fillna(-1).astype(int).values
            summary[name] = {**info, "folds": folds.fold_summary(f, tr.loc[f.index, col]),
                             "gold_unassigned": agg.count(int((out.loc[gold_idx, f"{name}_fold"] < 0).sum()))}
            k_out = int(f.max() + 1)
            for k in range(k_out):
                mask = pd.Series(False, index=tr.index)
                mask[f.index[f != k]] = True
                inner = folds.inner_folds(tr[col], mask)
                out[f"{name}_inner_{k}"] = -1
                out.loc[inner.index, f"{name}_inner_{k}"] = inner
            summary[name]["inner_k"] = [int(out[f"{name}_inner_{k}"].max() + 1) for k in range(k_out)]
        s2 = out.loc[out["train"] & (out["s2_fold"] >= 0), "s2_fold"]
        s1 = folds.seen_site_folds(s2, pairs)
        out["s1_fold"] = -1
        out.loc[s1.index, "s1_fold"] = s1
        summary["s1"] = {"sizes_match_s2": bool((s1.value_counts().sort_index() == s2.value_counts().sort_index()).all()),
                         "pairs_kept_together": len(pairs or [])}
        summary["train_studies"] = agg.count(int(out["train"].sum()))
        summary["scored_train_studies"] = agg.count(int((out["train"] & out["scored"]).sum()))
        out.reset_index().to_csv(od / "folds.csv", index=False)
        agg.write("run", agg.run_info("n3", {"pairs": len(pairs or [])}))
        agg.write("n3", summary)
    print({"s2": {k: summary["s2"][k] for k in ("scheme", "n_groups", "n_folds")},
           "vendor": {k: summary["vendor"][k] for k in ("scheme", "n_groups", "n_folds")}})
    return summary
