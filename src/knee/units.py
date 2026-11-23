"""Stage A unit folders (private/<name>/units/<encoder>_f<k>/): which ones count, the chain
guard, and the default-head settings recorded in each unit's own logs.

Chain guard. A unit stores its predictions with the UID, its S2 fold and the gold flag of
every study it predicted (n6._unit: the test fold and that fold's gold studies). The guard
compares those sets with the attached folds.csv.
  Guaranteed when it passes: each unit's test studies and gold studies are exactly S2 fold k
  of the attached folds.csv. With every fold present, each unit's outer training set (all
  other folds) is then the same set of studies too, given the same N0 and N5 files (their
  hashes are checked by `expect`).
  Not guaranteed: that the inner folds are the same (units do not store them; they only
  steer which config was chosen, never what a test score means), and that Stage A read the
  very same folds.csv file. Its recorded N3 hash differs by design when only the seen-site
  column (s1_fold) was rewritten; `folds.s2_sha16` is the hash that must stay the same.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import heads, paths
from .labels import UID
from .nb import common


def unit_dirs(root: Path | None = None) -> dict:
    """{(encoder, fold): folder} of every attached unit with predictions (a rerun replaces the original)."""
    seen: dict = {}
    for p in paths.glob_outputs("private/*/units/*_f*/predictions.csv", root):
        enc, fold = p.parent.name.rsplit("_f", 1)
        key = (enc, int(fold))
        if key in seen and not common.is_rerun(p.parent):
            continue
        seen[key] = p.parent
    return dict(sorted(seen.items()))


def check_units(d, dirs: dict) -> dict:
    """The chain guard: every unit's test and gold studies equal S2 fold k of the attached folds.csv."""
    from . import agg

    row_of = {u: i for i, u in enumerate(d.uids)}
    bad, how = [], {}
    for (enc, k), ud in dirs.items():
        df = pd.read_csv(ud / "predictions.csv", usecols=[UID, "s2_fold", "is_gold"])
        rows = df[UID].astype(str).map(row_of)
        known = rows.notna().to_numpy()
        have = set(rows[known].astype(int).tolist())
        gold = df["is_gold"].astype(bool).to_numpy()
        want = set(d.split(k)[1].tolist()) | set(d.gold_rows(k).tolist())
        ok = bool(known.all() and (df["s2_fold"] == k).all() and have == want
                  and set(rows[known & gold].astype(int).tolist()) == set(d.gold_rows(k).tolist()))
        if not ok:
            bad.append(f"{enc}_f{k}")
            # how it differs, as rounded counts: studies of fold k the unit lacks, and studies it has beyond fold k
            how[f"{enc}_f{k}"] = {"missing": agg.count(len(want - have)),
                                  "extra": agg.count(len(have - want) + int((~known).sum()))}
    # no unit at all is "nothing checked", never a pass
    return {"s2_folds_match_units": (not bad) if dirs else None, "units_checked": agg.count(len(dirs)),
            "mismatched_units": bad, "mismatch_counts": how}


class ChainError(AssertionError):
    """Stage A units were trained with other S2 folds than the attached folds.csv."""


def chain_summary(d, dirs: dict, root: Path | None = None) -> dict:
    """The chain block of a summary: the guard, the N3 hashes used here and those Stage A recorded."""
    from . import folds

    out = check_units(d, dirs)
    out["n3_folds_sha16"] = d.upstream["n3_folds"]["sha16"]
    out["s2_sha16"] = folds.s2_sha16(d.folds_path)
    rec = {}
    for name in sorted({ud.parent.parent.name for ud in dirs.values()}):
        p = paths.find_file(f"agg_{name}.json", root)
        if p is not None:
            rec[name] = ((json.loads(p.read_text()).get("upstream") or {}).get("n3_folds") or {}).get("sha16")
    out["stage_a_n3_folds_sha16"] = rec
    # False is allowed: a rerun of N3 that changed only s1_fold changes the file hash, not the S2 folds
    out["n3_hash_same_as_stage_a"] = {k: (v == out["n3_folds_sha16"]) if v else None for k, v in rec.items()}
    out["note"] = "n3_hash_same_as_stage_a may be false by design: an N3 rerun that only regroups s1_fold keeps s2_sha16"
    return out


def _epochs(values) -> int:
    """E as n6._unit sets it: the median best epoch of the inner fits, at least 1."""
    return max(1, int(np.median(values)))


def default_specs(unit_dir: Path, U: float) -> dict:
    """Default-head settings a unit's own logs support: {name: (config, E, source)}.

    "l2_defaults": the unit's L1 winner (pooling, planes) with the L2 defaults; E from BO trial 0
        (the enqueued defaults), else from the winner's row of l1.csv (the same config and seeds).
    "fixed": gated attention over all three planes with the L2 defaults (the pilot's head, no
        selection of any kind); E from that config's row of l1.csv. Absent if the grid lacked it.
    Only inner-fold results are read: no test score exists for either config until it is refit.
    """
    unit_dir = Path(unit_dir)
    win = json.loads((unit_dir / "config.json").read_text())["l1_winner"]
    l1 = pd.read_csv(unit_dir / "l1.csv") if (unit_dir / "l1.csv").exists() else pd.DataFrame()
    out = {}

    def l1_epochs(pooling, planes):
        if l1.empty:
            return None
        hit = l1[(l1["pooling"] == pooling) & (l1["planes"] == planes)]
        return _epochs([float(hit["median_best_epoch"].iloc[0])]) if len(hit) else None

    E, source = None, None
    tp = unit_dir / "trials.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        t0 = t[(t["number"] == 0) & (t["state"] == "COMPLETE")]
        if len(t0) and np.isclose(float(t0["lr"].iloc[0]), heads.Config().lr):
            eps = json.loads(str(t0["best_epochs"].iloc[0]))
            if eps:
                E, source = _epochs(eps), "BO trial 0"
    if E is None:
        E, source = l1_epochs(win["pooling"], win["planes"]), "l1.csv"
    if E is not None:
        out["l2_defaults"] = (heads.default_config(U, win["pooling"], win["planes"]), E, source)
    Ef = l1_epochs("gated", "all")
    if Ef is not None:
        out["fixed"] = (heads.default_config(U), Ef, "l1.csv")
    return out
