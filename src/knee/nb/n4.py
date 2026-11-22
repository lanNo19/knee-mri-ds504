"""N4a/N4b/N4c: the image cache for one plane (CPU). Runs in parallel with N0.

It rebuilds the study table with the same code as N0 (series choice and
laterality), so it needs no N0 output; N5 asserts that all copies match.
Writes private/n4_<plane>/: the memmap, its manifest, the per-study table and
study_table.csv; and agg_n4_<plane>.json.
"""
from __future__ import annotations

import time

from .. import agg, cache, paths, study_table
from . import common


def run(plane: str, budget_h: float = 8.0, workers: int = 4, limit: int | None = None,
        quiet: bool = True) -> dict:
    assert plane in ("sagittal", "coronal", "axial")
    s: dict = {}
    t0 = time.time()
    with common.notebook(f"n4_{plane}", {"budget_h": budget_h, "limit": limit}, quiet=quiet, budget_h=budget_h):
        root = paths.competition_root()
        od = paths.out_dir("private", f"n4_{plane}")
        st, _, hdr_errors = study_table.build(root, "train", workers=workers, limit=limit)
        st.to_csv(od / "study_table.csv", index=False)
        remaining = budget_h * 3600 - (time.time() - t0)
        s = cache.build(st, root / "train_series", plane, od, workers=workers, budget_s=remaining)
        s["header_read_errors"] = agg.counts(hdr_errors)
        s["laterality_source"] = agg.counts(st["laterality_source"].value_counts())
        s["study_table_sha16"] = common.sha16(od / "study_table.csv")
        agg.write(f"n4_{plane}", s)
    common.summary({"plane": plane, "checks": s.get("checks"), "failed": s.get("failed"),
                    "manifest_sha256": s.get("manifest_sha256"), "study_table_sha16": s.get("study_table_sha16")})
    return s
