"""N4a/N4b/N4c: the image cache for one plane (CPU). Runs in parallel with N0.

It rebuilds the study table with the same code as N0 (series choice and
laterality), so it needs no N0 output; N5 asserts that all copies match.
Writes n4_<plane>/cache_<plane>.u8 (memmap), its manifest, the per-study
table, study_table.csv, and agg_n4_<plane>.json.
"""
from __future__ import annotations

import time

from .. import agg, cache, paths, safe_log, study_table


def run(plane: str, budget_h: float = 8.0, workers: int = 4, limit: int | None = None,
        quiet: bool = True) -> dict:
    assert plane in ("sagittal", "coronal", "axial")
    summary: dict = {}
    t0 = time.time()
    with safe_log.step(f"n4_{plane}", quiet=quiet):
        root = paths.competition_root()
        od = paths.out_dir(f"n4_{plane}")
        st, _, hdr_errors = study_table.build(root, "train", workers=workers, limit=limit)
        st.to_csv(od / "study_table.csv", index=False)
        remaining = budget_h * 3600 - (time.time() - t0)
        summary = cache.build(st, root / "train_series", plane, od, workers=workers, budget_s=remaining)
        summary["header_read_errors"] = agg.counts(hdr_errors)
        summary["laterality_source"] = agg.counts(st["laterality_source"].value_counts())
        summary["checks"] = {
            "shape_ok": summary["written"] + summary["failed"] + summary["missing_series"] + summary["not_started_budget"] == summary["studies"],
            "no_failures": summary["failed"] == 0 and summary["not_started_budget"] == 0,
            "blank_under_1pct": summary["blank_volumes"] <= 0.01 * max(summary["studies"], 1),
            "position_order_share": round(summary["order_source"].get("position", 0) / max(summary["written"], 1), 4),
        }
        for k in ("blank_volumes", "failed", "missing_series", "missing_orientation", "mixed_orientation_series"):
            summary[k] = agg.count(summary[k])
        summary["error_types"] = agg.counts(summary["error_types"])
        summary["order_source"] = agg.counts(summary["order_source"])
        summary["decoders"] = agg.counts(summary["decoders"])
        agg.write("run", agg.run_info(f"n4_{plane}", {"budget_h": budget_h, "limit": limit}))
        agg.write(f"n4_{plane}", summary)
    print({"plane": plane, "checks": summary.get("checks"), "written": summary.get("written")})
    return summary
