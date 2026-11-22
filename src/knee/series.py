"""Series choice: one series per plane per study (plan, Image preparation).

Rule: fluid-sensitive and fat-suppressed; else fluid-sensitive; else any
series of that plane. Ties: slice count closest to 30; a remaining tie goes
to the larger count (more coverage), then to the series ID order, so the
choice never depends on file-system order.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PLANES = ("sagittal", "coronal", "axial")
TARGET_SLICES = 30


def truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float)):
        return v > 0
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def norm_plane(v) -> str | None:
    """Plane name from labels such as "Sagittal", "SAG", "Coronal", "COR", "Axial", "AX", "Transverse"."""
    s = str(v).strip().lower()
    if s.startswith("sag"):
        return "sagittal"
    if s.startswith("cor"):
        return "coronal"
    if s.startswith("ax") or s.startswith("tra"):
        return "axial"
    return None


def count_slices(series_dir: Path) -> int:
    try:
        return sum(1 for e in os.scandir(series_dir) if e.is_file() and e.name.lower().endswith(".dcm"))
    except FileNotFoundError:
        return 0


def add_slice_counts(series: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    """Add n_slices (files per series folder) to the series table."""
    s = series.copy()
    s["n_slices"] = [
        count_slices(images_root / str(st) / str(se))
        for st, se in zip(s["StudyInstanceUID"], s["SeriesInstanceUID"])
    ]
    return s


def choose(series: pd.DataFrame) -> pd.DataFrame:
    """One row per (study, plane) with the chosen series and how it was chosen.

    Input columns: StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive,
    Fat_Suppression, Anatomical_Plane, n_slices.
    Output columns: StudyInstanceUID, plane, SeriesInstanceUID, n_slices, tier
    (2 = fluid-sensitive and fat-suppressed, 1 = fluid-sensitive, 0 = any).
    """
    s = series.copy()
    s["plane"] = s["Anatomical_Plane"].map(norm_plane)
    s = s[s["plane"].notna() & (s["n_slices"] > 0)]
    fs = s["Fluid_Sensitive"].map(truthy)
    fat = s["Fat_Suppression"].map(truthy)
    s["tier"] = (fs & fat).astype(int) * 2 + (fs & ~fat).astype(int)
    s["dist"] = (s["n_slices"] - TARGET_SLICES).abs()
    s = s.sort_values(
        ["StudyInstanceUID", "plane", "tier", "dist", "n_slices", "SeriesInstanceUID"],
        ascending=[True, True, False, True, False, True],
        kind="mergesort",
    )
    out = s.groupby(["StudyInstanceUID", "plane"], sort=True).head(1)
    return out[["StudyInstanceUID", "plane", "SeriesInstanceUID", "n_slices", "tier"]].reset_index(drop=True)
