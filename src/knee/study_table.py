"""One row per study: chosen series per plane, laterality and scanner fingerprint.

N0 and every cache notebook (N4a-c) build this table with the same code, so
the caches can run in parallel with N0. N5 later asserts that all copies are
identical. Rows are sorted by StudyInstanceUID; that order is the cache row
order.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import dicom_io, safe_log, series

UID = "StudyInstanceUID"
GEOMETRY_MARGIN_MM = 20.0


def _info(args):
    study, plane, path = args
    try:
        return study, plane, dicom_io.series_info(Path(path)), None
    except Exception as e:  # noqa: BLE001
        return study, plane, None, type(e).__name__


def build(root: Path, split: str = "train", workers: int = 4, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (study table, chosen-series table with header facts, errors)."""
    ser = pd.read_csv(root / f"{split}_series.csv")
    if limit:
        keep = sorted(ser[UID].unique())[:limit]
        ser = ser[ser[UID].isin(keep)]
    images = root / f"{split}_series"
    ser = series.add_slice_counts(ser, images)
    chosen = series.choose(ser)
    jobs = [(r[UID], r["plane"], str(images / str(r[UID]) / str(r["SeriesInstanceUID"]))) for _, r in chosen.iterrows()]
    infos, errors = {}, Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=safe_log.worker_init) as ex:
        for study, plane, info, err in ex.map(_info, jobs, chunksize=64):
            if err:
                errors[err] += 1
            infos[(study, plane)] = info
    rows = []
    for _, r in chosen.iterrows():
        info = infos.get((r[UID], r["plane"]))
        rows.append({
            UID: r[UID], "plane": r["plane"], "SeriesInstanceUID": r["SeriesInstanceUID"],
            "n_slices": r["n_slices"], "tier": r["tier"],
            "transfer_syntax": getattr(info, "transfer_syntax", None),
            "manufacturer": getattr(info, "manufacturer", None),
            "model": getattr(info, "model", None),
            "field_strength": getattr(info, "field_strength", None),
            "laterality_tag": getattr(info, "laterality_tag", None),
            "has_ipp": getattr(info, "has_ipp", False),
            "has_iop": getattr(info, "has_iop", False),
            "center_x": getattr(info, "center_x", None),
            "plane_from_iop": getattr(info, "plane_from_iop", None),
        })
    ch = pd.DataFrame(rows)
    studies = sorted(ser[UID].unique())
    out = pd.DataFrame({UID: studies})
    for p in series.PLANES:
        sub = ch[ch["plane"] == p].set_index(UID)
        out[f"{p}_series"] = out[UID].map(sub["SeriesInstanceUID"])
        out[f"{p}_slices"] = out[UID].map(sub["n_slices"])
        out[f"{p}_tier"] = out[UID].map(sub["tier"])
    lat, src = [], []
    for st in studies:
        sub = ch[ch[UID] == st]
        tags = [t for t in sub["laterality_tag"] if t in ("R", "L")]
        geo = None
        for p in ("coronal", "axial"):
            cx = sub.loc[sub["plane"] == p, "center_x"]
            if len(cx) and pd.notna(cx.iloc[0]):
                geo = dicom_io.geometry_laterality(float(cx.iloc[0]), GEOMETRY_MARGIN_MM)
                if geo:
                    break
        if tags and len(set(tags)) == 1:
            lat.append(tags[0]); src.append("tag")
        elif geo:
            lat.append(geo); src.append("geometry" if not tags else "geometry_tag_conflict")
        else:
            lat.append(None); src.append("unresolved")
        # agreement between tag and geometry is summarised in N0
    out["laterality"] = lat
    out["laterality_source"] = src
    # Fingerprint from the sagittal series, else coronal, else axial
    fp, man = [], []
    for st in studies:
        sub = ch[ch[UID] == st].set_index("plane")
        val = None
        for p in series.PLANES:
            if p in sub.index:
                r = sub.loc[p]
                parts = [r["manufacturer"], r["model"], r["field_strength"]]
                if any(pd.notna(x) and x is not None for x in parts):
                    val = "|".join("" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in parts)
                    man.append(r["manufacturer"] if pd.notna(r["manufacturer"]) else None)
                    break
        if val is None:
            man.append(None)
        fp.append(val)
    out["fingerprint"] = fp
    out["manufacturer"] = man
    return out, ch, dict(errors)
