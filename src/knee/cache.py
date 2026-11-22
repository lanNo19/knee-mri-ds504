"""Image caches: one uint8 memmap [studies, 32, 224, 224] per plane (plan, Storage).

Rows follow the study order of the study table (sorted UIDs). Alongside the
memmap, in the private output folder: a manifest (sha256 of the memmap,
shape, dtype, plane), a per-study table (row, outcome, per-study sha256),
and the summary returned to N4 (counts and shares only).
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from . import agg, dicom_io, safe_log

SHAPE_TAIL = (dicom_io.N_SLICES, dicom_io.SIZE, dicom_io.SIZE)

_MM = None  # worker-local memmap


def _init_worker(mm_path: str, n: int):
    global _MM
    safe_log.worker_init()
    _MM = np.memmap(mm_path, dtype=np.uint8, mode="r+", shape=(n, *SHAPE_TAIL))


def _work(row: int, series_dir: str, plane: str, laterality: str | None) -> dict:
    try:
        res = dicom_io.prepare_volume(Path(series_dir), plane, laterality)
        _MM[row] = res.volume
        _MM.flush()
        return {
            "row": row, "ok": True, "order": res.order_source, "frame_ok": res.frame_ok,
            "decoders": ",".join(res.decoders), "blank": res.blank, "n_files": res.n_files,
            "mixed_iop": res.mixed_iop, "dup": res.duplicate_positions,
            "plane_mismatch": res.iop_plane_mismatch, "err": None,
            "mean": float(res.volume.mean()), "std": float(res.volume.std()),
            "sha256": hashlib.sha256(res.volume.tobytes()).hexdigest(),
        }
    except Exception as e:  # noqa: BLE001 - only the type is kept
        return {"row": row, "ok": False, "err": type(e).__name__}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build(study_table: pd.DataFrame, images_root: Path, plane: str, out: Path,
          workers: int = 4, limit: int | None = None, budget_s: float | None = None) -> dict:
    """study_table: rows in cache order with StudyInstanceUID, <plane>_series, laterality."""
    import time

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    tab = study_table.reset_index(drop=True)
    if limit:
        tab = tab.iloc[:limit]
    n = len(tab)
    mm_path = out / f"cache_{plane}.u8"
    mm = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(n, *SHAPE_TAIL))
    del mm
    jobs = []
    for i, r in tab.iterrows():
        se = r.get(f"{plane}_series")
        if se is None or (isinstance(se, float) and np.isnan(se)):
            continue
        lat = r.get("laterality")
        lat = lat if lat in ("L", "R") else None
        jobs.append((int(i), str(images_root / str(r["StudyInstanceUID"]) / str(se)), plane, lat))
    results = []
    stopped_early = 0
    window = max(2 * workers, 4)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(mm_path), n)) as ex:
        pending = set()
        it = iter(jobs)
        exhausted = False
        while pending or not exhausted:
            # Keep a small window in flight; stop taking new jobs at 0.75 x the budget.
            while not exhausted and len(pending) < window:
                j = next(it, None)
                if j is None:
                    exhausted = True
                    break
                if budget_s is not None and time.time() - t0 > 0.75 * budget_s:
                    stopped_early += 1
                    continue
                pending.add(ex.submit(_work, *j))
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for d in done:
                results.append(d.result())
    res = pd.DataFrame(results).sort_values("row") if results else pd.DataFrame(columns=["row", "ok", "err"])
    per = tab[["StudyInstanceUID"]].copy()
    per["row"] = np.arange(n)
    per = per.merge(res, on="row", how="left")
    per.to_csv(out / f"cache_{plane}_studies.csv", index=False)
    manifest = {"plane": plane, "shape": [n, *SHAPE_TAIL], "dtype": "uint8",
                "sha256": sha256_file(mm_path), "file": mm_path.name}
    (out / f"cache_{plane}_manifest.json").write_text(json.dumps(manifest, indent=1))
    ok = res["ok"].fillna(False).astype(bool) if len(res) else pd.Series([], dtype=bool)
    good = res[ok] if len(res) else res
    written = int(ok.sum())
    failed = int((~ok).sum())
    missing = n - len(jobs)
    summary = {
        "plane": plane,
        "studies": agg.count(n),
        "failed": agg.count(failed),
        "missing_series": agg.count(missing),
        "not_started_budget": agg.count(stopped_early),
        "error_types": agg.counts(res.loc[~ok, "err"].value_counts()) if len(res) else {},
        "blank_volumes": agg.count(int(good["blank"].sum())) if len(good) else 0,
        "order_source": agg.counts(good["order"].value_counts()) if len(good) else {},
        "missing_orientation": agg.count(int((~good["frame_ok"].astype(bool)).sum())) if len(good) else 0,
        "mixed_orientation_series": agg.count(int(good["mixed_iop"].astype(bool).sum())) if len(good) else 0,
        "series_with_duplicate_positions": agg.count(int((good["dup"] > 0).sum())) if len(good) else 0,
        "iop_plane_mismatch": agg.count(int(good["plane_mismatch"].astype(bool).sum())) if len(good) else 0,
        "decoders": agg.counts(good["decoders"].value_counts()) if len(good) else {},
        "files_per_series": agg.quantiles(good["n_files"]) if len(good) else {},
        "mean_intensity": agg.quantiles(good["mean"]) if len(good) else {},
        "manifest_sha256": manifest["sha256"][:16],
        "seconds": round(time.time() - t0, 1),
        "checks": {
            "shape_ok": written + failed + missing + stopped_early == n,
            "failures_under_1pct": failed + stopped_early <= 0.01 * max(n, 1),
            "blank_under_1pct": (int(good["blank"].sum()) if len(good) else 0) <= 0.01 * max(n, 1),
            "position_order_share": agg.share((good["order"] == "position").mean()) if len(good) else None,
        },
    }
    return summary
