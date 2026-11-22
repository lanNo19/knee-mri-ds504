"""Image caches: one uint8 memmap [studies, 32, 224, 224] per plane (plan, Storage).

Rows follow the study order of N0's study table. Alongside the memmap:
a manifest (sha256 of the memmap, shape, dtype, plane, commit), a per-study
hash table (sha256 of each study's block), per-study image statistics, and
agg_n4<plane>.json with the automatic checks (shape, blank volumes, slice-order
source, orientation tags) as counts only.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from . import dicom_io, safe_log

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
            "mixed_iop": res.mixed_iop, "err": None,
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
        jobs.append((int(i), str(images_root / str(r["StudyInstanceUID"]) / str(se)), plane, r.get("laterality")))
    results = []
    stopped_early = 0
    window = max(2 * workers, 4)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(mm_path), n)) as ex:
        pending = set()
        it = iter(jobs)
        exhausted = False
        while pending or not exhausted:
            # Keep a small window of jobs in flight, and stop taking new ones
            # at 0.75 x the time budget so the notebook ends normally.
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
    res = pd.DataFrame(results).sort_values("row") if results else pd.DataFrame(columns=["row", "ok"])
    # Per-study files (private outputs; UIDs never printed)
    per = tab[["StudyInstanceUID"]].copy()
    per["row"] = np.arange(n)
    per = per.merge(res, on="row", how="left")
    per.to_csv(out / f"cache_{plane}_studies.csv", index=False)
    manifest = {
        "plane": plane, "shape": [n, *SHAPE_TAIL], "dtype": "uint8",
        "sha256": sha256_file(mm_path), "file": mm_path.name,
    }
    (out / f"cache_{plane}_manifest.json").write_text(json.dumps(manifest, indent=1))
    ok = res["ok"].fillna(False).astype(bool) if len(res) else pd.Series([], dtype=bool)
    summary = {
        "plane": plane,
        "studies": n,
        "jobs": len(jobs),
        "written": int(ok.sum()),
        "failed": int((~ok).sum()),
        "missing_series": int(n - len(jobs)),
        "not_started_budget": int(stopped_early),
        "error_types": res.loc[~ok, "err"].value_counts().to_dict() if len(res) else {},
        "blank_volumes": int(res.loc[ok, "blank"].sum()) if len(res) else 0,
        "order_source": res.loc[ok, "order"].value_counts().to_dict() if len(res) else {},
        "missing_orientation": int((~res.loc[ok, "frame_ok"].astype(bool)).sum()) if len(res) else 0,
        "mixed_orientation_series": int(res.loc[ok, "mixed_iop"].astype(bool).sum()) if len(res) else 0,
        "decoders": res.loc[ok, "decoders"].value_counts().to_dict() if len(res) else {},
        "files_per_series": _quantiles(res.loc[ok, "n_files"]) if len(res) else {},
        "mean_intensity": _quantiles(res.loc[ok, "mean"]) if len(res) else {},
        "manifest_sha256": manifest["sha256"][:16],
        "seconds": round(time.time() - t0, 1),
    }
    return summary


def _quantiles(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([0.0, 0.05, 0.5, 0.95, 1.0])
    return {k: round(float(v), 2) for k, v in zip(["min", "p5", "median", "p95", "max"], q)}
