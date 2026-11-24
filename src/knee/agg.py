"""Summary files: the only files that leave Kaggle or reach Claude.

agg_<name>.json holds counts, shares, scores and pass/fail flags only.
Rules, enforced by write() before anything is saved:
1. Counts of studies go through count(): 1-4 appear as "<5" (zero stays 0,
   because it identifies nobody), and larger counts are rounded to the
   nearest 5, so a small count cannot be worked back out from a total minus
   the other categories.
2. No other integer from 1 to 4 may appear, except as the direct value of a
   structural key (fold numbers, number of groups, seeds and the like) or
   anywhere under "config" (cell-2 parameters), so a raw small count cannot
   slip through.
3. No string may look like a DICOM UID (digits with four or more dots) or be
   longer than 200 characters (report text).
Shares are rounded to 2 decimals and distributions are reported as p1-p99,
never min or max, because extremes and fine shares give single studies back.
run.json records versions, the repo commit, the seed and the config.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from . import paths

# Keys whose small integers are structure, not counts of studies.
STRUCTURAL = {
    "fold", "folds", "n_folds", "inner_k", "n_groups", "g_lang_groups", "g_vendor_groups",
    "languages", "language_merges", "seed", "limit", "n_columns", "tier", "k", "version",
    "workers", "repeat", "per_study", "planes", "n_tables", "budget_h", "n_tables_used",
    "min_support_per_language", "margin_mm", "inner_folds", "E", "epoch", "epochs",
    "seeds", "n_bins", "n_trials",
    "patience", "schedule_epochs",  # training settings (iteration 2), never counts of studies
}
UID_RE = re.compile(r"\d+(\.\d+){4,}")
MAX_STR = 200


class UnsafeSummary(ValueError):
    """A summary would carry a small count or a data-like string (message: the key path only)."""


def count(n) -> int | str:
    n = int(n)
    if n == 0:
        return 0
    if n < 5:
        return "<5"
    return int(5 * round(n / 5))


def counts(mapping) -> dict:
    """Apply count() to every value of a {key: int} mapping."""
    return {str(k): count(v) for k, v in dict(mapping).items()}


def share(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, 2)


def quantiles(values) -> dict:
    import pandas as pd

    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {k: round(float(v), 2) for k, v in zip(["p1", "p5", "median", "p95", "p99"], q)}


_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 _'()/.\-]{0,40}$")


def safe_columns(cols, uid: str = "StudyInstanceUID") -> dict:
    """Column names, only when they look like a real header.

    A headerless CSV turns its first row into column names, which can hold a
    UID, so names are written only if `uid` is among them and every name is
    short text with no dotted numbers.
    """
    names = [str(c) for c in cols]
    ok = uid in names and all(_SAFE_NAME.match(n) and not re.search(r"\d+\.\d+", n) for n in names)
    return {"header_ok": True, "columns": names[:60]} if ok else {"header_ok": False, "n_columns": len(names)}


def _check(o, path: str = "", ints: bool = True) -> None:
    """Refuse small ints (unless exempt) and data-like strings anywhere in the payload.

    A structural key exempts only its own int (or list of ints) value; dicts
    under it are still checked. Strings are always checked.
    """
    if isinstance(o, dict):
        for k, v in o.items():
            key = str(k)
            if UID_RE.search(key) or len(key) > MAX_STR:
                raise UnsafeSummary(f"unsafe key under {path or '/'}")
            exempt = key in STRUCTURAL and (
                isinstance(v, int) or (isinstance(v, list) and all(isinstance(x, int) for x in v)))
            _check(v, f"{path}/{key}", ints=ints and not exempt and key != "config")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            _check(v, f"{path}[{i}]", ints=ints)
    elif isinstance(o, bool) or o is None:
        return
    elif isinstance(o, int):
        if ints and 1 <= o <= 4:
            raise UnsafeSummary(f"small count at {path}")
    elif isinstance(o, float):
        return
    elif isinstance(o, str):
        if UID_RE.search(o) or len(o) > MAX_STR:
            raise UnsafeSummary(f"data-like string at {path}")


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def repo_commit() -> str | None:
    """Commit the package was installed from (pip writes it in direct_url.json)."""
    try:
        txt = metadata.distribution("knee").read_text("direct_url.json")
        if txt:
            return json.loads(txt).get("vcs_info", {}).get("commit_id")
    except Exception:  # noqa: BLE001
        return None
    return None


def run_info(step: str, config: dict | None = None, seed: int | None = None) -> dict:
    info: dict[str, Any] = {
        "step": step,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": repo_commit(),
        "seed": seed,
        "config": config or {},
        "versions": {
            p: _version(p)
            for p in (
                "knee", "numpy", "pandas", "pydicom", "scikit-learn", "scipy", "torch",
                "timm", "optuna", "lightgbm", "lingua-language-detector", "python-gdcm",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "dicomsdl", "pillow",
            )
        },
    }
    try:
        import torch  # type: ignore

        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        info["cuda"] = None
        info["gpus"] = []
    return info


def write(name: str, payload: dict, folder: Path | None = None) -> Path:
    """Write agg_<name>.json (or run.json when name == 'run') after the safety checks."""
    text = json.dumps(payload, indent=1, sort_keys=False, default=_json_default)
    _check(json.loads(text))
    folder = Path(folder or paths.WORKING)
    folder.mkdir(parents=True, exist_ok=True)
    fname = "run.json" if name == "run" else f"agg_{name}.json"
    p = folder / fname
    tmp = folder / f".{fname}.{os.getpid()}.tmp"  # same bytes, replaced in one step: a reader never sees half a file
    tmp.write_text(text)
    os.replace(tmp, p)
    return p


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(o, Path):
        return o.name
    raise TypeError(type(o).__name__)
