"""Summary files: the only files that leave Kaggle or reach Claude.

agg_<name>.json holds counts, scores and pass/fail flags only. Counts go
through count() so that 1-4 appear as "<5" (zero stays 0, because it
identifies nobody). run.json records versions, the repo commit, the seed and
the config, so every number can be traced to the code that made it.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from . import paths


def count(n) -> int | str:
    n = int(n)
    return n if (n == 0 or n >= 5) else "<5"


def counts(mapping) -> dict:
    """Apply count() to every value of a {key: int} mapping."""
    return {str(k): count(v) for k, v in dict(mapping).items()}


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
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg",
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
    """Write agg_<name>.json (or run.json when name == 'run')."""
    folder = Path(folder or paths.WORKING)
    folder.mkdir(parents=True, exist_ok=True)
    fname = "run.json" if name == "run" else f"agg_{name}.json"
    p = folder / fname
    p.write_text(json.dumps(payload, indent=1, sort_keys=False, default=_json_default))
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
