"""Shared notebook plumbing: run.json first, the safe step, the summary block, upstream checks."""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from .. import agg, paths, safe_log


@contextlib.contextmanager
def notebook(name: str, config: dict | None = None, quiet: bool = True, budget_h: float | None = None):
    """Write run.json at the output root first, then run the step safely.

    On failure the quiet log and traceback go to private/<name>_error.log (for
    Ablay only). A watchdog interrupts the step at 1.1 x budget_h.
    """
    agg.write("run", agg.run_info(name, config or {}))
    watchdog = 1.1 * budget_h * 3600 if budget_h else None
    err = paths.out_dir("private") / f"{name}_error.log"
    with safe_log.step(name, quiet=quiet, error_log=err, watchdog_s=watchdog):
        yield


def summary(payload: dict) -> None:
    """Print the step's summary between the markers (after the agg checks passed)."""
    text = json.dumps(payload, indent=1, default=agg._json_default)
    agg._check(json.loads(text))
    safe_log.block(text)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def upstream(file_name: str, explicit: Path | None = None) -> tuple[Path, dict]:
    """Locate an upstream output file and describe where it came from.

    Returns the path and {"commit": <upstream run.json commit>, "sha16": <file hash>},
    so a stale or wrong attached version is visible in the summary and can be asserted.
    """
    p = Path(explicit) if explicit else paths.find_file(file_name)
    if p is None or not Path(p).exists():
        raise FileNotFoundError("upstream output not attached")
    p = Path(p)
    commit = None
    for parent in [p.parent, *p.parents][:4]:
        rj = parent / "run.json"
        if rj.exists():
            try:
                commit = (json.loads(rj.read_text()).get("commit") or "")[:12] or None
            except Exception:  # noqa: BLE001
                commit = None
            break
    return p, {"commit": commit, "sha16": sha16(p)}


def is_rerun(path) -> bool:
    """A rerun notebook's output: its slug ends in -r or -r2... (knee-n6a-r), or its step folder under
    private/ does (n6a_r). Only those two path parts are checked, never an owner folder."""
    import re

    parts = Path(path).parts
    cands = [p for p in parts if p.startswith("knee-")]
    if "private" in parts:
        i = parts.index("private")
        if i + 1 < len(parts):
            cands.append(parts[i + 1])
    return any(re.search(r"[-_]r\d*$", p) for p in cands)


def expect(info: dict, sha16_prefix: str | None) -> None:
    """Fail loudly if the attached upstream file is not the expected version."""
    if sha16_prefix and not info["sha16"].startswith(sha16_prefix):
        raise AssertionError("attached upstream version differs from the expected one")
