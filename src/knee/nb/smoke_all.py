"""SMOKE-ALL: every written stage on a few real studies, in one session.

Why: Claude cannot see the real data, so each new stage meets it first here,
on about 20 studies in about 15 minutes, before the full run. Stages run in
order and write under private/smoke_all/; the summary lists each stage's
outcome (OK or the safe error line).
"""
from __future__ import annotations

from .. import agg, paths, safe_log
from . import common


def run(limit: int = 20, stages: tuple[str, ...] = ("n0", "n3", "n4"), workers: int = 4) -> dict:
    results: dict = {}
    with common.notebook("smoke_all", {"limit": limit, "stages": list(stages)}, quiet=False):
        base = paths.WORKING
        sub = paths.out_dir("private", "smoke_all")
        paths.WORKING = sub  # stages write under private/smoke_all
        try:
            if "n0" in stages:
                results["n0"] = _try(lambda: _n0(limit, workers))
            if "n3" in stages:
                results["n3"] = _try(lambda: _n3(sub))
            if "n4" in stages:
                for plane in ("sagittal", "coronal", "axial"):
                    results[f"n4_{plane}"] = _try(lambda p=plane: _n4(p, limit, workers))
        finally:
            paths.WORKING = base
        agg.write("smoke_all", {"limit": limit, "stages": results})
    common.summary({"smoke_all": results})
    return results


def _try(fn) -> str:
    try:
        fn()
        return "OK"
    except safe_log.StepFailed as e:
        return f"FAILED - {e}"


def _n0(limit, workers):
    from . import n0

    n0.run(limit=limit, workers=workers)


def _n3(sub):
    from . import n3

    n3.run(studies_csv=sub / "private" / "n0" / "studies.csv")


def _n4(plane, limit, workers):
    from . import n4

    n4.run(plane, budget_h=0.5, workers=workers, limit=limit)
