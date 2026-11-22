"""SMOKE: checks the platform before any real run.

Checks: GPU names (T4 x2 expected on GPU runs), package versions, the
safe-log handler (a synthetic UID in an error message must not appear in the
output), that the competition data and the public label tables are attached
(file names only), and each label table's layout (column names only, never
cell values). Writes run.json and agg_smoke.json.
"""
from __future__ import annotations

import io
import os
import sys

from .. import agg, labels, paths, safe_log


def _safe_log_test() -> bool:
    """Test 11 of the plan: an error carrying a fake UID prints only its type."""
    fake = "1.2.826.0.1.3680043.8.498.99999999999999999999"
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        try:
            with safe_log.step("selftest", quiet=False):
                raise KeyError(fake)
        except safe_log.StepFailed as e:
            msg = str(e)
    finally:
        sys.stdout = old
    shown = buf.getvalue() + msg
    return fake not in shown and "KeyError" in shown


def run(expect_gpu: bool = True) -> dict:
    safe_log.install()
    out: dict = {"checks": {}}
    with safe_log.step("smoke", quiet=False):
        info = agg.run_info("smoke")
        gpus = info.get("gpus") or []
        out["gpus"] = gpus
        out["checks"]["t4x2"] = (len(gpus) == 2 and all("T4" in g for g in gpus)) if expect_gpu else None
        out["checks"]["safe_log"] = _safe_log_test()
        try:
            root = paths.competition_root()
            names = sorted(p.name for p in root.iterdir())
            out["competition_files"] = [n for n in names if not n.startswith(".")][:20]
            out["checks"]["competition_attached"] = True
        except FileNotFoundError:
            out["checks"]["competition_attached"] = False
        tables = {}
        for key in paths.LABEL_TABLES:
            p = paths.find_label_table(key)
            if p is None:
                tables[key] = {"attached": False}
                continue
            import pandas as pd

            head = pd.read_csv(p, nrows=200)
            entry = {"attached": True, "columns": [str(c) for c in head.columns][:60]}
            try:
                entry["layout"] = labels.convert(key, head).layout
            except labels.SchemaError:
                entry["layout"] = "unrecognised"
            tables[key] = entry
        out["label_tables"] = tables
        out["versions"] = info["versions"]
        out["commit"] = info["commit"]
        agg.write("run", info)
        agg.write("smoke", out)
    print({"checks": out["checks"], "label_layouts": {k: v.get("layout") for k, v in out["label_tables"].items()}})
    return out


if __name__ == "__main__":  # pragma: no cover
    run(expect_gpu=os.environ.get("KNEE_EXPECT_GPU", "1") == "1")
