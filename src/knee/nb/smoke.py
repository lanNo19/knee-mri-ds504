"""SMOKE: checks the platform and the inputs' layouts before any real run.

Checks: GPU names (T4 x2 expected on GPU runs), package versions, the safe-log
handler (a synthetic UID in an error message must not appear), that the
competition data and the label tables are attached, the column names of
train.csv and train_series.csv (only if they look like a real header), the
value counts of the series-type columns, and each label table's layout and
cell counts (yes / no / unk / unmapped). Never cell values.
Writes run.json and agg_smoke.json and prints the summary block.
"""
from __future__ import annotations

import io
import os
import re
import sys

import pandas as pd

from .. import agg, labels, paths, safe_log
from . import common

_CAT = re.compile(r"^[A-Za-z0-9 _\-]{1,30}$")


def _safe_log_test() -> bool:
    """Test 11 of the plan: an error carrying a fake UID prints only its type."""
    fake = "1.2.826.0.1.3680043.8.498.99999999999999999999"
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    msg = ""
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


def _categories(series: pd.Series) -> dict:
    """Value counts of a categorical column, only for short plain labels."""
    vc = series.astype(str).value_counts()
    if len(vc) > 20 or not all(_CAT.match(v) for v in vc.index):
        return {"plain_labels": False, "n_values": agg.count(len(vc))}
    return {"plain_labels": True, "counts": agg.counts(vc)}


def _table_info(key: str, path) -> dict:
    raw = labels.read_table(path)
    entry = {"attached": True, **agg.safe_columns(raw.columns), "rows": agg.count(len(raw))}
    try:
        t = labels.convert(key, raw)
        entry["layout"] = t.layout
        entry["cells"] = agg.counts(t.cells or {})
        entry["has_unk"] = t.unk is not None
    except labels.SchemaError:
        entry["layout"] = "unrecognised"
    return entry


def run(expect_gpu: bool = True, dread_file: str | None = None) -> dict:
    out: dict = {"checks": {}}
    with common.notebook("smoke", {"expect_gpu": expect_gpu, "dread_file": bool(dread_file)}, quiet=False):
        info = agg.run_info("smoke")
        gpus = info.get("gpus") or []
        out["gpus"] = gpus
        out["checks"]["t4x2"] = (len(gpus) == 2 and all("T4" in g for g in gpus)) if expect_gpu else None
        out["checks"]["safe_log"] = _safe_log_test()
        try:
            root = paths.competition_root()
            names = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
            out["competition_files"] = [n for n in names if _CAT.match(n.replace(".", "_"))][:20]
            out["checks"]["competition_attached"] = True
            tr = pd.read_csv(root / "train.csv", nrows=50)
            out["train_csv"] = agg.safe_columns(tr.columns)
            ser = pd.read_csv(root / "train_series.csv")
            out["train_series_csv"] = agg.safe_columns(ser.columns)
            for col in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"):
                if col in ser.columns:
                    out[f"values_{col}"] = _categories(ser[col])
        except FileNotFoundError:
            out["checks"]["competition_attached"] = False
        tables = {}
        for key in paths.LABEL_TABLES:
            p = paths.find_label_table(key)
            tables[key] = _table_info(key, p) if p is not None else {"attached": False}
        if dread_file:
            p = paths.find_file(dread_file)
            tables["dread"] = _table_info("dread", p) if p is not None else {"attached": False}
        out["label_tables"] = tables
        out["checks"]["pilkwang_verdict_conf_unk"] = (
            tables.get("pilkwang", {}).get("layout") == "verdict+confidence" and tables["pilkwang"].get("has_unk", False)
        )
        out["versions"] = info["versions"]
        out["commit"] = (info["commit"] or "")[:12] or None
        agg.write("smoke", out)
    common.summary({"checks": out["checks"], "label_layouts": {k: v.get("layout") for k, v in out["label_tables"].items()}})
    return out


if __name__ == "__main__":  # pragma: no cover
    run(expect_gpu=os.environ.get("KNEE_EXPECT_GPU", "1") == "1")
