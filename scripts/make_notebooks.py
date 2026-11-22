"""Write the two-cell notebooks and their kernel-metadata.json (the CLI fallback).

The browser path pastes the same two cells (notebooks/RUNBOOK.md). This script
keeps both in step. Usage (fallback only, on Ablay's Mac):

    python scripts/make_notebooks.py --user <kaggle-username> --commit <sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
TABLES = ["pilkwang/rsna-knee-llm-labels", "stevenleehans/rsna-knee-llm-report-labels",
          "lixin73/rsna-knee-llm-report-labels-sol56"]

# name: (cell 2, gpu, uses competition, uses label tables, upstream notebooks, timeout s)
NOTEBOOKS = {
    "smoke": ("from knee.nb import smoke; smoke.run()", True, True, True, [], 1800),
    "smoke-all": ("from knee.nb import smoke_all; smoke_all.run(limit=20)", False, True, True, [], 3600),
    "n0": ("from knee.nb import n0; n0.run()", False, True, True, [], 21600),
    "n3": ("from knee.nb import n3; n3.run()", False, False, False, ["n0"], 3600),
    "n4-sagittal": ('from knee.nb import n4; n4.run("sagittal", budget_h=8)', False, True, False, [], 32400),
    "n4-coronal": ('from knee.nb import n4; n4.run("coronal", budget_h=8)', False, True, False, [], 32400),
    "n4-axial": ('from knee.nb import n4; n4.run("axial", budget_h=8)', False, True, False, [], 32400),
    "montage": ("from knee.nb import montage; montage.run()", False, False, False,
                ["n4-sagittal", "n4-coronal", "n4-axial"], 3600),
}


def cell1(commit: str) -> str:
    return (
        f'COMMIT = "{commit}"\n'
        '!pip install -q "knee[n0,decode] @ git+https://github.com/lanNo19/knee-mri-ds504@{COMMIT}"\n'
        "import knee.safe_log as sl; sl.install()"
    )


def notebook_json(commit: str, code: str) -> dict:
    def cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.splitlines(keepends=True)}

    return {"cells": [cell(cell1(commit)), cell(code)],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", default="notebooks")
    a = ap.parse_args()
    for name, (code, gpu, comp, tables, up, timeout) in NOTEBOOKS.items():
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "notebook.ipynb").write_text(json.dumps(notebook_json(a.commit, code), indent=1))
        meta = {
            "id": f"{a.user}/knee-{name}", "title": f"knee-{name}", "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": gpu, "enable_internet": True,
            "competition_sources": [COMPETITION] if comp else [],
            "dataset_sources": TABLES if tables else [],
            "kernel_sources": [f"{a.user}/knee-{u}" for u in up],
            "model_sources": [],
        }
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        (d / "timeout_s").write_text(f"{timeout}\n")
    print(f"wrote {len(NOTEBOOKS)} notebooks under {a.out}/")


if __name__ == "__main__":
    main()
