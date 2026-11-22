"""Write the two-cell notebooks and their kernel-metadata.json (the CLI fallback).

The browser path pastes the same two cells (notebooks/RUNBOOK.md). This script
keeps both in step. Usage (fallback only, on Ablay's Mac):

    python scripts/make_notebooks.py --user <kaggle-username> --commit <sha>
        [--dread-file <file>.csv [--dread-provenance-ok]]

--dread-file: the label file's name in the dreaddevelopment dataset, once Ablay
has recorded it. SMOKE and N0 then attach that dataset and check the table;
N0 uses it only with --dread-provenance-ok (Ablay recorded that its labels are
LLM output on report text with no gold, image-model or other-table input) and
only if it passes the agreement gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
TABLES = ["pilkwang/rsna-knee-llm-labels", "stevenleehans/rsna-knee-llm-report-labels",
          "lixin73/rsna-knee-llm-report-labels-sol56"]
DREAD = "dreaddevelopment/knee-raptor-corpus"

# name: (gpu, uses competition, uses label tables, upstream notebooks, timeout s)
NOTEBOOKS = {
    "smoke": (True, True, True, [], 1800),
    "smoke-all": (False, True, True, [], 3600),
    "n0": (False, True, True, [], 21600),
    "n3": (False, False, False, ["n0"], 3600),
    "n4-sagittal": (False, True, False, [], 32400),
    "n4-coronal": (False, True, False, [], 32400),
    "n4-axial": (False, True, False, [], 32400),
    "montage": (False, False, False, ["n4-sagittal", "n4-coronal", "n4-axial"], 3600),
}


def cell2(name: str, dread_file: str | None = None, dread_ok: bool = False) -> str:
    if name == "smoke":
        arg = f"dread_file={dread_file!r}" if dread_file else ""
        return f"from knee.nb import smoke; smoke.run({arg})"
    if name == "smoke-all":
        return "from knee.nb import smoke_all; smoke_all.run(limit=20)"
    if name == "n0":
        arg = f"dread_file={dread_file!r}, dread_provenance_ok={bool(dread_ok)}" if dread_file else ""
        return f"from knee.nb import n0; n0.run({arg})"
    if name == "n3":
        return "from knee.nb import n3; n3.run()"
    if name.startswith("n4-"):
        return f'from knee.nb import n4; n4.run("{name[3:]}", budget_h=8)'
    if name == "montage":
        return "from knee.nb import montage; montage.run()"
    raise KeyError(name)


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


def metadata(user: str, name: str, dread_file: str | None = None) -> dict:
    gpu, comp, tables, up, _ = NOTEBOOKS[name]
    datasets = list(TABLES) if tables else []
    if dread_file and name in ("smoke", "n0"):
        datasets.append(DREAD)
    meta = {
        "id": f"{user}/knee-{name}", "title": f"knee-{name}", "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_internet": True,
        "competition_sources": [COMPETITION] if comp else [],
        "dataset_sources": datasets,
        "kernel_sources": [f"{user}/knee-{u}" for u in up],
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", default="notebooks")
    ap.add_argument("--dread-file", default=None)
    ap.add_argument("--dread-provenance-ok", action="store_true")
    a = ap.parse_args(argv)
    for name in NOTEBOOKS:
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        code = cell2(name, a.dread_file, a.dread_provenance_ok)
        (d / "notebook.ipynb").write_text(json.dumps(notebook_json(a.commit, code), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps(metadata(a.user, name, a.dread_file), indent=1))
        (d / "timeout_s").write_text(f"{NOTEBOOKS[name][4]}\n")
    print(f"wrote {len(NOTEBOOKS)} notebooks under {a.out}/")


if __name__ == "__main__":
    main()
