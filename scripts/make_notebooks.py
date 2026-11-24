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
    "smoke-all-2": (True, True, True, [], 5400),
    "n5": (True, False, False, ["n4-sagittal", "n4-coronal", "n4-axial"], 14400),
    "n6p": (True, False, False, ["n0", "n3", "n5"], 14400),
    "n8": (False, False, False, ["n0", "n3", "n5"], 10800),
    "n6a-smoke": (True, False, False, ["n0", "n3", "n5"], 7200),
    "n6a": (True, False, False, ["n0", "n3", "n5"], 32400),
    "n6b": (True, False, False, ["n0", "n3", "n5"], 32400),
    # N7 and N9 attach the rerun knee-n3-r (S1 folds keep flagged pairs together; S2 unchanged), never
    # knee-n3 and never knee-n6a-smoke: exactly one output per upstream step
    "n7": (True, False, False, ["n0", "n3-r", "n5", "n6a", "n6b"], 14400),
    "n9": (False, False, False, ["n0", "n3-r", "n5", "n6p", "n6a", "n6b", "n7", "n8"], 7200),
    "bundle": (False, False, False, [], 3600),
    # internet OFF; the bundle dataset comes from --bundle-dataset
    "n10-canary": (False, True, False, ["n0"], 32400),
    "n10": (True, True, False, ["n0", "n6a"], 32400),
    "n10-convnext": (True, True, False, ["n0", "n6a"], 32400),
    "montage": (False, False, False, ["n4-sagittal", "n4-coronal", "n4-axial", "n6p"], 3600),
    # N12 (plan r2, error analysis; RUNBOOK "Error analysis N12"): CPU, descriptive, existing outputs only
    "n12": (False, False, False, ["n0", "n3-r", "n5", "n6a", "n6b-g", "n7-b"], 3600),
    # N14 (error hunt, n14_spec_r3): CPU, descriptive; the competition data for C5's header pass only
    "n14": (False, True, False, ["n0", "n3-r", "n5", "n6a", "n6b-g", "n7-b"], 10800),
}


# The first 8 characters of each upstream file's sha16 for the 23 Nov chain (n3_folds: knee-n3-r)
EXPECT = '{"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}'


MAIN_ONLY = 'encoder_names=("dinov2_vits14", "convnext_tiny"), '


def cell2(name: str, dread_file: str | None = None, dread_ok: bool = False, without_n6b: bool = False) -> str:
    if name == "smoke":
        arg = f"dread_file={dread_file!r}" if dread_file else ""
        return f"from knee.nb import smoke; smoke.run({arg})"
    if name == "smoke-all":
        return "from knee.nb import smoke_all; smoke_all.run(limit=20)"
    if name == "smoke-all-2":
        return 'from knee.nb import smoke_all; smoke_all.run(limit=20, stages=("n0", "n3", "n4", "n5", "n6p", "n8"))'
    if name == "n5":
        return "from knee.nb import n5; n5.run()"
    if name == "n6p":
        return "from knee.nb import n6p; n6p.run()"
    if name == "n8":
        return "from knee.nb import n8; n8.run()"
    if name == "n7":
        return f"from knee.nb import n7; n7.run({MAIN_ONLY if without_n6b else ''}expect={EXPECT})"
    if name == "n9":
        return f"from knee.nb import n9; n9.run(expect={EXPECT})"
    if name == "n6a-smoke":
        return 'from knee.nb import n6; n6.run("n6a_smoke", encoders=("dinov2_vits14",), folds=(0,))'
    if name == "n6a":
        return 'from knee.nb import n6; n6.run("n6a", encoders=("dinov2_vits14", "convnext_tiny"))'
    if name == "n6b":
        return 'from knee.nb import n6; n6.run("n6b", encoders=("vit_s16",), demo=True)'
    if name == "n0":
        arg = f"dread_file={dread_file!r}, dread_provenance_ok={bool(dread_ok)}" if dread_file else ""
        return f"from knee.nb import n0; n0.run({arg})"
    if name == "n3":
        return "from knee.nb import n3; n3.run()"
    if name.startswith("n4-"):
        return f'from knee.nb import n4; n4.run("{name[3:]}", budget_h=8)'
    if name == "montage":
        return "from knee.nb import montage; montage.run()"
    if name == "bundle":
        return "from knee.nb import bundle; bundle.run()"
    if name == "n10-canary":
        return "from knee.nb import n10; n10.run(canary=True)"
    if name == "n10":
        return 'from knee.nb import n10; n10.run("dinov2_vits14")'
    if name == "n10-convnext":
        return 'from knee.nb import n10; n10.run("convnext_tiny")'
    if name == "n12":
        return f"from knee.nb import n12; n12.run(expect={EXPECT})"
    if name == "n14":
        return f"from knee.nb import n14; n14.run(expect={EXPECT})"
    raise KeyError(name)


CELL1_OFFLINE = (
    "import glob, subprocess, sys\n"
    'W = sorted(glob.glob("/kaggle/input/*/private/bundle/wheels") + glob.glob("/kaggle/input/notebooks/*/*/private/bundle/wheels")'
    ' + glob.glob("/kaggle/input/datasets/*/*/private/bundle/wheels"))\n'
    'assert W, "attach the knee bundle dataset"\n'
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W[0], "knee", "timm==1.0.31",'
    ' "python-gdcm", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "dicomsdl"], check=True)\n'
    "import knee.safe_log as sl; sl.install()"
)
OFFLINE = ("n10", "n10-convnext", "n10-canary")


def cell1(commit: str, offline: bool = False) -> str:
    if offline:  # N10: internet off, everything from the bundle (its wheel is the pinned commit)
        return CELL1_OFFLINE
    return (
        f'COMMIT = "{commit}"\n'
        '!pip install -q "knee[n0,decode,models] @ git+https://github.com/lanNo19/knee-mri-ds504@{COMMIT}"\n'
        "import knee.safe_log as sl; sl.install()"
    )


def notebook_json(commit: str, code: str, offline: bool = False) -> dict:
    def cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.splitlines(keepends=True)}

    return {"cells": [cell(cell1(commit, offline)), cell(code)],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def metadata(user: str, name: str, dread_file: str | None = None, bundle: str | None = None,
             without_n6b: bool = False, cpu: bool = False) -> dict:
    """without_n6b: N7 and N9 leave knee-n6b out of their inputs (it is still queued or running); they
    are rerun in new notebooks once it has finished. cpu: N7 without an accelerator (its code runs on either)."""
    gpu, comp, tables, up, _ = NOTEBOOKS[name]
    if without_n6b and name in ("n7", "n9"):
        up = [u for u in up if u != "n6b"]
    if cpu and name == "n7":
        gpu = False
    datasets = list(TABLES) if tables else []
    if dread_file and name in ("smoke", "n0"):
        datasets.append(DREAD)
    if name in OFFLINE and bundle:
        datasets.append(bundle)
    meta = {
        "id": f"{user}/knee-{name}", "title": f"knee-{name}", "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_internet": name not in OFFLINE,
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
    ap.add_argument("--bundle-dataset", default=None, help="owner/slug of the private bundle dataset (N10)")
    ap.add_argument("--without-n6b", action="store_true",
                    help="N7 and N9 without knee-n6b attached (main encoders only); rerun both when N6b has finished")
    ap.add_argument("--n7-cpu", action="store_true", help="N7 without a GPU (the T4 queue is full)")
    a = ap.parse_args(argv)
    for name in NOTEBOOKS:
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        code = cell2(name, a.dread_file, a.dread_provenance_ok, a.without_n6b)
        (d / "notebook.ipynb").write_text(json.dumps(notebook_json(a.commit, code, offline=name in OFFLINE), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps(
            metadata(a.user, name, a.dread_file, a.bundle_dataset, a.without_n6b, a.n7_cpu), indent=1))
        (d / "timeout_s").write_text(f"{NOTEBOOKS[name][4]}\n")
    print(f"wrote {len(NOTEBOOKS)} notebooks under {a.out}/")


if __name__ == "__main__":
    main()
