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

N11_GPU_INPUTS = ["n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n5", "n6a"]
# "full" unless the smoke run says "halved" (plan 12.1 item 10): the same value in both sessions
N11_SCHEDULE = "full"

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
    # N10 with N11's B-lite fold models (only if N11's decision keeps the change): both sessions' outputs
    # (and knee-n11-c, knee-n11-d if they ran) beside knee-n6a, whose units hold the z-score statistics
    "n10-blite": (True, True, False, ["n0", "n6a", "n11-a", "n11-b"], 32400),
    "n10-convnext-blite": (True, True, False, ["n0", "n6a", "n11-a", "n11-b"], 32400),
    # the replay check before a b-lite submission (internet off: the bundle's wheel and checkpoint). Read: the
    # N4 caches, knee-n6a, the N11 outputs. knee-n0 and knee-n3-r are attached by the card's list, not read
    "n10-tails-check": (True, False, False, ["n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n6a", "n11-a", "n11-b"], 3600),
    "montage": (False, False, False, ["n4-sagittal", "n4-coronal", "n4-axial", "n6p"], 3600),
    # N12 (plan r2, error analysis; RUNBOOK "Error analysis N12"): CPU, descriptive, existing outputs only
    "n12": (False, False, False, ["n0", "n3-r", "n5", "n6a", "n6b-g", "n7-b"], 3600),
    # N14 (error hunt, n14_spec_r3): CPU, descriptive; the competition data for C5's header pass only
    "n14": (False, True, False, ["n0", "n3-r", "n5", "n6a", "n6b-g", "n7-b"], 10800),
    # N11 (iteration 3, Stage B-lite on all five folds; plan section 12). GPU steps: internet on (pip and the
    # encoder weights, as N5); the three N4 caches, N5's embeddings and index, knee-n6a's Stage A units.
    # Two sessions at the same time: a = folds 0, 1, 2; b = folds 3, 4. The CPU steps attach only N11's outputs.
    "n11-smoke": (True, False, False, N11_GPU_INPUTS, 3600),
    "n11-a": (True, False, False, N11_GPU_INPUTS, 43200),
    "n11-b": (True, False, False, N11_GPU_INPUTS, 43200),
    # sessions c and d: only after a partial session, one notebook per rerun call (a second call with the
    # same session name would delete the first call's folder); only units from units_missing; edit folds
    # and encoders in cell 2
    "n11-c": (True, False, False, N11_GPU_INPUTS, 43200),
    "n11-d": (True, False, False, N11_GPU_INPUTS, 43200),
    "n11-decide": (False, False, False, ["n11-a", "n11-b"], 3600),
    "n11-dev": (False, False, False, ["n11-a", "n11-b", "n11-decide"], 3600),
    # per-finding description of the same prediction files, behind dev()'s gate (descriptive, R8)
    "n11-dev-findings": (False, False, False, ["n11-a", "n11-b", "n11-decide"], 3600),
    # N11x (the learning-rate extension of iteration 3, it3_lr_extension_r3; RUNBOOK "Learning-rate extension
    # N11x"). GPU steps: the same inputs as n11-a and nothing of iteration 3 (R9: its numbers are constants in
    # the code). The CPU selection attaches the session and iteration 3's sessions (their state.json is read,
    # as n11-decide reads it). The refinement point runs only by the selection's refinement.rate.
    "n11x-smoke": (True, False, False, N11_GPU_INPUTS, 3600),
    "n11x-inner": (True, False, False, N11_GPU_INPUTS, 25200),
    "n11x-smoke-select": (False, False, False, ["n11x-smoke", "n11-a", "n11-b"], 1800),
    "n11x-select": (False, False, False, ["n11x-inner", "n11-a", "n11-b"], 3600),
    "n11x-refine": (True, False, False, N11_GPU_INPUTS, 14400),
    # N11x, delivery D2 (declaration, section 6). The final fits attach the GPU inputs and the selection's
    # output (the signed record), and nothing of iteration 3 (R9): frozen0 is recomputed there for check (b),
    # and no head-only seed is fitted. The CPU development step attaches iteration 3's sessions and its keep
    # record: frozen0 and head-only come from there, as in n11-dev. Timeout: above 1.1 x 3.8 h.
    "n11x-final": (True, False, False, N11_GPU_INPUTS + ["n11x-select"], 16200),
    "n11x-dev": (False, False, False, ["n11x-final", "n11x-select", "n11-a", "n11-b", "n11-decide"], 3600),
    # N10 with the refit model (internet off; the bundle must be rebuilt at the commit that holds this code):
    # as n10-tails-check and n10-blite, with knee-n11x-final and knee-n11x-select instead of knee-n11-a/-b
    "n10-tails-check-x": (True, False, False, ["n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n6a",
                                               "n11x-final", "n11x-select"], 3600),
    "n10-blite-x": (True, True, False, ["n0", "n6a", "n11x-final", "n11x-select"], 32400),
    # N13, the batch-floor check (it2b_batch_floor_r1.md; RUNBOOK "Batch-floor check"): CPU, three sessions at
    # the same time, added after the results. Inputs as the iteration-2 arms (knee-n0, knee-n5, knee-n3-r) plus
    # knee-n6a, whose per-unit summaries hold the chosen configurations. bc is the base (the chosen batch size).
    # Cell 1 of these three is pinned to BATCH_COMMIT (below), whatever --commit says. By hand: forks of
    # knee-n13-base0 with "Pin to original environment" kept, knee-n6a added (RUNBOOK "Batch-floor check").
    "n13-bc": (False, False, False, ["n0", "n5", "n3-r", "n6a"], 25200),
    "n13-b8": (False, False, False, ["n0", "n5", "n3-r", "n6a"], 25200),
    "n13-b4": (False, False, False, ["n0", "n5", "n3-r", "n6a"], 25200),
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
    if name == "n10-blite":
        return 'from knee.nb import n10; n10.run("dinov2_vits14", tails="n11")'
    if name == "n10-convnext-blite":
        return 'from knee.nb import n10; n10.run("convnext_tiny", tails="n11")'
    if name == "n10-tails-check":
        return 'from knee.nb import n10; n10.check_tails("dinov2_vits14", fold=0, n=200)'
    if name == "n12":
        return f"from knee.nb import n12; n12.run(expect={EXPECT})"
    if name == "n14":
        return f"from knee.nb import n14; n14.run(expect={EXPECT})"
    if name == "n11-smoke":
        return f"from knee.nb import n11; n11.run(smoke=True, expect={EXPECT})"
    if name in ("n11-a", "n11-b"):
        s = name[-1]
        folds = {"a": "(0, 1, 2)", "b": "(3, 4)"}[s]
        return f'from knee.nb import n11; n11.run(session="{s}", folds={folds}, expect={EXPECT}, schedule="{N11_SCHEDULE}")'
    if name in ("n11-c", "n11-d"):
        return (f'from knee.nb import n11; n11.run(session="{name[-1]}", folds=(0,), encoders=("convnext_tiny",), '
                f'expect={EXPECT}, schedule="{N11_SCHEDULE}")')
    if name == "n11-decide":
        return "from knee.nb import n11; n11.decide()"
    if name == "n11-dev":
        return "from knee.nb import n11; n11.dev()"
    if name == "n11-dev-findings":
        return "from knee.nb import n11; n11.dev_findings()"
    if name == "n11x-smoke":  # budget_h per session type: the declaration, section 9 (no default in the code)
        return f'from knee.nb import n11; n11.run_lrx("smoke", smoke=True, budget_h=0.5, expect={EXPECT})'
    if name == "n11x-inner":  # option A: both new rates, one per GPU, five folds
        return f'from knee.nb import n11; n11.run_lrx("inner", budget_h=5.8, expect={EXPECT})'
    if name == "n11x-smoke-select":
        return "from knee.nb import n11; n11.select_lrx(smoke=True)"
    if name == "n11x-select":
        return "from knee.nb import n11; n11.select_lrx()"
    if name == "n11x-refine":  # edit rates: refinement.rate of agg_n11x_select.json; budget_h: min(3.4, quota - 0.4)
        return (f'from knee.nb import n11; n11.run_lrx("refine", rates=(1.6e-4,), split_folds=True, replay=False, '
                f'budget_h=3.4, expect={EXPECT})')
    if name == "n11x-final":
        # BUDGET_H is not defined on purpose: the caller types the number in its place (no default anywhere).
        # Five folds: the smaller of 3.8 and (quota minus 0.4). Quota fallback (under 4.2 h left): add
        # folds=<the moved folds> and type the smaller of 2.7 (three or four folds) or 1.4 (one or two) and
        # (quota minus 0.4)
        return f"from knee.nb import n11; n11.run_final_lrx(budget_h=BUDGET_H, expect={EXPECT})"
    if name == "n11x-dev":
        return "from knee.nb import n11; n11.dev_lrx()"
    if name == "n10-tails-check-x":  # fold=None: the lowest fold that moved in the selection record
        return 'from knee.nb import n10; n10.check_tails("dinov2_vits14", fold=None, n=200, tails="n11x")'
    if name == "n10-blite-x":
        return 'from knee.nb import n10; n10.run("dinov2_vits14", tails="n11x")'
    if name in ("n13-bc", "n13-b8", "n13-b4"):  # budget_h: the soft stop; lower it if the session starts late
        return f'from knee.nb import n13; n13.run("{name[4:]}", budget_h=6.0, expect={EXPECT})'
    raise KeyError(name)


# The batch-floor arms (declaration it2b_batch_floor_r2.md): the reviewed commit of the three sessions.
# Later commits change only the local decide step, so the arms are never regenerated at another commit.
BATCH_COMMIT = "39c3c6fc7626e047ed643d5253e8ce9e662302af"
BATCH_NOTEBOOKS = ("n13-bc", "n13-b8", "n13-b4")


def commit_for(name: str, commit: str) -> str:
    """The commit of a notebook's cell 1: the pinned one for the batch-floor arms, else the caller's."""
    return BATCH_COMMIT if name in BATCH_NOTEBOOKS else commit


CELL1_OFFLINE = (
    "import glob, subprocess, sys\n"
    'W = sorted(glob.glob("/kaggle/input/*/private/bundle/wheels") + glob.glob("/kaggle/input/notebooks/*/*/private/bundle/wheels")'
    ' + glob.glob("/kaggle/input/datasets/*/*/private/bundle/wheels"))\n'
    'assert W, "attach the knee bundle dataset"\n'
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W[0], "knee", "timm==1.0.31",'
    ' "python-gdcm", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "dicomsdl"], check=True)\n'
    "import knee.safe_log as sl; sl.install()"
)
OFFLINE = ("n10", "n10-convnext", "n10-canary", "n10-blite", "n10-convnext-blite", "n10-tails-check",
           "n10-tails-check-x", "n10-blite-x")


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
        (d / "notebook.ipynb").write_text(json.dumps(notebook_json(commit_for(name, a.commit), code, offline=name in OFFLINE), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps(
            metadata(a.user, name, a.dread_file, a.bundle_dataset, a.without_n6b, a.n7_cpu), indent=1))
        (d / "timeout_s").write_text(f"{NOTEBOOKS[name][4]}\n")
    print(f"wrote {len(NOTEBOOKS)} notebooks under {a.out}/")


if __name__ == "__main__":
    main()
