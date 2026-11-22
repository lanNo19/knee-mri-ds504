"""End-to-end on a synthetic competition: N0 -> N3, and N4 for each plane.

The data-safety check: no UID and no report text may appear in any agg_*.json
or run.json, or in anything the steps print.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee import paths
from tests.synth import make_competition


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("kaggle")
    inp = base / "input"
    comp = inp / "rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    meta = make_competition(comp, n_per_lang=6, n_slices=12, size=48)
    (inp / "tables").mkdir(exist_ok=True)
    for f in (comp / "tables").iterdir():
        f.rename(inp / "tables" / f.name)
    work = base / "working"
    work.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(paths, "INPUT", inp)
    mp.setattr(paths, "WORKING", work)
    mp.delenv("KNEE_DATA_ROOT", raising=False)
    yield {"inp": inp, "work": work, "comp": comp, **meta}
    mp.undo()


def _leaks(text: str, meta: dict) -> list[str]:
    bad = [u for u in meta["uids"] if u in text]
    bad += [r[:40] for r in meta["reports"] if r[:40] in text]
    return bad


def test_n0(env, capsys):
    from knee.nb import n0

    s = n0.run(workers=2, quiet=True)
    printed = capsys.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    agg = (env["work"] / "agg_n0.json").read_text()
    assert not _leaks(agg, env)
    assert s["labels"]["gold"]["bug_check_passed"]
    assert s["groups"]["g_lang_groups"] == 6  # GERMAN and DUTCH merge on their shared scanner
    st = pd.read_csv(env["work"] / "n0" / "studies.csv")
    assert len(st) == env["n_studies"]
    assert (st["laterality_source"] != "unresolved").all()
    # laterality from tags or geometry matches the synthetic side (even index = right)
    side = {u: ("R" if i % 2 == 0 else "L") for i, u in enumerate(env["uids"])}
    assert all(st["laterality"] == st["StudyInstanceUID"].map(side))
    assert s["laterality"]["tag_geometry_agreement"] == 1.0
    soft = pd.read_csv(env["work"] / "n0" / "labels_soft.csv")
    assert soft.shape == (env["n_studies"], 13)


def test_n3(env, capsys):
    import shutil

    from knee.nb import n3

    # On Kaggle, N3 attaches N0's output under /kaggle/input/<n0 notebook>/
    shutil.copytree(env["work"] / "n0", env["inp"] / "knee-n0" / "n0", dirs_exist_ok=True)
    s = n3.run(quiet=True)
    printed = capsys.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    assert not _leaks((env["work"] / "agg_n3.json").read_text(), env)
    f = pd.read_csv(env["work"] / "n3" / "folds.csv")
    st = pd.read_csv(env["work"] / "n0" / "studies.csv").set_index("StudyInstanceUID")
    f = f.set_index("StudyInstanceUID")
    tr = f[f["train"]]
    for k in range(s["s2"]["n_folds"]):
        te_g = set(st.loc[tr.index[tr["s2_fold"] == k], "g_lang"])
        tr_g = set(st.loc[tr.index[tr["s2_fold"] != k], "g_lang"])
        assert not (te_g & tr_g)
    assert not f.loc[f["is_gold"], "train"].any()
    assert s["s1"]["sizes_match_s2"]


@pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
def test_n4(env, plane, capsys):
    from knee.nb import n4

    s = n4.run(plane, budget_h=1.0, workers=2, quiet=True)
    printed = capsys.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    assert not _leaks((env["work"] / f"agg_n4_{plane}.json").read_text(), env)
    assert s["checks"]["no_failures"] and s["checks"]["shape_ok"]
    od = env["work"] / f"n4_{plane}"
    man = json.loads((od / f"cache_{plane}_manifest.json").read_text())
    n = man["shape"][0]
    mm = np.memmap(od / man["file"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    assert n == env["n_studies"] and mm.shape[1:] == (32, 224, 224)
    assert mm.max() > 0
    # Slice order: synthetic intensity grows by 100 per slice along the stack axis
    # (k = 0..11 by position), so the mean rises along the 32 sampled slices for
    # right knees; sagittal left knees are reversed (medial/lateral swap).
    st = pd.read_csv(od / "study_table.csv")
    for row, lat in enumerate(st["laterality"]):
        m = mm[row].reshape(32, -1).mean(1)
        if plane == "sagittal" and lat == "L":
            assert m[0] > m[-1]
        else:
            assert m[0] < m[-1]
