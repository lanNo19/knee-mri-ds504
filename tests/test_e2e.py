"""End-to-end on a synthetic competition: N0 -> N3, N4 for each plane, SMOKE-ALL.

The data-safety check: no UID and no report text may appear in any agg_*.json
or run.json, or in anything the steps print (agg.write also refuses small
counts and UID-like strings on its own).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee import dicom_io, paths
from tests.synth import make_competition, write_dicom


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("kaggle")
    inp = base / "input"
    comp = inp / "rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    meta = make_competition(comp, n_per_lang=6, n_slices=12, size=48)
    # Edge cases: two studies without a report, one very short report (unsure language)
    tr = pd.read_csv(comp / "train.csv")
    tr.loc[3, "Report"] = ""
    tr.loc[4, "Report"] = ""
    tr.loc[5, "Report"] = "MRI knee."
    tr.to_csv(comp / "train.csv", index=False)
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
    bad += [r[:40] for r in meta["reports"] if len(r) > 40 and r[:40] in text]
    return bad


def _root_files(work: Path) -> list[str]:
    return sorted(p.name for p in work.iterdir() if p.is_file())


def test_n0(env, capfd):
    from knee.nb import n0

    s = n0.run(workers=2, quiet=True)
    printed = capfd.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    assert "=== KNEE SUMMARY BEGIN ===" in printed.out
    for f in env["work"].glob("*.json"):
        assert not _leaks(f.read_text(), env)
    assert set(_root_files(env["work"])) <= {"agg_n0.json", "run.json"}
    assert s["labels"]["gold"]["bug_check_passed"]
    assert s["groups"]["g_lang_groups"] == 6  # GERMAN and DUTCH merge on their shared scanner
    st = pd.read_csv(env["work"] / "private" / "n0" / "studies.csv")
    assert len(st) == env["n_studies"]
    side = {u: ("R" if i % 2 == 0 else "L") for i, u in enumerate(env["uids"])}
    assert all(st["laterality"] == st["StudyInstanceUID"].map(side))
    assert s["laterality"]["tag_vs_geometry"]["margin_20mm"]["agreement"] == 1.0
    no_rep = st[st["StudyInstanceUID"].isin([env["uids"][3], env["uids"][4]])]
    assert not no_rep["in_training"].any() and not no_rep["scored"].any()
    assert (env["work"] / "private" / "n0" / "tags.csv").exists()
    assert s["labels"]["pos_weight_cap_upper"] >= 20.0


def test_n3(env, capfd):
    from knee.nb import n3

    # On Kaggle, N3 attaches N0's output under /kaggle/input/<n0 notebook>/
    shutil.copytree(env["work"] / "private", env["inp"] / "knee-n0" / "private", dirs_exist_ok=True)
    shutil.copy(env["work"] / "run.json", env["inp"] / "knee-n0" / "run.json")
    s = n3.run(quiet=True)
    printed = capfd.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    assert not _leaks((env["work"] / "agg_n3.json").read_text(), env)
    f = pd.read_csv(env["work"] / "private" / "n3" / "folds.csv").set_index("StudyInstanceUID")
    st = pd.read_csv(env["work"] / "private" / "n0" / "studies.csv").set_index("StudyInstanceUID")
    tr = f[f["train"]]
    for k in range(s["s2"]["n_folds"]):
        assert not (set(st.loc[tr.index[tr["s2_fold"] == k], "g_lang"]) & set(st.loc[tr.index[tr["s2_fold"] != k], "g_lang"]))
    assert not f.loc[f["is_gold"], "train"].any()
    # every gold study is scored by the fold whose test fold holds its group
    for uid in f.index[f["is_gold"]]:
        k = f.at[uid, "s2_fold"]
        g = st.at[uid, "g_lang"]
        if k >= 0:
            assert g in set(st.loc[tr.index[tr["s2_fold"] == k], "g_lang"])
    assert s["s1"]["sizes_match_s2"]
    assert s["upstream_n0"]["sha16"]


@pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
def test_n4(env, plane, capfd):
    from knee.nb import n4

    s = n4.run(plane, budget_h=1.0, workers=2, quiet=True)
    printed = capfd.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    assert not _leaks((env["work"] / f"agg_n4_{plane}.json").read_text(), env)
    assert s["checks"]["failures_under_1pct"] and s["checks"]["shape_ok"]
    od = env["work"] / "private" / f"n4_{plane}"
    man = json.loads((od / f"cache_{plane}_manifest.json").read_text())
    mm = np.memmap(od / man["file"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    assert mm.shape == (env["n_studies"], 32, 224, 224) and mm.max() > 0
    st = pd.read_csv(od / "study_table.csv")
    for row, lat in enumerate(st["laterality"]):
        m = mm[row].reshape(32, -1).mean(1)
        if plane == "sagittal" and lat == "L":
            assert m[0] > m[-1]
        else:
            assert m[0] < m[-1]


def test_smoke_all(env, capfd):
    from knee.nb import smoke_all

    res = smoke_all.run(limit=20, workers=2)
    printed = capfd.readouterr()
    assert not _leaks(printed.out + printed.err, env)
    assert all(v == "OK" for v in res.values()), res


def _canon(h=64, w=96):
    a = np.full((h, w), 100, np.uint16)
    a[: h // 4, : w // 4] = 3000  # marker in the top-left quadrant
    return a


@pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
@pytest.mark.parametrize("lat", ["R", "L"])
def test_frame_and_mirror_through_dicom_files(tmp_path, plane, lat):
    """All 8 storage variants per plane come back in the canonical frame; left knees mirror (not sagittal)."""
    tgt = dicom_io.TARGET[plane]
    stack = np.array(tgt["stack"], float)
    for transpose in (False, True):
        for fr in (1, -1):
            for fc in (1, -1):
                img = _canon()
                r, c = np.array(tgt["along_row"], float), np.array(tgt["down_col"], float)
                if fc == -1:
                    img, r = img[:, ::-1], -r
                if fr == -1:
                    img, c = img[::-1, :], -c
                if transpose:
                    img, (r, c) = img.T, (c, r)
                d = tmp_path / f"{plane}_{lat}_{int(transpose)}{fr}{fc}"
                for k in range(6):
                    ipp = list(stack * (5.0 * k) + np.array([-80.0, 0, 0]))
                    write_dicom(d / f"{(k * 5) % 6:03d}.dcm", np.ascontiguousarray(img), ipp, list(r) + list(c),
                                k + 1, ("S", "M", 3.0), None)
                v = dicom_io.prepare_volume(d, plane, lat).volume[0].astype(float)
                left = v[:40, :40].mean() > v[:40, -40:].mean()
                assert left == (not (lat == "L" and plane != "sagittal")), (transpose, fr, fc)
