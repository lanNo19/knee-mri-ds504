"""Unit tests on synthetic data: safe logging, summaries, series choice, frame, labels, groups, folds."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

from knee import agg, dicom_io, folds, groups, labels, safe_log, series

FAKE_UID = "1.2.826.0.1.3680043.8.498.424242424242"


# ---------------------------------------------------------------- safe_log

def test_step_hides_error_message(capsys):
    with pytest.raises(safe_log.StepFailed) as ei:
        with safe_log.step("t", quiet=False):
            raise KeyError(FAKE_UID)
    out = capsys.readouterr().out
    assert FAKE_UID not in out and FAKE_UID not in str(ei.value)
    assert "KeyError" in out and "test_core.py" in out
    assert safe_log.BEGIN in out and safe_log.END in out


def test_step_quiet_hides_python_and_fd_output(capfd):
    with safe_log.step("q", quiet=True):
        print("secret report text " + FAKE_UID)
        sys.stderr.write("stderr " + FAKE_UID + "\n")
        os.write(1, ("fd1 " + FAKE_UID + "\n").encode())
        os.write(2, ("fd2 " + FAKE_UID + "\n").encode())
    out = capfd.readouterr()
    assert FAKE_UID not in out.out and FAKE_UID not in out.err
    assert "STEP q: OK" in out.out


def test_failed_step_writes_private_error_log_and_keeps_exception(tmp_path, capfd):
    err = tmp_path / "private" / "x_error.log"
    with pytest.raises(safe_log.StepFailed):
        with safe_log.step("x", quiet=True, error_log=err):
            os.write(2, ("detail " + FAKE_UID + "\n").encode())
            raise ValueError(FAKE_UID)
    shown = capfd.readouterr()
    assert FAKE_UID not in shown.out + shown.err
    text = err.read_text()
    assert FAKE_UID in text and "ValueError" in text  # private, for Ablay only
    assert safe_log._last is not None and safe_log._last[0] is ValueError


def test_watchdog_interrupts_step(capsys):
    t0 = time.time()
    with pytest.raises(safe_log.StepFailed) as ei:
        with safe_log.step("w", quiet=False, watchdog_s=0.3):
            while time.time() - t0 < 5:
                time.sleep(0.05)
    assert "watchdog" in str(ei.value) and time.time() - t0 < 3


def test_warning_logging_and_unraisable_hide_messages(capfd):
    import logging
    import warnings

    safe_log.install()
    warnings.warn("value " + FAKE_UID)
    logging.getLogger("pydicom").warning("bad value %s", FAKE_UID)
    logging.getLogger("some.other").error("bad %s", FAKE_UID)

    class Bad:
        def __del__(self):
            raise ValueError(FAKE_UID)

    b = Bad()
    del b
    out = capfd.readouterr()
    assert FAKE_UID not in out.err and FAKE_UID not in out.out


# ---------------------------------------------------------------- agg

def test_agg_write_refuses_small_counts_and_uids(tmp_path):
    with pytest.raises(agg.UnsafeSummary):
        agg.write("t", {"failed": 2}, tmp_path)
    with pytest.raises(agg.UnsafeSummary):
        agg.write("t", {"x": {"y": [7, 3]}}, tmp_path)
    with pytest.raises(agg.UnsafeSummary):
        agg.write("t", {"columns": [FAKE_UID]}, tmp_path)
    with pytest.raises(agg.UnsafeSummary):
        agg.write("t", {"text": "x" * 300}, tmp_path)
    p = agg.write("t", {"failed": agg.count(2), "n_folds": 5, "fold": 1, "ok": True, "share": 0.5, "none": 0}, tmp_path)
    assert p.exists()


def test_safe_columns_rejects_headerless_tables():
    assert agg.safe_columns(["StudyInstanceUID", "ACL", "Baker's"])["header_ok"]
    bad = agg.safe_columns([FAKE_UID, "0.9", "YES"])
    assert not bad["header_ok"] and "columns" not in bad


def test_count_and_share():
    assert agg.count(0) == 0 and agg.count(3) == "<5" and agg.count(5) == 5
    # counts of 5 or more are rounded to the nearest 5, so a total minus the other
    # categories cannot give a small count back exactly
    assert [agg.count(n) for n in (6, 7, 8, 12, 13, 58, 1002, 1003)] == [5, 5, 10, 10, 15, 60, 1000, 1005]
    assert agg.share(0.99981) == 1.0
    q = agg.quantiles(range(100))
    assert "min" not in q and "max" not in q and "p1" in q


def test_structural_keys_exempt_only_their_own_ints(tmp_path):
    # direct structural values pass
    agg.write("t", {"folds": 3, "languages": 2, "inner_k": [4, 4, 3], "n_groups": 4}, tmp_path)
    # counts nested under a structural key are still checked
    for bad in ({"languages": {"ENGLISH": 3}}, {"planes": {"sagittal": {"studies": 2}}},
                {"folds": [{"studies": 3}]}, {"languages": {"x": FAKE_UID}}):
        with pytest.raises(agg.UnsafeSummary):
            agg.write("t", bad, tmp_path)


def test_run_json_accepts_small_config_values(tmp_path):
    # cell-2 parameters are not counts of studies (N3 after the pilot, montage n=4)
    agg.write("run", agg.run_info("n3", {"confirmed_pairs": 2}), tmp_path)
    agg.write("run", agg.run_info("montage", {"n": 4, "seed": 0}), tmp_path)
    with pytest.raises(agg.UnsafeSummary):  # strings under config are still checked
        agg.write("run", agg.run_info("x", {"file": FAKE_UID}), tmp_path)
    with pytest.raises(agg.UnsafeSummary):  # and nothing outside config is exempt
        agg.write("run", {**agg.run_info("x", {}), "failed": 2}, tmp_path)


def test_find_file_reaches_notebook_output_mounts(tmp_path):
    from knee import paths

    # newer images mount notebook outputs at /kaggle/input/notebooks/<owner>/<slug>/
    f = tmp_path / "notebooks" / "owner" / "knee-n0" / "private" / "n0" / "studies.csv"
    f.parent.mkdir(parents=True)
    f.write_text("x\n")
    (tmp_path / paths.COMPETITION / "a" / "b").mkdir(parents=True)
    (tmp_path / paths.COMPETITION / "a" / "b" / "studies.csv").write_text("never\n")
    assert paths.find_file("studies.csv", tmp_path) == f


def _sleep_long(_):
    time.sleep(600)


def test_watchdog_ends_step_with_hung_worker(capfd):
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    with pytest.raises(safe_log.StepFailed) as ei:
        with safe_log.step("hang", quiet=True, watchdog_s=1.0):
            with ProcessPoolExecutor(2) as ex:
                list(ex.map(_sleep_long, range(2)))
    assert str(ei.value).startswith("watchdog:")
    assert time.time() - t0 < 1.0 + safe_log.WATCHDOG_GRACE_S + 10
    assert "STEP hang: FAILED - watchdog:" in capfd.readouterr().out


def test_notebook_cells_match_runbook():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_notebooks", root / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    runbook = (root / "notebooks" / "RUNBOOK.md").read_text()
    for name in mk.NOTEBOOKS:
        if name not in ("n3", "n6p", "n6a", "n6b", "n9", "n4-coronal", "n4-axial"):  # the card adds values; n4: "likewise"
            assert mk.cell2(name) in runbook, name
    assert mk.DREAD not in mk.metadata("u", "n0")["dataset_sources"]
    for name in ("smoke", "n0"):
        assert mk.DREAD in mk.metadata("u", name, "labels.csv")["dataset_sources"]
        assert "dread_file='labels.csv'" in mk.cell2(name, "labels.csv", True)
    assert "dread_provenance_ok=True" in mk.cell2("n0", "labels.csv", True)
    assert mk.DREAD not in mk.metadata("u", "n3", "labels.csv")["dataset_sources"]


# ---------------------------------------------------------------- series and frame

def test_series_choice_rule():
    s = pd.DataFrame({
        "StudyInstanceUID": ["a"] * 4,
        "SeriesInstanceUID": ["s1", "s2", "s3", "s4"],
        "Fluid_Sensitive": [0, 1, 1, 1],
        "Fat_Suppression": [0, 0, 1, 1],
        "Anatomical_Plane": ["Sagittal"] * 4,
        "n_slices": [30, 30, 20, 40],
    })
    ch = series.choose(s)
    assert ch.loc[0, "SeriesInstanceUID"] == "s4" and ch.loc[0, "tier"] == 2


@pytest.mark.parametrize("label,plane", [("AX", "axial"), ("Axial", "axial"), ("TRA", "axial"),
                                         ("SAG", "sagittal"), ("Coronal", "coronal"), ("xyz", None)])
def test_norm_plane(label, plane):
    assert series.norm_plane(label) == plane


@pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
def test_frame_ops_recover_canonical(plane):
    rng = np.random.default_rng(1)
    canon = rng.random((5, 7))
    tr = np.array(dicom_io.TARGET[plane]["along_row"], float)
    tc = np.array(dicom_io.TARGET[plane]["down_col"], float)
    for transpose in (False, True):
        for fr in (1, -1):
            for fc in (1, -1):
                img = canon.copy()
                r, c = tr.copy(), tc.copy()
                if fc == -1:
                    img = img[:, ::-1]
                    r = -r
                if fr == -1:
                    img = img[::-1, :]
                    c = -c
                if transpose:
                    img = img.T
                    r, c = c, r
                back = dicom_io.apply_frame(img, dicom_io.frame_ops(list(r) + list(c), plane))
                assert np.allclose(back, canon), (plane, transpose, fr, fc)


def test_order_slices_counts_duplicates_and_uses_slice_normal():
    iop = [0, 1, 0, 0, 0, -1]  # sagittal
    hs = [dicom_io.SliceHeader(path=f"{i}", ipp=[float(x), 0, 0], iop=iop, instance=10 - i, rows=4, cols=4,
                               spacing=[1, 1]) for i, x in enumerate([5, 1, 3, 3])]
    ordered, src, dup = dicom_io.order_slices(hs, "sagittal")
    assert src == "position" and dup == 1
    assert [h.ipp[0] for h in ordered] == [1, 3, 3, 5]


def test_monochrome1_uses_stored_range():
    class DS:
        PhotometricInterpretation = "MONOCHROME1"
        BitsStored = 12
        PixelRepresentation = 0
        RescaleSlope = None
        RescaleIntercept = None

    out = dicom_io._rescale(DS(), np.array([[0, 4095], [100, 200]], dtype=np.uint16))
    assert out[0, 0] == 4095 and out[0, 1] == 0 and out[1, 0] == 3995


def test_nearest_index_and_geometry():
    assert list(dicom_io.nearest_index(3, 5)) == [0, 0, 1, 2, 2]
    assert dicom_io.geometry_laterality(-80) == "R" and dicom_io.geometry_laterality(80) == "L"
    assert dicom_io.geometry_laterality(5) is None


# ---------------------------------------------------------------- labels

def _pil(uids, verdicts, conf):
    df = pd.DataFrame({"StudyInstanceUID": uids})
    for f in labels.FINDINGS:
        df[f + "_verdict"] = verdicts
        df[f + "_confidence"] = conf
    return df


def test_label_conversion_and_mean():
    uids = ["u1", "u2", "u3"]
    t1 = labels.convert("pilkwang", _pil(uids, ["YES", "NO", "UNK"], [0.9, 0.8, 0.05]))
    assert t1.layout == "verdict+confidence" and t1.cells["unk"] == 12
    assert np.allclose(t1.prob["ACL"].values, [0.9, 0.2, 0.0])
    num = pd.DataFrame({"StudyInstanceUID": uids, **{f: [1.0, 0.0, 0.5] for f in labels.FINDINGS}})
    t2 = labels.convert("steven_v2", num)
    m = labels.mean_label([t1, t2], uids)
    assert np.allclose(m["ACL"].values, [0.95, 0.1, 0.25])
    w = labels.weights(m, t1)
    assert np.allclose(w["ACL"].values, [1.0, 1.0, labels.UNK_WEIGHT])
    b = labels.binary(pd.DataFrame({f: [0.7, 0.5, 0.2] for f in labels.FINDINGS}, index=uids))
    assert b["ACL"].tolist()[0] == 1.0 and np.isnan(b["ACL"].tolist()[1]) and b["ACL"].tolist()[2] == 0.0


def test_pilkwang_real_layout_uses_verdict_and_confidence():
    # the real table: <finding> (derived number), <finding>__conf, <finding>__verdict
    uids = ["u1", "u2", "u3", "u4"]
    df = pd.DataFrame({"StudyInstanceUID": uids})
    for f in labels.FINDINGS:
        df[f] = [1, 0, 0, 1]
        df[f + "__conf"] = [0.9, 0.8, 0.05, 0.7]
        df[f + "__verdict"] = ["YES", "NO", "UNK", "YES"]
    t = labels.convert("pilkwang", df)
    assert t.layout == "verdict+confidence" and t.unk is not None and t.cells["unk"] == 12
    assert np.allclose(t.prob["ACL"].values, [0.9, 0.2, 0.0, 0.7])
    assert t.checks["value_verdict_agree"] == 1.0 and np.isclose(t.checks["conf_mean_no"], 0.8)
    # a confidence that is P(present) (low for NO) would be inverted by rule 2: refused, with the checks
    for f in labels.FINDINGS:
        df[f + "__conf"] = [0.9, 0.1, 0.05, 0.8]
    with pytest.raises(labels.SchemaError) as e:
        labels.convert("pilkwang", df)
    assert e.value.checks["conf_mean_no"] < 0.5


def test_label_conversion_refuses_bad_tables(tmp_path):
    uids = ["u1", "u2"]
    out_of_range = pd.DataFrame({"StudyInstanceUID": uids, **{f: [1.5, -1.0] for f in labels.FINDINGS}})
    with pytest.raises(labels.SchemaError):
        labels.convert("t", out_of_range)
    num_conf = pd.DataFrame({"StudyInstanceUID": uids, **{f: [1.0, 0.0] for f in labels.FINDINGS},
                             **{f + "_conf": [0.9, 0.9] for f in labels.FINDINGS}})
    with pytest.raises(labels.SchemaError):
        labels.convert("t", num_conf)
    garbage = _pil(uids, ["maybe", "perhaps"], [0.9, 0.9])
    with pytest.raises(labels.SchemaError):
        labels.convert("t", garbage)
    with pytest.raises(labels.SchemaError):
        labels.convert("t", pd.DataFrame({"StudyInstanceUID": ["x"], "foo": [1]}))


def test_read_table_keeps_na_strings_as_unk(tmp_path):
    p = tmp_path / "t.csv"
    _pil(["u1", "u2"], ["NA", "YES"], [0.05, 0.9]).to_csv(p, index=False)
    t = labels.convert("pilkwang", labels.read_table(p))
    assert t.cells["unk"] == 12 and t.prob.loc["u1", "ACL"] == 0.0 and t.unk.loc["u1", "ACL"]


def test_find_label_columns_and_cap():
    cols = ["StudyInstanceUID", "Report"] + [f.replace("'", "") for f in labels.FINDINGS]
    m = labels.find_label_columns(cols)
    assert m["Baker's"] == "Bakers"
    b = pd.DataFrame({f: [1, 0, 0, 0] * 10 for f in labels.FINDINGS})
    assert labels.pos_weight_cap_upper(b) == 20.0
    b["Fracture"] = [1] + [0] * 39
    assert labels.pos_weight_cap_upper(b) == 39.0


def test_macro_auc_empty_findings_is_nan():
    y = pd.DataFrame({f: [0, 1, 0, 1] for f in labels.FINDINGS})
    v, _ = labels.macro_auc(y, y.astype(float), [])
    assert np.isnan(v)


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(float)
    s = rng.random(200) + y * 0.3
    assert abs(labels.auc(y, s) - roc_auc_score(y, s)) < 1e-9


def _table(name, prob):
    return labels.Table(name=name, prob=prob, unk=None, layout="numeric")


def test_agreement_gate():
    rng = np.random.default_rng(0)
    n = 400
    uids = [f"s{i}" for i in range(n)]
    truth = rng.random((n, 12)) < 0.4

    def noisy(sd):
        return pd.DataFrame(np.clip(truth + rng.normal(0, sd, truth.shape), 0, 1), index=uids, columns=labels.FINDINGS)

    core = [_table("a", noisy(0.3)), _table("b", noisy(0.3)), _table("c", noisy(0.3))]
    good = labels.agreement_gate(core, _table("d", noisy(0.3)), uids)
    assert good["passed"]
    rand = labels.agreement_gate(core, _table("r", pd.DataFrame(rng.random((n, 12)), index=uids, columns=labels.FINDINGS)), uids)
    assert not rand["passed"]
    copy = labels.agreement_gate(core, _table("copy", core[0].prob.copy()), uids)
    assert not copy["passed"]  # Spearman 1.0 with one table marks a blend
    empty = labels.agreement_gate(core, _table("e", pd.DataFrame(np.nan, index=uids, columns=labels.FINDINGS)), uids)
    assert not empty["passed"]


# ---------------------------------------------------------------- groups and folds

def _studies(extra_rows=()):
    rows = []
    spec = [("ENGLISH", "A|1|1.5", 20), ("GERMAN", "B|X|3.0", 20), ("DUTCH", "B|X|3.0", 20),
            ("FRENCH", "C|1|1.5", 20), ("SPANISH", "D|1|3.0", 20), ("ITALIAN", "E|1|3.0", 20),
            ("POLISH", "F|1|1.5", 20)]
    k = 0
    for lang, fp, n in spec:
        for _ in range(n):
            rows.append({"uid": f"s{k}", "language": lang, "lang_conf": 0.99, "report_len": 200,
                         "fingerprint": fp, "manufacturer": fp.split("|")[0], "has_report": True})
            k += 1
    rows.append({"uid": "u1", "language": "ENGLISH", "lang_conf": 0.4, "report_len": 200, "fingerprint": "C|1|1.5",
                 "manufacturer": "C", "has_report": True})
    rows.append({"uid": "u2", "language": None, "lang_conf": 0.0, "report_len": 5, "fingerprint": "Z|9|1.0",
                 "manufacturer": "Z", "has_report": True})
    rows.extend(extra_rows)
    return pd.DataFrame(rows).set_index("uid")


def test_groups_merge_and_unsure():
    g, summ = groups.build_groups(_studies())
    assert g.loc["s20", "g_lang"] == g.loc["s40", "g_lang"]
    assert summ["g_lang_groups"] == 6 and summ["language_merges"] == 1
    assert g.loc["u1", "g_lang"] == "FRENCH" and not g.loc["u1", "scored"] and g.loc["u1", "in_training"]
    assert pd.isna(g.loc["u2", "g_lang"]) and not g.loc["u2", "in_training"]
    agg._check(summ)  # summary is safe as it stands


def test_one_study_cannot_merge_two_groups():
    stray = [{"uid": "x1", "language": "POLISH", "lang_conf": 0.99, "report_len": 200, "fingerprint": "A|1|1.5",
              "manufacturer": "A", "has_report": True}]
    g, summ = groups.build_groups(_studies(stray))
    assert g.loc["s0", "g_lang"] != g.loc["s120", "g_lang"]  # ENGLISH and POLISH stay apart


def test_folds_group_disjoint_s1_sizes_and_logo():
    g, _ = groups.build_groups(_studies())
    tr = g[g["in_training"]]
    f, info = folds.outer_folds(tr["g_lang"])
    assert info["scheme"] == "group_5fold" and info["n_folds"] == 5
    for k in range(5):
        assert not (set(tr.loc[f == k, "g_lang"]) & set(tr.loc[f != k, "g_lang"]))
        inner = folds.inner_folds(tr["g_lang"], f != k)
        for j in range(inner.max() + 1):
            a = set(tr.loc[inner[tr.index] == j, "g_lang"])
            b = set(tr.loc[(inner[tr.index] != j) & (inner[tr.index] >= 0), "g_lang"])
            assert not (a & b)
    s1 = folds.seen_site_folds(f, [("s0", "s1")])
    assert (s1.value_counts().sort_index() == f.value_counts().sort_index()).all()
    assert s1["s0"] == s1["s1"]
    few = tr[tr["g_lang"].isin(sorted(tr["g_lang"].unique())[:4])]
    f4, info4 = folds.outer_folds(few["g_lang"])
    assert info4["scheme"] == "leave_one_group_out" and info4["n_folds"] == 4
    three = tr[tr["g_lang"].isin(sorted(tr["g_lang"].unique())[:3])]
    assert folds.outer_folds(three["g_lang"])[1]["scheme"] == "not_testable"
