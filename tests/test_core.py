"""Unit tests on synthetic data: safe logging, series choice, frame, labels, groups, folds."""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import pytest

from knee import dicom_io, folds, groups, labels, safe_log, series

FAKE_UID = "1.2.826.0.1.3680043.8.498.424242424242"


def test_step_hides_error_message(capsys):
    with pytest.raises(safe_log.StepFailed) as ei:
        with safe_log.step("t", quiet=False):
            raise KeyError(FAKE_UID)
    out = capsys.readouterr().out
    assert FAKE_UID not in out and FAKE_UID not in str(ei.value)
    assert "KeyError" in out and "test_core.py" in out


def test_step_quiet_hides_prints(capsys):
    with safe_log.step("q", quiet=True):
        print("secret report text " + FAKE_UID)
        sys.stderr.write("stderr " + FAKE_UID + "\n")
    out = capsys.readouterr()
    assert FAKE_UID not in out.out and FAKE_UID not in out.err
    assert "STEP q: OK" in out.out


def test_warning_and_logging_hide_messages(capsys):
    import logging
    import warnings

    safe_log.install()
    warnings.warn("value " + FAKE_UID)
    logging.getLogger("pydicom").warning("bad value %s", FAKE_UID)
    err = capsys.readouterr().err
    assert FAKE_UID not in err


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
    # tier 2 wins over tier 1; among tier 2, |20-30| == |40-30| -> larger count wins
    assert ch.loc[0, "SeriesInstanceUID"] == "s4" and ch.loc[0, "tier"] == 2


@pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
def test_frame_ops_recover_canonical(plane):
    rng = np.random.default_rng(1)
    canon = rng.random((5, 7))
    tr = np.array(dicom_io.TARGET[plane]["along_row"], float)
    tc = np.array(dicom_io.TARGET[plane]["down_col"], float)
    # Stored variants: transposed and/or flipped, with the matching IOP
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
                ops = dicom_io.frame_ops(list(r) + list(c), plane)
                back = dicom_io.apply_frame(img, ops)
                assert np.allclose(back, canon), (plane, transpose, fr, fc)


def test_nearest_index():
    assert list(dicom_io.nearest_index(3, 5)) == [0, 0, 1, 2, 2]
    idx = dicom_io.nearest_index(30, 32)
    assert idx[0] == 0 and idx[-1] == 29 and len(idx) == 32


def test_geometry_laterality():
    assert dicom_io.geometry_laterality(-80) == "R"
    assert dicom_io.geometry_laterality(80) == "L"
    assert dicom_io.geometry_laterality(5) is None


def test_label_conversion_and_mean():
    uids = ["u1", "u2", "u3"]
    pil = pd.DataFrame({"StudyInstanceUID": uids})
    for f in labels.FINDINGS:
        pil[f + "_verdict"] = ["YES", "NO", "UNK"]
        pil[f + "_confidence"] = [0.9, 0.8, 0.05]
    t1 = labels.convert("pilkwang", pil)
    assert t1.layout == "verdict+confidence"
    assert np.allclose(t1.prob["ACL"].values, [0.9, 0.2, 0.0])
    num = pd.DataFrame({"StudyInstanceUID": uids, **{f: [1.0, 0.0, 0.5] for f in labels.FINDINGS}})
    t2 = labels.convert("steven_v2", num)
    m = labels.mean_label([t1, t2], uids)
    assert np.allclose(m["ACL"].values, [0.95, 0.1, 0.25])
    w = labels.weights(m, t1)
    assert np.allclose(w["ACL"].values, [1.0, 1.0, labels.UNK_WEIGHT])
    b = labels.binary(pd.DataFrame({f: [0.7, 0.5, 0.2] for f in labels.FINDINGS}, index=uids))
    assert b["ACL"].tolist()[0] == 1.0 and np.isnan(b["ACL"].tolist()[1]) and b["ACL"].tolist()[2] == 0.0


def test_schema_error_lists_columns_only():
    bad = pd.DataFrame({"StudyInstanceUID": ["x"], "foo": [1]})
    with pytest.raises(labels.SchemaError):
        labels.convert("t", bad)


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(float)
    s = rng.random(200) + y * 0.3
    assert abs(labels.auc(y, s) - roc_auc_score(y, s)) < 1e-9


def _studies():
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
    # one unsure study with a known scanner, one with an unknown scanner
    rows.append({"uid": "u1", "language": "ENGLISH", "lang_conf": 0.4, "report_len": 200, "fingerprint": "C|1|1.5",
                 "manufacturer": "C", "has_report": True})
    rows.append({"uid": "u2", "language": None, "lang_conf": 0.0, "report_len": 5, "fingerprint": "Z|9|1.0",
                 "manufacturer": "Z", "has_report": True})
    return pd.DataFrame(rows).set_index("uid")


def test_groups_merge_and_unsure():
    g, summ = groups.build_groups(_studies())
    assert g.loc["s20", "g_lang"] == g.loc["s40", "g_lang"]  # GERMAN and DUTCH share scanner B|X only
    assert summ["g_lang_groups"] == 6 and summ["language_merges"] == 1
    assert g.loc["u1", "g_lang"] == "FRENCH" and not g.loc["u1", "scored"] and g.loc["u1", "in_training"]
    assert pd.isna(g.loc["u2", "g_lang"]) and not g.loc["u2", "in_training"]


def test_folds_group_disjoint_and_s1_sizes():
    g, _ = groups.build_groups(_studies())
    tr = g[g["in_training"]]
    f, info = folds.outer_folds(tr["g_lang"])
    assert info["scheme"] == "group_5fold" and info["n_folds"] == 5
    for k in range(5):
        test_groups = set(tr.loc[f == k, "g_lang"])
        train_groups = set(tr.loc[f != k, "g_lang"])
        assert not (test_groups & train_groups)
        inner = folds.inner_folds(tr["g_lang"], f != k)
        for j in range(inner.max() + 1):
            a = set(tr.loc[inner[tr.index] == j, "g_lang"])
            b = set(tr.loc[(inner[tr.index] != j) & (inner[tr.index] >= 0), "g_lang"])
            assert not (a & b)
    s1 = folds.seen_site_folds(f, [("s0", "s1")])
    assert (s1.value_counts().sort_index() == f.value_counts().sort_index()).all()
    assert s1["s0"] == s1["s1"]
