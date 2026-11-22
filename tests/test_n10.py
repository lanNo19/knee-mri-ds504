"""N10 on a synthetic test split: caches, bundled checkpoint, fold heads, submission (CPU, no download)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from knee import encoders, heads, paths  # noqa: E402
from knee.labels import FINDINGS, UID  # noqa: E402
from tests.synth import make_competition  # noqa: E402


@pytest.fixture()
def comp_env(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    comp = inp / paths.COMPETITION
    comp.mkdir(parents=True)
    meta = make_competition(comp, n_per_lang=1, n_slices=12, size=48)
    # the test split: the same files under test names
    shutil.copytree(comp / "train_series", comp / "test_series")
    shutil.copy(comp / "train_series.csv", comp / "test_series.csv")
    uids = sorted(pd.read_csv(comp / "test_series.csv")[UID].astype(str).unique())
    pd.DataFrame({UID: uids, **{f: 0.5 for f in FINDINGS}}).to_csv(comp / "sample_submission.csv", index=False)
    # the offline bundle: a full-size checkpoint (random weights; nothing is downloaded) and its manifest
    ck = inp / "knee-bundle" / "private" / "bundle" / "checkpoints"
    ck.mkdir(parents=True)
    enc = encoders.Encoder("dinov2_vits14", "cpu", pretrained=False)
    torch.save(enc.model.state_dict(), ck / "dinov2_vits14.pth")
    import hashlib

    sha = hashlib.sha256((ck / "dinov2_vits14.pth").read_bytes()).hexdigest()
    (ck.parent / "manifest.json").write_text(json.dumps({"commit": "abcdef1234567890",
                                                         "files_sha256": {"dinov2_vits14.pth": sha}}))
    # N0's summary: the training prevalence for the per-study fallback
    n0 = inp / "knee-n0"
    n0.mkdir()
    (n0 / "agg_n0.json").write_text(json.dumps({"labels": {"prevalence_training": {f: 0.2 for f in FINDINGS}}}))
    # two Stage A units
    cfg = heads.Config(pooling="gated", planes="all")
    for k in (0, 1):
        ud = inp / "knee-n6a" / "private" / "n6a" / "units" / f"dinov2_vits14_f{k}"
        ud.mkdir(parents=True)
        torch.manual_seed(k)
        torch.save(heads.make_head(cfg, encoders.FEATURE_DIM, 3).state_dict(), ud / "head.pt")
        torch.save({p: (torch.zeros(encoders.FEATURE_DIM), torch.ones(encoders.FEATURE_DIM)) for p in heads.PLANES},
                   ud / "stats.pt")
        (ud / "config.json").write_text(json.dumps({"encoder": "dinov2_vits14", "fold": k, "config": cfg.to_dict()}))
    monkeypatch.setattr(paths, "INPUT", inp)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "working")
    monkeypatch.delenv("KNEE_DATA_ROOT", raising=False)
    return {"uids": uids, **meta}


def test_n10_writes_a_complete_submission(comp_env, capfd):
    from knee.nb import n10

    s = n10.run(workers=2, quiet=True)
    printed = capfd.readouterr()
    text = printed.out + printed.err + (paths.WORKING / "agg_n10.json").read_text()
    assert not [u for u in comp_env["uids"] if u in text]
    sub = pd.read_csv(paths.WORKING / "submission.csv")
    assert list(sub.columns) == [UID] + FINDINGS
    assert sorted(sub[UID].astype(str)) == comp_env["uids"]
    v = sub[FINDINGS].to_numpy()
    assert np.isfinite(v).all() and ((v > 0) & (v < 1)).all()
    assert s["submission"]["layout"] == "wide" and s["folds"] == [0, 1]
    assert s["bundle_commit"] == "abcdef123456" and s["failed_share"] == 0.0
    assert set(p.name for p in paths.WORKING.iterdir() if p.is_file()) <= {"submission.csv", "agg_n10.json", "run.json"}
    assert not (paths.WORKING / "private").exists() or not any((paths.WORKING / "private").rglob("*.u8"))  # caches in /tmp


def test_n10_canary_submits_prevalence_and_fails_on_purpose(comp_env):
    from knee import safe_log
    from knee.nb import n10

    s = n10.run(canary=True, workers=2, quiet=True)
    sub = pd.read_csv(paths.WORKING / "submission.csv")
    assert np.allclose(sub[FINDINGS].to_numpy(), 0.2) and s["folds"] == []
    # one study's coronal series all go missing: 1 of 7 studies (> 1%) cannot be decoded
    comp = paths.INPUT / paths.COMPETITION
    ser = pd.read_csv(comp / "test_series.csv")
    cor = ser[ser["Anatomical_Plane"].astype(str).str.lower().str.startswith("cor")]
    study = cor[UID].iloc[0]
    for se in cor.loc[cor[UID] == study, "SeriesInstanceUID"]:
        shutil.rmtree(comp / "test_series" / str(study) / str(se))
    with pytest.raises(safe_log.StepFailed):
        n10.run(canary=True, workers=2, quiet=True)
    out = json.loads((paths.WORKING / "agg_n10.json").read_text())
    assert out["failed_share"] > 0.01


def test_unknown_submission_layout_is_refused():
    from knee.nb import n10

    with pytest.raises(ValueError):
        n10.write_submission(pd.DataFrame({"row_id": ["a_ACL"], "target": [0.5]}), np.array(["a"]),
                             np.zeros((1, 12)), paths.WORKING / "x.csv")
