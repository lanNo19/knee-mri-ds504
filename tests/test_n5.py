"""N5 on a synthetic cache with small random encoders (CPU): row alignment, flips, missing planes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from knee import agg, paths, safe_log  # noqa: E402

UIDS = [f"1.2.826.0.1.3680043.8.498.{9000 + i}" for i in range(5)]
LAT = ["R", "L", None, "R", None]
FAILED = {("coronal", 3), ("axial", 4)}  # (plane, row) without a cache
SMALL = {"dinov2_vits14": {"depth": 1}, "convnext_tiny": {"depths": (1, 1, 1, 1)}, "vit_s16": {"depth": 1}}


def _volume(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros((32, 224, 224), np.uint8)
    v[:, :, :112] = rng.integers(0, 120, size=(32, 224, 112))  # left half darker: mirroring changes it
    v[:, :, 112:] = rng.integers(100, 255, size=(32, 224, 112))
    return v


def _write_cache(root, plane: str, tables_differ: bool = False):
    d = root / "notebooks" / "owner" / f"knee-n4-{plane}" / "private" / f"n4_{plane}"
    d.mkdir(parents=True)
    n = len(UIDS)
    mm = np.memmap(d / f"cache_{plane}.u8", dtype=np.uint8, mode="w+", shape=(n, 32, 224, 224))
    vols = {i: _volume(100 * PLANE_IDX[plane] + i) for i in range(n)}
    vols[1] = vols[0].copy()  # same input -> same features in rows 0 and 1
    vols[2] = vols[0][:, :, ::-1].copy()  # unresolved study = mirror of study 0
    ok = []
    for i in range(n):
        good = (plane, i) not in FAILED
        if good:
            mm[i] = vols[i]
        ok.append(good)
    mm.flush()
    (d / f"cache_{plane}_manifest.json").write_text(json.dumps({"plane": plane, "shape": [n, 32, 224, 224],
                                                                "dtype": "uint8", "file": f"cache_{plane}.u8"}))
    pd.DataFrame({"StudyInstanceUID": UIDS, "row": range(n), "ok": ok}).to_csv(d / f"cache_{plane}_studies.csv", index=False)
    lat = list(LAT)
    if tables_differ and plane == "axial":
        lat[0] = "L"
    pd.DataFrame({"StudyInstanceUID": UIDS, "laterality": lat}).to_csv(d / "study_table.csv", index=False)


PLANE_IDX = {"sagittal": 0, "coronal": 1, "axial": 2}


@pytest.fixture()
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "WORKING", tmp_path / "working")
    return tmp_path


def test_n5_rows_flips_and_missing_planes(work, capfd):
    from knee.nb import n5

    for p in n5.PLANES:
        _write_cache(work / "input", p)
    s = n5.run(root=work / "input", devices=["cpu"], pretrained=False, model_overrides=SMALL,
               studies_per_batch=2, quiet=True)
    printed = capfd.readouterr()
    for u in UIDS:
        assert u not in printed.out + printed.err
        assert u not in (paths.WORKING / "agg_n5.json").read_text()
    assert all(v for k, v in s["checks"].items() if k != "gpu_agreement")
    od = paths.WORKING / "private" / "n5"
    idx = pd.read_csv(od / "index.csv")
    assert idx["usable"].tolist() == [True, True, True, False, False]
    assert idx["flip_row"].tolist() == [-1, -1, 0, -1, 1]
    for name in SMALL:
        for p in n5.PLANES:
            a = np.load(od / f"emb_{name}_{p}.npy")
            assert a.shape == (5, 32, 768) and a.dtype == np.float16
            np.testing.assert_allclose(a[0].astype(np.float32), a[1].astype(np.float32), atol=1e-2)  # rows aligned
            assert not np.array_equal(a[0], a[2])  # the mirrored volume differs unflipped
            for q, r in FAILED:
                if q == p:
                    assert not a[r].any()
        for p in n5.FLIP_PLANES:
            a = np.load(od / f"emb_{name}_{p}.npy")
            f = np.load(od / f"emb_{name}_{p}_flip.npy")
            assert f.shape == (2, 32, 768)
            # flipping the mirrored study gives study 0 back
            np.testing.assert_allclose(f[0].astype(np.float32), a[0].astype(np.float32), atol=2e-2)
        assert not (od / f"emb_{name}_sagittal_flip.npy").exists()
        assert not np.load(od / f"emb_{name}_axial_flip.npy")[1].any()  # study 4 has no axial cache
        assert np.load(od / f"emb_{name}_coronal_flip.npy")[1].any()
    man = json.loads((od / "manifest.json").read_text())
    assert man["encoders"]["vit_s16"]["mean_std"] == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    assert s["index_sha16"] in printed.out
    agg._check(json.loads((paths.WORKING / "agg_n5.json").read_text()))


def test_n5_refuses_caches_from_different_study_tables(work):
    from knee.nb import n5

    for p in n5.PLANES:
        _write_cache(work / "input", p, tables_differ=True)
    with pytest.raises(safe_log.StepFailed):
        n5.run(root=work / "input", devices=["cpu"], pretrained=False, model_overrides=SMALL, quiet=True)


def test_encoder_normalisation_and_features():
    from knee import encoders

    for name, spec in encoders.SPECS.items():
        enc = encoders.Encoder(name, "cpu", pretrained=False, overrides=SMALL[name])
        assert tuple(map(tuple, enc.mean_std)) == spec.norm
        f = enc(torch.from_numpy(_volume(1)[:4]))
        assert f.shape == (4, encoders.FEATURE_DIM) and f.dtype == torch.float16


def test_montage_draws_flagged_pairs_without_ids(work, monkeypatch):
    from knee.nb import montage

    for p in ("sagittal", "coronal", "axial"):
        _write_cache(work / "input", p)
    pilot = work / "input" / "notebooks" / "owner" / "knee-n6p" / "private" / "n6p"
    pilot.mkdir(parents=True)
    pd.DataFrame({"pair_id": [0, 1], "uid_a": [UIDS[0], UIDS[2]], "uid_b": [UIDS[1], UIDS[3]],
                  "cosine": [0.999, 0.985]}).to_csv(pilot / "pairs.csv", index=False)
    monkeypatch.setattr(paths, "INPUT", work / "input")
    out = montage.run(n=3, pairs=2, quiet=True)
    od = paths.WORKING / "private" / "montage"
    assert out["pairs"] == "2 pairs" and (od / "pairs.png").exists()
    rows = (od / "pairs_rows.csv").read_text()
    assert rows.splitlines()[0] == "pair_id,row_a,row_b" and not any(u in rows for u in UIDS)
