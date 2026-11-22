"""KneeMRI (S5) on synthetic pickled volumes: cache, embeddings (small random encoders), in-domain test."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from knee import agg, kneemri  # noqa: E402

SMALL = {"dinov2_vits14": {"depth": 1}, "convnext_tiny": {"depths": (1, 1, 1, 1)}, "vit_s16": {"depth": 1}}


def test_kneemri_cache_embed_indomain(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "kneemri"
    (src / "vols").mkdir(parents=True)
    n = 60
    diag = rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2])
    for i in range(n):
        v = rng.normal(500, 50, (int(rng.integers(20, 40)), 64, 80))
        v[:, 20:40, 30:50] += 400 * (diag[i] > 0)  # a visible "tear" signal
        with open(src / "vols" / f"{i:04d}.pck", "wb") as f:
            pickle.dump(v.astype(np.uint16), f)
    pd.DataFrame({"examId": np.arange(n), "seriesNo": 1, "aclDiagnosis": diag, "kneeLR": rng.integers(0, 2, n),
                  "volumeFilename": [f"{i:04d}.pck" for i in range(n)]}).to_csv(src / "metadata.csv", index=False)
    c = kneemri.cache(str(src), str(tmp_path / "cache"))
    assert c["exams"] == n and c["cached"] == n
    mm = np.load(tmp_path / "cache" / "cache_sagittal.npy", mmap_mode="r")
    assert mm.shape == (n, 32, 224, 224) and mm.dtype == np.uint8
    kneemri.embed(str(tmp_path / "cache"), str(tmp_path / "emb"), "cpu", pretrained=False, overrides=SMALL)
    cfg = {"pooling": "mean", "planes": "all", "lr": 3e-3, "weight_decay": 3e-4, "dropout": 0.0,
           "attn_hidden": 192, "pos_cap": 4.47, "batch_size": 16}
    con = {e: {"config": cfg, "E": 10, "refit_steps": 30} for e in SMALL}
    (tmp_path / "consensus.json").write_text(json.dumps(con))
    s = kneemri.indomain(str(tmp_path / "emb"), str(tmp_path / "consensus.json"), str(tmp_path / "out"), "cpu",
                         n_boot=200)
    out = json.loads((tmp_path / "out" / "agg_kneemri.json").read_text())
    agg._check(out)  # N9 runs the same check
    assert set(out["in_domain_acl_auc"]) == set(SMALL)
    for v in s["by_encoder"].values():
        # 30 RSNA steps (E = 10, so 3 steps per RSNA epoch) on about 48 exams in batches of 16
        assert v["steps"] == 30 and v["epochs"] == 10 and v["final_lr"] > 0
    assert "delta_dinov2_minus_convnext" in out
