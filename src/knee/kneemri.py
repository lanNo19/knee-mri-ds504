"""KneeMRI, the second dataset (S5): cache, embeddings and the in-domain test.

Runs on Ablay's laptop (Apple GPU, --device mps), the lab cluster (cuda) or a
Kaggle notebook (from the unmodified public copy), with the same code as RSNA.
Data: Zenodo record 14789903 (Stajduhar et al. 2017, CC BY-NC-ND 4.0):
volumetric_data.7z (one pickled sagittal volume per exam) and metadata.csv
(917 exams: examId, seriesNo, aclDiagnosis 0 intact / 1 partial / 2 complete,
kneeLR, roi*, volumeFilename). Our copies stay private and unmodified.

  python -m knee.kneemri cache --src ~/data/kneemri --out ~/data/kneemri_cache
  python -m knee.kneemri embed --cache ~/data/kneemri_cache --out ~/data/kneemri_emb --device mps
  python -m knee.kneemri indomain --emb ~/data/kneemri_emb --configs configs/consensus.json --out ~/kaggle_out/kneemri

cache: the RSNA rules: 32 slices by nearest index, intensity clipped at the
  volume's 0.5/99.5 percentiles to 0-255, shorter side to 224 and a centre crop;
  no mirroring (laterality is known, and mirroring a sagittal stack only
  reverses slice order, which order-free pooling ignores).
indomain: stratified 5-fold CV by exam (there is no patient ID); each encoder's
  consensus config from Stage A, planes forced to sagittal, no tuning on
  KneeMRI, and the same number of optimiser steps as the RSNA refit. Label:
  partial or complete ACL tear; secondary: complete tear vs the rest. The
  positive control needs an in-domain ACL AUC of at least 0.75 per encoder.
  Intervals: a study-level bootstrap (one hospital, so they cover patients).
Output: agg_kneemri.json (summary numbers only) for the chat and N9.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dicom_io, encoders, heads, intervals, metrics

POSITIVE_CONTROL_MIN = 0.75
N_FOLDS = 5


def _load_volume(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        try:
            v = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            v = pickle.load(f, encoding="latin1")
    return np.asarray(v, dtype=np.float32)


def prepare(vol: np.ndarray) -> np.ndarray:
    """(slices, H, W) -> uint8 [32, 224, 224] with the RSNA rules (no mirroring)."""
    v = vol[dicom_io.nearest_index(vol.shape[0])]
    u8, _ = dicom_io.to_uint8(v)
    return np.stack([dicom_io.resize_crop(s) for s in u8])


def _extract(src: Path) -> Path:
    vols = [p for p in src.rglob("*.pck")]
    if vols:
        return src
    archive = src / "volumetric_data.7z"
    if not archive.exists():
        raise FileNotFoundError("volumetric_data.7z not found")
    import py7zr

    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(src / "extracted")
    return src / "extracted"


def cache(src: str, out: str, keep_extracted: bool = False) -> dict:
    src_p, out_p = Path(src).expanduser(), Path(out).expanduser()
    out_p.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(src_p / "metadata.csv")
    vol_dir = _extract(src_p)
    files = {p.name: p for p in vol_dir.rglob("*.pck")}
    n = len(meta)
    mm = np.lib.format.open_memmap(out_p / "cache_sagittal.npy", mode="w+", dtype=np.uint8,
                                   shape=(n, dicom_io.N_SLICES, dicom_io.SIZE, dicom_io.SIZE))
    ok = np.zeros(n, bool)
    for i, fn in enumerate(meta["volumeFilename"].astype(str)):
        p = files.get(Path(fn).name)
        if p is None:
            continue
        try:
            mm[i] = prepare(_load_volume(p))
            ok[i] = True
        except Exception:  # noqa: BLE001 - counted
            pass
    mm.flush()
    index = pd.DataFrame({"row": np.arange(n), "examId": meta["examId"], "acl": (meta["aclDiagnosis"] >= 1).astype(int),
                          "acl_complete": (meta["aclDiagnosis"] == 2).astype(int), "ok": ok})
    index.to_csv(out_p / "index.csv", index=False)
    if not keep_extracted and (src_p / "extracted").exists():
        import shutil

        shutil.rmtree(src_p / "extracted")
    s = {"exams": n, "cached": int(ok.sum()), "acl_positive_share": round(float(index["acl"].mean()), 3)}
    (out_p / "cache_summary.json").write_text(json.dumps(s, indent=1))
    return s


def embed(cache_dir: str, out: str, device: str = "cpu", encoder_names=tuple(encoders.SPECS),
          pretrained: bool = True, overrides: dict | None = None) -> dict:
    import torch

    cd, od = Path(cache_dir).expanduser(), Path(out).expanduser()
    od.mkdir(parents=True, exist_ok=True)
    mm = np.load(cd / "cache_sagittal.npy", mmap_mode="r")
    index = pd.read_csv(cd / "index.csv")
    rows = index.index[index["ok"]].to_numpy()
    s = {}
    for name in encoder_names:
        enc = encoders.Encoder(name, device, pretrained=pretrained, overrides=(overrides or {}).get(name))
        a = np.zeros((len(index), mm.shape[1], encoders.FEATURE_DIM), np.float16)
        t0 = time.time()
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            x = torch.from_numpy(np.ascontiguousarray(mm[b])).to(device)
            nb, ns = x.shape[:2]
            a[b] = enc(x.reshape(nb * ns, *x.shape[2:])).reshape(nb, ns, -1).cpu().numpy()
        np.save(od / f"emb_{name}_sagittal.npy", a)
        s[name] = {"seconds": round(time.time() - t0, 1), "mean_std": enc.mean_std}
    index.to_csv(od / "index.csv", index=False)
    (od / "embed_summary.json").write_text(json.dumps(s, indent=1))
    return s


def _boot_auc(p, y, n_boot=2000, seed=0, q=None):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        vals.append(q(w) if q else metrics.auc(p, y, w)[0])
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    return float(lo), float(hi)


def indomain(emb_dir: str, configs: str, out: str, device: str = "cpu", encoder_names=tuple(encoders.SPECS),
             seed: int = 0, n_boot: int = 2000) -> dict:
    """S5: out-of-fold ACL AUC per encoder with each encoder's consensus config (sagittal only)."""
    from sklearn.model_selection import StratifiedKFold

    ed, od = Path(emb_dir).expanduser(), Path(out).expanduser()
    od.mkdir(parents=True, exist_ok=True)
    con = json.loads(Path(configs).expanduser().read_text())
    index = pd.read_csv(ed / "index.csv")
    rows = np.flatnonzero(index["ok"].to_numpy())
    y_all = index["acl"].to_numpy(float)
    yc_all = index["acl_complete"].to_numpy(float)
    folds = np.full(len(index), -1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(rows, y_all[rows])):
        folds[rows[te]] = k
    soft = np.full((len(index), heads.N_OUT), np.nan, np.float32)
    soft[:, 0] = y_all  # ACL is the first finding; the other 11 are masked (weight 0)
    weight = np.zeros_like(soft)
    weight[:, 0] = 1.0
    binary = np.where(np.isnan(soft), np.nan, soft)
    res, preds = {}, {}
    for name in encoder_names:
        if name not in con:
            res[name] = {"status": "no consensus config"}
            continue
        c = con[name]
        cfg = heads.with_l1(heads.config_from(c["config"]), c["config"]["pooling"], "sag")
        emb = {"sagittal": np.load(ed / f"emb_{name}_sagittal.npy", mmap_mode="r")}
        bank = heads.Bank(emb, soft, weight, binary, device=device)
        P = np.full(len(index), np.nan)
        for k in range(N_FOLDS):
            tr, te = rows[folds[rows] != k], rows[folds[rows] == k]
            steps_per_epoch = math.ceil(len(tr) / cfg.batch_size)
            epochs = max(1, int(round(int(c["refit_steps"]) / steps_per_epoch))) if c.get("refit_steps") else int(c["E"])
            stats = bank.stats(tr)
            r = heads.fit(bank, cfg, tr, epochs=epochs, stats=stats, forbidden=te, seed=0)
            P[te] = heads.predict(bank, r.model, te, cfg, stats, use_flip=False)[:, 0]
        preds[name] = P
        a, _ = metrics.auc(P[rows], y_all[rows])
        lo, hi = _boot_auc(P[rows], y_all[rows], n_boot, seed)
        ac, _ = metrics.auc(P[rows], yc_all[rows])
        res[name] = {"acl_auc": round(a, 4), "acl_auc_ci": [round(lo, 4), round(hi, 4)],
                     "complete_vs_rest_auc": round(ac, 4), "epochs": epochs,
                     "positive_control_passed": bool(a >= POSITIVE_CONTROL_MIN)}
    out_s = {"dataset": "KneeMRI (Stajduhar et al. 2017, CC BY-NC-ND 4.0)", "exams_used": int(len(rows)),
             "acl_positive_share": round(float(y_all[rows].mean()), 3), "folds": N_FOLDS, "by_encoder": res,
             "in_domain_acl_auc": {k: v.get("acl_auc") for k, v in res.items()}}
    A, B = encoders.MAIN
    if A in preds and B in preds:
        pa, pb, yy = preds[A][rows], preds[B][rows], y_all[rows]

        def dstat(w):
            return metrics.auc(pa, yy, w)[0] - metrics.auc(pb, yy, w)[0]

        d = dstat(np.ones(len(rows)))
        lo, hi = _boot_auc(None, yy, n_boot, seed, q=dstat)
        out_s["delta_dinov2_minus_convnext"] = {"estimate": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                                                "category": intervals.decide(lo, hi, a="DINOv2 better",
                                                                             b="ConvNeXt better"),
                                                "interval": "study-level bootstrap (one hospital)"}
    (od / "agg_kneemri.json").write_text(json.dumps(out_s, indent=1))
    return out_s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m knee.kneemri")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cache")
    a.add_argument("--src", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("embed")
    b.add_argument("--cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--device", default="cpu")
    c = sub.add_parser("indomain")
    c.add_argument("--emb", required=True)
    c.add_argument("--configs", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.cmd == "cache":
        print(json.dumps(cache(args.src, args.out), indent=1))
    elif args.cmd == "embed":
        print(json.dumps(embed(args.cache, args.out, args.device), indent=1))
    else:
        print(json.dumps(indomain(args.emb, args.configs, args.out, args.device), indent=1))


if __name__ == "__main__":
    main()
