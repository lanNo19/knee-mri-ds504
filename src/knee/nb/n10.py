"""N10: inference on the competition's test studies and submission.csv (internet OFF; S4).

Inputs: the competition, the offline bundle dataset (private/bundle: wheels and
encoder checkpoints) and the Stage A outputs (N6a, or the one-unit smoke for the
canary): for the chosen encoder, each S2 fold unit's head.pt, stats.pt and config.json.
Steps, with the same code as training:
  1. study table and series choice for the test split (N0/N4 code);
  2. one uint8 cache per plane (N4's cache.build: decode, order, frame, laterality,
     intensity, 224 crop, 32 slices);
  3. slice embeddings with the encoder loaded from the bundle checkpoint (N5 code),
     plus flipped coronal/axial slices for unresolved laterality;
  4. each fold head predicts (flip-averaged for unresolved laterality); a plane
     whose cache failed gets that fold's training mean features (z = 0);
     the submission is the mean over the attached fold heads.
Writes submission.csv at the output root (Kaggle requires it; Claude never opens
it), private/n10/ (caches, embeddings, per-fold predictions) and agg_n10.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, cache, encoders, heads, paths, study_table
from ..labels import FINDINGS, UID
from . import common, n5

PLANES = heads.PLANES


def _find_bundle_ckpt(encoder: str, root: Path | None) -> Path:
    p = paths.find_file(f"{encoder}.pth", root, max_depth=7)
    if p is None:
        raise FileNotFoundError("offline bundle not attached")
    return p


def _units(encoder: str, root: Path | None) -> dict:
    base = Path(root) if root else paths.INPUT
    out = {}
    for c in sorted(base.glob(f"**/units/{encoder}_f*/config.json")):
        u = json.loads(c.read_text())
        k = int(u["fold"])
        if k not in out or common.is_rerun(c.parent):  # a rerun replaces the original unit
            out[k] = {"dir": c.parent, "config": heads.config_from(u["config"])}
    if not out:
        raise FileNotFoundError("Stage A heads not attached")
    return out


def write_submission(sample: pd.DataFrame, uids: np.ndarray, P: np.ndarray, out: Path) -> dict:
    """Fill the competition's sample submission with probabilities (wide layout: one row per study)."""
    cols = list(sample.columns)
    if UID in cols and all(f in cols for f in FINDINGS):
        sub = sample[[UID]].copy()
        sub[UID] = sub[UID].astype(str)
        pred = pd.DataFrame(P, columns=FINDINGS)
        pred[UID] = [str(u) for u in uids]
        sub = sub.merge(pred, on=UID, how="left")
        missing = int(sub[FINDINGS].isna().any(axis=1).sum())
        sub[FINDINGS] = sub[FINDINGS].fillna(0.5)
        sub = sub[cols]
        sub.to_csv(out, index=False)
        return {"layout": "wide", "rows": agg.count(len(sub)), "rows_without_prediction": agg.count(missing)}
    raise ValueError("unknown submission layout")


def run(encoder: str = "dinov2_vits14", root=None, workers: int = 4, budget_h: float = 8.5,
        quiet: bool = True) -> dict:
    s: dict = {}
    with common.notebook("n10", {"encoder": encoder, "workers": workers, "budget_h": budget_h},
                         quiet=quiet, budget_h=budget_h):
        import torch

        t0 = time.time()
        rootp = Path(root) if root else None
        comp = paths.competition_root()
        od = paths.out_dir("private", "n10")
        units = _units(encoder, rootp)
        ckpt = _find_bundle_ckpt(encoder, rootp)
        s["folds"] = sorted(units)

        # 1-2. Study table and caches for the test split (training code)
        st, _, hdr_err = study_table.build(comp, "test", workers=workers)
        st.to_csv(od / "study_table.csv", index=False)
        cache_sum = {}
        for p in PLANES:
            cd = od / f"cache_{p}"
            cache_sum[p] = cache.build(st, comp / "test_series", p, cd, workers=workers,
                                       budget_s=0.4 * budget_h * 3600)
            (cd / "study_table.csv").write_text((od / "study_table.csv").read_text())
        s["caches"] = {p: {"failed": v["failed"], "missing_series": v["missing_series"],
                           "checks": v["checks"]} for p, v in cache_sum.items()}
        s["header_read_errors"] = agg.counts(hdr_err)

        # 3. Embeddings with the bundled checkpoint (no download)
        devs = n5._devices(None)
        encs_by_dev = {}
        for dv in devs:
            e = encoders.Encoder(encoder, dv, pretrained=False)
            e.model.load_state_dict(torch.load(ckpt, map_location=dv))
            encs_by_dev[dv] = {encoder: e}
        n = len(st)
        lat = st["laterality"]
        unresolved = ~lat.isin(["L", "R"]).to_numpy()
        flip_row = np.full(n, -1)
        flip_row[unresolved] = np.arange(int(unresolved.sum()))
        emb, flip, ok = {}, {}, {}
        for p in PLANES:
            man = json.loads((od / f"cache_{p}" / f"cache_{p}_manifest.json").read_text())
            mm = np.memmap(od / f"cache_{p}" / man["file"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
            per = pd.read_csv(od / f"cache_{p}" / f"cache_{p}_studies.csv").set_index("row")
            ok[p] = per["ok"].reindex(np.arange(n)).fillna(False).astype(bool).to_numpy()
            rows = np.flatnonzero(ok[p]).tolist()
            emb[p] = np.zeros((n, mm.shape[1], encoders.FEATURE_DIM), np.float16)
            n5._embed(mm, rows, {encoder: emb[p]}, encs_by_dev, False, 8)
            if p in heads.FLIP_PLANES and unresolved.any():
                flip[p] = np.zeros((int(unresolved.sum()), mm.shape[1], encoders.FEATURE_DIM), np.float16)
                fr = np.flatnonzero(unresolved & ok[p]).tolist()
                if fr:
                    n5._embed(mm, fr, {encoder: flip[p]}, encs_by_dev, True, 8, out_rows=flip_row[fr].tolist())
        del encs_by_dev
        if devs[0].startswith("cuda"):
            torch.cuda.empty_cache()

        # 4. Fold heads, averaged
        dummy = np.zeros((n, len(FINDINGS)), np.float32)
        bank = heads.Bank(emb, dummy, dummy, dummy, flip=flip, flip_row=flip_row, device=devs[0])
        preds = []
        all_rows = np.arange(n)
        for k, u in sorted(units.items()):
            cfg = u["config"]
            planes = heads.PLANE_SETS[cfg.planes]
            stats = {p: (mu.to(bank.device), sd.to(bank.device))
                     for p, (mu, sd) in torch.load(u["dir"] / "stats.pt", map_location="cpu").items()}
            for p in planes:  # a failed plane gets this fold's training mean (z = 0)
                miss = np.flatnonzero(~ok[p])
                if len(miss):
                    bank.x[p][torch.as_tensor(miss, device=bank.device)] = stats[p][0].half()
            model = heads.make_head(cfg, encoders.FEATURE_DIM, len(planes)).to(bank.device)
            model.load_state_dict(torch.load(u["dir"] / "head.pt", map_location=bank.device))
            preds.append(heads.predict(bank, model, all_rows, cfg, stats))
        P = np.mean(preds, axis=0)
        np.save(od / "predictions_mean.npy", P)
        sample = pd.read_csv(comp / "sample_submission.csv")
        s["submission"] = write_submission(sample, st[UID].astype(str).to_numpy(), P, paths.WORKING / "submission.csv")
        s["studies"] = agg.count(n)
        s["all_planes_ok_share"] = agg.share(np.mean(ok["sagittal"] & ok["coronal"] & ok["axial"])) if n else None
        s["seconds"] = round(time.time() - t0, 1)
        agg.write("n10", s)
    common.summary({k: s.get(k) for k in ("submission", "studies", "all_planes_ok_share", "folds", "seconds")})
    return s
