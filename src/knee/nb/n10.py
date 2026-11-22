"""N10: inference on the competition's test studies and submission.csv (internet OFF; S4).

Inputs: the competition, the offline bundle dataset (private/bundle: wheels, encoder
checkpoints, manifest.json), N0's output (agg_n0.json: the training prevalence for
the per-study fallback) and, except for the canary, the Stage A output (N6a; or the
one-unit smoke): the chosen encoder's S2 fold units (head.pt, stats.pt, config.json).

Canary (canary=True, CPU, Monday): decodes every selected test series with the
training code and fails on purpose if more than 1% of test studies fail to decode
(any plane failed or missing), because a scored or failed canary is the only view of
hidden-test decoding before the real runs. Otherwise it submits the training
prevalence for every study.

Real run, with the same code as training:
  1. study table and series choice for the test split (N0/N4 code);
  2. one uint8 cache per plane (N4's cache.build), under /tmp (not saved);
  3. slice embeddings with the encoder loaded from the bundle checkpoint (its sha256
     is checked against the bundle manifest), plus flipped coronal/axial slices for
     unresolved laterality (N5 code);
  4. the mean over the attached fold heads (flip-averaged for unresolved laterality);
     a study missing any plane gets the training prevalence (per-study fallback).
Writes submission.csv at the output root (Kaggle requires it; Claude never opens it)
and agg_n10.json.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, cache, encoders, heads, paths, study_table
from ..labels import FINDINGS, UID
from . import common, n5

PLANES = heads.PLANES
MAX_FAILED_SHARE = 0.01


def _bundle(root: Path | None) -> tuple[Path, dict]:
    hits = paths.glob_outputs("private/bundle/manifest.json", root)
    if not hits:
        raise FileNotFoundError("offline bundle not attached")
    return hits[0].parent, json.loads(hits[0].read_text())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _units(encoder: str, root: Path | None) -> dict:
    out = {}
    for c in paths.glob_outputs(f"private/*/units/{encoder}_f*/config.json", root):
        u = json.loads(c.read_text())
        k = int(u["fold"])
        if k not in out or common.is_rerun(c.parent):  # a rerun replaces the original unit
            out[k] = {"dir": c.parent, "config": heads.config_from(u["config"]),
                      "head_sha16": common.sha16(c.parent / "head.pt")}
    if not out:
        raise FileNotFoundError("Stage A heads not attached")
    return out


def _prevalence(root: Path | None) -> np.ndarray:
    p = paths.find_file("agg_n0.json", root)
    if p is None:
        raise FileNotFoundError("N0's output (training prevalence) not attached")
    prev = json.loads(p.read_text())["labels"]["prevalence_training"]
    return np.array([float(prev[f]) for f in FINDINGS], np.float32)


def write_submission(sample: pd.DataFrame, uids: np.ndarray, P: np.ndarray, out: Path,
                     fallback: np.ndarray | None = None) -> dict:
    """Fill the competition's sample submission (wide layout: one row per study, 12 probabilities).

    A listed study without a prediction (no series in the test table) gets `fallback`
    (the training prevalence); other layouts are refused until SMOKE has shown the real one."""
    cols = list(sample.columns)
    if UID in cols and all(f in cols for f in FINDINGS):
        sub = sample[[UID]].copy()
        sub[UID] = sub[UID].astype(str)
        pred = pd.DataFrame(P, columns=FINDINGS)
        pred[UID] = [str(u) for u in uids]
        sub = sub.merge(pred, on=UID, how="left")
        missing = sub[FINDINGS].isna().any(axis=1).to_numpy()
        if missing.any():
            if fallback is None:
                raise ValueError("sample submission lists studies without a prediction")
            sub.loc[missing, FINDINGS] = np.tile(np.asarray(fallback, float), (int(missing.sum()), 1))
        sub = sub[cols]
        sub.to_csv(out, index=False)
        return {"layout": "wide", "rows": agg.count(len(sub)), "listed_without_series": agg.count(int(missing.sum()))}
    raise ValueError("unknown submission layout")


def run(encoder: str = "dinov2_vits14", canary: bool = False, root=None, workers: int = 4,
        budget_h: float = 8.0, max_failed_share: float = MAX_FAILED_SHARE, quiet: bool = True) -> dict:
    """budget_h 8.0: the watchdog ends the step at 8.8 h, inside the 9-hour submission limit."""
    s: dict = {"canary": canary}
    cfg = {"encoder": encoder, "canary": canary, "workers": workers, "budget_h": budget_h,
           "max_failed_share": max_failed_share}
    with common.notebook("n10", cfg, quiet=quiet, budget_h=budget_h):
        import torch

        t0 = time.time()
        rootp = Path(root) if root else None
        comp = paths.competition_root()
        bdir, man = _bundle(rootp)
        s["bundle_commit"] = (man.get("commit") or "")[:12] or None  # the installed wheel's commit
        prev = _prevalence(rootp)
        units = {} if canary else _units(encoder, rootp)
        if not canary:  # a wrong bundle fails in seconds, before hours of decoding
            ckpt = bdir / "checkpoints" / f"{encoder}.pth"
            want = man.get("files_sha256", {}).get(ckpt.name)
            if want is None or not ckpt.exists() or _sha256(ckpt) != want:
                raise ValueError("checkpoint does not match the bundle manifest")
        s["folds"] = sorted(units)
        s["heads_sha16"] = {f"fold_{k}": u["head_sha16"] for k, u in sorted(units.items())}
        work = Path(tempfile.mkdtemp(prefix="knee_n10_"))  # caches are large and never saved

        # 1-2. Study table and caches for the test split (training code); half the budget for decoding
        st, _, hdr_err = study_table.build(comp, "test", workers=workers)
        st.to_csv(work / "study_table.csv", index=False)
        n = len(st)
        decode_until = t0 + 0.5 * budget_h * 3600
        ok, cache_sum = {}, {}
        for i, p in enumerate(PLANES):
            cd = work / f"cache_{p}"
            left = max(60.0, (decode_until - time.time()) / (len(PLANES) - i))  # an equal share of what is left
            cache_sum[p] = cache.build(st, comp / "test_series", p, cd, workers=workers, budget_s=left)
            per = pd.read_csv(cd / f"cache_{p}_studies.csv").set_index("row")
            ok[p] = per["ok"].reindex(np.arange(n)).fillna(False).astype(bool).to_numpy()
        s["caches"] = {p: {"failed": v["failed"], "missing_series": v["missing_series"],
                           "not_started_budget": v["not_started_budget"]} for p, v in cache_sum.items()}
        s["header_read_errors"] = agg.counts(hdr_err)
        all_ok = ok["sagittal"] & ok["coronal"] & ok["axial"]
        failed_share = float(1 - all_ok.mean()) if n else 0.0
        s["studies"] = agg.count(n)
        s["failed_share"] = agg.share(failed_share)
        s["studies_with_fallback"] = agg.count(int((~all_ok).sum()))
        if failed_share > max_failed_share:
            agg.write("n10", s)
            raise RuntimeError("more than 1% of test studies failed to decode")  # deliberate (canary rule)

        P = np.tile(prev, (n, 1))  # per-study fallback: the training prevalence
        if not canary:
            # 3. Embeddings with the bundled checkpoint (no download; checked above)
            devs = n5._devices(None)
            encs_by_dev = {}
            for dv in devs:
                e = encoders.Encoder(encoder, dv, pretrained=False)
                e.model.load_state_dict(torch.load(ckpt, map_location=dv))
                encs_by_dev[dv] = {encoder: e}
            unresolved = ~st["laterality"].isin(["L", "R"]).to_numpy()
            flip_row = np.full(n, -1)
            flip_row[unresolved] = np.arange(int(unresolved.sum()))
            emb, flip = {}, {}
            rows = np.flatnonzero(all_ok).tolist()
            for p in PLANES:
                cd = work / f"cache_{p}"
                cm = json.loads((cd / f"cache_{p}_manifest.json").read_text())
                mm = np.memmap(cd / cm["file"], dtype=np.uint8, mode="r", shape=tuple(cm["shape"]))
                emb[p] = np.zeros((n, mm.shape[1], encoders.FEATURE_DIM), np.float16)
                n5._embed(mm, rows, {encoder: emb[p]}, encs_by_dev, False, 8)
                if p in heads.FLIP_PLANES and unresolved.any():
                    flip[p] = np.zeros((int(unresolved.sum()), mm.shape[1], encoders.FEATURE_DIM), np.float16)
                    fr = np.flatnonzero(unresolved & all_ok).tolist()
                    if fr:
                        n5._embed(mm, fr, {encoder: flip[p]}, encs_by_dev, True, 8, out_rows=flip_row[fr].tolist())
            del encs_by_dev
            if devs[0].startswith("cuda"):
                torch.cuda.empty_cache()

            # 4. Fold heads, averaged, for the studies with all three planes
            dummy = np.zeros((n, len(FINDINGS)), np.float32)
            bank = heads.Bank(emb, dummy, dummy, dummy, flip=flip, flip_row=flip_row, device=devs[0])
            good = np.flatnonzero(all_ok)
            preds = []
            for k, u in sorted(units.items()):
                cfg_k = u["config"]
                planes = heads.PLANE_SETS[cfg_k.planes]
                stats = {p: (mu.to(bank.device), sd.to(bank.device))
                         for p, (mu, sd) in torch.load(u["dir"] / "stats.pt", map_location="cpu").items()}
                model = heads.make_head(cfg_k, encoders.FEATURE_DIM, len(planes)).to(bank.device)
                model.load_state_dict(torch.load(u["dir"] / "head.pt", map_location=bank.device))
                preds.append(heads.predict(bank, model, good, cfg_k, stats))
            if len(good):
                P[good] = np.mean(preds, axis=0)
        sample = pd.read_csv(comp / "sample_submission.csv")
        s["submission"] = write_submission(sample, st[UID].astype(str).to_numpy(), P, paths.WORKING / "submission.csv",
                                           fallback=prev)
        s["seconds"] = round(time.time() - t0, 1)
        agg.write("n10", s)
    common.summary({k: s.get(k) for k in ("canary", "submission", "studies", "failed_share", "studies_with_fallback",
                                          "folds", "bundle_commit", "seconds")})
    return s
