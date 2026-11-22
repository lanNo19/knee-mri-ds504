"""Inputs of the modelling notebooks (N6p, N6a, N6b, N7, N8), aligned to N5's row order.

Reads, from attached notebook outputs (found by file name under /kaggle/input):
  N0: private/n0/studies.csv, labels_soft.csv, labels_weight.csv, labels_binary.csv, tags.csv
  N3: private/n3/folds.csv
  N5: private/n5/index.csv, emb_<encoder>_<plane>.npy (+ _flip for coronal, axial), manifest.json
Every upstream file's commit and hash go into the caller's summary, and an
expected hash prefix can be asserted (a stale attached version fails loudly).

Row i of every array here is row i of N5's index (the embedding row).
A study is usable only if all three planes have a cache (plan: a study missing
any plane after decoding is excluded for every encoder and config).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import heads, paths
from .labels import FINDINGS, UID
from .nb import common


def _find(name: str, root: Path | None, explicit: Path | None = None) -> tuple[Path, dict]:
    if explicit is not None:
        return common.upstream(name, explicit)
    p = paths.find_file(name, root)
    if p is None:
        raise FileNotFoundError("upstream output not attached")
    return common.upstream(name, p)


class StageData:
    def __init__(self, root: str | Path | None = None, expect: dict | None = None, need_embeddings: bool = True,
                 n0_dir: Path | None = None, n3_dir: Path | None = None, n5_dir: Path | None = None):
        root = Path(root) if root else None
        expect = expect or {}
        self.upstream: dict = {}

        def get(key, name, d):
            p, info = _find(name, root, (Path(d) / name) if d else None)
            common.expect(info, expect.get(key))
            self.upstream[key] = info
            return p

        st_path = get("n0_studies", "studies.csv", n0_dir)
        n0d = st_path.parent
        folds_path = get("n3_folds", "folds.csv", n3_dir)
        studies = pd.read_csv(st_path).set_index(UID)
        folds = pd.read_csv(folds_path).set_index(UID)
        if need_embeddings:
            n5d = Path(n5_dir) if n5_dir else None
            if n5d is None:
                p = paths.find_file("emb_dinov2_vits14_sagittal.npy", root)
                if p is None:
                    raise FileNotFoundError("upstream output not attached")
                n5d = p.parent
            idx_path, info = common.upstream("index.csv", n5d / "index.csv")
            common.expect(info, expect.get("n5_index"))
            self.upstream["n5_index"] = info
            index = pd.read_csv(idx_path)
            self.n5_dir = n5d
            self.usable_from_n5 = True
        else:  # N8: no embeddings; rows follow N0's studies, usable from N5's index when attached
            index = pd.DataFrame({"row": np.arange(len(studies)), UID: studies.index.astype(str),
                                  "usable": True, "flip_row": -1})
            self.n5_dir = None
            n5_index = Path(n5_dir) / "index.csv" if n5_dir is not None else None
            if n5_index is None:
                p = paths.find_file("emb_dinov2_vits14_sagittal.npy", root)
                n5_index = p.parent / "index.csv" if p is not None else None
            self.usable_from_n5 = bool(n5_index is not None and n5_index.exists())
            if self.usable_from_n5:
                ip, info = common.upstream("index.csv", n5_index)
                common.expect(info, expect.get("n5_index"))
                self.upstream["n5_index"] = info
                u = pd.read_csv(ip).astype({UID: str}).set_index(UID)["usable"]
                index["usable"] = index[UID].map(u).fillna(False).astype(bool).to_numpy()
        index[UID] = index[UID].astype(str)
        self.uids = index[UID].to_numpy()
        self.n = len(index)
        t = index.set_index(UID)
        st = studies.reindex(self.uids)
        fo = folds.reindex(self.uids)
        self.in_n0 = st.index.isin(studies.index)
        self.usable = t["usable"].fillna(False).astype(bool).to_numpy()
        self.flip_row = t["flip_row"].fillna(-1).astype(int).to_numpy()
        self.g_lang = st["g_lang"].to_numpy()
        self.g_vendor = st["g_vendor"].to_numpy()
        self.train = fo["train"].fillna(False).astype(bool).to_numpy() & self.usable
        self.scored = fo["scored"].fillna(False).astype(bool).to_numpy() & self.usable
        self.is_gold = fo["is_gold"].fillna(False).astype(bool).to_numpy() & self.usable
        self.folds = {c: fo[c].fillna(-1).astype(int).to_numpy() for c in fo.columns
                      if c.endswith("_fold") or "_inner_" in c}
        self.soft = labels_frame(n0d / "labels_soft.csv", self.uids)
        self.weight = labels_frame(n0d / "labels_weight.csv", self.uids, fill=0.0)
        self.binary = labels_frame(n0d / "labels_binary.csv", self.uids)
        # Scoring labels (plan): uncertain counts as negative; the NaN version is kept for the
        # sensitivity analysis (uncertain dropped) and for LightGBM training.
        self.eval_y = heads.scoring_labels(self.binary)
        gp = n0d / "labels_gold.csv"
        self.gold_y = labels_frame(gp, self.uids) if gp.exists() else np.full((self.n, len(FINDINGS)), np.nan)
        self.tags_path = n0d / "tags.csv"
        self.n0_dir = n0d
        self.n0_agg = {}
        for cand in (n0d.parent.parent / "agg_n0.json", n0d.parent / "agg_n0.json"):
            if cand.exists():
                import json

                self.n0_agg = json.loads(cand.read_text())
                break

    def pos_cap_upper(self) -> float:
        """U of the L2 table, as N0 set it before any head trained (recomputed only if N0's file is absent)."""
        from . import labels

        u = self.n0_agg.get("labels", {}).get("pos_weight_cap_upper")
        if u is not None:
            return float(u)
        return labels.pos_weight_cap_upper(pd.DataFrame(self.binary[self.train], columns=FINDINGS))

    # ---- index sets
    def outer(self, scheme: str = "s2"):
        """Outer fold array for the training studies (others -1)."""
        f = self.folds[f"{scheme}_fold"].copy()
        f[~(self.train | self.is_gold)] = -1
        return f

    def n_folds(self, scheme: str = "s2") -> int:
        f = self.folds[f"{scheme}_fold"][self.train]
        return int(f.max() + 1) if len(f) and f.max() >= 0 else 0

    def split(self, k: int, scheme: str = "s2"):
        """(outer-train rows, test rows, scored test rows) for outer fold k."""
        f = self.folds[f"{scheme}_fold"]
        tr = np.flatnonzero(self.train & (f >= 0) & (f != k))
        te = np.flatnonzero(self.train & (f == k))
        return tr, te, te[self.scored[te]]

    def inner(self, k: int, scheme: str = "s2") -> np.ndarray:
        return self.folds.get(f"{scheme}_inner_{k}", np.full(self.n, -1))

    def gold_rows(self, k: int | None = None, scheme: str = "s2") -> np.ndarray:
        g = self.is_gold.copy()
        if k is not None:
            g &= self.folds[f"{scheme}_fold"] == k
        return np.flatnonzero(g)

    # ---- embeddings
    def bank(self, encoder: str, device: str = "cpu", labels_override: tuple | None = None) -> heads.Bank:
        emb, flip = {}, {}
        for p in heads.PLANES:
            emb[p] = np.load(self.n5_dir / f"emb_{encoder}_{p}.npy", mmap_mode="r")
            fp = self.n5_dir / f"emb_{encoder}_{p}_flip.npy"
            if p in heads.FLIP_PLANES and fp.exists():
                flip[p] = np.load(fp, mmap_mode="r")
        soft, weight, binary = labels_override or (self.soft, self.weight, self.binary)
        return heads.Bank(emb, soft, weight, binary, flip=flip, flip_row=self.flip_row, device=device,
                          eval_y=None if labels_override else self.eval_y)

    def study_vectors(self, encoder: str) -> np.ndarray:
        """Per-plane mean of the slice features, three planes joined (float32 [N, 3D])."""
        out = []
        for p in heads.PLANES:
            a = np.load(self.n5_dir / f"emb_{encoder}_{p}.npy", mmap_mode="r")
            out.append(np.concatenate([np.asarray(a[i:i + 512], np.float32).mean(1) for i in range(0, len(a), 512)]))
        return np.concatenate(out, axis=1)


def labels_frame(path: Path, uids, fill: float | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    df[UID] = df[UID].astype(str)
    a = df.set_index(UID).reindex(uids)[FINDINGS].to_numpy(dtype=float)
    if fill is not None:
        a = np.nan_to_num(a, nan=fill)
    return a


def near_duplicates(vectors: np.ndarray, rows: np.ndarray, threshold: float = 0.98, device: str = "cpu",
                    chunk: int = 1024) -> pd.DataFrame:
    """Pairs (row_a < row_b) with cosine above threshold between z-scored study vectors."""
    import torch

    x = torch.from_numpy(np.asarray(vectors[rows], np.float32)).to(device)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    x = torch.nn.functional.normalize(x, dim=1)
    pairs = []
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk] @ x.T
        ii, jj = torch.nonzero(c > threshold, as_tuple=True)
        keep = (ii + i) < jj
        for a, b, v in zip((ii[keep] + i).tolist(), jj[keep].tolist(), c[ii[keep], jj[keep]].tolist()):
            pairs.append((int(rows[a]), int(rows[b]), float(v)))
    df = pd.DataFrame(pairs, columns=["row_a", "row_b", "cosine"]).sort_values("cosine", ascending=False)
    return df.reset_index(drop=True)
