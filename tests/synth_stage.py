"""Synthetic N0, N3 and N5 outputs for the modelling notebooks (labels, folds, embeddings with signal)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ENCODERS = ("dinov2_vits14", "convnext_tiny", "vit_s16")
PLANES = ("sagittal", "coronal", "axial")
UID = "StudyInstanceUID"


def make_stage(root: Path, n: int = 800, n_groups: int = 8, S: int = 8, D: int = 64, seed: int = 0,
               signal: dict | None = None, duplicates: int = 3) -> dict:
    """Writes root/n0, root/n3, root/n5 (Kaggle-like private folders). Returns facts for assertions."""
    rng = np.random.default_rng(seed)
    signal = signal or {"dinov2_vits14": 1.2, "convnext_tiny": 1.0, "vit_s16": 0.8}
    uids = np.array([f"1.2.826.0.1.3680043.8.498.{70000 + i}" for i in range(n)])
    grp = rng.integers(0, n_groups, n)
    lang = np.array([f"LANG{g}" for g in grp])
    shift = rng.normal(0, 0.4, (n_groups, 12))
    z = rng.normal(size=(n, 12)) + shift[grp]
    truth = (z > 0.55).astype(float)  # prevalence about 0.3
    soft = np.clip(truth * 0.8 + 0.1 + rng.normal(0, 0.08, truth.shape), 0, 1)
    soft[rng.random(soft.shape) < 0.02] = np.nan
    binary = np.where(np.isnan(soft), np.nan, np.where(soft > 0.5, 1.0, np.where(soft < 0.5, 0.0, np.nan)))
    weight = np.where(np.isnan(soft), 0.0, np.where(rng.random(soft.shape) < 0.1, 0.5, 1.0))
    is_gold = np.zeros(n, bool)
    is_gold[rng.choice(n, 58, replace=False)] = True
    scored = rng.random(n) < 0.9
    lat = rng.choice(np.array(["R", "L", None], dtype=object), size=n, p=[0.49, 0.49, 0.02])
    man = np.array(["SIEMENS", "GE", "PHILIPS"])[grp % 3]

    n0 = root / "n0" / "private" / "n0"
    n0.mkdir(parents=True)
    studies = pd.DataFrame({UID: uids, "g_lang": lang, "g_vendor": [f"{a}|{b}" for a, b in zip(lang, man)],
                            "scored": scored & ~is_gold, "in_training": ~is_gold, "is_gold": is_gold,
                            "has_report": True, "laterality": lat, "manufacturer": man})
    studies.to_csv(n0 / "studies.csv", index=False)
    for name, arr in (("labels_soft", soft), ("labels_weight", weight), ("labels_binary", binary)):
        pd.DataFrame(arr, columns=FINDINGS).assign(**{UID: uids})[[UID] + FINDINGS].to_csv(n0 / f"{name}.csv", index=False)
    pd.DataFrame(truth[is_gold].astype(int), columns=FINDINGS).assign(**{UID: uids[is_gold]})[[UID] + FINDINGS].to_csv(
        n0 / "labels_gold.csv", index=False)
    # floor-A tags: the scanner carries a little label signal through the group shift
    tags = pd.DataFrame({UID: uids, "manufacturer": man, "fingerprint": [f"{m}|M{g}|1.5" for m, g in zip(man, grp)],
                         "model": [f"M{g}" for g in grp], "field_strength": np.where(grp % 2, 3.0, 1.5)})
    for p in PLANES:
        for t in ("repetition_time", "echo_time", "slice_thickness", "pixel_spacing"):
            v = rng.normal(100 + 10 * grp, 5)
            if p == "sagittal" and t == "echo_time":
                v = v + 30 * truth[:, 0]  # a tag with label signal that transfers across sites (ACL)
            v = v.round(1)
            v[rng.random(n) < 0.05] = np.nan
            tags[f"{p}_{t}"] = v
    tags.to_csv(n0 / "tags.csv", index=False)
    (root / "n0" / "run.json").write_text(json.dumps({"commit": "synthetic000"}))

    # N3-like folds: groups -> 5 outer folds; inner folds by group inside each outer training set
    n3 = root / "n3" / "private" / "n3"
    n3.mkdir(parents=True)
    g_fold = {g: g % 5 for g in range(n_groups)}
    s2 = np.array([g_fold[g] for g in grp])
    train = ~is_gold
    folds = pd.DataFrame({UID: uids, "train": train, "scored": scored & ~is_gold, "is_gold": is_gold,
                          "s2_fold": s2, "vendor_fold": s2})
    for k in range(5):
        inner = np.full(n, -1)
        tr_groups = sorted(set(grp[train & (s2 != k)]))
        for i, g in enumerate(tr_groups):
            inner[train & (s2 != k) & (grp == g)] = i % min(4, len(tr_groups))
        folds[f"s2_inner_{k}"] = inner
        folds[f"vendor_inner_{k}"] = inner
    s1 = np.full(n, -1)
    tr_rows = np.flatnonzero(train)
    perm = rng.permutation(tr_rows)
    sizes = [int((s2[train] == k).sum()) for k in range(5)]
    start = 0
    for k, sz in enumerate(sizes):
        s1[perm[start:start + sz]] = k
        start += sz
    folds["s1_fold"] = s1
    folds.to_csv(n3 / "folds.csv", index=False)
    (root / "n3" / "run.json").write_text(json.dumps({"commit": "synthetic000"}))

    # N5-like embeddings: label signal through a fixed projection, plus slice noise
    n5 = root / "n5" / "private" / "n5"
    n5.mkdir(parents=True)
    usable = rng.random(n) > 0.02
    unresolved = pd.isna(lat)
    flip_row = np.full(n, -1)
    flip_row[unresolved] = np.arange(int(unresolved.sum()))
    index = pd.DataFrame({"row": np.arange(n), UID: uids, "laterality": lat, "usable": usable, "flip_row": flip_row})
    for p in PLANES:
        index[f"ok_{p}"] = usable
    index.to_csv(n5 / "index.csv", index=False)
    dup_pairs = [(int(a), int(b)) for a, b in rng.choice(np.flatnonzero(usable & ~is_gold), (duplicates, 2), replace=False)]
    for e in ENCODERS:
        for pi, p in enumerate(PLANES):
            W = rng.normal(0, 1, (12, D)) / np.sqrt(12)
            base = (truth - 0.3) @ W * signal[e] + shift[grp] @ W * 0.3
            x = base[:, None, :] + rng.normal(0, 1.0, (n, S, D))
            for a, b in dup_pairs:
                x[b] = x[a]
            x[~usable] = 0
            np.save(n5 / f"emb_{e}_{p}.npy", x.astype(np.float16))
            if p in ("coronal", "axial"):
                np.save(n5 / f"emb_{e}_{p}_flip.npy", (x[unresolved] + rng.normal(0, 0.05, (int(unresolved.sum()), S, D))).astype(np.float16))
    (root / "n5" / "run.json").write_text(json.dumps({"commit": "synthetic000"}))
    return {"uids": uids.tolist(), "dup_pairs": [(uids[a], uids[b]) for a, b in dup_pairs], "n": n}


def make_units(root: Path, name: str = "n6a", strengths: dict | None = None, seed: int = 1) -> None:
    """Synthetic Stage A unit outputs (predictions.csv, config.json, l1.csv and trials.csv per encoder and
    fold) from make_stage's files. As in n6._unit, a unit predicts its usable test-fold and gold studies."""
    rng = np.random.default_rng(seed)
    log_rng = np.random.default_rng(seed + 1)  # the search logs draw from their own stream
    strengths = strengths or {"dinov2_vits14": 1.6, "convnext_tiny": 0.9, "vit_s16": 1.1}
    n0 = root / "n0" / "private" / "n0"
    lab = pd.read_csv(n0 / "labels_binary.csv").set_index(UID)
    folds = pd.read_csv(root / "n3" / "private" / "n3" / "folds.csv").set_index(UID)
    y = lab.loc[folds.index, FINDINGS].fillna(0.5).to_numpy()
    usable = pd.read_csv(root / "n5" / "private" / "n5" / "index.csv").set_index(UID)["usable"].reindex(folds.index).to_numpy()
    for e, st in strengths.items():
        for k in range(5):
            rows = ((folds["s2_fold"] == k) & (folds["train"] | folds["is_gold"])).to_numpy() & usable
            ud = root / name / "private" / name / "units" / f"{e}_f{k}"
            ud.mkdir(parents=True, exist_ok=True)
            p = 1 / (1 + np.exp(-(st * (y[rows] - 0.3) * 3 + rng.normal(0, 1, (int(rows.sum()), 12)))))
            df = pd.DataFrame(p, columns=FINDINGS)
            df.insert(0, "is_gold", folds["is_gold"].to_numpy()[rows])
            df.insert(0, "s2_fold", k)
            df.insert(0, UID, folds.index[rows])
            df.to_csv(ud / "predictions.csv", index=False)
            cfg = {"pooling": "mean" if k % 2 else "gated", "planes": "all", "lr": 3e-3, "weight_decay": 3e-4,
                   "dropout": 0.25, "attn_hidden": 192, "pos_cap": 4.47, "batch_size": 48}
            (ud / "config.json").write_text(json.dumps({
                "encoder": e, "fold": k, "l1_winner": {"pooling": cfg["pooling"], "planes": "all"},
                "config": cfg, "best_inner_score": 0.7 + 0.01 * k, "E": 6}))
            _search_logs(ud, cfg["pooling"], 0.7 + 0.01 * k, log_rng)
    # floor A: a weak scanner signal
    fa = root / "n8" / "private" / "n8"
    fa.mkdir(parents=True, exist_ok=True)
    rows = ((folds["s2_fold"] >= 0) & (folds["train"] | folds["is_gold"])).to_numpy()
    p = 1 / (1 + np.exp(-(0.6 * (y[rows] - 0.3) + rng.normal(0, 1, (int(rows.sum()), 12)))))
    df = pd.DataFrame(p, columns=FINDINGS)
    df.insert(0, "s2_fold", folds["s2_fold"].to_numpy()[rows])
    df.insert(0, UID, folds.index[rows])
    df.to_csv(fa / "predictions.csv", index=False)


def _search_logs(ud: Path, pooling: str, best: float, rng, n_trials: int = 30) -> None:
    """l1.csv and trials.csv shaped like n6._unit's: trial 0 = the L2 defaults, the last trial is the best."""
    grid = [(p, s) for p in ("mean", "max", "gated") for s in ("sag", "sag+cor", "all")]
    l1 = pd.DataFrame([{"pooling": p, "planes": s, "inner_score": best - (0.02 if (p, s) == (pooling, "all") else 0.05),
                        "median_best_epoch": 5.0} for p, s in grid])
    l1.to_csv(ud / "l1.csv", index=False)
    t = pd.DataFrame({
        "number": np.arange(n_trials), "state": "COMPLETE",
        "value": np.r_[best - 0.02, best - 0.02 - rng.uniform(0, 0.05, n_trials - 2), best],
        "lr": np.r_[3e-4, np.exp(rng.uniform(np.log(1e-5), np.log(1e-2), n_trials - 1))],
        "weight_decay": np.r_[3e-4, np.exp(rng.uniform(np.log(1e-6), np.log(1e-1), n_trials - 1))],
        "dropout": np.r_[0.25, rng.uniform(0, 0.5, n_trials - 1)],
        "pos_cap": np.r_[1.5, np.exp(rng.uniform(0, np.log(2.0), n_trials - 1))],
        "batch_size": np.r_[48, np.exp(rng.uniform(np.log(16), np.log(128), n_trials - 1)).astype(int)],
        "best_epochs": json.dumps([4, 4, 5, 5])})
    if pooling == "gated":
        t["attn_hidden"] = np.r_[192, np.exp(rng.uniform(np.log(64), np.log(512), n_trials - 1)).astype(int)]
    t.to_csv(ud / "trials.csv", index=False)
