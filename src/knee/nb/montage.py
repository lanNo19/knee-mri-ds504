"""Montage notebook: pictures of the caches for Ablay's eyes only.

Claude starts this notebook's runs but never opens its pages or outputs,
because the images are RSNA pixels. Ablay looks at them in his own browser.

For each plane it writes private/montage/<plane>.png: a grid of 4 slices
(evenly spaced) for each of n studies, taken in cache row order from a
fixed random draw, with no IDs drawn on the image. The row numbers of the
drawn studies go to private/montage/<plane>_rows.csv. With the pilot's output
attached, private/montage/pairs.png shows the top flagged near-duplicate
pairs (sagittal, study A | study B per row), and pairs_rows.csv lists their
pair_id from top to bottom; Ablay passes the confirmed pair_ids to N3.
"""
from __future__ import annotations

import json

import numpy as np

from .. import paths
from . import common


def _grid(vol_rows: list[np.ndarray], per_study: int = 4) -> np.ndarray:
    tiles = []
    for v in vol_rows:
        idx = np.linspace(0, v.shape[0] - 1, per_study).round().astype(int)
        tiles.append(np.concatenate([v[i] for i in idx], axis=1))
    return np.concatenate(tiles, axis=0)


def _pairs(n_pairs: int, od) -> str:
    """Flagged near-duplicate pairs from the pilot: per pair, 4 sagittal slices of each study side by side."""
    import pandas as pd
    from PIL import Image

    pp = paths.find_file("pairs.csv")
    man_path = paths.find_file("cache_sagittal_manifest.json")
    if pp is None or man_path is None:
        return "pilot output or sagittal cache not attached"
    pairs = pd.read_csv(pp).head(n_pairs)
    if pairs.empty:
        return "no flagged pairs"
    man = json.loads(man_path.read_text())
    mm = np.memmap(man_path.parent / man["file"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    st = pd.read_csv(man_path.parent / "study_table.csv")
    row_of = {str(u): i for i, u in enumerate(st["StudyInstanceUID"].astype(str))}
    tiles, kept = [], []
    for _, r in pairs.iterrows():
        a, b = row_of.get(str(r["uid_a"])), row_of.get(str(r["uid_b"]))
        if a is None or b is None:
            continue
        ga, gb = _grid([np.asarray(mm[a])]), _grid([np.asarray(mm[b])])
        gap = np.full((ga.shape[0], 16), 255, np.uint8)
        tiles.append(np.concatenate([ga, gap, gb], axis=1))
        kept.append((int(r["pair_id"]), a, b))
    if not tiles:
        return "pairs not in the cache"
    sep = np.full((8, tiles[0].shape[1]), 255, np.uint8)
    img = np.concatenate([x for t in tiles for x in (t, sep)][:-1], axis=0)
    Image.fromarray(img).save(od / "pairs.png")
    # top to bottom: pair_id (pass confirmed ones to N3 as confirmed_pair_ids), cache rows
    (od / "pairs_rows.csv").write_text("pair_id,row_a,row_b\n" + "".join(f"{p},{a},{b}\n" for p, a, b in kept))
    return f"{len(kept)} pairs"


def run(n: int = 12, seed: int = 0, pairs: int = 10, quiet: bool = True) -> dict:
    """pairs: how many flagged near-duplicate pairs to draw (needs the pilot's output attached)."""
    out = {}
    with common.notebook("montage", {"n": n, "seed": seed, "pairs": pairs}, quiet=quiet):
        from PIL import Image

        od = paths.out_dir("private", "montage")
        for plane in ("sagittal", "coronal", "axial"):
            man_path = paths.find_file(f"cache_{plane}_manifest.json")
            if man_path is None:
                out[plane] = "cache not attached"
                continue
            man = json.loads(man_path.read_text())
            mm = np.memmap(man_path.parent / man["file"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
            rng = np.random.default_rng(seed)
            rows = sorted(rng.choice(mm.shape[0], size=min(n, mm.shape[0]), replace=False).tolist())
            img = _grid([np.asarray(mm[r]) for r in rows])
            Image.fromarray(img).save(od / f"{plane}.png")
            (od / f"{plane}_rows.csv").write_text("row\n" + "\n".join(map(str, rows)) + "\n")
            out[plane] = f"{len(rows)} studies"
        if pairs:
            out["pairs"] = _pairs(pairs, od)
    common.summary({"montages": out})
    return out
