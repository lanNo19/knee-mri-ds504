"""Montage notebook: pictures of the caches for Ablay's eyes only.

Claude starts this notebook's runs but never opens its pages or outputs,
because the images are RSNA pixels. Ablay looks at them in his own browser.

For each plane it writes private/montage/<plane>.png: a grid of 4 slices
(evenly spaced) for each of n studies, taken in cache row order from a
fixed random draw, with no IDs drawn on the image. The row numbers of the
drawn studies go to private/montage/<plane>_rows.csv.
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


def run(n: int = 12, seed: int = 0, quiet: bool = True) -> dict:
    out = {}
    with common.notebook("montage", {"n": n, "seed": seed}, quiet=quiet):
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
    common.summary({"montages": out})
    return out
