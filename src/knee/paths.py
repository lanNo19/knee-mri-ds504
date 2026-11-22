"""Where inputs live on Kaggle (and in tests).

Kaggle mounts competition data at /kaggle/input/<slug> on older images and at
/kaggle/input/competitions/<slug> on newer ones, so both are probed. Attached
datasets and notebook outputs are found by file name under /kaggle/input,
because their mount paths also differ between images.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
INPUT = Path(os.environ.get("KNEE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("KNEE_WORKING", "/kaggle/working"))

# Public LLM label tables (owner/slug, file name). Rule 1 of the label source.
LABEL_TABLES = {
    "pilkwang": ("pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv"),
    "steven_v2": ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v2.csv"),
    "lixin": ("lixin73/rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
}


def competition_root() -> Path:
    """The competition folder; KNEE_DATA_ROOT overrides it (tests)."""
    env = os.environ.get("KNEE_DATA_ROOT")
    if env:
        return Path(env)
    for cand in (INPUT / COMPETITION, INPUT / "competitions" / COMPETITION):
        if (cand / "train_series.csv").exists() or (cand / "test_series.csv").exists():
            return cand
    raise FileNotFoundError("competition data not attached")


def find_file(name: str, root: Path | None = None, max_depth: int = 6) -> Path | None:
    """First file called `name` under root (default /kaggle/input), or None.

    The competition folder is skipped, so the search never walks the DICOM tree.
    """
    root = Path(root or INPUT)
    if not root.exists():
        return None
    skip = {COMPETITION, "competitions"}
    tops = sorted(p for p in root.iterdir() if p.name not in skip)
    for depth in range(max_depth + 1):
        hits = []
        for top in tops:
            if depth == 0:
                if top.name == name and top.is_file():
                    hits.append(top)
                continue
            if top.is_dir():
                hits += glob.glob(str(top / ("*/" * (depth - 1)) / name))
        if hits:
            return Path(sorted(hits)[0])
    return None


def find_label_table(key: str, root: Path | None = None) -> Path | None:
    """Path of one public label table, or None if it is not attached."""
    _, fname = LABEL_TABLES[key]
    return find_file(fname, root)


def out_dir(*parts: str) -> Path:
    """A folder under /kaggle/working (created)."""
    p = WORKING.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
