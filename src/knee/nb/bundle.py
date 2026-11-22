"""Offline bundle for N10 (Kaggle notebook with internet ON; its output becomes a private dataset).

Kaggle submissions run with the internet off, so N10 installs from these files
(under private/bundle/, like every other output):
  wheels/        the project wheel (built from the pinned commit) and the pinned wheels:
                 timm 1.0.31, python-gdcm 3.2.6, pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.5.0,
                 pylibjpeg-openjpeg 2.6.0, dicomsdl 0.109.4 (no other dependencies: Kaggle has them)
  checkpoints/   <encoder>.pth: each encoder's state dict exactly as N5 used it (DINOv2 already
                 resampled to 224 x 224), plus a sha256 per file in manifest.json
Only public files: no RSNA data and no head weights (N10 attaches N6a's output for those).
With Ablay's OK, Claude saves this notebook's output as a private, versioned dataset.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from .. import agg, encoders, paths
from . import common

WHEELS = ["timm==1.0.31", "python-gdcm==3.2.6", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg==2.5.0",
          "pylibjpeg-openjpeg==2.6.0", "dicomsdl==0.109.4"]
REPO = "git+https://github.com/lanNo19/knee-mri-ds504@{commit}"


def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(commit: str | None = None, quiet: bool = True) -> dict:
    """commit: the repo commit N10 will run (default: the one this notebook installed)."""
    s: dict = {}
    with common.notebook("bundle", {"commit_given": bool(commit)}, quiet=quiet, budget_h=1.0):
        import torch

        commit = commit or agg.repo_commit()
        if not commit:
            raise RuntimeError("no commit to bundle")
        out = paths.out_dir("private", "bundle")
        wd, cd = out / "wheels", out / "checkpoints"
        wd.mkdir(exist_ok=True)
        cd.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", str(wd), *WHEELS],
                       check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wd), REPO.format(commit=commit)],
                       check=True, capture_output=True)
        files = {}
        for name in encoders.SPECS:
            enc = encoders.Encoder(name, "cpu", pretrained=True)
            p = cd / f"{name}.pth"
            torch.save(enc.model.state_dict(), p)
            files[p.name] = _sha256(p)
        for p in sorted(wd.glob("*.whl")):
            files[f"wheels/{p.name}"] = _sha256(p)
        manifest = {"commit": commit, "wheels": WHEELS, "files_sha256": files,
                    "encoders": {k: encoders.SPECS[k].timm_id for k in encoders.SPECS}}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        s = {"commit": commit[:12], "files": agg.count(len(files)), "wheels_ok": len(list(wd.glob("*.whl"))) >= len(WHEELS) + 1,
             "checkpoints_ok": all((cd / f"{n}.pth").exists() for n in encoders.SPECS),
             "manifest_sha16": common.sha16(out / "manifest.json")}
        agg.write("bundle", s)
    common.summary(s)
    return s
