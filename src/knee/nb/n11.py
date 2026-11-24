"""N11: Stage B-lite on all five outer folds, iteration 3 (plan section 12, it3_amendment_r3.md, which
replaces section 11, it3_declaration_r3.md, where the two differ).

The question: does training the last encoder blocks raise the score? Every number is "added after the
results" (R10). Outputs are attached only to this module's own decision and dev steps and, if the change is
kept, to N10; never to N7 or N9 (R9). Expert labels are not read: gold rows are only excluded.

Steps (each its own Kaggle notebook):
  run(smoke=True)   GPU. 200 micro-batches per encoder on fold 0: slices/s and peak memory, plus the code
                    paths that otherwise first run hours into a session. No score. Its slices/s decides the
                    schedule of both sessions ("halved" if the slower encoder is under 300 slices/s).
  run(session, folds, schedule)   GPU, two sessions at the same time (a: folds 0, 1, 2; b: folds 3, 4). One
                    process per encoder, one GPU each, working through its folds in order (one encoder only:
                    the two GPUs take different folds). A unit is one encoder on one fold; it uses its own
                    fold's Stage A config, E, head and statistics (knee-n6a only). Order inside a session:
                    1. frozen0 for every fold: this module's forward against N5 (check a) and against
                       knee-n6a's saved predictions (check b: fold macro AUC within 0.0005, else the run is
                       invalid and stops; the largest absolute probability gap is reported beside it).
                    2. all inner work for every fold, on the plan's inner validation group (VAL_TABLE):
                       the head refit on the inner training rows; that head's score with no training; the
                       inner head-only arm at seeds 0 and 100; successive halving over the encoder learning
                       rate (rung 0 = 3 twelfths for each of three rates, the survivor to the end); the
                       registered 2e-5 to the end if it is not the survivor. Per-study inner validation
                       predictions are saved privately for decide().
                    3. the final fits of every fold (head-only seeds 0-4 and b-lite at the chosen rate,
                       seed 0): the test-fold predictions and the b-lite weights go to the private output.
                       No test-fold metric of a trained model is computed or printed.
  decide()          CPU. Reads both sessions' outputs (inner files only; no test-fold file is opened) and
                    writes the keep decision per encoder, with time and sha256 (12.2).
  dev()             CPU. Refuses to run without a "decided" record; checks the prediction files' hashes
                    against it; computes the registered S2 metrics, Delta1 and N14-C2's side AUC, each
                    labelled as a development number (R6).

Schedules are counted in twelfths of a fit's optimiser steps: epochs = round(16,200 / the fold's training
size) (halved: half of it), warm-up = 1 twelfth, cosine to 0 at the end, rung 0 ends after 3 twelfths, the
training loss and the inner validation loss and macro AUC are logged at every twelfth.

Outputs of a session: agg_n11_<session>.json and run.json (aggregates only). Private:
private/n11_<session>/<encoder>_f<fold>/ holds state.json, inner_<arm>.csv, pred_<arm>.csv and
b_lite_seed_0.pt (tail and head, for N10).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import agg, encoders, heads, metrics, paths, safe_log, stage
from .. import units as unit_files
from ..labels import FINDINGS, UID
from . import common, n5

NAME = "n11"
MAIN = encoders.MAIN  # ("dinov2_vits14", "convnext_tiny")
DINO, CONV = MAIN
LRS = (5e-6, 2e-5, 8e-5)
REGISTERED_LR = 2e-5
PASSES = 16200  # study passes per fit: epochs = round(PASSES / the fold's training size)
TWELFTHS = 12
RUNG0 = 3  # rung 0 ends after 3 twelfths (the first quarter)
SCHEDULES = ("full", "halved")
SESSIONS = {"a": (0, 1, 2), "b": (3, 4)}
N_OUTER = 5
# 12.1 item 3: the inner validation group(s) per outer fold, with the fold's test group(s). g_lang values
# are the report-language names N0 writes. The run stops if a fold's test groups differ from this table.
VAL_TABLE = {0: {"test": ("ENGLISH",), "val": ("CROATIAN",)},
             1: {"test": ("CROATIAN",), "val": ("GREEK",)},
             2: {"test": ("GREEK",), "val": ("GERMAN",)},
             3: {"test": ("BOSNIAN", "GERMAN"), "val": ("BULGARIAN", "DUTCH")},
             4: {"test": ("BULGARIAN", "DUTCH"), "val": ("GREEK",)}}
VAL_RANGE = (250, 500)
TIE = 0.005  # rung scores within this are a tie
DECIMALS = 4  # every comparison of a rule is made at 4 decimals, the precision the scores are stored at
ENC_WEIGHT_DECAY = 0.05
MICRO = 2  # studies per micro-batch (1 if 2 do not fit in memory)
SMOKE_MICRO = 200  # micro-batches of the smoke run
MIN_SLICES_PER_S = 300.0
HEAD_ONLY_SEEDS = (0, 1, 2, 3, 4)  # final fits: a noise description only (test-fold numbers)
NULL_SEED = 100  # the inner head-only arm's second seed: s of the keep rule
CHECK_A_STUDIES = 200
CHECK_B_TOL = 0.0005
N_BOOT = 2000
MIN_GAIN = 0.005
MIN_FOLDS_UP = 4
FLIP_PLANES = heads.FLIP_PLANES
INNER_ARMS = ("start", "head_only_seed_0", f"head_only_seed_{NULL_SEED}", "b_lite")  # inner_<arm>.csv
DEV_LABEL = "development: these studies were test studies in iteration 1; likely optimistic"
DISPLAY = "grey; never in a title, never as a gain over iteration 1, never on slide 12 (R6)"
LIMITS = ["the folds share most training studies and two folds share a validation group: sqrt(5) and 4 of 5 are approximate",
          "each fold is judged on one small site", "s is a lower bound on the seed noise of a fine-tuned fit"]
SCOPE = "Stage B-lite on all five folds, iteration 3 (plan section 12): added after the results (R10)"
PLAN = "iteration plan section 12 (it3_amendment_r3)"
CURVE_NOTE = "training loss: dropout on, over the twelfth; validation loss: dropout off, at the twelfth's end"


class InvalidRun(RuntimeError):
    """frozen0 check (b) failed: the run is invalid and nothing else is read."""


class OutOfTime(RuntimeError):
    """The session's soft stop (budget_h) has passed: no new fit starts, and the step ends "partial"."""


class N11Error(RuntimeError):
    """An encoder's task failed (the safe description is in the summary's errors)."""


def _r(x, nd: int = DECIMALS):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def _lr_key(lr: float) -> str:
    return f"{float(lr):g}"


def _step_name(session: str) -> str:
    return f"{NAME}_{session}"


# ---------------------------------------------------------------- the schedule, in twelfths

def schedule_epochs(n_train: int, schedule: str = "full", passes: int = PASSES) -> int:
    """Epochs of a fold: PASSES study passes over its training size, rounded (12 for 1,350 studies, 4 for
    about 4,000); the halved schedule takes half (6 and 2)."""
    if schedule not in SCHEDULES:
        raise ValueError('schedule: "full" or "halved"')
    e = max(1, int(round(passes / max(1, int(n_train)))))
    return e if schedule == "full" else max(1, e // 2)


def twelfth_marks(total_steps: int) -> list:
    """The optimiser step counts at which each twelfth of a schedule ends (the last one is the total)."""
    total = int(total_steps)
    if total < TWELFTHS:
        raise ValueError("a schedule needs at least 12 optimiser steps")
    return [int(k * total / TWELFTHS + 0.5) for k in range(1, TWELFTHS + 1)]


def twelfth_factor(step: int, total_steps: int) -> float:
    """The learning rate at optimiser step `step` (from 0) as a fraction of its peak: linear warm-up over the
    first twelfth, then a cosine to 0 at the end. With 12 whole epochs this is heads.schedule_factor(step,
    steps_per_epoch, 12) step for step (a twelfth is then an epoch)."""
    warm = twelfth_marks(total_steps)[0]
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, int(total_steps) - warm))))


# ---------------------------------------------------------------- the encoder split

class Split:
    """An encoder as a frozen trunk and a trainable tail, on one device, always in eval mode.

    Subclasses give prep (uint8 slices -> model input), trunk (called under no_grad by the caller) and
    tail (trunk output -> [n, D] slice features). tail_prefixes: the parameter-name prefixes of the tail."""

    name = ""
    kind = ""
    expected_trainable: tuple | None = None  # (low, high) number of trainable parameters, if registered

    def __init__(self, model, device, tail_prefixes):
        import torch

        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.tail_prefixes = tuple(tail_prefixes)
        for n, p in self.model.named_parameters():
            p.requires_grad_(n.startswith(self.tail_prefixes))
        self._tail0 = self.tail_state()

    def tail_named_parameters(self) -> list:
        return [(n, p) for n, p in self.model.named_parameters() if n.startswith(self.tail_prefixes)]

    def tail_state(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.tail_named_parameters()}

    def load_tail_state(self, sd: dict) -> None:
        import torch

        with torch.no_grad():
            for n, p in self.tail_named_parameters():
                p.copy_(sd[n].to(p.device))

    def reset(self) -> None:
        """Back to the weights the encoder was built with (every run starts from them)."""
        self.load_tail_state(self._tail0)

    def n_trainable(self) -> int:
        return int(sum(p.numel() for _, p in self.tail_named_parameters()))

    @staticmethod
    def no_decay(name: str, p) -> bool:
        """No weight decay on biases, norm weights and layer-scale (all are one-dimensional)."""
        return p.ndim <= 1 or name.endswith(".bias") or name.endswith("gamma")

    def layer_scale_names(self) -> list:
        return [n for n, _ in self.tail_named_parameters() if n.endswith("gamma")]

    def drop_path_max(self) -> float:
        return float(max([float(getattr(m, "drop_prob", 0.0) or 0.0) for m in self.model.modules()
                          if type(m).__name__ == "DropPath"] or [0.0]))

    def check(self) -> dict:
        """Asserts of the declaration: no batch norm, drop-path 0, eval mode, the registered trainable size."""
        import torch

        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.model.modules()):
            raise AssertionError("the encoder has batch norm: accumulation would not equal one large batch")
        if self.drop_path_max() != 0.0:
            raise AssertionError("drop-path is not 0")
        if self.model.training:
            raise AssertionError("the encoder must stay in eval mode")
        if any(n.startswith(self.tail_prefixes) for n, _ in self.model.named_buffers()):
            raise AssertionError("the trainable tail holds buffers that a checkpoint would miss")
        n = self.n_trainable()
        if self.expected_trainable and not self.expected_trainable[0] <= n <= self.expected_trainable[1]:
            raise AssertionError("the trainable parameter count is outside the registered range")
        named = self.tail_named_parameters()
        return {"parameters": n, "frozen_parameters": int(sum(p.numel() for p in self.model.parameters()) - n),
                "no_decay_parameters": int(sum(p.numel() for k, p in named if self.no_decay(k, p))),
                "drop_path_max": self.drop_path_max(), "batch_norm": False,
                "layer_scale_excluded_from_decay": bool(self.layer_scale_names())}

    def prep(self, u8):
        raise NotImplementedError

    def trunk(self, x):
        raise NotImplementedError

    def tail(self, h):
        raise NotImplementedError


class TimmSplit(Split):
    """The two main encoders, split as registered (plan rev93): DINOv2 ViT-S/14 trains its last 3 blocks
    and the final norm (5.3M parameters); ConvNeXt-T its last stage-4 block and the head norm (4.8M).
    The features are those of encoders.Encoder: ViT = CLS and the mean of the patch tokens after the
    final norm; ConvNeXt = forward_head(pre_logits=True)."""

    VIT_BLOCKS = 3
    EXPECTED = {"vit": (5.2e6, 5.4e6), "convnext": (4.7e6, 4.9e6)}

    def __init__(self, name: str, device: str = "cpu", pretrained: bool = True):
        e = encoders.Encoder(name, device, pretrained=pretrained)
        m = e.model
        self.name, self.kind, self._enc = name, e.kind, e
        self.expected_trainable = self.EXPECTED[e.kind]
        if getattr(m, "grad_checkpointing", False):
            raise AssertionError("gradient checkpointing must be off")
        if e.kind == "vit":
            nb = len(m.blocks)
            if nb <= self.VIT_BLOCKS:
                raise AssertionError("the ViT has too few blocks to split")
            self._trunk_blocks, self._tail_blocks = m.blocks[:nb - self.VIT_BLOCKS], m.blocks[nb - self.VIT_BLOCKS:]
            prefixes = [f"blocks.{i}." for i in range(nb - self.VIT_BLOCKS, nb)] + ["norm."]
            self.n_prefix = e.n_prefix
        else:
            last = m.stages[-1]
            if getattr(last, "grad_checkpointing", False):
                raise AssertionError("gradient checkpointing must be off")
            nb = len(last.blocks)
            self._trunk_blocks, self._tail_block = last.blocks[:nb - 1], last.blocks[nb - 1]
            prefixes = [f"stages.{len(m.stages) - 1}.blocks.{nb - 1}.", "head.norm."]
        super().__init__(m, device, prefixes)

    def prep(self, u8):
        return self._enc._prep(u8)

    def trunk(self, x):
        m = self.model
        if self.kind == "vit":
            x = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
            return self._trunk_blocks(x)
        x = m.stem(x)
        for st in m.stages[:-1]:
            x = st(x)
        return self._trunk_blocks(m.stages[-1].downsample(x))

    def tail(self, h):
        import torch

        m = self.model
        if self.kind == "vit":
            t = m.norm(self._tail_blocks(h))
            return torch.cat([t[:, 0], t[:, self.n_prefix:].mean(1)], dim=1)
        return m.forward_head(m.norm_pre(self._tail_block(h)), pre_logits=True)


def _factory(task: dict):
    f = task.get("factory")
    return f(task["encoder"], task["device"]) if f is not None else TimmSplit(task["encoder"], task["device"])


# ---------------------------------------------------------------- feature sources

class BlockCache:
    """One plane's uint8 cache [studies, S, H, W], read one whole study per read call (pread).

    Why not index the memmap: a random study read through a memory map arrives as hundreds of page faults,
    each a round trip on a network-backed input disk (the first smoke run: 80 slices/s for both encoders,
    about 4 MB/s). One pread per study is one request of S x H x W bytes."""

    def __init__(self, path, shape):
        self.path, self.shape = str(path), tuple(int(x) for x in shape)
        self.row_bytes = int(np.prod(self.shape[1:]))
        self._fd = None

    def __getitem__(self, rows) -> np.ndarray:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        rows = np.atleast_1d(np.asarray(rows)).astype(np.int64)
        out = np.empty((len(rows), *self.shape[1:]), np.uint8)
        for i, r in enumerate(rows):
            if not 0 <= r < self.shape[0]:
                raise IndexError("cache row out of range")
            view = memoryview(out[i]).cast("B")
            done = 0
            while done < self.row_bytes:
                n = os.preadv(self._fd, [view[done:]], int(r) * self.row_bytes + done)
                if n <= 0:
                    raise OSError("short read from a cache file")
                done += n
        return out

    def __del__(self):
        try:
            if self._fd is not None:
                os.close(self._fd)
        except Exception:  # noqa: BLE001
            pass


class EncSource:
    """Slice features from the caches through the split encoder: the B-lite forward code.

    rows -> one float32 tensor [m, S, D] per plane. The trunk always runs under no_grad; the tail runs
    with a graph only when grad is true. flipped: coronal and axial slices mirrored left-right (as N5).
    The uint8 slices go to the GPU as they are; the float conversion and normalisation happen there."""

    def __init__(self, enc: Split, caches: dict, planes, amp: bool):
        self.enc, self.caches, self.planes, self.amp = enc, caches, tuple(planes), bool(amp)
        self.read_s = 0.0  # seconds spent reading the caches (host side)

    def read(self, rows) -> list:
        """One uint8 array [m, S, H, W] per plane, whole studies."""
        t0 = time.time()
        out = [np.ascontiguousarray(self.caches[p][rows]) for p in self.planes]
        self.read_s += time.time() - t0
        return out

    def to_device(self, arrays, flipped: bool = False):
        """(uint8 tensor [planes x m x S, H, W] on the device, m, S)."""
        import torch

        dev = self.enc.device
        parts = []
        for p, a in zip(self.planes, arrays):
            a = torch.from_numpy(a).to(dev)
            parts.append(a.flip(-1) if flipped and p in FLIP_PLANES else a)
        m, S = parts[0].shape[:2]
        return torch.cat(parts).reshape(len(self.planes) * m * S, *parts[0].shape[2:]), m, S

    def trunk(self, x):
        import torch

        with torch.autocast(device_type=self.enc.device.type, dtype=torch.float16, enabled=self.amp), torch.no_grad():
            return self.enc.trunk(self.enc.prep(x))

    def tail(self, h, m: int, S: int, grad: bool):
        import torch

        with torch.autocast(device_type=self.enc.device.type, dtype=torch.float16, enabled=self.amp):
            if grad:
                f = self.enc.tail(h)
            else:
                with torch.no_grad():
                    f = self.enc.tail(h)
        f = f.float().reshape(len(self.planes), m, S, -1)
        return [f[i] for i in range(len(self.planes))]

    def __call__(self, rows, grad: bool = False, flipped: bool = False):
        x, m, S = self.to_device(self.read(np.asarray(rows)), flipped)
        return self.tail(self.trunk(x), m, S, grad)


def localise_caches(root=None, local_dir=None, margin: int = 2 << 30, chunk: int = 32 << 20) -> tuple[dict, dict]:
    """Copy the three plane caches once, sequentially, from the input disk to local disk.

    Returns ({plane: path to read from}, a report). The target is the first of local_dir, $KNEE_LOCAL_CACHE,
    /kaggle/temp and the temp folder with room for the files plus `margin`. The copy is checked: the same
    size, the manifest's sha256 when it has one (hashed while copying), and 16 sampled studies equal to the
    source. No room, a failed copy or a failed check: the source files are read instead (in whole-study
    blocks), and the report says so. local_dir False switches the copy off."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    root = Path(root) if root else None
    src, mans = {}, {}
    for p in heads.PLANES:
        mp_ = paths.find_file(f"cache_{p}_manifest.json", root)
        if mp_ is None:
            raise FileNotFoundError("cache not attached")
        mans[p] = json.loads(mp_.read_text())
        src[p] = mp_.parent / mans[p]["file"]
    same = {p: str(v) for p, v in src.items()}
    total = sum(v.stat().st_size for v in src.values())
    info = {"gb": round(total / 2 ** 30, 2), "where": "source", "reads": "one pread per study"}
    if local_dir is False:
        return same, {**info, "note": "the local copy was switched off"}
    target = None
    for cand in ([local_dir] if local_dir else [os.environ.get("KNEE_LOCAL_CACHE"), "/kaggle/temp", tempfile.gettempdir()]):
        if not cand:
            continue
        try:
            d = Path(cand) / f"knee_n11_cache_{os.getpid()}"
            d.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(d).free >= total + int(margin):
                target = d
                break
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue
    if target is None:
        return same, {**info, "note": "no local folder with enough free space: the source files are read"}

    def copy(p):
        h = hashlib.sha256()
        with open(src[p], "rb", buffering=0) as fi, open(target / src[p].name, "wb", buffering=0) as fo:
            while True:
                b = fi.read(chunk)
                if not b:
                    break
                h.update(b)
                fo.write(b)
        return h.hexdigest()

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=len(src)) as ex:
            digests = dict(zip(src, ex.map(copy, list(src))))
        ok, sha_ok = True, {}
        for p in src:
            dst = target / src[p].name
            ok = ok and dst.stat().st_size == src[p].stat().st_size
            if mans[p].get("sha256"):
                sha_ok[p] = digests[p] == mans[p]["sha256"]
                ok = ok and sha_ok[p]
            shape = tuple(mans[p]["shape"])
            rows = np.unique(np.linspace(0, shape[0] - 1, min(16, shape[0])).round().astype(int))
            ok = ok and np.array_equal(BlockCache(dst, shape)[rows], BlockCache(src[p], shape)[rows])
    except OSError:
        ok, sha_ok = False, {}
    secs = time.time() - t0
    if not ok:
        shutil.rmtree(target, ignore_errors=True)
        return same, {**info, "note": "the local copy failed or did not equal the source: the source files are read",
                      "copy_seconds": round(secs, 1)}
    return ({p: str(target / src[p].name) for p in src},
            {**info, "where": "local copy", "local_dir": str(target), "copy_seconds": round(secs, 1),
             "copy_mb_per_s": round(total / 2 ** 20 / max(secs, 1e-6), 1),
             "sha256_equals_manifest": all(sha_ok.values()) if sha_ok else None, "sampled_studies_equal": True,
             "same_size": True})


class FeatSource:
    """The frozen0 features, kept on the device (the head-only arm trains on them)."""

    def __init__(self, n: int, rows, x: dict, flip_rows, flip: dict, planes):
        self.planes = tuple(planes)
        self.x, self.flip = x, flip
        self.pos = np.full(n, -1, int)
        self.pos[np.asarray(rows)] = np.arange(len(rows))
        self.fpos = np.full(n, -1, int)
        self.fpos[np.asarray(flip_rows, int)] = np.arange(len(flip_rows))

    def __call__(self, rows, grad: bool = False, flipped: bool = False):
        rows = np.asarray(rows)
        i = self.pos[rows]
        if (i < 0).any():
            raise KeyError("a study without frozen0 features")
        out = []
        for p in self.planes:
            if flipped and p in self.flip:
                j = self.fpos[rows]
                if (j < 0).any():
                    raise KeyError("a study without flipped frozen0 features")
                out.append(self.flip[p][j].float())
            else:
                out.append(self.x[p][i].float())
        return out


# ---------------------------------------------------------------- helpers kept from the fold-0 version

def _unit(enc: str, fold: int = 0) -> str:
    """A unit is one encoder on one outer fold (as Stage A's units): its folder and its run.json key."""
    return f"{enc}_f{int(fold)}"


def _guard(train_idx, val_idx, forbidden) -> None:
    """The plan's run-time leakage assertion, for every fit of this module (heads.fit calls it itself)."""
    heads._check_guard(train_idx, val_idx, forbidden)


def _sync(dev) -> None:
    if dev.type == "cuda":
        import torch

        torch.cuda.synchronize(dev)


def _rng_state(dev):
    import torch

    return torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.get_rng_state()


def _set_rng_state(dev, state) -> None:
    import torch

    if dev.type == "cuda":
        torch.cuda.set_rng_state(state, dev)
    else:
        torch.set_rng_state(state)


def _scaler(amp: bool):
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=amp)


def _z(c, xs):
    return [(x - mu) / sd for x, (mu, sd) in zip(xs, c.stats_l)]


def batch_backward(c, r, b: np.ndarray, micro: int | None = None, limit: int | None = None):
    """Gradients of one effective batch, accumulated over micro-batches of `micro` studies: the batch's
    weight sum is computed first and each micro-batch's loss sum is divided by it, so the result equals
    one large batch (no batch norm). No optimiser step. Returns (loss sum, weight sum, mean |z| sum,
    micro-batches done)."""
    import torch
    import torch.nn.functional as F

    micro = int(micro or c.micro)
    dev = c.device
    bt = torch.as_tensor(b, device=dev)
    wsum = c.bank.w[bt].sum().clamp_min(1.0)
    r.opt.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=dev)
    w_raw = torch.zeros((), device=dev)
    z_sum = torch.zeros((), device=dev)
    done = 0
    for j in range(0, len(b), micro):
        mb = b[j:j + micro]
        mt = bt[j:j + micro]
        zs = _z(c, r.src(mb, grad=r.with_encoder))
        w = c.bank.w[mt]
        ls = F.binary_cross_entropy_with_logits(r.head(zs), c.bank.y[mt], weight=w, pos_weight=r.pw, reduction="sum")
        r.scaler.scale(ls / wsum).backward()
        loss_sum += ls.detach()
        w_raw += w.sum()
        z_sum += sum(z.detach().abs().mean() for z in zs) / len(zs)
        done += 1
        r.slices += len(mb) * c.S * len(c.planes)
        if r.clock is not None:  # the probe: a time stamp per micro-batch
            _sync(dev)
            r.clock.append((time.time(), r.slices))
        if limit is not None and done >= limit:
            break
    return loss_sum, w_raw, z_sum, done


def _logits(c, src, head, rows, flipped: bool = False):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(rows), c.eval_studies):
            out.append(head(_z(c, src(rows[i:i + c.eval_studies], grad=False, flipped=flipped))).float())
    return torch.cat(out) if out else torch.zeros((0, heads.N_OUT), device=c.device)


def predict(c, src, head, rows, use_flip: bool = True) -> np.ndarray:
    """Probabilities [len(rows), 12], head in eval mode; unresolved-laterality studies (flip_row >= 0)
    average the stored and the mirrored coronal and axial slices, as heads.predict."""
    import torch

    rows = np.asarray(rows)
    head.eval()
    p = torch.sigmoid(_logits(c, src, head, rows)).cpu().numpy().astype(np.float32)
    if use_flip and any(q in FLIP_PLANES for q in c.planes):
        m = c.d.flip_row[rows] >= 0
        if m.any():
            pf = torch.sigmoid(_logits(c, src, head, rows[m], flipped=True)).cpu().numpy()
            p[m] = (p[m] + pf) / 2
    return p


def _score4(v) -> float:
    return round(float(v), DECIMALS) if v is not None and np.isfinite(v) else float("-inf")


def _tied(a: float, b: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and round(abs(a - b), DECIMALS) <= TIE)


def tie_fallback(pair) -> float:
    """A tied pair: 2e-5 if it is one of the two, else the lower of the two (the smaller change)."""
    pair = [float(x) for x in pair]
    return REGISTERED_LR if any(math.isclose(x, REGISTERED_LR) for x in pair) else min(pair)


def select_rung0(scores: dict, rung1_tie: bool = True) -> dict:
    """Which learning rates continue after rung 0. scores: {lr: validation macro AUC at the rung's end}.
    The best one continues; if the top two are within 0.005 both continue (rung1_tie false, the cut: the
    tie fallback alone). No finite score: the registered 2e-5."""
    order = sorted(scores, key=lambda lr: (-_score4(scores[lr]), not math.isclose(lr, REGISTERED_LR), lr))
    if not order or not np.isfinite(_score4(scores[order[0]])):
        return {"continue": [REGISTERED_LR], "tied": False, "note": "no finite rung-0 score: the registered rate"}
    top = order[0]
    if len(order) > 1 and _tied(_score4(scores[top]), _score4(scores[order[1]])):
        pair = sorted([top, order[1]])
        third = (len(order) > 2 and _tied(_score4(scores[top]), _score4(scores[order[2]])))
        more = "; a third rate within 0.005 was dropped by order (score, then 2e-5, then the lower)" if third else ""
        if rung1_tie:
            return {"continue": pair, "tied": True, "note": "top two within 0.005: both continue" + more}
        return {"continue": [tie_fallback(pair)], "tied": True, "pair": pair,
                "note": "top two within 0.005; rung 1 for the pair was cut: the tie fallback continues" + more}
    return {"continue": [top], "tied": False, "note": "the best rung-0 rate continues"}


def select_final(continued, final_scores: dict) -> tuple[float, str]:
    """The chosen learning rate after rung 1. One survivor: it. A pair: the higher final score decides; if
    the two are still within 0.005 (or not both finite), the tie fallback."""
    continued = [float(x) for x in continued]
    if len(continued) == 1:
        return continued[0], "the single rung-1 survivor"
    a, b = continued
    sa, sb = _score4(final_scores.get(a)), _score4(final_scores.get(b))
    if _tied(sa, sb) or not (np.isfinite(sa) and np.isfinite(sb)):
        return tie_fallback(continued), "tied at rung 0 and at the last epoch: 2e-5 if in the pair, else the lower"
    return (a if sa > sb else b), "tied at rung 0: the last epoch decided"


def record_chosen(working, enc: str, rec: dict) -> None:
    """Write one unit's chosen learning rate into run.json under its key (_unit; both encoders' processes
    write it: locked, and agg.write replaces the file atomically, so a reader never sees half a file)."""
    working = Path(working)
    lock = working / "private" / f"{NAME}_run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        p = working / "run.json"
        j = json.loads(p.read_text()) if p.exists() else {}
        j.setdefault("n11_chosen_encoder_lr", {})[enc] = rec
        agg.write("run", j, folder=working)


def chosen_recorded(working, enc: str, lr: float) -> bool:
    p = Path(working) / "run.json"
    if not p.exists():
        return False
    rec = (json.loads(p.read_text()).get("n11_chosen_encoder_lr") or {}).get(enc) or {}
    return rec.get("lr") is not None and math.isclose(float(rec["lr"]), float(lr))


def _device_name(dev) -> str:
    if dev.type != "cuda":
        return dev.type
    try:
        import torch

        return str(torch.cuda.get_device_name(dev))
    except Exception:  # noqa: BLE001 - descriptive only
        return "unknown"


def _peak_gb(dev):
    if dev.type != "cuda":
        return None
    import torch

    return round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)


def _features(c, rows, flip_rows) -> tuple[FeatSource, float]:
    """The frozen encoder's features of `rows` (and, mirrored, of `flip_rows`) through this module's forward,
    stored on the device (fp16 on a GPU, as N5 stores them). Returns the source and the seconds it took."""
    import torch

    dev = c.device
    c.enc.reset()
    rows, frows = np.asarray(rows), np.asarray(flip_rows, int)
    x = {p: torch.empty((len(rows), c.S, c.dim), dtype=c.fdtype, device=dev) for p in c.planes}
    _sync(dev)
    t0 = time.time()
    for i in range(0, len(rows), c.eval_studies):
        for p, f in zip(c.planes, c.src(rows[i:i + c.eval_studies])):
            x[p][i:i + f.shape[0]] = f.to(c.fdtype)
    flip = {p: torch.empty((len(frows), c.S, c.dim), dtype=c.fdtype, device=dev) for p in c.planes if p in FLIP_PLANES}
    for i in range(0, len(frows), c.eval_studies):
        for p, f in zip(c.planes, c.src(frows[i:i + c.eval_studies], flipped=True)):
            if p in flip:
                flip[p][i:i + f.shape[0]] = f.to(c.fdtype)
    _sync(dev)
    return FeatSource(c.d.n, rows, x, frows, flip, c.planes), time.time() - t0


def _map(tasks: list, parallel: bool) -> list:
    if not parallel:
        return [_task(t) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_task, {**t, "spawned": True}) for t in tasks]
        for t, f in zip(tasks, futs):
            try:
                out.append(f.result())
            except (KeyboardInterrupt, SystemExit):  # the watchdog or a manual stop: never swallowed
                raise
            except BaseException as e:  # noqa: BLE001 - a crashed worker
                out.append({"ok": False, "encoder": t["encoder"],
                            "errors": {_unit(t["encoder"], (t.get("folds") or [0])[0]): safe_log.describe(type(e), e.__traceback__)}})
    return out


def _pct(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return {"lo": None, "hi": None, "valid_share": 0.0}
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return {"lo": _r(lo), "hi": _r(hi), "valid_share": _r(float(ok.mean()), 3)}


# ---------------------------------------------------------------- units, state, index sets

def _unit_dir(working, step: str, enc: str, fold: int) -> Path:
    p = Path(working) / "private" / step / _unit(enc, fold)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_atomic(p: Path, text: str) -> None:
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")  # a reader never sees half a file
    tmp.write_text(text)
    os.replace(tmp, p)


def _load_state(udir: Path) -> dict:
    p = Path(udir) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(udir: Path, st: dict) -> None:
    text = json.dumps(st, indent=1, default=agg._json_default)
    agg._check(json.loads(text))  # the state becomes the summary's unit block: aggregates only
    _write_atomic(Path(udir) / "state.json", text)


def inner_split(d, fold: int, table: dict | None = None, val_range=VAL_RANGE) -> dict:
    """The fold's inner split by the plan's table (12.1 item 3): the validation set is one or two g_lang
    groups of the training set. Asserted: the fold's test groups are the table's, the validation groups are
    in the training set and not in the test set, and the validation set holds 250-500 studies. Any mismatch
    stops the run (the table is corrected first). Returns the row sets and the groups' sizes."""
    table = VAL_TABLE if table is None else table
    tr, te, _ = d.split(fold)
    g = d.g_lang.astype(str)
    test_groups = sorted(set(g[te].tolist()))
    row = table.get(fold) or table.get(str(fold))
    if row is None:
        raise AssertionError("the fold has no row in the validation table")
    if test_groups != sorted(str(x) for x in row["test"]):
        raise AssertionError("the fold's test groups differ from the plan's table: correct the table first")
    val = sorted(str(x) for x in row["val"])
    if set(val) & set(test_groups):
        raise AssertionError("a validation group is in the fold's test set")
    if not set(val) <= set(g[tr].tolist()):
        raise AssertionError("a validation group of the table is not in the fold's training set")
    m = np.isin(g[tr], val)
    iva_all = tr[m]
    if val_range and not val_range[0] <= len(iva_all) <= val_range[1]:
        raise AssertionError("the inner validation set is outside 250-500 studies")
    if set(iva_all.tolist()) & set(te.tolist()):
        raise AssertionError("a validation study is in the fold's test set")
    return {"itr": tr[~m], "iva_all": iva_all, "iva": iva_all[d.scored[iva_all]], "val_set": "+".join(val),
            "val_groups": {v: agg.count(int((g[tr] == v).sum())) for v in val},
            "test_groups": {v: agg.count(int((g[te] == v).sum())) for v in test_groups}}


def _shared(task: dict):
    """What every unit of one process shares: the data, the caches, the embeddings and the encoder."""
    import torch

    root = Path(task["root"]) if task.get("root") else None
    dev = torch.device(task["device"])
    d = stage.StageData(root, task.get("expect"))
    sh = SimpleNamespace(d=d, root=root, base=root or paths.INPUT, name=task["encoder"], device=dev,
                         amp=dev.type == "cuda", working=Path(task["working"]), step=task["step"],
                         micro=int(task.get("micro") or MICRO), eval_studies=int(task.get("eval_studies") or 4),
                         task=task)
    sh.caches, sh.upstream_n4 = {}, {}
    for p in heads.PLANES:
        mm, _, st, up, _ = n5._load_plane(p, root)
        if st[UID].astype(str).tolist() != d.uids.tolist():
            raise AssertionError("cache rows differ from N5's index")
        # the same bytes, read in whole-study blocks, from the local copy when the parent made one
        sh.caches[p] = BlockCache((task.get("cache_paths") or {}).get(p) or mm.filename, mm.shape)
        sh.upstream_n4[p] = up
    sh.S = int(sh.caches[heads.PLANES[0]].shape[1])
    sh.bank = d.bank(sh.name, str(dev))
    sh.dim = int(sh.bank.x[heads.PLANES[0]].shape[-1])
    sh.enc = _factory(task)
    sh.trainable = sh.enc.check()
    sh.fdtype = torch.float16 if sh.amp else torch.float32
    sh.unit_dirs = unit_files.unit_dirs(sh.base)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return sh


def _context(sh, fold: int):
    """One unit (this encoder on one outer fold): its index sets, its own fold's Stage A config, E, head and
    statistics (knee-n6a only) and its forbidden rows."""
    import torch

    d, task = sh.d, sh.task
    c = SimpleNamespace(**vars(sh))
    c.fold, c.unit = int(fold), _unit(sh.name, fold)
    c.dir = _unit_dir(sh.working, sh.step, sh.name, fold)
    c.tr, c.te, c.te_s = d.split(c.fold)
    c.gold = d.gold_rows()
    if d.is_gold[c.tr].any() or d.is_gold[c.te].any():
        raise AssertionError("a gold study is in the fold's training or test set")
    sp = inner_split(d, c.fold, task.get("val_table"), task.get("val_range"))
    c.itr, c.iva_all, c.iva, c.split = sp["itr"], sp["iva_all"], sp["iva"], sp
    c.forbid_outer = np.concatenate([c.te, c.gold])  # every fit: test rows and gold rows
    c.forbid_inner = np.concatenate([c.iva_all, c.te, c.gold])  # the inner refit: the validation rows too
    ud = sh.unit_dirs.get((sh.name, c.fold))
    if ud is None:
        raise FileNotFoundError("the fold's Stage A unit is not attached")
    if ud.parent.parent.name != "n6a":  # private/n6a/units/<encoder>_f<k>: never an iteration-2 unit or a rerun
        raise AssertionError("the Stage A unit must be knee-n6a's")
    c.unit_dir = ud
    c.chain = unit_files.check_units(d, {(sh.name, c.fold): ud})
    if not c.chain["s2_folds_match_units"]:
        raise unit_files.ChainError("the Stage A unit does not match the attached S2 folds")
    cj = json.loads((ud / "config.json").read_text())
    c.cfg, c.E = heads.config_from(cj["config"]), int(cj["E"])
    se = cj.get("schedule_epochs")  # the unit's schedule length (n6._unit); the registered 30 is passed as None
    c.stage_a_schedule = None if se is None or int(se) == heads.MAX_EPOCHS else int(se)
    c.planes = heads.PLANE_SETS[c.cfg.planes]
    raw = torch.load(ud / "stats.pt", map_location="cpu")
    c.stats = {p: (mu.float().to(sh.device), sd.float().to(sh.device)) for p, (mu, sd) in raw.items()}
    c.stats_l = [c.stats[p] for p in c.planes]
    c.head_state = torch.load(ud / "head.pt", map_location="cpu")
    c.src = EncSource(sh.enc, {p: sh.caches[p] for p in c.planes}, c.planes, sh.amp)
    c.schedule = task.get("schedule") or "full"
    c.epochs = schedule_epochs(len(c.tr), c.schedule, int(task.get("passes") or PASSES))
    return c


def inner_refit(c):
    """The head the inner runs start from: the fold's Stage A config, E and schedule length, seed 0, the
    cached N5 embeddings, the fixed Stage A statistics, on the inner training rows only. Forbidden: the
    validation rows, the test rows and the gold rows (the leakage guard raises if one reaches the fit)."""
    r = heads.fit(c.bank, c.cfg, c.itr, epochs=c.E, stats=c.stats, forbidden=c.forbid_inner, seed=0,
                  schedule_epochs=c.stage_a_schedule)
    state = {k: v.detach().cpu().clone() for k, v in r.model.state_dict().items()}
    if all(bool((state[k] == c.head_state[k]).all()) for k in state):
        raise AssertionError("the inner refit equals Stage A's head")
    return state


# ---------------------------------------------------------------- the training loop

def new_run(c, src, head_state: dict, rows, lr_enc: float, seed: int, val_rows=None, with_encoder: bool = True,
            epochs: int | None = None):
    """A fit of this module's loop, at step 0.

    AdamW (the heads' betas and eps). Groups: the trainable encoder weights (lr_enc, weight decay 0.05),
    the encoder's biases, norm weights and layer-scale (lr_enc, no decay), the head (its Stage A peak rate
    and weight decay). All share twelfth_factor over the fit's total steps = epochs (default: the fold's)
    x its steps per epoch. Effective batch = the head's Stage A batch size; pos_weight from the fit's own
    training rows, as heads.fit. with_encoder false (head-only on the frozen0 features): the head group alone."""
    import torch

    r = SimpleNamespace(src=src, rows=np.asarray(rows), lr_enc=float(lr_enc), seed=int(seed),
                        val_rows=None if val_rows is None else np.asarray(val_rows), with_encoder=bool(with_encoder),
                        step=0, k=0, history=[], slices=0, train_s=0.0, skipped=0, clock=None, perm=None, pos=0)
    dev = c.device
    r.head = heads.make_head(c.cfg, c.dim, len(c.planes))
    r.head.load_state_dict(head_state)
    r.head.to(dev)
    groups, r.peaks = [], []
    if with_encoder:
        c.enc.reset()
        dec, nod = [], []
        for n, p in c.enc.tail_named_parameters():
            (nod if c.enc.no_decay(n, p) else dec).append(p)
        for ps, wd in ((dec, ENC_WEIGHT_DECAY), (nod, 0.0)):
            if ps:
                groups.append({"params": ps, "lr": r.lr_enc, "weight_decay": wd})
                r.peaks.append(r.lr_enc)
    groups.append({"params": list(r.head.parameters()), "lr": c.cfg.lr, "weight_decay": c.cfg.weight_decay})
    r.peaks.append(float(c.cfg.lr))
    r.opt = torch.optim.AdamW(groups, lr=c.cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    r.scaler = _scaler(c.amp and with_encoder)
    r.gen = torch.Generator().manual_seed(r.seed)  # the batch order, as heads.fit
    if dev.type == "cuda":  # the head's dropout stream
        with torch.cuda.device(dev):
            torch.cuda.manual_seed(r.seed)
    else:
        torch.manual_seed(r.seed)
    r.bs = max(1, int(c.cfg.batch_size))
    r.spe = math.ceil(len(r.rows) / r.bs)
    r.epochs = int(c.epochs if epochs is None else epochs)
    r.total = r.epochs * r.spe
    r.marks = twelfth_marks(r.total)
    yb = c.bank.binary[r.rows]
    pos, neg = (yb == 1).sum(0), (yb == 0).sum(0)
    r.pw = torch.tensor(np.minimum(neg / np.maximum(pos, 1), c.cfg.pos_cap), dtype=torch.float32, device=dev)
    _zero_acc(c, r)
    return r


def _zero_acc(c, r) -> None:
    import torch

    r.acc = [torch.zeros((), device=c.device) for _ in range(3)]  # loss sum, weight sum, mean |z| sum
    r.acc_n = 0


def run_to(c, r, twelfth: int, limit: tuple | None = None) -> None:
    """Train r up to the end of twelfth `twelfth` (1-12) of its schedule. Head in train mode (dropout),
    encoder in eval mode. One optimiser step, one schedule step and one scaler update per effective batch;
    batches are drawn epoch by epoch in heads.fit's order. At every twelfth: the training loss (weighted
    mean over the twelfth, dropout on), mean |z| of the training features and, with validation rows, the
    validation loss and macro AUC (no flips, as Stage A's inner fits).
    limit: ("micro", n) stops after n micro-batches (the smoke run): no record, no score."""
    import torch

    dev, n = c.device, len(r.rows)
    target = r.marks[int(twelfth) - 1]
    micro_done = 0
    while r.step < target:
        if r.with_encoder and c.enc.model.training:
            raise AssertionError("the encoder must stay in eval mode")
        r.head.train()
        if r.perm is None or r.pos >= n:  # a new epoch
            r.perm, r.pos = r.rows[torch.randperm(n, generator=r.gen).numpy()], 0
        _sync(dev)
        t0 = time.time()
        f = twelfth_factor(r.step, r.total)
        for g, peak in zip(r.opt.param_groups, r.peaks):
            g["lr"] = peak * f
        left = None if not limit else limit[1] - micro_done
        ls, w, z, done = batch_backward(c, r, r.perm[r.pos:r.pos + r.bs], limit=left)
        micro_done += done
        if left is not None and micro_done >= limit[1]:
            _sync(dev)
            r.train_s += time.time() - t0
            return  # the smoke run ends inside a batch: no optimiser step for a partial batch
        scale0 = r.scaler.get_scale()
        r.scaler.step(r.opt)
        r.scaler.update()
        if r.scaler.get_scale() < scale0:
            r.skipped += 1  # a non-finite gradient: the scaler skipped this step
        r.step += 1
        r.pos += r.bs
        r.acc[0] += ls
        r.acc[1] += w
        r.acc[2] += z
        r.acc_n += done
        _sync(dev)
        r.train_s += time.time() - t0
        if r.step == r.marks[r.k]:
            rec = {"train_loss": float(r.acc[0] / r.acc[1].clamp_min(1e-6)), "mean_abs_z": float(r.acc[2] / max(r.acc_n, 1))}
            if r.val_rows is not None:
                rec.update(_validate(c, r))
            r.history.append(rec)
            r.k += 1
            _zero_acc(c, r)


def _validate(c, r) -> dict:
    """Inner validation: the same loss (the fit's pos_weight) and the macro AUC over the findings with both
    classes in the validation set's scored studies (report labels), no flip averaging."""
    import torch
    import torch.nn.functional as F

    r.head.eval()
    lg = _logits(c, r.src, r.head, r.val_rows)
    vt = torch.as_tensor(r.val_rows, device=c.device)
    w = c.bank.w[vt]
    loss = F.binary_cross_entropy_with_logits(lg, c.bank.y[vt], weight=w, pos_weight=r.pw, reduction="sum")
    r.val_pred = torch.sigmoid(lg).cpu().numpy()
    score = metrics.macro_available(r.val_pred, c.bank.eval_y[r.val_rows], list(range(heads.N_OUT)))
    return {"val_loss": float(loss / w.sum().clamp_min(1e-6)),
            "val_macro_auc": float(score) if np.isfinite(score) else None}


def save_checkpoint(c, r, path: Path) -> None:
    """Everything a run needs to continue exactly from a twelfth: the trainable weights, the optimiser, the
    scaler, the step, the epoch's batch order and position, its generator and the dropout stream."""
    import torch

    torch.save({"tail": c.enc.tail_state() if r.with_encoder else {}, "head": r.head.state_dict(),
                "opt": r.opt.state_dict(), "scaler": r.scaler.state_dict(), "step": r.step, "k": r.k,
                "perm": r.perm, "pos": r.pos, "gen": r.gen.get_state(), "rng": _rng_state(c.device),
                "history": r.history, "skipped": r.skipped, "slices": r.slices, "train_s": r.train_s,
                "lr_enc": r.lr_enc, "total": r.total, "seed": r.seed}, path)


def load_checkpoint(c, r, path: Path) -> None:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck["lr_enc"], ck["total"], ck["seed"]) != (r.lr_enc, r.total, r.seed):
        raise AssertionError("the checkpoint belongs to another run")
    if r.with_encoder:
        c.enc.load_tail_state(ck["tail"])
    r.head.load_state_dict(ck["head"])
    r.opt.load_state_dict(ck["opt"])
    r.scaler.load_state_dict(ck["scaler"])
    r.step, r.k, r.history = int(ck["step"]), int(ck["k"]), list(ck["history"])
    r.perm, r.pos = ck["perm"], int(ck["pos"])
    r.skipped, r.slices, r.train_s = int(ck["skipped"]), int(ck["slices"]), float(ck["train_s"])
    r.gen.set_state(ck["gen"])
    _set_rng_state(c.device, ck["rng"])
    _zero_acc(c, r)


def _curve(r) -> dict:
    keys = [k for k in ("train_loss", "val_loss", "val_macro_auc", "mean_abs_z") if any(k in h for h in r.history)]
    return {k: [_r(h.get(k)) for h in r.history] for k in keys}


def _fit_summary(r) -> dict:
    out = {"schedule_share": round(r.k / TWELFTHS, 3), "curve": _curve(r), "train_seconds": round(r.train_s, 1),
           "slices_per_s": round(r.slices / max(r.train_s, 1e-6), 1),
           "skipped_step_share": round(r.skipped / max(r.step, 1), 4),
           "config": {"epochs": r.epochs, "steps": r.total, "steps_per_epoch": r.spe, "seed": r.seed},
           "mean_abs_z_last": _r(r.history[-1]["mean_abs_z"]) if r.history else None}
    vl = [h.get("val_loss") for h in r.history]
    if vl and all(v is not None for v in vl):  # where the validation loss bottoms: reported, not used
        out["val_loss_min"] = {"schedule_share": round((int(np.argmin(vl)) + 1) / TWELFTHS, 3), "value": _r(min(vl))}
    return out


# ---------------------------------------------------------------- run.json, files

def predict_test(c, src, head, trained: bool, lr: float | None = None) -> np.ndarray:
    """The fold's scored test studies, with flip averaging. A trained model is refused until the unit's
    chosen learning rate stands in run.json. No metric is computed from a trained model's output here."""
    if trained and (lr is None or not chosen_recorded(c.working, c.unit, lr)):
        raise AssertionError("the chosen learning rate must be in run.json before a trained model sees the test fold")
    return predict(c, src, head, c.te_s, use_flip=True)


def sha16(p: Path) -> str:
    return common.sha16(p)


def _save_rows(c, name: str, rows, P: np.ndarray, st: dict, extra: dict) -> None:
    """A private per-study file: UID, the fold, the probabilities and the report labels (uncertain = negative)
    of the same studies, so that decide() and dev() need no other input. Its hash goes into the state."""
    df = pd.DataFrame(P, columns=FINDINGS)
    for j, f in enumerate(FINDINGS):
        df[f"y_{f}"] = c.d.eval_y[rows, j]
    for k, v in reversed(list(extra.items())):
        df.insert(0, k, v)
    df.insert(0, "s2_fold", c.fold)
    df.insert(0, UID, c.d.uids[rows])
    df.to_csv(c.dir / name, index=False)
    st.setdefault("files_sha16", {})[name] = sha16(c.dir / name)


def _save_pred(c, arm: str, P: np.ndarray, st: dict) -> None:
    _save_rows(c, f"pred_{arm}.csv", c.te_s, P, st, {"resolved": c.d.flip_row[c.te_s] < 0})


def _save_inner(c, arm: str, P: np.ndarray, st: dict) -> None:
    _save_rows(c, f"inner_{arm}.csv", c.iva, P, st, {"val_set": c.split["val_set"]})


# ---------------------------------------------------------------- a unit's blocks

def _invalid_flag(working, step: str) -> Path:
    return Path(working) / "private" / step / "INVALID"


def _abort_if_invalid(c) -> None:
    if _invalid_flag(c.working, c.step).exists():
        raise InvalidRun("frozen0 check (b) failed for a unit: nothing else is read")
    soft = c.task.get("soft_stop")
    if soft and time.time() > float(soft):  # called before every fit: a late session ends "partial", not FAILED
        raise OutOfTime("the session's budget is used up: no new fit starts")


def _base_block(c) -> dict:
    d = c.d
    pos = (d.eval_y[c.iva] == 1).sum(0)
    neg = (d.eval_y[c.iva] == 0).sum(0)
    bs = max(1, int(c.cfg.batch_size))
    return {
        "encoder": c.name, "fold": c.fold, "session": c.task.get("session"), "schedule": c.schedule,
        "stage_a_unit": "knee-n6a", "environment": c.task.get("environment"),
        "config": {"pooling": c.cfg.pooling, "planes": c.cfg.planes, "batch_size": bs, "E": c.E,
                   "head_lr": c.cfg.lr, "head_weight_decay": c.cfg.weight_decay, "dropout": c.cfg.dropout,
                   "encoder_weight_decay": ENC_WEIGHT_DECAY, "micro_batch_studies": c.micro,
                   "micro_batch_slices": c.micro * c.S * len(c.planes), "amp_fp16": c.amp,
                   "epochs": c.epochs, "passes": int(c.task.get("passes") or PASSES),
                   "steps_inner": c.epochs * math.ceil(len(c.itr) / bs),
                   "steps_final": c.epochs * math.ceil(len(c.tr) / bs),
                   "learning_rates": list(LRS), "rung0_twelfths": RUNG0, "warmup_twelfths": 1},
        "trainable": c.trainable, "kind": c.enc.kind, "device": c.device.type,
        "device_name": _device_name(c.device), "chain": c.chain, "upstream_n4": c.upstream_n4,
        "studies": {"training": agg.count(len(c.tr)), "test": agg.count(len(c.te)),
                    "test_scored": agg.count(len(c.te_s)), "inner_training": agg.count(len(c.itr)),
                    "inner_validation": agg.count(len(c.iva_all)), "inner_validation_scored": agg.count(len(c.iva)),
                    "gold_excluded": agg.count(len(c.gold))},
        "test_groups": c.split["test_groups"], "inner_validation_groups": c.split["val_groups"],
        "inner_validation_set": c.split["val_set"],
        "inner_validation_positives": {f: agg.count(int(v)) for f, v in zip(FINDINGS, pos)},
        "inner_validation_findings_with_both_classes": [f for f, a, b in zip(FINDINGS, pos, neg) if a > 0 and b > 0],
    }


def pick_micro(c) -> int:
    """2 studies per micro-batch; 1 if a forward and backward pass of 2 does not fit in memory."""
    import torch

    if c.micro <= 1 or c.device.type != "cuda":
        return c.micro
    r = new_run(c, c.src, c.head_state, c.tr[:max(c.micro, 24)], REGISTERED_LR, 0, epochs=12)
    try:
        batch_backward(c, r, r.rows[:c.micro], micro=c.micro)
        fits = True
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        fits = False
    r.opt.zero_grad(set_to_none=True)
    del r
    torch.cuda.empty_cache()
    return c.micro if fits else 1


def _breakdown(c, r, n: int = 20) -> dict:
    """Where the time of a b-lite micro-batch goes: n more micro-batches of random training studies after
    the timed run, with a device synchronisation after each stage (so the timed run itself is not slowed).
    Stages: reading the caches, host-to-GPU transfer, the frozen trunk, the tail forward with the head, the
    loss and the backward pass, and the optimiser step (once per effective batch)."""
    import torch
    import torch.nn.functional as F

    dev, src = c.device, c.src
    T = {"read": 0.0, "to_gpu": 0.0, "trunk": 0.0, "tail_and_backward": 0.0, "optimiser": 0.0}
    per_step = max(1, int(c.cfg.batch_size) // c.micro)
    rng = np.random.default_rng(0)
    r.head.train()
    r.opt.zero_grad(set_to_none=True)
    slices = 0
    for i in range(n):
        mb = rng.choice(c.tr, size=c.micro, replace=False)
        mt = torch.as_tensor(mb, device=dev)
        _sync(dev)
        t0 = time.time()
        arrays = src.read(mb)
        t1 = time.time()
        x, m, S = src.to_device(arrays)
        _sync(dev)
        t2 = time.time()
        h = src.trunk(x)
        _sync(dev)
        t3 = time.time()
        zs = _z(c, src.tail(h, m, S, True))
        w = c.bank.w[mt]
        ls = F.binary_cross_entropy_with_logits(r.head(zs), c.bank.y[mt], weight=w, pos_weight=r.pw, reduction="sum")
        r.scaler.scale(ls / w.sum().clamp_min(1.0)).backward()
        _sync(dev)
        t4 = time.time()
        if (i + 1) % per_step == 0:
            r.scaler.step(r.opt)
            r.scaler.update()
            r.opt.zero_grad(set_to_none=True)
            _sync(dev)
        t5 = time.time()
        for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            T[k] += v
        slices += x.shape[0]
    r.opt.zero_grad(set_to_none=True)
    total = sum(T.values())
    return {"micro_batches": agg.count(n), "seconds": {k: round(v, 3) for k, v in T.items()},
            "share": {k: round(v / max(total, 1e-9), 3) for k, v in T.items()},
            "slices_per_s": round(slices / max(total, 1e-9), 1),
            "note": "random training studies, a synchronisation after each stage; after the timed run"}


def _smoke(c, st: dict) -> None:
    """The smoke run (12.1 item 10): 200 micro-batches of the b-lite loop on fold 0's training studies:
    slices/s and peak memory. Then, on 8 training studies, the code that otherwise first runs hours into a
    session: the stored features, a head-only step, a flipped forward, a checkpoint round trip, the run.json
    lock. No score: no validation or test row is touched and nothing is compared with a label."""
    import torch

    _guard(c.tr, None, c.forbid_outer)
    r = new_run(c, c.src, c.head_state, c.tr, REGISTERED_LR, 0)
    r.clock = []
    run_to(c, r, TWELFTHS, limit=("micro", int(c.task.get("smoke_micro") or SMOKE_MICRO)))
    read_s = c.src.read_s
    last = r.clock[-101:]  # the last 100 micro-batches: without cold disk reads and cuDNN autotune
    steady = ((last[-1][1] - last[0][1]) / max(last[-1][0] - last[0][0], 1e-6)) if len(last) >= 2 else None
    st["smoke"] = {"micro_batches": agg.count(len(r.clock)), "slices_per_s": round(r.slices / max(r.train_s, 1e-6), 1),
                   "steady_slices_per_s": round(steady, 1) if steady is not None else None,
                   "steady_note": "the last 100 micro-batches; reported only, the rule uses slices_per_s",
                   "seconds": round(r.train_s, 1), "peak_memory_gb": _peak_gb(c.device),
                   "read_seconds": round(read_s, 1), "read_share": round(read_s / max(r.train_s, 1e-6), 3),
                   "micro_batch_slices": c.micro * c.S * len(c.planes), "reads_no_score": True,
                   "skipped_step_share": round(r.skipped / max(r.step, 1), 4)}
    st["smoke"]["time_breakdown"] = _breakdown(c, r)
    rows = c.tr[:8]
    feat, _ = _features(c, rows, rows)
    h = new_run(c, feat, c.head_state, c.tr[:max(24, 3 * int(c.cfg.batch_size))], 0.0, 0, with_encoder=False, epochs=12)
    h.rows = rows  # one step on the 8 studies
    h.perm, h.pos = rows, 0
    batch_backward(c, h, rows)
    h.opt.step()
    h.head.eval()
    flipped = [_logits(c, src, h.head, rows[:2], flipped=True) for src in (c.src, feat)]
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "ck.pt"
        before = (r.step, r.pos)
        save_checkpoint(c, r, ck)
        load_checkpoint(c, r, ck)
        record_chosen(tmp, "smoke", {"lr": REGISTERED_LR, "reason": "smoke: a dummy key in a temporary folder"})
        locked = chosen_recorded(tmp, "smoke", REGISTERED_LR)
    st["smoke"]["late_code_paths"] = {"feature_store": str(feat.x[c.planes[0]].dtype).replace("torch.", ""),
                                      "head_only_step": True,
                                      "flipped_forward_finite": bool(all(torch.isfinite(x).all() for x in flipped)),
                                      "checkpoint_round_trip": bool((r.step, r.pos) == before),
                                      "run_json_lock": bool(locked)}


def _frozen0(c, st: dict) -> FeatSource:
    """The frozen0 arm of a unit: features of its training and scored test studies through this module's
    forward, checks (a) and (b), and the arm's test predictions. Check (b) failing marks the run invalid."""
    import torch

    d, dev = c.d, c.device
    rows = np.unique(np.concatenate([c.tr, c.te_s]))
    frows = c.te_s[d.flip_row[c.te_s] >= 0]
    feat, secs = _features(c, rows, frows)
    # (a) features against the N5 embeddings, on 200 training studies (reported)
    k = min(CHECK_A_STUDIES, len(c.tr))
    sel = c.tr[np.unique(np.linspace(0, len(c.tr) - 1, k).round().astype(int))]
    sel_t = torch.as_tensor(sel, device=dev)
    gap, mean_gap, cos = 0.0, [], 1.0
    for p, a in zip(c.planes, feat(sel)):
        b = c.bank.x[p][sel_t].float()
        gap = max(gap, float((a - b).abs().max()))
        mean_gap.append(float((a - b).abs().mean()))
        cos = min(cos, float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min()))
    check_a = {"studies": agg.count(len(sel)), "max_abs_gap": round(gap, 5), "mean_abs_gap": round(float(np.mean(mean_gap)), 6),
               "min_cosine": round(cos, 5), "reported_only": True}
    # (b) the arm's fold macro AUC against knee-n6a's saved predictions, same rows (an iteration-1 number)
    head = heads.make_head(c.cfg, c.dim, len(c.planes))
    head.load_state_dict(c.head_state)
    head.to(dev)
    P0 = predict_test(c, feat, head, trained=False)
    ref = pd.read_csv(c.unit_dir / "predictions.csv")
    ref = ref[~ref["is_gold"].astype(bool)].astype({UID: str}).set_index(UID).reindex(d.uids[c.te_s])[FINDINGS].to_numpy(float)
    Y, zeros, find = d.eval_y[c.te_s], np.zeros(len(c.te_s), int), list(range(len(FINDINGS)))
    m0 = metrics.macro(P0, Y, zeros, find)
    mref = metrics.macro(ref, Y, zeros, find) if np.isfinite(ref).all() else float("nan")
    passed = bool(np.isfinite(m0) and np.isfinite(mref) and abs(m0 - mref) <= CHECK_B_TOL + 1e-12)
    check_b = {"macro_auc_frozen0": _r(m0, 5), "macro_auc_from_n6a_predictions": _r(mref, 5),
               "abs_difference": _r(abs(m0 - mref), 5), "tolerance": CHECK_B_TOL, "passed": passed,
               "max_abs_probability_gap": _r(float(np.nanmax(np.abs(P0 - ref))), 5) if np.isfinite(ref).any() else None,
               "rows": "the fold's scored test studies; gold rows dropped",
               "note": "the only test-fold number of this notebook: it equals an iteration-1 number"}
    tot, cnt = 0.0, 0  # the drift diagnostic's reference: mean |z| of the frozen training features
    for i in range(0, len(c.tr), 64):
        zs = _z(c, feat(c.tr[i:i + 64]))
        tot += float(sum(z.abs().mean() for z in zs) / len(zs)) * len(zs[0])
        cnt += len(zs[0])
    st["frozen0"] = {"check_a": check_a, "check_b": check_b, "valid": passed,
                     "mean_abs_z_training": _r(tot / max(cnt, 1)),
                     "forward_slices_per_s": round((len(rows) + len(frows)) * c.S * len(c.planes) / max(secs, 1e-6), 1),
                     "seconds": round(secs, 1)}
    _save_state(c.dir, st)
    if not passed:
        _invalid_flag(c.working, c.step).write_text("frozen0 check (b) failed\n")
        raise InvalidRun("frozen0 check (b) failed: the run is invalid")
    _save_pred(c, "frozen0", P0, st)
    _save_state(c.dir, st)
    return feat


def _inner(c, st: dict, feat: FeatSource) -> None:
    """All inner work of a unit (12.1 items 2 to 5, 12.2), on its inner split. Scores are inner validation
    scores only. Per-study validation predictions of the four arms decide() reads are saved privately."""
    task = c.task
    _abort_if_invalid(c)
    _guard(c.itr, c.iva, c.forbid_outer)
    head0 = inner_refit(c)
    s = {"refit": {"E": c.E, "seed": 0, "differs_from_stage_a_head": True,
                   "forbidden": "validation rows, test rows, gold rows"},
         "strategy": "successive halving, reduction factor 3; known bias: rung 0 favours fast learners",
         "curve_note": CURVE_NOTE, "rung0": {}, "to_the_end": {}, "head_only": {}}
    st["inner"] = s
    # the starting head with no further training (baseline i), on the frozen0 features
    hd = heads.make_head(c.cfg, c.dim, len(c.planes))
    hd.load_state_dict(head0)
    hd.to(c.device)
    P = predict(c, feat, hd, c.iva, use_flip=False)
    s["start"] = {"val_macro_auc": _r(metrics.macro_available(P, c.bank.eval_y[c.iva], list(range(heads.N_OUT))))}
    _save_inner(c, "start", P, st)
    # the inner head-only arm (baseline ii) at seed 0 and at the null seed: same schedule and batch order
    for seed in (0, NULL_SEED):
        r = new_run(c, feat, head0, c.itr, 0.0, seed, val_rows=c.iva, with_encoder=False)
        run_to(c, r, TWELFTHS)
        s["head_only"][f"seed_{seed}"] = {"val_macro_auc": _r(r.history[-1]["val_macro_auc"]), **_fit_summary(r)}
        _save_inner(c, f"head_only_seed_{seed}", r.val_pred, st)
    _save_state(c.dir, st)
    # rung 0: each learning rate for the first quarter of the schedule
    ck = {}
    for lr in LRS:
        _abort_if_invalid(c)
        r = new_run(c, c.src, head0, c.itr, lr, 0, val_rows=c.iva)
        run_to(c, r, RUNG0)
        ck[lr] = c.dir / f"rung0_{_lr_key(lr)}.pt"
        save_checkpoint(c, r, ck[lr])
        s["rung0"][_lr_key(lr)] = {"val_macro_auc": _r(r.history[-1]["val_macro_auc"]), **_fit_summary(r)}
        _save_state(c.dir, st)
    sel = select_rung0({lr: s["rung0"][_lr_key(lr)]["val_macro_auc"] for lr in LRS}, task.get("rung1_tie", True))
    s["selection"] = {"continue": [float(x) for x in sel["continue"]], "tied": sel["tied"], "note": sel["note"]}
    # to the end: the survivor (or the tied pair), and the registered 2e-5 whether or not it survived
    to_end = [float(x) for x in sel["continue"]]
    if not any(math.isclose(x, REGISTERED_LR) for x in to_end):
        to_end.append(REGISTERED_LR)
    final_scores = {}
    for lr in to_end:
        _abort_if_invalid(c)
        r = new_run(c, c.src, head0, c.itr, lr, 0, val_rows=c.iva)
        load_checkpoint(c, r, ck[lr])
        run_to(c, r, TWELFTHS)
        final_scores[lr] = r.history[-1]["val_macro_auc"]
        s["to_the_end"][_lr_key(lr)] = {"val_macro_auc": _r(final_scores[lr]), "resumed_from_rung0": True,
                                        "survivor": lr in sel["continue"], **_fit_summary(r)}
        if math.isclose(lr, REGISTERED_LR):  # the keep rule's b-lite arm: no choice on either side
            _save_inner(c, "b_lite", r.val_pred, st)
        _save_state(c.dir, st)
    chosen, why = select_final(sel["continue"], final_scores)
    s["chosen_lr"], s["chosen_reason"] = float(chosen), why
    s["keep_rule_arms"] = {"start": s["start"]["val_macro_auc"],
                           "head_only_seed_0": s["head_only"]["seed_0"]["val_macro_auc"],
                           f"head_only_seed_{NULL_SEED}": s["head_only"][f"seed_{NULL_SEED}"]["val_macro_auc"],
                           "b_lite_2e-05": s["to_the_end"][_lr_key(REGISTERED_LR)]["val_macro_auc"]}
    for p in ck.values():  # rung checkpoints are deleted when a unit's inner work is done (output limit)
        p.unlink(missing_ok=True)
    record_chosen(c.working, c.unit, {"lr": float(chosen), "reason": why,
                                      "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "before_any_test_pass_of_a_trained_model": True})
    s["chosen_lr_in_run_json"] = True
    s["complete"] = True
    _save_state(c.dir, st)


def _final(c, st: dict, feat: FeatSource) -> None:
    """The final fits of a unit, from Stage A's head on all its training studies: head-only (seeds 0-4) and
    b-lite at the unit's chosen rate (seed 0). They write test-fold predictions and the b-lite weights to the
    private output. No test-fold metric is computed."""
    import torch

    _abort_if_invalid(c)
    s_in = st.get("inner") or {}
    if not s_in.get("complete"):
        raise AssertionError("the unit's inner work is not complete")
    lr = float(s_in["chosen_lr"])
    _guard(c.tr, None, c.forbid_outer)
    fin = {"head_only": {}, "b_lite": {}, "curve_note": "training loss per twelfth only (dropout on)",
           "feature_note": "b-lite sees unrounded float32 features, head-only the fp16-stored frozen0 features "
                           "(a relative difference of about 5e-4)",
           "note": "predictions only: no test-fold metric of a trained model is computed in this notebook"}
    st["final"] = fin
    for k in c.task.get("head_only_seeds") or HEAD_ONLY_SEEDS:
        _abort_if_invalid(c)
        r = new_run(c, feat, c.head_state, c.tr, 0.0, k, with_encoder=False)
        run_to(c, r, TWELFTHS)
        _save_pred(c, f"head_only_seed_{k}", predict_test(c, feat, r.head, trained=True, lr=lr), st)
        fin["head_only"][f"seed_{k}"] = _fit_summary(r)
    _save_state(c.dir, st)
    _abort_if_invalid(c)
    r = new_run(c, c.src, c.head_state, c.tr, lr, 0)
    run_to(c, r, TWELFTHS)
    _save_pred(c, "b_lite", predict_test(c, c.src, r.head, trained=True, lr=lr), st)  # once, at the end
    torch.save({"tail": c.enc.tail_state(), "head": r.head.state_dict(), "encoder": c.name, "fold": c.fold,
                "tail_prefixes": list(c.enc.tail_prefixes)}, c.dir / "b_lite_seed_0.pt")  # kept for N10
    st["files_sha16"]["b_lite_seed_0.pt"] = sha16(c.dir / "b_lite_seed_0.pt")
    fin["b_lite"]["seed_0"] = {"encoder_lr": lr, **_fit_summary(r)}
    fin["drift"] = {"mean_abs_z_frozen0": st["frozen0"]["mean_abs_z_training"],
                    "mean_abs_z_b_lite_last": fin["b_lite"]["seed_0"]["mean_abs_z_last"],
                    "what": "mean absolute z of the training features under the fixed Stage A statistics"}
    fin["complete"] = True
    _save_state(c.dir, st)


def _task(task: dict) -> dict:
    """One process's part: one encoder on one device, through its folds in order. Returns only safe fields."""
    enc, working, step = task["encoder"], Path(task["working"]), task["step"]
    folds = [int(k) for k in task["folds"]]
    if task.get("spawned"):
        safe_log.worker_init()
        paths.WORKING = working
        try:
            import torch

            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # two processes share the CPU cores
        except Exception:  # noqa: BLE001
            pass
    cur = folds[0]
    states: dict = {}
    try:
        sh = _shared(task)
        if task["phase"] == "smoke":
            for k in range(sh.d.n_folds("s2")):  # every fold's unit loads (config, head, statistics, chain guard)
                cur = k
                _context(sh, k)
            cur = folds[0]
            c = _context(sh, cur)
            c.micro = sh.micro = pick_micro(c)
            st = _base_block(c)
            st["units_loaded"] = [_unit(enc, k) for k in range(sh.d.n_folds("s2"))]
            _smoke(c, st)
            return {"ok": True, "encoder": enc, "states": {cur: st}}
        feats = {}
        for cur in folds:  # 1. frozen0 for every fold: an invalid run stops before any training
            c = _context(sh, cur)
            if not task.get("micro") and cur == folds[0]:
                sh.micro = c.micro = pick_micro(c)
            st = _base_block(c)
            f = _frozen0(c, st)
            feats[cur] = SimpleNamespace(pos=f.pos, fpos=f.fpos, planes=f.planes,
                                         x={p: v.cpu() for p, v in f.x.items()}, flip={p: v.cpu() for p, v in f.flip.items()})
            states[cur] = st
            del f

        def feat_of(k):  # one fold's frozen0 features on the device at a time
            h = feats[k]
            f = FeatSource.__new__(FeatSource)
            f.planes, f.pos, f.fpos = h.planes, h.pos, h.fpos
            f.x = {p: v.to(sh.device) for p, v in h.x.items()}
            f.flip = {p: v.to(sh.device) for p, v in h.flip.items()}
            return f

        errors: dict = {}

        def unit_step(fn, k) -> None:
            """One unit's block. A unit's own failure costs only that unit (the other folds go on); an invalid
            run, the soft stop and an interrupt end the process."""
            try:
                c = _context(sh, k)
                fn(c, states[k], feat_of(k))
                if fn is _final:
                    states[k]["leakage_assertion_passed"] = True
                    _save_state(c.dir, states[k])
            except (InvalidRun, OutOfTime, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001 - only the type and the location are kept
                errors[_unit(enc, k)] = safe_log.describe(type(e), e.__traceback__)
                try:
                    (_unit_dir(working, step, enc, k) / "error.log").write_text(
                        "".join(traceback.format_exception(type(e), e, e.__traceback__)))
                except Exception:  # noqa: BLE001
                    pass

        for cur in folds:  # 2. all inner work for every fold
            unit_step(_inner, cur)
        for cur in folds:  # 3. then the final fits of every fold
            if _unit(enc, cur) not in errors:
                unit_step(_final, cur)
            feats.pop(cur, None)
        return {"ok": not errors, "encoder": enc, "states": states, "errors": errors}
    except BaseException as e:  # noqa: BLE001 - only the type and the location leave the process
        try:
            (_unit_dir(working, step, enc, cur) / "error.log").write_text(
                "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        except Exception:  # noqa: BLE001
            pass
        if isinstance(e, (KeyboardInterrupt, SystemExit)):  # the watchdog or a manual stop ends the whole step
            raise
        return {"ok": False, "encoder": enc, "invalid": isinstance(e, InvalidRun),
                "out_of_time": isinstance(e, OutOfTime),
                "errors": {_unit(enc, cur): safe_log.describe(type(e), e.__traceback__)}}
    finally:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass



# ---------------------------------------------------------------- the GPU step

def _unit_summary(b: dict) -> dict:
    def g(o, *path):
        for k in path:
            o = o.get(k) if isinstance(o, dict) else None
        return o

    return {"frozen0_valid": g(b, "frozen0", "valid"), "check_b_abs_difference": g(b, "frozen0", "check_b", "abs_difference"),
            "max_abs_probability_gap": g(b, "frozen0", "check_b", "max_abs_probability_gap"),
            "check_a_min_cosine": g(b, "frozen0", "check_a", "min_cosine"),
            "inner_validation_set": b.get("inner_validation_set"), "inner_validation_groups": b.get("inner_validation_groups"),
            "test_groups": b.get("test_groups"),
            "rung0_val_macro_auc": {k: v.get("val_macro_auc") for k, v in (g(b, "inner", "rung0") or {}).items()},
            "keep_rule_arms_inner": g(b, "inner", "keep_rule_arms"), "chosen_lr": g(b, "inner", "chosen_lr"),
            "inner_complete": g(b, "inner", "complete"), "final_complete": g(b, "final", "complete"),
            "smoke": b.get("smoke"), "trainable_parameters": g(b, "trainable", "parameters")}


def _summary(s: dict) -> dict:
    return {"status": s.get("status"), "errors": s.get("errors"), "session": s.get("session"),
            "units_complete": s.get("units_complete"), "units_missing": s.get("units_missing"),
            "out_of_time": s.get("out_of_time"),
            "schedule": s.get("schedule"), "suggested_schedule": s.get("suggested_schedule"),
            "cache_copy": s.get("cache_copy"),
            "environment": s.get("environment"), "folds": s.get("folds"),
            "validation_table": s.get("validation_table"),
            "note": "no test-fold metric of a trained model is computed here; frozen0's check equals an iteration-1 number",
            "units": {u: _unit_summary(b) for u, b in (s.get("units") or {}).items()}}


def run(session: str | None = None, folds=None, schedule: str | None = None, root=None, expect: dict | None = None,
        encoders: tuple[str, ...] = MAIN, smoke: bool = False, budget_h: float = 10.5, quiet: bool = True,
        devices=None, processes: bool | None = None, rung1_tie: bool = True, head_only_seeds=HEAD_ONLY_SEEDS,
        val_table: dict | None = None, val_range=VAL_RANGE, passes: int = PASSES, encoder_factory=None,
        eval_studies: int = 4, smoke_micro: int = SMOKE_MICRO, local_cache=None, local_margin: int = 2 << 30) -> dict:
    """One GPU session of Stage B-lite (see the module docstring). expect: the upstream hash prefixes of the card.

    smoke=True: the smoke run (step n11_smoke; fold 0; 200 micro-batches per encoder; no score). Its summary
      gives suggested_schedule: "halved" if the slower encoder is under 300 slices/s, else "full".
    session, folds: "a" (folds 0, 1, 2) or "b" (folds 3, 4); folds defaults to the session's. The step is
      n11_<session>; a call first deletes what an earlier call of the same session wrote.
    schedule: "full" or "halved", required: the value the smoke run decided, the same in both sessions. It
      is recorded in run.json's config.
    encoders: both main encoders (one process and one GPU each), or ConvNeXt-T only (12.5's trigger: the two
      GPUs then take different folds).
    The cut order's items that live here (defaults = nothing cut): head_only_seeds (the final head-only
      seeds; (0,) cuts seeds 1-4) and rung1_tie (False: a tied pair at rung 0 does not continue as a pair).
    budget_h: the watchdog stops the step at 1.1 x (Kaggle's limit is 12 h).
    local_cache: None = the three caches are first copied once to local disk (/kaggle/temp or the temp folder,
      if there is room; checked against the source) and read from there; False = read the input files; a
      path = copy there. Either way the caches are read one whole study per read call. The copy is deleted
      when the step ends. cache_copy in the summary says what happened and how fast.
    val_table, val_range, passes, encoder_factory, eval_studies, smoke_micro, devices, processes, local_margin:
      for tests."""
    from . import n13

    encs = tuple(encoders)
    if smoke:
        session, folds, schedule = "smoke", (0,), schedule or "full"
    if not session or not str(session).isalnum():
        raise ValueError('session: "a" or "b" (letters and digits)')
    if schedule not in SCHEDULES:
        raise ValueError('schedule: "full" or "halved", as the smoke run decided')
    folds = tuple(int(k) for k in (folds if folds is not None else SESSIONS.get(session, ())))
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("folds: the session's outer folds, each once")
    hos = tuple(int(k) for k in head_only_seeds)
    if 0 not in hos:
        raise ValueError("head_only_seeds must hold seed 0")
    step = _step_name(session)
    overrides = bool(encoder_factory is not None or val_table is not None or tuple(val_range or ()) != VAL_RANGE
                     or passes != PASSES or smoke_micro != SMOKE_MICRO)
    env = n13.environment()
    cfg = {"session": session, "folds": list(folds), "schedule": schedule, "encoders": list(encs), "smoke": bool(smoke),
           "budget_h": budget_h, "rung1_tie": rung1_tie, "head_only_seeds": list(hos), "passes": passes,
           "eval_studies": eval_studies, "environment": env, "test_overrides": overrides}
    s: dict = {"written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scope": SCOPE, "plan": PLAN,
               "added_after_results": True, "session": session, "folds": list(folds), "schedule": schedule,
               "environment": env, "knee_commit": agg.repo_commit(),
               "cuts": {"head_only_seeds_1_to_4": hos != HEAD_ONLY_SEEDS, "rung1_for_a_tied_pair": not rung1_tie}}
    with common.notebook(step, cfg, quiet=quiet, budget_h=budget_h):
        import shutil

        t0 = time.time()
        working = paths.WORKING
        d = stage.StageData(root, expect)
        s["upstream"] = d.upstream
        # the plan's validation table against the attached folds, before anything runs (the smoke run checks
        # all five folds): groups as names and rounded sizes; any mismatch stops the step here
        s["validation_table"] = {}
        for k in (range(d.n_folds("s2")) if smoke else folds):
            sp = inner_split(d, k, val_table, val_range)
            s["validation_table"][f"fold_{k}"] = {"test_groups": sp["test_groups"], "inner_validation_groups": sp["val_groups"],
                                                  "inner_validation_studies": agg.count(len(sp["iva_all"]))}
        devs = n5._devices(devices)
        parallel = bool(len(devs) >= 2) if processes is None else bool(processes)
        shutil.rmtree(Path(working) / "private" / step, ignore_errors=True)  # nothing of an earlier call survives
        base = {"root": str(root) if root else None, "expect": expect, "working": str(working), "step": step,
                "session": session, "schedule": schedule, "factory": encoder_factory, "val_table": val_table,
                "val_range": list(val_range) if val_range else None, "passes": int(passes),
                "eval_studies": eval_studies, "rung1_tie": rung1_tie, "head_only_seeds": list(hos),
                "smoke_micro": smoke_micro, "environment": env, "phase": "smoke" if smoke else "session",
                "soft_stop": None if smoke else t0 + budget_h * 3600}
        if len(encs) == 1 and len(devs) >= 2 and len(folds) >= 2 and not smoke:  # one encoder: the GPUs split the folds
            jobs = [(encs[0], devs[i], folds[i::2]) for i in range(2)]
        else:
            jobs = [(e, devs[i % len(devs)], folds) for i, e in enumerate(encs)]
        s["processes"] = [f"{e} on {dv.split(':')[0]}: folds {', '.join(str(k) for k in fs)}" for e, dv, fs in jobs]
        base["cache_paths"], s["cache_copy"] = localise_caches(root, local_cache, local_margin)
        local_dir = s["cache_copy"].pop("local_dir", None)
        try:
            results = _map([{**base, "encoder": e, "device": dv, "folds": list(fs)} for e, dv, fs in jobs],
                           parallel and len(jobs) >= 2)
        except BaseException:  # the watchdog or a manual stop: what finished is still written, then the stop goes on
            if not smoke:
                s["units"] = {u: b for u, b in ((_unit(e, k), _load_state(_unit_dir(working, step, e, k)))
                                                for e in encs for k in folds) if b}
                s["status"] = "interrupted"
                try:
                    agg.write(step, s)
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            if local_dir:
                shutil.rmtree(local_dir, ignore_errors=True)
        on_disk = {_unit(e, k): _load_state(_unit_dir(working, step, e, k)) for e in encs for k in folds}
        bad = {u: err for r in results if not r.get("ok") for u, err in (r.get("errors") or {}).items()}
        if any(r.get("invalid") for r in results) or _invalid_flag(working, step).exists():
            s["units"] = {u: {"frozen0": b.get("frozen0")} for u, b in on_disk.items() if b.get("frozen0")}
            s["status"] = "invalid: frozen0 check (b) failed for a unit; nothing else is read"
            s["errors"] = bad
            agg.write(step, s)
            raise InvalidRun("frozen0 check (b) failed: the run is invalid")
        if smoke:
            s["units"] = {_unit(r["encoder"], k): b for r in results if r.get("ok") for k, b in r["states"].items()}
        else:
            s["units"] = {u: b for u, b in on_disk.items() if b}  # every inner run that finished is reported (R10)
        if bad and smoke:
            s["errors"] = bad
            s["status"] = "failed"
            agg.write(step, s)
            raise N11Error("an encoder's task failed")
        if smoke:
            rates = {u: b["smoke"]["slices_per_s"] for u, b in s["units"].items()}
            s["slices_per_s"] = rates
            s["suggested_schedule"] = "halved" if min(rates.values()) < MIN_SLICES_PER_S else "full"
            s["rule"] = "the slower encoder under 300 slices/s over 200 micro-batches: every fold gets the halved schedule"
            s["status"] = "ok (smoke: no score was read)"
        else:
            # a failed or late unit must not turn the notebook FAILED (its output might then not attach):
            # the step ends OK with status "partial"; the missing units are rerun whole in session c
            done = sorted(u for u, b in s["units"].items() if (b.get("final") or {}).get("complete"))
            want_units = [_unit(e, k) for e in encs for k in folds]
            s["units_complete"] = done
            s["units_missing"] = [u for u in want_units if u not in done]
            s["out_of_time"] = any(r.get("out_of_time") for r in results)
            if bad or s["units_missing"]:
                s["errors"] = bad
                s["status"] = "partial"
            else:
                s["status"] = "ok"
        s["seconds"] = round(time.time() - t0, 1)
        agg.write(step, s)
    try:
        common.summary(_summary(s))
    except agg.UnsafeSummary:
        common.summary({"status": s.get("status"), "errors": s.get("errors"),
                        "summary_error": f"the summary block was refused by the checks; read agg_{step}.json"})
    return s


# ---------------------------------------------------------------- decide (CPU; inner validation only)

def find_units(root=None) -> dict:
    """{(encoder, fold): folder} of the attached sessions' units. The smoke run's folder is not a session.
    A unit rerun whole in a later session (knee-n11-c) replaces an unfinished one: the folder that got
    further counts (final fits complete > inner work complete > neither). Two folders that got equally far
    are refused."""
    def rank(folder: Path) -> int:
        b = _load_state(folder)
        return 2 if (b.get("final") or {}).get("complete") else 1 if (b.get("inner") or {}).get("complete") else 0

    out: dict = {}
    for p in paths.glob_outputs(f"private/{NAME}_*/*_f*/state.json", Path(root) if root else None):
        if p.parent.parent.name == _step_name("smoke"):
            continue
        enc, k = p.parent.name.rsplit("_f", 1)
        key = (enc, int(k))
        if key in out:
            a, b = rank(out[key]), rank(p.parent)
            if a == b:
                raise ValueError(f"two units for {enc} fold {k} that got equally far: attach each session once")
            if a > b:
                continue
        out[key] = p.parent
    return dict(sorted(out.items()))


def _session_run(ud: Path) -> dict:
    """run.json of the session a unit folder belongs to ({} if it is not there)."""
    p = ud.parent.parent.parent / "run.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def _read_rows(p: Path) -> dict:
    df = pd.read_csv(p).astype({UID: str})
    out = {"uid": df[UID].to_numpy(), "P": df[FINDINGS].to_numpy(float),
           "Y": df[[f"y_{f}" for f in FINDINGS]].to_numpy(float), "fold": df["s2_fold"].to_numpy(int)}
    for k in ("val_set", "resolved"):
        if k in df.columns:
            out[k] = df[k].to_numpy()
    return out


def _wmacro(P, Y, w=None) -> float:
    """The inner validation score (metrics.macro_available) with study weights: the mean AUC over the
    findings that have both classes in the (resampled) validation set."""
    vals = [metrics.auc(P[:, j], Y[:, j], w)[0] for j in range(P.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def decision(units: dict, n_boot: int = N_BOOT, seed: int = 0, min_gain: float = MIN_GAIN, encoders=None) -> dict:
    """The keep rule of 12.2, per encoder, on inner validation only. units: find_units()'s result. Only
    state.json and the inner_<arm>.csv files are opened; the test-fold files' hashes are copied from the states.

    Per encoder, over its five folds, every comparison at 4 decimals:
      b-lite    the run at the registered 2e-5 to the end of the schedule;
      baseline  the arm with the higher five-fold mean of (i) the inner starting head with no training and
                (ii) the inner head-only arm at seed 0 (a tie: head-only); both are reported;
      s         sample SD (n - 1) of the per-unit differences head-only seed 100 minus seed 0, over all units
                (10; 5 if only ConvNeXt-T ran);
      kept      (a) mean difference > max(min_gain, 2 s / sqrt(5)); (b) b-lite higher in at least 4 of 5 folds;
                (c) the study-bootstrap interval of the mean difference excludes 0 (studies resampled inside
                each distinct validation set, one draw used in every fold that has it, paired across arms).
    The sessions' environments must be equal field by field, else the status is "env_mismatch" and nothing
    is kept; the same when the sessions' knee commits differ or one is unknown (env_differences holds
    "knee_commit"). A unit of a session that ran with test overrides is refused.
    encoders: None = every encoder with a unit folder. Given (needs Ablay's OK: the plan defines the
    one-encoder case only for the 23:00 trigger): only these are read, the record names the encoder left out,
    and s uses the remaining units' differences (5 for one encoder).
    Returns the record with its time and sha256."""
    left_out = {}
    if encoders is not None:
        keep_e = set(encoders)
        left_out = {e: "not read: left out by the encoders argument of decide" for e in sorted({e for e, _ in units} - keep_e)}
        units = {key: ud for key, ud in units.items() if key[0] in keep_e}
    if not units:
        raise ValueError("decide: no unit found: attach both sessions' outputs")
    encs = sorted({e for e, _ in units}, key=lambda e: (e not in MAIN, MAIN.index(e) if e in MAIN else 0))
    for e in encs:
        if sorted(k for ee, k in units if ee == e) != list(range(N_OUTER)):
            raise ValueError(f"decide: {e} must have exactly one unit for each of the five folds (never fewer folds)")
    st = {key: _load_state(ud) for key, ud in units.items()}
    for (e, k), b in st.items():
        if not (b.get("inner") or {}).get("complete") or not (b.get("frozen0") or {}).get("valid"):
            raise ValueError(f"decide: the inner work of {e} fold {k} is not complete")
    if len({b.get("schedule") for b in st.values()}) != 1:
        raise ValueError("decide: the units ran under different schedules")
    runs = {key: _session_run(ud) for key, ud in units.items()}
    if any((r.get("config") or {}).get("test_overrides") for r in runs.values()):
        raise ValueError("decide: a unit's session ran with test overrides")
    commits = sorted({str(r.get("commit")) for r in runs.values()})
    envs = [b.get("environment") for b in st.values()]
    if not all(isinstance(x, dict) for x in envs):
        raise ValueError("decide: a unit has no environment record")
    differ = sorted(f for f in set().union(*(x.keys() for x in envs))
                    if len({json.dumps(x.get(f), sort_keys=True) for x in envs}) > 1)
    rec: dict = {"rule": "plan section 12.2, the B-lite keep rule (n11.decision)", "plan": PLAN,
                 "decided_from": "inner validation scores only (R4, R5); no test-fold file was opened",
                 "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "added_after_results": True,
                 "encoders_run": encs, "schedule": next(iter(st.values())).get("schedule"),
                 "min_gain": float(min_gain), "n_boot": int(n_boot), "limits": LIMITS,
                 "development_delta1_reported": all(e in encs for e in MAIN),
                 "sessions": sorted({str(b.get("session")) for b in st.values()}),
                 "knee_commits": commits, "encoders_left_out": left_out,
                 "units": {_unit(e, k): {"session": b.get("session"), "chosen_lr": b["inner"]["chosen_lr"],
                                         "inner_validation_set": b.get("inner_validation_set")}
                           for (e, k), b in st.items()},
                 "prediction_files_sha16": {_unit(e, k): {n: h for n, h in (b.get("files_sha16") or {}).items()
                                                          if n.startswith(("pred_", "b_lite_"))}
                                            for (e, k), b in st.items()}}
    if len(commits) != 1 or "None" in commits:  # 12.1 item 9: the same commit in every session, and a known one
        differ.append("knee_commit")
    if differ:
        rec.update({"status": "env_mismatch", "env_differences": differ,
                    "encoders": {e: {"keep": False} for e in encs}})
        return _signed(rec)
    # the inner files (hash-checked against the states) and the scores, recomputed and compared with the states
    files, score = {}, {}
    names = {"start": "start", "head_only": "head_only_seed_0", "null": f"head_only_seed_{NULL_SEED}", "b_lite": "b_lite"}
    logged = {"start": "start", "head_only": "head_only_seed_0", "null": f"head_only_seed_{NULL_SEED}", "b_lite": "b_lite_2e-05"}
    for key, ud in units.items():
        for arm, fname in names.items():
            p = ud / f"inner_{fname}.csv"
            if sha16(p) != (st[key].get("files_sha16") or {}).get(p.name):
                raise ValueError("decide: an inner file is not the one its state records")
            files[key + (arm,)] = _read_rows(p)
            v = _wmacro(files[key + (arm,)]["P"], files[key + (arm,)]["Y"])
            if st[key]["inner"]["keep_rule_arms"].get(logged[arm]) is None or not np.isfinite(v):
                raise ValueError("decide: a unit has no finite inner score")
            if abs(v - float(st[key]["inner"]["keep_rule_arms"][logged[arm]])) > 2e-4:
                raise ValueError("decide: an inner file's score differs from the score its state logged")
            score[key + (arm,)] = round(v, DECIMALS)
        sets = {tuple(files[key + (a,)]["uid"].tolist()) for a in names}
        if len(sets) != 1:
            raise ValueError("decide: a unit's inner files hold different studies")
    folds = list(range(N_OUTER))
    null = [score[(e, k, "null")] - score[(e, k, "head_only")] for e in encs for k in folds]
    s = float(np.std(null, ddof=1))
    thr_raw = max(float(min_gain), 2 * s / math.sqrt(N_OUTER))
    thr = round(thr_raw, DECIMALS)
    # one draw per distinct validation set, used in every fold that has it
    vsets: dict = {}
    for (e, k), ud in units.items():
        f = files[(e, k, "b_lite")]
        name = str(f["val_set"][0])
        if vsets.setdefault(name, f["uid"].tolist()) != f["uid"].tolist():
            raise ValueError("decide: two folds list different studies for one validation set")
    W = {}
    for i, name in enumerate(sorted(vsets)):
        rng = np.random.default_rng([int(seed), i])
        n = len(vsets[name])
        W[name] = [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(int(n_boot))]
    out_e = {}
    for e in encs:
        means = {a: float(np.mean([score[(e, k, a)] for k in folds])) for a in ("start", "head_only")}
        base = "start" if round(means["start"], DECIMALS) > round(means["head_only"], DECIMALS) else "head_only"
        diffs = [round(score[(e, k, "b_lite")] - score[(e, k, base)], DECIMALS) for k in folds]
        gain_raw = float(np.mean(diffs))
        gain = round(gain_raw, DECIMALS)
        up = sum(1 for x in diffs if x > 0)
        boot = np.empty(int(n_boot))
        for b in range(int(n_boot)):
            tot = 0.0
            for k in folds:
                fb, f0 = files[(e, k, "b_lite")], files[(e, k, base)]
                w = W[str(fb["val_set"][0])][b]
                tot += _wmacro(fb["P"], fb["Y"], w) - _wmacro(f0["P"], f0["Y"], w)
            boot[b] = tot / len(folds)
        ci = _pct(boot)
        excl = bool(ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0))
        out_e[e] = {"keep": bool(gain > thr and up >= MIN_FOLDS_UP and excl),
                    "gain": gain, "threshold": thr, "folds_up": up, "interval": ci, "interval_excludes_0": excl,
                    "conditions": {"a_gain_above_threshold": bool(gain > thr), "b_at_least_4_of_5_folds_up": bool(up >= MIN_FOLDS_UP),
                                   "c_bootstrap_interval_excludes_0": excl},
                    "baseline": base, "baseline_means": {a: _r(v) for a, v in means.items()},
                    "gain_unrounded": gain_raw,
                    "per_fold": {f"fold_{k}": {"b_lite": score[(e, k, "b_lite")], "start": score[(e, k, "start")],
                                               "head_only": score[(e, k, "head_only")],
                                               "head_only_null_seed": score[(e, k, "null")], "difference": dv}
                                 for k, dv in zip(folds, diffs)}}
    rec.update({"status": "decided", "environment": envs[0], "s": s, "s_divisor": "n - 1 (sample SD)",
                "n_null": len(null), "null_mean": float(np.mean(null)), "threshold": thr,
                "threshold_unrounded": thr_raw, "validation_sets": {n: agg.count(len(u)) for n, u in vsets.items()},
                "bootstrap": "studies resampled inside each distinct validation set; one draw per set, used in every "
                             "fold that has it; paired across arms; percentile 2.5-97.5",
                "encoders": out_e})
    return _signed(rec)


def _canon(rec: dict) -> bytes:
    body = {k: v for k, v in rec.items() if k != "sha256"}
    return json.dumps(body, sort_keys=True, default=agg._json_default).encode()


def _signed(rec: dict) -> dict:
    rec = json.loads(json.dumps(rec, default=agg._json_default))
    rec["sha256"] = hashlib.sha256(_canon(rec)).hexdigest()
    return rec


def decide(root=None, n_boot: int = N_BOOT, seed: int = 0, quiet: bool = True, budget_h: float = 1.0,
           encoders=None) -> dict:
    """The CPU decision step: the sessions' outputs attached (a, b and, after a partial session, c), nothing
    else. Writes agg_n11_decide.json (the decision record, with its time and sha256) and prints it. No
    test-fold file is opened. encoders: see decision()."""
    rec: dict = {}
    cfg = {"n_boot": int(n_boot), "seed": seed, "encoders": list(encoders) if encoders is not None else None}
    with common.notebook(f"{NAME}_decide", cfg, quiet=quiet, budget_h=budget_h):
        rec = decision(find_units(root), n_boot=n_boot, seed=seed, encoders=encoders)
        agg.write(f"{NAME}_decide", rec)
    common.summary(rec)
    return rec


# ---------------------------------------------------------------- dev (CPU; after the decision)

def _fold_draws(folds, n_boot: int, seed: int):
    """Study-bootstrap weights inside each test fold, as intervals.within_fold_bootstrap draws them."""
    f = np.asarray(folds)
    parts = [np.flatnonzero(f == k) for k in np.unique(f) if k >= 0]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        w = np.zeros(len(f))
        for rows in parts:
            w += np.bincount(rng.choice(rows, size=len(rows)), minlength=len(f))
        yield w


def _decision(decision_record, root=None) -> dict:
    """The written decision: decision()'s output as a dict, its JSON text, the path of its file, or (None)
    the attached agg_n11_decide.json. Only status "decided" with an intact sha256 is accepted (R5)."""
    d = decision_record
    if d is None:
        p = paths.find_file(f"agg_{NAME}_decide.json", Path(root) if root else None)
        if p is None:
            raise ValueError("dev needs the written decision (decide()'s output): R5")
        d = p
    if isinstance(d, (str, Path)):
        text = str(d)
        d = json.loads(text) if text.lstrip().startswith("{") else json.loads(Path(text).read_text())
    status = d.get("status") if isinstance(d, dict) else None
    if status == "env_mismatch":
        raise ValueError("dev: the record says env_mismatch, so there is no decision; no S2 number is computed (R5)")
    if status != "decided":
        raise ValueError('dev needs a decision written by n11.decide, with status "decided" (R5)')
    if d.get("sha256") != hashlib.sha256(_canon(d)).hexdigest():
        raise ValueError("dev: the decision record was changed after it was written (sha256)")
    if not all(isinstance(((d.get("encoders") or {}).get(e) or {}).get("keep"), bool) for e in d.get("encoders_run") or [None]):
        raise ValueError("dev: this is not a decision written by n11.decide")
    return d


def dev(decision_record=None, root=None, n_boot: int = N_BOOT, seed: int = 0, side_auc: bool = True,
        quiet: bool = True, budget_h: float = 1.0) -> dict:
    """The S2 development numbers of B-lite (12.3), run only after the decision is written.

    Inputs attached: both sessions' outputs and the decision (knee-n11-decide's output, or decision_record).
    Refused without a "decided" record with an intact sha256, and when a prediction file's hash differs from
    the record's. Per encoder and arm (frozen0, head-only seed 0, b-lite), on every fold's scored test
    studies: the mean of fold macros and the fold-stratified macro AUC; b-lite minus each baseline; the
    head-only seeds' SD; Delta1 (both encoders only) and N14-C2's side AUC. Every number is a development
    number (R6), whatever the decision (R10)."""
    dec = _decision(decision_record, root)
    s: dict = {"label": DEV_LABEL, "display": DISPLAY, "descriptive": True, "added_after_results": True, "plan": PLAN,
               "decision": {"status": dec["status"], "sha256": dec["sha256"], "written_at_utc": dec.get("written_at_utc"),
                            "keep": {e: dec["encoders"][e]["keep"] for e in dec["encoders_run"]}}}
    with common.notebook(f"{NAME}_dev", {"decision_sha256": dec["sha256"], "n_boot": int(n_boot), "seed": seed},
                         quiet=quiet, budget_h=budget_h):
        encs = list(dec["encoders_run"])
        units = {key: ud for key, ud in find_units(root).items() if key[0] in encs}
        want = dec["prediction_files_sha16"]
        if {_unit(e, k) for e, k in units} != set(want):
            raise ValueError("dev: the attached units are not the ones the decision was written on")
        data: dict = {}
        for (e, k), ud in units.items():
            for fname, h in want[_unit(e, k)].items():
                if sha16(ud / fname) != h:
                    raise ValueError("dev: a prediction file is not the one the decision record lists")
                if fname.startswith("pred_"):
                    data.setdefault(e, {}).setdefault(fname[len("pred_"):-len(".csv")], []).append(_read_rows(ud / fname))
        find = list(range(len(FINDINGS)))
        arms_of = {e: {a: {k: np.concatenate([x[k] for x in parts]) for k in ("uid", "P", "Y", "fold", "resolved")}
                       for a, parts in data.get(e, {}).items()} for e in encs}
        out: dict = {}
        ok_encs = []
        draws: dict = {}
        for e in encs:
            A = arms_of[e]
            need = ("frozen0", "head_only_seed_0", "b_lite")
            if not all(a in A and len(set(A[a]["fold"].tolist())) == N_OUTER for a in need):
                out[e] = {"status": "an arm or a fold is missing"}
                continue
            if any(not np.isfinite(A[a]["P"]).all() for a in A):
                out[e] = {"status": "b-lite predictions not finite"}
                continue
            if any(A[a]["uid"].tolist() != A["frozen0"]["uid"].tolist() for a in A):
                raise ValueError("dev: the arms of an encoder hold different studies")

            def m(a, A=A):
                x = A[a]
                return {"mean_of_fold_macros": metrics.per_fold_macro_mean(x["P"], x["Y"], x["fold"], find),
                        "fold_stratified_macro_auc": metrics.macro(x["P"], x["Y"], x["fold"], find)}

            vals = {a: m(a) for a in need}
            # study bootstrap inside each test fold (n9's secondary interval), one set of draws for every arm
            draws[e] = np.array([[metrics.macro(A[a]["P"], A[a]["Y"], A[a]["fold"], find, w) for a in need]
                                 for w in _fold_draws(A["frozen0"]["fold"], int(n_boot), seed)])
            col = {a: draws[e][:, i] for i, a in enumerate(need)}
            fs = "fold_stratified_macro_auc"
            seeds = sorted(a for a in A if a.startswith("head_only_seed_"))
            ho = [m(a)["mean_of_fold_macros"] for a in seeds]
            out[e] = {"label": DEV_LABEL, "studies": agg.count(len(A["frozen0"]["uid"])),
                      "decision_keep": dec["encoders"][e]["keep"],
                      **{a: {**{k: _r(v) for k, v in vals[a].items()}, "fold_stratified_interval": _pct(col[a])} for a in need},
                      "b_lite_minus_frozen0": {**{k: _r(vals["b_lite"][k] - vals["frozen0"][k]) for k in vals["b_lite"]},
                                               "fold_stratified_interval": _pct(col["b_lite"] - col["frozen0"])},
                      "b_lite_minus_head_only": {**{k: _r(vals["b_lite"][k] - vals["head_only_seed_0"][k]) for k in vals["b_lite"]},
                                                 "fold_stratified_interval": _pct(col["b_lite"] - col["head_only_seed_0"])},
                      "head_only_seed_values": [_r(v) for v in ho],
                      "head_only_seed_sd": _r(float(np.std(ho, ddof=1)), 5) if len(ho) >= 2 else None,
                      "head_only_seed_note": "mean of fold macros per final head-only seed: a noise description only"}
            ok_encs.append(e)
        s["encoders"] = out
        s["interval"] = ("fold_stratified_interval: study bootstrap inside each test fold, paired across arms and "
                         "encoders, percentile 2.5-97.5; descriptive (sites and fold models held fixed)")
        s["n_boot"] = int(n_boot)
        s["metrics"] = {"mean_of_fold_macros": "mean over the S2 folds of each fold's macro AUC",
                        "fold_stratified_macro_auc": "pairs inside one S2 test fold, mean over the 12 findings"}
        if all(e in ok_encs for e in MAIN):
            if arms_of[DINO]["frozen0"]["uid"].tolist() != arms_of[CONV]["frozen0"]["uid"].tolist():
                raise ValueError("dev: the two encoders hold different studies")
            d1 = {a: {k: out[DINO][a][k] - out[CONV][a][k] for k in s["metrics"]} for a in ("frozen0", "b_lite")}
            dd = {"frozen0": draws[DINO][:, 0] - draws[CONV][:, 0], "b_lite": draws[DINO][:, 2] - draws[CONV][:, 2]}
            ci = _pct(dd["b_lite"] - dd["frozen0"])
            s["delta1"] = {"what": "DINOv2 minus ConvNeXt-T", "label": DEV_LABEL,
                           **{a: {**{k: _r(v) for k, v in d1[a].items()}, "fold_stratified_interval": _pct(dd[a])} for a in d1},
                           "change": {**{k: _r(d1["b_lite"][k] - d1["frozen0"][k]) for k in s["metrics"]},
                                      "fold_stratified_interval": ci},
                           "change_interval_excludes_0": bool(ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0)),
                           "note": "a development Delta1 is not a new H1 result"}
        else:
            s["delta1"] = {"status": "not reported (needs both main encoders)"}
        if side_auc and ok_encs:
            from . import n14

            ref = arms_of[ok_encs[0]]["frozen0"]
            fake = SimpleNamespace(folds={"s2_fold": ref["fold"]}, flip_row=np.where(ref["resolved"].astype(bool), -1, 0))
            s["side_discrimination"] = {"what": "N14-C2 on every fold's scored test studies: a description of FM3"}
            same = all(arms_of[e]["frozen0"]["uid"].tolist() == ref["uid"].tolist() for e in ok_encs)
            for arm in ("frozen0", "b_lite"):
                r = n14.c2_side_discrimination(SimpleNamespace(d=fake, ev=np.arange(len(ref["uid"])), Y=ref["Y"],
                                                               units={e: arms_of[e][arm]["P"] for e in ok_encs if same}),
                                               n_boot, seed)
                r["scope"] = f"side information in the {arm} outputs, all folds; report labels"
                r["label"] = DEV_LABEL
                s["side_discrimination"][arm] = r
        else:
            s["side_discrimination"] = {"status": "cut" if not side_auc else "not available"}
        s["status"] = "ok" if len(ok_encs) == len(encs) else "ok (an encoder has no result: see its status)"
        agg.write(f"{NAME}_dev", s)
    common.summary(s)
    return s
