"""N11: Stage B-lite, iteration 3 (GPU T4 x2; the declaration it3_declaration_r3.md, plan section 11).

One question: does training the last encoder blocks raise the fold-0 score? Outer fold 0 only, both
main encoders, the N4 uint8 caches, the fold-0 Stage A head and its fixed z-scoring statistics.
Every number is "added after the results" (R10) and a development number (R6). Nothing is kept, no
submission comes from it, and its output is attached to no other notebook (R9). Expert labels are
not read: gold rows are only excluded.

Arms per encoder (section 3):
  frozen0    no training; this module's own forward (frozen trunk + trainable tail, eval mode, fp16).
             Check (a): features of 200 training studies against the N5 embeddings (reported).
             Check (b): fold-0 macro AUC within 0.0005 of the value recomputed here from knee-n6a's
             saved predictions; if it fails the run is invalid and nothing else is read.
  head-only  the encoder stays frozen; the head trains on in this module's loop on the frozen0
             features (seeds 0-4; their SD is the noise floor).
  b-lite     the last blocks train with the head (seed 0; seed 1 if the run is ahead of the budget).
Search (section 4): successive halving over the encoder learning rate {5e-6, 2e-5, 8e-5} on one inner
split (the largest g_lang group of the training set is the inner validation group); inner runs start
from a head refit on the inner training rows only. The test fold never chooses anything; the chosen
rate is written to run.json before any test-fold forward pass of a trained model.

Phases of run() (one process per encoder, one GPU each, when two GPUs exist):
  probe    the first 200 micro-batches of the first rung-0 run: slices/s and peak memory (no score).
           Below 300 slices/s for the slower encoder: both run the short schedule (6 epochs; rungs 2, 6).
  main     frozen0 (checks), search (refit, rung 0, rung 1, the chosen rate), final (head-only seeds
           0-4 and b-lite seed 0; the test fold scored once, at the last epoch).
  seed1    b-lite seed 1 for both encoders, only if the run is ahead of the budget (cut first).
  combine  the parent, CPU: macro AUCs with paired study bootstraps, the reading per encoder and the
           joint reading (section 7, in its order), fold-0 Delta1 and its change, N14-C2's side AUC.

Outputs: agg_n11.json and run.json (aggregates only; curves as per-epoch aggregates). Private:
private/n11/<encoder>/ holds state.json, pred_<arm>.csv (per-study, for the bootstrap), the b-lite
weights and, during the search, the rung checkpoints.
"""
from __future__ import annotations

import fcntl
import json
import math
import multiprocessing as mp
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
LONG = {"name": "long", "epochs": 12, "rung0": 3}
SHORT = {"name": "short", "epochs": 6, "rung0": 2}
SCHEDULES = {"long": LONG, "short": SHORT}
TIE = 0.005  # rung scores within this are a tie
MARGIN = 0.02  # the plan's margin (reading rule)
DECIMALS = 4  # every comparison of a rule is made at 4 decimals, the precision the scores are stored at
ENC_WEIGHT_DECAY = 0.05
MICRO = 2  # studies per micro-batch (1 if 2 do not fit in memory)
PROBE_MICRO = 200  # micro-batches of the throughput rule
MIN_SLICES_PER_S = 300.0
SMOKE_STEPS = 20
HEAD_ONLY_SEEDS = (0, 1, 2, 3, 4)
CHECK_A_STUDIES = 200
CHECK_B_TOL = 0.0005
N_BOOT = 2000
INNER_VAL_RANGE = (300, 500)  # "about 400 studies" (declaration, section 4 step 1)
SEED1_SHARE = 0.8  # seed 1 starts only if it is projected to end before this share of the budget
FLIP_PLANES = heads.FLIP_PLANES
BLOCKS = ("probe", "frozen0", "search", "final", "seed1", "combine")
DEV_LABEL = "development: these studies were test studies in iteration 1; likely optimistic"
DISPLAY = "grey; never in a title, never as a gain over iteration 1, never on slide 12 (R6)"
NOISE = "not distinguishable from seed noise"
HELPS = "training the last blocks helps on fold 0"
LOWERED = "training the last blocks lowered the fold-0 score"
SMALL = "small or mixed"
LIMITS = ["a positive result does not single out the ceiling (training can also undo dilution)",
          "a null result rules out neither (one fold, 1,350 training studies, 12 epochs, one depth)",
          "nothing is kept; no S4 submission comes from B-lite; D11 stands"]
SCOPE = ("Stage B-lite, iteration 3 (plan section 11): a diagnostic on outer fold 0; no number here keeps or "
         "switches on a change (R8)")


class InvalidRun(RuntimeError):
    """frozen0 check (b) failed: the run is invalid and nothing else is read."""


class N11Error(RuntimeError):
    """An encoder's task failed (the safe description is in agg_n11.json's errors)."""


def _r(x, nd: int = DECIMALS):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def _lr_key(lr: float) -> str:
    return f"{float(lr):g}"


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

class EncSource:
    """Slice features from the caches through the split encoder: the B-lite forward code.

    rows -> one float32 tensor [m, S, D] per plane. The trunk always runs under no_grad; the tail runs
    with a graph only when grad is true. flipped: coronal and axial slices mirrored left-right (as N5)."""

    def __init__(self, enc: Split, caches: dict, planes, amp: bool):
        self.enc, self.caches, self.planes, self.amp = enc, caches, tuple(planes), bool(amp)

    def __call__(self, rows, grad: bool = False, flipped: bool = False):
        import torch

        rows = np.asarray(rows)
        dev = self.enc.device
        parts = []
        for p in self.planes:
            a = torch.from_numpy(np.ascontiguousarray(self.caches[p][rows])).to(dev)
            parts.append(a.flip(-1) if flipped and p in FLIP_PLANES else a)
        m, S = parts[0].shape[:2]
        x = torch.cat(parts).reshape(len(self.planes) * m * S, *parts[0].shape[2:])
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.amp):
            with torch.no_grad():
                h = self.enc.trunk(self.enc.prep(x))
            if grad:
                f = self.enc.tail(h)
            else:
                with torch.no_grad():
                    f = self.enc.tail(h)
        f = f.float().reshape(len(self.planes), m, S, -1)
        return [f[i] for i in range(len(self.planes))]


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


# ---------------------------------------------------------------- one encoder's context

def _enc_dir(working, enc: str) -> Path:
    p = Path(working) / "private" / NAME / enc
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_state(working, enc: str) -> dict:
    p = _enc_dir(working, enc) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(working, enc: str, st: dict) -> None:
    text = json.dumps(st, indent=1, default=agg._json_default)
    agg._check(json.loads(text))  # the state becomes agg_n11.json's encoder block: aggregates only
    (_enc_dir(working, enc) / "state.json").write_text(text)


def inner_split(d, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inner training rows, the inner validation group's rows, its scored rows): the validation group is
    the largest g_lang group of the training set (ties: the first name in sorted order)."""
    g = d.g_lang[tr].astype(str)
    names, cnt = np.unique(g, return_counts=True)
    if len(names) < 2:
        raise ValueError("the training set needs at least two site groups for an inner split")
    G = names[int(np.argmax(cnt))]
    iva_all = tr[g == G]
    return tr[g != G], iva_all, iva_all[d.scored[iva_all]]


def _context(task: dict):
    """Everything one encoder's task needs: data, index sets, caches, the Stage A head and the encoder."""
    import torch

    root = Path(task["root"]) if task.get("root") else None
    base = root or paths.INPUT
    enc_name, dev = task["encoder"], torch.device(task["device"])
    d = stage.StageData(root, task.get("expect"))
    c = SimpleNamespace(d=d, name=enc_name, device=dev, amp=dev.type == "cuda", working=Path(task["working"]),
                        micro=int(task.get("micro") or MICRO), eval_studies=int(task.get("eval_studies") or 4))
    c.tr, c.te, c.te_s = d.split(0)
    c.gold = d.gold_rows()
    if d.is_gold[c.tr].any() or d.is_gold[c.te].any():
        raise AssertionError("a gold study is in the fold-0 training or test set")
    c.itr, c.iva_all, c.iva = inner_split(d, c.tr)
    rng = task.get("inner_val_range")
    if rng and not rng[0] <= len(c.iva_all) <= rng[1]:
        raise AssertionError("the inner validation group is not about 400 studies")
    c.forbid_outer = np.concatenate([c.te, c.gold])  # every fit: test rows and gold rows
    c.forbid_inner = np.concatenate([c.iva_all, c.te, c.gold])  # the inner refit: validation rows too
    ud = unit_files.unit_dirs(base).get((enc_name, 0))
    if ud is None:
        raise FileNotFoundError("the fold-0 Stage A unit is not attached")
    c.unit_dir = ud
    c.chain = unit_files.check_units(d, {(enc_name, 0): ud})
    if not c.chain["s2_folds_match_units"]:
        raise unit_files.ChainError("the Stage A unit does not match the attached S2 folds")
    cj = json.loads((ud / "config.json").read_text())
    c.cfg, c.E = heads.config_from(cj["config"]), int(cj["E"])
    c.planes = heads.PLANE_SETS[c.cfg.planes]
    c.caches, c.upstream_n4 = {}, {}
    for p in c.planes:
        mm, _, st, up, _ = n5._load_plane(p, root)
        if st[UID].astype(str).tolist() != d.uids.tolist():
            raise AssertionError("cache rows differ from N5's index")
        c.caches[p], c.upstream_n4[p] = mm, up
    c.S = int(c.caches[c.planes[0]].shape[1])
    c.bank = d.bank(enc_name, str(dev))
    c.dim = int(c.bank.x[c.planes[0]].shape[-1])
    raw = torch.load(ud / "stats.pt", map_location="cpu")
    c.stats = {p: (mu.float().to(dev), sd.float().to(dev)) for p, (mu, sd) in raw.items()}
    c.stats_l = [c.stats[p] for p in c.planes]
    c.head_state = torch.load(ud / "head.pt", map_location="cpu")
    c.enc = _factory(task)
    c.trainable = c.enc.check()
    c.src = EncSource(c.enc, c.caches, c.planes, c.amp)
    c.fdtype = torch.float16 if c.amp else torch.float32
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return c


def _guard(train_idx, val_idx, forbidden) -> None:
    """The plan's run-time leakage assertion, for every fit of this module (heads.fit calls it itself)."""
    heads._check_guard(train_idx, val_idx, forbidden)


def inner_refit(c):
    """The head the inner runs start from: fold 0's Stage A config and E, seed 0, the cached N5 embeddings,
    the fixed Stage A statistics, on the inner training rows only. Forbidden: the validation group's rows,
    the test rows and the gold rows (the leakage guard raises if one reaches the fit)."""
    r = heads.fit(c.bank, c.cfg, c.itr, epochs=c.E, stats=c.stats, forbidden=c.forbid_inner, seed=0)
    state = {k: v.detach().cpu().clone() for k, v in r.model.state_dict().items()}
    if all(bool((state[k] == c.head_state[k]).all()) for k in state):
        raise AssertionError("the inner refit equals Stage A's head")
    return state


# ---------------------------------------------------------------- the training loop

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


def new_run(c, src, head_state: dict, rows, lr_enc: float, schedule_epochs: int, seed: int, val_rows=None,
            with_encoder: bool = True):
    """A fit of this module's loop, at epoch 0.

    AdamW (the heads' betas and eps). Groups: the trainable encoder weights (lr_enc, weight decay 0.05),
    the encoder's biases, norm weights and layer-scale (lr_enc, no decay), the head (its Stage A peak rate
    and weight decay). All share heads.schedule_factor over schedule_epochs. Effective batch = the head's
    Stage A batch size; pos_weight from the fit's own training rows, as heads.fit. with_encoder false
    (the head-only arm on the frozen0 features): the head group alone."""
    import torch

    r = SimpleNamespace(src=src, rows=np.asarray(rows), lr_enc=float(lr_enc), sched_epochs=int(schedule_epochs),
                        seed=int(seed), val_rows=None if val_rows is None else np.asarray(val_rows),
                        with_encoder=bool(with_encoder), step=0, epoch=0, history=[], slices=0, train_s=0.0,
                        skipped=0)
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
    yb = c.bank.binary[r.rows]
    pos, neg = (yb == 1).sum(0), (yb == 0).sum(0)
    r.pw = torch.tensor(np.minimum(neg / np.maximum(pos, 1), c.cfg.pos_cap), dtype=torch.float32, device=dev)
    return r


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
        if limit is not None and done >= limit:
            break
    return loss_sum, w_raw, z_sum, done


def run_epochs(c, r, until: int, limit: tuple | None = None) -> None:
    """Train r up to epoch `until` of its schedule. Head in train mode (dropout), encoder in eval mode.
    One optimiser step, one schedule step and one scaler update per effective batch. Per epoch: the
    training loss (weighted mean over the epoch, dropout on), mean |z| of the training features and, with
    validation rows, the validation loss and macro AUC (no flips, as Stage A's inner fits).
    limit: ("micro", n) stops after n micro-batches (the throughput probe); ("steps", n) after n optimiser
    steps (the smoke run). A limited run records no epoch and reads no score."""
    import torch

    dev = c.device
    n = len(r.rows)
    micro_done, steps0, stop = 0, r.step, False
    while r.epoch < until and not stop:
        if r.with_encoder and c.enc.model.training:
            raise AssertionError("the encoder must stay in eval mode")
        r.head.train()
        perm = r.rows[torch.randperm(n, generator=r.gen).numpy()]
        loss_sum = torch.zeros((), device=dev)
        w_sum = torch.zeros((), device=dev)
        z_sum = torch.zeros((), device=dev)
        z_n = 0
        _sync(dev)
        t0 = time.time()
        for i in range(0, n, r.bs):
            f = heads.schedule_factor(r.step, r.spe, r.sched_epochs)
            for g, peak in zip(r.opt.param_groups, r.peaks):
                g["lr"] = peak * f
            left = None if not (limit and limit[0] == "micro") else limit[1] - micro_done
            ls, w, z, done = batch_backward(c, r, perm[i:i + r.bs], limit=left)
            micro_done += done
            if left is not None and micro_done >= limit[1]:
                stop = True  # the probe ends inside a batch: no optimiser step for a partial batch
                break
            scale0 = r.scaler.get_scale()
            r.scaler.step(r.opt)
            r.scaler.update()
            if r.scaler.get_scale() < scale0:
                r.skipped += 1  # a non-finite gradient: the scaler skipped this step
            r.step += 1
            loss_sum += ls
            w_sum += w
            z_sum += z
            z_n += done
            if limit and limit[0] == "steps" and r.step - steps0 >= limit[1]:
                stop = True
                break
        _sync(dev)
        r.train_s += time.time() - t0
        if stop:
            break
        r.epoch += 1
        rec = {"train_loss": float(loss_sum / w_sum.clamp_min(1e-6)), "mean_abs_z": float(z_sum / max(z_n, 1))}
        if r.val_rows is not None:
            rec.update(_validate(c, r))
        r.history.append(rec)


def _logits(c, src, head, rows, flipped: bool = False):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(rows), c.eval_studies):
            out.append(head(_z(c, src(rows[i:i + c.eval_studies], grad=False, flipped=flipped))).float())
    return torch.cat(out) if out else torch.zeros((0, heads.N_OUT), device=c.device)


def _validate(c, r) -> dict:
    """Inner validation: the same loss (the fit's pos_weight) and the macro AUC over the findings with both
    classes in the validation group's scored studies (report labels), no flip averaging."""
    import torch
    import torch.nn.functional as F

    r.head.eval()
    lg = _logits(c, r.src, r.head, r.val_rows)
    vt = torch.as_tensor(r.val_rows, device=c.device)
    w = c.bank.w[vt]
    loss = F.binary_cross_entropy_with_logits(lg, c.bank.y[vt], weight=w, pos_weight=r.pw, reduction="sum")
    pv = torch.sigmoid(lg).cpu().numpy()
    score = metrics.macro_available(pv, c.bank.eval_y[r.val_rows], list(range(heads.N_OUT)))
    return {"val_loss": float(loss / w.sum().clamp_min(1e-6)),
            "val_macro_auc": float(score) if np.isfinite(score) else None}


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


def save_checkpoint(c, r, path: Path) -> None:
    """Everything a rung-1 run needs to continue exactly: the trainable weights, the optimiser, the
    scaler, the schedule step, the batch-order generator and the dropout stream."""
    import torch

    torch.save({"tail": c.enc.tail_state() if r.with_encoder else {}, "head": r.head.state_dict(),
                "opt": r.opt.state_dict(), "scaler": r.scaler.state_dict(), "step": r.step, "epoch": r.epoch,
                "gen": r.gen.get_state(), "rng": _rng_state(c.device), "history": r.history,
                "skipped": r.skipped, "slices": r.slices, "train_s": r.train_s,
                "lr_enc": r.lr_enc, "sched_epochs": r.sched_epochs, "seed": r.seed}, path)


def load_checkpoint(c, r, path: Path) -> None:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck["lr_enc"], ck["sched_epochs"], ck["seed"]) != (r.lr_enc, r.sched_epochs, r.seed):
        raise AssertionError("the checkpoint belongs to another run")
    if r.with_encoder:
        c.enc.load_tail_state(ck["tail"])
    r.head.load_state_dict(ck["head"])
    r.opt.load_state_dict(ck["opt"])
    r.scaler.load_state_dict(ck["scaler"])
    r.step, r.epoch, r.history = int(ck["step"]), int(ck["epoch"]), list(ck["history"])
    r.skipped, r.slices, r.train_s = int(ck["skipped"]), int(ck["slices"]), float(ck["train_s"])
    r.gen.set_state(ck["gen"])
    _set_rng_state(c.device, ck["rng"])


def _curve(r) -> dict:
    keys = [k for k in ("train_loss", "val_loss", "val_macro_auc", "mean_abs_z") if any(k in h for h in r.history)]
    return {k: [_r(h.get(k)) for h in r.history] for k in keys}


def _fit_summary(r) -> dict:
    out = {"epochs": int(r.epoch), "curve": _curve(r), "train_seconds": round(r.train_s, 1),
           "slices_per_s": round(r.slices / max(r.train_s, 1e-6), 1),
           "skipped_step_share": round(r.skipped / max(r.step, 1), 4),
           "mean_abs_z_last_epoch": _r(r.history[-1]["mean_abs_z"]) if r.history else None}
    vl = [h.get("val_loss") for h in r.history]
    if vl and all(v is not None for v in vl):
        out["val_loss_min"] = {"epoch": int(np.argmin(vl)) + 1, "value": _r(min(vl))}  # reported, not used
    return out


# ---------------------------------------------------------------- selection rules (pure)

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
        if rung1_tie:
            return {"continue": pair, "tied": True, "note": "top two within 0.005: both continue"}
        return {"continue": [tie_fallback(pair)], "tied": True, "pair": pair,
                "note": "top two within 0.005; rung 1 for the pair was cut: the tie fallback continues"}
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


def schedule_for(slices_per_s, minimum: float = MIN_SLICES_PER_S) -> dict:
    """The throughput rule: the slower encoder below `minimum` slices/s puts both on the short schedule."""
    vals = [float(v) for v in slices_per_s if v is not None]
    if not vals:
        raise ValueError("no throughput was measured")
    return dict(SHORT if min(vals) < minimum else LONG)


def reading(b_lite: float, frozen0: float, head_only: float, seed_sd, ci_frozen0: dict, ci_head_only: dict) -> str:
    """Section 7, in its order; the first line that fits is the reading.
    1. |b-lite minus the stricter baseline| smaller than twice the head-only seed SD: seed noise.
    2. b-lite beats both by at least 0.02 and both paired intervals exclude 0: helps.
    3. b-lite below both by 0.02 or more: lowered.
    4. otherwise: small or mixed."""
    def r4(x):
        return round(float(x), DECIMALS)

    def excludes0(ci):
        lo, hi = (ci or {}).get("lo"), (ci or {}).get("hi")
        return lo is not None and hi is not None and (lo > 0 or hi < 0)

    df, dh = r4(b_lite - frozen0), r4(b_lite - head_only)
    diff = r4(b_lite - max(frozen0, head_only))
    if seed_sd is not None and np.isfinite(seed_sd) and abs(diff) < r4(2 * seed_sd):
        return NOISE
    if df >= MARGIN and dh >= MARGIN and excludes0(ci_frozen0) and excludes0(ci_head_only):
        return HELPS
    if df <= -MARGIN and dh <= -MARGIN:
        return LOWERED
    return SMALL


def joint_reading(readings: dict) -> str:
    """ "helps" only if both encoders read "helps"; if the two disagree, "mixed"."""
    vals = [readings.get(e) for e in MAIN]
    if any(v is None for v in vals):
        return "not available"
    return vals[0] if len(set(vals)) == 1 else "mixed"


# ---------------------------------------------------------------- run.json: the chosen rate

def record_chosen(working, enc: str, rec: dict) -> None:
    """Write one encoder's chosen learning rate into run.json (both encoders' processes write it: locked)."""
    working = Path(working)
    lock = working / "private" / NAME / "run.lock"
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


def predict_test(c, src, head, trained: bool, lr: float | None = None) -> np.ndarray:
    """The fold-0 scored test studies, with flip averaging. A trained model is refused until the chosen
    learning rate stands in run.json."""
    if trained and (lr is None or not chosen_recorded(c.working, c.name, lr)):
        raise AssertionError("the chosen learning rate must be in run.json before a trained model sees the test fold")
    return predict(c, src, head, c.te_s, use_flip=True)


def _save_pred(c, arm: str, P: np.ndarray) -> None:
    df = pd.DataFrame(P, columns=FINDINGS)
    df.insert(0, "s2_fold", 0)
    df.insert(0, UID, c.d.uids[c.te_s])
    df.to_csv(_enc_dir(c.working, c.name) / f"pred_{arm}.csv", index=False)


# ---------------------------------------------------------------- an encoder's blocks

def _invalid_flag(working) -> Path:
    return Path(working) / "private" / NAME / "INVALID"


def _abort_if_invalid(working) -> None:
    if _invalid_flag(working).exists():
        raise InvalidRun("frozen0 check (b) failed for an encoder: nothing else is read")


def _base_block(c, task: dict) -> dict:
    d, sch = c.d, task.get("schedule") or {}
    pos = (d.eval_y[c.iva] == 1).sum(0)
    return {
        "config": {"pooling": c.cfg.pooling, "planes": c.cfg.planes, "batch_size": int(c.cfg.batch_size), "E": c.E,
                   "head_lr": c.cfg.lr, "head_weight_decay": c.cfg.weight_decay, "dropout": c.cfg.dropout,
                   "encoder_weight_decay": ENC_WEIGHT_DECAY, "micro_batch_studies": c.micro,
                   "micro_batch_slices": c.micro * c.S * len(c.planes), "amp_fp16": c.amp,
                   "schedule_epochs": sch.get("epochs"), "rung0_epochs": sch.get("rung0"),
                   "steps_per_epoch_inner": math.ceil(len(c.itr) / max(1, int(c.cfg.batch_size))),
                   "steps_per_epoch_final": math.ceil(len(c.tr) / max(1, int(c.cfg.batch_size))),
                   "learning_rates": list(LRS), "head_only_seeds": list(HEAD_ONLY_SEEDS)},
        "trainable": c.trainable, "kind": c.enc.kind, "device": c.device.type,
        "device_name": _device_name(c.device), "chain": c.chain, "upstream_n4": c.upstream_n4,
        "studies": {"training": agg.count(len(c.tr)), "test": agg.count(len(c.te)),
                    "test_scored": agg.count(len(c.te_s)), "inner_training": agg.count(len(c.itr)),
                    "inner_validation": agg.count(len(c.iva_all)), "inner_validation_scored": agg.count(len(c.iva)),
                    "gold_excluded": agg.count(len(c.gold))},
        "inner_validation_positives": {f: agg.count(int(v)) for f, v in zip(FINDINGS, pos)},
        "inner_validation_findings_with_both_classes": agg.count(
            int(((pos > 0) & ((d.eval_y[c.iva] == 0).sum(0) > 0)).sum())),
    }


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


def pick_micro(c, head_state) -> int:
    """2 studies per micro-batch; 1 if a forward and backward pass of 2 does not fit in memory."""
    import torch

    if c.micro <= 1 or c.device.type != "cuda":
        return c.micro
    r = new_run(c, c.src, head_state, c.tr[:c.micro], REGISTERED_LR, LONG["epochs"], 0)
    try:
        batch_backward(c, r, r.rows, micro=c.micro)
        fits = True
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        fits = False
    r.opt.zero_grad(set_to_none=True)
    del r
    torch.cuda.empty_cache()
    return c.micro if fits else 1


def _probe(c, task: dict, st: dict) -> None:
    """The throughput rule's measurement: the first 200 micro-batches of the first rung-0 run (the refit
    head, the first learning rate, the 12-epoch schedule). No validation pass, no score."""
    _guard(c.itr, c.iva, c.forbid_outer)
    head0 = inner_refit(c)
    lr = LRS[0] if task.get("search", True) else REGISTERED_LR
    r = new_run(c, c.src, head0, c.itr, lr, LONG["epochs"], 0)
    run_epochs(c, r, LONG["rung0"], limit=("micro", int(task.get("probe_micro") or PROBE_MICRO)))
    st["probe"] = {"slices_per_s": round(r.slices / max(r.train_s, 1e-6), 1), "slices": agg.count(r.slices),
                   "seconds": round(r.train_s, 1), "peak_memory_gb": _peak_gb(c.device),
                   "micro_batch_slices": c.micro * c.S * len(c.planes), "learning_rate": lr}
    st["micro"] = {"config": {"micro_batch_studies": c.micro}}


def _smoke(c, task: dict, st: dict) -> None:
    """20 optimiser steps of the b-lite loop on training studies: memory and slices/s only, no score."""
    _guard(c.tr, None, c.forbid_outer)
    r = new_run(c, c.src, c.head_state, c.tr, REGISTERED_LR, LONG["epochs"], 0)
    run_epochs(c, r, LONG["epochs"], limit=("steps", int(task.get("smoke_steps") or SMOKE_STEPS)))
    st["smoke"] = {"optimiser_steps": agg.count(r.step), "slices_per_s": round(r.slices / max(r.train_s, 1e-6), 1),
                   "seconds": round(r.train_s, 1), "peak_memory_gb": _peak_gb(c.device),
                   "micro_batch_slices": c.micro * c.S * len(c.planes), "reads_no_score": True,
                   "skipped_step_share": round(r.skipped / max(r.step, 1), 4)}


def _frozen0(c, st: dict) -> FeatSource:
    """The frozen0 arm: features of the training and scored test studies through this module's forward,
    checks (a) and (b), and the arm's test predictions. Check (b) failing marks the run invalid."""
    import torch

    d, dev = c.d, c.device
    c.enc.reset()
    rows = np.unique(np.concatenate([c.tr, c.te_s]))
    x = {p: torch.empty((len(rows), c.S, c.dim), dtype=c.fdtype, device=dev) for p in c.planes}
    _sync(dev)
    t0 = time.time()
    for i in range(0, len(rows), c.eval_studies):
        for p, f in zip(c.planes, c.src(rows[i:i + c.eval_studies])):
            x[p][i:i + f.shape[0]] = f.to(c.fdtype)
    frows = c.te_s[d.flip_row[c.te_s] >= 0]
    flip = {p: torch.empty((len(frows), c.S, c.dim), dtype=c.fdtype, device=dev) for p in c.planes if p in FLIP_PLANES}
    for i in range(0, len(frows), c.eval_studies):
        for p, f in zip(c.planes, c.src(frows[i:i + c.eval_studies], flipped=True)):
            if p in flip:
                flip[p][i:i + f.shape[0]] = f.to(c.fdtype)
    _sync(dev)
    secs = time.time() - t0
    feat = FeatSource(d.n, rows, x, frows, flip, c.planes)
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
    # (b) the arm's fold-0 macro AUC against knee-n6a's saved predictions, same rows
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
               "rows": "fold 0's scored test studies; gold rows dropped"}
    # the drift diagnostic's reference: mean |z| of the frozen training features
    tot, cnt = 0.0, 0
    for i in range(0, len(c.tr), 64):
        zs = _z(c, feat(c.tr[i:i + 64]))
        tot += float(sum(z.abs().mean() for z in zs) / len(zs)) * len(zs[0])
        cnt += len(zs[0])
    st["frozen0"] = {"check_a": check_a, "check_b": check_b, "valid": passed,
                     "mean_abs_z_training": _r(tot / max(cnt, 1)),
                     "forward_slices_per_s": round((len(rows) + len(frows)) * c.S * len(c.planes) / max(secs, 1e-6), 1),
                     "seconds": round(secs, 1)}
    _save_state(c.working, c.name, st)
    if not passed:
        _invalid_flag(c.working).write_text("frozen0 check (b) failed\n")
        raise InvalidRun("frozen0 check (b) failed: the run is invalid")
    _save_pred(c, "frozen0", P0)
    return feat


def _search(c, task: dict, st: dict) -> None:
    """Successive halving over the encoder learning rate on the inner split (section 4)."""
    sch = task["schedule"]
    total, r0 = int(sch["epochs"]), int(sch["rung0"])
    ed = _enc_dir(c.working, c.name)
    _abort_if_invalid(c.working)
    _guard(c.itr, c.iva, c.forbid_outer)
    head0 = inner_refit(c)
    prev = st.get("search") or {}
    s = {"refit": {"E": c.E, "seed": 0, "differs_from_stage_a_head": True,
                   "forbidden": "validation group rows, test rows, gold rows"},
         "strategy": "successive halving, reduction factor 3; known bias: rung 0 favours fast learners",
         "rung0": {}, "rung1": {}}
    st["search"] = s
    ck = {}
    if task.get("search", True):
        for lr in LRS:
            _abort_if_invalid(c.working)
            r = new_run(c, c.src, head0, c.itr, lr, total, 0, val_rows=c.iva)
            run_epochs(c, r, r0)
            ck[lr] = ed / f"rung0_{_lr_key(lr)}.pt"
            save_checkpoint(c, r, ck[lr])
            s["rung0"][_lr_key(lr)] = {"val_macro_auc": _r(r.history[-1]["val_macro_auc"]), **_fit_summary(r)}
            _save_state(c.working, c.name, st)
        sel = select_rung0({lr: s["rung0"][_lr_key(lr)]["val_macro_auc"] for lr in LRS}, task.get("rung1_tie", True))
    else:  # the cut: the registered rate; rung-0 values of an earlier call of this session are kept
        s["rung0"] = prev.get("rung0") or {}
        sel = {"continue": [REGISTERED_LR], "tied": False, "note": "the search was cut: the registered 2e-5"}
    s["selection"] = {"continue": [float(x) for x in sel["continue"]], "tied": sel["tied"], "note": sel["note"]}
    final_scores = {}
    for lr in sel["continue"]:
        _abort_if_invalid(c.working)
        r = new_run(c, c.src, head0, c.itr, lr, total, 0, val_rows=c.iva)
        if lr in ck:
            load_checkpoint(c, r, ck[lr])
        run_epochs(c, r, total)
        final_scores[lr] = r.history[-1]["val_macro_auc"]
        s["rung1"][_lr_key(lr)] = {"val_macro_auc": _r(final_scores[lr]), "resumed_from_rung0": lr in ck,
                                   **_fit_summary(r)}
        _save_state(c.working, c.name, st)
    chosen, why = select_final(sel["continue"], final_scores)
    if not task.get("search", True):
        why = "the search was cut: the registered 2e-5"
    s["chosen_lr"], s["chosen_reason"] = float(chosen), why
    s["graph"] = "the 12-epoch training-and-validation graph is rung1[chosen_lr].curve"
    for p in ck.values():  # the rung checkpoints are not needed any more
        p.unlink(missing_ok=True)
    _save_state(c.working, c.name, st)
    record_chosen(c.working, c.name, {"lr": float(chosen), "reason": why,
                                      "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "before_any_test_pass_of_a_trained_model": True})
    s["chosen_lr_in_run_json"] = True
    _save_state(c.working, c.name, st)


def _chosen(c, st: dict) -> float:
    s = st.get("search") or {}
    if s.get("chosen_lr") is None:
        raise AssertionError("no chosen learning rate: run the search block first")
    lr = float(s["chosen_lr"])
    if not chosen_recorded(c.working, c.name, lr):  # a later call of run() rewrote run.json: record it again
        record_chosen(c.working, c.name, {"lr": lr, "reason": s.get("chosen_reason"),
                                          "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "before_any_test_pass_of_a_trained_model": True,
                                          "copied_from": "state.json of the search block"})
    return lr


def _b_lite(c, task: dict, st: dict, seed: int) -> None:
    import torch

    total = int(task["schedule"]["epochs"])
    lr = _chosen(c, st)
    _abort_if_invalid(c.working)
    _guard(c.tr, None, c.forbid_outer)
    r = new_run(c, c.src, c.head_state, c.tr, lr, total, seed)
    run_epochs(c, r, total)
    t0 = time.time()
    P = predict_test(c, c.src, r.head, trained=True, lr=lr)  # the test fold, once, at the last epoch
    _save_pred(c, f"b_lite_seed_{seed}", P)
    torch.save({"tail": c.enc.tail_state(), "head": r.head.state_dict()},
               _enc_dir(c.working, c.name) / f"b_lite_seed_{seed}.pt")
    st.setdefault("final", {}).setdefault("b_lite", {})[f"seed_{seed}"] = {
        "encoder_lr": lr, **_fit_summary(r), "test_seconds": round(time.time() - t0, 1)}
    _save_state(c.working, c.name, st)


def _final(c, task: dict, st: dict, feat: FeatSource) -> None:
    """Final fits from Stage A's fold-0 head on all training studies: head-only seeds 0-4, b-lite seed 0."""
    total = int(task["schedule"]["epochs"])
    _abort_if_invalid(c.working)
    lr = _chosen(c, st)
    _guard(c.tr, None, c.forbid_outer)
    st["final"] = {"head_only": {}, "b_lite": {},
                   "note": "training loss per epoch only; the test fold is scored once, at the last epoch"}
    for k in HEAD_ONLY_SEEDS:
        r = new_run(c, feat, c.head_state, c.tr, 0.0, total, k, with_encoder=False)
        run_epochs(c, r, total)
        _save_pred(c, f"head_only_seed_{k}", predict_test(c, feat, r.head, trained=True, lr=lr))
        st["final"]["head_only"][f"seed_{k}"] = _fit_summary(r)
    _save_state(c.working, c.name, st)
    _b_lite(c, task, st, 0)
    st["final"]["drift"] = {"mean_abs_z_frozen0": st["frozen0"]["mean_abs_z_training"],
                            "mean_abs_z_b_lite_last_epoch": st["final"]["b_lite"]["seed_0"]["mean_abs_z_last_epoch"],
                            "what": "mean absolute z of the training features under the fixed Stage A statistics"}
    _save_state(c.working, c.name, st)


def _task(task: dict) -> dict:
    """One encoder's part of a phase, in its own process when two GPUs exist. Returns only safe fields."""
    enc, working = task["encoder"], Path(task["working"])
    if task.get("spawned"):
        safe_log.worker_init()
        paths.WORKING = working
        try:
            import torch

            torch.set_num_threads(2)  # two processes share 4 CPU cores
        except Exception:  # noqa: BLE001
            pass
    try:
        phase, want = task["phase"], set(task["blocks"])
        st = _load_state(working, enc) if phase != "smoke" else {}
        c = _context(task)
        if not task.get("micro"):
            c.micro = pick_micro(c, c.head_state)
        if phase in ("smoke", "probe"):
            st.update(_base_block(c, task))
            (_smoke if phase == "smoke" else _probe)(c, task, st)
        elif phase == "main":
            st.update(_base_block(c, task))
            st["schedule"] = task["schedule"]["name"]
            feat = None
            if want & {"frozen0", "final"}:
                feat = _frozen0(c, st)
            elif not (st.get("frozen0") or {}).get("valid"):
                raise AssertionError("frozen0 has not passed: run the frozen0 block first")
            if "search" in want:
                _search(c, task, st)
            if "final" in want:
                _final(c, task, st, feat)
        elif phase == "seed1":
            _b_lite(c, task, st, 1)
        st["leakage_assertion_passed"] = True
        if phase != "smoke":
            _save_state(working, enc, st)
        return {"ok": True, "encoder": enc, "state": st}
    except BaseException as e:  # noqa: BLE001 - only the type and the location leave the process
        try:
            (_enc_dir(working, enc) / "error.log").write_text(
                "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "encoder": enc, "invalid": isinstance(e, InvalidRun),
                "error": safe_log.describe(type(e), e.__traceback__)}
    finally:
        try:  # encoders in turn on one GPU: free this encoder's tensors
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _map(tasks: list, parallel: bool) -> list:
    if not parallel:
        return [_task(t) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_task, {**t, "spawned": True}) for t in tasks]
        for t, f in zip(tasks, futs):
            try:
                out.append(f.result())
            except BaseException as e:  # noqa: BLE001 - a crashed worker
                out.append({"ok": False, "encoder": t["encoder"], "error": safe_log.describe(type(e), e.__traceback__)})
    return out


# ---------------------------------------------------------------- combine (the parent, CPU)

def _draws(n: int, n_boot: int, seed: int):
    """Study-bootstrap weights (n12._draws): n_boot times, n studies drawn with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def _pct(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return {"lo": None, "hi": None, "valid_share": 0.0}
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return {"lo": _r(lo), "hi": _r(hi), "valid_share": _r(float(ok.mean()), 3)}


def _load_pred(working, enc: str, arm: str, n: int, row_of: dict):
    p = _enc_dir(working, enc) / f"pred_{arm}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    P = np.full((n, len(FINDINGS)), np.nan)
    rows = df[UID].astype(str).map(row_of)
    ok = rows.notna().to_numpy()
    P[rows[ok].astype(int).to_numpy()] = df.loc[ok, FINDINGS].to_numpy(float)
    return P


def combine(d, working, encs, n_boot: int = N_BOOT, seed: int = 0, side_auc: bool = True) -> dict:
    """Scores, paired study bootstraps and the readings from the arms' saved fold-0 predictions.

    Rows: fold 0's scored test studies (gold is never in the test set), the same for every arm and both
    encoders, so every difference is paired. Metric: macro AUC over the 12 findings, report labels."""
    row_of = {u: i for i, u in enumerate(d.uids)}
    te_s = d.split(0)[2]
    Y, find = d.eval_y, list(range(len(FINDINGS)))
    arms = ["frozen0"] + [f"head_only_seed_{k}" for k in HEAD_ONLY_SEEDS] + ["b_lite_seed_0", "b_lite_seed_1"]
    P = {e: {a: _load_pred(working, e, a, d.n, row_of) for a in arms} for e in encs}
    P = {e: {a: v for a, v in pa.items() if v is not None} for e, pa in P.items()}
    need = ["frozen0", "head_only_seed_0", "b_lite_seed_0"]
    done = [e for e in encs if all(a in P[e] for a in need)]
    out: dict = {"label": DEV_LABEL, "display": DISPLAY, "added_after_results": True, "descriptive": True,
                 "metric": "fold-0 macro AUC over the 12 findings, scored test studies, report labels",
                 "interval": "study bootstrap, paired across arms and encoders, percentile 2.5-97.5",
                 "n_boot": int(n_boot), "encoders": {}, "limits": LIMITS,
                 "reading_rule": ["1: |b-lite minus the stricter baseline| under twice the head-only seed SD = seed noise",
                                  "2: b-lite beats frozen0 and head-only by 0.02 or more, both paired intervals exclude 0 = helps",
                                  "3: b-lite below both by 0.02 or more = lowered", "4: otherwise small or mixed",
                                  "the first line that fits is the reading; joint: helps only if both; mixed if they disagree"]}
    if not done:
        return {**out, "status": "not available (no encoder has all three arms)"}
    have = np.ones(d.n, bool)
    for e in done:
        for v in P[e].values():
            have &= np.isfinite(v).all(axis=1)
    ev = te_s[have[te_s]]
    out["evaluation_studies"] = agg.count(len(ev))
    out["all_scored_test_studies_used"] = bool(len(ev) == len(te_s))
    zeros = np.zeros(len(ev), int)
    W = list(_draws(len(ev), int(n_boot), seed))

    def mac(X, w=None):
        return metrics.macro(X[ev], Y[ev], zeros, find, w)

    point, draws, two_seeds = {}, {}, {}
    for e in done:
        point[e] = {a: mac(X) for a, X in P[e].items()}
        boot = [a for a in ("frozen0", "head_only_seed_0", "b_lite_seed_0", "b_lite_seed_1") if a in P[e]]
        draws[e] = {a: np.array([mac(P[e][a], w) for w in W]) for a in boot}
        two_seeds[e] = "b_lite_seed_1" in P[e]

    def b_lite(e, both: bool):
        """(point, draws) of b-lite: seed 0, or the mean of the two seeds' macro AUCs."""
        if both:
            return ((point[e]["b_lite_seed_0"] + point[e]["b_lite_seed_1"]) / 2,
                    (draws[e]["b_lite_seed_0"] + draws[e]["b_lite_seed_1"]) / 2)
        return point[e]["b_lite_seed_0"], draws[e]["b_lite_seed_0"]

    readings = {}
    for e in done:
        pt, dr = point[e], draws[e]
        bl, bl_d = b_lite(e, two_seeds[e])
        ho = [pt[f"head_only_seed_{k}"] for k in HEAD_ONLY_SEEDS if f"head_only_seed_{k}" in pt]
        sd = float(np.std(ho, ddof=1)) if len(ho) >= 2 else None
        f0, h0 = pt["frozen0"], pt["head_only_seed_0"]
        ci_f = {"estimate": _r(bl - f0), **_pct(bl_d - dr["frozen0"])}
        ci_h = {"estimate": _r(bl - h0), **_pct(bl_d - dr["head_only_seed_0"])}
        stricter = "frozen0" if f0 >= h0 else "head_only"
        read = reading(bl, f0, h0, sd, ci_f, ci_h)
        readings[e] = read
        out["encoders"][e] = {
            "macro_auc": {"frozen0": {"estimate": _r(f0), **_pct(dr["frozen0"])},
                          "head_only_seed_0": {"estimate": _r(h0), **_pct(dr["head_only_seed_0"])},
                          "b_lite_seed_0": {"estimate": _r(pt["b_lite_seed_0"]), **_pct(dr["b_lite_seed_0"])},
                          "b_lite_seed_1": ({"estimate": _r(pt["b_lite_seed_1"]), **_pct(dr["b_lite_seed_1"])}
                                            if two_seeds[e] else None),
                          "b_lite": _r(bl), "b_lite_is": "the mean of seeds 0 and 1" if two_seeds[e] else "seed 0"},
            "head_only_seed_macro_aucs": [_r(v) for v in ho],
            "head_only_seed_sd": _r(sd, 5), "twice_head_only_seed_sd": _r(2 * sd) if sd is not None else None,
            "seed_sd_note": "sample SD (n - 1) over the head-only seeds: a lower bound on a fine-tuned fit's seed noise",
            "b_lite_minus_frozen0": ci_f, "b_lite_minus_head_only": ci_h,
            "stricter_baseline": stricter,
            "stricter_baseline_note": "the better of frozen0 and head-only, chosen on the test fold: it can only shrink the effect",
            "effect": ci_f if stricter == "frozen0" else ci_h,
            "reading": read}
    out["readings"] = readings
    out["joint_reading"] = joint_reading(readings)
    # fold-0 Delta1 (DINOv2 minus ConvNeXt-T) for frozen0 and b-lite, and the paired interval of its change
    if all(e in done for e in MAIN):
        both = two_seeds[DINO] and two_seeds[CONV]
        (bd, bd_d), (bc, bc_d) = b_lite(DINO, both), b_lite(CONV, both)
        d_f = point[DINO]["frozen0"] - point[CONV]["frozen0"]
        d_fd = draws[DINO]["frozen0"] - draws[CONV]["frozen0"]
        change = {"estimate": _r((bd - bc) - d_f), **_pct((bd_d - bc_d) - d_fd)}
        excl = change["lo"] is not None and (change["lo"] > 0 or change["hi"] < 0)
        out["delta1_fold0"] = {
            "what": "DINOv2 minus ConvNeXt-T, fold-0 macro AUC", "label": DEV_LABEL,
            "frozen0": {"estimate": _r(d_f), **_pct(d_fd)}, "b_lite": {"estimate": _r(bd - bc), **_pct(bd_d - bc_d)},
            "b_lite_is": "the mean of seeds 0 and 1" if both else "seed 0", "change": change,
            "change_interval_excludes_0": bool(excl),
            "reading": ("the change's paired interval excludes 0" if excl else
                        "not read as a ranking change (the paired interval of the change includes 0)")}
    else:
        out["delta1_fold0"] = {"status": "not available (needs both main encoders)"}
    if side_auc:
        out["side_discrimination"] = _side(d, ev, Y, {a: {e: P[e][src] for e in done}
                                                      for a, src in (("frozen0", "frozen0"), ("b_lite", "b_lite_seed_0"))},
                                           n_boot, seed)
    else:
        out["side_discrimination"] = {"status": "cut (the cut order's second item)"}
    out["status"] = "ok"
    return out


def _side(d, ev, Y, units_by_arm: dict, n_boot: int, seed: int) -> dict:
    """N14-C2's side-discrimination AUC (its function) for the b-lite outputs beside the frozen value."""
    from . import n14

    out = {"what": "N14-C2 on fold 0's scored test studies: a description of FM3", "b_lite_is": "seed 0"}
    for arm, units in units_by_arm.items():
        r = n14.c2_side_discrimination(SimpleNamespace(d=d, ev=ev, Y=Y, units=units), n_boot, seed)
        r["scope"] = f"side information in the {arm} outputs, fold 0; report labels"
        r["studies"] = "fold 0's scored test studies with exactly one side positive; unresolved laterality left out"
        out[arm] = r
    return out


# ---------------------------------------------------------------- run

def _summary(s: dict) -> dict:
    enc = s.get("encoders") or {}
    res = (s.get("results") or {}).get("encoders") or {}

    def g(o, *path):
        for k in path:
            o = o.get(k) if isinstance(o, dict) else None
        return o

    out = {"status": s.get("status"), "errors": s.get("errors"), "schedule": g(s, "schedule", "name"),
           "throughput_slices_per_s": g(s, "schedule", "slices_per_s"), "seed1": g(s, "seed1", "ran"),
           "diagnostic_only": "nothing is kept; every number is a development number (R6, R8)",
           "encoders": {}, "joint_reading": g(s, "results", "joint_reading"),
           "delta1_fold0": {k: g(s, "results", "delta1_fold0", k) for k in ("frozen0", "b_lite", "change", "reading")},
           "side_discrimination": {arm: {k: {"reading": v.get("reading"), "status": v.get("status"),
                                             "side_auc": {e: x.get("side_auc") for e, x in (v.get("encoders") or {}).items()}}
                                         for k, v in (g(s, "results", "side_discrimination", arm, "pairs") or {}).items()}
                                   for arm in ("frozen0", "b_lite")}}
    for e, b in enc.items():
        out["encoders"][e] = {
            "trainable_parameters": g(b, "trainable", "parameters"), "peak_memory_gb": g(b, "probe", "peak_memory_gb"),
            "smoke": b.get("smoke"),
            "frozen0_valid": g(b, "frozen0", "valid"), "check_a_max_abs_gap": g(b, "frozen0", "check_a", "max_abs_gap"),
            "check_b_abs_difference": g(b, "frozen0", "check_b", "abs_difference"),
            "rung0_val_macro_auc": {k: v.get("val_macro_auc") for k, v in (g(b, "search", "rung0") or {}).items()},
            "rung1_val_macro_auc": {k: v.get("val_macro_auc") for k, v in (g(b, "search", "rung1") or {}).items()},
            "chosen_lr": g(b, "search", "chosen_lr"), "chosen_lr_in_run_json": g(b, "search", "chosen_lr_in_run_json"),
            "macro_auc": {k: (v.get("estimate") if isinstance(v, dict) else v)
                          for k, v in (g(res, e, "macro_auc") or {}).items()},
            "b_lite_minus_frozen0": g(res, e, "b_lite_minus_frozen0"),
            "b_lite_minus_head_only": g(res, e, "b_lite_minus_head_only"),
            "twice_head_only_seed_sd": g(res, e, "twice_head_only_seed_sd"), "reading": g(res, e, "reading")}
    return out


def run(root=None, expect: dict | None = None, encoders: tuple[str, ...] = MAIN, blocks=None, smoke: bool = False,
        n_boot: int = N_BOOT, seed: int = 0, budget_h: float = 5.0, quiet: bool = True, devices=None,
        processes: bool | None = None, seed1: bool | None = None, side_auc: bool = True, rung1_tie: bool = True,
        search: bool = True, schedule: str | None = None, min_slices_per_s: float = MIN_SLICES_PER_S,
        deadline_utc: str | None = None, inner_val_range=INNER_VAL_RANGE, encoder_factory=None,
        eval_studies: int = 4, probe_micro: int = PROBE_MICRO) -> dict:
    """Stage B-lite on outer fold 0 (see the module docstring). expect: the upstream hash prefixes of the card.

    smoke: 20 optimiser steps per encoder; reads no score; reports memory and slices/s (step n11_smoke).
    blocks: which of ("probe", "frozen0", "search", "final", "seed1", "combine") run (default all); a later
      call in the same session reuses private/n11/<encoder>/state.json and the saved predictions.
    The cut order, first to last (defaults = the declaration, nothing cut):
      seed1      None = b-lite seed 1 runs for both encoders only if it is projected to end before 0.8 x
                 budget_h (and before deadline_utc); True forces it; False cuts it.
      side_auc   False cuts N14-C2's side-discrimination AUC.
      rung1_tie  False: a tied pair at rung 0 does not continue as a pair (2e-5 if in the pair, else the lower).
      search     False cuts the learning-rate search: the registered 2e-5 (its inner 12-epoch run still runs).
    schedule: None = the throughput rule decides ("short" if the slower encoder is under min_slices_per_s
      over the first 200 micro-batches); "long" or "short" forces it (recorded as an override).
    processes: None = one process per encoder when at least two GPUs exist; False = in this process.
    budget_h: the wall-clock budget (the watchdog stops the step at 1.1 x). seed: the bootstrap's seed.
    encoder_factory, inner_val_range, min_slices_per_s, probe_micro, eval_studies: for tests (a module-level
      factory(name, device) -> Split; no weights are downloaded)."""
    if schedule is not None and schedule not in SCHEDULES:
        raise ValueError('schedule: None, "long" or "short"')
    want = set(blocks or BLOCKS)
    if not want <= set(BLOCKS):
        raise ValueError(f"blocks: a subset of {BLOCKS}")
    encs = tuple(encoders)
    name = f"{NAME}_smoke" if smoke else NAME
    overrides = bool(encoder_factory is not None or schedule is not None or tuple(inner_val_range or ()) != INNER_VAL_RANGE
                     or min_slices_per_s != MIN_SLICES_PER_S or probe_micro != PROBE_MICRO or encs != MAIN)
    cfg = {"encoders": list(encs), "blocks": sorted(want), "smoke": bool(smoke), "n_boot": int(n_boot), "seed": seed,
           "budget_h": budget_h, "seed1": seed1, "side_auc": side_auc, "rung1_tie": rung1_tie, "search": search,
           "schedule": schedule, "min_slices_per_s": min_slices_per_s, "deadline_utc": deadline_utc,
           "eval_studies": eval_studies, "test_overrides": overrides}
    s: dict = {"written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scope": SCOPE,
               "plan": "iteration plan section 11 (it3_declaration_r3)", "added_after_results": True,
               "descriptive": True, "label": DEV_LABEL, "display": DISPLAY}
    with common.notebook(name, cfg, quiet=quiet, budget_h=budget_h):
        t0 = time.time()
        working = paths.WORKING
        d = stage.StageData(root, expect)
        s["upstream"] = d.upstream
        devs = n5._devices(devices)
        parallel = bool(len(devs) >= 2 and len(encs) >= 2) if processes is None else bool(processes)
        s["processes"] = "one per encoder" if parallel else "one (encoders in turn)"
        s["cuts"] = {"seed1": seed1 is False, "side_auc": not side_auc, "rung1_for_a_tied_pair": not rung1_tie,
                     "search": not search}
        base = {"root": str(root) if root else None, "expect": expect, "working": str(working),
                "factory": encoder_factory, "inner_val_range": list(inner_val_range) if inner_val_range else None,
                "eval_studies": eval_studies, "search": search, "rung1_tie": rung1_tie, "probe_micro": probe_micro}

        def tasks(phase: str, **extra) -> list:
            return [{**base, "encoder": e, "device": devs[i % len(devs)], "phase": phase,
                     "blocks": sorted(want), "micro": micro.get(e), **extra} for i, e in enumerate(encs)]

        def collect(results: list, what: str) -> None:
            for r in results:
                if r.get("ok"):
                    s.setdefault("encoders", {})[r["encoder"]] = r["state"]
            bad = {r["encoder"]: r.get("error") for r in results if not r.get("ok")}
            invalid = any(r.get("invalid") for r in results) or _invalid_flag(working).exists()
            if invalid:  # nothing else is read: only each encoder's frozen0 checks are written
                s["encoders"] = {e: {"frozen0": _load_state(working, e).get("frozen0")} for e in encs}
                s["status"] = "invalid: frozen0 check (b) failed; nothing else is read"
                s["errors"] = bad
                agg.write(name, s)
                raise InvalidRun("frozen0 check (b) failed: the run is invalid")
            if bad:
                s["errors"] = bad
                s["status"] = f"failed in {what}"
                agg.write(name, s)
                raise N11Error(f"an encoder's task failed in {what}")
            s["status"] = f"running ({what} done)"
            agg.write(name, s)

        micro: dict = {}
        if smoke:
            collect(_map(tasks("smoke"), parallel), "smoke")
            s["status"] = "ok (smoke: no score was read)"
            agg.write(name, s)
        else:
            _invalid_flag(working).unlink(missing_ok=True)
            sched_file = Path(working) / "private" / NAME / "schedule.json"
            # 1. the throughput rule
            if want == set(BLOCKS):  # a full run starts from an empty state
                for e in encs:
                    (_enc_dir(working, e) / "state.json").unlink(missing_ok=True)
            sch = None
            if "probe" in want:
                collect(_map(tasks("probe", schedule=LONG), parallel), "probe")
                rates = {e: s["encoders"][e]["probe"]["slices_per_s"] for e in encs}
                sch = dict(SCHEDULES[schedule]) if schedule else schedule_for(rates.values(), min_slices_per_s)
                s["schedule"] = {"name": sch["name"], "config": {"epochs": sch["epochs"], "rung0": sch["rung0"]},
                                 "slices_per_s": rates, "minimum_slices_per_s": min_slices_per_s,
                                 "rule": "the slower encoder under the minimum over the first 200 micro-batches of the "
                                         "first rung-0 run: both run 6 epochs (rungs 2 and 6)",
                                 "forced": schedule is not None,
                                 "note": "the first rung-0 run restarts under either schedule (the probe is not reused)"}
                sched_file.parent.mkdir(parents=True, exist_ok=True)
                sched_file.write_text(json.dumps(s["schedule"]))
            elif schedule:
                sch = dict(SCHEDULES[schedule])
                s["schedule"] = {"name": sch["name"], "config": {"epochs": sch["epochs"], "rung0": sch["rung0"]}, "forced": True}
            elif sched_file.exists():
                s["schedule"] = json.loads(sched_file.read_text())
                sch = dict(SCHEDULES[s["schedule"]["name"]])
            elif want & {"frozen0", "search", "final", "seed1"}:
                raise ValueError("no schedule: run the probe block or pass schedule")
            micro = {e: ((_load_state(working, e).get("micro") or {}).get("config") or {}).get("micro_batch_studies")
                     for e in encs}
            # 2. frozen0, the search and the final fits
            if want & {"frozen0", "search", "final"}:
                collect(_map(tasks("main", schedule=sch), parallel), "frozen0, search and final fits")
            # 3. b-lite seed 1, for both encoders or neither (the first cut)
            if "seed1" in want:
                st = {e: _load_state(working, e) for e in encs}
                fit = [((st[e].get("final") or {}).get("b_lite") or {}).get("seed_0") or {} for e in encs]
                secs = [f.get("train_seconds", 0.0) + f.get("test_seconds", 0.0) for f in fit]
                need_s = 1.15 * (max(secs) if parallel else sum(secs))
                limit = t0 + SEED1_SHARE * budget_h * 3600
                if deadline_utc:
                    from .n6 import _deadline_epoch

                    limit = min(limit, _deadline_epoch(deadline_utc))
                ahead = bool(all(fit) and time.time() + need_s <= limit)
                go = bool(all(fit) and (seed1 is True or (seed1 is None and ahead)))
                s["seed1"] = {"ran": go, "ahead_of_budget": ahead, "projected_seconds": round(need_s, 1),
                              "rule": "seed 1 runs for both encoders only if it is projected to end before 0.8 x the "
                                      "budget (the first cut); forced or cut by the seed1 argument",
                              "argument": seed1}
                if go:
                    collect(_map(tasks("seed1", schedule=sch), parallel), "b-lite seed 1")
            for e in encs:
                s.setdefault("encoders", {})[e] = _load_state(working, e)
            # 4. scores, intervals and readings
            if "combine" in want:
                s["results"] = combine(d, working, encs, n_boot=n_boot, seed=seed, side_auc=side_auc)
                s["status"] = "ok" if s["results"].get("status") == "ok" else "ok (blocks so far; results not available)"
            else:
                s["status"] = "ok (blocks so far)"
            s["seconds"] = round(time.time() - t0, 1)
            agg.write(name, s)
    try:
        common.summary(_summary(s))
    except agg.UnsafeSummary:
        common.summary({"status": s.get("status"), "errors": s.get("errors"),
                        "summary_error": "the summary block was refused by the checks; read agg_n11.json"})
    return s
