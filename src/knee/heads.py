"""Stage A heads on frozen slice embeddings (plan, Phase 4: Stage A, search spaces, fixed settings).

A head pools each plane's 32 slice features (mean, max, or gated attention
with one attention map per finding), joins the planes and gives 12 logits.
Loss: binary cross-entropy on the soft labels, with per-cell weights (0.5
where pilkwang says UNK, 0 where masked) and a positive weight per finding
(negatives / positives on the training set, capped by the tuned cap).
AdamW (0.9/0.999, eps 1e-8); 1 epoch linear warm-up, then cosine to 0 over a
30-epoch schedule. Inner fits stop early after 5 epochs without a gain in the
inner validation macro AUC; refits run exactly E epochs of the same schedule.
The schedule length and the patience can be set per fit (fit's schedule_epochs and
patience, iteration 2); the defaults are the registered 30 and 5, bit for bit.
Features are z-scored with statistics from the outer training set only.
Embeddings stay on the GPU; there is no DataLoader.

Leakage guard: every fit receives the outer test indices as `forbidden` and
refuses to start if any of them is in its training or validation set
(the run-time assertion of the plan); guard_checks counts the checks passed.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import metrics

PLANES = ("sagittal", "coronal", "axial")
FLIP_PLANES = ("coronal", "axial")
POOLINGS = ("mean", "max", "gated")
PLANE_SETS = {"sag": ("sagittal",), "sag+cor": ("sagittal", "coronal"), "all": PLANES}
L1_GRID = [(p, s) for p in POOLINGS for s in PLANE_SETS]
MAX_EPOCHS = 30
PATIENCE = 5
WARMUP_EPOCHS = 1
N_OUT = 12

import threading

guard_checks = 0  # fits whose index sets passed the leakage assertion
_INIT_LOCK = threading.Lock()  # serialises seeded weight initialisation across threads (N6p runs two GPUs)


def scoring_labels(binary) -> np.ndarray:
    """The plan's scoring rule: uncertain (exactly 0.5 or masked) counts as negative."""
    return np.nan_to_num(np.asarray(binary, float), nan=0.0)


class LeakageError(AssertionError):
    """A test-fold index reached a fit (the plan's run-time leakage assertion)."""


@dataclass(frozen=True)
class Config:
    pooling: str = "gated"
    planes: str = "all"
    lr: float = 3e-4
    weight_decay: float = 3e-4
    dropout: float = 0.25
    attn_hidden: int = 192
    pos_cap: float = math.sqrt(20.0)
    batch_size: int = 48

    def to_dict(self) -> dict:
        return asdict(self)


def default_config(pos_cap_upper: float, pooling: str = "gated", planes: str = "all") -> Config:
    """L2 defaults: the geometric centre of each range; the cap default is sqrt(U)."""
    return Config(pooling=pooling, planes=planes, pos_cap=float(math.sqrt(pos_cap_upper)))


@dataclass
class FitResult:
    best_epoch: int
    best_score: float
    history: list = field(default_factory=list)
    epochs_run: int = 0
    seconds: float = 0.0
    model: object = None
    pruned: bool = False
    steps_run: int = 0
    final_lr: float = float("nan")


class Bank:
    """One encoder's slice embeddings and the labels, on one device.

    emb[plane]: float16 [N, S, D]; soft, weight: float32 [N, 12] (soft NaN = masked);
    binary: float [N, 12] with NaN for uncertain (positive weights only); eval_y: the
    scoring labels (plan: uncertain counts as negative), used for inner validation;
    flip[plane]: float16 [M, S, D] for unresolved-laterality studies (flip_row >= 0),
    coronal and axial only.
    """

    def __init__(self, emb: dict, soft, weight, binary, flip: dict | None = None, flip_row=None,
                 device: str = "cpu", eval_y=None):
        import torch

        self.device = torch.device(device)
        self.x = {p: torch.from_numpy(np.ascontiguousarray(emb[p])).to(self.device) for p in emb}
        soft = np.asarray(soft, np.float32)
        wt = np.asarray(weight, np.float32) * ~np.isnan(soft)
        self.y = torch.from_numpy(np.nan_to_num(soft, nan=0.0)).to(self.device)
        self.w = torch.from_numpy(np.nan_to_num(wt, nan=0.0).astype(np.float32)).to(self.device)
        self.binary = np.asarray(binary, float)
        self.eval_y = scoring_labels(self.binary) if eval_y is None else np.asarray(eval_y, float)
        self.n = len(soft)
        self.flip = {p: torch.from_numpy(np.ascontiguousarray(a)).to(self.device) for p, a in (flip or {}).items()}
        self.flip_row = np.full(self.n, -1, int) if flip_row is None else np.asarray(flip_row, int)

    def with_labels(self, soft, weight, binary) -> "Bank":
        """The same embeddings with other labels (no copy of the embeddings)."""
        import torch

        b = object.__new__(Bank)
        b.__dict__.update(self.__dict__)
        soft = np.asarray(soft, np.float32)
        wt = np.asarray(weight, np.float32) * ~np.isnan(soft)
        b.y = torch.from_numpy(np.nan_to_num(soft, nan=0.0)).to(self.device)
        b.w = torch.from_numpy(np.nan_to_num(wt, nan=0.0).astype(np.float32)).to(self.device)
        b.binary = np.asarray(binary, float)
        b.eval_y = scoring_labels(b.binary)
        return b

    def stats(self, idx) -> dict:
        """Per-plane mean and std of the features over the slices of studies idx (float32)."""
        import torch

        idx_t = torch.as_tensor(np.asarray(idx), device=self.device)
        dt = torch.float32 if self.device.type == "mps" else torch.float64  # MPS has no float64
        out = {}
        for p, x in self.x.items():
            s1 = torch.zeros(x.shape[-1], device=self.device, dtype=dt)
            s2 = torch.zeros_like(s1)
            cnt = 0
            for i in range(0, len(idx_t), 256):
                xb = x[idx_t[i:i + 256]].to(dt).reshape(-1, x.shape[-1])
                s1 += xb.sum(0)
                s2 += (xb * xb).sum(0)
                cnt += xb.shape[0]
            mu = s1 / max(cnt, 1)
            sd = (s2 / max(cnt, 1) - mu * mu).clamp_min(1e-12).sqrt()
            out[p] = (mu.float(), sd.float().clamp_min(1e-6))
        return out


def make_head(cfg: Config, dim: int, n_planes: int):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(cfg.dropout)
            if cfg.pooling == "gated":
                h = int(cfg.attn_hidden)
                self.V = nn.ModuleList([nn.Linear(dim, h) for _ in range(n_planes)])
                self.U = nn.ModuleList([nn.Linear(dim, h) for _ in range(n_planes)])
                self.A = nn.ModuleList([nn.Linear(h, N_OUT) for _ in range(n_planes)])
                fan_in = dim * n_planes
                self.out_w = nn.Parameter(torch.empty(N_OUT, fan_in))
                self.out_b = nn.Parameter(torch.empty(N_OUT))
                nn.init.kaiming_uniform_(self.out_w, a=math.sqrt(5))  # as nn.Linear
                nn.init.uniform_(self.out_b, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
            else:
                self.fc = nn.Linear(dim * n_planes, N_OUT)

        def attention(self, xs):
            """Per plane: attention weights over slices, [B, S, 12] (gated pooling only)."""
            return [torch.softmax(self.A[i](torch.tanh(self.V[i](x)) * torch.sigmoid(self.U[i](x))), dim=1)
                    for i, x in enumerate(xs)]

        def forward(self, xs):
            if cfg.pooling == "mean":
                return self.fc(self.drop(torch.cat([x.mean(1) for x in xs], -1)))
            if cfg.pooling == "max":
                return self.fc(self.drop(torch.cat([x.amax(1) for x in xs], -1)))
            pooled = [torch.einsum("bsk,bsd->bkd", a, x) for a, x in zip(self.attention(xs), xs)]
            z = self.drop(torch.cat(pooled, -1))
            return (z * self.out_w).sum(-1) + self.out_b

    return Head()


def _inputs(bank: Bank, idx_t, planes, stats, flipped: bool = False):
    import torch

    xs = []
    for p in planes:
        mu, sd = stats[p]
        if flipped and p in bank.flip:
            rows = torch.as_tensor(bank.flip_row[idx_t.cpu().numpy()], device=bank.device)
            x = bank.flip[p][rows]
        else:
            x = bank.x[p][idx_t]
        xs.append((x.float() - mu) / sd)
    return xs


def _check_guard(train_idx, val_idx, forbidden) -> None:
    global guard_checks
    tr = set(np.asarray(train_idx).tolist())
    va = set(np.asarray(val_idx).tolist()) if val_idx is not None else set()
    fb = set(np.asarray(forbidden).tolist()) if forbidden is not None else set()
    if tr & fb or va & fb or tr & va:
        raise LeakageError("test or validation indices reached a fit")
    guard_checks += 1


def predict(bank: Bank, model, idx, cfg: Config, stats: dict, use_flip: bool = True, batch: int = 256) -> np.ndarray:
    """Probabilities [len(idx), 12]; unresolved-laterality studies average stored and flipped slices."""
    import torch

    planes = PLANE_SETS[cfg.planes]
    idx = np.asarray(idx)
    out = np.zeros((len(idx), N_OUT), np.float32)
    model.eval()
    flip_any = use_flip and any(p in bank.flip for p in planes)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            bt = torch.as_tensor(b, device=bank.device)
            p = torch.sigmoid(model(_inputs(bank, bt, planes, stats))).float().cpu().numpy()
            if flip_any:
                m = bank.flip_row[b] >= 0
                if m.any():
                    bf = torch.as_tensor(b[m], device=bank.device)
                    pf = torch.sigmoid(model(_inputs(bank, bf, planes, stats, flipped=True))).float().cpu().numpy()
                    p[m] = (p[m] + pf) / 2
            out[i:i + batch] = p
    return out


def schedule_factor(step: int, steps_per_epoch: int, schedule_epochs: int | None = None) -> float:
    """The learning rate at optimiser step `step` (from 0) as a fraction of its peak: linear warm-up
    over WARMUP_EPOCHS, then a cosine to 0 at the end of a schedule of `schedule_epochs` epochs
    (None: MAX_EPOCHS, the registered 30). fit() uses exactly this, so a refit stopped at E epochs
    ends at schedule_factor(E * steps_per_epoch - 1, ...)."""
    total = (MAX_EPOCHS if schedule_epochs is None else int(schedule_epochs)) * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def fit(bank: Bank, cfg: Config, train_idx, val_idx=None, findings=None, epochs: int | None = None,
        seed: int = 0, stats: dict | None = None, forbidden=None, max_epochs: int | None = None,
        patience: int | None = PATIENCE, on_epoch=None, steps: int | None = None,
        schedule_steps_per_epoch: int | None = None, schedule_epochs: int | None = None) -> FitResult:
    """Inner fit (val_idx given): early stopping on the validation macro AUC over `findings`, after
    `patience` epochs without a gain (None: no stopping rule).
    Refit (epochs given): exactly that many epochs of the schedule; no stopping rule.
    Fixed-steps refit (steps given, KneeMRI S5): exactly `steps` optimiser steps of a schedule whose
    epoch is `schedule_steps_per_epoch` steps (the RSNA refit's), so learning rate and step count
    match the RSNA refit even though the dataset is smaller.
    schedule_epochs: length of the cosine schedule in epochs (None: MAX_EPOCHS = 30, as registered).
    max_epochs: the most epochs a fit without `epochs` or `steps` runs; None means schedule_epochs
    when that is given, else MAX_EPOCHS. So an inner fit never runs past its own schedule, and a
    refit with epochs=E and the same schedule_epochs stops at E of that schedule (E above
    schedule_epochs is refused: those epochs would train at learning rate 0).
    With schedule_epochs, max_epochs and patience left at their defaults the fit is the registered
    one, bit for bit (tests/test_iteration2.py compares it with commit a260cde).
    on_epoch(epoch, score) -> True stops the fit as pruned (the HPO demo's successive halving)."""
    import torch
    import torch.nn.functional as F

    if schedule_epochs is not None and int(schedule_epochs) < 1:
        raise ValueError("schedule_epochs must be at least 1")
    if schedule_epochs is not None and epochs is not None and int(epochs) > int(schedule_epochs):
        raise ValueError("a refit cannot run past its schedule (epochs > schedule_epochs)")
    _check_guard(train_idx, val_idx, forbidden)
    t0 = time.time()
    gen = torch.Generator().manual_seed(seed)
    planes = PLANE_SETS[cfg.planes]
    stats = stats or bank.stats(train_idx)
    dim = bank.x[planes[0]].shape[-1]
    with _INIT_LOCK:  # the CPU generator only, so another device's dropout stream is untouched
        torch.default_generator.manual_seed(seed)
        model = make_head(cfg, dim, len(planes))
    if bank.device.type == "cuda":
        with torch.cuda.device(bank.device):
            torch.cuda.manual_seed(seed)
    elif bank.device.type == "mps":
        torch.mps.manual_seed(seed)
    model = model.to(bank.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=cfg.weight_decay)
    train_idx = np.asarray(train_idx)
    n = len(train_idx)
    bs = max(1, int(cfg.batch_size))
    steps_per_epoch = math.ceil(n / bs)
    spe_sched = int(schedule_steps_per_epoch) if schedule_steps_per_epoch else steps_per_epoch

    def lr_at(step):
        return schedule_factor(step, spe_sched, schedule_epochs)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    yb = bank.binary[train_idx]
    pos, neg = (yb == 1).sum(0), (yb == 0).sum(0)
    pw = torch.tensor(np.minimum(neg / np.maximum(pos, 1), cfg.pos_cap), dtype=torch.float32, device=bank.device)
    tr_t = torch.as_tensor(train_idx, device=bank.device)
    if max_epochs is None:  # an inner fit runs at most its own schedule (registered: 30 epochs)
        max_epochs = MAX_EPOCHS if schedule_epochs is None else int(schedule_epochs)
    if steps is not None:  # fixed number of optimiser steps: as many epochs as it takes
        n_epochs = math.ceil(int(steps) / steps_per_epoch)
    else:
        n_epochs = int(epochs) if epochs is not None else int(max_epochs)
    res = FitResult(best_epoch=n_epochs, best_score=float("nan"))
    best, since = -np.inf, 0
    done = 0
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(n, generator=gen).to(bank.device)]
        for i in range(0, n, bs):
            if steps is not None and done >= int(steps):
                break
            b = perm[i:i + bs]
            logits = model(_inputs(bank, b, planes, stats))
            w = bank.w[b]
            loss = F.binary_cross_entropy_with_logits(logits, bank.y[b], weight=w, pos_weight=pw,
                                                      reduction="sum") / w.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            res.final_lr = float(opt.param_groups[0]["lr"])
            sched.step()
            done += 1
        res.epochs_run = ep
        res.steps_run = done
        if val_idx is not None and epochs is None:
            pv = predict(bank, model, val_idx, cfg, stats, use_flip=False)
            yv = bank.eval_y[np.asarray(val_idx)]
            score = metrics.macro_available(pv, yv, findings or list(range(N_OUT)))
            res.history.append(round(float(score), 4) if np.isfinite(score) else None)
            if np.isfinite(score) and score > best:
                best, since = score, 0
                res.best_epoch, res.best_score = ep, float(score)
            else:
                since += 1
                if patience is not None and since >= patience:
                    break
            if on_epoch is not None and on_epoch(ep, float(score) if np.isfinite(score) else None):
                res.pruned = True
                break
    res.seconds = time.time() - t0
    res.model = model
    return res


def config_from(d: dict) -> Config:
    keys = Config.__dataclass_fields__
    return Config(**{k: v for k, v in d.items() if k in keys})


def with_l1(cfg: Config, pooling: str, planes: str) -> Config:
    return replace(cfg, pooling=pooling, planes=planes)
