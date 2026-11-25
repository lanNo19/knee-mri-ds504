"""N13s: the slice-count check (it2c_slice_count_r2.md), "added after the results" (R10).

The project proposal promised a search over the slice count. It was never searched: every model reads
32 slices per plane, the number fixed when the image cache was built. Question: does the head, at its
chosen settings, score lower on inner validation when it reads fewer of the 32 cached slices per plane?

Five arms, each its own CPU notebook (a fork of knee-n13-bc): s32, s16, s16b, s8, s4. Every arm fits, per unit
(main encoder x outer fold), the unit's chosen Stage A configuration at its chosen batch size on the
unit's inner folds, exactly as n13's batch-floor arm bc does (n13._batch_unit with the chosen batch size):
the same inner sets, leakage guard, seed 0 + j, 30-epoch schedule, patience 5 and scoring. The only
change in an arm is the slices the head sees. Arm s<n> keeps the cached slice positions
floor((i + 0.5) * 32 / n), i = 0..n-1, the same positions in all three planes, in training and in
validation, for every study:
  s16: 1, 3, ..., 31    s8: 2, 6, ..., 30    s4: 4, 12, 20, 28    s32: 0..31 (the stored arrays, untouched).
  s16b: the 16 positions s16 leaves out (0, 2, ..., 30). It is descriptive only (s16b minus s16: the size
  of the position effect at a fixed count) and never enters a verdict.
The flipped embeddings of unresolved-laterality studies are cut at the same positions for consistency;
the inner score never reads them (inner validation predicts without flips). The standardisation statistics are recomputed over the kept slices of the outer training set. s32 is the
base of the rule; it repeats arm bc at this module's commit. The encoder is not rerun: the slices were
encoded one by one, so dropping a slice does not change the embedding of another slice.
No arm refits on the outer training set, predicts a test fold or writes a unit folder. Only the inner
score is kept. The test fold and the gold studies reach each fit only as forbidden indices.

This module only adds: nothing in n13.py, n6.py or heads.py is changed, and the worker's cached bank
(n6._W) is never modified; an arm with fewer slices reads its own copy of the kept slices.
Output of an arm: agg_n13_<arm>_inner.json (step n13_<arm>) and run.json.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .. import agg, heads, paths, safe_log, stage
from ..labels import FINDINGS
from . import common, n6, n13

MAIN = n13.MAIN
SLICE_PLAN = "it2c_slice_count_r2.md: the slice-count check, declared before the run, added after the results"
SLICE_REASON = ("it2c slice-count check: the chosen Stage A configuration on the inner folds; "
                "only the slices the head reads differ")
N_CACHED = 32  # slices per plane in the embedding cache (dicom_io.N_SLICES): the most that can be tested
SLICE_ARMS = {"s32": 32, "s16": 16, "s16b": 16, "s8": 8, "s4": 4}  # slices per plane the head reads
SLICE_SEED = 0  # arm bc's seed: inner fit j uses seed + j
SLICE_BUDGET_H = 4.0  # the soft stop of a session
SLICE_SCORE = ("inner score: mean over the unit's inner folds of the validation macro AUC at each inner fit's "
               "best epoch (Stage A's objective), for the unit's chosen configuration on the arm's slices")
_S: dict = {}  # worker state of this module: the cut bank of the current encoder (n6._W is left alone)


def _arm(arm: str) -> int:
    if arm not in SLICE_ARMS:
        raise ValueError("unknown slice-count arm: use s32, s16, s16b, s8 or s4")
    return SLICE_ARMS[arm]


def slice_positions(arm: str) -> list:
    """The cached slice positions an arm keeps: floor((i + 0.5) * 32 / n), i = 0..n-1, for arm s<n>;
    for s16b the 16 positions that s16 leaves out (0, 2, ..., 30)."""
    n = _arm(arm)
    pos = [((2 * i + 1) * N_CACHED) // (2 * n) for i in range(n)]  # the same floor, in whole numbers
    if arm == "s16b":
        pos = [q for q in range(N_CACHED) if q not in pos]
    return pos


def positions_text(arm: str) -> str:
    """The positions as a short string (first, second, ..., last): no list of small numbers in a summary."""
    p = slice_positions(arm)
    return f"{p[0]},{p[1]},...,{p[-1]}"


def slice_config(arm: str) -> dict:
    """What an arm's head reads (it sits under a "config" key in every summary: settings, not counts)."""
    return {"slices": int(_arm(arm)), "slices_cached": N_CACHED, "slice_positions": positions_text(arm)}


def slice_settings(arm: str) -> dict:
    """The settings block of a slice-count arm: arm bc's settings plus the slices under "config"."""
    base = n13.batch_settings("bc")
    return {**base, "config": {**base["config"], **slice_config(arm)}}


def slice_bank(bank, arm: str):
    """The bank an arm fits on.

    Every plane of the bank (stored and flipped embeddings) must hold exactly 32 slices: ValueError
    otherwise. s32 returns the bank object itself, untouched. Any other arm returns a new Bank object
    that shares the labels, weights, flip rows and scoring labels with the given bank and holds its own
    copy of the kept slices (x and flip, the same positions in every plane). The given bank is not
    modified. The flipped embeddings are cut for consistency only: an inner fit never reads them
    (heads.fit trains on the stored slices and validates with use_flip=False)."""
    pos = slice_positions(arm)
    for group in (bank.x, bank.flip):
        for p, a in group.items():
            if a.dim() != 3 or int(a.shape[1]) != N_CACHED:
                raise ValueError(f"the slice-count check needs {N_CACHED} cached slices in every plane ({p} differs)")
    if arm == "s32":
        return bank
    import torch

    idx = torch.as_tensor(pos, dtype=torch.long, device=bank.device)
    cut = object.__new__(heads.Bank)
    cut.__dict__.update(bank.__dict__)
    cut.x = {p: a.index_select(1, idx).contiguous() for p, a in bank.x.items()}
    cut.flip = {p: a.index_select(1, idx).contiguous() for p, a in bank.flip.items()}
    return cut


def _arm_bank(enc: str, arm: str):
    """The worker's bank for one encoder and arm (the cut copy is kept until the encoder or the arm changes)."""
    bank = n6._bank(enc)
    if arm == "s32":
        return slice_bank(bank, arm)  # the cached bank itself, after the check of its shape
    if _S.get("source") is not bank or _S.get("arm") != arm:
        _S.clear()
        _S.update(source=bank, arm=arm, bank=slice_bank(bank, arm))
    return _S["bank"]


def _slice_task(task: dict) -> dict:
    """Worker entry (spawned): one unit; a failure keeps its traceback under private/<step>/errors/."""
    return n6._guarded(_slice_unit, task, Path(task["dir"]))


def _slice_unit(task: dict, od: Path) -> dict:
    """The inner fits of one unit: n13._batch_unit at the chosen batch size, on the arm's slices.

    Past the deadline no fit starts, and a running fit stops at the end of its epoch (heads.fit's
    on_epoch): the unit then returns stopped, with no score. Nothing is written to disk."""
    t0 = time.time()
    if t0 > task["deadline"]:
        return {"task": task["id"], "ok": False, "stopped": True}
    enc, k, seed, elig = task["encoder"], task["fold"], task["seed"], task["eligible"]
    arm = task["arm"]
    n = _arm(arm)
    chosen = heads.config_from(task["config_chosen"])
    cfg = chosen  # the chosen configuration, its batch size included
    d, bank = n6._data(), _arm_bank(enc, arm)
    tr, te, _ = d.split(k)
    forbid = np.concatenate([te, d.gold_rows()])  # the test fold and every gold study: forbidden in each fit
    inner = d.inner(k)
    folds_in = sorted(set(inner[tr][inner[tr] >= 0].tolist()))
    stats = bank.stats(tr)  # over the kept slices of the outer training set
    checks0 = heads.guard_checks
    sc, best_eps, ran, spe, spe_chosen = [], [], [], [], []
    for j in folds_in:
        if time.time() > task["deadline"]:
            return {"task": task["id"], "ok": False, "stopped": True}
        itr = tr[inner[tr] != j]
        iva = tr[(inner[tr] == j) & d.scored[tr]]
        r = heads.fit(bank, cfg, itr, iva, elig, stats=stats, forbidden=forbid, seed=seed + j,
                      patience=heads.PATIENCE, schedule_epochs=None,
                      on_epoch=lambda ep, score: time.time() > task["deadline"])
        if r.pruned and r.epochs_run < heads.MAX_EPOCHS:  # stopped by the deadline before its own end
            return {"task": task["id"], "ok": False, "stopped": True}
        sc.append(r.best_score)
        best_eps.append(r.best_epoch)
        ran.append(int(r.epochs_run))
        spe.append(math.ceil(len(itr) / max(1, int(cfg.batch_size))))
        spe_chosen.append(math.ceil(len(itr) / max(1, int(chosen.batch_size))))
    fits = len(folds_in)
    sc = [x for x in sc if np.isfinite(x)]
    if not sc:
        raise ValueError("no finite inner score")
    used = cfg.to_dict()
    if {key for key in used if used[key] != task["config_chosen"][key]}:
        raise AssertionError("a setting differs from the chosen configuration")
    seen = {p: int(bank.x[p].shape[1]) for p in heads.PLANE_SETS[cfg.planes]}
    if set(seen.values()) != {n} or {int(a.shape[1]) for a in bank.flip.values()} - {n}:
        raise AssertionError("the head did not read the arm's number of slices in every plane")
    summary = {"encoder": enc, "fold": k, "inner_score": round(float(np.mean(sc)), n13.DECIMALS),
               "E": max(1, int(np.median(best_eps))), "epochs": ran, "inner_k": fits, "seed": seed,
               "schedule_epochs": heads.MAX_EPOCHS, "patience": heads.PATIENCE,
               "device": n6._W["device"], "device_name": n6._device_name(n6._W["device"]),
               "config": used, "batch_chosen": int(chosen.batch_size),
               "slices": {"config": slice_config(arm)},
               "steps_per_epoch": n13._r5(np.mean(spe)), "steps_per_epoch_chosen": n13._r5(np.mean(spe_chosen)),
               "step_ratio": round(float(np.sum(spe) / max(1, np.sum(spe_chosen))), 2),
               "stage_a_best_inner_score": task["stage_a_best"], "chosen_sha16": task["chosen_sha16"],
               "seconds": round(time.time() - t0, 1)}
    return {"task": task["id"], "ok": True, "summary": summary, "fits": fits,
            "guard_checks": heads.guard_checks - checks0}


def slice_summary(arm: str, step: str, results: dict, task_ids: list, environment: dict | None,
                  encoders, budget_h: float | None = None, seconds: float | None = None,
                  ref_checked: bool = True) -> dict:
    """The arm's summary block (agg_<step>_inner.json) from the units' results {unit id: worker result}.

    status: "ok" when every unit of the session finished, else "partial" (a unit that was stopped by the
    soft stop, failed, or has no result yet). The finished units are always written."""
    units, scores, bad_guard = {}, {}, []
    for uid in task_ids:
        r = results.get(uid)
        if r is None:
            units[uid] = {"ok": False, "stopped": True, "error": None}
        elif not r.get("ok"):
            units[uid] = {"ok": False, "stopped": bool(r.get("stopped")), "error": r.get("error")}
        else:
            u = r["summary"]
            units[uid] = {"ok": True, **{key: v for key, v in u.items() if key not in ("encoder", "fold")}}
            scores.setdefault(u["encoder"], {})[int(u["fold"])] = u["inner_score"]
            if r.get("guard_checks") != r.get("fits"):
                bad_guard.append(uid)
    done = [uid for uid in task_ids if units[uid]["ok"]]
    errors = [str(units[uid].get("error")) for uid in task_ids if not units[uid]["ok"]]
    return {"arm": arm, "step": step, "plan": SLICE_PLAN, "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": SLICE_REASON,
            "encoders": list(encoders), "settings": slice_settings(arm),
            "chosen_ref_checked": bool(ref_checked),
            "environment": environment, "knee_commit": agg.repo_commit(),
            "status": "ok" if len(done) == len(task_ids) else "partial",
            "all_units_ok": len(done) == len(task_ids),
            "units_finished": done, "units_missing": [uid for uid in task_ids if not units[uid]["ok"]],
            "leakage_assertion_passed": not bad_guard and not any("LeakageError" in e for e in errors),
            "test_fold_use": "none: no refit, no prediction, no unit folder; the test fold is only forbidden",
            "device_names": sorted({units[uid]["device_name"] for uid in done}), "score": SLICE_SCORE,
            "budget_h": budget_h, "seconds": seconds, "inner_scores": scores, "units": units}


def run_slices(arm: str, expect: dict | None = None, root=None, workers: int = 4,
               budget_h: float = SLICE_BUDGET_H, quiet: bool = True, folds: tuple[int, ...] | None = None,
               rerun: int = 0, encoders: tuple[str, ...] | None = None,
               chosen_ref: dict | None = n13.CHOSEN_REF) -> dict:
    """One arm of the slice-count check: s32, s16, s16b, s8 or s4 (see the module docstring). CPU only.

    Inputs attached: knee-n0, knee-n5, knee-n3-r and knee-n6a, as arm bc (from knee-n6a only its
    agg_n6a_<encoder>_f<k>.json files are read: the chosen configurations). expect: the card's hashes.
    budget_h is the soft stop: after budget_h hours no inner fit starts and a running fit ends with its
    epoch, the session ends cleanly with status "partial", and every finished unit is in the summary
    (agg_<step>_inner.json, rewritten as each unit ends). The watchdog of the step is at 1.1 x budget_h.
    Units are queued encoder by encoder (DINOv2 first). A worker crash is not retried.
    A rerun (rerun=1: step n13_<arm>_r) names the folds and encoders of the missing units.
    chosen_ref=None (tests) skips the check against the values read on 25 Nov; the summary says so."""
    _arm(arm)
    encs = tuple(MAIN if encoders is None else encoders)
    if not encs or not set(encs) <= set(MAIN):
        raise ValueError("the arms run the two main encoders (a rerun names the missing ones)")
    r = int(rerun)
    if r < 0 or (r and (not folds or encoders is None)):
        raise ValueError("a rerun names the folds and encoders of the missing units")
    if budget_h is None or float(budget_h) < 0:
        raise ValueError("budget_h is the soft stop in hours: a number, 0 or more")
    name = f"n13_{arm}" + ("" if not r else "_r" if r == 1 else f"_r{r}")
    env = n13.environment()
    cfg = {"name": name, "arm": arm, "rerun": r, "encoders": list(encs), "folds": list(folds) if folds else None,
           "workers": workers, "seed": SLICE_SEED, "budget_h": budget_h,
           "batch_size": "chosen", "config": slice_config(arm),
           "test_overrides": True, "test_overrides_reason": SLICE_REASON, "added_after_results": True,
           "plan": SLICE_PLAN, "chosen_ref_checked": chosen_ref is not None, "environment": env}
    out: dict = {}
    with common.notebook(name, cfg, quiet=quiet, budget_h=budget_h):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        import torch

        if torch.cuda.device_count():
            raise ValueError("the slice-count arms run on CPU (accelerator None): the five arms share one device type")
        t0 = time.time()
        d = stage.StageData(root, expect)
        K = d.n_folds("s2")
        fold_list = list(folds) if folds else list(range(K))
        if K != n13.N_OUTER or not set(fold_list) <= set(range(n13.N_OUTER)):
            raise ValueError("the slice-count check needs the 5 S2 outer folds")
        chosen = n13.chosen_configs(root, encs, fold_list, chosen_ref)
        od = paths.out_dir("private", name, "errors")
        tasks = [{"id": f"{e}_f{k}", "encoder": e, "fold": k, "seed": SLICE_SEED,
                  "eligible": list(range(len(FINDINGS))), "arm": arm,
                  "config_chosen": chosen[(e, k)]["config"], "stage_a_best": chosen[(e, k)]["best_inner_score"],
                  "chosen_sha16": chosen[(e, k)]["sha16"], "deadline": t0 + float(budget_h) * 3600,
                  "dir": str(od / f"{e}_f{k}")}
                 for e in encs for k in fold_list]  # encoder by encoder
        ids = [t["id"] for t in tasks]
        results: dict = {}

        def write(final: bool = False) -> dict:
            s = slice_summary(arm, name, results, ids, env, encs, budget_h=budget_h,
                              seconds=round(time.time() - t0, 1) if final else None,
                              ref_checked=chosen_ref is not None)
            s["upstream"] = d.upstream
            agg.write(f"{name}_inner", s)
            return s

        write()
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=n6._init,
                                 initargs=(counter, 0, str(root) if root else None, expect, None)) as ex:
            futs = {ex.submit(_slice_task, t): t for t in tasks}
            for f in as_completed(futs):
                t = futs[f]
                try:
                    res = f.result()
                except BrokenProcessPool:
                    res = {"task": t["id"], "ok": False, "error": "worker crashed"}
                except BaseException as e:  # noqa: BLE001
                    res = {"task": t["id"], "ok": False, "error": safe_log.describe(type(e), e.__traceback__)}
                results[t["id"]] = res
                write()  # the finished units are on disk before the next one ends
        out = write(final=True)
    common.summary(out)
    return out
