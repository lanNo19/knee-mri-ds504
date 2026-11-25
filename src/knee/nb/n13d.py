"""N13d: the dropout-ceiling check (it2d_dropout_r2.md), "added after the results" (R10).

The head search drew dropout from 0 to 0.5 and the ceiling was never extended; two units chose the
ceiling exactly. Question: does dropout above the searched ceiling (0.6, 0.7) score higher on inner
validation than each unit's chosen dropout?

Four arms, each its own CPU notebook (a fork of knee-n13-bc): dc, p5, p6, p7. Every arm fits, per unit
(main encoder x outer fold), the unit's chosen Stage A configuration at its chosen batch size on the
unit's inner folds, exactly as n13's batch-floor arm bc does (n13._batch_unit with the chosen batch size):
the same bank, inner sets, leakage guard, seed 0 + j, 30-epoch schedule, patience 5 and scoring. The only
change in an arm is the dropout of the head (heads.make_head: one nn.Dropout on the 12 per-finding
pooled vectors, all planes joined, before each finding's output weights; the slice features and the
attention branch get none; it acts in training only, the inner score is computed in eval mode):
  dc: the unit's chosen dropout, unchanged (the base of the rule; it repeats arm bc at this module's commit);
  p5: dropout 0.5 in every unit (the end of the searched range);  p6: dropout 0.6;  p7: dropout 0.7.
Each unit's chosen dropout is checked against the value read on 25 Nov (3 decimals, DROPOUT_REF), beside
n13's check of its batch size and best inner score.
The batch order comes from fit's own generator (seed + j), so it is the same in every arm; only the
dropout masks differ, by construction.
No arm refits on the outer training set, predicts a test fold or writes a unit folder. Only the inner
score is kept. The test fold and the gold studies reach each fit only as forbidden indices.

This module only adds: nothing in n13.py, n13s.py, n6.py or heads.py is changed, and the worker's cached
bank (n6._W) is used as it is and never modified.
Output of an arm: agg_n13_<arm>_inner.json (step n13_<arm>) and run.json.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import agg, heads, paths, safe_log, stage
from ..labels import FINDINGS
from . import common, n6, n13

MAIN = n13.MAIN
DROPOUT_PLAN = "it2d_dropout_r2.md: the dropout-ceiling check, declared before the run, added after the results"
DROPOUT_REASON = ("it2d dropout-ceiling check: the chosen Stage A configuration on the inner folds; "
                  "only the dropout of the head differs")
DROPOUT_ARMS = {"dc": None, "p5": 0.5, "p6": 0.6, "p7": 0.7}  # None: the unit's chosen dropout (the base)
# The chosen dropout per unit at 3 decimals, as read on 25 Nov (n9_r3_numbers.md, "The optimum found by the
# head search, per unit"): with n13.CHOSEN_REF, a check that the attached knee-n6a is that version.
DROPOUT_REF = {
    "dinov2_vits14": {0: 0.472, 1: 0.000, 2: 0.500, 3: 0.053, 4: 0.000},
    "convnext_tiny": {0: 0.472, 1: 0.500, 2: 0.386, 3: 0.000, 4: 0.222},
}
SEARCH_CEILING = 0.5  # the head search drew dropout from 0 to 0.5 (n6.suggest)
DROPOUT_SEED = 0  # arm bc's seed: inner fit j uses seed + j
DROPOUT_BUDGET_H = 3.0  # the soft stop of a session
DROPOUT_SCORE = ("inner score: mean over the unit's inner folds of the validation macro AUC at each inner fit's "
                 "best epoch (Stage A's objective), for the unit's chosen configuration at the arm's dropout")


def _arm(arm: str):
    if arm not in DROPOUT_ARMS:
        raise ValueError("unknown dropout-ceiling arm: use dc, p5, p6 or p7")
    return DROPOUT_ARMS[arm]


def dropout_config(arm: str) -> dict:
    """The dropout an arm's head uses (it sits under a "config" key in every summary: a setting)."""
    p = _arm(arm)
    return {"dropout": "chosen" if p is None else float(p)}


def dropout_settings(arm: str) -> dict:
    """The settings block of a dropout-ceiling arm: arm bc's settings plus the dropout under "config"."""
    base = n13.batch_settings("bc")
    return {**base, "config": {**base["config"], **dropout_config(arm)}}


def arm_config(chosen: heads.Config, arm: str) -> heads.Config:
    """The configuration an arm fits: the chosen one itself for dc, else the chosen one with the arm's dropout."""
    p = _arm(arm)
    return chosen if p is None else replace(chosen, dropout=float(p))


def check_dropout_ref(chosen: dict, ref: dict | None = DROPOUT_REF) -> None:
    """Each unit's chosen dropout at 3 decimals must be the one read on 25 Nov: ValueError otherwise (before any fit).
    chosen: n13.chosen_configs()'s {(encoder, fold): {"config", ...}}. ref=None skips the check."""
    if ref is None:
        return
    for (e, k), c in chosen.items():
        want = (ref.get(e) or {}).get(int(k))
        if want is None or round(float(c["config"]["dropout"]), 3) != round(float(want), 3):
            raise ValueError(f"chosen_configs: {e} fold {k} is not the Stage A unit read on 25 Nov (dropout differs)")


def _dropout_task(task: dict) -> dict:
    """Worker entry (spawned): one unit; a failure keeps its traceback under private/<step>/errors/."""
    return n6._guarded(_dropout_unit, task, Path(task["dir"]))


def _dropout_unit(task: dict, od: Path) -> dict:
    """The inner fits of one unit: n13._batch_unit at the chosen batch size, at the arm's dropout.

    Past the deadline no fit starts, and a running fit stops at the end of its epoch (heads.fit's
    on_epoch): the unit then returns stopped, with no score. Nothing is written to disk."""
    t0 = time.time()
    if t0 > task["deadline"]:
        return {"task": task["id"], "ok": False, "stopped": True}
    enc, k, seed, elig = task["encoder"], task["fold"], task["seed"], task["eligible"]
    arm = task["arm"]
    chosen = heads.config_from(task["config_chosen"])
    cfg = arm_config(chosen, arm)  # dc: the chosen configuration itself, its batch size included
    d, bank = n6._data(), n6._bank(enc)  # the worker's cached bank, as arm bc
    tr, te, _ = d.split(k)
    forbid = np.concatenate([te, d.gold_rows()])  # the test fold and every gold study: forbidden in each fit
    inner = d.inner(k)
    folds_in = sorted(set(inner[tr][inner[tr] >= 0].tolist()))
    stats = bank.stats(tr)
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
    if {key for key in used if used[key] != task["config_chosen"][key]} - {"dropout"}:
        raise AssertionError("a setting other than the dropout differs from the chosen configuration")
    want = float(chosen.dropout) if DROPOUT_ARMS[arm] is None else float(DROPOUT_ARMS[arm])
    if float(used["dropout"]) != want:
        raise AssertionError("the head did not use the arm's dropout")
    summary = {"encoder": enc, "fold": k, "inner_score": round(float(np.mean(sc)), n13.DECIMALS),
               "E": max(1, int(np.median(best_eps))), "epochs": ran, "inner_k": fits, "seed": seed,
               "schedule_epochs": heads.MAX_EPOCHS, "patience": heads.PATIENCE,
               "device": n6._W["device"], "device_name": n6._device_name(n6._W["device"]),
               "config": used, "batch_chosen": int(chosen.batch_size),
               "dropout": {"config": {"dropout_chosen": float(chosen.dropout), "dropout_used": float(cfg.dropout)}},
               "steps_per_epoch": n13._r5(np.mean(spe)), "steps_per_epoch_chosen": n13._r5(np.mean(spe_chosen)),
               "step_ratio": round(float(np.sum(spe) / max(1, np.sum(spe_chosen))), 2),
               "stage_a_best_inner_score": task["stage_a_best"], "chosen_sha16": task["chosen_sha16"],
               "seconds": round(time.time() - t0, 1)}
    return {"task": task["id"], "ok": True, "summary": summary, "fits": fits,
            "guard_checks": heads.guard_checks - checks0}


def dropout_summary(arm: str, step: str, results: dict, task_ids: list, environment: dict | None,
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
    return {"arm": arm, "step": step, "plan": DROPOUT_PLAN, "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": DROPOUT_REASON,
            "encoders": list(encoders), "settings": dropout_settings(arm),
            "chosen_ref_checked": bool(ref_checked),
            "environment": environment, "knee_commit": agg.repo_commit(),
            "status": "ok" if len(done) == len(task_ids) else "partial",
            "all_units_ok": len(done) == len(task_ids),
            "units_finished": done, "units_missing": [uid for uid in task_ids if not units[uid]["ok"]],
            "leakage_assertion_passed": not bad_guard and not any("LeakageError" in e for e in errors),
            "test_fold_use": "none: no refit, no prediction, no unit folder; the test fold is only forbidden",
            "device_names": sorted({units[uid]["device_name"] for uid in done}), "score": DROPOUT_SCORE,
            "budget_h": budget_h, "seconds": seconds, "inner_scores": scores, "units": units}


def run_dropout(arm: str, expect: dict | None = None, root=None, workers: int = 4,
                budget_h: float = DROPOUT_BUDGET_H, quiet: bool = True, folds: tuple[int, ...] | None = None,
                rerun: int = 0, encoders: tuple[str, ...] | None = None,
                chosen_ref: dict | None = n13.CHOSEN_REF) -> dict:
    """One arm of the dropout-ceiling check: dc, p5, p6 or p7 (see the module docstring). CPU only.

    Inputs attached: knee-n0, knee-n5, knee-n3-r and knee-n6a, as arm bc (from knee-n6a only its
    agg_n6a_<encoder>_f<k>.json files are read: the chosen configurations). expect: the card's hashes.
    budget_h is the soft stop: after budget_h hours no inner fit starts and a running fit ends with its
    epoch, the session ends cleanly with status "partial", and every finished unit is in the summary
    (agg_<step>_inner.json, rewritten as each unit ends). The watchdog of the step is at 1.1 x budget_h.
    Units are queued encoder by encoder (DINOv2 first). A worker crash is not retried.
    A rerun (rerun=1: step n13_<arm>_r) names the folds and encoders of the missing units.
    chosen_ref=None (tests) skips the check against the values read on 25 Nov (batch size, best inner score
    and dropout); the summary says so."""
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
           "workers": workers, "seed": DROPOUT_SEED, "budget_h": budget_h,
           "batch_size": "chosen", "config": dropout_config(arm),
           "test_overrides": True, "test_overrides_reason": DROPOUT_REASON, "added_after_results": True,
           "plan": DROPOUT_PLAN, "chosen_ref_checked": chosen_ref is not None, "environment": env}
    out: dict = {}
    with common.notebook(name, cfg, quiet=quiet, budget_h=budget_h):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        import torch

        if torch.cuda.device_count():
            raise ValueError("the dropout-ceiling arms run on CPU (accelerator None): the four arms share one device type")
        t0 = time.time()
        d = stage.StageData(root, expect)
        K = d.n_folds("s2")
        fold_list = list(folds) if folds else list(range(K))
        if K != n13.N_OUTER or not set(fold_list) <= set(range(n13.N_OUTER)):
            raise ValueError("the dropout-ceiling check needs the 5 S2 outer folds")
        chosen = n13.chosen_configs(root, encs, fold_list, chosen_ref)
        check_dropout_ref(chosen, None if chosen_ref is None else DROPOUT_REF)
        od = paths.out_dir("private", name, "errors")
        tasks = [{"id": f"{e}_f{k}", "encoder": e, "fold": k, "seed": DROPOUT_SEED,
                  "eligible": list(range(len(FINDINGS))), "arm": arm,
                  "config_chosen": chosen[(e, k)]["config"], "stage_a_best": chosen[(e, k)]["best_inner_score"],
                  "chosen_sha16": chosen[(e, k)]["sha16"], "deadline": t0 + float(budget_h) * 3600,
                  "dir": str(od / f"{e}_f{k}")}
                 for e in encs for k in fold_list]  # encoder by encoder
        ids = [t["id"] for t in tasks]
        results: dict = {}

        def write(final: bool = False) -> dict:
            s = dropout_summary(arm, name, results, ids, env, encs, budget_h=budget_h,
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
            futs = {ex.submit(_dropout_task, t): t for t in tasks}
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
