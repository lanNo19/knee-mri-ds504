"""N6a and N6b: Stage A, nested HPO on frozen embeddings (GPU T4 x2, 4 worker processes).

A unit is one encoder on one S2 outer fold (independent: one failure does not
stop the others; a failed unit is rerun in a new notebook that N9 also attaches).
Per unit, with only the outer training set (the test fold and every gold study
are passed to each fit as forbidden indices):
  L1 (grid): the 9 pooling x planes configs at the L2 defaults, each scored by
     the mean inner macro AUC over the inner folds (S2 macro findings).
  L2 (Bayesian optimisation) for the L1 winner: Optuna GPSampler, 30 trials,
     trial 0 = the defaults, 9 random, then 20 GP; each trial scored on all
     inner folds. Gated-attention hidden size is searched only for gated pooling.
  Refit: the best trial's config on the whole outer training set for
     E = the median best epoch of its inner fits (same schedule, stopped at E,
     seed 0), then predict the test fold and the gold studies in it.
N6b also runs the HPO demo (DINOv2, outer fold 0, gated attention over all
planes, inner fold 0 as validation): random vs random + successive halving vs
GP-BO at 900 epochs per arm, one repeat, plus fANOVA on the random arm.

Writes private/<name>/units/<encoder>_f<k>/ (l1.csv, trials.csv, predictions.csv,
head.pt, config.json, stats.pt), private/<name>/demo/ and, at the output root,
agg_<name>.json plus one agg_<name>_<encoder>_f<k>.json per unit (configs and
inner scores only).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, heads, paths, safe_log, stage
from ..labels import FINDINGS, UID
from . import common

N_TRIALS = 30
N_STARTUP = 10  # trial 0 (the defaults) + 9 random
# Simplified Stage A (plan: pilot projection above 6 h, or the backstop): this grid, BO on fold 0 only
SIMPLE_GRID = [("max", "sag"), ("max", "all"), ("gated", "sag"), ("gated", "all")]
DEMO_EPOCHS_PER_ARM = 900
_W: dict = {}  # worker state: device, data, banks


def space_defaults(U: float, gated: bool) -> dict:
    d = {"lr": 3e-4, "weight_decay": 3e-4, "dropout": 0.25, "pos_cap": float(np.sqrt(U)), "batch_size": 48}
    if gated:
        d["attn_hidden"] = 192
    return d


def suggest(trial, U: float, gated: bool) -> dict:
    """The plan's L2 space (log scale where marked)."""
    p = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "pos_cap": trial.suggest_float("pos_cap", 1.0, float(U), log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 128, log=True),
    }
    if gated:
        p["attn_hidden"] = trial.suggest_int("attn_hidden", 64, 512, log=True)
    return p


# ---------------------------------------------------------------- workers

def _init(counter, n_gpus: int, root, expect, shared=None):
    safe_log.worker_init()
    _W["shared"] = shared
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    _W["device"] = f"cuda:{wid % n_gpus}" if n_gpus else "cpu"
    _W["root"], _W["expect"] = root, expect
    try:  # several workers share few CPU cores: one thread each avoids oversubscription
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:  # noqa: BLE001
        pass


def _data():
    if "data" not in _W:
        _W["data"] = stage.StageData(_W.get("root"), _W.get("expect"))
    return _W["data"]


def _bank(enc: str):
    if _W.get("bank_enc") != enc:
        _W.pop("bank", None)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        _W["bank"] = _data().bank(enc, _W["device"])
        _W["bank_enc"] = enc
    return _W["bank"]


def _guarded(fn, task, out_dir: Path):
    """Run one task; on failure keep the full traceback privately and return only the safe line."""
    try:
        return fn(task, out_dir)
    except BaseException as e:  # noqa: BLE001
        _signal_fold0(task, "failed")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.log").write_text("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        except Exception:  # noqa: BLE001
            pass
        return {"task": task["id"], "ok": False, "error": safe_log.describe(type(e), e.__traceback__)}


def _unit_task(task: dict) -> dict:
    return _guarded(_unit, task, Path(task["dir"]))


def _demo_task(task: dict) -> dict:
    return _guarded(_demo, task, Path(task["dir"]))


# ---------------------------------------------------------------- one unit

def _fold_mode(task: dict) -> str:
    """The tuning mode of a fold, decided once and shared by both encoders' units of that fold.

    "simplified" when the run is simplified (pilot decision) or when the unit is projected to
    finish after the backstop deadline; "skipped" past the soft stop (0.75 x budget); else "nested".
    """
    shared = _W.get("shared")
    k = task["fold"]
    if shared is None:  # tests without a manager
        return task.get("mode", "nested")
    with shared["lock"]:
        modes = shared["modes"]
        if k not in modes:
            now = time.time()
            done = list(shared["durations"])
            est = float(np.median(done)) if done else (task.get("unit_s_est") or None)
            if task.get("mode") == "simplified":
                modes[k] = "simplified"
            elif now > task["soft_stop"]:
                modes[k] = "skipped"
            elif task.get("backstop") and est is not None and now + est > task["backstop"]:
                modes[k] = "simplified"
            else:
                modes[k] = "nested"
        return modes[k]


def _signal_fold0(task: dict, state: str) -> None:
    """Fold 0 tells the other folds' units that its L2 settings are ready ("done") or will not come."""
    shared = _W.get("shared")
    if shared is not None and task.get("fold") == 0 and "encoder" in task:
        shared["fold0"][task["encoder"]] = state


def _fold0_l2(task: dict, enc: str) -> tuple[dict | None, str]:
    """L2 settings found on fold 0 for this encoder (simplified tuning reuses them for every fold).

    Waits until fold 0's unit has finished or failed, bounded by the soft stop (0.75 x budget)."""
    p = Path(task["dir"]).parent / f"{enc}_f0" / "config.json"
    if not task.get("fold0_in_run", True):  # e.g. a rerun of one fold: use the attached original fold 0
        hits = paths.glob_outputs(f"private/*/units/{enc}_f0/config.json", Path(_W["root"]) if _W.get("root") else None)
        p = hits[0] if hits else p
    shared = _W.get("shared")
    while task.get("fold0_in_run", True) and not p.exists() and time.time() < task["soft_stop"]:
        if shared is not None and shared["fold0"].get(enc) == "failed":
            break
        if shared is None and not Path(task["dir"]).parent.joinpath(f"{enc}_f0").exists():
            break  # tests without a manager: nothing to wait for
        time.sleep(15)
    if not p.exists():
        return None, "defaults"
    c = json.loads(p.read_text())["config"]
    return {k: c[k] for k in ("lr", "weight_decay", "dropout", "attn_hidden", "pos_cap", "batch_size")}, "fold 0"


def _unit(task: dict, od: Path) -> dict:
    import optuna
    import torch

    t0 = time.time()
    mode = _fold_mode(task)
    if mode == "skipped":
        return {"task": task["id"], "ok": False, "skipped": True, "mode": mode}
    enc, k, seed, elig = task["encoder"], task["fold"], task["seed"], task["eligible"]
    d, bank = _data(), _bank(enc)
    od.mkdir(parents=True, exist_ok=True)
    tr, te, _ = d.split(k)
    gold_all, gold_k = d.gold_rows(), d.gold_rows(k)
    forbid = np.concatenate([te, gold_all])
    inner = d.inner(k)
    folds_in = sorted(set(inner[tr][inner[tr] >= 0].tolist()))
    stats = bank.stats(tr)
    U = d.pos_cap_upper()
    n_fits = 0

    def score(cfg):
        nonlocal n_fits
        sc, eps = [], []
        for j in folds_in:
            itr = tr[inner[tr] != j]
            iva = tr[(inner[tr] == j) & d.scored[tr]]
            r = heads.fit(bank, cfg, itr, iva, elig, stats=stats, forbidden=forbid, seed=seed + j)
            n_fits += 1
            sc.append(r.best_score)
            eps.append(r.best_epoch)
        sc = [x for x in sc if np.isfinite(x)]
        return (float(np.mean(sc)) if sc else float("nan")), eps

    # L1 grid at the L2 defaults (the simplified grid: max or gated, sagittal or all three planes)
    base = heads.default_config(U)
    grid = task.get("l1_grid") or (SIMPLE_GRID if mode == "simplified" else heads.L1_GRID)
    l1 = []
    for pooling, planes in grid:
        m, eps = score(heads.with_l1(base, pooling, planes))
        l1.append({"pooling": pooling, "planes": planes, "inner_score": m, "median_best_epoch": float(np.median(eps))})
    l1df = pd.DataFrame(l1)
    l1df.to_csv(od / "l1.csv", index=False)
    win = l1df.loc[l1df["inner_score"].fillna(-1).idxmax()]
    win_cfg = heads.with_l1(base, str(win["pooling"]), str(win["planes"]))
    gated = win_cfg.pooling == "gated"

    l2_from = "bayesian optimisation on this fold"
    study = None
    if mode == "simplified" and k != 0:
        # simplified: Bayesian optimisation ran on fold 0 only; its L2 settings are reused
        l2, l2_from = _fold0_l2(task, enc)
        cfg_best = replace(win_cfg, **l2) if l2 else win_cfg
        best_score, eps = score(cfg_best)
        E = max(1, int(np.median(eps)))
        pd.DataFrame([{"number": 0, "state": "REUSED", "value": best_score, **(l2 or {}),
                       "best_epochs": json.dumps([int(e) for e in eps])}]).to_csv(od / "trials.csv", index=False)
        default_score = None
    else:
        # L2 Bayesian optimisation for the L1 winner
        sampler = optuna.samplers.GPSampler(seed=seed + 1000 * k, n_startup_trials=N_STARTUP)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.enqueue_trial(space_defaults(U, gated))

        def objective(trial):
            cfg = replace(win_cfg, **suggest(trial, U, gated))
            m, eps = score(cfg)
            trial.set_user_attr("best_epochs", [int(e) for e in eps])
            return m

        study.optimize(objective, n_trials=int(task.get("n_trials") or N_TRIALS), catch=(ValueError,))
        rows = []
        for t in study.trials:
            rows.append({"number": t.number, "state": t.state.name, "value": t.value, **t.params,
                         "best_epochs": json.dumps(t.user_attrs.get("best_epochs", []))})
        pd.DataFrame(rows).to_csv(od / "trials.csv", index=False)
        best = study.best_trial
        cfg_best = replace(win_cfg, **best.params)
        best_score = float(best.value)
        E = max(1, int(np.median(best.user_attrs["best_epochs"])))
        default_score = round(float(study.trials[0].value), 4) if study.trials[0].value is not None else None

    # Refit on the outer training set, then the test fold and its gold studies
    r = heads.fit(bank, cfg_best, tr, epochs=E, stats=stats, forbidden=forbid, seed=0)
    rows_out = np.concatenate([te, gold_k])
    P = heads.predict(bank, r.model, rows_out, cfg_best, stats)
    pred = pd.DataFrame(P, columns=FINDINGS)
    pred.insert(0, "is_gold", d.is_gold[rows_out])
    pred.insert(0, "s2_fold", k)
    pred.insert(0, UID, d.uids[rows_out])
    pred.to_csv(od / "predictions.csv", index=False)
    torch.save(r.model.state_dict(), od / "head.pt")
    torch.save({p: (mu.cpu(), sd.cpu()) for p, (mu, sd) in stats.items()}, od / "stats.pt")
    summary = {"encoder": enc, "fold": k, "tuning": "nested" if mode == "nested" else "simplified (not nested)",
               "l2_from": l2_from,
               "l1_winner": {"pooling": win_cfg.pooling, "planes": win_cfg.planes},
               "l1_scores": {f"{a}|{b}": (round(v, 4) if np.isfinite(v) else None)
                             for a, b, v in zip(l1df["pooling"], l1df["planes"], l1df["inner_score"])},
               "config": cfg_best.to_dict(), "best_inner_score": round(best_score, 4), "E": E,
               # optimiser steps of the refit (KneeMRI S5 uses the same number of steps)
               "refit_steps": int(E * int(np.ceil(len(tr) / max(1, int(cfg_best.batch_size))))),
               "default_trial_score": default_score,
               "inner_k": len(folds_in), "seconds": round(time.time() - t0, 1),
               "predictions_sha16": common.sha16(od / "predictions.csv")}
    (od / "config.json").write_text(json.dumps(summary, indent=1))
    _signal_fold0(task, "done")
    shared = _W.get("shared")
    if shared is not None and mode == "nested":
        shared["durations"].append(time.time() - t0)
    return {"task": task["id"], "ok": True, "summary": summary, "fits": n_fits, "mode": mode}


# ---------------------------------------------------------------- the HPO demo

def _demo(task: dict, od: Path) -> dict:
    import optuna
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    t0 = time.time()
    enc, seed, elig = task["encoder"], task["seed"], task["eligible"]
    d, bank = _data(), _bank(enc)
    od.mkdir(parents=True, exist_ok=True)
    k = 0
    tr, te, _ = d.split(k)
    forbid = np.concatenate([te, d.gold_rows()])
    inner = d.inner(k)
    itr, iva = tr[inner[tr] != 0], tr[(inner[tr] == 0) & d.scored[tr]]
    stats = bank.stats(tr)
    U = d.pos_cap_upper()
    base = heads.with_l1(heads.default_config(U), "gated", "all")
    log: list[dict] = []

    def train_trial(trial, arm, prune: bool):
        cfg = replace(base, **suggest(trial, U, True))
        best = -np.inf
        res = heads.fit(bank, cfg, itr, iva, elig, stats=stats, forbidden=forbid, seed=seed, patience=None,
                        max_epochs=heads.MAX_EPOCHS, on_epoch=(lambda ep, sc: _report(trial, ep, sc, prune)))
        for ep, sc in enumerate(res.history, start=1):
            if sc is not None and sc > best:
                best = sc
            log.append({"arm": arm, "trial": trial.number, "epoch": ep, "score": sc})
        if res.pruned:
            raise optuna.TrialPruned()
        return best

    arms = {}
    for arm in ("random", "successive_halving", "gp"):
        if arm == "gp":
            sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=N_STARTUP)
            pruner = optuna.pruners.NopPruner()
        else:
            sampler = optuna.samplers.RandomSampler(seed=seed)
            pruner = (optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3)
                      if arm == "successive_halving" else optuna.pruners.NopPruner())
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        prune = arm == "successive_halving"
        budget = int(task.get("demo_epochs") or DEMO_EPOCHS_PER_ARM)
        max_trials = budget // heads.MAX_EPOCHS if arm != "successive_halving" else budget // 2 + 10
        while sum(1 for x in log if x["arm"] == arm) < budget and len(study.trials) < max_trials:
            study.optimize(lambda t: train_trial(t, arm, prune), n_trials=1, catch=(ValueError,))
        arms[arm] = study
    lg = pd.DataFrame(log)
    lg.to_csv(od / "demo_epochs.csv", index=False)
    curves = {}
    for arm, g in lg.groupby("arm"):
        bsf = g["score"].astype(float).cummax().to_numpy()
        marks = [100, 300, 600, 900]
        curves[arm] = {f"best_at_{m}_epochs": (round(float(bsf[min(m, len(bsf)) - 1]), 4) if len(bsf) else None)
                       for m in marks}
        curves[arm]["best_so_far_every_30_epochs"] = [round(float(x), 4) if np.isfinite(x) else None for x in bsf[29::30]]  # for the slide
        curves[arm]["epochs_spent"] = agg.count(len(g))
        curves[arm]["trials"] = agg.count(len(arms[arm].trials))
    imp = {}
    try:
        done = [t for t in arms["random"].trials if t.state.name == "COMPLETE"]
        if len(done) >= 5:
            imp = {kk: round(float(v), 3) for kk, v in get_param_importances(
                arms["random"], evaluator=FanovaImportanceEvaluator(seed=seed)).items()}
    except Exception:  # noqa: BLE001 - importance is descriptive
        imp = {}
    summary = {"encoder": enc, "fold": k, "curves": curves, "fanova_random_arm": imp,
               "seconds": round(time.time() - t0, 1)}
    (od / "demo.json").write_text(json.dumps(summary, indent=1))
    return {"task": task["id"], "ok": True, "summary": summary}


def _report(trial, epoch: int, score, prune: bool) -> bool:
    """Report an epoch's validation score; True asks the fit to stop (pruned)."""
    if not prune or score is None:
        return False
    trial.report(float(score), epoch)
    return bool(trial.should_prune())


# ---------------------------------------------------------------- the notebook

def _deadline_epoch(deadline_utc: str | None) -> float | None:
    if not deadline_utc:
        return None
    from datetime import datetime, timezone

    return datetime.strptime(deadline_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def run(name: str = "n6a", encoders: tuple[str, ...] = ("dinov2_vits14", "convnext_tiny"),
        folds: tuple[int, ...] | None = None, demo: bool = False, root=None, expect: dict | None = None,
        workers: int = 4, seed: int = 0, budget_h: float = 8.0, quiet: bool = True,
        mode: str = "nested", backstop_utc: str | None = None, unit_hours_est: float | None = None,
        n_trials: int | None = None, l1_grid: list | None = None, demo_epochs: int | None = None) -> dict:
    """name: n6a (DINOv2 and ConvNeXt-T) or n6b (ViT-S/16 and the demo); folds=(0,) for the one-unit smoke.
    mode: "nested" (the plan) or "simplified" (the pilot projected N6a above 6 h; labelled "not nested").
    backstop_utc: the backstop as UTC, e.g. "2026-11-24T03:00:00Z" (Tue 08:00 Almaty): a fold whose units
    are projected to finish after it runs simplified, for both encoders together. unit_hours_est: the
    pilot's unit projection, used until a unit has finished. Units are queued fold by fold, and a fold
    not started by 0.75 x budget_h is skipped for every encoder (never one encoder alone).
    n_trials, l1_grid and demo_epochs exist for tests only (the plan fixes 30, the 9-config grid and 900)."""
    s: dict = {}
    cfg = {"name": name, "encoders": list(encoders), "folds": list(folds) if folds else None, "demo": demo,
           "workers": workers, "seed": seed, "budget_h": budget_h, "mode": mode, "backstop_utc": backstop_utc,
           "unit_hours_est": unit_hours_est, "test_overrides": bool(n_trials or l1_grid or demo_epochs)}
    with common.notebook(name, cfg, quiet=quiet, budget_h=budget_h):
        import torch

        t0 = time.time()
        d = stage.StageData(root, expect)
        K = d.n_folds("s2")
        elig = list(range(len(FINDINGS))) if K >= 2 else []  # the S2 macro is over all 12 findings
        s["upstream"] = d.upstream
        s["findings"] = [FINDINGS[j] for j in elig]
        s["n_folds"] = K
        if K < 2 or not elig:
            s["status"] = "not_testable"
        else:
            od = paths.out_dir("private", name)
            fold_list = list(folds) if folds is not None else list(range(K))
            base_task = {"seed": seed, "eligible": elig, "soft_stop": t0 + 0.75 * budget_h * 3600,
                         "backstop": _deadline_epoch(backstop_utc), "mode": mode,
                         "unit_s_est": unit_hours_est * 3600 if unit_hours_est else None,
                         "n_trials": n_trials, "l1_grid": l1_grid}
            # fold by fold, so both encoders of a fold start (and are switched or skipped) together
            tasks = [{**base_task, "id": f"{e}_f{k}", "encoder": e, "fold": k, "dir": str(od / "units" / f"{e}_f{k}"),
                      "fold0_in_run": 0 in fold_list}
                     for k in fold_list for e in encoders]
            n_gpus = torch.cuda.device_count()
            ctx = mp.get_context("spawn")
            counter = ctx.Value("i", 0)
            mgr = ctx.Manager()
            shared = {"lock": mgr.Lock(), "modes": mgr.dict(), "durations": mgr.list(), "fold0": mgr.dict()}
            units, fails = {}, {}
            demo_task = ({"id": "demo", "encoder": "dinov2_vits14", "seed": seed, "eligible": elig,
                          "dir": str(od / "demo"), "demo_epochs": demo_epochs} if demo else None)
            pending = list(tasks)
            try:
                for attempt in (1, 2):
                    # a hard worker crash breaks the whole pool: the tasks it took down run once more
                    # in a fresh pool (finished units keep their outputs)
                    crashed = []
                    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=_init,
                                             initargs=(counter, n_gpus, str(root) if root else None, expect,
                                                       shared)) as ex:
                        futs = {ex.submit(_unit_task, t): t for t in pending}
                        if demo_task is not None:
                            futs[ex.submit(_demo_task, demo_task)] = demo_task
                        for f in as_completed(futs):
                            t = futs[f]
                            try:
                                r = f.result()
                            except BrokenProcessPool:
                                crashed.append(t)
                                continue
                            except BaseException as e:  # noqa: BLE001
                                r = {"task": t["id"], "ok": False, "error": safe_log.describe(type(e), e.__traceback__)}
                            if r["task"] == "demo":
                                s["demo"] = r.get("summary") or {"ok": False, "error": r.get("error")}
                                demo_task = None
                                continue
                            if r.get("ok"):
                                u = r["summary"]
                                agg.write(f"{name}_{r['task']}", u)  # written as each unit completes
                                units[r["task"]] = {"ok": True, "tuning": u["tuning"], "l1_winner": u["l1_winner"],
                                                    "E": u["E"], "best_inner_score": u["best_inner_score"],
                                                    "seconds": u["seconds"]}
                            else:
                                units[r["task"]] = {"ok": False, "skipped": bool(r.get("skipped")), "error": r.get("error")}
                                fails[r["task"]] = r.get("error")
                    crashed_ids = {t["id"] for t in crashed}
                    pending = [t for t in tasks if t["id"] in crashed_ids]  # original (fold-major) order
                    if not crashed:
                        break
                for t in pending:  # crashed twice
                    units[t["id"]] = {"ok": False, "skipped": False, "error": "worker crashed twice"}
                if demo_task is not None and "demo" not in s:
                    s["demo"] = {"ok": False, "error": "worker crashed twice"}
                s["fold_modes"] = {f"fold_{k}": v for k, v in dict(shared["modes"]).items()}
            finally:
                mgr.shutdown()
            s["units"] = units
            s["units_ok"] = agg.count(sum(1 for v in units.values() if v["ok"]))
            s["all_units_ok"] = len(units) == len(tasks) and all(v["ok"] for v in units.values())
            s["all_nested"] = all(v.get("tuning") == "nested" for v in units.values() if v["ok"])
            s["leakage_assertion_passed"] = not any("LeakageError" in str(v) for v in fails.values())
        s["seconds"] = round(time.time() - t0, 1)
        agg.write(name, s)
    common.summary({k: s.get(k) for k in ("all_units_ok", "all_nested", "fold_modes", "units", "demo",
                                          "leakage_assertion_passed", "status") if s.get(k) is not None})
    return s


if __name__ == "__main__":  # pragma: no cover
    run(os.environ.get("KNEE_N6_NAME", "n6a"))
