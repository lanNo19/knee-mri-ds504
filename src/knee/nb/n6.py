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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, heads, paths, safe_log, stage
from ..labels import FINDINGS, UID
from . import common

N_TRIALS = 30
N_STARTUP = 10  # trial 0 (the defaults) + 9 random
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

def _init(counter, n_gpus: int, root, expect):
    safe_log.worker_init()
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

def _unit(task: dict, od: Path) -> dict:
    import optuna
    import torch

    t0 = time.time()
    if time.time() > task["deadline"]:
        return {"task": task["id"], "ok": False, "skipped": True}
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

    # L1 grid at the L2 defaults
    base = heads.default_config(U)
    l1 = []
    for pooling, planes in task.get("l1_grid") or heads.L1_GRID:
        m, eps = score(heads.with_l1(base, pooling, planes))
        l1.append({"pooling": pooling, "planes": planes, "inner_score": m, "median_best_epoch": float(np.median(eps))})
    l1df = pd.DataFrame(l1)
    l1df.to_csv(od / "l1.csv", index=False)
    win = l1df.loc[l1df["inner_score"].fillna(-1).idxmax()]
    win_cfg = heads.with_l1(base, str(win["pooling"]), str(win["planes"]))
    gated = win_cfg.pooling == "gated"

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
    E = max(1, int(np.median(best.user_attrs["best_epochs"])))

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
    summary = {"encoder": enc, "fold": k, "l1_winner": {"pooling": win_cfg.pooling, "planes": win_cfg.planes},
               "l1_scores": {f"{a}|{b}": (round(v, 4) if np.isfinite(v) else None)
                             for a, b, v in zip(l1df["pooling"], l1df["planes"], l1df["inner_score"])},
               "config": cfg_best.to_dict(), "best_inner_score": round(float(best.value), 4), "E": E,
               "default_trial_score": round(float(study.trials[0].value), 4) if study.trials[0].value is not None else None,
               "inner_k": len(folds_in), "seconds": round(time.time() - t0, 1),
               "predictions_sha16": common.sha16(od / "predictions.csv")}
    (od / "config.json").write_text(json.dumps(summary, indent=1))
    return {"task": task["id"], "ok": True, "summary": summary, "fits": n_fits}


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

def run(name: str = "n6a", encoders: tuple[str, ...] = ("dinov2_vits14", "convnext_tiny"),
        folds: tuple[int, ...] | None = None, demo: bool = False, root=None, expect: dict | None = None,
        workers: int = 4, seed: int = 0, budget_h: float = 8.0, quiet: bool = True,
        n_trials: int | None = None, l1_grid: list | None = None, demo_epochs: int | None = None) -> dict:
    """name: n6a (DINOv2 and ConvNeXt-T) or n6b (ViT-S/16 and the demo); folds=(0,) for the one-unit smoke.
    n_trials, l1_grid and demo_epochs exist for tests only (the plan fixes 30, the 9-config grid and 900)."""
    s: dict = {}
    cfg = {"name": name, "encoders": list(encoders), "folds": list(folds) if folds else None, "demo": demo,
           "workers": workers, "seed": seed, "budget_h": budget_h, "test_overrides": bool(n_trials or l1_grid or demo_epochs)}
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
            deadline = t0 + 0.75 * budget_h * 3600
            tasks = [{"id": f"{e}_f{k}", "encoder": e, "fold": k, "seed": seed, "eligible": elig,
                      "deadline": deadline, "dir": str(od / "units" / f"{e}_f{k}"),
                      "n_trials": n_trials, "l1_grid": l1_grid}
                     for e in encoders for k in (folds if folds is not None else range(K))]
            n_gpus = torch.cuda.device_count()
            ctx = mp.get_context("spawn")
            counter = ctx.Value("i", 0)
            results = []
            with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=_init,
                                     initargs=(counter, n_gpus, str(root) if root else None, expect)) as ex:
                futs = [ex.submit(_unit_task, t) for t in tasks]
                if demo:
                    futs.append(ex.submit(_demo_task, {"id": "demo", "encoder": "dinov2_vits14", "seed": seed,
                                                       "eligible": elig, "dir": str(od / "demo"),
                                                       "demo_epochs": demo_epochs}))
                for f in as_completed(futs):
                    results.append(f.result())
            units = {}
            for r in results:
                if r["task"] == "demo":
                    s["demo"] = r.get("summary") or {"ok": False, "error": r.get("error")}
                    continue
                if r.get("ok"):
                    u = r["summary"]
                    agg.write(f"{name}_{r['task']}", u)
                    units[r["task"]] = {"ok": True, "l1_winner": u["l1_winner"], "E": u["E"],
                                        "best_inner_score": u["best_inner_score"], "seconds": u["seconds"]}
                else:
                    units[r["task"]] = {"ok": False, "skipped": bool(r.get("skipped")), "error": r.get("error")}
            s["units"] = units
            s["units_ok"] = agg.count(sum(1 for v in units.values() if v["ok"]))
            s["all_units_ok"] = all(v["ok"] for v in units.values())
            s["leakage_assertion_passed"] = not any("LeakageError" in str(v.get("error")) for v in units.values())
        s["seconds"] = round(time.time() - t0, 1)
        agg.write(name, s)
    common.summary({k: s.get(k) for k in ("all_units_ok", "units", "demo", "leakage_assertion_passed", "status")
                    if s.get(k) is not None})
    return s


if __name__ == "__main__":  # pragma: no cover
    run(os.environ.get("KNEE_N6_NAME", "n6a"))
