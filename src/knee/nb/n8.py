"""N8: LightGBM floor A on acquisition tags only (H3), CPU.

Features (N0's private tags.csv): manufacturer and model as categories (a
missing value is its own category), field strength, and for each plane the
repetition time, echo time, slice thickness and pixel spacing (numeric;
LightGBM sends missing values down their own branch). No demographic tags.
Labels: the binary labels; uncertain cells are left out of training for that
finding (plan). Scoring, early stopping and selection use the plan's scoring
rule (uncertain counts as negative). The S2 macro is over all 12 findings.
Studies are those N5 can embed (all planes decoded), so floor A is scored on
the same studies as the image models.

Nested, per S2 outer fold: 40 random-search trials, each scored by the mean
over the inner folds of the inner validation macro AUC (the S2 macro
findings, fixed from labels and folds exactly as in the pilot); one model
per finding per inner fold, at most 2,000 trees, early stop after 50 rounds
on the inner validation fold. The best trial is refit on the outer training
set with each finding's median best number of trees, and predicts the test
fold and the gold studies in it. No test study enters a fit, a stopping rule
or a selection (asserted).

Writes private/n8/predictions.csv and trials.csv, and agg_n8.json.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import agg, heads, metrics, paths, stage
from ..labels import FINDINGS, UID
from . import common

N_TRIALS = 40
MAX_TREES = 2000
EARLY_STOP = 50
NUMERIC = [f"{p}_{t}" for p in heads.PLANES
           for t in ("repetition_time", "echo_time", "slice_thickness", "pixel_spacing")] + ["field_strength"]
CATEGORICAL = ["manufacturer", "model"]


def sample_params(rng) -> dict:
    """The plan's LightGBM space (log scale where marked)."""
    return {
        "num_leaves": int(round(np.exp(rng.uniform(np.log(8), np.log(256))))),
        "learning_rate": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.3)))),
        "min_child_samples": int(round(np.exp(rng.uniform(np.log(5), np.log(100))))),
        "feature_fraction": float(rng.uniform(0.5, 1.0)),
        "lambda_l2": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
    }


def features(tags: pd.DataFrame, uids) -> pd.DataFrame:
    t = tags.copy()
    t[UID] = t[UID].astype(str)
    t = t.set_index(UID).reindex(uids)
    X = pd.DataFrame(index=t.index)
    for c in CATEGORICAL:
        v = t[c].astype("string").fillna("MISSING") if c in t else pd.Series("MISSING", index=t.index)
        X[c] = pd.Categorical(v)
    for c in NUMERIC:
        X[c] = pd.to_numeric(t[c], errors="coerce") if c in t else np.nan
    return X.reset_index(drop=True)


def _fit_one(X, y, tr, va, params, rounds=None, seed=0, y_eval=None):
    """y: training labels (NaN = uncertain, left out); y_eval: validation labels (scoring rule)."""
    import lightgbm as lgb

    p = {"objective": "binary", "verbosity": -1, "num_threads": 1, "seed": seed, "deterministic": True,
         "force_row_wise": True, **params}
    m_tr = tr[~np.isnan(y[tr])]
    dtr = lgb.Dataset(X.iloc[m_tr], label=y[m_tr], free_raw_data=False)
    if rounds is not None:
        return lgb.train(p, dtr, num_boost_round=int(rounds)), int(rounds)
    ye = y if y_eval is None else y_eval
    m_va = va[~np.isnan(ye[va])]
    yv = ye[m_va]
    if len(m_va) == 0 or yv.min() == yv.max():
        return lgb.train(p, dtr, num_boost_round=200), 200
    dva = lgb.Dataset(X.iloc[m_va], label=yv, reference=dtr)
    b = lgb.train({**p, "metric": "auc"}, dtr, num_boost_round=MAX_TREES, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return b, int(b.best_iteration or MAX_TREES)


def run(root=None, expect: dict | None = None, n_trials: int = N_TRIALS, workers: int = 4, seed: int = 0,
        budget_h: float = 2.0, quiet: bool = True) -> dict:
    s: dict = {}
    cfg = {"n_trials": n_trials, "workers": workers, "seed": seed, "budget_h": budget_h}
    with common.notebook("n8", cfg, quiet=quiet, budget_h=budget_h):
        t0 = time.time()
        d = stage.StageData(root, expect, need_embeddings=False)
        od = paths.out_dir("private", "n8")
        tags = pd.read_csv(d.tags_path)
        X = features(tags, d.uids)
        Y = d.binary  # training labels (uncertain left out)
        YE = d.eval_y  # scoring labels (uncertain = negative)
        f2 = d.folds["s2_fold"]
        K = d.n_folds("s2")
        tr_all = np.flatnonzero(d.train & (f2 >= 0))
        ev = tr_all[d.scored[tr_all]]
        elig = list(range(len(FINDINGS))) if K >= 2 else []
        s["upstream"] = d.upstream
        s["usable_from_n5"] = d.usable_from_n5
        s["findings"] = [FINDINGS[j] for j in elig]
        s["features"] = {"categorical": CATEGORICAL, "numeric": NUMERIC,
                         "missing_share": {c: agg.share(X[c].isna().mean()) for c in NUMERIC}}
        P = np.full((d.n, len(FINDINGS)), np.nan, np.float32)
        trials, chosen = [], {}
        guard_ok = True
        for k in range(K if elig else 0):
            tr, te, _ = d.split(k)
            forbidden = set(te.tolist()) | set(d.gold_rows().tolist())
            inner = d.inner(k)
            folds_in = sorted(set(inner[tr][inner[tr] >= 0].tolist()))
            rng = np.random.default_rng(seed + k)
            best = None
            for t in range(n_trials):
                params = sample_params(rng)
                scores, iters = [], {j: [] for j in elig}
                for j_in in folds_in:
                    itr, iva = tr[inner[tr] != j_in], tr[(inner[tr] == j_in) & d.scored[tr]]
                    if forbidden & (set(itr.tolist()) | set(iva.tolist())):
                        guard_ok = False
                        raise AssertionError("a test study reached floor A's tuning")
                    with ThreadPoolExecutor(workers) as ex:
                        res = list(ex.map(lambda j: _fit_one(X, Y[:, j], itr, iva, params, seed=seed,
                                                             y_eval=YE[:, j]), elig))
                    pv = np.full((len(iva), len(FINDINGS)), np.nan)
                    for j, (b, it) in zip(elig, res):
                        pv[:, j] = b.predict(X.iloc[iva], num_iteration=it)
                        iters[j].append(it)
                    scores.append(metrics.macro_available(pv, YE[iva], elig))
                score = float(np.nanmean(scores)) if scores else float("nan")
                trials.append({"fold": k, "trial": t, **params, "inner_score": round(score, 5)})
                if np.isfinite(score) and (best is None or score > best[0]):
                    best = (score, params, {j: int(np.median(v)) for j, v in iters.items()})
            if best is None:
                continue
            score, params, rounds = best
            chosen[f"fold_{k}"] = {**{a: (round(v, 5) if isinstance(v, float) else v) for a, v in params.items()},
                                   "inner_score": round(score, 4)}
            rows = np.concatenate([te, d.gold_rows(k)])
            if forbidden & set(tr.tolist()):
                guard_ok = False
                raise AssertionError("a test study reached floor A's refit")
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(lambda j: _fit_one(X, Y[:, j], tr, None, params, rounds=max(1, rounds[j]), seed=seed),
                                  elig))
            for j, (b, _) in zip(elig, res):
                P[rows, j] = b.predict(X.iloc[rows])
        pd.DataFrame(trials).to_csv(od / "trials.csv", index=False)
        out = pd.DataFrame(P, columns=FINDINGS)
        out.insert(0, "s2_fold", f2)
        out.insert(0, UID, d.uids)
        out = out[np.isfinite(P).any(axis=1)]
        out.to_csv(od / "predictions.csv", index=False)
        if elig and K >= 2:
            s["s2_macro_auc"] = round(metrics.macro(P[ev], YE[ev], f2[ev], elig), 4)
            s["mean_of_fold_macros"] = round(metrics.per_fold_macro_mean(P[ev], YE[ev], f2[ev], elig), 4)
        else:
            # SMOKE-ALL's 20 studies: still exercise the features and one fit on the real tags
            s["status"] = "not_testable"
            rows = np.flatnonzero(d.train)
            j = next((j for j in range(len(FINDINGS)) if len(set(Y[rows, j][~np.isnan(Y[rows, j])])) == 2), None)
            if j is not None:
                b, _ = _fit_one(X, Y[:, j], rows, None, sample_params(np.random.default_rng(seed)), rounds=5, seed=seed)
                s["smoke"] = {"fit_ok": bool(np.isfinite(b.predict(X.iloc[rows])).all())}
        s["chosen"] = chosen
        s["n_folds"] = K
        s["leakage_assertion_passed"] = guard_ok
        s["predictions_sha16"] = common.sha16(od / "predictions.csv")
        s["seconds"] = round(time.time() - t0, 1)
        agg.write("n8", s)
    common.summary({k: s.get(k) for k in ("s2_macro_auc", "mean_of_fold_macros", "n_folds", "leakage_assertion_passed",
                                          "predictions_sha16", "status") if s.get(k) is not None})
    return s
