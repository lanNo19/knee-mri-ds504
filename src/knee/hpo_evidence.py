"""Descriptive evidence about Stage A's hyperparameter search, from the unit logs only.

Nothing here trains, selects or scores on a test fold: every number is an INNER validation score
that Stage A already logged (l1.csv, trials.csv, config.json of each unit). The outputs are
aggregates over trials and units, so they hold no per-study value.

1. boundary: is a chosen value (or a top-5 trial) within the outer 5% of its search range? Many
   hits at one end mean the range was too narrow on that side. Log-scaled settings are measured
   on the log scale.
2. best_so_far: the best inner score after each trial (mean over units), and the first trial that
   reached the final best. A curve still rising at the end means 30 trials were too few.
3. hierarchy: the L1 winner's margin over the L1 runner-up against the gain from L2 tuning. It
   shows which level of the search mattered.
4. profiles: the inner score against each setting, in 6 equal-width bins of its (log) range, from
   the start-up trials only (trial number below n6.N_STARTUP: the defaults and the random draws).
   Later trials are placed by the optimiser where scores are already good, so they would mix the
   setting's effect with the search phase.
Trials are always identified by their `number` column, never by their position among the
completed trials (a failed trial keeps its number).
Units tuned in simplified mode have no search on their own fold and are listed, not used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import agg

EDGE = 0.05
N_BINS = 6
TOP = 5


def load(dirs: dict) -> list[dict]:
    """One record per unit folder: its config.json, l1.csv and trials.csv (those that exist)."""
    out = []
    for (enc, k), ud in dirs.items():
        ud = Path(ud)
        if not (ud / "config.json").exists() or not (ud / "trials.csv").exists():
            continue
        out.append({"encoder": enc, "fold": int(k), "config": json.loads((ud / "config.json").read_text()),
                    "trials": pd.read_csv(ud / "trials.csv"),
                    "l1": pd.read_csv(ud / "l1.csv") if (ud / "l1.csv").exists() else None})
    return out


def searched(unit: dict) -> pd.DataFrame:
    """The unit's completed search trials in trial order (empty for a unit that reused fold 0's settings)."""
    t = unit["trials"]
    t = t[(t["state"] == "COMPLETE") & np.isfinite(pd.to_numeric(t["value"], errors="coerce"))]
    return t.sort_values("number").reset_index(drop=True)


def position(values, lo: float, hi: float, log: bool) -> np.ndarray:
    """Where each value sits in its range: 0 = lower end, 1 = upper end (on the log scale if log)."""
    v = np.asarray(values, dtype=float)
    if log:
        v, lo, hi = np.log(v), np.log(lo), np.log(hi)
    return (v - lo) / (hi - lo)


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def _n_startup() -> int:
    from .nb import n6

    return n6.N_STARTUP  # trial 0 (the defaults) + 9 random, then the GP sampler


def _curve(unit: dict) -> np.ndarray:
    """Best inner score so far, indexed by trial number (NaN until the first completed trial)."""
    t = searched(unit)
    v = np.full(int(t["number"].max()) + 1, np.nan)
    v[t["number"].to_numpy(int)] = t["value"].to_numpy(float)
    return np.fmax.accumulate(v)


def _edge_shares(pos) -> dict:
    p = np.asarray(pos, dtype=float)
    p = p[np.isfinite(p)]
    if not len(p):
        return {"share_outer_5pct": None, "share_low_end": None, "share_high_end": None}
    return {"share_outer_5pct": agg.share(((p < EDGE) | (p > 1 - EDGE)).mean()),
            "share_low_end": agg.share((p < EDGE).mean()), "share_high_end": agg.share((p > 1 - EDGE).mean())}


def _applies(unit: dict, hp: str) -> bool:
    return hp != "attn_hidden" or unit["config"]["config"].get("pooling") == "gated"


def boundary(units: list[dict], space_of) -> dict:
    """Per setting: the share of units whose chosen value, and of top-5 trials, in the outer 5% of the range."""
    out = {}
    for hp, (lo, hi, log) in space_of(True).items():
        us = [u for u in units if _applies(u, hp) and hp in u["trials"]]

        def block(sel):
            chosen = [position([u["config"]["config"][hp]], lo, hi, log)[0] for u in sel]
            top = [position(searched(u).nlargest(TOP, "value")[hp], lo, hi, log) for u in sel]
            return {"units": agg.count(len(sel)), "chosen": _edge_shares(chosen),
                    "top5_trials": _edge_shares(np.concatenate(top) if top else [])}

        out[hp] = {"scale": "log" if log else "linear", "range": [lo, hi], "all": block(us),
                   "by_encoder": {e: block([u for u in us if u["encoder"] == e])
                                  for e in sorted({u["encoder"] for u in us})}}
    return out


def best_so_far(units: list[dict]) -> dict:
    """Per encoder: the mean over units of the best inner score after each trial, and when the best came."""
    out = {}
    for e in sorted({u["encoder"] for u in units}):
        curves, first = [], []
        for u in units:
            if u["encoder"] != e:
                continue
            c = _curve(u)
            curves.append(c)
            first.append(int(np.argmax(c >= c[-1])))  # the trial number that first reached the final best
        n = max(len(c) for c in curves)
        grid = np.full((len(curves), n), np.nan)
        for i, c in enumerate(curves):
            grid[i, :len(c)] = c
        cnt = np.isfinite(grid).sum(axis=0)
        mean = np.where(cnt > 0, np.nansum(grid, axis=0) / np.maximum(cnt, 1), np.nan)
        ns = _n_startup()
        gp = [c[-1] - c[ns - 1] for c in curves if len(c) > ns and np.isfinite(c[ns - 1])]
        out[e] = {"mean_best_so_far": [_r(x) for x in mean], "n_trials": int(n),
                  "first_trial_of_final_best_median": float(np.median(first)),
                  "share_units_best_is_trial_0": agg.share(np.mean([f == 0 for f in first])),
                  "mean_gain_after_startup_trials": _r(np.mean(gp)) if gp else None}
    return out


def hierarchy(units: list[dict]) -> dict:
    """Per unit: the L1 winner's margin over the runner-up, and the L2 gain over the winner at the defaults."""
    out = {}
    for e in sorted({u["encoder"] for u in units}):
        rows = {}
        for u in units:
            if u["encoder"] != e or u["l1"] is None:
                continue
            sc = np.sort(u["l1"]["inner_score"].dropna().to_numpy(float))[::-1]
            if len(sc) < 2:
                continue
            rows[f"fold_{u['fold']}"] = {"l1_margin": _r(sc[0] - sc[1]),
                                         "l2_gain": _r(searched(u)["value"].max() - sc[0])}
        if rows:
            m, g = [v["l1_margin"] for v in rows.values()], [v["l2_gain"] for v in rows.values()]
            out[e] = {"per_unit": rows, "mean_l1_margin": _r(np.mean(m)), "mean_l2_gain": _r(np.mean(g)),
                      "share_units_l2_gain_above_l1_margin": agg.share(np.mean([b > a for a, b in zip(m, g)]))}
    return out


def profiles(units: list[dict], space_of) -> dict:
    """Per encoder and setting: the mean inner score per bin (6 equal-width bins of the (log) range),
    from the start-up trials only (number < n6.N_STARTUP), which the optimiser did not place.

    "mean" is the raw mean; "mean_centred" first subtracts each unit's own mean score (over these
    trials), so folds that are simply harder do not bend the curve. "trial_share" is each bin's
    share of the trials used."""
    out = {}
    for e in sorted({u["encoder"] for u in units}):
        out[e] = {}
        for hp, (lo, hi, log) in space_of(True).items():
            pos, val, cen = [], [], []
            for u in units:
                t = searched(u)
                t = t[t["number"] < _n_startup()]
                if u["encoder"] != e or not _applies(u, hp) or hp not in t or not len(t):
                    continue
                pos.append(position(t[hp], lo, hi, log))
                val.append(t["value"].to_numpy(float))
                cen.append(val[-1] - val[-1].mean())
            if not pos:
                continue
            pos, val, cen = np.concatenate(pos), np.concatenate(val), np.concatenate(cen)
            b = np.clip((pos * N_BINS).astype(int), 0, N_BINS - 1)
            out[e][hp] = {
                "mean": [_r(val[b == i].mean()) if (b == i).any() else None for i in range(N_BINS)],
                "mean_centred": [_r(cen[b == i].mean()) if (b == i).any() else None for i in range(N_BINS)],
                "trial_share": [agg.share((b == i).mean()) for i in range(N_BINS)]}
    return out


def summary(units: list[dict], U: float) -> dict:
    """The hpo_evidence block of agg_results.json (inner scores only; decides nothing)."""
    from .nb import n6

    def space_of(gated: bool) -> dict:
        return n6.space(U, gated)

    bo = [u for u in units if len(searched(u)) >= 2]
    skipped: dict = {}
    for u in units:
        if len(searched(u)) < 2:
            skipped.setdefault(u["encoder"], {"folds": []})["folds"].append(u["fold"])
    out = {"score": "inner validation macro AUC (never a test fold)", "units_without_own_search": skipped,
           "n_bins": N_BINS, "edge": EDGE,
           "profiles_from": "start-up trials only (trial number below 10: the defaults and 9 random draws)"}
    if bo:
        out.update({"boundary": boundary(bo, space_of), "best_so_far": best_so_far(bo),
                    "hierarchy": hierarchy(bo), "profiles": profiles(bo, space_of)})
    return out
