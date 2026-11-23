"""N13: iteration 2 of the iteration plan (iteration_plan_r2.md, sections 2, 5 and 7).

Written before any iteration-2 number exists. Every arm is "added after the results" (R10) and
each one is n6.run with gated attention over all three planes (n6's N10-ready unit folders):
  a1    A1: the default head settings (trial 0 = the L2 defaults, batch 48), no search, a cosine
        schedule over 8 epochs (1-epoch warm-up) and patience 3; seed 0.
  null  trial 0 with the registered schedule (30 epochs, patience 5) at seed 100: its differences
        from Stage A's trial 0 give the null spread s of the keep rule.
  cpu0  trial 0 with the registered schedule at seed 0: Stage A's trial 0 on the arms' device. Run
        only if the arms run on CPU; A1 is then paired with it instead of Stage A (R4).
  a7    A7 (conditional, iteration 3): the registered 30-trial GP search for gated attention over
        all planes, batch size in [8, 128] and weight decay in [1e-6, 1], under the schedule chosen
        in iteration 2 (pass schedule_epochs and patience). Trial 0 stays the registered defaults.
Inner fit j of a unit uses seed + j and the refit seed 0, as in Stage A. The keep rule is decide();
it reads inner scores only (R4) and is written down before any S2 number of A1 is read (R5).

Outputs, under the arm's own step name (nothing an iteration-1 notebook reads):
  private/n13_<arm>/units/<encoder>_f<k>/ (n6's layout: head.pt, stats.pt, config.json with the
  schedule length, patience, seed and device of the unit), agg_n13_<arm>.json and
  agg_n13_<arm>_<encoder>_f<k>.json (n6's), and agg_n13_<arm>_inner.json: this module's summary
  block (per unit the trial-0 inner score, E, the settings used and head_sha16 for the N10 check).
Attach rules: never attach an N13 output to N7 or N9 (R9). N10 gets only knee-n13-a1 (and its
rerun, if any), never knee-n6a beside it; check_heads() then compares N10's heads with N13's.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import agg, heads, paths, safe_log
from . import common, n6

ENCODERS = ("dinov2_vits14", "convnext_tiny")
L1_GRID = [("gated", "all")]  # the pilot's head: gated attention over all three planes, no L1 choice
A7_SPACE = {"batch_size": [8, 128], "weight_decay": [1e-6, 1.0]}
REASON = "iteration 2, plan r2: defaults only, see iteration_plan_r2.md"
REASON_A7 = "iteration 3 (A7), plan r2: widened batch-size and weight-decay ranges, see iteration_plan_r2.md"
# None = the registered value (schedule 30 epochs, patience 5); "given" = passed to run() (a7 only)
ARMS = {
    "a1": {"seed": 0, "n_trials": 1, "schedule_epochs": 8, "patience": 3, "space_overrides": None, "reason": REASON},
    "null": {"seed": 100, "n_trials": 1, "schedule_epochs": None, "patience": None, "space_overrides": None,
             "reason": REASON},
    "cpu0": {"seed": 0, "n_trials": 1, "schedule_epochs": None, "patience": None, "space_overrides": None,
             "reason": REASON},
    "a7": {"seed": 0, "n_trials": None, "schedule_epochs": "given", "patience": "given", "space_overrides": A7_SPACE,
           "reason": REASON_A7},
}
# The A1 keep rule (plan r2 section 5)
MIN_GAIN = 0.005
N_OUTER = 5
MIN_FOLDS_UP = 4
SCORE = ("trial-0 inner score: mean over the unit's inner folds of the validation macro AUC at each inner "
         "fit's best epoch (Stage A's objective; default_trial_score in config.json)")


def arm_spec(arm: str, schedule_epochs: int | None = None, patience: int | None = None) -> dict:
    """The n6 settings of one arm. Only a7 takes schedule_epochs and patience (both required): the
    other arms are fixed by plan r2."""
    if arm not in ARMS:
        raise ValueError("unknown arm: use a1, null, cpu0 or a7")
    spec = dict(ARMS[arm])
    if arm == "a7":
        if schedule_epochs is None or patience is None:
            raise ValueError("a7 needs schedule_epochs and patience: the schedule chosen in iteration 2")
        spec["schedule_epochs"], spec["patience"] = int(schedule_epochs), int(patience)
    elif schedule_epochs is not None or patience is not None:
        raise ValueError("only a7 takes schedule_epochs and patience; the other arms are fixed by plan r2")
    spec["name"] = f"n13_{arm}"
    return spec


def run(arm: str, expect: dict | None = None, root=None, workers: int = 4, budget_h: float = 8.0,
        quiet: bool = True, schedule_epochs: int | None = None, patience: int | None = None,
        folds: tuple[int, ...] | None = None, rerun: bool = False,
        encoders: tuple[str, ...] = ENCODERS) -> dict:
    """One arm (see the module docstring). expect: the upstream hashes of the run card.
    rerun=True with folds=(k,): a failed unit again, as step n13_<arm>_r (its unit replaces the
    original in N10, as knee-n6a-r does). encoders: tests only (the plan runs both main encoders)."""
    spec = arm_spec(arm, schedule_epochs, patience)
    if rerun and not folds:
        raise ValueError("a rerun names its folds, e.g. folds=(2,)")
    name = spec["name"] + ("_r" if rerun else "")
    s6 = n6.run(name, encoders=tuple(encoders), folds=tuple(folds) if folds else None, root=root, expect=expect,
                workers=workers, seed=spec["seed"], budget_h=budget_h, quiet=quiet, n_trials=spec["n_trials"],
                l1_grid=list(L1_GRID), schedule_epochs=spec["schedule_epochs"], patience=spec["patience"],
                space_overrides=spec["space_overrides"], overrides_reason=spec["reason"])
    out: dict = {}
    with safe_log.step(f"{name}_inner", quiet=quiet, error_log=paths.out_dir("private") / f"{name}_inner_error.log"):
        out = inner_summary(arm, spec, s6, paths.WORKING / "private" / name / "units", step=name)
        agg.write(f"{name}_inner", out)
    common.summary(out)
    return out


def inner_summary(arm: str, spec: dict, s6: dict, units_dir: Path, step: str | None = None) -> dict:
    """The arm's summary block from n6's result and unit folders (inner scores and settings only)."""
    units_dir = Path(units_dir)
    units, scores, devices = {}, {}, set()
    for uid, st in sorted((s6.get("units") or {}).items()):
        if not st.get("ok"):
            units[uid] = {"ok": False, "skipped": bool(st.get("skipped")), "error": st.get("error")}
            continue
        u = json.loads((units_dir / uid / "config.json").read_text())
        rec = {"ok": True, "trial0_inner_score": u["default_trial_score"], "E": int(u["E"]),
               "schedule_epochs": int(u["schedule_epochs"]), "patience": int(u["patience"]), "seed": int(u["seed"]),
               "device": u["device"], "device_name": u["device_name"], "tuning": u["tuning"],
               "head_sha16": common.sha16(units_dir / uid / "head.pt")}
        if spec.get("n_trials") != 1:  # a search (a7): its best trial is a maximum over the trials
            rec["best_inner_score"] = u["best_inner_score"]
            rec["config"] = u["config"]
        units[uid] = rec
        devices.add(u["device_name"])
        scores.setdefault(u["encoder"], {})[int(u["fold"])] = u["default_trial_score"]
    sched, pat = spec.get("schedule_epochs"), spec.get("patience")
    return {"arm": arm, "step": step or spec["name"], "plan": "iteration_plan_r2.md", "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": spec["reason"],
            "settings": {"l1": "gated|all", "n_trials": spec.get("n_trials") or n6.N_TRIALS,
                         "schedule_epochs": heads.MAX_EPOCHS if sched is None else int(sched),
                         "patience": heads.PATIENCE if pat is None else int(pat), "seed": spec["seed"],
                         "space_overrides": n6.check_space_overrides(spec.get("space_overrides"))},
            "all_units_ok": s6.get("all_units_ok"), "all_nested": s6.get("all_nested"),
            "leakage_assertion_passed": s6.get("leakage_assertion_passed"), "status": s6.get("status"),
            "device_names": sorted(devices), "score": SCORE,
            "trial0_inner_scores": scores, "units": units}


def trial0_scores(unit_summaries) -> dict:
    """{encoder: {fold: trial-0 inner score}} from per-unit summaries: n6's agg_<name>_<encoder>_f<k>.json
    (e.g. knee-n6a's, for Stage A's trial 0) or config.json files, already loaded as dicts."""
    out: dict = {}
    for u in unit_summaries:
        out.setdefault(u["encoder"], {})[int(u["fold"])] = u["default_trial_score"]
    return out


def _fold(k) -> int:
    return int(str(k).replace("fold_", ""))


def decide(a1: dict, base: dict, null: dict, min_gain: float = MIN_GAIN) -> dict:
    """The A1 keep rule of plan r2 section 5, on inner scores {encoder: {fold: score}} (R4, R5).

    a1: the A1 arm's trial-0 inner score per unit. base: Stage A's trial 0 (on CPU: the cpu0 arm).
    null: the seed-100 rerun of trial 0 (the null arm). For every encoder of a1:
      gain      = mean over the 5 outer folds of (a1 - base), unit by unit;
      s         = SD (n - 1 divisor) of the per-unit differences null - base over the encoders of a1
                  (both main encoders: 10 differences);
      threshold = max(min_gain, 2 s / sqrt(5));
      folds_up  = outer folds where a1 > base;
      keep      = gain > threshold and folds_up >= 4.
    Fold keys may be ints, "0" or "fold_0" (JSON). Every input must hold outer folds 0-4 for every
    encoder of a1, with finite scores: the rule is not defined on fewer units (ValueError), so a failed
    unit is rerun, never dropped. per_fold holds a1 - base for each fold."""
    def norm(d):
        return {str(e): {_fold(k): (math.nan if v is None else float(v)) for k, v in m.items()}
                for e, m in dict(d).items()}

    a1, base, null = norm(a1), norm(base), norm(null)
    folds = list(range(N_OUTER))
    encs = sorted(a1)
    if not encs:
        raise ValueError("decide: a1 holds no encoder")
    for label, d in (("a1", a1), ("base", base), ("null", null)):
        for e in encs:
            m = d.get(e, {})
            if sorted(m) != folds or not all(math.isfinite(v) for v in m.values()):
                raise ValueError(f"decide: {label} needs a finite score for each of the {N_OUTER} outer folds")
    null_diffs = [null[e][k] - base[e][k] for e in encs for k in folds]
    s = float(np.std(null_diffs, ddof=1))
    threshold = max(float(min_gain), 2 * s / math.sqrt(N_OUTER))
    out = {}
    for e in encs:
        diffs = [a1[e][k] - base[e][k] for k in folds]
        gain = float(np.mean(diffs))
        up = int(sum(x > 0 for x in diffs))
        out[e] = {"gain": gain, "threshold": threshold, "folds_up": up,
                  "keep": bool(gain > threshold and up >= MIN_FOLDS_UP), "s": s,
                  "per_fold": dict(zip(folds, diffs))}
    return out


def check_heads(n10_agg: dict, inner: dict, encoder: str = "convnext_tiny") -> dict:
    """Before an iteration-2 submission (plan r2 section 5, code rule): the heads N10 loaded are exactly
    N13's. n10_agg: N10's agg_n10.json (heads_sha16 per fold: n10.py:126, written at n10.py:201);
    inner: agg_n13_a1_inner.json (head_sha16 per unit). match needs the same folds on both sides,
    all of them N13 units that finished, and equal hashes."""
    got = {_fold(k): v for k, v in (n10_agg.get("heads_sha16") or {}).items()}
    want = {}
    for uid, u in (inner.get("units") or {}).items():
        enc, k = uid.rsplit("_f", 1)
        if enc == encoder and u.get("ok"):
            want[int(k)] = u["head_sha16"]
    folds = sorted(set(got) | set(want))
    bad = [k for k in folds if got.get(k) is None or got.get(k) != want.get(k)]
    return {"encoder": encoder, "folds": folds, "mismatched_folds": bad,
            "match": bool(want) and not bad and len(want) == N_OUTER}
