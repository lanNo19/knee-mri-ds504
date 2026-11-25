"""N13: iteration 2 of the iteration plan (iteration_plan_r2.md sections 2, 5 and 7, with the plan r3
changes of the iteration plan's section 9, written after review r1 and before any iteration-2 run).

Every arm is "added after the results" (R10) and is its own Kaggle notebook: n6.run with gated
attention over all three planes, writing n6's N10-ready unit folders.
  a1     A1: the default head settings (trial 0 = the L2 defaults, batch 48), no search, a cosine
         schedule over 8 epochs (1-epoch warm-up) and patience 3; seed 0; both main encoders.
  base0  trial 0 with the registered schedule (30 epochs, patience 5) at seed 0: the base of both
         differences of the keep rule, on GPU and on CPU alike. It runs on the same device type,
         Kaggle image and code as a1 and null (plan r3; it replaces the CPU-only cpu0 arm, and
         Stage A's logged trial 0 is kept only as a descriptive check, stage_a_check()).
  null   trial 0 with the registered schedule at seed 100: its differences from base0 give s.
  a7     A7 (conditional, iteration 3), descriptive only (plan r3: no keep decision, no
         submission): ConvNeXt-T only, the registered 30-trial GP search with batch size in
         [8, 128] and weight decay in [1e-6, 1], under the schedule passed in (A1's 8 and 3 if A1
         was kept for ConvNeXt-T, else the registered 30 and 5; nothing is chosen automatically).
         Trial 0 stays the registered defaults.
Inner fit j of a unit uses seed + j and the refit seed 0, as in Stage A. Each arm records its
environment (environment()) in run.json and in its summary block.

Claude runs these outside Kaggle, on the downloaded summaries (agg_n13_<arm>_inner.json):
  merged()         one arm's units from its summaries: a rerun replaces a unit only if it failed;
  decide()         the A1 keep rule on inner scores only (R4), written before any S2 number (R5);
  stage_a_check()  base0 minus Stage A's logged trial 0 per unit (descriptive, never in the rule);
  check_heads()    N10's heads_sha16 (agg_n10.json) against the a1 arm's heads_sha16.
dev() is a CPU notebook: A1's S2 development numbers, paired with base0. It needs the written decision.

Outputs of an arm, under its own step name (no file name that an iteration-1 notebook writes):
private/n13_<arm>/units/<encoder>_f<k>/ (n6's layout), agg_n13_<arm>.json and
agg_n13_<arm>_<encoder>_f<k>.json (n6's), agg_n13_<arm>_inner.json (this module's summary block) and
run.json. Never attach an N13 output to N7 or N9 (R9): its units look like Stage A's and they take
every unit they find. N10 gets only knee-n13-a1 (and its reruns), never knee-n6a beside it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, encoders, heads, hpo_evidence, metrics, paths, safe_log, stage, units as unit_files
from ..labels import FINDINGS, UID
from . import common, n6

MAIN = encoders.MAIN  # ("dinov2_vits14", "convnext_tiny"): the keep rule needs both
L1_GRID = [("gated", "all")]  # the pilot's head: gated attention over all three planes, no L1 choice
A7_SPACE = {"batch_size": [8, 128], "weight_decay": [1e-6, 1.0]}
A7_ENCODERS = ("convnext_tiny",)
PLAN = "iteration plan section 9 (plan r3): iteration_plan_r2.md with the review-r1 changes"
REASON = "iteration 2, plan r3: defaults only, see the iteration plan, section 9"
REASON_A7 = "iteration 3 (A7), plan r3: descriptive only, ConvNeXt-T, widened batch-size and weight-decay ranges"
A7_LABEL = "none (plan r3): A7 is descriptive only; no keep decision and no submission"
DEV_LABEL = "development: these studies were test studies in iteration 1; likely optimistic"
# None = the registered value (schedule 30 epochs, patience 5); "given" = passed to run() (a7 only)
ARMS = {
    "a1": {"seed": 0, "n_trials": 1, "schedule_epochs": 8, "patience": 3, "space_overrides": None,
           "encoders": MAIN, "reason": REASON},
    "base0": {"seed": 0, "n_trials": 1, "schedule_epochs": None, "patience": None, "space_overrides": None,
              "encoders": MAIN, "reason": REASON},
    "null": {"seed": 100, "n_trials": 1, "schedule_epochs": None, "patience": None, "space_overrides": None,
             "encoders": MAIN, "reason": REASON},
    "a7": {"seed": 0, "n_trials": None, "schedule_epochs": "given", "patience": "given", "space_overrides": A7_SPACE,
           "encoders": A7_ENCODERS, "reason": REASON_A7},
}
# The A1 keep rule (plan r3)
MIN_GAIN = 0.005
N_OUTER = 5
MIN_FOLDS_UP = 4
DECIMALS = 4  # scores are stored at 4 decimals; every comparison of the rule is made at 4 decimals
ENV_FIELDS = ("kaggle_image_git_commit", "kaggle_image_build_date", "python", "torch", "cuda", "cudnn", "device_name")
SCORE = ("trial-0 inner score: mean over the unit's inner folds of the validation macro AUC at each inner "
         "fit's best epoch (Stage A's objective; default_trial_score in config.json)")


# ---------------------------------------------------------------- arms

def arm_spec(arm: str, schedule_epochs: int | None = None, patience: int | None = None) -> dict:
    """The n6 settings of one arm. Only a7 takes schedule_epochs and patience (both required): the
    other arms are fixed by the plan."""
    if arm == "cpu0":
        raise ValueError("cpu0 is now base0 (plan r3): the same arm on every device")
    if arm not in ARMS:
        raise ValueError("unknown arm: use a1, base0, null or a7")
    spec = dict(ARMS[arm])
    if arm == "a7":
        if schedule_epochs is None or patience is None:
            raise ValueError("a7 needs schedule_epochs and patience: A1's if A1 was kept for ConvNeXt-T, else 30 and 5")
        spec["schedule_epochs"], spec["patience"] = int(schedule_epochs), int(patience)
    elif schedule_epochs is not None or patience is not None:
        raise ValueError("only a7 takes schedule_epochs and patience; the other arms are fixed by the plan")
    spec["name"] = f"n13_{arm}"
    return spec


def settings(spec: dict) -> dict:
    """The settings block of an arm's summary (the values used; None in ARMS = the registered value)."""
    sched, pat = spec.get("schedule_epochs"), spec.get("patience")
    return {"l1": "gated|all", "n_trials": spec.get("n_trials") or n6.N_TRIALS,
            "schedule_epochs": heads.MAX_EPOCHS if sched is None else int(sched),
            "patience": heads.PATIENCE if pat is None else int(pat), "seed": spec["seed"],
            "space_overrides": n6.check_space_overrides(spec.get("space_overrides"))}


def environment() -> dict:
    """What the arms of one decision must share, field by field (plan r3): the Kaggle image (the
    GIT_COMMIT and BUILD_DATE that Kaggle's docker-python image sets; None elsewhere), python, torch,
    the CUDA and cuDNN builds, and the device ("cpu", or the GPU model)."""
    import platform

    import torch

    def image(var: str, path: str):
        v = os.environ.get(var)
        if not v:
            try:
                v = Path(path).read_text().strip()
            except OSError:
                v = None
        return v or None

    try:
        gpus = sorted({str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())})
    except Exception:  # noqa: BLE001 - recorded as no GPU
        gpus = []
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        cudnn = None
    return {"kaggle_image_git_commit": image("GIT_COMMIT", "/etc/git_commit"),
            "kaggle_image_build_date": image("BUILD_DATE", "/etc/build_date"),
            "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": cudnn, "device_name": " + ".join(gpus) if gpus else "cpu"}


def run(arm: str, expect: dict | None = None, root=None, workers: int = 4, budget_h: float = 8.0,
        quiet: bool = True, schedule_epochs: int | None = None, patience: int | None = None,
        folds: tuple[int, ...] | None = None, rerun: int = 0, encoders: tuple[str, ...] | None = None) -> dict:
    """One arm (see the module docstring). expect: the upstream hashes of the run card.
    A rerun (rerun=1: step n13_<arm>_r; 2: n13_<arm>_r2) runs only the units that failed: pass their
    folds and encoders (folds=(k,), encoders=("<encoder>",)). Its units replace failed units only.
    encoders: a subset of both main encoders for reruns and tests; a7 runs ConvNeXt-T only."""
    if arm in BATCH_ARMS:  # the batch-floor check (it2b, below): its own runner; budget_h is its soft stop,
        # at most BATCH_BUDGET_H through this door (run()'s default of 8.0 is past tonight's limit)
        if schedule_epochs is not None or patience is not None:
            raise ValueError("the batch-floor arms keep the registered schedule (30 epochs, patience 5)")
        return run_batch(arm, expect=expect, root=root, workers=workers, budget_h=min(budget_h, BATCH_BUDGET_H),
                         quiet=quiet, folds=folds, rerun=rerun, encoders=encoders)
    spec = arm_spec(arm, schedule_epochs, patience)
    encs = tuple(spec["encoders"] if encoders is None else encoders)
    if arm == "a7" and encs != A7_ENCODERS:
        raise ValueError("a7 runs ConvNeXt-T only (plan r3)")
    if not encs or not set(encs) <= set(spec["encoders"]):
        raise ValueError("the arms run the two main encoders (a rerun names the failed ones)")
    r = int(rerun)
    if r < 0 or (r and (not folds or encoders is None)):
        raise ValueError("a rerun names the folds and encoders of the failed units, e.g. folds=(2,), encoders=('convnext_tiny',)")
    name = spec["name"] + ("" if not r else "_r" if r == 1 else f"_r{r}")
    env = environment()
    s6 = n6.run(name, encoders=encs, folds=tuple(folds) if folds else None, root=root, expect=expect,
                workers=workers, seed=spec["seed"], budget_h=budget_h, quiet=quiet, n_trials=spec["n_trials"],
                l1_grid=list(L1_GRID), schedule_epochs=spec["schedule_epochs"], patience=spec["patience"],
                space_overrides=spec["space_overrides"], overrides_reason=spec["reason"],
                extra_config={"arm": arm, "rerun": r, "environment": env})
    out: dict = {}
    with safe_log.step(f"{name}_inner", quiet=quiet, error_log=paths.out_dir("private") / f"{name}_inner_error.log"):
        U = stage.StageData(root, expect, need_embeddings=False).pos_cap_upper() if arm == "a7" else None
        out = inner_summary(arm, spec, s6, paths.WORKING / "private" / name / "units", step=name, environment=env,
                            U=U, encoders=encs)
        agg.write(f"{name}_inner", out)
    common.summary(out)
    return out


def _unit_key(uid: str) -> tuple[str, int]:
    enc, k = uid.rsplit("_f", 1)
    return enc, int(k)


def inner_summary(arm: str, spec: dict, s6: dict, units_dir: Path, step: str | None = None,
                  environment: dict | None = None, U: float | None = None, encoders=None) -> dict:
    """The arm's summary block from n6's result and its unit folders (inner scores and settings only)."""
    units_dir = Path(units_dir)
    units, scores, best, heads_sha, devices, dirs = {}, {}, {}, {}, set(), {}
    for uid, st in sorted((s6.get("units") or {}).items()):
        if not st.get("ok"):
            units[uid] = {"ok": False, "skipped": bool(st.get("skipped")), "error": st.get("error")}
            continue
        u = json.loads((units_dir / uid / "config.json").read_text())
        enc, k = u["encoder"], int(u["fold"])
        sha = common.sha16(units_dir / uid / "head.pt")
        rec = {"ok": True, "trial0_inner_score": u["default_trial_score"], "E": int(u["E"]),
               "schedule_epochs": int(u["schedule_epochs"]), "patience": int(u["patience"]), "seed": int(u["seed"]),
               "device": u["device"], "device_name": u["device_name"], "tuning": u["tuning"], "head_sha16": sha}
        if spec.get("n_trials") != 1:  # a search (a7): its best trial is a maximum over the trials
            rec["best_inner_score"] = u["best_inner_score"]
            rec["best_minus_trial0"] = (round(u["best_inner_score"] - u["default_trial_score"], DECIMALS)
                                        if u["default_trial_score"] is not None else None)
            rec["config"] = u["config"]
            best.setdefault(enc, {})[k] = u["best_inner_score"]
        units[uid] = rec
        devices.add(u["device_name"])
        dirs[(enc, k)] = units_dir / uid
        scores.setdefault(enc, {})[k] = u["default_trial_score"]
        heads_sha.setdefault(enc, {})[f"fold_{k}"] = sha  # the format of agg_n10.json's heads_sha16
    out = {"arm": arm, "step": step or spec["name"], "plan": PLAN, "added_after_results": True,
           "test_overrides": True, "test_overrides_reason": spec["reason"],
           "encoders": list(encoders or spec["encoders"]), "settings": settings(spec),
           "environment": environment, "knee_commit": agg.repo_commit(),
           "all_units_ok": s6.get("all_units_ok"), "all_nested": s6.get("all_nested"),
           "leakage_assertion_passed": s6.get("leakage_assertion_passed"), "status": s6.get("status"),
           "device_names": sorted(devices), "score": SCORE,
           "trial0_inner_scores": scores, "heads_sha16": heads_sha, "units": units}
    if arm == "a7":
        out["descriptive"] = True
        out["keep_decision"] = A7_LABEL
        out["best_inner_scores"] = best  # a maximum over 30 trials: favoured by selection
        if U is not None and dirs:  # where the chosen values sit in the widened ranges
            recs = hpo_evidence.load(dirs)
            out["boundary_widened_ranges"] = hpo_evidence.boundary(
                recs, lambda gated: n6.space(U, gated, spec["space_overrides"]))
    return out


# ---------------------------------------------------------------- the keep rule (run outside Kaggle)

def merged(summaries) -> dict:
    """One arm's units from its summaries in run order (the original first, then its reruns).

    A rerun's unit replaces the original's only when the original failed. A second finished unit for
    the same encoder and fold is refused (ValueError), as are summaries of different arms or settings.
    Returns arm, steps, settings, environments, knee_commits and units {(encoder, fold): unit}."""
    items = [summaries] if isinstance(summaries, dict) else list(summaries)
    if not items:
        raise ValueError("merged: no summary given")
    arm, sett = items[0].get("arm"), items[0].get("settings")
    units: dict = {}
    for s in items:
        if s.get("arm") != arm:
            raise ValueError("merged: summaries of different arms")
        if s.get("settings") != sett:
            raise ValueError("merged: a rerun with other settings than the original")
        for uid, u in (s.get("units") or {}).items():
            key = _unit_key(uid)
            prev = units.get(key)
            if prev is not None and prev.get("ok") and u.get("ok"):
                raise ValueError(f"merged: two finished units for {key[0]} fold {key[1]}: "
                                 "a rerun replaces only a failed unit")
            if prev is None or not prev.get("ok"):
                units[key] = {**u, "step": s.get("step")}
    return {"arm": arm, "steps": [s.get("step") for s in items], "settings": sett,
            "environments": [s.get("environment") for s in items],
            "knee_commits": [s.get("knee_commit") for s in items], "units": units}


def _complete(label: str, m: dict) -> dict:
    """{encoder: {fold: score}} of an arm that holds exactly one finished unit per main encoder and fold."""
    want = {(e, k) for e in MAIN for k in range(N_OUTER)}
    if set(m["units"]) != want:
        raise ValueError(f"decide: {label} must hold exactly one unit for each of {', '.join(MAIN)} "
                         f"and each outer fold 0-{N_OUTER - 1} (nothing else)")
    out: dict = {e: {} for e in MAIN}
    for (e, k), u in sorted(m["units"].items()):
        if not u.get("ok"):
            raise ValueError(f"decide: {label} unit {e} fold {k} did not finish: rerun it "
                             "(the rerun replaces only a failed unit)")
        v = u.get("trial0_inner_score")
        if v is None or not math.isfinite(float(v)):
            raise ValueError(f"decide: {label} unit {e} fold {k} has no finite trial-0 inner score")
        out[e][k] = float(v)
    return out


def _heads(m: dict) -> dict:
    out: dict = {}
    for (e, k), u in sorted(m["units"].items()):
        if u.get("ok"):
            out.setdefault(e, {})[f"fold_{k}"] = u["head_sha16"]
    return out


def decide(a1, base0, null, min_gain: float = MIN_GAIN) -> dict:
    """The A1 keep rule of plan r3 (iteration plan section 9), on inner scores only (R4, R5).

    Inputs: each arm's summary, agg_n13_<arm>_inner.json as a dict, or the list [original, rerun, ...]
    (see merged()). The arms are a1, base0 and null, with the plan's settings. Each must hold exactly
    one finished unit for each of the two main encoders (encoders.MAIN) and each of the 5 outer
    folds: anything else raises ValueError (the rule is not defined on other units).
    Environment: every summary's environment dict (environment()) must be equal field by field.
    Otherwise the status is "env_mismatch" and neither encoder is kept (no numbers are computed).
    The rule, with every comparison made at 4 decimals (the precision the scores are stored at):
      s         = sample SD (n - 1 divisor) of the 10 per-unit differences null - base0 (both encoders);
      threshold = max(min_gain, 2 s / sqrt(5)), rounded to 4 decimals;
      gain      = mean over the 5 outer folds of (a1 - base0), rounded to 4 decimals;
      folds_up  = folds with round(a1 - base0, 4) > 0;
      keep      = gain > threshold (a 4-decimal tie is not kept) and folds_up >= 4.
    Also returns n_null, null_mean (with its standard error s / sqrt(10)), and per arm its steps and
    heads (dev() checks the attached units against them)."""
    arms = {"a1": merged(a1), "base0": merged(base0), "null": merged(null)}
    for label, m in arms.items():
        if m["arm"] != label:
            raise ValueError(f"decide: the {label} input holds the summary of arm {m['arm']}")
        if m["settings"] != settings(arm_spec(label)):
            raise ValueError(f"decide: the {label} summaries do not have the plan's settings for {label}")
    S = {label: _complete(label, m) for label, m in arms.items()}
    envs = [e for m in arms.values() for e in m["environments"]]
    if not all(isinstance(e, dict) for e in envs):
        raise ValueError("decide: a summary has no environment record")
    fields = sorted(set(ENV_FIELDS).union(*(e.keys() for e in envs)))
    differ = [f for f in fields if len({json.dumps(e.get(f), sort_keys=True) for e in envs}) > 1]
    record = {"rule": "plan r3, A1 keep rule (n13.decide)", "decided_from": "inner validation scores only (R4, R5)",
              "min_gain": float(min_gain),
              "arms": {label: {"steps": m["steps"], "heads_sha16": _heads(m)} for label, m in arms.items()},
              "same_knee_commit": len({c for m in arms.values() for c in m["knee_commits"]}) == 1}
    if differ:
        return {"status": "env_mismatch", "env_differences": differ,
                "environments": {label: m["environments"] for label, m in arms.items()},
                "encoders": {e: {"gain": None, "threshold": None, "folds_up": None, "keep": False, "s": None}
                             for e in MAIN}, **record}
    folds = list(range(N_OUTER))
    null_diffs = [S["null"][e][k] - S["base0"][e][k] for e in MAIN for k in folds]
    s = float(np.std(null_diffs, ddof=1))
    threshold_raw = max(float(min_gain), 2 * s / math.sqrt(N_OUTER))
    threshold = round(threshold_raw, DECIMALS)
    out_e = {}
    for e in MAIN:
        diffs = [S["a1"][e][k] - S["base0"][e][k] for k in folds]
        gain_raw = float(np.mean(diffs))
        gain = round(gain_raw, DECIMALS)
        up = sum(1 for x in diffs if round(x, DECIMALS) > 0)
        out_e[e] = {"gain": gain, "threshold": threshold, "folds_up": up,
                    "keep": bool(gain > threshold and up >= MIN_FOLDS_UP), "s": s,
                    "gain_unrounded": gain_raw, "per_fold": {k: round(x, DECIMALS) for k, x in zip(folds, diffs)}}
    return {"status": "decided", "environment": envs[0], "s": s, "s_divisor": "n - 1 (sample SD)",
            "n_null": len(null_diffs), "null_mean": float(np.mean(null_diffs)),
            "null_mean_se": s / math.sqrt(len(null_diffs)), "threshold": threshold,
            "threshold_unrounded": threshold_raw, "encoders": out_e, **record}


def stage_a_check(base0, stage_a_units) -> dict:
    """Descriptive reproducibility check (plan r3), never used in the rule: base0's trial-0 inner score
    minus Stage A's logged trial 0, per unit, at 4 decimals. stage_a_units: knee-n6a's ten per-unit
    summaries (agg_n6a_<encoder>_f<k>.json, as dicts). Each must be nested with the L1 winner gated|all
    (trial 0 is then the same configuration), with one unit per main encoder and fold. base0 must be
    the base0 arm's summaries, with its settings."""
    m = merged(base0)
    if m["arm"] != "base0" or m["settings"] != settings(arm_spec("base0")):
        raise ValueError("stage_a_check: the first input must be the base0 arm's summaries, with base0's settings")
    b = _complete("base0", m)
    a: dict = {e: {} for e in MAIN}
    for u in stage_a_units:
        e, k = u["encoder"], int(u["fold"])
        if e not in a or not 0 <= k < N_OUTER:
            raise ValueError("stage_a_check: a unit outside the two main encoders and folds 0-4")
        if k in a[e]:
            raise ValueError("stage_a_check: two Stage A units for one encoder and fold")
        if u.get("tuning") != "nested" or u.get("l1_winner") != {"pooling": "gated", "planes": "all"}:
            raise ValueError("stage_a_check: a Stage A unit whose trial 0 is not gated|all from a nested search")
        if u.get("default_trial_score") is None:
            raise ValueError("stage_a_check: a Stage A unit without a trial-0 score")
        a[e][k] = float(u["default_trial_score"])
    if any(sorted(a[e]) != list(range(N_OUTER)) for e in MAIN):
        raise ValueError("stage_a_check: Stage A needs one unit per main encoder and fold 0-4")
    diff = {f"{e}_f{k}": round(b[e][k] - a[e][k], DECIMALS) for e in MAIN for k in range(N_OUTER)}
    return {"descriptive": True, "used_in_rule": False,
            "what": "base0 minus Stage A's logged trial 0 (same configuration and seeds; other session and image)",
            "caveat": ("Stage A read knee-n3 and base0 reads knee-n3-r: their S2 and inner folds are the same by "
                       "design, asserted but not hash-checked, so a difference can also come from there"),
            "per_unit": diff, "max_abs_difference": max(abs(v) for v in diff.values()),
            "all_equal_at_4_decimals": all(v == 0 for v in diff.values())}


def check_heads(n10_agg: dict, n13_summary, encoder: str = "convnext_tiny") -> dict:
    """Run outside Kaggle, before an iteration-2 submission (N10 runs the bundle's wheel, which has no
    N13 code): N10's heads are exactly the a1 arm's. n10_agg: agg_n10.json (heads_sha16 =
    {"fold_k": sha16}, n10.py:126, written at :201). n13_summary: agg_n13_a1_inner.json, or the list
    [original, rerun, ...]; their heads_sha16 blocks list the finished units only, so a rerun fills a
    failed fold and a second finished unit for one fold is refused. match needs the same five folds
    on both sides and equal hashes."""
    items = [n13_summary] if isinstance(n13_summary, dict) else list(n13_summary)
    want: dict = {}
    for s in items:
        if s.get("arm") != "a1":
            raise ValueError("check_heads compares N10 with the a1 arm's summaries")
        for fk, sha in ((s.get("heads_sha16") or {}).get(encoder) or {}).items():
            k = _fold(fk)
            if k in want:
                raise ValueError("check_heads: two finished units for one fold (a rerun replaces only a failed unit)")
            want[k] = sha
    got = {_fold(k): v for k, v in (n10_agg.get("heads_sha16") or {}).items()}
    folds = sorted(set(got) | set(want))
    bad = [k for k in folds if got.get(k) is None or got.get(k) != want.get(k)]
    return {"encoder": encoder, "folds": folds, "mismatched_folds": bad,
            "match": not bad and folds == list(range(N_OUTER))}


def _fold(k) -> int:
    return int(str(k).replace("fold_", ""))


# ---------------------------------------------------------------- A1's S2 development numbers (CPU notebook)

def _decision(decision) -> dict:
    """The written decision: decide()'s output as a dict, its JSON text, or the path of its JSON file.
    Only status "decided" is accepted: an "env_mismatch" record means there is no A1 decision, so no
    S2 number of A1 may be computed from it (R5)."""
    if decision is None:
        raise ValueError("dev needs the written A1 decision (decide()'s output, its JSON text or its file): R5")
    if isinstance(decision, (str, Path)):
        text = str(decision)
        decision = json.loads(text) if text.lstrip().startswith("{") else json.loads(Path(text).read_text())
    status = decision.get("status") if isinstance(decision, dict) else None
    if status == "env_mismatch":
        raise ValueError("dev: the record says env_mismatch, so there is no A1 decision on these arms; "
                         "no S2 number of A1 is computed (R5)")
    if status != "decided":
        raise ValueError('dev needs an A1 decision written by n13.decide, with status "decided" (R5)')
    ok = True
    for e in MAIN:
        ok = ok and isinstance(((decision.get("encoders") or {}).get(e) or {}).get("keep"), bool)
    for arm in ("a1", "base0"):
        h = ((decision.get("arms") or {}).get(arm) or {}).get("heads_sha16") or {} if ok else {}
        ok = ok and all(sorted(_fold(k) for k in (h.get(e) or {})) == list(range(N_OUTER)) for e in MAIN)
    if not ok:
        raise ValueError("dev: this is not a decision written by n13.decide (status, keep per encoder, the arms' heads)")
    return decision


def _decided_units(base: Path | None, decision: dict, arm: str) -> dict:
    """{(encoder, fold): unit folder} of the attached units the decision was written on (head hash)."""
    steps = decision["arms"][arm]["steps"]
    want = decision["arms"][arm]["heads_sha16"]
    found: dict = {}
    for step in steps:
        for p in paths.glob_outputs(f"private/{step}/units/*_f*/head.pt", base):
            found.setdefault(_unit_key(p.parent.name), []).append(p.parent)
    out = {}
    for e in MAIN:
        for k in range(N_OUTER):
            sha = {_fold(f): v for f, v in want[e].items()}[k]
            hits = [ud for ud in found.get((e, k), []) if common.sha16(ud / "head.pt") == sha]
            if not hits:
                raise ValueError(f"dev: the attached {arm} outputs do not hold the units the decision was written on")
            ud = hits[0]
            cfg = json.loads((ud / "config.json").read_text())
            if cfg.get("predictions_sha16") != common.sha16(ud / "predictions.csv"):
                raise ValueError(f"dev: a {arm} unit's predictions.csv is not the one its config.json records")
            out[(e, k)] = ud
    return out


def _check_numbers(decision: dict, dirs: dict) -> None:
    """The decision's numbers follow from the attached a1 and base0 units (their config.json
    default_trial_score, as the summaries carried it) and from its own s and min_gain, by decide()'s
    rule: per fold round(a1 - base0, 4), the gain, folds_up, the threshold and keep. Else ValueError."""
    thr = decision.get("threshold")
    s, mg = decision.get("s"), decision.get("min_gain", MIN_GAIN)
    if thr is None or s is None or round(max(float(mg), 2 * float(s) / math.sqrt(N_OUTER)), DECIMALS) != thr:
        raise ValueError("dev: the decision's threshold does not follow from its s and min_gain")
    for e in MAIN:
        rec = decision["encoders"][e]
        sc = {arm: [json.loads((dirs[arm][(e, k)] / "config.json").read_text())["default_trial_score"]
                    for k in range(N_OUTER)] for arm in ("a1", "base0")}
        diffs = [a - b for a, b in zip(sc["a1"], sc["base0"])]
        per_fold = {_fold(k): v for k, v in (rec.get("per_fold") or {}).items()}
        gain = round(float(np.mean(diffs)), DECIMALS)
        up = sum(1 for x in diffs if round(x, DECIMALS) > 0)
        if ([per_fold.get(k) for k in range(N_OUTER)] != [round(x, DECIMALS) for x in diffs]
                or rec.get("gain") != gain or rec.get("folds_up") != up or rec.get("threshold") != thr
                or rec.get("keep") is not bool(gain > thr and up >= MIN_FOLDS_UP)):
            raise ValueError(f"dev: the decision's numbers for {e} do not follow from the attached units")


def _predictions(dirs: dict, n: int, row_of: dict) -> dict:
    """{encoder: [n, 12] S2 predictions} from one arm's unit folders."""
    out = {e: np.full((n, len(FINDINGS)), np.nan) for e in MAIN}
    for (e, _), ud in sorted(dirs.items()):
        df = pd.read_csv(ud / "predictions.csv")
        rows = df[UID].astype(str).map(row_of)
        ok = rows.notna().to_numpy()
        out[e][rows[ok].astype(int).to_numpy()] = df.loc[ok, FINDINGS].to_numpy(float)
    return out


def dev(decision, root=None, expect: dict | None = None, quiet: bool = True, budget_h: float = 1.0) -> dict:
    """A1's S2 development numbers (plan R5 and R6), a CPU notebook run only after the decision is written.

    decision: decide()'s output (a dict, its JSON text or the path of its JSON file) with status
    "decided". Refused without it, on any other status (an env_mismatch record holds no decision), and
    when its numbers do not follow from the attached units (_check_numbers).
    Inputs attached: knee-n0, knee-n3-r, knee-n5 and the a1 and base0 outputs (with their reruns). The
    units are those the decision was written on (their head hashes). With N9's evaluation set (usable,
    scored training studies with an S2 fold, predicted by every model here) and scoring rule (uncertain =
    negative), per main encoder: the fold-stratified macro AUC and the mean of fold macros of a1 and of
    base0, and a1 minus base0 (paired: same studies). Every number is labelled as development."""
    dec = _decision(decision)
    canon = json.dumps(dec, sort_keys=True, default=str).encode()
    cfg = {"decision_status": dec["status"], "decision_keep": {e: dec["encoders"][e]["keep"] for e in MAIN},
           "decision_sha16": hashlib.sha256(canon).hexdigest()[:16]}
    s: dict = {"label": DEV_LABEL, "descriptive": True, "added_after_results": True, "plan": PLAN,
               "decision": {"status": dec["status"], "keep": cfg["decision_keep"], "sha16": cfg["decision_sha16"]}}
    with common.notebook("n13_dev", cfg, quiet=quiet, budget_h=budget_h):
        base = Path(root) if root else None
        d = stage.StageData(root, expect)
        row_of = {u: i for i, u in enumerate(d.uids)}
        s["upstream"] = d.upstream
        dirs = {arm: _decided_units(base, dec, arm) for arm in ("a1", "base0")}
        _check_numbers(dec, dirs)  # an edited record (e.g. a keep flag) is refused
        s["chain"] = {arm: unit_files.check_units(d, dirs[arm]) for arm in dirs}
        if not all(c["s2_folds_match_units"] for c in s["chain"].values()):
            s["status"] = "chain mismatch"
            agg.write("n13_dev", s)
            raise unit_files.ChainError("N13 units do not match the attached S2 folds")
        P = {arm: _predictions(dirs[arm], d.n, row_of) for arm in dirs}
        f2, Y = d.folds["s2_fold"], d.eval_y  # uncertain = negative
        tr_all = np.flatnonzero(d.train & (f2 >= 0))
        ev_all = tr_all[d.scored[tr_all]]
        have = np.ones(d.n, bool)
        for arm in P:
            for e in MAIN:
                have &= np.isfinite(P[arm][e]).all(axis=1)
        ev = ev_all[have[ev_all]]
        elig = list(range(len(FINDINGS)))
        s["evaluation_studies"] = agg.count(len(ev))
        s["evaluation_share_of_scored"] = agg.share(len(ev) / max(len(ev_all), 1))
        out = {}
        for e in MAIN:
            row = {}
            for arm in ("a1", "base0"):
                X = P[arm][e]
                row[arm] = {"fold_stratified_macro_auc": metrics.macro(X[ev], Y[ev], f2[ev], elig),
                            "mean_of_fold_macros": metrics.per_fold_macro_mean(X[ev], Y[ev], f2[ev], elig)}
            row["a1_minus_base0"] = {m: row["a1"][m] - row["base0"][m] for m in row["a1"]}
            out[e] = {arm: {m: (round(v, DECIMALS) if math.isfinite(v) else None) for m, v in vals.items()}
                      for arm, vals in row.items()}
        s["s2"] = out
        s["metrics"] = {"fold_stratified_macro_auc": "pairs inside one S2 test fold, mean over the 12 findings",
                        "mean_of_fold_macros": "mean over the S2 folds of each fold's macro AUC (each fold one fifth)"}
        s["status"] = "ok"
        agg.write("n13_dev", s)
    common.summary(s)
    return s


# ---------------------------------------------------------------- the batch-floor check (it2b)
# Declaration: it2b_batch_floor_r1.md, written before any of these runs; "added after the results" (R10).
# Finding: the head search chose a batch size on its floor (16) in 6 of 10 units and the floor was never
# extended. Question: does a batch size below the floor score higher on inner validation?
# Three arms, each its own CPU notebook. Every arm fits, per unit (main encoder x outer fold), the unit's
# CHOSEN Stage A configuration (knee-n6a's agg_n6a_<encoder>_f<k>.json, field "config") on the unit's inner
# folds, with the fits of n6's score(): the same inner sets, standardisation, leakage guard, seed + j,
# 30-epoch schedule and patience 5. Only the batch size differs between the arms:
#   bc  the chosen batch size, unchanged: the base of both differences. It is needed because base0 is NOT
#       the chosen configuration: base0, a1 and null fit trial 0 (the L2 defaults, batch 48).
#   b8  batch size 8.      b4  batch size 4.
# No arm refits on the outer training set, predicts a test fold or writes a unit folder: there is no
# head.pt, no predictions.csv and so nothing that N7, N9 or N10 could take for a unit. The test fold and
# the gold studies reach each fit only as forbidden indices.
# Nothing above this line changed except the six lines of run() that hand these three arm names over.
import time
from dataclasses import replace

BATCH_PLAN = "it2b_batch_floor_r1.md: the batch-floor check, declared before the run, added after the results"
BATCH_REASON = "it2b batch-floor check: the chosen Stage A configuration on the inner folds; only the batch size differs"
BATCH_ARMS = {"bc": None, "b8": 8, "b4": 4}  # None: the unit's chosen batch size
BATCH_SEED = 0  # base0's seed: inner fit j uses seed + j
BATCH_BUDGET_H = 6.0  # the soft stop of a session (start about 15:30, end by about 22:00)
SEARCH_RANGE = (16, 128)  # Stage A's registered batch-size range (n6.suggest)
S_DECLARED = 0.0030  # iteration 2's s (n13_numbers.md): the SD of the 10 null minus base0 differences
MATTERED, NO_SUPPORT, NO_VERDICT = "the floor mattered", "no support", "no verdict"
# The chosen batch size and best inner score per unit, as read on 25 Nov (n9_r3_numbers.md, "The optimum
# found by the head search, per unit"): a check that the attached knee-n6a is that version.
CHOSEN_REF = {
    "dinov2_vits14": {0: (33, 0.7249), 1: (16, 0.7419), 2: (16, 0.7362), 3: (16, 0.7435), 4: (16, 0.7368)},
    "convnext_tiny": {0: (33, 0.7256), 1: (16, 0.7497), 2: (16, 0.7466), 3: (27, 0.7522), 4: (23, 0.7460)},
}
BATCH_SCORE = ("inner score: mean over the unit's inner folds of the validation macro AUC at each inner fit's "
               "best epoch (Stage A's objective), for the unit's chosen configuration at the arm's batch size")


def _r5(x) -> int:
    """To the nearest 5 (as agg.count rounds): steps per epoch at batch 4 would give a study count back."""
    return int(5 * round(float(x) / 5))


def batch_settings(arm: str) -> dict:
    """The settings block of a batch-floor arm (the batch size sits under "config": a setting, not a count)."""
    if arm not in BATCH_ARMS:
        raise ValueError("unknown batch-floor arm: use bc, b8 or b4")
    b = BATCH_ARMS[arm]
    return {"l1": "gated|all", "n_trials": 1, "schedule_epochs": heads.MAX_EPOCHS, "patience": heads.PATIENCE,
            "seed": BATCH_SEED, "space_overrides": None,
            "config": {"source": "chosen (knee-n6a)", "batch_size": "chosen" if b is None else int(b)}}


def chosen_configs(root=None, encoders=MAIN, folds=None, ref: dict | None = CHOSEN_REF) -> dict:
    """{(encoder, fold): {"config", "best_inner_score", "sha16"}}: each unit's chosen Stage A configuration.

    Read from the attached knee-n6a output: agg_n6a_<encoder>_f<k>.json at its root (the per-unit summary
    n6 writes; never a file under private/). Exactly one file per unit, from a nested search whose winner
    is gated attention over all planes, with a batch size inside the registered range. With ref (the
    default: CHOSEN_REF) the batch size and the best inner score must be the ones read on 25 Nov.
    Anything else raises ValueError before any fit."""
    base = Path(root) if root else None
    out = {}
    for e in encoders:
        for k in (range(N_OUTER) if folds is None else folds):
            hits = paths.glob_outputs(f"agg_n6a_{e}_f{int(k)}.json", base)
            if len(hits) != 1:
                raise ValueError(f"chosen_configs: attach exactly one knee-n6a output (unit {e} fold {k}: "
                                 f"{len(hits)} summaries found)")
            u = json.loads(hits[0].read_text())
            c = u.get("config") or {}
            if (u.get("encoder") != e or int(u.get("fold", -1)) != int(k) or u.get("tuning") != "nested"
                    or u.get("l1_winner") != {"pooling": "gated", "planes": "all"}
                    or set(c) != set(heads.Config.__dataclass_fields__)
                    or (c["pooling"], c["planes"]) != ("gated", "all")):
                raise ValueError(f"chosen_configs: {e} fold {k} is not a nested gated|all Stage A unit summary")
            b = c["batch_size"]
            if int(b) != b or not SEARCH_RANGE[0] <= b <= SEARCH_RANGE[1]:
                raise ValueError(f"chosen_configs: {e} fold {k} has a batch size outside the registered range")
            if ref is not None:
                want = (ref.get(e) or {}).get(int(k))
                if want is None or (int(b), round(float(u["best_inner_score"]), DECIMALS)) != want:
                    raise ValueError(f"chosen_configs: {e} fold {k} is not the Stage A unit read on 25 Nov "
                                     "(batch size or best inner score differs)")
            out[(e, int(k))] = {"config": heads.config_from(c).to_dict(),
                                "best_inner_score": float(u["best_inner_score"]), "sha16": common.sha16(hits[0])}
    return out


def _batch_task(task: dict) -> dict:
    """Worker entry (spawned): one unit; a failure keeps its traceback under private/<step>/errors/."""
    return n6._guarded(_batch_unit, task, Path(task["dir"]))


def _batch_unit(task: dict, od: Path) -> dict:
    """The inner fits of one unit: n6's score() for the chosen configuration at the arm's batch size.

    Past the deadline no fit starts, and a running fit stops at the end of its epoch (heads.fit's
    on_epoch): the unit then returns stopped, with no score. Nothing is written to disk."""
    t0 = time.time()
    if t0 > task["deadline"]:
        return {"task": task["id"], "ok": False, "stopped": True}
    enc, k, seed, elig = task["encoder"], task["fold"], task["seed"], task["eligible"]
    chosen = heads.config_from(task["config_chosen"])
    b = task["batch_size"]
    cfg = chosen if b is None else replace(chosen, batch_size=int(b))
    d, bank = n6._data(), n6._bank(enc)
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
    if {key for key in used if used[key] != task["config_chosen"][key]} - {"batch_size"}:
        raise AssertionError("a setting other than the batch size differs from the chosen configuration")
    summary = {"encoder": enc, "fold": k, "inner_score": round(float(np.mean(sc)), DECIMALS),
               "E": max(1, int(np.median(best_eps))), "epochs": ran, "inner_k": fits, "seed": seed,
               "schedule_epochs": heads.MAX_EPOCHS, "patience": heads.PATIENCE,
               "device": n6._W["device"], "device_name": n6._device_name(n6._W["device"]),
               "config": used, "batch_chosen": int(chosen.batch_size),
               "steps_per_epoch": _r5(np.mean(spe)), "steps_per_epoch_chosen": _r5(np.mean(spe_chosen)),
               "step_ratio": round(float(np.sum(spe) / max(1, np.sum(spe_chosen))), 2),
               "stage_a_best_inner_score": task["stage_a_best"], "chosen_sha16": task["chosen_sha16"],
               "seconds": round(time.time() - t0, 1)}
    return {"task": task["id"], "ok": True, "summary": summary, "fits": fits,
            "guard_checks": heads.guard_checks - checks0}


def batch_summary(arm: str, step: str, results: dict, task_ids: list, environment: dict | None,
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
    return {"arm": arm, "step": step, "plan": BATCH_PLAN, "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": BATCH_REASON,
            "encoders": list(encoders), "settings": batch_settings(arm),
            "chosen_ref_checked": bool(ref_checked),
            "environment": environment, "knee_commit": agg.repo_commit(),
            "status": "ok" if len(done) == len(task_ids) else "partial",
            "all_units_ok": len(done) == len(task_ids),
            "units_finished": done, "units_missing": [uid for uid in task_ids if not units[uid]["ok"]],
            "leakage_assertion_passed": not bad_guard and not any("LeakageError" in e for e in errors),
            "test_fold_use": "none: no refit, no prediction, no unit folder; the test fold is only forbidden",
            "device_names": sorted({units[uid]["device_name"] for uid in done}), "score": BATCH_SCORE,
            "budget_h": budget_h, "seconds": seconds, "inner_scores": scores, "units": units}


def run_batch(arm: str, expect: dict | None = None, root=None, workers: int = 4,
              budget_h: float = BATCH_BUDGET_H, quiet: bool = True, folds: tuple[int, ...] | None = None,
              rerun: int = 0, encoders: tuple[str, ...] | None = None, chosen_ref: dict | None = CHOSEN_REF) -> dict:
    """One arm of the batch-floor check: bc, b8 or b4 (see the section comment above). CPU only.

    Inputs attached: knee-n0, knee-n5, knee-n3-r (as the other arms) and knee-n6a (the chosen
    configurations; only its agg_n6a_<encoder>_f<k>.json files are read). expect: the card's hashes.
    budget_h is the soft stop: after budget_h hours no inner fit starts and a running fit ends with its
    epoch, the session ends cleanly with status "partial", and every finished unit is in the summary
    (agg_<step>_inner.json, rewritten as each unit ends). The watchdog of the step is at 1.1 x budget_h.
    Units are queued encoder by encoder (DINOv2 first), so a stopped session tends to hold one encoder
    whole. A rerun (rerun=1: step n13_<arm>_r) names the folds and encoders of the missing units.
    chosen_ref=None (tests) skips the check against the values read on 25 Nov; the summary says so."""
    if arm not in BATCH_ARMS:
        raise ValueError("unknown batch-floor arm: use bc, b8 or b4")
    encs = tuple(MAIN if encoders is None else encoders)
    if not encs or not set(encs) <= set(MAIN):
        raise ValueError("the arms run the two main encoders (a rerun names the missing ones)")
    r = int(rerun)
    if r < 0 or (r and (not folds or encoders is None)):
        raise ValueError("a rerun names the folds and encoders of the missing units")
    if budget_h is None or float(budget_h) < 0:
        raise ValueError("budget_h is the soft stop in hours: a number, 0 or more")
    name = f"n13_{arm}" + ("" if not r else "_r" if r == 1 else f"_r{r}")
    env = environment()
    cfg = {"name": name, "arm": arm, "rerun": r, "encoders": list(encs), "folds": list(folds) if folds else None,
           "workers": workers, "seed": BATCH_SEED, "budget_h": budget_h,
           "batch_size": batch_settings(arm)["config"]["batch_size"],
           "test_overrides": True, "test_overrides_reason": BATCH_REASON, "added_after_results": True,
           "plan": BATCH_PLAN, "chosen_ref_checked": chosen_ref is not None, "environment": env}
    out: dict = {}
    with common.notebook(name, cfg, quiet=quiet, budget_h=budget_h):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        import torch

        if torch.cuda.device_count():
            raise ValueError("the batch-floor arms run on CPU (accelerator None): the three arms share one device type")
        t0 = time.time()
        d = stage.StageData(root, expect)
        K = d.n_folds("s2")
        fold_list = list(folds) if folds else list(range(K))
        if K != N_OUTER or not set(fold_list) <= set(range(N_OUTER)):
            raise ValueError("the batch-floor check needs the 5 S2 outer folds")
        chosen = chosen_configs(root, encs, fold_list, chosen_ref)
        od = paths.out_dir("private", name, "errors")
        tasks = [{"id": f"{e}_f{k}", "encoder": e, "fold": k, "seed": BATCH_SEED,
                  "eligible": list(range(len(FINDINGS))), "batch_size": BATCH_ARMS[arm],
                  "config_chosen": chosen[(e, k)]["config"], "stage_a_best": chosen[(e, k)]["best_inner_score"],
                  "chosen_sha16": chosen[(e, k)]["sha16"], "deadline": t0 + float(budget_h) * 3600,
                  "dir": str(od / f"{e}_f{k}")}
                 for e in encs for k in fold_list]  # encoder by encoder
        ids = [t["id"] for t in tasks]
        results: dict = {}

        def write(final: bool = False) -> dict:
            s = batch_summary(arm, name, results, ids, env, encs, budget_h=budget_h,
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
            futs = {ex.submit(_batch_task, t): t for t in tasks}
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


def _batch_arm(label: str, summaries) -> dict | None:
    """merged() for a batch-floor arm, with its label and settings checked; None when the arm did not run."""
    if summaries is None:
        return None
    m = merged(summaries)
    if m["arm"] != label:
        raise ValueError(f"decide_batch: the {label} input holds the summary of arm {m['arm']}")
    if m["settings"] != batch_settings(label):
        raise ValueError(f"decide_batch: the {label} summaries do not have the declared settings for {label}")
    want = {(e, k) for e in MAIN for k in range(N_OUTER)}
    if not set(m["units"]) <= want:
        raise ValueError(f"decide_batch: {label} holds a unit outside {', '.join(MAIN)} and folds 0-{N_OUTER - 1}")
    if not all(isinstance(e, dict) for e in m["environments"]):
        raise ValueError(f"decide_batch: a {label} summary has no environment record")
    for (e, k), u in m["units"].items():
        if u.get("ok"):
            v = u.get("inner_score")
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f"decide_batch: {label} unit {e} fold {k} is finished without a finite inner score")
            b, want_b = (u.get("config") or {}).get("batch_size"), BATCH_ARMS[label]
            if b is None or (want_b is None and b != u.get("batch_chosen")) or (want_b is not None and b != want_b):
                raise ValueError(f"decide_batch: {label} unit {e} fold {k} did not use the arm's batch size")
    return m


def _env_differences(envs: list) -> list:
    fields = sorted(set(ENV_FIELDS).union(*(e.keys() for e in envs)))
    return [f for f in fields if len({json.dumps(e.get(f), sort_keys=True) for e in envs}) > 1]


def decide_batch(bc, b8, b4, base0, null, min_gain: float = MIN_GAIN) -> dict:
    """The rule of the batch-floor check (it2b_batch_floor_r1.md), on inner scores only (R4, R5).

    Inputs: each arm's summary, agg_n13_<arm>_inner.json as a dict, or the list [original, rerun, ...]
    (merged()). bc, b8, b4: the three new arms; b8 or b4 may be None (not run), and a new arm may be
    partial. base0, null: iteration 2's summaries, complete, used only for s (they fit the default
    configuration at batch 48, so they are not the base of any difference here).
    Environment: the environments of every bc, b8 and b4 session must be equal field by field (the Kaggle
    image's commit and build date, python, torch, CUDA, cuDNN, device). If not: status "env_mismatch"
    and no verdict; rerun the arms on one image. A difference from iteration 2's environment is only
    reported (same_environment_as_iteration2): s then comes from another image.
    The rule is iteration 2's keep rule with no new number, every comparison at 4 decimals:
      s         = sample SD (n - 1) of the 10 differences null - base0; it must be S_DECLARED at 4 decimals;
      threshold = max(min_gain, 2 s / sqrt(5)), at 4 decimals;
      gain      = mean over the 5 outer folds of (arm - bc), at 4 decimals, per encoder and arm;
      folds_up  = folds with round(arm - bc, 4) > 0;
      verdict   = "the floor mattered" if gain > threshold and folds_up >= 4, else "no support".
    An encoder with fewer than 5 folds finished in the arm and in bc gets "no verdict" for that arm: its
    finished folds are listed with their differences, and no gain is given.
    Also, descriptive: the per-unit table, the units split by chosen batch size (on the floor of 16, or
    above it), and bc minus Stage A's best inner score. The batch sizes used sit under "config" keys."""
    arms = {"bc": _batch_arm("bc", bc), "b8": _batch_arm("b8", b8), "b4": _batch_arm("b4", b4)}
    it2 = {"base0": merged(base0), "null": merged(null)}
    for label, m in it2.items():
        if m["arm"] != label or m["settings"] != settings(arm_spec(label)):
            raise ValueError(f"decide_batch: the {label} input is not iteration 2's {label} arm with its settings")
    S2 = {label: _complete(label, m) for label, m in it2.items()}
    folds = list(range(N_OUTER))
    null_diffs = [S2["null"][e][k] - S2["base0"][e][k] for e in MAIN for k in folds]
    s = float(np.std(null_diffs, ddof=1))
    if round(s, DECIMALS) != round(S_DECLARED, DECIMALS):
        raise ValueError("decide_batch: base0 and null do not give the declared s (0.0030): not iteration 2's summaries")
    threshold_raw = max(float(min_gain), 2 * s / math.sqrt(N_OUTER))
    threshold = round(threshold_raw, DECIMALS)
    ran = {label: m for label, m in arms.items() if m is not None}
    given = [x for a in (bc, b8, b4) if a is not None for x in ([a] if isinstance(a, dict) else list(a))]
    envs = [e for m in ran.values() for e in m["environments"]]
    envs2 = [e for m in it2.values() for e in m["environments"] if isinstance(e, dict)]
    record = {"rule": "it2b batch-floor rule (n13.decide_batch): iteration 2's keep rule, base = the bc arm",
              "plan": BATCH_PLAN, "added_after_results": True,
              "decided_from": "inner validation scores only (R4, R5); no test-fold number exists for these arms",
              "min_gain": float(min_gain), "s": s, "s_declared": S_DECLARED,
              "s_source": "iteration 2: sample SD of the 10 null minus base0 differences (defaults, batch 48, seeds 0 and 100)",
              "threshold": threshold, "threshold_unrounded": threshold_raw,
              "arms": {label: (None if m is None else {
                  "steps": m["steps"], "config": {"batch_size": batch_settings(label)["config"]["batch_size"]},
                  "units_finished": sorted(f"{e}_f{k}" for (e, k), u in m["units"].items() if u.get("ok"))})
                  for label, m in arms.items()},
              "same_knee_commit": len({c for m in ran.values() for c in m["knee_commits"]}) == 1,
              "chosen_ref_checked": all(bool(x.get("chosen_ref_checked")) for x in given),
              "limits": ["one seed (0); one setting changed at the chosen point, not a new search",
                         "the learning rate is not rescaled: a smaller batch is also more optimiser steps (step_ratio)",
                         "four comparisons (2 arms x 2 encoders) against one base, no correction",
                         "s was measured on the default configuration at batch 48, not at batch 8 or 4"]}
    empty = {e: {arm: {"verdict": NO_VERDICT, "reason": None, "gain": None, "threshold": threshold,
                       "folds_up": None, "folds": [], "per_fold": {}} for arm in ("b8", "b4")} for e in MAIN}
    if arms["bc"] is None:
        for e in MAIN:
            for arm in empty[e]:
                empty[e][arm]["reason"] = "the bc arm (the base) did not run"
        return {"status": "no_base", "encoders": empty, **record}
    differ = _env_differences(envs)
    if differ:
        for e in MAIN:
            for arm in empty[e]:
                empty[e][arm]["reason"] = "the arms' environments differ"
        return {"status": "env_mismatch", "env_differences": differ,
                "environments": {label: m["environments"] for label, m in ran.items()},
                "encoders": empty, **record}
    differ2 = _env_differences([envs[0], *envs2]) if envs2 else ["no environment record in iteration 2's summaries"]

    def score(label, e, k):
        u = (arms[label] or {"units": {}})["units"].get((e, k))
        return float(u["inner_score"]) if u and u.get("ok") else None

    def words(arm, e, rec):
        b = BATCH_ARMS[arm]
        if rec["verdict"] == NO_VERDICT:
            return f"Batch {b}, {e}: no verdict ({rec['reason']})."
        tail = f"{e}: gain {rec['gain']:+.4f}, higher in {rec['folds_up']} of 5 folds; threshold {threshold:.4f}"
        if rec["verdict"] == MATTERED:
            return (f"Open edge: batch {b} scored higher than the chosen batch on inner validation ({tail}). "
                    "The head search was not rerun.")
        return f"Batch {b} did not score higher than the chosen batch on inner validation ({tail})."

    out_e: dict = {}
    for e in MAIN:
        out_e[e] = {}
        for arm in ("b8", "b4"):
            have = [k for k in folds if score("bc", e, k) is not None and score(arm, e, k) is not None]
            diffs = {k: score(arm, e, k) - score("bc", e, k) for k in have}
            rec = {"verdict": NO_VERDICT, "reason": None, "gain": None, "threshold": threshold, "folds_up": None,
                   "folds": have, "per_fold": {k: round(x, DECIMALS) for k, x in diffs.items()}}
            if arms[arm] is None:
                rec["reason"] = "the arm did not run"
            elif len(have) < N_OUTER:
                rec["reason"] = "fewer than 5 folds finished in the arm and in bc"
            else:
                gain_raw = float(np.mean([diffs[k] for k in folds]))
                gain = round(gain_raw, DECIMALS)
                up = sum(1 for k in folds if round(diffs[k], DECIMALS) > 0)
                rec.update(gain=gain, gain_unrounded=gain_raw, folds_up=up,
                           verdict=MATTERED if (gain > threshold and up >= MIN_FOLDS_UP) else NO_SUPPORT)
            rec["words"] = words(arm, e, rec)
            out_e[e][arm] = rec

    table, split = {}, {}
    for e in MAIN:
        for k in folds:
            us = {label: (arms[label] or {"units": {}})["units"].get((e, k)) or {} for label in arms}
            us = {label: u if u.get("ok") else {} for label, u in us.items()}
            base = us["bc"].get("inner_score")
            row = {"batch_chosen": next((u["batch_chosen"] for u in us.values() if u.get("batch_chosen")), None),
                   "stage_a_best_inner_score": next((u["stage_a_best_inner_score"] for u in us.values()
                                                     if u.get("stage_a_best_inner_score") is not None), None)}
            for label, u in us.items():
                row[label] = {"inner_score": u.get("inner_score"), "E": u.get("E"),
                              "steps_per_epoch": u.get("steps_per_epoch"), "step_ratio": u.get("step_ratio"),
                              "minus_bc": (round(u["inner_score"] - base, DECIMALS)
                                           if label != "bc" and base is not None and u.get("inner_score") is not None
                                           else None)}
            row["bc_minus_stage_a"] = (round(base - row["stage_a_best_inner_score"], DECIMALS)
                                       if base is not None and row["stage_a_best_inner_score"] is not None else None)
            table[f"{e}_f{k}"] = row
    for arm in ("b8", "b4"):
        split[arm] = {}
        for name, pick in (("chosen_on_floor", lambda b: b == SEARCH_RANGE[0]),
                           ("chosen_above_floor", lambda b: b > SEARCH_RANGE[0])):
            uids = [uid for uid, row in table.items() if row["batch_chosen"] is not None
                    and pick(row["batch_chosen"]) and row[arm]["minus_bc"] is not None]
            vals = [table[uid][arm]["minus_bc"] for uid in uids]
            split[arm][name] = {"units": uids, "mean_minus_bc": round(float(np.mean(vals)), DECIMALS) if vals else None,
                                "units_up": [uid for uid in uids if table[uid][arm]["minus_bc"] > 0]}
    gaps = [row["bc_minus_stage_a"] for row in table.values() if row["bc_minus_stage_a"] is not None]
    verdicts = [out_e[e][arm]["verdict"] for e in MAIN for arm in ("b8", "b4")]
    return {"status": "decided" if NO_VERDICT not in verdicts else "partial",
            "environment": envs[0], "same_environment_as_iteration2": not differ2,
            "env_differences_from_iteration2": differ2, "encoders": out_e, "units": table,
            "by_chosen_batch": {"descriptive": True, "note": "no verdict from this split; for a unit above the floor "
                                "batch 8 or 4 is a larger change", **split},
            "bc_minus_stage_a": {"descriptive": True, "mean": round(float(np.mean(gaps)), DECIMALS) if gaps else None,
                                 "note": "Stage A's score is the best of 30 trials on a T4: favoured by selection"},
            **record}
