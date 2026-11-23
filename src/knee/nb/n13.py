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
    (trial 0 is then the same configuration), with one unit per main encoder and fold."""
    b = _complete("base0", merged(base0))
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
    """The written decision: decide()'s output as a dict, its JSON text, or the path of its JSON file."""
    if decision is None:
        raise ValueError("dev needs the written A1 decision (decide()'s output, its JSON text or its file): R5")
    if isinstance(decision, (str, Path)):
        text = str(decision)
        decision = json.loads(text) if text.lstrip().startswith("{") else json.loads(Path(text).read_text())
    ok = isinstance(decision, dict) and decision.get("status") in ("decided", "env_mismatch")
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

    decision: decide()'s output (a dict, its JSON text or the path of its JSON file); refused without it.
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
