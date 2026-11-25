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


# ---- the local decide step of the dropout-ceiling check (declaration r2, sections 4 to 6) ----
# Everything from here on runs outside Kaggle only. The four arms run at DROPOUT_ARM_COMMIT; nothing that
# run_dropout reaches was changed after that commit (tests/test_n13_dropout.py proves it on the text).
DROPOUT_DECLARATION = "it2d_dropout_r2.md: r2, after review r1; the arms run at DROPOUT_ARM_COMMIT, the decide step later"
DROPOUT_ARM_COMMIT = "a88659486b5ba3bb4aeba189fb605899b57aeddd"  # the commit the four Kaggle sessions install
VERDICT_ARMS = ("p6", "p7")  # p5 is descriptive (p5 minus dc) and decides only the words of a "scored higher"
HIGHER, LOWER, NO_DIFF, NO_VERDICT = ("higher dropout scored higher", "higher dropout scored lower",
                                      "no difference shown", "no verdict")
CEILING, INSIDE, NO_P5 = "the ceiling mattered", "inside the searched range", "p5 not available"
RERUN_TOL = 0.0005  # a unit with |dc - bc| above it is listed as rerun noise (a tie at 0.0005 is inside)
SPLIT_AT = 0.4  # descriptive split: units whose chosen dropout is at least 0.4, and the others
WORDS_ENDING = "One seed. The other head settings were not re-tuned. Differences under 0.005 cannot be shown by this check."
WORDS_NOT_RERUN = "The head search was not rerun."
WORDS_NONE = "Raising dropout above the searched range did not score higher by the declared margin."
WORDS_RANGE = "The searched dropout range ended at 0.5 and was not extended."  # in a no-verdict sentence
N_COMPARISONS = len(MAIN) * len(VERDICT_ARMS)  # four verdict comparisons


def _p_text(arm: str) -> str:
    return f"{DROPOUT_ARMS[arm]:g}"


def _higher_line(arm: str, e: str, ceiling: str) -> str:
    """The summary line of one "higher dropout scored higher" (declaration r2, sections 5 and 6)."""
    p, name = _p_text(arm), n13.ENC_WORDS[e]
    if ceiling == CEILING:  # arm minus dc and arm minus p5 both meet the rule
        return f"Open point: the ceiling mattered: dropout {p} scored higher than the chosen dropout and than 0.5 ({name})."
    if ceiling == INSIDE:  # arm minus dc meets the rule, arm minus p5 does not
        return f"Dropout {p} scored higher than the chosen dropout but not than 0.5, inside the searched range ({name})."
    return (f"Open point: dropout {p} scored higher than the chosen dropout ({name}). This check does not show "
            "whether the range end or the search's pick inside the range is the cause.")


def _summary_lines(recs: dict) -> list:
    """The summary lines of the slide (declaration r2, section 6) from {(encoder, arm): comparison record}.
    They never name a dropout as the one to use."""
    out = [_higher_line(arm, e, recs[(e, arm)]["ceiling"]) for e in MAIN for arm in VERDICT_ARMS
           if recs[(e, arm)]["verdict"] == HIGHER]
    k = sum(1 for r in recs.values() if r["verdict"] == NO_VERDICT)
    if not out and not k:
        out.append(WORDS_NONE)
    elif not out:
        out.append(f"No comparison with a verdict scored higher by the declared margin; {k} of {N_COMPARISONS} "
                   "comparisons have no verdict.")
    return out


def _dropout_arm(label: str, summaries) -> dict | None:
    """n13.merged() for a dropout-ceiling arm, with its label, settings and dropout checked; None when it did not run."""
    if summaries is None:
        return None
    m = n13.merged(summaries)
    if m["arm"] != label:
        raise ValueError(f"decide_dropout: the {label} input holds the summary of arm {m['arm']}")
    if m["settings"] != dropout_settings(label):
        raise ValueError(f"decide_dropout: the {label} summaries do not have the declared settings for {label}")
    want = {(e, k) for e in MAIN for k in range(n13.N_OUTER)}
    if not set(m["units"]) <= want:
        raise ValueError(f"decide_dropout: {label} holds a unit outside {', '.join(MAIN)} and folds 0-{n13.N_OUTER - 1}")
    if not all(isinstance(e, dict) for e in m["environments"]):
        raise ValueError(f"decide_dropout: a {label} summary has no environment record")
    for (e, k), u in m["units"].items():
        if u.get("ok"):
            v = u.get("inner_score")
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f"decide_dropout: {label} unit {e} fold {k} is finished without a finite inner score")
            c = u.get("config") or {}
            if c.get("batch_size") is None or c.get("batch_size") != u.get("batch_chosen"):
                raise ValueError(f"decide_dropout: {label} unit {e} fold {k} did not use its chosen batch size")
            dr = (u.get("dropout") or {}).get("config") or {}
            chosen, used = dr.get("dropout_chosen"), dr.get("dropout_used")
            if chosen is None or used is None or c.get("dropout") is None:
                raise ValueError(f"decide_dropout: {label} unit {e} fold {k} does not record its dropout")
            if round(float(chosen), 3) != DROPOUT_REF[e][k]:
                raise ValueError(f"decide_dropout: {label} unit {e} fold {k} is not the unit read on 25 Nov (dropout)")
            p = DROPOUT_ARMS[label]
            if float(used) != float(c["dropout"]) or float(used) != (float(chosen) if p is None else float(p)):
                raise ValueError(f"decide_dropout: {label} unit {e} fold {k} did not use the arm's dropout")
    return m


def _same_but_dropout(arms: dict) -> None:
    """Per unit, every arm's configuration equals dc's except the dropout (and the chosen dropout is the same)."""
    base = arms["dc"]
    for label, m in arms.items():
        if m is None or label == "dc" or base is None:
            continue
        for key, u in m["units"].items():
            b = base["units"].get(key)
            if not (u.get("ok") and b and b.get("ok")):
                continue
            strip = [{k: v for k, v in x["config"].items() if k != "dropout"} for x in (u, b)]
            chosen = [x["dropout"]["config"]["dropout_chosen"] for x in (u, b)]
            if strip[0] != strip[1] or chosen[0] != chosen[1] or u["batch_chosen"] != b["batch_chosen"]:
                raise ValueError(f"decide_dropout: {label} unit {key[0]} fold {key[1]} differs from dc in more than the dropout")


def decide_dropout(dc, p6, p7, bc, base0, null, p5=None, min_gain: float = n13.MIN_GAIN,
                   arm_commit: str | None = DROPOUT_ARM_COMMIT, replaced: dict | None = None) -> dict:
    """The rule of the dropout-ceiling check (it2d_dropout_r2.md, section 5), on inner scores only (R4, R5).

    Inputs: each arm's summary, agg_n13_<arm>_inner.json as a dict, or the list [original, rerun, ...]
    (n13.merged()). dc is the base. p6, p7 and p5 may be None (not run) and any arm may be partial; a
    missing p5 never blocks a verdict. bc: the batch-floor arm's summary (it ran at commit 39c3c6f), used only
    for the rerun check; None leaves it empty. base0, null: iteration 2's summaries, complete, used only for s.
    Enforced on every dc, p5, p6 and p7 summary given (reruns too), in this order, each with its own status
    and no verdict:
      "no_base"                 dc did not run;
      "leakage_check_failed"    a summary's leakage_assertion_passed is not true;
      "chosen_ref_not_checked"  a summary's chosen_ref_checked is not true (not the 25 Nov configurations);
      "commit_mismatch"         the summaries hold more than one knee commit, or not arm_commit
                                (default DROPOUT_ARM_COMMIT; None skips only the comparison with arm_commit);
      "env_mismatch"            the environments differ in any field.
    The rule, every comparison at 4 decimals:
      s         = sample SD (n - 1) of the 10 differences null - base0; it must be 0.0030 at 4 decimals;
      threshold = max(min_gain, 2 s / sqrt(5)), at 4 decimals (0.005);
      gain      = mean over the 5 outer folds of (arm - dc), at 4 decimals, per encoder and arm (p6, p7);
      lower / higher = the folds with round(arm - dc, 4) < 0 / > 0 (a fold at 0.0000 is neither);
      verdict   = "higher dropout scored higher" if gain > threshold and higher in at least 4 folds;
                  "higher dropout scored lower" if gain < -threshold and lower in at least 4 folds;
                  else "no difference shown" (a tie at 0.0050 is inside).
    An encoder with fewer than 5 folds finished in the arm and in dc gets "no verdict" for that arm.
    A "scored higher" is then compared with p5 by the same rule (arm minus p5): "the ceiling mattered" when
    that meets the rule too, "inside the searched range" when it does not, "p5 not available" when p5 has
    fewer than 5 folds for the encoder. These words change no verdict.
    Descriptive, never in a verdict: dc minus bc per unit (rerun_check; units above RERUN_TOL are listed and
    the slide says so), p5 minus dc per encoder, the split by chosen dropout (at least 0.4 and the others),
    the per-unit table, the mean inner score and the median best epoch per arm.
    replaced: {arm: the summaries of a whole arm that was rerun after an environment mismatch}; they are
    only reported (replaced_runs, R10). The dropout values sit under "config" keys."""
    given_in = {"dc": dc, "p5": p5, "p6": p6, "p7": p7}
    arms = {label: _dropout_arm(label, x) for label, x in given_in.items()}
    _same_but_dropout(arms)
    m_bc = n13._batch_arm("bc", bc)
    it2 = {"base0": n13.merged(base0), "null": n13.merged(null)}
    for label, m in it2.items():
        if m["arm"] != label or m["settings"] != n13.settings(n13.arm_spec(label)):
            raise ValueError(f"decide_dropout: the {label} input is not iteration 2's {label} arm with its settings")
    S2 = {label: n13._complete(label, m) for label, m in it2.items()}
    folds = list(range(n13.N_OUTER))
    null_diffs = [S2["null"][e][k] - S2["base0"][e][k] for e in MAIN for k in folds]
    s = float(np.std(null_diffs, ddof=1))
    if round(s, n13.DECIMALS) != round(n13.S_DECLARED, n13.DECIMALS):
        raise ValueError("decide_dropout: base0 and null do not give the declared s (0.0030): not iteration 2's summaries")
    threshold_raw = max(float(min_gain), 2 * s / math.sqrt(n13.N_OUTER))
    threshold = round(threshold_raw, n13.DECIMALS)
    ran = {label: m for label, m in arms.items() if m is not None}
    given = [x for a in given_in.values() if a is not None for x in ([a] if isinstance(a, dict) else list(a))]
    envs = [e for m in ran.values() for e in m["environments"]]
    commits = sorted({str(c) for m in ran.values() for c in m["knee_commits"]})
    old_runs = {}
    for label, summaries in sorted((replaced or {}).items()):
        if label not in DROPOUT_ARMS:
            raise ValueError("decide_dropout: replaced names the arms dc, p5, p6 or p7")
        m = _dropout_arm(label, summaries)
        if m is None:
            continue
        old_runs[label] = {
            "steps": m["steps"], "environments": m["environments"], "knee_commits": m["knee_commits"],
            "inner_scores": {f"{e}_f{k}": u["inner_score"] for (e, k), u in sorted(m["units"].items()) if u.get("ok")},
            "why": "environment mismatch; the arm to rerun was chosen from the environment fields only",
            "use": "reported only (R10); the rerun replaces the arm whole in the rule"}
    record = {"rule": "it2d dropout-ceiling rule (n13d.decide_dropout): iteration 2's margin, base = the dc arm",
              "plan": DROPOUT_PLAN, "declaration": DROPOUT_DECLARATION, "added_after_results": True,
              "decided_from": "inner validation scores only (R4, R5); no test-fold number exists for these arms",
              "min_gain": float(min_gain), "s": s, "s_declared": n13.S_DECLARED,
              "s_source": "iteration 2: sample SD of the 10 null minus base0 differences (defaults, dropout 0.25, seeds 0 and 100)",
              "threshold": threshold, "threshold_unrounded": threshold_raw,
              "verdict_arms": list(VERDICT_ARMS), "descriptive_arms": ["p5"], "search_ceiling": SEARCH_CEILING,
              "arms": {label: (None if m is None else {
                  "steps": m["steps"], "config": dropout_config(label),
                  "units_finished": sorted(f"{e}_f{k}" for (e, k), u in m["units"].items() if u.get("ok"))})
                  for label, m in arms.items()},
              "same_knee_commit": len(commits) == 1, "knee_commits": commits, "arm_commit_declared": arm_commit,
              "knee_commit_is_declared": None if arm_commit is None else commits == [arm_commit],
              "chosen_ref_checked": all(x.get("chosen_ref_checked") is True for x in given),
              "leakage_assertion_passed": all(x.get("leakage_assertion_passed") is True for x in given),
              "replaced_runs": old_runs,
              "limits": ["one seed (0); one setting changed at the chosen point, not a new search",
                         "a fixed 0.6 or 0.7 is a larger step for units that chose 0 than for units that chose 0.5",
                         "the other five head settings were tuned at the chosen dropout and not re-tuned: this favours it",
                         "differences under 0.005 cannot be shown",
                         "four verdict comparisons (p6, p7 x 2 encoders) plus the p5 comparisons, no correction",
                         "s was measured at dropout 0.25; noise at 0.7 may be larger (a difference is called too easily)"]}

    def score(m, e, k):
        u = (m or {"units": {}})["units"].get((e, k))
        return float(u["inner_score"]) if u and u.get("ok") else None

    def finished(label, e):
        return sum(1 for k in folds if score(arms[label], e, k) is not None)

    def no_verdict_words(arm, e, reason) -> tuple:
        """(outcome, extra, own ending) of a no-verdict comparison (declaration r2, section 6)."""
        return (f"Dropout {_p_text(arm)}, {n13.ENC_WORDS[e]}: no verdict ({reason}).", WORDS_RANGE,
                f"Finished: dc {finished('dc', e)} of 5 folds, {arm} {finished(arm, e)} of 5 folds.")

    def stop(status: str, reason: str, **extra) -> dict:
        enc = {e: {arm: dict(zip(("words", "words_extra", "words_limits"), no_verdict_words(arm, e, reason)),
                             verdict=NO_VERDICT, reason=reason, gain=None, threshold=threshold, folds_up=None,
                             lower={"folds": []}, higher={"folds": []}, folds=[], per_fold={}, ceiling=None,
                             minus_p5=None)
                   for arm in VERDICT_ARMS} for e in MAIN}
        recs = {(e, arm): enc[e][arm] for e in MAIN for arm in VERDICT_ARMS}
        return {"status": status, **extra, "encoders": enc, "summary_words": _summary_lines(recs),
                "words_ending": None, **record}

    if arms["dc"] is None:
        return stop("no_base", "the dc arm (the base) did not run")
    if not record["leakage_assertion_passed"]:
        return stop("leakage_check_failed", "an arm's leakage check did not pass",
                    leakage_failed_steps=[x.get("step") for x in given if x.get("leakage_assertion_passed") is not True])
    if not record["chosen_ref_checked"]:
        return stop("chosen_ref_not_checked", "an arm did not check its configurations against 25 Nov",
                    chosen_ref_unchecked_steps=[x.get("step") for x in given if x.get("chosen_ref_checked") is not True])
    if len(commits) != 1 or record["knee_commit_is_declared"] is False:
        return stop("commit_mismatch", "the arms did not run at one knee commit" if len(commits) != 1
                    else "the arms did not run at the declared knee commit",
                    knee_commits_by_step={str(x.get("step")): str(x.get("knee_commit")) for x in given})
    differ = n13._env_differences(envs)
    if differ:
        return stop("env_mismatch", "the arms' environments differ", env_differences=differ,
                    environments={label: m["environments"] for label, m in ran.items()})

    def compare(a, b, e) -> dict:
        """a minus b for one encoder: the folds both finished, per fold and mean at 4 decimals, and the rule."""
        have = [k for k in folds if score(a, e, k) is not None and score(b, e, k) is not None]
        diffs = {k: score(a, e, k) - score(b, e, k) for k in have}
        per = {k: round(x, n13.DECIMALS) + 0.0 for k, x in diffs.items()}
        out = {"folds": have, "per_fold": per, "lower": {"folds": [k for k in have if per[k] < 0]},
               "higher": {"folds": [k for k in have if per[k] > 0]}, "all_folds": len(have) == n13.N_OUTER,
               "gain": None, "meets_higher": None, "meets_lower": None}
        if have:
            raw = float(np.mean([diffs[k] for k in have]))
            out["gain"] = round(raw, n13.DECIMALS) + 0.0  # + 0.0: no -0.0
            out["gain_unrounded"] = raw
        if out["all_folds"]:
            out["meets_higher"] = out["gain"] > threshold and len(out["higher"]["folds"]) >= n13.MIN_FOLDS_UP
            out["meets_lower"] = out["gain"] < -threshold and len(out["lower"]["folds"]) >= n13.MIN_FOLDS_UP
        return out

    def words(arm, e, rec) -> tuple:
        """(outcome, extra, own ending): one sentence set per encoder and arm (declaration r2, section 6).
        Only a no-verdict set has its own ending; the common ending is given once per slide (words_ending)."""
        if rec["verdict"] == NO_VERDICT:
            return no_verdict_words(arm, e, rec["reason"])
        g, p, name = rec["gain"], _p_text(arm), n13.ENC_WORDS[e]
        if g == 0:  # at 4 decimals
            return f"Dropout {p} scored the same as the chosen dropout on inner validation ({name}).", None, None
        side = "lower" if g < 0 else "higher"
        outcome = (f"Dropout {p} scored {abs(g):.4f} {side} than the chosen dropout on inner validation "
                   f"({name}, {side} in {len(rec[side]['folds'])} of 5 folds).")
        if rec["verdict"] == HIGHER:
            return outcome, WORDS_NOT_RERUN, None
        if rec["verdict"] == LOWER:
            return outcome, None, None
        return outcome, (f"This does not meet the declared rule (more than {threshold:g} at 4 decimals, and "
                         f"{n13.MIN_FOLDS_UP} of 5 folds), so it is not counted as an effect."), None

    out_e: dict = {}
    for e in MAIN:
        out_e[e] = {}
        for arm in VERDICT_ARMS:
            c = compare(arms[arm], arms["dc"], e)
            rec = {"verdict": NO_VERDICT, "reason": None, "gain": None, "threshold": threshold, "folds_up": None,
                   "lower": c["lower"], "higher": c["higher"], "folds": c["folds"], "per_fold": c["per_fold"],
                   "ceiling": None, "minus_p5": None}
            if arms[arm] is None:
                rec["reason"] = "the arm did not run"
            elif not c["all_folds"]:
                rec["reason"] = "fewer than 5 folds finished in the arm and in dc"
            else:
                verdict = HIGHER if c["meets_higher"] else LOWER if c["meets_lower"] else NO_DIFF
                rec.update(gain=c["gain"], gain_unrounded=c["gain_unrounded"], folds_up=len(c["higher"]["folds"]),
                           verdict=verdict)
                if arms["p5"] is not None:  # descriptive: arm minus p5, by the same rule (words only)
                    rec["minus_p5"] = compare(arms[arm], arms["p5"], e)
                if verdict == HIGHER:
                    v5 = rec["minus_p5"]
                    rec["ceiling"] = (NO_P5 if v5 is None or not v5["all_folds"]
                                      else CEILING if v5["meets_higher"] else INSIDE)
            rec["words"], rec["words_extra"], rec["words_limits"] = words(arm, e, rec)
            out_e[e][arm] = rec

    # descriptive from here on: nothing below enters a verdict
    p5_minus_dc = {"descriptive": True, "used_in_rule": False, "p5_ran": arms["p5"] is not None,
                   "what": "p5 (dropout 0.5 in every unit, the end of the searched range) minus dc, per encoder",
                   **{e: compare(arms["p5"], arms["dc"], e) for e in MAIN}}
    table, gaps = {}, {}
    for e in MAIN:
        for k in folds:
            uid = f"{e}_f{k}"
            base, old = score(arms["dc"], e, k), score(m_bc, e, k)
            row = {"config": {"dropout_chosen": None}, "batch_chosen": None}
            for label, m in arms.items():
                u = ((m or {"units": {}})["units"].get((e, k)) or {})
                u = u if u.get("ok") else {}
                if u:
                    row["config"]["dropout_chosen"] = u["dropout"]["config"]["dropout_chosen"]
                    row["batch_chosen"] = row["batch_chosen"] or u.get("batch_chosen")
                row[label] = {"inner_score": u.get("inner_score"), "E": u.get("E"),
                              "minus_dc": (round(u["inner_score"] - base, n13.DECIMALS) + 0.0
                                           if label != "dc" and base is not None and u.get("inner_score") is not None
                                           else None)}
            row["bc_inner_score"] = old
            row["dc_minus_bc"] = round(base - old, n13.DECIMALS) + 0.0 if base is not None and old is not None else None
            if row["dc_minus_bc"] is not None:
                gaps[uid] = row["dc_minus_bc"]
            table[uid] = row
    outside = [uid for uid, v in gaps.items() if abs(v) > RERUN_TOL]  # at 4 decimals: 0.0005 is inside
    rerun = {"descriptive": True, "used_in_rule": False, "tolerance": RERUN_TOL,
             "what": "dc minus bc per unit: the same fits (configuration, seeds, rows, dropout) at two commits and sessions",
             "bc_given": m_bc is not None,
             "bc_steps": None if m_bc is None else m_bc["steps"],
             "bc_knee_commits": None if m_bc is None else sorted({str(c) for c in m_bc["knee_commits"]}),
             "bc_env_differences": (None if m_bc is None else
                                    n13._env_differences([envs[0], *[x for x in m_bc["environments"] if isinstance(x, dict)]])),
             "units_checked": sorted(gaps), "per_unit": gaps,
             "max_abs_difference": max(abs(v) for v in gaps.values()) if gaps else None,
             "identical": bool(gaps) and all(v == 0 for v in gaps.values()),
             "units_outside": outside, "units_outside_difference": {uid: gaps[uid] for uid in outside},
             "consequence": "none for the rule: the verdict is still given, because all arms share one commit and image",
             "words": (f"Rerun noise: the chosen-dropout arm (dc) differed from the earlier run of the same fits (arm bc) "
                       f"by more than {RERUN_TOL:g} in {len(outside)} of {len(gaps)} units (largest difference "
                       f"{max(abs(gaps[u]) for u in outside):.4f}).") if outside else None}

    def group(label, e, high: bool) -> dict:
        units = [(enc, k) for enc in (MAIN if e is None else (e,)) for k in folds
                 if table[f"{enc}_f{k}"]["config"]["dropout_chosen"] is not None
                 and (table[f"{enc}_f{k}"]["config"]["dropout_chosen"] >= SPLIT_AT) == high]
        have = [(enc, k) for enc, k in units if score(arms[label], enc, k) is not None and score(arms["dc"], enc, k) is not None]
        return {"units": [f"{enc}_f{k}" for enc, k in units], "units_compared": [f"{enc}_f{k}" for enc, k in have],
                "mean_minus_dc": (round(float(np.mean([score(arms[label], enc, k) - score(arms["dc"], enc, k)
                                                       for enc, k in have])), n13.DECIMALS) + 0.0 if have else None)}

    split = {"descriptive": True, "used_in_rule": False, "config": {"split_at": SPLIT_AT},
             "what": "arm minus dc, mean over the units whose chosen dropout is at least 0.4 (high) and over the others (rest)",
             **{label: {key: {"high": group(label, e, True), "rest": group(label, e, False)}
                        for key, e in (("all", None), *((enc, enc) for enc in MAIN))}
                for label in ("p5", "p6", "p7")}}

    def over_units(label, e, field, fn):
        vals = [u[field] for (enc, _), u in (arms[label] or {"units": {}})["units"].items()
                if (e is None or enc == e) and u.get("ok") and u.get(field) is not None]
        return float(fn(vals)) if vals else None

    best_epoch = {label: {"all": over_units(label, None, "E", np.median),
                          **{e: over_units(label, e, "E", np.median) for e in MAIN}} for label in arms}
    means = {e: {label: (round(float(np.mean([score(m, e, k) for k in folds])), n13.DECIMALS)
                         if all(score(m, e, k) is not None for k in folds) else None)
                 for label, m in arms.items()} for e in MAIN}
    recs = {(e, arm): out_e[e][arm] for e in MAIN for arm in VERDICT_ARMS}
    with_verdict = any(r["verdict"] != NO_VERDICT for r in recs.values())
    return {"status": "decided" if with_verdict and all(r["verdict"] != NO_VERDICT for r in recs.values()) else "partial",
            "environment": envs[0], "encoders": out_e, "summary_words": _summary_lines(recs),
            "words_rerun": rerun["words"] if with_verdict else None,
            "words_ending": WORDS_ENDING if with_verdict else None,
            "units": table, "mean_inner_score": means,
            "median_best_epoch": {"descriptive": True, "what": "median over the arm's finished units of the unit's E "
                                  "(the median best epoch of its inner fits)", **best_epoch},
            "p5_minus_dc": p5_minus_dc, "by_chosen_dropout": split,
            "dc_reruns_bc": rerun["identical"], "rerun_check": rerun, **record}


def dropout_slide_sentences(record: dict) -> list:
    """The slide sentences of a decide_dropout record: one per encoder and arm (outcome, extra, and for a
    no-verdict set its own ending), then the summary line(s), then the rerun-noise sentence when a dc unit
    is outside the tolerance, then the common ending once, as the last line. With no verdict in any of the
    four comparisons the summary line is the only line after them: no rerun-noise line, no common ending."""
    out = []
    for e in MAIN:
        for arm in VERDICT_ARMS:
            rec = record["encoders"][e][arm]
            out.append(" ".join(p for p in (rec.get("words"), rec.get("words_extra"), rec.get("words_limits")) if p))
    out += list(record.get("summary_words") or [])
    for extra in (record.get("words_rerun"), record.get("words_ending")):
        if extra:
            out.append(extra)
    return out
