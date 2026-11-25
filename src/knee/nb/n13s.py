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


# ---- the local decide step of the slice-count check (declaration r3, sections 4 to 7; code review r1) ----
# Everything from here on runs outside Kaggle only. The five arms run at SLICE_ARM_COMMIT; nothing that
# run_slices reaches was changed after that commit (tests/test_n13_slices.py proves it on the text).
SLICE_DECLARATION = "it2c_slice_count_r3.md: r3 (the arm commit, the summary lines); the arms run at SLICE_ARM_COMMIT, the decide step later"
SLICE_ARM_COMMIT = "ecf264fcc11b780e53940b29de86e5a0fc512ae2"  # the commit the five Kaggle sessions install
VERDICT_ARMS = ("s16", "s8", "s4")  # s16b is descriptive only: never a verdict
COST, HIGHER, NO_DIFF, NO_VERDICT = "fewer slices cost", "fewer slices scored higher", "no difference shown", "no verdict"
RERUN_TOL = 0.0005  # a unit with |s32 - bc| above it is listed as rerun noise (a tie at 0.0005 is inside)
WORDS_ENDING = ("One seed. Head settings were tuned at 32 slices and not re-tuned. More than 32 slices was not "
                "tested: it needs a new cache. Differences under 0.005 cannot be shown by this check.")
WORDS_NOT_CHANGED = "The models were not changed."
WORDS_FIXED = "The slice count was fixed at 32 and not searched."
WORDS_NONE = ("No count from 4 to 32 differed by the declared margin. This check does not show that 32 is needed "
              "and does not show that fewer is enough.")
WORDS_NO_CACHE = "More than 32 slices was not tested: it needs a new cache."  # the ending of a no-verdict sentence set
WORDS_EDGE = "32 is the largest count tested, so the curve may still rise beyond it."
N_COMPARISONS = len(MAIN) * len(VERDICT_ARMS)  # six verdict comparisons


def _counts_text(counts) -> str:
    """Counts, largest first: "16", "16 and 4", "16, 8 and 4"."""
    c = [str(x) for x in sorted(set(counts), reverse=True)]
    return c[0] if len(c) == 1 else ", ".join(c[:-1]) + " and " + c[-1]


def _summary_lines(verdicts: dict) -> list:
    """The summary lines of the slide (declaration r3, section 7; code review r1, S1 and S2), from
    {(encoder, arm): verdict}. They never name a count as the one to use."""
    out = []
    for e in MAIN:
        cost = [SLICE_ARMS[arm] for arm in VERDICT_ARMS if verdicts[(e, arm)] == COST]
        if cost:
            out.append(f"Fewer slices cost at {_counts_text(cost)} ({n13.ENC_WORDS[e]}). {WORDS_EDGE}")
    any_cost = bool(out)
    for e in MAIN:
        for arm in VERDICT_ARMS:
            if verdicts[(e, arm)] == HIGHER:
                out.append(f"Open point: with {SLICE_ARMS[arm]} of the 32 cached slices per plane the head scored "
                           f"higher ({n13.ENC_WORDS[e]}). {WORDS_NOT_CHANGED}")
    k = sum(1 for v in verdicts.values() if v == NO_VERDICT)
    if all(v == NO_DIFF for v in verdicts.values()):
        out.append(WORDS_NONE)
    elif k and not out and k < N_COMPARISONS:
        out.append(f"No comparison with a verdict differed by the declared margin; {k} of {N_COMPARISONS} "
                   "comparisons have no verdict.")
    elif k:
        out.append(f"{k} of {N_COMPARISONS} comparisons have no verdict.")
    return out


def _slice_arm(label: str, summaries) -> dict | None:
    """n13.merged() for a slice-count arm, with its label and settings checked; None when the arm did not run."""
    if summaries is None:
        return None
    m = n13.merged(summaries)
    if m["arm"] != label:
        raise ValueError(f"decide_slices: the {label} input holds the summary of arm {m['arm']}")
    if m["settings"] != slice_settings(label):
        raise ValueError(f"decide_slices: the {label} summaries do not have the declared settings for {label}")
    want = {(e, k) for e in MAIN for k in range(n13.N_OUTER)}
    if not set(m["units"]) <= want:
        raise ValueError(f"decide_slices: {label} holds a unit outside {', '.join(MAIN)} and folds 0-{n13.N_OUTER - 1}")
    if not all(isinstance(e, dict) for e in m["environments"]):
        raise ValueError(f"decide_slices: a {label} summary has no environment record")
    for (e, k), u in m["units"].items():
        if u.get("ok"):
            v = u.get("inner_score")
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f"decide_slices: {label} unit {e} fold {k} is finished without a finite inner score")
            b = (u.get("config") or {}).get("batch_size")
            if b is None or b != u.get("batch_chosen"):
                raise ValueError(f"decide_slices: {label} unit {e} fold {k} did not use its chosen batch size")
            if (u.get("slices") or {}).get("config") != slice_config(label):
                raise ValueError(f"decide_slices: {label} unit {e} fold {k} did not read the arm's slices")
    return m


def decide_slices(s32, s16, s8, s4, bc, base0, null, s16b=None, min_gain: float = n13.MIN_GAIN,
                  arm_commit: str | None = SLICE_ARM_COMMIT, replaced: dict | None = None) -> dict:
    """The rule of the slice-count check (it2c_slice_count_r2.md, section 5), on inner scores only (R4, R5).

    Inputs: each arm's summary, agg_n13_<arm>_inner.json as a dict, or the list [original, rerun, ...]
    (n13.merged()). s32 is the base. s16, s8, s4 and s16b may be None (not run) and any arm may be partial.
    bc: the batch-floor arm's summary (it ran at commit 39c3c6f), used only for the rerun check; None
    leaves the rerun check empty. base0, null: iteration 2's summaries, complete, used only for s.
    Enforced on every s32, s16, s16b, s8 and s4 summary given (reruns too), in this order, each with its
    own status and no verdict:
      "no_base"                 s32 did not run;
      "leakage_check_failed"    a summary's leakage_assertion_passed is not true;
      "chosen_ref_not_checked"  a summary's chosen_ref_checked is not true (not the 25 Nov configurations);
      "commit_mismatch"         the summaries hold more than one knee commit, or not arm_commit
                                (default SLICE_ARM_COMMIT; None skips only the comparison with arm_commit);
      "env_mismatch"            the environments differ in any field.
    The rule, every comparison at 4 decimals:
      s         = sample SD (n - 1) of the 10 differences null - base0; it must be 0.0030 at 4 decimals;
      threshold = max(min_gain, 2 s / sqrt(5)), at 4 decimals (0.005);
      diff      = mean over the 5 outer folds of (arm - s32), at 4 decimals, per encoder and arm (s16, s8, s4);
      lower / higher = the folds with round(arm - s32, 4) < 0 / > 0 (a fold at 0.0000 is neither);
      verdict   = "fewer slices cost" if diff < -threshold and lower in at least 4 folds;
                  "fewer slices scored higher" if diff > threshold and higher in at least 4 folds;
                  else "no difference shown" (a tie at 0.0050 is inside).
    An encoder with fewer than 5 folds finished in the arm and in s32 gets "no verdict" for that arm.
    Sentences per encoder and arm: words (the outcome), words_extra and words_limits (only a no-verdict set
    has its own ending); summary_words (the summary lines) and words_ending (the common ending, given once
    per slide when at least one comparison has a verdict); slice_slide_sentences() joins them.
    Descriptive, never in a verdict: s32 minus bc per unit (rerun_check; units above RERUN_TOL are listed),
    s16b minus s16 and s16b minus s32 per encoder (position_effect), the per-unit table, the mean inner
    score and the median best epoch per arm.
    replaced: {arm: the summaries of a whole arm that was rerun after an environment mismatch}; they are
    only reported (replaced_runs, R10). The slices read sit under "config" keys."""
    given_in = {"s32": s32, "s16": s16, "s16b": s16b, "s8": s8, "s4": s4}
    arms = {label: _slice_arm(label, x) for label, x in given_in.items()}
    m_bc = n13._batch_arm("bc", bc)
    it2 = {"base0": n13.merged(base0), "null": n13.merged(null)}
    for label, m in it2.items():
        if m["arm"] != label or m["settings"] != n13.settings(n13.arm_spec(label)):
            raise ValueError(f"decide_slices: the {label} input is not iteration 2's {label} arm with its settings")
    S2 = {label: n13._complete(label, m) for label, m in it2.items()}
    folds = list(range(n13.N_OUTER))
    null_diffs = [S2["null"][e][k] - S2["base0"][e][k] for e in MAIN for k in folds]
    s = float(np.std(null_diffs, ddof=1))
    if round(s, n13.DECIMALS) != round(n13.S_DECLARED, n13.DECIMALS):
        raise ValueError("decide_slices: base0 and null do not give the declared s (0.0030): not iteration 2's summaries")
    threshold_raw = max(float(min_gain), 2 * s / math.sqrt(n13.N_OUTER))
    threshold = round(threshold_raw, n13.DECIMALS)
    ran = {label: m for label, m in arms.items() if m is not None}
    given = [x for a in given_in.values() if a is not None for x in ([a] if isinstance(a, dict) else list(a))]
    envs = [e for m in ran.values() for e in m["environments"]]
    commits = sorted({str(c) for m in ran.values() for c in m["knee_commits"]})
    old_runs = {}
    for label, summaries in sorted((replaced or {}).items()):
        if label not in SLICE_ARMS:
            raise ValueError("decide_slices: replaced names the arms s32, s16, s16b, s8 or s4")
        m = _slice_arm(label, summaries)
        if m is None:
            continue
        old_runs[label] = {
            "steps": m["steps"], "environments": m["environments"], "knee_commits": m["knee_commits"],
            "inner_scores": {f"{e}_f{k}": u["inner_score"] for (e, k), u in sorted(m["units"].items()) if u.get("ok")},
            "why": "environment mismatch; the arm to rerun was chosen from the environment fields only",
            "use": "reported only (R10); the rerun replaces the arm whole in the rule"}
    record = {"rule": "it2c slice-count rule (n13s.decide_slices): iteration 2's margin, base = the s32 arm",
              "plan": SLICE_PLAN, "declaration": SLICE_DECLARATION, "added_after_results": True,
              "decided_from": "inner validation scores only (R4, R5); no test-fold number exists for these arms",
              "min_gain": float(min_gain), "s": s, "s_declared": n13.S_DECLARED,
              "s_source": "iteration 2: sample SD of the 10 null minus base0 differences (defaults, 32 slices, seeds 0 and 100)",
              "threshold": threshold, "threshold_unrounded": threshold_raw,
              "verdict_arms": list(VERDICT_ARMS), "descriptive_arms": ["s16b"],
              "arms": {label: (None if m is None else {
                  "steps": m["steps"], "config": slice_config(label),
                  "units_finished": sorted(f"{e}_f{k}" for (e, k), u in m["units"].items() if u.get("ok"))})
                  for label, m in arms.items()},
              "same_knee_commit": len(commits) == 1, "knee_commits": commits, "arm_commit_declared": arm_commit,
              "knee_commit_is_declared": None if arm_commit is None else commits == [arm_commit],
              "chosen_ref_checked": all(x.get("chosen_ref_checked") is True for x in given),
              "leakage_assertion_passed": all(x.get("leakage_assertion_passed") is True for x in given),
              "replaced_runs": old_runs,
              "limits": ["one seed (0); one setting changed at the chosen point, not a new search",
                         "of the 32 cached slices: every second (fourth, eighth) cached position, not a new cache",
                         "head settings were tuned at 32 slices and not re-tuned: this favours 32",
                         "more than 32 slices was not tested: it needs a new cache and new encoder passes",
                         "differences under 0.005 cannot be shown; the check cannot show that a smaller count is as good",
                         "six verdict comparisons (3 arms x 2 encoders) against one base, no correction",
                         "s was measured at 32 slices; it may understate noise at 4 slices (a difference is called too easily)"]}

    def stop(status: str, reason: str, **extra) -> dict:
        ran_words = f"Status: {status}. Arms given: {', '.join(ran) if ran else 'none'}."  # what ran
        enc = {e: {arm: {"verdict": NO_VERDICT, "reason": reason, "diff": None, "threshold": threshold,
                         "folds_up": None, "lower": {"folds": []}, "higher": {"folds": []}, "folds": [], "per_fold": {},
                         "words": f"{SLICE_ARMS[arm]} slices, {n13.ENC_WORDS[e]}: no verdict ({reason}).",
                         "words_extra": f"{WORDS_FIXED} {ran_words}", "words_limits": WORDS_NO_CACHE}
                   for arm in VERDICT_ARMS} for e in MAIN}
        return {"status": status, **extra, "encoders": enc,
                "summary_words": _summary_lines({(e, arm): NO_VERDICT for e in MAIN for arm in VERDICT_ARMS}),
                "words_ending": None, **record}

    if arms["s32"] is None:
        return stop("no_base", "the s32 arm (the base) did not run")
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

    def score(m, e, k):
        u = (m or {"units": {}})["units"].get((e, k))
        return float(u["inner_score"]) if u and u.get("ok") else None

    def words(arm, e, rec) -> tuple:
        """(outcome, extra, own ending): one sentence set per encoder and arm (declaration r3, section 6).
        Only a no-verdict set has its own ending; the common ending is given once per slide (words_ending)."""
        n, name = SLICE_ARMS[arm], n13.ENC_WORDS[e]
        if rec["verdict"] == NO_VERDICT:
            done = {label: sum(1 for k in folds if score(arms[label], e, k) is not None) for label in ("s32", arm)}
            ran_words = f"Finished: s32 {done['s32']} of 5 folds, {arm} {done[arm]} of 5 folds."  # what ran
            return f"{n} slices, {name}: no verdict ({rec['reason']}).", f"{WORDS_FIXED} {ran_words}", WORDS_NO_CACHE
        d = rec["diff"]
        if d == 0:  # at 4 decimals
            return (f"With {n} of the 32 cached slices per plane the head scored the same on inner validation "
                    f"({name})."), None, None
        side = "lower" if d < 0 else "higher"
        outcome = (f"With {n} of the 32 cached slices per plane the head scored {abs(d):.4f} {side} on inner "
                   f"validation ({name}, {side} in {len(rec[side]['folds'])} of 5 folds).")
        if rec["verdict"] == COST:
            return outcome, None, None
        if rec["verdict"] == HIGHER:
            return outcome, WORDS_NOT_CHANGED, None
        return outcome, (f"This does not meet the declared rule (more than {threshold:g} at 4 decimals, and "
                         f"{n13.MIN_FOLDS_UP} of 5 folds), so it is not counted as an effect."), None

    out_e: dict = {}
    for e in MAIN:
        out_e[e] = {}
        for arm in VERDICT_ARMS:
            have = [k for k in folds if score(arms["s32"], e, k) is not None and score(arms[arm], e, k) is not None]
            diffs = {k: score(arms[arm], e, k) - score(arms["s32"], e, k) for k in have}
            per_fold = {k: round(x, n13.DECIMALS) + 0.0 for k, x in diffs.items()}
            lower, higher = [k for k in have if per_fold[k] < 0], [k for k in have if per_fold[k] > 0]
            rec = {"verdict": NO_VERDICT, "reason": None, "diff": None, "threshold": threshold, "folds_up": None,
                   "lower": {"folds": lower}, "higher": {"folds": higher}, "folds": have, "per_fold": per_fold}
            if arms[arm] is None:
                rec["reason"] = "the arm did not run"
            elif len(have) < n13.N_OUTER:
                rec["reason"] = "fewer than 5 folds finished in the arm and in s32"
            else:
                diff_raw = float(np.mean([diffs[k] for k in folds]))
                diff = round(diff_raw, n13.DECIMALS) + 0.0  # + 0.0: no -0.0
                if diff < -threshold and len(lower) >= n13.MIN_FOLDS_UP:
                    verdict = COST
                elif diff > threshold and len(higher) >= n13.MIN_FOLDS_UP:
                    verdict = HIGHER
                else:
                    verdict = NO_DIFF
                rec.update(diff=diff, diff_unrounded=diff_raw, folds_up=len(higher), verdict=verdict)
            rec["words"], rec["words_extra"], rec["words_limits"] = words(arm, e, rec)
            out_e[e][arm] = rec

    # descriptive from here on: nothing below enters a verdict
    def minus(a, b, e) -> dict:
        have = [k for k in folds if score(a, e, k) is not None and score(b, e, k) is not None]
        per = {k: round(score(a, e, k) - score(b, e, k), n13.DECIMALS) + 0.0 for k in have}
        return {"folds": have, "per_fold": per,
                "mean": round(float(np.mean([score(a, e, k) - score(b, e, k) for k in have])), n13.DECIMALS) + 0.0
                if have else None,
                "all_folds": len(have) == n13.N_OUTER}

    position = {"descriptive": True, "used_in_rule": False,
                "what": "s16b reads the 16 cached positions that s16 leaves out: the position effect at a fixed count",
                "s16b_ran": arms["s16b"] is not None,
                "s16b_minus_s16": {e: minus(arms["s16b"], arms["s16"], e) for e in MAIN},
                "s16b_minus_s32": {e: minus(arms["s16b"], arms["s32"], e) for e in MAIN}}
    table, gaps = {}, {}
    for e in MAIN:
        for k in folds:
            uid = f"{e}_f{k}"
            base, old = score(arms["s32"], e, k), score(m_bc, e, k)
            row = {"batch_chosen": None}
            for label, m in arms.items():
                u = ((m or {"units": {}})["units"].get((e, k)) or {})
                u = u if u.get("ok") else {}
                row["batch_chosen"] = row["batch_chosen"] or u.get("batch_chosen")
                row[label] = {"inner_score": u.get("inner_score"), "E": u.get("E"),
                              "minus_s32": (round(u["inner_score"] - base, n13.DECIMALS) + 0.0
                                            if label != "s32" and base is not None and u.get("inner_score") is not None
                                            else None)}
            row["bc_inner_score"] = old
            row["s32_minus_bc"] = round(base - old, n13.DECIMALS) + 0.0 if base is not None and old is not None else None
            if row["s32_minus_bc"] is not None:
                gaps[uid] = row["s32_minus_bc"]
            table[uid] = row
    outside = [uid for uid, v in gaps.items() if abs(v) > RERUN_TOL]  # at 4 decimals: 0.0005 is inside
    rerun = {"descriptive": True, "used_in_rule": False, "tolerance": RERUN_TOL,
             "what": "s32 minus bc per unit: the same fits (configuration, seeds, rows, slices) at two commits and sessions",
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
             "words": (f"Rerun noise: the 32-slice arm differed from the earlier run of the same fits (arm bc) by more "
                       f"than {RERUN_TOL:g} in {len(outside)} of {len(gaps)} units (largest difference "
                       f"{max(abs(gaps[u]) for u in outside):.4f}).") if outside else None}

    def over_units(label, e, field, fn):
        vals = [u[field] for (enc, _), u in (arms[label] or {"units": {}})["units"].items()
                if (e is None or enc == e) and u.get("ok") and u.get(field) is not None]
        return float(fn(vals)) if vals else None

    best_epoch = {label: {"all": over_units(label, None, "E", np.median),
                          **{e: over_units(label, e, "E", np.median) for e in MAIN}} for label in arms}
    means = {e: {label: (round(float(np.mean([score(m, e, k) for k in folds])), n13.DECIMALS)
                         if all(score(m, e, k) is not None for k in folds) else None)
                 for label, m in arms.items()} for e in MAIN}
    verdicts = {(e, arm): out_e[e][arm]["verdict"] for e in MAIN for arm in VERDICT_ARMS}
    summary = _summary_lines(verdicts)
    with_verdict = any(v != NO_VERDICT for v in verdicts.values())
    return {"status": "decided" if NO_VERDICT not in verdicts.values() else "partial",
            "environment": envs[0], "encoders": out_e, "summary_words": summary,
            "words_ending": WORDS_ENDING if with_verdict else None,
            "units": table, "mean_inner_score": means,
            "median_best_epoch": {"descriptive": True, "what": "median over the arm's finished units of the unit's E "
                                  "(the median best epoch of its inner fits)", **best_epoch},
            "position_effect": position, "s32_reruns_bc": rerun["identical"], "rerun_check": rerun,
            **record}


def slice_slide_sentences(record: dict) -> list:
    """The slide sentences of a decide_slices record: one per encoder and arm (outcome, extra, and for a
    no-verdict set its own ending), then the summary line(s), then the rerun-noise sentence when an s32
    unit is outside the tolerance, then the common ending once, as the last line (not given when no
    comparison has a verdict)."""
    out = []
    for e in MAIN:
        for arm in VERDICT_ARMS:
            rec = record["encoders"][e][arm]
            out.append(" ".join(p for p in (rec.get("words"), rec.get("words_extra"), rec.get("words_limits")) if p))
    out += list(record.get("summary_words") or [])
    for extra in ((record.get("rerun_check") or {}).get("words"), record.get("words_ending")):
        if extra:
            out.append(extra)
    return out
