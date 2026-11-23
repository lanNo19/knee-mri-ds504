"""N9's blocks added after the results critique (23 Nov, late). All descriptive, all new keys.

N7 had already run (on CPU) and is not rerun, so every block reads only what N7 and Stage A wrote:
private/n7/{s1,vendor,s2seeds,default,default_fixed}_<encoder>.csv, triage_thresholds.csv,
agg_n7.json, and each unit's predictions.csv, trials.csv, l1.csv, config.json. No block fits a
model. Each runs under n9x.add_all's wrapper, so a failure costs that block only. Each block
carries `descriptive: true` and a one-line `scope`; none changes a decision or an existing key.

  baselines.same_device_gain        tuned minus default, both refit by N7 on N7's device
  refit_agreement                   N7's seed-0 refit against Stage A's fit, with a seed-1 yardstick
  triage_secondary                  any-finding AUC, specificity interval, threshold overshoot
  largest_site_decomposition        the largest site: unseen (S2) against seen (S1) at equal and larger training size
  best_epoch_spread                 inner best epochs, and the learning rate reached at E
  gold_uncertainty                  bootstrap standard errors on the gold studies
  h2_without_lookalikes             Delta2 without look-alike studies (post hoc)
  h1.per_finding_exploratory_note   how to read the 12 unadjusted intervals
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import agg, encoders, heads, hpo_evidence, metrics, paths, stage
from ..labels import FINDINGS
from . import n9 as core
from . import n9x

LOOKALIKE_COSINE = 0.90
GAP_FACTOR = 1.5  # refit_agreement: the Stage A gap may be at most this many times the seed-0-to-seed-1 gap
TARGET_SENSITIVITY = 0.95


def _stage_a_device(c) -> str:
    """Where Stage A's main units were fit, from the run.json at each unit's output root ("unknown" if absent)."""
    seen = set()
    for (enc, _), ud in c.dirs.items():
        if enc not in encoders.MAIN:
            continue
        rj = ud.parents[3] / "run.json"
        if not rj.exists():
            return "unknown"
        gpus = json.loads(rj.read_text()).get("gpus")
        seen.add("unknown" if gpus is None else ("cuda" if gpus else "cpu"))
    return seen.pop() if len(seen) == 1 else "unknown"


def _delta1(c, A, B, w=None) -> float:
    return c.macro(A, w) - c.macro(B, w)


# ---------------------------------------------------------------- 1. same-device HPO gain

def same_device_gain(c, s) -> dict:
    """Tuned minus default head with both refit by N7 on one device and with the same seed 0.

    baselines.rows.tuned_*.gain_over_own_default takes the tuned score from Stage A's fit and the
    default from N7's refit. When the two ran on different devices that gain holds one extra
    seed-like draw (dropout masks come from the device's generator). Here both sides are N7's refits.
    The default head has one seed only; the tuned head has six, so the mean over them is shown too."""
    n7_dev, sa_dev = n9x._n7_device(c) or "unknown", _stage_a_device(c)
    mixed = None if "unknown" in (n7_dev, sa_dev) else n7_dev != sa_dev
    out = {"descriptive": True,
           "scope": "tuned minus default head, both refit by N7 on one device; default has one seed; decides nothing",
           "device": n7_dev, "stage_a_device": sa_dev,
           "gain_over_own_default_mixes_devices": mixed if mixed is not None else "unknown",
           "gain_over_own_default_note": "that gain is Stage A's fit minus N7's default refit; across devices it holds an extra seed-like draw",
           "encoders": {}, "not_available": {}}
    for e in (core.CONV, core.VIT, core.DINO):
        seeds = {sd: P for sd, P in n9x._seed_predictions(c, e).items() if c.complete(P)}
        default = c.n7_csv(f"default_{e}.csv")
        if 0 not in seeds or not c.complete(default):
            out["not_available"][e] = "N7 wrote no seed refits or no default refit for this encoder"
            continue
        T0 = seeds[0]
        per_seed = np.array([c.macro(seeds[sd]) for sd in sorted(seeds)])
        out["encoders"][e] = {
            "seed0_minus_default": n9x._slim(core._interval(lambda w, T0=T0, D=default: c.macro(T0, w) - c.macro(D, w),
                                                            c.g, c.n_boot, c.seed)),
            "mean_of_seeds_minus_default": core._r(per_seed.mean() - c.macro(default)),
            "tuned_seed_sd": core._r(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
            "seeds": sorted(seeds), "default_seeds": "one (seed 0)"}
    return out


# ---------------------------------------------------------------- 2. refit agreement

def _gap_stats(a, b) -> dict:
    g = np.abs(a - b).ravel()
    q = np.percentile(g, [50, 90, 99])
    return {"median": core._r(q[0], 5), "p90": core._r(q[1], 5), "p99": core._r(q[2], 5), "max": core._r(g.max(), 5)}


def refit_agreement(c, s) -> dict:
    """Is N7's seed-0 refit as close to Stage A's stored fit as two seeds of N7 are to each other?

    Per main encoder: the absolute probability gap over all evaluation studies and findings between
    N7 seed 0 and Stage A, and, as the yardstick, between N7 seed 0 and N7 seed 1.
    gap_like_another_seed = for every encoder the median and the 99th percentile of the first are at
    most 1.5 times those of the second. The seven fits of Delta1 are listed with their device."""
    n7_dev, sa_dev = n9x._n7_device(c) or "unknown", _stage_a_device(c)
    preds = {e: n9x._seed_predictions(c, e) for e in (core.DINO, core.CONV)}
    seeds = sorted(sd for sd in set(preds[core.DINO]) & set(preds[core.CONV])
                   if c.complete(preds[core.DINO][sd]) and c.complete(preds[core.CONV][sd]))
    if 0 not in seeds or 1 not in seeds:
        return {"descriptive": True, "status": "not available (N7 wrote no seed 0 and seed 1 refits)"}
    out = {"descriptive": True,
           "scope": "N7's refits against Stage A's stored predictions; a check of the refit, decides nothing",
           "factor": GAP_FACTOR, "encoders": {}}
    flags = []
    for e in (core.DINO, core.CONV):
        a = _gap_stats(preds[e][0][c.ev], c.units[e][c.ev])
        b = _gap_stats(preds[e][0][c.ev], preds[e][1][c.ev])
        like = bool(a["median"] <= GAP_FACTOR * b["median"] and a["p99"] <= GAP_FACTOR * b["p99"])
        flags.append(like)
        out["encoders"][e] = {"n7_seed0_vs_stage_a": a, "n7_seed0_vs_n7_seed1": b, "gap_like_another_seed": like}
    out["gap_like_another_seed"] = all(flags)
    same = [_delta1(c, preds[core.DINO][sd], preds[core.CONV][sd]) for sd in seeds]
    out["fits"] = {"stage_a": {"delta1": c.h1["estimate"], "device": sa_dev, "role": "the decision fit"},
                   **{f"n7_seed_{sd}": {"delta1": core._r(v), "device": n7_dev} for sd, v in zip(seeds, same)}}
    every = [v for v in [c.h1["estimate"], *same] if v is not None]
    out["delta1_low_high_all_fits"] = [core._r(min(every)), core._r(max(every))]
    out["delta1_low_high_n7_fits"] = [core._r(min(same)), core._r(max(same))]
    out["delta1_sd_n7_fits"] = core._r(np.std(same, ddof=1)) if len(same) > 1 else None
    out["sign_same_all_fits"] = bool(len({np.sign(v) for v in every}) == 1)
    return out


# ---------------------------------------------------------------- 3. triage

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval (95%) for a share k / n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def triage_secondary(c, s) -> dict:
    """How good the triage score itself is, beside the confusion matrix in `triage`.

    any_finding_auc: AUC of the highest of the 12 probabilities against "any finding present",
    pairs inside a test fold only, pair-weighted (as the primary metric). specificity_interval: Wilson
    95% from the exact counts (not shown when a cell it uses holds 1 to 4 studies).
    threshold_overshoot: achieved test sensitivity minus the 0.95 the inner threshold aimed at."""
    pos = (c.Y[:, c.elig] == 1).any(axis=1)
    p = (c.n7_dir / "triage_thresholds.csv") if c.n7_dir else None
    thr = pd.read_csv(p) if p is not None and p.exists() else None
    K = c.d.n_folds("s2")
    out = {"descriptive": True,
           "scope": "any finding = max of 12 probabilities; uncertain labels count as absent; decides nothing",
           "any_finding_prevalence": agg.share(pos[c.ev].mean()), "target_sensitivity": TARGET_SENSITIVITY, "models": {}}
    for e, P in c.units.items():
        if not c.complete(P):
            continue
        score = P[:, c.elig].max(axis=1)
        row = {"any_finding_auc": core._r(metrics.stratified_auc(score[c.ev], pos[c.ev].astype(float), c.f2[c.ev])[0])}
        t = thr[thr["encoder"] == e].set_index("fold")["threshold"] if thr is not None else pd.Series(dtype=float)
        if set(t.index) == set(range(K)):
            tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for k, tk in t.items():
                rows = c.ev[c.f2[c.ev] == k]
                cm = metrics.confusion(score[rows] >= tk, pos[rows])
                tot = {a: tot[a] + cm[a] for a in tot}
            r = metrics.rates(tot, suppress_small=True)
            if r["specificity"] is not None:
                lo, hi = wilson(tot["tn"], tot["tn"] + tot["fp"])
                row["specificity_interval"] = [agg.share(lo), agg.share(hi)]
            else:
                row["specificity_interval"] = None
            row["threshold_overshoot"] = (core._r(r["sensitivity"] - TARGET_SENSITIVITY, 3)
                                          if r["sensitivity"] is not None else None)
        out["models"][e] = row
    return out


# ---------------------------------------------------------------- 4. the largest site, unseen against seen

def _minus(a, b):
    return core._r(a - b) if a is not None and b is not None else None


def largest_site_decomposition(c, s) -> dict:
    """The largest site group: its AUC when unseen (S2) and when seen (S1), at equal and at larger training size.

    What the S1 design gives (folds.seen_site_folds, n7._refits): S1 fold j is a random fold with
    exactly the size of S2 fold j, refit with S2 fold j's config and E. So the S1 fold with the index
    of the group's S2 fold (k) has a training set of the same size as the S2 model that never saw the
    group, but a random mix of sites. The other S1 folds have larger, mixed training sets.
      a  S2 predictions, the group's studies           unseen site, small training set
      b  S1 predictions, the group's studies in S1 fold k    seen site, the same training-set size and config
      c  S1 predictions, the group's studies in the other S1 folds   seen site, larger training sets
    a_on_b_studies and a_on_c_studies score the S2 predictions on exactly b's and c's studies.
    b minus a_on_b_studies: same studies, same size, same config -> the cost of the site being unseen.
    c minus b: different studies and other folds' configs -> only a rough view of training-set size.
    S1 can split one patient across folds, so every "seen" number is an upper bound."""
    d, f1 = c.d, c.d.folds["s1_fold"]
    g = pd.Series(c.g).astype(str)
    focus = g.value_counts().idxmax()
    rows_g = c.ev[(g == focus).to_numpy() & (f1[c.ev] >= 0)]
    k = int(pd.Series(c.f2[rows_g]).mode().iloc[0])
    train = d.train & (f1 >= 0)
    in_group = d.g_lang.astype(str) == focus

    def size(mask):
        return {"studies": agg.count(int(mask.sum())), "group_share": agg.share((mask & in_group).sum() / max(mask.sum(), 1))}

    others = sorted(set(f1[train].tolist()) - {k})
    out = {"descriptive": True,
           "scope": "one site group, existing S1 and S2 predictions; separates unseen site from training size; decides nothing",
           "group": focus, "fold": k,
           "training_sets": {"a_s2_fold_k": size(np.isin(np.arange(d.n), d.split(k)[0])),
                             "b_s1_fold_k": size(train & (f1 != k)),
                             "c_other_s1_folds_mean": {
                                 "studies": agg.count(int(np.mean([(train & (f1 != j)).sum() for j in others]))) if others else None,
                                 "group_share": agg.share(np.mean([(train & (f1 != j) & in_group).sum() / max((train & (f1 != j)).sum(), 1)
                                                                   for j in others])) if others else None}},
           "caveats": ["b and c are different studies; c also uses the other folds' configs",
                       "S1 folds are random, so one patient can sit on both sides: seen-site numbers are upper bounds"],
           "encoders": {}}
    for e, P2 in c.units.items():
        P1 = c.n7_csv(f"s1_{e}.csv")
        if P1 is None or not c.complete(P2):
            continue
        rows = rows_g[np.isfinite(P1[rows_g]).all(axis=1)]
        rb, rc = rows[f1[rows] == k], rows[f1[rows] != k]

        def m(P, r, strata):
            return core._r(metrics.macro(P[r], c.Y[r], strata, c.elig)) if len(r) else None

        zb, zc, za = np.zeros(len(rb), int), np.zeros(len(rc), int), np.zeros(len(rows), int)
        v = {"a_unseen": m(P2, rows, za), "a_on_b_studies": m(P2, rb, zb), "a_on_c_studies": m(P2, rc, zc),
             "b_seen_equal_training_size": m(P1, rb, zb), "c_seen_larger_training_size": m(P1, rc, f1[rc]),
             "studies": {"a": agg.count(len(rows)), "b": agg.count(len(rb)), "c": agg.count(len(rc))}}
        v["unseen_site_cost_at_equal_training_size"] = _minus(v["b_seen_equal_training_size"], v["a_on_b_studies"])
        v["larger_training_set_seen_site_rough"] = _minus(v["c_seen_larger_training_size"], v["b_seen_equal_training_size"])
        out["encoders"][e] = v
    if not out["encoders"]:
        out["status"] = "not available (no S1 predictions from N7)"
    return out


# ---------------------------------------------------------------- 5. best-epoch spread

def _epoch_stats(values) -> dict | None:
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    return {"lowest": float(v.min()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "share_at_or_below_6": agg.share((v <= 6).mean()),
            "share_at_cap": agg.share((v >= heads.MAX_EPOCHS).mean())}


def best_epoch_spread(c, s) -> dict:
    """Inner best epochs from Stage A's trial logs, and how far the schedule had come at E.

    chosen_config: the inner fits of each unit's best trial (their median is E). all_trials: every
    completed trial's inner fits. lr_fraction_of_peak_at_median_E: heads.schedule_factor at the last
    step of epoch E (1-epoch warm-up, cosine over the 30-epoch schedule): near 1 means the cosine had
    barely started when training stopped. Fits are counted, not studies."""
    units = hpo_evidence.load(c.dirs)
    out = {"descriptive": True, "scope": "inner validation fits only (Stage A logs); decides nothing",
           "scheduled_epochs": heads.MAX_EPOCHS, "patience": heads.PATIENCE, "encoders": {}}
    for e in sorted({u["encoder"] for u in units}):
        chosen, every, Es, spe = [], [], [], []
        for u in units:
            t = hpo_evidence.searched(u)
            if u["encoder"] != e or not len(t) or "best_epochs" not in t:
                continue
            eps = [json.loads(str(x)) for x in t["best_epochs"]]
            chosen += eps[int(t["value"].to_numpy(float).argmax())]
            every += [x for row in eps for x in row]
            E = int(u["config"]["E"])
            Es.append(E)
            if u["config"].get("refit_steps"):
                spe.append(int(u["config"]["refit_steps"]) // E)
        if not Es:
            continue
        medE = float(np.median(Es))
        steps = int(np.median(spe)) if spe else 100
        out["encoders"][e] = {
            "chosen_config": _epoch_stats(chosen), "all_trials": _epoch_stats(every),
            "refit_epochs": {"E": sorted(Es)}, "median_E": medE,
            "lr_fraction_of_peak_at_median_E": core._r(heads.schedule_factor(int(round(medE * steps)) - 1, steps), 3),
            "warmup_share_of_training": core._r(heads.WARMUP_EPOCHS / medE, 2)}
    return out


# ---------------------------------------------------------------- 6. gold

def gold_uncertainty(c, s) -> dict:
    """Approximate standard errors on the gold studies: a study bootstrap (2,000 draws, studies drawn
    with replacement, both encoders on the same draw). It covers the choice of gold studies only,
    not sites or seeds. The exact gold count goes to private/n9/gold_count.json; here it is rounded."""
    gold, GY = c.gold, c.d.gold_y
    if not len(gold) or not c.g_find:
        return {"descriptive": True, "status": "not available (no gold findings)"}
    exact = {"gold_studies_scored_by_both_main_encoders": int(len(gold)),
             "gold_studies_usable": int(len(c.d.gold_rows())), "gold_findings": [FINDINGS[j] for j in c.g_find]}
    (paths.out_dir("private", "n9") / "gold_count.json").write_text(json.dumps(exact, indent=1))
    zeros = np.zeros(len(gold), int)
    models = {e: P for e, P in c.units.items() if np.isfinite(P[gold]).all()}

    def macro(P, w=None):
        return metrics.macro(P[gold], GY[gold], zeros, c.g_find, w)

    n_draws = 2000
    rng = np.random.default_rng(c.seed)
    draws = {e: np.empty(n_draws) for e in models}
    for b in range(n_draws):
        w = np.bincount(rng.integers(0, len(gold), len(gold)), minlength=len(gold)).astype(float)
        for e, P in models.items():
            draws[e][b] = macro(P, w)

    def se(v):
        ok = np.isfinite(v)
        return core._r(v[ok].std(ddof=1)) if ok.sum() > 1 else None

    out = {"descriptive": True, "scope": "gold studies only; covers which gold studies were drawn, not sites or seeds",
           "studies": agg.count(len(gold)), "exact_count_file": "private/n9/gold_count.json", "n_draws": n_draws,
           "valid_share": core._r(float(np.mean([np.isfinite(v).mean() for v in draws.values()])), 3),
           "macro_auc": {e: {"estimate": core._r(macro(P)), "se": se(draws[e]),
                             # pairs inside one fold model only: offsets between the five fold models cannot lower it
                             "fold_stratified": core._r(metrics.macro(P[gold], GY[gold], c.f2[gold], c.g_find))}
                         for e, P in models.items()}}
    if core.DINO in models and core.CONV in models:
        dd = draws[core.DINO] - draws[core.CONV]
        out["delta1"] = {"estimate": core._r(macro(models[core.DINO]) - macro(models[core.CONV])), "se": se(dd)}
    return out


# ---------------------------------------------------------------- 7. Delta2 without look-alike studies

def h2_without_lookalikes(c, s) -> dict:
    """Post hoc: H2's Delta2 again without every study that is a member of a look-alike pair.

    N7 wrote only counts of the pairs, not who they are. N9 has the N5 embeddings attached (StageData
    needs them), and the pairs take seconds on CPU, so they are found again here with the pilot's
    vectors and code (stage.near_duplicates, DINOv2 study vectors, cosine above 0.90). The pair count
    is compared with N7's probe. Membership stays in memory; only counts are reported."""
    out = {"descriptive": True, "post_hoc": True,
           "scope": "sensitivity of Delta2 to look-alike studies; added after results were seen; decides nothing",
           "cosine_above": LOOKALIKE_COSINE}
    s1 = {e: c.n7_csv(f"s1_{e}.csv") for e in (core.DINO, core.CONV)}
    if any(v is None for v in s1.values()) or c.h2.get("estimate") is None:
        return {**out, "status": "not available (H2 was not run: no S1 predictions)"}
    d, f1 = c.d, c.d.folds["s1_fold"]
    pairs = stage.near_duplicates(d.study_vectors(core.DINO), np.flatnonzero(d.usable), LOOKALIKE_COSINE, "cpu")
    members = np.unique(np.r_[pairs["row_a"].to_numpy(int), pairs["row_b"].to_numpy(int)])
    A, B, sA, sB = c.units[core.DINO], c.units[core.CONV], s1[core.DINO], s1[core.CONV]
    ev2 = c.ev[(f1[c.ev] >= 0) & np.isfinite(sA[c.ev]).all(axis=1) & np.isfinite(sB[c.ev]).all(axis=1)]
    keep = ev2[~np.isin(ev2, members)]

    def delta2(rows):
        st1, st2 = metrics.strata_codes(f1[rows], d.g_lang[rows]), metrics.strata_codes(c.f2[rows], d.g_lang[rows])

        def ws(P, st, w):
            return metrics.macro(P[rows], c.Y[rows], st, c.h2_find, w)

        return lambda w: (ws(sA, st1, w) - ws(A, st2, w)) - (ws(sB, st1, w) - ws(B, st2, w))

    iv = n9x._slim(core._interval(delta2(keep), d.g_lang[keep], c.n_boot, c.seed))
    probe = (s.get("near_duplicate_probe") or {}).get(f"cosine_above_{LOOKALIKE_COSINE:.2f}") or {}
    out.update({"pair_source": "found again in N9 from the attached N5 embeddings with the pilot's code (N7 wrote counts only)",
                "pairs": agg.count(len(pairs)),
                "pairs_match_n7_probe": (probe.get("pairs") == agg.count(len(pairs))) if probe else None,
                "studies_removed": agg.count(int(len(ev2) - len(keep))),
                "share_removed": agg.share((len(ev2) - len(keep)) / max(len(ev2), 1)),
                "delta2_registered": {k: c.h2.get(k) for k in ("estimate", "lo", "hi")},
                "delta2_without_lookalikes": iv,
                "change": core._r(iv["estimate"] - c.h2["estimate"]) if iv["estimate"] is not None else None})
    return out


# ---------------------------------------------------------------- 8. per-finding note

def per_finding_note(c, s) -> dict:
    """How to read h1.per_finding_exploratory: 12 unadjusted 95% intervals; some exclude 0 by chance."""
    pf = s["h1"]["per_finding_exploratory"]
    n = sum(1 for v in pf.values() if v["lo"] is not None and (v["lo"] > 0 or v["hi"] < 0))
    return {"descriptive": True, "scope": "the 12 per-finding intervals are unadjusted and exploratory; they decide nothing",
            "intervals_excluding_zero": f"{n} of {len(pf)}",
            "note": "unadjusted for 12 comparisons: show all 12 or none; never quote a single finding"}


def _root(s):
    return s


BLOCKS = [
    ("baselines.same_device_gain", same_device_gain, lambda s: s.setdefault("baselines", {}), "same_device_gain"),
    ("refit_agreement", refit_agreement, _root, "refit_agreement"),
    ("triage_secondary", triage_secondary, _root, "triage_secondary"),
    ("largest_site_decomposition", largest_site_decomposition, _root, "largest_site_decomposition"),
    ("best_epoch_spread", best_epoch_spread, _root, "best_epoch_spread"),
    ("gold_uncertainty", gold_uncertainty, _root, "gold_uncertainty"),
    ("h2_without_lookalikes", h2_without_lookalikes, _root, "h2_without_lookalikes"),
    ("h1.per_finding_exploratory_note", per_finding_note, lambda s: s["h1"], "per_finding_exploratory_note"),
]
