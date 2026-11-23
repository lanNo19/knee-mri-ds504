"""N9's secondary outputs (added 23 Nov after the pilot reviews). All of them are descriptive.

Rule: nothing here changes H1, H2 or H3, the primary metric, a chosen setting or an existing
number in agg_results.json. Every block has its own new key. A block that fails is recorded under
"secondary_errors" (error type and place only) and the freeze file is still written.

Blocks and what they read:
  h1.secondary                   per-fold Delta1 and its unweighted mean (Stage A predictions)
  h1.g_lang.interval_within_fold_study_bootstrap   a second interval for Delta1
  average_precision              macro AP per model (Stage A and floor A predictions)
  seed_sensitivity               N7's s2seeds_<encoder>.csv (refits with seeds 0..5)
  triage                         N7's triage_thresholds.csv applied to the test folds
  baselines                      one table, B0 to the tuned models, on the same studies and folds
  hpo_evidence                   Stage A's l1.csv, trials.csv, config.json (inner scores only)
  near_duplicate_probe           copied from agg_n7.json (counts only)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, hpo_evidence, intervals, metrics, paths, safe_log
from ..labels import FINDINGS, UID
from . import n9 as core

ROLES = {core.CONV: "baseline model", core.VIT: "control", core.DINO: "target model"}
SEED_REPRODUCED_TOL = 1e-3  # largest probability gap at which N7's seed-0 refit counts as Stage A's refit again


@dataclass
class Ctx:
    """What n9.run already computed, handed to the secondary blocks (nothing is recomputed differently)."""
    d: object
    ev: np.ndarray  # the fixed evaluation studies (rows)
    Y: np.ndarray  # scoring labels (uncertain = negative)
    units: dict  # {encoder: [N, 12] Stage A predictions}
    elig: list  # the S2 macro findings (all 12)
    n_boot: int
    seed: int
    root: Path | None
    row_of: dict
    floor: np.ndarray | None  # floor A predictions
    gold: np.ndarray  # gold rows scored by both main encoders
    g_find: list  # gold findings
    h1: dict  # the G-lang decision interval of Delta1
    cat_lang: str  # its category
    n7_dir: Path | None
    dirs: dict = field(default_factory=dict)  # Stage A unit folders
    h2_find: list = field(default_factory=list)  # H2's findings
    h2: dict = field(default_factory=dict)  # the H2 block as decided

    @property
    def f2(self):
        return self.d.folds["s2_fold"]

    @property
    def g(self):
        return self.d.g_lang[self.ev]

    def macro(self, P, w=None) -> float:
        """The primary metric: fold-stratified macro AUC on the evaluation studies."""
        return metrics.macro(P[self.ev], self.Y[self.ev], self.f2[self.ev], self.elig, w)

    def complete(self, P) -> bool:
        return P is not None and bool(np.isfinite(P[self.ev]).all())

    def n7_csv(self, name: str) -> np.ndarray | None:
        return core._load_csv(self.n7_dir / name, self.d.n, self.row_of) if self.n7_dir else None


def _slim(iv: dict) -> dict:
    return {k: iv[k] for k in ("estimate", "lo", "hi", "half_width")}


def _sign(x) -> float:
    return float(np.sign(x or 0))


# ---------------------------------------------------------------- B and C: views of Delta1

def per_fold_delta1(c: Ctx) -> dict:
    """Delta1 inside each test fold, and its unweighted mean over folds.

    A fold's Delta1 is the mean over the findings with both classes in that fold of
    AUC(DINOv2) - AUC(ConvNeXt) (a finding without positives there has no AUC; it is listed).
    The primary Delta1 weights folds by their pairs; this mean gives every fold the same weight."""
    A, B = c.units[core.DINO], c.units[core.CONV]
    K = c.d.n_folds("s2")
    deltas, shares, missing = [], [], {}
    pw = np.zeros(K)
    for k in range(K):
        rows = c.ev[c.f2[c.ev] == k]
        diffs = []
        for j in c.elig:
            a, w = metrics.auc(A[rows, j], c.Y[rows, j])
            b, _ = metrics.auc(B[rows, j], c.Y[rows, j])
            pw[k] += w
            if w > 0:
                diffs.append(a - b)
            else:
                missing.setdefault(f"fold_{k}", []).append(FINDINGS[j])
        deltas.append(float(np.mean(diffs)) if diffs else float("nan"))
    shares = pw / pw.sum() if pw.sum() > 0 else np.full(K, np.nan)
    ok = np.isfinite(deltas)
    mean = float(np.mean(np.asarray(deltas)[ok])) if ok.any() else float("nan")
    return {"per_fold_delta1": [core._r(x) for x in deltas],
            "pair_share": [agg.share(x) for x in shares],
            "findings_without_pairs": missing,
            "unweighted_mean_of_fold_delta1": core._r(mean),
            "primary_delta1": c.h1["estimate"],
            "sign_agrees_with_primary": bool(np.isfinite(mean) and _sign(mean) == _sign(c.h1["estimate"]))}


def study_bootstrap(c: Ctx) -> dict:
    """Delta1's second interval: studies resampled inside each test fold (see intervals.within_fold_bootstrap)."""
    A, B = c.units[core.DINO], c.units[core.CONV]
    iv = intervals.within_fold_bootstrap(lambda w: c.macro(A, w) - c.macro(B, w), c.f2[c.ev], c.n_boot, c.seed)
    return {"estimate": c.h1["estimate"], "lo": core._r(iv["lo"]), "hi": core._r(iv["hi"]),
            "half_width": core._r(iv["half_width"]), "valid_share": core._r(iv["valid_share"], 3),
            "n_boot": iv["n_boot"], "captures": "study sampling noise inside folds; sites and fold models fixed",
            "used_for_decisions": False}


# ---------------------------------------------------------------- F: average precision, triage

def ap_of(c: Ctx, P) -> float:
    return metrics.macro_ap(P[c.ev], c.Y[c.ev], c.f2[c.ev], c.elig)


def ap_chance(c: Ctx) -> tuple[float, float]:
    """(chance, prevalence_weighted) of the macro AP: labels and folds only."""
    return metrics.macro_ap_chance(c.Y[c.ev], c.f2[c.ev], c.elig)


def average_precision(c: Ctx) -> dict:
    """Macro AP on S2 per model, ranked inside each test fold (metrics.stratified_ap).

    chance = the expected AP of a random ranking (metrics.random_ap per fold, folds weighted by their
    positives like the AP). It is above the prevalence for rare findings, so "times chance" uses it.
    prevalence_weighted = the same weighting of each fold's plain prevalence, kept for reference."""
    models = {**{e: P for e, P in c.units.items() if c.complete(P)},
              **({"floor_a": c.floor} if c.complete(c.floor) else {})}
    chance, prev = ap_chance(c)
    out = {"definition": "per finding: AP inside each test fold, folds weighted by positives; mean over the 12 findings",
           "chance": core._r(chance), "chance_definition": "expected AP of a random ranking, same fold weighting",
           "prevalence_weighted": core._r(prev),
           "mean_prevalence": core._r(float(np.mean((c.Y[c.ev][:, c.elig] == 1).mean(axis=0)))), "models": {}}
    for e, P in models.items():
        out["models"][e] = {"macro_ap": core._r(ap_of(c, P)), "per_finding": {
            FINDINGS[j]: core._r(metrics.stratified_ap(P[c.ev, j], c.Y[c.ev, j], c.f2[c.ev])[0], 3) for j in c.elig}}
    out["chance_per_finding"] = {FINDINGS[j]: core._r(metrics.stratified_ap_chance(c.Y[c.ev, j], c.f2[c.ev])[0], 3)
                                 for j in c.elig}
    return out


def triage(c: Ctx) -> dict:
    """The worklist triage simulation at the 95%-sensitivity threshold N7 chose on inner out-of-fold
    predictions: "any finding" = the highest of the 12 probabilities; each fold's threshold is applied
    to its own test fold; the confusion matrix is pooled over folds (counts rounded to 5). Rates come
    from the exact counts; a rate that uses a cell of 1 to 4 studies is not shown (None)."""
    p = (c.n7_dir / "triage_thresholds.csv") if c.n7_dir else None
    if p is None or not p.exists():
        return {"status": "not run (no thresholds from N7)"}
    thr = pd.read_csv(p)
    pos_all = (c.Y[:, c.elig] == 1).any(axis=1)
    out = {"rule": "any finding = max of 12 probabilities; threshold from inner out-of-fold predictions",
           "target_sensitivity": 0.95, "descriptive": True, "models": {}}
    for e, P in c.units.items():
        t = thr[thr["encoder"] == e].set_index("fold")["threshold"]
        if not c.complete(P) or set(t.index) != set(range(c.d.n_folds("s2"))):
            continue
        tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for k, tk in t.items():
            rows = c.ev[c.f2[c.ev] == k]
            cm = metrics.confusion(P[rows][:, c.elig].max(axis=1) >= tk, pos_all[rows])
            tot = {a: tot[a] + cm[a] for a in tot}
        r = metrics.rates(tot, suppress_small=True)
        out["models"][e] = {"confusion": agg.counts(tot), **{k: agg.share(v) for k, v in r.items()},
                            "normal_studies_moved_back_share": agg.share(r["specificity"]),
                            "inner_sensitivity_mean": agg.share(thr.loc[thr["encoder"] == e, "inner_sensitivity"].mean())}
    return out


# ---------------------------------------------------------------- D: seed sensitivity

def _seed_predictions(c: Ctx, enc: str) -> dict:
    p = (c.n7_dir / f"s2seeds_{enc}.csv") if c.n7_dir else None
    if p is None or not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for sd, g in df.groupby("seed"):
        P = np.full((c.d.n, len(FINDINGS)), np.nan)
        rows = g[UID].astype(str).map(c.row_of)
        ok = rows.notna().to_numpy()
        P[rows[ok].astype(int).to_numpy()] = g.loc[ok, FINDINGS].to_numpy(float)
        out[int(sd)] = P
    return out


def seed_sensitivity(c: Ctx) -> dict:
    """Delta1 with each fold's chosen config refit under seeds 1..5 (N7), next to the decision seed 0.

    decision_per_seed applies the unchanged rule (intervals.decide on the G-lang decision interval)
    to each seed's paired predictions. It covers the G-lang category only: G-vendor has no seed refits.
    seed_sensitive = some seed's category differs from seed 0's. Decisions stay on seed 0.
    Two companions keep that flag from being over-read: share_of_extra_seeds_same_category (how many
    of seeds 1..5 gave seed 0's category) and sign_same_all_seeds (Delta1 has one sign in all six).

    seed0_refit_check compares N7's own seed-0 refit with Stage A's stored predictions. It is a
    numerical check of the refit code path, not the chain guard (that is the "chain" block). On the
    device Stage A used (a T4), the gap should be below the tolerance. With N7 on CPU it will not
    be: heads.fit draws weights and batch order from the CPU generator on every device, but dropout
    masks come from the device's own generator, so a CPU refit with seed 0 is in effect one more
    seed. Nothing is wrong then. Read delta1_gap against delta1_sd (delta1_gap_within_seed_sd)."""
    pa, pb = _seed_predictions(c, core.DINO), _seed_predictions(c, core.CONV)
    seeds = sorted(sd for sd in set(pa) & set(pb) if sd != 0 and c.complete(pa[sd]) and c.complete(pb[sd]))
    if not seeds:
        return {"status": "not run (no seed refits from N7)"}
    delta = {"seed_0": c.h1["estimate"]}
    dec = {"seed_0": c.cat_lang}
    ivs, auc = {}, {core.DINO: {"seed_0": core._r(c.macro(c.units[core.DINO]))},
                    core.CONV: {"seed_0": core._r(c.macro(c.units[core.CONV]))}}
    for sd in seeds:
        A, B = pa[sd], pb[sd]
        iv = core._interval(lambda w: c.macro(A, w) - c.macro(B, w), c.g, c.n_boot, c.seed)
        key = f"seed_{sd}"
        delta[key], ivs[key] = iv["estimate"], _slim(iv)
        dec[key] = intervals.decide(iv["lo"], iv["hi"], a="DINOv2 better", b="ConvNeXt better")
        auc[core.DINO][key], auc[core.CONV][key] = core._r(c.macro(A)), core._r(c.macro(B))
    vals = np.array([v for v in delta.values() if v is not None], float)
    out = {"decision_seed": 0, "seeds": [0, *seeds], "delta1_per_seed": delta,
           "delta1_sd": core._r(vals.std(ddof=1)) if len(vals) > 1 else None,
           "delta1_low_high": [core._r(vals.min()), core._r(vals.max())],
           "interval_per_seed": ivs, "decision_per_seed": dec,
           "seed_sensitive": any(v != c.cat_lang for v in dec.values()),
           "share_of_extra_seeds_same_category": agg.share(np.mean([dec[f"seed_{sd}"] == c.cat_lang for sd in seeds])),
           "sign_same_all_seeds": bool(len({_sign(v) for v in vals}) == 1),
           "macro_auc_per_seed": auc, "scope": "G-lang category of H1 only; decisions use seed 0"}
    if 0 in pa and 0 in pb and c.complete(pa[0]) and c.complete(pb[0]):
        gap = max(float(np.abs(pa[0][c.ev] - c.units[core.DINO][c.ev]).max()),
                  float(np.abs(pb[0][c.ev] - c.units[core.CONV][c.ev]).max()))
        d0 = c.macro(pa[0]) - c.macro(pb[0])
        dgap = abs(d0 - (c.h1["estimate"] or 0.0))
        sd_all = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        out["seed0_refit_check"] = {
            "refit_device": _n7_device(c), "tolerance": SEED_REPRODUCED_TOL,
            "max_abs_probability_gap": core._r(gap, 5), "delta1_of_refit": core._r(d0),
            "delta1_gap": core._r(dgap, 5),
            "reproduces_stage_a": bool(gap <= SEED_REPRODUCED_TOL),
            "delta1_gap_within_seed_sd": bool(dgap <= sd_all) if np.isfinite(sd_all) else None,
            "reading": "check of the refit code, not the chain guard; a CPU refit differs from a GPU one like another seed "
                       "(dropout stream): judge delta1_gap against delta1_sd"}
    return out


def _n7_device(c: Ctx) -> str | None:
    p = paths.find_file("agg_n7.json", c.root)
    return json.loads(p.read_text()).get("device") if p else None


# ---------------------------------------------------------------- E: baselines table

def _gold_auc(c: Ctx, P) -> float | None:
    if not c.g_find or P is None or not len(c.gold) or not np.isfinite(P[c.gold]).all():
        return None
    return core._r(metrics.macro(P[c.gold], c.d.gold_y[c.gold], np.zeros(len(c.gold), int), c.g_find))


def _row(c: Ctx, P, role: str, default=None) -> dict:
    """One model's row: every column on the same evaluation studies and S2 folds."""
    if not c.complete(P):
        return {"role": role, "complete": False}
    ap = ap_of(c, P)
    row = {"role": role, "complete": True,
           "s2_macro_auc": _slim(core._interval(lambda w: c.macro(P, w), c.g, c.n_boot, c.seed)),
           "mean_of_fold_macros": core._r(metrics.per_fold_macro_mean(P[c.ev], c.Y[c.ev], c.f2[c.ev], c.elig)),
           "gain_over_b1": core._r(c.macro(P) - c.macro(c.floor)) if c.complete(c.floor) else None,
           "macro_ap": core._r(ap), "gold58_macro_auc": _gold_auc(c, P)}
    if c.complete(default):
        row["gain_over_own_default"] = _slim(core._interval(lambda w: c.macro(P, w) - c.macro(default, w),
                                                            c.g, c.n_boot, c.seed))
    return row


def baselines(c: Ctx) -> dict:
    """B0 to B3 and the tuned models as one table (the reference row is added by hand on the slides)."""
    pilot = paths.find_file("agg_pilot.json", c.root)
    null = ((json.loads(pilot.read_text()).get("overlap_control") or {}).get("null_macro_auc")) if pilot else None
    chance_ap, _ = ap_chance(c)  # the expected AP of a random ranking: labels and folds only
    rows = {"b0_chance": {"role": "chance", "s2_macro_auc": {"estimate": 0.5},
                          "measured_permuted_label_null": null, "macro_ap": core._r(chance_ap),
                          "gold58_macro_auc": 0.5 if c.g_find else None}}
    floor = _row(c, c.floor, "scanner tags only (LightGBM, floor A)")
    floor.pop("gain_over_b1", None)
    rows["b1_scanner_tags"] = floor
    defaults = {}
    for e in (core.CONV, core.VIT, core.DINO):
        defaults[e] = c.n7_csv(f"default_{e}.csv")
        rows[f"b2_default_{e}"] = _row(c, defaults[e], "default head: fold's L1 winner at the L2 defaults (no L2 tuning)")
        fixed = c.n7_csv(f"default_fixed_{e}.csv")
        if fixed is not None:
            rows[f"b2_default_fixed_{e}"] = _row(c, fixed, "default head: gated attention, all planes, L2 defaults (no HPO)")
    for e in (core.CONV, core.VIT, core.DINO):
        rows[f"tuned_{e}"] = _row(c, c.units.get(e), f"tuned, {ROLES[e]}", defaults[e])
    return {"columns": {
                "s2_macro_auc": "primary definition: fold-stratified macro AUC, with the group-level decision interval",
                "mean_of_fold_macros": "mean over S2 folds of each fold's macro AUC (comparable with published fold means)",
                "gain_over_b1": "S2 macro AUC minus floor A's (point difference)",
                "gain_over_own_default": "tuned minus the same encoder's b2_default row, paired, with the decision interval",
                "macro_ap": "macro average precision (see average_precision); chance is b0_chance.macro_ap",
                "gold58_macro_auc": "macro AUC on the gold studies over the gold findings (fold models pooled)"},
            "rows": rows, "descriptive": True,
            # every model row is scored on the one fixed evaluation set, or it is marked incomplete and empty
            "same_studies_and_folds_for_every_row": all(r.get("complete", True) for r in rows.values()),
            "incomplete_rows": [k for k, r in rows.items() if not r.get("complete", True)],
            "reference_row": "not computed here (published numbers are added by hand on the slides)"}


# ---------------------------------------------------------------- G and H

def hpo(c: Ctx) -> dict:
    return hpo_evidence.summary(hpo_evidence.load(c.dirs), c.d.pos_cap_upper())


def near_duplicate_probe(c: Ctx) -> dict:
    p = paths.find_file("agg_n7.json", c.root)
    probe = json.loads(p.read_text()).get("near_duplicate_probe") if p else None
    return probe if probe else {"status": "not run (no probe in agg_n7.json)"}


def add_all(s: dict, c: Ctx) -> None:
    """Add every secondary block to the summary; a failing block is recorded and skipped."""
    errors = {}

    def put(name: str, fn, target: dict, key: str) -> None:
        try:
            val = fn(c)
            # the same check agg.write applies to the whole file: an unsafe value (a small count, a
            # data-like string) costs this block only, never the decisions
            agg._check(json.loads(json.dumps(val, default=agg._json_default)), path=name)
            target[key] = val
        except Exception as e:  # noqa: BLE001 - descriptive extras must not cost the freeze file
            errors[name] = safe_log.describe(type(e), e.__traceback__)

    put("h1.secondary", per_fold_delta1, s["h1"], "secondary")
    put("interval_within_fold_study_bootstrap", study_bootstrap, s["h1"]["g_lang"],
        "interval_within_fold_study_bootstrap")
    put("average_precision", average_precision, s, "average_precision")
    put("seed_sensitivity", seed_sensitivity, s, "seed_sensitivity")
    put("triage", triage, s, "triage")
    put("baselines", baselines, s, "baselines")
    put("hpo_evidence", hpo, s, "hpo_evidence")
    put("near_duplicate_probe", near_duplicate_probe, s, "near_duplicate_probe")
    # blocks added after the results critique (knee.nb.n9c): the same wrapper, new keys only
    from . import n9c

    for name, fn, where, key in n9c.BLOCKS:
        put(name, lambda c, fn=fn: fn(c, s), where(s), key)
    s["secondary_errors"] = errors
