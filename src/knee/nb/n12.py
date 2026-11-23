"""N12: error analysis after the iteration-1 results (CPU; plan r2, section 4). Descriptive only.

Reads existing outputs and nothing else: knee-n0 (studies.csv with the report language, the soft
and binary report labels, the expert gold labels, agg_n0.json), knee-n3-r (folds.csv), knee-n5
(index.csv), the Stage A units of knee-n6a and knee-n6b-g (predictions.csv and config.json, and
agg_<run>.json at each output root) and N7's seed refits (knee-n7-b: private/n7/s2seeds_<encoder>.csv).
It fits nothing, decides nothing and switches nothing on (plan r2, R8); N7, N9 and N10 never read
its output. It writes agg_n12.json and run.json only: no per-study row, no identifier.

The evaluation studies, the predictions, the findings and the scoring rule (uncertain counts as
negative) are N9's: the same StageData, the same unit loading (n9._load_units), the same chain guard
(a mismatch fails the step, as in N9), the same fixed evaluation set (the scored training studies with
a test fold that both main encoders predicted) and N9's gold set (S3: the gold studies both main
encoders predicted, each by its own fold model, pooled over the fold models).

Blocks (keys of agg_n12.json). Each carries descriptive: true, added_after_results: true and a
one-line scope. A block that raises, or returns a value the summary checks refuse, is recorded under
errors (error type and place only) and the other blocks still run.
  gold_paired_check           item 1 (FM5b): gold macro AUC against expert and against report labels
  gold_label_quality          item 2 (FM5, FM3): the report labels against the expert labels, per finding
  fm5b_attenuation_check      added after review r1: the FM5b relation on gold, per finding and macro
  per_finding_table           item 3 (FM3): S2 AUC per finding, training prevalence, label quality
  english_group_by_language   item 4 (FM1, FM11): the English group's fold model by report language
  development                 item 5: a2_seed_ensemble, a3_encoder_ensemble, a4_calibration, each
                              labelled "development: ..." (plan r2, R6)

Suppression (plan r2, section 4): a finding enters a stratum only with at least 5 positives and 5
negatives there; a stratum needs at least 50 studies; a macro uses one fixed finding list for every
stratum it is compared with; counts go through agg.count, so the gold count shows as "about 60".
On gold (reviews r1 and r2, B1), the standard is: no cell of 1-4 studies comes back, also for a reader who
knows the exact gold total (agg_n0's is_58). Every rate from a finding's 2x2 table (PPV, NPV,
PPV + NPV - 1 and what is built on them) is blank when any of its four cells holds 1-4 studies. Every
per-finding number has 2 decimals and a macro keeps 4 only over at least 5 findings (AUCs at 3-4 decimals
that share one denominator, positives x negatives, can pin a count). Exact counts of 5 or more can still
come back (for example from a fully shown table); the agg rule does not hide those. Gold studies without
any report label are left out of every comparison that needs the report label (counted and flagged),
never read as "absent".
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, intervals, metrics, paths, safe_log, stage
from .. import units as unit_files
from ..labels import FINDINGS, UID
from . import common
from . import n9 as core
from . import n9x

MIN_EACH = 5  # positives and negatives a finding needs inside a stratum
MIN_STUDIES = 50  # studies a stratum needs
MIN_FINDINGS = 5  # findings a Spearman correlation across findings needs
PER_FINDING_DECIMALS = 2  # every per-finding number on gold (B1): AUCs at 3-4 decimals sharing a denominator can pin a count
CORE_LANGUAGES = ("ENGLISH", "SPANISH", "TURKISH")  # item 4: the rows plan r2 names; they fix the finding list
ENGLISH = "ENGLISH"  # report language (lingua's name) that identifies the English G-lang group
ENCODERS = (core.DINO, core.CONV, core.VIT)
DEV_LABEL = "development: these studies were test studies in iteration 1; likely optimistic"
SCOPE = ("error analysis after iteration 1 (plan r2, section 4): existing outputs only; fits nothing, "
         "decides nothing, switches nothing on")
NOT_RUN = {  # plan r2, section 4: not run, with the reason
    "label_agreement_strata": "the stratum is defined by the label it is scored against",
    "g_vendor_strata": "a site grouping (G-lang x manufacturer), not a scanner grouping",
    "missing_plane_fallback": "no such study in S2: a study is usable only with all three planes",
    "per_finding_x_language": "84 cells, mostly suppressed",
    "laterality": "70 unresolved studies; at chance they would move macro AUC by under 0.01",
}


def _flags(scope: str, **extra) -> dict:
    return {"descriptive": True, "added_after_results": True, "scope": scope, **extra}


def _dev(scope: str, **extra) -> dict:
    return _flags(scope, label=DEV_LABEL, **extra)


def _draws(n: int, n_boot: int, seed: int):
    """Study-bootstrap weights: n_boot times, n studies drawn with replacement (multiplicities)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def _percentile(vals, nd: int = 4) -> dict:
    """95% percentile interval of the defined draws (nd decimals), and the share of draws that were defined."""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return {"lo": None, "hi": None, "valid_share": 0.0}
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return {"lo": core._r(lo, nd), "hi": core._r(hi, nd), "valid_share": core._r(float(ok.mean()), 3)}


def _enough(y) -> bool:
    """At least MIN_EACH positives and MIN_EACH negatives (labels 1 and 0; NaN counts as neither)."""
    y = np.asarray(y, dtype=float)
    return bool((y == 1).sum() >= MIN_EACH and (y == 0).sum() >= MIN_EACH)


def _about(n: int) -> str:
    c = agg.count(n)
    return f"about {c}" if isinstance(c, int) else c


# ---------------------------------------------------------------- gold: shared rules (items 1, 2 and the FM5b check)

def _small(x) -> bool:
    """A cell of 1 to 4 studies: agg shows such a count as "<5", so no written number may give it back."""
    return 1 <= int(x) <= 4


def _cells(t, r) -> dict:
    """The exact 2x2 table of the report label r against the expert label t (1 present, 0 absent). Never written."""
    t, r = np.asarray(t, dtype=float), np.asarray(r, dtype=float)
    return {"tp": int(((r == 1) & (t == 1)).sum()), "fp": int(((r == 1) & (t == 0)).sum()),
            "fn": int(((r == 0) & (t == 1)).sum()), "tn": int(((r == 0) & (t == 0)).sum())}


def _ppv_npv(cl: dict) -> tuple[float | None, float | None]:
    """PPV and NPV from the exact cells. Both are blank (None) when any of the four cells holds 1 to 4
    studies (review r2, R2-B1), and a rate is blank when it is undefined.
    Why both: the gold total is exact (agg_n0.json writes is_58), so a shown rate's denominator carries the
    other two cells (NPV = TN / (n - TP - FP)) and, with the blank pattern, gives a small cell back. N9's
    rule for rates blanks each rate on its own cells, which is enough only when the total is unknown.
    What this guarantees: no cell of 1-4 studies comes back. Exact counts of 5 or more can come back (a
    fully shown table gives its positives back), as with N9's rates."""
    if any(_small(v) for v in cl.values()):  # the gold total is exact: one shown rate carries the rest
        return None, None
    ppv = None if cl["tp"] + cl["fp"] == 0 else cl["tp"] / (cl["tp"] + cl["fp"])
    npv = None if cl["tn"] + cl["fn"] == 0 else cl["tn"] / (cl["tn"] + cl["fn"])
    return ppv, npv


def _gold_with_report(c) -> tuple[np.ndarray, int]:
    """N9's gold studies that have a report label (at least one soft label), and how many have none.
    A study without any report label is never read as "report says absent": S2 never scores such a study."""
    has = ~np.isnan(c.d.soft[c.gold]).all(axis=1)
    return c.gold[has], int((~has).sum())


def _gold_head(gold, n_without: int) -> dict:
    return {"studies": agg.count(len(gold)), "studies_text": _about(len(gold)),
            "studies_note": "gold studies with a report label, rounded to 5; the exact count is not written",
            "gold_studies_without_report_labels": agg.count(n_without),
            "studies_without_report_labels_left_out": bool(n_without > 0)}


def _paired_findings(c, gold) -> tuple[list, dict]:
    """Item 1's findings: at least 5 positives and 5 negatives among these studies in the expert labels and
    in the report labels; the others with the reason."""
    zeros = np.zeros(len(gold), int)
    f_exp = metrics.eligible(c.d.gold_y[gold], zeros, MIN_EACH, MIN_EACH)
    f_rep = metrics.eligible(c.Y[gold], zeros, MIN_EACH, MIN_EACH)
    both = [j for j in f_exp if j in f_rep]
    left = {}
    for j in range(len(FINDINGS)):
        if j not in both:
            where = [name for name, fl in (("expert", f_exp), ("report", f_rep)) if j not in fl]
            left[FINDINGS[j]] = f"under 5 positives or negatives in the {' and '.join(where)} labels"
    return both, left


def _macro_decimals(findings) -> int:
    """A macro over at least MIN_FINDINGS findings keeps 4 decimals; over fewer it is close to a per-finding
    number and gets the per-finding precision."""
    return 4 if len(findings) >= MIN_FINDINGS else PER_FINDING_DECIMALS


# ---------------------------------------------------------------- 1. gold, expert against report labels

def gold_paired_check(c, n_boot: int, seed: int) -> dict:
    """Item 1 (FM5b): each encoder's macro AUC on the gold studies against the expert labels and
    against the report labels. Same studies, same predictions, same findings.

    Studies and predictions are N9's S3 (the gold studies both main encoders predicted, each by the fold
    model whose test fold holds it, pooled over the fold models) without the studies that have no report
    label at all: those are left out of the comparison, never read as "absent" (counted and flagged).
    Findings: at least 5 positives and 5 negatives among these studies in the expert labels and in the
    report labels. Report labels follow N9's scoring rule (the binary label, uncertain counted as
    negative). Interval: a study bootstrap (n_boot draws of these studies with replacement; both label
    versions and every encoder on the same draw), percentile 2.5-97.5.
    expert_labels_on_n9_findings is N9's S3 number recomputed on N9's own studies and findings, so the
    cross-check against N9's file always applies, whatever the paired list holds."""
    GY, RY = c.d.gold_y, c.Y
    out = _flags("gold studies only: macro AUC against expert and against report labels, same studies, "
                 "predictions and findings; decides nothing",
                 failure_mode="FM5b", predictions="Stage A, pooled over the fold models (as N9's s3_gold)",
                 report_labels=("binary report label, uncertain counted as negative (N9's scoring rule); gold studies "
                                "without any report label are left out"),
                 interval="study bootstrap, paired, percentile 2.5-97.5", n_boot=int(n_boot))
    g9 = c.gold
    z9 = np.zeros(len(g9), int)
    out["n9_findings"] = [FINDINGS[j] for j in c.g_find]
    out["encoders"] = {e: {"expert_labels_on_n9_findings": core._r(metrics.macro(P[g9], GY[g9], z9, c.g_find))
                           if c.g_find else None}
                       for e, P in c.units.items() if len(g9) and np.isfinite(P[g9]).all()}  # as N9's s3_gold
    gold, n_without = _gold_with_report(c)
    out.update(_gold_head(gold, n_without))
    if len(gold) < MIN_STUDIES:
        return {**out, "status": "suppressed (fewer than 50 gold studies with a report label)"}
    both, left = _paired_findings(c, gold)
    out.update({"findings": [FINDINGS[j] for j in both], "findings_left_out": left,
                "findings_same_as_n9_s3": [FINDINGS[j] for j in both] == [FINDINGS[j] for j in c.g_find]})
    if not both:
        return {**out, "status": "not available (no finding with 5 positives and 5 negatives in both label versions)"}
    nd = _macro_decimals(both)
    zeros = np.zeros(len(gold), int)
    models = {e: P for e, P in c.units.items() if np.isfinite(P[gold]).all()}

    def macro(P, L, w=None):
        return metrics.macro(P[gold], L[gold], zeros, both, w)

    vals = {e: np.full((int(n_boot), 2), np.nan) for e in models}  # per draw: expert, report
    for b, w in enumerate(_draws(len(gold), int(n_boot), seed)):
        for e, P in models.items():
            vals[e][b] = macro(P, GY, w), macro(P, RY, w)
    for e, P in models.items():
        ex, rp = macro(P, GY), macro(P, RY)
        out["encoders"].setdefault(e, {}).update({
            "expert_labels": core._r(ex, nd), "report_labels": core._r(rp, nd),
            "expert_labels_interval": _percentile(vals[e][:, 0], nd),
            "report_labels_interval": _percentile(vals[e][:, 1], nd),
            "expert_minus_report": {"estimate": core._r(ex - rp, nd), **_percentile(vals[e][:, 0] - vals[e][:, 1], nd)}})
    return out


# ---------------------------------------------------------------- 2. label quality on gold

def label_quality_raw(c, n_boot: int, seed: int) -> dict:
    """{finding: {"soft_label_auc", "soft_label_auc_draws", "soft_label_note", "ppv", "npv"}} unrounded, on the
    gold studies of item 1 (N9's, with a report label), or {"suppressed": why}.

    soft_label_auc: AUC of the soft report label (mean of the label tables) against the expert label, on the
    studies with a soft label for that finding (masked cells left out; ties count one half); it needs 5
    expert positives and 5 negatives among those studies, else it is null with soft_label_note. Its draws
    are item 1's (same studies, same seed). ppv, npv: of the binary report label under the scoring rule
    (uncertain counted as negative) against the expert label, blank by _ppv_npv's cell rule. A finding with
    under 5 expert positives or negatives is suppressed; with fewer than 50 gold studies every finding is."""
    gold, _ = _gold_with_report(c)
    GY, RY, soft = c.d.gold_y, c.Y, c.d.soft
    W = list(_draws(len(gold), int(n_boot), seed)) if len(gold) >= MIN_STUDIES else []
    out = {}
    for j, f in enumerate(FINDINGS):
        ye = GY[gold, j]
        if len(gold) < MIN_STUDIES:
            out[f] = {"suppressed": "fewer than 50 gold studies with a report label"}
            continue
        if not _enough(ye):
            out[f] = {"suppressed": "under 5 expert positives or negatives"}
            continue
        ppv, npv = _ppv_npv(_cells(ye, RY[gold, j]))
        sj = soft[gold, j]
        if _enough(ye[~np.isnan(sj)]):
            a, draws, note = metrics.auc(sj, ye)[0], [metrics.auc(sj, ye, w)[0] for w in W], None
        else:
            a, draws, note = float("nan"), [], "under 5 expert positives or negatives among the studies with a soft label"
        out[f] = {"soft_label_auc": a, "soft_label_auc_draws": draws, "soft_label_note": note, "ppv": ppv, "npv": npv}
    return out


def _quality_view(v: dict) -> dict:
    """One finding's label quality as items 2 and 3 show it: every number at 2 decimals (B1)."""
    if "suppressed" in v:
        return dict(v)
    both = v["ppv"] is not None and v["npv"] is not None
    out = {"soft_label_auc": core._r(v["soft_label_auc"], PER_FINDING_DECIMALS),
           "soft_label_auc_interval": _percentile(v["soft_label_auc_draws"], PER_FINDING_DECIMALS),
           "ppv": agg.share(v["ppv"]), "npv": agg.share(v["npv"]),
           "ppv_plus_npv_minus_1": agg.share(v["ppv"] + v["npv"] - 1) if both else None}
    if v.get("soft_label_note"):
        out["soft_label_note"] = v["soft_label_note"]
    return out


def gold_label_quality(c, n_boot: int, seed: int) -> dict:
    """Item 2 (FM5, FM3): how well the report labels match the expert labels, per finding, on the gold
    studies of item 1. ppv_plus_npv_minus_1 is the factor of the FM5b relation (if the model's errors
    are independent of the report's given the truth): AUC on report labels - 0.5 =
    (PPV + NPV - 1) x (AUC on true labels - 0.5); fm5b_attenuation_check tests it."""
    out = _flags("gold studies only: the report labels against the expert labels, per finding; decides nothing",
                 failure_mode="FM5, FM3",
                 soft_label="mean of the label tables' P(present) (N0), masked cells left out",
                 binary_label="present if the soft label is above 0.5; uncertain (0.5 or masked) counted as negative",
                 ppv_npv_rule=("both rates null when any cell of the finding's 2x2 table holds 1-4 studies (the gold "
                               "total is exact), and a rate null when undefined"),
                 privacy="no cell of 1-4 studies comes back; per-finding numbers at 2 decimals",
                 interval="soft_label_auc_interval: study bootstrap, item 1's draws, percentile 2.5-97.5",
                 n_boot=int(n_boot))
    gold, n_without = _gold_with_report(c)
    out.update(_gold_head(gold, n_without))
    if len(gold) < MIN_STUDIES:
        return {**out, "status": "suppressed (fewer than 50 gold studies with a report label)"}
    raw = label_quality_raw(c, n_boot, seed)
    per = {f: _quality_view(v) for f, v in raw.items()}
    shown = [v["soft_label_auc"] for v in raw.values() if "suppressed" not in v and np.isfinite(v["soft_label_auc"])]
    out.update({"soft_label_cells_masked": agg.count(int(np.isnan(c.d.soft[gold]).sum())),
                "binary_label_cells_uncertain": agg.count(int(np.isnan(c.d.binary[gold]).sum())),
                "per_finding": per,
                "soft_label_macro_auc": core._r(float(np.mean(shown)), 3 if len(shown) >= MIN_FINDINGS else 2) if shown else None,
                "soft_label_macro_auc_note": "mean over the findings shown; N0's 0.892 used every gold study",
                "findings_suppressed": [f for f, v in per.items() if "suppressed" in v],
                "rates_blank": [f for f, v in per.items() if "suppressed" not in v and (v["ppv"] is None or v["npv"] is None)]})
    return out


# ---------------------------------------------------------------- S3. the FM5b relation on gold

def fm5b_attenuation_check(c, n_boot: int, seed: int) -> dict:
    """Added after review r1 (S3): does each encoder's report-label AUC on gold sit where the FM5b relation
    puts it? Per finding j of item 1's list, on item 1's studies:
      report_labels_auc               AUC against the report labels (N9's scoring rule)
      report_labels_if_independent    0.5 + (PPV_j + NPV_j - 1) x (expert_labels_auc - 0.5)
    and at macro level the means over the list (the macro of the first is item 1's report_labels).
    The relation is exact when the model's errors are independent of the report's errors given the expert
    label. report_minus_if_independent above 0: the model's errors line up with the report's (it learned
    the reports' habits); below 0: the report's errors cost the model more than independence would.
    Interval: item 1's paired study bootstrap; per draw, weighted PPV and NPV and both AUCs.
    B1: per finding, the if-independent value and its difference are blank when a cell of that finding's
    2x2 table holds 1-4 studies (item 2's rule), and per-finding numbers have 2 decimals. The macro uses
    every finding of the list (unrounded inside), so it is there even when per-finding values are blank."""
    GY, RY = c.d.gold_y, c.Y
    out = _flags("gold studies only: the report-label AUC against what the FM5b relation predicts from the "
                 "expert-label AUC and PPV + NPV - 1; decides nothing",
                 failure_mode="FM5b", n_boot=int(n_boot),
                 assumption=("the model's errors are independent of the report's errors given the expert label (FM5b "
                             "row); above 0 = the model's errors line up with the report's"),
                 reading="read the macro: per-finding values on about 60 studies are noisy even under exact independence",
                 per_finding_blank_rule="a cell of 1-4 studies in that finding's 2x2 table (item 2's rule)")
    gold, n_without = _gold_with_report(c)
    out.update(_gold_head(gold, n_without))
    if len(gold) < MIN_STUDIES:
        return {**out, "status": "suppressed (fewer than 50 gold studies with a report label)"}
    both, _ = _paired_findings(c, gold)
    if not both:
        return {**out, "status": "not available (no finding with 5 positives and 5 negatives in both label versions)"}
    nd = _macro_decimals(both)
    zeros = np.zeros(len(gold), int)
    t = {j: GY[gold, j] for j in both}
    r = {j: RY[gold, j] for j in both}
    shown = {j: None not in _ppv_npv(_cells(t[j], r[j])) for j in both}
    out.update({"findings": [FINDINGS[j] for j in both],
                "per_finding_blank": [FINDINGS[j] for j in both if not shown[j]], "encoders": {}})
    models = {e: P for e, P in c.units.items() if np.isfinite(P[gold]).all()}

    def per_finding(P, w=None) -> dict:
        """{j: (report-label AUC, if-independent value, expert-label AUC)} on the data (w None) or one draw."""
        ww = np.ones(len(gold)) if w is None else w
        res = {}
        for j in both:
            rp, rn = ww * (r[j] == 1), ww * (r[j] == 0)
            if not (rp.sum() > 0 and rn.sum() > 0):
                res[j] = (np.nan, np.nan, np.nan)
                continue
            ppv, npv = (rp * (t[j] == 1)).sum() / rp.sum(), (rn * (t[j] == 0)).sum() / rn.sum()
            a_e, a_r = metrics.auc(P[gold, j], t[j], w)[0], metrics.auc(P[gold, j], r[j], w)[0]
            res[j] = (a_r, 0.5 + (ppv + npv - 1) * (a_e - 0.5), a_e)
        return res

    def macro_diff(res) -> float:
        v = np.array([res[j][0] - res[j][1] for j in both], dtype=float)
        return float(v.mean()) if np.isfinite(v).all() else float("nan")

    W = list(_draws(len(gold), int(n_boot), seed))
    for e, P in models.items():
        base = per_finding(P)
        draws = [per_finding(P, w) for w in W]
        obs = metrics.macro(P[gold], RY[gold], zeros, both)  # = item 1's report_labels
        pred = float(np.mean([base[j][1] for j in both]))
        pf = {}
        for j in both:
            a_r, p_j, a_e = base[j]
            row = {"expert_labels_auc": core._r(a_e, PER_FINDING_DECIMALS), "report_labels_auc": core._r(a_r, PER_FINDING_DECIMALS),
                   "report_labels_if_independent": None, "report_minus_if_independent": None}
            if shown[j]:
                row["report_labels_if_independent"] = core._r(p_j, PER_FINDING_DECIMALS)
                row["report_minus_if_independent"] = {"estimate": core._r(a_r - p_j, PER_FINDING_DECIMALS),
                                                      **_percentile([d[j][0] - d[j][1] for d in draws], PER_FINDING_DECIMALS)}
            pf[FINDINGS[j]] = row
        out["encoders"][e] = {"report_labels": core._r(obs, nd), "report_labels_if_independent": core._r(pred, nd),
                              "report_minus_if_independent": {"estimate": core._r(obs - pred, nd),
                                                              **_percentile([macro_diff(d) for d in draws], nd)},
                              "per_finding": pf}
    return out


# ---------------------------------------------------------------- 3. the per-finding table

def _training_prevalence(d) -> tuple[dict, str]:
    """N0's training prevalence per finding (agg_n0.json), else the same quantity recomputed."""
    pv = (d.n0_agg.get("labels") or {}).get("prevalence_training") if isinstance(d.n0_agg, dict) else None
    if isinstance(pv, dict) and set(FINDINGS) <= set(pv):
        return {f: pv[f] for f in FINDINGS}, "agg_n0.json labels.prevalence_training"
    b = d.binary[d.train]
    return ({f: agg.share(np.nan_to_num(b[:, j], nan=0.0).mean()) for j, f in enumerate(FINDINGS)},
            "recomputed from labels_binary.csv (uncertain as absent) over the training studies usable in N5; "
            "N0's own figure uses every training study")


def _spearman_band(k: int, n_perm: int = 20000, seed: int = 0) -> float | None:
    """|rho| that 5% of random orderings of k findings exceed (two-sided, no ties, permutations with a fixed
    seed): a rho inside the band fits no association. About 0.58 for 12 findings."""
    if k < MIN_FINDINGS:
        return None
    perm = np.argsort(np.random.default_rng(seed).random((n_perm, k)), axis=1).astype(float)
    rho = 1 - 6 * ((perm - np.arange(k)) ** 2).sum(axis=1) / (k * (k * k - 1))
    return core._r(float(np.percentile(np.abs(rho), 95)), 2)


def per_finding_table(c, n_boot: int, seed: int) -> dict:
    """Item 3 (FM3): all 12 findings. S2 fold-stratified AUC per encoder (the numbers N9 writes in
    s2.<encoder>.per_finding), N0's training prevalence, item 2's label quality (recomputed here the
    same way, so this block stands without item 2), and per encoder the Spearman correlation across
    findings between label quality (the soft label's AUC on gold) and the S2 AUC. The correlation
    uses the findings item 2 did not suppress (at least 5; else None)."""
    from scipy.stats import spearmanr

    prev, source = _training_prevalence(c.d)
    models = {e: P for e, P in c.units.items() if c.complete(P)}  # as N9's s2 block
    auc = {e: {j: metrics.stratified_auc(P[c.ev, j], c.Y[c.ev, j], c.f2[c.ev])[0] for j in c.elig}
           for e, P in models.items()}
    out = _flags("all 12 findings: S2 fold-stratified AUC (mostly the English group), training prevalence, "
                 "label quality on gold; decides nothing",
                 failure_mode="FM3", s2_auc="fold-stratified AUC per finding, as N9's s2.<encoder>.per_finding",
                 prevalence_source=source, prevalence_population="the training studies, not the evaluation studies",
                 metric_note="fold-stratified: the English group carries most of the pairs")
    try:
        raw, raw_error = label_quality_raw(c, n_boot, seed), None
    except Exception as e:  # noqa: BLE001 - the AUC and prevalence columns stand without it
        raw, raw_error = {}, safe_log.describe(type(e), e.__traceback__)
    rows = {}
    for j in c.elig:
        f = FINDINGS[j]
        rows[f] = {"s2_auc": {e: core._r(auc[e][j], 3) for e in models}, "training_prevalence": prev.get(f),
                   "label_quality": _quality_view(raw[f]) if f in raw else None}
    out["rows"] = rows
    if raw_error:
        out["label_quality_error"] = raw_error
    used = [j for j in c.elig if "soft_label_auc" in (raw.get(FINDINGS[j]) or {})
            and np.isfinite(raw[FINDINGS[j]]["soft_label_auc"])]
    sp = {}
    for e in models:
        x = [raw[FINDINGS[j]]["soft_label_auc"] for j in used]
        y = [auc[e][j] for j in used]
        rho = float(spearmanr(x, y).statistic) if len(used) >= MIN_FINDINGS and np.isfinite(y).all() else float("nan")
        sp[e] = {"rho": core._r(rho, 3)}
    out["spearman_label_quality_vs_s2_auc"] = {
        "encoders": sp, "findings_used": [FINDINGS[j] for j in used],
        "no_association_band": _spearman_band(len(used)),
        "reading": ("descriptive, across findings; label quality = the soft report label's AUC against expert labels "
                    "on gold; |rho| below no_association_band fits no association (5%, two-sided)"),
        "populations": "they differ: the gold case mix against S2, which is mostly the English group"}
    return out


# ---------------------------------------------------------------- 4. the English group by report language

def _languages(d) -> np.ndarray | None:
    """Report language per row of the index, upper case (N0's studies.csv), or None without that column."""
    p = Path(d.n0_dir) / "studies.csv"
    if "language" not in pd.read_csv(p, nrows=0).columns:
        return None
    st = pd.read_csv(p, usecols=[UID, "language"])
    st[UID] = st[UID].astype(str)
    return st.drop_duplicates(UID).set_index(UID)["language"].reindex(d.uids).astype(str).str.upper().to_numpy()


def english_group_by_language(c, language, n_boot: int, seed: int) -> dict:
    """Item 4 (FM1, FM11): inside the S2 fold that holds the English G-lang group, the macro AUC of that
    fold's Stage A model on the group's studies of each report language.

    The group is the G-lang group of the evaluation studies whose report language is English (found by
    group, not by fold number); its fold is the S2 fold of those studies. One model per encoder and one
    fold, so the training set is the same in every row (training_studies, rounded). A row needs at least
    50 evaluation studies. One fixed finding list, set on the rows plan r2 names (CORE_LANGUAGES: English,
    Spanish, Turkish) that have 50 studies, written as list_fixed_on: the findings with at least 5
    positives and 5 negatives in every one of them. Any other language of the group (French, Latin) is reported only on that list,
    and only if every finding of the list has 5 positives and 5 negatives there; else it is suppressed
    with the reason. A smaller row never shrinks the list. The reference row is the whole group (all its
    languages) on the same list. Intervals: a study bootstrap inside each row (n_boot draws, every
    encoder on the same draw), percentile 2.5-97.5."""
    out = _flags("one S2 fold (the English group's), its own fold model, rows by report language; decides nothing",
                 failure_mode="FM1, FM11", n_boot=int(n_boot), min_studies_per_row=MIN_STUDIES,
                 note="one model and one fold per encoder: the training set is the same in every row",
                 named_rows=list(CORE_LANGUAGES))
    if language is None:
        return {**out, "status": "not available (studies.csv has no language column)"}
    ev, Y = c.ev, c.Y
    lang = np.asarray(language).astype(str)
    eng = ev[lang[ev] == ENGLISH]
    if not len(eng):
        return {**out, "status": "not available (no evaluation study with an English report)"}
    group = str(pd.Series(c.d.g_lang[eng]).astype(str).mode().iloc[0])
    in_group = ev[pd.Series(c.d.g_lang[ev]).astype(str).to_numpy() == group]
    group_folds = sorted(set(c.f2[in_group].tolist()))
    k = int(pd.Series(c.f2[in_group]).mode().iloc[0])
    rows_all = in_group[c.f2[in_group] == k]
    sizes = pd.Series(lang[rows_all]).value_counts()
    suppressed = {str(L): {"reason": "under 50 evaluation studies"} for L, n in sizes.items() if n < MIN_STUDIES}
    core_rows = [L for L in CORE_LANGUAGES if sizes.get(L, 0) >= MIN_STUDIES]
    strata = {L: rows_all[lang[rows_all] == L] for L in core_rows}
    find = [j for j in c.elig if all(_enough(Y[r, j]) for r in strata.values())] if strata else []
    for L, n in sizes.items():
        L = str(L)
        if L in CORE_LANGUAGES or n < MIN_STUDIES:
            continue
        r = rows_all[lang[rows_all] == L]
        lacking = [FINDINGS[j] for j in find if not _enough(Y[r, j])]
        if lacking:
            suppressed[L] = {"reason": "under 5 positives or 5 negatives for a finding of the fixed list",
                             "findings": lacking}
        else:
            strata[L] = r
    tr = c.d.split(k)[0] if hasattr(c.d, "split") else None
    out.update({"group": group, "fold": k, "group_in_one_fold": len(group_folds) == 1,
                "list_fixed_on": core_rows,  # the named rows with 50 studies: they set the list
                "training_studies": agg.count(len(tr)) if tr is not None else None,
                "group_languages": sorted(str(L) for L in sizes.index),
                "languages_reported": list(strata), "languages_suppressed": suppressed,
                "findings": [FINDINGS[j] for j in find],
                # per finding left out of the list: the named rows where it has under 5 positives or negatives
                "findings_left_out": {FINDINGS[j]: [L for L in core_rows if not _enough(Y[strata[L], j])]
                                      for j in c.elig if j not in find}})
    if not core_rows or not find:
        return {**out, "status": "not available (no named row with 50 studies, or no finding with 5 positives and "
                                 "5 negatives in every named row)"}
    models = {e: P for e, P in c.units.items() if np.isfinite(P[rows_all]).all()}

    def row(r) -> dict:
        zeros = np.zeros(len(r), int)
        vals = {e: np.full(int(n_boot), np.nan) for e in models}
        for b, w in enumerate(_draws(len(r), int(n_boot), seed)):
            for e, P in models.items():
                vals[e][b] = metrics.macro(P[r], Y[r], zeros, find, w)
        return {"studies": agg.count(len(r)),
                "encoders": {e: {"macro_auc": core._r(metrics.macro(P[r], Y[r], zeros, find)), **_percentile(vals[e])}
                             for e, P in models.items()}}

    out["rows"] = {L: row(r) for L, r in strata.items()}
    out["reference_whole_group"] = row(rows_all)
    return out


# ---------------------------------------------------------------- 5. development numbers (A2, A3, A4)

def _levels_raw(c, P, rows=None) -> tuple[float, float]:
    """The two S2 metrics on the evaluation studies (or a subset of them), unrounded."""
    r = c.ev if rows is None else rows
    return (metrics.macro(P[r], c.Y[r], c.f2[r], c.elig), metrics.per_fold_macro_mean(P[r], c.Y[r], c.f2[r], c.elig))


def _levels(c, P, rows=None) -> dict:
    fs, mf = _levels_raw(c, P, rows)
    return {"fold_stratified_macro_auc": core._r(fs), "mean_of_fold_macros": core._r(mf)}


def _members_mean(vals) -> dict:
    """The mean of the members' own two metrics: the like-for-like reference for reading an ensemble."""
    v = np.asarray(vals, dtype=float)
    return {"fold_stratified_macro_auc": core._r(float(v[:, 0].mean())), "mean_of_fold_macros": core._r(float(v[:, 1].mean()))}


def a2_seed_ensemble(c) -> dict:
    """A2: per encoder, the mean probability of N7's seed refits (seeds 0-5: each fold's chosen config
    and E refit with that seed), scored on the evaluation studies. Nothing is fitted or chosen.
    members_mean: the mean of the single seed refits' own numbers (same device, same seeds), the reference
    to read the ensemble against; Stage A's GPU fit is not one of the members."""
    out = _dev("S2 evaluation studies: mean probability over N7's seed refits per encoder; nothing fitted or chosen",
               action="A2", failure_mode="FM8",
               members_note="members_mean = the single seed refits (same device, same seeds); not iteration 1's number",
               s4_note="on S2 one fold model predicts each study; N10 already averages five fold heads on S4, so an S4 gain would be smaller",
               encoders={}, not_available={})
    for e in ENCODERS:
        seeds = {sd: P for sd, P in n9x._seed_predictions(c, e).items() if c.complete(P)}
        if len(seeds) < 2:
            out["not_available"][e] = "N7 wrote no seed refits (or fewer than two complete ones) for this encoder"
            continue
        M = np.mean(np.stack([seeds[sd] for sd in sorted(seeds)]), axis=0)
        out["encoders"][e] = {"seeds": sorted(int(sd) for sd in seeds), **_levels(c, M),
                              "members_mean": _members_mean([_levels_raw(c, seeds[sd]) for sd in sorted(seeds)])}
    return out


def a3_encoder_ensemble(c) -> dict:
    """A3: the equal-weight mean of the three encoders' Stage A probabilities, on the evaluation studies
    all three predicted. Not evidence about H1. Nothing is fitted or chosen. members_mean: the mean of the
    three encoders' own numbers on the same studies; no difference is computed (R6)."""
    out = _dev("S2 evaluation studies all three encoders predicted: equal-weight mean of their Stage A "
               "probabilities; not evidence about H1", action="A3", failure_mode="FM2, FM4",
               encoders=list(ENCODERS),
               members_note="members_mean = the three encoders alone on the same studies; no difference is computed (R6)")
    missing = [e for e in ENCODERS if e not in c.units]
    if missing:
        return {**out, "status": "not available (an encoder's units are not attached)", "missing": missing}
    ok = np.ones(c.d.n, bool)
    for e in ENCODERS:
        ok &= np.isfinite(c.units[e]).all(axis=1)
    rows = c.ev[ok[c.ev]]
    if not len(rows):
        return {**out, "status": "not available (no study predicted by all three encoders)"}
    M = np.mean(np.stack([c.units[e] for e in ENCODERS]), axis=0)
    return {**out, "studies": agg.count(len(rows)), "share_of_evaluation_studies": agg.share(len(rows) / len(c.ev)),
            **_levels(c, M, rows), "members_mean": _members_mean([_levels_raw(c, c.units[e], rows) for e in ENCODERS])}


def _stage_a_upstream(name: str, root) -> dict | None:
    p = paths.find_file(f"agg_{name}.json", root)
    return (json.loads(p.read_text()).get("upstream") or {}) if p is not None else None


def _labels_from_attached_n0(d) -> bool | None:
    """True if labels_binary.csv and studies.csv are the two files of the attached N0 run (agg_n0.json's
    files_sha16); None without that record."""
    rec = (d.n0_agg or {}).get("files_sha16") if isinstance(d.n0_agg, dict) else None
    if not rec or "labels_binary_sha16" not in rec or "studies_sha16" not in rec:
        return None
    return bool(common.sha16(Path(d.n0_dir) / "labels_binary.csv") == rec["labels_binary_sha16"]
                and d.upstream["n0_studies"]["sha16"] == rec["studies_sha16"])


# The A4 gate. A check in A4_MUST_BE_TRUE must be true; one in A4_MUST_NOT_BE_FALSE may be null (not
# recorded, so not checkable) but never false. stage_a_n5_index_same is shown under info and never gates:
# the weights do not use N5 (the chain guard already fixes each training set).
A4_MUST_BE_TRUE = {
    "chain_guard_passed": "the chain guard did not pass: the training sets are not known",
    "unit_for_every_s2_fold": "a fold has no unit: the training sets are not known",
    "stage_a_n0_studies_same": "Stage A's agg file is missing or records another N0 version: its training labels are not confirmed",
    "pos_cap_recorded": "a unit's config.json has no pos_cap",
}
A4_MUST_NOT_BE_FALSE = {
    "refit_steps_match_training_size": "a unit's refit_steps do not match the current training-set size",
    "labels_binary_from_attached_n0_run": "labels_binary.csv is not the file of the attached N0 run",
}


def positive_weights(c, enc: str, chain_ok: bool) -> tuple[dict | None, dict, dict, str | None]:
    """Each fold's positive weight per finding, exactly as heads.fit computed it for Stage A's refit:
    min(negatives / max(positives, 1), pos_cap) over the binary report labels of the outer training
    set (uncertain counts as neither), as float32. Returns ({fold: weights}, checks, info, reason to skip).

    The weights depend on three things only: the training studies, labels_binary and pos_cap. With the
    chain guard passed and a unit for every S2 fold, each unit's training set is exactly the current one
    (the union of the other folds' checked test sets, knee.units); the labels are N0's (Stage A recorded
    the same studies.csv hash; labels_binary.csv belongs to that N0 run where agg_n0.json says so); pos_cap
    is the unit's own config.json value; where the unit recorded refit_steps, it must equal
    E x ceil(training studies / batch size).
    Gate (A4_MUST_BE_TRUE, A4_MUST_NOT_BE_FALSE): chain_guard_passed, unit_for_every_s2_fold,
    stage_a_n0_studies_same and pos_cap_recorded must be true; refit_steps_match_training_size and
    labels_binary_from_attached_n0_run must not be false (null = not recorded). info.stage_a_n5_index_same
    never gates."""
    K = c.d.n_folds("s2")
    folds = sorted(k for (e, k) in c.dirs if e == enc)
    checks = {"chain_guard_passed": bool(chain_ok), "unit_for_every_s2_fold": folds == list(range(K)),
              "stage_a_n0_studies_same": None, "pos_cap_recorded": None, "refit_steps_match_training_size": None,
              "labels_binary_from_attached_n0_run": _labels_from_attached_n0(c.d)}
    info = {"stage_a_n5_index_same": None}
    pw = {}
    if checks["chain_guard_passed"] and checks["unit_for_every_s2_fold"]:
        n0_same, n5_same, steps_ok, cap_ok = [], [], [], []
        for k in range(K):
            ud = c.dirs[(enc, k)]
            rec = _stage_a_upstream(ud.parent.parent.name, c.root)
            n0_same.append(None if rec is None else
                           (rec.get("n0_studies") or {}).get("sha16") == c.d.upstream["n0_studies"]["sha16"])
            n5_same.append(None if rec is None or "n5_index" not in rec else
                           (rec.get("n5_index") or {}).get("sha16") == c.d.upstream["n5_index"]["sha16"])
            u = json.loads((ud / "config.json").read_text())
            cap = (u.get("config") or {}).get("pos_cap")
            cap_ok.append(cap is not None)
            tr = c.d.split(k)[0]
            bs = (u.get("config") or {}).get("batch_size")
            if u.get("refit_steps") is not None and bs is not None and u.get("E") is not None:
                # n6._unit: refit_steps = E x ceil(training studies / batch size)
                steps_ok.append(int(u["refit_steps"]) == int(u["E"]) * math.ceil(len(tr) / max(1, int(bs))))
            else:
                steps_ok.append(None)
            if cap is None:
                continue
            yb = c.d.binary[tr]
            pos, neg = (yb == 1).sum(0), (yb == 0).sum(0)
            pw[k] = np.minimum(neg / np.maximum(pos, 1), float(cap)).astype(np.float32)  # heads.fit's tensor, float32

        def all_of(v):
            return None if any(x is None for x in v) else all(v)

        checks.update({"stage_a_n0_studies_same": all_of(n0_same), "pos_cap_recorded": all(cap_ok),
                       "refit_steps_match_training_size": (False if any(x is False for x in steps_ok) else
                                                           None if all(x is None for x in steps_ok) else True)})
        info["stage_a_n5_index_same"] = all_of(n5_same)
    for name, why in A4_MUST_BE_TRUE.items():
        if checks[name] is not True:
            return None, checks, info, why
    for name, why in A4_MUST_NOT_BE_FALSE.items():
        if checks[name] is False:
            return None, checks, info, why
    return pw, checks, info, None


def a4_calibration(c, chain_ok: bool) -> dict:
    """A4 (optional, closed form): subtract log(positive weight) from each logit of Stage A's predictions,
    per fold and finding, with the weight the fold's refit trained with (positive_weights), then ECE
    (N9's _ece) and the any-finding AUC (max of 12, pairs inside a test fold) before and after. Nothing
    is fitted. A monotone shift per fold and finding cannot change a fold-stratified AUC per finding, so
    the S2 macro (and S4) cannot move; the probabilities and the max-of-12 score can. For studies with
    unresolved laterality (stored and flipped slices averaged) the shift applies to the averaged
    probability. Every encoder is either in encoders or in skipped (with the reason): not attached,
    predictions incomplete, the gate of positive_weights, or a zero weight."""
    from scipy.special import expit, logit

    out = _dev("S2 evaluation studies: closed-form shift of each logit by -log(positive weight); ECE and "
               "any-finding AUC before and after; nothing fitted", action="A4", failure_mode="FM7",
               positive_weight_rule="min(negatives / max(positives, 1), pos_cap) on the outer training set (heads.fit), float32",
               gate_must_be_true=list(A4_MUST_BE_TRUE), gate_must_not_be_false=list(A4_MUST_NOT_BE_FALSE),
               info_never_gates=["stage_a_n5_index_same"], encoders={}, skipped={})
    pos_any = (c.Y[:, c.elig] == 1).any(axis=1).astype(float)

    def any_auc(P):
        return core._r(metrics.stratified_auc(P[c.ev][:, c.elig].max(axis=1), pos_any[c.ev], c.f2[c.ev])[0])

    for e in ENCODERS:
        P = c.units.get(e)
        if P is None:
            out["skipped"][e] = {"reason": "not attached"}
            continue
        if not c.complete(P):
            out["skipped"][e] = {"reason": "Stage A predictions incomplete on the evaluation studies"}
            continue
        pw, checks, info, why = positive_weights(c, e, chain_ok)
        if why is not None:
            out["skipped"][e] = {"reason": why, "checks": checks, "info": info}
            continue
        if not all(np.isfinite(w).all() and (w > 0).all() for w in pw.values()):
            out["skipped"][e] = {"reason": "a positive weight is 0 (a finding without negatives in training)",
                                 "checks": checks, "info": info}
            continue
        Q = np.array(P, dtype=float, copy=True)
        for k, w in pw.items():
            rows = c.ev[c.f2[c.ev] == k]
            Q[rows] = expit(logit(np.asarray(P[rows], dtype=float)) - np.log(w.astype(np.float64)))
        before = metrics.macro(P[c.ev], c.Y[c.ev], c.f2[c.ev], c.elig)
        after = metrics.macro(Q[c.ev], c.Y[c.ev], c.f2[c.ev], c.elig)
        logw = np.mean(np.log(np.stack([pw[k] for k in sorted(pw)]).astype(np.float64)), axis=0)
        out["encoders"][e] = {
            "checks": checks, "info": info,
            "ece": {"before": core._ece(P[c.ev], c.Y[c.ev]), "after": core._ece(Q[c.ev], c.Y[c.ev])},
            "any_finding_auc": {"before": any_auc(P), "after": any_auc(Q)},
            "fold_stratified_macro_auc_unchanged": bool(abs(before - after) < 1e-12),
            "log_positive_weight_mean_over_folds": {FINDINGS[j]: core._r(logw[j], 3) for j in c.elig}}
    return out


# ---------------------------------------------------------------- the notebook

def _n7_dir(root) -> Path | None:
    for name in (f"s2seeds_{core.DINO}.csv", f"s2seeds_{core.CONV}.csv"):
        p = paths.find_file(name, root)
        if p is not None:
            return p.parent
    return None


def _inputs(c) -> dict:
    """Which N7 seed-refit files were read (their hashes) and whether studies.csv has the report language."""
    n7 = c.n7_dir
    return {"n7_seed_refits_found": n7 is not None, "n7_device": n9x._n7_device(c),
            "s2seeds_sha16": {e: (common.sha16(n7 / f"s2seeds_{e}.csv")
                                  if n7 is not None and (n7 / f"s2seeds_{e}.csv").exists() else None) for e in ENCODERS},
            "language_column": "language" in pd.read_csv(Path(c.d.n0_dir) / "studies.csv", nrows=0).columns}


def _summary_values(s: dict) -> dict:
    def get(*path):
        v = core._at(s, path)
        return v if isinstance(v, dict) else {}

    gp, lq, pf, eg = get("gold_paired_check"), get("gold_label_quality"), get("per_finding_table"), \
        get("english_group_by_language")
    dev = get("development")

    def trio(v):
        return [v.get("estimate"), v.get("lo"), v.get("hi")]

    fm = get("fm5b_attenuation_check")
    sp = get("per_finding_table", "spearman_label_quality_vs_s2_auc")
    return {
        "gold_paired_check": {"studies": gp.get("studies_text"), "findings_same_as_n9_s3": gp.get("findings_same_as_n9_s3"),
                              "gold_studies_without_report_labels": gp.get("gold_studies_without_report_labels"),
                              "encoders": {e: {"expert_on_n9_findings": v.get("expert_labels_on_n9_findings"),
                                               "expert": v.get("expert_labels"), "report": v.get("report_labels"),
                                               "expert_minus_report": trio(v.get("expert_minus_report") or {})}
                                           for e, v in (gp.get("encoders") or {}).items()},
                              "status": gp.get("status")},
        "gold_label_quality": {"soft_label_macro_auc": lq.get("soft_label_macro_auc"),
                               "findings_suppressed": lq.get("findings_suppressed"), "rates_blank": lq.get("rates_blank")},
        "fm5b_report_minus_if_independent": {e: trio(v.get("report_minus_if_independent") or {})
                                             for e, v in (fm.get("encoders") or {}).items()},
        "spearman_label_quality_vs_s2_auc": {e: v.get("rho") for e, v in (sp.get("encoders") or {}).items()},
        "spearman_no_association_band": sp.get("no_association_band"),
        "per_finding_table_has_12_rows": len(pf.get("rows") or {}) == len(FINDINGS),
        "english_group": {"group": eg.get("group"), "fold": eg.get("fold"), "training_studies": eg.get("training_studies"),
                          "languages_reported": eg.get("languages_reported"),
                          "languages_suppressed": eg.get("languages_suppressed"), "findings_left_out": eg.get("findings_left_out"),
                          "rows": {L: {e: [v.get("macro_auc"), v.get("lo"), v.get("hi")] for e, v in (r.get("encoders") or {}).items()}
                                   for L, r in (eg.get("rows") or {}).items()},
                          "status": eg.get("status")},
        "development": {
            "label": DEV_LABEL,
            "a2_seed_ensemble": {e: [v.get("fold_stratified_macro_auc"), v.get("mean_of_fold_macros")]
                                 for e, v in (get("development", "a2_seed_ensemble", "encoders")).items()},
            "a3_encoder_ensemble": [get("development", "a3_encoder_ensemble").get("fold_stratified_macro_auc"),
                                    get("development", "a3_encoder_ensemble").get("mean_of_fold_macros")],
            "a4_calibration": {e: {"ece": [v["ece"]["before"], v["ece"]["after"]],
                                   "any_finding_auc": [v["any_finding_auc"]["before"], v["any_finding_auc"]["after"]]}
                               for e, v in (get("development", "a4_calibration", "encoders")).items()},
            "a4_skipped": {e: v.get("reason") for e, v in (get("development", "a4_calibration", "skipped")).items()},
            "metrics": "a2 and a3: [fold-stratified macro AUC, mean of fold macros]; a4: [before, after]"} if dev else None}


FINAL_KEEP = ("written_at_utc", "descriptive", "added_after_results", "scope", "upstream", "chain", "units_found",
              "findings", "main_complete", "evaluation_studies", "errors")


def _write_final(s: dict) -> None:
    """agg_n12.json with every block. If that write fails (each block passed the checks alone, so this is not
    expected), the file keeps the run's frame and the errors instead, and says so; the summary then shows it."""
    try:
        agg.write("n12", s)
    except Exception as e:  # noqa: BLE001 - never lose the record of the run
        errors = dict(s.get("errors") or {})
        errors["final_write"] = safe_log.describe(type(e), e.__traceback__)
        keep = {k: s[k] for k in FINAL_KEEP if k in s}
        s.clear()
        s.update({**keep, "errors": errors, "status": "final write failed: the blocks are not in this file"})
        agg.write("n12", s)


def _summary(s: dict) -> dict:
    out = {"status": s.get("status"), "chain_ok": (s.get("chain") or {}).get("s2_folds_match_units"),
           "main_complete": s.get("main_complete"), "errors": s.get("errors"),
           "descriptive_only": "N12 decides nothing and switches nothing on"}
    try:
        out.update(_summary_values(s))
    except Exception as e:  # noqa: BLE001 - the summary block must print whatever the blocks left
        out["summary_error"] = safe_log.describe(type(e), e.__traceback__)
    return out


def run(root=None, expect: dict | None = None, n_boot: int = intervals.N_BOOT, seed: int = 0,
        budget_h: float = 1.0, quiet: bool = True) -> dict:
    """The five items of plan r2, section 4, on the attached outputs. expect: the upstream hash prefixes
    {"n0_studies", "n3_folds", "n5_index"}, as in N9's card. n_boot: draws of every study bootstrap."""
    s: dict = {"written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **_flags(SCOPE),
               "suppression": {"min_positives": MIN_EACH, "min_negatives": MIN_EACH, "min_studies": MIN_STUDIES},
               "not_run": NOT_RUN}
    with common.notebook("n12", {"n_boot": n_boot, "seed": seed}, quiet=quiet, budget_h=budget_h):
        base = Path(root) if root else paths.INPUT
        rootp = Path(root) if root else None
        # N9's loading, line for line (n9.run): the same data, units, chain guard and evaluation set
        d = stage.StageData(root, expect)
        row_of = {u: i for i, u in enumerate(d.uids)}
        f2, Y = d.folds["s2_fold"], d.eval_y  # uncertain = negative
        dirs = unit_files.unit_dirs(base)
        units = core._load_units(base, d.n, row_of, dirs)
        unit_folds = units.pop("_folds")
        K = d.n_folds("s2")
        s["upstream"] = d.upstream
        s["chain"] = unit_files.chain_summary(d, dirs, rootp)
        if s["chain"]["s2_folds_match_units"] is False:
            s["status"] = "chain mismatch"
            agg.write("n12", s)
            raise unit_files.ChainError("Stage A units do not match the attached S2 folds")
        s["units_found"] = {e: {"folds": v} for e, v in unit_folds.items()}
        tr_all = np.flatnonzero(d.train & (f2 >= 0))
        ev_all = tr_all[d.scored[tr_all]]
        elig = list(range(len(FINDINGS)))
        s["findings"] = [FINDINGS[j] for j in elig]
        main_ok = all(e in units and len(unit_folds.get(e, [])) == K for e in (core.DINO, core.CONV))
        s["main_complete"] = main_ok
        if not main_ok:
            s["status"] = "main units missing"
            agg.write("n12", s)
        else:
            have = np.ones(d.n, bool)
            for e in (core.DINO, core.CONV):
                have &= np.isfinite(units[e]).all(axis=1)
            ev = ev_all[have[ev_all]]
            s["evaluation_studies"] = agg.count(len(ev))
            gold = d.gold_rows()
            gold = gold[np.isfinite(units[core.DINO][gold]).all(axis=1) & np.isfinite(units[core.CONV][gold]).all(axis=1)]
            g_find = metrics.eligible(d.gold_y[gold], np.zeros(len(gold), int)) if len(gold) else []
            c = n9x.Ctx(d=d, ev=ev, Y=Y, units=units, elig=elig, n_boot=n_boot, seed=seed, root=rootp, row_of=row_of,
                        floor=None, gold=gold, g_find=g_find, h1={}, cat_lang="not used", n7_dir=_n7_dir(rootp), dirs=dirs)
            chain_ok = s["chain"]["s2_folds_match_units"] is True
            errors: dict = {}
            s["errors"] = errors

            def put(name: str, fn, holder: dict, key: str) -> None:
                """One block on its own copy of the inputs; a failure or an unsafe value costs this block only."""
                try:
                    val = fn(copy.deepcopy(c))
                    agg._check(json.loads(json.dumps(val, default=agg._json_default)), path=name)
                    holder[key] = val
                except Exception as e:  # noqa: BLE001 - descriptive blocks must not stop each other
                    errors[name] = safe_log.describe(type(e), e.__traceback__)

            put("inputs", _inputs, s, "inputs")
            put("gold_paired_check", lambda c: gold_paired_check(c, n_boot, seed), s, "gold_paired_check")
            put("gold_label_quality", lambda c: gold_label_quality(c, n_boot, seed), s, "gold_label_quality")
            put("fm5b_attenuation_check", lambda c: fm5b_attenuation_check(c, n_boot, seed), s, "fm5b_attenuation_check")
            put("per_finding_table", lambda c: per_finding_table(c, n_boot, seed), s, "per_finding_table")
            put("english_group_by_language", lambda c: english_group_by_language(c, _languages(c.d), n_boot, seed), s,
                "english_group_by_language")
            s["development"] = _dev("S2 development numbers of A2, A3 and A4 (plan r2, section 5); "
                                    "never a gain over iteration 1, never in a slide title")
            put("development.a2_seed_ensemble", a2_seed_ensemble, s["development"], "a2_seed_ensemble")
            put("development.a3_encoder_ensemble", a3_encoder_ensemble, s["development"], "a3_encoder_ensemble")
            put("development.a4_calibration", lambda c: a4_calibration(c, chain_ok), s["development"], "a4_calibration")
            s["status"] = "ok" if not errors else "ok, some blocks failed (see errors)"
            _write_final(s)
    try:
        common.summary(_summary(s))
    except agg.UnsafeSummary:  # every value in it passed the checks already; never let the print cost the run
        common.summary({"status": s.get("status"), "errors": s.get("errors"),
                        "summary_error": "the summary block was refused by the checks; read agg_n12.json"})
    return s
