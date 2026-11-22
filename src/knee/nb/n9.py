"""N9: statistics and decisions (CPU). Its agg_results.json is the freeze file.

Inputs (attached notebook outputs): N0, N3, N5 (index only), N6a and N6b
units (S2 and gold predictions), N7 (S1 and G-vendor), N8 (floor A), the
pilot (its S2 macro findings must equal the ones recomputed here); optional
cell-2 values: the hidden-test scores (S4) and KneeMRI's agg_kneemri.json (S5).

Scoring rule (plan): uncertain labels count as negative; the sensitivity
analysis drops them. Fixed sets: the evaluation studies are the usable,
scored training studies with a test fold, the same for every encoder and
floor A; the S2 macro is the mean over all 12 findings (sensitivity: the
findings with at least 5 positives and 5 negatives in every test fold); H2's
findings are those with at least 5 positives among the evaluation studies;
gold's are those with at least 5 positives and 5 negatives among the gold
studies.

Decisions (plan, Phase 1), each with the union of a paired cluster bootstrap
over G-lang groups and a delete-one-group jackknife t-interval:
  H1: Delta1 = AUC(DINOv2) - AUC(ConvNeXt), fold-stratified macro AUC, under
      G-lang folds and under G-vendor folds (clustered on G-lang); the same
      category is the decision, else "inconclusive (grouping-sensitive)".
  H2: Delta2 = gap(DINOv2) - gap(ConvNeXt), gap = within-site AUC on S1 minus
      on S2; "supported" = upper bound below 0 and not equivalent; "opposite" =
      lower bound above 0.
  H3: floor A's macro AUC; supported if its lower bound is above 0.5.
  Anti-claim 1: dArch = ViT-S/16 - ConvNeXt, dPre = DINOv2 - ViT-S/16.
  S3 (gold) and S4 (hidden test): "consistent" = the same sign as Delta1.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, intervals, metrics, paths, stage
from ..labels import FINDINGS, UID
from . import common

TARGETS = {"s2_mean_of_fold_macros": 0.78, "s2_stretch": 0.82, "gold_macro_auc": 0.85, "kneemri_acl": 0.80}
DINO, CONV, VIT = "dinov2_vits14", "convnext_tiny", "vit_s16"


def _load_units(base: Path, n: int, row_of: dict) -> dict:
    """{encoder: [N, 12] S2 predictions} from every attached unit (a rerun replaces the original)."""
    out: dict = {}
    seen: dict = {}
    for p in sorted(base.glob("**/units/*_f*/predictions.csv")):
        enc, fold = p.parent.name.rsplit("_f", 1)
        key = (enc, int(fold))
        if key in seen and not common.is_rerun(p.parent):
            continue
        seen[key] = p
    for (enc, _), p in sorted(seen.items()):
        df = pd.read_csv(p)
        P = out.setdefault(enc, np.full((n, len(FINDINGS)), np.nan))
        rows = df[UID].astype(str).map(row_of)
        ok = rows.notna().to_numpy()
        P[rows[ok].astype(int).to_numpy()] = df.loc[ok, FINDINGS].to_numpy(float)
    out["_folds"] = {e: sorted(int(f) for (ee, f) in seen if ee == e) for e in {e for e, _ in seen}}
    return out


def _load_csv(path: Path | None, n: int, row_of: dict) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    P = np.full((n, len(FINDINGS)), np.nan)
    rows = df[UID].astype(str).map(row_of)
    ok = rows.notna().to_numpy()
    P[rows[ok].astype(int).to_numpy()] = df.loc[ok, FINDINGS].to_numpy(float)
    return P


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def _interval(stat, groups, n_boot, seed) -> dict:
    di = intervals.decision_interval(stat, groups, n_boot=n_boot, seed=seed)
    return {"estimate": _r(di["estimate"]), "lo": _r(di["lo"]), "hi": _r(di["hi"]),
            "half_width": _r(di["half_width"]),
            "bootstrap": {"lo": _r(di["bootstrap"]["lo"]), "hi": _r(di["bootstrap"]["hi"]),
                          "valid_share": _r(di["bootstrap"]["valid_share"], 3)},
            "jackknife": {"lo": _r(di["jackknife"]["lo"]), "hi": _r(di["jackknife"]["hi"]),
                          "se": _r(di["jackknife"]["se"])},
            "n_groups": di["n_groups"], "approximate": di["approximate"]}


def _controls(root) -> dict:
    """The pilot's controls and every notebook's leakage flag, copied into the freeze file."""
    out = {}
    p = paths.find_file("agg_pilot.json", root)
    if p is not None:
        pj = json.loads(p.read_text())
        pc, oc = pj.get("positive_control") or {}, pj.get("overlap_control") or {}
        out["pilot"] = {"positive_control_all_passed": pc.get("all_passed"),
                        "overlap_control_passed": oc.get("passed"), "known_answer_detected": oc.get("known_answer_detected"),
                        "leakage_assertion_passed": pj.get("leakage_assertion_passed"),
                        "half_width_delta1": (pj.get("half_width_delta1") or {}).get("union")}
    for name in ("n6a", "n6b", "n8"):
        p = paths.find_file(f"agg_{name}.json", root)
        if p is not None:
            j = json.loads(p.read_text())
            out[name] = {"leakage_assertion_passed": j.get("leakage_assertion_passed"),
                         **({"all_units_ok": j.get("all_units_ok"), "all_nested": j.get("all_nested")}
                            if name != "n8" else {"usable_from_n5": j.get("usable_from_n5")})}
    flags = [v.get("leakage_assertion_passed") is True for v in out.values() if isinstance(v, dict)]
    out["all_leakage_assertions_passed"] = all(flags) if flags else None
    return out


def _h2_category(lo, hi) -> str:
    c = intervals.decide(lo, hi)
    return {"B better": "supported", "A better": "opposite"}.get(c, c)


def _ece(p, y, bins: int = 10) -> float | None:
    vals = []
    for j in range(y.shape[1]):
        m = ~np.isnan(y[:, j]) & np.isfinite(p[:, j])
        if m.sum() < 20:
            continue
        pj, yj = p[m, j], y[m, j]
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(pj, edges) - 1, 0, bins - 1)
        e = sum(abs(pj[idx == b].mean() - yj[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
        vals.append(e)
    return round(float(np.mean(vals)), 4) if vals else None


def run(root=None, expect: dict | None = None, hidden: dict | None = None, kneemri_json: str | None = None,
        n_boot: int = intervals.N_BOOT, seed: int = 0, budget_h: float = 2.0, quiet: bool = True) -> dict:
    """hidden: {"dinov2_vits14": score, "convnext_tiny": score} from the Kaggle late submissions (S4).
    kneemri_json: path or https URL (the repo's raw file) of agg_kneemri.json (S5), if KneeMRI ran."""
    s: dict = {"frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cfg = {"n_boot": n_boot, "seed": seed, "hidden": bool(hidden), "kneemri": bool(kneemri_json)}
    with common.notebook("n9", cfg, quiet=quiet, budget_h=budget_h):
        base = Path(root) if root else paths.INPUT
        d = stage.StageData(root, expect)
        row_of = {u: i for i, u in enumerate(d.uids)}
        f2, fv, f1 = d.folds["s2_fold"], d.folds["vendor_fold"], d.folds["s1_fold"]
        Y = d.eval_y  # uncertain = negative
        Y_drop = d.binary  # sensitivity: uncertain dropped
        units = _load_units(base, d.n, row_of)
        unit_folds = units.pop("_folds")
        K = d.n_folds("s2")
        s["upstream"] = d.upstream
        s["units_found"] = {e: {"folds": v} for e, v in unit_folds.items()}
        tr_all = np.flatnonzero(d.train & (f2 >= 0))
        ev_all = tr_all[d.scored[tr_all]]
        elig = list(range(len(FINDINGS)))
        five = metrics.eligible(Y[ev_all], f2[ev_all])
        pilot = paths.find_file("agg_pilot.json", Path(root) if root else None)
        pilot_f = json.loads(pilot.read_text()).get("findings") if pilot else None
        s["findings"] = {"s2": [FINDINGS[j] for j in elig], "sensitivity_5x5": [FINDINGS[j] for j in five],
                         "matches_pilot": (pilot_f == list(FINDINGS)) if pilot_f is not None else None}
        main_ok = all(e in units and len(unit_folds.get(e, [])) == K for e in (DINO, CONV))
        s["main_complete"] = main_ok
        if not main_ok:
            s["status"] = "main units missing"
            agg.write("results", s)
        else:
            # fixed evaluation set: studies every available model scored
            have = np.ones(d.n, bool)
            for e in (DINO, CONV):
                have &= np.isfinite(units[e]).all(axis=1)
            ev = ev_all[have[ev_all]]
            s["evaluation_studies"] = agg.count(len(ev))
            s["evaluation_share_of_scored"] = agg.share(len(ev) / max(len(ev_all), 1))
            g = d.g_lang[ev]

            def m(P, rows, folds, fl, w=None):
                return metrics.macro(P[rows], Y[rows], folds[rows], fl, w)

            # per-encoder S2 summaries
            enc_s2 = {}
            for e, P in units.items():
                full = np.isfinite(P[ev]).all(axis=1).all()
                enc_s2[e] = {
                    "complete": bool(full),
                    "fold_stratified_macro_auc": _r(m(P, ev, f2, elig)) if full else None,
                    "mean_of_fold_macros": _r(metrics.per_fold_macro_mean(P[ev], Y[ev], f2[ev], elig)) if full else None,
                    "pooled_macro_auc": _r(metrics.macro(P[ev], Y[ev], np.zeros(len(ev), int), elig)) if full else None,
                    "per_finding": {FINDINGS[j]: _r(metrics.stratified_auc(P[ev, j], Y[ev, j], f2[ev])[0], 3) for j in elig} if full else None,
                    "ece": _ece(P[ev], Y[ev]) if full else None,
                }
            s["s2"] = enc_s2
            s["positives_per_finding"] = {FINDINGS[j]: agg.count(int((Y[ev, j] == 1).sum())) for j in elig}

            # H1 under G-lang
            A, B = units[DINO], units[CONV]

            def d1(w):
                return m(A, ev, f2, elig, w) - m(B, ev, f2, elig, w)

            h1_lang = _interval(d1, g, n_boot, seed)
            cat_lang = intervals.decide(h1_lang["lo"], h1_lang["hi"], a="DINOv2 better", b="ConvNeXt better")

            # sensitivity analyses of H1 (G-lang): uncertain dropped; the 5/5 finding set
            def d1_drop(w):
                return (metrics.macro(A[ev], Y_drop[ev], f2[ev], elig, w)
                        - metrics.macro(B[ev], Y_drop[ev], f2[ev], elig, w))

            sens = {"uncertain_dropped": _interval(d1_drop, g, n_boot, seed)}
            if five and five != elig:
                sens["findings_5x5"] = _interval(lambda w: m(A, ev, f2, five, w) - m(B, ev, f2, five, w), g, n_boot, seed)
            for v in sens.values():
                v["category"] = intervals.decide(v["lo"], v["hi"], a="DINOv2 better", b="ConvNeXt better")
            # each fold's share of H1's pairs, and the per-fold Delta1
            per_fold, pw = {}, {}
            for k in range(K):
                rows = ev[f2[ev] == k]
                one = np.zeros(len(rows), int)
                pw[k] = sum(metrics.stratified_auc(A[rows, j], Y[rows, j], one)[1] for j in elig)
                per_fold[f"fold_{k}"] = {
                    "delta1": _r(metrics.macro(A[rows], Y[rows], one, elig) - metrics.macro(B[rows], Y[rows], one, elig)),
                    "groups": sorted(set(map(str, d.g_lang[rows])))}
            tot = sum(pw.values()) or 1.0
            for k in range(K):
                per_fold[f"fold_{k}"]["pair_share"] = agg.share(pw[k] / tot)
            heavy = [v["groups"] for v in per_fold.values() if (v["pair_share"] or 0) > 0.5]

            # H1 under G-vendor (consensus configs, not nested), clustered on G-lang
            nv = paths.find_file(f"vendor_{DINO}.csv", Path(root) if root else None)
            VA = _load_csv(nv, d.n, row_of)
            VB = _load_csv(nv.parent / f"vendor_{CONV}.csv", d.n, row_of) if nv else None
            if VA is not None and VB is not None:
                evv = ev[(fv[ev] >= 0) & np.isfinite(VA[ev]).all(axis=1) & np.isfinite(VB[ev]).all(axis=1)]

                def d1v(w):
                    return m(VA, evv, fv, elig, w) - m(VB, evv, fv, elig, w)

                h1_vendor = _interval(d1v, d.g_lang[evv], n_boot, seed)
                cat_vendor = intervals.decide(h1_vendor["lo"], h1_vendor["hi"], a="DINOv2 better", b="ConvNeXt better")
            else:
                h1_vendor, cat_vendor = None, "not run"
            s["h1"] = {"g_lang": {**h1_lang, "category": cat_lang}, "g_vendor": (
                {**h1_vendor, "category": cat_vendor, "label": "not nested"} if h1_vendor else {"category": cat_vendor}),
                "decision": intervals.combine_groupings(cat_lang, cat_vendor) if h1_vendor else f"{cat_lang} (G-vendor not run)",
                "per_fold": per_fold, "fold_over_half_of_pairs": heavy, "sensitivity": sens}

            # H2: within-site AUC gap, S1 minus S2
            s1A = _load_csv(nv.parent / f"s1_{DINO}.csv", d.n, row_of) if nv else None
            s1B = _load_csv(nv.parent / f"s1_{CONV}.csv", d.n, row_of) if nv else None
            h2_find = [j for j in range(len(FINDINGS)) if (Y[ev, j] == 1).sum() >= 5]
            if s1A is not None and s1B is not None:
                ev2 = ev[(f1[ev] >= 0) & np.isfinite(s1A[ev]).all(axis=1) & np.isfinite(s1B[ev]).all(axis=1)]
                st_s1 = metrics.strata_codes(f1[ev2], d.g_lang[ev2])
                st_s2 = metrics.strata_codes(f2[ev2], d.g_lang[ev2])

                def ws(P, st, w):
                    return metrics.macro(P[ev2], Y[ev2], st, h2_find, w)

                def d2(w):
                    return (ws(s1A, st_s1, w) - ws(A, st_s2, w)) - (ws(s1B, st_s1, w) - ws(B, st_s2, w))

                h2 = _interval(d2, d.g_lang[ev2], n_boot, seed)
                s["h2"] = {**h2, "category": _h2_category(h2["lo"], h2["hi"]), "findings": [FINDINGS[j] for j in h2_find],
                           "gap": {e: _r(ws(Pa, st_s1, None) - ws(Pb, st_s2, None))
                                   for e, Pa, Pb in ((DINO, s1A, A), (CONV, s1B, B))}}
            else:
                s["h2"] = {"category": "not run (cut)"}

            # H3: floor A
            FA = None
            for cand in sorted(base.glob("**/private/n8/predictions.csv")):
                FA = _load_csv(cand, d.n, row_of)
                break
            if FA is not None and np.isfinite(FA[ev]).all():
                h3 = _interval(lambda w: m(FA, ev, f2, elig, w), g, n_boot, seed)
                s["h3"] = {**h3, "supported": bool(h3["lo"] is not None and h3["lo"] > 0.5)}
            else:
                s["h3"] = {"supported": None, "status": "floor A missing or incomplete"}

            # Anti-claim 1: architecture vs pretraining
            if VIT in units and np.isfinite(units[VIT][ev]).all():
                C = units[VIT]
                s["anti_claim_1"] = {
                    "d_arch": _interval(lambda w: m(C, ev, f2, elig, w) - m(B, ev, f2, elig, w), g, n_boot, seed),
                    "d_pre": _interval(lambda w: m(A, ev, f2, elig, w) - m(C, ev, f2, elig, w), g, n_boot, seed),
                }
            else:
                s["anti_claim_1"] = None

            # S3 gold (descriptive): pooled over fold models, gold findings fixed
            gold = d.gold_rows()
            gold = gold[np.isfinite(A[gold]).all(axis=1) & np.isfinite(B[gold]).all(axis=1)]
            GY = d.gold_y
            g_find = metrics.eligible(GY[gold], np.zeros(len(gold), int)) if len(gold) else []
            if g_find:
                gm = {e: _r(metrics.macro(P[gold], GY[gold], np.zeros(len(gold), int), g_find)) for e, P in units.items()
                      if np.isfinite(P[gold]).all()}
                dg = (gm.get(DINO) or 0) - (gm.get(CONV) or 0)
                s["s3_gold"] = {"studies": agg.count(len(gold)), "findings": [FINDINGS[j] for j in g_find],
                                "macro_auc": gm, "delta1": _r(dg),
                                "consistent": bool(np.sign(dg) == np.sign(h1_lang["estimate"] or 0))}
            else:
                s["s3_gold"] = {"status": "no gold findings"}

            # S4 hidden test (descriptive)
            if hidden and DINO in hidden and CONV in hidden:
                dh = float(hidden[DINO]) - float(hidden[CONV])
                s["s4_hidden"] = {"scores": {k: _r(v) for k, v in hidden.items()}, "delta": _r(dh),
                                  "consistent": bool(np.sign(dh) == np.sign(h1_lang["estimate"] or 0))}
            else:
                s["s4_hidden"] = {"status": "not run"}

            # Per-language AUC (descriptive)
            per_lang = {}
            for lg in sorted(set(map(str, d.g_lang[ev]))):
                rows = ev[d.g_lang[ev].astype(str) == lg]
                small = len(rows) < 5  # suppressed, like any count below 5
                per_lang[lg] = {"studies": agg.count(len(rows)),
                                **{e: (None if small else _r(metrics.macro_available(P[rows], Y[rows], elig), 3))
                                   for e, P in units.items()}}
            s["per_language"] = per_lang

            # S5 KneeMRI (from the laptop)
            km = None
            if kneemri_json and str(kneemri_json).startswith("https://"):
                import urllib.request

                with urllib.request.urlopen(str(kneemri_json), timeout=60) as r:  # the repo's committed summary
                    km = json.loads(r.read().decode())
            elif kneemri_json and Path(kneemri_json).exists():
                km = json.loads(Path(kneemri_json).read_text())
            if km is not None:
                agg._check(km)
                s["s5_kneemri"] = km
            else:
                s["s5_kneemri"] = {"status": "not run"}

            # Per-finding Delta1 (exploratory: the same intervals, unadjusted; they decide nothing)
            s["h1"]["per_finding_exploratory"] = {
                FINDINGS[j]: _interval(lambda w, j=j: m(A, ev, f2, [j], w) - m(B, ev, f2, [j], w), g, n_boot, seed)
                for j in elig}

            # Controls and leakage flags from the other notebooks
            s["controls"] = _controls(Path(root) if root else None)

            # Targets vs achieved (the better encoder on S2)
            better = DINO if (h1_lang["estimate"] or 0) >= 0 else CONV
            s["targets"] = {
                "encoder": better,
                "s2_mean_of_fold_macros": {"target": TARGETS["s2_mean_of_fold_macros"], "stretch": TARGETS["s2_stretch"],
                                           "achieved": enc_s2[better]["mean_of_fold_macros"]},
                "gold_macro_auc": {"target": TARGETS["gold_macro_auc"],
                                   "achieved": s["s3_gold"].get("macro_auc", {}).get(better) if isinstance(s["s3_gold"].get("macro_auc"), dict) else None},
                "kneemri_acl": {"target": TARGETS["kneemri_acl"],
                                "achieved": (s["s5_kneemri"].get("in_domain_acl_auc") or {}).get(better)},
            }
            s["decisions"] = {"H1": s["h1"]["decision"], "H2": s["h2"]["category"],
                              "H3": ("supported" if s["h3"].get("supported") else
                                     "not supported" if s["h3"].get("supported") is False else "not run"),
                              "H4": "not run (cut 22 Nov)"}
            agg.write("results", s)
    common.summary({"decisions": s.get("decisions"), "main_complete": s.get("main_complete"),
                    "controls_ok": (s.get("controls") or {}).get("all_leakage_assertions_passed"),
                    "findings_match_pilot": s.get("findings", {}).get("matches_pilot"),
                    "status": s.get("status")})
    return s
