"""N14: error analysis after the error hunt (CPU; it3_error_hunt_r3.md, n14_spec_r3.md). Descriptive only.

Reads existing outputs: knee-n0 (studies.csv with each plane's chosen series and tier, the labels, the
expert gold labels), knee-n3-r (folds.csv), knee-n5 (embeddings), the Stage A units of knee-n6a and
knee-n6b-g (predictions.csv, config.json, head.pt, stats.pt) and knee-n7-b (agg_n7.json's consensus
configs, private/n7/s1_<encoder>.csv). C5 also reads the first DICOM header of each chosen series.

What it is for (plan r2, R8, as Ablay set the order on 24 Nov at 11:15): error analysis first, then the
next change is chosen from the failure-mode report and written into the plan before it runs (R3); a
change is kept or dropped on inner validation only (R4). So these blocks may point at a failure to
address; no S2 or gold number here keeps or switches on any change. N7, N9, N10 and N13 never read
this output. C1 fits heads (gold scored by an all-data model); nothing else is fitted.

Blocks (keys of agg_n14.json), each with descriptive true, added_after_results true and its reading rule
applied as written in n14_spec_r3.md before any number existed:
  c1_gold_all_data          FM1 / gold target: gold scored by heads fit on every training study (development)
  c2_side_discrimination    side information in the S2 outputs (menisci; OA)
  c3_sagittal_attention     slice localisation of the S2 heads (ACL, medial and lateral meniscus)
  c4_fallback_contrast      S2 macro AUC: fluid fat-suppressed series in all three planes vs the rest
  c5_centre_crop            header pass: square share, longer axis, cut fraction, mm per pixel
  c6_s1_level               S1 mean of fold macros beside S2's (site part only)
Privacy as N12: counts through agg.count, gold macros at 4 decimals only over at least 5 findings and no
per-finding number on gold, nothing per study, no identifier.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import agg, dicom_io, heads, intervals, metrics, paths, safe_log, stage
from .. import units as unit_files
from ..labels import FINDINGS, UID
from . import common
from . import n9 as core
from . import n12

ENCODERS = (core.DINO, core.CONV, core.VIT)
MAIN = (core.DINO, core.CONV)
PLANES = heads.PLANES
LOOKALIKE_COSINE = 0.90  # N9's look-alike probe (n9c.LOOKALIKE_COSINE)
MIN_SIDE = 50  # C2: studies needed on each side of a pair
SIDE_PAIRS = {"menisci": ("Medial Meniscus", "Lateral Meniscus"), "oa": ("Medial OA", "Lateral OA")}
C3_FINDINGS = ("ACL", "Medial Meniscus", "Lateral Meniscus")
CUT_RULE = 0.15  # C5: cut fraction
SHARE_RULE = 0.20  # C5: share of studies
DEV_LABEL = n12.DEV_LABEL
SCOPE = ("error analysis after the error hunt (it3_error_hunt_r3, n14_spec_r3): existing outputs and C1's "
         "all-data refit; no number here keeps or switches on a change (R8)")


def _flags(scope: str, **extra) -> dict:
    return {"descriptive": True, "added_after_results": True, "scope": scope, **extra}


def _r(x, nd=4):
    return core._r(x, nd)


def _band(values: list, lo: float, hi: float, names=("lost", "partial", "present")) -> str:
    """The C2 band rule: under lo, between, at least hi; it holds only if every value falls in the same band."""
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return "not available"
    bands = {names[0] if v < lo else names[2] if v >= hi else names[1] for v in values}
    return bands.pop() if len(bands) == 1 else "mixed"


# ---------------------------------------------------------------- C1: gold scored by an all-data model

def _consensus(root) -> dict:
    p = paths.find_file("agg_n7.json", root)
    if p is None:
        raise FileNotFoundError("agg_n7.json not attached")
    j = json.loads(p.read_text())
    return {"consensus": j.get("consensus") or {}, "device": j.get("device"),
            "sha16": common.sha16(p)}


def gold_lookalikes(d, gold) -> np.ndarray:
    """Training rows whose DINOv2 study vector has cosine above 0.90 with a gold study (N9's probe code)."""
    pairs = stage.near_duplicates(d.study_vectors(core.DINO), np.flatnonzero(d.usable), LOOKALIKE_COSINE, "cpu")
    ra, rb = pairs["row_a"].astype(int).to_numpy(), pairs["row_b"].astype(int).to_numpy()
    g = np.zeros(d.n, bool)
    g[gold] = True
    hit = set(rb[g[ra] & d.train[rb]].tolist()) | set(ra[g[rb] & d.train[ra]].tolist())
    return np.array(sorted(hit), dtype=int)


def c1_gold_all_data(c, n_boot: int, seed: int, device: str = "cpu") -> dict:
    """Gold scored by heads fit on every training study (look-alikes of gold removed), beside N9's S3 values.

    Fit: per encoder, N7's consensus config and E (agg_n7.json of knee-n7-b), seed 0 (Stage A's refit seed),
    every training study except the training look-alikes of gold; gold is passed as `forbidden`, as in every
    fit. Score: N9's gold studies and gold findings, expert labels (uncertain does not exist on gold).
    Beside it: N9's two S2 values on the same studies (pooled over the fold models; fold-stratified by the
    S2 fold) and the paired difference all-data minus pooled, with a study bootstrap (percentile 2.5-97.5).
    Development (R6): gold was scored in iteration 1. Config (consensus vs each fold's tuned one) and device
    (CPU vs Stage A's GPU) differ; each moved S2 by up to about 0.007, so a difference under 0.02 is within them."""
    d, gold, g_find = c.d, c.gold, c.g_find
    out = _flags("gold studies scored by heads fit on every training study; development", label=DEV_LABEL,
                 failure_mode="FM1 / gold target", n_boot=int(n_boot),
                 interval="study bootstrap, paired, percentile 2.5-97.5",
                 caveat="consensus config on CPU vs each fold's tuned config on GPU: differences under 0.02 are "
                        "within config and device effects")
    if not len(gold) or not g_find:
        return {**out, "status": "not available (no gold studies or findings)"}
    con = _consensus(c.root)
    out["n7"] = {"device": con["device"], "agg_n7_sha16": con["sha16"]}
    f2 = d.folds["s2_fold"]
    K = d.n_folds("s2")
    out["gold_studies"] = agg.count(len(gold))
    out["gold_per_s2_fold"] = {f"fold_{k}": agg.count(int((f2[gold] == k).sum())) for k in range(K)}
    out["findings"] = [FINDINGS[j] for j in g_find]
    look = gold_lookalikes(d, gold)
    tr = np.flatnonzero(d.train)
    tr_fit = np.setdiff1d(tr, look)
    out["lookalikes_removed"] = agg.count(len(look))
    out["training_studies"] = agg.count(len(tr_fit))
    nd = n12._macro_decimals(g_find)
    GY = d.gold_y
    zeros = np.zeros(len(gold), int)
    f2g = f2[gold]

    def mac(P, w=None, strata=None):
        return metrics.macro(P[gold], GY[gold], zeros if strata is None else strata, g_find, w)

    out["encoders"] = {}
    for e in ENCODERS:
        if e not in con["consensus"] or e not in c.units or not np.isfinite(c.units[e][gold]).all():
            continue
        cc = con["consensus"][e]
        cfg, E = heads.config_from(cc["config"]), int(cc["E"])
        bank = d.bank(e, device)
        stats = bank.stats(tr_fit)
        r = heads.fit(bank, cfg, tr_fit, epochs=E, stats=stats, forbidden=gold, seed=0)
        P = np.full((d.n, len(FINDINGS)), np.nan)
        P[gold] = heads.predict(bank, r.model, gold, cfg, stats)
        del bank
        S2 = c.units[e]
        vals = np.full((int(n_boot), 4), np.nan)  # one draw for all four numbers (paired)
        for b, w in enumerate(n12._draws(len(gold), int(n_boot), seed)):
            vals[b] = mac(P, w), mac(S2, w), mac(S2, w, f2g), mac(P, w, f2g)
        a, p, fs, afs = mac(P), mac(S2), mac(S2, None, f2g), mac(P, None, f2g)
        out["encoders"][e] = {
            "all_data": _r(a, nd), "all_data_interval": n12._percentile(vals[:, 0], nd),
            "s2_pooled": _r(p, nd), "s2_fold_stratified": _r(fs, nd),
            "all_data_fold_stratified": _r(afs, nd),
            "all_data_minus_s2_pooled": {"estimate": _r(a - p, nd), **n12._percentile(vals[:, 0] - vals[:, 1], nd)},
            "all_data_minus_s2_fold_stratified": {"estimate": _r(afs - fs, nd),
                                                  **n12._percentile(vals[:, 3] - vals[:, 2], nd),
                                                  "note": "both scored on pairs inside one S2 fold"},
            "E": E, "config_source": "N7 consensus", "pooling": cfg.pooling, "planes": cfg.planes}
    diffs = [out["encoders"].get(e, {}).get("all_data_minus_s2_pooled", {}).get("estimate") for e in MAIN]
    if any(x is None for x in diffs):
        reading = "not available"
    elif all(x >= 0.05 for x in diffs):
        reading = "the protocol explains a large part of the gold miss"
    elif all(x < 0.02 for x in diffs):
        reading = "the protocol does not explain the gold miss"
    else:
        reading = "the protocol explains part of the gold miss"
    out["reading_rule"] = ("all-data minus S2 pooled at least 0.05 for both main encoders = large part; under 0.02 "
                           "for both = does not; otherwise partly (descriptive; gold SE about 0.024)")
    out["reading"] = reading
    return out


# ---------------------------------------------------------------- C2: side discrimination

def c2_side_discrimination(c, n_boot: int, seed: int) -> dict:
    """Among scored S2 studies with exactly one side positive (report labels, uncertain = negative), the
    fold-stratified AUC of p_medial - p_lateral for "medial positive" against "lateral positive". The
    unresolved-laterality studies (flip_row >= 0) are left out. 0.5 = no side information."""
    d, f2 = c.d, c.d.folds["s2_fold"]
    out = _flags("side information in the S2 Stage A outputs; report labels",
                 studies="scored S2 evaluation studies with exactly one side positive; unresolved laterality left out",
                 min_studies_per_side=MIN_SIDE, n_boot=int(n_boot),
                 interval="study bootstrap, percentile 2.5-97.5",
                 reading_rule="per pair: under 0.65 = side information largely lost; at least 0.75 = present; "
                              "between = partial; a band holds only if both main encoders fall in it, else mixed")
    resolved = d.flip_row < 0
    base = c.ev[resolved[c.ev]]
    out["unresolved_left_out"] = agg.count(int((~resolved[c.ev]).sum()))
    out["pairs"] = {}
    for name, (fm, fl) in SIDE_PAIRS.items():
        jm, jl = FINDINGS.index(fm), FINDINGS.index(fl)
        one = (c.Y[base, jm] + c.Y[base, jl]) == 1
        rows = base[one]
        lab = c.Y[rows, jm]
        n_med, n_lat = int((lab == 1).sum()), int((lab == 0).sum())
        entry = {"findings": [fm, fl], "medial_only": agg.count(n_med), "lateral_only": agg.count(n_lat)}
        if min(n_med, n_lat) < MIN_SIDE:
            out["pairs"][name] = {**entry, "status": f"suppressed (under {MIN_SIDE} studies on a side)"}
            continue
        entry["encoders"] = {}
        draws = list(n12._draws(len(rows), int(n_boot), seed))
        for e, P in c.units.items():
            if not np.isfinite(P[rows][:, [jm, jl]]).all():
                continue
            score = P[rows, jm] - P[rows, jl]
            auc = metrics.stratified_auc(score, lab, f2[rows])[0]
            boot = [metrics.stratified_auc(score, lab, f2[rows], w)[0] for w in draws]
            entry["encoders"][e] = {"side_auc": _r(auc), **n12._percentile(boot)}
        entry["reading"] = _band([entry["encoders"].get(e, {}).get("side_auc") for e in MAIN], 0.65, 0.75)
        out["pairs"][name] = entry
    return out


# ---------------------------------------------------------------- C3: sagittal attention profiles

def _load_head(ud: Path, dim: int):
    import torch

    cfg = heads.config_from(json.loads((ud / "config.json").read_text())["config"])
    planes = heads.PLANE_SETS[cfg.planes]
    model = heads.make_head(cfg, dim, len(planes))
    model.load_state_dict(torch.load(ud / "head.pt", map_location="cpu"))
    model.eval()
    raw = torch.load(ud / "stats.pt", map_location="cpu")
    stats = {p: (mu.float(), sd.float()) for p, (mu, sd) in raw.items()}
    return cfg, planes, model, stats


def c3_sagittal_attention(c) -> dict:
    """Each main encoder's S2 fold heads: the attention over the sagittal slices for its test studies
    (evaluation studies with resolved laterality), mean profile per finding by report label.
    Index 0 = lateral, last = medial after laterality normalisation (dicom_io.TARGET, prepare_volume).
    ACL: share of the attention on the middle third (slices 11-21 of 32; uniform = 0.34). Menisci: centre of
    mass of the positives' profile (slice index)."""
    import torch

    d = c.d
    out = _flags("slice localisation of the S2 Stage A heads, sagittal plane; report labels",
                 studies="S2 evaluation studies of each fold's test set; unresolved laterality left out",
                 index="0 = lateral, last = medial (after laterality normalisation)",
                 reading_rule=["ACL: middle-third share for positives under 0.45 for both main encoders = the heads "
                               "do not find the ACL slices",
                               "menisci: medial minus lateral centre of mass at least 5 slices (of 32) for both main "
                               "encoders = sagittal side information survives"])
    in_ev = np.zeros(d.n, bool)
    in_ev[c.ev] = True
    out["encoders"], out["skipped_units"] = {}, []
    for e in MAIN:
        bank = d.bank(e, "cpu")
        S = int(bank.x["sagittal"].shape[1])
        dim = int(bank.x["sagittal"].shape[-1])
        acc = {f: {"pos": np.zeros(S), "neg": np.zeros(S), "n_pos": 0, "n_neg": 0} for f in C3_FINDINGS}
        for (enc, k), ud in sorted(c.dirs.items()):
            if enc != e:
                continue
            cfg, planes, model, stats = _load_head(ud, dim)
            if cfg.pooling != "gated" or "sagittal" not in planes:
                out["skipped_units"].append(f"{e}_f{k}")
                continue
            si = planes.index("sagittal")
            te = d.split(k)[1]
            rows = te[in_ev[te] & (d.flip_row[te] < 0)]
            # the rebuilt head must reproduce the unit's own predictions (no flips on these rows)
            gap = float(np.nanmax(np.abs(heads.predict(bank, model, rows, cfg, stats, use_flip=False)
                                         - c.units[e][rows]))) if len(rows) else float("nan")
            out.setdefault("max_abs_gap_vs_predictions", {})[f"{e}_f{k}"] = round(gap, 5) if np.isfinite(gap) else None
            if not np.isfinite(gap) or gap > 1e-3:
                out["skipped_units"].append(f"{e}_f{k}")
                continue
            for i in range(0, len(rows), 256):
                b = rows[i:i + 256]
                xs = heads._inputs(bank, torch.as_tensor(b), planes, stats)
                with torch.no_grad():
                    att = model.attention(xs)[si].float().numpy()  # [B, S, 12]
                for f in C3_FINDINGS:
                    j = FINDINGS.index(f)
                    y = c.Y[b, j]
                    acc[f]["pos"] += att[y == 1, :, j].sum(0)
                    acc[f]["neg"] += att[y == 0, :, j].sum(0)
                    acc[f]["n_pos"] += int((y == 1).sum())
                    acc[f]["n_neg"] += int((y == 0).sum())
        del bank
        lo, hi = int(round(S * 11 / 32)), int(round(S * 22 / 32))
        idx = np.arange(S)
        enc_out = {"slices": S, "middle_third_slices": f"{lo}-{hi - 1}", "uniform_middle_share": _r((hi - lo) / S, 2)}
        for f in C3_FINDINGS:
            a = acc[f]
            pos = a["pos"] / max(a["n_pos"], 1)
            neg = a["neg"] / max(a["n_neg"], 1)
            enc_out[f] = {"positives": agg.count(a["n_pos"]), "negatives": agg.count(a["n_neg"]),
                          "profile_positives": [round(float(v), 3) for v in pos] if a["n_pos"] >= MIN_SIDE else None,
                          "profile_negatives": [round(float(v), 3) for v in neg] if a["n_neg"] >= MIN_SIDE else None,
                          "middle_share_positives": _r(pos[lo:hi].sum(), 3) if a["n_pos"] >= MIN_SIDE else None,
                          "centre_of_mass_positives": _r(float((idx * pos).sum()), 2) if a["n_pos"] >= MIN_SIDE else None}
        mm, lm = enc_out["Medial Meniscus"]["centre_of_mass_positives"], enc_out["Lateral Meniscus"]["centre_of_mass_positives"]
        enc_out["medial_minus_lateral_centre"] = _r(mm - lm, 2) if mm is not None and lm is not None else None
        out["encoders"][e] = enc_out
    shares = [out["encoders"].get(e, {}).get("ACL", {}).get("middle_share_positives") for e in MAIN]
    gaps = [out["encoders"].get(e, {}).get("medial_minus_lateral_centre") for e in MAIN]
    S = out["encoders"].get(core.DINO, {}).get("slices") or 32
    out["reading"] = {
        "acl": ("not available" if any(v is None for v in shares) else
                "the heads do not find the ACL slices" if all(v < 0.45 for v in shares) else
                "rule not met (at least one main encoder at 0.45 or above)"),
        "menisci": ("not available" if any(v is None for v in gaps) else
                    "sagittal side information survives the order-blind head" if all(v >= 5 * S / 32 for v in gaps)
                    else "rule not met (gap under 5 slices for at least one main encoder)")}
    return out


# ---------------------------------------------------------------- C4: fallback contrast

def _tiers(d) -> np.ndarray | None:
    st = pd.read_csv(Path(d.n0_dir) / "studies.csv")
    cols = [f"{p}_tier" for p in PLANES]
    if not all(col in st.columns for col in cols):
        return None
    st[UID] = st[UID].astype(str)
    t = st.set_index(UID).reindex(d.uids)[cols]
    return (t.to_numpy(dtype=float) == 2).all(axis=1)


def c4_fallback_contrast(c, n_boot: int, seed: int) -> dict:
    """S2 macro AUC (strata: S2 fold x G-lang group, so pairs never cross a site group) for evaluation studies
    whose three chosen series are all fluid fat-suppressed (tier 2) and for the rest, inside the G-lang groups
    that hold fallback studies. Findings: at least 5 positives and 5 negatives in each group. Descriptive:
    fallback series may cluster by site, and this does not measure the fallback's cost."""
    d, f2 = c.d, c.d.folds["s2_fold"]
    out = _flags("S2 macro AUC by series tier; descriptive, not the fallback's cost",
                 strata="S2 fold x G-lang group", n_boot=int(n_boot), interval="study bootstrap, percentile 2.5-97.5",
                 reading_rule=("the rest lower by more than 0.03 for both main encoders, within the groups that hold "
                               "fallback studies = goes on the possible-actions list; it chooses nothing (R8)"))
    all_fs = _tiers(d)
    if all_fs is None:
        return {**out, "status": "not available (studies.csv has no tier columns)"}
    fb_rows = c.ev[~all_fs[c.ev]]
    groups = sorted({str(g) for g in d.g_lang[fb_rows]})
    in_g = np.isin(d.g_lang.astype(str), groups)
    fs_rows = c.ev[all_fs[c.ev] & in_g[c.ev]]
    out.update({"fallback_studies": agg.count(len(fb_rows)), "fs_studies_in_those_groups": agg.count(len(fs_rows)),
                "n_groups": len(groups),
                "fallback_share_of_evaluation": agg.share(len(fb_rows) / max(len(c.ev), 1))})
    zf, zb = np.zeros(len(fs_rows), int), np.zeros(len(fb_rows), int)
    findings = [j for j in metrics.eligible(c.Y[fs_rows], zf) if j in metrics.eligible(c.Y[fb_rows], zb)]
    out["findings"] = [FINDINGS[j] for j in findings]
    if len(fb_rows) < MIN_SIDE or len(fs_rows) < MIN_SIDE or len(findings) < n12.MIN_FINDINGS:
        return {**out, "status": "suppressed (too few fallback studies or findings)"}
    strata = metrics.strata_codes(f2, d.g_lang)
    fs_all = c.ev[all_fs[c.ev]]  # every evaluation study with the fluid fat-suppressed series in all planes
    out["fs_studies_all_groups"] = agg.count(len(fs_all))
    out["encoders"] = {}
    dfs = list(n12._draws(len(fs_rows), int(n_boot), seed))
    dfb = list(n12._draws(len(fb_rows), int(n_boot), seed + 1))
    dfa = list(n12._draws(len(fs_all), int(n_boot), seed + 2))
    for e, P in c.units.items():
        if not (np.isfinite(P[fs_all]).all() and np.isfinite(P[fb_rows]).all()):
            continue

        def m(rows, w=None, st=strata):
            return metrics.macro(P[rows], c.Y[rows], st[rows], findings, w)

        a, b = m(fs_rows), m(fb_rows)
        diff = [m(fs_rows, wa) - m(fb_rows, wb) for wa, wb in zip(dfs, dfb)]
        a2, b2 = m(fs_all, None, f2), m(fb_rows, None, f2)
        diff2 = [m(fs_all, wa, f2) - m(fb_rows, wb, f2) for wa, wb in zip(dfa, dfb)]
        out["encoders"][e] = {"all_fluid_fatsat": _r(a), "fallback": _r(b),
                              "all_fluid_fatsat_minus_fallback": {"estimate": _r(a - b), **n12._percentile(diff)},
                              "fold_stratified_all_groups": {
                                  "all_fluid_fatsat": _r(a2), "fallback": _r(b2),
                                  "difference": {"estimate": _r(a2 - b2), **n12._percentile(diff2)}}}
    gaps = [out["encoders"].get(e, {}).get("all_fluid_fatsat_minus_fallback", {}).get("estimate") for e in MAIN]
    out["reading"] = ("not available" if any(v is None for v in gaps) else
                      "goes on the possible-actions list" if all(v > 0.03 for v in gaps) else
                      "rule not met: not on the possible-actions list")
    return out


# ---------------------------------------------------------------- C5: centre crop

def _header_geometry(series_dir: Path, plane: str):
    files = dicom_io.list_dcm(series_dir)
    if not files:
        return None
    h = dicom_io.slice_header(files[0])
    if h.rows is None or h.cols is None:
        return None
    transpose = dicom_io.frame_ops(h.iop, plane)[0]
    H, W = (h.cols, h.rows) if transpose else (h.rows, h.cols)
    sp = float(np.mean(h.spacing)) if h.spacing else float("nan")
    return H, W, sp


def c5_centre_crop(c, images_root: Path | None, workers: int = 8) -> dict:
    """First header of each chosen series (usable studies): the image size in the canonical frame, which axis
    is longer, the cut fraction of the centre crop (1 - short/long) and the effective mm per pixel at 224
    (mean pixel spacing x short side / 224). The anterior-posterior axis is the columns in sagittal and the rows
    in axial (dicom_io.TARGET); coronal has none."""
    d = c.d
    out = _flags("geometry of the chosen series (headers only); what the shorter-side resize and centre crop cut",
                 reading_rule=[f"in sagittal or axial: cut fraction at least {CUT_RULE} with the anterior-posterior "
                               f"axis the longer one, in at least {SHARE_RULE:.0%} of studies",
                               "= the crop can remove the patella or the popliteal fossa in a material share of studies"])
    root = images_root
    if root is None:
        try:
            root = paths.competition_root() / "train_series"
        except Exception:  # noqa: BLE001 - no competition data attached
            root = None
    if root is None or not Path(root).exists():
        return {**out, "status": "not available (competition data not attached)"}
    st = pd.read_csv(Path(d.n0_dir) / "studies.csv")
    st[UID] = st[UID].astype(str)
    st = st.set_index(UID).reindex(d.uids)
    rows = np.flatnonzero(d.usable)
    out["planes"] = {}
    material = {}
    for p in PLANES:
        col = f"{p}_series"
        if col not in st.columns:
            out["planes"][p] = {"status": "no series column"}
            continue
        jobs = [(Path(root) / d.uids[i] / str(st[col].iloc[i]), p) for i in rows if pd.notna(st[col].iloc[i])]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            geo = list(ex.map(lambda a: _safe_geometry(*a), jobs))
        ok = [g for g in geo if g is not None]
        if not ok:
            out["planes"][p] = {"status": "no header read", "unread": agg.count(len(geo))}
            continue
        H = np.array([g[0] for g in ok], float)
        W = np.array([g[1] for g in ok], float)
        sp = np.array([g[2] for g in ok], float)
        short, long_ = np.minimum(H, W), np.maximum(H, W)
        cut = 1 - short / long_
        ap_longer = (W > H) if p == "sagittal" else (H > W) if p == "axial" else np.zeros(len(H), bool)
        big = (cut >= CUT_RULE) & ap_longer
        out["planes"][p] = {
            "series_read": agg.count(len(ok)), "unread": agg.count(len(geo) - len(ok)),
            "square_share": agg.share((H == W).mean()),
            "anterior_posterior_longer_share": agg.share(ap_longer.mean()) if p != "coronal" else None,
            "cut_fraction": agg.quantiles(cut),
            "cut_at_least_015_share": agg.share((cut >= CUT_RULE).mean()),
            "cut_at_least_015_with_ap_longer_share": agg.share(big.mean()) if p != "coronal" else None,
            "mm_per_pixel_at_224": agg.quantiles(sp * short / heads_size())}
        if p != "coronal":
            material[p] = float(big.mean()) >= SHARE_RULE
    out["reading"] = ("not available" if not material else
                      "material share: " + ", ".join(p for p, v in material.items() if v) if any(material.values())
                      else "no material share (rule not met)")
    return out


def heads_size() -> int:
    return dicom_io.SIZE


def _safe_geometry(series_dir: Path, plane: str):
    try:
        return _header_geometry(series_dir, plane)
    except Exception:  # noqa: BLE001 - a header that cannot be read counts as unread
        return None


# ---------------------------------------------------------------- C6: S1 level

def c6_s1_level(c) -> dict:
    """The S1 mean of fold macros (N7's S1 predictions; S1 folds have S2's sizes) beside S2's mean of fold
    macros, on the same evaluation studies. Site part only (n9c.py:226-229); S1 folds are random, so the gap is
    an upper bound (one patient can sit on both sides)."""
    d = c.d
    f1, f2 = d.folds["s1_fold"], d.folds["s2_fold"]
    out = _flags("S1 (seen sites, S2's fold sizes) against S2 (unseen sites), mean of fold macros; site part only",
                 reading_rule="descriptive only")
    if c.n7_dir is None:
        return {**out, "status": "not available (N7's S1 predictions not attached)"}
    out["encoders"] = {}
    for e in ENCODERS:
        S1 = core._load_csv(c.n7_dir / f"s1_{e}.csv", d.n, c.row_of)
        if S1 is None or e not in c.units:
            continue
        P = c.units[e]
        rows = c.ev[(f1[c.ev] >= 0) & np.isfinite(S1[c.ev]).all(axis=1) & np.isfinite(P[c.ev]).all(axis=1)]
        s1 = metrics.per_fold_macro_mean(S1[rows], c.Y[rows], f1[rows], c.elig)
        s2 = metrics.per_fold_macro_mean(P[rows], c.Y[rows], f2[rows], c.elig)
        out["encoders"][e] = {"studies": agg.count(len(rows)),
                              "same_studies_as_evaluation_set": bool(len(rows) == len(c.ev)),
                              "s1_mean_of_fold_macros": _r(s1),
                              "s2_mean_of_fold_macros": _r(s2), "s1_minus_s2": _r(s1 - s2)}
    return out


# ---------------------------------------------------------------- run

def _summary(s: dict) -> dict:
    def g(*path):
        v = core._at(s, path)
        return v if isinstance(v, dict) else {}

    out = {"status": s.get("status"), "chain_ok": g("chain").get("s2_folds_match_units"),
           "main_complete": s.get("main_complete"), "errors": s.get("errors"),
           "descriptive_only": "N14 keeps nothing and switches nothing on (R8)"}
    try:
        c1 = g("c1_gold_all_data")
        out["c1"] = {"label": c1.get("label"), "reading": c1.get("reading"), "gold_per_s2_fold": c1.get("gold_per_s2_fold"),
                     "lookalikes_removed": c1.get("lookalikes_removed"),
                     "encoders": {e: [v.get("all_data"), v.get("s2_pooled"),
                                      (v.get("all_data_minus_s2_pooled") or {}).get("estimate"),
                                      v.get("all_data_fold_stratified"), v.get("s2_fold_stratified"),
                                      (v.get("all_data_minus_s2_fold_stratified") or {}).get("estimate")]
                                  for e, v in (c1.get("encoders") or {}).items()},
                     "columns": "pooled: all-data, S2, difference; fold-stratified: all-data, S2, difference"}
        c2 = g("c2_side_discrimination")
        out["c2"] = {k: {"reading": v.get("reading"), "status": v.get("status"),
                         "side_auc": {e: x.get("side_auc") for e, x in (v.get("encoders") or {}).items()}}
                     for k, v in (c2.get("pairs") or {}).items()}
        c3 = g("c3_sagittal_attention")
        out["c3"] = {"reading": c3.get("reading"),
                     "encoders": {e: {"acl_middle_share": (v.get("ACL") or {}).get("middle_share_positives"),
                                      "medial_minus_lateral_centre": v.get("medial_minus_lateral_centre")}
                                  for e, v in (c3.get("encoders") or {}).items()}}
        c4 = g("c4_fallback_contrast")
        out["c4"] = {"reading": c4.get("reading"), "status": c4.get("status"),
                     "fallback_share_of_evaluation": c4.get("fallback_share_of_evaluation"),
                     "encoders": {e: (v.get("all_fluid_fatsat_minus_fallback") or {}).get("estimate")
                                  for e, v in (c4.get("encoders") or {}).items()}}
        c5 = g("c5_centre_crop")
        out["c5"] = {"reading": c5.get("reading"), "status": c5.get("status"),
                     "planes": {p: {"square": v.get("square_share"), "ap_longer": v.get("anterior_posterior_longer_share"),
                                    "cut_015_ap_longer": v.get("cut_at_least_015_with_ap_longer_share"),
                                    "cut_median": (v.get("cut_fraction") or {}).get("median"),
                                    "mm_px_median": (v.get("mm_per_pixel_at_224") or {}).get("median")}
                                for p, v in (c5.get("planes") or {}).items()}}
        c6 = g("c6_s1_level")
        out["c6"] = {e: [v.get("s1_mean_of_fold_macros"), v.get("s2_mean_of_fold_macros"), v.get("s1_minus_s2")]
                     for e, v in (c6.get("encoders") or {}).items()}
    except Exception as e:  # noqa: BLE001 - the summary must print whatever the blocks left
        out["summary_error"] = safe_log.describe(type(e), e.__traceback__)
    return out


def run(root=None, expect: dict | None = None, n_boot: int = intervals.N_BOOT, seed: int = 0,
        budget_h: float = 2.0, quiet: bool = True, images_root=None, blocks=None) -> dict:
    """The six blocks of n14_spec_r3.md on the attached outputs. expect: upstream hash prefixes {"n0_studies",
    "n3_folds", "n5_index"} as in N12's card. blocks: names to run (default all)."""
    want = set(blocks or ("c1", "c2", "c3", "c4", "c5", "c6"))
    s: dict = {"written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **_flags(SCOPE)}
    with common.notebook("n14", {"n_boot": n_boot, "seed": seed, "blocks": sorted(want)}, quiet=quiet, budget_h=budget_h):
        base = Path(root) if root else paths.INPUT
        rootp = Path(root) if root else None
        d = stage.StageData(root, expect)
        row_of = {u: i for i, u in enumerate(d.uids)}
        f2, Y = d.folds["s2_fold"], d.eval_y
        dirs = unit_files.unit_dirs(base)
        units = core._load_units(base, d.n, row_of, dirs)
        unit_folds = units.pop("_folds")
        K = d.n_folds("s2")
        s["upstream"] = d.upstream
        s["chain"] = unit_files.chain_summary(d, dirs, rootp)
        if s["chain"]["s2_folds_match_units"] is False:
            s["status"] = "chain mismatch"
            agg.write("n14", s)
            raise unit_files.ChainError("Stage A units do not match the attached S2 folds")
        s["units_found"] = {e: {"folds": v} for e, v in unit_folds.items()}
        main_ok = all(e in units and len(unit_folds.get(e, [])) == K for e in MAIN)
        s["main_complete"] = main_ok
        if not main_ok:
            s["status"] = "main units missing"
            agg.write("n14", s)
        else:
            tr_all = np.flatnonzero(d.train & (f2 >= 0))
            ev_all = tr_all[d.scored[tr_all]]
            have = np.ones(d.n, bool)
            for e in MAIN:
                have &= np.isfinite(units[e]).all(axis=1)
            ev = ev_all[have[ev_all]]
            s["evaluation_studies"] = agg.count(len(ev))
            gold = d.gold_rows()
            gold = gold[np.isfinite(units[core.DINO][gold]).all(axis=1) & np.isfinite(units[core.CONV][gold]).all(axis=1)]
            g_find = metrics.eligible(d.gold_y[gold], np.zeros(len(gold), int)) if len(gold) else []
            c = SimpleNamespace(d=d, ev=ev, Y=Y, units=units, elig=list(range(len(FINDINGS))), root=rootp,
                                row_of=row_of, dirs=dirs, gold=gold, g_find=g_find, n7_dir=n12._n7_dir(rootp))
            if c.n7_dir is None:
                p = paths.find_file(f"s1_{core.DINO}.csv", rootp)
                c.n7_dir = p.parent if p is not None else None
            errors: dict = {}
            s["errors"] = errors

            def put(name: str, fn) -> None:
                try:
                    val = fn(c)
                    agg._check(json.loads(json.dumps(val, default=agg._json_default)), path=name)
                    s[name] = val
                except Exception as e:  # noqa: BLE001 - descriptive blocks must not stop each other
                    errors[name] = safe_log.describe(type(e), e.__traceback__)
                s["status"] = "running (blocks so far)"
                agg.write("n14", s)  # a late interrupt keeps the blocks already done

            if "c6" in want:
                put("c6_s1_level", c6_s1_level)
            if "c2" in want:
                put("c2_side_discrimination", lambda c: c2_side_discrimination(c, n_boot, seed))
            if "c4" in want:
                put("c4_fallback_contrast", lambda c: c4_fallback_contrast(c, n_boot, seed))
            if "c3" in want:
                put("c3_sagittal_attention", c3_sagittal_attention)
            if "c5" in want:
                put("c5_centre_crop", lambda c: c5_centre_crop(c, Path(images_root) if images_root else None))
            if "c1" in want:
                put("c1_gold_all_data", lambda c: c1_gold_all_data(c, n_boot, seed))
            s["status"] = "ok" if not errors else "ok, some blocks failed (see errors)"
            agg.write("n14", s)
    try:
        common.summary(_summary(s))
    except agg.UnsafeSummary:
        common.summary({"status": s.get("status"), "errors": s.get("errors"),
                        "summary_error": "the summary block was refused by the checks; read agg_n14.json"})
    return s
