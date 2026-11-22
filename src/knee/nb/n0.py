"""N0: data understanding, labels, site groups and floor-A tags (CPU).

Writes to /kaggle/working:
  private/n0/studies.csv        one row per study: chosen series, laterality, fingerprint,
                                language, groups, flags
  private/n0/labels_soft.csv, labels_weight.csv, labels_binary.csv
  private/n0/labels_gold.csv    expert labels of the gold studies (S3 in N9; never trained on)
  private/n0/tags.csv           acquisition tags for floor A
  agg_n0.json, run.json         summaries only (agg checks enforced)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import agg, dicom_io, groups, labels, paths, series, study_table
from ..labels import FINDINGS, UID
from . import common

TAGS = ("repetition_time", "echo_time", "slice_thickness", "pixel_spacing")
PLANES = ("sagittal", "coronal", "axial")


def ts_name(ts) -> str:
    """Transfer syntax by name (its UID looks like a data UID to the summary checks)."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or str(ts) == "":
        return "missing"
    try:
        from pydicom.uid import UID as _UID

        name = _UID(str(ts)).name
        return name if name and name != str(ts) else "other"
    except Exception:  # noqa: BLE001
        return "other"


def _decode_tests(chosen: pd.DataFrame, images, max_per_ts: int = 3) -> dict:
    out = {}
    for ts, sub in chosen.dropna(subset=["transfer_syntax"]).groupby("transfer_syntax"):
        ok, tried, decs = True, 0, set()
        for _, r in sub.head(max_per_ts).iterrows():
            files = dicom_io.list_dcm(images / str(r[UID]) / str(r["SeriesInstanceUID"]))
            if not files:
                continue
            tried += 1
            try:
                _, dec = dicom_io.decode(files[len(files) // 2])
                decs.add(dec)
            except Exception:  # noqa: BLE001
                ok = False
        key = ts_name(ts)
        prev = out.get(key, {"series": 0, "all_decoded": True, "decoders": []})
        out[key] = {"series": agg.count(len(sub)), "all_decoded": bool(ok and tried > 0 and prev["all_decoded"]),
                    "decoders": sorted(set(prev["decoders"]) | decs)}
    return out


def _series_types(ser: pd.DataFrame) -> dict:
    """Series rows by plane and type, over every row of train_series.csv (not only chosen series)."""
    plane = ser["Anatomical_Plane"].map(series.norm_plane).fillna("unrecognised")
    fs = ser["Fluid_Sensitive"].map(series.truthy).astype(bool)
    fat = ser["Fat_Suppression"].map(series.truthy).astype(bool)
    kind = pd.Series(np.select([fs & fat, fs & ~fat, ~fs & fat], ["fluid+fatsat", "fluid", "fatsat_only"],
                               "neither"), index=ser.index)
    tab = pd.crosstab(plane, kind)
    return {str(p): agg.counts(tab.loc[p]) for p in tab.index}


def _top_fingerprints_by_language(tr: pd.DataFrame, top: int = 5) -> dict:
    """Most common scanner fingerprints per report language (sure studies; only counts of 5 or more)."""
    out = {}
    sure = tr[tr["sure"].astype(bool)].dropna(subset=["fingerprint", "language"])
    for lg, sub in sure.groupby("language"):
        vc = sub["fingerprint"].value_counts()
        vc = vc[vc >= 5].head(top)
        if len(vc):
            out[str(lg)] = agg.counts(vc)
    return out


def _report_column(columns) -> str:
    for c in columns:
        if str(c).strip().lower() == "report":
            return c
    raise labels.SchemaError("train.csv: no Report column")


def _geometry_agreement(chosen: pd.DataFrame) -> dict:
    """Tag-vs-geometry laterality agreement at several margins (the plan names none; 20 mm is used)."""
    tag = (chosen.dropna(subset=["laterality_tag"]).groupby(UID)["laterality_tag"]
           .agg(lambda x: x.iloc[0] if x.nunique() == 1 else np.nan).dropna())
    cx = {}
    for uid, sub in chosen.groupby(UID):
        for p in ("coronal", "axial"):
            v = sub.loc[sub["plane"] == p, "center_x"]
            if len(v) and pd.notna(v.iloc[0]):
                cx[uid] = float(v.iloc[0])
                break
    out = {}
    for m in (0, 10, 20):
        pairs = [(tag[u], dicom_io.geometry_laterality(cx[u], m)) for u in tag.index if u in cx]
        decided = [(a, b) for a, b in pairs if b is not None]
        out[f"margin_{m}mm"] = {
            "pairs": agg.count(len(decided)),
            "inside_margin": agg.count(len(pairs) - len(decided)),
            "agreement": agg.share(np.mean([a == b for a, b in decided])) if decided else None,
        }
    return out


def run(limit: int | None = None, workers: int = 4, dread_file: str | None = None,
        dread_provenance_ok: bool = False, quiet: bool = True) -> dict:
    """dread_file: the label file's name in the dreaddevelopment dataset (Ablay reads it on the
    dataset page). dread_provenance_ok: True only if Ablay recorded that those labels are LLM output
    on report text with no gold, image-model or other-table input."""
    s: dict = {}
    cfg = {"limit": limit, "dread_file": bool(dread_file), "dread_provenance_ok": dread_provenance_ok}
    with common.notebook("n0", cfg, quiet=quiet):
        root = paths.competition_root()
        od = paths.out_dir("private", "n0")
        train = pd.read_csv(root / "train.csv", keep_default_na=True)
        lab_cols = labels.find_label_columns(train.columns)
        rep_col = _report_column(train.columns)
        train = train.rename(columns={v: k for k, v in lab_cols.items()} | {rep_col: "Report"})
        if limit:
            keep = sorted(train[UID].unique())[:limit]
            train = train[train[UID].isin(keep)]
        train = train.drop_duplicates(UID).set_index(UID).sort_index()
        has_report = train["Report"].fillna("").astype(str).str.strip().str.len() > 0
        gold = train[FINDINGS].dropna(how="any")
        vals = set(np.unique(gold.to_numpy()).tolist()) if len(gold) else set()
        if not vals <= {0, 1, 0.0, 1.0}:
            raise labels.SchemaError("gold labels are not 0/1")
        gold = gold.astype(int)
        is_gold = train.index.isin(gold.index)

        # 1. Images: series choice, header facts, laterality, fingerprint, tags
        st, chosen, hdr_errors = study_table.build(root, "train", workers=workers, limit=limit)
        st = st.set_index(UID).reindex(train.index)
        ser = pd.read_csv(root / "train_series.csv")
        if limit:
            ser = ser[ser[UID].isin(train.index)]
        planes = {}
        for p in PLANES:
            sub = chosen[chosen["plane"] == p]
            planes[p] = {
                "studies_with_plane": agg.count(sub[UID].nunique()),
                "chosen_tier": agg.counts(sub["tier"].map({2: "fluid+fatsat", 1: "fluid", 0: "any"}).value_counts()),
                "slices_chosen": agg.quantiles(sub["n_slices"]),
                "iop_plane_matches_label": agg.share((sub["plane_from_iop"] == p).mean()) if len(sub) else None,
            }
        s["images"] = {
            "studies": agg.count(len(train)),
            "series_rows": agg.count(len(ser)),
            "fluid_equals_fatsat_rows": agg.share((ser["Fluid_Sensitive"].astype(str) == ser["Fat_Suppression"].astype(str)).mean()),
            "series_types_all_rows": _series_types(ser),
            "planes": planes,
            "transfer_syntax_chosen": agg.counts(chosen["transfer_syntax"].map(ts_name).value_counts()),
            "decode_test": _decode_tests(chosen, root / "train_series"),
            "header_read_errors": agg.counts(hdr_errors),
            "tag_coverage_chosen": {
                "ImagePositionPatient": agg.share(chosen["has_ipp"].mean()),
                "ImageOrientationPatient": agg.share(chosen["has_iop"].mean()),
                "Laterality": agg.share(chosen["laterality_tag"].notna().mean()),
                "Manufacturer": agg.share(chosen["manufacturer"].notna().mean()),
                "ManufacturerModelName": agg.share(chosen["model"].notna().mean()),
                "MagneticFieldStrength": agg.share(chosen["field_strength"].notna().mean()),
                **{t: agg.share(chosen[t].notna().mean()) for t in TAGS},
            },
        }
        s["laterality"] = {
            "source": agg.counts(st["laterality_source"].value_counts()),
            "left_knees": agg.count((st["laterality"] == "L").sum()),
            "right_knees": agg.count((st["laterality"] == "R").sum()),
            "tag_vs_geometry": _geometry_agreement(chosen),
            "margin_mm": study_table.GEOMETRY_MARGIN_MM,
        }

        # 2. Reports: language and length
        lang = groups.detect_languages(train["Report"])
        st = st.join(lang)
        st["has_report"] = has_report
        st["is_gold"] = is_gold
        s["reports"] = {
            "with_report": agg.count(has_report.sum()),
            "length_chars": agg.quantiles(lang.loc[has_report, "report_len"]),
            "languages": agg.counts(lang.loc[has_report, "language"].fillna("none").value_counts()),
            "confidence": agg.quantiles(lang.loc[has_report, "lang_conf"]),
            "below_0.80": agg.count(((lang["lang_conf"] < groups.MIN_CONF) & has_report).sum()),
            "under_20_chars": agg.count(((lang["report_len"] < groups.MIN_CHARS) & has_report).sum()),
        }

        # 3. Labels (before groups, so all-masked studies can be left unscored)
        non_gold = train.index[~is_gold & has_report]
        tables, tinfo = {}, {}
        for key in paths.LABEL_TABLES:
            p = paths.find_label_table(key)
            if p is None:
                tinfo[key] = {"attached": False}
                continue
            raw = labels.read_table(p)
            try:
                t = labels.convert(key, raw)
            except labels.SchemaError as e:
                tinfo[key] = {"attached": True, "layout": "unrecognised", "reason": str(e),
                              "checks": {k: agg.share(v) for k, v in (getattr(e, "checks", None) or {}).items()},
                              **agg.safe_columns(raw.columns)}
                continue
            cov = labels.coverage(t, non_gold)
            tinfo[key] = {"attached": True, "layout": t.layout, "cells": agg.counts(t.cells or {}),
                          "checks": {k: agg.share(v) for k, v in (t.checks or {}).items()},
                          "coverage_non_gold": agg.share(cov), "kept": cov >= labels.COVERAGE_MIN}
            if cov >= labels.COVERAGE_MIN:
                tables[key] = t
        core = [tables[k] for k in ("pilkwang", "steven_v2", "lixin") if k in tables]
        used = list(core)
        if dread_file:
            dp = paths.find_file(dread_file)
            entry = {"attached": dp is not None, "provenance_ok": bool(dread_provenance_ok), "used": False}
            if dp is not None and len(core) >= 2:
                try:
                    dt = labels.convert("dread", labels.read_table(dp))
                    cov = labels.coverage(dt, non_gold)
                    gate = labels.agreement_gate(core, dt, non_gold)
                    entry.update({"layout": dt.layout, "coverage_non_gold": agg.share(cov), "gate": gate})
                    if dread_provenance_ok and gate["passed"] and cov >= labels.COVERAGE_MIN:
                        used.append(dt)
                        entry["used"] = True
                except labels.SchemaError:
                    entry["layout"] = "unrecognised"
            tinfo["dread"] = entry
        else:
            tinfo["dread"] = {"attached": False, "reason": "no file named in cell 2"}
        s["labels"] = {"tables": tinfo, "used": [t.name for t in used], "flag_fewer_than_3": len(used) < 3}
        if not used:
            raise RuntimeError("no label table attached")
        uids = train.index
        mean = labels.mean_label(used, uids)
        w = labels.weights(mean, tables.get("pilkwang"))
        b = labels.binary(mean)
        all_masked = mean.isna().all(axis=1)

        # 4. Site groups
        gt, gsum = groups.build_groups(st)
        gt.loc[gt["is_gold"], "in_training"] = False  # gold is never trained on
        masked = all_masked.reindex(gt.index).fillna(True).astype(bool)
        gt.loc[masked.to_numpy(), "in_training"] = False
        gt["scored"] = gt["scored"].astype(bool) & ~masked
        s["groups"] = gsum
        tr = gt[gt["in_training"]]
        s["groups"]["g_lang_sizes"] = agg.counts(tr["g_lang"].value_counts())
        top_fp = tr["fingerprint"].fillna("missing").value_counts()
        s["groups"]["top_fingerprints"] = agg.counts(top_fp[top_fp >= 5].head(15))
        s["groups"]["top_fingerprints_by_language"] = _top_fingerprints_by_language(tr)
        s["groups"]["fingerprint_coverage_training"] = agg.share(tr["fingerprint"].notna().mean()) if len(tr) else None

        # 5. Label summaries and the gold bug check (never chooses anything)
        from scipy.stats import spearmanr

        agree = {}
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                a = used[i].prob.reindex(non_gold)[FINDINGS].to_numpy(float).ravel()
                c = used[j].prob.reindex(non_gold)[FINDINGS].to_numpy(float).ravel()
                ok = ~(np.isnan(a) | np.isnan(c))
                agree[f"{used[i].name}~{used[j].name}"] = round(float(spearmanr(a[ok], c[ok]).statistic), 3) if ok.sum() > 10 else None
        s["labels"]["spearman_between_tables"] = agree
        elig = labels.eligible_findings(gold)
        g_auc, _ = labels.macro_auc(gold.astype(float), mean.loc[gold.index], elig)
        s["labels"]["gold"] = {
            "studies": agg.count(len(gold)),
            "is_58": len(gold) == 58,
            "with_report": agg.count(has_report[gold.index].sum()),
            "positives": agg.counts({f: int((gold[f] == 1).sum()) for f in FINDINGS}),
            "negatives": agg.counts({f: int((gold[f] == 0).sum()) for f in FINDINGS}),
            "eligible_findings": elig,
            "macro_auc_eligible": round(g_auc, 3) if np.isfinite(g_auc) else None,
            "bug_check_passed": bool(len(elig) > 0 and np.isfinite(g_auc) and g_auc >= 0.80),
        }
        tr_ids = gt.index[gt["in_training"]]
        s["labels"]["prevalence_training"] = {f: agg.share(b.loc[tr_ids, f].fillna(0).mean()) for f in FINDINGS}
        s["labels"]["uncertain_cells_training"] = agg.count(int(b.loc[tr_ids].isna().sum().sum()))
        s["labels"]["unk_weighted_cells_training"] = agg.count(int((w.loc[tr_ids] == labels.UNK_WEIGHT).sum().sum()))
        s["labels"]["all_masked_studies"] = agg.count(int(all_masked.sum()))
        s["labels"]["pos_weight_cap_upper"] = labels.pos_weight_cap_upper(b.loc[tr_ids])

        # 6. Floor-A tags (private) and private outputs
        tag_cols = [UID, "manufacturer", "fingerprint"] + [f"{p}_{t}" for p in PLANES for t in TAGS]
        tags = st.reset_index()[[c for c in tag_cols if c in st.reset_index().columns]]
        fs = chosen[chosen["plane"] == "sagittal"].set_index(UID)
        tags["model"] = tags[UID].map(fs["model"])
        tags["field_strength"] = tags[UID].map(fs["field_strength"])
        tags.to_csv(od / "tags.csv", index=False)
        gt.reset_index().to_csv(od / "studies.csv", index=False)
        mean.reset_index().to_csv(od / "labels_soft.csv", index=False)
        w.reset_index().to_csv(od / "labels_weight.csv", index=False)
        b.reset_index().to_csv(od / "labels_binary.csv", index=False)
        gold.reset_index().to_csv(od / "labels_gold.csv", index=False)  # expert labels, for S3 in N9
        s["training_studies"] = agg.count(int(gt["in_training"].sum()))
        s["scored_training_studies"] = agg.count(int((gt["scored"] & gt["in_training"]).sum()))
        # Hashes of the private files, so later steps can assert they read this version
        # (N3's expect_n0 takes the first characters of studies_sha16).
        s["files_sha16"] = {f"{n}_sha16": common.sha16(od / f"{n}.csv")
                            for n in ("studies", "labels_soft", "labels_weight", "labels_binary", "labels_gold", "tags")}
        agg.write("n0", s)
    common.summary({
        "gold_bug_check": s["labels"]["gold"]["bug_check_passed"],
        "gold_macro_auc": s["labels"]["gold"]["macro_auc_eligible"],
        "tables_used": s["labels"]["used"],
        "g_lang_groups": s["groups"]["g_lang_groups"],
        "training_studies": s["training_studies"],
        "studies_sha16": s["files_sha16"]["studies_sha16"],
    })
    return s
