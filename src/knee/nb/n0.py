"""N0: data understanding, labels and site groups (CPU).

Writes to /kaggle/working:
  n0/studies.csv   one row per study: chosen series, laterality, fingerprint,
                   language, groups, flags (private; never printed)
  n0/labels_soft.csv, n0/labels_weight.csv, n0/labels_binary.csv (private)
  agg_n0.json      counts, coverage and checks only
  run.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import agg, dicom_io, groups, labels, paths, safe_log, study_table
from ..labels import FINDINGS, UID


def _q(s) -> dict:
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {k: round(float(v), 3) for k, v in zip(["p5", "p25", "median", "p75", "p95"], q)}


def _decode_tests(chosen: pd.DataFrame, images, max_per_ts: int = 3) -> dict:
    out = {}
    for ts, sub in chosen.dropna(subset=["transfer_syntax"]).groupby("transfer_syntax"):
        ok, decs = 0, set()
        tried = 0
        for _, r in sub.head(max_per_ts).iterrows():
            files = dicom_io.list_dcm(images / str(r[UID]) / str(r["SeriesInstanceUID"]))
            if not files:
                continue
            tried += 1
            try:
                _, dec = dicom_io.decode(files[len(files) // 2])
                ok += 1
                decs.add(dec)
            except Exception:  # noqa: BLE001
                pass
        out[str(ts)] = {"series": agg.count(len(sub)), "tried": tried, "decoded": ok, "decoders": sorted(decs)}
    return out


def run(limit: int | None = None, workers: int = 4, dread_provenance_ok: bool = False,
        quiet: bool = True) -> dict:
    """dread_provenance_ok: set True only if Ablay recorded that dreaddevelopment's
    labels are LLM output on report text with no gold, image-model or other-table input."""
    summary: dict = {}
    with safe_log.step("n0", quiet=quiet):
        root = paths.competition_root()
        od = paths.out_dir("n0")
        train = pd.read_csv(root / "train.csv")
        if limit:
            keep = sorted(train[UID].unique())[:limit]
            train = train[train[UID].isin(keep)]
        train = train.drop_duplicates(UID).set_index(UID).sort_index()
        has_report = train["Report"].fillna("").astype(str).str.strip().str.len() > 0
        gold = train[FINDINGS].dropna(how="any")
        is_gold = train.index.isin(gold.index)

        # 1. Images: series choice, header facts, laterality, fingerprint
        st, chosen, hdr_errors = study_table.build(root, "train", workers=workers, limit=limit)
        st = st.set_index(UID).reindex(train.index)
        ser = pd.read_csv(root / "train_series.csv")
        if limit:
            ser = ser[ser[UID].isin(train.index)]
        fs_eq = (ser["Fluid_Sensitive"].astype(str) == ser["Fat_Suppression"].astype(str)).mean()
        planes = {}
        for p in ("sagittal", "coronal", "axial"):
            sub = chosen[chosen["plane"] == p]
            planes[p] = {
                "studies_with_plane": agg.count(sub[UID].nunique()),
                "series_in_csv": agg.count((ser["Anatomical_Plane"].astype(str).str.lower().str[:3] == p[:3]).sum()),
                "chosen_tier": agg.counts(sub["tier"].map({2: "fluid+fatsat", 1: "fluid", 0: "any"}).value_counts()),
                "slices_chosen": _q(sub["n_slices"]),
                "iop_plane_matches_label": round(float((sub["plane_from_iop"] == p).mean()), 4) if len(sub) else None,
            }
        summary["images"] = {
            "studies": agg.count(len(train)),
            "fluid_equals_fatsat_rows": round(float(fs_eq), 4),
            "planes": planes,
            "transfer_syntax_chosen": agg.counts(chosen["transfer_syntax"].fillna("missing").value_counts()),
            "decode_test": _decode_tests(chosen, root / "train_series"),
            "header_read_errors": agg.counts(hdr_errors),
            "tag_coverage_chosen": {
                "ImagePositionPatient": round(float(chosen["has_ipp"].mean()), 4),
                "ImageOrientationPatient": round(float(chosen["has_iop"].mean()), 4),
                "Laterality": round(float(chosen["laterality_tag"].notna().mean()), 4),
                "Manufacturer": round(float(chosen["manufacturer"].notna().mean()), 4),
                "ManufacturerModelName": round(float(chosen["model"].notna().mean()), 4),
                "MagneticFieldStrength": round(float(chosen["field_strength"].notna().mean()), 4),
            },
        }
        # Laterality sources and tag-vs-geometry agreement
        geo = {}
        for uid, sub in chosen.groupby(UID):
            g = None
            for p in ("coronal", "axial"):
                cx = sub.loc[sub["plane"] == p, "center_x"]
                if len(cx) and pd.notna(cx.iloc[0]):
                    g = dicom_io.geometry_laterality(float(cx.iloc[0]), study_table.GEOMETRY_MARGIN_MM)
                    if g:
                        break
            geo[uid] = g
        tag = chosen.dropna(subset=["laterality_tag"]).groupby(UID)["laterality_tag"].agg(
            lambda x: x.iloc[0] if x.nunique() == 1 else None)
        both = [(tag[u], geo.get(u)) for u in tag.index if tag[u] and geo.get(u)]
        summary["laterality"] = {
            "source": agg.counts(st["laterality_source"].value_counts()),
            "tag_geometry_pairs": agg.count(len(both)),
            "tag_geometry_agreement": round(float(np.mean([a == b for a, b in both])), 4) if both else None,
            "left_knees": agg.count((st["laterality"] == "L").sum()),
            "right_knees": agg.count((st["laterality"] == "R").sum()),
        }

        # 2. Reports: language and length
        lang = groups.detect_languages(train["Report"])
        st = st.join(lang)
        st["has_report"] = has_report
        st["is_gold"] = is_gold
        summary["reports"] = {
            "with_report": agg.count(has_report.sum()),
            "length_chars": _q(lang.loc[has_report, "report_len"]),
            "languages": agg.counts(lang.loc[has_report, "language"].fillna("none").value_counts()),
            "confidence": _q(lang.loc[has_report, "lang_conf"]),
            "below_0.80": agg.count(((lang["lang_conf"] < groups.MIN_CONF) & has_report).sum()),
            "under_20_chars": agg.count(((lang["report_len"] < groups.MIN_CHARS) & has_report).sum()),
        }

        # 3. Site groups
        gt, gsum = groups.build_groups(st)
        gt.loc[gt["is_gold"], "in_training"] = False  # gold is never trained on
        summary["groups"] = gsum
        tr = gt[gt["in_training"]]
        summary["groups"]["g_lang_sizes"] = agg.counts(tr["g_lang"].value_counts())
        fp_per_group = tr.dropna(subset=["fingerprint"]).groupby("g_lang")["fingerprint"].nunique()
        summary["groups"]["fingerprints_per_group"] = {k: int(v) for k, v in fp_per_group.items()}
        top_fp = tr["fingerprint"].fillna("missing").value_counts()
        summary["groups"]["top_fingerprints"] = agg.counts(top_fp[top_fp >= 5].head(15))
        summary["groups"]["fingerprint_coverage_training"] = round(float(tr["fingerprint"].notna().mean()), 4) if len(tr) else None

        # 4. Labels
        non_gold = train.index[~is_gold & has_report]
        tables, tinfo = {}, {}
        for key in paths.LABEL_TABLES:
            p = paths.find_label_table(key)
            if p is None:
                tinfo[key] = {"attached": False}
                continue
            raw = pd.read_csv(p)
            try:
                t = labels.convert(key, raw)
            except labels.SchemaError:
                tinfo[key] = {"attached": True, "layout": "unrecognised", "columns": [str(c) for c in raw.columns][:60]}
                continue
            cov = labels.coverage(t, non_gold)
            tinfo[key] = {"attached": True, "layout": t.layout, "coverage_non_gold": round(cov, 4),
                          "kept": cov >= labels.COVERAGE_MIN}
            if cov >= labels.COVERAGE_MIN:
                tables[key] = t
        core = [tables[k] for k in ("pilkwang", "steven_v2", "lixin") if k in tables]
        used = list(core)
        dread_path = paths.find_file("dread_labels.csv")  # placeholder name; see run card
        if dread_path is not None and dread_provenance_ok and len(core) >= 2:
            try:
                dt = labels.convert("dread", pd.read_csv(dread_path))
                gate = labels.agreement_gate(core, dt, non_gold)
                tinfo["dread"] = {"attached": True, "layout": dt.layout, "gate": gate}
                if gate["passed"] and labels.coverage(dt, non_gold) >= labels.COVERAGE_MIN:
                    used.append(dt)
            except labels.SchemaError:
                tinfo["dread"] = {"attached": True, "layout": "unrecognised"}
        else:
            tinfo["dread"] = {"attached": dread_path is not None, "provenance_ok": bool(dread_provenance_ok), "used": False}
        summary["labels"] = {"tables": tinfo, "used": [t.name for t in used], "flag_fewer_than_3": len(used) < 3}
        if not used:
            raise RuntimeError("no label table attached")
        uids = train.index
        mean = labels.mean_label(used, uids)
        pil = tables.get("pilkwang")
        w = labels.weights(mean, pil)
        b = labels.binary(mean)
        # Agreement between the used tables (Spearman over all cells, non-gold)
        from scipy.stats import spearmanr

        agree = {}
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                a = used[i].prob.reindex(non_gold)[FINDINGS].to_numpy(float).ravel()
                c = used[j].prob.reindex(non_gold)[FINDINGS].to_numpy(float).ravel()
                ok = ~(np.isnan(a) | np.isnan(c))
                agree[f"{used[i].name}~{used[j].name}"] = round(float(spearmanr(a[ok], c[ok]).statistic), 4) if ok.sum() > 10 else None
        summary["labels"]["spearman_between_tables"] = agree
        # Gold bug check (never chooses anything)
        elig = labels.eligible_findings(gold)
        g_auc, g_per = labels.macro_auc(gold.astype(float), mean.loc[gold.index], elig)
        summary["labels"]["gold"] = {
            "studies": agg.count(len(gold)),
            "with_report": agg.count(has_report[gold.index].sum()),
            "positives": agg.counts({f: int((gold[f] == 1).sum()) for f in FINDINGS}),
            "negatives": agg.counts({f: int((gold[f] == 0).sum()) for f in FINDINGS}),
            "eligible_findings": elig,
            "macro_auc_eligible": round(g_auc, 4) if np.isfinite(g_auc) else None,
            "bug_check_passed": bool(np.isfinite(g_auc) and g_auc >= 0.80),
        }
        tr_ids = gt.index[gt["in_training"]]
        summary["labels"]["prevalence_training"] = {
            f: round(float(b.loc[tr_ids, f].fillna(0).mean()), 4) for f in FINDINGS
        }
        summary["labels"]["uncertain_cells_training"] = agg.count(int(b.loc[tr_ids].isna().sum().sum()))
        summary["labels"]["unk_weighted_cells_training"] = agg.count(int((w.loc[tr_ids] == labels.UNK_WEIGHT).sum().sum()))
        summary["labels"]["masked_studies"] = agg.count(int(mean.isna().all(axis=1).sum()))

        # 5. Save private outputs
        gt.reset_index().to_csv(od / "studies.csv", index=False)
        mean.reset_index().to_csv(od / "labels_soft.csv", index=False)
        w.reset_index().to_csv(od / "labels_weight.csv", index=False)
        b.reset_index().to_csv(od / "labels_binary.csv", index=False)
        summary["outputs"] = sorted(p.name for p in od.iterdir())
        summary["training_studies"] = agg.count(int(gt["in_training"].sum()))
        summary["scored_studies"] = agg.count(int((gt["scored"] & gt["in_training"]).sum()))
        agg.write("run", agg.run_info("n0", {"limit": limit, "dread_provenance_ok": dread_provenance_ok}))
        agg.write("n0", summary)
    print({"gold_bug_check": summary["labels"]["gold"], "groups": summary["groups"]["g_lang_groups"],
           "tables_used": summary["labels"]["used"]})
    return summary
