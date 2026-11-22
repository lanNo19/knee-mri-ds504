"""Labels from the public LLM label tables (plan, "Labels from reports").

Rules fixed before any data were seen:
1. Tables: pilkwang report_labels_v2, stevenleehans llm_labels_v2, lixin73
   labels_llm_gpt56sol. dreaddevelopment only if it passes its provenance
   check (recorded by hand) and the agreement gate below.
2. Each table becomes P(present) per cell. pilkwang: YES = confidence,
   NO = 1 - confidence, UNK = 0 (kept). Verdict-only tables: YES = 1, NO = 0.
   Tables that already hold numbers in [0, 1] are used as they are.
3. Label = mean over the tables present for that cell; all missing = masked.
4. Training weight 0.5 where pilkwang says UNK; masked cells weight 0.
5. Binary label: present if mean > 0.5, absent if < 0.5, uncertain if exactly
   0.5 or masked. Scoring counts uncertain as negative.
6. Coverage: a table covering under 98% of the non-gold training studies is
   dropped.

Schema detection reads column names only and never prints cell values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
UID = "StudyInstanceUID"
YES = {"yes", "y", "present", "positive", "pos", "true", "1", "abnormal"}
NO = {"no", "n", "absent", "negative", "neg", "false", "0", "normal"}
UNK = {"unk", "unknown", "not_mentioned", "not mentioned", "na", "n/a", "none", "silent", ""}
COVERAGE_MIN = 0.98
UNK_WEIGHT = 0.5


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


FINDING_KEYS = {_key(f): f for f in FINDINGS}


class SchemaError(ValueError):
    """The table's columns do not match a known layout (message: column names only)."""


@dataclass
class Table:
    name: str
    prob: pd.DataFrame  # index UID, columns FINDINGS, float in [0, 1] or NaN
    unk: pd.DataFrame | None  # bool, True where the table says the report is silent
    layout: str
    cells: dict | None = None  # counts of yes / no / unk / unmapped / numeric / empty cells


def _match_columns(cols) -> dict[str, dict[str, str]]:
    """For each finding: which columns hold its value, verdict and confidence."""
    found: dict[str, dict[str, str]] = {f: {} for f in FINDINGS}
    for c in cols:
        k = _key(c)
        for fk, f in sorted(FINDING_KEYS.items(), key=lambda kv: -len(kv[0])):
            if k == fk:
                found[f]["value"] = c
                break
            if k.startswith(fk) or k.endswith(fk):
                rest = k[len(fk):] if k.startswith(fk) else k[: -len(fk)]
                if rest in {"conf", "confidence", "prob", "p", "score"}:
                    found[f]["conf"] = c
                elif rest in {"verdict", "label", "status", "state", "answer", "cat", "category"}:
                    found[f]["verdict"] = c
                elif rest in {"y"} and "value" not in found[f]:
                    found[f]["value"] = c
                break
    return found


def _as_verdict(series: pd.Series) -> tuple[pd.Series, dict]:
    """Map strings to 1 (yes), 0 (no), -1 (unknown or silent); other strings to NaN.

    Returns the mapped series and cell counts (yes, no, unk, unmapped, empty).
    """
    n = {"yes": 0, "no": 0, "unk": 0, "unmapped": 0, "empty": 0}

    def m(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            n["empty"] += 1
            return np.nan
        s = str(v).strip().lower()
        if s in YES:
            n["yes"] += 1
            return 1.0
        if s in NO:
            n["no"] += 1
            return 0.0
        if s in UNK:
            n["unk"] += 1
            return -1.0
        n["unmapped"] += 1
        return np.nan
    out = series.map(m)
    return out, n


def read_table(path) -> pd.DataFrame:
    """Read a label table keeping strings such as "NA" or "None" (they can mean UNK)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def convert(name: str, df: pd.DataFrame) -> Table:
    """Turn one raw table into P(present) per finding.

    Refuses (SchemaError) rather than guesses: numeric values outside [0, 1],
    a numeric value column that also has a confidence column, or more than
    0.5% of verdict cells that map to nothing.
    """
    if UID not in df.columns:
        raise SchemaError(f"{name}: no {UID} column")
    df = df.drop_duplicates(UID).set_index(UID)
    cols = _match_columns(df.columns)
    missing = [f for f in FINDINGS if not cols[f]]
    if missing:
        raise SchemaError(f"{name}: findings not found: {missing}")
    prob = pd.DataFrame(index=df.index, columns=FINDINGS, dtype=float)
    unk = pd.DataFrame(False, index=df.index, columns=FINDINGS)
    layouts = set()
    cells = {"yes": 0, "no": 0, "unk": 0, "unmapped": 0, "numeric": 0, "empty": 0}
    for f in FINDINGS:
        c = cols[f]
        val = df[c["value"]] if "value" in c else None
        if val is not None and pd.api.types.is_numeric_dtype(val):
            if "conf" in c:
                raise SchemaError(f"{name}: numeric {f} with a confidence column")
            v = val.astype(float)
            if ((v < 0) | (v > 1)).any():
                raise SchemaError(f"{name}: numeric {f} outside [0, 1]")
            if "verdict" in c:
                ver, n = _as_verdict(df[c["verdict"]])
                unk[f] = ver == -1
                for k in cells:
                    cells[k] += n.get(k, 0)
            prob[f] = v
            cells["numeric"] += int(v.notna().sum())
            cells["empty"] += int(v.isna().sum())
            layouts.add("numeric")
            continue
        verdict_col = c.get("verdict") or c.get("value")
        if verdict_col is None:
            raise SchemaError(f"{name}: no value or verdict column for {f}")
        ver, n = _as_verdict(df[verdict_col])
        for k in cells:
            cells[k] += n.get(k, 0)
        if "conf" in c:
            conf = pd.to_numeric(df[c["conf"]], errors="coerce")
            if ((conf < 0) | (conf > 1)).any():
                raise SchemaError(f"{name}: confidence for {f} outside [0, 1]")
            p = np.where(ver == 1, conf, np.where(ver == 0, 1 - conf, np.where(ver == -1, 0.0, np.nan)))
            layouts.add("verdict+confidence")
        else:
            p = np.where(ver == 1, 1.0, np.where(ver == 0, 0.0, np.where(ver == -1, 0.0, np.nan)))
            layouts.add("verdict")
        prob[f] = p
        unk[f] = ver == -1
    total = max(1, sum(cells.values()) - cells["empty"])
    if cells["unmapped"] > 0.005 * total:
        raise SchemaError(f"{name}: too many unmapped verdict cells")
    layout = "+".join(sorted(layouts))
    return Table(name=name, prob=prob, unk=unk if unk.values.any() else None, layout=layout, cells=cells)


def coverage(table: Table, uids) -> float:
    uids = pd.Index(uids)
    if len(uids) == 0:
        return 0.0
    present = table.prob.reindex(uids).notna().any(axis=1)
    return float(present.mean())


def mean_label(tables: list[Table], uids) -> pd.DataFrame:
    uids = pd.Index(uids)
    stack = np.stack([t.prob.reindex(uids)[FINDINGS].to_numpy(dtype=float) for t in tables])
    with np.errstate(invalid="ignore"):
        m = np.nanmean(stack, axis=0) if len(tables) else np.full((len(uids), len(FINDINGS)), np.nan)
    return pd.DataFrame(m, index=uids, columns=FINDINGS)


def weights(mean: pd.DataFrame, pilkwang: Table | None) -> pd.DataFrame:
    w = pd.DataFrame(1.0, index=mean.index, columns=FINDINGS)
    w[mean.isna()] = 0.0
    if pilkwang is not None and pilkwang.unk is not None:
        u = pilkwang.unk.reindex(mean.index).fillna(False).astype(bool)
        w[u & mean.notna()] = UNK_WEIGHT
    return w


def binary(mean: pd.DataFrame) -> pd.DataFrame:
    """1 present, 0 absent, NaN uncertain (exactly 0.5 or masked)."""
    b = pd.DataFrame(np.nan, index=mean.index, columns=FINDINGS)
    b[mean > 0.5] = 1.0
    b[mean < 0.5] = 0.0
    return b


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC with average ranks for ties; NaN when one class is missing."""
    from scipy.stats import rankdata

    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_auc(y: pd.DataFrame, s: pd.DataFrame, findings=None) -> tuple[float, dict]:
    findings = list(FINDINGS if findings is None else findings)
    per = {f: auc(y[f].to_numpy(dtype=float), s[f].to_numpy(dtype=float)) for f in findings}
    vals = [v for v in per.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def eligible_findings(gold: pd.DataFrame, min_each: int = 5) -> list[str]:
    out = []
    for f in FINDINGS:
        v = gold[f].dropna()
        if (v == 1).sum() >= min_each and (v == 0).sum() >= min_each:
            out.append(f)
    return out


def agreement_gate(core: list[Table], extra: Table, uids) -> dict:
    """Agreement gate for an extra table (dreaddevelopment) on non-gold studies.

    Each core table's macro AUC against the binarised mean of the other core
    tables; the extra table's against the binarised mean of all core tables.
    Pass if extra >= min(core) - 0.02, no finding below 0.5, and Spearman with
    any single core table <= 0.9.
    """
    from scipy.stats import spearmanr

    uids = pd.Index(uids)
    core_vals = {}
    for i, t in enumerate(core):
        others = [o for j, o in enumerate(core) if j != i]
        ref = binary(mean_label(others, uids))
        core_vals[t.name], _ = macro_auc(ref, t.prob.reindex(uids))
    ref_all = binary(mean_label(core, uids))
    ex_val, ex_per = macro_auc(ref_all, extra.prob.reindex(uids))
    rhos = {}
    for t in core:
        a = extra.prob.reindex(uids)[FINDINGS].to_numpy(dtype=float).ravel()
        b = t.prob.reindex(uids)[FINDINGS].to_numpy(dtype=float).ravel()
        ok = ~(np.isnan(a) | np.isnan(b))
        rhos[t.name] = float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 10 else float("nan")
    lowest = min(core_vals.values()) if core_vals else float("nan")
    passed = bool(
        np.isfinite(ex_val)
        and np.isfinite(lowest)
        and ex_val >= lowest - 0.02
        and all(np.isfinite(v) and v >= 0.5 for v in ex_per.values())
        and len(rhos) > 0
        and all(np.isfinite(r) and r <= 0.9 for r in rhos.values())
    )
    return {
        "core_macro_auc": {k: round(v, 4) for k, v in core_vals.items()},
        "extra_macro_auc": round(ex_val, 4) if np.isfinite(ex_val) else None,
        "extra_min_finding_auc": round(min(v for v in ex_per.values() if np.isfinite(v)), 4) if any(np.isfinite(v) for v in ex_per.values()) else None,
        "max_spearman": round(max(r for r in rhos.values() if np.isfinite(r)), 4) if any(np.isfinite(r) for r in rhos.values()) else None,
        "passed": passed,
    }


def find_label_columns(columns) -> dict[str, str]:
    """Map each finding to its column in train.csv (robust to spelling such as "Bakers")."""
    cols = _match_columns(columns)
    out = {f: c["value"] for f, c in cols.items() if "value" in c}
    missing = [f for f in FINDINGS if f not in out]
    if missing:
        raise SchemaError(f"train.csv: label columns not found for {missing}")
    return out


def pos_weight_cap_upper(binary_train: pd.DataFrame) -> float:
    """U = the larger of 20 and the rarest finding's negatives/positives (plan, L2 table)."""
    ratios = []
    for f in FINDINGS:
        v = binary_train[f].fillna(0)
        pos = float((v == 1).sum())
        neg = float((v == 0).sum())
        if pos > 0:
            ratios.append(neg / pos)
    return round(max([20.0] + ratios), 1)
