"""N12, the error analysis of plan r2 (section 4), on synthetic stage data (CPU).

Every block is present and labelled; the privacy scan passes; every block equals an independent
recomputation (sklearn AUCs, pandas counts); item 3 equals N9's own s2.<encoder>.per_finding and the gold
cross-check equals N9's s3_gold; nothing on gold gives a small cell or an exact positive count back (the
back-calculation is tried and fails, while it succeeds on the old rule); gold studies without report
labels are left out, never read as "absent"; a failing block costs only itself; A4's gate does what it
says; item 4's list is fixed on the named rows; a chain mismatch fails the step; the CLI entry matches."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from knee import agg, paths
from knee.labels import FINDINGS, UID

sk = pytest.importorskip("sklearn.metrics")
MAIN = ("dinov2_vits14", "convnext_tiny")
ENCODERS = ("dinov2_vits14", "convnext_tiny", "vit_s16")
N_BOOT = 100
DEV = "development: these studies were test studies in iteration 1; likely optimistic"


def _sha16(p):
    from knee.nb.common import sha16

    return sha16(p)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """make_stage + make_units (three encoders), then what N12 reads beyond them: a report language per
    study with an English G-lang group of several languages (two synthetic groups of S2 fold 0 merged),
    N7's seed refits for the main encoders, agg_n0.json and Stage A's agg_n6a.json, refit_steps."""
    from tests.synth_stage import make_stage, make_units

    root = tmp_path_factory.mktemp("n12")
    inp = root / "input"
    facts = make_stage(inp, n=1200)
    make_units(inp)
    n0 = inp / "n0" / "private" / "n0"
    st = pd.read_csv(n0 / "studies.csv")
    eng = st["g_lang"].isin(["LANG0", "LANG5"]).to_numpy()  # both groups sit in S2 fold 0 (group % 5)
    rng = np.random.default_rng(11)
    st["language"] = st["g_lang"]
    st.loc[eng, "language"] = rng.choice(["ENGLISH", "SPANISH", "TURKISH", "FRENCH"], size=int(eng.sum()),
                                         p=[0.5, 0.3, 0.12, 0.08])
    st.loc[eng, "g_lang"] = "ENGLISH"
    st.loc[eng, "g_vendor"] = "ENGLISH|" + st.loc[eng, "manufacturer"].astype(str)
    st.to_csv(n0 / "studies.csv", index=False)
    # the gold studies' report labels get noise that grows with the finding, so label quality varies
    # (gold is never trained on, so nothing else changes; the binary label follows the soft one, as in N0)
    soft = pd.read_csv(n0 / "labels_soft.csv").set_index(UID)
    binary = pd.read_csv(n0 / "labels_binary.csv").set_index(UID)
    g_ids = st.loc[st["is_gold"].astype(bool), UID]
    noise = rng.normal(0, 1, (len(g_ids), 12)) * (0.05 + 0.05 * np.arange(12))
    sg = np.clip(soft.loc[g_ids, FINDINGS].to_numpy() + noise, 0, 1)
    soft.loc[g_ids, FINDINGS] = sg
    binary.loc[g_ids, FINDINGS] = np.where(np.isnan(sg), np.nan, np.where(sg > 0.5, 1.0, np.where(sg < 0.5, 0.0, np.nan)))
    soft.reset_index().to_csv(n0 / "labels_soft.csv", index=False)
    binary.reset_index().to_csv(n0 / "labels_binary.csv", index=False)

    # N7's seed refits (main encoders only, as N7 writes them): the unit predictions plus seed noise
    n7 = inp / "n7" / "private" / "n7"
    n7.mkdir(parents=True)
    for e in MAIN:
        base = pd.concat(pd.read_csv(inp / "n6a" / "private" / "n6a" / "units" / f"{e}_f{k}" / "predictions.csv")
                         for k in range(5))
        parts = []
        for sd in range(6):
            r = np.random.default_rng(100 + sd)
            df = base[[UID, "s2_fold"]].copy()
            df["seed"] = sd
            df[FINDINGS] = np.clip(base[FINDINGS].to_numpy() + r.normal(0, 0.05, (len(base), 12)), 0, 1)
            parts.append(df)
        pd.concat(parts).to_csv(n7 / f"s2seeds_{e}.csv", index=False)
    (inp / "n7" / "agg_n7.json").write_text(json.dumps({"device": "cpu"}))

    # N0's summary: the training prevalence (as N0 rounds it) and its file hashes
    lab = pd.read_csv(n0 / "labels_binary.csv").set_index(UID)
    tr_ids = st.loc[st["in_training"].astype(bool), UID]
    prev = {f: agg.share(lab.loc[tr_ids, f].fillna(0).mean()) for f in FINDINGS}
    (inp / "n0" / "agg_n0.json").write_text(json.dumps({
        "labels": {"prevalence_training": prev},
        "files_sha16": {"studies_sha16": _sha16(n0 / "studies.csv"), "labels_binary_sha16": _sha16(n0 / "labels_binary.csv")}}))

    # Stage A's own records: agg_n6a.json (upstream hashes) and each unit's refit_steps
    n5i = inp / "n5" / "private" / "n5" / "index.csv"
    (inp / "n6a" / "agg_n6a.json").write_text(json.dumps({"upstream": {
        "n0_studies": {"sha16": _sha16(n0 / "studies.csv")}, "n3_folds": {"sha16": "a21b342fec48d2b6"},
        "n5_index": {"sha16": _sha16(n5i)}}}))
    folds = pd.read_csv(inp / "n3" / "private" / "n3" / "folds.csv").set_index(UID)
    usable = pd.read_csv(n5i).set_index(UID)["usable"].reindex(folds.index).fillna(False).astype(bool)
    for e in ENCODERS:
        for k in range(5):
            cp = inp / "n6a" / "private" / "n6a" / "units" / f"{e}_f{k}" / "config.json"
            u = json.loads(cp.read_text())
            n_tr = int((folds["train"].astype(bool) & usable & (folds["s2_fold"] >= 0) & (folds["s2_fold"] != k)).sum())
            u["refit_steps"] = int(u["E"]) * math.ceil(n_tr / int(u["config"]["batch_size"]))
            cp.write_text(json.dumps(u))
    expect = {"n0_studies": _sha16(n0 / "studies.csv")[:8],
              "n3_folds": _sha16(inp / "n3" / "private" / "n3" / "folds.csv")[:8], "n5_index": _sha16(n5i)[:8]}
    return {"root": root, "inp": inp, "facts": facts, "expect": expect, "prev": prev}


def _copy_input(env, tmp_path) -> Path:
    inp = tmp_path / "input"
    shutil.copytree(env["inp"], inp)
    return inp


def _refresh_n0_record(inp: Path) -> None:
    """After a label file is edited: agg_n0.json's hashes again (as N0 would have written them)."""
    n0 = inp / "n0" / "private" / "n0"
    a0 = json.loads((inp / "n0" / "agg_n0.json").read_text())
    a0["files_sha16"]["labels_binary_sha16"] = _sha16(n0 / "labels_binary.csv")
    (inp / "n0" / "agg_n0.json").write_text(json.dumps(a0))


# ---------------------------------------------------------------- independent recomputation

class Ref:
    """The inputs read back with pandas only (no knee code): rows in N5's index order, as StageData has them."""

    def __init__(self, inp):
        n0 = inp / "n0" / "private" / "n0"
        idx = pd.read_csv(inp / "n5" / "private" / "n5" / "index.csv")
        self.uids = idx[UID].astype(str).to_numpy()
        folds = pd.read_csv(inp / "n3" / "private" / "n3" / "folds.csv").set_index(UID).reindex(self.uids)
        st = pd.read_csv(n0 / "studies.csv").set_index(UID).reindex(self.uids)
        usable = idx["usable"].astype(bool).to_numpy()
        self.f2 = folds["s2_fold"].to_numpy()
        self.train = folds["train"].astype(bool).to_numpy() & usable
        self.scored = folds["scored"].astype(bool).to_numpy() & usable
        self.gold_flag = folds["is_gold"].astype(bool).to_numpy() & usable
        self.g_lang = st["g_lang"].astype(str).to_numpy()
        self.lang = st["language"].astype(str).str.upper().to_numpy()
        self.binary = pd.read_csv(n0 / "labels_binary.csv").set_index(UID).reindex(self.uids)[FINDINGS].to_numpy(float)
        self.soft = pd.read_csv(n0 / "labels_soft.csv").set_index(UID).reindex(self.uids)[FINDINGS].to_numpy(float)
        self.Y = np.nan_to_num(self.binary, nan=0.0)  # the scoring rule: uncertain counts as negative
        self.GY = pd.read_csv(n0 / "labels_gold.csv").set_index(UID).reindex(self.uids)[FINDINGS].to_numpy(float)
        self.P = {}
        for e in ENCODERS:
            df = pd.concat(pd.read_csv(p) for p in sorted((inp / "n6a" / "private" / "n6a" / "units").glob(f"{e}_f*/predictions.csv")))
            self.P[e] = df.set_index(UID).reindex(self.uids)[FINDINGS].to_numpy(float)
        both = np.isfinite(self.P[MAIN[0]]).all(1) & np.isfinite(self.P[MAIN[1]]).all(1)
        self.ev = np.flatnonzero(self.train & (self.f2 >= 0) & self.scored & both)
        self.gold = np.flatnonzero(self.gold_flag & both)  # N9's S3 studies
        self.gold_r = self.gold[~np.isnan(self.soft[self.gold]).all(axis=1)]  # those with a report label


def _auc(y, s, w=None):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = np.isfinite(s) & ~np.isnan(y) & (np.ones(len(s), bool) if w is None else np.asarray(w, float) > 0)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return float("nan"), 0.0
    return float(sk.roc_auc_score(y, s, sample_weight=None if w is None else np.asarray(w, float)[m])), \
        float((y == 1).sum() * (y == 0).sum())


def _fold_stratified(s, y, folds):
    num = den = 0.0
    for k in np.unique(folds):
        m = folds == k
        a, pw = _auc(y[m], s[m])
        if pw > 0:
            num, den = num + a * pw, den + pw
    return num / den


def _macro_pooled(P, Y, rows, findings, w=None):
    vals = [_auc(Y[rows, j], P[rows, j], w)[0] for j in findings]
    return float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")


def _enough(y):
    return (y == 1).sum() >= 5 and (y == 0).sum() >= 5


def _draws(n, n_boot, seed=0):
    rng = np.random.default_rng(seed)  # N12's study bootstrap: n studies with replacement, as weights
    return [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(n_boot)]


def _cells(t, r):
    return {"tp": int(((r == 1) & (t == 1)).sum()), "fp": int(((r == 1) & (t == 0)).sum()),
            "fn": int(((r == 0) & (t == 1)).sum()), "tn": int(((r == 0) & (t == 0)).sum())}


def _small(x):
    return 1 <= x <= 4


def _rates(cl):
    """PPV and NPV under N9's rule for rates: blank when a cell the rate uses holds 1-4 studies."""
    ppv = None if _small(cl["tp"]) or _small(cl["fp"]) or cl["tp"] + cl["fp"] == 0 else cl["tp"] / (cl["tp"] + cl["fp"])
    npv = None if _small(cl["tn"]) or _small(cl["fn"]) or cl["tn"] + cl["fn"] == 0 else cl["tn"] / (cl["tn"] + cl["fn"])
    return ppv, npv


def _ints(o, key=None):
    if isinstance(o, dict):
        return [x for k, v in o.items() for x in _ints(v, k)]
    if isinstance(o, list):
        return [x for v in o for x in _ints(v, key)]
    return [(key, o)] if isinstance(o, int) and not isinstance(o, bool) else []


def _floats(o):
    if isinstance(o, dict):
        return [x for v in o.values() for x in _floats(v)]
    if isinstance(o, list):
        return [x for v in o for x in _floats(v)]
    return [o] if isinstance(o, float) else []


def _lists(o):
    if isinstance(o, dict):
        return [x for v in o.values() for x in _lists(v)]
    if isinstance(o, list):
        return [len(o)] + [x for v in o for x in _lists(v)]
    return []


# ---------------------------------------------------------------- the main run

BLOCKS = ("gold_paired_check", "gold_label_quality", "fm5b_attenuation_check", "per_finding_table",
          "english_group_by_language", "development")


def test_n12_blocks_privacy_and_independent_recomputation(env, monkeypatch, capfd):
    from knee import safe_log
    from knee.nb import n9, n12

    inp, root = env["inp"], env["root"]
    monkeypatch.setattr(paths, "WORKING", root / "w9")
    s9 = n9.run(root=inp, n_boot=50, quiet=True)
    capfd.readouterr()
    monkeypatch.setattr(paths, "WORKING", root / "w12")
    s = n12.run(root=inp, expect=env["expect"], n_boot=N_BOOT, quiet=True)
    printed = capfd.readouterr()
    text = (paths.WORKING / "agg_n12.json").read_text()
    saved = json.loads(text)

    # every block exists, is labelled, and nothing failed
    assert s["status"] == "ok" and s["errors"] == {} and saved["errors"] == {}
    assert s["chain"]["s2_folds_match_units"] is True and s["main_complete"] is True
    for key in BLOCKS:
        b = saved[key]
        assert b["descriptive"] is True and b["added_after_results"] is True and len(b["scope"]) > 20, key
        assert "status" not in b, key
    for key in ("a2_seed_ensemble", "a3_encoder_ensemble", "a4_calibration"):
        b = saved["development"][key]
        assert b["descriptive"] is True and b["added_after_results"] is True and len(b["scope"]) > 20, key
        assert b["label"] == DEV and "status" not in b, key
    assert saved["development"]["label"] == DEV
    assert saved["descriptive"] is True and saved["added_after_results"] is True and "decides nothing" in saved["scope"]
    assert set(saved["not_run"]) == {"label_agreement_strata", "g_vendor_strata", "missing_plane_fallback",
                                     "per_finding_x_language", "laterality"}
    assert saved["upstream"]["n0_studies"]["sha16"].startswith(env["expect"]["n0_studies"])
    ins = saved["inputs"]
    assert ins["n7_seed_refits_found"] is True and ins["language_column"] is True and ins["n7_device"] == "cpu"
    assert all(len(ins["s2seeds_sha16"][e]) == 16 for e in MAIN) and ins["s2seeds_sha16"]["vit_s16"] is None

    # privacy: no identifier anywhere, the summary checks pass, no per-study list, nothing private written
    uids = env["facts"]["uids"]
    assert not [u for u in uids if u in printed.out + printed.err + text]
    agg._check(saved)
    assert max(_lists(saved)) <= len(FINDINGS)
    assert all(v == 0 or v >= 5 or k in agg.STRUCTURAL for k, v in _ints(saved))
    assert sorted(p.name for p in paths.WORKING.iterdir() if p.is_file()) == ["agg_n12.json", "run.json"]
    assert not [p for p in (paths.WORKING / "private").rglob("*") if p.is_file()]
    # B1: every per-finding number on gold has at most 2 decimals
    per_finding_gold = [saved["gold_label_quality"]["per_finding"],
                        {f: r["label_quality"] for f, r in saved["per_finding_table"]["rows"].items()},
                        {e: v["per_finding"] for e, v in saved["fm5b_attenuation_check"]["encoders"].items()}]
    vals = _floats(per_finding_gold)
    assert len(vals) > 100 and all(round(x, 2) == x for x in vals)
    block = json.loads(printed.out.split(safe_log.BEGIN)[-1].split(safe_log.END)[0])
    agg._check(block)
    assert block["status"] == "ok" and block["chain_ok"] is True and block["errors"] == {}
    assert block["development"]["label"].startswith("development:")
    assert set(block["fm5b_report_minus_if_independent"]) == set(ENCODERS)

    ref = Ref(inp)
    assert len(ref.ev) > 500 and s["evaluation_studies"] == agg.count(len(ref.ev)) == s9["evaluation_studies"]
    assert len(ref.gold_r) == len(ref.gold)  # the fixture's gold studies all have report labels
    _item1(s, s9, ref)
    _item2(s, ref)
    _fm5b(s, ref)
    _item3(s, s9, ref, env)
    _item4(s, ref)
    _item5(s, s9, ref, inp)


def _item1(s, s9, ref):
    g = s["gold_paired_check"]
    gold = ref.gold_r
    n = len(gold)
    assert 50 <= n <= 58 and g["studies"] == agg.count(n) and g["studies_text"] == f"about {agg.count(n)}"
    assert g["gold_studies_without_report_labels"] == 0 and g["studies_without_report_labels_left_out"] is False
    both = [j for j in range(12) if _enough(ref.GY[gold, j]) and _enough(ref.Y[gold, j])]
    assert g["findings"] == [FINDINGS[j] for j in both] and len(both) >= 6
    assert set(g["findings_left_out"]) == set(FINDINGS) - set(g["findings"])
    # S1: the fixture is built so that the lists agree, so the digit-for-digit checks below cannot skip themselves
    assert g["findings_same_as_n9_s3"] is True and g["findings"] == s9["s3_gold"]["findings"] == g["n9_findings"]
    assert set(g["encoders"]) == set(ENCODERS) and g["n_boot"] == N_BOOT
    draws = _draws(n, N_BOOT)
    n9_find = [FINDINGS.index(f) for f in s9["s3_gold"]["findings"]]
    for e in ENCODERS:
        P = ref.P[e]
        v = g["encoders"][e]
        assert v["expert_labels_on_n9_findings"] == s9["s3_gold"]["macro_auc"][e]  # always N9's number
        assert v["expert_labels_on_n9_findings"] == pytest.approx(_macro_pooled(P, ref.GY, ref.gold, n9_find), abs=6e-5)
        ex, rp = _macro_pooled(P, ref.GY, gold, both), _macro_pooled(P, ref.Y, gold, both)
        assert v["expert_labels"] == pytest.approx(ex, abs=6e-5) and v["report_labels"] == pytest.approx(rp, abs=6e-5)
        assert v["expert_labels"] == s9["s3_gold"]["macro_auc"][e]
        assert v["expert_minus_report"]["estimate"] == pytest.approx(ex - rp, abs=1.1e-4)
        exd = np.array([_macro_pooled(P, ref.GY, gold, both, w) for w in draws])
        rpd = np.array([_macro_pooled(P, ref.Y, gold, both, w) for w in draws])
        for got, vv in ((v["expert_minus_report"], exd - rpd), (v["expert_labels_interval"], exd),
                        (v["report_labels_interval"], rpd)):
            ok = np.isfinite(vv)
            lo, hi = np.percentile(vv[ok], [2.5, 97.5])
            assert got["lo"] == pytest.approx(lo, abs=1.1e-4) and got["hi"] == pytest.approx(hi, abs=1.1e-4)
            assert got["valid_share"] == pytest.approx(ok.mean(), abs=1e-3) and got["hi"] > got["lo"]
        assert v["expert_minus_report"]["lo"] <= v["expert_minus_report"]["estimate"] <= v["expert_minus_report"]["hi"]


def _label_quality(ref, draws=()):
    gold, out = ref.gold_r, {}
    for j, f in enumerate(FINDINGS):
        ye = ref.GY[gold, j]
        if not _enough(ye):
            continue
        ppv, npv = _rates(_cells(ye, ref.Y[gold, j]))
        out[f] = {"soft": _auc(ye, ref.soft[gold, j])[0],
                  "soft_draws": np.array([_auc(ye, ref.soft[gold, j], w)[0] for w in draws]), "ppv": ppv, "npv": npv}
    return out


def _item2(s, ref):
    q = s["gold_label_quality"]
    want = _label_quality(ref, _draws(len(ref.gold_r), N_BOOT))
    assert set(q["per_finding"]) == set(FINDINGS) and q["n_boot"] == N_BOOT
    assert q["findings_suppressed"] == [f for f in FINDINGS if f not in want]
    blank = [f for f, w in want.items() if w["ppv"] is None or w["npv"] is None]
    assert q["rates_blank"] == blank and 0 < len(blank) < len(want)  # both kinds occur here
    for f, w in want.items():
        v = q["per_finding"][f]
        assert v["soft_label_auc"] == pytest.approx(w["soft"], abs=5.1e-3)
        ok = np.isfinite(w["soft_draws"])
        lo, hi = np.percentile(w["soft_draws"][ok], [2.5, 97.5])
        iv = v["soft_label_auc_interval"]
        assert iv["lo"] == pytest.approx(lo, abs=5.1e-3) and iv["hi"] == pytest.approx(hi, abs=5.1e-3)
        assert iv["valid_share"] == pytest.approx(ok.mean(), abs=1e-3)
        for key in ("ppv", "npv"):
            assert (v[key] is None) if w[key] is None else v[key] == pytest.approx(w[key], abs=5.1e-3)
        if w["ppv"] is not None and w["npv"] is not None:
            assert v["ppv_plus_npv_minus_1"] == pytest.approx(w["ppv"] + w["npv"] - 1, abs=5.1e-3)
        else:
            assert v["ppv_plus_npv_minus_1"] is None
    assert q["soft_label_macro_auc"] == pytest.approx(np.mean([w["soft"] for w in want.values()]), abs=6e-4)


def _fm5b(s, ref):
    b = s["fm5b_attenuation_check"]
    gold = ref.gold_r
    n = len(gold)
    both = [j for j in range(12) if _enough(ref.GY[gold, j]) and _enough(ref.Y[gold, j])]
    assert b["findings"] == [FINDINGS[j] for j in both] and b["n_boot"] == N_BOOT
    shown = {j: None not in _rates(_cells(ref.GY[gold, j], ref.Y[gold, j])) for j in both}
    assert b["per_finding_blank"] == [FINDINGS[j] for j in both if not shown[j]]
    assert b["per_finding_blank"] == [f for f in s["gold_label_quality"]["rates_blank"] if f in b["findings"]]  # item 2's rule
    draws = _draws(n, N_BOOT)

    def per_finding(P, w=None):
        ww = np.ones(n) if w is None else w
        res = {}
        for j in both:
            t, r = ref.GY[gold, j], ref.Y[gold, j]
            rp, rn = ww * (r == 1), ww * (r == 0)
            if rp.sum() <= 0 or rn.sum() <= 0:
                res[j] = (np.nan, np.nan, np.nan)
                continue
            ppv, npv = (rp * (t == 1)).sum() / rp.sum(), (rn * (t == 0)).sum() / rn.sum()
            a_e, a_r = _auc(t, P[gold, j], w)[0], _auc(r, P[gold, j], w)[0]
            res[j] = (a_r, 0.5 + (ppv + npv - 1) * (a_e - 0.5), a_e)
        return res

    for e in ENCODERS:
        P, v = ref.P[e], b["encoders"][e]
        base = per_finding(P)
        obs, pred = np.mean([base[j][0] for j in both]), np.mean([base[j][1] for j in both])
        assert v["report_labels"] == s["gold_paired_check"]["encoders"][e]["report_labels"]
        assert v["report_labels"] == pytest.approx(obs, abs=6e-5) and v["report_labels_if_independent"] == pytest.approx(pred, abs=6e-5)
        d = v["report_minus_if_independent"]
        assert d["estimate"] == pytest.approx(obs - pred, abs=1.1e-4)
        per_draw = [per_finding(P, w) for w in draws]
        macro = np.array([np.mean([x[j][0] - x[j][1] for j in both]) for x in per_draw])
        ok = np.isfinite(macro)
        lo, hi = np.percentile(macro[ok], [2.5, 97.5])
        assert d["lo"] == pytest.approx(lo, abs=1.1e-4) and d["hi"] == pytest.approx(hi, abs=1.1e-4)
        assert d["valid_share"] == pytest.approx(ok.mean(), abs=1e-3) and d["lo"] < d["hi"]
        for j in both:
            row = v["per_finding"][FINDINGS[j]]
            a_r, p_j, a_e = base[j]
            assert row["expert_labels_auc"] == pytest.approx(a_e, abs=5.1e-3)
            assert row["report_labels_auc"] == pytest.approx(a_r, abs=5.1e-3)
            if shown[j]:
                assert row["report_labels_if_independent"] == pytest.approx(p_j, abs=5.1e-3)
                diffs = np.array([x[j][0] - x[j][1] for x in per_draw])
                okj = np.isfinite(diffs)
                lo, hi = np.percentile(diffs[okj], [2.5, 97.5])
                rd = row["report_minus_if_independent"]
                assert rd["estimate"] == pytest.approx(a_r - p_j, abs=6e-3)
                assert rd["lo"] == pytest.approx(lo, abs=5.1e-3) and rd["hi"] == pytest.approx(hi, abs=5.1e-3)
            else:
                assert row["report_labels_if_independent"] is None and row["report_minus_if_independent"] is None
    assert "independent" in b["assumption"]


def _item3(s, s9, ref, env):
    from scipy.stats import spearmanr

    t = s["per_finding_table"]
    assert list(t["rows"]) == FINDINGS and t["prevalence_source"] == "agg_n0.json labels.prevalence_training"
    assert "training studies" in t["prevalence_population"]
    ev = ref.ev
    lq = _label_quality(ref)
    sp = t["spearman_label_quality_vs_s2_auc"]
    used = [f for f in FINDINGS if f in lq]
    assert sp["findings_used"] == used and 0.5 < sp["no_association_band"] < 0.66  # about 0.58 for 12 findings
    assert "no_association_band" in sp["reading"] and "English group" in sp["populations"]
    for e in ENCODERS:
        auc = {}
        for j, f in enumerate(FINDINGS):
            auc[f] = _fold_stratified(ref.P[e][ev, j], ref.Y[ev, j], ref.f2[ev])
            got = t["rows"][f]["s2_auc"][e]
            assert got == pytest.approx(auc[f], abs=6e-4)
            assert got == s9["s2"][e]["per_finding"][f]  # the very numbers N9 writes
        rho = spearmanr([lq[f]["soft"] for f in used], [auc[f] for f in used]).statistic
        assert sp["encoders"][e]["rho"] == pytest.approx(rho, abs=1.1e-3)
    for f in FINDINGS:
        row = t["rows"][f]
        assert row["training_prevalence"] == env["prev"][f]
        assert row["label_quality"] == s["gold_label_quality"]["per_finding"][f]  # item 2 beside it


def _item4(s, ref):
    g = s["english_group_by_language"]
    ev = ref.ev
    grp = ev[ref.g_lang[ev] == "ENGLISH"]
    k = int(pd.Series(ref.f2[grp]).mode().iloc[0])
    assert g["group"] == "ENGLISH" and g["fold"] == k == 0 and g["group_in_one_fold"] is True
    assert g["list_fixed_on"] == ["ENGLISH", "SPANISH", "TURKISH"]
    assert g["training_studies"] == agg.count(int((ref.train & (ref.f2 >= 0) & (ref.f2 != k)).sum()))
    rows_all = grp[ref.f2[grp] == k]
    sizes = pd.Series(ref.lang[rows_all]).value_counts()
    core = [L for L in ("ENGLISH", "SPANISH", "TURKISH") if sizes.get(L, 0) >= 50]
    assert core == ["ENGLISH", "SPANISH"]  # TURKISH and FRENCH have under 50 here
    assert g["languages_reported"] == core
    assert g["languages_suppressed"] == {L: {"reason": "under 50 evaluation studies"} for L in sizes.index if sizes[L] < 50}
    strata = {L: rows_all[ref.lang[rows_all] == L] for L in core}
    find = [j for j in range(12) if all(_enough(ref.Y[r, j]) for r in strata.values())]
    assert g["findings"] == [FINDINGS[j] for j in find] and find
    assert g["findings_left_out"] == {FINDINGS[j]: [L for L, r in strata.items() if not _enough(ref.Y[r, j])]
                                      for j in range(12) if j not in find}
    for L, r in [*strata.items(), ("whole group", rows_all)]:
        got = g["rows"][L] if L in g["rows"] else g["reference_whole_group"]
        assert got["studies"] == agg.count(len(r))
        for e in ENCODERS:
            v = got["encoders"][e]
            assert v["macro_auc"] == pytest.approx(_macro_pooled(ref.P[e], ref.Y, r, find), abs=6e-5)
            assert v["lo"] <= v["macro_auc"] <= v["hi"] and v["valid_share"] > 0.9


def _item5(s, s9, ref, inp):
    from scipy.special import expit, logit

    from knee.nb import n9

    dev = s["development"]
    ev = ref.ev

    def levels(P, rows):
        fs = np.mean([_fold_stratified(P[rows, j], ref.Y[rows, j], ref.f2[rows]) for j in range(12)])
        mf = np.mean([_macro_pooled(P, ref.Y, rows[ref.f2[rows] == k], range(12)) for k in range(5)])
        return fs, mf

    # A2: the mean probability over N7's six seeds, and the members' own mean (M6)
    a2 = dev["a2_seed_ensemble"]
    assert set(a2["encoders"]) == set(MAIN) and set(a2["not_available"]) == {"vit_s16"}
    for e in MAIN:
        df = pd.read_csv(inp / "n7" / "private" / "n7" / f"s2seeds_{e}.csv")
        members = [df[df["seed"] == sd].set_index(UID).reindex(ref.uids)[FINDINGS].to_numpy(float) for sd in range(6)]
        fs, mf = levels(np.mean(members, axis=0), ev)
        v = a2["encoders"][e]
        assert v["seeds"] == [0, 1, 2, 3, 4, 5]
        assert v["fold_stratified_macro_auc"] == pytest.approx(fs, abs=6e-5) and v["mean_of_fold_macros"] == pytest.approx(mf, abs=6e-5)
        mm = np.mean([levels(M, ev) for M in members], axis=0)
        assert v["members_mean"]["fold_stratified_macro_auc"] == pytest.approx(mm[0], abs=6e-5)
        assert v["members_mean"]["mean_of_fold_macros"] == pytest.approx(mm[1], abs=6e-5)
    # A3: the equal-weight mean of the three encoders, and the three encoders alone (M6)
    a3 = dev["a3_encoder_ensemble"]
    fs, mf = levels(np.mean([ref.P[e] for e in ENCODERS], axis=0), ev)
    assert a3["studies"] == agg.count(len(ev)) and a3["share_of_evaluation_studies"] == 1.0
    assert a3["fold_stratified_macro_auc"] == pytest.approx(fs, abs=6e-5) and a3["mean_of_fold_macros"] == pytest.approx(mf, abs=6e-5)
    mm = np.mean([levels(ref.P[e], ev) for e in ENCODERS], axis=0)
    assert a3["members_mean"]["fold_stratified_macro_auc"] == pytest.approx(mm[0], abs=6e-5)
    assert a3["members_mean"]["fold_stratified_macro_auc"] == pytest.approx(
        np.mean([s9["s2"][e]["fold_stratified_macro_auc"] for e in ENCODERS]), abs=1e-4)
    # A4: positive weights from the outer training labels and each unit's cap; logits shifted by -log(w)
    a4 = dev["a4_calibration"]
    assert set(a4["encoders"]) == set(ENCODERS) and a4["skipped"] == {}
    assert a4["gate_must_be_true"] == ["chain_guard_passed", "unit_for_every_s2_fold", "stage_a_n0_studies_same",
                                       "pos_cap_recorded"]
    assert a4["gate_must_not_be_false"] == ["refit_steps_match_training_size", "labels_binary_from_attached_n0_run"]
    pos_any = (ref.Y == 1).any(axis=1).astype(float)
    for e in ENCODERS:
        v = a4["encoders"][e]
        assert all(x is True for x in v["checks"].values()), v["checks"]
        assert set(v["checks"]) == set(a4["gate_must_be_true"]) | set(a4["gate_must_not_be_false"])
        assert v["info"] == {"stage_a_n5_index_same": True}
        Q = ref.P[e].copy()
        for k in range(5):
            cfg = json.loads((inp / "n6a" / "private" / "n6a" / "units" / f"{e}_f{k}" / "config.json").read_text())
            tr = np.flatnonzero(ref.train & (ref.f2 >= 0) & (ref.f2 != k))
            yb = ref.binary[tr]
            w = np.minimum((yb == 0).sum(0) / np.maximum((yb == 1).sum(0), 1), cfg["config"]["pos_cap"]).astype(np.float32)
            rows = ev[ref.f2[ev] == k]
            Q[rows] = expit(logit(ref.P[e][rows]) - np.log(w.astype(float)))
        assert v["ece"]["before"] == s9["s2"][e]["ece"] and v["ece"]["after"] == n9._ece(Q[ev], ref.Y[ev])
        assert v["ece"]["after"] != v["ece"]["before"]
        assert v["any_finding_auc"]["before"] == s9["triage_secondary"]["models"][e]["any_finding_auc"]
        assert v["any_finding_auc"]["after"] == pytest.approx(_fold_stratified(Q[ev].max(1), pos_any[ev], ref.f2[ev]), abs=6e-5)
        assert v["fold_stratified_macro_auc_unchanged"] is True


# ---------------------------------------------------------------- B1: the back-calculation fails

def _gold_ctx(rng, n=58):
    """A hand-made gold set of n studies and 12 findings (expert labels, report labels with the sensitivity
    and specificity of probe4, soft labels, three encoders' predictions), as N12's gold blocks read it."""
    GY, RY = np.zeros((n, 12)), np.zeros((n, 12))
    for j in range(12):
        GY[rng.permutation(n)[:int(rng.integers(8, 36))], j] = 1.0
        sens, spec = rng.uniform(0.6, 0.95), rng.uniform(0.8, 0.98)
        RY[:, j] = np.where(GY[:, j] == 1, rng.random(n) < sens, rng.random(n) > spec)
    soft = np.clip(0.5 * RY + 0.3 * GY + rng.normal(0, 0.15, (n, 12)), 0, 1)
    units = {e: np.clip(0.3 + 0.4 * GY + rng.normal(0, 0.3, (n, 12)), 0, 1) for e in ENCODERS}
    return SimpleNamespace(gold=np.arange(n), d=SimpleNamespace(gold_y=GY, soft=soft, binary=RY.copy()), Y=RY,
                           units=units, g_find=list(range(12)))


def _tables(p_window, n):
    for p in p_window:
        for tp in range(p + 1):
            for fp in range(n - p + 1):
                yield {"tp": tp, "fp": fp, "fn": p - tp, "tn": n - p - fp}


def _consistent(cl, pub, rule):
    """Whether a candidate table could have produced the published ppv, npv and factor under a rule."""
    if rule == "cells":
        ppv_shown = not (_small(cl["tp"]) or _small(cl["fp"])) and cl["tp"] + cl["fp"] > 0
        npv_shown = not (_small(cl["tn"]) or _small(cl["fn"])) and cl["tn"] + cl["fn"] > 0
    else:  # the first version's rule: a denominator of at least 5
        ppv_shown, npv_shown = cl["tp"] + cl["fp"] >= 5, cl["tn"] + cl["fn"] >= 5
    if (pub["ppv"] is not None) != ppv_shown or (pub["npv"] is not None) != npv_shown:
        return False
    ppv = cl["tp"] / (cl["tp"] + cl["fp"]) if ppv_shown else None
    npv = cl["tn"] / (cl["tn"] + cl["fn"]) if npv_shown else None
    if ppv is not None and agg.share(ppv) != pub["ppv"]:
        return False
    if npv is not None and agg.share(npv) != pub["npv"]:
        return False
    if pub.get("factor") is not None and agg.share(ppv + npv - 1) != pub["factor"]:
        return False
    return True


def _recovered_small_cells(true, pub, p_window, n, rule) -> int:
    """How many of the true table's cells of 1-4 an attacker gets back exactly (one value left)."""
    cands = [cl for cl in _tables(p_window, n) if _consistent(cl, pub, rule)]
    assert true in cands
    return sum(1 for k, v in true.items() if _small(v) and {cl[k] for cl in cands} == {v})


def _auc_feasible(a, p, n, nd):
    """Whether some AUC m / (2 p (n - p)) with integer m rounds to a at nd decimals (ties count one half)."""
    D = 2 * p * (n - p)
    m0 = int(round(a * D))
    return any(0 <= m <= D and round(m / D, nd) == a for m in range(m0 - 3, m0 + 4))


def test_n12_gold_numbers_cannot_give_a_small_count_back():
    """Review r1, B1. An attacker who knows everything public (the exact gold count, agg_n0's positives rounded
    to 5, N12's rules) tries to get the 2x2 cells and the exact positive counts back from N12's gold numbers.
    Under the first version's rule the cells of 1-4 come back, and AUCs at 4 decimals give the exact positive
    count; under the fixed rules neither does."""
    from knee import metrics
    from knee.nb import n12

    rng = np.random.default_rng(7)
    n = 58
    tried = recovered_new = recovered_old = with_small = 0
    p_unique_new = p_unique_4dp = n_findings = 0
    for _ in range(8):
        c = _gold_ctx(rng, n)
        q = n12.gold_label_quality(c, 10, 0)
        fm = n12.fm5b_attenuation_check(c, 10, 0)
        raw = n12.label_quality_raw(c, 10, 0)
        agg._check(json.loads(json.dumps(q)))
        agg._check(json.loads(json.dumps(fm)))
        for j, f in enumerate(FINDINGS):
            v = q["per_finding"][f]
            if "suppressed" in v:
                continue
            p_true = int((c.d.gold_y[:, j] == 1).sum())
            shown = agg.count(p_true)  # what agg_n0.json publishes
            window = [p for p in range(5, n - 4) if agg.count(p) == shown]
            true = _cells(c.d.gold_y[:, j], c.Y[:, j])
            pub = {"ppv": v["ppv"], "npv": v["npv"], "factor": v["ppv_plus_npv_minus_1"]}
            old_ppv = true["tp"] / (true["tp"] + true["fp"]) if true["tp"] + true["fp"] >= 5 else None
            old_npv = true["tn"] / (true["tn"] + true["fn"]) if true["tn"] + true["fn"] >= 5 else None
            pub_old = {"ppv": agg.share(old_ppv) if old_ppv is not None else None,
                       "npv": agg.share(old_npv) if old_npv is not None else None}
            tried += 1
            with_small += any(_small(x) for x in true.values())
            recovered_new += _recovered_small_cells(true, pub, window, n, "cells")
            recovered_old += _recovered_small_cells(true, pub_old, window, n, "old")
            # the exact positive count from the per-finding AUCs against the expert labels (one denominator)
            aucs = [v["soft_label_auc"]] + [fm["encoders"][e]["per_finding"][f]["expert_labels_auc"]
                                            for e in ENCODERS if f in fm["encoders"][e]["per_finding"]]
            exact = [raw[f]["soft_label_auc"]] + [metrics.auc(c.units[e][:, j], c.d.gold_y[:, j])[0] for e in ENCODERS]
            left_new = [p for p in window if all(_auc_feasible(a, p, n, 2) for a in aucs)]
            left_4dp = [p for p in window if all(_auc_feasible(round(a, 4), p, n, 4) for a in exact)]
            assert p_true in left_new and p_true in left_4dp
            n_findings += 1
            p_unique_new += len(left_new) == 1
            p_unique_4dp += len(left_4dp) == 1
    assert tried >= 80 and with_small > tried / 2  # most findings have a cell of 1-4 here, as on the real gold
    assert recovered_new == 0  # the fixed rule gives no small cell back
    assert recovered_old > tried / 4  # the first version's rule did (probe4), so the attack is real
    assert p_unique_new == 0  # 2 decimals leave every rounded positive count open
    assert p_unique_4dp > n_findings / 2  # 4 decimals would give the exact count back


# ---------------------------------------------------------------- item 4's rules

def test_n12_english_group_list_is_fixed_on_the_named_rows():
    """Review r1, S5: the finding list is fixed on English, Spanish and Turkish. Another language with 50
    studies is reported on that list only if it meets it; else it is suppressed with the reason. A row under
    50 studies is suppressed. The group is found by its English reports, not by fold number."""
    from knee.nb import n12

    rng = np.random.default_rng(3)
    lang = np.array(["ENGLISH"] * 250 + ["SPANISH"] * 150 + ["TURKISH"] * 120 + ["FRENCH"] * 70 + ["LATIN"] * 60
                    + ["CATALAN"] * 40 + ["CROATIAN"] * 200)
    n = len(lang)
    g_lang = np.where(lang == "CROATIAN", "CROATIAN", "ENGLISH")
    f2 = np.where(g_lang == "ENGLISH", 2, 0)  # the English group sits in S2 fold 2
    Y = (rng.random((n, 12)) < 0.3).astype(float)
    turkish, latin = np.flatnonzero(lang == "TURKISH"), np.flatnonzero(lang == "LATIN")
    Y[turkish, 1] = 0.0
    Y[turkish[:4], 1] = 1.0  # MCL: 4 positives among the Turkish reports -> MCL leaves the list
    Y[latin, 11] = 0.0
    Y[latin[:3], 11] = 1.0  # Fracture: 3 positives among the Latin reports -> LATIN is suppressed, Fracture stays
    units = {e: np.clip(Y * 0.3 + rng.random((n, 12)) * 0.7, 0, 1) for e in MAIN}
    c = SimpleNamespace(ev=np.arange(n), Y=Y, f2=f2, elig=list(range(12)), units=units, d=SimpleNamespace(g_lang=g_lang))
    out = n12.english_group_by_language(c, lang, n_boot=50, seed=0)
    agg._check(json.loads(json.dumps(out)))
    assert out["group"] == "ENGLISH" and out["fold"] == 2 and out["group_in_one_fold"] is True
    assert out["findings_left_out"] == {"MCL": ["TURKISH"]}
    find = [j for j in range(12) if j != 1]
    assert out["findings"] == [FINDINGS[j] for j in find] and "Fracture" in out["findings"]
    assert out["languages_reported"] == ["ENGLISH", "SPANISH", "TURKISH", "FRENCH"]
    assert out["languages_suppressed"] == {
        "LATIN": {"reason": "under 5 positives or 5 negatives for a finding of the fixed list", "findings": ["Fracture"]},
        "CATALAN": {"reason": "under 50 evaluation studies"}}
    for L in out["languages_reported"]:
        r = np.flatnonzero(lang == L)
        want = np.mean([sk.roc_auc_score(Y[r, j], units["convnext_tiny"][r, j]) for j in find])
        assert out["rows"][L]["encoders"]["convnext_tiny"]["macro_auc"] == pytest.approx(want, abs=6e-5)
    assert out["reference_whole_group"]["studies"] == agg.count(690)


# ---------------------------------------------------------------- faults and gates

def test_n12_a_failing_block_costs_only_itself_and_a4_says_why_it_is_skipped(env, tmp_path, monkeypatch):
    from knee.nb import n12

    inp = _copy_input(env, tmp_path)
    # one unit's refit_steps disagree with the training-set size: A4 is skipped for that encoder only
    cp = inp / "n6a" / "private" / "n6a" / "units" / "vit_s16_f2" / "config.json"
    u = json.loads(cp.read_text())
    u["refit_steps"] += 1
    cp.write_text(json.dumps(u))
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w1")
    monkeypatch.setattr(n12, "gold_label_quality", lambda c, n_boot, seed: 1 / 0)
    monkeypatch.setattr(n12, "a3_encoder_ensemble", lambda c: {"descriptive": True, "studies": 3})  # an unsafe count
    monkeypatch.setattr(n12, "_languages", lambda d: 1 / 0)  # the report languages cannot be read
    monkeypatch.setattr(n12, "_inputs", lambda c: 1 / 0)
    s = n12.run(root=inp, n_boot=20, quiet=True)
    saved = json.loads((paths.WORKING / "agg_n12.json").read_text())
    assert set(saved["errors"]) == {"gold_label_quality", "development.a3_encoder_ensemble", "english_group_by_language",
                                    "inputs"}
    assert "ZeroDivisionError" in saved["errors"]["gold_label_quality"]
    assert "UnsafeSummary" in saved["errors"]["development.a3_encoder_ensemble"]
    assert not {"gold_label_quality", "english_group_by_language", "inputs"} & set(saved)
    assert "a3_encoder_ensemble" not in saved["development"]
    assert s["status"].startswith("ok, some blocks failed")
    for key in ("gold_paired_check", "fm5b_attenuation_check", "per_finding_table"):
        assert saved[key]["descriptive"] is True, key
    assert saved["per_finding_table"]["rows"]["ACL"]["label_quality"]["soft_label_auc"] is not None  # its own copy
    a4 = saved["development"]["a4_calibration"]
    assert set(a4["encoders"]) == set(MAIN) and set(a4["skipped"]) == {"vit_s16"}
    assert "refit_steps" in a4["skipped"]["vit_s16"]["reason"]
    assert a4["skipped"]["vit_s16"]["checks"]["refit_steps_match_training_size"] is False
    monkeypatch.undo()

    # without Stage A's agg file the N0 version it trained on is unknown: A4 is skipped for every encoder
    (inp / "n6a" / "agg_n6a.json").unlink()
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w2")
    saved = n12.run(root=inp, n_boot=20, quiet=True)
    a4 = saved["development"]["a4_calibration"]
    assert a4["encoders"] == {} and set(a4["skipped"]) == set(ENCODERS) and saved["errors"] == {}
    assert all(v["checks"]["stage_a_n0_studies_same"] is None and "N0" in v["reason"] for v in a4["skipped"].values())
    assert saved["development"]["a2_seed_ensemble"]["encoders"]  # the other development numbers stand


def test_n12_a4_gate_n5_is_information_and_missing_encoders_are_skipped(env, tmp_path, monkeypatch):
    """Review r1, S4: a false N5 check never stops A4 (the weights do not use N5) and is shown under info;
    null refit_steps and labels records do not stop it; an encoder that is not attached is listed as skipped."""
    from knee.nb import n12

    inp = _copy_input(env, tmp_path)
    a6 = json.loads((inp / "n6a" / "agg_n6a.json").read_text())
    a6["upstream"]["n5_index"]["sha16"] = "0" * 16
    (inp / "n6a" / "agg_n6a.json").write_text(json.dumps(a6))
    for cp in (inp / "n6a" / "private" / "n6a" / "units").glob("*/config.json"):
        u = json.loads(cp.read_text())
        u.pop("refit_steps")
        cp.write_text(json.dumps(u))
    a0 = json.loads((inp / "n0" / "agg_n0.json").read_text())
    a0.pop("files_sha16")
    (inp / "n0" / "agg_n0.json").write_text(json.dumps(a0))
    for ud in (inp / "n6a" / "private" / "n6a" / "units").glob("vit_s16_f*"):
        shutil.rmtree(ud)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    s = n12.run(root=inp, n_boot=20, quiet=True)
    assert s["errors"] == {}
    a4 = s["development"]["a4_calibration"]
    assert set(a4["encoders"]) == set(MAIN) and a4["skipped"] == {"vit_s16": {"reason": "not attached"}}
    for e in MAIN:
        v = a4["encoders"][e]
        assert v["info"] == {"stage_a_n5_index_same": False}
        assert v["checks"]["refit_steps_match_training_size"] is None and v["checks"]["labels_binary_from_attached_n0_run"] is None
        assert all(v["checks"][k] is True for k in a4["gate_must_be_true"])
    assert s["development"]["a3_encoder_ensemble"]["missing"] == ["vit_s16"]


def test_n12_gold_studies_without_report_labels_and_the_n9_cross_check(env, tmp_path, monkeypatch):
    """Review r1, S2 and S1. Gold studies with no report label are left out of every report comparison (counted,
    flagged), never read as "absent"; the expert macro on N9's studies and findings stays N9's number, also when
    the paired finding list differs from N9's."""
    from knee.nb import n12

    inp = _copy_input(env, tmp_path)
    n0 = inp / "n0" / "private" / "n0"
    st = pd.read_csv(n0 / "studies.csv")
    gold_ids = set(st.loc[st["is_gold"].astype(bool), UID].astype(str))
    ref0 = Ref(inp)
    drop = set(ref0.uids[ref0.gold[:5]])  # five of N9's gold studies lose every report label
    for name in ("labels_soft", "labels_binary"):
        df = pd.read_csv(n0 / f"{name}.csv")
        df.loc[df[UID].astype(str).isin(drop), FINDINGS] = np.nan
        df.to_csv(n0 / f"{name}.csv", index=False)
    # and Fracture keeps only 3 report positives on gold: the paired list loses it, N9's list does not
    b = pd.read_csv(n0 / "labels_binary.csv")
    pos = b.index[b[UID].astype(str).isin(gold_ids) & (b["Fracture"] == 1)]
    b.loc[pos[3:], "Fracture"] = 0.0
    b.to_csv(n0 / "labels_binary.csv", index=False)
    _refresh_n0_record(inp)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    s = n12.run(root=inp, n_boot=20, quiet=True)
    assert s["errors"] == {}
    ref = Ref(inp)
    assert len(ref.gold_r) == len(ref.gold) - 5
    g = s["gold_paired_check"]
    assert g["gold_studies_without_report_labels"] == 5 and g["studies_without_report_labels_left_out"] is True
    assert g["studies"] == agg.count(len(ref.gold_r))
    both = [j for j in range(12) if _enough(ref.GY[ref.gold_r, j]) and _enough(ref.Y[ref.gold_r, j])]
    assert "Fracture" not in g["findings"] and g["findings"] == [FINDINGS[j] for j in both]
    assert g["findings_same_as_n9_s3"] is False and g["n9_findings"] == FINDINGS
    for e in ENCODERS:
        v = g["encoders"][e]
        assert v["report_labels"] == pytest.approx(_macro_pooled(ref.P[e], ref.Y, ref.gold_r, both), abs=6e-5)
        assert v["expert_labels"] == pytest.approx(_macro_pooled(ref.P[e], ref.GY, ref.gold_r, both), abs=6e-5)
        # S1: on N9's own studies (with the five) and N9's twelve findings: N9's s3_gold number
        assert v["expert_labels_on_n9_findings"] == pytest.approx(_macro_pooled(ref.P[e], ref.GY, ref.gold, range(12)), abs=6e-5)
        assert v["expert_labels_on_n9_findings"] != v["expert_labels"]
    for key in ("gold_label_quality", "fm5b_attenuation_check"):
        assert s[key]["gold_studies_without_report_labels"] == 5 and s[key]["studies_without_report_labels_left_out"] is True
    # item 2 uses the same studies: the rates come from the gold studies with a report label only
    acl = s["gold_label_quality"]["per_finding"]["ACL"]
    want_ppv, want_npv = _rates(_cells(ref.GY[ref.gold_r, 0], ref.Y[ref.gold_r, 0]))
    assert (acl["ppv"] is None) if want_ppv is None else acl["ppv"] == agg.share(want_ppv)
    assert (acl["npv"] is None) if want_npv is None else acl["npv"] == agg.share(want_npv)


def test_n12_a_failed_final_write_keeps_the_record(env, tmp_path, monkeypatch):
    """Review r1, M7: if the full agg_n12.json cannot be written, the file keeps the run's frame and says why."""
    from knee.nb import n12

    original = agg.write

    def flaky(name, payload, folder=None):
        if name == "n12" and "gold_paired_check" in payload:
            raise RuntimeError("boom")
        return original(name, payload, folder)

    monkeypatch.setattr(agg, "write", flaky)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    s = n12.run(root=env["inp"], n_boot=20, quiet=True)
    saved = json.loads((paths.WORKING / "agg_n12.json").read_text())
    assert saved["status"] == "final write failed: the blocks are not in this file"
    assert saved == json.loads(json.dumps(s, default=agg._json_default))
    assert "RuntimeError" in saved["errors"]["final_write"] and "boom" not in saved["errors"]["final_write"]
    assert saved["chain"]["s2_folds_match_units"] is True and "gold_paired_check" not in saved


def test_n12_chain_mismatch_fails_the_step(env, tmp_path, monkeypatch):
    from knee import safe_log
    from knee.nb import n12

    inp = _copy_input(env, tmp_path)
    fp = inp / "n3" / "private" / "n3" / "folds.csv"
    f = pd.read_csv(fp)
    usable = pd.read_csv(inp / "n5" / "private" / "n5" / "index.csv").set_index(UID)["usable"].reindex(f[UID]).to_numpy()
    i = int(np.flatnonzero((f["train"] & (f["s2_fold"] == 0)).to_numpy() & usable)[0])
    f.loc[i, "s2_fold"] = 1  # one training study moved to another S2 fold
    f.to_csv(fp, index=False)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    with pytest.raises(safe_log.StepFailed, match="ChainError"):
        n12.run(root=inp, n_boot=20, quiet=True)
    saved = json.loads((paths.WORKING / "agg_n12.json").read_text())
    assert saved["status"] == "chain mismatch" and saved["chain"]["s2_folds_match_units"] is False
    assert not {"gold_paired_check", "per_finding_table", "development"} & set(saved)


def test_n12_cli_entry_matches_the_runbook_card():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_notebooks", root / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (root / "notebooks" / "RUNBOOK.md").read_text().split("## Error analysis N12 (plan r2)")[1]
    assert mk.cell2("n12") in card
    meta = mk.metadata("u", "n12")
    assert meta["kernel_sources"] == [f"u/knee-{x}" for x in ("n0", "n3-r", "n5", "n6a", "n6b-g", "n7-b")]
    assert meta["enable_gpu"] is False and meta["enable_internet"] is True and meta["competition_sources"] == []
    for x in ("knee-n0", "knee-n3-r", "knee-n5", "knee-n6a", "knee-n6b-g", "knee-n7-b"):
        assert x in card
