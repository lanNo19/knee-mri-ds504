"""N12, the error analysis of plan r2 (section 4), on synthetic stage data (CPU).

Every block is present and labelled; the privacy scan passes; items 1 to 5 equal an independent
recomputation (sklearn AUCs, pandas counts); item 3 equals N9's own s2.<encoder>.per_finding and item
1's expert-label macro equals N9's s3_gold; a failing block costs only itself; A4 is skipped, with
the reason, when the positive weights cannot be confirmed; a chain mismatch fails the step."""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest

from knee import agg, paths
from knee.labels import FINDINGS, UID

sk = pytest.importorskip("sklearn.metrics")
MAIN = ("dinov2_vits14", "convnext_tiny")
ENCODERS = ("dinov2_vits14", "convnext_tiny", "vit_s16")
N_BOOT = 100


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
        self.gold = np.flatnonzero(self.gold_flag & both)


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


def _ints(o, key=None):
    if isinstance(o, dict):
        return [x for k, v in o.items() for x in _ints(v, k)]
    if isinstance(o, list):
        return [x for v in o for x in _ints(v, key)]
    return [(key, o)] if isinstance(o, int) and not isinstance(o, bool) else []


def _lists(o):
    if isinstance(o, dict):
        return [x for v in o.values() for x in _lists(v)]
    if isinstance(o, list):
        return [len(o)] + [x for v in o for x in _lists(v)]
    return []


# ---------------------------------------------------------------- the main run

def test_n12_blocks_privacy_and_independent_recomputation(env, monkeypatch, capfd):
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
    for key in ("gold_paired_check", "gold_label_quality", "per_finding_table", "english_group_by_language", "development"):
        b = saved[key]
        assert b["descriptive"] is True and b["added_after_results"] is True and len(b["scope"]) > 20, key
        assert "status" not in b, key
    for key in ("a2_seed_ensemble", "a3_encoder_ensemble", "a4_calibration"):
        b = saved["development"][key]
        assert b["descriptive"] is True and b["added_after_results"] is True and len(b["scope"]) > 20, key
        assert b["label"] == "development: these studies were test studies in iteration 1; likely optimistic"
    assert saved["development"]["label"] == saved["development"]["a2_seed_ensemble"]["label"]
    assert saved["descriptive"] is True and saved["added_after_results"] is True and "decides nothing" in saved["scope"]
    assert set(saved["not_run"]) == {"label_agreement_strata", "g_vendor_strata", "missing_plane_fallback",
                                     "per_finding_x_language", "laterality"}
    assert saved["upstream"]["n0_studies"]["sha16"].startswith(env["expect"]["n0_studies"])

    # privacy: no identifier anywhere, the summary checks pass, no per-study list, nothing private written
    uids = env["facts"]["uids"]
    assert not [u for u in uids if u in printed.out + printed.err + text]
    agg._check(saved)
    assert max(_lists(saved)) <= len(FINDINGS)
    assert all(v == 0 or v >= 5 or k in agg.STRUCTURAL for k, v in _ints(saved))
    assert sorted(p.name for p in paths.WORKING.iterdir() if p.is_file()) == ["agg_n12.json", "run.json"]
    assert not [p for p in (paths.WORKING / "private").rglob("*") if p.is_file()]
    from knee import safe_log

    block = json.loads(printed.out.split(safe_log.BEGIN)[-1].split(safe_log.END)[0])
    agg._check(block)
    assert block["status"] == "ok" and block["chain_ok"] is True and block["errors"] == {}
    assert block["development"]["label"].startswith("development:")

    ref = Ref(inp)
    assert len(ref.ev) > 500 and s["evaluation_studies"] == agg.count(len(ref.ev)) == s9["evaluation_studies"]
    _item1(s, s9, ref)
    _item2(s, ref)
    _item3(s, s9, ref, env)
    _item4(s, ref)
    _item5(s, s9, ref, inp)


def _item1(s, s9, ref):
    g = s["gold_paired_check"]
    gold = ref.gold
    n = len(gold)
    assert 50 <= n <= 58 and g["studies"] == agg.count(n) and g["studies_text"] == f"about {agg.count(n)}"
    both = [j for j in range(12) if _enough(ref.GY[gold, j]) and _enough(ref.Y[gold, j])]
    assert g["findings"] == [FINDINGS[j] for j in both] and len(both) >= 6
    assert set(g["findings_left_out"]) == set(FINDINGS) - set(g["findings"])
    assert g["findings_same_as_n9_s3"] == (g["findings"] == s9["s3_gold"]["findings"])
    assert set(g["encoders"]) == set(ENCODERS) and g["n_boot"] == N_BOOT
    rng = np.random.default_rng(0)  # the same draws: n studies with replacement, as weights
    draws = [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(N_BOOT)]
    for e in ENCODERS:
        P = ref.P[e]
        ex, rp = _macro_pooled(P, ref.GY, gold, both), _macro_pooled(P, ref.Y, gold, both)
        v = g["encoders"][e]
        assert v["expert_labels"] == pytest.approx(ex, abs=6e-5) and v["report_labels"] == pytest.approx(rp, abs=6e-5)
        assert v["expert_minus_report"]["estimate"] == pytest.approx(ex - rp, abs=1.1e-4)
        if g["findings_same_as_n9_s3"]:  # the expert-label macro is N9's S3 number, digit for digit
            assert v["expert_labels"] == s9["s3_gold"]["macro_auc"][e]
        exd = np.array([_macro_pooled(P, ref.GY, gold, both, w) for w in draws])
        rpd = np.array([_macro_pooled(P, ref.Y, gold, both, w) for w in draws])
        for got, vals in ((v["expert_minus_report"], exd - rpd), (v["expert_labels_interval"], exd),
                          (v["report_labels_interval"], rpd)):
            ok = np.isfinite(vals)
            lo, hi = np.percentile(vals[ok], [2.5, 97.5])
            assert got["lo"] == pytest.approx(lo, abs=1.1e-4) and got["hi"] == pytest.approx(hi, abs=1.1e-4)
            assert got["valid_share"] == pytest.approx(ok.mean(), abs=1e-3) and got["hi"] > got["lo"]
        iv = v["expert_minus_report"]
        assert iv["lo"] <= iv["estimate"] <= iv["hi"]
        assert v["expert_labels_interval"]["lo"] <= v["expert_labels"] <= v["expert_labels_interval"]["hi"]


def _label_quality(ref, draws=()):
    gold, out = ref.gold, {}
    for j, f in enumerate(FINDINGS):
        ye = ref.GY[gold, j]
        if not _enough(ye):
            continue
        rb = ref.Y[gold, j]
        out[f] = {"soft": _auc(ye, ref.soft[gold, j])[0],
                  "soft_draws": np.array([_auc(ye, ref.soft[gold, j], w)[0] for w in draws]),
                  "ppv": (ye[rb == 1] == 1).mean() if (rb == 1).sum() >= 5 else None,
                  "npv": (ye[rb == 0] == 0).mean() if (rb == 0).sum() >= 5 else None}
    return out


def _item2(s, ref):
    q = s["gold_label_quality"]
    n = len(ref.gold)
    rng = np.random.default_rng(0)  # item 1's draws
    draws = [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(N_BOOT)]
    want = _label_quality(ref, draws)
    assert set(q["per_finding"]) == set(FINDINGS) and q["n_boot"] == N_BOOT
    assert q["findings_suppressed"] == [f for f in FINDINGS if f not in want]
    for f, w in want.items():
        v = q["per_finding"][f]
        assert v["soft_label_auc"] == pytest.approx(w["soft"], abs=6e-4)
        ok = np.isfinite(w["soft_draws"])
        lo, hi = np.percentile(w["soft_draws"][ok], [2.5, 97.5])
        iv = v["soft_label_auc_interval"]
        assert iv["lo"] == pytest.approx(lo, abs=6e-4) and iv["hi"] == pytest.approx(hi, abs=6e-4)
        assert iv["lo"] <= v["soft_label_auc"] <= iv["hi"] and iv["valid_share"] == pytest.approx(ok.mean(), abs=1e-3)
        for key in ("ppv", "npv"):
            assert (v[key] is None) if w[key] is None else v[key] == pytest.approx(w[key], abs=5.1e-3)
        if w["ppv"] is not None and w["npv"] is not None:
            assert v["ppv_plus_npv_minus_1"] == pytest.approx(w["ppv"] + w["npv"] - 1, abs=5.1e-3)
    assert sum(v.get("ppv") is not None for v in q["per_finding"].values()) >= 6  # the comparison is not empty
    assert q["soft_label_macro_auc"] == pytest.approx(np.mean([w["soft"] for w in want.values()]), abs=6e-4)


def _item3(s, s9, ref, env):
    from scipy.stats import spearmanr

    t = s["per_finding_table"]
    assert list(t["rows"]) == FINDINGS and t["prevalence_source"] == "agg_n0.json labels.prevalence_training"
    ev = ref.ev
    lq = _label_quality(ref)
    for e in ENCODERS:
        auc = {}
        for j, f in enumerate(FINDINGS):
            auc[f] = _fold_stratified(ref.P[e][ev, j], ref.Y[ev, j], ref.f2[ev])
            got = t["rows"][f]["s2_auc"][e]
            assert got == pytest.approx(auc[f], abs=6e-4)
            assert got == s9["s2"][e]["per_finding"][f]  # the very numbers N9 writes
        used = [f for f in FINDINGS if f in lq]
        rho = spearmanr([lq[f]["soft"] for f in used], [auc[f] for f in used]).statistic
        assert t["spearman_label_quality_vs_s2_auc"]["encoders"][e]["rho"] == pytest.approx(rho, abs=1.1e-3)
        assert t["spearman_label_quality_vs_s2_auc"]["findings_used"] == used
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
    rows_all = grp[ref.f2[grp] == k]
    sizes = pd.Series(ref.lang[rows_all]).value_counts()
    reported = [L for L, n in sizes.items() if n >= 50]
    assert g["languages_reported"] == reported and g["languages_suppressed"] == sorted(L for L in sizes.index if L not in reported)
    assert {"ENGLISH", "SPANISH"} <= set(reported) and g["languages_suppressed"]  # both paths are exercised here
    strata = {L: rows_all[ref.lang[rows_all] == L] for L in reported}
    find = [j for j in range(12) if all(_enough(ref.Y[r, j]) for r in strata.values())]
    assert g["findings"] == [FINDINGS[j] for j in find] and find
    assert g["findings_left_out"] == {FINDINGS[j]: [L for L, r in strata.items() if not _enough(ref.Y[r, j])]
                                      for j in range(12) if j not in find}
    assert set(g["rows"]) == set(reported)
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

    # A2: the mean probability over N7's six seeds
    a2 = dev["a2_seed_ensemble"]
    assert set(a2["encoders"]) == set(MAIN) and set(a2["not_available"]) == {"vit_s16"}
    for e in MAIN:
        df = pd.read_csv(inp / "n7" / "private" / "n7" / f"s2seeds_{e}.csv")
        M = np.mean([df[df["seed"] == sd].set_index(UID).reindex(ref.uids)[FINDINGS].to_numpy(float) for sd in range(6)], axis=0)
        fs, mf = levels(M, ev)
        v = a2["encoders"][e]
        assert v["seeds"] == [0, 1, 2, 3, 4, 5]
        assert v["fold_stratified_macro_auc"] == pytest.approx(fs, abs=6e-5) and v["mean_of_fold_macros"] == pytest.approx(mf, abs=6e-5)
    # A3: the equal-weight mean of the three encoders
    a3 = dev["a3_encoder_ensemble"]
    M = np.mean([ref.P[e] for e in ENCODERS], axis=0)
    fs, mf = levels(M, ev)
    assert a3["studies"] == agg.count(len(ev)) and a3["share_of_evaluation_studies"] == 1.0
    assert a3["fold_stratified_macro_auc"] == pytest.approx(fs, abs=6e-5) and a3["mean_of_fold_macros"] == pytest.approx(mf, abs=6e-5)
    # A4: positive weights from the outer training labels and each unit's cap; logits shifted by -log(w)
    a4 = dev["a4_calibration"]
    assert set(a4["encoders"]) == set(ENCODERS) and a4["skipped"] == {}
    pos_any = (ref.Y == 1).any(axis=1).astype(float)
    for e in ENCODERS:
        v = a4["encoders"][e]
        assert all(x is True for x in v["checks"].values()), v["checks"]
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


def test_n12_english_group_rows_share_one_finding_list():
    """Item 4's rules on a hand-made group: found by its English reports, not by fold number; a row under
    50 studies is suppressed; a small row that passes can take a rare finding out of every row, and the
    block names that row."""
    from types import SimpleNamespace

    from knee.nb import n12

    rng = np.random.default_rng(3)
    lang = np.array(["ENGLISH"] * 250 + ["SPANISH"] * 150 + ["LATIN"] * 60 + ["FRENCH"] * 40 + ["CROATIAN"] * 200)
    n = len(lang)
    g_lang = np.where(lang == "CROATIAN", "CROATIAN", "ENGLISH")
    f2 = np.where(g_lang == "ENGLISH", 2, 0)  # the English group sits in S2 fold 2
    Y = (rng.random((n, 12)) < 0.3).astype(float)
    latin = np.flatnonzero(lang == "LATIN")
    Y[latin, 11] = 0.0
    Y[latin[:3], 11] = 1.0  # Fracture: 3 positives among the Latin reports
    units = {e: np.clip(Y * 0.3 + rng.random((n, 12)) * 0.7, 0, 1) for e in MAIN}
    c = SimpleNamespace(ev=np.arange(n), Y=Y, f2=f2, elig=list(range(12)), units=units, d=SimpleNamespace(g_lang=g_lang))
    out = n12.english_group_by_language(c, lang, n_boot=50, seed=0)
    assert out["group"] == "ENGLISH" and out["fold"] == 2 and out["group_in_one_fold"] is True
    assert out["languages_reported"] == ["ENGLISH", "SPANISH", "LATIN"] and out["languages_suppressed"] == ["FRENCH"]
    assert out["findings_left_out"] == {"Fracture": ["LATIN"]} and out["findings"] == FINDINGS[:11]
    for L in ("ENGLISH", "SPANISH", "LATIN"):
        r = np.flatnonzero(lang == L)
        want = np.mean([sk.roc_auc_score(Y[r, j], units["convnext_tiny"][r, j]) for j in range(11)])
        assert out["rows"][L]["encoders"]["convnext_tiny"]["macro_auc"] == pytest.approx(want, abs=6e-5)
    assert out["reference_whole_group"]["studies"] == agg.count(500) and set(out["rows"]) == {"ENGLISH", "SPANISH", "LATIN"}
    agg._check(json.loads(json.dumps(out)))


# ---------------------------------------------------------------- faults

def test_n12_a_failing_block_costs_only_itself_and_a4_says_why_it_is_skipped(env, tmp_path, monkeypatch):
    from knee.nb import n12

    inp = tmp_path / "input"
    shutil.copytree(env["inp"], inp)
    # one unit's refit_steps disagree with the training-set size: A4 is skipped for that encoder only
    cp = inp / "n6a" / "private" / "n6a" / "units" / "vit_s16_f2" / "config.json"
    u = json.loads(cp.read_text())
    u["refit_steps"] += 1
    cp.write_text(json.dumps(u))
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w1")
    monkeypatch.setattr(n12, "gold_label_quality", lambda c, n_boot, seed: 1 / 0)
    monkeypatch.setattr(n12, "a3_encoder_ensemble", lambda c: {"descriptive": True, "studies": 3})  # an unsafe count
    s = n12.run(root=inp, n_boot=20, quiet=True)
    saved = json.loads((paths.WORKING / "agg_n12.json").read_text())
    assert set(saved["errors"]) == {"gold_label_quality", "development.a3_encoder_ensemble"}
    assert "ZeroDivisionError" in saved["errors"]["gold_label_quality"]
    assert "UnsafeSummary" in saved["errors"]["development.a3_encoder_ensemble"]
    assert "gold_label_quality" not in saved and "a3_encoder_ensemble" not in saved["development"]
    assert s["status"].startswith("ok, some blocks failed")
    for key in ("gold_paired_check", "per_finding_table", "english_group_by_language"):
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


def test_n12_chain_mismatch_fails_the_step(env, tmp_path, monkeypatch):
    from knee import safe_log
    from knee.nb import n12

    inp = tmp_path / "input"
    shutil.copytree(env["inp"], inp)
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
