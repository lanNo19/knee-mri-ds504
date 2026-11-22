"""Metrics, decision intervals, heads and the N6p pilot on synthetic stage data (CPU)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from knee import intervals, metrics, paths

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- metrics

def test_auc_matches_sklearn_with_ties_and_weights():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 300).astype(float)  # many ties
    y = (rng.random(300) < 0.4).astype(float)
    assert metrics.auc(s, y)[0] == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 3, 300)
    rep = np.repeat(np.arange(300), w)  # integer weights = duplicated studies
    assert metrics.auc(s, y, w)[0] == pytest.approx(roc_auc_score(y[rep], s[rep]))
    y2 = y.copy()
    y2[:10] = np.nan  # uncertain labels are left out
    m = ~np.isnan(y2)
    assert metrics.auc(s, y2)[0] == pytest.approx(roc_auc_score(y2[m], s[m]))


def test_stratified_auc_ignores_offsets_between_folds():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(float)
    folds = np.repeat([0, 1], 200)
    s = y + rng.normal(0, 1, 400)
    a0, _ = metrics.stratified_auc(s, y, folds)
    s2 = s + np.where(folds == 1, 5.0, 0.0)  # a calibration offset in fold 1 only
    a1, _ = metrics.stratified_auc(s2, y, folds)
    assert a0 == pytest.approx(a1)
    assert metrics.auc(s2, y)[0] != pytest.approx(a0)  # the pooled AUC moves


def test_macro_is_nan_when_a_fixed_finding_is_undefined():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], float)
    s = np.random.default_rng(0).random((4, 2))
    assert np.isnan(metrics.macro(s, y, np.zeros(4, int), [0, 1]))
    assert np.isfinite(metrics.macro(s, y, np.zeros(4, int), [0]))


# ---------------------------------------------------------------- intervals

def test_intervals_refuse_studies_without_a_group():
    with pytest.raises(ValueError):
        intervals.cluster_bootstrap(lambda w: 0.0, np.array(["A", None, "B"], dtype=object), n_boot=5)


def test_scoring_rule_counts_uncertain_as_negative():
    from knee import heads

    b = np.array([[1.0, np.nan], [0.0, 1.0]])
    assert heads.scoring_labels(b).tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_decision_rule_order():
    assert intervals.decide(-0.01, 0.015) == "equivalent"
    assert intervals.decide(0.001, 0.05) == "A better"
    assert intervals.decide(-0.05, -0.001) == "B better"
    assert intervals.decide(-0.03, 0.04) == "inconclusive"
    assert intervals.decide(None, None) == "not testable"
    assert intervals.combine_groupings("A better", "A better") == "A better"
    assert intervals.combine_groupings("A better", "inconclusive") == "inconclusive (grouping-sensitive)"


def test_interval_covers_a_known_difference():
    rng = np.random.default_rng(2)
    n, G = 1200, 10
    groups = rng.integers(0, G, n)
    y = (rng.random((n, 3)) < 0.3).astype(float)
    folds = groups % 5
    a = y + rng.normal(0, 0.8, (n, 3))  # better
    b = y + rng.normal(0, 1.2, (n, 3))

    def delta(w):
        return metrics.macro(a, y, folds, [0, 1, 2], w) - metrics.macro(b, y, folds, [0, 1, 2], w)

    di = intervals.decision_interval(delta, groups, n_boot=300, seed=0)
    assert di["lo"] <= di["estimate"] <= di["hi"]
    assert di["lo"] > 0  # the true difference is clearly positive
    assert di["hi"] - di["lo"] >= di["bootstrap"]["hi"] - di["bootstrap"]["lo"]
    assert di["n_groups"] == G and not di["approximate"]


# ---------------------------------------------------------------- heads

def _bank(n=300, D=16, S=4, seed=0):
    from knee import heads

    rng = np.random.default_rng(seed)
    y = (rng.random((n, 12)) < 0.3).astype(float)
    W = rng.normal(0, 1, (12, D))
    emb = {p: ((y @ W)[:, None, :] + rng.normal(0, 1, (n, S, D))).astype(np.float16) for p in heads.PLANES}
    return heads.Bank(emb, y.astype(np.float32), np.ones_like(y, np.float32), y), y


@pytest.mark.parametrize("pooling", ["mean", "max", "gated"])
def test_heads_learn_and_respect_the_leakage_guard(pooling):
    from knee import heads

    bank, y = _bank()
    cfg = heads.Config(pooling=pooling, planes="all", batch_size=32, lr=3e-3)
    tr, va, te = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)
    r = heads.fit(bank, cfg, tr, va, list(range(12)), forbidden=te, max_epochs=15)
    p = heads.predict(bank, r.model, te, cfg, bank.stats(tr))
    assert metrics.macro(p, y[te], np.zeros(len(te), int), list(range(12))) > 0.75
    with pytest.raises(heads.LeakageError):
        heads.fit(bank, cfg, np.arange(0, 260), va, forbidden=te)
    with pytest.raises(heads.LeakageError):
        heads.fit(bank, cfg, tr, np.arange(190, 250), forbidden=te)  # validation overlaps training


def test_fixed_steps_fit_follows_the_rsna_schedule():
    import math

    from knee import heads

    bank, _ = _bank(n=80)
    cfg = heads.Config(pooling="mean", batch_size=16, lr=1e-3)  # 5 steps per epoch on this data
    r = heads.fit(bank, cfg, np.arange(80), steps=60, schedule_steps_per_epoch=20)
    assert r.steps_run == 60 and r.epochs_run == 12  # 12 small epochs = 3 RSNA epochs of 20 steps
    warm, total = 20, heads.MAX_EPOCHS * 20
    want = 1e-3 * 0.5 * (1 + math.cos(math.pi * (59 - warm) / (total - warm)))
    assert r.final_lr == pytest.approx(want, rel=1e-6)  # the RSNA refit's learning rate at its last step


def test_rerun_rule_and_output_search(tmp_path):
    from knee import paths as kp
    from knee.nb.common import is_rerun

    assert is_rerun("/kaggle/input/knee-n6a-r/private/n6a/units/x_f0")
    assert is_rerun("/w/private/n6a_r/units/x_f0")
    assert not is_rerun("/kaggle/input/notebooks/mr-r/knee-n6a/private/n6a/units/x_f0")  # owner folders are ignored
    for d in ("knee-n6a/private/n6a/units/a_f0", "notebooks/o/knee-n6b/private/n6b/units/b_f1",
              f"{kp.COMPETITION}/private/x/units/c_f2"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "config.json").write_text("{}")
    found = [p.parent.name for p in kp.glob_outputs("private/*/units/*_f*/config.json", tmp_path)]
    assert found == ["a_f0", "b_f1"]  # the competition folder is never searched


def test_refit_runs_exactly_e_epochs():
    from knee import heads

    bank, _ = _bank()
    r = heads.fit(bank, heads.Config(pooling="mean"), np.arange(200), epochs=3)
    assert r.epochs_run == 3 and not r.history


# ---------------------------------------------------------------- N6p pilot

@pytest.fixture(scope="module")
def stage_env(tmp_path_factory):
    from tests.synth_stage import make_stage

    root = tmp_path_factory.mktemp("stage")
    facts = make_stage(root / "input")
    return root, facts


def test_n6p_pilot(stage_env, monkeypatch, capfd):
    from knee.nb import n6p

    root, facts = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working")
    s = n6p.run(root=root / "input", device="cpu", n_boot=200, quiet=True)
    printed = capfd.readouterr()
    text = printed.out + printed.err + (paths.WORKING / "agg_pilot.json").read_text()
    assert not [u for u in facts["uids"] if u in text]
    assert s["findings"] == list(__import__("knee.labels", fromlist=["FINDINGS"]).FINDINGS)  # all 12
    pc = s["positive_control"]
    assert pc["all_passed"] and set(pc) == {"min", "passed", "all_passed", "lowest_s2_macro_auc"}
    assert (paths.WORKING / "private" / "n6p" / "positive_control.json").exists()  # values stay private
    hw = s["half_width_delta1"]
    assert hw["union"] >= max(hw["bootstrap"], hw["jackknife"]) - 1e-9
    assert "estimate" not in hw and "lo" not in hw  # the pilot shows no direction for H1
    oc = s["overlap_control"]
    assert abs(oc["null_macro_auc"] - 0.5) < 0.05
    assert s["leakage_assertion_passed"]
    pairs = pd.read_csv(paths.WORKING / "private" / "n6p" / "pairs.csv")
    got = {frozenset(x) for x in zip(pairs["uid_a"], pairs["uid_b"])}
    assert all(frozenset(p) in got for p in facts["dup_pairs"])
    wall = s["projection"]["n6a_wall_hours"]
    assert 0 < wall["low"] <= wall["high"]
    json.loads((paths.WORKING / "agg_pilot.json").read_text())


# ---------------------------------------------------------------- N8 floor A

def test_n8_floor_a(stage_env, monkeypatch, capfd):
    pytest.importorskip("lightgbm")
    from knee.nb import n8

    root, facts = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_n8")
    s = n8.run(root=root / "input", n_trials=3, workers=2, quiet=True)
    printed = capfd.readouterr()
    text = printed.out + printed.err + (paths.WORKING / "agg_n8.json").read_text()
    assert not [u for u in facts["uids"] if u in text]
    assert s["leakage_assertion_passed"] and s["n_folds"] == 5
    assert 0.5 < s["s2_macro_auc"] < 0.7  # only one tag carries label signal that transfers to unseen sites
    pred = pd.read_csv(paths.WORKING / "private" / "n8" / "predictions.csv")
    lab = pd.read_csv(root / "input" / "n0" / "private" / "n0" / "labels_binary.csv").set_index("StudyInstanceUID")
    te = pred[pred["s2_fold"] >= 0]
    acl, _ = metrics.stratified_auc(te["ACL"].to_numpy(), lab.loc[te["StudyInstanceUID"], "ACL"].to_numpy(),
                                    te["s2_fold"].to_numpy())
    assert acl > 0.65
    st = pd.read_csv(root / "input" / "n0" / "private" / "n0" / "studies.csv")
    gold = set(st.loc[st["is_gold"], "StudyInstanceUID"])
    assert gold & set(pred["StudyInstanceUID"])  # gold studies are scored by their fold's model
    trials = pd.read_csv(paths.WORKING / "private" / "n8" / "trials.csv")
    assert len(trials) == 5 * 3


# ---------------------------------------------------------------- N6 Stage A

def test_n6_stage_a_units_and_demo(stage_env, monkeypatch, capfd):
    pytest.importorskip("optuna")
    from knee.nb import n6

    root, facts = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_n6")
    s = n6.run("n6a", encoders=("dinov2_vits14",), folds=(0, 1), demo=True, root=root / "input", workers=2,
               n_trials=3, l1_grid=[("mean", "sag"), ("gated", "all")], demo_epochs=60, quiet=True)
    # (slow on 2 CPU cores: two units, the L1 grid and 30 trials are cut down; the logic is the same)
    printed = capfd.readouterr()
    agg_files = list(paths.WORKING.glob("agg_*.json"))
    text = printed.out + printed.err + "".join(f.read_text() for f in agg_files)
    assert not [u for u in facts["uids"] if u in text]
    assert s["all_units_ok"], s["units"]
    assert s["leakage_assertion_passed"]
    assert {f.name for f in agg_files} >= {"agg_n6a.json", "agg_n6a_dinov2_vits14_f0.json", "agg_n6a_dinov2_vits14_f1.json"}
    folds = pd.read_csv(root / "input" / "n3" / "private" / "n3" / "folds.csv").set_index("StudyInstanceUID")
    for k in (0, 1):
        ud = paths.WORKING / "private" / "n6a" / "units" / f"dinov2_vits14_f{k}"
        pred = pd.read_csv(ud / "predictions.csv").set_index("StudyInstanceUID")
        f = folds.loc[pred.index]
        # only the test fold and the gold studies of that fold are predicted by this unit
        assert ((f["s2_fold"] == k) & (f["train"] | f["is_gold"])).all()
        assert (ud / "head.pt").exists() and (ud / "trials.csv").exists()
        trials = pd.read_csv(ud / "trials.csv")
        assert len(trials) == 3 and trials.loc[0, "lr"] == pytest.approx(3e-4)  # trial 0 = the defaults
    demo = s["demo"]
    assert set(demo["curves"]) == {"random", "successive_halving", "gp"}


# ---------------------------------------------------------------- N7 and N9

def test_n7_consensus_and_n9_decisions(stage_env, monkeypatch, capfd):
    from knee.nb import n7, n9
    from tests.synth_stage import make_units

    root, facts = stage_env
    make_units(root / "input")
    monkeypatch.setattr(paths, "WORKING", root / "input" / "n7out")  # N9 then finds N7's csvs under input
    s7 = n7.run(root=root / "input", device="cpu", quiet=True)
    con = s7["consensus"]["dinov2_vits14"]
    assert con["config"]["pooling"] == "gated" and con["E"] == 6  # gated won 3 of 5 folds
    assert not any(v["folds"] for v in s7["missing_units"].values())
    monkeypatch.setattr(paths, "WORKING", root / "working_n9")
    s9 = n9.run(root=root / "input", n_boot=200, hidden={"dinov2_vits14": 0.9, "convnext_tiny": 0.85}, quiet=True)
    printed = capfd.readouterr()
    text = printed.out + printed.err + (paths.WORKING / "agg_results.json").read_text()
    assert not [u for u in facts["uids"] if u in text]
    assert s9["main_complete"]
    assert s9["h1"]["g_lang"]["category"] == "DINOv2 better"
    assert s9["h1"]["decision"] in ("DINOv2 better", "inconclusive (grouping-sensitive)")
    assert s9["h2"]["category"] in ("supported", "opposite", "equivalent", "inconclusive")
    assert s9["h3"]["supported"] is True and s9["decisions"]["H3"] == "supported"
    assert s9["s3_gold"]["consistent"] and s9["s4_hidden"]["consistent"]
    assert s9["anti_claim_1"]["d_pre"]["estimate"] > 0
    assert s9["decisions"]["H4"].startswith("not run")



def test_n6_backstop_switches_both_encoders_of_a_fold_together(stage_env, monkeypatch):
    pytest.importorskip("optuna")
    from knee.nb import n6

    root, _ = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_n6b")
    # a backstop already past: every fold is projected late, so each runs simplified for both encoders
    s = n6.run("n6a", encoders=("dinov2_vits14", "convnext_tiny"), folds=(0, 1), root=root / "input", workers=2,
               backstop_utc="2000-01-01T00:00:00Z", unit_hours_est=1.0, n_trials=2,
               l1_grid=[("mean", "sag"), ("gated", "all")], quiet=True)
    assert s["all_units_ok"], s["units"]
    assert s["fold_modes"] == {"fold_0": "simplified", "fold_1": "simplified"}
    assert not s["all_nested"]
    for k in (0, 1):
        tunings = {s["units"][f"{e}_f{k}"]["tuning"] for e in ("dinov2_vits14", "convnext_tiny")}
        assert tunings == {"simplified (not nested)"}
    cfg1 = json.loads((paths.WORKING / "private" / "n6a" / "units" / "dinov2_vits14_f1" / "config.json").read_text())
    assert cfg1["l2_from"] == "fold 0"  # BO ran on fold 0 only; fold 1 reuses its L2 settings
