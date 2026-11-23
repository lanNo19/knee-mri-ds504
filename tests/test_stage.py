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
    # the corrected projection (code review finding 1): old keys kept, corrected ones beside them
    cor = s["projection"]["corrected"]
    assert cor["wall_divisor"] == 2.0 and cor["batch_factor"] == 1.26
    assert all(v >= 1.0 for v in cor["epoch_factor"].values())
    assert len(cor["inner_fit_size_factor"]["factor"]) == 5 and min(cor["inner_fit_size_factor"]["factor"]) > 0
    assert cor["n6a_wall_hours"] >= 1.2 * wall["high"]  # 30 epochs, x1.26, divided by 2 only
    assert isinstance(cor["n6a_over_6h"], bool) and "n6a_over_6h_gate" in s["projection"]
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
    ms = s["features"]["missing_share"]
    assert set(ms) == set(n8.CATEGORICAL + n8.NUMERIC) and ms["manufacturer"] == 0.0 and 0 < ms["sagittal_echo_time"] < 0.2
    assert s["features"]["columns_absent"] == []
    tags = pd.read_csv(root / "input" / "n0" / "private" / "n0" / "tags.csv")
    X = n8.features(tags.drop(columns=["model"]), tags["StudyInstanceUID"].astype(str))
    assert (X["model"].astype(str) == "MISSING").all()  # an absent column is all-missing: now visible in the summary
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

    # N7's seed refit is Stage A's refit: with seed 0 it gives the unit's stored predictions back
    from knee import heads, stage, units
    from knee.nb import n7

    d = stage.StageData(root / "input")
    ud = paths.WORKING / "private" / "n6a" / "units" / "dinov2_vits14_f0"
    u = json.loads((ud / "config.json").read_text())
    bank = d.bank("dinov2_vits14", "cpu")
    spec = {0: (heads.config_from(u["config"]), int(u["E"]))}
    P = n7.s2_refits(d, bank, spec.get, seed=0)
    pred = pd.read_csv(ud / "predictions.csv")
    rows = pred["StudyInstanceUID"].astype(str).map({x: i for i, x in enumerate(d.uids)}).to_numpy()
    FIND = [c for c in pred.columns if c not in ("StudyInstanceUID", "s2_fold", "is_gold")]
    assert np.isfinite(P[rows]).all() and np.isnan(np.delete(P, rows, axis=0)).all()  # fold 0 and its gold only
    assert np.abs(P[rows] - pred[FIND].to_numpy()).max() < 1e-4
    assert np.abs(n7.s2_refits(d, bank, spec.get, seed=1)[rows] - pred[FIND].to_numpy()).max() > 1e-4  # seeds differ
    # the default-head settings come from the unit's own inner logs
    specs = units.default_specs(ud, d.pos_cap_upper())
    cfg, E, source = specs["l2_defaults"]
    t0 = pd.read_csv(ud / "trials.csv").iloc[0]
    assert source == "BO trial 0" and E == max(1, int(np.median(json.loads(t0["best_epochs"]))))
    assert (cfg.pooling, cfg.planes) == (u["l1_winner"]["pooling"], u["l1_winner"]["planes"])
    assert cfg.lr == heads.Config().lr and cfg.pos_cap == pytest.approx(np.sqrt(d.pos_cap_upper()))
    assert specs["fixed"][0] == heads.default_config(d.pos_cap_upper()) and specs["fixed"][2] == "l1.csv"
    # the chain guard accepts real units and the attached folds
    dirs = units.unit_dirs(paths.WORKING)
    assert set(dirs) == {("dinov2_vits14", 0), ("dinov2_vits14", 1)}
    ok = units.check_units(d, dirs)
    assert ok["s2_folds_match_units"] is True and ok["mismatched_units"] == [] and ok["units_checked"] == "<5"
    none = units.check_units(d, {})
    assert none["s2_folds_match_units"] is None and none["units_checked"] == 0  # nothing checked is not a pass


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
    assert s7["chain"]["s2_folds_match_units"] and s7["leakage_assertion_passed"] and s7["device"] == "cpu"
    assert s7["extras_ok"] is True and s7["extras_errors"] == {} and s7["chain"]["units_checked"] == 15
    assert s7["extras_requested"] == ["seeds", "defaults", "triage", "probe"] and s7["base_seconds"] <= s7["seconds"]
    assert {"dinov2_vits14.seeds", "vit_s16.triage", "probe"} <= set(s7["extras_seconds"])
    assert s7["extras"]["dinov2_vits14"]["seeds"] == [0, 1, 2, 3, 4, 5] and "seeds" not in s7["extras"]["vit_s16"]
    assert s7["extras"]["vit_s16"]["default_l2_defaults"]["E"]["epochs"] == [4] * 5  # BO trial 0's inner epochs
    assert s7["extras"]["vit_s16"]["default_fixed"]["E"]["epochs"] == [5] * 5  # l1.csv's row
    n7d = paths.WORKING / "private" / "n7"
    seeds = pd.read_csv(n7d / "s2seeds_convnext_tiny.csv")
    assert sorted(seeds["seed"].unique()) == [0, 1, 2, 3, 4, 5] and seeds.groupby("seed").size().nunique() == 1
    thr = pd.read_csv(n7d / "triage_thresholds.csv")
    assert len(thr) == 15 and (thr["inner_sensitivity"] >= 0.95).all() and thr["threshold"].between(0, 1).all()
    probe = s7["near_duplicate_probe"]
    assert set(probe) == {"encoder", "cosine_above_0.90", "cosine_above_0.95", "cosine_above_0.98"}
    assert probe["cosine_above_0.98"]["pairs"] == "<5"  # the three planted duplicates
    assert set(probe["cosine_above_0.90"]) == {"pairs", "across_language_groups", "across_vendor_groups",
                                               "pairs_both_training", "across_s2_folds", "across_vendor_folds",
                                               "across_s1_folds"}
    assert not [u for u in facts["uids"] if u in (paths.WORKING / "agg_n7.json").read_text()]
    # Stage A's run.json says where its units were fit (a T4 here; N7 above ran on CPU)
    (root / "input" / "n6a" / "run.json").write_text(json.dumps({"commit": "synthetic000", "gpus": ["Tesla T4", "Tesla T4"]}))
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
    _check_secondary(s9, root)


def _check_secondary(s9, root):
    """The 23 Nov additions: new keys only, descriptive, privacy-safe."""
    assert s9["secondary_errors"] == {}
    # A. chain guard
    ch = s9["chain"]
    assert ch["s2_folds_match_units"] is True and ch["mismatched_units"] == [] and ch["units_checked"] == 15
    assert ch["n3_folds_sha16"] == s9["upstream"]["n3_folds"]["sha16"] and len(ch["s2_sha16"]) == 16
    # B. per-fold Delta1
    sec = s9["h1"]["secondary"]
    assert len(sec["per_fold_delta1"]) == 5 and sum(sec["pair_share"]) == pytest.approx(1.0, abs=0.03)
    assert sec["unweighted_mean_of_fold_delta1"] == pytest.approx(np.mean(sec["per_fold_delta1"]), abs=1e-4)
    assert sec["sign_agrees_with_primary"] is True and sec["primary_delta1"] == s9["h1"]["g_lang"]["estimate"]
    old = [s9["h1"]["per_fold"][f"fold_{k}"]["delta1"] for k in range(5)]
    assert sec["per_fold_delta1"] == pytest.approx(old, abs=1e-4)  # all findings present: equals the existing view
    # C. the second interval sits beside the first and decides nothing
    gl = s9["h1"]["g_lang"]
    wb = gl["interval_within_fold_study_bootstrap"]
    assert wb["lo"] < gl["estimate"] < wb["hi"] and wb["used_for_decisions"] is False and wb["n_boot"] == 200
    assert {"estimate", "lo", "hi", "half_width", "bootstrap", "jackknife", "category"} <= set(gl)
    # D. seed sensitivity
    sd = s9["seed_sensitivity"]
    assert sd["seeds"] == [0, 1, 2, 3, 4, 5] and sd["decision_seed"] == 0
    assert sd["delta1_per_seed"]["seed_0"] == gl["estimate"] and sd["decision_per_seed"]["seed_0"] == gl["category"]
    assert len(sd["delta1_per_seed"]) == 6 and sd["delta1_sd"] > 0
    same = [sd["decision_per_seed"][f"seed_{k}"] == gl["category"] for k in range(1, 6)]
    assert sd["seed_sensitive"] == (not all(same)) and sd["share_of_extra_seeds_same_category"] == np.mean(same)
    signs = {np.sign(v) for v in sd["delta1_per_seed"].values()}
    assert sd["sign_same_all_seeds"] == (len(signs) == 1)
    rc = sd["seed0_refit_check"]
    assert rc["reproduces_stage_a"] is False  # the synthetic units are not real refits
    assert rc["refit_device"] == "cpu" and rc["tolerance"] == 1e-3 and rc["delta1_gap"] > 0
    assert rc["delta1_gap"] == pytest.approx(abs(rc["delta1_of_refit"] - gl["estimate"]), abs=2e-4)
    assert isinstance(rc["delta1_gap_within_seed_sd"], bool)
    # F. average precision and the triage confusion matrix
    ap = s9["average_precision"]
    assert 0.2 < ap["prevalence_weighted"] < ap["chance"] < 0.45  # a random ranking beats the plain prevalence
    assert ap["models"]["dinov2_vits14"]["macro_ap"] > ap["models"]["convnext_tiny"]["macro_ap"] > ap["chance"]
    assert len(ap["chance_per_finding"]) == 12 and s9["baselines"]["rows"]["b0_chance"]["macro_ap"] == ap["chance"]
    assert s9["baselines"]["same_studies_and_folds_for_every_row"] is True and s9["baselines"]["incomplete_rows"] == []
    assert len(ap["models"]["floor_a"]["per_finding"]) == 12
    tr = s9["triage"]["models"]["dinov2_vits14"]
    assert set(tr["confusion"]) == {"tp", "fp", "fn", "tn"} and all(v == "<5" or v % 5 == 0 for v in tr["confusion"].values())
    assert {"sensitivity", "specificity", "precision", "f1", "normal_studies_moved_back_share"} <= set(tr)
    # E. the baselines table: one structure, the same studies and folds for every row
    rows = s9["baselines"]["rows"]
    assert list(rows)[:2] == ["b0_chance", "b1_scanner_tags"] and rows["b0_chance"]["s2_macro_auc"]["estimate"] == 0.5
    for e in ("convnext_tiny", "vit_s16", "dinov2_vits14"):
        for name in (f"b2_default_{e}", f"b2_default_fixed_{e}", f"tuned_{e}"):
            assert rows[name]["complete"], name
            assert {"estimate", "lo", "hi"} <= set(rows[name]["s2_macro_auc"]) and rows[name]["macro_ap"] is not None
            assert rows[name]["gold58_macro_auc"] is not None
        t = rows[f"tuned_{e}"]
        assert t["s2_macro_auc"]["estimate"] == s9["s2"][e]["fold_stratified_macro_auc"]
        assert t["mean_of_fold_macros"] == s9["s2"][e]["mean_of_fold_macros"]
        assert t["gain_over_b1"] == pytest.approx(t["s2_macro_auc"]["estimate"] - s9["h3"]["estimate"], abs=2e-4)
        assert t["gain_over_own_default"]["estimate"] == pytest.approx(
            t["s2_macro_auc"]["estimate"] - rows[f"b2_default_{e}"]["s2_macro_auc"]["estimate"], abs=2e-4)
    assert rows["b1_scanner_tags"]["s2_macro_auc"]["estimate"] == s9["h3"]["estimate"]
    assert rows["tuned_dinov2_vits14"]["role"] == "tuned, target model"
    assert rows["tuned_dinov2_vits14"]["gold58_macro_auc"] == s9["s3_gold"]["macro_auc"]["dinov2_vits14"]
    # G. HPO evidence (inner scores only)
    hp = s9["hpo_evidence"]
    assert set(hp["boundary"]) == {"lr", "weight_decay", "dropout", "pos_cap", "batch_size", "attn_hidden"}
    assert len(hp["best_so_far"]["dinov2_vits14"]["mean_best_so_far"]) == 30
    assert hp["best_so_far"]["dinov2_vits14"]["first_trial_of_final_best_median"] == 29.0
    assert hp["hierarchy"]["vit_s16"]["per_unit"]["fold_0"] == {"l1_margin": 0.03, "l2_gain": 0.02}
    assert len(hp["profiles"]["convnext_tiny"]["lr"]["mean_centred"]) == 6
    # H. the probe is copied from N7
    assert s9["near_duplicate_probe"]["cosine_above_0.98"]["pairs"] == "<5"
    _check_round3(s9, root)
    # the freeze file itself
    saved = json.loads((paths.WORKING / "agg_results.json").read_text())
    assert saved["baselines"]["rows"].keys() == rows.keys() and saved["chain"] == ch


def _check_round3(s9, root):
    """The blocks added after the results critique: descriptive, from files N7 and Stage A already wrote."""
    gl = s9["h1"]["g_lang"]
    for key in ("refit_agreement", "triage_secondary", "largest_site_decomposition", "best_epoch_spread",
                "gold_uncertainty", "h2_without_lookalikes"):
        assert s9[key]["descriptive"] is True and len(s9[key]["scope"]) > 10, key
    # 1. same-device gain: both sides are N7's refits; the old gain is untouched and flagged as mixed
    g = s9["baselines"]["same_device_gain"]
    assert g["device"] == "cpu" and g["stage_a_device"] == "cuda" and g["gain_over_own_default_mixes_devices"] is True
    assert set(g["encoders"]) == {"dinov2_vits14", "convnext_tiny"} and set(g["not_available"]) == {"vit_s16"}
    rows = s9["baselines"]["rows"]
    seeds = s9["seed_sensitivity"]["macro_auc_per_seed"]
    for e, v in g["encoders"].items():
        d0 = rows[f"b2_default_{e}"]["s2_macro_auc"]["estimate"]
        assert v["seeds"] == [0, 1, 2, 3, 4, 5] and v["seed0_minus_default"]["lo"] <= v["seed0_minus_default"]["estimate"]
        per = [seeds[e][f"seed_{k}"] for k in range(1, 6)]  # seeds 1-5 as N9 already reports them
        t0 = v["seed0_minus_default"]["estimate"] + d0
        assert v["mean_of_seeds_minus_default"] == pytest.approx(np.mean([t0, *per]) - d0, abs=3e-4)
        assert v["tuned_seed_sd"] == pytest.approx(np.std([t0, *per], ddof=1), abs=3e-4)
        # the older, mixed-device gain is still Stage A's fit minus the default, untouched
        assert rows[f"tuned_{e}"]["gain_over_own_default"]["estimate"] == pytest.approx(
            rows[f"tuned_{e}"]["s2_macro_auc"]["estimate"] - d0, abs=2e-4)
    # 2. refit agreement: synthetic units are not refits, so the gap is far from seed-like, and it is said
    ra = s9["refit_agreement"]
    assert ra["gap_like_another_seed"] is False and len(ra["fits"]) == 7
    assert ra["fits"]["stage_a"] == {"delta1": gl["estimate"], "device": "cuda", "role": "the decision fit"}
    assert ra["fits"]["n7_seed_3"]["device"] == "cpu"
    assert ra["fits"]["n7_seed_3"]["delta1"] == s9["seed_sensitivity"]["delta1_per_seed"]["seed_3"]
    e0 = ra["encoders"]["dinov2_vits14"]
    assert e0["n7_seed0_vs_stage_a"]["median"] > 1.5 * e0["n7_seed0_vs_n7_seed1"]["median"] > 0
    assert e0["n7_seed0_vs_stage_a"]["median"] <= e0["n7_seed0_vs_stage_a"]["p99"] <= e0["n7_seed0_vs_stage_a"]["max"]
    assert max(v["n7_seed0_vs_stage_a"]["max"] for v in ra["encoders"].values()) == pytest.approx(
        s9["seed_sensitivity"]["seed0_refit_check"]["max_abs_probability_gap"], abs=1e-4)
    assert ra["delta1_low_high_n7_fits"][0] <= ra["delta1_low_high_n7_fits"][1] and ra["delta1_sd_n7_fits"] > 0
    assert ra["delta1_low_high_all_fits"][1] == pytest.approx(gl["estimate"], abs=1e-4)  # Stage A's is the largest here
    # 3. triage: the score's own AUC, a specificity interval, the overshoot
    ts = s9["triage_secondary"]
    assert 0.9 < ts["any_finding_prevalence"] <= 1.0 and ts["target_sensitivity"] == 0.95
    for e, v in ts["models"].items():
        assert 0.5 < v["any_finding_auc"] <= 1.0
        tr = s9["triage"]["models"][e]
        if tr["sensitivity"] is not None:
            assert v["threshold_overshoot"] == pytest.approx(tr["sensitivity"] - 0.95, abs=6e-3)
        if tr["specificity"] is None:
            assert v["specificity_interval"] is None
        else:
            assert v["specificity_interval"][0] <= tr["specificity"] <= v["specificity_interval"][1]
    # 4. the largest site: unseen (S2) against seen (S1), at equal and larger training size
    ls = s9["largest_site_decomposition"]
    assert ls["group"].startswith("LANG") and ls["fold"] in range(5)
    tsz = ls["training_sets"]
    assert tsz["a_s2_fold_k"]["group_share"] == 0.0 and tsz["b_s1_fold_k"]["group_share"] > 0.05  # unseen against seen
    assert abs(tsz["a_s2_fold_k"]["studies"] - tsz["b_s1_fold_k"]["studies"]) <= 15  # the same size (usable studies, rounded)
    for e in ("dinov2_vits14", "convnext_tiny", "vit_s16"):
        v = ls["encoders"][e]
        assert v["studies"]["a"] >= v["studies"]["b"] and 0.5 < v["b_seen_equal_training_size"] <= 1
        assert v["unseen_site_cost_at_equal_training_size"] == pytest.approx(
            v["b_seen_equal_training_size"] - v["a_on_b_studies"], abs=2e-4)
    # 5. best-epoch spread and the schedule
    be = s9["best_epoch_spread"]["encoders"]["dinov2_vits14"]
    assert be["chosen_config"] == {"lowest": 4.0, "median": 4.5, "p90": 5.0, "share_at_or_below_6": 1.0, "share_at_cap": 0.0}
    assert be["refit_epochs"] == {"E": [6] * 5} and be["median_E"] == 6.0
    assert be["lr_fraction_of_peak_at_median_E"] == pytest.approx(0.93, abs=0.005) and be["warmup_share_of_training"] == 0.17
    assert s9["best_epoch_spread"]["scheduled_epochs"] == 30
    # 6. gold: a standard error, and the exact count only in a private file
    gu = s9["gold_uncertainty"]
    exact = json.loads((paths.WORKING / "private" / "n9" / "gold_count.json").read_text())
    n = exact["gold_studies_scored_by_both_main_encoders"]
    assert 50 <= n <= 58 and gu["studies"] == 5 * round(n / 5) and gu["n_draws"] == 2000
    assert str(n) not in json.dumps(gu) or n % 5 == 0  # the summary holds only the rounded count
    for e in ("dinov2_vits14", "convnext_tiny"):
        assert gu["macro_auc"][e]["estimate"] == s9["s3_gold"]["macro_auc"][e] and 0 < gu["macro_auc"][e]["se"] < 0.1
    assert gu["delta1"]["estimate"] == pytest.approx(s9["s3_gold"]["delta1"], abs=2e-4) and gu["delta1"]["se"] > 0
    # 7. Delta2 without the look-alike studies (post hoc); the pairs are the ones N7's probe counted
    h2 = s9["h2_without_lookalikes"]
    assert h2["post_hoc"] is True and h2["pairs_match_n7_probe"] is True and h2["studies_removed"] in ("<5", 5)
    assert h2["delta2_registered"]["estimate"] == s9["h2"]["estimate"]
    assert h2["delta2_without_lookalikes"]["estimate"] == pytest.approx(s9["h2"]["estimate"], abs=0.01)
    assert h2["change"] == pytest.approx(h2["delta2_without_lookalikes"]["estimate"] - s9["h2"]["estimate"], abs=2e-4)
    # 8. the per-finding note
    note = s9["h1"]["per_finding_exploratory_note"]
    k = sum(1 for v in s9["h1"]["per_finding_exploratory"].values() if v["lo"] > 0 or v["hi"] < 0)
    assert note["intervals_excluding_zero"] == f"{k} of 12" and "single" in note["note"]


def test_secondary_faults_never_cost_the_freeze_file_or_the_base_outputs(stage_env, tmp_path, monkeypatch):
    """Review round 2, findings 1 and 2: a failing or unsafe secondary block is recorded and skipped."""
    import shutil

    from knee.nb import n7, n9, n9x
    from tests.synth_stage import make_units

    root, facts = stage_env
    inp = tmp_path / "input"
    for part in ("n0", "n3", "n5"):
        shutil.copytree(root / "input" / part, inp / part)
    make_units(inp, strengths={"dinov2_vits14": 1.6, "convnext_tiny": 0.9})  # the two main encoders only: no N6b
    # N7 without N6b, a reduced extras set, and one block that fails for every encoder
    monkeypatch.setattr(paths, "WORKING", inp / "n7out")
    seen_agg = {}

    def broken(*a, **k):
        seen_agg["base"] = json.loads((paths.WORKING / "agg_n7.json").read_text())  # already on disk
        raise RuntimeError("boom")

    monkeypatch.setattr(n7, "_default_block", broken)
    s7 = n7.run(("dinov2_vits14", "convnext_tiny"), root=inp, device="cpu", quiet=True,
                extras=("seeds", "defaults"), seeds=(0, 1))
    assert set(s7["extras_errors"]) == {"dinov2_vits14.defaults", "convnext_tiny.defaults"} and s7["extras_ok"] is False
    assert all("RuntimeError" in v and "boom" not in v for v in s7["extras_errors"].values())
    assert s7["extras"]["convnext_tiny"] == {"seeds": [0, 1]}  # the other block still ran
    assert "near_duplicate_probe" not in s7 and s7["leakage_assertion_passed"] is True
    n7d = paths.WORKING / "private" / "n7"
    assert {p.name for p in n7d.glob("*.csv")} == {f"{a}_{e}.csv" for a in ("s1", "vendor", "s2seeds")
                                                   for e in ("dinov2_vits14", "convnext_tiny")}
    assert seen_agg["base"]["consensus"] == s7["consensus"] and seen_agg["base"]["extras_ok"] is None
    assert set(seen_agg["base"]["files_sha16"]) == {f"{a}_{e}.csv" for a in ("s1", "vendor")
                                                    for e in ("dinov2_vits14", "convnext_tiny")}
    saved7 = json.loads((paths.WORKING / "agg_n7.json").read_text())
    assert saved7["extras_errors"] == s7["extras_errors"] and saved7["files_sha16"] == s7["files_sha16"]
    with pytest.raises(ValueError):
        n7.run(root=inp, device="cpu", extras=("nonsense",))
    # N9: one block raises, one returns a small count, one returns report-like text
    monkeypatch.setattr(paths, "WORKING", tmp_path / "working")
    monkeypatch.setattr(n9x, "triage", lambda c: 1 / 0)
    monkeypatch.setattr(n9x, "hpo", lambda c: {"units": 3})
    monkeypatch.setattr(n9x, "near_duplicate_probe", lambda c: {"note": "x" * 300})
    s9 = n9.run(root=inp, n_boot=50, quiet=True, compare_with={"H1": "DINOv2 better", "H2": "supported", "H3": "supported"})
    saved = json.loads((paths.WORKING / "agg_results.json").read_text())  # the freeze file exists
    assert set(saved["secondary_errors"]) == {"triage", "hpo_evidence", "near_duplicate_probe"}
    assert "ZeroDivisionError" in saved["secondary_errors"]["triage"] and "UnsafeSummary" in saved["secondary_errors"]["hpo_evidence"]
    assert not {"triage", "hpo_evidence", "near_duplicate_probe"} & set(saved)
    assert set(saved["decisions"]) == {"H1", "H2", "H3", "H4"} and saved["decisions"] == s9["decisions"]
    assert saved["h1"]["g_lang"]["category"] == "DINOv2 better" and saved["anti_claim_1"] is None  # no control encoder
    cmp = saved["compare_with_previous"]
    assert cmp["same_h1_h2_h3"] == (not cmp["differs"]) and set(cmp["previous"]) == {"H1", "H2", "H3"}
    assert cmp["differs"] == [h for h in ("H1", "H2", "H3") if cmp["previous"][h] != saved["decisions"][h]]
    # the blocks that depend on what N7 did or did not write
    rows = saved["baselines"]["rows"]
    assert rows["tuned_vit_s16"]["complete"] is False and rows["b2_default_dinov2_vits14"]["complete"] is False
    assert saved["baselines"]["same_studies_and_folds_for_every_row"] is False
    assert "tuned_vit_s16" in saved["baselines"]["incomplete_rows"]
    assert saved["seed_sensitivity"]["seeds"] == [0, 1] and "seed0_refit_check" in saved["seed_sensitivity"]
    # the blocks added after the results critique say what N7's files cannot support, and invent nothing
    sdg = saved["baselines"]["same_device_gain"]
    assert sdg["encoders"] == {} and set(sdg["not_available"]) == {"dinov2_vits14", "convnext_tiny", "vit_s16"}  # no default refits
    assert sdg["stage_a_device"] == "unknown" and sdg["gain_over_own_default_mixes_devices"] == "unknown"  # no run.json
    assert len(saved["refit_agreement"]["fits"]) == 3 and saved["refit_agreement"]["fits"]["stage_a"]["device"] == "unknown"
    assert set(saved["triage_secondary"]["models"]["dinov2_vits14"]) == {"any_finding_auc"}  # no thresholds: the AUC only
    assert saved["h2_without_lookalikes"]["pairs_match_n7_probe"] is None  # the probe did not run in N7
    assert set(saved["largest_site_decomposition"]["encoders"]) == {"dinov2_vits14", "convnext_tiny"}
    assert not [u for u in facts["uids"] if u in json.dumps(saved)]


def test_chain_guard_refuses_units_from_other_s2_folds(stage_env, tmp_path, monkeypatch):
    import shutil

    from knee import folds as kfolds, stage, units
    from knee.nb import n9
    from tests.synth_stage import make_units

    root, _ = stage_env
    inp = tmp_path / "input"
    for part in ("n0", "n3", "n5"):
        shutil.copytree(root / "input" / part, inp / part)
    make_units(inp)
    fp = inp / "n3" / "private" / "n3" / "folds.csv"
    before = stage.StageData(inp)
    dirs = units.unit_dirs(inp)
    assert len(dirs) == 15 and units.check_units(before, dirs)["s2_folds_match_units"]
    # a rerun of N3 that only regroups the seen-site folds: another file hash, the same S2 hash, guard passes
    f = pd.read_csv(fp)
    f["s1_fold"] = f["s1_fold"].to_numpy()[::-1]
    f.to_csv(fp, index=False)
    after = stage.StageData(inp)
    old = before.upstream["n3_folds"]["sha16"]
    assert after.upstream["n3_folds"]["sha16"] != old
    (inp / "n6a" / "agg_n6a.json").write_text(json.dumps({"upstream": {"n3_folds": {"sha16": old}}}))  # as Stage A wrote it
    ch = units.chain_summary(after, dirs, inp)
    assert ch["stage_a_n3_folds_sha16"] == {"n6a": old} and ch["n3_hash_same_as_stage_a"] == {"n6a": False}  # by design
    assert ch["s2_folds_match_units"] and ch["s2_sha16"] == kfolds.s2_sha16(root / "input" / "n3" / "private" / "n3" / "folds.csv")
    # one training study moved to another S2 fold: two units no longer match, and N9 refuses to freeze
    i = int(np.flatnonzero((f["train"] & (f["s2_fold"] == 0)).to_numpy() & after.usable)[0])
    f.loc[i, "s2_fold"] = 1
    f.to_csv(fp, index=False)
    moved = stage.StageData(inp)
    bad = units.check_units(moved, dirs)
    assert not bad["s2_folds_match_units"]
    assert set(bad["mismatched_units"]) == {f"{e}_f{k}" for e in ("dinov2_vits14", "convnext_tiny", "vit_s16") for k in (0, 1)}
    monkeypatch.setattr(paths, "WORKING", tmp_path / "working")
    from knee import safe_log

    with pytest.raises(safe_log.StepFailed, match="ChainError"):  # fails loudly: the step, with the error type
        n9.run(root=inp, n_boot=50, quiet=True)
    saved = json.loads((paths.WORKING / "agg_results.json").read_text())
    assert saved["status"] == "chain mismatch" and saved["chain"]["s2_folds_match_units"] is False
    assert "decisions" not in saved and "h1" not in saved



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
