"""The secondary, descriptive additions of 23 Nov: AP, the study-level bootstrap, the chain guard,
the default-head settings, the HPO evidence, the pair facts of N3 (fast, synthetic, CPU)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from knee import agg, folds, hpo_evidence, intervals, metrics

UID = "StudyInstanceUID"


# ---------------------------------------------------------------- metrics

def test_average_precision_matches_sklearn_with_ties_and_weights():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    s = rng.integers(0, 6, 400).astype(float)  # many ties
    y = (rng.random(400) < 0.2).astype(float)
    assert metrics.average_precision(s, y)[0] == pytest.approx(average_precision_score(y, s))
    w = rng.integers(0, 3, 400)
    assert metrics.average_precision(s, y, w)[0] == pytest.approx(average_precision_score(y, s, sample_weight=w))
    assert np.isnan(metrics.average_precision(s, np.zeros(400))[0])  # no positives: undefined


def test_stratified_ap_never_ranks_across_folds_and_has_a_chance_level():
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.25).astype(float)
    f = np.repeat([0, 1, -1], 200)
    s = y + rng.normal(0, 1, 600)
    ap, P = metrics.stratified_ap(s, y, f)
    chance, prev_w = metrics.stratified_ap_chance(y, f)
    shifted, _ = metrics.stratified_ap(s + np.where(f == 1, 9.0, 0.0), y, f)  # an offset in one fold model
    assert ap == pytest.approx(shifted)
    parts = [metrics.average_precision(s[f == k], y[f == k]) for k in (0, 1)]
    assert ap == pytest.approx(sum(a * p for a, p in parts) / sum(p for _, p in parts))
    assert P == (y[f >= 0] == 1).sum()  # the fold -1 studies are left out
    prev = [(y[f == k] == 1).mean() for k in (0, 1)]
    assert prev_w == pytest.approx(sum(pr * p for pr, (_, p) in zip(prev, parts)) / P)
    assert chance == pytest.approx(sum(metrics.random_ap(p, 200) * p for _, p in parts) / P) and chance > prev_w
    # a random ranking scores the chance level
    rand = np.mean([metrics.stratified_ap(rng.random(600), y, f)[0] for _ in range(2000)])
    assert rand == pytest.approx(chance, abs=0.005)
    y2 = np.c_[y, np.zeros(600)]
    assert np.isnan(metrics.macro_ap(np.c_[s, s], y2, f, [0, 1]))  # a finding without positives: undefined, not dropped
    assert np.isnan(metrics.macro_ap_chance(y2, f, [0, 1])[0])
    assert metrics.macro_ap_chance(y2, f, [0]) == pytest.approx((chance, prev_w))


def test_random_ap_is_the_simulated_chance_level_also_for_rare_findings():
    rng = np.random.default_rng(5)
    for P, n in ((5, 700), (20, 700), (150, 300)):
        y = np.r_[np.ones(P), np.zeros(n - P)]
        sim = [metrics.average_precision(rng.random(n), y)[0] for _ in range(6000)]
        se = np.std(sim) / np.sqrt(len(sim))
        assert abs(np.mean(sim) - metrics.random_ap(P, n)) < 4 * se  # the formula, within simulation noise
    assert metrics.random_ap(5, 700) > 2 * 5 / 700  # rare finding: chance is over twice the prevalence
    assert metrics.random_ap(1, 50) == pytest.approx(sum(1 / i for i in range(1, 51)) / 50)  # one positive: E[1/rank]


def test_threshold_confusion_and_rates():
    rng = np.random.default_rng(2)
    pos = rng.random(1000) < 0.4
    s = pos + rng.normal(0, 1, 1000)
    t = metrics.threshold_at_sensitivity(s, pos, 0.95)
    assert (s[pos] >= t).mean() >= 0.95 > (s[pos] > t).mean()  # the highest threshold that keeps 95%
    c = metrics.confusion(s >= t, pos)
    assert sum(c.values()) == 1000 and c["tp"] + c["fn"] == pos.sum()
    r = metrics.rates(c)
    assert r["sensitivity"] == pytest.approx(c["tp"] / pos.sum())
    assert r["f1"] == pytest.approx(2 * r["precision"] * r["sensitivity"] / (r["precision"] + r["sensitivity"]))
    assert metrics.rates({"tp": 0, "fp": 0, "fn": 0, "tn": 5})["precision"] is None
    # privacy: a rate that uses a cell of 1 to 4 studies is not shown (the cell itself shows as "<5")
    small = metrics.rates({"tp": 90, "fp": 10, "fn": 3, "tn": 40}, suppress_small=True)
    assert small["sensitivity"] is None and small["f1"] is None
    assert small["specificity"] == pytest.approx(0.8) and small["precision"] == pytest.approx(0.9)
    assert metrics.rates({"tp": 90, "fp": 0, "fn": 10, "tn": 40}, suppress_small=True)["precision"] == 1.0  # zero is shown
    assert np.isnan(metrics.threshold_at_sensitivity(s, np.zeros(1000, bool)))


# ---------------------------------------------------------------- the second interval

def test_within_fold_bootstrap_sees_noise_the_group_interval_cannot():
    rng = np.random.default_rng(3)
    n = 900
    fold = np.r_[np.zeros(700, int), np.ones(100, int), np.full(100, -1)]
    x = rng.normal(0, 1, n)

    def stat(w):  # the mean of fold 0 only: one fold, one group
        m = fold == 0
        return float((w[m] * x[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float("nan")

    iv = intervals.within_fold_bootstrap(stat, fold, n_boot=400, seed=0)
    assert iv["lo"] < stat(np.ones(n)) < iv["hi"] and iv["valid_share"] == 1.0 and iv["n_boot"] == 400
    assert iv["half_width"] == pytest.approx(1.96 / np.sqrt(700), rel=0.25)  # the usual standard error
    assert iv == intervals.within_fold_bootstrap(stat, fold, n_boot=400, seed=0)  # deterministic
    # the group bootstrap keeps or drops fold 0 whole, so its value there never moves
    groups = np.where(fold == 0, "A", np.where(fold == 1, "B", "C"))
    cb = intervals.cluster_bootstrap(stat, groups, n_boot=200, seed=0)
    assert cb["hi"] - cb["lo"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- folds hash and N3's pair facts

def _folds_table():
    idx = pd.Index([f"u{i}" for i in range(12)], name=UID)
    return pd.DataFrame({"train": [True] * 10 + [False] * 2, "s2_fold": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, 0],
                         "vendor_fold": [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, -1, 0],
                         "s1_fold": [0, 0, 1, 2, 2, 3, 3, 4, 4, 1, -1, -1]}, index=idx)


def test_s2_sha16_ignores_only_the_seen_site_column(tmp_path):
    from knee.nb.common import sha16

    t = _folds_table()
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    t.reset_index().to_csv(a, index=False)
    t.assign(s1_fold=t["s1_fold"].to_numpy()[::-1]).reset_index().to_csv(b, index=False)
    t2 = t.copy()
    t2.loc["u3", "s2_fold"] = 2
    t2.reset_index().to_csv(c, index=False)
    assert sha16(a) != sha16(b) and folds.s2_sha16(a) == folds.s2_sha16(b)  # S1 regrouped: same S2 hash
    assert folds.s2_sha16(a) != folds.s2_sha16(c)  # one S2 fold changed: another hash
    assert len(folds.s2_sha16(a)) == 16


def test_pair_facts_count_applied_together_and_across():
    from knee.nb import n3

    t = _folds_table()
    pairs = [("u0", "u1"), ("u2", "u3"), ("u8", "u11"), ("u4", "missing")]
    f = n3.pair_facts(t, pairs)
    # u0-u1 and u2-u3 are training pairs; u8-u11 has a non-training study; the last has an unknown UID
    assert f["pairs_applied"] == agg.count(2) and f["pairs_not_applied"] == agg.count(2)
    assert f["pairs_together"] is False  # u2 and u3 sit in different S1 folds
    assert f["pairs_across_vendor_folds"] == agg.count(2)  # u0-u1 and u8-u11
    assert f["pairs_across_s2_folds"] == agg.count(1)  # u8-u11
    assert n3.pair_facts(t, [("u0", "u1")])["pairs_together"] is True
    assert n3.pair_facts(t, [])["pairs_together"] is None
    s1 = folds.seen_site_folds(t.loc[t["train"], "s2_fold"], pairs)
    t.loc[s1.index, "s1_fold"] = s1
    assert n3.pair_facts(t, pairs)["pairs_together"] is True  # N3's own S1 folds keep applied pairs together


# ---------------------------------------------------------------- HPO evidence

def _unit(enc, k, pooling, values, lr, l1_scores, chosen_lr):
    n = len(values)
    t = pd.DataFrame({"number": range(n), "state": "COMPLETE", "value": values, "lr": lr,
                      "weight_decay": [1e-3] * n, "dropout": np.linspace(0.0, 0.5, n), "pos_cap": [2.0] * n,
                      "batch_size": [48] * n})
    if pooling == "gated":
        t["attn_hidden"] = [192] * n
    l1 = pd.DataFrame({"pooling": ["mean", "max", "gated"], "planes": "all", "inner_score": l1_scores})
    cfg = {"pooling": pooling, "planes": "all", "lr": chosen_lr, "weight_decay": 1e-3, "dropout": 0.25,
           "attn_hidden": 192, "pos_cap": 2.0, "batch_size": 48}
    return {"encoder": enc, "fold": k, "config": {"config": cfg}, "trials": t, "l1": l1}


def test_space_matches_what_suggest_draws():
    optuna = pytest.importorskip("optuna")
    from knee.nb import n6

    for gated in (True, False):
        trial = optuna.create_study().ask()
        n6.suggest(trial, 20.0, gated)
        got = {k: (d.low, d.high, bool(d.log)) for k, d in trial.distributions.items()}
        assert got == n6.space(20.0, gated)


def test_hpo_evidence_boundary_curve_hierarchy_and_profiles():
    pytest.importorskip("optuna")
    lr = [3e-4, 1e-5, 1e-2, 1e-3, 2e-4, 5e-3]  # two of six at the ends of [1e-5, 1e-2]
    units = [
        _unit("enc_a", 0, "gated", [0.70, 0.60, 0.72, 0.71, 0.72, 0.69], lr, [0.60, 0.65, 0.70], chosen_lr=1e-2),
        _unit("enc_a", 1, "mean", [0.80, 0.70, 0.75, 0.78, 0.79, 0.77], lr, [0.80, 0.70, 0.60], chosen_lr=3e-4),
        {**_unit("enc_a", 2, "mean", [0.5], [3e-4], [0.5, 0.4, 0.3], 3e-4),
         "trials": pd.DataFrame({"number": [0], "state": ["REUSED"], "value": [0.5], "lr": [3e-4]})},
    ]
    s = hpo_evidence.summary(units, U=20.0)
    assert s["units_without_own_search"] == {"enc_a": {"folds": [2]}}  # the simplified unit is listed, not used
    b = s["boundary"]["lr"]
    assert b["scale"] == "log" and b["all"]["chosen"] == {"share_outer_5pct": 0.5, "share_low_end": 0.0, "share_high_end": 0.5}
    assert b["all"]["top5_trials"]["share_outer_5pct"] == pytest.approx(0.2)  # 2 of the 10 top-5 trials (lr 1e-2 in each unit)
    assert s["boundary"]["dropout"]["all"]["top5_trials"]["share_outer_5pct"] > 0  # linear scale, ends at 0 and 0.5
    assert s["boundary"]["attn_hidden"]["all"]["units"] == "<5"  # only the gated unit searched it
    assert hpo_evidence.position([1e-5, 10 ** -3.5, 1e-2], 1e-5, 1e-2, True) == pytest.approx([0, 0.5, 1])
    c = s["best_so_far"]["enc_a"]
    assert c["mean_best_so_far"] == [0.75, 0.75, 0.76, 0.76, 0.76, 0.76] and c["n_trials"] == 6
    assert c["first_trial_of_final_best_median"] == 1.0  # trial 2 for the first unit, trial 0 for the second
    assert c["share_units_best_is_trial_0"] == 0.5
    h = s["hierarchy"]["enc_a"]
    assert h["per_unit"]["fold_0"] == {"l1_margin": 0.05, "l2_gain": 0.02}
    assert h["per_unit"]["fold_1"] == {"l1_margin": 0.1, "l2_gain": 0.0}
    assert h["share_units_l2_gain_above_l1_margin"] == 0.0
    p = s["profiles"]["enc_a"]["lr"]
    assert len(p["mean"]) == hpo_evidence.N_BINS and sum(p["trial_share"]) == pytest.approx(1.0, abs=0.03)
    assert p["mean"][0] == pytest.approx(0.65) and p["mean_centred"][0] < 0  # lr = 1e-5 scored worst in both units
    assert "attn_hidden" in s["profiles"]["enc_a"]
    agg._check(json.loads(json.dumps(s)))  # safe to write: no small counts, no long strings


def test_hpo_evidence_uses_trial_numbers_and_start_up_trials_only():
    pytest.importorskip("optuna")
    from knee.nb import n6

    n = 14
    values = [0.70] * n
    values[1], values[12] = 0.74, 0.90  # trial 1 is the best start-up trial; a late GP trial is the best of all
    lr = [3e-4] * 10 + [1e-2] * 4  # the optimiser's late trials all sit at the top of the lr range
    u = _unit("enc_a", 0, "mean", values, lr, [0.70, 0.6, 0.5], chosen_lr=1e-2)
    u["trials"].loc[0, "state"] = "FAIL"  # trial 0 failed: it keeps its number
    s = hpo_evidence.summary([u], U=20.0)
    c = s["best_so_far"]["enc_a"]
    assert c["n_trials"] == n and c["mean_best_so_far"][0] is None and c["mean_best_so_far"][1] == 0.74
    assert c["first_trial_of_final_best_median"] == 12.0  # the trial's number, not its place among completed trials
    assert c["share_units_best_is_trial_0"] == 0.0
    assert c["mean_gain_after_startup_trials"] == pytest.approx(0.16)
    p = s["profiles"]["enc_a"]["lr"]
    assert p["mean"][-1] is None and p["trial_share"][-1] == 0.0  # the late lr = 1e-2 trials are not in the profile
    assert sum(x is not None for x in p["mean"]) == 1 and "start-up" in s["profiles_from"]
    assert n6.N_STARTUP == 10
