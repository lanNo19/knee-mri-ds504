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


def test_wilson_interval_and_schedule_factor():
    from knee import heads
    from knee.nb import n9c

    lo, hi = n9c.wilson(8, 100)
    assert (lo, hi) == pytest.approx((0.0411, 0.1500), abs=5e-4)  # the textbook Wilson interval for 8 of 100
    lo0, hi0 = n9c.wilson(0, 50)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0 < hi0 < 0.08
    for k, n in ((0, 1), (0, 7), (0, 50), (1, 1), (50, 50), (3, 1000)):  # never outside [0, 1]
        a, b = n9c.wilson(k, n)
        assert 0.0 <= a <= k / n + 1e-12 and k / n - 1e-12 <= b <= 1.0
    assert all(np.isnan(v) for v in n9c.wilson(0, 0))  # no studies: no interval, and no ZeroDivisionError
    # the summary's interval: 2 decimals, and none from fewer than 20 normal studies or beside a "<5" cell
    assert n9c.specificity_interval(400, 35) == [0.89, 0.94]
    assert n9c.specificity_interval(6, 0) is None and n9c.specificity_interval(19, 0) is None
    assert n9c.specificity_interval(200, 3) is None and n9c.specificity_interval(20, 0) == [0.84, 1.0]
    spe = 76
    # 1-epoch warm-up, then a cosine over the 30-epoch schedule: at E = 4, 5, 6 the rate is still near its peak
    got = [heads.schedule_factor(E * spe - 1, spe) for E in (4, 5, 6)]
    assert got == pytest.approx([0.974, 0.954, 0.930], abs=3e-3)
    assert heads.schedule_factor(0, spe) == pytest.approx(1 / spe) and heads.schedule_factor(spe - 1, spe) == 1.0
    assert heads.schedule_factor(30 * spe, spe) == pytest.approx(0.0, abs=1e-12)

    def old_closure(spe_sched):  # heads.fit's inline schedule before it became schedule_factor (commit d8212d5)
        import math

        total = heads.MAX_EPOCHS * spe_sched
        warm = heads.WARMUP_EPOCHS * spe_sched

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

        return lr_at

    for s in (1, 2, 7, 76, 500):
        lr_at = old_closure(s)
        assert all(heads.schedule_factor(i, s) == lr_at(i) for i in range(0, 32 * s + 2))  # bit for bit


def test_refit_agreement_says_when_the_device_gap_is_like_another_seed(monkeypatch):
    from types import SimpleNamespace

    from knee.nb import n9c, n9x

    rng = np.random.default_rng(7)
    n = 400
    base = {e: rng.random((n, 12)) for e in ("dinov2_vits14", "convnext_tiny")}

    def refits(c, e):  # six N7 seeds: the same fit plus seed-sized noise
        r = np.random.default_rng(len(e))
        return {sd: np.clip(base[e] + r.normal(0, 0.02, (n, 12)), 0, 1) for sd in range(6)}

    monkeypatch.setattr(n9x, "_seed_predictions", refits)
    monkeypatch.setattr(n9x, "_n7_device", lambda c: "cpu")

    def ctx(stage_a_noise):
        units = {e: np.clip(base[e] + rng.normal(0, stage_a_noise, (n, 12)), 0, 1) for e in base}
        return SimpleNamespace(units=units, ev=np.arange(n), dirs={}, h1={"estimate": 0.01},
                               complete=lambda P: True, macro=lambda P, w=None: float(P.mean()))

    like = n9c.refit_agreement(ctx(0.02), {})  # Stage A differs from N7 seed 0 as two seeds do
    assert like["gap_like_another_seed"] is True and like["fits"]["stage_a"]["device"] == "unknown"
    assert list(like["fits"]) == ["stage_a"] + [f"n7_seed_{k}" for k in range(6)]
    far = n9c.refit_agreement(ctx(0.2), {})  # ten times further away: not seed-like
    assert far["gap_like_another_seed"] is False
    yard = like["encoders"]["dinov2_vits14"]["n7_seed_pairs_median"]
    assert far["encoders"]["dinov2_vits14"]["n7_seed_pairs_median"] == yard and like["seeds"] == [0, 1, 2, 3, 4, 5]
    # the yardstick is the median over all 15 seed pairs, not seeds 0 and 1 alone
    P = refits(None, "dinov2_vits14")
    med = [np.median(np.abs(P[a] - P[b])) for a in range(6) for b in range(a + 1, 6)]
    assert len(med) == 15 and yard["median"] == pytest.approx(np.median(med), abs=1e-5)
    assert "not a test" in like["flag_is"]


def test_primary_numbers_are_compared_at_the_decimals_supplied():
    from knee.nb import n9

    s = {"h1": {"g_lang": {"estimate": -0.0213, "lo": -0.053, "hi": 0.0105},
                "g_vendor": {"estimate": -0.0173, "lo": -0.0563, "hi": 0.0224}},
         "h2": {"estimate": 0.0142, "lo": -0.0061, "hi": 0.0338}, "h3": {"estimate": 0.5061, "lo": 0.4897, "hi": 0.5172}}
    card = {"delta1_g_lang": ["-0.0213", "-0.0530", "+0.0105"], "delta1_g_vendor": ["-0.0173", "-0.056", "+0.022"],
            "delta2": ["+0.0142", "-0.006", "+0.034"], "h3": ["0.506", "0.490", "0.517"]}
    m = n9.primary_matches(s, card)
    assert m["all_match"] is True and m["complete_card"] is True and m["unknown_keys_present"] is False
    assert "error" not in m
    assert m["numbers"]["delta1_g_vendor"] == {"estimate": True, "lo": True, "hi": True}
    assert m["compared_at_decimals"] == {"delta1_g_lang": "4, 4, 4", "delta1_g_vendor": "4, 3, 3",
                                         "delta2": "4, 3, 3", "h3": "3, 3, 3"}
    agg._check(json.loads(json.dumps(m)))  # booleans and short strings only
    # one unit of the last supplied decimal off: that number, and only that number, is false
    off = n9.primary_matches(s, {**card, "delta1_g_lang": ["-0.0214", "-0.0530", "+0.0105"], "h3": ["0.506", "0.491", "0.517"]})
    assert off["all_match"] is False
    assert off["numbers"]["delta1_g_lang"] == {"estimate": False, "lo": True, "hi": True}
    assert off["numbers"]["h3"] == {"estimate": True, "lo": False, "hi": True} and off["numbers"]["delta2"]["hi"] is True
    # refit noise of about 0.006 is far outside every tolerance
    assert n9.primary_matches(s, {"delta2": ["+0.0202", "-0.006", "+0.034"]})["numbers"]["delta2"]["estimate"] is False
    # dict form, numbers, a missing block, nonsense: reported, never raised
    assert n9.primary_matches(s, {"h3": {"estimate": 0.5061}})["numbers"]["h3"] == {"estimate": True}
    assert n9.primary_matches({"h1": {"g_vendor": {"category": "not run"}}}, card)["all_match"] is False
    for bad in ({"h3": ["x", None, "0.5"]}, {"nonsense": [1, 2, 3]}, {"h3": 7}, {}):
        r = n9.primary_matches(s, bad)
        assert r["all_match"] is False
    # review r2, N1: an incomplete card can never pass, even when every number it has matches
    short = {k: v for k, v in card.items() if k != "h3"}
    r = n9.primary_matches(s, short)
    assert r["complete_card"] is False and r["all_match"] is False and all(v for f in r["numbers"].values() for v in f.values())
    r = n9.primary_matches(s, {**card, "h3": ["0.506", "0.490"]})
    assert r["complete_card"] is False and r["all_match"] is False
    # review r2, N2: an unknown name is marked, never echoed; a UID-like one cannot make the file unwritable
    uid_like = "1.2.840.113619.2.55.3"
    r = n9.primary_matches(s, {**card, uid_like: ["0", "0", "0"]})
    assert r["unknown_keys_present"] is True and r["all_match"] is False and uid_like not in json.dumps(r)
    agg._check(json.loads(json.dumps(r)))
    assert n9.primary_matches(s, {"nonsense": [1, 2, 3]})["unknown_keys_present"] is True
    # review r2, N6: exponent notation and unreadable input
    assert n9._decimals("1e-3") == 3 and n9._decimals("5e-05") == 4 and n9._decimals("-0.0530") == 4
    assert n9._decimals("x") == 0 and n9._decimals(None) == 0 and n9._decimals("nan") == 0 and n9._decimals("2") == 0


def test_a_changed_primary_block_is_put_back_from_the_snapshot():
    from knee.nb import n9

    def summary():
        return {"h1": {"g_lang": {"estimate": 0.01, "lo": -0.01, "hi": 0.03, "category": "inconclusive"}, "decision": "inconclusive",
                       "per_finding_exploratory": {"acl": {"lo": 0.0, "hi": 0.1}}},
                "h2": {"estimate": 0.02, "category": "inconclusive"}, "h3": {"estimate": 0.5, "supported": False},
                "decisions": {"H1": "inconclusive", "H2": "inconclusive", "H3": "not supported"}, "s2": {"x": 1.0}}

    s = summary()
    snap = n9._primary_view(s)
    # the three places where secondary blocks add keys inside H1 are not changes
    s["h1"]["secondary"] = {"a": 0.5}
    s["h1"]["g_lang"]["interval_within_fold_study_bootstrap"] = {"lo": 0.0}
    s["h1"]["per_finding_exploratory_note"] = {"note": "n"}
    s["baselines"] = {"rows": {}}
    assert n9._restore_primary(s, snap) is False and s["h1"]["secondary"] == {"a": 0.5}
    # a changed decision, a changed number, a removed block, an added key: each is seen and undone
    for change in (lambda s: s["decisions"].update(H1="DINOv2 better"), lambda s: s["h2"].update(estimate=0.5),
                   lambda s: s.pop("h3"), lambda s: s["h1"]["g_lang"].update(extra=1.0)):
        change(s)
        assert n9._restore_primary(s, snap) is True
        assert n9._primary_view(s) == snap and s["decisions"]["H1"] == "inconclusive" and s["h2"]["estimate"] == 0.02
        assert s["h1"]["secondary"] == {"a": 0.5} and s["h1"]["per_finding_exploratory_note"] == {"note": "n"}  # the additions stay
        assert s["h1"]["g_lang"]["interval_within_fold_study_bootstrap"] == {"lo": 0.0} and s["baselines"] == {"rows": {}}
        assert n9._restore_primary(s, snap) is False
    s["h1"]["g_lang"] = object()  # not even comparable: put back, without raising
    assert n9._restore_primary(s, snap) is True and n9._primary_view(s) == snap


def test_every_key_written_before_the_blocks_is_put_back_and_a_failed_final_write_keeps_the_flag(monkeypatch):
    """Review r2, N3 and N5."""
    import copy

    from knee.nb import n9

    s = {"h1": {"g_lang": {"estimate": 0.01}, "decision": "inconclusive"}, "h2": {"estimate": 0.02},
         "h3": {"estimate": 0.5}, "decisions": {"H1": "inconclusive"}, "s2": {"x": 1.0}, "chain": {"ok": True}}
    before, snap = copy.deepcopy(s), n9._primary_view(s)
    s["average_precision"] = {"a": 0.5}  # a new key: what the blocks are for
    assert n9._restore_earlier(s, before, snap) is False and s["average_precision"] == {"a": 0.5}
    for change in (lambda s: s.update(s2={"x": 9.0}), lambda s: s["chain"].update(extra=True), lambda s: s.pop("chain"),
                   lambda s: s.update(chain=object())):
        change(s)
        assert n9._restore_earlier(s, before, snap) is True
        assert s["s2"] == {"x": 1.0} and s["chain"] == {"ok": True} and s["average_precision"] == {"a": 0.5}
    s["decisions"]["H1"] = "DINOv2 better"
    assert n9._restore_earlier(s, before, snap) is True and s["decisions"]["H1"] == "inconclusive"

    # the second write fails: the file gets the primary results, and the detected change is not hidden
    written = []

    def write(name, payload):
        if not written and "average_precision" in payload:
            written.append("refused")
            raise RuntimeError("cannot write")
        written.append(copy.deepcopy(payload))

    monkeypatch.setattr(n9.agg, "write", write)
    s["secondary_mutated_primary"] = True
    n9._write_final(s, before)
    assert written[0] == "refused" and "average_precision" not in written[1]
    assert written[1]["secondary_mutated_primary"] is True and "RuntimeError" in written[1]["secondary_errors"]["final_write"]
    assert written[1]["s2"] == {"x": 1.0} and written[1]["decisions"] == {"H1": "inconclusive"}


def test_each_block_sees_h1_and_h2_as_decided(monkeypatch):
    """Review r2, N4: a block that writes into its H1 or H2 copy changes nothing a later block reads."""
    from knee.nb import n9c, n9x

    seen = {}

    def spoil(c, s):
        c.h1["estimate"], c.h2["estimate"] = 9.0, 9.0
        return {"descriptive": True}

    def read(c, s):
        seen["h1"], seen["h2"] = c.h1["estimate"], c.h2["estimate"]
        return {"descriptive": True}

    for name in ("per_fold_delta1", "study_bootstrap", "average_precision", "seed_sensitivity", "triage", "baselines",
                 "hpo", "near_duplicate_probe"):
        monkeypatch.setattr(n9x, name, lambda c: {"descriptive": True})
    monkeypatch.setattr(n9c, "BLOCKS", [("spoil", spoil, lambda s: s, "spoil"), ("read", read, lambda s: s, "read")])
    c = n9x.Ctx(d=None, ev=np.arange(3), Y=None, units={}, elig=[], n_boot=10, seed=0, root=None, row_of={}, floor=None,
                gold=np.arange(0), g_find=[], h1={"estimate": 0.1}, cat_lang="inconclusive", n7_dir=None,
                h2={"estimate": 0.2})
    s = {"h1": {"g_lang": {}}, "decisions": {"H1": "inconclusive"}}
    n9x.add_all(s, c)
    assert seen == {"h1": 0.1, "h2": 0.2} and s["secondary_errors"] == {} and "read" in s and "spoil" in s


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
