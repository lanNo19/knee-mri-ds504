"""The dropout-ceiling check (it2d_dropout_r2.md): n13d's arms dc, p5, p6 and p7 and n13d.decide_dropout.

Synthetic data, CPU. The arms fit the unit's chosen Stage A configuration at its chosen batch size on its
inner folds, as n13's arm bc does; only the dropout of the head differs (dc: the chosen one, the base).
The Kaggle sessions install commit DROPOUT_ARM_COMMIT: one test shows that everything n13d.run_dropout
reaches is textually that commit's, so the later commit changed the local decide step only.
"""
from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from knee import agg, heads, paths

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
MAIN = ("dinov2_vits14", "convnext_tiny")
DN, CX = MAIN
FIND = list(range(12))
ARMS = ("dc", "p5", "p6", "p7")
P = {"dc": None, "p5": 0.5, "p6": 0.6, "p7": 0.7}
# the chosen dropout per unit as stored by knee-n6a (unrounded; 25 Nov at 3 decimals: n13d.DROPOUT_REF)
CHOSEN_DROPOUT = {DN: [0.4718740392573121, 0.0, 0.5, 0.0525307933392619, 0.0],
                  CX: [0.4718740392573121, 0.5, 0.3857844902577407, 0.0, 0.22173021826688713]}
# synthetic chosen configurations; batch size and best score of each unit are CHOSEN_REF's, dropout as above
CHOSEN = {"pooling": "gated", "planes": "all", "lr": 1e-3, "weight_decay": 1e-4, "attn_hidden": 32, "pos_cap": 4.0}


def _git(*args) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_stage_a(inp: Path, dropout=None) -> None:
    """knee-n6a's per-unit summaries (agg_n6a_<encoder>_f<k>.json at the output root), synthetic."""
    from knee.nb import n13

    d = inp / "n6a"
    d.mkdir(parents=True, exist_ok=True)
    for e in MAIN:
        for k, (b, best) in n13.CHOSEN_REF[e].items():
            u = {"encoder": e, "fold": k, "tuning": "nested", "l2_from": "bayesian optimisation on this fold",
                 "l1_winner": {"pooling": "gated", "planes": "all"},
                 "config": {**CHOSEN, "dropout": (dropout or CHOSEN_DROPOUT)[e][k], "batch_size": b},
                 "best_inner_score": best, "E": 5, "default_trial_score": round(best - 0.002, 4)}
            (d / f"agg_n6a_{e}_f{k}.json").write_text(json.dumps(u, indent=1))


@pytest.fixture(scope="module")
def stage_env(tmp_path_factory):
    from tests.synth_stage import make_stage

    root = tmp_path_factory.mktemp("it2dstage")
    facts = make_stage(root / "input", S=8)
    _write_stage_a(root / "input")
    return root, facts


@pytest.fixture()
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # as in the workers
    yield
    torch.set_num_threads(n)


@pytest.fixture()
def worker(stage_env, monkeypatch):
    """This process as one of n6's workers on the synthetic stage."""
    from knee.nb import n6

    root, _ = stage_env
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    return root


def _task(arm, chosen, enc=DN, fold=0, deadline=None):
    return {"id": f"{enc}_f{fold}", "encoder": enc, "fold": fold, "seed": 0, "eligible": FIND, "arm": arm,
            "config_chosen": chosen, "stage_a_best": 0.7, "chosen_sha16": "0" * 16,
            "deadline": time.time() + 3600 if deadline is None else deadline, "dir": "unused"}


def _bc_task(chosen, enc=DN, fold=0):
    return {**_task("dc", chosen, enc, fold), "batch_size": None}  # n13's arm bc: the chosen batch size


# ---------------------------------------------------------------- the arms

def test_dropout_arms_and_settings_are_the_declared_ones():
    from knee.nb import n13, n13d

    assert n13d.DROPOUT_ARMS == {"dc": None, "p5": 0.5, "p6": 0.6, "p7": 0.7}
    assert n13d.DROPOUT_PLAN == "it2d_dropout_r2.md: the dropout-ceiling check, declared before the run, added after the results"
    assert n13d.DROPOUT_REF == {DN: {0: 0.472, 1: 0.0, 2: 0.5, 3: 0.053, 4: 0.0},
                                CX: {0: 0.472, 1: 0.5, 2: 0.386, 3: 0.0, 4: 0.222}}
    assert all(round(CHOSEN_DROPOUT[e][k], 3) == n13d.DROPOUT_REF[e][k] for e in MAIN for k in range(5))
    for arm in ARMS:
        s = n13d.dropout_settings(arm)
        assert {k: v for k, v in s.items() if k != "config"} == {k: v for k, v in n13.batch_settings("bc").items() if k != "config"}
        assert s["config"] == {"source": "chosen (knee-n6a)", "batch_size": "chosen",
                               "dropout": "chosen" if arm == "dc" else P[arm]}
        assert n13d.dropout_config(arm) == {"dropout": "chosen" if arm == "dc" else P[arm]}
        agg._check(json.loads(json.dumps(s)))
    c = heads.Config(dropout=0.2, batch_size=16)
    assert n13d.arm_config(c, "dc") is c  # dc: the chosen configuration object itself
    for arm in ("p5", "p6", "p7"):
        d = n13d.arm_config(c, arm)
        assert d.dropout == P[arm] and {k: v for k, v in d.to_dict().items() if k != "dropout"} == {
            k: v for k, v in c.to_dict().items() if k != "dropout"}
    assert (n13d.DROPOUT_SEED, n13d.DROPOUT_BUDGET_H, n13d.SEARCH_CEILING) == (0, 3.0, 0.5)
    assert set(n13.BATCH_ARMS) == {"bc", "b8", "b4"} and set(n13.ARMS) == {"a1", "base0", "null", "a7"}  # untouched
    for bad in (lambda: n13d.dropout_config("p8"), lambda: n13d.dropout_settings("bc"), lambda: n13d.run_dropout("p4"),
                lambda: n13d.run_dropout("s32"), lambda: n13d.run_dropout("p6", encoders=("vit_s16",)),
                lambda: n13d.run_dropout("p6", rerun=1, folds=(2,)), lambda: n13d.run_dropout("p6", budget_h=None)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):  # n13.run does not know these arms: nothing in n13 was changed
        n13.run("p6")


def test_the_head_applies_the_arm_dropout_on_the_pooled_vectors_in_training_only():
    """Where dropout acts (heads.make_head): one nn.Dropout with p = cfg.dropout, on the pooled per-finding
    vectors before the output weights; in eval mode (the inner score) it does nothing."""
    for p in (0.0, 0.5, 0.6, 0.7):
        m = heads.make_head(heads.Config(dropout=p, attn_hidden=8), 6, 3)
        drops = [x for x in m.modules() if isinstance(x, torch.nn.Dropout)]
        assert len(drops) == 1 and drops[0] is m.drop and m.drop.p == p
        xs = [torch.randn(4, 5, 6) for _ in range(3)]
        m.eval()
        a, b = m(xs), m(xs)
        assert torch.equal(a, b)


def test_check_dropout_ref():
    from knee.nb import n13d

    ok = {(e, k): {"config": {"dropout": CHOSEN_DROPOUT[e][k]}} for e in MAIN for k in range(5)}
    n13d.check_dropout_ref(ok)
    n13d.check_dropout_ref({(DN, 2): {"config": {"dropout": 0.5004}}})  # 0.500 at 3 decimals
    for bad in ({(DN, 2): {"config": {"dropout": 0.4994}}}, {(CX, 4): {"config": {"dropout": 0.25}}},
                {("vit_s16", 0): {"config": {"dropout": 0.0}}}):
        with pytest.raises(ValueError):
            n13d.check_dropout_ref(bad)
    n13d.check_dropout_ref({(CX, 4): {"config": {"dropout": 0.25}}}, ref=None)  # skipped


def test_dc_is_arm_bc_on_the_untouched_bank(worker, tmp_path, monkeypatch, one_thread):
    """dc fits on the worker's cached bank object and returns arm bc's unit score exactly."""
    from knee.nb import n6, n13, n13d

    chosen = n13.chosen_configs(worker / "input")[(DN, 1)]["config"]
    bc = n13._batch_unit(_bc_task(chosen, fold=1), tmp_path / "never_made")
    cached = n6._W["bank"]
    seen = []
    real_fit = heads.fit

    def fit(bank, cfg, *a, **kw):
        seen.append((bank, cfg))
        return real_fit(bank, cfg, *a, **kw)

    monkeypatch.setattr(heads, "fit", fit)
    dc = n13d._dropout_unit(_task("dc", chosen, fold=1), tmp_path / "never_made")
    assert len(seen) == 4 and all(b is cached for b, _ in seen) and n6._W["bank"] is cached
    assert all(c == heads.config_from(chosen) for _, c in seen)
    assert dc["ok"] and dc["fits"] == dc["guard_checks"] == bc["fits"] == 4
    a, b = dc["summary"], bc["summary"]
    assert a["inner_score"] == b["inner_score"] and (a["E"], a["epochs"]) == (b["E"], b["epochs"])
    assert set(a) - set(b) == {"dropout"} and set(b) <= set(a)
    assert {k: v for k, v in a.items() if k not in ("dropout", "seconds")} == {k: v for k, v in b.items() if k != "seconds"}
    assert a["dropout"] == {"config": {"dropout_chosen": 0.0, "dropout_used": 0.0}}
    assert a["config"] == chosen and a["step_ratio"] == 1.0 and isinstance(a["step_ratio"], float)
    assert not (tmp_path / "never_made").exists()


def test_each_arm_passes_its_dropout_to_the_head_and_nothing_else_differs(worker, tmp_path, monkeypatch, one_thread):
    from knee import stage
    from knee.nb import n6, n13, n13d

    chosen_all = n13.chosen_configs(worker / "input")
    cached = n6._bank(DN)
    before = {p: a.clone() for p, a in cached.x.items()}
    ids = {p: id(a) for p, a in cached.x.items()}
    d = stage.StageData(worker / "input")
    fits = []
    real_fit = heads.fit

    def fit(bank, cfg, train_idx, val_idx=None, findings=None, **kw):
        r = real_fit(bank, cfg, train_idx, val_idx, findings, **kw)
        fits.append({"bank": bank, "cfg": cfg.to_dict(), "train": np.asarray(train_idx).copy(),
                     "val": np.asarray(val_idx).copy(), "findings": findings, "head_p": r.model.drop.p,
                     **{k: v for k, v in kw.items() if k != "on_epoch"}})
        return r

    monkeypatch.setattr(heads, "fit", fit)
    for fold in (0, 2):  # chosen dropout 0.472 and 0.5 (the range end: p5 is then dc itself)
        chosen = chosen_all[(DN, fold)]["config"]
        tr, te, _ = d.split(fold)
        out, calls = {}, {}
        for arm in ARMS:
            fits.clear()
            r = n13d._dropout_unit(_task(arm, chosen, fold=fold), tmp_path / "never_made")
            assert r["ok"] and r["fits"] == r["guard_checks"] == 4  # every inner fit passed the leakage guard
            out[arm], calls[arm] = r["summary"], list(fits)
        for arm in ARMS:
            want = chosen["dropout"] if arm == "dc" else P[arm]
            u = out[arm]
            assert u["dropout"] == {"config": {"dropout_chosen": float(chosen["dropout"]), "dropout_used": float(want)}}
            assert u["config"] == {**chosen, "dropout": want}
            for c, base in zip(calls[arm], calls["dc"]):
                assert c["cfg"] == {**chosen, "dropout": want} and c["head_p"] == want  # the head's nn.Dropout
                assert {k: v for k, v in c["cfg"].items() if k != "dropout"} == {k: v for k, v in base["cfg"].items() if k != "dropout"}
                assert c["bank"] is cached and base["bank"] is cached  # the worker's bank, as arm bc
                for p in heads.PLANES:
                    assert torch.equal(c["stats"][p][0], base["stats"][p][0]) and torch.equal(c["stats"][p][1], base["stats"][p][1])
                assert np.array_equal(c["train"], base["train"]) and np.array_equal(c["val"], base["val"])
                assert np.array_equal(c["forbidden"], base["forbidden"]) and c["findings"] == FIND
                assert set(te.tolist()) | set(d.gold_rows().tolist()) == set(c["forbidden"].tolist())
                assert set(c["train"].tolist()) | set(c["val"].tolist()) <= set(tr.tolist())
                same = {k for k in c if k not in ("cfg", "head_p", "bank", "stats", "train", "val", "forbidden")}
                assert {k: c[k] for k in same} == {k: base[k] for k in same}  # seed, patience, schedule, epochs, ...
                assert (c.get("epochs"), c["patience"], c["schedule_epochs"], c.get("max_epochs"), c.get("steps")) == (None, 5, None, None, None)
            assert [c["seed"] for c in calls[arm]] == [0, 1, 2, 3]
            # the unit summary differs from dc's only in the dropout, the scores and the times
            free = {"dropout", "config", "inner_score", "E", "epochs", "seconds"}
            assert {k: v for k, v in u.items() if k not in free} == {k: v for k, v in out["dc"].items() if k not in free}
            assert (u["schedule_epochs"], u["patience"], u["seed"], u["inner_k"], u["device"], u["step_ratio"]) == (30, 5, 0, 4, "cpu", 1.0)
            assert 0.5 < u["inner_score"] <= 1 and round(u["inner_score"], 4) == u["inner_score"]
            agg._check(json.loads(json.dumps(u)))  # every arm's unit summary passes the privacy check
        assert len({out[a]["inner_score"] for a in ("dc", "p6", "p7")}) > 1 or fold == 2  # the arms differ
        if fold == 2:  # chosen 0.5: p5 fits the chosen configuration itself and returns dc's score
            assert out["p5"]["inner_score"] == out["dc"]["inner_score"] and out["p5"]["epochs"] == out["dc"]["epochs"]
    # the worker's cached bank: the same object, the same tensors, the same values
    assert n6._W["bank"] is cached and n6._bank(DN) is cached
    for p, a in cached.x.items():
        assert id(a) == ids[p] and torch.equal(a, before[p])
    assert not (tmp_path / "never_made").exists()  # a unit writes nothing


def test_every_arm_end_to_end_passes_the_privacy_check_and_writes_no_test_fold_file(stage_env, monkeypatch, capfd):
    from knee.nb import n13, n13d

    root, facts = stage_env
    for arm in ARMS:
        monkeypatch.setattr(paths, "WORKING", root / f"working_{arm}")
        out = n13d.run_dropout(arm, root=root / "input", workers=1, encoders=(DN,), folds=(2,), budget_h=1.0)
        printed = capfd.readouterr()
        every = sorted(p for p in paths.WORKING.rglob("*") if p.is_file())
        assert [p.relative_to(paths.WORKING).as_posix() for p in every] == [f"agg_n13_{arm}_inner.json", "run.json"]
        text = printed.out + printed.err + "".join(p.read_text() for p in every)
        assert not [u for u in facts["uids"] if u in text]  # no study identifier anywhere Claude reads
        assert f"STEP n13_{arm}: OK" in printed.out
        assert sorted(p.name for p in (paths.WORKING / "private" / f"n13_{arm}").iterdir()) == ["errors"]  # no unit folder
        rj = json.loads((paths.WORKING / "run.json").read_text())
        c = rj["config"]
        assert rj["step"] == f"n13_{arm}" and (c["arm"], c["batch_size"], c["seed"], c["budget_h"], c["rerun"]) == (arm, "chosen", 0, 1.0, 0)
        assert c["config"] == n13d.dropout_config(arm)
        assert c["test_overrides"] is True and c["test_overrides_reason"] == n13d.DROPOUT_REASON
        assert c["added_after_results"] is True and c["chosen_ref_checked"] is True and c["plan"] == n13d.DROPOUT_PLAN
        assert c["environment"] == out["environment"] == n13.environment()
        back = json.loads((paths.WORKING / f"agg_n13_{arm}_inner.json").read_text())
        assert back == json.loads(json.dumps(out))
        agg._check(back)  # the aggregate of every arm passes the privacy check (it was written through agg.write)
        assert (out["arm"], out["step"], out["status"], out["all_units_ok"]) == (arm, f"n13_{arm}", "ok", True)
        assert out["plan"] == n13d.DROPOUT_PLAN and out["settings"] == n13d.dropout_settings(arm)
        assert out["leakage_assertion_passed"] is True and out["chosen_ref_checked"] is True and out["device_names"] == ["cpu"]
        assert out["test_fold_use"].startswith("none") and out["seconds"] > 0 and "knee_commit" in out
        assert out["units_finished"] == [f"{DN}_f2"] and out["units_missing"] == []
        u = out["units"][f"{DN}_f2"]
        want = 0.5 if arm == "dc" else P[arm]
        assert u["config"] == {**CHOSEN, "dropout": want, "batch_size": 16} and u["batch_chosen"] == 16
        assert u["dropout"] == {"config": {"dropout_chosen": 0.5, "dropout_used": want}}
        assert u["stage_a_best_inner_score"] == n13.CHOSEN_REF[DN][2][1]
        m = n13.merged(out)  # the reader of the rule takes it
        assert m["arm"] == arm and sorted(m["units"]) == [(DN, 2)]
        leak = copy.deepcopy(back)
        leak["units"][f"{DN}_f2"]["n_fits"] = 4  # outside "config" a 4 is a small count
        with pytest.raises(agg.UnsafeSummary):
            agg._check(leak)


def test_a_knee_n6a_with_another_dropout_is_refused_before_any_fit(tmp_path_factory, monkeypatch, capfd):
    from tests.synth_stage import make_stage

    from knee.nb import n13d

    root = tmp_path_factory.mktemp("it2dwrong")
    make_stage(root / "input", S=8)
    wrong = copy.deepcopy(CHOSEN_DROPOUT)
    wrong[CX][4] = 0.25
    _write_stage_a(root / "input", dropout=wrong)
    monkeypatch.setattr(paths, "WORKING", root / "working")
    called = []
    monkeypatch.setattr(heads, "fit", lambda *a, **k: called.append(1))
    with pytest.raises(BaseException):
        n13d.run_dropout("p6", root=root / "input", workers=1, folds=(4,), budget_h=1.0)
    assert "STEP n13_p6: FAILED - ValueError" in capfd.readouterr().out and not called
    assert not (paths.WORKING / "agg_n13_p6_inner.json").exists()


def test_soft_stop_gpu_refusal_and_the_summary_flags(stage_env, worker, tmp_path, monkeypatch, capfd, one_thread):
    from knee.nb import n6, n13, n13d

    root, _ = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_p7_stop")
    out = n13d.run_dropout("p7", root=root / "input", workers=2, encoders=(DN,), folds=(0, 1), budget_h=0.0)
    assert "STEP n13_p7: OK" in capfd.readouterr().out  # a stopped session is not a failed step
    assert (out["status"], out["all_units_ok"], out["units_finished"]) == ("partial", False, [])
    assert out["units_missing"] == [f"{DN}_f0", f"{DN}_f1"] and out["inner_scores"] == {}
    assert all(u == {"ok": False, "stopped": True, "error": None} for u in out["units"].values())
    assert json.loads((paths.WORKING / "agg_n13_p7_inner.json").read_text()) == json.loads(json.dumps(out))
    # a GPU session is refused before any data is read
    monkeypatch.setattr(paths, "WORKING", root / "working_p6_gpu")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(BaseException):
        n13d.run_dropout("p6", root=root / "input", workers=1, encoders=(DN,), folds=(0,), budget_h=1.0)
    assert "STEP n13_p6: FAILED - ValueError" in capfd.readouterr().out
    assert not (paths.WORKING / "agg_n13_p6_inner.json").exists()
    monkeypatch.undo()
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    # a unit past its deadline does not start; a deadline passing during the second fit ends it with its epoch
    chosen = n13.chosen_configs(root / "input")[(DN, 0)]["config"]
    assert n13d._dropout_unit(_task("p6", chosen, deadline=0.0), tmp_path / "x") == {"task": f"{DN}_f0", "ok": False, "stopped": True}
    task = _task("p6", chosen)
    real_fit, runs = heads.fit, []

    def fit(*a, **kw):
        if runs:
            task["deadline"] = 0.0  # the budget ends while the second fit runs
        r = real_fit(*a, **kw)
        runs.append(r.epochs_run)
        return r

    monkeypatch.setattr(heads, "fit", fit)
    r = n13d._dropout_unit(task, tmp_path / "x")
    assert r == {"task": f"{DN}_f0", "ok": False, "stopped": True} and len(runs) == 2 and runs[1] == 1
    monkeypatch.setattr(heads, "fit", real_fit)
    done = n13d._dropout_unit(_task("p6", chosen), tmp_path / "x")
    s = n13d.dropout_summary("p6", "n13_p6", {f"{DN}_f0": done, f"{DN}_f1": r},
                             [f"{DN}_f0", f"{DN}_f1", f"{DN}_f2"], n13.environment(), (DN,), budget_h=3.0)
    assert (s["status"], s["units_finished"]) == ("partial", [f"{DN}_f0"])  # the finished unit is written
    assert s["units_missing"] == [f"{DN}_f1", f"{DN}_f2"] and s["leakage_assertion_passed"] is True
    assert s["inner_scores"] == {DN: {0: done["summary"]["inner_score"]}}
    agg.write("n13_p6_inner", s, folder=tmp_path / "agg")
    failed = {"task": f"{DN}_f1", "ok": False, "error": "LeakageError at heads.py:209 in _check_guard()"}
    s2 = n13d.dropout_summary("p6", "n13_p6", {f"{DN}_f0": done, f"{DN}_f1": failed}, [f"{DN}_f0", f"{DN}_f1"], None, (DN,))
    assert s2["leakage_assertion_passed"] is False and s2["units"][f"{DN}_f1"]["stopped"] is False
    s3 = n13d.dropout_summary("p6", "n13_p6", {f"{DN}_f0": {**done, "guard_checks": 3}}, [f"{DN}_f0"], None, (DN,))
    assert s3["leakage_assertion_passed"] is False  # a fit that did not pass the guard
    s4 = n13d.dropout_summary("p6", "n13_p6", {}, [f"{DN}_f0"], None, (DN,), ref_checked=False)
    assert s4["chosen_ref_checked"] is False


# ---------------------------------------------------------------- decide_dropout

ENV = {"kaggle_image_git_commit": "e1dbc17d", "kaggle_image_build_date": "20261117-122508", "python": "3.13.18",
       "torch": "2.11.0+cpu", "cuda": None, "cudnn": None, "device_name": "cpu"}
BC_COMMIT = "39c3c6fc7626e047ed643d5253e8ce9e662302af"  # the commit arm bc ran at
# iteration 2's inner scores (n13_numbers.md): s = 0.0030
BASE0 = {"convnext_tiny": [0.7240, 0.7444, 0.7393, 0.7476, 0.7418], "dinov2_vits14": [0.7227, 0.7398, 0.7327, 0.7398, 0.7367]}
NULL = {"convnext_tiny": [0.7192, 0.7463, 0.7432, 0.7472, 0.7381], "dinov2_vits14": [0.7188, 0.7390, 0.7327, 0.7404, 0.7318]}
DC = {DN: [0.7243, 0.7419, 0.7364, 0.7433, 0.7368], CX: [0.7265, 0.7463, 0.7459, 0.7522, 0.7463]}  # bc's scores
ENDING = "One seed. The other head settings were not re-tuned. Differences under 0.005 cannot be shown by this check."
NOT_MET = ("This does not meet the declared rule (more than 0.005 at 4 decimals, and 4 of 5 folds), so it is not "
           "counted as an effect.")
RANGE = "The searched dropout range ended at 0.5 and was not extended."
NONE_LINE = "Raising dropout above the searched range did not score higher by the declared margin."
DOWN4 = [-0.008, -0.007, -0.006, -0.009, 0.001]  # mean -0.0058, lower in 4 of 5 folds
DOWN3 = [-0.020, -0.020, -0.010, 0.010, 0.010]  # mean -0.0060, lower in 3 of 5 folds
UP4 = [0.008, 0.007, 0.006, 0.009, -0.001]  # mean +0.0058, higher in 4 of 5 folds
UP3 = [0.020, 0.020, 0.010, -0.010, -0.010]
TIE_DOWN, TIE_UP = [-0.005] * 5, [0.005] * 5  # exactly 0.0050: inside "no difference shown"
SMALL_DOWN = [-0.004, -0.002, 0.001, -0.003, -0.006]  # mean -0.0028
ZERO = [0.001, -0.001, 0.0, 0.0, 0.0]  # mean exactly 0
ZERO_FOLDS = [-0.020, -0.010, -0.005, 0.0, 0.0]  # mean -0.0070, but lower in 3 folds only: a 0.0000 fold is neither
FLAT = [0.0] * 5


def _it2(arm, scores, env=ENV):
    from knee.nb import n13

    units = {f"{e}_f{k}": {"ok": True, "trial0_inner_score": v, "head_sha16": "0" * 16}
             for e, vals in scores.items() for k, v in enumerate(vals)}
    return {"arm": arm, "step": f"n13_{arm}", "settings": n13.settings(n13.arm_spec(arm)), "environment": env,
            "knee_commit": "9ecea01", "units": units}


def _units(arm, diffs, stopped, only):
    """Unit summaries as _dropout_unit writes them (arm None: bc's, without the dropout block)."""
    from knee.nb import n13

    units, scores, ids = {}, {}, []
    for e in MAIN:
        for k in range(5):
            uid = f"{e}_f{k}"
            if only is not None and (e, k) not in only:
                continue
            ids.append(uid)
            if (e, k) in stopped:
                units[uid] = {"ok": False, "stopped": True, "error": None}
                continue
            b, best = n13.CHOSEN_REF[e][k]
            chosen = CHOSEN_DROPOUT[e][k]
            used = chosen if arm in (None, "dc") else P[arm]
            v = round(DC[e][k] + ((diffs or {}).get(e) or FLAT)[k], 4)
            u = {"ok": True, "inner_score": v, "E": 4 if arm != "p7" else 6, "epochs": [9, 9, 8, 3], "inner_k": 4,
                 "seed": 0, "schedule_epochs": 30, "patience": 5, "device": "cpu", "device_name": "cpu",
                 "config": {**CHOSEN, "dropout": used, "batch_size": b}, "batch_chosen": b}
            if arm is not None:
                u["dropout"] = {"config": {"dropout_chosen": float(chosen), "dropout_used": float(used)}}
            u.update({"steps_per_epoch": int(5 * round(3000 / b / 5)), "steps_per_epoch_chosen": int(5 * round(3000 / b / 5)),
                      "step_ratio": 1.0, "stage_a_best_inner_score": best, "chosen_sha16": "0" * 16, "seconds": 100.0})
            units[uid] = u
            scores.setdefault(e, {})[k] = v
    return units, scores, ids


def _s(arm, diffs=None, env=ENV, stopped=(), step=None, only=None, commit="ARM"):
    """A dropout-arm summary shaped as dropout_summary writes it: DC plus diffs {encoder: [5 differences]}."""
    from knee.nb import n13d

    units, scores, ids = _units(arm, diffs, stopped, only)
    done = [u for u in ids if units[u]["ok"]]
    return {"arm": arm, "step": step or f"n13_{arm}", "plan": n13d.DROPOUT_PLAN, "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": n13d.DROPOUT_REASON, "encoders": list(MAIN),
            "settings": n13d.dropout_settings(arm), "chosen_ref_checked": True, "environment": env,
            "knee_commit": n13d.DROPOUT_ARM_COMMIT if commit == "ARM" else commit,
            "status": "ok" if len(done) == len(ids) else "partial", "all_units_ok": len(done) == len(ids),
            "units_finished": done, "units_missing": [u for u in ids if not units[u]["ok"]],
            "leakage_assertion_passed": True, "device_names": ["cpu"], "score": n13d.DROPOUT_SCORE, "budget_h": 3.0,
            "seconds": 2000.0, "inner_scores": scores, "units": units}


def _bc(diffs=None, env=ENV, stopped=()):
    """Arm bc's summary as n13.batch_summary wrote it (commit 39c3c6f): DC plus diffs."""
    from knee.nb import n13

    units, scores, ids = _units(None, diffs, stopped, None)
    done = [u for u in ids if units[u]["ok"]]
    return {"arm": "bc", "step": "n13_bc", "plan": n13.BATCH_PLAN, "added_after_results": True, "encoders": list(MAIN),
            "settings": n13.batch_settings("bc"), "chosen_ref_checked": True, "environment": env, "knee_commit": BC_COMMIT,
            "status": "ok", "units_finished": done, "leakage_assertion_passed": True, "inner_scores": scores, "units": units}


def _decide(p6=None, p7=None, p5=None, dc="base", bc="same", **kw):
    """decide_dropout on the given difference tables {encoder: [5]} or ready summaries; None: the arm did not run."""
    from knee.nb import n13d

    def arm(name, x):
        return _s(name, x) if isinstance(x, dict) and "arm" not in x else x

    return n13d.decide_dropout(_s("dc") if dc == "base" else dc, arm("p6", p6), arm("p7", p7),
                               _bc() if bc == "same" else bc, _it2("base0", BASE0), _it2("null", NULL),
                               p5=arm("p5", p5), **kw)


def _strings(o):
    if isinstance(o, dict):
        return [s for k, v in o.items() for s in [str(k), *_strings(v)]]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return [o] if isinstance(o, str) else []


def _checked(r, folder):
    """The record passes the aggregate checker and holds no long string and no forbidden word."""
    from knee.nb import n13d

    p = agg.write("n13_dropout_decide", r, folder=folder)
    back = json.loads(p.read_text())
    assert max(len(x) for x in _strings(back)) <= 200
    for line in n13d.dropout_slide_sentences(r):
        assert "optimal" not in line.lower() and "best" not in line.lower()
    return back


def test_decide_dropout_verdicts_by_the_declared_rule(tmp_path):
    from knee.nb import n13, n13d

    assert (n13d.HIGHER, n13d.LOWER, n13d.NO_DIFF, n13d.NO_VERDICT) == (
        "higher dropout scored higher", "higher dropout scored lower", "no difference shown", "no verdict")
    assert n13d.VERDICT_ARMS == ("p6", "p7") and n13d.RERUN_TOL == 0.0005 and n13d.SPLIT_AT == 0.4
    r = _decide(p6={DN: UP4, CX: DOWN4}, p7={DN: DOWN3, CX: TIE_UP}, p5={DN: SMALL_DOWN, CX: FLAT})
    assert r["status"] == "decided" and r["added_after_results"] is True and r["same_knee_commit"] is True
    assert round(r["s"], 4) == 0.0030 == r["s_declared"] and r["threshold"] == 0.005 and r["min_gain"] == 0.005
    assert abs(r["threshold_unrounded"] - 0.005) < 1e-12  # 2 s / sqrt(5) = 0.0027 does not bind
    assert r["plan"] == n13d.DROPOUT_PLAN and r["declaration"].startswith("it2d_dropout_r2.md")
    assert r["knee_commits"] == [n13d.DROPOUT_ARM_COMMIT] and r["knee_commit_is_declared"] is True
    assert r["leakage_assertion_passed"] is True and r["chosen_ref_checked"] is True and r["replaced_runs"] == {}
    e = r["encoders"]
    assert set(e) == set(MAIN) and all(set(v) == {"p6", "p7"} for v in e.values())  # never a verdict for p5

    def got(enc, arm):
        x = e[enc][arm]
        return x["verdict"], x["gain"], len(x["lower"]["folds"]), len(x["higher"]["folds"])

    assert got(DN, "p6") == ("higher dropout scored higher", 0.0058, 1, 4)
    assert got(CX, "p6") == ("higher dropout scored lower", -0.0058, 4, 1)
    assert got(DN, "p7") == ("no difference shown", -0.006, 3, 2)  # the difference alone is not enough
    assert got(CX, "p7") == ("no difference shown", 0.005, 0, 5)  # a 4-decimal tie at 0.0050 is inside
    assert e[DN]["p6"]["folds_up"] == 4 and e[DN]["p6"]["folds"] == [0, 1, 2, 3, 4]
    # p5 given: p6 minus p5 (0.0058 + 0.0028) meets the rule too: the ceiling mattered
    assert e[DN]["p6"]["ceiling"] == "the ceiling mattered" and e[DN]["p6"]["minus_p5"]["meets_higher"] is True
    assert e[DN]["p6"]["minus_p5"]["gain"] == 0.0086 and e[CX]["p6"]["ceiling"] is None
    # section 6's sentences
    w = {(enc, arm): (e[enc][arm]["words"], e[enc][arm]["words_extra"], e[enc][arm]["words_limits"]) for enc in MAIN for arm in ("p6", "p7")}
    assert w[(DN, "p6")] == ("Dropout 0.6 scored 0.0058 higher than the chosen dropout on inner validation "
                             "(DINOv2, higher in 4 of 5 folds).", "The head search was not rerun.", None)
    assert w[(CX, "p6")] == ("Dropout 0.6 scored 0.0058 lower than the chosen dropout on inner validation "
                             "(ConvNeXt-T, lower in 4 of 5 folds).", None, None)
    assert w[(DN, "p7")] == ("Dropout 0.7 scored 0.0060 lower than the chosen dropout on inner validation "
                             "(DINOv2, lower in 3 of 5 folds).", NOT_MET, None)
    assert w[(CX, "p7")] == ("Dropout 0.7 scored 0.0050 higher than the chosen dropout on inner validation "
                             "(ConvNeXt-T, higher in 5 of 5 folds).", NOT_MET, None)
    lines = n13d.dropout_slide_sentences(r)
    assert lines[:4] == [" ".join(p for p in w[(enc, arm)] if p) for enc in MAIN for arm in ("p6", "p7")]
    assert lines[4:] == ["Open point: the ceiling mattered (DINOv2): dropout 0.6 scored higher than the chosen dropout "
                         "and than dropout 0.5, the end of the searched range.", ENDING] == [*r["summary_words"], r["words_ending"]]
    assert not any(ENDING in x for x in lines[:-1])  # the common ending once, as the last line
    # descriptive: p5 minus dc, the split by chosen dropout, the rerun check, the table, the medians
    pm = r["p5_minus_dc"]
    assert pm["used_in_rule"] is False and pm["p5_ran"] is True
    assert (pm[DN]["gain"], pm[DN]["all_folds"], pm[CX]["gain"]) == (-0.0028, True, 0.0)
    sp = r["by_chosen_dropout"]
    assert sp["used_in_rule"] is False and sp["config"] == {"split_at": 0.4}
    assert sp["p6"]["all"]["high"]["units"] == [f"{DN}_f0", f"{DN}_f2", f"{CX}_f0", f"{CX}_f1"]  # 0.472, 0.5, 0.472, 0.5
    assert sp["p6"]["all"]["rest"]["units"] == [f"{DN}_f1", f"{DN}_f3", f"{DN}_f4", f"{CX}_f2", f"{CX}_f3", f"{CX}_f4"]
    assert sp["p6"][DN]["high"]["mean_minus_dc"] == round((0.008 + 0.006) / 2, 4)
    assert sp["p6"][CX]["rest"]["mean_minus_dc"] == round((-0.006 - 0.009 + 0.001) / 3, 4)
    assert sp["p7"]["all"]["high"]["mean_minus_dc"] == round((-0.020 - 0.010 + 0.005 + 0.005) / 4, 4)
    rc = r["rerun_check"]
    assert (rc["used_in_rule"], rc["identical"], rc["units_outside"], rc["words"], rc["max_abs_difference"]) == (False, True, [], None, 0.0)
    assert r["dc_reruns_bc"] is True and len(rc["units_checked"]) == 10 and rc["bc_knee_commits"] == [BC_COMMIT]
    assert rc["bc_env_differences"] == [] and r["words_rerun"] is None
    u = r["units"][f"{DN}_f0"]
    assert u["batch_chosen"] == 33 and u["config"] == {"dropout_chosen": CHOSEN_DROPOUT[DN][0]}
    assert u["dc"] == {"inner_score": 0.7243, "E": 4, "minus_dc": None} and u["p6"]["minus_dc"] == 0.008
    assert (u["bc_inner_score"], u["dc_minus_bc"]) == (0.7243, 0.0)
    assert r["median_best_epoch"]["p7"] == {"all": 6.0, DN: 6.0, CX: 6.0} and r["median_best_epoch"]["dc"]["all"] == 4.0
    assert r["mean_inner_score"][DN]["dc"] == round(float(np.mean(DC[DN])), 4)
    assert r["arms"]["p6"]["config"] == {"dropout": 0.6} and r["arms"]["dc"]["config"] == {"dropout": "chosen"}
    back = _checked(r, tmp_path)
    assert back["encoders"][DN]["p6"]["verdict"] == n13d.HIGHER
    # the summaries as files give the same record; a rerun fills a stopped unit
    tabs = {"p6": {DN: UP4, CX: DOWN4}, "p7": {DN: DOWN3, CX: TIE_UP}, "p5": {DN: SMALL_DOWN, CX: FLAT}}
    f = {a: json.loads(json.dumps(_s(a, tabs.get(a)))) for a in ARMS}
    it2 = [json.loads(json.dumps(x)) for x in (_bc(), _it2("base0", BASE0), _it2("null", NULL))]
    again = n13d.decide_dropout(f["dc"], f["p6"], f["p7"], *it2, p5=f["p5"])
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(r))
    part = _s("p7", tabs["p7"], stopped={(DN, 2)})
    fill = _s("p7", tabs["p7"], only={(DN, 2)}, step="n13_p7_r")
    rr = _decide(p6=tabs["p6"], p7=[part, fill], p5=tabs["p5"])
    assert rr["encoders"] == r["encoders"] and rr["arms"]["p7"]["steps"] == ["n13_p7", "n13_p7_r"]
    assert n13.MIN_FOLDS_UP == 4


def test_a_higher_score_is_compared_with_p5_and_a_missing_p5_blocks_nothing(tmp_path):
    from knee.nb import n13d

    up = {DN: UP4, CX: UP4}
    # p5 scores as p6 does: p6 minus p5 is 0: inside the searched range
    inside = _decide(p6=up, p7={}, p5=up)
    x = inside["encoders"][DN]["p6"]
    assert (x["verdict"], x["ceiling"], x["minus_p5"]["gain"], x["minus_p5"]["meets_higher"]) == (n13d.HIGHER, "inside the searched range", 0.0, False)
    assert inside["summary_words"] == [
        "Open point: dropout 0.6 scored higher than the chosen dropout but not than dropout 0.5, inside the searched range (DINOv2).",
        "Open point: dropout 0.6 scored higher than the chosen dropout but not than dropout 0.5, inside the searched range (ConvNeXt-T)."]
    assert inside["encoders"][DN]["p6"]["words_extra"] == "The head search was not rerun."
    _checked(inside, tmp_path / "inside")
    # no p5: the verdicts stand, and the open point says the cause is not shown
    nop5 = _decide(p6=up, p7={DN: UP4})
    assert nop5["status"] == "decided" and nop5["arms"]["p5"] is None and nop5["p5_minus_dc"]["p5_ran"] is False
    assert [nop5["encoders"][e]["p6"]["verdict"] for e in MAIN] == [n13d.HIGHER] * 2
    assert nop5["encoders"][DN]["p6"]["ceiling"] == "p5 not available" and nop5["encoders"][DN]["p6"]["minus_p5"] is None
    assert nop5["summary_words"] == [
        "Open point: dropout 0.6 scored higher than the chosen dropout (DINOv2). This check does not show whether the "
        "range end or the search's pick inside the range is the cause.",
        "Open point: dropout 0.7 scored higher than the chosen dropout (DINOv2). This check does not show whether the "
        "range end or the search's pick inside the range is the cause.",
        "Open point: dropout 0.6 scored higher than the chosen dropout (ConvNeXt-T). This check does not show whether the "
        "range end or the search's pick inside the range is the cause."]
    assert n13d.dropout_slide_sentences(nop5)[-1] == ENDING and len(n13d.dropout_slide_sentences(nop5)) == 8
    assert nop5["p5_minus_dc"][DN] == {"folds": [], "per_fold": {}, "lower": {"folds": []}, "higher": {"folds": []},
                                       "all_folds": False, "gain": None, "meets_higher": None, "meets_lower": None}
    _checked(nop5, tmp_path / "nop5")
    # a partial p5 (4 folds): the same open point, and the verdicts stand
    part = _decide(p6=up, p7={}, p5=_s("p5", stopped={(CX, 3)}))
    assert part["status"] == "decided" and part["encoders"][CX]["p6"]["ceiling"] == "p5 not available"
    assert part["encoders"][DN]["p6"]["ceiling"] == "the ceiling mattered"  # DINOv2's p5 is whole
    assert part["encoders"][CX]["p6"]["minus_p5"]["folds"] == [0, 1, 2, 4]
    # p6 minus p5 above the margin but higher in 3 folds only: not the ceiling
    three = _decide(p6=up, p7={}, p5={DN: [0.02, 0.02, 0.0, 0.0, -0.06]})
    v5 = three["encoders"][DN]["p6"]["minus_p5"]
    assert (v5["gain"], len(v5["higher"]["folds"]), v5["meets_higher"]) == (0.0098, 3, False)
    assert three["encoders"][DN]["p6"]["ceiling"] == "inside the searched range"
    # p5 never gets a verdict of its own, even when it differs from dc by a lot
    big = _decide(p6={}, p7={}, p5={DN: [0.02] * 5})
    assert big["summary_words"] == [NONE_LINE] and big["p5_minus_dc"][DN]["meets_higher"] is True
    assert all(big["encoders"][e][a]["verdict"] == n13d.NO_DIFF for e in MAIN for a in ("p6", "p7"))


def test_decide_dropout_no_difference_sentences_ties_and_the_summary_line(tmp_path):
    from knee.nb import n13d

    neg0 = [-0.00004, 0.0, 0.0, 0.0, 0.0]  # rounds to 0.0000: "the same", never "-0.0000"
    r = _decide(p6={DN: ZERO, CX: neg0}, p7={DN: TIE_DOWN, CX: ZERO_FOLDS})
    e = r["encoders"]
    assert r["status"] == "decided" and all(e[enc][a]["verdict"] == "no difference shown" for enc in MAIN for a in ("p6", "p7"))
    assert (e[DN]["p6"]["words"], e[DN]["p6"]["words_extra"], e[DN]["p6"]["gain"]) == (
        "Dropout 0.6 scored the same as the chosen dropout on inner validation (DINOv2).", None, 0.0)
    assert e[CX]["p6"]["words"] == "Dropout 0.6 scored the same as the chosen dropout on inner validation (ConvNeXt-T)."
    assert str(e[CX]["p6"]["gain"]) == "0.0" and e[CX]["p6"]["lower"] == {"folds": []}  # -0.00004 is 0.0000 at 4 decimals
    assert (e[DN]["p7"]["gain"], e[DN]["p7"]["words"], e[DN]["p7"]["words_extra"]) == (
        -0.005, "Dropout 0.7 scored 0.0050 lower than the chosen dropout on inner validation (DINOv2, lower in 5 of 5 folds).", NOT_MET)
    assert e[CX]["p7"]["gain"] == -0.007 and e[CX]["p7"]["lower"] == {"folds": [0, 1, 2]}  # folds at 0.0000 are neither
    assert e[CX]["p7"]["per_fold"] == {0: -0.02, 1: -0.01, 2: -0.005, 3: 0.0, 4: 0.0} and e[CX]["p7"]["higher"] == {"folds": []}
    assert e[CX]["p7"]["words"] == ("Dropout 0.7 scored 0.0070 lower than the chosen dropout on inner validation "
                                    "(ConvNeXt-T, lower in 3 of 5 folds).")
    lines = n13d.dropout_slide_sentences(r)
    assert len(lines) == 6 and lines[4] == NONE_LINE == r["summary_words"][0] and lines[5] == ENDING
    assert lines[0] == e[DN]["p6"]["words"] and lines[1] == f"{e[DN]['p7']['words']} {NOT_MET}"
    _checked(r, tmp_path / "none")
    # "scored lower" is not "scored higher": the summary line stands; four folds lower and one at 0.0000 is 4 of 5
    four = [-0.010, -0.010, -0.010, -0.010, 0.0]
    r2 = _decide(p6={DN: four, CX: [-x for x in four]}, p7={DN: SMALL_DOWN, CX: UP3})
    assert (r2["encoders"][DN]["p6"]["verdict"], r2["encoders"][CX]["p6"]["verdict"]) == (n13d.LOWER, n13d.HIGHER)
    assert "(DINOv2, lower in 4 of 5 folds)" in r2["encoders"][DN]["p6"]["words"]
    assert "(ConvNeXt-T, higher in 4 of 5 folds)" in r2["encoders"][CX]["p6"]["words"]
    assert r2["encoders"][CX]["p7"]["verdict"] == n13d.NO_DIFF and "higher in 3 of 5 folds" in r2["encoders"][CX]["p7"]["words"]
    assert len(r2["summary_words"]) == 1 and r2["summary_words"][0].startswith("Open point: dropout 0.6 scored higher than the chosen dropout (ConvNeXt-T).")
    r3 = _decide(p6={DN: DOWN4, CX: DOWN4}, p7={DN: DOWN4, CX: DOWN4})
    assert all(r3["encoders"][e][a]["verdict"] == n13d.LOWER for e in MAIN for a in ("p6", "p7"))
    assert r3["summary_words"] == [NONE_LINE] and n13d.dropout_slide_sentences(r3)[-1] == ENDING
    _checked(r3, tmp_path / "lower")


def test_a_mean_of_minus_0_00504_is_a_tie_at_4_decimals():
    near = [-0.0051, -0.0051, -0.0050, -0.0050, -0.0050]  # the sum is -0.0252
    r = _decide(p6={DN: near, CX: [-x for x in near]}, p7={DN: [-0.0051] * 5})
    x = r["encoders"][DN]["p6"]
    assert abs(x["gain_unrounded"] - (-0.00504)) < 1e-9 and x["gain"] == -0.005 and x["lower"] == {"folds": [0, 1, 2, 3, 4]}
    assert x["verdict"] == "no difference shown" and x["words_extra"] == NOT_MET
    y = r["encoders"][CX]["p6"]
    assert abs(y["gain_unrounded"] - 0.00504) < 1e-9 and (y["verdict"], y["gain"]) == ("no difference shown", 0.005)
    assert "scored 0.0050 higher" in y["words"]
    z = r["encoders"][DN]["p7"]  # -0.0051 is beyond the tie
    assert (z["verdict"], z["gain"]) == ("higher dropout scored lower", -0.0051)


def test_decide_dropout_gives_no_verdict_with_fewer_than_5_folds(tmp_path):
    from knee.nb import n13d

    down = {DN: DOWN4, CX: DOWN4}
    r = _decide(p6=_s("p6", down, stopped={(CX, 3), (CX, 4)}), p7=None, p5=_s("p5", down, stopped={(DN, 0)}))
    assert r["status"] == "partial"
    c6 = r["encoders"][CX]["p6"]
    assert (c6["verdict"], c6["gain"], c6["folds_up"], c6["folds"]) == ("no verdict", None, None, [0, 1, 2])
    assert c6["reason"] == "fewer than 5 folds finished in the arm and in dc" and "gain_unrounded" not in c6
    assert c6["per_fold"] == {0: -0.008, 1: -0.007, 2: -0.006} and c6["lower"] == {"folds": [0, 1, 2]}
    assert (c6["words"], c6["words_extra"], c6["words_limits"]) == (
        "Dropout 0.6, ConvNeXt-T: no verdict (fewer than 5 folds finished in the arm and in dc).", RANGE,
        "Finished: dc 5 of 5 folds, p6 3 of 5 folds.")
    assert r["encoders"][DN]["p6"]["verdict"] == n13d.LOWER  # the other encoder is whole
    p7 = r["encoders"][DN]["p7"]
    assert (p7["verdict"], p7["reason"]) == ("no verdict", "the arm did not run") and r["arms"]["p7"] is None
    lines = n13d.dropout_slide_sentences(r)
    assert lines[1] == f"Dropout 0.7, DINOv2: no verdict (the arm did not run). {RANGE} Finished: dc 5 of 5 folds, p7 0 of 5 folds."
    assert lines[4:] == ["No comparison with a verdict scored higher by the declared margin; 3 of 4 comparisons have no verdict.",
                         ENDING]
    assert r["p5_minus_dc"][DN]["folds"] == [1, 2, 3, 4] and r["units"][f"{CX}_f3"]["p6"] == {"inner_score": None, "E": None, "minus_dc": None}
    assert r["mean_inner_score"][CX]["p6"] is None and r["mean_inner_score"][DN]["p6"] is not None
    _checked(r, tmp_path / "part")
    # a "scored higher" beside a missing verdict: the open point only
    rh = _decide(p6={CX: UP4}, p7=None)
    assert len(rh["summary_words"]) == 2 and rh["summary_words"][0].startswith("Open point: dropout 0.6 scored higher")
    assert rh["summary_words"][1].endswith("comparisons have no verdict.")
    # all four without a verdict (dc partial in both encoders): only that line, no rerun line, no common ending
    rz = _decide(p6={}, p7={}, dc=_s("dc", stopped={(DN, 0), (CX, 4)}), bc=_bc({DN: [0, 0.002, 0, 0, 0]}))
    assert rz["status"] == "partial" and rz["words_ending"] is None and rz["words_rerun"] is None
    assert rz["summary_words"] == ["No comparison with a verdict scored higher by the declared margin; 4 of 4 comparisons have no verdict."]
    assert rz["rerun_check"]["units_outside"] == [f"{DN}_f1"]  # still in the record
    lz = n13d.dropout_slide_sentences(rz)
    assert len(lz) == 5 and ENDING not in lz and not any("cannot be shown" in x or "Rerun noise" in x for x in lz)
    assert lz[0] == (f"Dropout 0.6, DINOv2: no verdict (fewer than 5 folds finished in the arm and in dc). {RANGE} "
                     "Finished: dc 4 of 5 folds, p6 5 of 5 folds.")
    _checked(rz, tmp_path / "none")
    # a fold missing in the base takes the verdict from both arms for that encoder
    rb = _decide(p6=down, p7=down, dc=_s("dc", stopped={(DN, 0)}))
    assert [rb["encoders"][DN][a]["verdict"] for a in ("p6", "p7")] == ["no verdict"] * 2
    assert rb["encoders"][CX]["p7"]["verdict"] == n13d.LOWER and len(rb["rerun_check"]["units_checked"]) == 9
    # dc did not run: no verdict at all
    r0 = _decide(p6=down, p7=down, p5=down, dc=None)
    assert r0["status"] == "no_base" and "units" not in r0 and r0["words_ending"] is None
    assert r0["summary_words"] == ["No comparison with a verdict scored higher by the declared margin; 4 of 4 comparisons have no verdict."]
    recs = [v[a] for v in r0["encoders"].values() for a in ("p6", "p7")]
    assert all((x["verdict"], x["gain"], x["reason"]) == ("no verdict", None, "the dc arm (the base) did not run") for x in recs)
    l0 = n13d.dropout_slide_sentences(r0)
    assert l0[0] == f"Dropout 0.6, DINOv2: no verdict (the dc arm (the base) did not run). {RANGE} Finished: dc 0 of 5 folds, p6 5 of 5 folds."
    assert len(l0) == 5 and not any("cannot be shown" in x for x in l0)
    _checked(r0, tmp_path / "no_base")


def _all_none(r, status, reason):
    assert r["status"] == status and "units" not in r and "rerun_check" not in r and r["words_ending"] is None
    assert r["summary_words"] == ["No comparison with a verdict scored higher by the declared margin; 4 of 4 comparisons have no verdict."]
    recs = [v[a] for v in r["encoders"].values() for a in ("p6", "p7")]
    assert len(recs) == 4
    assert all((x["verdict"], x["gain"], x["folds_up"], x["reason"]) == ("no verdict", None, None, reason) for x in recs)
    assert all(x["words"].endswith(f"no verdict ({reason}).") and x["words_extra"] == RANGE for x in recs)
    assert all(x["words_limits"].startswith("Finished: dc ") for x in recs)
    return r


def test_decide_dropout_statuses_leakage_chosen_ref_commit_environment(tmp_path):
    from knee.nb import n13d

    down = {DN: DOWN4, CX: DOWN4}
    other = {**ENV, "kaggle_image_build_date": "20261124-000000", "torch": "2.12.0+cpu"}

    def edited(arm, **fields):
        return {**_s(arm, down if arm != "dc" else None), **fields}

    for bad in (False, None):  # the leakage flag of every arm summary given (p5's and a rerun's too)
        r = _all_none(_decide(p6=edited("p6", leakage_assertion_passed=bad), p7=down),
                      "leakage_check_failed", "an arm's leakage check did not pass")
        assert r["leakage_failed_steps"] == ["n13_p6"] and r["leakage_assertion_passed"] is False
    _checked(r, tmp_path / "leak")
    assert _decide(p6=down, p7=down, p5=edited("p5", leakage_assertion_passed=False))["leakage_failed_steps"] == ["n13_p5"]
    assert _decide(p6=down, p7=down, dc=edited("dc", leakage_assertion_passed=False))["status"] == "leakage_check_failed"
    part = _s("p7", down, stopped={(DN, 2)})
    fill = {**_s("p7", down, only={(DN, 2)}, step="n13_p7_r"), "leakage_assertion_passed": False}
    assert _decide(p6=down, p7=[part, fill])["leakage_failed_steps"] == ["n13_p7_r"]
    for bad in (False, None):
        r = _all_none(_decide(p6=down, p7=edited("p7", chosen_ref_checked=bad)),
                      "chosen_ref_not_checked", "an arm did not check its configurations against 25 Nov")
        assert r["chosen_ref_unchecked_steps"] == ["n13_p7"] and r["chosen_ref_checked"] is False
    _checked(r, tmp_path / "ref")
    assert _decide(p6=down, p7=down, p5=edited("p5", chosen_ref_checked=False))["status"] == "chosen_ref_not_checked"
    r = _all_none(_decide(p6=down, p7=_s("p7", down, commit="0ther")), "commit_mismatch", "the arms did not run at one knee commit")
    assert r["same_knee_commit"] is False and r["knee_commits"] == sorted(["0ther", n13d.DROPOUT_ARM_COMMIT])
    assert r["knee_commits_by_step"]["n13_p7"] == "0ther" and r["knee_commits_by_step"]["n13_dc"] == n13d.DROPOUT_ARM_COMMIT
    _checked(r, tmp_path / "commit")
    assert _decide(p6=down, p7=down, p5=_s("p5", down, commit="0ther"))["status"] == "commit_mismatch"
    old = {a: _s(a, down if a != "dc" else None, commit=BC_COMMIT) for a in ARMS}
    r = _all_none(_decide(p6=old["p6"], p7=old["p7"], p5=old["p5"], dc=old["dc"]),
                  "commit_mismatch", "the arms did not run at the declared knee commit")
    assert r["same_knee_commit"] is True and r["knee_commit_is_declared"] is False
    r = _decide(p6=old["p6"], p7=old["p7"], dc=old["dc"], arm_commit=None)  # only that comparison is skipped
    assert r["status"] == "decided" and r["knee_commit_is_declared"] is None
    r = _all_none(_decide(p6=down, p7=_s("p7", down, env=other)), "env_mismatch", "the arms' environments differ")
    assert r["env_differences"] == ["kaggle_image_build_date", "torch"] and set(r["environments"]) == {"dc", "p6", "p7"}
    _checked(r, tmp_path / "env")
    assert _decide(p6=down, p7=down, p5=_s("p5", down, env=other))["status"] == "env_mismatch"
    fill = _s("p7", down, only={(DN, 2)}, step="n13_p7_r", env=other)
    assert _decide(p6=down, p7=[part, fill])["status"] == "env_mismatch"
    # all arms on one other image: decided; bc on another image is only reported
    ro = _decide(p6=_s("p6", down, env=other), p7=_s("p7", down, env=other), dc=_s("dc", env=other))
    assert ro["status"] == "decided" and ro["rerun_check"]["bc_env_differences"] == ["kaggle_image_build_date", "torch"]
    # the order: no base, leakage, chosen_ref, commit, environment
    worst = {**_s("p7", down, env=other, commit="0ther"), "leakage_assertion_passed": False, "chosen_ref_checked": False}
    assert _decide(p6=down, p7=worst)["status"] == "leakage_check_failed"
    assert _decide(p6=down, p7={**worst, "leakage_assertion_passed": True})["status"] == "chosen_ref_not_checked"
    worst.update(leakage_assertion_passed=True, chosen_ref_checked=True)
    assert _decide(p6=down, p7=worst)["status"] == "commit_mismatch"
    assert _decide(p6=down, p7=worst, dc=None)["status"] == "no_base"
    # an arm rerun whole after an environment mismatch: the first run is reported, the rerun enters the rule
    first = _s("p7", {DN: UP4, CX: UP4}, env=other)
    rr = _decide(p6=down, p7=down, replaced={"p7": first, "p6": None})
    assert rr["status"] == "decided" and rr["encoders"][DN]["p7"]["verdict"] == n13d.LOWER
    assert list(rr["replaced_runs"]) == ["p7"] and rr["replaced_runs"]["p7"]["environments"] == [other]
    _checked(rr, tmp_path / "replaced")
    for bad in ({"b8": first}, {"p6": first}):
        with pytest.raises(ValueError):
            _decide(p6=down, p7=down, replaced=bad)


def test_decide_dropout_refuses_wrong_inputs_and_any_other_s():
    from knee.nb import n13d

    down = {DN: DOWN4, CX: DOWN4}
    dc, p6, p7, bc = _s("dc"), _s("p6", down), _s("p7", down), _bc()
    base0, null = _it2("base0", BASE0), _it2("null", NULL)

    def broken(arm, fn):
        x = _s(arm, down if arm != "dc" else None)
        fn(x["units"][f"{DN}_f1"])
        return x

    wrong_batch = broken("p7", lambda u: u["config"].update(batch_size=8))  # not its chosen 16
    wrong_used = broken("p7", lambda u: u["dropout"]["config"].update(dropout_used=0.6))
    wrong_cfg = broken("p7", lambda u: u["config"].update(dropout=0.6))
    no_dropout = broken("p7", lambda u: u.pop("dropout"))
    off_ref = broken("p7", lambda u: u["dropout"]["config"].update(dropout_chosen=0.25))  # not 0.000 of 25 Nov
    other_lr = broken("p6", lambda u: u["config"].update(lr=2e-3))  # differs from dc in more than the dropout
    dc_moved = broken("dc", lambda u: (u["config"].update(dropout=0.6), u["dropout"]["config"].update(dropout_used=0.6)))
    no_env = _s("p7", down, env=None)
    extra = _s("p7", down)
    extra["units"]["vit_s16_f0"] = dict(extra["units"][f"{DN}_f0"])
    nan = _s("p7", down)
    nan["units"][f"{DN}_f0"]["inner_score"] = None
    other_settings = _s("p7", down)
    other_settings["settings"] = {**other_settings["settings"], "patience": 3}
    wide = _it2("null", {e: [v + d for v, d in zip(vals, (0.02, -0.02, 0.01, -0.01, 0.0))] for e, vals in NULL.items()})
    twice = [_s("p7", down), _s("p7", down, only={(DN, 2)}, step="n13_p7_r")]
    for args, kw in (((dc, p7, p6, bc, base0, null), {}),  # arms in the wrong places
                     ((dc, p6, p7, bc, base0, null), {"p5": p6}),
                     ((dc, p6, p7, dc, base0, null), {}),  # bc must be arm bc's summary
                     ((p6, p6, p7, bc, base0, null), {}),
                     ((dc, p6, wrong_batch, bc, base0, null), {}), ((dc, p6, wrong_used, bc, base0, null), {}),
                     ((dc, p6, wrong_cfg, bc, base0, null), {}), ((dc, p6, no_dropout, bc, base0, null), {}),
                     ((dc, p6, off_ref, bc, base0, null), {}), ((dc, other_lr, p7, bc, base0, null), {}),
                     ((dc_moved, p6, p7, bc, base0, null), {}), ((dc, p6, no_env, bc, base0, null), {}),
                     ((dc, p6, extra, bc, base0, null), {}), ((dc, p6, nan, bc, base0, null), {}),
                     ((dc, p6, other_settings, bc, base0, null), {}),
                     ((dc, p6, p7, bc, null, base0), {}),
                     ((dc, p6, p7, bc, base0, wide), {}),  # not iteration 2's s (0.0030)
                     ((dc, p6, p7, bc, base0, _it2("null", {e: v[:4] for e, v in NULL.items()})), {}),
                     ((dc, p6, twice, bc, base0, null), {})):  # a second finished unit for one fold
        with pytest.raises(ValueError):
            n13d.decide_dropout(*args, **kw)
    assert n13d.decide_dropout(dc, p6, p7, None, base0, null)["rerun_check"]["bc_given"] is False  # bc may be missing


def test_decide_dropout_rerun_check_lists_units_above_the_tolerance(tmp_path):
    from knee.nb import n13d

    down = {DN: DOWN4, CX: SMALL_DOWN}
    tie = _decide(p6=down, p7=down, bc=_bc({DN: [0.0005, 0, 0, 0, 0]}))  # bc 0.0005 above dc in one unit: inside
    rc = tie["rerun_check"]
    assert tie["units"][f"{DN}_f0"]["dc_minus_bc"] == -0.0005 and rc["units_outside"] == [] and rc["words"] is None
    assert rc["identical"] is False and tie["dc_reruns_bc"] is False and rc["max_abs_difference"] == 0.0005
    assert len(n13d.dropout_slide_sentences(tie)) == 6
    out = _decide(p6=down, p7=down, bc=_bc({DN: [0.0006, 0, 0, 0, 0], CX: [0, 0, -0.0020, 0, 0]}))
    rc = out["rerun_check"]
    assert rc["units_outside"] == [f"{DN}_f0", f"{CX}_f2"]  # listed; the verdict is still given
    assert rc["units_outside_difference"] == {f"{DN}_f0": -0.0006, f"{CX}_f2": 0.002}
    assert rc["words"] == out["words_rerun"] == (
        "Rerun noise: the chosen-dropout arm (dc) differed from the earlier run of the same fits (arm bc) by more "
        "than 0.0005 in 2 of 10 units (largest difference 0.0020).")
    assert out["status"] == "decided" and out["encoders"] == tie["encoders"]  # bc never enters a verdict
    lines = n13d.dropout_slide_sentences(out)
    assert len(lines) == 7 and lines[4] == NONE_LINE and lines[5] == rc["words"] and lines[6] == ENDING
    _checked(out, tmp_path)
    part = _decide(p6=down, p7=down, bc=_bc(stopped={(CX, 1)}))
    assert part["units"][f"{CX}_f1"]["dc_minus_bc"] is None and len(part["rerun_check"]["units_checked"]) == 9
    none = _decide(p6=down, p7=down, bc=None)
    assert none["dc_reruns_bc"] is False and none["rerun_check"]["max_abs_difference"] is None
    assert none["rerun_check"]["per_unit"] == {} and none["encoders"] == tie["encoders"]
    _checked(none, tmp_path / "none")


def test_the_decide_record_of_every_arm_passes_the_privacy_check(tmp_path):
    """Dropout values and fold lists sit under config or structural keys; a small count elsewhere is refused."""
    r = _decide(p6={DN: UP4}, p7={CX: DOWN4}, p5={DN: FLAT})
    back = _checked(r, tmp_path / "all")
    assert back["units"][f"{DN}_f2"]["config"] == {"dropout_chosen": 0.5}
    for arm in ARMS:
        agg._check(json.loads(json.dumps(_s(arm, {DN: UP4}))))  # every arm's summary as dropout_summary writes it
    leak = copy.deepcopy(r)
    leak["by_chosen_dropout"]["p6"]["all"]["high"]["n_units"] = 4  # a count of units outside "config"
    with pytest.raises(agg.UnsafeSummary):
        agg.write("n13_dropout_decide", leak, folder=tmp_path / "leak")


# ---------------------------------------------------------------- the arm run path is DROPOUT_ARM_COMMIT's

DECIDE_NEW = {"DROPOUT_DECLARATION", "DROPOUT_ARM_COMMIT", "VERDICT_ARMS", "HIGHER", "LOWER", "NO_DIFF", "NO_VERDICT",
              "CEILING", "INSIDE", "NO_P5", "RERUN_TOL", "SPLIT_AT", "WORDS_ENDING", "WORDS_NOT_RERUN", "WORDS_NONE",
              "WORDS_RANGE", "N_COMPARISONS", "_p_text", "_higher_line", "_summary_lines", "_dropout_arm",
              "_same_but_dropout", "decide_dropout", "dropout_slide_sentences"}  # added after the arm commit


def _top(src: str) -> list:
    """[(names, text)] of a module's top-level statements, in order."""
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            names = set()
        out.append((frozenset(names), ast.get_source_segment(src, node)))
    return out


def _refs(text: str) -> set:
    tree = ast.parse(text)
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


def test_everything_reachable_from_run_dropout_is_textually_the_arm_commit():
    """The Kaggle sessions of the four arms install DROPOUT_ARM_COMMIT. The later commit may change the local
    decide step only: (1) every file under src/ and pyproject.toml is that commit's, except n13d.py, and no
    module was added; (2) in n13d.py every top-level statement of that commit is unchanged in text and order
    (the module docstring and the imports too) and the later statements only follow them; (3) nothing that
    run_dropout reaches, by name, is a name added later."""
    from knee.nb import n13d

    arm = n13d.DROPOUT_ARM_COMMIT
    assert len(arm) == 40 and int(arm, 16) >= 0
    if _git("rev-parse", "--verify", f"{arm}^{{commit}}") is None:
        pytest.skip("git history with the arm commit is not available")
    f13d = "src/knee/nb/n13d.py"
    tracked = _git("ls-tree", "-r", "--name-only", arm, "--", "src", "pyproject.toml").decode().splitlines()
    assert {f13d, "src/knee/nb/n13.py", "src/knee/nb/n13s.py", "src/knee/heads.py", "src/knee/nb/n6.py",
            "src/knee/agg.py", "pyproject.toml"} <= set(tracked)
    for f in tracked:
        if f != f13d:
            assert _git("show", f"{arm}:{f}") == (REPO / f).read_bytes(), f
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / "src").rglob("*.py")}
    assert on_disk <= set(tracked)  # no module was added
    old, new = _top(_git("show", f"{arm}:{f13d}").decode()), _top((REPO / f13d).read_text())
    assert [text for _, text in new[:len(old)]] == [text for _, text in old]  # same text, same order
    added = set().union(*(names for names, _ in new[len(old):]))
    assert added == DECIDE_NEW and not added & set().union(*(names for names, _ in old))  # nothing is redefined
    assert all(names for names, _ in new[len(old):])  # the later statements are definitions only: no import, no call
    by_name = {name: text for names, text in new for name in names}
    reach, todo = set(), ["run_dropout"]
    while todo:  # every module-level name that run_dropout can reach, by reference
        name = todo.pop()
        if name in reach:
            continue
        reach.add(name)
        todo += [x for x in _refs(by_name[name]) if x in by_name]
    assert {"run_dropout", "_dropout_task", "_dropout_unit", "arm_config", "check_dropout_ref", "dropout_config",
            "dropout_settings", "dropout_summary", "DROPOUT_ARMS", "DROPOUT_PLAN", "DROPOUT_REF", "_arm"} <= reach
    assert not reach & DECIDE_NEW
    old_text = [text for _, text in old]
    assert all(by_name[name] in old_text for name in reach)  # each is the text of the arm commit
    # the arm commit left n13.py, n13s.py, n6.py, heads.py and agg.py as they were at the commit the branch started from
    if _git("rev-parse", "--verify", "fe91086^{commit}") is not None:
        for f in ("src/knee/nb/n13.py", "src/knee/nb/n13s.py", "src/knee/nb/n6.py", "src/knee/heads.py",
                  "src/knee/agg.py", "src/knee/stage.py", "src/knee/nb/__init__.py"):
            assert _git("show", f"fe91086:{f}") == (REPO / f).read_bytes(), f
        diff = _git("diff", "--name-only", "fe91086", arm, "--", "src", "pyproject.toml").decode().split()
        assert diff == [f13d]  # the arm commit added one module and changed nothing else under src/
    for f in ("src/knee/nb/n13.py", "src/knee/nb/n13s.py", "src/knee/nb/__init__.py"):
        assert "n13d" not in (REPO / f).read_text()  # no other module imports it
    assert n13d.DROPOUT_PLAN.startswith("it2d_dropout_r2.md")  # the arms' own label


# ---------------------------------------------------------------- the notebooks

def test_the_four_notebooks_are_registered_as_cpu_notebooks():
    from knee.nb import n13d

    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import make_notebooks as mk
    finally:
        sys.path.remove(str(REPO / "scripts"))
    card = (REPO / "notebooks" / "RUNBOOK.md").read_text().split("## Dropout-ceiling check (it2d; added after the results)")[1]
    card = card.split("## If a run fails")[0]
    arm_commit = n13d.DROPOUT_ARM_COMMIT
    assert mk.DROPOUT_COMMIT == arm_commit and mk.DROPOUT_NOTEBOOKS == tuple(f"n13-{a}" for a in ARMS)
    for arm in ARMS:
        name = f"n13-{arm}"
        gpu, comp, tables, up, timeout = mk.NOTEBOOKS[name]
        assert (gpu, comp, tables) == (False, False, False) and up == mk.NOTEBOOKS["n13-bc"][3] == ["n0", "n5", "n3-r", "n6a"]
        assert mk.cell2(name) == f'from knee.nb import n13d; n13d.run_dropout("{arm}", budget_h=3.0, expect={mk.EXPECT})'
        assert mk.cell2(name).split("expect=")[1] == mk.cell2("n13-bc").split("expect=")[1]  # bc's hashes
        meta = mk.metadata("someone", name)
        assert meta["enable_gpu"] is False and "machine_shape" not in meta and meta["enable_internet"] is True
        assert meta["kernel_sources"] == mk.metadata("someone", "n13-bc")["kernel_sources"]
        assert timeout >= 1.1 * 3.0 * 3600  # above the step's watchdog
        assert f"| knee-{name} | None (CPU) |" in card and f"`{mk.cell2(name)}`" in card
        assert mk.commit_for(name, "0123abc") == arm_commit  # cell 1 is pinned, whatever --commit says
        assert mk.notebook_json(mk.commit_for(name, "0123abc"), mk.cell2(name))["cells"][0]["source"][0] == f'COMMIT = "{arm_commit}"\n'
    assert mk.commit_for("n13-s32", "0123abc") == mk.SLICE_COMMIT and mk.commit_for("n13-bc", "0123abc") == mk.BATCH_COMMIT
    assert mk.commit_for("n13-base0", "0123abc") == "0123abc"
    assert f'`COMMIT = "{arm_commit}"`' in card and "it2d_dropout_r2.md" in card
    assert "fork of knee-n13-bc" in card and "original environment" in card and "replace cell 2" in card
    assert "A worker crash is not retried" in card and "n13d.decide_dropout(dc, p6, p7, bc, base0, null, p5=p5)" in card
    for status in ("no_base", "leakage_check_failed", "chosen_ref_not_checked", "commit_mismatch", "env_mismatch"):
        assert f'"{status}"' in card
