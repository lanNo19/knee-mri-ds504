"""The batch-floor check (it2b_batch_floor_r2.md): n13's arms bc, b8 and b4 and n13.decide_batch.

Synthetic data, CPU. The arms fit the unit's chosen Stage A configuration on its inner folds with n6's
inner fits; only the batch size differs. The arms a1, base0 and null must be what they were at commit
79e95c5 (the commit this branch started from): one test runs them from that commit's tree and from
this tree and compares every output file. It skips when git or the commit is not available.
The three Kaggle sessions ran at commit 39c3c6f (ARM): another test shows that everything n13.run("<arm>")
reaches is textually that commit's, so the later commits changed the local decide step only.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest

from knee import agg, heads, paths

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
REF = "79e95c5"  # the commit this branch started from
ARM = "39c3c6fc7626e047ed643d5253e8ce9e662302af"  # the commit the three Kaggle arms ran at
MAIN = ("dinov2_vits14", "convnext_tiny")
FIND = list(range(12))
# synthetic chosen configurations; the batch size and best score of each unit are CHOSEN_REF's
CHOSEN = {"pooling": "gated", "planes": "all", "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
          "attn_hidden": 32, "pos_cap": 4.0}


def _git(*args) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_stage_a(inp: Path, edit=None) -> None:
    """knee-n6a's per-unit summaries (agg_n6a_<encoder>_f<k>.json at the output root), synthetic."""
    from knee.nb import n13

    d = inp / "n6a"
    d.mkdir(parents=True, exist_ok=True)
    for e in MAIN:
        for k, (b, best) in n13.CHOSEN_REF[e].items():
            u = {"encoder": e, "fold": k, "tuning": "nested", "l2_from": "bayesian optimisation on this fold",
                 "l1_winner": {"pooling": "gated", "planes": "all"}, "config": {**CHOSEN, "batch_size": b},
                 "best_inner_score": best, "E": 5, "default_trial_score": round(best - 0.002, 4)}
            if edit:
                edit(u)
            (d / f"agg_n6a_{e}_f{k}.json").write_text(json.dumps(u, indent=1))


@pytest.fixture(scope="module")
def stage_env(tmp_path_factory):
    from tests.synth_stage import make_stage

    root = tmp_path_factory.mktemp("it2bstage")
    facts = make_stage(root / "input")
    _write_stage_a(root / "input")
    return root, facts


@pytest.fixture()
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # as in the workers
    yield
    torch.set_num_threads(n)


def _task(arm, chosen, enc="dinov2_vits14", fold=0, deadline=None, od=None):
    from knee.nb import n13

    return {"id": f"{enc}_f{fold}", "encoder": enc, "fold": fold, "seed": n13.BATCH_SEED, "eligible": FIND,
            "batch_size": n13.BATCH_ARMS[arm], "config_chosen": chosen, "stage_a_best": 0.7,
            "chosen_sha16": "0" * 16, "deadline": time.time() + 3600 if deadline is None else deadline,
            "dir": str(od) if od else "unused"}


# ---------------------------------------------------------------- the arms

def test_batch_arms_are_declared_and_leave_the_old_arms_alone(monkeypatch):
    from knee.nb import n13

    assert n13.BATCH_ARMS == {"bc": None, "b8": 8, "b4": 4}
    assert set(n13.ARMS) == {"a1", "base0", "null", "a7"}  # the new arms are not n6 arms
    for arm, b in (("bc", "chosen"), ("b8", 8), ("b4", 4)):
        assert n13.batch_settings(arm) == {
            "l1": "gated|all", "n_trials": 1, "schedule_epochs": 30, "patience": 5, "seed": 0,
            "space_overrides": None, "config": {"source": "chosen (knee-n6a)", "batch_size": b}}
        agg._check(n13.batch_settings(arm))  # batch size 4 is a setting under "config", not a small count
    assert (n13.S_DECLARED, n13.MIN_GAIN, n13.MIN_FOLDS_UP, n13.SEARCH_RANGE) == (0.0030, 0.005, 4, (16, 128))
    floor = [(e, k) for e in MAIN for k, (b, _) in n13.CHOSEN_REF[e].items() if b == 16]
    assert len(floor) == 6  # the finding: the floor in 6 of 10 units
    for bad in (lambda: n13.batch_settings("b2"), lambda: n13.run_batch("b16"),
                lambda: n13.run("b8", schedule_epochs=8, patience=3),  # the registered schedule only
                lambda: n13.run_batch("b8", encoders=("vit_s16",)),
                lambda: n13.run_batch("b8", rerun=1, folds=(2,)),  # a rerun names its encoders too
                lambda: n13.run_batch("b8", budget_h=None)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):  # unknown names still fail in run() as before
        n13.run("a2")
    seen = []
    monkeypatch.setattr(n13, "run_batch", lambda arm, **kw: seen.append((arm, kw)) or {})
    n13.run("b8")  # run()'s default budget of 8 h is cut to the session budget of this check
    n13.run("b4", budget_h=2.5, folds=(1,), encoders=("convnext_tiny",), rerun=1)
    assert [(a, kw["budget_h"], kw["folds"], kw["rerun"]) for a, kw in seen] == [("b8", 6.0, None, 0), ("b4", 2.5, (1,), 1)]


def test_chosen_configs_are_read_from_stage_a_summaries_and_checked(stage_env, tmp_path):
    from knee.nb import n13

    root, _ = stage_env
    c = n13.chosen_configs(root / "input")
    assert sorted(c) == sorted((e, k) for e in MAIN for k in range(5))
    assert c[("dinov2_vits14", 0)]["config"] == {**CHOSEN, "batch_size": 33}
    assert c[("convnext_tiny", 3)]["config"]["batch_size"] == 27
    assert all(len(v["sha16"]) == 16 for v in c.values())

    def case(edit, **kw):
        inp = tmp_path / edit.__name__
        _write_stage_a(inp, edit)
        return lambda: n13.chosen_configs(inp, **kw)

    def other_batch(u):
        u["config"]["batch_size"] = 17 if u["fold"] == 2 else u["config"]["batch_size"]

    def other_score(u):
        u["best_inner_score"] += 0.0001 if u["fold"] == 4 else 0.0

    def below_floor(u):
        u["config"]["batch_size"] = 8

    def not_nested(u):
        u["tuning"] = "simplified (not nested)"

    def not_gated(u):
        u["l1_winner"] = {"pooling": "mean", "planes": "all"}

    def no_dropout(u):
        u["config"].pop("dropout")

    for bad in (case(other_batch), case(other_score), case(below_floor, ref=None), case(not_nested, ref=None),
                case(not_gated, ref=None), case(no_dropout, ref=None)):
        with pytest.raises(ValueError):
            bad()
    assert case(other_batch, ref=None)()[("dinov2_vits14", 2)]["config"]["batch_size"] == 17  # only the reference check
    two = tmp_path / "two"
    _write_stage_a(two)
    (two / "n6a_again").mkdir()
    (two / "n6a_again" / "agg_n6a_dinov2_vits14_f0.json").write_text((two / "n6a" / "agg_n6a_dinov2_vits14_f0.json").read_text())
    with pytest.raises(ValueError):  # two attached versions of one unit
        n13.chosen_configs(two)
    with pytest.raises(ValueError):  # knee-n6a not attached
        n13.chosen_configs(tmp_path / "nothing_here")


def test_only_the_batch_size_differs_between_the_arms(stage_env, tmp_path, monkeypatch, one_thread):
    from knee import stage
    from knee.nb import n6, n13

    root, _ = stage_env
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    chosen = n13.chosen_configs(root / "input")[("dinov2_vits14", 0)]["config"]
    fits = []
    real_fit = heads.fit

    def fit(bank, cfg, train_idx, val_idx=None, findings=None, **kw):
        fits.append({"cfg": cfg.to_dict(), "train": np.asarray(train_idx).copy(), "val": np.asarray(val_idx).copy(),
                     "findings": findings, **{k: kw.get(k) for k in ("epochs", "seed", "patience", "schedule_epochs",
                                                                     "max_epochs", "steps", "forbidden")}})
        return real_fit(bank, cfg, train_idx, val_idx, findings, **kw)

    monkeypatch.setattr(heads, "fit", fit)
    d = stage.StageData(root / "input")
    tr, te, _ = d.split(0)
    out, calls = {}, {}
    for arm in ("bc", "b8", "b4"):
        fits.clear()
        r = n13._batch_unit(_task(arm, chosen, od=tmp_path / "never_made"), tmp_path / "never_made")
        assert r["ok"] and r["fits"] == r["guard_checks"] == 4  # every inner fit passed the leakage guard
        out[arm], calls[arm] = r["summary"], list(fits)
    assert not (tmp_path / "never_made").exists()  # a unit writes nothing
    for arm, b in (("bc", 33), ("b8", 8), ("b4", 4)):
        u = out[arm]
        assert [c["cfg"] for c in calls[arm]] == [{**chosen, "batch_size": b}] * 4  # only the batch size
        assert u["config"] == {**chosen, "batch_size": b} and u["batch_chosen"] == 33
        for c in calls[arm]:  # base0's inner fit: seed + j, registered schedule and patience, no refit
            assert (c["epochs"], c["patience"], c["schedule_epochs"], c["max_epochs"], c["steps"]) == (None, 5, None, None, None)
            assert c["findings"] == FIND
            assert set(te.tolist()) | set(d.gold_rows().tolist()) == set(c["forbidden"].tolist())
            assert not set(c["train"].tolist()) & set(te.tolist()) and not set(c["val"].tolist()) & set(te.tolist())
            assert set(c["train"].tolist()) | set(c["val"].tolist()) <= set(tr.tolist())
        assert [c["seed"] for c in calls[arm]] == [0, 1, 2, 3]
        for a, c in zip(calls["bc"], calls[arm]):  # the same inner training and validation rows in every arm
            assert np.array_equal(a["train"], c["train"]) and np.array_equal(a["val"], c["val"])
        # steps per epoch: ceil(n / batch) per inner fit (the last, smaller batch is kept), reported to the nearest 5
        spe = [int(np.ceil(len(c["train"]) / b)) for c in calls[arm]]
        assert u["steps_per_epoch"] == int(5 * round(np.mean(spe) / 5)) and u["steps_per_epoch"] % 5 == 0
        assert (u["schedule_epochs"], u["patience"], u["seed"], u["inner_k"], u["device"]) == (30, 5, 0, 4, "cpu")
        assert len(u["epochs"]) == 4 and all(1 <= x <= 30 for x in u["epochs"]) and 1 <= u["E"] <= 30
        assert 0.5 < u["inner_score"] <= 1 and round(u["inner_score"], 4) == u["inner_score"]
        agg._check(json.loads(json.dumps(u)))  # batch 4 and small epoch counts sit under exempt keys
    assert out["bc"]["step_ratio"] == 1.0 and out["bc"]["steps_per_epoch"] == out["bc"]["steps_per_epoch_chosen"]
    assert 3.9 < out["b8"]["step_ratio"] < 4.3 and 7.9 < out["b4"]["step_ratio"] < 8.5  # 33/8 and 33/4
    assert out["b8"]["steps_per_epoch_chosen"] == out["bc"]["steps_per_epoch"]


def test_a_batch_arm_at_the_default_configuration_is_base0s_trial_0(stage_env, tmp_path, monkeypatch, one_thread):
    """The inner fits are n6's: with the default configuration as "chosen", bc gives base0's unit score."""
    pytest.importorskip("optuna")
    from knee.nb import n6, n13

    root, _ = stage_env
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    od = tmp_path / "dinov2_vits14_f1"
    spec = n13.arm_spec("base0")
    task = {"id": "dinov2_vits14_f1", "encoder": "dinov2_vits14", "fold": 1, "seed": spec["seed"], "eligible": FIND,
            "soft_stop": time.time() + 3600, "backstop": None, "mode": "nested", "unit_s_est": None,
            "n_trials": spec["n_trials"], "l1_grid": list(n13.L1_GRID), "schedule_epochs": spec["schedule_epochs"],
            "patience": spec["patience"], "space_overrides": spec["space_overrides"], "dir": str(od),
            "fold0_in_run": False}
    b0 = n6._unit(task, od)["summary"]  # base0's unit, as n13.run("base0") runs it
    assert b0["config"]["batch_size"] == 48
    r = n13._batch_unit(_task("bc", b0["config"], fold=1), tmp_path / "x")
    assert r["summary"]["inner_score"] == b0["default_trial_score"] == b0["best_inner_score"]
    assert r["summary"]["E"] == b0["E"] and r["summary"]["config"] == b0["config"]


def test_run_b8_end_to_end_records_the_override_and_touches_no_test_fold_file(stage_env, monkeypatch, capfd):
    from knee.nb import n13

    root, facts = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_b8")
    out = n13.run("b8", root=root / "input", workers=2, encoders=("dinov2_vits14",), folds=(0, 1), budget_h=1.0)
    printed = capfd.readouterr()
    every = sorted(p for p in paths.WORKING.rglob("*") if p.is_file())
    assert [p.relative_to(paths.WORKING).as_posix() for p in every] == ["agg_n13_b8_inner.json", "run.json"]
    text = printed.out + printed.err + "".join(p.read_text() for p in every)
    assert not [u for u in facts["uids"] if u in text]  # no study identifier anywhere Claude reads
    assert "STEP n13_b8: OK" in printed.out
    # no unit folder, no head, no prediction: nothing N7, N9 or N10 could take, and no test-fold number
    assert sorted(p.name for p in (paths.WORKING / "private" / "n13_b8").iterdir()) == ["errors"]
    rj = json.loads((paths.WORKING / "run.json").read_text())
    c = rj["config"]
    assert rj["step"] == "n13_b8" and (c["arm"], c["batch_size"], c["seed"], c["budget_h"], c["rerun"]) == ("b8", 8, 0, 1.0, 0)
    assert c["test_overrides"] is True and c["test_overrides_reason"] == n13.BATCH_REASON
    assert c["added_after_results"] is True and c["chosen_ref_checked"] is True
    assert c["environment"] == out["environment"] == n13.environment()
    assert json.loads((paths.WORKING / "agg_n13_b8_inner.json").read_text()) == json.loads(json.dumps(out))
    assert (out["arm"], out["step"], out["status"], out["all_units_ok"]) == ("b8", "n13_b8", "ok", True)
    assert out["added_after_results"] is True and out["test_overrides"] is True and out["plan"] == n13.BATCH_PLAN
    assert out["leakage_assertion_passed"] is True and out["device_names"] == ["cpu"]
    assert out["settings"] == n13.batch_settings("b8") and out["chosen_ref_checked"] is True
    assert out["units_finished"] == ["dinov2_vits14_f0", "dinov2_vits14_f1"] and out["units_missing"] == []
    assert set(out["upstream"]) == {"n0_studies", "n3_folds", "n5_index"}
    for k, chosen_b in ((0, 33), (1, 16)):
        u = out["units"][f"dinov2_vits14_f{k}"]
        assert u["ok"] and u["config"] == {**CHOSEN, "batch_size": 8} and u["batch_chosen"] == chosen_b
        assert u["steps_per_epoch"] > u["steps_per_epoch_chosen"] > 0 and u["step_ratio"] > 1.9
        assert out["inner_scores"]["dinov2_vits14"][k] == u["inner_score"]
        assert u["stage_a_best_inner_score"] == n13.CHOSEN_REF["dinov2_vits14"][k][1]
    assert "trial0_inner_scores" not in out and "heads_sha16" not in out  # not an n6 arm's summary
    m = n13.merged(out)  # the reader of the keep rule takes it (reruns, steps, environments)
    assert m["arm"] == "b8" and sorted(m["units"]) == [("dinov2_vits14", 0), ("dinov2_vits14", 1)]


# ---------------------------------------------------------------- the soft stop

def test_soft_stop_ends_cleanly_with_status_partial(stage_env, tmp_path, monkeypatch, capfd, one_thread):
    from knee.nb import n6, n13

    root, _ = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_b4_stop")
    out = n13.run("b4", root=root / "input", workers=2, encoders=("dinov2_vits14",), folds=(0, 1), budget_h=0.0)
    assert "STEP n13_b4: OK" in capfd.readouterr().out  # a stopped session is not a failed step
    assert (out["status"], out["all_units_ok"], out["units_finished"]) == ("partial", False, [])
    assert out["units_missing"] == ["dinov2_vits14_f0", "dinov2_vits14_f1"] and out["inner_scores"] == {}
    assert all(u == {"ok": False, "stopped": True, "error": None} for u in out["units"].values())
    assert json.loads((paths.WORKING / "agg_n13_b4_inner.json").read_text()) == json.loads(json.dumps(out))
    assert out["leakage_assertion_passed"] is True and out["budget_h"] == 0.0

    # a unit whose deadline passes during its second inner fit stops at that fit's next epoch end: no score
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    chosen = n13.chosen_configs(root / "input")[("dinov2_vits14", 0)]["config"]
    task = _task("b4", chosen)
    real_fit, runs = heads.fit, []

    def fit(*a, **kw):
        if runs:
            task["deadline"] = 0.0  # the budget ends while the second fit runs
        r = real_fit(*a, **kw)
        runs.append(r.epochs_run)
        return r

    monkeypatch.setattr(heads, "fit", fit)
    r = n13._batch_unit(task, tmp_path / "x")
    assert r == {"task": "dinov2_vits14_f0", "ok": False, "stopped": True}
    assert len(runs) == 2 and runs[1] == 1  # the second fit ended with its first epoch; no third fit started
    monkeypatch.setattr(heads, "fit", real_fit)
    done = n13._batch_unit(_task("b4", chosen), tmp_path / "x")
    s = n13.batch_summary("b4", "n13_b4", {"dinov2_vits14_f0": done, "dinov2_vits14_f1": r},
                          ["dinov2_vits14_f0", "dinov2_vits14_f1", "dinov2_vits14_f2"], n13.environment(),
                          ("dinov2_vits14",), budget_h=6.0)
    assert (s["status"], s["units_finished"]) == ("partial", ["dinov2_vits14_f0"])  # the finished unit is written
    assert s["units_missing"] == ["dinov2_vits14_f1", "dinov2_vits14_f2"]
    assert s["inner_scores"] == {"dinov2_vits14": {0: done["summary"]["inner_score"]}}
    assert s["units"]["dinov2_vits14_f0"]["config"]["batch_size"] == 4
    agg.write("n13_b4_inner", s, folder=tmp_path / "agg")  # the checker takes a partial summary with batch 4
    failed = {"task": "dinov2_vits14_f1", "ok": False, "error": "LeakageError at heads.py:209 in _check_guard()"}
    s2 = n13.batch_summary("b4", "n13_b4", {"dinov2_vits14_f0": done, "dinov2_vits14_f1": failed},
                           ["dinov2_vits14_f0", "dinov2_vits14_f1"], None, ("dinov2_vits14",))
    assert s2["leakage_assertion_passed"] is False and s2["units"]["dinov2_vits14_f1"]["stopped"] is False
    s3 = n13.batch_summary("b4", "n13_b4", {"dinov2_vits14_f0": {**done, "guard_checks": 3}}, ["dinov2_vits14_f0"],
                           None, ("dinov2_vits14",))
    assert s3["leakage_assertion_passed"] is False  # a fit that did not pass the guard


# ---------------------------------------------------------------- the old arms are unchanged

RUN_SCRIPT = """
import sys
from knee.nb import n13
if __name__ == "__main__":  # N6's workers are spawned and import this file again
    n13.run(sys.argv[2], root=sys.argv[1], workers=1, encoders=("dinov2_vits14",), folds=(0,))
"""
TIMED = {"seconds", "time_utc"}


def _untimed(o):
    if isinstance(o, dict):
        return {k: _untimed(v) for k, v in o.items() if k not in TIMED}
    return [_untimed(v) for v in o] if isinstance(o, list) else o


def test_files_of_the_old_arms_are_untouched_and_n13_only_gained_lines():
    if _git("rev-parse", "--verify", f"{REF}^{{commit}}") is None:
        pytest.skip("git history with commit 79e95c5 is not available")
    for f in ("src/knee/heads.py", "src/knee/nb/n6.py", "src/knee/agg.py", "src/knee/stage.py", "src/knee/metrics.py",
              "src/knee/nb/common.py", "src/knee/paths.py", "src/knee/safe_log.py"):
        assert _git("show", f"{REF}:{f}") == (REPO / f).read_bytes(), f
    diff = _git("diff", "--unified=0", REF, "--", "src/knee/nb/n13.py").decode()
    removed = [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert removed == []  # additions only


@pytest.mark.parametrize("arm", ["base0", "a1", "null"])
def test_old_arm_outputs_are_byte_identical_to_79e95c5(stage_env, tmp_path, arm):
    pytest.importorskip("optuna")
    root, _ = stage_env
    archive = _git("archive", "--format=tar", REF, "src")
    if archive is None:
        pytest.skip("git history with commit 79e95c5 is not available")
    ref = tmp_path / "ref"
    ref.mkdir()
    (tmp_path / "ref.tar").write_bytes(archive)
    with tarfile.open(tmp_path / "ref.tar") as tf:
        tf.extractall(ref, filter="data")
    script_dir = tmp_path / "script"
    script_dir.mkdir()
    (script_dir / "run_arm.py").write_text(RUN_SCRIPT)
    for label, src in (("old", ref / "src"), ("new", REPO / "src")):
        env = {**os.environ, "PYTHONPATH": str(src), "KNEE_WORKING": str(tmp_path / label)}
        subprocess.run([sys.executable, str(script_dir / "run_arm.py"), str(root / "input"), arm],
                       env=env, check=True, capture_output=True, timeout=900)
    old_w, new_w = tmp_path / "old", tmp_path / "new"
    files = {w: sorted(p.relative_to(w).as_posix() for p in w.rglob("*") if p.is_file()) for w in (old_w, new_w)}
    assert files[old_w] == files[new_w]
    assert f"agg_n13_{arm}_inner.json" in files[new_w] and f"private/n13_{arm}/units/dinov2_vits14_f0/head.pt" in files[new_w]
    timed = {"run.json", f"agg_n13_{arm}.json", f"agg_n13_{arm}_dinov2_vits14_f0.json",
             f"private/n13_{arm}/units/dinov2_vits14_f0/config.json"}  # they hold a time or a duration
    for f in files[new_w]:
        if f in timed:
            assert _untimed(json.loads((old_w / f).read_text())) == _untimed(json.loads((new_w / f).read_text())), f
        else:  # the arm's summary block, predictions, trials, the head and its statistics: byte for byte
            assert (old_w / f).read_bytes() == (new_w / f).read_bytes(), f


# ---------------------------------------------------------------- decide_batch

ENV = {"kaggle_image_git_commit": "e1dbc17d", "kaggle_image_build_date": "20261117-122508", "python": "3.13.18",
       "torch": "2.11.0+cpu", "cuda": None, "cudnn": None, "device_name": "cpu"}
# iteration 2's inner scores (n13_numbers.md): s = 0.0030
BASE0 = {"convnext_tiny": [0.7240, 0.7444, 0.7393, 0.7476, 0.7418], "dinov2_vits14": [0.7227, 0.7398, 0.7327, 0.7398, 0.7367]}
NULL = {"convnext_tiny": [0.7192, 0.7463, 0.7432, 0.7472, 0.7381], "dinov2_vits14": [0.7188, 0.7390, 0.7327, 0.7404, 0.7318]}
BC = {"dinov2_vits14": [0.7240, 0.7410, 0.7350, 0.7420, 0.7360], "convnext_tiny": [0.7250, 0.7480, 0.7450, 0.7500, 0.7440]}


def _it2(arm, scores, env=ENV):
    from knee.nb import n13

    units = {f"{e}_f{k}": {"ok": True, "trial0_inner_score": v, "head_sha16": "0" * 16}
             for e, vals in scores.items() for k, v in enumerate(vals)}
    return {"arm": arm, "step": f"n13_{arm}", "settings": n13.settings(n13.arm_spec(arm)), "environment": env,
            "knee_commit": "9ecea01", "units": units}


def _b(arm, diffs=None, env=ENV, stopped=(), step=None, only=None, commit=ARM):
    """A batch-arm summary shaped as batch_summary writes it: BC plus diffs {encoder: [5 differences]}."""
    from knee.nb import n13

    b_arm = n13.BATCH_ARMS[arm]
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
            chosen_b, best = n13.CHOSEN_REF[e][k]
            b = chosen_b if b_arm is None else b_arm
            v = round(BC[e][k] + (diffs[e][k] if diffs else 0.0), 4)
            units[uid] = {"ok": True, "inner_score": v, "E": 4, "epochs": [9, 9, 8, 3], "inner_k": 4, "seed": 0,
                          "schedule_epochs": 30, "patience": 5, "device": "cpu", "device_name": "cpu",
                          "config": {**CHOSEN, "batch_size": b}, "batch_chosen": chosen_b,
                          "steps_per_epoch": int(5 * round(3000 / b / 5)),
                          "steps_per_epoch_chosen": int(5 * round(3000 / chosen_b / 5)),
                          "step_ratio": round(chosen_b / b, 2), "stage_a_best_inner_score": best,
                          "chosen_sha16": "0" * 16, "seconds": 100.0}
            scores.setdefault(e, {})[k] = v
    done = [u for u in ids if units[u]["ok"]]
    return {"arm": arm, "step": step or f"n13_{arm}", "plan": n13.BATCH_PLAN, "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": n13.BATCH_REASON, "encoders": list(MAIN),
            "settings": n13.batch_settings(arm), "chosen_ref_checked": True, "environment": env, "knee_commit": commit,
            "status": "ok" if len(done) == len(ids) else "partial", "all_units_ok": len(done) == len(ids),
            "units_finished": done, "units_missing": [u for u in ids if not units[u]["ok"]],
            "leakage_assertion_passed": True, "device_names": ["cpu"], "score": n13.BATCH_SCORE, "budget_h": 6.0,
            "seconds": 3000.0, "inner_scores": scores, "units": units}


UP4 = [0.008, 0.007, 0.006, 0.009, -0.001]  # mean 0.0058, higher in 4 of 5 folds
UP3 = [0.020, 0.020, 0.010, -0.010, -0.010]  # mean 0.0060, higher in 3 of 5 folds
TIE = [0.005, 0.005, 0.005, 0.005, 0.005]  # mean exactly 0.0050: not above the threshold
DOWN = [-0.004, -0.002, 0.001, -0.003, -0.006]


def _decide(b8, b4, bc=None, **kw):
    from knee.nb import n13

    return n13.decide_batch(_b("bc") if bc is None else bc, b8, b4, _it2("base0", BASE0), _it2("null", NULL), **kw)


def test_decide_batch_verdicts_by_the_declared_rule(tmp_path):
    from knee.nb import n13

    r = _decide(_b("b8", {"dinov2_vits14": UP4, "convnext_tiny": UP3}),
                _b("b4", {"dinov2_vits14": TIE, "convnext_tiny": DOWN}))
    assert r["status"] == "decided" and r["added_after_results"] is True and r["same_knee_commit"] is True
    assert round(r["s"], 4) == 0.0030 == r["s_declared"] and r["threshold"] == 0.005 and r["min_gain"] == 0.005
    assert abs(r["threshold_unrounded"] - 0.005) < 1e-12  # 2 s / sqrt(5) = 0.0027 does not bind
    assert r["same_environment_as_iteration2"] is True and r["env_differences_from_iteration2"] == []
    d8, c8 = r["encoders"]["dinov2_vits14"]["b8"], r["encoders"]["convnext_tiny"]["b8"]
    d4, c4 = r["encoders"]["dinov2_vits14"]["b4"], r["encoders"]["convnext_tiny"]["b4"]
    assert (d8["verdict"], d8["gain"], d8["folds_up"], d8["folds"]) == ("the floor mattered", 0.0058, 4, [0, 1, 2, 3, 4])
    assert d8["per_fold"] == {0: 0.008, 1: 0.007, 2: 0.006, 3: 0.009, 4: -0.001}
    assert (c8["verdict"], c8["gain"], c8["folds_up"]) == ("no support", 0.006, 3)  # the gain alone is not enough
    assert (d4["verdict"], d4["gain"], d4["folds_up"]) == ("no support", 0.005, 5)  # a 4-decimal tie is not above
    assert (c4["verdict"], c4["gain"], c4["folds_up"]) == ("no support", -0.0028, 1)
    # one sentence set per encoder and arm (declaration r2, section 7)
    limits, cannot = "One seed. Learning rate not re-tuned for the smaller batch.", "Gains under 0.005 cannot be shown by this check."
    margin = "This does not meet the declared margin (more than 0.005 and 4 of 5 folds), so it is not counted as an effect."
    edge = ("Open edge: batch 8 scored +0.0058 above the chosen batch on inner validation (DINOv2, 4 of 5 folds). "
            "The head search was not rerun.")
    assert (d8["words"], d8["words_margin"], d8["words_limits"]) == (edge, None, limits)
    assert (c8["words"], c8["words_margin"], c8["words_limits"]) == (  # positive, under the margin (3 folds)
        "Batch 8 scored 0.0060 higher than the chosen batch on inner validation (ConvNeXt-T, 3 of 5 folds).",
        margin, f"{limits} {cannot}")
    assert (d4["words"], d4["words_margin"]) == (  # positive, under the margin (the tie): never "did not score higher"
        "Batch 4 scored 0.0050 higher than the chosen batch on inner validation (DINOv2, 5 of 5 folds).", margin)
    none = ("Batch 4 did not score higher than the chosen batch on inner validation (ConvNeXt-T: gain -0.0028, "
            "higher in 1 of 5 folds).")
    assert (c4["words"], c4["words_margin"], c4["words_limits"]) == (none, None, f"{limits} {cannot}")  # gain <= 0
    assert n13.batch_slide_sentences(r) == [
        f"{d8['words']} {limits}", f"{d4['words']} {margin} {limits} {cannot}",
        f"{c8['words']} {margin} {limits} {cannot}", f"{c4['words']} {limits} {cannot}"]
    # the enforced flags, the commits and the declaration are in the record
    assert r["leakage_assertion_passed"] is True and r["chosen_ref_checked"] is True and r["replaced_runs"] == {}
    assert r["knee_commits"] == [ARM] == [n13.ARM_COMMIT] and r["knee_commit_is_declared"] is True
    assert r["declaration"].startswith("it2b_batch_floor_r2.md") and r["plan"] == n13.BATCH_PLAN
    assert "gains under 0.005 cannot be shown by this check" in r["limits"] and len(r["limits"]) == 6
    # the per-unit table and the split by chosen batch size (descriptive)
    u = r["units"]["dinov2_vits14_f0"]
    assert u["batch_chosen"] == 33 and u["bc"]["inner_score"] == 0.724 and u["bc"]["minus_bc"] is None
    assert (u["b8"]["inner_score"], u["b8"]["minus_bc"], u["b4"]["minus_bc"]) == (0.732, 0.008, 0.005)
    assert u["bc_minus_stage_a"] == round(0.7240 - 0.7249, 4) and r["bc_minus_stage_a"]["descriptive"] is True
    rp = r["bc_minus_stage_a"]  # bc reproduces Stage A's best inner score inside 0.006 in every unit
    assert (rp["tolerance"], rp["used_in_rule"], rp["reproduces"], rp["units_outside"], rp["words"]) == (0.006, False, True, [], None)
    assert r["bc_reproduces_stage_a"] is True and u["bc_reproduces_stage_a"] is True and len(rp["units_checked"]) == 10
    assert rp["max_abs_difference"] == 0.0022 and rp["mean"] == -0.0013 and "T4" in rp["cause"]
    above = ["dinov2_vits14_f0", "convnext_tiny_f0", "convnext_tiny_f3", "convnext_tiny_f4"]
    sp = r["by_chosen_batch"]
    assert sp["descriptive"] is True and sp["b8"]["chosen_above_floor"]["units"] == above
    assert len(sp["b8"]["chosen_on_floor"]["units"]) == 6
    assert sp["b8"]["chosen_above_floor"]["mean_minus_bc"] == round((0.008 + 0.020 - 0.010 - 0.010) / 4, 4)
    assert sp["b8"]["chosen_above_floor"]["units_up"] == ["dinov2_vits14_f0", "convnext_tiny_f0"]
    assert r["arms"]["b4"]["config"] == {"batch_size": 4} and r["arms"]["bc"]["config"] == {"batch_size": "chosen"}
    # only aggregate numbers, and the checker takes them (batch 4, fold lists and counts of folds included)
    p = agg.write("n13_batch_decide", r, folder=tmp_path)
    assert p.name == "agg_n13_batch_decide.json"
    back = json.loads(p.read_text())
    assert back["encoders"]["dinov2_vits14"]["b8"]["verdict"] == n13.MATTERED and max(len(x) for x in _strings(back)) <= 200
    leak = copy.deepcopy(r)
    leak["units"]["dinov2_vits14_f0"]["b4"]["batch_size"] = 4  # outside "config" a 4 is a small count
    with pytest.raises(agg.UnsafeSummary):
        agg.write("n13_batch_decide", leak, folder=tmp_path / "leak")
    assert not (tmp_path / "leak" / "agg_n13_batch_decide.json").exists()
    # the summaries as files give the same record; a rerun fills a stopped unit
    files = [json.loads(json.dumps(x)) for x in (_b("bc"), _b("b8", {"dinov2_vits14": UP4, "convnext_tiny": UP3}),
                                                  _b("b4", {"dinov2_vits14": TIE, "convnext_tiny": DOWN}),
                                                  _it2("base0", BASE0), _it2("null", NULL))]
    assert json.loads(json.dumps(n13.decide_batch(*files))) == json.loads(json.dumps(r))
    part = _b("b8", {"dinov2_vits14": UP4, "convnext_tiny": UP3}, stopped={("dinov2_vits14", 2)})
    fill = _b("b8", {"dinov2_vits14": UP4, "convnext_tiny": UP3}, only={("dinov2_vits14", 2)}, step="n13_b8_r")
    rr = _decide([part, fill], _b("b4", {"dinov2_vits14": TIE, "convnext_tiny": DOWN}))
    assert rr["encoders"] == r["encoders"] and rr["arms"]["b8"]["steps"] == ["n13_b8", "n13_b8_r"]


def _strings(o):
    if isinstance(o, dict):
        return [s for k, v in o.items() for s in [str(k), *_strings(v)]]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return [o] if isinstance(o, str) else []


def test_decide_batch_gives_no_verdict_with_fewer_than_5_folds(tmp_path):
    up = {"dinov2_vits14": UP4, "convnext_tiny": UP4}
    r = _decide(_b("b8", up, stopped={("convnext_tiny", 3), ("convnext_tiny", 4)}), _b("b4", up))
    assert r["status"] == "partial"
    c8 = r["encoders"]["convnext_tiny"]["b8"]
    assert (c8["verdict"], c8["gain"], c8["folds_up"], c8["folds"]) == ("no verdict", None, None, [0, 1, 2])
    assert c8["reason"] == "fewer than 5 folds finished in the arm and in bc" and "gain_unrounded" not in c8
    assert c8["per_fold"] == {0: 0.008, 1: 0.007, 2: 0.006} and "no verdict" in c8["words"]
    assert r["encoders"]["dinov2_vits14"]["b8"]["verdict"] == "the floor mattered"  # the other encoder is whole
    assert r["encoders"]["convnext_tiny"]["b4"]["verdict"] == "the floor mattered"  # and so is the other arm
    assert r["units"]["convnext_tiny_f3"]["b8"] == {"inner_score": None, "E": None, "steps_per_epoch": None,
                                                    "step_ratio": None, "minus_bc": None}
    agg.write("n13_batch_decide", r, folder=tmp_path)  # 3 finished folds are a list of folds, not a count
    # a fold missing in the base takes the verdict from both arms for that encoder
    rb = _decide(_b("b8", up), _b("b4", up), bc=_b("bc", stopped={("dinov2_vits14", 0)}))
    assert [rb["encoders"]["dinov2_vits14"][a]["verdict"] for a in ("b8", "b4")] == ["no verdict"] * 2
    assert rb["encoders"]["dinov2_vits14"]["b8"]["folds"] == [1, 2, 3, 4]
    assert rb["encoders"]["convnext_tiny"]["b8"]["verdict"] == "the floor mattered"
    # an arm that did not run, and a missing base
    rn = _decide(_b("b8", up), None)
    assert rn["status"] == "partial" and rn["arms"]["b4"] is None
    assert rn["encoders"]["dinov2_vits14"]["b4"]["reason"] == "the arm did not run"
    assert rn["encoders"]["dinov2_vits14"]["b8"]["verdict"] == "the floor mattered"
    from knee.nb import n13

    r0 = n13.decide_batch(None, _b("b8", up), _b("b4", up), _it2("base0", BASE0), _it2("null", NULL))
    assert r0["status"] == "no_base" and all(v[a]["verdict"] == "no verdict" for v in r0["encoders"].values() for a in v)
    agg.write("n13_batch_decide", r0, folder=tmp_path / "no_base")


def test_decide_batch_environment_and_input_checks(tmp_path):
    from knee.nb import n13

    up = {"dinov2_vits14": UP4, "convnext_tiny": UP4}
    other = {**ENV, "kaggle_image_build_date": "20261124-000000", "torch": "2.12.0+cpu"}
    # one new arm on another image: no verdict at all
    r = _decide(_b("b8", up, env=other), _b("b4", up))
    assert r["status"] == "env_mismatch" and r["env_differences"] == ["kaggle_image_build_date", "torch"]
    assert all(v[a]["verdict"] == "no verdict" and v[a]["gain"] is None and v[a]["reason"] == "the arms' environments differ"
               for v in r["encoders"].values() for a in v)
    assert "units" not in r and set(r["environments"]) == {"bc", "b8", "b4"}
    agg.write("n13_batch_decide", r, folder=tmp_path)
    # all three new arms on one image that is not iteration 2's: the verdict stands and the record says so
    r2 = _decide(_b("b8", up, env=other), _b("b4", up, env=other), bc=_b("bc", env=other))
    assert r2["status"] == "decided" and r2["same_environment_as_iteration2"] is False
    assert r2["env_differences_from_iteration2"] == ["kaggle_image_build_date", "torch"]
    assert r2["encoders"]["dinov2_vits14"]["b8"]["verdict"] == "the floor mattered"
    # a rerun on another image is an environment difference too
    part = _b("b8", up, stopped={("dinov2_vits14", 2)})
    fill = _b("b8", up, only={("dinov2_vits14", 2)}, step="n13_b8_r", env=other)
    assert _decide([part, fill], _b("b4", up))["status"] == "env_mismatch"
    # refused inputs
    wrong_batch = _b("b8", up)
    wrong_batch["units"]["dinov2_vits14_f1"]["config"]["batch_size"] = 16
    moved_base = _b("bc")
    moved_base["units"]["convnext_tiny_f3"]["config"]["batch_size"] = 16  # not its chosen 27
    no_env = _b("b4", up, env=None)
    extra = _b("b4", up)
    extra["units"]["vit_s16_f0"] = dict(extra["units"]["dinov2_vits14_f0"])
    nan = _b("b4", up)
    nan["units"]["dinov2_vits14_f0"]["inner_score"] = None
    other_settings = _b("b4", up)
    other_settings["settings"] = {**other_settings["settings"], "patience": 3}
    b8, b4, bc, base0, null = _b("b8", up), _b("b4", up), _b("bc"), _it2("base0", BASE0), _it2("null", NULL)
    wide = _it2("null", {e: [v + d for v, d in zip(vals, (0.02, -0.02, 0.01, -0.01, 0.0))] for e, vals in NULL.items()})
    twice = [_b("b8", up), _b("b8", up, only={("dinov2_vits14", 2)}, step="n13_b8_r")]
    for args in ((bc, b4, b8, base0, null),  # arms in the wrong places
                 (bc, wrong_batch, b4, base0, null), (moved_base, b8, b4, base0, null), (bc, b8, no_env, base0, null),
                 (bc, b8, extra, base0, null), (bc, b8, nan, base0, null), (bc, b8, other_settings, base0, null),
                 (bc, b8, b4, null, base0), (bc, b8, b4, base0, wide),  # not iteration 2's s
                 (bc, b8, b4, base0, _it2("null", {e: v[:4] for e, v in NULL.items()})),
                 (bc, twice, b4, base0, null)):  # a second finished unit for one fold
        with pytest.raises(ValueError):
            n13.decide_batch(*args)



def _all_none(r, status, reason):
    assert r["status"] == status and "units" not in r and "bc_minus_stage_a" not in r
    recs = [v[a] for v in r["encoders"].values() for a in ("b8", "b4")]
    assert all((x["verdict"], x["gain"], x["folds_up"], x["reason"]) == ("no verdict", None, None, reason) for x in recs)
    assert all(x["words"].endswith(f"no verdict ({reason}).") and x["words_limits"] == "The batch-size floor was reached and not extended."
               for x in recs)
    return r


def test_decide_batch_enforces_leakage_flag_chosen_ref_and_one_commit(tmp_path):
    from knee.nb import n13

    up = {"dinov2_vits14": UP4, "convnext_tiny": UP4}

    def edited(arm, **fields):
        return {**_b(arm, up if arm != "bc" else None), **fields}

    # the leakage flag of every arm summary given (a rerun's too) must be true
    for bad in (False, None):
        r = _all_none(_decide(edited("b8", leakage_assertion_passed=bad), _b("b4", up)),
                      "leakage_check_failed", "an arm's leakage check did not pass")
        assert r["leakage_failed_steps"] == ["n13_b8"] and r["leakage_assertion_passed"] is False
    agg.write("n13_batch_decide", r, folder=tmp_path / "leak")
    r = _decide(_b("b8", up), _b("b4", up), bc=edited("bc", leakage_assertion_passed=False))
    assert r["status"] == "leakage_check_failed" and r["leakage_failed_steps"] == ["n13_bc"]
    part = _b("b8", up, stopped={("dinov2_vits14", 2)})
    fill = {**_b("b8", up, only={("dinov2_vits14", 2)}, step="n13_b8_r"), "leakage_assertion_passed": False}
    assert _decide([part, fill], _b("b4", up))["leakage_failed_steps"] == ["n13_b8_r"]
    # chosen_ref_checked: the arm checked its configurations against the values read on 25 Nov
    for bad in (False, None):
        r = _all_none(_decide(_b("b8", up), edited("b4", chosen_ref_checked=bad)),
                      "chosen_ref_not_checked", "an arm did not check its configurations against 25 Nov")
        assert r["chosen_ref_unchecked_steps"] == ["n13_b4"] and r["chosen_ref_checked"] is False
    agg.write("n13_batch_decide", r, folder=tmp_path / "ref")
    # one knee commit across the arm files, and it is the declared one
    r = _all_none(_decide(_b("b8", up, commit="0ther"), _b("b4", up)), "commit_mismatch",
                  "the arms did not run at one knee commit")
    assert r["same_knee_commit"] is False and r["knee_commits"] == sorted(["0ther", ARM])
    assert r["knee_commits_by_step"] == {"n13_bc": ARM, "n13_b8": "0ther", "n13_b4": ARM}
    agg.write("n13_batch_decide", r, folder=tmp_path / "commit")
    old = [_b(a, up if a != "bc" else None, commit="9ecea01") for a in ("bc", "b8", "b4")]
    r = _all_none(n13.decide_batch(*old, _it2("base0", BASE0), _it2("null", NULL)), "commit_mismatch",
                  "the arms did not run at the declared knee commit")
    assert r["same_knee_commit"] is True and r["knee_commit_is_declared"] is False and r["arm_commit_declared"] == ARM
    r = n13.decide_batch(*old, _it2("base0", BASE0), _it2("null", NULL), arm_commit=None)  # only that comparison is skipped
    assert r["status"] == "decided" and r["knee_commit_is_declared"] is None
    fill = _b("b8", up, only={("dinov2_vits14", 2)}, step="n13_b8_r", commit="0ther")  # a rerun at another commit
    assert _decide([part, fill], _b("b4", up))["status"] == "commit_mismatch"
    # the order: no base, leakage, chosen_ref, commit, environment
    other = {**ENV, "torch": "2.12.0+cpu"}
    worst = {**_b("b8", up, env=other, commit="0ther"), "leakage_assertion_passed": False, "chosen_ref_checked": False}
    assert _decide(worst, _b("b4", up))["status"] == "leakage_check_failed"
    assert _decide({**worst, "leakage_assertion_passed": True}, _b("b4", up))["status"] == "chosen_ref_not_checked"
    worst.update(leakage_assertion_passed=True, chosen_ref_checked=True)
    assert _decide(worst, _b("b4", up))["status"] == "commit_mismatch"
    assert _decide(_b("b8", up, env=other), _b("b4", up))["status"] == "env_mismatch"
    r0 = n13.decide_batch(None, worst, _b("b4", up), _it2("base0", BASE0), _it2("null", NULL))
    assert r0["status"] == "no_base" and r0["encoders"]["dinov2_vits14"]["b8"]["words_limits"].startswith("The batch-size floor")


def test_decide_batch_bc_against_stage_a_tolerance_and_slide_sentence(tmp_path):
    from knee.nb import n13

    assert n13.REPRO_TOL == 0.006
    up = {"dinov2_vits14": UP4, "convnext_tiny": DOWN}
    flat = [0.0] * 5
    # DINOv2 fold 0 is -0.0009 from Stage A: a shift of -0.0051 is the tie (-0.0060, inside), -0.0052 is outside
    tie = _decide(_b("b8", up), _b("b4", up), bc=_b("bc", {"dinov2_vits14": [-0.0051, 0, 0, 0, 0], "convnext_tiny": flat}))
    assert tie["units"]["dinov2_vits14_f0"]["bc_minus_stage_a"] == -0.006 and tie["bc_reproduces_stage_a"] is True
    assert tie["bc_minus_stage_a"]["units_outside"] == [] and tie["bc_minus_stage_a"]["words"] is None
    assert len(n13.batch_slide_sentences(tie)) == 4
    low = _decide(_b("b8", up), _b("b4", up), bc=_b("bc", {"dinov2_vits14": [-0.0052, 0, 0, -0.0100, 0], "convnext_tiny": flat}))
    rp = low["bc_minus_stage_a"]
    assert low["status"] == "decided" and low["bc_reproduces_stage_a"] is False and rp["reproduces"] is False
    assert rp["units_outside"] == ["dinov2_vits14_f0", "dinov2_vits14_f3"]  # listed; the verdict is still given
    assert rp["units_outside_difference"] == {"dinov2_vits14_f0": -0.0061, "dinov2_vits14_f3": -0.0115}
    assert rp["words"] == ("The chosen configuration scored 0.0088 lower when refit on CPU in 2 of 10 units "
                           "(mean over those units; tolerance 0.006).")
    assert low["units"]["dinov2_vits14_f0"]["bc_reproduces_stage_a"] is False
    assert low["units"]["dinov2_vits14_f1"]["bc_reproduces_stage_a"] is True
    assert all(low["encoders"][e][a]["verdict"] != "no verdict" for e in MAIN for a in ("b8", "b4"))
    assert n13.batch_slide_sentences(low)[4] == rp["words"] and len(n13.batch_slide_sentences(low)) == 5
    agg.write("n13_batch_decide", low, folder=tmp_path / "low")
    assert max(len(x) for x in _strings(json.loads((tmp_path / "low" / "agg_n13_batch_decide.json").read_text()))) <= 200
    # a unit above Stage A by more than the tolerance is listed too, with its own sentence
    both = _decide(_b("b8", up), _b("b4", up), bc=_b("bc", {"dinov2_vits14": [-0.0052, 0, 0, 0, 0],
                                                            "convnext_tiny": [0, 0, 0.0090, 0, 0]}))
    assert both["bc_minus_stage_a"]["units_outside"] == ["dinov2_vits14_f0", "convnext_tiny_f2"]
    assert both["bc_minus_stage_a"]["words"] == (
        "The chosen configuration scored 0.0061 lower when refit on CPU in 1 of 10 units (mean over those units; "
        "tolerance 0.006). The chosen configuration scored 0.0074 higher when refit on CPU in 1 of 10 units "
        "(mean over those units; tolerance 0.006).")
    # a missing bc unit is not checked (and not called a failure); no bc unit at all: nothing reproduces
    part = _decide(_b("b8", up), _b("b4", up), bc=_b("bc", stopped={("convnext_tiny", 1)}))
    assert part["units"]["convnext_tiny_f1"]["bc_reproduces_stage_a"] is None
    assert len(part["bc_minus_stage_a"]["units_checked"]) == 9 and part["bc_reproduces_stage_a"] is True
    none = _decide(_b("b8", up), _b("b4", up), bc=_b("bc", stopped={(e, k) for e in MAIN for k in range(5)}))
    assert none["bc_reproduces_stage_a"] is False and none["bc_minus_stage_a"]["mean"] is None
    assert none["bc_minus_stage_a"]["max_abs_difference"] is None and none["status"] == "partial"


def test_decide_batch_words_for_every_outcome_and_a_replaced_arm(tmp_path):
    from knee.nb import n13

    zero = [0.001, -0.001, 0.0, 0.0, 0.0]  # gain exactly 0: "did not score higher", printed +0.0000
    neg0 = [-0.00004, 0.0, 0.0, 0.0, 0.0]  # rounds to -0.0: printed +0.0000, never -0.0000
    r = _decide(_b("b8", {"dinov2_vits14": zero, "convnext_tiny": neg0}),
                _b("b4", {"dinov2_vits14": UP4, "convnext_tiny": [0.004] * 5}, stopped={("dinov2_vits14", 4)}))
    e = r["encoders"]
    assert e["dinov2_vits14"]["b8"]["words"] == (
        "Batch 8 did not score higher than the chosen batch on inner validation (DINOv2: gain +0.0000, higher in 1 of 5 folds).")
    assert "gain +0.0000, higher in 0 of 5 folds" in e["convnext_tiny"]["b8"]["words"]
    assert e["convnext_tiny"]["b4"]["words"] == (  # the review's case: small, in 5 of 5 folds, not an effect
        "Batch 4 scored 0.0040 higher than the chosen batch on inner validation (ConvNeXt-T, 5 of 5 folds).")
    assert e["convnext_tiny"]["b4"]["verdict"] == "no support" and e["convnext_tiny"]["b4"]["words_margin"] is not None
    nv = e["dinov2_vits14"]["b4"]
    assert (nv["words"], nv["words_margin"], nv["words_limits"]) == (
        "Batch 4, DINOv2: no verdict (fewer than 5 folds finished in the arm and in bc).", None,
        "The batch-size floor was reached and not extended.")
    lines = n13.batch_slide_sentences(r)
    assert lines[1] == f"{nv['words']} The batch-size floor was reached and not extended." and len(lines) == 4
    assert all("did not score higher" not in x for x in lines[1:2] + lines[3:])
    rn = _decide(_b("b8", {"dinov2_vits14": zero, "convnext_tiny": neg0}), None)
    assert rn["encoders"]["convnext_tiny"]["b4"]["words"] == "Batch 4, ConvNeXt-T: no verdict (the arm did not run)."
    # an arm rerun whole after an environment mismatch: the first run is reported, the rerun enters the rule
    up = {"dinov2_vits14": UP4, "convnext_tiny": UP4}
    other = {**ENV, "kaggle_image_build_date": "20261124-000000"}
    first = _b("b8", {"dinov2_vits14": DOWN, "convnext_tiny": DOWN}, env=other)
    assert _decide(first, _b("b4", up))["status"] == "env_mismatch"
    rr = _decide(_b("b8", up), _b("b4", up), replaced={"b8": first, "b4": None})
    assert rr["status"] == "decided" and rr["encoders"]["dinov2_vits14"]["b8"]["verdict"] == "the floor mattered"
    old = rr["replaced_runs"]
    assert list(old) == ["b8"] and old["b8"]["steps"] == ["n13_b8"] and old["b8"]["environments"] == [other]
    assert old["b8"]["inner_scores"]["dinov2_vits14_f0"] == round(BC["dinov2_vits14"][0] + DOWN[0], 4)
    assert len(old["b8"]["inner_scores"]) == 10 and "reported only" in old["b8"]["use"]
    agg.write("n13_batch_decide", rr, folder=tmp_path)
    with pytest.raises(ValueError):
        _decide(_b("b8", up), _b("b4", up), replaced={"b2": first})
    with pytest.raises(ValueError):  # the replaced file must be that arm's
        _decide(_b("b8", up), _b("b4", up), replaced={"b4": first})


# ---------------------------------------------------------------- the arm run path is commit 39c3c6f's

DECIDE_ONLY = {"decide_batch"}  # changed after ARM
DECIDE_NEW = {"BATCH_DECLARATION", "ARM_COMMIT", "REPRO_TOL", "ENC_WORDS", "WORDS_LIMITS", "WORDS_CANNOT",
              "WORDS_FLOOR", "batch_slide_sentences"}  # added after ARM


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


def test_everything_reachable_from_an_arm_run_is_textually_commit_39c3c6f():
    """The Kaggle sessions of bc, b8 and b4 install commit 39c3c6f. Later commits may change the local decide
    step only: (1) every file under src/ and pyproject.toml is that commit's, except n13.py; (2) in n13.py every
    top-level statement of that commit is unchanged in text and order, except decide_batch; (3) nothing that
    n13.run reaches, by name, is decide_batch or a name added later."""
    if _git("rev-parse", "--verify", f"{ARM}^{{commit}}") is None:
        pytest.skip("git history with commit 39c3c6f is not available")
    tracked = [f for f in _git("ls-tree", "-r", "--name-only", ARM, "--", "src", "pyproject.toml").decode().splitlines()]
    assert "src/knee/nb/n13.py" in tracked and "src/knee/heads.py" in tracked and "src/knee/nb/n6.py" in tracked
    for f in tracked:
        if f != "src/knee/nb/n13.py":
            assert _git("show", f"{ARM}:{f}") == (REPO / f).read_bytes(), f
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / "src").rglob("*.py")}
    assert on_disk <= set(tracked)  # no module was added
    old, new = _top(_git("show", f"{ARM}:src/knee/nb/n13.py").decode()), _top((REPO / "src/knee/nb/n13.py").read_text())
    kept_old = [text for names, text in old if not names & DECIDE_ONLY]
    kept_new = [text for names, text in new if not names & (DECIDE_ONLY | DECIDE_NEW)]
    assert kept_old == kept_new and len(kept_old) == len(old) - 1  # same text, same order; one statement changed
    added = set().union(*(names for names, _ in new)) - set().union(*(names for names, _ in old))
    assert added == DECIDE_NEW
    by_name = {name: text for names, text in new for name in names}
    reach, todo = set(), ["run"]
    while todo:  # every module-level name that n13.run can reach, by reference
        name = todo.pop()
        if name in reach:
            continue
        reach.add(name)
        todo += [x for x in _refs(by_name[name]) if x in by_name]
    assert {"run", "run_batch", "_batch_task", "_batch_unit", "batch_summary", "batch_settings", "chosen_configs",
            "environment", "BATCH_ARMS", "BATCH_PLAN", "CHOSEN_REF", "BATCH_BUDGET_H"} <= reach
    assert not reach & (DECIDE_ONLY | DECIDE_NEW)
    assert all(by_name[name] in kept_old for name in reach)  # each is the text of commit 39c3c6f
    from knee.nb import n13

    assert n13.ARM_COMMIT == ARM and n13.BATCH_PLAN.startswith("it2b_batch_floor_r1.md")  # the arms' own label


# ---------------------------------------------------------------- the notebooks

def test_the_three_notebooks_are_registered_as_cpu_notebooks():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import make_notebooks as mk
    finally:
        sys.path.remove(str(REPO / "scripts"))
    card = (REPO / "notebooks" / "RUNBOOK.md").read_text().split("## Batch-floor check (it2b; added after the results)")[1]
    card = card.split("## If a run fails")[0]
    for arm in ("bc", "b8", "b4"):
        name = f"n13-{arm}"
        gpu, comp, tables, up, timeout = mk.NOTEBOOKS[name]
        assert (gpu, comp, tables) == (False, False, False) and up == ["n0", "n5", "n3-r", "n6a"]
        assert mk.cell2(name) == (f'from knee.nb import n13; n13.run("{arm}", budget_h=6.0, expect={mk.EXPECT})')
        meta = mk.metadata("someone", name)
        assert meta["enable_gpu"] is False and "machine_shape" not in meta and meta["enable_internet"] is True
        assert meta["kernel_sources"] == ["someone/knee-n0", "someone/knee-n5", "someone/knee-n3-r", "someone/knee-n6a"]
        assert meta["competition_sources"] == [] and meta["dataset_sources"] == []
        assert timeout >= 1.1 * 6.0 * 3600  # above the step's watchdog
        assert f"| knee-{name} | None (CPU) |" in card and f"`{mk.cell2(name)}`" in card
    assert "knee-n0, knee-n5, knee-n3-r and knee-n6a" in card and "n13.decide_batch(bc, b8, b4, base0, null)" in card
    # the card of declaration r2: the commit of the three sessions, the notebook to fork, no retry after a crash
    assert mk.BATCH_COMMIT == ARM and f'`COMMIT = "{ARM}"`' in card and "it2b_batch_floor_r2.md" in card
    assert "fork of knee-n13-base0" in card and "original environment" in card and "replace cell 2" in card
    assert "A worker crash is not retried" in card and "Do not restart the session" in card
    for status in ("no_base", "leakage_check_failed", "chosen_ref_not_checked", "commit_mismatch", "env_mismatch"):
        assert f'"{status}"' in card
    for name in ("n13-bc", "n13-b8", "n13-b4"):  # the generator pins cell 1 of the three, whatever --commit says
        assert mk.commit_for(name, "0123abc") == ARM
        assert mk.notebook_json(mk.commit_for(name, "0123abc"), mk.cell2(name))["cells"][0]["source"][0] == f'COMMIT = "{ARM}"\n'
    assert mk.commit_for("n13-base0", "0123abc") == "0123abc"
