"""The slice-count check (it2c_slice_count_r2.md): n13s's arms s32, s16, s16b, s8 and s4 and n13s.decide_slices.

Synthetic data with 32 slices per plane, CPU. The arms fit the unit's chosen Stage A configuration at its
chosen batch size on its inner folds, as n13's arm bc does; only the slices the head reads differ.
The Kaggle sessions install commit SLICE_ARM_COMMIT (ARM below): one test shows that everything
n13s.run_slices reaches is textually that commit's, so the later commit changed the local decide step only.
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
FIND = list(range(12))
ARMS = ("s32", "s16", "s16b", "s8", "s4")
POS = {"s32": list(range(32)), "s16": list(range(1, 32, 2)), "s16b": list(range(0, 32, 2)),
       "s8": [2, 6, 10, 14, 18, 22, 26, 30], "s4": [4, 12, 20, 28]}  # the declaration's lists, written out
# synthetic chosen configurations; the batch size and best score of each unit are CHOSEN_REF's
CHOSEN = {"pooling": "gated", "planes": "all", "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
          "attn_hidden": 32, "pos_cap": 4.0}


def _git(*args) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_stage_a(inp: Path) -> None:
    """knee-n6a's per-unit summaries (agg_n6a_<encoder>_f<k>.json at the output root), synthetic."""
    from knee.nb import n13

    d = inp / "n6a"
    d.mkdir(parents=True, exist_ok=True)
    for e in MAIN:
        for k, (b, best) in n13.CHOSEN_REF[e].items():
            u = {"encoder": e, "fold": k, "tuning": "nested", "l2_from": "bayesian optimisation on this fold",
                 "l1_winner": {"pooling": "gated", "planes": "all"}, "config": {**CHOSEN, "batch_size": b},
                 "best_inner_score": best, "E": 5, "default_trial_score": round(best - 0.002, 4)}
            (d / f"agg_n6a_{e}_f{k}.json").write_text(json.dumps(u, indent=1))


@pytest.fixture(scope="module")
def stage_env(tmp_path_factory):
    from tests.synth_stage import make_stage

    root = tmp_path_factory.mktemp("it2cstage")
    facts = make_stage(root / "input", S=32)  # the cache's 32 slices per plane
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
    """This process as one of n6's workers on the synthetic stage; n13s's own worker state starts empty."""
    from knee.nb import n6, n13s

    root, _ = stage_env
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    monkeypatch.setattr(n13s, "_S", {})
    return root


def _task(arm, chosen, enc="dinov2_vits14", fold=0, deadline=None):
    return {"id": f"{enc}_f{fold}", "encoder": enc, "fold": fold, "seed": 0, "eligible": FIND, "arm": arm,
            "config_chosen": chosen, "stage_a_best": 0.7, "chosen_sha16": "0" * 16,
            "deadline": time.time() + 3600 if deadline is None else deadline, "dir": "unused"}


def _bc_task(chosen, enc="dinov2_vits14", fold=0):
    return {**_task("s32", chosen, enc, fold), "batch_size": None}  # n13's arm bc: the chosen batch size


# ---------------------------------------------------------------- the arms

def test_slice_arms_positions_and_settings_are_the_declared_ones():
    from knee.nb import n13, n13s

    assert n13s.SLICE_ARMS == {"s32": 32, "s16": 16, "s16b": 16, "s8": 8, "s4": 4} and n13s.N_CACHED == 32
    assert n13s.SLICE_PLAN == "it2c_slice_count_r2.md: the slice-count check, declared before the run, added after the results"
    for arm, n in n13s.SLICE_ARMS.items():
        pos = n13s.slice_positions(arm)
        assert pos == POS[arm] and len(pos) == n
        if arm != "s16b":  # the declared formula: floor((i + 0.5) * 32 / n)
            assert pos == [int(np.floor((i + 0.5) * 32 / n)) for i in range(n)]
        s = n13s.slice_settings(arm)
        assert {k: v for k, v in s.items() if k != "config"} == {k: v for k, v in n13.batch_settings("bc").items() if k != "config"}
        assert s["config"] == {"source": "chosen (knee-n6a)", "batch_size": "chosen", "slices": n, "slices_cached": 32,
                               "slice_positions": f"{pos[0]},{pos[1]},...,{pos[-1]}"}
        agg._check(json.loads(json.dumps(s)))  # 4 slices is a setting under "config", not a small count
    assert sorted(POS["s16"] + POS["s16b"]) == list(range(32))  # s16b: the 16 positions s16 leaves out
    assert n13s.slice_settings("s16") != n13s.slice_settings("s16b")
    assert (n13s.SLICE_SEED, n13s.SLICE_BUDGET_H) == (0, 4.0)
    assert set(n13.BATCH_ARMS) == {"bc", "b8", "b4"} and set(n13.ARMS) == {"a1", "base0", "null", "a7"}  # untouched
    for bad in (lambda: n13s.slice_positions("s2"), lambda: n13s.slice_settings("s64"), lambda: n13s.run_slices("s12"),
                lambda: n13s.run_slices("bc"), lambda: n13s.run_slices("s8", encoders=("vit_s16",)),
                lambda: n13s.run_slices("s8", rerun=1, folds=(2,)), lambda: n13s.run_slices("s8", budget_h=None)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):  # n13.run does not know these arms: nothing in n13 was changed
        n13.run("s16")


def test_slice_bank_cuts_a_copy_and_refuses_a_bank_without_32_slices():
    from knee.nb import n13s

    rng = np.random.default_rng(0)
    N, M, D = 6, 2, 5

    def bank(S, S_flip=None):
        emb = {p: rng.normal(size=(N, S, D)).astype(np.float16) for p in heads.PLANES}
        flip = {p: rng.normal(size=(M, S if S_flip is None else S_flip, D)).astype(np.float16) for p in heads.FLIP_PLANES}
        y = rng.random((N, 12)).astype(np.float32)
        return heads.Bank(emb, y, np.ones((N, 12), np.float32), (y > 0.5).astype(float), flip=flip,
                          flip_row=np.array([0, -1, -1, 1, -1, -1]))

    b = bank(32)
    before = {"x": {p: a.clone() for p, a in b.x.items()}, "flip": {p: a.clone() for p, a in b.flip.items()}}
    ids = {"x": {p: id(a) for p, a in b.x.items()}, "flip": {p: id(a) for p, a in b.flip.items()}}
    assert n13s.slice_bank(b, "s32") is b  # the bank object itself: no indexing, no copy
    for arm in ("s16", "s16b", "s8", "s4"):
        c = n13s.slice_bank(b, arm)
        assert c is not b and isinstance(c, heads.Bank)
        assert set(c.x) == set(heads.PLANES) and set(c.flip) == set(heads.FLIP_PLANES)
        for p in heads.PLANES:
            assert tuple(c.x[p].shape) == (N, len(POS[arm]), D) and c.x[p].dtype == b.x[p].dtype
            assert torch.equal(c.x[p], before["x"][p][:, POS[arm], :])
        for p in heads.FLIP_PLANES:  # the flipped embeddings are cut at the same positions
            assert tuple(c.flip[p].shape) == (M, len(POS[arm]), D)
            assert torch.equal(c.flip[p], before["flip"][p][:, POS[arm], :])
        # labels, weights, flip rows and the scoring labels are the same objects
        assert c.y is b.y and c.w is b.w and c.binary is b.binary and c.eval_y is b.eval_y and c.flip_row is b.flip_row
        assert (c.n, c.device) == (b.n, b.device)
        c.x["sagittal"].zero_()  # a copy: writing to it does not reach the source
    for kind in ("x", "flip"):  # the source bank is not modified
        for p, a in getattr(b, kind).items():
            assert id(a) == ids[kind][p] and torch.equal(a, before[kind][p]) and a.shape[1] == 32
    # statistics over the kept slices only
    st = n13s.slice_bank(b, "s4").stats(np.arange(N))
    want = before["x"]["coronal"][:, POS["s4"], :].double().reshape(-1, D).mean(0).float()
    assert torch.allclose(st["coronal"][0], want, atol=1e-6)
    for bad in (bank(8), bank(16), bank(32, S_flip=8)):
        for arm in ARMS:  # s32 too: the arm's name says 32
            with pytest.raises(ValueError):
                n13s.slice_bank(bad, arm)


def test_s32_is_arm_bc_on_the_untouched_bank(worker, tmp_path, monkeypatch, one_thread):
    """s32 fits on the worker's cached bank object and returns arm bc's unit score exactly."""
    from knee.nb import n6, n13, n13s

    chosen = n13.chosen_configs(worker / "input")[("dinov2_vits14", 1)]["config"]
    bc = n13._batch_unit(_bc_task(chosen, fold=1), tmp_path / "never_made")
    cached = n6._W["bank"]
    seen = []
    real_fit = heads.fit

    def fit(bank, cfg, *a, **kw):
        seen.append((bank, kw["stats"]))
        return real_fit(bank, cfg, *a, **kw)

    monkeypatch.setattr(heads, "fit", fit)
    s32 = n13s._slice_unit(_task("s32", chosen, fold=1), tmp_path / "never_made")
    assert len(seen) == 4 and all(b is cached for b, _ in seen) and n6._W["bank"] is cached and n13s._S == {}
    assert s32["ok"] and s32["fits"] == s32["guard_checks"] == bc["fits"] == 4
    a, b = s32["summary"], bc["summary"]
    assert a["inner_score"] == b["inner_score"] and (a["E"], a["epochs"]) == (b["E"], b["epochs"])
    extra = {"slices"}
    assert set(a) - set(b) == extra and set(b) <= set(a)
    assert {k: v for k, v in a.items() if k not in extra | {"seconds"}} == {k: v for k, v in b.items() if k != "seconds"}
    assert a["slices"] == {"config": {"slices": 32, "slices_cached": 32, "slice_positions": "0,1,...,31"}}
    assert a["config"] == chosen and a["step_ratio"] == 1.0 and isinstance(a["step_ratio"], float)
    assert not (tmp_path / "never_made").exists()


def test_an_arm_passes_exactly_its_slices_to_the_head_and_leaves_the_cached_bank_alone(worker, tmp_path, monkeypatch, one_thread):
    from knee import stage
    from knee.nb import n6, n13, n13s

    chosen = n13.chosen_configs(worker / "input")[("dinov2_vits14", 0)]["config"]
    cached = n6._bank("dinov2_vits14")
    assert n6._W["bank"] is cached and all(a.shape[1] == 32 for a in cached.x.values())
    assert cached.flip and (cached.flip_row >= 0).any()  # the stage has unresolved-laterality studies
    before = {"x": {p: a.clone() for p, a in cached.x.items()}, "flip": {p: a.clone() for p, a in cached.flip.items()}}
    ids = {"x": {p: id(a) for p, a in cached.x.items()}, "flip": {p: id(a) for p, a in cached.flip.items()}}
    d = stage.StageData(worker / "input")
    tr, te, _ = d.split(0)
    fits = []
    real_fit = heads.fit

    def fit(bank, cfg, train_idx, val_idx=None, findings=None, **kw):
        fits.append({"bank": bank, "cfg": cfg.to_dict(), "train": np.asarray(train_idx).copy(),
                     "val": np.asarray(val_idx).copy(), "findings": findings, **kw})
        return real_fit(bank, cfg, train_idx, val_idx, findings, **kw)

    monkeypatch.setattr(heads, "fit", fit)
    out, calls = {}, {}
    for arm in ARMS:
        fits.clear()
        r = n13s._slice_unit(_task(arm, chosen), tmp_path / "never_made")
        assert r["ok"] and r["fits"] == r["guard_checks"] == 4  # every inner fit passed the leakage guard
        out[arm], calls[arm] = r["summary"], list(fits)
    for arm in ARMS:
        n, pos, u = len(POS[arm]), POS[arm], out[arm]
        bank = calls[arm][0]["bank"]
        assert all(c["bank"] is bank for c in calls[arm])  # one bank for the unit's inner fits
        assert (bank is cached) == (arm == "s32")
        for p in heads.PLANES:  # exactly n slices per plane, at the declared positions, the same in every plane
            assert tuple(bank.x[p].shape) == (cached.n, n, before["x"][p].shape[2])
            assert torch.equal(bank.x[p], before["x"][p][:, pos, :])
        for p in heads.FLIP_PLANES:  # and the flipped embeddings are cut the same way
            assert tuple(bank.flip[p].shape) == (before["flip"][p].shape[0], n, before["flip"][p].shape[2])
            assert torch.equal(bank.flip[p], before["flip"][p][:, pos, :])
        assert bank.y is cached.y and bank.w is cached.w and bank.binary is cached.binary
        assert bank.eval_y is cached.eval_y and bank.flip_row is cached.flip_row
        want = bank.stats(tr)  # the statistics: recomputed over the kept slices of the outer training set
        for c in calls[arm]:
            for p in heads.PLANES:
                assert torch.equal(c["stats"][p][0], want[p][0]) and torch.equal(c["stats"][p][1], want[p][1])
                direct = before["x"][p][torch.as_tensor(tr)][:, pos, :].double().reshape(-1, before["x"][p].shape[2])
                assert torch.allclose(c["stats"][p][0], direct.mean(0).float(), atol=1e-5)
            # everything else is arm bc's inner fit: the chosen configuration, seed + j, schedule, patience, rows
            assert c["cfg"] == chosen and c["findings"] == FIND
            assert (c.get("epochs"), c["patience"], c["schedule_epochs"], c.get("max_epochs"), c.get("steps")) == (None, 5, None, None, None)
            assert set(te.tolist()) | set(d.gold_rows().tolist()) == set(c["forbidden"].tolist())
            assert set(c["train"].tolist()) | set(c["val"].tolist()) <= set(tr.tolist())
        assert [c["seed"] for c in calls[arm]] == [0, 1, 2, 3]
        for a, c in zip(calls["s32"], calls[arm]):  # the same inner training and validation rows in every arm
            assert np.array_equal(a["train"], c["train"]) and np.array_equal(a["val"], c["val"])
        assert u["slices"] == {"config": {"slices": n, "slices_cached": 32, "slice_positions": f"{pos[0]},{pos[1]},...,{pos[-1]}"}}
        assert u["config"] == chosen and u["batch_chosen"] == 33 and u["step_ratio"] == 1.0
        assert u["steps_per_epoch"] == u["steps_per_epoch_chosen"] == out["s32"]["steps_per_epoch"]
        assert (u["schedule_epochs"], u["patience"], u["seed"], u["inner_k"], u["device"]) == (30, 5, 0, 4, "cpu")
        assert 0.5 < u["inner_score"] <= 1 and round(u["inner_score"], 4) == u["inner_score"]
        agg._check(json.loads(json.dumps(u)))  # 4 slices sits under a "config" key
    assert len({out[a]["inner_score"] for a in ARMS}) > 1  # the arms do not all see the same input
    assert not torch.equal(calls["s16"][0]["bank"].x["sagittal"], calls["s16b"][0]["bank"].x["sagittal"])
    # the worker's cached bank: the same object, the same tensors, the same values
    assert n6._W["bank"] is cached and n6._bank("dinov2_vits14") is cached
    for kind in ("x", "flip"):
        for p, a in getattr(cached, kind).items():
            assert id(a) == ids[kind][p] and a.shape[1] == 32 and torch.equal(a, before[kind][p])
    # the cut copy is kept for the next unit of the same encoder and arm, and replaced when the arm changes
    b4 = n13s._arm_bank("dinov2_vits14", "s4")
    assert n13s._arm_bank("dinov2_vits14", "s4") is b4 and n13s._arm_bank("dinov2_vits14", "s8") is not b4
    assert not (tmp_path / "never_made").exists()  # a unit writes nothing


def test_run_s4_end_to_end_passes_the_privacy_check_and_writes_no_test_fold_file(stage_env, monkeypatch, capfd):
    from knee.nb import n13, n13s

    root, facts = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_s4")
    out = n13s.run_slices("s4", root=root / "input", workers=2, encoders=("dinov2_vits14",), folds=(0, 1), budget_h=1.0)
    printed = capfd.readouterr()
    every = sorted(p for p in paths.WORKING.rglob("*") if p.is_file())
    assert [p.relative_to(paths.WORKING).as_posix() for p in every] == ["agg_n13_s4_inner.json", "run.json"]
    text = printed.out + printed.err + "".join(p.read_text() for p in every)
    assert not [u for u in facts["uids"] if u in text]  # no study identifier anywhere Claude reads
    assert "STEP n13_s4: OK" in printed.out
    assert sorted(p.name for p in (paths.WORKING / "private" / "n13_s4").iterdir()) == ["errors"]  # no unit folder
    rj = json.loads((paths.WORKING / "run.json").read_text())
    c = rj["config"]
    assert rj["step"] == "n13_s4" and (c["arm"], c["batch_size"], c["seed"], c["budget_h"], c["rerun"]) == ("s4", "chosen", 0, 1.0, 0)
    assert c["config"] == {"slices": 4, "slices_cached": 32, "slice_positions": "4,12,...,28"}
    assert c["test_overrides"] is True and c["test_overrides_reason"] == n13s.SLICE_REASON
    assert c["added_after_results"] is True and c["chosen_ref_checked"] is True and c["plan"] == n13s.SLICE_PLAN
    assert c["environment"] == out["environment"] == n13.environment()
    back = json.loads((paths.WORKING / "agg_n13_s4_inner.json").read_text())
    assert back == json.loads(json.dumps(out))
    agg._check(back)  # the aggregate of an s4 arm passes the privacy check (it was written through agg.write)
    assert (out["arm"], out["step"], out["status"], out["all_units_ok"]) == ("s4", "n13_s4", "ok", True)
    assert out["added_after_results"] is True and out["test_overrides"] is True and out["plan"] == n13s.SLICE_PLAN
    assert out["leakage_assertion_passed"] is True and out["device_names"] == ["cpu"] and out["budget_h"] == 1.0
    assert out["settings"] == n13s.slice_settings("s4") and out["chosen_ref_checked"] is True
    assert out["test_fold_use"].startswith("none") and out["seconds"] > 0 and "knee_commit" in out
    assert out["units_finished"] == ["dinov2_vits14_f0", "dinov2_vits14_f1"] and out["units_missing"] == []
    assert set(out["upstream"]) == {"n0_studies", "n3_folds", "n5_index"}
    for k, chosen_b in ((0, 33), (1, 16)):
        u = out["units"][f"dinov2_vits14_f{k}"]
        assert u["ok"] and u["config"] == {**CHOSEN, "batch_size": chosen_b} and u["batch_chosen"] == chosen_b
        assert u["slices"] == {"config": {"slices": 4, "slices_cached": 32, "slice_positions": "4,12,...,28"}}
        assert out["inner_scores"]["dinov2_vits14"][k] == u["inner_score"]
        assert u["stage_a_best_inner_score"] == n13.CHOSEN_REF["dinov2_vits14"][k][1]
    m = n13.merged(out)  # the reader of the rule takes it (reruns, steps, environments)
    assert m["arm"] == "s4" and sorted(m["units"]) == [("dinov2_vits14", 0), ("dinov2_vits14", 1)]
    leak = copy.deepcopy(back)
    leak["units"]["dinov2_vits14_f0"]["slices"] = 4  # outside "config" a 4 is a small count
    with pytest.raises(agg.UnsafeSummary):
        agg._check(leak)


def test_soft_stop_gpu_refusal_and_the_summary_flags(stage_env, worker, tmp_path, monkeypatch, capfd, one_thread):
    from knee.nb import n13, n13s

    root, _ = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_s16b_stop")
    out = n13s.run_slices("s16b", root=root / "input", workers=2, encoders=("dinov2_vits14",), folds=(0, 1), budget_h=0.0)
    assert "STEP n13_s16b: OK" in capfd.readouterr().out  # a stopped session is not a failed step
    assert (out["status"], out["all_units_ok"], out["units_finished"]) == ("partial", False, [])
    assert out["units_missing"] == ["dinov2_vits14_f0", "dinov2_vits14_f1"] and out["inner_scores"] == {}
    assert all(u == {"ok": False, "stopped": True, "error": None} for u in out["units"].values())
    assert json.loads((paths.WORKING / "agg_n13_s16b_inner.json").read_text()) == json.loads(json.dumps(out))
    # a GPU session is refused before any data is read
    monkeypatch.setattr(paths, "WORKING", root / "working_s8_gpu")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(BaseException):
        n13s.run_slices("s8", root=root / "input", workers=1, encoders=("dinov2_vits14",), folds=(0,), budget_h=1.0)
    assert "STEP n13_s8: FAILED - ValueError" in capfd.readouterr().out
    assert not (paths.WORKING / "agg_n13_s8_inner.json").exists()
    monkeypatch.undo()
    # a unit past its deadline does not start; a deadline passing during the second fit ends it with its epoch
    monkeypatch.setattr(n13s, "_S", {})
    from knee.nb import n6

    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    chosen = n13.chosen_configs(root / "input")[("dinov2_vits14", 0)]["config"]
    assert n13s._slice_unit(_task("s8", chosen, deadline=0.0), tmp_path / "x") == {"task": "dinov2_vits14_f0", "ok": False, "stopped": True}
    task = _task("s8", chosen)
    real_fit, runs = heads.fit, []

    def fit(*a, **kw):
        if runs:
            task["deadline"] = 0.0  # the budget ends while the second fit runs
        r = real_fit(*a, **kw)
        runs.append(r.epochs_run)
        return r

    monkeypatch.setattr(heads, "fit", fit)
    r = n13s._slice_unit(task, tmp_path / "x")
    assert r == {"task": "dinov2_vits14_f0", "ok": False, "stopped": True} and len(runs) == 2 and runs[1] == 1
    monkeypatch.setattr(heads, "fit", real_fit)
    done = n13s._slice_unit(_task("s8", chosen), tmp_path / "x")
    s = n13s.slice_summary("s8", "n13_s8", {"dinov2_vits14_f0": done, "dinov2_vits14_f1": r},
                           ["dinov2_vits14_f0", "dinov2_vits14_f1", "dinov2_vits14_f2"], n13.environment(),
                           ("dinov2_vits14",), budget_h=4.0)
    assert (s["status"], s["units_finished"]) == ("partial", ["dinov2_vits14_f0"])  # the finished unit is written
    assert s["units_missing"] == ["dinov2_vits14_f1", "dinov2_vits14_f2"] and s["leakage_assertion_passed"] is True
    assert s["inner_scores"] == {"dinov2_vits14": {0: done["summary"]["inner_score"]}}
    agg.write("n13_s8_inner", s, folder=tmp_path / "agg")
    failed = {"task": "dinov2_vits14_f1", "ok": False, "error": "LeakageError at heads.py:209 in _check_guard()"}
    s2 = n13s.slice_summary("s8", "n13_s8", {"dinov2_vits14_f0": done, "dinov2_vits14_f1": failed},
                            ["dinov2_vits14_f0", "dinov2_vits14_f1"], None, ("dinov2_vits14",))
    assert s2["leakage_assertion_passed"] is False and s2["units"]["dinov2_vits14_f1"]["stopped"] is False
    s3 = n13s.slice_summary("s8", "n13_s8", {"dinov2_vits14_f0": {**done, "guard_checks": 3}}, ["dinov2_vits14_f0"],
                            None, ("dinov2_vits14",))
    assert s3["leakage_assertion_passed"] is False  # a fit that did not pass the guard


# ---------------------------------------------------------------- decide_slices

ENV = {"kaggle_image_git_commit": "e1dbc17d", "kaggle_image_build_date": "20261117-122508", "python": "3.13.18",
       "torch": "2.11.0+cpu", "cuda": None, "cudnn": None, "device_name": "cpu"}
BC_COMMIT = "39c3c6fc7626e047ed643d5253e8ce9e662302af"  # the commit arm bc ran at
# iteration 2's inner scores (n13_numbers.md): s = 0.0030
BASE0 = {"convnext_tiny": [0.7240, 0.7444, 0.7393, 0.7476, 0.7418], "dinov2_vits14": [0.7227, 0.7398, 0.7327, 0.7398, 0.7367]}
NULL = {"convnext_tiny": [0.7192, 0.7463, 0.7432, 0.7472, 0.7381], "dinov2_vits14": [0.7188, 0.7390, 0.7327, 0.7404, 0.7318]}
S32 = {"dinov2_vits14": [0.7240, 0.7410, 0.7350, 0.7420, 0.7360], "convnext_tiny": [0.7250, 0.7480, 0.7450, 0.7500, 0.7440]}
DN, CX = MAIN
ENDING = ("One seed. Head settings were tuned at 32 slices and not re-tuned. More than 32 slices was not tested: "
          "it needs a new cache. Differences under 0.005 cannot be shown by this check.")
NOT_MET = "This does not meet the declared rule (more than 0.005 and 4 of 5 folds), so it is not counted as an effect."
FIXED = "The slice count was fixed at 32 and not searched."
NONE_LINE = ("No count from 4 to 32 differed by the declared margin. This check does not show that 32 is needed and "
             "does not show that fewer is enough.")
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


def _units(diffs, stopped, only, extra):
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
            v = round(S32[e][k] + ((diffs or {}).get(e) or FLAT)[k], 4)
            units[uid] = {"ok": True, "inner_score": v, "E": 4, "epochs": [9, 9, 8, 3], "inner_k": 4, "seed": 0,
                          "schedule_epochs": 30, "patience": 5, "device": "cpu", "device_name": "cpu",
                          "config": {**CHOSEN, "batch_size": b}, "batch_chosen": b, **extra,
                          "steps_per_epoch": int(5 * round(3000 / b / 5)),
                          "steps_per_epoch_chosen": int(5 * round(3000 / b / 5)), "step_ratio": 1.0,
                          "stage_a_best_inner_score": best, "chosen_sha16": "0" * 16, "seconds": 100.0}
            scores.setdefault(e, {})[k] = v
    return units, scores, ids


def _s(arm, diffs=None, env=ENV, stopped=(), step=None, only=None, commit="ARM"):
    """A slice-arm summary shaped as slice_summary writes it: S32 plus diffs {encoder: [5 differences]}."""
    from knee.nb import n13s

    units, scores, ids = _units(diffs, stopped, only, {"slices": {"config": n13s.slice_config(arm)}})
    done = [u for u in ids if units[u]["ok"]]
    return {"arm": arm, "step": step or f"n13_{arm}", "plan": n13s.SLICE_PLAN, "added_after_results": True,
            "test_overrides": True, "test_overrides_reason": n13s.SLICE_REASON, "encoders": list(MAIN),
            "settings": n13s.slice_settings(arm), "chosen_ref_checked": True, "environment": env,
            "knee_commit": n13s.SLICE_ARM_COMMIT if commit == "ARM" else commit,
            "status": "ok" if len(done) == len(ids) else "partial", "all_units_ok": len(done) == len(ids),
            "units_finished": done, "units_missing": [u for u in ids if not units[u]["ok"]],
            "leakage_assertion_passed": True, "device_names": ["cpu"], "score": n13s.SLICE_SCORE, "budget_h": 4.0,
            "seconds": 2000.0, "inner_scores": scores, "units": units}


def _bc(diffs=None, env=ENV, stopped=()):
    """Arm bc's summary as n13.batch_summary wrote it (commit 39c3c6f): S32 plus diffs."""
    from knee.nb import n13

    units, scores, ids = _units(diffs, stopped, None, {})
    done = [u for u in ids if units[u]["ok"]]
    return {"arm": "bc", "step": "n13_bc", "plan": n13.BATCH_PLAN, "added_after_results": True, "encoders": list(MAIN),
            "settings": n13.batch_settings("bc"), "chosen_ref_checked": True, "environment": env, "knee_commit": BC_COMMIT,
            "status": "ok", "units_finished": done, "leakage_assertion_passed": True, "inner_scores": scores, "units": units}


def _decide(s16=None, s8=None, s4=None, s16b=None, s32="base", bc="same", **kw):
    """decide_slices on the given difference tables {encoder: [5]} or ready summaries; None: the arm did not run."""
    from knee.nb import n13s

    def arm(name, x):
        return _s(name, x) if isinstance(x, dict) and "arm" not in x else x

    return n13s.decide_slices(_s("s32") if s32 == "base" else s32, arm("s16", s16), arm("s8", s8), arm("s4", s4),
                              _bc() if bc == "same" else bc, _it2("base0", BASE0), _it2("null", NULL),
                              s16b=arm("s16b", s16b), **kw)


def _strings(o):
    if isinstance(o, dict):
        return [s for k, v in o.items() for s in [str(k), *_strings(v)]]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return [o] if isinstance(o, str) else []


def _checked(r, folder):
    """The record passes the aggregate checker and holds no long string and no forbidden word."""
    from knee.nb import n13s

    p = agg.write("n13_slices_decide", r, folder=folder)
    back = json.loads(p.read_text())
    assert max(len(x) for x in _strings(back)) <= 200
    for line in n13s.slice_slide_sentences(r):
        assert "optimal" not in line.lower() and "best" not in line.lower()
    return back


def test_decide_slices_verdicts_by_the_declared_rule(tmp_path):
    from knee.nb import n13, n13s

    assert (n13s.COST, n13s.HIGHER, n13s.NO_DIFF, n13s.NO_VERDICT) == (
        "fewer slices cost", "fewer slices scored higher", "no difference shown", "no verdict")
    assert n13s.VERDICT_ARMS == ("s16", "s8", "s4") and n13s.RERUN_TOL == 0.0005
    r = _decide(s16={DN: SMALL_DOWN, CX: TIE_DOWN}, s8={DN: DOWN3, CX: UP4}, s4={DN: DOWN4, CX: ZERO_FOLDS},
                s16b={DN: SMALL_DOWN, CX: FLAT})
    assert r["status"] == "decided" and r["added_after_results"] is True and r["same_knee_commit"] is True
    assert round(r["s"], 4) == 0.0030 == r["s_declared"] and r["threshold"] == 0.005 and r["min_gain"] == 0.005
    assert abs(r["threshold_unrounded"] - 0.005) < 1e-12  # 2 s / sqrt(5) = 0.0027 does not bind
    assert r["plan"] == n13s.SLICE_PLAN and r["declaration"].startswith("it2c_slice_count_r2.md")
    assert r["knee_commits"] == [n13s.SLICE_ARM_COMMIT] and r["knee_commit_is_declared"] is True
    assert r["leakage_assertion_passed"] is True and r["chosen_ref_checked"] is True and r["replaced_runs"] == {}
    e = r["encoders"]
    assert set(e) == set(MAIN) and all(set(v) == {"s16", "s8", "s4"} for v in e.values())  # never a verdict for s16b

    def got(enc, arm):
        x = e[enc][arm]
        return x["verdict"], x["diff"], len(x["lower"]["folds"]), len(x["higher"]["folds"])

    assert got(DN, "s4") == ("fewer slices cost", -0.0058, 4, 1)
    assert got(DN, "s8") == ("no difference shown", -0.006, 3, 2)  # the difference alone is not enough
    assert got(DN, "s16") == ("no difference shown", -0.0028, 4, 1)
    assert got(CX, "s16") == ("no difference shown", -0.005, 5, 0)  # a 4-decimal tie at 0.0050 is inside
    assert got(CX, "s8") == ("fewer slices scored higher", 0.0058, 1, 4)
    assert got(CX, "s4") == ("no difference shown", -0.007, 3, 0)  # folds at 0.0000 are neither lower nor higher
    assert e[CX]["s4"]["per_fold"] == {0: -0.02, 1: -0.01, 2: -0.005, 3: 0.0, 4: 0.0}
    assert e[CX]["s4"]["lower"] == {"folds": [0, 1, 2]} and e[CX]["s4"]["higher"] == {"folds": []}
    assert e[DN]["s4"]["folds"] == [0, 1, 2, 3, 4] and e[CX]["s8"]["folds_up"] == 4
    # section 6's sentences
    w = {(enc, arm): (e[enc][arm]["words"], e[enc][arm]["words_extra"], e[enc][arm]["words_limits"]) for enc in MAIN for arm in ("s16", "s8", "s4")}
    assert w[(DN, "s4")] == ("With 4 of the 32 cached slices per plane the head scored 0.0058 lower on inner validation "
                             "(DINOv2, lower in 4 of 5 folds).", None, ENDING)
    assert w[(CX, "s8")] == ("With 8 of the 32 cached slices per plane the head scored 0.0058 higher on inner validation "
                             "(ConvNeXt-T, higher in 4 of 5 folds).", "The models were not changed.", ENDING)
    assert w[(DN, "s8")] == ("With 8 of the 32 cached slices per plane the head scored 0.0060 lower on inner validation "
                             "(DINOv2, lower in 3 of 5 folds).", NOT_MET, ENDING)
    assert w[(CX, "s16")] == ("With 16 of the 32 cached slices per plane the head scored 0.0050 lower on inner validation "
                              "(ConvNeXt-T, lower in 5 of 5 folds).", NOT_MET, ENDING)
    lines = n13s.slice_slide_sentences(r)
    assert lines[:6] == [" ".join(p for p in w[(enc, arm)] if p) for enc in MAIN for arm in ("s16", "s8", "s4")]
    assert all(x.endswith(ENDING) for x in lines[:6])
    # the summary line: a cost verdict names the largest such count; a "scored higher" is an open point
    assert r["summary_words"] == lines[6:] == [
        "Fewer slices cost from 4 down. 32 is the largest count tested, so the curve may still rise beyond it.",
        "Open point: with 8 of the 32 cached slices per plane the head scored higher (ConvNeXt-T). The models were not changed."]
    # descriptive: the rerun check, the position effect, the table, the medians
    rc = r["rerun_check"]
    assert (rc["used_in_rule"], rc["identical"], rc["units_outside"], rc["words"], rc["max_abs_difference"]) == (False, True, [], None, 0.0)
    assert r["s32_reruns_bc"] is True and len(rc["units_checked"]) == 10 and rc["bc_knee_commits"] == [BC_COMMIT]
    assert rc["bc_env_differences"] == []
    pe = r["position_effect"]
    assert pe["used_in_rule"] is False and pe["s16b_ran"] is True
    assert pe["s16b_minus_s16"][DN] == {"folds": [0, 1, 2, 3, 4], "per_fold": {k: 0.0 for k in range(5)}, "mean": 0.0, "all_folds": True}
    assert pe["s16b_minus_s16"][CX]["mean"] == 0.005 and pe["s16b_minus_s16"][CX]["per_fold"] == {k: 0.005 for k in range(5)}
    assert pe["s16b_minus_s32"][DN]["mean"] == -0.0028 and pe["s16b_minus_s32"][CX]["mean"] == 0.0
    u = r["units"][f"{DN}_f0"]
    assert u["batch_chosen"] == 33 and u["s32"] == {"inner_score": 0.724, "E": 4, "minus_s32": None}
    assert (u["s4"]["inner_score"], u["s4"]["minus_s32"], u["s16b"]["minus_s32"]) == (0.716, -0.008, -0.004)
    assert (u["bc_inner_score"], u["s32_minus_bc"]) == (0.724, 0.0)
    assert r["median_best_epoch"]["s4"] == {"all": 4.0, DN: 4.0, CX: 4.0} and isinstance(r["median_best_epoch"]["s4"]["all"], float)
    assert r["mean_inner_score"][DN]["s32"] == round(float(np.mean(S32[DN])), 4)
    assert r["arms"]["s4"]["config"] == {"slices": 4, "slices_cached": 32, "slice_positions": "4,12,...,28"}
    assert r["arms"]["s16b"]["config"]["slice_positions"] == "0,2,...,30" and len(r["limits"]) == 7
    back = _checked(r, tmp_path)
    assert back["encoders"][DN]["s4"]["verdict"] == n13s.COST
    leak = copy.deepcopy(r)
    leak["units"][f"{DN}_f0"]["s4"]["slices"] = 4  # outside "config" a 4 is a small count
    with pytest.raises(agg.UnsafeSummary):
        agg.write("n13_slices_decide", leak, folder=tmp_path / "leak")
    # the summaries as files give the same record; a rerun fills a stopped unit
    tabs = {"s16": {DN: SMALL_DOWN, CX: TIE_DOWN}, "s8": {DN: DOWN3, CX: UP4}, "s4": {DN: DOWN4, CX: ZERO_FOLDS},
            "s16b": {DN: SMALL_DOWN, CX: FLAT}}
    f = {a: json.loads(json.dumps(_s(a, tabs.get(a)))) for a in ARMS}
    it2 = [json.loads(json.dumps(x)) for x in (_bc(), _it2("base0", BASE0), _it2("null", NULL))]
    again = n13s.decide_slices(f["s32"], f["s16"], f["s8"], f["s4"], *it2, s16b=f["s16b"])
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(r))
    part = _s("s8", tabs["s8"], stopped={(DN, 2)})
    fill = _s("s8", tabs["s8"], only={(DN, 2)}, step="n13_s8_r")
    rr = _decide(s16=tabs["s16"], s8=[part, fill], s4=tabs["s4"], s16b=tabs["s16b"])
    assert rr["encoders"] == r["encoders"] and rr["arms"]["s8"]["steps"] == ["n13_s8", "n13_s8_r"]
    assert n13.MIN_FOLDS_UP == 4


def test_decide_slices_no_difference_sentences_ties_and_the_summary_line(tmp_path):
    from knee.nb import n13s

    neg0 = [-0.00004, 0.0, 0.0, 0.0, 0.0]  # rounds to 0.0000: "the same", never "-0.0000"
    r = _decide(s16={DN: ZERO, CX: neg0}, s8={DN: TIE_UP, CX: UP3}, s4={DN: SMALL_DOWN, CX: [0.004] * 5})
    e = r["encoders"]
    assert r["status"] == "decided" and all(e[enc][a]["verdict"] == "no difference shown" for enc in MAIN for a in ("s16", "s8", "s4"))
    # diff = 0.0000: "scored the same"
    assert (e[DN]["s16"]["words"], e[DN]["s16"]["words_extra"], e[DN]["s16"]["diff"]) == (
        "With 16 of the 32 cached slices per plane the head scored the same on inner validation (DINOv2).", None, 0.0)
    assert e[CX]["s16"]["words"] == "With 16 of the 32 cached slices per plane the head scored the same on inner validation (ConvNeXt-T)."
    assert str(e[CX]["s16"]["diff"]) == "0.0" and e[CX]["s16"]["lower"] == {"folds": []}  # -0.00004 is 0.0000 at 4 decimals
    assert len(e[DN]["s16"]["lower"]["folds"]) == len(e[DN]["s16"]["higher"]["folds"]) == 1
    # diff > 0: "higher" in both places; the tie at +0.0050 is inside
    assert (e[DN]["s8"]["diff"], e[DN]["s8"]["words"], e[DN]["s8"]["words_extra"]) == (
        0.005, "With 8 of the 32 cached slices per plane the head scored 0.0050 higher on inner validation "
               "(DINOv2, higher in 5 of 5 folds).", NOT_MET)
    assert e[CX]["s8"]["words"] == ("With 8 of the 32 cached slices per plane the head scored 0.0060 higher on inner "
                                    "validation (ConvNeXt-T, higher in 3 of 5 folds).")
    assert e[CX]["s4"]["words"].startswith("With 4 of the 32 cached slices per plane the head scored 0.0040 higher")
    # diff < 0
    assert (e[DN]["s4"]["words"], e[DN]["s4"]["words_extra"]) == (
        "With 4 of the 32 cached slices per plane the head scored 0.0028 lower on inner validation "
        "(DINOv2, lower in 4 of 5 folds).", NOT_MET)
    lines = n13s.slice_slide_sentences(r)
    assert len(lines) == 7 and lines[6] == NONE_LINE and r["summary_words"] == [NONE_LINE]  # no arm differed
    assert lines[0] == f"{e[DN]['s16']['words']} {ENDING}" and lines[2] == f"{e[DN]['s4']['words']} {NOT_MET} {ENDING}"
    assert r["position_effect"]["s16b_ran"] is False and r["arms"]["s16b"] is None  # s16b missing: the verdicts stand
    assert r["position_effect"]["s16b_minus_s16"][DN] == {"folds": [], "per_fold": {}, "mean": None, "all_folds": False}
    assert r["units"][f"{DN}_f0"]["s16b"] == {"inner_score": None, "E": None, "minus_s32": None}
    _checked(r, tmp_path / "none")
    # the largest count that cost is named; ties at -0.0050 with every fold lower stay inside
    r2 = _decide(s16={DN: DOWN4, CX: TIE_DOWN}, s8={DN: DOWN4, CX: DOWN4}, s4={DN: FLAT, CX: DOWN4})
    assert [r2["encoders"][DN][a]["verdict"] for a in ("s16", "s8", "s4")] == ["fewer slices cost"] * 2 + ["no difference shown"]
    assert r2["encoders"][CX]["s16"]["verdict"] == "no difference shown"
    assert r2["summary_words"] == ["Fewer slices cost from 16 down. 32 is the largest count tested, so the curve may still rise beyond it."]
    # four folds lower and one at 0.0000: lower in 4 of 5 folds
    four = [-0.010, -0.010, -0.010, -0.010, 0.0]
    r3 = _decide(s16={DN: four, CX: [-x for x in four]}, s8={}, s4={})
    assert (r3["encoders"][DN]["s16"]["verdict"], r3["encoders"][CX]["s16"]["verdict"]) == ("fewer slices cost", "fewer slices scored higher")
    assert "(DINOv2, lower in 4 of 5 folds)" in r3["encoders"][DN]["s16"]["words"]
    assert "(ConvNeXt-T, higher in 4 of 5 folds)" in r3["encoders"][CX]["s16"]["words"]
    # a "scored higher" without any cost: the open point only; "no count differed" would be false
    r4 = _decide(s16={CX: UP4}, s8={}, s4={})
    assert r4["summary_words"] == ["Open point: with 16 of the 32 cached slices per plane the head scored higher "
                                   "(ConvNeXt-T). The models were not changed."]
    _checked(r4, tmp_path / "open")


def test_decide_slices_gives_no_verdict_with_fewer_than_5_folds(tmp_path):
    from knee.nb import n13s

    down = {DN: DOWN4, CX: DOWN4}
    r = _decide(s16=_s("s16", down, stopped={(CX, 3), (CX, 4)}), s8=down, s4=None, s16b=_s("s16b", down, stopped={(DN, 0)}))
    assert r["status"] == "partial"
    c16 = r["encoders"][CX]["s16"]
    assert (c16["verdict"], c16["diff"], c16["folds_up"], c16["folds"]) == ("no verdict", None, None, [0, 1, 2])
    assert c16["reason"] == "fewer than 5 folds finished in the arm and in s32" and "diff_unrounded" not in c16
    assert c16["per_fold"] == {0: -0.008, 1: -0.007, 2: -0.006} and c16["lower"] == {"folds": [0, 1, 2]}
    assert (c16["words"], c16["words_extra"]) == (
        "16 slices, ConvNeXt-T: no verdict (fewer than 5 folds finished in the arm and in s32).", FIXED)
    assert r["encoders"][DN]["s16"]["verdict"] == "fewer slices cost"  # the other encoder is whole
    assert r["encoders"][CX]["s8"]["verdict"] == "fewer slices cost"  # and so is the other arm
    s4 = r["encoders"][DN]["s4"]
    assert (s4["verdict"], s4["reason"], s4["words"]) == ("no verdict", "the arm did not run", "4 slices, DINOv2: no verdict (the arm did not run).")
    assert r["arms"]["s4"] is None and r["units"][f"{CX}_f3"]["s16"] == {"inner_score": None, "E": None, "minus_s32": None}
    lines = n13s.slice_slide_sentences(r)
    assert lines[2] == f"4 slices, DINOv2: no verdict (the arm did not run). {FIXED} {ENDING}"
    assert lines[6:] == ["Fewer slices cost from 16 down. 32 is the largest count tested, so the curve may still rise beyond it."]
    pe = r["position_effect"]["s16b_minus_s16"]  # a partial s16b: its finished folds, and it changes no verdict
    assert pe[DN]["folds"] == [1, 2, 3, 4] and pe[DN]["all_folds"] is False and pe[CX]["folds"] == [0, 1, 2]
    assert r["mean_inner_score"][CX]["s16"] is None and r["mean_inner_score"][CX]["s8"] is not None
    _checked(r, tmp_path / "part")
    # without a cost and with a missing verdict there is no summary line: "no count differed" is not known
    rp = _decide(s16={}, s8={}, s4=None)
    assert rp["status"] == "partial" and rp["summary_words"] == []
    # a fold missing in the base takes the verdict from every arm for that encoder
    rb = _decide(s16=down, s8=down, s4=down, s32=_s("s32", stopped={(DN, 0)}))
    assert [rb["encoders"][DN][a]["verdict"] for a in ("s16", "s8", "s4")] == ["no verdict"] * 3
    assert rb["encoders"][DN]["s8"]["folds"] == [1, 2, 3, 4] and rb["encoders"][CX]["s8"]["verdict"] == "fewer slices cost"
    assert rb["rerun_check"]["units_checked"] == sorted(f"{e}_f{k}" for e in MAIN for k in range(5) if (e, k) != (DN, 0))
    # s32 did not run: no verdict at all
    r0 = _decide(s16=down, s8=down, s4=down, s32=None)
    assert r0["status"] == "no_base" and "units" not in r0 and r0["summary_words"] == []
    recs = [v[a] for v in r0["encoders"].values() for a in ("s16", "s8", "s4")]
    assert all((x["verdict"], x["diff"], x["reason"]) == ("no verdict", None, "the s32 arm (the base) did not run") for x in recs)
    assert n13s.slice_slide_sentences(r0)[0] == f"16 slices, DINOv2: no verdict (the s32 arm (the base) did not run). {FIXED} {ENDING}"
    assert len(n13s.slice_slide_sentences(r0)) == 6
    _checked(r0, tmp_path / "no_base")


def _all_none(r, status, reason):
    assert r["status"] == status and "units" not in r and "rerun_check" not in r and r["summary_words"] == []
    recs = [v[a] for v in r["encoders"].values() for a in ("s16", "s8", "s4")]
    assert len(recs) == 6
    assert all((x["verdict"], x["diff"], x["folds_up"], x["reason"]) == ("no verdict", None, None, reason) for x in recs)
    assert all(x["words"].endswith(f"no verdict ({reason}).") and x["words_extra"] == FIXED for x in recs)
    return r


def test_decide_slices_statuses_leakage_chosen_ref_commit_environment(tmp_path):
    from knee.nb import n13s

    down = {DN: DOWN4, CX: DOWN4}
    other = {**ENV, "kaggle_image_build_date": "20261124-000000", "torch": "2.12.0+cpu"}

    def edited(arm, **fields):
        return {**_s(arm, down if arm != "s32" else None), **fields}

    # the leakage flag of every arm summary given (s16b's and a rerun's too) must be true
    for bad in (False, None):
        r = _all_none(_decide(s16=edited("s16", leakage_assertion_passed=bad), s8=down, s4=down),
                      "leakage_check_failed", "an arm's leakage check did not pass")
        assert r["leakage_failed_steps"] == ["n13_s16"] and r["leakage_assertion_passed"] is False
    _checked(r, tmp_path / "leak")
    assert _decide(s16=down, s8=down, s4=down, s16b=edited("s16b", leakage_assertion_passed=False))["leakage_failed_steps"] == ["n13_s16b"]
    assert _decide(s16=down, s8=down, s4=down, s32=edited("s32", leakage_assertion_passed=False))["status"] == "leakage_check_failed"
    part = _s("s8", down, stopped={(DN, 2)})
    fill = {**_s("s8", down, only={(DN, 2)}, step="n13_s8_r"), "leakage_assertion_passed": False}
    assert _decide(s16=down, s8=[part, fill], s4=down)["leakage_failed_steps"] == ["n13_s8_r"]
    # chosen_ref_checked
    for bad in (False, None):
        r = _all_none(_decide(s16=down, s8=down, s4=edited("s4", chosen_ref_checked=bad)),
                      "chosen_ref_not_checked", "an arm did not check its configurations against 25 Nov")
        assert r["chosen_ref_unchecked_steps"] == ["n13_s4"] and r["chosen_ref_checked"] is False
    _checked(r, tmp_path / "ref")
    # one knee commit across the arm files (s16b included), and it is the declared one; bc's commit is not compared
    r = _all_none(_decide(s16=down, s8=_s("s8", down, commit="0ther"), s4=down), "commit_mismatch",
                  "the arms did not run at one knee commit")
    assert r["same_knee_commit"] is False and r["knee_commits"] == sorted(["0ther", n13s.SLICE_ARM_COMMIT])
    assert r["knee_commits_by_step"]["n13_s8"] == "0ther" and r["knee_commits_by_step"]["n13_s32"] == n13s.SLICE_ARM_COMMIT
    _checked(r, tmp_path / "commit")
    assert _decide(s16=down, s8=down, s4=down, s16b=_s("s16b", down, commit="0ther"))["status"] == "commit_mismatch"
    old = {a: _s(a, down if a != "s32" else None, commit=BC_COMMIT) for a in ARMS}
    r = _all_none(_decide(s16=old["s16"], s8=old["s8"], s4=old["s4"], s16b=old["s16b"], s32=old["s32"]),
                  "commit_mismatch", "the arms did not run at the declared knee commit")
    assert r["same_knee_commit"] is True and r["knee_commit_is_declared"] is False and r["arm_commit_declared"] == n13s.SLICE_ARM_COMMIT
    r = _decide(s16=old["s16"], s8=old["s8"], s4=old["s4"], s32=old["s32"], arm_commit=None)  # only that comparison is skipped
    assert r["status"] == "decided" and r["knee_commit_is_declared"] is None
    # the environment: all arms present must be equal in every field (s16b and reruns too)
    r = _all_none(_decide(s16=down, s8=_s("s8", down, env=other), s4=down), "env_mismatch", "the arms' environments differ")
    assert r["env_differences"] == ["kaggle_image_build_date", "torch"] and set(r["environments"]) == {"s32", "s16", "s8", "s4"}
    _checked(r, tmp_path / "env")
    assert _decide(s16=down, s8=down, s4=down, s16b=_s("s16b", down, env=other))["status"] == "env_mismatch"
    fill = _s("s8", down, only={(DN, 2)}, step="n13_s8_r", env=other)
    assert _decide(s16=down, s8=[part, fill], s4=down)["status"] == "env_mismatch"
    # all arms on one other image: decided; bc on another image is only reported
    ro = _decide(s16=_s("s16", down, env=other), s8=_s("s8", down, env=other), s4=_s("s4", down, env=other),
                 s32=_s("s32", env=other))
    assert ro["status"] == "decided" and ro["rerun_check"]["bc_env_differences"] == ["kaggle_image_build_date", "torch"]
    # the order: no base, leakage, chosen_ref, commit, environment
    worst = {**_s("s8", down, env=other, commit="0ther"), "leakage_assertion_passed": False, "chosen_ref_checked": False}
    assert _decide(s16=down, s8=worst, s4=down)["status"] == "leakage_check_failed"
    assert _decide(s16=down, s8={**worst, "leakage_assertion_passed": True}, s4=down)["status"] == "chosen_ref_not_checked"
    worst.update(leakage_assertion_passed=True, chosen_ref_checked=True)
    assert _decide(s16=down, s8=worst, s4=down)["status"] == "commit_mismatch"
    assert _decide(s16=down, s8=worst, s4=down, s32=None)["status"] == "no_base"
    # an arm rerun whole after an environment mismatch: the first run is reported, the rerun enters the rule
    first = _s("s8", {DN: UP4, CX: UP4}, env=other)
    rr = _decide(s16=down, s8=down, s4=down, replaced={"s8": first, "s4": None})
    assert rr["status"] == "decided" and rr["encoders"][DN]["s8"]["verdict"] == "fewer slices cost"
    assert list(rr["replaced_runs"]) == ["s8"] and rr["replaced_runs"]["s8"]["environments"] == [other]
    assert len(rr["replaced_runs"]["s8"]["inner_scores"]) == 10
    _checked(rr, tmp_path / "replaced")
    for bad in ({"b8": first}, {"s4": first}):
        with pytest.raises(ValueError):
            _decide(s16=down, s8=down, s4=down, replaced=bad)


def test_decide_slices_refuses_wrong_inputs_and_any_other_s():
    from knee.nb import n13s

    down = {DN: DOWN4, CX: DOWN4}
    s32, s16, s8, s4, bc = _s("s32"), _s("s16", down), _s("s8", down), _s("s4", down), _bc()
    base0, null = _it2("base0", BASE0), _it2("null", NULL)
    wrong_batch = _s("s8", down)
    wrong_batch["units"][f"{DN}_f1"]["config"]["batch_size"] = 8  # not its chosen 16
    wrong_slices = _s("s8", down)
    wrong_slices["units"][f"{DN}_f1"]["slices"] = {"config": n13s.slice_config("s4")}
    no_slices = _s("s8", down)
    no_slices["units"][f"{DN}_f1"].pop("slices")
    no_env = _s("s4", down, env=None)
    extra = _s("s4", down)
    extra["units"]["vit_s16_f0"] = dict(extra["units"][f"{DN}_f0"])
    nan = _s("s4", down)
    nan["units"][f"{DN}_f0"]["inner_score"] = None
    other_settings = _s("s4", down)
    other_settings["settings"] = {**other_settings["settings"], "patience": 3}
    wide = _it2("null", {e: [v + d for v, d in zip(vals, (0.02, -0.02, 0.01, -0.01, 0.0))] for e, vals in NULL.items()})
    twice = [_s("s8", down), _s("s8", down, only={(DN, 2)}, step="n13_s8_r")]
    for args, kw in (((s32, s8, s16, s4, bc, base0, null), {}),  # arms in the wrong places
                     ((s32, s16, s8, s4, bc, base0, null), {"s16b": s16}),  # s16 is not s16b: other positions
                     ((s32, s16, s8, s4, s32, base0, null), {}),  # bc must be arm bc's summary
                     ((s32, s16, wrong_batch, s4, bc, base0, null), {}), ((s32, s16, wrong_slices, s4, bc, base0, null), {}),
                     ((s32, s16, no_slices, s4, bc, base0, null), {}), ((s32, s16, s8, no_env, bc, base0, null), {}),
                     ((s32, s16, s8, extra, bc, base0, null), {}), ((s32, s16, s8, nan, bc, base0, null), {}),
                     ((s32, s16, s8, other_settings, bc, base0, null), {}),
                     ((s32, s16, s8, s4, bc, null, base0), {}),
                     ((s32, s16, s8, s4, bc, base0, wide), {}),  # not iteration 2's s (0.0030)
                     ((s32, s16, s8, s4, bc, base0, _it2("null", {e: v[:4] for e, v in NULL.items()})), {}),
                     ((s32, s16, twice, s4, bc, base0, null), {})):  # a second finished unit for one fold
        with pytest.raises(ValueError):
            n13s.decide_slices(*args, **kw)
    assert n13s.decide_slices(s32, s16, s8, s4, None, base0, null)["rerun_check"]["bc_given"] is False  # bc may be missing


def test_decide_slices_rerun_check_lists_units_above_the_tolerance(tmp_path):
    from knee.nb import n13s

    down = {DN: DOWN4, CX: SMALL_DOWN}
    # bc 0.0005 above s32 in one unit: the tie is inside
    tie = _decide(s16=down, s8=down, s4=down, bc=_bc({DN: [0.0005, 0, 0, 0, 0]}))
    rc = tie["rerun_check"]
    assert tie["units"][f"{DN}_f0"]["s32_minus_bc"] == -0.0005 and rc["units_outside"] == [] and rc["words"] is None
    assert rc["identical"] is False and tie["s32_reruns_bc"] is False and rc["max_abs_difference"] == 0.0005
    assert len(n13s.slice_slide_sentences(tie)) == 7
    out = _decide(s16=down, s8=down, s4=down, bc=_bc({DN: [0.0006, 0, 0, 0, 0], CX: [0, 0, -0.0020, 0, 0]}))
    rc = out["rerun_check"]
    assert rc["units_outside"] == [f"{DN}_f0", f"{CX}_f2"]  # listed; the verdict is still given
    assert rc["units_outside_difference"] == {f"{DN}_f0": -0.0006, f"{CX}_f2": 0.002}
    assert rc["words"] == ("Rerun noise: the 32-slice arm differed from the earlier run of the same fits (arm bc) by more "
                           "than 0.0005 in 2 of 10 units (largest difference 0.0020).")
    assert out["status"] == "decided" and out["encoders"] == tie["encoders"]  # bc never enters a verdict
    lines = n13s.slice_slide_sentences(out)
    assert len(lines) == 8 and lines[7] == rc["words"] and lines[6].startswith("Fewer slices cost from 16 down.")
    _checked(out, tmp_path)
    # a missing bc unit is not checked; no bc at all: nothing to compare
    part = _decide(s16=down, s8=down, s4=down, bc=_bc(stopped={(CX, 1)}))
    assert part["units"][f"{CX}_f1"]["s32_minus_bc"] is None and len(part["rerun_check"]["units_checked"]) == 9
    none = _decide(s16=down, s8=down, s4=down, bc=None)
    assert none["s32_reruns_bc"] is False and none["rerun_check"]["max_abs_difference"] is None
    assert none["rerun_check"]["per_unit"] == {} and none["encoders"] == tie["encoders"]
    _checked(none, tmp_path / "none")


# ---------------------------------------------------------------- the arm run path is SLICE_ARM_COMMIT's

DECIDE_NEW = {"SLICE_DECLARATION", "SLICE_ARM_COMMIT", "VERDICT_ARMS", "COST", "HIGHER", "NO_DIFF", "NO_VERDICT",
              "RERUN_TOL", "WORDS_ENDING", "WORDS_NOT_CHANGED", "WORDS_FIXED", "WORDS_NONE", "_slice_arm",
              "decide_slices", "slice_slide_sentences"}  # added after the arm commit


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


def test_everything_reachable_from_run_slices_is_textually_the_arm_commit():
    """The Kaggle sessions of the five arms install SLICE_ARM_COMMIT. The later commit may change the local decide
    step only: (1) every file under src/ and pyproject.toml is that commit's, except n13s.py, and no module was
    added; (2) in n13s.py every top-level statement of that commit is unchanged in text and order (the module
    docstring and the imports too) and the later statements only follow them; (3) nothing that run_slices
    reaches, by name, is a name added later."""
    from knee.nb import n13s

    arm = n13s.SLICE_ARM_COMMIT
    assert len(arm) == 40 and int(arm, 16) >= 0
    if _git("rev-parse", "--verify", f"{arm}^{{commit}}") is None:
        pytest.skip("git history with the arm commit is not available")
    f13s = "src/knee/nb/n13s.py"
    tracked = _git("ls-tree", "-r", "--name-only", arm, "--", "src", "pyproject.toml").decode().splitlines()
    assert {f13s, "src/knee/nb/n13.py", "src/knee/heads.py", "src/knee/nb/n6.py", "src/knee/agg.py", "pyproject.toml"} <= set(tracked)
    for f in tracked:
        if f != f13s:
            assert _git("show", f"{arm}:{f}") == (REPO / f).read_bytes(), f
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / "src").rglob("*.py")}
    assert on_disk <= set(tracked)  # no module was added
    old, new = _top(_git("show", f"{arm}:{f13s}").decode()), _top((REPO / f13s).read_text())
    assert [text for _, text in new[:len(old)]] == [text for _, text in old]  # same text, same order
    added = set().union(*(names for names, _ in new[len(old):]))
    assert added == DECIDE_NEW and not added & set().union(*(names for names, _ in old))  # nothing is redefined
    assert all(names for names, _ in new[len(old):])  # the later statements are definitions only: no import, no call
    by_name = {name: text for names, text in new for name in names}
    reach, todo = set(), ["run_slices"]
    while todo:  # every module-level name that run_slices can reach, by reference
        name = todo.pop()
        if name in reach:
            continue
        reach.add(name)
        todo += [x for x in _refs(by_name[name]) if x in by_name]
    assert {"run_slices", "_slice_task", "_slice_unit", "_arm_bank", "slice_bank", "slice_positions", "slice_config",
            "slice_settings", "slice_summary", "SLICE_ARMS", "SLICE_PLAN", "N_CACHED", "_S"} <= reach
    assert not reach & DECIDE_NEW
    old_text = [text for _, text in old]
    assert all(by_name[name] in old_text for name in reach)  # each is the text of the arm commit
    # the arm commit left n13.py, n6.py and heads.py as they were at the commit the branch started from
    if _git("rev-parse", "--verify", "b0da7b7^{commit}") is not None:
        for f in ("src/knee/nb/n13.py", "src/knee/nb/n6.py", "src/knee/heads.py", "src/knee/agg.py", "src/knee/stage.py"):
            assert _git("show", f"b0da7b7:{f}") == (REPO / f).read_bytes(), f
        diff = _git("diff", "--name-only", "b0da7b7", arm, "--", "src", "pyproject.toml").decode().split()
        assert diff == [f13s]  # the arm commit added one module and changed nothing else under src/


# ---------------------------------------------------------------- the notebooks

def test_the_five_notebooks_are_registered_as_cpu_notebooks():
    from knee.nb import n13s

    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import make_notebooks as mk
    finally:
        sys.path.remove(str(REPO / "scripts"))
    card = (REPO / "notebooks" / "RUNBOOK.md").read_text().split("## Slice-count check (it2c; added after the results)")[1]
    card = card.split("## If a run fails")[0]
    arm_commit = n13s.SLICE_ARM_COMMIT
    assert mk.SLICE_COMMIT == arm_commit and mk.SLICE_NOTEBOOKS == tuple(f"n13-{a}" for a in ARMS)
    for arm in ARMS:
        name = f"n13-{arm}"
        gpu, comp, tables, up, timeout = mk.NOTEBOOKS[name]
        assert (gpu, comp, tables) == (False, False, False) and up == mk.NOTEBOOKS["n13-bc"][3] == ["n0", "n5", "n3-r", "n6a"]
        assert mk.cell2(name) == f'from knee.nb import n13s; n13s.run_slices("{arm}", budget_h=4.0, expect={mk.EXPECT})'
        assert mk.cell2(name).split("expect=")[1] == mk.cell2("n13-bc").split("expect=")[1]  # bc's hashes
        meta = mk.metadata("someone", name)
        assert meta["enable_gpu"] is False and "machine_shape" not in meta and meta["enable_internet"] is True
        assert meta["kernel_sources"] == mk.metadata("someone", "n13-bc")["kernel_sources"]
        assert timeout >= 1.1 * 4.0 * 3600  # above the step's watchdog
        assert f"| knee-{name} | None (CPU) |" in card and f"`{mk.cell2(name)}`" in card
        assert mk.commit_for(name, "0123abc") == arm_commit  # cell 1 is pinned, whatever --commit says
        assert mk.notebook_json(mk.commit_for(name, "0123abc"), mk.cell2(name))["cells"][0]["source"][0] == f'COMMIT = "{arm_commit}"\n'
    assert mk.commit_for("n13-bc", "0123abc") == mk.BATCH_COMMIT and mk.commit_for("n13-base0", "0123abc") == "0123abc"
    assert f'`COMMIT = "{arm_commit}"`' in card and "it2c_slice_count_r2.md" in card
    assert "fork of knee-n13-bc" in card and "original environment" in card and "replace cell 2" in card
    assert "A worker crash is not retried" in card and "n13s.decide_slices(s32, s16, s8, s4, bc, base0, null, s16b=s16b)" in card
    for status in ("no_base", "leakage_check_failed", "chosen_ref_not_checked", "commit_mismatch", "env_mismatch"):
        assert f'"{status}"' in card
