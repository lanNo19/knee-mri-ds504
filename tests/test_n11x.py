"""N11x, the learning-rate extension of iteration 3 (it3_lr_extension_r3.md), on synthetic data (CPU, the fake
encoder of tests/test_n11.py). Tested: the loop's files against commit 095c91f; a run at 8e-5 through the new
entry against iteration 3's `_inner`, bit for bit; the two processes' folders; the start check (pass, one
failed fold, three failed folds); the replay and the one stop flag; invalid runs and the swap; the soft
stop; Rule S and every line of Rule D on hand-made scores; the selection record; the smoke run; the
privacy checks of every written aggregate; iteration 3's find_units; the notebook entries and their card."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn.metrics")

from tests.test_n11 import MAIN, PASSES, TABLE, _make_inputs, fake_factory  # noqa: E402
from tests.test_n11 import _shared as _shared11  # noqa: E402

DINO = MAIN[0]
ROOT = Path(__file__).resolve().parents[1]
BASE = "095c91f"  # the commit of iteration 3's sessions
END = {"8e-05": {f"fold_{k}": v for k, v in enumerate((0.60, 0.61, 0.62, 0.63, 0.64))},
       "2e-05": {f"fold_{k}": v for k, v in enumerate((0.55, 0.56, 0.57, 0.58, 0.59))}}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    inp = tmp_path_factory.mktemp("n11x") / "input"
    _make_inputs(inp)
    return inp


@pytest.fixture(scope="module")
def refs(inputs):
    """The fixture's own reference numbers, in EXT_REF's form: each fold's start score and fold 0's 8e-5 run
    to the quarter mark, computed with the module's functions (what iteration 3 would have logged)."""
    from knee.nb import n11

    sh = _shared11(inputs, inputs.parent / "wref")
    start = {}
    for k in range(5):
        c = n11._context(sh, k)
        st = {"ext": {}}
        head0 = n11._ext_start(c, st, None)
        start[f"fold_{k}"] = st["ext"]["start"]["val_macro_auc"]
        if k == 0:
            r = n11.new_run(c, c.src, head0, c.itr, 8e-5, 0, val_rows=c.iva)
            n11.run_to(c, r, n11.RUNG0)
            replay = {"fold": 0, "lr": 8e-5, "val_macro_auc": [n11._r(h["val_macro_auc"]) for h in r.history],
                      "train_loss": [n11._r(h["train_loss"]) for h in r.history]}
    return {"source": "the fixture", "start_val_macro_auc": start, "replay": replay, "end_val_macro_auc": copy.deepcopy(END)}


def _lrx(inputs, work, monkeypatch, **kw):
    from knee import paths
    from knee.nb import n11

    monkeypatch.setattr(paths, "WORKING", Path(work))
    args = {"root": inputs, "val_table": TABLE, "val_range": None, "passes": PASSES, "encoder_factory": fake_factory,
            "quiet": True, "budget_h": 1.0, "devices": ["cpu"]}
    return n11.run_lrx(**{**args, **kw})


@pytest.fixture(scope="module")
def session(inputs, refs):
    """The inner session as declared: both new rates, five folds, one device (the replay process goes first)."""
    work = inputs.parent / "wx"
    with pytest.MonkeyPatch.context() as mp:
        s = _lrx(inputs, work, mp, session="inner", reference=refs)
    return s, work


def _states(work, step):
    return {int(p.parent.name.rsplit("_f", 1)[1]): json.loads(p.read_text())
            for p in sorted((Path(work) / "private" / step).glob("*_f*/state.json"))}


# ---------------------------------------------------------------- the loop is iteration 3's

def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def test_loop_files_unchanged_since_iteration3():
    """The files the loop uses are byte-equal to 095c91f, and n11.py has no removed or changed line above
    find_units (additions only). The loop itself (the schedule to the fit summary) is byte-equal."""
    import difflib

    if _git("cat-file", "-e", f"{BASE}^{{commit}}").returncode != 0:
        pytest.skip("commit 095c91f is not in this checkout")
    for f in ("heads.py", "encoders.py", "metrics.py", "stage.py", "units.py", "nb/n5.py"):
        old = _git("show", f"{BASE}:src/knee/{f}")
        assert old.returncode == 0 and old.stdout == (ROOT / "src" / "knee" / f).read_text(), f
    old = _git("show", f"{BASE}:src/knee/nb/n11.py").stdout
    new = (ROOT / "src" / "knee" / "nb" / "n11.py").read_text()
    mark = "\ndef find_units("
    a, b = old.split(mark)[0].splitlines(), new.split(mark)[0].splitlines()
    ops = {tag for tag, *_ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()}
    assert ops <= {"equal", "insert"}, ops

    def loop(text):  # from the schedule to the end of the fit summary: new_run, run_to, _validate, checkpoints
        return text[text.index("# ---------------------------------------------------------------- the schedule, in twelfths"):
                    text.index("# ---------------------------------------------------------------- run.json, files")]

    assert loop(old) == loop(new)
    for name in ("_inner", "_frozen0", "_final", "_task", "inner_refit", "_context", "_shared", "_abort_if_invalid"):
        def body(text, name=name):
            i = text.index(f"\ndef {name}(")
            return text[i:text.index("\n\n\n", i)]

        assert body(old) == body(new), name


def test_a_run_at_8e5_reproduces_iteration3_inner_bit_for_bit(inputs, refs, tmp_path, monkeypatch):
    """Iteration 3's `_inner` (8e-5 as the survivor: rung 0, checkpoint, then to the end) against the same
    unit through run_lrx at 8e-5: every logged number, the validation predictions and the trained weights
    are equal, bit for bit, on the CPU."""
    from knee.nb import n11

    seen = []
    real = n11.run_to

    def spy(c, r, twelfth, limit=None):
        real(c, r, twelfth, limit)
        if r.with_encoder and r.k in (n11.RUNG0, n11.TWELFTHS) and r.step == r.marks[r.k - 1]:
            seen.append({"lr": r.lr_enc, "k": r.k, "history": copy.deepcopy(r.history), "pred": r.val_pred.copy(),
                         "tail": {n: v.clone() for n, v in c.enc.tail_state().items()},
                         "head": {n: v.detach().clone() for n, v in r.head.state_dict().items()},
                         "total": r.total, "bs": r.bs, "peaks": list(r.peaks)})

    monkeypatch.setattr(n11, "run_to", spy)
    monkeypatch.setattr(n11, "select_rung0", lambda scores, tie=True: {"continue": [8e-5], "tied": False, "note": "test"})
    c = n11._context(_shared11(inputs, tmp_path / "w3"), 0)
    st = n11._base_block(c)
    n11._inner(c, st, n11._frozen0(c, st))
    old = {x["k"]: x for x in seen if x["lr"] == 8e-5}
    assert set(old) == {n11.RUNG0, n11.TWELFTHS}
    seen.clear()
    s = _lrx(inputs, tmp_path / "wx", monkeypatch, session="rep", rates=(8e-5,), folds=(0,), replay=False, reference=refs)
    assert s["status"] == "ok", s["errors"]
    new = {x["k"]: x for x in seen}
    for k in (n11.RUNG0, n11.TWELFTHS):
        assert new[k]["history"] == old[k]["history"]  # raw floats: training loss, validation loss, AUC, mean |z|
        assert np.array_equal(new[k]["pred"], old[k]["pred"])
        assert all(torch.equal(new[k]["tail"][n], old[k]["tail"][n]) for n in old[k]["tail"])
        assert all(torch.equal(new[k]["head"][n], old[k]["head"][n]) for n in old[k]["head"])
        assert (new[k]["total"], new[k]["bs"], new[k]["peaks"]) == (old[k]["total"], old[k]["bs"], old[k]["peaks"])
    run = _states(tmp_path / "wx", "n11x_rep_8e-05")[0]["ext"]
    it3 = st["inner"]
    assert run["run"]["curve"] == it3["to_the_end"]["8e-05"]["curve"]  # the same curve keys, the same numbers
    assert run["run"]["val_macro_auc"] == it3["to_the_end"]["8e-05"]["val_macro_auc"]
    assert run["run"]["quarter_val_macro_auc"] == it3["rung0"]["8e-05"]["val_macro_auc"]
    assert run["run"]["config"] == it3["to_the_end"]["8e-05"]["config"]
    assert run["start"]["val_macro_auc"] == it3["start"]["val_macro_auc"] and run["start"]["passed"] is True
    import pandas as pd

    a = pd.read_csv(tmp_path / "wx" / "private" / "n11x_rep_8e-05" / f"{DINO}_f0" / "inner_start.csv")
    b = pd.read_csv(c.dir / "inner_start.csv")
    assert a.equals(b)  # the starting head's validation predictions, study for study


# ---------------------------------------------------------------- the session

def test_session_runs_checks_and_files(session, refs, inputs):
    from knee import agg, stage
    from knee.nb import n11

    s, work = session
    assert s["status"] == "ok" and s["errors"] == {} and s["runs_missing"] == [] and s["out_of_time"] is False
    assert s["processes"] == ["3.2e-4 on cpu: folds 0, 1, 2, 3, 4", "1.28e-3 on cpu: folds 0, 1, 2, 3, 4; the replay first"]
    assert s["test_overrides"] is True and s["added_after_results"] is True and s["stop_flag"] is None
    d = stage.StageData(inputs)
    for key in ("0.00032", "0.00128"):
        sts = _states(work, f"n11x_inner_{key}")
        assert sorted(sts) == [0, 1, 2, 3, 4]
        for k, b in sts.items():
            e, ud = b["ext"], work / "private" / f"n11x_inner_{key}" / f"{DINO}_f{k}"
            assert e["lr_key"] == key and e["start"]["passed"] and e["start"]["abs_difference"] == 0.0
            assert e["start"]["reference"] == refs["start_val_macro_auc"][f"fold_{k}"] and e["tail_rms_before_training"] > 0
            run = e["run"]
            assert run["status"] == "finished" and run["valid"] is True and run["invalid"] is None and run["non_finite"] is False
            assert set(run["curve"]) == {"train_loss", "val_loss", "val_macro_auc", "mean_abs_z"}  # iteration 3's keys
            assert all(len(v) == 12 for v in run["curve"].values()) and run["val_macro_auc"] == run["curve"]["val_macro_auc"][-1]
            assert run["quarter_val_macro_auc"] == run["curve"]["val_macro_auc"][2]
            assert run["best"]["val_macro_auc"] == max(run["curve"]["val_macro_auc"])
            assert run["curve"]["val_macro_auc"][run["best"]["k"] - 1] == run["best"]["val_macro_auc"]
            pt = run["per_twelfth"]
            assert all(len(pt[x]) == 12 for x in ("skipped_step_share", "non_finite", "train_seconds"))
            assert pt["train_seconds"] == sorted(pt["train_seconds"]) and run["skipped_step_share"] == 0.0
            assert run["encoder_lr"] == float(key) and run["head_lr"] == 3e-3 and run["config"]["seed"] == 0
            # private per-study files: inner validation only, the fold's scored validation studies
            names = sorted(p.name for p in ud.iterdir())
            want = ["inner_start.csv", f"inner_b_lite_{key}.csv", "state.json"]
            if key == "0.00128" and k == 0:
                want.append("inner_replay_8e-05_quarter.csv")
            assert names == sorted(want) and set(b["files_sha16"]) == set(want) - {"state.json"}
            import pandas as pd

            iva = n11.inner_split(d, k, TABLE, None)["iva"]
            df = pd.read_csv(ud / f"inner_b_lite_{key}.csv").astype({"StudyInstanceUID": str})
            assert df["StudyInstanceUID"].tolist() == d.uids[iva].tolist() and not set(df["StudyInstanceUID"]) & set(d.uids[d.split(k)[1]])
    assert not list(work.rglob("pred_*.csv")) and not list(work.rglob("*.pt"))  # no test-fold file, no weights
    rp = s["replay"]
    assert rp["passed"] is True and rp["max_abs_gap"] == 0.0 and rp["val_macro_auc"] == refs["replay"]["val_macro_auc"]
    assert rp["train_loss_abs_gaps"] == [0.0, 0.0, 0.0] and rp["stop_rule_applied"] is True and rp["above_half_the_band"] is False
    assert "replay" in _states(work, "n11x_inner_0.00128")[0] and "replay" not in _states(work, "n11x_inner_0.00032")[0]
    # the aggregates: one per rate, the session's, run.json; all pass the privacy checks and hold no UID
    files = sorted(p.name for p in work.glob("*.json"))
    assert files == ["agg_n11x_inner.json", "agg_n11x_inner_0.00032.json", "agg_n11x_inner_0.00128.json", "run.json"]
    uids = set(d.uids.tolist())
    for p in [*work.glob("*.json"), *work.rglob("state.json")]:
        j = json.loads(p.read_text())
        agg._check(j)
        assert not any(u in p.read_text() for u in list(uids)[:50]) and "write_error" not in j
    one = json.loads((work / "agg_n11x_inner_0.00032.json").read_text())
    assert one["lr"] == 3.2e-4 and one["folds"] == [0, 1, 2, 3, 4] and set(one["units"]) == {f"{DINO}_f{k}" for k in range(5)}
    assert one["overview"]["fold_3"]["status"] == "finished" and one["start_check"]["fold_3"]["passed"] is True
    assert json.loads((work / "run.json").read_text())["config"]["test_overrides"] is True
    assert n11.find_units(work) == {}  # iteration 3's reader does not see the extension's folders
    assert sorted(n11.find_ext_units(work)) == [(key, k) for key in ("0.00032", "0.00128") for k in range(5)]


def test_no_test_row_is_read_and_the_starting_head_is_the_inner_refit(inputs, refs, tmp_path, monkeypatch):
    from knee import heads, stage
    from knee.nb import n11

    d = stage.StageData(inputs)
    te, rows, fits = set(d.split(1)[1].tolist()), [], []
    for cls in (n11.EncSource, n11.FeatSource):
        real = cls.__call__

        def call(self, r, grad=False, flipped=False, real=real):
            rows.append(np.asarray(r).copy())
            return real(self, r, grad=grad, flipped=flipped)

        monkeypatch.setattr(cls, "__call__", call)
    real_fit = heads.fit

    def spy_fit(bank, cfg, train_idx, *a, **k):
        fits.append((set(np.asarray(train_idx).tolist()), set(np.asarray(k["forbidden"]).tolist()), k.get("seed")))
        return real_fit(bank, cfg, train_idx, *a, **k)

    monkeypatch.setattr(heads, "fit", spy_fit)
    for name in ("predict_test", "_frozen0", "_inner", "_final", "record_chosen"):
        monkeypatch.setattr(n11, name, lambda *a, **k: pytest.fail("iteration 3's blocks and the test pass are not part of the extension"))
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", rates=(3.2e-4,), folds=(1,), replay=False, reference=refs)
    assert s["status"] == "ok" and rows
    sp = n11.inner_split(d, 1, TABLE, None)
    assert not te & set(np.concatenate(rows).tolist())
    assert set(np.concatenate(rows).tolist()) <= set(sp["itr"].tolist()) | set(sp["iva"].tolist())
    assert len(fits) == 1 and fits[0][0] == set(sp["itr"].tolist()) and fits[0][2] == 0
    assert set(sp["iva_all"].tolist()) | te <= fits[0][1]
    with pytest.raises(ValueError):
        _lrx(inputs, tmp_path / "w", monkeypatch, encoders=(MAIN[1],))
    with pytest.raises(TypeError):  # budget_h has no default
        n11.run_lrx("inner", root=inputs)
    # two processes need two GPUs: refused before the cache copy (devices is the tests' argument)
    monkeypatch.setattr(n11, "localise_caches", lambda *a, **k: pytest.fail("refused before the cache copy"))
    for kw in ({}, {"rates": (1.6e-4,), "split_folds": True, "replay": False}):
        with pytest.raises(Exception) as ei:
            _lrx(inputs, tmp_path / "w9", monkeypatch, session="inner", reference=refs, devices=None, **kw)
        assert "AssertionError" in str(ei.value) or isinstance(ei.value, AssertionError)
    assert not list((tmp_path / "w9").glob("agg_*.json"))


def test_two_processes_do_not_collide(inputs, refs, tmp_path, monkeypatch):
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", folds=(0, 1), processes=True, devices=["cpu", "cpu"],
             reference=refs)
    assert s["status"] == "ok", s["errors"]
    steps = sorted(p.name for p in (tmp_path / "w" / "private").iterdir() if p.is_dir())
    assert steps == ["n11x_inner_0.00032", "n11x_inner_0.00128"]
    for key in ("0.00032", "0.00128"):
        sts = _states(tmp_path / "w", f"n11x_inner_{key}")
        assert sorted(sts) == [0, 1] and all(b["ext"]["lr_key"] == key and b["ext"]["run"]["status"] == "finished" for b in sts.values())
    assert s["replay"]["passed"] is True
    assert sorted(p.name for p in (tmp_path / "w").glob("agg_*.json")) == [
        "agg_n11x_inner.json", "agg_n11x_inner_0.00032.json", "agg_n11x_inner_0.00128.json"]


# ---------------------------------------------------------------- start check, replay, the stop flag

def test_start_check_one_failed_fold_goes_on_three_stop_the_session(inputs, refs, tmp_path, monkeypatch, capfd):
    from knee.nb import n11

    bad = copy.deepcopy(refs)
    bad["start_val_macro_auc"]["fold_1"] += 0.0006  # outside 0.0005
    bad["start_val_macro_auc"]["fold_2"] += 0.0005  # at the tolerance: passes
    s = _lrx(inputs, tmp_path / "w1", monkeypatch, session="inner", rates=(3.2e-4,), folds=(0, 1, 2), reference=bad)
    assert s["status"] == "ok" and s["stop_flag"] is None
    chk = s["start_check"]["0.00032"]
    assert [chk[f"fold_{k}"]["passed"] for k in range(3)] == [True, False, True]
    assert chk["fold_1"]["abs_difference"] == 0.0006 and chk["fold_2"]["abs_difference"] == 0.0005
    assert all(s["overview"]["0.00032"][f"fold_{k}"]["status"] == "finished" for k in range(3))  # reported and marked
    # three failed folds: the one stop flag, nothing is trained, the notebook still ends OK with its files
    for k in (0, 1, 3):
        bad["start_val_macro_auc"][f"fold_{k}"] = refs["start_val_macro_auc"][f"fold_{k}"] + 0.01
    monkeypatch.setattr(n11, "new_run", lambda *a, **k: pytest.fail("nothing may train after three failed start checks"))
    s = _lrx(inputs, tmp_path / "w2", monkeypatch, session="inner", reference=bad)
    assert "STEP n11x_inner: OK" in capfd.readouterr().out
    assert s["status"] == "stopped: the start check failed in three or more folds"
    flag = json.loads((tmp_path / "w2" / "private" / "n11x_STOP").read_text())
    assert flag["reason"] == "start_check" and flag["folds"] == [0, 1, 3] and s["stop_flag"]["reason"] == "start_check"
    a = json.loads((tmp_path / "w2" / "agg_n11x_inner.json").read_text())
    assert all(v["status"] == "no run" for r in a["overview"].values() for v in r.values()) and a["replay"] is None
    assert [a["start_check"]["0.00128"][f"fold_{k}"]["passed"] for k in range(5)] == [False, False, True, False, True]
    sel = n11.selection_ext(n11.find_ext_units(tmp_path / "w2"), None, flag, reference=bad, smoke=True, n_boot=0)
    assert sel["status"] == "stopped_at_start_check" and set(sel["picks"].values()) == {8e-5} and not sel["refinement"]["may_run"]


def test_failed_replay_writes_the_one_flag_and_stops_the_other_process(inputs, refs, tmp_path, monkeypatch):
    from knee.nb import n11

    bad = copy.deepcopy(refs)
    bad["replay"]["val_macro_auc"][1] += 0.0051
    trained = []
    real = n11._inner_ext
    monkeypatch.setattr(n11, "_inner_ext", lambda c, *a, **k: (trained.append(c.step), real(c, *a, **k))[1])
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", folds=(0, 1, 2), reference=bad)
    assert s["status"] == "inconclusive: reused and new scores cannot be compared at 0.005"
    assert s["replay"]["passed"] is False and s["replay"]["max_abs_gap"] == 0.0051 and trained == []
    flag = json.loads((tmp_path / "w" / "private" / "n11x_STOP").read_text())  # one path, outside the step folders
    assert flag["reason"] == "replay" and flag["max_abs_gap"] == 0.0051
    assert not list((tmp_path / "w" / "private").glob("n11x_inner_*/INVALID"))
    ok = copy.deepcopy(refs)
    ok["replay"]["val_macro_auc"][1] += 0.005  # 0.0050 is not above 0.005
    s = _lrx(inputs, tmp_path / "w5", monkeypatch, session="inner", rates=(3.2e-4,), folds=(0,), reference=ok)
    assert s["status"] == "ok" and s["replay"]["passed"] is True and s["replay"]["above_half_the_band"] is True


def test_a_flag_written_by_another_step_stops_a_run_at_its_next_twelfth(inputs, refs, tmp_path, monkeypatch):
    """The flag is written while the 3.2e-4 process is inside twelfth 2 of fold 0, as the other rate's process
    would write it after a failed replay: the run stops with two twelfths, and no other run starts."""
    from knee.nb import n11

    real = n11.run_to

    def run_to(c, r, twelfth, limit=None):
        if c.fold == 0 and twelfth == 2 and r.with_encoder:
            n11._ext_stop(c.working, "inner", "replay", {"lr_key": "0.00128", "fold": 0, "max_abs_gap": 0.02})
        real(c, r, twelfth, limit)

    monkeypatch.setattr(n11, "run_to", run_to)
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", rates=(3.2e-4,), folds=(0, 1), replay=False, reference=refs)
    assert s["status"].startswith("inconclusive") and s["stop_flag"]["lr_key"] == "0.00128"
    sts = _states(tmp_path / "w", "n11x_inner_0.00032")
    run = sts[0]["ext"]["run"]
    assert run["status"] == "unfinished: stop flag" and run["finished"] is False and run["valid"] is False
    assert len(run["curve"]["val_macro_auc"]) == 2 and run["val_macro_auc"] is None and "run" not in sts[1]["ext"]
    assert s["runs_missing"] == [f"{DINO}_f0@0.00032", f"{DINO}_f1@0.00032"]
    assert not (tmp_path / "w" / "private" / "n11x_inner_0.00032" / "INVALID").exists()


# ---------------------------------------------------------------- invalid runs, the swap, the soft stop

def test_invalid_runs_stop_at_their_twelfth_and_cost_only_their_fold(inputs, refs, tmp_path, monkeypatch):
    from knee.nb import n11

    real_val, real_run = n11._validate, n11.run_to

    def validate(c, r):
        out = real_val(c, r)
        if c.fold == 0 and r.k == 3:  # r.k counts the finished twelfths: this is the fourth validation
            out["val_loss"] = float("nan")
        if c.fold == 3 and r.k == 11:
            out["val_macro_auc"] = None
        return out

    def run_to(c, r, twelfth, limit=None):
        real_run(c, r, twelfth, limit)
        if c.fold == 1 and twelfth == 5:
            r.skipped = int(0.05 * r.step) + 1  # just above the limit

    monkeypatch.setattr(n11, "_validate", validate)
    monkeypatch.setattr(n11, "run_to", run_to)
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", rates=(3.2e-4,), folds=(0, 1, 2, 3, 4), replay=False,
             reference=refs, force_invalid=(4, 2))
    assert s["status"] == "ok" and s["errors"] == {} and s["runs_missing"] == []  # every run reached an end: reported
    ov = s["overview"]["0.00032"]
    assert ov["fold_0"]["invalid"] == {"k": 4, "reason": "non-finite val loss"} and ov["fold_0"]["non_finite"] is True
    assert ov["fold_1"]["invalid"] == {"k": 5, "reason": "skipped optimiser steps above the limit"}
    assert ov["fold_3"]["invalid"] == {"k": 12, "reason": "the end score is not finite"}
    assert ov["fold_4"]["invalid"] == {"k": 2, "reason": "forced by the smoke run: no real fault"}
    assert ov["fold_2"]["status"] == "finished" and ov["fold_2"]["valid"] is True
    sts = _states(tmp_path / "w", "n11x_inner_0.00032")
    for k, n in ((0, 4), (1, 5), (3, 12), (4, 2)):
        run = sts[k]["ext"]["run"]
        assert run["status"] == "invalid" and run["valid"] is False and len(run["curve"]["train_loss"]) == n
        assert run["val_macro_auc"] is None and len(run["per_twelfth"]["non_finite"]) == n
    assert sts[0]["ext"]["run"]["per_twelfth"]["non_finite"] == [False, False, False, True]
    assert sts[1]["ext"]["run"]["per_twelfth"]["skipped_step_share"][-1] > 0.05
    assert "validation_predictions" not in sts[0]["ext"]["run"] and sts[2]["ext"]["run"]["validation_predictions"]
    sel = n11.selection_ext(n11.find_ext_units(tmp_path / "w"), None, None, reference=refs, smoke=True, n_boot=0)
    assert {k: v["runs"]["0.00032"]["eligible"] for k, v in sel["folds"].items()} == {
        "fold_0": False, "fold_1": False, "fold_2": True, "fold_3": False, "fold_4": False}
    assert sel["rule_d"]["line"] == {"k": 1} and "invalid in folds 0, 1, 3, 4" in sel["rule_d"]["words"]
    assert sel["graphs"]["g1"]["invalid"] == [f"0.00032 fold_{k}" for k in (0, 1, 3, 4)] and sel["missing"] == []


def test_early_stop_rule_swaps_the_upper_rate(inputs, refs, tmp_path, monkeypatch):
    from knee.nb import n11

    real = n11._validate

    def validate(c, r):
        out = real(c, r)
        if r.with_encoder and r.lr_enc == 1.28e-3 and c.fold in (0, 1) and r.k == 1:
            out["val_loss"] = float("inf")
        return out

    monkeypatch.setattr(n11, "_validate", validate)
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", folds=(0, 1, 2, 3), reference=refs)
    assert s["status"] == "ok" and s["swapped"] == [{"from": 1.28e-3, "to": 6.4e-4}] and s["runs_missing"] == []
    ov = s["overview"]
    assert sorted(ov) == ["0.00032", "0.00064", "0.00128"]
    assert [ov["0.00128"][f"fold_{k}"]["status"] for k in range(4)] == [
        "invalid", "invalid"] + ["not run: the rate was stopped by the early-stop rule"] * 2
    assert all(v["status"] == "finished" for v in ov["0.00064"].values()) and all(v["status"] == "finished" for v in ov["0.00032"].values())
    sw = _states(tmp_path / "w", "n11x_inner_0.00064")
    assert sw[2]["ext"]["swapped_from"] == 1.28e-3 and sw[2]["ext"]["start"]["copied_from_step"] == "n11x_inner_0.00128"
    assert (tmp_path / "w" / "agg_n11x_inner_0.00064.json").exists()
    sel = n11.selection_ext(n11.find_ext_units(tmp_path / "w"), None, None, reference=refs, smoke=True, n_boot=0)
    assert sel["rule_d"]["upper"] == 6.4e-4 and sel["rule_d"]["words_swap"] == (
        "1.28e-3 was unstable in folds 0 and 1 and was not run in the other three")
    assert "unstable" not in sel["rule_d"]["words"]
    # the swap followed by a soft stop and an invalid fold of 6.4e-4: every string of the record stays writable
    from knee import agg

    for k, status in ((2, "invalid"), (3, "unfinished: out of time")):
        p = tmp_path / "w" / "private" / "n11x_inner_0.00064" / f"{DINO}_f{k}" / "state.json"
        b = json.loads(p.read_text())
        b["ext"]["run"].update({"status": status, "finished": False, "valid": False, "val_macro_auc": None})
        p.write_text(json.dumps(b))
    sel = n11.selection_ext(n11.find_ext_units(tmp_path / "w"), None, None, reference=refs, smoke=True, n_boot=0)
    agg._check(sel)
    rd = sel["rule_d"]
    assert rd["c2"] == "UNSTABLE" and rd["reasons"]["0.00064"] == ["invalid in fold 2", "not finished in fold 3", "no unit for fold 4"]
    assert rd["line"] == {"k": 1} and rd["reasons"]["0.00032"] == ["no unit for fold 4"]  # the session had four folds
    assert _longest(sel) <= 200
    # late failures do not swap: fold 1 invalid after the quarter mark

    def late(c, r):
        out = real(c, r)
        if r.with_encoder and r.lr_enc == 1.28e-3 and ((c.fold == 0 and r.k == 0) or (c.fold == 1 and r.k == 3)):
            out["val_loss"] = float("inf")
        return out

    monkeypatch.setattr(n11, "_validate", late)
    s = _lrx(inputs, tmp_path / "w2", monkeypatch, session="inner", rates=(1.28e-3,), folds=(0, 1, 2), replay=False, reference=refs)
    assert s["swapped"] == [] and [s["overview"]["0.00128"][f"fold_{k}"]["status"] for k in range(3)] == ["invalid", "invalid", "finished"]


def test_soft_stop_ends_partial_with_everything_written(inputs, refs, tmp_path, monkeypatch, capfd):
    from knee.nb import n11

    real = n11.run_to

    def run_to(c, r, twelfth, limit=None):
        if c.fold == 1 and twelfth == 4 and r.with_encoder:
            c.task["soft_stop"] = 1.0  # the budget is used up while twelfth 4 trains
        real(c, r, twelfth, limit)

    monkeypatch.setattr(n11, "run_to", run_to)
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", rates=(3.2e-4,), folds=(0, 1, 2), replay=False, reference=refs)
    assert "STEP n11x_inner: OK" in capfd.readouterr().out
    assert s["status"] == "partial" and s["out_of_time"] is True and s["stop_flag"] is None
    assert s["runs_missing"] == [f"{DINO}_f1@0.00032", f"{DINO}_f2@0.00032"]
    assert s["errors"][f"{DINO}_f1@0.00032"].startswith("OutOfTime at")
    sts = _states(tmp_path / "w", "n11x_inner_0.00032")
    assert sts[0]["ext"]["run"]["status"] == "finished"
    run = sts[1]["ext"]["run"]  # the twelfths it has
    assert run["status"] == "unfinished: out of time" and len(run["curve"]["val_macro_auc"]) == 4 and run["val_macro_auc"] is None
    assert "run" not in sts[2]["ext"] and sts[2]["ext"]["start"]["passed"] is True
    a = json.loads((tmp_path / "w" / "agg_n11x_inner_0.00032.json").read_text())
    assert a["status"] == "partial" and a["overview"]["fold_1"]["status"] == "unfinished: out of time"
    # the selection accepts a partial session and says what is missing; a finished fold can still pick
    sel = n11.selection_ext(n11.find_ext_units(tmp_path / "w"), None, None, reference=refs, smoke=True, n_boot=0)
    assert sel["missing"] == ["0.00032 fold_1: unfinished: out of time", "0.00032 fold_2: no run",
                              "0.00032 fold_3: no unit", "0.00032 fold_4: no unit"]
    assert sel["folds"]["fold_0"]["runs"]["0.00032"]["eligible"] is True and sel["rule_d"]["c1"] == "NONE"


# ---------------------------------------------------------------- Rule S and Rule D on hand-made scores

def test_rule_s_per_fold():
    from knee.nb import n11

    S = n11.ext_rule_s
    lo, mid, up = 8e-5, 3.2e-4, 1.28e-3
    yes = {mid: True, up: True}
    assert S({lo: 0.7000, mid: 0.7051, up: 0.7000}, yes)["pick"] == mid  # more than 0.005
    assert S({lo: 0.7000, mid: 0.7050, up: 0.7050}, yes) == {"pick": lo, "moved": False, "steps": [
        {"rate": mid, "against": lo, "difference": 0.005, "more_than_the_band": False},
        {"rate": up, "against": lo, "difference": 0.005, "more_than_the_band": False}]}  # 0.0050 is not more than 0.005
    assert S({lo: 0.7000, mid: 0.7051, up: 0.7101}, yes)["pick"] == mid  # inside the band of the pick: the lower rate stays
    assert S({lo: 0.7000, mid: 0.7051, up: 0.7102}, yes)["pick"] == up
    assert S({lo: 0.7000, mid: 0.7040, up: 0.7051}, yes)["pick"] == up  # the upper rate against 8e-5
    assert S({lo: 0.7000, mid: 0.7000, up: 0.6000}, yes)["pick"] == lo
    assert S({lo: 0.7000, mid: 0.9000, up: 0.7060}, {mid: False, up: True})["pick"] == up  # an ineligible rate is skipped
    assert S({lo: 0.7000, mid: 0.9000, up: 0.9000}, {})["moved"] is False
    assert S({lo: 0.7000, mid: None, up: 0.7100}, yes)["pick"] == up and S({lo: None, mid: 0.9}, yes)["pick"] == lo
    assert S({lo: 0.7, mid: 0.8, 6.4e-4: 0.81, up: 0.9}, {mid: True, 6.4e-4: True, up: True})["pick"] == up
    assert S({2e-5: 0.99, lo: 0.7, mid: 0.7}, yes)["pick"] == lo  # a lower rate is never a candidate


def _longest(o) -> int:
    if isinstance(o, dict):
        return max([_longest(v) for v in o.values()] + [len(str(k)) for k in o] + [0])
    if isinstance(o, list):
        return max([_longest(v) for v in o] + [0])
    return len(o) if isinstance(o, str) else 0


def _five(base, deltas):
    return {k: round(base[k] + d, 4) for k, d in enumerate(deltas)}


def test_rule_d_every_line_and_the_classes():
    from knee.nb import n11

    lo, mid, up = 8e-5, 3.2e-4, 1.28e-3
    b = dict(enumerate((0.70, 0.72, 0.74, 0.76, 0.78)))
    low2 = _five(b, [-0.03] * 5)
    C = n11.ext_compare
    assert C(_five(b, [0.0051] * 5), b)["class"] == "HIGHER" and C(_five(b, [0.005] * 5), b)["class"] == "BAND"
    assert C(_five(b, [-0.0051] * 5), b)["class"] == "LOWER" and C(_five(b, [-0.005] * 5), b)["class"] == "BAND"
    assert C(_five(b, [0.02, 0.02, 0.02, 0.02, -0.01]), b)["class"] == "HIGHER"  # 4 of 5 folds
    sp = C(_five(b, [0.05, 0.05, 0.05, -0.01, -0.01]), b)
    assert sp["class"] == "SPLIT" and sp["d"] == 0.026 and sp["a_higher_in"] == {"folds": [0, 1, 2]}
    assert C(_five(b, [-0.05, -0.05, -0.05, 0.01, 0.01]), b)["class"] == "SPLIT"

    def D(dm, du, **kw):
        end = {2e-5: low2, lo: b}
        if dm is not None:
            end[mid] = _five(b, dm)
        if du is not None:
            end[kw.get("upper", up)] = _five(b, du)
        return n11.ext_rule_d(end, mid, **kw)

    up5, dn5, flat = [0.02] * 5, [-0.02] * 5, [0.001] * 5
    r = D(None, up5, reasons={mid: "invalid in fold 1"})
    assert (r["line"], r["c1"], r["optimum"], r["r_star"]) == ({"k": 1}, "NONE", None, up)  # c2 is compare(U, 8e-5)
    assert r["words"] == "no five-fold mean for 3.2e-4 (invalid in fold 1); per-fold lines only; the question is not answered"
    r = D([0.05, 0.05, 0.05, -0.01, -0.01], dn5)
    assert r["line"] == {"k": 2} and r["words"] == ("unresolved: the folds disagree on 3.2e-4 against 8e-5 "
                                                    "(mean difference +0.0260, 3 of 5 folds); no optimum is named")
    r = D(up5, [0.02 + x for x in (-0.05, -0.05, -0.05, 0.01, 0.01)])  # c2 SPLIT, against 3.2e-4
    assert (r["line"], r["c1"], r["c2"]) == ({"k": 2}, "HIGHER", "SPLIT") and "1.28e-3 against 3.2e-4" in r["words"]
    r = D(dn5, up5)
    assert (r["line"], r["optimum"], r["r_star"]) == ({"k": 3}, None, up) and r["words"] == "not monotone; no optimum is named; reported as is"
    for du, c2 in ((dn5, "LOWER"), (flat, "BAND")):
        r = D(dn5, du)
        assert (r["line"], r["c2"], r["optimum"], r["r_star"]) == ({"k": 4}, c2, lo, lo)
        assert r["words"] == "optimum of the grid: 8e-5, bracketed (2e-5 and 3.2e-4 are both lower)"
    r = D(dn5, None, upper_invalid=True, reasons={up: "invalid in folds 0, 2"})
    assert (r["line"], r["c2"]) == ({"k": 4}, "UNSTABLE") and r["words"].endswith(
        "are both lower); 1.28e-3 has no five-fold mean (invalid in folds 0, 2)")
    assert D(dn5, None, reasons={up: "not run"})["c2"] == "NOMEAN"
    for dm, cur in ((flat, lo), (up5, mid)):
        r = D(dm, None, upper_invalid=True)
        assert (r["line"], r["c2"], r["r_star"], r["optimum"]) == ({"k": 5}, "UNSTABLE", cur, None)
        assert r["words"] == f"closed by instability above {n11._lr_text(cur)} (warm-up one twelfth, no clipping, fp16), not by score"
        r = D(dm, None, reasons={up: "not run (quota)"})
        assert (r["line"], r["c2"], r["r_star"]) == ({"k": 6}, "NOMEAN", cur)
        assert r["words"] == "no five-fold mean for 1.28e-3 (not run (quota)); the upper side is open"
    for dm, du in ((flat, up5), (up5, [0.04] * 5)):
        r = D(dm, du)
        assert (r["line"], r["c2"], r["r_star"], r["optimum"]) == ({"k": 7}, "HIGHER", up, None)
        assert r["words"] == "not bracketed: the optimum is at or above 1.28e-3"
    r = D(flat, dn5)
    assert (r["line"], r["optimum"], r["r_star"]) == ({"k": 8}, lo, lo)
    assert r["words"] == "flat from 8e-5 to 3.2e-4, lower at 1.28e-3; the grid optimum is named as 8e-5, the lower rate of the tie"
    r = D(flat, [0.004] * 5)
    assert (r["line"], r["optimum"], r["r_star"]) == ({"k": 9}, None, lo)
    assert r["words"] == "flat from 8e-5 to 1.28e-3 inside 0.005; no optimum is named; the upper side is open"
    r = D(up5, flat)  # U against 3.2e-4: 0.019 lower
    assert (r["line"], r["cur"], r["optimum"], r["r_star"]) == ({"k": 10}, mid, mid, mid)
    assert r["words"] == "optimum of the grid: 3.2e-4, bracketed (8e-5 and 1.28e-3 are both lower)"
    r = D(up5, [0.022] * 5)
    assert (r["line"], r["optimum"], r["r_star"]) == ({"k": 11}, None, mid) and r["words"] == "rises to 3.2e-4, flat to 1.28e-3; the upper side is open"
    # a rate with four folds has no mean; the swap adds its sentence and U is 6.4e-4
    four = {k: v for k, v in _five(b, up5).items() if k != 4}
    assert n11.ext_rule_d({lo: b, mid: four, up: _five(b, up5)}, mid, up)["c1"] == "NONE"
    r = D(up5, flat, upper=6.4e-4, swapped_from=up)
    assert r["upper"] == 6.4e-4 and r["words"] == "optimum of the grid: 3.2e-4, bracketed (8e-5 and 6.4e-4 are both lower)"
    assert r["words_swap"] == "1.28e-3 was unstable in folds 0 and 1 and was not run in the other three"
    assert D(up5, flat)["words_swap"] is None
    # d by the one convention: the mean of the per-fold differences at 4 decimals (the keep rule's way)
    A = dict(enumerate((0.7951, 0.7950, 0.7950, 0.7950, 0.7951)))  # mean 0.79504: shown as 0.7950
    B = dict(enumerate((0.7900, 0.7900, 0.7900, 0.7899, 0.7899)))  # mean 0.78996: shown as 0.7900
    c = C(A, B)
    assert (c["mean_a"], c["mean_b"], c["d"], c["class"]) == (0.795, 0.79, 0.0051, "HIGHER")
    assert C(B, A)["d"] == -0.0051 and C(B, A)["class"] == "LOWER"
    A[0] = 0.7947  # the mean difference is now 0.00500: not more than 0.005
    assert (C(A, B)["d"], C(A, B)["class"]) == (0.005, "BAND")
    assert "mean over the folds of the differences" in r["compare"]["convention"] == n11.EXT_D_CONVENTION
    # no outcome writes a string above the summaries' 200 characters: line 1, the split line, line 4 with a reason
    long = "invalid in folds 0, 1; not run in folds 2, 3; not finished in fold 4; start check failed in fold 3"
    outs = [D(None, up5, reasons={mid: long}, swapped_from=up, upper=6.4e-4), D([0.05, 0.05, 0.05, -0.01, -0.01], dn5, swapped_from=up, upper=6.4e-4),
            D(dn5, None, upper_invalid=True, reasons={6.4e-4: long}, swapped_from=up, upper=6.4e-4),
            D(flat, None, reasons={up: long}), D(dn5, None, reasons={up: "invalid in fold 2; not run in fold 4"}, upper_invalid=True)]
    for o in outs:
        assert _longest(o) <= 200
    assert outs[0]["words"] == "no five-fold mean for 3.2e-4 (several causes: see rule_d.reasons); per-fold lines only; the question is not answered"
    assert outs[4]["words"].endswith("1.28e-3 has no five-fold mean (invalid in fold 2; not run in fold 4)")
    from knee import agg

    agg._check(json.loads(json.dumps(r)))
    # the refinement point: the geometric midpoint between r* and its better neighbour
    R = n11.ext_refinement
    assert R({2e-5: 0.76, lo: 0.79, mid: 0.78, up: 0.70}, lo)["rate"] == 1.6e-4
    assert R({2e-5: 0.79, lo: 0.79, mid: 0.78}, lo)["rate"] == 4e-5  # a tie: the lower neighbour
    assert R({2e-5: 0.76, lo: 0.79, mid: 0.80, up: 0.78}, mid)["rate"] == 1.6e-4
    assert R({2e-5: 0.76, lo: 0.79, mid: 0.80, up: 0.795}, mid)["rate"] == 6.4e-4
    assert R({lo: 0.78, mid: 0.80, 6.4e-4: 0.79}, mid, ran=(mid, 6.4e-4))["rate_text"] == "4.5e-4"
    x = R({lo: 0.78, mid: 0.79, up: 0.80}, up, ran=(mid, 6.4e-4, up))
    assert (x["rate"], x["neighbour"], x["already_ran"]) == (6.4e-4, mid, True)
    assert R({lo: 0.78}, lo)["rate"] is None and [n11._lr_text(v) for v in (8e-5, 3.2e-4, 1.28e-3, 2e-5)] == ["8e-5", "3.2e-4", "1.28e-3", "2e-5"]


# ---------------------------------------------------------------- the selection record

ENV = {"python": "3.x", "torch": "2.x", "device_name": "T4"}


def _attached(tmp_path, work, refs, env3=None, scores=None, commit="095c91f", mutate=None, b_lite=False):
    """A copy of the session's output as a notebook input (test_overrides off, a known environment), and
    hand-made iteration-3 units whose states carry the reference numbers. scores: {lr key: {fold: end}} to
    write into the extension's states."""
    root = tmp_path / "in"
    shutil.copytree(work, root / "knee-n11x-inner")
    rj = root / "knee-n11x-inner" / "run.json"
    j = json.loads(rj.read_text())
    j["config"]["test_overrides"] = False
    j["commit"] = "abc1234"
    rj.write_text(json.dumps(j))
    for p in (root / "knee-n11x-inner").rglob("state.json"):
        b = json.loads(p.read_text())
        b["environment"] = dict(ENV)
        key, k = p.parent.parent.name.rsplit("_", 1)[1], int(b["fold"])
        if scores and key in scores and k in scores[key]:
            b["ext"]["run"]["val_macro_auc"] = scores[key][k]
        if mutate:
            mutate(key, k, b)
        p.write_text(json.dumps(b))
    for k in range(5):
        sess = "a" if k < 3 else "b"
        ud = root / f"knee-n11-{sess}" / "private" / f"n11_{sess}" / f"{DINO}_f{k}"
        ud.mkdir(parents=True)
        (root / f"knee-n11-{sess}" / "run.json").write_text(json.dumps({"commit": commit, "config": {"test_overrides": False}}))
        f = f"fold_{k}"
        curve = {"train_loss": [1.0 - 0.01 * i for i in range(12)], "val_loss": [1.2] * 12,
                 "val_macro_auc": [0.5 + 0.001 * i for i in range(11)] + [refs["end_val_macro_auc"]["8e-05"][f]]}
        rung = {"val_macro_auc": 0.5, "curve": {"val_macro_auc": [0.5, 0.5, 0.5], "train_loss": [1.0, 1.0, 1.0]}}
        if k == 0:
            rung = {"val_macro_auc": refs["replay"]["val_macro_auc"][2],
                    "curve": {"val_macro_auc": refs["replay"]["val_macro_auc"], "train_loss": refs["replay"]["train_loss"]}}
        st = {"encoder": DINO, "fold": k, "session": sess, "environment": dict(env3 or ENV), "files_sha16": {},
              "inner": {"complete": True, "start": {"val_macro_auc": refs["start_val_macro_auc"][f]},
                        "head_only": {"seed_0": {"val_macro_auc": 0.5}},
                        "rung0": {"5e-06": {"val_macro_auc": 0.49}, "2e-05": {"val_macro_auc": 0.5}, "8e-05": rung},
                        "to_the_end": {"8e-05": {"val_macro_auc": refs["end_val_macro_auc"]["8e-05"][f], "curve": curve},
                                       "2e-05": {"val_macro_auc": refs["end_val_macro_auc"]["2e-05"][f],
                                                 "curve": {**curve, "val_macro_auc": [0.5] * 11 + [refs["end_val_macro_auc"]["2e-05"][f]]}}}}}
        if b_lite:  # iteration 3's 2e-5 validation predictions: every scored validation study of the fold
            from knee.labels import FINDINGS
            from knee.nb import n11
            from tests.test_n11 import _write_rows

            rng = np.random.default_rng(k)
            vs = "+".join(sorted(TABLE[k]["val"]))
            n = 60
            _write_rows(ud / "inner_b_lite.csv", [f"{vs}_{i}" for i in range(n)], k, rng.random((n, len(FINDINGS))),
                        (rng.random((n, len(FINDINGS))) < 0.3).astype(float), {"val_set": vs})
            st["files_sha16"]["inner_b_lite.csv"] = n11.sha16(ud / "inner_b_lite.csv")
        (ud / "state.json").write_text(json.dumps(st))
    return root


def test_selection_record_from_the_session_and_iteration3(session, refs, tmp_path, monkeypatch, capfd):
    from knee import agg, paths
    from knee.nb import n11

    s, work = session
    scores = {"0.00032": dict(enumerate([0.6051, 0.6150, 0.6400, 0.6300, 0.6500])),  # folds 0, 2, 4 move to 3.2e-4
              "0.00128": dict(enumerate([0.6000, 0.6000, 0.6460, 0.6000, 0.6300]))}  # fold 2: +0.006 on its pick
    root = _attached(tmp_path, work, refs, scores=scores)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "out")
    rec = n11.select_lrx(root=root, reference=refs, n_boot=20)
    assert "STEP n11x_select: OK" in capfd.readouterr().out
    assert rec["status"] == "selected" and rec["no_fold_moves_because"] == [] and rec["missing"] == []
    assert rec["picks"] == {"fold_0": 3.2e-4, "fold_1": 8e-5, "fold_2": 1.28e-3, "fold_3": 8e-5, "fold_4": 3.2e-4}
    assert rec["folds_moved"] == {"folds": [0, 2, 4]} and rec["any_fold_moves"] is True
    f2 = rec["folds"]["fold_2"]
    assert f2["scores"] == {"2e-05": 0.57, "8e-05": 0.62, "0.00032": 0.64, "0.00128": 0.646} and f2["pick_key"] == "0.00128"
    assert f2["runs"]["0.00032"] == {"status": "finished", "valid": True, "invalid": None, "eligible": True,
                                     "why_not_eligible": None, "skipped_step_share": 0.0, "head_lr": 3e-3}
    assert set(f2["best_twelfth"]) == {"2e-05", "8e-05", "0.00032", "0.00128"} and f2["start_check"]["0.00128"]["passed"] is True
    assert set(f2["quarter_scores"]) == {"5e-06", "2e-05", "8e-05", "0.00032", "0.00128"}
    rd = rec["rule_d"]
    assert rd["means"] == {"2e-05": 0.57, "8e-05": 0.62, "0.00032": 0.628, "0.00128": 0.6152}
    assert (rd["c1"], rd["c2"], rd["cur"], rd["r_star"], rd["line"]) == ("HIGHER", "LOWER", 3.2e-4, 3.2e-4, {"k": 10})
    assert rd["words"] == "optimum of the grid: 3.2e-4, bracketed (8e-5 and 1.28e-3 are both lower)" and rd["optimum"] == 3.2e-4
    assert rd["compare"]["c1"]["d"] == 0.008 and rd["compare"]["c1"]["a_higher_in"] == {"folds": [0, 1, 2, 4]}
    assert rd["words_as_shown"] == rd["words"] and rd["label"] is None and rd["replay_max_abs_gap"] == 0.0
    assert set(rd["mean_curve"]) == {"2e-05", "8e-05", "0.00032", "0.00128"} and len(rd["mean_curve"]["0.00032"]["val_macro_auc"]) == 12
    assert rec["refinement"] == {**rec["refinement"], "r_star": 3.2e-4, "neighbour": 8e-5, "rate": 1.6e-4, "may_run": True}
    assert rec["iteration3"]["attached"] and rec["iteration3"]["constants_equal_the_states"] and rec["iteration3"]["knee_commits"] == ["095c91f"]
    assert rec["extension_commits"] == ["abc1234"] and rec["environment"] == {
        "compared_with_iteration3": True, "differences": [], "extension": ENV, "label": None}
    g1, g2 = rec["graphs"]["g1"], rec["graphs"]["g2"]
    assert g1["rates"] == [5e-6, 2e-5, 8e-5, 3.2e-4, 1.28e-3] and g1["mean_end"]["0.00032"] == 0.628 and g1["picks"] == rec["picks"]
    assert g1["end"]["0.00128"]["fold_2"] == 0.646 and g1["start_head"] == refs["start_val_macro_auc"] and g1["head_only_mean"] == 0.5
    assert g2["folds"]["fold_0"]["panels"] == ["8e-05", "0.00032", "0.00128"] and g2["folds"]["fold_1"]["panels"] == ["2e-05", "8e-05", "0.00032"]
    assert set(g2["folds"]["fold_0"]["curves"]["0.00032"]) == {"train_loss", "val_loss", "val_macro_auc"}
    assert rec["intervals"]["arms"] == ["0.00032", "0.00128"] and set(rec["intervals"]["pairs"]) == {"0.00128_minus_0.00032"}
    # signed: time and sha256; the written file is the record and passes the privacy checks
    assert rec["sha256"] == n11._signed({k: v for k, v in rec.items() if k != "sha256"})["sha256"] and rec["written_at_utc"].endswith("Z")
    on_disk = json.loads((tmp_path / "out" / "agg_n11x_select.json").read_text())
    assert on_disk == rec
    agg._check(on_disk)
    # the gates: test overrides, the environment, the constants, the replay record, nothing of iteration 3
    with pytest.raises(Exception):
        n11.select_lrx(root=work, reference=refs, reference_only=True, n_boot=0)  # the session ran with test overrides
    ext = n11.find_ext_units(root)

    def sel(**kw):
        return n11.selection_ext(ext, n11.find_units(root), **{"stop_flag": None, "reference": refs, "n_boot": 0, **kw})

    assert sel()["picks"] == rec["picks"]
    r = n11.selection_ext(ext, None, None, reference=refs, n_boot=0)
    assert (r["status"], r["inconclusive_because"], set(r["picks"].values())) == ("inconclusive", ["environment_not_compared"], {8e-5})
    assert r["refinement"]["may_run"] is False and r["folds"]["fold_2"]["runs"]["0.00032"]["why_not_eligible"].startswith("no fold moves")
    assert n11.selection_ext(ext, None, None, reference=refs, n_boot=0, expect_env=ENV)["status"] == "selected"
    r = n11.selection_ext(ext, None, None, reference=refs, n_boot=0, expect_env={**ENV, "torch": "other"})
    assert r["status"] == "inconclusive" and r["environment"]["differences"] == ["torch"] and r["rule_d"]["label"] == "across environments"
    assert r["rule_d"]["words_as_shown"] == r["rule_d"]["words"]  # Rule D's words are shown with that label
    wrong = copy.deepcopy(refs)
    wrong["end_val_macro_auc"]["8e-05"]["fold_3"] += 0.0001
    r = sel(reference=wrong)
    assert r["inconclusive_because"] == ["reference_mismatch"] and r["iteration3"]["constants_that_differ"] == ["end 8e-05 fold_3"]
    r = sel(stop_flag={"reason": "replay", "max_abs_gap": 0.02})
    assert r["status"] == "inconclusive" and r["rule_d"]["words_as_shown"] == "inconclusive: reused and new scores cannot be compared at 0.005"
    r = sel(describe=("inner",))
    assert set(r["picks"].values()) == {8e-5} and r["folds"]["fold_2"]["runs"]["0.00128"]["why_not_eligible"] == "a refinement point is never eligible"
    assert r["status"] == "selected"


def test_selection_start_check_per_rate_and_missing_replay(session, refs, tmp_path):
    from knee.nb import n11

    s, work = session
    scores = {"0.00032": dict(enumerate([0.70] * 5)), "0.00128": dict(enumerate([0.75] * 5))}

    def fail_start(key, k, b):
        if key == "0.00128" and k == 1:
            b["ext"]["start"]["passed"] = False

    root = _attached(tmp_path, work, refs, scores=scores, mutate=fail_start)
    r = n11.selection_ext(n11.find_ext_units(root), n11.find_units(root), None, reference=refs, n_boot=0)
    assert r["status"] == "selected"
    assert r["picks"] == {**{f"fold_{k}": 1.28e-3 for k in range(5)}, "fold_1": 3.2e-4}  # fold 1 cannot pick that rate
    assert r["folds"]["fold_1"]["runs"]["0.00128"]["why_not_eligible"] == "the start check of this rate's process failed in this fold"
    rd = r["rule_d"]
    assert "0.00128" not in rd["means"] and (rd["c1"], rd["c2"], rd["line"]) == ("HIGHER", "NOMEAN", {"k": 6})
    assert rd["words"] == "no five-fold mean for 1.28e-3 (start check failed in fold 1); the upper side is open"
    assert r["graphs"]["g1"]["mean_end"]["0.00128"] is None and "0.00128" not in rd["mean_curve"]

    def no_replay(key, k, b):
        b.pop("replay", None)

    root2 = _attached(tmp_path / "2", work, refs, scores=scores, mutate=no_replay)
    r = n11.selection_ext(n11.find_ext_units(root2), n11.find_units(root2), None, reference=refs, n_boot=0)
    assert (r["status"], r["inconclusive_because"]) == ("inconclusive", ["no_replay_record"]) and set(r["picks"].values()) == {8e-5}


# ---------------------------------------------------------------- the smoke run

def test_smoke_runs_both_processes_a_forced_invalid_run_and_the_selection(inputs, refs, tmp_path, monkeypatch, capfd):
    from knee import agg
    from knee.nb import n11

    off = copy.deepcopy(refs)
    off["replay"]["val_macro_auc"] = [0.9, 0.9, 0.9]  # a tiny schedule cannot match the logged replay: no stop in smoke mode
    s = _lrx(inputs, tmp_path / "w", monkeypatch, smoke=True, budget_h=0.5, reference=off, smoke_val=20)
    assert "STEP n11x_smoke: OK" in capfd.readouterr().out
    assert s["status"] == "ok (smoke: no score is meaningful)" and s["session"] == "smoke" and s["test_overrides"] is True
    assert s["smoke_passed"] is True and s["stop_flag"] is None and not (tmp_path / "w" / "private" / "n11x_STOP").exists()
    ck = s["smoke_checks"]
    assert [v for k, v in ck.items() if k.startswith("c") and k != "c7_environment_equals_iteration3"] == [True] * 7
    assert ck["c7_environment_equals_iteration3"] is None
    assert s["replay"]["passed"] is False and s["replay"]["stop_rule_applied"] is False and len(s["replay"]["val_macro_auc_abs_gaps"]) == 3
    ov = s["overview"]
    assert ov["0.00032"]["fold_2"]["invalid"] == {"k": 2, "reason": "forced by the smoke run: no real fault"}
    assert all(v["status"] == "finished" for key in ov for f, v in ov[key].items() if (key, f) != ("0.00032", "fold_2"))
    assert all(v["passed"] and v["abs_difference"] == 0.0 for r in s["start_check"].values() for v in r.values())  # the real check
    sts = _states(tmp_path / "w", "n11x_smoke_0.00128")
    for b in sts.values():
        cfg = b["ext"]["run"]["config"]
        assert cfg["steps"] == 12 and cfg["epochs"] == 1 and len(b["ext"]["run"]["curve"]["val_macro_auc"]) == 12
    import pandas as pd

    assert len(pd.read_csv(tmp_path / "w" / "private" / "n11x_smoke_0.00128" / f"{DINO}_f1" / "inner_b_lite_0.00128.csv")) == 20
    names = sorted(p.name for p in (tmp_path / "w").glob("*.json"))
    assert names == ["agg_n11x_smoke.json", "agg_n11x_smoke_0.00032.json", "agg_n11x_smoke_0.00128.json",
                     "agg_n11x_smoke_select.json", "run.json"]
    for n in names:
        agg._check(json.loads((tmp_path / "w" / n).read_text()))
    sel = json.loads((tmp_path / "w" / "agg_n11x_smoke_select.json").read_text())
    assert sel["smoke"] is True and sel["status"] == "selected" and s["selection"]["sha256"] == sel["sha256"]
    assert sel["folds"]["fold_2"]["runs"]["0.00032"]["eligible"] is False
    assert json.loads((tmp_path / "w" / "agg_n11x_smoke.json").read_text())["smoke_checks"] == ck
    assert n11.find_units(tmp_path / "w") == {}
    # the CPU step on the smoke output; a smoke output is refused as a result
    rec = n11.select_lrx(root=tmp_path / "w", smoke=True, reference_only=True, reference=off, n_boot=0)
    assert rec["smoke"] is True and (tmp_path / "w" / "agg_n11x_smoke_select.json").exists()
    with pytest.raises(Exception):
        n11.select_lrx(root=tmp_path / "w", sessions=("smoke",), reference_only=True, reference=off, n_boot=0)
    # a failed check is said: the forced run is not invalid
    real, real_run = n11._inner_ext, n11.run_to

    def run_to(c, r, twelfth, limit=None):
        real_run(c, r, twelfth, limit)
        if c.fold == 1 and r.with_encoder and r.lr_enc == 1.28e-3:
            r.skipped = r.step // 2  # skipped steps do not invalidate a smoke run: reported only

    monkeypatch.setattr(n11, "run_to", run_to)
    monkeypatch.setattr(n11, "_inner_ext", lambda c, st, lr, head0, force_k=None: real(c, st, lr, head0, None))
    s = _lrx(inputs, tmp_path / "w2", monkeypatch, smoke=True, budget_h=0.5, reference=off, folds=(0, 1, 2))
    assert s["smoke_skipped_step_share"]["0.00128"] == {"fold_0": 0.0, "fold_1": 0.5, "fold_2": 0.0}
    assert s["overview"]["0.00128"]["fold_1"]["status"] == "finished" and s["swapped"] == []
    assert s["smoke_checks"]["c5_every_other_run_has_12_records_and_a_finite_end_score"] is True
    assert s["smoke_passed"] is False and s["smoke_checks"]["c4_forced_run_invalid_with_twelfth_and_reason"] is False
    assert s["smoke_checks"]["c1_every_process_lists_its_folds"] is False  # three folds, not five


def test_split_folds_for_one_rate_and_the_reference_constants(inputs, refs, tmp_path, monkeypatch):
    from knee.nb import n11

    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="refine", rates=(1.6e-4,), split_folds=True, replay=False,
             reference=refs, devices=["cpu", "cpu"], processes=False)
    assert s["status"] == "ok" and s["processes"] == ["1.6e-4 on cpu: folds 0, 2, 4", "1.6e-4 on cpu: folds 1, 3"]
    assert sorted(_states(tmp_path / "w", "n11x_refine_0.00016")) == [0, 1, 2, 3, 4] and s["replay"] is None
    assert sorted(p.name for p in (tmp_path / "w").glob("agg_*.json")) == ["agg_n11x_refine.json", "agg_n11x_refine_0.00016.json"]
    R = n11.EXT_REF  # the constants of the declaration, section 4 and section 2
    assert list(R["start_val_macro_auc"].values()) == [0.7295, 0.7753, 0.7600, 0.7516, 0.7732]
    assert R["replay"] == {"fold": 0, "lr": 8e-5, "val_macro_auc": [0.7269, 0.7408, 0.7449], "train_loss": [0.9672, 0.9726, 0.9057]}
    assert list(R["end_val_macro_auc"]["8e-05"].values()) == [0.7246, 0.8236, 0.7900, 0.7978, 0.8115]
    assert list(R["end_val_macro_auc"]["2e-05"].values()) == [0.7138, 0.7951, 0.7654, 0.7695, 0.7820]
    assert [(v["k"], v["val_macro_auc"]) for v in R["best_twelfth"]["8e-05"].values()] == [
        (5, 0.7498), (10, 0.8245), (6, 0.7984), (6, 0.8000), (10, 0.8126)]
    assert (n11.EXT_LRS, n11.EXT_SWAP_LR, n11.EXT_BAND, n11.EXT_MAX_SKIPPED, n11.EXT_REPLAY_TOL, n11.EXT_START_TOL, n11.EXT_START_STOP) == (
        (3.2e-4, 1.28e-3), 6.4e-4, 0.005, 0.05, 0.005, 0.0005, 3)
    assert n11.LRS == (5e-6, 2e-5, 8e-5) and n11.REGISTERED_LR == 2e-5 and n11.run_extension is n11.run_lrx
    assert n11._ext_step("inner", 3.2e-4) == "n11x_inner_0.00032" and n11._ext_step("inner", 8e-5) == "n11x_inner_8e-05"
    from knee import agg

    agg._check(json.loads(json.dumps(R)))


# ---------------------------------------------------------------- the notebook entries and their card

def test_n11x_cli_entries_match_the_runbook_card():
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (ROOT / "notebooks" / "RUNBOOK.md").read_text().split("## Learning-rate extension N11x")[1]
    gpu_inputs = [f"u/knee-{u}" for u in ("n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n5", "n6a")]
    assert mk.cell2("n11x-smoke") == f'from knee.nb import n11; n11.run_lrx("smoke", smoke=True, budget_h=0.5, expect={mk.EXPECT})'
    assert mk.cell2("n11x-inner") == f'from knee.nb import n11; n11.run_lrx("inner", budget_h=5.8, expect={mk.EXPECT})'
    assert mk.cell2("n11x-select") == "from knee.nb import n11; n11.select_lrx()"
    assert mk.cell2("n11x-smoke-select") == "from knee.nb import n11; n11.select_lrx(smoke=True)"
    assert "split_folds=True, replay=False, budget_h=3.4" in mk.cell2("n11x-refine")
    for name in ("n11x-smoke", "n11x-inner", "n11x-refine"):
        meta = mk.metadata("u", name)
        assert mk.cell2(name) in card and meta["enable_gpu"] is True and meta["enable_internet"] is True
        assert meta["kernel_sources"] == gpu_inputs and meta["kernel_sources"] == mk.metadata("u", "n11-a")["kernel_sources"]
        assert meta["machine_shape"] == "NvidiaTeslaT4" and not any("n11-" in x for x in meta["kernel_sources"])  # R9
    for name, first in (("n11x-select", "n11x-inner"), ("n11x-smoke-select", "n11x-smoke")):
        meta = mk.metadata("u", name)
        assert mk.cell2(name) in card and meta["enable_gpu"] is False
        assert meta["kernel_sources"] == [f"u/knee-{first}", "u/knee-n11-a", "u/knee-n11-b"]
    assert "TODO D2" in card and mk.NOTEBOOKS["n11x-inner"][4] >= 1.1 * 5.8 * 3600


# ---------------------------------------------------------------- review r1 of bdc49c3

def test_smoke_select_notebook_call_with_iteration3_prediction_files(inputs, refs, tmp_path, monkeypatch, capfd):
    """knee-n11x-smoke-select's exact call on a smoke output beside iteration-3 units that hold inner_b_lite.csv
    (all validation studies; the smoke's files hold 32): no interval is computed and iteration 3's file is
    not opened, so the record is written and the environment check (criterion 7) can be read."""
    from knee import agg, paths
    from knee.nb import n11

    s = _lrx(inputs, tmp_path / "w", monkeypatch, smoke=True, budget_h=0.5, reference=refs)
    assert s["smoke_checks"]["c7_selection_ran_in_smoke_mode"] is True
    root = _attached(tmp_path, tmp_path / "w", refs, b_lite=True)
    monkeypatch.setattr(paths, "INPUT", root)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "out")
    monkeypatch.setattr(n11, "_read_rows", lambda p: pytest.fail("no per-study file is opened by the smoke selection"))
    rec = n11.select_lrx(smoke=True)  # cell 2 of knee-n11x-smoke-select
    assert "STEP n11x_smoke_select: OK" in capfd.readouterr().out
    assert rec["environment"]["compared_with_iteration3"] is True and rec["environment"]["differences"] == []
    assert rec["intervals"]["pairs"] == {} and rec["intervals"]["n_boot"] == 0 and rec["smoke"] is True
    assert rec["iteration3"]["attached"] and rec["inconclusive_because"] == ["reference_mismatch"]  # the fixture is not EXT_REF
    agg._check(json.loads((tmp_path / "out" / "agg_n11x_smoke_select.json").read_text()))
    rec = n11.select_lrx(smoke=True, reference=refs)
    assert rec["status"] == "selected" and rec["iteration3"]["constants_equal_the_states"] is True


def test_a_described_refinement_point_at_the_swap_rate_is_not_a_swap(session, refs, tmp_path):
    from knee.nb import n11

    s, work = session
    root = _attached(tmp_path, work, refs)
    src = root / "knee-n11x-inner" / "private" / "n11x_inner_0.00032"
    dst = root / "knee-n11x-refine" / "private" / "n11x_refine_0.00064"
    shutil.copytree(src, dst)
    (root / "knee-n11x-refine" / "run.json").write_text(json.dumps({"commit": "abc1234", "config": {"test_overrides": False}}))
    for p in dst.rglob("state.json"):
        b = json.loads(p.read_text())
        b["session"] = "refine"
        b["ext"].update({"lr": 6.4e-4, "lr_key": "0.00064"})
        b["ext"]["run"]["val_macro_auc"] = 0.99
        p.write_text(json.dumps(b))
    units = n11.find_ext_units(root, ("inner", "refine"))
    r = n11.selection_ext(units, n11.find_units(root), None, reference=refs, n_boot=0, describe=("refine",))
    plain = n11.selection_ext(n11.find_ext_units(root), n11.find_units(root), None, reference=refs, n_boot=0)
    assert r["rule_d"]["upper"] == 1.28e-3 and r["rule_d"]["words_swap"] is None and r["new_rates"] == [3.2e-4, 6.4e-4, 1.28e-3]
    assert (r["rule_d"]["words"], r["rule_d"]["c1"], r["rule_d"]["c2"], r["picks"]) == (
        plain["rule_d"]["words"], plain["rule_d"]["c1"], plain["rule_d"]["c2"], plain["picks"])
    assert all(v["runs"]["0.00064"]["why_not_eligible"] == "a refinement point is never eligible" for v in r["folds"].values())
    assert r["rule_d"]["means"]["0.00064"] == 0.99 and r["refinement"]["already_ran"] in (True, False)


def test_replay_process_ending_without_its_record_stops_the_session(inputs, refs, tmp_path, monkeypatch):
    from knee.nb import n11

    trained = []
    real = n11._inner_ext
    monkeypatch.setattr(n11, "_inner_ext", lambda c, *a, **k: (trained.append(c.step), real(c, *a, **k))[1])
    monkeypatch.setattr(n11, "_ext_replay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic failure in the replay")))
    s = _lrx(inputs, tmp_path / "w", monkeypatch, session="inner", folds=(0, 1), reference=refs)
    flag = json.loads((tmp_path / "w" / "private" / "n11x_STOP").read_text())
    assert flag["reason"] == "replay: no record" and flag["lr_key"] == "0.00128" and trained == []
    assert s["status"].startswith("inconclusive") and s["replay"] is None
    assert s["errors"][f"{DINO}_f0@8e-05"].startswith("RuntimeError at")
    env = _states(tmp_path / "w", "n11x_inner_0.00128")[0]["environment"]
    for p in (tmp_path / "w").glob("run.json"):  # as a real session's output: no test overrides
        j = json.loads(p.read_text())
        j["config"]["test_overrides"] = False
        p.write_text(json.dumps(j))
    sel = n11.selection_ext(n11.find_ext_units(tmp_path / "w"), None, s["stop_flag"], reference=refs, n_boot=0, expect_env=env)
    assert (sel["status"], sel["inconclusive_because"]) == ("inconclusive", ["no_replay_record"]) and not sel["refinement"]["may_run"]
    # a process without the replay that fails writes no flag: the other process goes on
    monkeypatch.undo()
    from knee import paths

    monkeypatch.setattr(paths, "WORKING", tmp_path / "w2")
    monkeypatch.setattr(n11, "_inner_ext", lambda c, *a, **k: (_ for _ in ()).throw(RuntimeError("x")) if c.step.endswith("0.00032") else real(c, *a, **k))
    s = _lrx(inputs, tmp_path / "w2", monkeypatch, session="inner", folds=(0, 1), reference=refs)
    assert s["status"] == "partial" and s["stop_flag"] is None and s["replay"]["passed"] is True
    assert all(v["status"] == "finished" for v in s["overview"]["0.00128"].values())
