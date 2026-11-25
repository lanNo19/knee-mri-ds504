"""N11x edge point (it3_lr_edge_r1): fold 0 above 1.28e-3. The GPU step is run_lrx with arguments only, at the
inner session's commit. Tested: everything run_lrx reaches is the text of 2f5b27a (the inner session's commit)
at 79e95c5 and here; run_lrx takes the edge call as declared (one fold, two rates, replay in the 5.12e-3
process, no swap); every line of the declared words on made-up scores; the record's refusals and its
inconclusive cases; the end-to-end path on synthetic data (edge session and inner session -> aggregates only ->
edge_lrx); the privacy checks of the written record; the card's cells."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn.metrics")

from tests.test_n11 import MAIN, PASSES, TABLE, _make_inputs, fake_factory  # noqa: E402
from tests.test_n11 import _shared as _shared11  # noqa: E402

DINO = MAIN[0]
ROOT = Path(__file__).resolve().parents[1]
INNER = "2f5b27aeb7de774dc5729be42d94bf1e7dc29f1c"
D2 = "79e95c5320e83442cbbc6ea42b0ea31a5b443082"


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


# ---------------------------------------------------------------- the GPU step is the inner session's code

def test_run_lrx_text_is_the_inner_sessions_at_both_commits_and_here():
    """2f5b27a ran the inner session. At 79e95c5 and on this branch, every src/knee file except nb/n10.py
    (not imported by n11) is byte-equal to 2f5b27a's, and n11.py only gained lines at its end."""
    for c in (INNER, D2):
        if _git("cat-file", "-e", f"{c}^{{commit}}").returncode != 0:
            pytest.skip("the commits are not in this checkout")
    from knee.nb import n11

    assert set(n11.EDGE_COMMITS) == {INNER, D2}
    files = _git("ls-tree", "-r", "--name-only", INNER, "src/knee").stdout.split()
    assert "src/knee/nb/n11.py" in files and "src/knee/nb/n13.py" in files
    for f in files:
        old = _git("show", f"{INNER}:{f}").stdout
        for rev in (D2, None):
            new = _git("show", f"{rev}:{f}").stdout if rev else (ROOT / f).read_text()
            if f == "src/knee/nb/n10.py":
                continue
            if f == "src/knee/nb/n11.py":
                assert new.splitlines()[:len(old.splitlines())] == old.splitlines(), rev
            else:
                assert new == old, (f, rev)
    src11 = (ROOT / "src/knee/nb/n11.py").read_text()
    assert "import n10" not in src11 and "n10." not in src11.split("\ndef run_lrx(")[1].split("\ndef ext_rule_s(")[0]
    # 79e95c5's new files under src/knee: none
    assert sorted(_git("ls-tree", "-r", "--name-only", D2, "src/knee").stdout.split()) == sorted(files)


# ---------------------------------------------------------------- the words, on made-up scores

REF = {0.00008: 0.7246, 0.00032: 0.7471, 0.00128: 0.7614}


def _words(**new):
    """new: {"mid"/"top": score, or a class name for a rate without a counted score}."""
    from knee.nb import n11

    rates = {"mid": n11.EDGE_MID, "top": n11.EDGE_TOP}
    scores, classes, unstable = dict(REF), {}, {}
    for k, v in new.items():
        lr = rates[k]
        if isinstance(v, str):
            classes[lr] = v
            if v == "UNSTABLE":
                unstable[lr] = (2, "non-finite train loss")
        else:
            scores[lr] = v
            classes[lr] = n11._edge_class({"finished": True, "valid": True, "val_macro_auc": v}, {"passed": True}, REF[0.00128])[0]
    return n11.edge_words(scores, classes, unstable)


def test_words_single_rate_the_three_outcomes_and_instability():
    w = _words(top=0.7400)  # lower by 0.0214
    assert w["line"] == {"k": 4} and w["best_point_tested"] == 1.28e-3
    assert w["words"].startswith("fold 0's best rate lies between 3.2e-4 and 5.12e-3; 1.28e-3 is the best grid point")
    w = _words(top=0.7700)  # higher by 0.0086
    assert w["line"] == {"k": 3} and w["words"].startswith("still rising: 5.12e-3 is +0.0086 over 1.28e-3") and "open edge" in w["words"]
    w = _words(top=0.7664)  # +0.0050: inside the band (more than 0.005 is needed)
    assert w["line"] == {"k": 5} and w["words"].startswith("flat above 1.28e-3: 5.12e-3 is within 0.005") and "+0.0050" in w["words"]
    w = _words(top=0.7564)  # -0.0050: inside the band too
    assert w["line"] == {"k": 5} and "-0.0050" in w["words"]
    w = _words(top=0.7563)  # -0.0051: lower
    assert w["line"] == {"k": 4}
    w = _words(top="UNSTABLE")
    assert w["line"] == {"k": 2} and w["words"].startswith("above 1.28e-3 closed by instability")
    assert w["words_top"] == "5.12e-3 was unstable in fold 0 (twelfth 2: non-finite train loss)"
    for c in ("UNFINISHED", "START_FAILED", "NO_RUN"):
        w = _words(top=c)
        assert w["line"] == {"k": 1} and w["words"].startswith("not answered") and c.lower().replace("_", " ") in w["words"]


def test_words_two_rates_every_line():
    w = _words(mid=0.7500, top=0.7300)  # both lower: the tighter bracket
    assert w["line"] == {"k": 4} and "between 3.2e-4 and 2.56e-3; 1.28e-3 is the best point tested" in w["words"]
    w = _words(mid=0.7700, top=0.7500)  # 2.56e-3 is the best point, 5.12e-3 lower than it
    assert w["best_point_tested"] == 2.56e-3 and w["line"] == {"k": 4}
    assert "between 1.28e-3 and 5.12e-3; 2.56e-3 is the best point tested" in w["words"]
    w = _words(mid=0.7700, top=0.7800)
    assert w["best_point_tested"] == 5.12e-3 and w["line"] == {"k": 3} and w["words"].startswith("still rising: 5.12e-3")
    w = _words(mid=0.7700, top=0.7720)  # rises to 2.56e-3, flat to 5.12e-3
    assert w["best_point_tested"] == 2.56e-3 and w["line"] == {"k": 5} and "flat above 2.56e-3" in w["words"]
    w = _words(mid=0.7700, top="UNSTABLE")
    assert w["line"] == {"k": 3} and "closed by instability above 2.56e-3" in w["words"] and w["words_top"]
    w = _words(mid=0.7620, top=0.7400)  # flat to 2.56e-3, lower at 5.12e-3
    assert w["line"] == {"k": 5} and "lower at 5.12e-3" in w["words"] and "between 3.2e-4 and 5.12e-3" in w["words"]
    w = _words(mid=0.7620, top=0.7600)
    assert w["line"] == {"k": 5} and "5.12e-3 is within 0.005" in w["words"]
    w = _words(mid=0.7500, top=0.7610)  # lower, then back inside the band
    assert w["line"] == {"k": 6} and w["words"].startswith("not monotone above 1.28e-3")
    w = _words(mid=0.7500, top="UNFINISHED")
    assert w["line"] == {"k": 4} and "2.56e-3" in w["words"] and "unfinished" in w["words_top"]
    w = _words(mid="UNSTABLE", top="UNSTABLE")
    assert w["line"] == {"k": 2}
    for w in (_words(mid=0.7700, top=0.7500), _words(mid=0.7620, top=0.7400), _words(mid=0.75, top=0.7610)):
        assert len(w["words"]) <= 200


# ---------------------------------------------------------------- the record, from hand-made aggregates

ENV = {"kaggle_image_git_commit": "abc", "kaggle_image_build_date": "20261120", "python": "3.11.13", "torch": "2.6.0",
       "cuda": "12.4", "cudnn": 90100, "device_name": "Tesla T4"}


def _file(session, lr, score, status="finished", passed=True, replay=True, commit=INNER, env=None, k=None):
    run = {"status": status, "finished": status in ("finished",), "valid": status == "finished",
           "val_macro_auc": score if status == "finished" else None, "invalid": {"k": k, "reason": "non-finite train loss"} if k else None,
           "best": {"k": 11, "val_macro_auc": score} if score else None, "curve": {"val_macro_auc": [score] * 12} if score else None}
    unit = {"fold": 0, "session": session, "environment": env or ENV,
            "config": {"micro_batch_studies": 2, "epochs": 12, "steps_inner": 348, "batch_size": 33, "head_lr": 6.86e-4},
            "ext": {"run": run, "start": {"passed": passed, "abs_difference": 0.0 if passed else 0.0012},
                    "tail_rms_before_training": 0.11519}}
    rp = {"fold": 0, "lr": 8e-5, "max_abs_gap": 0.0, "passed": True, "above_half_the_band": False}
    return {"session": session, "smoke": False, "test_overrides": False, "folds": [0], "status": "ok", "knee_commit": commit,
            "environment": env or ENV, "replay": rp if replay else None, "lr_key": str(lr), "units": {f"{DINO}_f0": unit}}


def _inner():
    return {"0.00032": _file("inner", 3.2e-4, 0.7471), "0.00128": _file("inner", 1.28e-3, 0.7614)}


def test_record_words_labels_and_privacy():
    from knee import agg
    from knee.nb import n11

    edge = {"0.00256": _file("edge", 2.56e-3, 0.7650, replay=False), "0.00512": _file("edge", 5.12e-3, 0.7400)}
    rec = n11.edge_record(edge, _inner())
    assert rec["status"] == "described" and rec["label"] is None and rec["line"] == {"k": 5}
    assert rec["words"] == rec["words_as_shown"] and rec["words"].startswith("flat above 1.28e-3 to 2.56e-3 (within 0.005), lower at 5.12e-3")
    assert "between 3.2e-4 and 5.12e-3; 1.28e-3 is the best point tested, the lower rate of the tie" in rec["words"]
    assert rec["runs"]["0.00512"]["class"] == "LOWER" and rec["runs"]["0.00512"]["difference_to_1.28e-3"] == -0.0214
    assert rec["runs"]["0.00256"]["class"] == "BAND" and rec["runs"]["0.00512"]["largest_move_per_weight"] == round(5.12e-3 * 174, 3)
    assert rec["pick"]["changed"] is False and rec["pick"]["fold_0"] == 1.28e-3 and rec["inner"]["constants_equal_the_files"]
    assert rec["environment"]["differences"] == [] and rec["edge_commits"] == [INNER]
    agg._check(json.loads(json.dumps(rec)))
    assert rec["sha256"] == n11.hashlib.sha256(n11._canon(rec)).hexdigest()
    # without the inner session: the constants, labelled
    rec = n11.edge_record(edge, None)
    assert rec["status"] == "described" and rec["label"] == "environment not compared" and rec["words_as_shown"].endswith("[environment not compared]")
    # another environment: labelled, the words still given
    env2 = {**ENV, "kaggle_image_build_date": "20261124"}
    rec = n11.edge_record({"0.00512": _file("edge", 5.12e-3, 0.7700, env=env2)}, _inner())
    assert rec["label"] == "across environments" and rec["environment"]["differences"] == ["kaggle_image_build_date"]
    assert rec["status"] == "described" and rec["words"].startswith("still rising")
    # a failed start check: that rate does not count
    rec = n11.edge_record({"0.00512": _file("edge", 5.12e-3, 0.7700, passed=False)}, _inner())
    assert rec["runs"]["0.00512"]["class"] == "START_FAILED" and rec["line"] == {"k": 1}
    # an unstable run
    rec = n11.edge_record({"0.00512": _file("edge", 5.12e-3, None, status="invalid", k=2)}, _inner())
    assert rec["line"] == {"k": 2} and "twelfth 2" in rec["words_top"]
    agg._check(json.loads(json.dumps(rec)))


def test_record_inconclusive_and_refused():
    from knee.nb import n11

    one = {"0.00512": _file("edge", 5.12e-3, 0.7700, replay=False)}
    rec = n11.edge_record(one, _inner())
    assert rec["status"] == "inconclusive" and rec["inconclusive_because"] == ["no_replay_record"] and rec["words"] is None
    assert rec["words_as_shown"].startswith("inconclusive")
    bad = copy.deepcopy({"0.00512": _file("edge", 5.12e-3, 0.7700)})
    bad["0.00512"]["replay"].update({"passed": False, "max_abs_gap": 0.0061})
    assert n11.edge_record(bad, _inner())["inconclusive_because"] == ["replay_gap_above_tolerance"]
    assert n11.edge_record({"0.00512": _file("edge", 5.12e-3, 0.77, commit="deadbeef")}, _inner())["inconclusive_because"] == ["unknown_commit"]
    assert n11.edge_record({"0.00512": _file("edge", 5.12e-3, 0.77, commit=D2)}, _inner())["status"] == "described"
    inner = _inner()
    inner["0.00128"]["units"][f"{DINO}_f0"]["ext"]["run"]["val_macro_auc"] = 0.7600
    assert n11.edge_record({"0.00512": _file("edge", 5.12e-3, 0.77)}, inner)["inconclusive_because"] == ["reference_mismatch"]
    inner = _inner()
    inner["0.00128"]["test_overrides"] = True  # a test session of the inner rates is not the reference
    assert n11.edge_record({"0.00512": _file("edge", 5.12e-3, 0.77)}, inner)["inconclusive_because"] == ["reference_mismatch"]
    with pytest.raises(ValueError, match="declared"):
        n11.edge_record({"0.00064": _file("edge", 6.4e-4, 0.77)}, _inner())
    with pytest.raises(ValueError, match="declared"):
        n11.edge_record({"0.00256": _file("edge", 2.56e-3, 0.77)}, _inner())
    with pytest.raises(ValueError, match="session"):
        n11.edge_record({"0.00512": _file("inner", 5.12e-3, 0.77)}, _inner())
    f = _file("edge", 5.12e-3, 0.77)
    with pytest.raises(ValueError, match="test overrides"):
        n11.edge_record({"0.00512": {**f, "test_overrides": True}}, _inner())
    with pytest.raises(ValueError, match="fold 0 only"):
        n11.edge_record({"0.00512": {**f, "folds": [0, 1]}}, _inner())
    with pytest.raises(ValueError, match="no unit blocks"):
        n11.edge_record({"0.00512": {k: v for k, v in f.items() if k != "units"}}, _inner())


# ---------------------------------------------------------------- end to end on synthetic data

def test_end_to_end_edge_session_then_words_from_aggregates_only(tmp_path, monkeypatch, capfd):
    """The declared call (fold 0, EDGE_LRS, swap None) through run_lrx on the fake encoder; a one-fold inner
    session for the reference; both outputs attached as notebook folders; edge_lrx reads the aggregates
    only (private/ is removed first) and writes agg_n11x_edge_words.json."""
    from knee import agg, paths
    from knee.nb import n11

    inp = tmp_path / "input"
    _make_inputs(inp)
    sh = _shared11(inp, tmp_path / "wref")
    c = n11._context(sh, 0)
    st = {"ext": {}}
    head0 = n11._ext_start(c, st, None)
    r = n11.new_run(c, c.src, head0, c.itr, 8e-5, 0, val_rows=c.iva)
    n11.run_to(c, r, n11.RUNG0)
    ref = {"source": "the fixture", "start_val_macro_auc": {"fold_0": st["ext"]["start"]["val_macro_auc"]},
           "replay": {"fold": 0, "lr": 8e-5, "val_macro_auc": [n11._r(h["val_macro_auc"]) for h in r.history],
                      "train_loss": [n11._r(h["train_loss"]) for h in r.history]},
           "end_val_macro_auc": {}}
    monkeypatch.setattr(agg, "repo_commit", lambda: INNER)
    args = {"root": inp, "val_table": TABLE, "val_range": None, "passes": PASSES, "encoder_factory": fake_factory,
            "quiet": True, "budget_h": 1.0, "devices": ["cpu"], "reference": ref}
    att = tmp_path / "attached"
    for session, rates in (("inner", n11.EXT_LRS), ("edge", n11.EDGE_LRS)):
        work = tmp_path / f"w_{session}"
        monkeypatch.setattr(paths, "WORKING", work)
        s = n11.run_lrx(session, folds=(0,), rates=rates, swap=None, **args)
        assert s["status"] == "ok" and s["runs_missing"] == [] and s["swapped"] == []
        assert len(s["processes"]) == 2 and s["processes"][1].endswith("; the replay first")
        assert s["replay"]["passed"] and s["start_check"][n11._lr_key(rates[1])]["fold_0"]["passed"]
        out = att / f"knee-n11x-{session}"
        out.mkdir(parents=True)
        for p in work.glob("agg_*.json"):
            shutil.copy(p, out / p.name)
    assert sorted(p.name for p in (att / "knee-n11x-edge").iterdir()) == [
        "agg_n11x_edge.json", "agg_n11x_edge_0.00256.json", "agg_n11x_edge_0.00512.json"]
    inner = {k: json.loads((att / "knee-n11x-inner" / f"agg_n11x_inner_{k}.json").read_text()) for k in ("0.00032", "0.00128")}
    end = {k: v["units"][f"{DINO}_f0"]["ext"]["run"]["val_macro_auc"] for k, v in inner.items()}
    eref = {**n11.EDGE_REF, "end_val_macro_auc": {"8e-05": 0.5, **end}}
    work = tmp_path / "w_words"
    monkeypatch.setattr(paths, "WORKING", work)
    rec = n11.edge_lrx(root=att, reference=eref, allow_overrides=True)
    assert rec["status"] == "described" and rec["environment"]["differences"] == [] and rec["inner"]["constants_equal_the_files"]
    assert rec["edge_commits"] == [INNER] and rec["test_overrides"] is True and rec["replay"]["passed"]
    assert set(rec["runs"]) == {"0.00256", "0.00512"} and rec["words"] and len(rec["words"]) <= 200
    written = json.loads((work / "agg_n11x_edge_words.json").read_text())
    assert written["sha256"] == rec["sha256"] and not list(work.glob("private/n11x_edge_*"))
    edge = {k: json.loads((att / "knee-n11x-edge" / f"agg_n11x_edge_{k}.json").read_text()) for k in ("0.00256", "0.00512")}
    for k, f in edge.items():
        u = f["units"][f"{DINO}_f0"]
        assert u["session"] == "edge" and u["config"]["micro_batch_studies"] == 2 and u["ext"]["lr"] == float(k)
        assert rec["runs"][k]["val_macro_auc"] == u["ext"]["run"]["val_macro_auc"]
    # the words are those of edge_words on the same scores
    w = n11.edge_words({float(k): v for k, v in rec["scores"].items()}, {float(k): v["class"] for k, v in rec["runs"].items()})
    assert w["words"] == rec["words"]
    assert "=== KNEE SUMMARY BEGIN ===" in capfd.readouterr().out


def test_card_cells_in_the_runbook():
    from knee.nb import n11

    card = (ROOT / "notebooks" / "RUNBOOK.md").read_text().split("## Edge point N11x-edge")[1]
    gpu = ('from knee.nb import n11; n11.run_lrx("edge", folds=(0,), rates=(2.56e-3, 5.12e-3), swap=None, budget_h=2.0, '
           'expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})')
    assert gpu in card and "from knee.nb import n11; n11.edge_lrx()" in card
    assert f'COMMIT = "{INNER}"' in card
    assert tuple(sorted((2.56e-3, 5.12e-3))) == n11.EDGE_LRS and "5.12e-3" == n11._lr_text(n11.EDGE_TOP)
