"""N11x, delivery D2 (it3_lr_extension_r3.md, section 6 and code list items 11 to 14), end to end on synthetic
data (CPU, the fake encoder of tests/test_n11.py): iteration 3's two sessions and its keep record -> the
inner extension session -> the signed selection record -> the final fits at each fold's pick -> the
development step -> N10's source switch and its replay check. Tested on the way: the refusals without, or
with a changed, selection record; all five folds against the moved folds only; an invalid and an unfinished
final fit (per-fold numbers only, N10 refuses); the pick in run.json before a trained model sees a test
fold; no test-fold metric in the GPU step; a refit at 8e-5 equal to iteration 3's final fit, bit for bit;
the privacy checks of every written aggregate; the notebook entries and their card; the graph script."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn.metrics")

from tests.test_n11 import MAIN, PASSES, TABLE, _make_inputs, fake_factory  # noqa: E402

DINO = MAIN[0]
ROOT = Path(__file__).resolve().parents[1]
PICKS = {0: 3.2e-4, 1: 8e-5, 2: 1.28e-3, 3: 8e-5, 4: 3.2e-4}  # what the made-up end scores below select
MOVED = [0, 2, 4]
KW = {"val_table": TABLE, "val_range": None, "passes": PASSES, "encoder_factory": fake_factory, "quiet": True}


def _fix_run(folder: Path, commit: str) -> None:
    """A copied working folder as a real session's output: a known commit, no test overrides."""
    p = folder / "run.json"
    j = json.loads(p.read_text())
    j["commit"] = commit
    j["config"]["test_overrides"] = False
    p.write_text(json.dumps(j))


def _attach_final(root: Path, work: Path, name: str = "knee-n11x-final") -> Path:
    shutil.rmtree(root / name, ignore_errors=True)
    shutil.copytree(work, root / name)
    _fix_run(root / name, "d2c0de1")
    return root / name


def _final(inputs, work, mp, selection, **kw):
    from knee import paths
    from knee.nb import n11

    mp.setattr(paths, "WORKING", Path(work))
    return n11.run_final_lrx(selection, **{"root": inputs, "budget_h": 1.0, "devices": ["cpu"], **KW, **kw})


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Everything before the final fits, once: the inputs, iteration 3's sessions a and b for DINOv2 (8e-5
    the survivor in every fold, as it was), the inner extension session with the reference numbers taken from
    those states, an input folder holding all of them, the keep record and the signed selection record."""
    from knee import paths
    from knee.nb import n11

    base = tmp_path_factory.mktemp("n11xf")
    inputs, root = base / "input", base / "attached"
    _make_inputs(inputs)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(n11, "select_rung0", lambda scores, tie=True: {"continue": [8e-5], "tied": False, "note": "test"})
        for s, folds in (("a", (0, 1, 2)), ("b", (3, 4))):
            mp.setattr(paths, "WORKING", base / f"w{s}")
            r = n11.run(session=s, folds=folds, schedule="full", root=inputs, encoders=(DINO,), devices=["cpu"], **KW)
            assert r["status"] == "ok", r.get("errors")
    it3 = {}
    for s in ("a", "b"):
        shutil.copytree(base / f"w{s}", root / f"knee-n11-{s}")
        _fix_run(root / f"knee-n11-{s}", "095c91f")
        for p in (root / f"knee-n11-{s}").rglob("state.json"):
            b = json.loads(p.read_text())
            it3[int(b["fold"])] = b
    assert all(b["inner"]["chosen_lr"] == 8e-5 for b in it3.values())
    q = it3[0]["inner"]["rung0"]["8e-05"]["curve"]
    refs = {"source": "the fixture's iteration-3 states",
            "start_val_macro_auc": {f"fold_{k}": b["inner"]["start"]["val_macro_auc"] for k, b in it3.items()},
            "replay": {"fold": 0, "lr": 8e-5, "val_macro_auc": q["val_macro_auc"], "train_loss": q["train_loss"]},
            "end_val_macro_auc": {key: {f"fold_{k}": b["inner"]["to_the_end"][key]["val_macro_auc"] for k, b in it3.items()}
                                  for key in ("8e-05", "2e-05")}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "WORKING", base / "wx")
        s = n11.run_lrx(session="inner", root=inputs, budget_h=1.0, devices=["cpu"], reference=refs, **KW)
        assert s["status"] == "ok" and s["replay"]["max_abs_gap"] == 0.0, s
    shutil.copytree(base / "wx", root / "knee-n11x-inner")
    _fix_run(root / "knee-n11x-inner", "2f5b27a")
    end = refs["end_val_macro_auc"]["8e-05"]
    for p in (root / "knee-n11x-inner").rglob("state.json"):  # made-up end scores: folds 0, 2, 4 move
        b = json.loads(p.read_text())
        key, k = b["ext"]["lr_key"], int(b["fold"])
        up = {"0.00032": 0.02 if k in MOVED else 0.0, "0.00128": 0.03 if k == 2 else -0.01}[key]
        b["ext"]["run"]["val_macro_auc"] = round(end[f"fold_{k}"] + up, 4)
        p.write_text(json.dumps(b))
    dec = n11.decision(n11.find_units(root), n_boot=20)
    assert dec["status"] == "decided" and dec["encoders_run"] == [DINO]
    (root / "knee-n11-decide").mkdir()
    (root / "knee-n11-decide" / "agg_n11_decide.json").write_text(json.dumps(dec))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "WORKING", root / "knee-n11x-select")
        sel = n11.select_lrx(root=root, reference=refs, n_boot=20)
    assert sel["status"] == "selected" and sel["picks"] == {f"fold_{k}": v for k, v in PICKS.items()}
    assert sel["folds_moved"] == {"folds": MOVED} and sel["iteration3"]["constants_equal_the_states"] is True
    return {"base": base, "inputs": inputs, "root": root, "sel": sel, "dec": dec, "refs": refs,
            "sel_path": root / "knee-n11x-select" / "agg_n11x_select.json"}


@pytest.fixture(scope="module")
def final(chain):
    """The final-fit session as declared (all five folds), with spies on the test pass, the fits and the
    metric calls; then its output attached beside the other sessions."""
    from knee import metrics
    from knee.nb import n11

    seen = {"test": [], "fits": [], "macro": 0}
    real_test, real_new, real_macro = n11.predict_test, n11.new_run, metrics.macro

    def spy_test(c, src, head, trained, lr=None):
        seen["test"].append({"fold": c.fold, "trained": trained, "lr": lr,
                             "recorded": lr is not None and n11.chosen_recorded(c.working, c.unit, lr)})
        return real_test(c, src, head, trained, lr)

    def spy_new(c, src, head_state, rows, lr_enc, seed, **kw):
        seen["fits"].append({"fold": c.fold, "lr": lr_enc, "seed": seed, "rows": np.asarray(rows).copy(),
                             "with_encoder": kw.get("with_encoder", True), "val": kw.get("val_rows") is not None,
                             "stage_a_head": all(bool((head_state[k] == c.head_state[k]).all()) for k in head_state)})
        return real_new(c, src, head_state, rows, lr_enc, seed, **kw)

    def spy_macro(*a, **k):
        seen["macro"] += 1
        return real_macro(*a, **k)

    work = chain["base"] / "wf"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(n11, "predict_test", spy_test)
        mp.setattr(n11, "new_run", spy_new)
        mp.setattr(metrics, "macro", spy_macro)
        for name in ("_wmacro", "_inner", "_final", "inner_refit", "_ext_start"):
            mp.setattr(n11, name, lambda *a, **k: pytest.fail("not part of the final fits of the extension"))
        mp.setattr(metrics, "macro_available", lambda *a, **k: pytest.fail("no validation score in a final fit"))
        s = _final(chain["inputs"], work, mp, chain["sel_path"])
    _attach_final(chain["root"], work)
    return {"s": s, "work": work, "seen": seen}


def _states(work):
    return {int(p.parent.name.rsplit("_f", 1)[1]): json.loads(p.read_text())
            for p in sorted((Path(work) / "private" / "n11x_final").glob("*_f*/state.json"))}


# ---------------------------------------------------------------- the gate

def test_final_fits_are_refused_without_or_with_a_changed_selection_record(chain, tmp_path, monkeypatch):
    from knee.nb import n11

    sel, inputs = chain["sel"], chain["inputs"]
    monkeypatch.setattr(n11, "localise_caches", lambda *a, **k: pytest.fail("refused before anything runs"))
    monkeypatch.setattr(n11, "_shared", lambda *a, **k: pytest.fail("refused before anything runs"))

    def refused(selection, exc=ValueError, **kw):
        with pytest.raises(exc):
            _final(inputs, tmp_path / "w", monkeypatch, selection, **kw)

    refused(None)  # no agg_n11x_select.json among the inputs
    tampered = copy.deepcopy(sel)
    tampered["picks"]["fold_1"] = 3.2e-4  # a pick changed after the record was written
    refused(tampered)
    refused(json.dumps(tampered))
    refused({k: v for k, v in sel.items() if k != "sha256"})
    resign = lambda rec: n11._signed({k: v for k, v in rec.items() if k != "sha256"})  # noqa: E731
    refused(resign({**sel, "status": "inconclusive"}))  # a signed record that selected nothing
    refused(resign({**sel, "smoke": True}))
    refused(resign({**sel, "picks": {**sel["picks"], "fold_0": 1.6e-4}}))  # a refinement point is never a pick
    refused(resign({**sel, "picks": {f"fold_{k}": 8e-5 for k in range(5)}}))  # no fold moved: nothing is refit
    refused({"status": "selected"})
    # the record is good: what else is refused before the step starts
    refused(sel, folds=(0, 1))  # neither all five nor exactly the moved folds
    refused(sel, folds=())
    refused(sel, budget_h=3.9)  # above the bound of five final fits plus 0.1
    refused(sel, folds=MOVED, budget_h=2.8)
    refused(sel, budget_h=None)
    refused(sel, budget_h=0)
    refused(sel, encoders=(MAIN[1],))
    with pytest.raises(TypeError):  # budget_h has no default
        n11.run_final_lrx(sel, root=inputs)
    assert not list((tmp_path / "w").glob("*.json")) and not (tmp_path / "w" / "private").exists()
    # two processes need two GPUs: refused inside the step, before the cache copy (devices is the tests' argument)
    with pytest.raises(Exception) as ei:
        _final(inputs, tmp_path / "w2", monkeypatch, sel, devices=None)
    assert "AssertionError" in str(ei.value) or isinstance(ei.value, AssertionError)
    assert not (tmp_path / "w2" / "agg_n11x_final.json").exists()
    assert n11._selection(chain["sel_path"]) == sel and n11._selection(json.dumps(sel)) == sel
    assert n11._selection(None, chain["root"]) == sel  # the attached file
    assert n11.selection_picks(sel) == PICKS and n11._selection_moved(sel) == MOVED
    assert n11.run_final_extension is n11.run_final_lrx and n11.dev_extension is n11.dev_lrx


# ---------------------------------------------------------------- the final fits

def test_final_session_fits_each_fold_at_its_pick(chain, final):
    from knee import agg, stage
    from knee.nb import n11

    s, work, seen, sel = final["s"], final["work"], final["seen"], chain["sel"]
    assert s["status"] == "ok" and s["errors"] == {} and s["units_missing"] == [] and s["fits_invalid"] == []
    assert s["processes"] == [f"{DINO} on cpu: folds 0, 1, 4", f"{DINO} on cpu: folds 2, 3"]  # section 6's assignment
    assert s["refit_model"]["complete"] is True and s["all_five_folds"] is True and s["out_of_time"] is False
    assert s["selection"]["sha256"] == sel["sha256"] and s["selection"]["folds_moved"] == {"folds": MOVED}
    d = stage.StageData(chain["inputs"])
    sts = _states(work)
    assert sorted(sts) == [0, 1, 2, 3, 4]
    for k, b in sts.items():
        ud = work / "private" / "n11x_final" / f"{DINO}_f{k}"
        x, fin = b["ext_final"], b["final"]
        fit = fin["b_lite"]["seed_0"]
        assert x["selection_sha256"] == sel["sha256"] and x["pick"] == PICKS[k] and x["moved"] is (k in MOVED)
        assert b["session"] == "final" and b["schedule"] == "full" and b["frozen0"]["valid"] is True
        assert b["frozen0"]["check_b"]["passed"] is True and b["leakage_assertion_passed"] is True
        assert fin["complete"] is True and fin["valid"] is True and "head_only" not in fin and "inner" not in b
        assert fit["encoder_lr"] == PICKS[k] and fit["status"] == "finished" and fit["valid"] is True and fit["invalid"] is None
        assert set(fit["curve"]) == {"train_loss", "mean_abs_z"} and len(fit["curve"]["train_loss"]) == 12  # no validation
        pt = fit["per_twelfth"]
        assert all(len(pt[n]) == 12 for n in pt) and not any(pt["non_finite"]) and fit["skipped_step_share"] == 0.0
        assert fit["config"]["seed"] == 0 and fit["config"]["epochs"] == n11.schedule_epochs(len(d.split(k)[0]), "full", PASSES)
        names = sorted(p.name for p in ud.iterdir())
        assert names == ["b_lite_seed_0.pt", "pred_b_lite.csv", "pred_frozen0.csv", "state.json"]
        assert set(b["files_sha16"]) == set(names) - {"state.json"}
        assert b["files_sha16"]["b_lite_seed_0.pt"] == n11.sha16(ud / "b_lite_seed_0.pt")
        ck = torch.load(ud / "b_lite_seed_0.pt")
        assert set(ck) == {"tail", "head", "encoder", "fold", "tail_prefixes"} and ck["fold"] == k and ck["encoder"] == DINO
        import pandas as pd

        df = pd.read_csv(ud / "pred_b_lite.csv").astype({"StudyInstanceUID": str})
        assert df["StudyInstanceUID"].tolist() == d.uids[d.split(k)[2]].tolist()  # the fold's scored test studies
    # one fit per fold: b-lite, seed 0, the fold's pick, Stage A's head, all outer training rows, no validation rows
    assert [(f["fold"], f["lr"]) for f in seen["fits"]] == [(k, PICKS[k]) for k in (0, 1, 4, 2, 3)]
    for f in seen["fits"]:
        assert f["with_encoder"] and f["seed"] == 0 and f["stage_a_head"] and not f["val"]
        assert sorted(f["rows"].tolist()) == sorted(d.split(f["fold"])[0].tolist())
    # the pick stands in run.json before a trained model sees the test fold; frozen0 came first
    trained = [t for t in seen["test"] if t["trained"]]
    assert [(t["fold"], t["lr"], t["recorded"]) for t in trained] == [(k, PICKS[k], True) for k in (0, 1, 4, 2, 3)]
    assert [t["fold"] for t in seen["test"]][:3] == [0, 1, 4] and not any(t["trained"] for t in seen["test"][:3])
    run = json.loads((work / "run.json").read_text())
    rec = run["n11_chosen_encoder_lr"]
    assert {u: v["lr"] for u, v in rec.items()} == {f"{DINO}_f{k}": PICKS[k] for k in range(5)}
    assert all(v["selection_sha256"] == sel["sha256"] and v["before_any_test_pass_of_a_trained_model"] for v in rec.values())
    assert run["config"]["selection_sha256"] == sel["sha256"] and run["config"]["test_overrides"] is True
    # no test-fold metric of a trained model: the only metric calls are frozen0's check (b), two per fold
    assert seen["macro"] == 2 * 5
    a = json.loads((work / "agg_n11x_final.json").read_text())

    def keys(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield f"{path}/{k}"
                yield from keys(v, f"{path}/{k}")

    assert not [k for k in keys(a) if "auc" in k.lower() and "/frozen0/check_b/" not in k]
    assert a["overview"][f"{DINO}_f2"] == {**a["overview"][f"{DINO}_f2"], "pick": 1.28e-3, "moved": True, "status": "finished",
                                           "valid": True, "final_complete": True, "frozen0_valid": True}
    # aggregates only, and iteration 3's reader does not see the folder
    uids = d.uids.tolist()[:60]
    files = sorted(p.name for p in work.glob("*.json"))
    assert files == ["agg_n11x_final.json", "run.json"]
    for p in [*work.glob("*.json"), *work.rglob("state.json")]:
        agg._check(json.loads(p.read_text()))
        assert not any(u in p.read_text() for u in uids) and "write_error" not in p.read_text()
    assert n11.find_units(work) == {} and n11.find_ext_units(work, ("inner", "final")) == {}
    assert sorted(n11.find_final_units(work)) == [0, 1, 2, 3, 4]


def test_a_refit_at_8e5_is_iteration3s_final_fit_bit_for_bit(chain, final):
    """Folds 1 and 3 keep 8e-5: the extension's final fit and iteration 3's `_final` give the same weights
    and the same test-fold predictions on the CPU (the same loop, seed, head and rows)."""
    import pandas as pd

    for k in (1, 3):
        new = final["work"] / "private" / "n11x_final" / f"{DINO}_f{k}"
        old = next(chain["root"].glob(f"knee-n11-*/private/n11_*/{DINO}_f{k}"))
        a, b = torch.load(new / "b_lite_seed_0.pt"), torch.load(old / "b_lite_seed_0.pt")
        assert all(torch.equal(a["tail"][n], b["tail"][n]) for n in b["tail"]) and set(a["tail"]) == set(b["tail"])
        assert all(torch.equal(a["head"][n], b["head"][n]) for n in b["head"])
        for name in ("pred_b_lite.csv", "pred_frozen0.csv"):
            assert pd.read_csv(new / name).equals(pd.read_csv(old / name)), name
        fa = json.loads((new / "state.json").read_text())["final"]["b_lite"]["seed_0"]
        fb = json.loads((old / "state.json").read_text())["final"]["b_lite"]["seed_0"]
        assert fa["curve"] == fb["curve"] and fa["config"] == fb["config"] and fa["encoder_lr"] == fb["encoder_lr"] == 8e-5
    # a moved fold is another model
    new = torch.load(final["work"] / "private" / "n11x_final" / f"{DINO}_f0" / "b_lite_seed_0.pt")
    old = torch.load(next(chain["root"].glob(f"knee-n11-*/private/n11_*/{DINO}_f0")) / "b_lite_seed_0.pt")
    assert not all(torch.equal(new["tail"][n], old["tail"][n]) for n in old["tail"])


# ---------------------------------------------------------------- the development step

def _dev(chain, mp, out, **kw):
    from knee import paths
    from knee.nb import n11

    mp.setattr(paths, "WORKING", Path(out))
    return n11.dev_lrx(**{"root": chain["root"], "n_boot": 20, "expect_decision": chain["dec"]["sha256"], **kw})


def test_dev_numbers_of_the_refit_model(chain, final, tmp_path, monkeypatch, capfd):
    from knee import agg, paths
    from knee.nb import n11

    s = _dev(chain, monkeypatch, tmp_path / "out")
    assert "STEP n11x_dev: OK" in capfd.readouterr().out
    assert s["status"] == "ok" and s["label"] == n11.DEV_LABEL + "; rate chosen after iteration 3's results were read"
    assert s["label_parts"] == [n11.DEV_LABEL, "rate chosen after iteration 3's results were read"]
    assert s["decision"]["sha256"] == chain["dec"]["sha256"] and s["selection"]["sha256"] == chain["sel"]["sha256"]
    assert s["folds_refit"] == {"folds": [0, 1, 2, 3, 4]} and s["report_final_model"]["model"] == "refit"
    assert s["report_final_model"]["picks_text"] == "f0 3.2e-4, f1 8e-5, f2 1.28e-3, f3 8e-5, f4 3.2e-4"
    assert s["final_fits"] == {"knee_commits": ["d2c0de1"], "one_known_commit": True,
                               "environment_differences_from_iteration3": [], "environment_label": None}
    # iteration 3's arms are dev()'s numbers: the same files, folds, draws and seed
    monkeypatch.setattr(paths, "WORKING", tmp_path / "out3")
    d3 = n11.dev(chain["dec"], root=chain["root"], n_boot=20, side_auc=False)["encoders"][DINO]
    m = s["model"]
    assert m["status"] == "ok" and m["label"] == s["label"]
    assert m["frozen0"] == d3["frozen0"] and m["head_only_seed_0"] == d3["head_only_seed_0"] and m["b_lite_iteration3"] == d3["b_lite"]
    for key in ("b_lite", "b_lite_minus_frozen0", "b_lite_minus_head_only", "b_lite_minus_iteration3_b_lite"):
        assert set(m[key]) >= {"mean_of_fold_macros", "fold_stratified_macro_auc", "fold_stratified_interval"}
        ci = m[key]["fold_stratified_interval"]
        assert ci["lo"] <= m[key]["fold_stratified_macro_auc"] <= ci["hi"]
    assert m["b_lite_minus_iteration3_b_lite"]["mean_of_fold_macros"] == pytest.approx(
        m["b_lite"]["mean_of_fold_macros"] - m["b_lite_iteration3"]["mean_of_fold_macros"], abs=2e-4)
    assert m["b_lite"] != m["b_lite_iteration3"]  # three folds hold another model
    # per fold; and the two 8e-5 final fits of an unmoved fold: on the CPU the same model, so the gap is 0
    for k in range(5):
        f = s["per_fold"][f"fold_{k}"]
        assert f["pick"] == PICKS[k] and f["moved"] is (k in MOVED) and f["iteration3_lr"] == 8e-5 and f["final_fit"] == "finished"
        assert f["label"] == s["label"] and f["frozen0"] is not None and f["b_lite"] is not None
        assert f["b_lite_minus_iteration3_b_lite"]["macro_auc"] == pytest.approx(f["b_lite"] - f["b_lite_iteration3"], abs=2e-4)
        assert set(f["b_lite_minus_frozen0"]["interval"]) == {"lo", "hi", "valid_share"}
        assert ("same_seed_noise_of_a_final_fit" in f) is (k not in MOVED)
    for k in (1, 3):
        g = s["per_fold"][f"fold_{k}"]["same_seed_noise_of_a_final_fit"]
        assert g["what"].startswith("same-seed noise of a final fit") and g["encoder_lr"] == 8e-5
        assert (g["abs_macro_auc_gap"], g["max_abs_probability_gap"], g["mean_abs_probability_gap"]) == (0.0, 0.0, 0.0)
    assert s["per_fold"]["fold_0"]["b_lite"] != s["per_fold"]["fold_0"]["b_lite_iteration3"]
    assert s["same_seed_noise"]["folds"] == [1, 3] and s["same_seed_noise"]["max_abs_macro_auc_gap"] == 0.0
    out = json.loads((tmp_path / "out" / "agg_n11x_dev.json").read_text())
    agg._check(out)
    text = (tmp_path / "out" / "agg_n11x_dev.json").read_text()
    assert not agg.UID_RE.search(text) and sorted(p.name for p in (tmp_path / "out").iterdir() if p.is_file()) == [
        "agg_n11x_dev.json", "run.json"]
    # the gates: the keep record (and which one), the selection record, the files
    root = chain["root"]
    with pytest.raises(ValueError):  # not the keep record of iteration 3 (the default expects 0a8cce4c...)
        n11.dev_lrx(root=root, n_boot=5)
    assert n11.EXT_KEEP_SHA256.startswith("0a8cce4c") and len(n11.EXT_KEEP_SHA256) == 64
    with pytest.raises(ValueError):
        _dev(chain, monkeypatch, tmp_path / "o2", decision_record={**chain["dec"], "status": "env_mismatch"})
    bad = copy.deepcopy(chain["sel"])
    bad["picks"]["fold_1"] = 3.2e-4
    with pytest.raises(ValueError):
        _dev(chain, monkeypatch, tmp_path / "o2", selection=bad)
    other = n11._signed({**{k: v for k, v in chain["sel"].items() if k != "sha256"}, "written_at_utc": "2026-11-25T00:00:00Z"})
    with pytest.raises(Exception):  # another record than the fits ran under
        _dev(chain, monkeypatch, tmp_path / "o2", selection=other)
    p = root / "knee-n11x-final" / "private" / "n11x_final" / f"{DINO}_f3" / "pred_b_lite.csv"
    good = p.read_text()
    p.write_text(good.replace("0.", "0.0", 1))
    try:
        with pytest.raises(Exception):  # a prediction file that is not the one its state records
            _dev(chain, monkeypatch, tmp_path / "o2")
    finally:
        p.write_text(good)
    assert not (tmp_path / "o2" / "agg_n11x_dev.json").exists()
    # dev() itself is as it was: it reads iteration 3's units only
    assert set(d3) >= {"frozen0", "head_only_seed_0", "b_lite", "head_only_seed_values"}


# ---------------------------------------------------------------- N10: the source switch on the real final fits

def _n10_root(chain, tmp_path, final_from: Path | None = None) -> Path:
    """The inputs of knee-n10-tails-check-x: the bundle, the N4 caches, knee-n6a, knee-n11x-final and
    knee-n11x-select. Nothing of iteration 3."""
    root = tmp_path / "n10in"
    for name in ("n4-sagittal", "n4-coronal", "n4-axial", "n6a"):
        shutil.copytree(chain["inputs"] / name, root / f"knee-{name}")
    shutil.copytree(final_from or chain["root"] / "knee-n11x-final", root / "knee-n11x-final")
    shutil.copytree(chain["root"] / "knee-n11x-select", root / "knee-n11x-select")
    ck = root / "knee-bundle" / "private" / "bundle" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / f"{DINO}.pth").write_bytes(b"not read: the test passes its own encoder")
    sha = hashlib.sha256((ck / f"{DINO}.pth").read_bytes()).hexdigest()
    (ck.parent / "manifest.json").write_text(json.dumps({"commit": "abcdef1234567890", "files_sha256": {f"{DINO}.pth": sha}}))
    return root


def test_n10_reads_the_refit_model_and_replays_a_moved_fold(chain, final, tmp_path, monkeypatch):
    pytest.importorskip("timm")
    from knee import agg, paths
    from knee.nb import n10

    root = _n10_root(chain, tmp_path)
    t = n10.tail_units(DINO, root, "n11x")
    assert sorted(t["units"]) == [0, 1, 2, 3, 4] and t["commit"] == "d2c0de1" and t["schedule"] == "full" and t["source"] == "n11x"
    assert [t["units"][k]["encoder_lr"] for k in range(5)] == [PICKS[k] for k in range(5)]
    assert t["selection"]["sha256"] == chain["sel"]["sha256"] and t["lr_rule"]["selection_sha256"] == chain["sel"]["sha256"]
    assert all(u["session"] == "final" and u["dir"].parent.name == "n11x_final" for u in t["units"].values())
    with pytest.raises(n10.TailError):  # iteration 3's reader finds nothing here: the two sources are not mixed
        n10.tail_units(DINO, root)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "wc")
    s = n10.check_tails(DINO, fold=None, n=200, root=root, encoder_factory=fake_factory, tails="n11x")
    assert s["passed"] is True and s["fold"] == 0 and s["source"] == "n11x" and s["encoder_lr"] == 3.2e-4  # the lowest moved fold
    assert s["max_abs_probability_gap"] < 1e-5 and s["selection_sha256"] == chain["sel"]["sha256"] and s["n11_commit"] == "d2c0de1"
    out = json.loads((tmp_path / "wc" / "agg_n10_tails_check.json").read_text())
    agg._check(out)
    assert json.loads((tmp_path / "wc" / "run.json").read_text())["config"]["tails"] == "n11x"
    s = n10.check_tails(DINO, fold=3, n=200, root=root, encoder_factory=fake_factory, tails="n11x")  # an unmoved fold too
    assert s["passed"] is True and s["fold"] == 3 and s["encoder_lr"] == 8e-5
    enc = fake_factory(DINO)
    P, info = n10.predict_tails(enc, t, {p: np.zeros((2, 4, 16, 16), np.uint8) for p in ("sagittal", "coronal", "axial")},
                                np.arange(2), np.array([-1, 0]))
    assert P.shape == (5, 2, 12) and np.isfinite(P).all() and len({v["tail_sha16"] for v in info["per_fold"].values()}) == 5
    # a later version of the selection record (another sha256) is refused
    p = root / "knee-n11x-select" / "agg_n11x_select.json"
    from knee.nb import n11

    p.write_text(json.dumps(n11._signed({**{k: v for k, v in chain["sel"].items() if k != "sha256"}, "written_at_utc": "later"})))
    with pytest.raises(n10.TailError, match="attached selection record"):
        n10.tail_units(DINO, root, "n11x")


# ---------------------------------------------------------------- the moved folds only (the quota fallback)

def test_moved_folds_only_gives_per_fold_numbers_and_no_n10(chain, final, tmp_path, monkeypatch):
    pytest.importorskip("timm")
    from knee import agg
    from knee.nb import n10

    s = _final(chain["inputs"], tmp_path / "wm", monkeypatch, chain["sel"], folds=MOVED, budget_h=2.7)
    assert s["status"] == "ok" and s["folds"] == MOVED and s["all_five_folds"] is False
    assert s["processes"] == [f"{DINO} on cpu: folds 0, 4", f"{DINO} on cpu: folds 2"]
    assert s["refit_model"] == {"complete": False, "note": "the refit model is incomplete: per-fold development numbers only, "
                                                           "no N10 preparation"}
    assert sorted(_states(tmp_path / "wm")) == MOVED and s["units_missing"] == []
    for k in MOVED:  # the same fits as in the five-fold session
        a = torch.load(tmp_path / "wm" / "private" / "n11x_final" / f"{DINO}_f{k}" / "b_lite_seed_0.pt")
        b = torch.load(final["work"] / "private" / "n11x_final" / f"{DINO}_f{k}" / "b_lite_seed_0.pt")
        assert all(torch.equal(a["tail"][n], b["tail"][n]) for n in b["tail"])
    full = chain["root"] / "knee-n11x-final"
    keep = tmp_path / "kept"
    shutil.move(str(full), str(keep))
    try:
        _attach_final(chain["root"], tmp_path / "wm")
        d = _dev(chain, monkeypatch, tmp_path / "out")
        with pytest.raises(n10.TailError, match="exactly folds 0-4"):
            n10.tail_units(DINO, _n10_root(chain, tmp_path), "n11x")
    finally:
        shutil.rmtree(full, ignore_errors=True)
        shutil.move(str(keep), str(full))
    assert d["status"] == "ok (per-fold numbers only: the refit model is incomplete)"
    assert d["model"]["status"] == "per-fold numbers only" and "mean_of_fold_macros" not in json.dumps(d["model"])
    assert d["model"]["why"] == "fold 1: not refit; fold 3: not refit" and d["folds_refit"] == {"folds": MOVED}
    assert d["report_final_model"]["model"] == "iteration3_8e-5"
    assert d["report_final_model"]["sentence"] == "the picks were not applied to a full model: fewer than five refit folds"
    for k in range(5):
        f = d["per_fold"][f"fold_{k}"]
        assert ("b_lite" in f) is (k in MOVED) and f["final_fit"] == ("finished" if k in MOVED else "not refit")
    assert d["same_seed_noise"]["folds"] == [] and d["same_seed_noise"]["max_abs_macro_auc_gap"] is None
    agg._check(json.loads((tmp_path / "out" / "agg_n11x_dev.json").read_text()))


# ---------------------------------------------------------------- invalid and unfinished final fits

def test_an_invalid_final_fit_is_reported_not_repeated_and_blocks_the_model(chain, final, tmp_path, monkeypatch, capfd):
    pytest.importorskip("timm")
    from knee import agg
    from knee.nb import n10, n11

    real = n11.run_to
    fits = []

    def run_to(c, r, twelfth, limit=None):
        real(c, r, twelfth, limit)
        if c.fold == 2 and twelfth == 4:
            r.history[-1]["train_loss"] = float("nan")
        if c.fold == 4 and twelfth == 6:
            r.skipped = int(0.05 * r.step) + 1  # just above the limit
        if twelfth == 1:
            fits.append(c.fold)

    monkeypatch.setattr(n11, "run_to", run_to)
    s = _final(chain["inputs"], tmp_path / "wi", monkeypatch, chain["sel"])
    assert "STEP n11x_final: OK" in capfd.readouterr().out
    assert s["status"] == "partial" and s["errors"] == {} and s["out_of_time"] is False
    assert s["fits_invalid"] == [f"{DINO}_f2", f"{DINO}_f4"] == s["units_missing"] and s["fits_unfinished"] == []
    assert s["units_complete"] == [f"{DINO}_f{k}" for k in (0, 1, 3)] and s["refit_model"]["complete"] is False
    assert sorted(fits) == [0, 1, 2, 3, 4]  # each fit started once: an invalid fit is not repeated
    ov = s["overview"]
    assert ov[f"{DINO}_f2"]["invalid"] == {"k": 4, "reason": "non-finite train loss"} and ov[f"{DINO}_f2"]["status"] == "invalid"
    assert ov[f"{DINO}_f4"]["invalid"] == {"k": 6, "reason": "skipped optimiser steps above the limit"}
    sts = _states(tmp_path / "wi")
    for k, n in ((2, 4), (4, 6)):
        fin = sts[k]["final"]
        fit = fin["b_lite"]["seed_0"]
        assert fin["complete"] is False and fin["valid"] is False and fit["valid"] is False and fit["finished"] is False
        assert len(fit["per_twelfth"]["non_finite"]) == n and fit["encoder_lr"] == PICKS[k]
        ud = tmp_path / "wi" / "private" / "n11x_final" / f"{DINO}_f{k}"
        assert sorted(p.name for p in ud.iterdir()) == ["pred_frozen0.csv", "state.json"]  # no trained model saw the test fold
    assert sts[2]["final"]["b_lite"]["seed_0"]["per_twelfth"]["non_finite"] == [False, False, False, True]
    assert sts[4]["final"]["b_lite"]["seed_0"]["per_twelfth"]["skipped_step_share"][-1] > 0.05
    for p in [*(tmp_path / "wi").glob("*.json"), *(tmp_path / "wi").rglob("state.json")]:
        agg._check(json.loads(p.read_text()))
        assert "NaN" not in p.read_text()
    monkeypatch.undo()
    full = chain["root"] / "knee-n11x-final"
    keep = tmp_path / "kept"
    shutil.move(str(full), str(keep))
    try:
        _attach_final(chain["root"], tmp_path / "wi")
        d = _dev(chain, monkeypatch, tmp_path / "out")
        with pytest.raises(n10.TailError, match="final fit of fold 2 is not finished and valid"):
            n10.tail_units(DINO, _n10_root(chain, tmp_path), "n11x")
    finally:
        shutil.rmtree(full, ignore_errors=True)
        shutil.move(str(keep), str(full))
    assert d["status"].startswith("ok (per-fold numbers only") and d["model"]["status"] == "per-fold numbers only"
    assert d["model"]["why"] == "fold 2: invalid; fold 4: invalid" and d["folds_refit"] == {"folds": [0, 1, 3]}
    assert d["report_final_model"] == {**d["report_final_model"], "model": "iteration3_8e-5",
                                       "sentence": "the picks were not applied to a full model: a final fit is invalid or unfinished"}
    assert "b_lite" not in d["per_fold"]["fold_2"] and d["per_fold"]["fold_2"]["final_fit"] == "invalid"
    assert d["per_fold"]["fold_0"]["b_lite"] is not None and d["same_seed_noise"]["folds"] == [1, 3]
    agg._check(json.loads((tmp_path / "out" / "agg_n11x_dev.json").read_text()))


def test_soft_stop_ends_partial_and_a_failed_frozen0_check_ends_invalid(chain, tmp_path, monkeypatch, capfd):
    from knee import agg, safe_log
    from knee.nb import n11

    real = n11.run_to

    def run_to(c, r, twelfth, limit=None):
        if c.fold == 1 and twelfth == 3:
            c.task["soft_stop"] = 1.0  # the budget of this process is used up while twelfth 3 trains
        real(c, r, twelfth, limit)

    monkeypatch.setattr(n11, "run_to", run_to)
    s = _final(chain["inputs"], tmp_path / "ws", monkeypatch, chain["sel"])
    assert "STEP n11x_final: OK" in capfd.readouterr().out
    assert s["status"] == "partial" and s["out_of_time"] is True and s["fits_invalid"] == []
    assert s["units_complete"] == [f"{DINO}_f{k}" for k in (0, 2, 3)] and s["fits_unfinished"] == [f"{DINO}_f1", f"{DINO}_f4"]
    assert s["errors"][f"{DINO}_f1"].startswith("OutOfTime at")
    sts = _states(tmp_path / "ws")
    fit = sts[1]["final"]["b_lite"]["seed_0"]
    assert fit["status"] == "unfinished: out of time" and len(fit["curve"]["train_loss"]) == 3 and fit["valid"] is False
    assert "final" not in sts[4] and sts[4]["frozen0"]["valid"] is True  # frozen0 ran for every fold first
    assert s["overview"][f"{DINO}_f4"]["status"] == "not started" and s["overview"][f"{DINO}_f1"]["final_complete"] is False
    assert not list((tmp_path / "ws" / "private" / "n11x_final" / f"{DINO}_f1").glob("*.pt"))
    agg._check(json.loads((tmp_path / "ws" / "agg_n11x_final.json").read_text()))
    # check (b) of frozen0 fails: the run is invalid, the step fails, nothing is trained
    monkeypatch.undo()
    monkeypatch.setattr(n11, "CHECK_B_TOL", -1.0)
    monkeypatch.setattr(n11, "new_run", lambda *a, **k: pytest.fail("nothing may train after a failed check (b)"))
    with pytest.raises(safe_log.StepFailed):
        _final(chain["inputs"], tmp_path / "wb", monkeypatch, chain["sel"])
    a = json.loads((tmp_path / "wb" / "agg_n11x_final.json").read_text())
    assert a["status"].startswith("invalid: frozen0 check (b) failed") and (tmp_path / "wb" / "private" / "n11x_final" / "INVALID").exists()
    assert all(set(b) == {"frozen0"} and b["frozen0"]["valid"] is False for b in a["units"].values())
    assert not list((tmp_path / "wb").rglob("pred_b_lite.csv")) and not list((tmp_path / "wb").rglob("*.pt"))


def test_two_processes_share_the_step_folder_and_run_json(chain, tmp_path, monkeypatch):
    s = _final(chain["inputs"], tmp_path / "w", monkeypatch, chain["sel"], folds=MOVED, processes=True, devices=["cpu", "cpu"])
    assert s["status"] == "ok", s["errors"]
    assert s["processes"] == [f"{DINO} on cpu: folds 0, 4", f"{DINO} on cpu: folds 2"]
    assert sorted(_states(tmp_path / "w")) == MOVED
    rec = json.loads((tmp_path / "w" / "run.json").read_text())["n11_chosen_encoder_lr"]
    assert {u: v["lr"] for u, v in rec.items()} == {f"{DINO}_f{k}": PICKS[k] for k in MOVED}


# ---------------------------------------------------------------- the notebook entries and their card

def test_d2_notebook_entries_match_the_runbook_card():
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (ROOT / "notebooks" / "RUNBOOK.md").read_text().split("## Learning-rate extension N11x")[1]
    gpu = [f"u/knee-{u}" for u in ("n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n5", "n6a")]
    assert mk.cell2("n11x-final") == f"from knee.nb import n11; n11.run_final_lrx(budget_h=BUDGET_H, expect={mk.EXPECT})"
    with pytest.raises(NameError):  # the caller must type the number: the cell as generated does not run
        exec(mk.cell2("n11x-final"), {})
    meta = mk.metadata("u", "n11x-final")
    assert meta["kernel_sources"] == gpu + ["u/knee-n11x-select"] and not any("n11-" in x for x in meta["kernel_sources"])  # R9
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert mk.NOTEBOOKS["n11x-final"][4] >= 1.1 * 3.8 * 3600
    assert mk.cell2("n11x-dev") == "from knee.nb import n11; n11.dev_lrx()"
    meta = mk.metadata("u", "n11x-dev")
    assert meta["enable_gpu"] is False and meta["kernel_sources"] == [
        f"u/knee-{x}" for x in ("n11x-final", "n11x-select", "n11-a", "n11-b", "n11-decide")]
    assert mk.cell2("n10-tails-check-x") == 'from knee.nb import n10; n10.check_tails("dinov2_vits14", fold=None, n=200, tails="n11x")'
    assert mk.cell2("n10-blite-x") == 'from knee.nb import n10; n10.run("dinov2_vits14", tails="n11x")'
    for name, like in (("n10-tails-check-x", "n10-tails-check"), ("n10-blite-x", "n10-blite")):
        meta, old = mk.metadata("u", name, bundle="u/knee-bundle"), mk.metadata("u", like, bundle="u/knee-bundle")
        assert meta["enable_gpu"] is True and meta["enable_internet"] is False and name in mk.OFFLINE
        assert meta["dataset_sources"] == ["u/knee-bundle"] and meta["competition_sources"] == old["competition_sources"]
        assert meta["kernel_sources"] == [x for x in old["kernel_sources"] if "n11-" not in x] + ["u/knee-n11x-final", "u/knee-n11x-select"]
        assert mk.cell1("x", offline=name in mk.OFFLINE) == mk.CELL1_OFFLINE and mk.NOTEBOOKS[name][4] == mk.NOTEBOOKS[like][4]
    for name in ("n11x-final", "n11x-dev", "n10-tails-check-x", "n10-blite-x"):
        assert mk.cell2(name) in card, name
    for text in ("the smaller of 3.8 and (quota minus 0.4)", "the smaller of 2.7 and (quota minus 0.4)",
                 "the smaller of 1.4 and (quota minus 0.4)", "must be rebuilt first", "GPU 0: folds 0, 1, 4. GPU 1: folds 2, 3",
                 "rate chosen after iteration 3's results were read", "same-seed noise of a final fit", "scripts/plot_lrx.py"):
        assert text in card, text
    from knee.nb import n11

    assert n11.EXT_FINAL_GPUS == ((0, 1, 4), (2, 3)) and n11.EXT_FINAL_BOUND_H == {5: 3.7, 4: 2.6, 3: 2.6, 2: 1.3, 1: 1.3}
    for name in ("n11x-inner", "n11x-select", "n11-dev", "n10-blite", "n10-tails-check"):  # D1's and iteration 3's entries stay
        assert "lrx(budget_h=BUDGET_H" not in mk.cell2(name) and "n11x-final" not in mk.metadata("u", name)["kernel_sources"]


# ---------------------------------------------------------------- the graphs

def _python_with_matplotlib():
    for exe in (sys.executable, "python3", "/usr/local/bin/python3", "/usr/bin/python3"):
        try:
            if subprocess.run([exe, "-c", "import matplotlib"], capture_output=True, timeout=120).returncode == 0:
                return exe
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def test_graph_script_draws_g1_and_g2_from_the_aggregates(chain, tmp_path, monkeypatch):
    """scripts/plot_lrx.py on the aggregates of this module's sessions (synthetic): the three files are
    written. It gets aggregate JSON files only; one run is marked invalid and a refinement file is made up."""
    exe = _python_with_matplotlib()
    if exe is None:
        pytest.skip("no Python with matplotlib here")
    from knee import paths
    from knee.nb import n11

    base, script = chain["base"], ROOT / "scripts" / "plot_lrx.py"
    inner = json.loads((base / "wx" / "agg_n11x_inner.json").read_text())
    run = inner["units"]["0.00128"][f"{DINO}_f3"]["ext"]["run"]  # an invalid run, drawn up to its last twelfth
    run.update({"status": "invalid", "valid": False, "finished": False, "val_macro_auc": None,
                "invalid": {"k": 5, "reason": "non-finite val loss"},
                "curve": {k: v[:5] for k, v in run["curve"].items()}})
    (tmp_path / "agg_n11x_inner.json").write_text(json.dumps(inner))
    refine = {"session": "refine", "units": {"0.00016": copy.deepcopy(inner["units"]["0.00032"])}}
    (tmp_path / "agg_n11x_refine.json").write_text(json.dumps(refine))
    args = [exe, str(script), "--it3", str(base / "wa" / "agg_n11_a.json"), str(base / "wb" / "agg_n11_b.json"),
            "--inner", str(tmp_path / "agg_n11x_inner.json"), "--select", str(chain["sel_path"])]
    for extra, out in (([], tmp_path / "g"), (["--refine", str(tmp_path / "agg_n11x_refine.json")], tmp_path / "g_refine")):
        r = subprocess.run([*args, *extra, "--out", str(out)], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        for name in ("g1_steps_to_the_optimum.png", "g2_fold0.png", "g2_all_folds.png"):
            assert (out / name).stat().st_size > 5000, name
    # the selection of a session with an invalid run and without iteration 3 attached is drawn too
    env = json.loads((base / "wx" / "private" / "n11x_inner_0.00032" / f"{DINO}_f0" / "state.json").read_text())["environment"]
    root = tmp_path / "in"
    shutil.copytree(chain["root"] / "knee-n11x-inner", root / "knee-n11x-inner")
    p = root / "knee-n11x-inner" / "private" / "n11x_inner_0.00128" / f"{DINO}_f3" / "state.json"
    b = json.loads(p.read_text())
    b["ext"]["run"].update({"status": "invalid", "valid": False, "finished": False, "val_macro_auc": None})
    p.write_text(json.dumps(b))
    sel = n11.selection_ext(n11.find_ext_units(root), None, None, reference=chain["refs"], n_boot=0, expect_env=env)
    assert sel["rule_d"]["c2"] == "UNSTABLE" and "0.00128" not in sel["rule_d"]["means"]
    (tmp_path / "sel2.json").write_text(json.dumps(sel))
    r = subprocess.run([exe, str(script), "--it3", str(base / "wa" / "agg_n11_a.json"), str(base / "wb" / "agg_n11_b.json"),
                        "--inner", str(tmp_path / "agg_n11x_inner.json"), "--select", str(tmp_path / "sel2.json"),
                        "--out", str(tmp_path / "g2")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "g2" / "g1_steps_to_the_optimum.png").exists()
    text = script.read_text()  # matplotlib and the standard library only; it opens what it is given, nothing else
    assert "import pandas" not in text and "import numpy" not in text and "glob" not in text and ".csv" not in text
