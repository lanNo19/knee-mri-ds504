"""N11's per-finding description of the S2 test folds (n11.dev_findings), on hand-made prediction files.
Tested: dev()'s gate (no decision record, an edited record, a changed prediction file); the per-finding AUCs
against scikit-learn; the mean of the 12 findings against dev()'s macro; each fold's macro against dev()'s
mean; the summary lists; the written file against agg's checks, with a finding that has under 5 positives."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from knee.labels import FINDINGS

sk = pytest.importorskip("sklearn.metrics")
from tests.test_n11 import MAIN, _sessions, safe_log_failed  # noqa: E402 - the hand-made sessions of test_n11

ARMS = ("frozen0", "head_only_seed_0", "b_lite")
RARE, GAP, THIN = FINDINGS.index("Fracture"), FINDINGS.index("Synovitis"), FINDINGS.index("Baker's")


def _thin(root: Path) -> dict:
    """Rewrite the test-fold labels of every unit (the same change in every arm and encoder): Fracture keeps
    one positive study in all (fold 0), Synovitis has no positive in fold 1, Baker's keeps 3 positives in
    fold 2. The states' hashes follow. Returns {(encoder, fold, arm): (P, Y)}."""
    from knee.nb import n11

    truth = {}
    for (e, k), ud in n11.find_units(root).items():
        st = json.loads((ud / "state.json").read_text())
        for p in sorted(ud.glob("pred_*.csv")):
            df = pd.read_csv(p)

            def keep(j, n, df=df):
                col = f"y_{FINDINGS[j]}"
                pos = np.flatnonzero(df[col].to_numpy() == 1)
                df.loc[pos[n:], col] = 0.0

            keep(RARE, 1 if k == 0 else 0)
            if k == 1:
                keep(GAP, 0)
            if k == 2:
                keep(THIN, 3)
            df.to_csv(p, index=False)
            st["files_sha16"][p.name] = n11.sha16(p)
            truth[(e, k, p.stem[len("pred_"):])] = (df[FINDINGS].to_numpy(float),
                                                    df[[f"y_{f}" for f in FINDINGS]].to_numpy(float))
        (ud / "state.json").write_text(json.dumps(st))
    return truth


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    from knee import paths
    from knee.nb import n11

    tmp = tmp_path_factory.mktemp("n11_findings")
    root = tmp / "in"
    _sessions(root)
    truth = _thin(root)
    rec = n11.decision(n11.find_units(root), n_boot=20)
    mp = pytest.MonkeyPatch()
    mp.setattr(paths, "WORKING", tmp / "w")
    try:
        s = n11.dev_findings(rec, root=root, n_boot=40, quiet=True)
        text = (tmp / "w" / "agg_n11_dev_findings.json").read_text()
        d = n11.dev(rec, root=root, n_boot=40, side_auc=False, quiet=True)
    finally:
        mp.undo()
    return SimpleNamespace(root=root, truth=truth, rec=rec, s=s, text=text, dev=d)



def test_gate_is_devs(tmp_path, monkeypatch):
    from knee import paths
    from knee.nb import n11

    root = tmp_path / "in"
    _sessions(root)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    with pytest.raises(ValueError):  # no decision attached
        n11.dev_findings(root=root, quiet=True)
    rec = n11.decision(n11.find_units(root), n_boot=20)
    for bad in ({**rec, "status": "env_mismatch"}, {k: v for k, v in rec.items() if k != "status"},
                {**rec, "encoders": {e: {**v, "keep": not v["keep"]} for e, v in rec["encoders"].items()}}):
        with pytest.raises(ValueError):  # no decision, or a record edited after it was written
            n11.dev_findings(bad, root=root, quiet=True)
    assert not (tmp_path / "w" / "agg_n11_dev_findings.json").exists()
    (root / "knee-n11-decide").mkdir()
    (root / "knee-n11-decide" / "agg_n11_decide.json").write_text(json.dumps(rec))
    s = n11.dev_findings(root=root, n_boot=10, quiet=True)  # the attached decision file is found
    assert s["status"] == "ok" and s["decision"]["sha256"] == rec["sha256"]
    (tmp_path / "w" / "agg_n11_dev_findings.json").unlink()
    # a prediction file that is not the one the record lists is refused, and nothing is written
    p = n11.find_units(root)[(MAIN[1], 3)] / "pred_frozen0.csv"
    p.write_text(p.read_text().replace("t3_0,", "t3_x,"))
    with pytest.raises(safe_log_failed()):
        n11.dev_findings(rec, root=root, quiet=True)
    assert not (tmp_path / "w" / "agg_n11_dev_findings.json").exists()


def test_per_finding_aucs_equal_sklearn(made):
    for e in MAIN:
        F = made.s["encoders"][e]["findings"]
        assert list(F) == FINDINGS
        for arm in ARMS:
            parts = [made.truth[(e, k, arm)] for k in range(5)]
            for j, name in enumerate(FINDINGS):
                num = den = 0.0
                per_fold, used = [], []
                for k, (P, Y) in enumerate(parts):
                    pos, neg = (Y[:, j] == 1).sum(), (Y[:, j] == 0).sum()
                    if pos == 0 or neg == 0:
                        continue
                    a = sk.roc_auc_score(Y[:, j], P[:, j])
                    num, den = num + a * pos * neg, den + pos * neg
                    per_fold.append(a)
                    used.append(k)
                assert F[name][arm]["fold_stratified_auc"] == pytest.approx(num / den, abs=6e-5), (e, arm, name)
                assert F[name][arm]["mean_of_fold_aucs"] == pytest.approx(np.mean(per_fold), abs=6e-5)
                assert F[name]["folds"] == used and F[name]["n_folds"] == len(used)
        assert F[FINDINGS[RARE]]["folds"] == [0] and F[FINDINGS[GAP]]["folds"] == [0, 2, 3, 4]
        assert F[FINDINGS[THIN]]["n_folds"] == 5
        for name in FINDINGS:
            for key, base in (("b_lite_minus_frozen0", "frozen0"), ("b_lite_minus_head_only", "head_only_seed_0")):
                for m in ("fold_stratified_auc", "mean_of_fold_aucs"):
                    assert F[name][key][m] == pytest.approx(F[name]["b_lite"][m] - F[name][base][m], abs=2e-4)
                ci = F[name][key]["fold_stratified_interval"]
                side = F[name][key]["interval_side"]
                assert side == ("above_0" if ci["lo"] > 0 else "below_0" if ci["hi"] < 0 else "includes_0")
                if name != FINDINGS[RARE]:  # one positive study: a draw without it has no pair
                    assert ci["valid_share"] == 1.0 and ci["lo"] <= F[name][key]["fold_stratified_auc"] <= ci["hi"]
        assert F[FINDINGS[RARE]]["b_lite_minus_frozen0"]["fold_stratified_interval"]["valid_share"] < 1.0


def test_macro_of_the_findings_and_fold_macros_equal_dev(made):
    for e in MAIN:
        r, d = made.s["encoders"][e], made.dev["encoders"][e]
        for arm in ARMS:
            vals = [r["findings"][f][arm]["fold_stratified_auc"] for f in FINDINGS]
            assert np.mean(vals) == pytest.approx(d[arm]["fold_stratified_macro_auc"], abs=6e-5)
        pf = r["per_fold_macro"]
        assert [k for k in pf if k.startswith("fold_")] == [f"fold_{k}" for k in range(5)]
        for k in range(5):
            for arm in ARMS:
                P, Y = made.truth[(e, k, arm)]
                both = all((Y[:, j] == 1).any() and (Y[:, j] == 0).any() for j in range(12))
                ref = np.mean([sk.roc_auc_score(Y[:, j], P[:, j]) for j in range(12)]) if both else None
                assert pf[f"fold_{k}"][arm] == (pytest.approx(ref, abs=6e-5) if both else None)
        assert pf["fold_0"]["b_lite_minus_frozen0"] == pytest.approx(pf["fold_0"]["b_lite"] - pf["fold_0"]["frozen0"], abs=2e-4)
        assert pf["fold_1"]["b_lite_minus_frozen0"] is None  # Fracture has no positive study outside fold 0
        for arm in ARMS:  # dev() leaves out the folds that have no macro
            assert pf["fold_0"][arm] == pytest.approx(d[arm]["mean_of_fold_macros"], abs=6e-5)


def test_fold_macros_on_complete_folds_average_to_devs_mean(tmp_path, monkeypatch):
    from knee import paths
    from knee.nb import n11

    root = tmp_path / "in"
    _sessions(root, encs=(MAIN[1],))
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    rec = n11.decision(n11.find_units(root), n_boot=20)
    s = n11.dev_findings(rec, root=root, n_boot=10, quiet=True)
    d = n11.dev(rec, root=root, n_boot=10, side_auc=False, quiet=True)
    assert set(s["encoders"]) == {MAIN[1]} and s["status"] == "ok"
    pf = s["encoders"][MAIN[1]]["per_fold_macro"]
    for arm in ARMS:
        assert np.mean([pf[f"fold_{k}"][arm] for k in range(5)]) == pytest.approx(d["encoders"][MAIN[1]][arm]["mean_of_fold_macros"], abs=6e-5)
    assert np.mean([pf[f"fold_{k}"]["b_lite_minus_frozen0"] for k in range(5)]) == pytest.approx(
        d["encoders"][MAIN[1]]["b_lite_minus_frozen0"]["mean_of_fold_macros"], abs=2e-4)


def test_summary_groups_labels_and_agg_checks(made):
    from knee import agg
    from knee.nb import n11

    payload = json.loads(made.text)
    agg._check(payload)  # the file as written
    assert payload == json.loads(json.dumps(made.s, default=agg._json_default))
    assert not agg.UID_RE.search(made.text) and "t0_" not in made.text and "resolved" not in made.text
    tag = {"label": n11.DEV_LABEL, "descriptive": True, "added_after_results": True, "note": n11.FINDINGS_NOTE}
    assert n11.DEV_LABEL == "development: these studies were test studies in iteration 1; likely optimistic"
    assert n11.FINDINGS_NOTE == "12 intervals are shown without a multiplicity correction; they describe, they decide nothing (R8)"
    assert tag.items() <= payload.items() and payload["arms"] == list(ARMS) and payload["n_boot"] == 40
    groups = n11.finding_groups()
    assert groups["meniscus_pair"] == ["Medial Meniscus", "Lateral Meniscus"] and groups["oa_pair"] == ["Medial OA", "Lateral OA"]
    assert sorted(sum(groups.values(), [])) == sorted(FINDINGS) and len(groups["all_other"]) == 8
    sides = ("above_0", "below_0", "includes_0", "undefined")
    for e in MAIN:
        r = payload["encoders"][e]
        for block in (r, r["per_fold_macro"], r["summary"]):
            assert tag.items() <= block.items()
        F = r["findings"]
        assert F[FINDINGS[RARE]]["positives"] == "<5"  # one positive study in all: suppressed
        for j, name in enumerate(FINDINGS):
            if j != RARE:
                n = sum(int((made.truth[(e, k, "frozen0")][1][:, j] == 1).sum()) for k in range(5))
                assert F[name]["positives"] == agg.count(n) and F[name]["positives"] % 5 == 0
        allf = r["summary"]["all_findings"]
        assert sorted(sum((allf[c] for c in sides), [])) == sorted(FINDINGS)
        for c in sides:
            assert allf[c] == [f for f in FINDINGS if F[f]["b_lite_minus_frozen0"]["interval_side"] == c]
        assert len(allf["above_0"]) >= 6  # b-lite was made much stronger than frozen0 in the fixture
        for g, names in groups.items():
            blk = r["summary"]["groups"][g]
            assert blk["findings"] == names and sorted(sum((blk[c] for c in sides), [])) == sorted(names)
            for c in sides:
                assert set(blk[c]) == set(allf[c]) & set(names)


def test_cli_entry_matches_the_runbook_card():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_notebooks", root / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (root / "notebooks" / "RUNBOOK.md").read_text().split("### 4b. knee-n11-dev-findings")[1].split("Rules for reading N11:")[0]
    assert mk.cell2("n11-dev-findings") == "from knee.nb import n11; n11.dev_findings()" and mk.cell2("n11-dev-findings") in card
    meta = mk.metadata("u", "n11-dev-findings")
    assert meta["kernel_sources"] == mk.metadata("u", "n11-dev")["kernel_sources"] == ["u/knee-n11-a", "u/knee-n11-b", "u/knee-n11-decide"]
    assert meta["enable_gpu"] is False and meta["enable_internet"] == mk.metadata("u", "n11-dev")["enable_internet"]
