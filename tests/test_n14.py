"""N14, the error analysis after the error hunt (n14_spec_r3.md), on synthetic stage data (CPU).

Every block is present, labelled and passes the privacy checks; the numbers equal independent
recomputations (sklearn AUCs, numpy counts); C1 leaves gold and its training look-alikes out of the fit;
C3 skips non-gated units and its profiles are attention means; C5 reads the planted geometry; a failing
block costs only itself; a chain mismatch fails the step."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.labels import FINDINGS, UID

sk = pytest.importorskip("sklearn.metrics")
torch = pytest.importorskip("torch")
MAIN = ("dinov2_vits14", "convnext_tiny")
ENCODERS = ("dinov2_vits14", "convnext_tiny", "vit_s16")
N_BOOT = 50
GATED = {"pooling": "gated", "planes": "all", "lr": 3e-3, "weight_decay": 3e-4, "dropout": 0.25,
         "attn_hidden": 16, "pos_cap": 4.47, "batch_size": 48}


def _write_heads(inp: Path) -> None:
    """head.pt and stats.pt for every synthetic unit, fit like n6._unit's refit (2 epochs, seed 0)."""
    from knee import heads, stage

    d = stage.StageData(inp)
    for e in ENCODERS:
        bank = d.bank(e, "cpu")
        for k in range(5):
            ud = inp / "n6a" / "private" / "n6a" / "units" / f"{e}_f{k}"
            cfg_d = json.loads((ud / "config.json").read_text())["config"]
            cfg_d["attn_hidden"] = 16
            cfgj = json.loads((ud / "config.json").read_text())
            cfgj["config"] = cfg_d
            (ud / "config.json").write_text(json.dumps(cfgj))
            cfg = heads.config_from(cfg_d)
            tr, te, _ = d.split(k)
            stats = bank.stats(tr)
            r = heads.fit(bank, cfg, tr, epochs=2, stats=stats, forbidden=np.concatenate([te, d.gold_rows()]), seed=0)
            torch.save(r.model.state_dict(), ud / "head.pt")
            torch.save({p: (mu.cpu(), sd.cpu()) for p, (mu, sd) in stats.items()}, ud / "stats.pt")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from tests.synth import write_dicom
    from tests.synth_stage import make_stage, make_units

    root = tmp_path_factory.mktemp("n14")
    inp = root / "input"
    facts = make_stage(inp, n=700, S=8, D=32)
    make_units(inp)
    n0 = inp / "n0" / "private" / "n0"
    st = pd.read_csv(n0 / "studies.csv")
    rng = np.random.default_rng(5)
    for p in ("sagittal", "coronal", "axial"):
        st[f"{p}_tier"] = 2
        st[f"{p}_series"] = [f"1.2.3.{i}.{p[:3]}" for i in range(len(st))]
    fb = rng.random(len(st)) < 0.15
    st.loc[fb, "sagittal_tier"] = 0
    st.to_csv(n0 / "studies.csv", index=False)
    # a planted look-alike: one training study gets a gold study's DINOv2 embeddings in every plane
    folds = pd.read_csv(inp / "n3" / "private" / "n3" / "folds.csv")
    idx = pd.read_csv(inp / "n5" / "private" / "n5" / "index.csv")
    usable = idx["usable"].to_numpy()
    gold_rows = np.flatnonzero(folds["is_gold"].to_numpy() & usable)
    train_rows = np.flatnonzero(folds["train"].to_numpy() & usable)
    g0, t0 = int(gold_rows[0]), int(train_rows[0])
    n5 = inp / "n5" / "private" / "n5"
    for p in ("sagittal", "coronal", "axial"):
        a = np.load(n5 / f"emb_dinov2_vits14_{p}.npy")
        a[t0] = a[g0]
        np.save(n5 / f"emb_dinov2_vits14_{p}.npy", a)
    _write_heads(inp)
    # N7: consensus configs and S1 predictions (knee-n7-b layout)
    n7 = inp / "n7-b" / "private" / "n7"
    n7.mkdir(parents=True)
    (inp / "n7-b" / "agg_n7.json").write_text(json.dumps(
        {"device": "cpu", "consensus": {e: {"config": GATED, "E": 3, "fold": 0} for e in ENCODERS}}))
    lab = pd.read_csv(n0 / "labels_binary.csv").set_index(UID).reindex(folds[UID])[FINDINGS].fillna(0.5).to_numpy()
    s1 = folds["s1_fold"].to_numpy()
    for i, e in enumerate(ENCODERS):
        rows = s1 >= 0
        P = 1 / (1 + np.exp(-((1.2 + 0.2 * i) * (lab[rows] - 0.3) * 3 + rng.normal(0, 1, (int(rows.sum()), 12)))))
        df = pd.DataFrame(P, columns=FINDINGS)
        df.insert(0, "s2_fold", folds["s2_fold"].to_numpy()[rows])
        df.insert(0, UID, folds[UID].to_numpy()[rows])
        df.to_csv(n7 / f"s1_{e}.csv", index=False)
    # C5: one header per chosen series for the first 60 usable studies; sagittal 64 x 48 (rows x cols) with
    # the canonical frame (columns run anterior-posterior), so 48 is the short side and A-P is NOT longer;
    # axial 48 x 64 rows x cols (A-P along rows is shorter); coronal square
    images = root / "train_series"
    sag_iop = [0, 1, 0, 0, 0, -1]
    cor_iop = [1, 0, 0, 0, 0, -1]
    ax_iop = [1, 0, 0, 0, 1, 0]
    planted = 0
    for i in np.flatnonzero(usable)[:60]:
        uid = folds[UID].iloc[i]
        for p, iop, shape in (("sagittal", sag_iop, (64, 48)), ("coronal", cor_iop, (64, 64)), ("axial", ax_iop, (48, 64))):
            ser = st.loc[st[UID] == uid, f"{p}_series"].iloc[0]
            write_dicom(images / uid / ser / "0001.dcm", np.zeros(shape), (0, 0, 0), iop, 1,
                        ("SIEMENS", "X", 1.5), "R", spacing=(0.5, 0.5))
        planted += 1
    return {"inp": inp, "facts": facts, "lookalike": (g0, t0), "images": images, "planted": planted, "fb": fb}


def _run(env, monkeypatch, root=None, **kw):
    from knee import paths
    from knee.nb import n14

    out = Path(root or env["inp"]).parent / "w14"
    monkeypatch.setattr(paths, "WORKING", out)
    return n14.run(root=root or env["inp"], n_boot=N_BOOT, images_root=env["images"], quiet=True, **kw)


def _auc(y, s, w=None):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return sk.roc_auc_score(y, s, sample_weight=w)


def _strat_auc(s, y, strata):
    """Pairs only within strata: the pair-weighted mean of the per-stratum AUCs."""
    num, den = 0.0, 0.0
    for g in np.unique(strata):
        m = strata == g
        npos, nneg = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if npos and nneg:
            num += _auc(y[m], s[m]) * npos * nneg
            den += npos * nneg
    return num / den if den else np.nan


def test_n14_blocks_flags_and_recomputation(env, monkeypatch, capfd):
    from knee import heads, stage
    from knee.nb import n9 as core
    from knee import units as unit_files

    seen = {}
    real_fit = heads.fit

    def spy(bank, cfg, train_idx, *a, **k):
        seen.setdefault("calls", []).append((np.asarray(train_idx).copy(), k.get("forbidden")))
        return real_fit(bank, cfg, train_idx, *a, **k)

    monkeypatch.setattr(heads, "fit", spy)
    s = _run(env, monkeypatch)
    assert s["status"] == "ok", s.get("errors")
    for b in ("c1_gold_all_data", "c2_side_discrimination", "c3_sagittal_attention", "c4_fallback_contrast",
              "c5_centre_crop", "c6_s1_level"):
        assert s[b]["descriptive"] is True and s[b]["added_after_results"] is True, b
    assert "STEP n14: OK" in capfd.readouterr().out

    d = stage.StageData(env["inp"])
    row_of = {u: i for i, u in enumerate(d.uids)}
    units = core._load_units(env["inp"], d.n, row_of, unit_files.unit_dirs(env["inp"]))
    units.pop("_folds")
    f2, Y = d.folds["s2_fold"], d.eval_y
    gold = d.gold_rows()
    gold = gold[np.isfinite(units[MAIN[0]][gold]).all(1) & np.isfinite(units[MAIN[1]][gold]).all(1)]

    # C1: three fits (one per encoder); gold forbidden; the planted look-alike left out; numbers as recomputed
    c1 = s["c1_gold_all_data"]
    assert c1["label"].startswith("development")
    assert len(seen["calls"]) == 3
    g0, t0 = env["lookalike"]
    for tr, forb in seen["calls"]:
        assert t0 not in set(tr.tolist()) and not set(tr.tolist()) & set(gold.tolist())
        assert set(np.asarray(forb).tolist()) == set(gold.tolist())
    assert c1["lookalikes_removed"] in ("<5", 0) or isinstance(c1["lookalikes_removed"], int)
    gy = d.gold_y
    g_find = [FINDINGS.index(f) for f in c1["findings"]]
    for e in ENCODERS:
        P = units[e]
        pooled = np.mean([_auc(gy[gold, j], P[gold, j]) for j in g_find])
        fs = np.mean([_strat_auc(P[gold, j], gy[gold, j], f2[gold]) for j in g_find])
        assert c1["encoders"][e]["s2_pooled"] == pytest.approx(pooled, abs=6e-5)
        assert c1["encoders"][e]["s2_fold_stratified"] == pytest.approx(fs, abs=6e-5)
        v = c1["encoders"][e]["all_data_minus_s2_pooled"]
        assert v["estimate"] == pytest.approx(c1["encoders"][e]["all_data"] - c1["encoders"][e]["s2_pooled"], abs=2e-4)
    assert c1["reading"] in ("the protocol explains a large part of the gold miss",
                             "the protocol does not explain the gold miss",
                             "the protocol explains part of the gold miss")

    # the fixed evaluation set (as N9)
    tr_all = np.flatnonzero(d.train & (f2 >= 0))
    ev = tr_all[d.scored[tr_all]]
    ev = ev[np.isfinite(units[MAIN[0]][ev]).all(1) & np.isfinite(units[MAIN[1]][ev]).all(1)]

    # C2: side AUC recomputed per pair and encoder
    c2 = s["c2_side_discrimination"]
    base = ev[d.flip_row[ev] < 0]
    for name, (fm, fl) in (("menisci", ("Medial Meniscus", "Lateral Meniscus")), ("oa", ("Medial OA", "Lateral OA"))):
        jm, jl = FINDINGS.index(fm), FINDINGS.index(fl)
        rows = base[(Y[base, jm] + Y[base, jl]) == 1]
        lab = Y[rows, jm]
        entry = c2["pairs"][name]
        if min((lab == 1).sum(), (lab == 0).sum()) < 50:
            assert entry["status"].startswith("suppressed")
            continue
        for e in ENCODERS:
            ref = _strat_auc(units[e][rows, jm] - units[e][rows, jl], lab, f2[rows])
            assert entry["encoders"][e]["side_auc"] == pytest.approx(ref, abs=6e-5)
        assert entry["reading"] in ("lost", "partial", "present", "mixed")

    # C3: non-gated units skipped; each mean profile is a distribution over the slices
    c3 = s["c3_sagittal_attention"]
    assert set(c3["skipped_units"]) == {f"{e}_f{k}" for e in MAIN for k in (1, 3)}
    for e in MAIN:
        for f in ("ACL", "Medial Meniscus", "Lateral Meniscus"):
            prof = c3["encoders"][e][f]["profile_positives"]
            if prof is not None:
                assert sum(prof) == pytest.approx(1.0, abs=0.01)
                lo, hi = (int(x) for x in c3["encoders"][e]["middle_third_slices"].split("-"))
                if f == "ACL":
                    assert c3["encoders"][e][f]["middle_share_positives"] == pytest.approx(sum(prof[lo:hi + 1]), abs=0.01)

    # C4: the planted fallback share and the macro recomputed inside the groups holding fallback studies
    c4 = s["c4_fallback_contrast"]
    st = pd.read_csv(Path(d.n0_dir) / "studies.csv").astype({UID: str}).set_index(UID).reindex(d.uids)
    all_fs = (st[[f"{p}_tier" for p in ("sagittal", "coronal", "axial")]].to_numpy() == 2).all(1)
    fb_rows = ev[~all_fs[ev]]
    assert c4["fallback_share_of_evaluation"] == pytest.approx(len(fb_rows) / len(ev), abs=0.006)
    groups = sorted({str(g) for g in d.g_lang[fb_rows]})
    fs_rows = ev[all_fs[ev] & np.isin(d.g_lang.astype(str), groups)[ev]]
    strata = np.array([f"{a}|{b}" for a, b in zip(f2, d.g_lang)])
    if "encoders" in c4:
        js = [FINDINGS.index(f) for f in c4["findings"]]
        for e in ENCODERS:
            a = np.mean([_strat_auc(units[e][fs_rows, j], Y[fs_rows, j], strata[fs_rows]) for j in js])
            b = np.mean([_strat_auc(units[e][fb_rows, j], Y[fb_rows, j], strata[fb_rows]) for j in js])
            assert c4["encoders"][e]["all_fluid_fatsat"] == pytest.approx(a, abs=6e-5)
            assert c4["encoders"][e]["fallback"] == pytest.approx(b, abs=6e-5)

    # C5: the planted geometry (60 studies with headers; the rest unread)
    c5 = s["c5_centre_crop"]["planes"]
    assert c5["coronal"]["square_share"] == 1.0
    assert c5["sagittal"]["square_share"] == 0.0 and c5["sagittal"]["anterior_posterior_longer_share"] == 0.0
    assert c5["axial"]["anterior_posterior_longer_share"] == 0.0
    assert c5["sagittal"]["cut_fraction"]["median"] == pytest.approx(0.25, abs=0.01)
    assert c5["sagittal"]["mm_per_pixel_at_224"]["median"] == pytest.approx(0.5 * 48 / 224, abs=0.01)
    assert c5["sagittal"]["series_read"] == 60
    assert s["c5_centre_crop"]["reading"] == "no material share (rule not met)"

    # C6: S1 and S2 mean of fold macros recomputed
    c6 = s["c6_s1_level"]["encoders"]
    f1 = d.folds["s1_fold"]
    n7 = env["inp"] / "n7-b" / "private" / "n7"
    for e in ENCODERS:
        S1 = core._load_csv(n7 / f"s1_{e}.csv", d.n, row_of)
        rows = ev[(f1[ev] >= 0) & np.isfinite(S1[ev]).all(1)]
        s1v = np.mean([np.mean([_auc(Y[rows][f1[rows] == k, j], S1[rows][f1[rows] == k, j]) for j in range(12)])
                       for k in range(5)])
        assert c6[e]["s1_mean_of_fold_macros"] == pytest.approx(s1v, abs=6e-5)


def test_n14_a_failing_block_costs_only_itself(env, monkeypatch):
    from knee.nb import n14

    def boom(c):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(n14, "c6_s1_level", boom)
    s = _run(env, monkeypatch, blocks=("c2", "c6"))
    assert "c6_s1_level" in s["errors"] and "c2_side_discrimination" in s
    assert s["status"].startswith("ok, some blocks failed")


def test_n14_chain_mismatch_fails_the_step(env, monkeypatch, tmp_path, capfd):
    from knee import safe_log

    inp = tmp_path / "input"
    shutil.copytree(env["inp"], inp)
    p = inp / "n6a" / "private" / "n6a" / "units" / "convnext_tiny_f2" / "predictions.csv"
    df = pd.read_csv(p)
    df.loc[0, "s2_fold"] = 3
    df.to_csv(p, index=False)
    with pytest.raises(safe_log.StepFailed):
        _run(env, monkeypatch, root=inp, blocks=("c2",))
    assert "STEP n14: FAILED - ChainError" in capfd.readouterr().out
