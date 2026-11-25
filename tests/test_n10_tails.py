"""N10 with N11's B-lite fold models (tails="n11"), on synthetic data (CPU, no download).

1. Without tails, N10 writes the same submission bytes as the registered code (tests/n10_reference.py, the
   file as at 095c91f).
2. With tails, the submission equals a plain computation written here: per fold the whole encoder with the
   fold's tail loaded, the fold's statistics and head, the mirrored pass for unresolved laterality, then the
   mean over the five folds. The trunk runs once per study (and once more, mirrored, where needed).
3. Refusals: a missing fold, two commits, an incomplete unit, a changed weights file, two schedules, two
   learning-rate rules, test overrides, another Stage A config.
4. agg_n10.json passes the privacy checks and holds the series tiers as rounded counts."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from knee import agg, cache, encoders, heads, paths, safe_log, study_table  # noqa: E402
from knee.labels import FINDINGS, UID  # noqa: E402
from tests.synth import make_competition  # noqa: E402
from tests.test_n10 import comp_env  # noqa: E402,F401 - the registered path's fixture

ENC = "dinov2_vits14"
D = 32
PLANES = ("sagittal", "coronal", "axial")
CFG = {"pooling": "gated", "planes": "all", "lr": 3e-3, "weight_decay": 3e-4, "dropout": 0.25,
       "attn_hidden": 16, "pos_cap": 4.47, "batch_size": 16}
SESSIONS = {"a": (0, 1, 2), "b": (3, 4)}
LRS = {0: 2e-5, 1: 8e-5, 2: 2e-5, 3: 5e-6, 4: 2e-5}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TRUNK_SLICES = []  # slices per trunk call of the fake encoder


# ---------------------------------------------------------------- 1. bit-identity without tails

def test_without_tails_the_submission_bytes_equal_the_registered_code(comp_env):  # noqa: F811
    from knee.nb import n10
    from tests import n10_reference as old

    s_old = old.run(workers=2, quiet=True)
    b_old = (paths.WORKING / "submission.csv").read_bytes()
    (paths.WORKING / "submission.csv").unlink()
    s_new = n10.run(workers=2, quiet=True, tails=None)
    assert (paths.WORKING / "submission.csv").read_bytes() == b_old
    for k in s_old:  # every field of the registered summary is still there, with the same value
        if k != "seconds":
            assert s_new[k] == s_old[k], k
    assert set(s_new) - set(s_old) == {"mode", "series_tiers"} and s_new["mode"] == "heads"
    run = json.loads((paths.WORKING / "run.json").read_text())
    assert set(run["config"]) == {"encoder", "canary", "workers", "budget_h", "max_failed_share"}  # as registered
    # the canary too
    old.run(canary=True, workers=2, quiet=True)
    b_old = (paths.WORKING / "submission.csv").read_bytes()
    assert n10.run(canary=True, workers=2, quiet=True)["mode"] == "canary"
    assert (paths.WORKING / "submission.csv").read_bytes() == b_old


def test_without_tails_the_bytes_are_equal_in_a_harder_case(comp_env, monkeypatch):  # noqa: F811
    """Two studies of unresolved laterality, one study without an axial series (prevalence row) and one
    listed study with no series at all (prevalence row), with the limit on failed studies raised."""
    from knee.nb import n10
    from tests import n10_reference as old

    comp = paths.INPUT / paths.COMPETITION
    ser = pd.read_csv(comp / "test_series.csv")
    uids = sorted(ser[UID].astype(str).unique())
    ax = ser["Anatomical_Plane"].astype(str).str.lower().str.startswith("ax")
    ser = ser[~((ser[UID].astype(str) == uids[2]) & ax) & (ser[UID].astype(str) != uids[5])]
    ser.to_csv(comp / "test_series.csv", index=False)
    real = study_table.build

    def build(*a, **kw):
        st, ch, err = real(*a, **kw)
        st.loc[[0, 3], "laterality"] = None
        return st, ch, err

    monkeypatch.setattr(study_table, "build", build)
    s_old = old.run(workers=2, quiet=True, max_failed_share=0.5)
    b_old = (paths.WORKING / "submission.csv").read_bytes()
    (paths.WORKING / "submission.csv").unlink()
    s_new = n10.run(workers=2, quiet=True, max_failed_share=0.5)
    assert (paths.WORKING / "submission.csv").read_bytes() == b_old
    assert all(s_new[k] == s_old[k] for k in s_old if k != "seconds")
    sub = pd.read_csv(paths.WORKING / "submission.csv").astype({UID: str}).set_index(UID)
    assert len(sub) == len(uids) and np.allclose(sub.loc[[uids[2], uids[5]], FINDINGS].to_numpy(), 0.2)
    assert not np.allclose(sub.loc[uids[0], FINDINGS].to_numpy(), 0.2)
    assert s_new["series_tiers"]["axial"]["no_series"] == "<5"
    # at the registered limit both stop on purpose
    for mod in (old, n10):
        (paths.WORKING / "submission.csv").unlink(missing_ok=True)
        with pytest.raises(safe_log.StepFailed):
            mod.run(workers=2, quiet=True)
        assert not (paths.WORKING / "submission.csv").exists()


def test_a_failing_tier_count_never_stops_a_submission(comp_env, monkeypatch):  # noqa: F811
    from knee.nb import n10
    from tests import n10_reference as old

    old.run(canary=True, workers=2, quiet=True)
    b_old = (paths.WORKING / "submission.csv").read_bytes()
    (paths.WORKING / "submission.csv").unlink()

    def boom(st):
        raise KeyError("sagittal_tier")

    monkeypatch.setattr(n10, "series_tiers", boom)
    s = n10.run(canary=True, workers=2, quiet=True)
    assert (paths.WORKING / "submission.csv").read_bytes() == b_old
    assert s["series_tiers"] is None
    assert json.loads((paths.WORKING / "agg_n10.json").read_text())["series_tiers"] is None


def test_tails_argument_is_checked():
    from knee.nb import n10

    with pytest.raises(ValueError):
        n10.run(tails="n13")
    with pytest.raises(ValueError):
        n10.run(canary=True, tails="n11")
    with pytest.raises(ValueError):
        n10.run(encoder_factory=fake_factory)


# ---------------------------------------------------------------- the fake encoder and N11's files

def fake_factory(name: str, device: str = "cpu"):
    """The factory hook of N11's tests: (name, device) -> an n11.Split. Frozen trunk (conv stem, pooling,
    linear) and a trainable tail (a residual block with layer-scale, a norm); any slice size."""
    from knee.nb import n11

    class LayerScale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = torch.nn.Parameter(torch.full((D,), 0.5))

    class FakeSplit(n11.Split):
        def __init__(self):
            self.name, self.kind = name, "fake"
            g = torch.Generator().manual_seed(7)
            model = torch.nn.ModuleDict({"stem": torch.nn.Conv2d(3, 8, 16, 16), "pool": torch.nn.AdaptiveAvgPool2d(4),
                                         "mid": torch.nn.Linear(8 * 16, D), "block": torch.nn.Linear(D, D),
                                         "ls": LayerScale(), "norm": torch.nn.LayerNorm(D)})
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.ndim > 1:
                        p.copy_(torch.randn(p.shape, generator=g) / p[0].numel() ** 0.5)
                    elif n.endswith("bias"):
                        p.copy_(0.1 * torch.randn(p.shape, generator=g))
            super().__init__(model, device, ("block.", "ls.", "norm."))

        def prep(self, u8):
            return u8.float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)

        def trunk(self, x):
            TRUNK_SLICES.append(int(x.shape[0]))
            m = self.model
            return torch.nn.functional.gelu(m["mid"](m["pool"](torch.nn.functional.gelu(m["stem"](x))).flatten(1)))

        def tail(self, h):
            m = self.model
            return m["norm"](h + m["ls"].gamma * torch.nn.functional.gelu(m["block"](h)))

    return FakeSplit()


def _plain_features(model, u8):
    """The whole fake encoder written out once more, without Split's trunk and tail: [n, H, W] -> [n, D]."""
    F = torch.nn.functional
    x = u8.float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    h = F.gelu(model["mid"](model["pool"](F.gelu(model["stem"](x))).flatten(1)))
    return model["norm"](h + model["ls"].gamma * F.gelu(model["block"](h)))


def _tail_of(k: int) -> dict:
    """Fold k's fine-tuned tail: the fake encoder's tail, moved by a seeded amount."""
    g = torch.Generator().manual_seed(100 + k)
    return {n: v + 0.3 * torch.randn(v.shape, generator=g) for n, v in fake_factory(ENC).tail_state().items()}


def _cfg_of(k: int) -> dict:
    return {**CFG, "planes": "sag+cor"} if k == 3 else dict(CFG)  # one fold's head reads two planes


def _write_n11(inp: Path, enc: str = ENC, sessions=SESSIONS, commit=COMMIT) -> None:
    """What N11's final fits leave for N10: per unit b_lite_seed_0.pt and state.json, per session run.json."""
    for s, folds in sessions.items():
        out = inp / f"knee-n11-{s}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps({"step": f"n11_{s}", "commit": commit, "config": {
            "session": s, "folds": list(folds), "schedule": "full", "rung1_tie": True, "passes": 16200,
            "test_overrides": False}}))
        for k in folds:
            cfg = heads.config_from(_cfg_of(k))
            ud = out / "private" / f"n11_{s}" / f"{enc}_f{k}"
            ud.mkdir(parents=True)
            torch.manual_seed(50 + k)
            head = heads.make_head(cfg, D, len(heads.PLANE_SETS[cfg.planes]))
            torch.save({"tail": _tail_of(k), "head": head.state_dict(), "encoder": enc, "fold": k,
                        "tail_prefixes": ["block.", "ls.", "norm."]}, ud / "b_lite_seed_0.pt")
            sha = hashlib.sha256((ud / "b_lite_seed_0.pt").read_bytes()).hexdigest()[:16]
            (ud / "state.json").write_text(json.dumps({
                "encoder": enc, "fold": k, "session": s, "schedule": "full", "stage_a_unit": "knee-n6a",
                "config": {"pooling": cfg.pooling, "planes": cfg.planes, "dropout": cfg.dropout, "head_lr": cfg.lr,
                           "head_weight_decay": cfg.weight_decay, "amp_fp16": False},
                "frozen0": {"valid": True}, "inner": {"complete": True, "chosen_lr": LRS[k]},
                "final": {"complete": True, "b_lite": {"seed_0": {"encoder_lr": LRS[k]}}},
                "files_sha16": {"b_lite_seed_0.pt": sha}}))


def _write_n6a(inp: Path, enc: str = ENC) -> None:
    """knee-n6a's units: each fold's own z-score statistics and head config (and a head N10 must not use)."""
    for k in range(5):
        cfg = heads.config_from(_cfg_of(k))
        ud = inp / "knee-n6a" / "private" / "n6a" / "units" / f"{enc}_f{k}"
        ud.mkdir(parents=True)
        g = torch.Generator().manual_seed(200 + k)
        torch.save({p: (0.2 * torch.randn(D, generator=g), 0.5 + torch.rand(D, generator=g)) for p in PLANES}, ud / "stats.pt")
        torch.manual_seed(900 + k)
        torch.save(heads.make_head(cfg, D, len(heads.PLANE_SETS[cfg.planes])).state_dict(), ud / "head.pt")
        (ud / "config.json").write_text(json.dumps({"encoder": enc, "fold": k, "config": _cfg_of(k), "E": 3}))


@pytest.fixture()
def blite_env(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    comp = inp / paths.COMPETITION
    comp.mkdir(parents=True)
    make_competition(comp, n_per_lang=1, n_slices=12, size=48)
    shutil.copytree(comp / "train_series", comp / "test_series")
    shutil.copy(comp / "train_series.csv", comp / "test_series.csv")
    uids = sorted(pd.read_csv(comp / "test_series.csv")[UID].astype(str).unique())
    pd.DataFrame({UID: uids, **{f: 0.5 for f in FINDINGS}}).to_csv(comp / "sample_submission.csv", index=False)
    ck = inp / "knee-bundle" / "private" / "bundle" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / f"{ENC}.pth").write_bytes(b"not read: the test passes its own encoder")
    sha = hashlib.sha256((ck / f"{ENC}.pth").read_bytes()).hexdigest()
    (ck.parent / "manifest.json").write_text(json.dumps({"commit": "abcdef1234567890", "files_sha256": {f"{ENC}.pth": sha}}))
    (inp / "knee-n0").mkdir()
    (inp / "knee-n0" / "agg_n0.json").write_text(json.dumps({"labels": {"prevalence_training": {f: 0.2 for f in FINDINGS}}}))
    _write_n6a(inp)
    _write_n11(inp)
    monkeypatch.setattr(paths, "INPUT", inp)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "working")
    monkeypatch.delenv("KNEE_DATA_ROOT", raising=False)
    # two studies of unresolved laterality (the synthetic headers resolve every side)
    real = study_table.build

    def build(*a, **kw):
        st, ch, err = real(*a, **kw)
        st.loc[[1, 4], "laterality"] = None
        return st, ch, err

    monkeypatch.setattr(study_table, "build", build)
    return {"uids": uids, "inp": inp, "comp": comp, "tmp": tmp_path}


# ---------------------------------------------------------------- 2. with tails

def test_blite_submission_equals_a_plain_computation(blite_env, capfd):
    from knee.nb import n10

    TRUNK_SLICES.clear()
    s = n10.run(ENC, tails="n11", encoder_factory=fake_factory, workers=2, quiet=True, eval_studies=3)
    printed = capfd.readouterr()
    text = printed.out + printed.err + (paths.WORKING / "agg_n10.json").read_text()
    assert not [u for u in blite_env["uids"] if u in text]
    sub = pd.read_csv(paths.WORKING / "submission.csv").astype({UID: str}).set_index(UID)

    # the plain computation: the same table and caches (training code), then every fold's whole encoder
    st, _, _ = study_table.build(blite_env["comp"], "test", workers=2)
    n = len(st)
    unresolved = ~st["laterality"].isin(["L", "R"]).to_numpy()
    assert unresolved.sum() == 2 and n == 7
    x = {}
    for p in PLANES:
        cd = blite_env["tmp"] / f"ref_{p}"
        cache.build(st, blite_env["comp"] / "test_series", p, cd, workers=2)
        cm = json.loads((cd / f"cache_{p}_manifest.json").read_text())
        x[p] = torch.from_numpy(np.array(np.memmap(cd / cm["file"], dtype=np.uint8, mode="r", shape=tuple(cm["shape"]))))
    S = x[PLANES[0]].shape[1]
    per_fold, no_flip = [], []
    for k in range(5):
        cfg = heads.config_from(_cfg_of(k))
        planes = heads.PLANE_SETS[cfg.planes]
        unit = [p for p in blite_env["inp"].glob(f"knee-n11-*/private/*/{ENC}_f{k}/b_lite_seed_0.pt")]
        ck = torch.load(unit[0], map_location="cpu")
        model = fake_factory(ENC).model
        model.load_state_dict(ck["tail"], strict=False)
        stats = torch.load(blite_env["inp"] / "knee-n6a" / "private" / "n6a" / "units" / f"{ENC}_f{k}" / "stats.pt")
        head = heads.make_head(cfg, D, len(planes))
        head.load_state_dict(ck["head"])
        head.eval()
        P, P0 = np.zeros((n, 12)), np.zeros((n, 12))
        with torch.no_grad():
            for i in range(n):
                def prob(flip):
                    zs = []
                    for p in planes:
                        u8 = x[p][i].flip(-1) if flip and p in ("coronal", "axial") else x[p][i]
                        zs.append(((_plain_features(model, u8) - stats[p][0]) / stats[p][1]).reshape(1, S, D))
                    return torch.sigmoid(head(zs))[0].numpy().astype(float)

                P0[i] = prob(False)
                P[i] = (P0[i] + prob(True)) / 2 if unresolved[i] else P0[i]
        per_fold.append(P)
        no_flip.append(P0)
    want = np.mean(per_fold, axis=0)
    got = sub.reindex(st[UID].astype(str))[FINDINGS].to_numpy()
    assert np.abs(got - want).max() < 2e-6
    assert np.abs(per_fold[0] - per_fold[1]).max() > 1e-3  # the folds differ, so the mean is a real check
    assert np.abs(got - np.mean(no_flip, axis=0))[unresolved].max() > 1e-4  # and the mirrored pass is in it
    assert np.abs(got - np.mean(no_flip, axis=0))[~unresolved].max() < 2e-6

    # one trunk pass per study, one more (mirrored) for the two unresolved studies; batches of 3 studies
    assert sum(TRUNK_SLICES) == (n + 2) * 3 * S and max(TRUNK_SLICES) == 3 * 3 * S
    b = s["b_lite"]
    assert s["mode"] == "b-lite" and s["folds"] == [0, 1, 2, 3, 4] and s["failed_share"] == 0.0
    assert b["n11_commit"] == COMMIT[:12] and b["schedule"] == "full" and b["lr_rule"] == {"rung1_tie": True, "passes": 16200}
    assert b["trunk_passes_per_study"] == round((n + 2) / n, 2) and b["amp_fp16"] is False
    assert "heads_sha16" not in s and b["test_overrides"] is True
    assert b["trunk_sha16"] == n10.trunk_sha16(fake_factory(ENC)) and len(b["trunk_sha16"]) == 16
    moved = fake_factory(ENC)
    moved.load_tail_state(_tail_of(0))
    assert n10.trunk_sha16(moved) == b["trunk_sha16"]  # the tail is not part of it
    with torch.no_grad():
        moved.model["mid"].bias.add_(1e-3)
    assert n10.trunk_sha16(moved) != b["trunk_sha16"]  # any trunk weight is
    for k in range(5):
        u = b["units"][f"fold_{k}"]
        state = json.loads(next(blite_env["inp"].glob(f"knee-n11-*/private/*/{ENC}_f{k}/state.json")).read_text())
        assert u["weights_file_sha16"] == state["files_sha16"]["b_lite_seed_0.pt"]  # the hash N11 logged
        assert u["encoder_lr"] == LRS[k] and u["session"] == ("a" if k < 3 else "b")
        ck = torch.load(next(blite_env["inp"].glob(f"knee-n11-*/private/*/{ENC}_f{k}/b_lite_seed_0.pt")))
        assert u["tail_sha16"] == n10._state_sha16(ck["tail"]) and u["head_sha16"] == n10._state_sha16(ck["head"])
        assert len(u["tail_sha16"]) == 16 and u["tail_sha16"] != u["head_sha16"]
    assert len({b["units"][f"fold_{k}"]["tail_sha16"] for k in range(5)}) == 5

    # 4. the summary: the privacy checks, the tiers as rounded counts, nothing per study
    out = json.loads((paths.WORKING / "agg_n10.json").read_text())
    agg._check(out)
    tiers = out["series_tiers"]
    assert set(tiers) == set(PLANES) | {"studies_all_planes_top_tier"}
    for p in PLANES:  # 7 studies, every plane at the top tier: 7 is written as 5
        assert tiers[p] == {"fluid_sensitive_fat_suppressed": 5, "fluid_sensitive_only": 0, "any_series": 0, "no_series": 0}
    assert tiers["studies_all_planes_top_tier"] == 5

    def leaves(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from leaves(v)
        elif isinstance(o, list):
            for v in o:
                yield from leaves(v)
        else:
            yield o

    vals = list(leaves(out))
    assert all(len(v) <= 200 for v in vals if isinstance(v, str))
    assert len(vals) < 120  # nothing per study: a fixed number of fields, whatever the number of studies
    run = json.loads((paths.WORKING / "run.json").read_text())
    assert run["config"]["tails"] == "n11" and run["config"]["test_overrides"] is True
    assert set(p.name for p in paths.WORKING.iterdir() if p.is_file()) <= {"submission.csv", "agg_n10.json", "run.json"}


def test_series_tiers_are_rounded_counts():
    from knee.nb import n10

    st = pd.DataFrame({"sagittal_tier": [2] * 23 + [1] * 3 + [0] * 6 + [np.nan] * 2,
                       "coronal_tier": [2] * 34, "axial_tier": [2] * 20 + [np.nan] * 14})
    t = n10.series_tiers(st)
    assert t["sagittal"] == {"fluid_sensitive_fat_suppressed": 25, "fluid_sensitive_only": "<5", "any_series": 5, "no_series": "<5"}
    assert t["axial"]["no_series"] == 15 and t["studies_all_planes_top_tier"] == 20
    agg._check(json.loads(json.dumps({"series_tiers": t})))


# ---------------------------------------------------------------- 3. refusals

def _edit_state(inp: Path, k: int, fn) -> None:
    p = next(inp.glob(f"knee-n11-*/private/*/{ENC}_f{k}/state.json"))
    st = json.loads(p.read_text())
    fn(st)
    p.write_text(json.dumps(st))


def _edit_run(inp: Path, s: str, fn) -> None:
    p = inp / f"knee-n11-{s}" / "run.json"
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))


@pytest.fixture()
def n11_files(tmp_path):
    inp = tmp_path / "input"
    _write_n6a(inp)
    _write_n11(inp)
    return inp


def test_tail_units_reads_the_five_final_fits(n11_files):
    from knee.nb import n10

    t = n10.tail_units(ENC, n11_files)
    assert sorted(t["units"]) == [0, 1, 2, 3, 4] and t["commit"] == COMMIT and t["schedule"] == "full"
    assert [t["units"][k]["encoder_lr"] for k in range(5)] == [LRS[k] for k in range(5)]
    assert t["units"][3]["config"].planes == "sag+cor"
    # a unit rerun whole in session c replaces the unfinished one of session b
    _edit_state(n11_files, 4, lambda st: st["final"].update({"complete": False}))
    _write_n11(n11_files, sessions={"c": (4,)})
    assert n10.tail_units(ENC, n11_files)["units"][4]["session"] == "c"


def test_refused_missing_fold(n11_files):
    from knee.nb import n10

    shutil.rmtree(n11_files / "knee-n11-b")
    with pytest.raises(n10.TailError, match="exactly folds 0-4"):
        n10.tail_units(ENC, n11_files)
    with pytest.raises(n10.TailError, match="exactly folds 0-4"):  # the other encoder has no unit at all
        n10.tail_units("convnext_tiny", n11_files)


def test_refused_state_of_another_unit_and_duplicate_units(n11_files):
    from knee.nb import n10

    _edit_state(n11_files, 1, lambda st: st.update({"fold": 2}))
    with pytest.raises(n10.TailError, match="fold 1 names another encoder or fold"):
        n10.tail_units(ENC, n11_files)
    _edit_state(n11_files, 1, lambda st: st.update({"fold": 1, "encoder": "convnext_tiny"}))
    with pytest.raises(n10.TailError, match="fold 1 names another encoder or fold"):
        n10.tail_units(ENC, n11_files)
    _edit_state(n11_files, 1, lambda st: st.update({"encoder": ENC}))
    n10.tail_units(ENC, n11_files)
    _write_n11(n11_files, sessions={"c": (4,)})  # fold 4 complete in session b and in session c
    with pytest.raises(n10.TailError, match="equally far"):
        n10.tail_units(ENC, n11_files)


def test_refused_mixed_or_unknown_commit(n11_files):
    from knee.nb import n10

    _edit_run(n11_files, "b", lambda r: r.update({"commit": "f" * 40}))
    with pytest.raises(n10.TailError, match="one known commit"):
        n10.tail_units(ENC, n11_files)
    _edit_run(n11_files, "a", lambda r: r.update({"commit": None}))
    _edit_run(n11_files, "b", lambda r: r.update({"commit": None}))
    with pytest.raises(n10.TailError, match="one known commit"):
        n10.tail_units(ENC, n11_files)


@pytest.mark.parametrize("edit", [lambda st: st["final"].update({"complete": False}),
                                  lambda st: st.pop("final"),
                                  lambda st: st["frozen0"].update({"valid": False})])
def test_refused_incomplete_unit(n11_files, edit):
    from knee.nb import n10

    _edit_state(n11_files, 2, edit)
    with pytest.raises(n10.TailError, match="fold 2 is not complete"):
        n10.tail_units(ENC, n11_files)


def test_refused_weights_missing_or_changed(n11_files):
    from knee.nb import n10

    p = next(n11_files.glob(f"knee-n11-*/private/*/{ENC}_f1/b_lite_seed_0.pt"))
    ck = torch.load(p)
    ck["head"] = {k: v + 1 for k, v in ck["head"].items()}
    torch.save(ck, p)
    with pytest.raises(n10.TailError, match="weights of fold 1"):
        n10.tail_units(ENC, n11_files)
    p.unlink()
    with pytest.raises(n10.TailError, match="weights of fold 1"):
        n10.tail_units(ENC, n11_files)


def test_refused_schedule_rule_overrides_and_config(n11_files):
    from knee.nb import n10

    def fresh():
        shutil.rmtree(n11_files)
        _write_n6a(n11_files)
        _write_n11(n11_files)

    _edit_state(n11_files, 3, lambda st: st.update({"schedule": "halved"}))
    with pytest.raises(n10.TailError, match="different schedules"):
        n10.tail_units(ENC, n11_files)
    fresh()
    _edit_run(n11_files, "b", lambda r: r["config"].update({"rung1_tie": False}))
    with pytest.raises(n10.TailError, match="different learning-rate rules"):
        n10.tail_units(ENC, n11_files)
    fresh()
    _edit_state(n11_files, 0, lambda st: st["final"]["b_lite"]["seed_0"].update({"encoder_lr": 8e-5}))
    with pytest.raises(n10.TailError, match="chosen learning rate"):
        n10.tail_units(ENC, n11_files)
    fresh()
    _edit_run(n11_files, "a", lambda r: r["config"].update({"test_overrides": True}))
    with pytest.raises(n10.TailError, match="test overrides"):
        n10.tail_units(ENC, n11_files)
    fresh()
    _edit_state(n11_files, 4, lambda st: st["config"].update({"dropout": 0.1}))
    with pytest.raises(n10.TailError, match="not the one N11 trained with"):
        n10.tail_units(ENC, n11_files)
    fresh()
    shutil.rmtree(n11_files / "knee-n6a" / "private" / "n6a" / "units" / f"{ENC}_f2")
    with pytest.raises(n10.TailError, match="knee-n6a's unit of fold 2"):
        n10.tail_units(ENC, n11_files)


def test_a_refused_run_stops_before_decoding_and_writes_no_submission(blite_env):
    from knee.nb import n10

    shutil.rmtree(blite_env["inp"] / "knee-n11-b")
    calls = []
    real = study_table.build
    study_table.build = lambda *a, **kw: calls.append(1) or real(*a, **kw)
    try:
        with pytest.raises(safe_log.StepFailed):
            n10.run(ENC, tails="n11", encoder_factory=fake_factory, workers=2, quiet=True)
    finally:
        study_table.build = real
    assert not calls and not (paths.WORKING / "submission.csv").exists()


def test_a_tail_of_another_encoder_or_fold_is_refused(n11_files):
    from knee.nb import n10

    t = n10.tail_units(ENC, n11_files)
    enc = fake_factory(ENC)
    caches = {p: np.zeros((2, 2, 32, 32), np.uint8) for p in PLANES}
    P, _ = n10.predict_tails(enc, t, caches, np.arange(2), np.array([-1, 0]))
    assert P.shape == (5, 2, 12) and np.isfinite(P).all() and not enc.model.training
    p = t["units"][1]["weights"]
    ck = torch.load(p)
    torch.save({**ck, "fold": 2}, p)
    with pytest.raises(n10.TailError, match="another unit"):
        n10.predict_tails(enc, t, caches, np.arange(2), np.array([-1, 0]))
    torch.save({**ck, "tail_prefixes": ["block.", "norm."]}, p)
    with pytest.raises(n10.TailError, match="trainable tail"):
        n10.predict_tails(enc, t, caches, np.arange(2), np.array([-1, 0]))
    torch.save({**ck, "tail": {k: (v[:1] if v.ndim == 1 else v) for k, v in ck["tail"].items()}}, p)
    with pytest.raises(n10.TailError, match="another shape"):  # copy_ would broadcast a 1-element tensor
        n10.predict_tails(enc, t, caches, np.arange(2), np.array([-1, 0]))


# ---------------------------------------------------------------- the replay check against N11's predictions

CHECK_ROWS = [7, 2, 5, 0, 9, 3]  # fold 0's scored test studies, as cache rows, in the prediction file's order
CHECK_RESOLVED = [True, False, True, True, False, True]


@pytest.fixture()
def check_env(tmp_path, monkeypatch):
    """The bundle, three N4-like caches, knee-n6a, the N11 outputs and fold 0's pred_b_lite.csv, computed
    here with the plain forward (the whole fake encoder with fold 0's tail)."""
    from tests.synth import uid

    inp = tmp_path / "input"
    _write_n6a(inp)
    _write_n11(inp)
    ck = inp / "knee-bundle" / "private" / "bundle" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / f"{ENC}.pth").write_bytes(b"not read: the test passes its own encoder")
    sha = hashlib.sha256((ck / f"{ENC}.pth").read_bytes()).hexdigest()
    (ck.parent / "manifest.json").write_text(json.dumps({"commit": "abcdef1234567890", "files_sha256": {f"{ENC}.pth": sha}}))
    n, S = 10, 2
    uids = [uid(i) for i in range(n)]
    rng = np.random.default_rng(5)
    x = {}
    for p in PLANES:
        d = inp / f"knee-n4-{p}" / "private" / f"n4_{p}"
        d.mkdir(parents=True)
        x[p] = rng.integers(0, 256, size=(n, S, 32, 32), dtype=np.uint8)
        x[p].tofile(d / f"cache_{p}.u8")
        (d / f"cache_{p}_manifest.json").write_text(json.dumps({"plane": p, "shape": list(x[p].shape), "dtype": "uint8",
                                                               "file": f"cache_{p}.u8"}))
        pd.DataFrame({UID: uids, "row": range(n), "ok": True}).to_csv(d / f"cache_{p}_studies.csv", index=False)
    ud = inp / "knee-n11-a" / "private" / "n11_a" / f"{ENC}_f0"
    w = torch.load(ud / "b_lite_seed_0.pt")
    model = fake_factory(ENC).model
    model.load_state_dict(w["tail"], strict=False)
    cfg = heads.config_from(_cfg_of(0))
    head = heads.make_head(cfg, D, 3)
    head.load_state_dict(w["head"])
    head.eval()
    stats = torch.load(inp / "knee-n6a" / "private" / "n6a" / "units" / f"{ENC}_f0" / "stats.pt")
    P = np.zeros((len(CHECK_ROWS), 12))
    with torch.no_grad():
        for j, (i, res) in enumerate(zip(CHECK_ROWS, CHECK_RESOLVED)):
            def prob(flip):
                zs = []
                for p in PLANES:
                    u8 = torch.from_numpy(x[p][i])
                    u8 = u8.flip(-1) if flip and p in ("coronal", "axial") else u8
                    zs.append(((_plain_features(model, u8) - stats[p][0]) / stats[p][1]).reshape(1, S, D))
                return torch.sigmoid(head(zs))[0].numpy().astype(float)

            P[j] = prob(False) if res else (prob(False) + prob(True)) / 2
    df = pd.DataFrame(P.astype(np.float32), columns=FINDINGS)
    for f in FINDINGS:
        df[f"y_{f}"] = 0.0
    df.insert(0, "resolved", CHECK_RESOLVED)
    df.insert(0, "s2_fold", 0)
    df.insert(0, UID, [uids[i] for i in CHECK_ROWS])
    df.to_csv(ud / "pred_b_lite.csv", index=False)
    _edit_state(inp, 0, lambda st: st["files_sha16"].update(
        {"pred_b_lite.csv": hashlib.sha256((ud / "pred_b_lite.csv").read_bytes()).hexdigest()[:16]}))
    monkeypatch.setattr(paths, "INPUT", inp)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "working")
    monkeypatch.delenv("KNEE_DATA_ROOT", raising=False)
    return {"inp": inp, "uids": uids, "unit": ud}


def test_check_tails_replays_n11s_predictions_and_writes_aggregates_only(check_env, capfd):
    from knee.nb import n10

    s = n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory)
    printed = capfd.readouterr()
    text = printed.out + printed.err + (paths.WORKING / "agg_n10_tails_check.json").read_text()
    assert not [u for u in check_env["uids"] if u in text] and not agg.UID_RE.search(text)
    assert s["passed"] is True and s["max_abs_probability_gap"] < 1e-5 and s["mean_abs_probability_gap"] < 1e-5
    assert s["studies"] == 5 and s["studies_unresolved_laterality"] == "<5" and s["fold"] == 0 and s["trunk_sha16"] == n10.trunk_sha16(fake_factory(ENC))
    out = json.loads((paths.WORKING / "agg_n10_tails_check.json").read_text())
    agg._check(out)
    assert set(out) == {"encoder", "fold", "bundle_commit", "studies", "studies_unresolved_laterality", "max_abs_probability_gap", "mean_abs_probability_gap",
                        "tolerance", "passed", "trunk_sha16", "n11_commit", "weights_file_sha16", "stage_a_stats_sha16",
                        "prediction_file_sha16", "tail_sha16", "head_sha16", "amp_fp16", "amp_fp16_in_training", "device",
                        "reads", "seconds"}
    assert set(p.name for p in paths.WORKING.iterdir() if p.is_file()) <= {"agg_n10_tails_check.json", "run.json"}
    assert not list(paths.WORKING.rglob("*.csv"))
    # the first n studies only
    s = n10.check_tails(ENC, fold=0, n=4, encoder_factory=fake_factory)
    assert s["passed"] is True and s["studies"] == "<5"


def test_check_tails_fails_on_other_statistics_another_flip_rule_or_a_changed_file(check_env):
    from knee.nb import n10

    # another version of knee-n6a's statistics (same config): the gap shows it
    sp = check_env["inp"] / "knee-n6a" / "private" / "n6a" / "units" / f"{ENC}_f0" / "stats.pt"
    good = sp.read_bytes()
    st = torch.load(sp)
    torch.save({p: (mu + 0.5, sd) for p, (mu, sd) in st.items()}, sp)
    def failed(**kw):
        with pytest.raises(safe_log.StepFailed):  # the summary is written first, then the step fails
            n10.check_tails(ENC, fold=0, n=200, **{"encoder_factory": fake_factory, **kw})
        return json.loads((paths.WORKING / "agg_n10_tails_check.json").read_text())

    s = failed()
    assert s["passed"] is False and s["max_abs_probability_gap"] > 0.01
    sp.write_bytes(good)
    # another trunk
    def other(name, device="cpu"):
        e = fake_factory(name, device)
        with torch.no_grad():
            e.model["mid"].weight.mul_(1.3)
        return e

    s2 = failed(encoder_factory=other)
    assert s2["passed"] is False and s2["trunk_sha16"] != n10.trunk_sha16(fake_factory(ENC))
    # the prediction file is not the one the state records
    pp = check_env["unit"] / "pred_b_lite.csv"
    df = pd.read_csv(pp)
    df.loc[0, "resolved"] = not bool(df.loc[0, "resolved"])
    df.to_csv(pp, index=False)
    with pytest.raises(safe_log.StepFailed):
        n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory)
    # ... and with its hash recorded, the other flip rule for one study is a gap
    _edit_state(check_env["inp"], 0, lambda st: st["files_sha16"].update(
        {"pred_b_lite.csv": hashlib.sha256(pp.read_bytes()).hexdigest()[:16]}))
    try:
        s3 = n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory)
    except safe_log.StepFailed:
        s3 = json.loads((paths.WORKING / "agg_n10_tails_check.json").read_text())
    assert s3["max_abs_probability_gap"] > 1e-4
    # a prediction that is not finite: null in strict JSON, and a fail
    df[FINDINGS[0]] = df[FINDINGS[0]].astype(float)
    df.loc[1, FINDINGS[0]] = np.nan
    df.to_csv(pp, index=False)
    _edit_state(check_env["inp"], 0, lambda st: st["files_sha16"].update(
        {"pred_b_lite.csv": hashlib.sha256(pp.read_bytes()).hexdigest()[:16]}))
    with pytest.raises(safe_log.StepFailed):
        n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory)
    text = (paths.WORKING / "agg_n10_tails_check.json").read_text()
    s4 = json.loads(text)
    assert "NaN" not in text and s4["passed"] is False and s4["max_abs_probability_gap"] is None
    assert s4["mean_abs_probability_gap"] is None
    # a study that is in no cache
    for p in PLANES:
        c = check_env["inp"] / f"knee-n4-{p}" / "private" / f"n4_{p}" / f"cache_{p}_studies.csv"
        per = pd.read_csv(c)
        per.loc[7, "ok"] = False
        per.to_csv(c, index=False)
    with pytest.raises(safe_log.StepFailed):
        n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory)


@pytest.mark.parametrize("name", ["dinov2_vits14", "convnext_tiny"])
def test_real_split_equals_the_whole_timm_encoder_with_the_tail_loaded(tmp_path, name):
    """The path of a real run (n11.TimmSplit, random weights, no download): trunk once, then each fold's
    tail, against encoders.Encoder (the whole model in one forward) holding the same weights."""
    from knee.nb import n10, n11

    torch.manual_seed(0)
    enc = n11.TimmSplit(name, "cpu", pretrained=False)
    enc.check()
    base = {k: v.clone() for k, v in enc.model.state_dict().items()}
    rng = np.random.default_rng(1)
    S, n = 2, 3
    caches = {p: rng.integers(0, 256, size=(n, S, 224, 224), dtype=np.uint8) for p in PLANES}
    flip_row = np.array([-1, 0, -1])
    cfg = heads.Config(pooling="gated", planes="all")
    units, want = {}, []
    for k in (0, 1):
        g = torch.Generator().manual_seed(10 + k)
        tail = {n_: v + 0.02 * torch.randn(v.shape, generator=g) for n_, v in enc.tail_state().items()}
        torch.manual_seed(20 + k)
        head = heads.make_head(cfg, encoders.FEATURE_DIM, 3).eval()
        stats = {p: (0.1 * torch.randn(encoders.FEATURE_DIM, generator=g), 0.5 + torch.rand(encoders.FEATURE_DIM, generator=g))
                 for p in PLANES}
        torch.save({"tail": tail, "head": head.state_dict(), "encoder": name, "fold": k,
                    "tail_prefixes": list(enc.tail_prefixes)}, tmp_path / f"w{k}.pt")
        torch.save(stats, tmp_path / f"s{k}.pt")
        units[k] = {"weights": tmp_path / f"w{k}.pt", "stats": tmp_path / f"s{k}.pt", "config": cfg}
        whole = encoders.Encoder(name, "cpu", pretrained=False)
        whole.model.load_state_dict({**base, **tail})
        P = np.zeros((n, 12))
        with torch.no_grad():
            for i in range(n):
                def prob(flip):
                    zs = []
                    for p in PLANES:
                        u8 = torch.from_numpy(caches[p][i])
                        f = whole(u8.flip(-1) if flip and p in ("coronal", "axial") else u8).float()
                        zs.append(((f - stats[p][0]) / stats[p][1]).reshape(1, S, -1))
                    return torch.sigmoid(head(zs))[0].numpy().astype(float)

                P[i] = (prob(False) + prob(True)) / 2 if flip_row[i] >= 0 else prob(False)
        want.append(P)
    got, info = n10.predict_tails(enc, {"units": units}, caches, np.arange(n), flip_row, eval_studies=2)
    assert got.shape == (2, n, 12)
    assert np.abs(got - np.array(want)).max() < 2e-3  # Encoder stores its features as float16
    assert np.abs(got[0] - got[1]).max() > 1e-3
    assert info["trunk_slices"] == (n + 1) * 3 * S and info["amp_fp16"] is False


# ---------------------------------------------------------------- the notebook entries and their card

def test_blite_notebook_entries_match_the_runbook_card():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_notebooks", root / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (root / "notebooks" / "RUNBOOK.md").read_text().split("## N10 with N11's B-lite models")[1]
    for name, enc in (("n10-blite", "dinov2_vits14"), ("n10-convnext-blite", "convnext_tiny")):
        assert mk.cell2(name) == f'from knee.nb import n10; n10.run("{enc}", tails="n11")' and mk.cell2(name) in card
        meta = mk.metadata("u", name, bundle="u/knee-bundle")
        assert meta["enable_gpu"] is True and meta["enable_internet"] is False
        assert meta["kernel_sources"] == ["u/knee-n0", "u/knee-n6a", "u/knee-n11-a", "u/knee-n11-b"]
        assert meta["dataset_sources"] == ["u/knee-bundle"] and meta["competition_sources"] == [mk.COMPETITION]
        assert mk.cell1("x", offline=name in mk.OFFLINE) == mk.CELL1_OFFLINE
        assert mk.NOTEBOOKS[name] == mk.NOTEBOOKS["n10"][:3] + (["n0", "n6a", "n11-a", "n11-b"], mk.NOTEBOOKS["n10"][4])
    name = "n10-tails-check"
    assert mk.cell2(name) == 'from knee.nb import n10; n10.check_tails("dinov2_vits14", fold=0, n=200)' and mk.cell2(name) in card
    meta = mk.metadata("u", name, bundle="u/knee-bundle")
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False and meta["competition_sources"] == []
    assert meta["kernel_sources"] == [f"u/knee-{x}" for x in ("n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n6a",
                                                              "n11-a", "n11-b")]
    assert meta["dataset_sources"] == ["u/knee-bundle"] and name in mk.OFFLINE
    for name in ("n10", "n10-convnext", "n10-canary"):  # the registered entries are unchanged
        assert "tails" not in mk.cell2(name)
    assert mk.NOTEBOOKS["n10"] == (True, True, False, ["n0", "n6a"], 32400)


# ---------------------------------------------------------------- tails="n11x": the refit model of the extension

PICKS_X = {0: 3.2e-4, 1: 8e-5, 2: 1.28e-3, 3: 8e-5, 4: 3.2e-4}
COMMIT_X = "fedcba9876543210fedcba9876543210fedcba98"


def _selection_x(picks=PICKS_X, **over) -> dict:
    """A signed selection record with the fields the readers use (n11._signed, as select_lrx writes it)."""
    from knee.nb import n11

    return n11._signed({"status": "selected", "smoke": False, "plan": "it3_lr_extension_r3", "rule_d": {"words": "test"},
                        "written_at_utc": "2026-11-25T11:15:00Z", "picks": {f"fold_{k}": v for k, v in picks.items()},
                        **over})


def _write_n11x(inp: Path, sel: dict | None = None, enc: str = ENC, folds=range(5), commit=COMMIT_X) -> dict:
    """What knee-n11x-final and knee-n11x-select leave for N10: the fold models of _write_n11 (the same
    tensors) under private/n11x_final/, each state with the selection's sha256 and the fold's pick."""
    sel = sel or _selection_x()
    (inp / "knee-n11x-select").mkdir(parents=True, exist_ok=True)
    (inp / "knee-n11x-select" / "agg_n11x_select.json").write_text(json.dumps(sel))
    out = inp / "knee-n11x-final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"step": "n11x_final", "commit": commit, "config": {
        "session": "final", "folds": list(folds), "schedule": "full", "passes": 16200, "selection_sha256": sel["sha256"],
        "test_overrides": False}}))
    for k in folds:
        cfg = heads.config_from(_cfg_of(k))
        ud = out / "private" / "n11x_final" / f"{enc}_f{k}"
        ud.mkdir(parents=True)
        torch.manual_seed(50 + k)
        head = heads.make_head(cfg, D, len(heads.PLANE_SETS[cfg.planes]))
        torch.save({"tail": _tail_of(k), "head": head.state_dict(), "encoder": enc, "fold": k,
                    "tail_prefixes": ["block.", "ls.", "norm."]}, ud / "b_lite_seed_0.pt")
        sha = hashlib.sha256((ud / "b_lite_seed_0.pt").read_bytes()).hexdigest()[:16]
        lr = PICKS_X[k]
        (ud / "state.json").write_text(json.dumps({
            "encoder": enc, "fold": k, "session": "final", "schedule": "full", "stage_a_unit": "knee-n6a",
            "config": {"pooling": cfg.pooling, "planes": cfg.planes, "dropout": cfg.dropout, "head_lr": cfg.lr,
                       "head_weight_decay": cfg.weight_decay, "amp_fp16": False},
            "ext_final": {"selection_sha256": sel["sha256"], "pick": lr, "moved": lr != 8e-5},
            "frozen0": {"valid": True},
            "final": {"complete": True, "valid": True,
                      "b_lite": {"seed_0": {"encoder_lr": lr, "status": "finished", "finished": True, "valid": True}}},
            "files_sha16": {"b_lite_seed_0.pt": sha}}))
    return sel


def _edit_x(inp: Path, k: int, fn) -> None:
    p = inp / "knee-n11x-final" / "private" / "n11x_final" / f"{ENC}_f{k}" / "state.json"
    st = json.loads(p.read_text())
    fn(st)
    p.write_text(json.dumps(st))


@pytest.fixture()
def n11x_files(tmp_path):
    inp = tmp_path / "input"
    _write_n6a(inp)
    return inp, _write_n11x(inp)


def test_tail_units_reads_the_five_final_fits_of_the_extension(n11x_files):
    from knee.nb import n10

    inp, sel = n11x_files
    assert n10.TAILS == ("n11", "n11x")
    t = n10.tail_units(ENC, inp, "n11x")
    assert sorted(t["units"]) == [0, 1, 2, 3, 4] and t["commit"] == COMMIT_X and t["schedule"] == "full" and t["source"] == "n11x"
    assert [t["units"][k]["encoder_lr"] for k in range(5)] == [PICKS_X[k] for k in range(5)]
    assert t["selection"] == {"sha256": sel["sha256"], "written_at_utc": sel["written_at_utc"], "folds_moved": [0, 2, 4]}
    assert t["lr_rule"] == {"selection_sha256": sel["sha256"], "passes": 16200} and t["units"][3]["config"].planes == "sag+cor"
    with pytest.raises(n10.TailError, match="exactly folds 0-4"):  # the N11 source sees none of these units
        n10.tail_units(ENC, inp)
    with pytest.raises(ValueError):
        n10.tail_units(ENC, inp, "n12")
    with pytest.raises(n10.TailError, match="DINOv2 only"):
        n10.tail_units("convnext_tiny", inp, "n11x")
    # the two sources stay apart: iteration 3's units beside the extension's change neither reading
    _write_n11(inp)
    old = n10.tail_units(ENC, inp)
    assert set(old) == {"units", "commit", "schedule", "lr_rule"} and old["commit"] == COMMIT
    assert [old["units"][k]["encoder_lr"] for k in range(5)] == [LRS[k] for k in range(5)]
    assert n10.tail_units(ENC, inp, "n11x")["commit"] == COMMIT_X


def test_refused_extension_units_selection_rate_validity_commit(n11x_files):
    from knee.nb import n10

    inp, sel = n11x_files
    selp = inp / "knee-n11x-select" / "agg_n11x_select.json"

    def fresh(**kw):
        shutil.rmtree(inp)
        _write_n6a(inp)
        return _write_n11x(inp, **kw)

    def refused(match):
        with pytest.raises(n10.TailError, match=match):
            n10.tail_units(ENC, inp, "n11x")

    selp.unlink()  # no selection record attached
    refused("selection record")
    fresh()
    bad = json.loads(selp.read_text())
    bad["picks"]["fold_1"] = 3.2e-4  # a pick changed after the record was written
    selp.write_text(json.dumps(bad))
    refused("changed after it was written")
    for over in ({"status": "inconclusive"}, {"smoke": True}):
        fresh(sel=_selection_x(**over))
        refused("selection record")
    fresh()
    selp.write_text(json.dumps(_selection_x(written_at_utc="2026-11-25T15:00:00Z")))  # another (later) record
    refused("fold 0 was not run under the attached selection record")
    fresh()
    _edit_x(inp, 2, lambda st: st["ext_final"].update({"selection_sha256": "0" * 64}))
    refused("fold 2 was not run under the attached selection record")
    fresh()
    _edit_x(inp, 4, lambda st: st["final"]["b_lite"]["seed_0"].update({"encoder_lr": 8e-5}))  # iteration 3's rate, not the pick
    refused("fold 4 was not run at the fold's pick")
    fresh(sel=_selection_x({**PICKS_X, 1: 3.2e-4}))  # the record picks another rate than the fit of fold 1 used
    refused("fold 1 was not run at the fold's pick")
    for edit in (lambda st: st["final"].update({"valid": False}), lambda st: st["final"].update({"complete": False}),
                 lambda st: st["final"]["b_lite"]["seed_0"].update({"valid": False, "status": "invalid"}),
                 lambda st: st["final"]["b_lite"]["seed_0"].update({"finished": False, "status": "unfinished: out of time"}),
                 lambda st: st["frozen0"].update({"valid": False})):
        fresh()
        _edit_x(inp, 3, edit)
        refused("final fit of fold 3 is not finished and valid")
    fresh(folds=(0, 2, 4))  # the moved folds only: no N10 preparation
    refused("exactly folds 0-4")
    fresh()
    (inp / "knee-n11x-final" / "private" / "n11x_final" / f"{ENC}_f1" / "b_lite_seed_0.pt").write_bytes(b"other")
    refused("weights of fold 1")
    fresh()
    shutil.copytree(inp / "knee-n11x-final", inp / "knee-n11x-final-2")  # the same units attached twice
    refused("two final-fit units")
    fresh(commit=None)
    refused("one known commit")
    fresh()
    rj = inp / "knee-n11x-final" / "run.json"
    r = json.loads(rj.read_text())
    rj.write_text(json.dumps({**r, "config": {**r["config"], "test_overrides": True}}))
    refused("test overrides")
    rj.write_text(json.dumps({**r, "config": {**r["config"], "selection_sha256": "0" * 64}}))
    refused("session did not run under the attached selection record")
    fresh()
    _edit_x(inp, 4, lambda st: st["config"].update({"dropout": 0.1}))
    refused("not the one N11 trained with")
    fresh()
    _edit_x(inp, 0, lambda st: st.update({"fold": 1}))
    refused("names another encoder or fold")
    fresh()
    shutil.rmtree(inp / "knee-n6a" / "private" / "n6a" / "units" / f"{ENC}_f2")
    refused("knee-n6a's unit of fold 2")


def test_refit_model_submission_equals_the_same_fold_models_read_from_n11(blite_env):
    """The same five fold models once under knee-n11-a/-b and once under knee-n11x-final: tails="n11x" writes
    the same submission bytes as tails="n11", and says in its summary where the models came from."""
    from knee.nb import n10

    sel = _write_n11x(blite_env["inp"])
    s_old = n10.run(ENC, tails="n11", encoder_factory=fake_factory, workers=2, quiet=True, eval_studies=3)
    b_old = (paths.WORKING / "submission.csv").read_bytes()
    (paths.WORKING / "submission.csv").unlink()
    assert "source" not in s_old["b_lite"] and "selection_sha256" not in s_old["b_lite"]  # the n11 summary is as it was
    s = n10.run(ENC, tails="n11x", encoder_factory=fake_factory, workers=2, quiet=True, eval_studies=3)
    assert (paths.WORKING / "submission.csv").read_bytes() == b_old
    b = s["b_lite"]
    assert s["mode"] == "b-lite" and s["folds"] == [0, 1, 2, 3, 4] and b["source"] == "n11x"
    assert b["selection_sha256"] == sel["sha256"] and b["n11_commit"] == COMMIT_X[:12] and b["schedule"] == "full"
    assert b["lr_rule"] == {"selection_sha256": sel["sha256"], "passes": 16200}
    for k in range(5):
        u, o = b["units"][f"fold_{k}"], s_old["b_lite"]["units"][f"fold_{k}"]
        assert u["encoder_lr"] == PICKS_X[k] and u["session"] == "final"
        assert (u["tail_sha16"], u["head_sha16"], u["stage_a_stats_sha16"]) == (o["tail_sha16"], o["head_sha16"], o["stage_a_stats_sha16"])
    assert set(b) - set(s_old["b_lite"]) == {"source", "selection_sha256", "selection_written_at_utc"}
    out = json.loads((paths.WORKING / "agg_n10.json").read_text())
    agg._check(out)
    text = (paths.WORKING / "agg_n10.json").read_text()
    assert not [u for u in blite_env["uids"] if u in text]
    run = json.loads((paths.WORKING / "run.json").read_text())
    assert run["config"]["tails"] == "n11x"
    # refused before any decoding when a final fit is not valid; no submission is written
    (paths.WORKING / "submission.csv").unlink()
    _edit_x(blite_env["inp"], 2, lambda st: st["final"].update({"valid": False}))
    calls = []
    real = study_table.build
    study_table.build = lambda *a, **kw: calls.append(1) or real(*a, **kw)
    try:
        with pytest.raises(safe_log.StepFailed):
            n10.run(ENC, tails="n11x", encoder_factory=fake_factory, workers=2, quiet=True)
    finally:
        study_table.build = real
    assert not calls and not (paths.WORKING / "submission.csv").exists()
    with pytest.raises(ValueError):
        n10.run(canary=True, tails="n11x")


def test_check_tails_replays_one_refit_fold(check_env):
    from knee.nb import n10

    inp = check_env["inp"]
    s_old = n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory)
    run_old = json.loads((paths.WORKING / "run.json").read_text())
    assert "tails" not in run_old["config"] and "source" not in s_old  # the n11 check is as it was
    sel = _write_n11x(inp)
    ud = inp / "knee-n11x-final" / "private" / "n11x_final" / f"{ENC}_f0"
    shutil.copy(check_env["unit"] / "pred_b_lite.csv", ud / "pred_b_lite.csv")  # the same model: the same predictions
    _edit_x(inp, 0, lambda st: st["files_sha16"].update(
        {"pred_b_lite.csv": hashlib.sha256((ud / "pred_b_lite.csv").read_bytes()).hexdigest()[:16]}))
    s = n10.check_tails(ENC, fold=None, n=200, encoder_factory=fake_factory, tails="n11x")  # None: the lowest moved fold
    assert s["passed"] is True and s["fold"] == 0 and s["source"] == "n11x" and s["encoder_lr"] == 3.2e-4
    assert s["selection_sha256"] == sel["sha256"] and s["n11_commit"] == COMMIT_X[:12]
    for key in ("max_abs_probability_gap", "mean_abs_probability_gap", "tail_sha16", "head_sha16", "trunk_sha16",
                "weights_file_sha16", "stage_a_stats_sha16", "prediction_file_sha16", "studies"):
        assert s[key] == s_old[key], key
    out = json.loads((paths.WORKING / "agg_n10_tails_check.json").read_text())
    agg._check(out)
    assert set(out) - set(s_old) == {"source", "selection_sha256", "encoder_lr"}
    assert json.loads((paths.WORKING / "run.json").read_text())["config"]["tails"] == "n11x"
    assert not list(paths.WORKING.rglob("*.csv"))
    with pytest.raises(safe_log.StepFailed):  # fold 1 holds no prediction file here
        n10.check_tails(ENC, fold=1, n=200, encoder_factory=fake_factory, tails="n11x")
    with pytest.raises(ValueError):
        n10.check_tails(ENC, fold=None, n=200, encoder_factory=fake_factory)  # fold=None belongs to the extension
    with pytest.raises(ValueError):
        n10.check_tails(ENC, fold=0, tails="n12")
    # a later selection record: the check is refused, and the summary is not written as passed
    (inp / "knee-n11x-select" / "agg_n11x_select.json").write_text(json.dumps(_selection_x(written_at_utc="later")))
    (paths.WORKING / "agg_n10_tails_check.json").unlink()
    with pytest.raises(safe_log.StepFailed):
        n10.check_tails(ENC, fold=0, n=200, encoder_factory=fake_factory, tails="n11x")
    assert not (paths.WORKING / "agg_n10_tails_check.json").exists()
