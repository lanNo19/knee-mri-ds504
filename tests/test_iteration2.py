"""Iteration 2 (plan r2): the settable schedule and patience, N6's new arguments, N13 and its keep rule.

The registered code paths must stay bit for bit those of commit a260cde (iteration 1). Two tests
compare with that commit directly: heads.fit (the a260cde heads.py loaded beside the current one)
and a whole N6 run (the a260cde tree from `git archive`, run in a subprocess on the same data).
Both skip when git or the commit is not available.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee import agg, heads, paths

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
REF = "a260cde"  # iteration 1's code: main when iteration 2 started
FIND = list(range(12))


def _git(*args) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _ref_heads():
    """heads.py exactly as committed at a260cde, as a module of its own (knee.metrics is shared)."""
    src = _git("show", f"{REF}:src/knee/heads.py")
    if src is None:
        pytest.skip("git history with commit a260cde is not available")
    name = "knee._heads_a260cde"
    mod = types.ModuleType(name)
    mod.__package__ = "knee"
    mod.__file__ = f"<{REF}:src/knee/heads.py>"
    sys.modules[name] = mod  # dataclasses look the module up while a class is built
    try:
        exec(compile(src.decode(), mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def _arrays(n=300, D=16, S=4, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 12)) < 0.3).astype(float)
    W = rng.normal(0, 1, (12, D))
    emb = {p: ((y @ W)[:, None, :] + rng.normal(0, 1, (n, S, D))).astype(np.float16) for p in heads.PLANES}
    return emb, y


def _bank(mod, emb, y):
    return mod.Bank(emb, y.astype(np.float32), np.ones_like(y, np.float32), y)


def _fingerprint(r) -> str:
    """One hash over all weights and everything the fit reports."""
    h = hashlib.sha256()
    for k, v in r.model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps([r.best_epoch, repr(r.best_score), r.history, r.epochs_run, r.steps_run,
                         repr(r.final_lr), r.pruned]).encode())
    return h.hexdigest()


@pytest.fixture()
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # as in N6's workers: no thread-order effects in the comparison
    yield
    torch.set_num_threads(n)


# ---------------------------------------------------------------- heads: defaults = a260cde

def test_default_fits_are_bit_identical_to_a260cde(one_thread):
    ref = _ref_heads()
    emb, y = _arrays()
    old_bank, new_bank = _bank(ref, emb, y), _bank(heads, emb, y)
    tr, va, te = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)

    def cases(m):
        C = m.Config
        return {
            "inner gated all b48": lambda b: m.fit(b, C(lr=3e-3), tr, va, FIND, forbidden=te, seed=3),
            "inner mean sag b16": lambda b: m.fit(b, C(pooling="mean", planes="sag", batch_size=16, lr=1e-3),
                                                  tr, va, FIND, forbidden=te, seed=0),
            "inner max sag+cor b128": lambda b: m.fit(b, C(pooling="max", planes="sag+cor", batch_size=128, lr=3e-3),
                                                      tr, va, FIND, forbidden=te, seed=1),
            "inner, max_epochs 15": lambda b: m.fit(b, C(pooling="mean", batch_size=32, lr=3e-3), tr, va, FIND,
                                                    forbidden=te, max_epochs=15, seed=2),
            "demo-like, no stopping rule": lambda b: m.fit(b, C(batch_size=40, lr=1e-3), tr, va, FIND, forbidden=te,
                                                           patience=None, max_epochs=m.MAX_EPOCHS, seed=4,
                                                           on_epoch=lambda ep, sc: False),
            "refit E 7": lambda b: m.fit(b, C(batch_size=24, lr=3e-3), tr, epochs=7, forbidden=te, seed=0),
            # KneeMRI S5 (kneemri.py:193): fixed steps on the RSNA refit's schedule, and E epochs without steps
            "S5 fixed steps": lambda b: m.fit(b, C(planes="sag", batch_size=16, lr=1e-3), np.arange(80), steps=60,
                                              schedule_steps_per_epoch=20, forbidden=te, seed=0),
            "S5 epochs": lambda b: m.fit(b, C(planes="sag", batch_size=16), np.arange(80), epochs=5,
                                         forbidden=te, seed=0),
        }

    old = {k: _fingerprint(f(old_bank)) for k, f in cases(ref).items()}
    new = {k: _fingerprint(f(new_bank)) for k, f in cases(heads).items()}
    assert new == old
    # the registered values passed explicitly take the same path
    C = heads.Config
    explicit = {
        "inner gated all b48": _fingerprint(heads.fit(new_bank, C(lr=3e-3), tr, va, FIND, forbidden=te, seed=3,
                                                      schedule_epochs=30, patience=5)),
        "refit E 7": _fingerprint(heads.fit(new_bank, C(batch_size=24, lr=3e-3), tr, epochs=7, forbidden=te, seed=0,
                                            schedule_epochs=heads.MAX_EPOCHS)),
    }
    assert explicit == {k: old[k] for k in explicit}
    # the schedule itself, bit for bit, including steps past its end
    for spe in (1, 2, 7, 76, 500):
        assert all(heads.schedule_factor(i, spe) == ref.schedule_factor(i, spe)
                   == heads.schedule_factor(i, spe, 30) for i in range(0, 32 * spe + 2))


# ---------------------------------------------------------------- heads: the A1 schedule

def test_schedule_of_8_epochs_warms_up_and_ends_at_zero():
    for spe in (1, 7, 76, 500):
        def f(i, spe=spe):
            return heads.schedule_factor(i, spe, 8)

        # warm-up over the first epoch: the registered linear ramp
        assert all(f(i) == (i + 1) / spe == heads.schedule_factor(i, spe) for i in range(spe))
        assert f(spe - 1) == 1.0 and f(spe) == 1.0
        # then a cosine that falls at every step and is exactly 0 once epoch 8 has ended
        after = [f(i) for i in range(spe, 8 * spe + 1)]
        assert all(a > b for a, b in zip(after, after[1:]))
        assert f(8 * spe - 1) > 0 and f(8 * spe) == 0.0
        assert all(f(i) == 0.0 for i in range(8 * spe, 10 * spe))
        if spe % 2 == 0:  # halfway through the cosine (epoch 4.5) the rate is half its peak
            assert f(9 * spe // 2) == pytest.approx(0.5, abs=1e-12)
    # at E = 4 of 8 the rate has fallen to about 0.6 of its peak (the 30-epoch schedule: about 0.97)
    want = 0.5 * (1 + math.cos(math.pi * (4 * 76 - 1 - 76) / (8 * 76 - 76)))  # = 0.6141
    assert heads.schedule_factor(4 * 76 - 1, 76, 8) == want == pytest.approx(0.614, abs=1e-3)
    assert heads.schedule_factor(4 * 76 - 1, 76) == pytest.approx(0.974, abs=3e-3)


def test_fit_runs_its_own_schedule_and_patience():
    emb, y = _arrays()
    bank = _bank(heads, emb, y)
    tr, va, te = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)
    cfg = heads.Config(pooling="mean", batch_size=32, lr=3e-3)
    spe = math.ceil(len(tr) / cfg.batch_size)
    # an inner fit without a stopping rule runs exactly its schedule: 8 epochs, or the registered 30
    r8 = heads.fit(bank, cfg, tr, va, FIND, forbidden=te, patience=None, schedule_epochs=8)
    assert r8.epochs_run == 8 and len(r8.history) == 8 and r8.steps_run == 8 * spe
    assert r8.final_lr == cfg.lr * heads.schedule_factor(8 * spe - 1, spe, 8)  # the last step's rate
    assert heads.fit(bank, cfg, tr, va, FIND, forbidden=te, patience=None).epochs_run == heads.MAX_EPOCHS
    # a refit of E epochs keeps the 8-epoch schedule and stops at E
    rr = heads.fit(bank, cfg, tr, epochs=3, forbidden=te, schedule_epochs=8)
    assert rr.epochs_run == 3 and not rr.history
    assert rr.final_lr == cfg.lr * heads.schedule_factor(3 * spe - 1, spe, 8)
    assert rr.final_lr < cfg.lr * heads.schedule_factor(3 * spe - 1, spe)
    # patience: with lr 0 the weights and the validation score never change, so the fit stops
    # `patience` epochs after epoch 1; the schedule length caps it
    flat = heads.Config(pooling="mean", batch_size=32, lr=0.0)
    assert heads.fit(bank, flat, tr, va, FIND, forbidden=te, patience=3, schedule_epochs=8).epochs_run == 4
    assert heads.fit(bank, flat, tr, va, FIND, forbidden=te).epochs_run == 1 + heads.PATIENCE
    assert heads.fit(bank, flat, tr, va, FIND, forbidden=te, patience=20, schedule_epochs=8).epochs_run == 8
    with pytest.raises(ValueError):
        heads.fit(bank, cfg, tr, epochs=1, forbidden=te, schedule_epochs=0)


# ---------------------------------------------------------------- N6: search ranges

def test_search_space_overrides_draw_the_registered_trials_unless_widened():
    optuna = pytest.importorskip("optuna")
    from knee.nb import n6

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    U = 20.0

    def trials(draw, n=13):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GPSampler(seed=3, n_startup_trials=10))
        study.enqueue_trial(n6.space_defaults(U, True))

        def objective(t):
            p = draw(t)
            return -abs(math.log10(p["lr"]) + 3) - abs(p["dropout"] - 0.2) - abs(math.log2(p["batch_size"]) - 5) / 4

        study.optimize(objective, n_trials=n)
        return [t.params for t in study.trials], [t.distributions for t in study.trials]

    reg, reg_d = trials(lambda t: n6.suggest(t, U, True))
    # the override code path with the registered range: the same 13 trials (10 start-up, 3 GP)
    same, same_d = trials(lambda t: n6.suggest(t, U, True, {"batch_size": [16, 128]}))
    assert same == reg and same_d == reg_d
    wide_space = {"batch_size": [8, 128], "weight_decay": [1e-6, 1.0]}
    wide, wide_d = trials(lambda t: n6.suggest(t, U, True, wide_space))
    d, r = wide_d[-1], reg_d[-1]
    assert (d["batch_size"].low, d["batch_size"].high, d["batch_size"].log) == (8, 128, True)
    assert (d["weight_decay"].low, d["weight_decay"].high, d["weight_decay"].log) == (1e-6, 1.0, True)
    assert all(d[k] == r[k] for k in ("lr", "dropout", "pos_cap", "attn_hidden"))
    assert wide[0] == reg[0] == n6.space_defaults(U, True)  # trial 0 stays the registered defaults
    assert n6.space(U, True, wide_space) == {**n6.space(U, True), "batch_size": (8, 128, True),
                                              "weight_decay": (1e-6, 1.0, True)}
    assert n6.space(U, False, {"attn_hidden": [32, 256]}) == n6.space(U, False)  # attn_hidden: gated only
    assert n6.check_space_overrides(None) is None and n6.check_space_overrides({}) is None
    for bad in ({"momentum": [0.1, 0.9]}, {"batch_size": [128, 8]}, {"batch_size": [8.5, 128]},
                {"weight_decay": [0, 1]}, {"dropout": [0.0, 1.5]}, {"batch_size": [2, 128]}):
        with pytest.raises(ValueError):  # (a small int would be refused by the agg rules at the end)
            n6.check_space_overrides(bad)


# ---------------------------------------------------------------- N6: a default run = a260cde

@pytest.fixture(scope="module")
def stage_env(tmp_path_factory):
    from tests.synth_stage import make_stage

    root = tmp_path_factory.mktemp("it2stage")
    facts = make_stage(root / "input")
    return root, facts


# the existing synthetic N6 call (tests/test_stage.py), without the demo
N6_ARGS = {"encoders": ["dinov2_vits14"], "folds": [0, 1], "workers": 2, "n_trials": 3,
           "l1_grid": [["mean", "sag"], ["gated", "all"]], "quiet": True}
NEW_UNIT_KEYS = {"schedule_epochs": 30, "patience": 5, "space_overrides": None, "seed": 0, "device": "cpu",
                 "device_name": "cpu"}
RUN_SCRIPT = """
import json, sys
from knee.nb import n6
if __name__ == "__main__":  # N6's workers are spawned and import this file again
    a = json.loads(sys.argv[2])
    a["encoders"], a["folds"], a["l1_grid"] = tuple(a["encoders"]), tuple(a["folds"]), [tuple(g) for g in a["l1_grid"]]
    n6.run("n6a", root=sys.argv[1], **a)
"""


def _without(d: dict, keys) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def _no_seconds(o):
    if isinstance(o, dict):
        return {k: _no_seconds(v) for k, v in o.items() if k != "seconds"}
    return o


def test_n6_run_with_defaults_reproduces_a260cde(stage_env, tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from knee.nb import n6

    root, _ = stage_env
    archive = _git("archive", "--format=tar", REF, "src")
    if archive is None:
        pytest.skip("git history with commit a260cde is not available")
    ref = tmp_path / "ref"
    ref.mkdir()
    (tmp_path / "ref.tar").write_bytes(archive)
    with tarfile.open(tmp_path / "ref.tar") as tf:
        tf.extractall(ref, filter="data")
    script_dir = tmp_path / "script"
    script_dir.mkdir()
    (script_dir / "run_ref.py").write_text(RUN_SCRIPT)
    env = {**os.environ, "PYTHONPATH": str(ref / "src"), "KNEE_WORKING": str(tmp_path / "old")}
    subprocess.run([sys.executable, str(script_dir / "run_ref.py"), str(root / "input"), json.dumps(N6_ARGS)],
                   env=env, check=True, capture_output=True, timeout=1500)
    # the new code, in this process, with no new argument
    monkeypatch.setattr(paths, "WORKING", tmp_path / "new")
    a = {**N6_ARGS, "encoders": tuple(N6_ARGS["encoders"]), "folds": tuple(N6_ARGS["folds"]),
         "l1_grid": [tuple(g) for g in N6_ARGS["l1_grid"]]}
    s = n6.run("n6a", root=root / "input", **a)
    assert s["all_units_ok"]
    old_w, new_w = tmp_path / "old", tmp_path / "new"
    for k in N6_ARGS["folds"]:
        uo, un = (w / "private" / "n6a" / "units" / f"dinov2_vits14_f{k}" for w in (old_w, new_w))
        for f in ("predictions.csv", "trials.csv", "l1.csv", "head.pt", "stats.pt"):
            assert (uo / f).read_bytes() == (un / f).read_bytes(), f
        co, cn = (json.loads((u / "config.json").read_text()) for u in (uo, un))
        assert _without(cn, NEW_UNIT_KEYS) | {"seconds": None} == co | {"seconds": None}
        assert {k2: cn[k2] for k2 in NEW_UNIT_KEYS} == NEW_UNIT_KEYS  # the values used, recorded
        ao, an = (json.loads((w / f"agg_n6a_dinov2_vits14_f{k}.json").read_text()) for w in (old_w, new_w))
        assert _without(an, NEW_UNIT_KEYS) | {"seconds": None} == ao | {"seconds": None}
    ro, rn = (json.loads((w / "agg_n6a.json").read_text()) for w in (old_w, new_w))
    assert _no_seconds(rn) == _no_seconds(ro)
    jo, jn = (json.loads((w / "run.json").read_text())["config"] for w in (old_w, new_w))
    assert jn == jo and jn["test_overrides"] is True  # (true from the test-only n_trials and l1_grid, as before)


# ---------------------------------------------------------------- N6: the new settings reach every fit

def test_n6_unit_passes_schedule_patience_and_ranges_to_every_fit(stage_env, tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from knee.nb import n6

    root, _ = stage_env
    fits, drawn = [], []
    real_fit, real_suggest = heads.fit, n6.suggest

    def fit(*a, **kw):
        fits.append({k: kw.get(k) for k in ("epochs", "schedule_epochs", "patience", "max_epochs", "seed")})
        return real_fit(*a, **kw)

    def suggest(trial, U, gated, overrides=None):
        drawn.append(overrides)
        return real_suggest(trial, U, gated, overrides)

    monkeypatch.setattr(heads, "fit", fit)
    monkeypatch.setattr(n6, "suggest", suggest)
    monkeypatch.setattr(n6, "_W", {"root": str(root / "input"), "expect": None, "device": "cpu", "shared": None})
    # (batch sizes from 64 up keep this CPU test fast; the A7 range itself is checked in the test above)
    ovr = n6.check_space_overrides({"batch_size": [64, 128], "weight_decay": [1e-6, 1.0]})
    od = tmp_path / "dinov2_vits14_f0"
    task = {"id": "dinov2_vits14_f0", "encoder": "dinov2_vits14", "fold": 0, "seed": 100, "eligible": FIND,
            "soft_stop": time.time() + 3600, "backstop": None, "mode": "nested", "unit_s_est": None,
            "n_trials": 2, "l1_grid": [("gated", "all")], "schedule_epochs": 8, "patience": 3,
            "space_overrides": ovr, "dir": str(od), "fold0_in_run": True}
    r = n6._unit(task, od)
    assert r["ok"]
    inner = [c for c in fits if c["epochs"] is None]
    refit = [c for c in fits if c["epochs"] is not None]
    assert len(inner) == 4 * 3  # 4 inner folds x (one L1 config + two trials)
    assert all((c["schedule_epochs"], c["patience"], c["max_epochs"]) == (8, 3, None) for c in inner)
    assert {c["seed"] for c in inner} == {100, 101, 102, 103}  # seed + inner fold
    assert len(refit) == 1 and (refit[0]["schedule_epochs"], refit[0]["seed"]) == (8, 0)
    assert drawn == [ovr, ovr]  # every trial drew from the widened ranges
    u = json.loads((od / "config.json").read_text())
    assert (u["schedule_epochs"], u["patience"], u["seed"], u["device"], u["device_name"]) == (8, 3, 100, "cpu", "cpu")
    assert u["space_overrides"] == {"batch_size": [64, 128], "weight_decay": [1e-06, 1.0]}
    t1 = pd.read_csv(od / "trials.csv").iloc[1]  # the one drawn trial lies in the widened ranges
    assert 64 <= t1["batch_size"] <= 128 and 1e-6 <= t1["weight_decay"] <= 1.0
    assert 1 <= u["E"] <= 8 and refit[0]["epochs"] == u["E"]
    assert all(1 <= e <= 8 for b in pd.read_csv(od / "trials.csv")["best_epochs"] for e in json.loads(b))
    agg._check(r["summary"])  # patience 3 is a setting, not a count: the per-unit agg file accepts it
    monkeypatch.setattr(paths, "WORKING", tmp_path / "work")
    for bad in ({"schedule_epochs": 0}, {"patience": 2.5}, {"overrides_reason": "two\nlines"},
                {"space_overrides": {"batch_size": [1, 64]}}):
        with pytest.raises(ValueError):  # refused before run.json or any fit
            n6.run("n6x", root=root / "input", **bad)
    assert not (tmp_path / "work").exists()


# ---------------------------------------------------------------- N13

def test_n13_arms_are_the_plan_r2_settings():
    from knee.nb import common, n13

    a1, null, cpu0 = n13.arm_spec("a1"), n13.arm_spec("null"), n13.arm_spec("cpu0")
    assert (a1["seed"], a1["n_trials"], a1["schedule_epochs"], a1["patience"]) == (0, 1, 8, 3)
    assert (null["seed"], null["n_trials"], null["schedule_epochs"], null["patience"]) == (100, 1, None, None)
    assert (cpu0["seed"], cpu0["n_trials"], cpu0["schedule_epochs"], cpu0["patience"]) == (0, 1, None, None)
    assert a1["space_overrides"] is None and null["space_overrides"] is None
    a7 = n13.arm_spec("a7", schedule_epochs=8, patience=3)
    assert (a7["seed"], a7["n_trials"], a7["schedule_epochs"], a7["patience"]) == (0, None, 8, 3)
    assert a7["space_overrides"] == {"batch_size": [8, 128], "weight_decay": [1e-6, 1.0]}
    assert a1["reason"] == "iteration 2, plan r2: defaults only, see iteration_plan_r2.md"
    names = {n13.arm_spec(a, *((8, 3) if a == "a7" else ()))["name"] for a in n13.ARMS}
    assert names == {"n13_a1", "n13_null", "n13_cpu0", "n13_a7"}  # never n6a or n6b
    assert not any(common.is_rerun(Path("private") / n / "units" / "convnext_tiny_f0") for n in names)
    assert common.is_rerun(Path("private") / "n13_a1_r" / "units" / "convnext_tiny_f0")  # a unit rerun
    with pytest.raises(ValueError):
        n13.arm_spec("a7")  # A7 needs the schedule chosen in iteration 2
    with pytest.raises(ValueError):
        n13.arm_spec("a1", schedule_epochs=30)  # fixed by the plan
    with pytest.raises(ValueError):
        n13.arm_spec("a2")
    with pytest.raises(ValueError):
        n13.run("a1", rerun=True)  # a rerun names its folds


def test_n13_a1_arm_end_to_end(stage_env, monkeypatch, capfd):
    pytest.importorskip("optuna")
    from knee.nb import common, n13

    root, facts = stage_env
    monkeypatch.setattr(paths, "WORKING", root / "working_n13")
    out = n13.run("a1", root=root / "input", workers=2, encoders=("dinov2_vits14",), folds=(0, 1))
    printed = capfd.readouterr()
    files = sorted(p.name for p in paths.WORKING.iterdir() if p.is_file())
    text = printed.out + printed.err + "".join((paths.WORKING / f).read_text() for f in files)
    assert not [u for u in facts["uids"] if u in text]  # no study identifier anywhere Claude reads
    assert files == ["agg_n13_a1.json", "agg_n13_a1_dinov2_vits14_f0.json", "agg_n13_a1_dinov2_vits14_f1.json",
                     "agg_n13_a1_inner.json", "run.json"]
    assert [p.name for p in (paths.WORKING / "private").iterdir() if p.is_dir()] == ["n13_a1"]
    rj = json.loads((paths.WORKING / "run.json").read_text())
    c = rj["config"]
    assert rj["step"] == "n13_a1" and c["test_overrides"] is True and c["test_overrides_reason"] == n13.REASON
    assert (c["schedule_epochs"], c["patience"], c["seed"], c["encoders"]) == (8, 3, 0, ["dinov2_vits14"])
    assert "space_overrides" not in c  # recorded only when given
    assert out["all_units_ok"] and out["leakage_assertion_passed"] and out["all_nested"]
    assert out["settings"] == {"l1": "gated|all", "n_trials": 1, "schedule_epochs": 8, "patience": 3, "seed": 0,
                               "space_overrides": None}
    assert json.loads((paths.WORKING / "agg_n13_a1_inner.json").read_text()) == json.loads(json.dumps(out))
    for k in (0, 1):
        ud = paths.WORKING / "private" / "n13_a1" / "units" / f"dinov2_vits14_f{k}"
        u, cfg = out["units"][f"dinov2_vits14_f{k}"], json.loads((ud / "config.json").read_text())
        assert (u["schedule_epochs"], u["patience"], u["seed"], u["device"], u["device_name"]) == (8, 3, 0, "cpu", "cpu")
        # one trial, the defaults: trial 0 is the unit's whole search
        assert u["trial0_inner_score"] == cfg["default_trial_score"] == cfg["best_inner_score"]
        assert out["trial0_inner_scores"]["dinov2_vits14"][k] == u["trial0_inner_score"]
        defaults = heads.Config().to_dict()  # the L2 defaults (batch 48), gated attention over all planes
        assert _without(cfg["config"], {"pos_cap"}) == _without(defaults, {"pos_cap"})
        assert 1 <= u["E"] <= 8 and u["E"] == cfg["E"]
        trials, l1 = pd.read_csv(ud / "trials.csv"), pd.read_csv(ud / "l1.csv")
        assert len(trials) == 1 and len(l1) == 1 and all(1 <= e <= 8 for e in json.loads(trials["best_epochs"][0]))
        assert trials["value"][0] == l1["inner_score"][0]  # L1's one config is trial 0: same fits, same seeds
        assert u["head_sha16"] == common.sha16(ud / "head.pt")
    # N10 reads these units with its own, unchanged loader (n10._units, as at the bundle's commit)
    from knee.nb import n10

    found = n10._units("dinov2_vits14", paths.WORKING)
    assert sorted(found) == [0, 1]
    for u in found.values():
        assert (u["config"].pooling, u["config"].planes, u["config"].batch_size) == ("gated", "all", 48)
        model = heads.make_head(u["config"], 64, 3)  # the synthetic embeddings have 64 features
        model.load_state_dict(torch.load(u["dir"] / "head.pt", map_location="cpu"))
    n10_like = {"heads_sha16": {f"fold_{k}": u["head_sha16"] for k, u in sorted(found.items())}}  # as n10.py:126
    chk = n13.check_heads(n10_like, out, encoder="dinov2_vits14")
    assert chk["mismatched_folds"] == [] and chk["folds"] == [0, 1]
    assert chk["match"] is False  # equal hashes, but a submission needs all five S2 folds


def _unit_scores(base: float, diffs) -> dict:
    return {k: base + d for k, d in enumerate(diffs)}


def test_n13_keep_rule_on_hand_made_numbers():
    from knee.nb import n13

    base = {"dinov2_vits14": {0: 0.70, 1: 0.72, 2: 0.74, 3: 0.71, 4: 0.73},
            "convnext_tiny": {0: 0.75, 1: 0.76, 2: 0.74, 3: 0.77, 4: 0.75}}
    nd = {"dinov2_vits14": [0.002, -0.002, 0.001, -0.001, 0.0], "convnext_tiny": [0.003, -0.003, 0.0, 0.001, -0.001]}
    null = {e: {k: base[e][k] + nd[e][k] for k in range(5)} for e in base}
    s_hand = math.sqrt(30e-6 / 9)  # the 10 differences have mean 0 and squares summing to 30e-6
    a1 = {"dinov2_vits14": {k: base["dinov2_vits14"][k] + d for k, d in enumerate([0.010, 0.008, 0.006, 0.012, -0.001])},
          "convnext_tiny": {k: base["convnext_tiny"][k] + d for k, d in enumerate([0.004, 0.006, 0.003, 0.005, 0.006])}}
    r = n13.decide(a1, base, null)
    assert r["dinov2_vits14"]["s"] == pytest.approx(s_hand, rel=1e-9) == r["convnext_tiny"]["s"]
    assert 2 * s_hand / math.sqrt(5) == pytest.approx(0.001633, abs=1e-6)  # below 0.005: min_gain binds
    for e in base:
        assert r[e]["threshold"] == 0.005
    assert r["dinov2_vits14"]["gain"] == pytest.approx(0.007) and r["dinov2_vits14"]["folds_up"] == 4
    assert r["dinov2_vits14"]["keep"] is True
    assert r["convnext_tiny"]["gain"] == pytest.approx(0.0048) and r["convnext_tiny"]["folds_up"] == 5
    assert r["convnext_tiny"]["keep"] is False  # every fold up, but the mean gain is under 0.005
    assert r["dinov2_vits14"]["per_fold"][4] == pytest.approx(-0.001)
    # a large gain in only 3 of 5 folds is not kept
    a1b = {**a1, "convnext_tiny": {k: base["convnext_tiny"][k] + d for k, d in enumerate([0.02, 0.02, 0.02, -0.001, -0.001])}}
    rb = n13.decide(a1b, base, null)["convnext_tiny"]
    assert rb["gain"] == pytest.approx(0.0116) and rb["folds_up"] == 3 and rb["keep"] is False
    # a wide null spread raises the bar above min_gain: the same 0.007 gain is then not kept
    wide = {e: {k: base[e][k] + d for k, d in enumerate([0.01, -0.01, 0.01, -0.01, 0.0])} for e in base}
    rw = n13.decide(a1, base, wide)["dinov2_vits14"]
    assert rw["s"] == pytest.approx(math.sqrt(8e-4 / 9), rel=1e-9)
    assert rw["threshold"] == pytest.approx(2 * math.sqrt(8e-4 / 9) / math.sqrt(5), rel=1e-9)  # about 0.0084
    assert rw["keep"] is False
    # strictly greater: a gain equal to the threshold is not kept (dyadic numbers, so exact)
    b0 = {e: {k: 0.5 for k in range(5)} for e in base}
    up = {e: {k: 0.5078125 for k in range(5)} for e in base}
    assert n13.decide(up, b0, b0, min_gain=0.0078125)["dinov2_vits14"]["keep"] is False
    assert n13.decide(up, b0, b0, min_gain=0.0078)["dinov2_vits14"]["keep"] is True
    # JSON fold keys ("0" or "fold_0") give the same result
    as_json = {e: {f"fold_{k}": v for k, v in m.items()} for e, m in a1.items()}
    assert n13.decide(as_json, json.loads(json.dumps(base)), null) == r
    # the rule needs every unit: a missing fold, a missing score or a missing null encoder is refused
    for bad in (({**a1, "convnext_tiny": {k: v for k, v in a1["convnext_tiny"].items() if k != 2}}, base, null),
                ({**a1, "dinov2_vits14": {**a1["dinov2_vits14"], 1: None}}, base, null),
                (a1, base, {"dinov2_vits14": null["dinov2_vits14"]}),
                ({}, base, null)):
        with pytest.raises(ValueError):
            n13.decide(*bad)
    # trial0_scores reads n6's per-unit summaries (e.g. knee-n6a's agg_n6a_<encoder>_f<k>.json)
    units = [{"encoder": e, "fold": k, "default_trial_score": v} for e, m in base.items() for k, v in m.items()]
    assert n13.trial0_scores(units) == base


def test_n13_check_heads():
    from knee.nb import n13

    inner = {"units": {**{f"convnext_tiny_f{k}": {"ok": True, "head_sha16": f"c{k}"} for k in range(5)},
                       **{f"dinov2_vits14_f{k}": {"ok": True, "head_sha16": f"d{k}"} for k in range(5)}}}
    n10 = {"heads_sha16": {f"fold_{k}": f"c{k}" for k in range(5)}}
    assert n13.check_heads(n10, inner) == {"encoder": "convnext_tiny", "folds": [0, 1, 2, 3, 4],
                                           "mismatched_folds": [], "match": True}
    other = {"heads_sha16": {**n10["heads_sha16"], "fold_3": "x"}}  # e.g. a knee-n6a head picked up beside N13
    assert n13.check_heads(other, inner)["mismatched_folds"] == [3] and not n13.check_heads(other, inner)["match"]
    short = {"heads_sha16": {f"fold_{k}": f"c{k}" for k in range(4)}}
    assert n13.check_heads(short, inner)["mismatched_folds"] == [4]
    failed = {"units": {**inner["units"], "convnext_tiny_f1": {"ok": False, "error": "x"}}}
    assert not n13.check_heads(n10, failed)["match"]
    assert not n13.check_heads(n10, inner, encoder="dinov2_vits14")["match"]
