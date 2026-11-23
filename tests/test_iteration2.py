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
    with pytest.raises(ValueError):  # a refit past its schedule would train at learning rate 0
        heads.fit(bank, cfg, tr, epochs=9, forbidden=te, schedule_epochs=8)
    assert heads.fit(bank, cfg, tr, epochs=9, forbidden=te).epochs_run == 9  # no schedule given: as registered


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
    # (batch sizes from 32 up keep this CPU test fast and hold trial 0's 48; the A7 range is checked above)
    ovr = n6.check_space_overrides({"batch_size": [32, 128], "weight_decay": [1e-6, 1.0]})
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
    assert u["space_overrides"] == {"batch_size": [32, 128], "weight_decay": [1e-06, 1.0]}
    t1 = pd.read_csv(od / "trials.csv").iloc[1]  # the one drawn trial lies in the widened ranges
    assert 32 <= t1["batch_size"] <= 128 and 1e-6 <= t1["weight_decay"] <= 1.0
    assert 1 <= u["E"] <= 8 and refit[0]["epochs"] == u["E"]
    assert all(1 <= e <= 8 for b in pd.read_csv(od / "trials.csv")["best_epochs"] for e in json.loads(b))
    agg._check(r["summary"])  # patience 3 is a setting, not a count: the per-unit agg file accepts it
    monkeypatch.setattr(paths, "WORKING", tmp_path / "work")
    for bad in ({"schedule_epochs": 0}, {"patience": 2.5}, {"overrides_reason": "two\nlines"},
                {"space_overrides": {"batch_size": [1, 64]}}, {"extra_config": {"seed": 1}}):
        with pytest.raises(ValueError):  # refused before run.json or any fit
            n6.run("n6x", root=root / "input", **bad)
    assert not (tmp_path / "work").exists()
    from knee import safe_log

    with pytest.raises(safe_log.StepFailed):  # a range without the registered default (trial 0): no unit starts
        n6.run("n6x", root=root / "input", space_overrides={"batch_size": [64, 128]}, n_trials=1,
               l1_grid=[("gated", "all")], quiet=True)
    assert not (tmp_path / "work" / "private" / "n6x" / "units").exists()


# ---------------------------------------------------------------- N13

MAIN = ("dinov2_vits14", "convnext_tiny")
ENV = {"kaggle_image_git_commit": "0f1e2d3c", "kaggle_image_build_date": "20251101-000000", "python": "3.11.13",
       "torch": "2.6.0+cu124", "cuda": "12.4", "cudnn": 90100, "device_name": "Tesla T4"}


def _summary(arm, scores, env=ENV, failed=(), step=None, sett=None):
    """An N13 summary shaped as inner_summary writes it, from {encoder: {fold: score}} (hand-made)."""
    from knee.nb import n13

    units, heads_sha = {}, {}
    spec = n13.arm_spec(arm, *((8, 3) if arm == "a7" else ()))
    step = step or spec["name"]
    for e, m in scores.items():
        for k, v in m.items():
            uid = f"{e}_f{k}"
            if (e, k) in failed:
                units[uid] = {"ok": False, "skipped": False, "error": "RuntimeError at n6.py:1 in _unit()"}
                continue
            sha = hashlib.sha256(f"{step}/{uid}".encode()).hexdigest()[:16]
            units[uid] = {"ok": True, "trial0_inner_score": v, "head_sha16": sha}
            heads_sha.setdefault(e, {})[f"fold_{k}"] = sha
    return {"arm": arm, "step": step, "settings": sett or n13.settings(spec), "environment": env,
            "knee_commit": "c0ffee", "heads_sha16": heads_sha, "units": units}


def _scores(base: dict, diffs: dict) -> dict:
    """4-decimal scores (as stored): base + diffs, per encoder and fold."""
    return {e: {k: round(base[e][k] + d, 4) for k, d in enumerate(diffs[e])} for e in base}


BASE = {"dinov2_vits14": {0: 0.70, 1: 0.72, 2: 0.74, 3: 0.71, 4: 0.73},
        "convnext_tiny": {0: 0.75, 1: 0.76, 2: 0.74, 3: 0.77, 4: 0.75}}
NULL_D = {"dinov2_vits14": [0.002, -0.002, 0.001, -0.001, 0.0], "convnext_tiny": [0.003, -0.003, 0.0, 0.001, -0.001]}


def test_n13_arms_are_the_plan_r3_settings():
    from knee.nb import common, n13

    a1, base0, null = (n13.arm_spec(a) for a in ("a1", "base0", "null"))
    assert (a1["seed"], a1["n_trials"], a1["schedule_epochs"], a1["patience"]) == (0, 1, 8, 3)
    assert (base0["seed"], base0["n_trials"], base0["schedule_epochs"], base0["patience"]) == (0, 1, None, None)
    assert (null["seed"], null["n_trials"], null["schedule_epochs"], null["patience"]) == (100, 1, None, None)
    assert a1["encoders"] == base0["encoders"] == null["encoders"] == MAIN
    assert a1["space_overrides"] is base0["space_overrides"] is null["space_overrides"] is None
    assert n13.settings(base0) == {"l1": "gated|all", "n_trials": 1, "schedule_epochs": 30, "patience": 5, "seed": 0,
                                   "space_overrides": None}
    assert n13.settings(a1)["schedule_epochs"] == 8 and n13.settings(null)["seed"] == 100
    a7 = n13.arm_spec("a7", schedule_epochs=8, patience=3)
    assert (a7["seed"], a7["n_trials"], a7["schedule_epochs"], a7["patience"]) == (0, None, 8, 3)
    assert a7["encoders"] == ("convnext_tiny",) and n13.settings(a7)["n_trials"] == 30
    assert a7["space_overrides"] == {"batch_size": [8, 128], "weight_decay": [1e-6, 1.0]}
    assert a1["reason"] == "iteration 2, plan r3: defaults only, see the iteration plan, section 9"
    names = {n13.arm_spec(a, *((8, 3) if a == "a7" else ()))["name"] for a in n13.ARMS}
    assert names == {"n13_a1", "n13_base0", "n13_null", "n13_a7"}  # never n6a or n6b
    assert not any(common.is_rerun(Path("private") / n / "units" / "convnext_tiny_f0") for n in names)
    assert all(common.is_rerun(Path("private") / r / "units" / "convnext_tiny_f0") for r in ("n13_a1_r", "n13_a1_r2"))
    for bad in (lambda: n13.arm_spec("a7"),  # A7 needs the schedule written in the plan
                lambda: n13.arm_spec("a1", schedule_epochs=30),  # fixed by the plan
                lambda: n13.arm_spec("cpu0"),  # now base0
                lambda: n13.arm_spec("a2"),
                lambda: n13.run("a1", rerun=True, folds=(2,)),  # a rerun names its encoders ...
                lambda: n13.run("a1", rerun=1, encoders=("convnext_tiny",)),  # ... and its folds
                lambda: n13.run("a7", schedule_epochs=8, patience=3, encoders=MAIN),  # ConvNeXt-T only
                lambda: n13.run("a1", encoders=("vit_s16",))):
        with pytest.raises(ValueError):
            bad()


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
    # the environment: in run.json and in the summary block, the same fields
    assert (c["arm"], c["rerun"]) == ("a1", 0) and c["environment"] == out["environment"] == n13.environment()
    assert set(out["environment"]) == set(n13.ENV_FIELDS) and out["environment"]["device_name"] == "cpu"
    assert out["all_units_ok"] and out["leakage_assertion_passed"] and out["all_nested"]
    assert out["settings"] == n13.settings(n13.arm_spec("a1")) and out["encoders"] == ["dinov2_vits14"]
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
        assert u["head_sha16"] == common.sha16(ud / "head.pt") == out["heads_sha16"]["dinov2_vits14"][f"fold_{k}"]
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


def test_n13_keep_rule_on_hand_made_numbers():
    from knee.nb import n13

    null = _scores(BASE, NULL_D)
    a1 = _scores(BASE, {"dinov2_vits14": [0.010, 0.008, 0.006, 0.012, -0.001],
                        "convnext_tiny": [0.004, 0.006, 0.003, 0.005, 0.006]})
    r = n13.decide(_summary("a1", a1), _summary("base0", BASE), _summary("null", null))
    s_hand = math.sqrt(30e-6 / 9)  # the 10 differences: mean 0, squares summing to 30e-6, n - 1 divisor
    assert r["status"] == "decided" and r["environment"] == ENV and r["same_knee_commit"] is True
    assert r["s"] == pytest.approx(s_hand, rel=1e-6) and r["s_divisor"] == "n - 1 (sample SD)"
    assert r["n_null"] == 10 and r["null_mean"] == pytest.approx(0.0, abs=1e-12)
    assert r["null_mean_se"] == pytest.approx(s_hand / math.sqrt(10), rel=1e-6)
    assert 2 * s_hand / math.sqrt(5) == pytest.approx(0.001633, abs=1e-6)  # below 0.005: min_gain binds
    dino, conv = r["encoders"]["dinov2_vits14"], r["encoders"]["convnext_tiny"]
    assert r["threshold"] == dino["threshold"] == conv["threshold"] == 0.005
    assert (dino["gain"], dino["folds_up"], dino["keep"]) == (0.007, 4, True)
    assert (conv["gain"], conv["folds_up"], conv["keep"]) == (0.0048, 5, False)  # every fold up, gain under 0.005
    assert dino["s"] == conv["s"] == r["s"] and dino["per_fold"] == {0: 0.01, 1: 0.008, 2: 0.006, 3: 0.012, 4: -0.001}
    # a large gain in only 3 of 5 folds is not kept
    a1b = _scores(BASE, {"dinov2_vits14": [0.0] * 5, "convnext_tiny": [0.02, 0.02, 0.02, -0.001, -0.001]})
    rb = n13.decide(_summary("a1", a1b), _summary("base0", BASE), _summary("null", null))["encoders"]["convnext_tiny"]
    assert (rb["gain"], rb["folds_up"], rb["keep"]) == (0.0116, 3, False)
    # a wide null spread raises the bar above min_gain: threshold 2 s / sqrt(5) = 0.0084 at 4 decimals
    wide = _scores(BASE, {e: [0.01, -0.01, 0.01, -0.01, 0.0] for e in MAIN})
    rw = n13.decide(_summary("a1", a1), _summary("base0", BASE), _summary("null", wide))
    assert rw["s"] == pytest.approx(math.sqrt(8e-4 / 9), rel=1e-6) and rw["threshold"] == 0.0084
    assert rw["threshold_unrounded"] == pytest.approx(2 * math.sqrt(8e-4 / 9) / math.sqrt(5), rel=1e-6)
    assert rw["encoders"]["dinov2_vits14"]["keep"] is False  # the same 0.007 gain
    # the result is the same from the JSON files
    files = [json.loads(json.dumps(_summary(a, sc))) for a, sc in (("a1", a1), ("base0", BASE), ("null", null))]
    assert json.loads(json.dumps(n13.decide(*files))) == json.loads(json.dumps(r))


def test_n13_keep_rule_compares_at_4_decimals():
    from knee.nb import n13

    null = _scores(BASE, NULL_D)  # threshold 0.005 (min_gain binds)
    # review r1, S2: decimal gain exactly 0.0050 (float 0.0050000000000000044) and 5 folds up: a tie, not kept
    tie_a1, tie_b = [0.6974, 0.7422, 0.773, 0.7649, 0.7664], [0.6937, 0.7382, 0.7667, 0.7621, 0.7582]
    assert sum(a - b for a, b in zip(tie_a1, tie_b)) / 5 > 0.005  # float noise above the bar
    base = {**BASE, "convnext_tiny": dict(enumerate(tie_b))}
    a1 = {"dinov2_vits14": dict(BASE["dinov2_vits14"]), "convnext_tiny": dict(enumerate(tie_a1))}
    null_t = _scores(base, NULL_D)
    r = n13.decide(_summary("a1", a1), _summary("base0", base), _summary("null", null_t))["encoders"]["convnext_tiny"]
    assert (r["gain"], r["threshold"], r["folds_up"], r["keep"]) == (0.005, 0.005, 5, False)
    # one unit of the 4th decimal above the bar is kept
    a1_up = {**a1, "convnext_tiny": {k: round(v + (0.0005 if k == 0 else 0.0), 4) for k, v in a1["convnext_tiny"].items()}}
    r_up = n13.decide(_summary("a1", a1_up), _summary("base0", base), _summary("null", null_t))["encoders"]["convnext_tiny"]
    assert (r_up["gain"], r_up["keep"]) == (0.0051, True)
    # a gain that rounds to the threshold at 4 decimals is a tie too (0.00502 -> 0.0050)
    a1_r = _scores(BASE, {"dinov2_vits14": [0.0051, 0.0051, 0.0050, 0.0049, 0.0050], "convnext_tiny": [0.0] * 5})
    rr = n13.decide(_summary("a1", a1_r), _summary("base0", BASE), _summary("null", null))["encoders"]["dinov2_vits14"]
    assert rr["gain_unrounded"] == pytest.approx(0.00502) and (rr["gain"], rr["keep"]) == (0.005, False)
    # a threshold tie from s: gain 0.0084 against threshold 0.0084 is not kept, 0.0085 is
    wide = _scores(BASE, {e: [0.01, -0.01, 0.01, -0.01, 0.0] for e in MAIN})
    for d, kept in ((0.0084, False), (0.0085, True)):
        a1_w = _scores(BASE, {"dinov2_vits14": [d] * 5, "convnext_tiny": [0.0] * 5})
        rw = n13.decide(_summary("a1", a1_w), _summary("base0", BASE), _summary("null", wide))["encoders"]["dinov2_vits14"]
        assert (rw["threshold"], rw["gain"], rw["keep"]) == (0.0084, d, kept)
    # "higher in a fold" is round(a1 - base0, 4) > 0: a difference below half a unit of the 4th decimal is a tie
    a1_f = _scores(BASE, {"dinov2_vits14": [0.01, 0.01, 0.01, 0.0, 0.0], "convnext_tiny": [0.0] * 5})
    a1_f["dinov2_vits14"][3] += 0.00004  # not a 4-decimal value: up to float rounding, still level
    rf = n13.decide(_summary("a1", a1_f), _summary("base0", BASE), _summary("null", null))["encoders"]["dinov2_vits14"]
    assert rf["per_fold"][3] == 0.0 and rf["folds_up"] == 3 and rf["keep"] is False


def test_n13_keep_rule_refuses_other_units_and_environments():
    from knee.nb import n13

    null = _scores(BASE, NULL_D)
    a1 = _scores(BASE, {"dinov2_vits14": [0.010, 0.008, 0.006, 0.012, -0.001],
                        "convnext_tiny": [0.004, 0.006, 0.003, 0.005, 0.006]})
    b, n = _summary("base0", BASE), _summary("null", null)
    ok = n13.decide(_summary("a1", a1), b, n)
    bad_inputs = [
        (_summary("a1", {"convnext_tiny": a1["convnext_tiny"]}), b, n),  # review r1, S1: one encoder only
        (_summary("a1", {**a1, "vit_s16": a1["convnext_tiny"]}), b, n),  # S1: a third encoder
        (_summary("a1", {**a1, "convnext_tiny": {k: v for k, v in a1["convnext_tiny"].items() if k != 2}}), b, n),
        (_summary("a1", {**a1, "convnext_tiny": {**a1["convnext_tiny"], 5: 0.7}}), b, n),  # a sixth fold
        (_summary("a1", {**a1, "dinov2_vits14": {**a1["dinov2_vits14"], 1: None}}), b, n),  # no score
        (_summary("a1", a1, failed={("convnext_tiny", 2)}), b, n),  # an unfinished unit, not rerun
        (_summary("a1", a1), _summary("null", null), n),  # the null arm passed as base0
        (_summary("a1", a1, sett={**n13.settings(n13.arm_spec("a1")), "schedule_epochs": 30}), b, n),
        ({**_summary("a1", a1), "environment": None}, b, n),  # no environment record
        (_summary("a1", a1), b, {"dinov2_vits14": null["dinov2_vits14"]}),  # not a summary
    ]
    for args in bad_inputs:
        with pytest.raises(ValueError):
            n13.decide(*args)
    with pytest.raises(ValueError, match="exactly one unit"):
        n13.decide(*bad_inputs[0])
    # a rerun replaces a failed unit (and only a failed one)
    orig = _summary("a1", a1, failed={("convnext_tiny", 2)})
    rerun = _summary("a1", {"convnext_tiny": {2: a1["convnext_tiny"][2]}}, step="n13_a1_r")
    r = n13.decide([orig, rerun], b, n)
    assert r["encoders"] == ok["encoders"] and r["arms"]["a1"]["steps"] == ["n13_a1", "n13_a1_r"]
    assert r["arms"]["a1"]["heads_sha16"]["convnext_tiny"]["fold_2"] == rerun["heads_sha16"]["convnext_tiny"]["fold_2"]
    twice = _summary("a1", {"convnext_tiny": {3: a1["convnext_tiny"][3]}}, step="n13_a1_r")
    with pytest.raises(ValueError, match="two finished units"):
        n13.decide([_summary("a1", a1), twice], b, n)  # a second finished unit for one fold: refused
    # environments: every field equal, or no decision
    other = dict(ENV, torch="2.6.1+cu124")
    for args in ((_summary("a1", a1), b, _summary("null", null, env=other)),
                 ([orig, _summary("a1", {"convnext_tiny": {2: a1["convnext_tiny"][2]}}, step="n13_a1_r", env=other)], b, n)):
        r = n13.decide(*args)
        assert r["status"] == "env_mismatch" and r["env_differences"] == ["torch"]
        assert all(v["keep"] is False and v["gain"] is None for v in r["encoders"].values())
    r = n13.decide(_summary("a1", a1), b, _summary("null", null, env=dict(ENV, device_name="cpu")))
    assert r["status"] == "env_mismatch" and r["env_differences"] == ["device_name"]


def test_n13_stage_a_check_is_descriptive():
    from knee.nb import n13

    stage_a = [{"encoder": e, "fold": k, "tuning": "nested", "l1_winner": {"pooling": "gated", "planes": "all"},
                "default_trial_score": v, "best_inner_score": v + 0.003} for e, m in BASE.items() for k, v in m.items()]
    same = n13.stage_a_check(_summary("base0", BASE), stage_a)
    assert same["descriptive"] is True and same["used_in_rule"] is False
    assert same["all_equal_at_4_decimals"] is True and same["max_abs_difference"] == 0
    moved = _scores(BASE, {"dinov2_vits14": [0.0, 0.0003, 0.0, 0.0, 0.0], "convnext_tiny": [0.0, 0.0, 0.0, 0.0, -0.0012]})
    r = n13.stage_a_check(_summary("base0", moved), stage_a)
    assert r["per_unit"]["dinov2_vits14_f1"] == 0.0003 and r["per_unit"]["convnext_tiny_f4"] == -0.0012
    assert r["all_equal_at_4_decimals"] is False and r["max_abs_difference"] == 0.0012
    for bad in ([{**u, "l1_winner": {"pooling": "max", "planes": "all"}} if i == 3 else u for i, u in enumerate(stage_a)],
                [{**u, "tuning": "simplified (not nested)"} if i == 0 else u for i, u in enumerate(stage_a)],
                stage_a[:-1], stage_a + stage_a[:1],
                [{**u, "default_trial_score": None} if i == 5 else u for i, u in enumerate(stage_a)]):
        with pytest.raises(ValueError):
            n13.stage_a_check(_summary("base0", BASE), bad)


def test_n13_check_heads():
    from knee.nb import n13

    a1 = _summary("a1", _scores(BASE, {e: [0.0] * 5 for e in MAIN}))
    conv = a1["heads_sha16"]["convnext_tiny"]
    n10 = {"heads_sha16": dict(conv)}
    assert n13.check_heads(n10, a1) == {"encoder": "convnext_tiny", "folds": [0, 1, 2, 3, 4],
                                        "mismatched_folds": [], "match": True}
    other = {"heads_sha16": {**conv, "fold_3": "x"}}  # e.g. a knee-n6a head picked up beside N13
    assert n13.check_heads(other, a1)["mismatched_folds"] == [3] and not n13.check_heads(other, a1)["match"]
    short = {"heads_sha16": {f"fold_{k}": conv[f"fold_{k}"] for k in range(4)}}
    assert n13.check_heads(short, a1)["mismatched_folds"] == [4]
    assert not n13.check_heads(n10, a1, encoder="dinov2_vits14")["match"]
    # a rerun fills the failed fold; N10 then loads the rerun's head
    orig = _summary("a1", _scores(BASE, {e: [0.0] * 5 for e in MAIN}), failed={("convnext_tiny", 1)})
    rerun = _summary("a1", {"convnext_tiny": {1: 0.76}}, step="n13_a1_r")
    n10_r = {"heads_sha16": {**conv, "fold_1": rerun["heads_sha16"]["convnext_tiny"]["fold_1"]}}
    assert n13.check_heads(n10_r, [orig, rerun])["match"] is True
    assert n13.check_heads(n10_r, orig)["mismatched_folds"] == [1]  # without the rerun's summary
    with pytest.raises(ValueError):
        n13.check_heads(n10, [a1, rerun])  # the rerun repeats a finished unit
    with pytest.raises(ValueError):
        n13.check_heads(n10, _summary("base0", BASE))  # N10 submits the a1 arm only


@pytest.fixture(scope="module")
def dev_env(tmp_path_factory):
    """A synthetic stage with a1 and base0 outputs shaped like N13's (both main encoders, 5 folds)."""
    from knee.nb import n13
    from tests.synth_stage import make_stage, make_units

    root = tmp_path_factory.mktemp("it2dev") / "input"
    facts = make_stage(root)
    summ = {}
    for arm, strengths, seed in (("a1", {"dinov2_vits14": 1.5, "convnext_tiny": 1.1}, 5),
                                 ("base0", {"dinov2_vits14": 1.3, "convnext_tiny": 1.0}, 6)):
        spec = n13.arm_spec(arm)
        make_units(root, name=spec["name"], strengths=strengths, seed=seed)
        ud_all = root / spec["name"] / "private" / spec["name"] / "units"
        for ud in sorted(ud_all.iterdir()):
            cfg = json.loads((ud / "config.json").read_text())
            st = n13.settings(spec)
            cfg.update({"tuning": "nested", "default_trial_score": cfg["best_inner_score"],
                        "schedule_epochs": st["schedule_epochs"], "patience": st["patience"], "seed": st["seed"],
                        "device": "cuda:0", "device_name": "Tesla T4",
                        "predictions_sha16": common_sha16(ud / "predictions.csv")})
            (ud / "config.json").write_text(json.dumps(cfg))
            (ud / "head.pt").write_bytes(hashlib.sha256(f"{arm}/{ud.name}".encode()).digest())
        s6 = {"units": {ud.name: {"ok": True} for ud in ud_all.iterdir()}}
        summ[arm] = json.loads(json.dumps(n13.inner_summary(arm, spec, s6, ud_all, environment=ENV)))
    null = {e: {k: round(v + 0.001 * (-1) ** k, 4) for k, v in m.items()}
            for e, m in {e: {int(k): v for k, v in sc.items()} for e, sc in summ["base0"]["trial0_inner_scores"].items()}.items()}
    summ["null"] = _summary("null", null)
    return root, facts, summ


def common_sha16(p):
    from knee.nb import common

    return common.sha16(p)


def _independent_s2(root: Path, arm_dirs: list[Path]):
    """S2 metrics recomputed without knee.metrics or StageData: sklearn per fold, pairs inside folds."""
    from sklearn.metrics import roc_auc_score

    from tests.synth_stage import FINDINGS as F

    folds = pd.read_csv(root / "n3" / "private" / "n3" / "folds.csv").set_index("StudyInstanceUID")
    usable = pd.read_csv(root / "n5" / "private" / "n5" / "index.csv").set_index("StudyInstanceUID")["usable"]
    binary = pd.read_csv(root / "n0" / "private" / "n0" / "labels_binary.csv").set_index("StudyInstanceUID")[F]
    keep = folds["train"] & folds["scored"] & usable.reindex(folds.index).fillna(False) & (folds["s2_fold"] >= 0)
    preds = {}
    for d in arm_dirs:  # {(arm, encoder): frame indexed by UID}
        for ud in sorted(d.iterdir()):
            enc = ud.name.rsplit("_f", 1)[0]
            p = pd.read_csv(ud / "predictions.csv").set_index("StudyInstanceUID")[F]
            preds.setdefault((d.parent.name, enc), []).append(p)
    preds = {k: pd.concat(v) for k, v in preds.items()}
    ev = folds.index[keep]
    for p in preds.values():
        ev = ev[np.isfinite(p.reindex(ev).to_numpy()).all(axis=1)]
    y = binary.reindex(ev).fillna(0.0).to_numpy()  # uncertain counts as negative
    f = folds.loc[ev, "s2_fold"].to_numpy()
    out = {}
    for key, p in preds.items():
        s = p.reindex(ev).to_numpy()
        strat, per_fold = [], []
        for j in range(len(F)):
            num = den = 0.0
            for k in np.unique(f):
                m = f == k
                pos, neg = (y[m, j] == 1).sum(), (y[m, j] == 0).sum()
                if pos and neg:
                    num += roc_auc_score(y[m, j], s[m, j]) * pos * neg
                    den += pos * neg
            strat.append(num / den)
        for k in np.unique(f):
            m = f == k
            per_fold.append(np.mean([roc_auc_score(y[m, j], s[m, j]) for j in range(len(F))]))
        out[key] = {"fold_stratified_macro_auc": float(np.mean(strat)), "mean_of_fold_macros": float(np.mean(per_fold))}
    return out, len(ev)


def test_n13_dev_recomputes_the_development_numbers(dev_env, tmp_path, monkeypatch):
    from knee import safe_log
    from knee.nb import n13

    root, facts, summ = dev_env
    decision = n13.decide(summ["a1"], summ["base0"], summ["null"])
    assert decision["status"] == "decided"
    monkeypatch.setattr(paths, "WORKING", tmp_path / "work")
    # refused without the written decision, or with one written on other units
    for bad in (None, {}, {"status": "decided"}, json.dumps({"status": "maybe"})):
        with pytest.raises(ValueError):
            n13.dev(bad, root=root)
    other = json.loads(json.dumps(decision))
    other["arms"]["a1"]["heads_sha16"]["convnext_tiny"]["fold_3"] = "0" * 16
    with pytest.raises(safe_log.StepFailed):  # the attached units are not those the decision was written on
        n13.dev(other, root=root)
    assert not (paths.WORKING / "agg_n13_dev.json").exists()
    # from the dict, from its JSON text and from its file: the same numbers
    dec_file = tmp_path / "a1_decision.json"
    dec_file.write_text(json.dumps(decision))
    out = n13.dev(decision, root=root)
    assert n13.dev(dec_file.read_text(), root=root)["s2"] == out["s2"] == n13.dev(str(dec_file), root=root)["s2"]
    saved = json.loads((paths.WORKING / "agg_n13_dev.json").read_text())
    assert saved["s2"] == json.loads(json.dumps(out["s2"]))
    assert out["label"] == "development: these studies were test studies in iteration 1; likely optimistic"
    assert out["descriptive"] is True and out["added_after_results"] is True and out["status"] == "ok"
    assert out["decision"]["keep"] == {e: decision["encoders"][e]["keep"] for e in MAIN}
    assert all(c["s2_folds_match_units"] for c in out["chain"].values())
    text = (paths.WORKING / "agg_n13_dev.json").read_text() + (paths.WORKING / "run.json").read_text()
    assert not [u for u in facts["uids"] if u in text]
    # the same numbers, recomputed independently (sklearn per fold; pairs inside folds)
    want, n_ev = _independent_s2(root, [root / "n13_a1" / "private" / "n13_a1" / "units",
                                        root / "n13_base0" / "private" / "n13_base0" / "units"])
    assert out["evaluation_studies"] == agg.count(n_ev)
    for e in MAIN:
        for arm, name in (("a1", "n13_a1"), ("base0", "n13_base0")):
            for m in ("fold_stratified_macro_auc", "mean_of_fold_macros"):
                assert out["s2"][e][arm][m] == pytest.approx(want[(name, e)][m], abs=5.1e-5)
        for m in ("fold_stratified_macro_auc", "mean_of_fold_macros"):
            diff = want[("n13_a1", e)][m] - want[("n13_base0", e)][m]
            assert out["s2"][e]["a1_minus_base0"][m] == pytest.approx(diff, abs=5.1e-5)


def test_n13_a7_summary_is_descriptive(dev_env):
    from knee.nb import n13
    from tests.synth_stage import make_units

    root, _, summ = dev_env
    spec = n13.arm_spec("a7", schedule_epochs=8, patience=3)
    make_units(root, name="n13_a7", strengths={"convnext_tiny": 1.0}, seed=9)  # 30 trials per unit
    ud_all = root / "n13_a7" / "private" / "n13_a7" / "units"
    for ud in ud_all.iterdir():
        cfg = json.loads((ud / "config.json").read_text())
        cfg.update({"tuning": "nested", "default_trial_score": round(cfg["best_inner_score"] - 0.02, 4),
                    "schedule_epochs": 8, "patience": 3, "seed": 0, "device": "cuda:0", "device_name": "Tesla T4"})
        (ud / "config.json").write_text(json.dumps(cfg))
        (ud / "head.pt").write_bytes(ud.name.encode())
    s6 = {"units": {ud.name: {"ok": True} for ud in ud_all.iterdir()}}
    out = n13.inner_summary("a7", spec, s6, ud_all, environment=ENV, U=20.0, encoders=("convnext_tiny",))
    agg._check(json.loads(json.dumps(out)))  # passes the agg rules
    assert out["descriptive"] is True and out["keep_decision"].startswith("none (plan r3)")
    assert out["encoders"] == ["convnext_tiny"] and out["settings"]["n_trials"] == 30
    assert out["settings"]["space_overrides"] == {"batch_size": [8, 128], "weight_decay": [1e-06, 1.0]}
    assert sorted(out["best_inner_scores"]["convnext_tiny"]) == [0, 1, 2, 3, 4]
    assert all(u["best_minus_trial0"] == 0.02 for u in out["units"].values())
    b = out["boundary_widened_ranges"]
    assert b["batch_size"]["range"] == [8, 128] and b["weight_decay"]["range"] == [1e-06, 1.0]
    assert b["lr"]["range"] == [1e-05, 0.01]  # the other ranges stay registered
    with pytest.raises(ValueError):  # A7 never enters the keep rule
        n13.decide(json.loads(json.dumps(out)), summ["base0"], summ["null"])
