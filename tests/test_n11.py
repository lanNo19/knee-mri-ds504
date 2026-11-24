"""N11, Stage B-lite on all five folds (it3_amendment_r3.md, plan section 12), on synthetic data (CPU, no
weights downloaded). A tiny fake encoder pair with the same interface split as the real ones (frozen trunk +
trainable tail) stands in for DINOv2 and ConvNeXt-T. Tested: the twelfths schedule; the validation table and
its assertions; the GPU step (order, arms, files, no test-fold metric, the chosen rate recorded before a
trained model sees a test fold); the loop (accumulation, exact resume, head-only, against heads.fit);
decide() and dev() on hand-made files with known answers; the real split with timm."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from knee.labels import FINDINGS, UID

torch = pytest.importorskip("torch")
sk = pytest.importorskip("sklearn.metrics")

MAIN = ("dinov2_vits14", "convnext_tiny")
PLANES = ("sagittal", "coronal", "axial")
S, D, HW = 4, 32, 16
GATED = {"pooling": "gated", "planes": "all", "lr": 3e-3, "weight_decay": 3e-4, "dropout": 0.25,
         "attn_hidden": 16, "pos_cap": 4.47, "batch_size": 16}


class _LayerScale(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.full((dim,), 0.5))


def _fake_split(name: str, device: str = "cpu"):
    from knee.nb import n11

    class FakeSplit(n11.Split):
        """Frozen trunk (conv stem + linear) and a trainable tail (a residual block with layer-scale, a norm)."""

        def __init__(self):
            self.name, self.kind = name, "fake"
            g = torch.Generator().manual_seed(11 + MAIN.index(name))
            model = torch.nn.ModuleDict({"stem": torch.nn.Conv2d(3, 8, 4, 4), "mid": torch.nn.Linear(8 * 16, D),
                                         "block": torch.nn.Linear(D, D), "ls": _LayerScale(D),
                                         "norm": torch.nn.LayerNorm(D)})
            with torch.no_grad():
                for n, p in model.named_parameters():  # every weight from the seeded generator
                    if p.ndim > 1:
                        p.copy_(torch.randn(p.shape, generator=g) / p[0].numel() ** 0.5)
                    elif n.endswith("bias"):
                        p.copy_(0.1 * torch.randn(p.shape, generator=g))
            super().__init__(model, device, ("block.", "ls.", "norm."))

        def prep(self, u8):
            return u8.float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)

        def trunk(self, x):
            m = self.model
            return torch.nn.functional.gelu(m["mid"](torch.nn.functional.gelu(m["stem"](x)).flatten(1)))

        def tail(self, h):
            m = self.model
            return m["norm"](h + m["ls"].gamma * torch.nn.functional.gelu(m["block"](h)))

    return FakeSplit()


def fake_factory(name: str, device: str = "cpu"):
    return _fake_split(name, device)


def _make_inputs(inp: Path, n: int = 360) -> None:
    """N0, N3 and N5 files (tests/synth_stage.py), N4-like caches whose images carry the labels, N5-like
    embeddings computed by the fake encoders, and a fold-0 Stage A unit per encoder (n6._unit's files)."""
    from knee import heads, stage
    from tests.synth_stage import make_stage

    make_stage(inp, n=n, S=S, D=D)
    idx = pd.read_csv(inp / "n5" / "private" / "n5" / "index.csv").astype({UID: str})
    lab = pd.read_csv(inp / "n0" / "private" / "n0" / "labels_binary.csv").astype({UID: str}).set_index(UID)
    y = lab.reindex(idx[UID])[FINDINGS].fillna(0).to_numpy()
    usable = idx["usable"].to_numpy()
    flip_row = idx["flip_row"].to_numpy()
    rng = np.random.default_rng(3)
    caches = {}
    for p in PLANES:
        img = rng.integers(0, 80, size=(n, S, HW, HW)).astype(np.float32)
        for j in range(len(FINDINGS)):  # finding j brightens cell j of a 4 x 4 grid (not left-right symmetric)
            r0, c0 = 4 * (j // 4), 4 * (j % 4)
            on = (y[:, j] == 1)[:, None] & (rng.random((n, S)) < 0.8)
            img[:, :, r0:r0 + 4, c0:c0 + 4] += 120 * on[:, :, None, None]
        img = np.clip(img, 0, 255).astype(np.uint8)
        img[~usable] = 0
        d = inp / f"n4-{p}" / "private" / f"n4_{p}"
        d.mkdir(parents=True)
        mm = np.memmap(d / f"cache_{p}.u8", dtype=np.uint8, mode="w+", shape=img.shape)
        mm[:] = img
        mm.flush()
        (d / f"cache_{p}_manifest.json").write_text(json.dumps(
            {"plane": p, "shape": list(img.shape), "dtype": "uint8", "file": f"cache_{p}.u8"}))
        pd.DataFrame({UID: idx[UID], "row": range(n), "ok": usable}).to_csv(d / f"cache_{p}_studies.csv", index=False)
        idx[[UID, "laterality"]].to_csv(d / "study_table.csv", index=False)
        caches[p] = img
    n5d = inp / "n5" / "private" / "n5"
    for e in MAIN:
        enc = fake_factory(e)
        for p in PLANES:
            with torch.no_grad():
                def feats(a):
                    x = torch.from_numpy(a).reshape(-1, HW, HW)
                    return enc.tail(enc.trunk(enc.prep(x))).reshape(a.shape[0], S, D).numpy()

                f = feats(caches[p])
                f[~usable] = 0
                np.save(n5d / f"emb_{e}_{p}.npy", f.astype(np.float16))
                if p in ("coronal", "axial"):
                    un = np.flatnonzero(flip_row >= 0)
                    ff = np.zeros((len(un), S, D), np.float32)
                    ff[flip_row[un]] = feats(np.ascontiguousarray(caches[p][un][..., ::-1]))
                    np.save(n5d / f"emb_{e}_{p}_flip.npy", ff.astype(np.float16))
    d = stage.StageData(inp)
    cfg = heads.config_from(GATED)
    for e in MAIN:
        bank = d.bank(e, "cpu")
        for k in range(5):  # every unit has its own fold's head, statistics and predictions
            tr, te, _ = d.split(k)
            stats = bank.stats(tr)
            r = heads.fit(bank, cfg, tr, epochs=2 + k % 2, stats=stats, forbidden=np.concatenate([te, d.gold_rows()]), seed=0)
            ud = inp / "n6a" / "private" / "n6a" / "units" / f"{e}_f{k}"
            ud.mkdir(parents=True)
            rows_out = np.concatenate([te, d.gold_rows(k)])
            pred = pd.DataFrame(heads.predict(bank, r.model, rows_out, cfg, stats), columns=FINDINGS)
            pred.insert(0, "is_gold", d.is_gold[rows_out])
            pred.insert(0, "s2_fold", k)
            pred.insert(0, UID, d.uids[rows_out])
            pred.to_csv(ud / "predictions.csv", index=False)
            torch.save(r.model.state_dict(), ud / "head.pt")
            torch.save({p: (mu.cpu(), sd.cpu()) for p, (mu, sd) in stats.items()}, ud / "stats.pt")
            (ud / "config.json").write_text(json.dumps({"encoder": e, "fold": k, "config": GATED, "E": 2 + k % 2}))


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    inp = tmp_path_factory.mktemp("n11") / "input"
    _make_inputs(inp)
    return inp



TABLE = {0: {"test": ("LANG0", "LANG5"), "val": ("LANG1",)}, 1: {"test": ("LANG1", "LANG6"), "val": ("LANG2",)},
         2: {"test": ("LANG2", "LANG7"), "val": ("LANG3",)}, 3: {"test": ("LANG3",), "val": ("LANG4", "LANG0")},
         4: {"test": ("LANG4",), "val": ("LANG2",)}}
PASSES = 460  # 2 epochs on about 230 training studies
KW = {"inner_table": None}


def _run(inputs, **kw):
    from knee.nb import n11

    args = {"root": inputs, "val_table": TABLE, "val_range": None, "passes": PASSES, "encoder_factory": fake_factory,
            "schedule": "full", "quiet": True, "smoke_micro": 12}
    return n11.run(**{**args, **kw})


@pytest.fixture(scope="module")
def e2e(inputs):
    """Session "a" on folds 0 and 1, both fake encoders, in this process, with spies on the inner refit, the
    test pass, the feature sources and the order of the blocks."""
    from knee import heads, paths, stage
    from knee.nb import n11

    work = inputs.parent / "w11"
    seen = {"fit": [], "test": [], "rows": [], "order": []}
    real_fit, real_test = heads.fit, n11.predict_test
    where, in_test = ["-"], [0]

    def spy_fit(bank, cfg, train_idx, *a, **k):
        seen["fit"].append((np.asarray(train_idx).copy(), np.asarray(k.get("forbidden")).copy()))
        return real_fit(bank, cfg, train_idx, *a, **k)

    def spy_test(c, src, head, trained, lr=None):
        in_test[0] += 1
        try:
            P = real_test(c, src, head, trained, lr)
        finally:
            in_test[0] -= 1
        seen["test"].append({"unit": c.unit, "trained": trained, "lr": lr,
                             "recorded": lr is not None and n11.chosen_recorded(c.working, c.unit, lr)})
        return P

    def spy_source(mp, cls):
        real = cls.__call__

        def call(self, rows, grad=False, flipped=False):
            seen["rows"].append(SimpleNamespace(src=cls.__name__, stage=where[-1], in_test=bool(in_test[0]),
                                                grad=bool(grad), flipped=bool(flipped), rows=np.asarray(rows).copy()))
            return real(self, rows, grad=grad, flipped=flipped)

        mp.setattr(cls, "__call__", call)

    def spy_stage(mp, name):
        real = getattr(n11, name)

        def f(c, *a, **k):
            where.append((name, c.fold))
            seen["order"].append((c.name, name, c.fold))
            try:
                return real(c, *a, **k)
            finally:
                where.pop()

        mp.setattr(n11, name, f)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "WORKING", work)
        mp.setattr(heads, "fit", spy_fit)
        mp.setattr(n11, "predict_test", spy_test)
        for cls in (n11.EncSource, n11.FeatSource):
            spy_source(mp, cls)
        for name in ("_frozen0", "_inner", "_final"):
            spy_stage(mp, name)
        s = _run(inputs, session="a", folds=(0, 1))
    return SimpleNamespace(s=s, work=work, seen=seen, d=stage.StageData(inputs))


def _shared(inputs, work, enc=MAIN[0], **kw):
    from knee.nb import n11

    return n11._shared({"root": str(inputs), "expect": None, "working": str(work), "step": "n11_t", "factory": fake_factory,
                        "encoder": enc, "device": "cpu", "val_table": TABLE, "val_range": None, "eval_studies": 4,
                        "micro": 2, "passes": PASSES, "schedule": "full", **kw})


@pytest.fixture()
def ctx(inputs, tmp_path):
    from knee.nb import n11

    return n11._context(_shared(inputs, tmp_path / "w"), 0)


def _macro(P, Y):
    return float(np.mean([sk.roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


# ---------------------------------------------------------------- the schedule and the validation table

def test_schedule_in_twelfths():
    from knee import heads
    from knee.nb import n11

    assert n11.schedule_epochs(1350) == 12 and n11.schedule_epochs(1350, "halved") == 6
    assert [n11.schedule_epochs(n) for n in (3950, 4090)] == [4, 4] and n11.schedule_epochs(4000, "halved") == 2
    spe = 85  # fold 0: 1,350 studies, batch 16: the full schedule is the old 12-epoch one, step for step
    total = 12 * spe
    assert n11.twelfth_marks(total) == [spe * k for k in range(1, 13)]
    assert [n11.twelfth_factor(i, total) for i in range(total)] == [heads.schedule_factor(i, spe, 12) for i in range(total)]
    total = 4 * 257  # folds 1-4: twelfths that are no whole number of steps
    marks = n11.twelfth_marks(total)
    assert marks[-1] == total and marks[2] == 257 and all(b > a for a, b in zip(marks, marks[1:]))
    f = [n11.twelfth_factor(i, total) for i in range(total)]
    warm = marks[0]
    assert f[warm - 1] == 1.0 and f[0] == pytest.approx(1 / warm) and all(b > a for a, b in zip(f[:warm], f[1:warm]))
    assert all(b < a for a, b in zip(f[warm:], f[warm + 1:])) and 0 < f[-1] < 1e-4
    with pytest.raises(ValueError):
        n11.twelfth_marks(11)
    with pytest.raises(ValueError):
        n11.schedule_epochs(1350, "short")


def test_validation_table_and_its_assertions(inputs):
    from knee import stage
    from knee.nb import n11

    t = n11.VAL_TABLE  # the plan's table: five folds, four distinct validation sets, English never held out
    assert sorted(t) == [0, 1, 2, 3, 4] and len({v["val"] for v in t.values()}) == 4
    assert t[1]["val"] == t[4]["val"] == ("GREEK",) and t[3]["val"] == ("BULGARIAN", "DUTCH")
    assert not any("ENGLISH" in v["val"] for v in t.values()) and all(not set(v["val"]) & set(v["test"]) for v in t.values())
    assert n11.VAL_RANGE == (250, 500) and n11.SESSIONS == {"a": (0, 1, 2), "b": (3, 4)}
    d = stage.StageData(inputs)
    g = d.g_lang.astype(str)
    for k in range(5):
        sp = n11.inner_split(d, k, TABLE, None)
        tr, te, _ = d.split(k)
        assert set(g[sp["iva_all"]]) == set(TABLE[k]["val"]) and not set(g[sp["itr"]]) & set(TABLE[k]["val"])
        assert set(sp["itr"]) | set(sp["iva_all"]) == set(tr) and not set(sp["iva_all"]) & set(te)
        assert set(sp["iva"]) <= set(sp["iva_all"]) and d.scored[sp["iva"]].all()
        assert sp["val_set"] == "+".join(sorted(TABLE[k]["val"])) and set(sp["val_groups"]) == set(TABLE[k]["val"])
    assert len(n11.inner_split(d, 3, TABLE, None)["val_groups"]) == 2  # fold 3: two groups
    with pytest.raises(AssertionError):  # the registered table does not fit this data: the run stops
        n11.inner_split(d, 0, None, None)
    with pytest.raises(AssertionError):  # another fold-to-group order
        n11.inner_split(d, 0, {0: {"test": ("LANG0",), "val": ("LANG1",)}}, None)
    with pytest.raises(AssertionError):  # a validation group inside the test fold
        n11.inner_split(d, 0, {0: {"test": ("LANG0", "LANG5"), "val": ("LANG5",)}}, None)
    with pytest.raises(AssertionError):  # outside 250-500 studies
        n11.inner_split(d, 0, TABLE, (250, 500))
    with pytest.raises(AssertionError):
        n11.inner_split(d, 0, {0: {"test": ("LANG0", "LANG5"), "val": ("NOPE",)}}, None)


# ---------------------------------------------------------------- the GPU step

def test_session_units_order_and_outputs(e2e):
    from knee import agg
    from knee.nb import n11

    s, d = e2e.s, e2e.d
    assert s["status"] == "ok", s.get("errors")
    assert s["session"] == "a" and s["schedule"] == "full" and set(s["units"]) == {f"{e}_f{k}" for e in MAIN for k in (0, 1)}
    assert json.loads((e2e.work / "run.json").read_text())["config"]["schedule"] == "full"
    # order inside a session: frozen0 for every fold, then all inner work, then the final fits
    for e in MAIN:
        assert [(n, k) for ee, n, k in e2e.seen["order"] if ee == e] == [
            ("_frozen0", 0), ("_frozen0", 1), ("_inner", 0), ("_inner", 1), ("_final", 0), ("_final", 1)]
    row_of = {u: i for i, u in enumerate(d.uids)}
    for u, b in s["units"].items():
        e, k = u.rsplit("_f", 1)
        k = int(k)
        ud = e2e.work / "private" / "n11_a" / u
        te_s = d.split(k)[2]
        assert b["fold"] == k and b["config"]["E"] == 2 + k % 2 and b["config"]["epochs"] == 2  # its own fold's unit
        assert set(b["inner_validation_groups"]) == set(TABLE[k]["val"]) and set(b["test_groups"]) == set(TABLE[k]["test"])
        f0 = b["frozen0"]
        assert f0["valid"] and f0["check_b"]["abs_difference"] <= 0.0005 and f0["check_b"]["max_abs_probability_gap"] < 2e-3
        mine = pd.read_csv(ud / "pred_frozen0.csv").astype({UID: str})
        assert [row_of[x] for x in mine[UID]] == te_s.tolist()
        assert np.array_equal(mine[[f"y_{f}" for f in FINDINGS]].to_numpy(), d.eval_y[te_s])
        assert np.array_equal(mine["resolved"].to_numpy(), d.flip_row[te_s] < 0)
        inner = b["inner"]
        assert set(inner["rung0"]) == {"5e-06", "2e-05", "8e-05"} and inner["complete"] is True
        for v in inner["rung0"].values():  # the first quarter: 3 twelfths, logged at every twelfth
            assert v["schedule_share"] == 0.25 and [len(x) for x in v["curve"].values()] == [3, 3, 3, 3]
        assert "2e-05" in inner["to_the_end"] and n11._lr_key(inner["chosen_lr"]) in inner["to_the_end"]
        for key, v in inner["to_the_end"].items():
            assert v["schedule_share"] == 1.0 and len(v["curve"]["val_loss"]) == 12 and v["resumed_from_rung0"]
            assert v["curve"]["val_macro_auc"][:3] == inner["rung0"][key]["curve"]["val_macro_auc"]
        assert set(inner["head_only"]) == {"seed_0", "seed_100"} and inner["start"]["val_macro_auc"] is not None
        arms = inner["keep_rule_arms"]
        assert set(arms) == {"start", "head_only_seed_0", "head_only_seed_100", "b_lite_2e-05"}
        for arm, key in (("start", "start"), ("head_only_seed_0", "head_only_seed_0"),
                         ("head_only_seed_100", "head_only_seed_100"), ("b_lite", "b_lite_2e-05")):
            f = n11._read_rows(ud / f"inner_{arm}.csv")  # the per-study inner files give the logged scores back
            iva = n11.inner_split(d, k, TABLE, None)["iva"]
            assert [row_of[x] for x in f["uid"]] == iva.tolist() and set(f["val_set"]) == {b["inner_validation_set"]}
            assert n11._wmacro(f["P"], f["Y"]) == pytest.approx(arms[key], abs=6e-5)
        fin = b["final"]
        assert set(fin["head_only"]) == {f"seed_{i}" for i in range(5)} and set(fin["b_lite"]) == {"seed_0"}
        assert fin["complete"] is True and len(fin["b_lite"]["seed_0"]["curve"]["train_loss"]) == 12
        assert fin["b_lite"]["seed_0"]["encoder_lr"] == inner["chosen_lr"]
        assert "auc" not in json.dumps(fin).lower()  # no test-fold metric of a trained model
        files = sorted(p.name for p in ud.iterdir())
        assert files == sorted(["state.json", "b_lite_seed_0.pt", "pred_frozen0.csv", "pred_b_lite.csv"]
                               + [f"pred_head_only_seed_{i}.csv" for i in range(5)]
                               + [f"inner_{a}.csv" for a in n11.INNER_ARMS])  # rung checkpoints are gone
        assert set(b["files_sha16"]) == set(files) - {"state.json"}
        assert all(b["files_sha16"][n] == n11.sha16(ud / n) for n in b["files_sha16"])
        w = torch.load(ud / "b_lite_seed_0.pt", map_location="cpu")
        assert set(w["tail"]) == {"block.weight", "block.bias", "ls.gamma", "norm.weight", "norm.bias"} and w["fold"] == k
    # the only test-fold numbers anywhere in the outputs are frozen0's check
    text = (e2e.work / "agg_n11_a.json").read_text()
    a = json.loads(text)
    agg._check(a)

    def keys(o, path=""):
        if isinstance(o, dict):
            for kk, v in o.items():
                yield f"{path}/{kk}"
                yield from keys(v, f"{path}/{kk}")

    auc_keys = [p for p in keys(a) if "auc" in p.rsplit("/", 1)[-1].lower()]
    assert auc_keys and all("/inner/" in p or "/frozen0/check_b/" in p for p in auc_keys)
    assert not any(w in text for w in ("mean_of_fold_macros", "fold_stratified", "delta1", "side_auc"))
    assert not any(x in text + (e2e.work / "run.json").read_text() for x in d.uids[:50])
    assert sorted(p.name for p in e2e.work.iterdir()) == ["agg_n11_a.json", "private", "run.json"]
    with pytest.raises(ValueError):  # never fewer folds: two folds are no basis for a decision
        n11.decision(n11.find_units(e2e.work))


def test_chosen_rate_recorded_before_any_trained_test_pass_and_rows_seen(e2e):
    from knee.nb import n11

    d = e2e.d
    rj = json.loads((e2e.work / "run.json").read_text())["n11_chosen_encoder_lr"]
    for u, b in e2e.s["units"].items():
        chosen = b["inner"]["chosen_lr"]
        assert rj[u]["lr"] == chosen
        calls = [c for c in e2e.seen["test"] if c["unit"] == u]
        assert [c["trained"] for c in calls] == [False] + [True] * 6  # frozen0, head-only x 5, b-lite
        assert all(c["recorded"] and c["lr"] == chosen for c in calls if c["trained"])
    gold = set(d.gold_rows().tolist())
    for x in e2e.seen["rows"]:
        name, k = x.stage
        tr, te, te_s = d.split(k)
        sp = n11.inner_split(d, k, TABLE, None)
        rows = set(x.rows.tolist())
        assert rows <= set(tr) | set(te_s) and not rows & gold
        if rows & set(te):  # test rows: only inside predict_test, or the frozen0 feature build
            assert x.in_test or (name == "_frozen0" and x.src == "EncSource" and not x.grad)
        if name == "_inner":  # inner work never touches a test row; a graph only on inner training rows
            assert not x.in_test and not x.flipped and rows <= set(tr)
            if x.src == "EncSource":
                assert rows <= (set(sp["itr"]) if x.grad else set(sp["iva"]))
            else:
                assert rows <= set(sp["itr"]) or rows <= set(sp["iva"])
        elif name == "_final" and not x.in_test:
            assert rows <= set(tr) and x.grad is (x.src == "EncSource")
    # every fit's leakage sets: the refit never sees its validation, test or gold rows
    assert len(e2e.seen["fit"]) == 4
    for train, forb in e2e.seen["fit"]:
        assert not set(train.tolist()) & set(forb.tolist()) and gold <= set(forb.tolist())


def test_inner_refit_guard_and_unit_inputs(inputs, tmp_path, monkeypatch):
    from knee import heads
    from knee.nb import n11

    sh = _shared(inputs, tmp_path / "w")
    c0, c3 = n11._context(sh, 0), n11._context(sh, 3)
    assert c0.E == 2 and c3.E == 3 and c0.unit_dir.name == f"{MAIN[0]}_f0" and c3.unit_dir.name == f"{MAIN[0]}_f3"
    assert not torch.equal(c0.stats["sagittal"][0], c3.stats["sagittal"][0])  # each fold's own statistics
    assert set(c3.forbid_inner) == set(c3.iva_all) | set(c3.te) | set(c3.gold)
    for bad in (c3.iva_all[0], c3.te[0], c3.gold[0]):
        c2 = copy.copy(c3)
        c2.itr = np.append(c3.itr, bad)
        with pytest.raises(heads.LeakageError):
            n11.inner_refit(c2)
    inp = tmp_path / "input"
    shutil.copytree(inputs, inp)
    cfg_path = inp / "n6a" / "private" / "n6a" / "units" / f"{MAIN[0]}_f0" / "config.json"
    cj = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps({**cj, "schedule_epochs": 8}))
    assert n11._context(_shared(inp, tmp_path / "w2"), 0).stage_a_schedule == 8 and c0.stage_a_schedule is None
    (inp / "n6a" / "private" / "n6a").rename(inp / "n6a" / "private" / "n13_a1")  # an iteration-2 unit
    with pytest.raises(AssertionError):
        n11._context(_shared(inp, tmp_path / "w3"), 0)


def test_invalid_run_stops_before_any_training(inputs, tmp_path, monkeypatch, capfd):
    from knee import paths, safe_log
    from knee.nb import n11

    inp = tmp_path / "input"
    shutil.copytree(inputs, inp)
    p = inp / "n6a" / "private" / "n6a" / "units" / "dinov2_vits14_f1" / "predictions.csv"
    df = pd.read_csv(p)
    df[FINDINGS] = df[FINDINGS].to_numpy()[::-1]  # other studies' probabilities: check (b) must fail
    df.to_csv(p, index=False)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    monkeypatch.setattr(n11, "new_run", lambda *a, **k: pytest.fail("nothing may train in an invalid run"))
    with pytest.raises(safe_log.StepFailed):
        _run(inp, session="a", folds=(0, 1))
    assert "STEP n11_a: FAILED - InvalidRun" in capfd.readouterr().out
    a = json.loads((tmp_path / "w" / "agg_n11_a.json").read_text())
    assert a["status"].startswith("invalid") and all(set(v) == {"frozen0"} for v in a["units"].values())
    assert a["units"]["dinov2_vits14_f1"]["frozen0"]["check_b"]["passed"] is False
    assert a["units"]["dinov2_vits14_f0"]["frozen0"]["check_b"]["passed"] is True


def test_smoke_reads_no_score(inputs, tmp_path, monkeypatch):
    from knee import paths
    from knee.nb import n11

    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    for name in ("_validate", "predict", "_frozen0", "_inner", "_final", "decision"):
        monkeypatch.setattr(n11, name, lambda *a, **k: pytest.fail("the smoke run must not score anything"))
    s = _run(inputs, smoke=True, schedule=None)
    assert s["status"].startswith("ok (smoke") and s["suggested_schedule"] in ("full", "halved")
    assert set(s["units"]) == {f"{e}_f0" for e in MAIN}
    for b in s["units"].values():
        sm = b["smoke"]
        assert sm["micro_batches"] == 10 and sm["slices_per_s"] > 0 and sm["reads_no_score"] is True
        assert sm["late_code_paths"] == {"feature_store": "float32", "head_only_step": True, "flipped_forward_finite": True,
                                         "checkpoint_round_trip": True, "run_json_lock": True}
        assert not set(b) & {"frozen0", "inner", "final"}
    assert set(s["environment"]) >= {"torch", "python", "device_name"}
    assert set(s["validation_table"]) == {f"fold_{k}" for k in range(5)}  # the table is checked for every fold
    assert set(s["validation_table"]["fold_3"]["inner_validation_groups"]) == {"LANG4", "LANG0"}
    with pytest.raises(Exception):  # the registered table does not fit this data: the smoke run stops
        _run(inputs, smoke=True, schedule=None, val_table=None)
    assert "n11_chosen_encoder_lr" not in json.loads((tmp_path / "w" / "run.json").read_text())
    assert (tmp_path / "w" / "agg_n11_smoke.json").exists()
    assert n11.find_units(tmp_path / "w") == {}  # the smoke run is no session


def test_one_encoder_two_processes_split_the_folds(inputs, tmp_path, monkeypatch):
    from knee import paths

    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    s = _run(inputs, session="b", folds=(3, 4), encoders=(MAIN[1],), processes=True, devices=["cpu", "cpu"],
             head_only_seeds=(0,))
    assert s["status"] == "ok" and set(s["units"]) == {f"{MAIN[1]}_f3", f"{MAIN[1]}_f4"}
    assert s["processes"] == [f"{MAIN[1]} on cpu: folds 3", f"{MAIN[1]} on cpu: folds 4"]
    assert s["cuts"]["head_only_seeds_1_to_4"] is True
    rj = json.loads((tmp_path / "w" / "run.json").read_text())["n11_chosen_encoder_lr"]
    assert set(rj) == set(s["units"])
    assert all(set(b["final"]["head_only"]) == {"seed_0"} for b in s["units"].values())
    with pytest.raises(ValueError):
        _run(inputs, session="b", schedule=None)  # the schedule is an argument, set from the smoke run


def test_a_failed_or_late_unit_ends_the_step_partial(inputs, tmp_path, monkeypatch, capfd):
    """One unit fails: the step ends OK with status "partial", the other units finish and are listed.
    The soft stop raises OutOfTime before a fit, never inside one."""
    import time

    from knee import paths
    from knee.nb import n11

    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    real = n11._inner

    def inner(c, st, feat):
        if c.fold == 1:
            raise RuntimeError("synthetic failure of one unit")
        real(c, st, feat)

    monkeypatch.setattr(n11, "_inner", inner)
    s = _run(inputs, session="a", folds=(0, 1), encoders=(MAIN[1],), head_only_seeds=(0,))
    assert "STEP n11_a: OK" in capfd.readouterr().out
    u0, u1 = f"{MAIN[1]}_f0", f"{MAIN[1]}_f1"
    assert s["status"] == "partial" and s["units_complete"] == [u0] and s["units_missing"] == [u1]
    assert s["errors"][u1].startswith("RuntimeError at") and s["out_of_time"] is False
    assert s["units"][u0]["final"]["complete"] is True and s["units"][u1]["frozen0"]["valid"] is True
    a = json.loads((tmp_path / "w" / "agg_n11_a.json").read_text())
    assert a["status"] == "partial" and set(a["units"]) == {u0, u1}
    # the soft stop
    c = SimpleNamespace(working=tmp_path / "w", step="n11_a", task={"soft_stop": time.time() - 1})
    with pytest.raises(n11.OutOfTime):
        n11._abort_if_invalid(c)
    c.task = {"soft_stop": time.time() + 60}
    n11._abort_if_invalid(c)
    monkeypatch.setattr(n11, "_inner", real)
    monkeypatch.setattr(n11, "_final", lambda c, st, feat: (_ for _ in ()).throw(n11.OutOfTime("late")))
    s = _run(inputs, session="a", folds=(0, 1), encoders=(MAIN[1],), head_only_seeds=(0,))
    assert s["status"] == "partial" and s["out_of_time"] is True and s["units_complete"] == []
    assert all(b["inner"]["complete"] for b in s["units"].values())  # the finished inner work is kept and listed
    assert s["errors"][u0].startswith("OutOfTime at")


def test_an_interrupt_is_not_swallowed(inputs, tmp_path, monkeypatch):
    from knee.nb import n11

    def stop(task):
        raise KeyboardInterrupt

    monkeypatch.setattr(n11, "_shared", stop)
    with pytest.raises(KeyboardInterrupt):
        n11._task({"encoder": MAIN[0], "working": str(tmp_path / "w"), "step": "n11_a", "folds": [0], "phase": "session"})
    assert (tmp_path / "w" / "private" / "n11_a" / f"{MAIN[0]}_f0" / "error.log").exists()


# ---------------------------------------------------------------- the loop

def test_gradient_accumulation_equals_one_large_batch(ctx):
    from knee.nb import n11

    b = ctx.tr[:16]
    grads = []
    for micro in (2, len(b), 5):
        r = n11.new_run(ctx, ctx.src, ctx.head_state, ctx.tr, 2e-5, 0)
        r.head.eval()  # dropout off: the passes must see the same function
        n11.batch_backward(ctx, r, b, micro=micro)
        named = ctx.enc.tail_named_parameters() + list(r.head.named_parameters())
        grads.append({n: p.grad.detach().clone() for n, p in named})
    assert any(float(g.abs().max()) > 1e-4 for n, g in grads[0].items() if n.startswith("block."))
    for other in grads[1:]:
        for n, g in grads[0].items():
            assert g.dtype == torch.float32 and float((g - other[n]).abs().max()) < 1e-6, n


def test_rung_resume_inside_an_epoch_equals_an_uninterrupted_run(ctx, tmp_path):
    from knee.nb import n11

    rows, val = ctx.itr[:60], ctx.iva

    def start():
        return n11.new_run(ctx, ctx.src, ctx.head_state, rows, 1e-3, 0, val_rows=val, epochs=3)

    a = start()
    n11.run_to(ctx, a, 12)
    tail_a, head_a = ctx.enc.tail_state(), {k: v.clone() for k, v in a.head.state_dict().items()}
    b = start()
    n11.run_to(ctx, b, 3)
    assert b.step == b.marks[2] == 3 and 0 < b.pos < len(rows) and len(b.history) == 3  # the rung is inside epoch 1
    n11.save_checkpoint(ctx, b, tmp_path / "ck.pt")
    other = n11.new_run(ctx, ctx.src, ctx.head_state, rows, 8e-5, 0, val_rows=val, epochs=3)  # another run in between
    n11.run_to(ctx, other, 5)
    b2 = start()
    n11.load_checkpoint(ctx, b2, tmp_path / "ck.pt")
    n11.run_to(ctx, b2, 12)
    assert b2.history == a.history and len(a.history) == 12 and b2.step == a.step == a.total == 12
    tail_b = ctx.enc.tail_state()
    assert any(not torch.equal(tail_a[n], ctx.enc._tail0[n]) for n in tail_a)
    assert all(torch.equal(tail_a[n], tail_b[n]) for n in tail_a)
    assert all(torch.equal(head_a[k], v) for k, v in b2.head.state_dict().items())
    with pytest.raises(AssertionError):  # a checkpoint of another learning rate is refused
        n11.load_checkpoint(ctx, other, tmp_path / "ck.pt")


def test_head_only_with_encoder_rate_0_and_groups(ctx):
    from knee.nb import n11

    rows = ctx.tr[:48]
    r = n11.new_run(ctx, ctx.src, ctx.head_state, rows, 0.0, 0, epochs=4)  # the b-lite loop, encoder rate 0
    named = dict(ctx.enc.tail_named_parameters())
    no_decay = {n for n, p in named.items() if ctx.enc.no_decay(n, p)}
    assert no_decay == {"block.bias", "ls.gamma", "norm.weight", "norm.bias"}
    assert [g["weight_decay"] for g in r.opt.param_groups] == [0.05, 0.0, ctx.cfg.weight_decay]
    assert {id(p) for p in r.opt.param_groups[1]["params"]} == {id(named[n]) for n in no_decay}
    n11.run_to(ctx, r, 6)
    f = n11.twelfth_factor(r.step - 1, r.total)
    assert [g["lr"] for g in r.opt.param_groups] == pytest.approx([0.0, 0.0, ctx.cfg.lr * f])
    now = ctx.enc.tail_state()
    assert all(torch.equal(now[n], ctx.enc._tail0[n]) for n in now) and not ctx.enc.model.training
    feat, _ = n11._features(ctx, rows, [])
    r2 = n11.new_run(ctx, feat, ctx.head_state, rows, 0.0, 0, with_encoder=False, epochs=4)
    n11.run_to(ctx, r2, 6)
    for k, v in r.head.state_dict().items():
        assert float((v - r2.head.state_dict()[k]).abs().max()) < 1e-5, k
    assert any(not torch.equal(v, ctx.head_state[k]) for k, v in r.head.state_dict().items())


def test_loop_equals_heads_fit_at_dropout_0(ctx):
    """12 whole epochs: a twelfth is an epoch, so the loop on cached features must equal heads.fit with a
    12-epoch schedule (loss normalisation, pos_weight, batch order, schedule)."""
    from dataclasses import replace

    from knee import heads
    from knee.nb import n11

    c = copy.copy(ctx)
    c.cfg = replace(ctx.cfg, dropout=0.0, batch_size=17)
    rows = ctx.tr
    torch.default_generator.manual_seed(0)  # the head heads.fit builds for seed 0
    init = {k: v.clone() for k, v in heads.make_head(c.cfg, c.dim, 3).state_dict().items()}
    ref = heads.fit(c.bank, c.cfg, rows, epochs=12, stats=c.stats, forbidden=c.forbid_outer, seed=0, schedule_epochs=12)
    want = heads.predict(c.bank, ref.model, rows, c.cfg, c.stats, use_flip=False)
    rt = torch.as_tensor(rows)
    feat = n11.FeatSource(c.d.n, rows, {p: c.bank.x[p][rt] for p in c.planes}, [], {}, c.planes)
    for micro in (2, 16):
        c.micro = micro
        r = n11.new_run(c, feat, init, rows, 0.0, 0, with_encoder=False, epochs=12)
        n11.run_to(c, r, 12)
        assert r.step == ref.steps_run
        assert np.abs(n11.predict(c, feat, r.head, rows, use_flip=False) - want).max() < 1e-5, micro


def test_trained_model_refused_on_the_test_fold_until_the_rate_is_recorded(ctx):
    from knee import heads
    from knee.nb import n11

    head = heads.make_head(ctx.cfg, ctx.dim, 3)
    head.load_state_dict(ctx.head_state)
    for lr in (2e-5, None):
        with pytest.raises(AssertionError):
            n11.predict_test(ctx, ctx.src, head, trained=True, lr=lr)
    n11.record_chosen(ctx.working, ctx.unit, {"lr": 2e-5, "reason": "test"})
    with pytest.raises(AssertionError):
        n11.predict_test(ctx, ctx.src, head, trained=True, lr=8e-5)
    P = n11.predict_test(ctx, ctx.src, head, trained=True, lr=2e-5)
    ref = heads.predict(ctx.bank, head, ctx.te_s, ctx.cfg, ctx.stats)  # flip averaging as heads.predict
    assert (ctx.d.flip_row[ctx.te_s] >= 0).any() and np.abs(P - ref).max() < 2e-3


def test_tie_rules():
    from knee.nb import n11

    lo, mid, hi = n11.LRS
    sel = n11.select_rung0({lo: 0.70, mid: 0.72, hi: 0.76})
    assert sel["continue"] == [hi] and sel["tied"] is False and n11.select_final(sel["continue"], {hi: 0.60})[0] == hi
    sel = n11.select_rung0({lo: 0.70, mid: 0.757, hi: 0.76})
    assert sel["continue"] == [mid, hi] and sel["tied"] is True
    assert n11.select_final(sel["continue"], {mid: 0.78, hi: 0.80})[0] == hi
    assert n11.select_final([mid, hi], {mid: 0.78, hi: 0.785})[0] == mid
    assert n11.select_final([lo, hi], {lo: 0.78, hi: 0.784})[0] == lo and n11.select_final([lo, hi], {lo: 0.78, hi: 0.79})[0] == hi
    assert n11.select_rung0({lo: 0.70, mid: 0.755, hi: 0.76})["tied"] is True
    assert n11.select_rung0({lo: 0.70, mid: 0.7549, hi: 0.76})["tied"] is False
    three = n11.select_rung0({lo: 0.758, mid: 0.757, hi: 0.76})
    assert three["continue"] == [lo, hi] and "third rate" in three["note"]
    assert n11.select_rung0({lo: 0.70, mid: 0.757, hi: 0.76}, rung1_tie=False)["continue"] == [mid]
    assert n11.select_rung0({lo: None, mid: None, hi: None})["continue"] == [mid]


# ---------------------------------------------------------------- decide and dev on hand-made files

ENV = {"kaggle_image_git_commit": "abc", "kaggle_image_build_date": "d", "python": "3.11", "torch": "2.4",
       "cuda": "12.1", "cudnn": 90100, "device_name": "Tesla T4"}
VSET = {0: "CROATIAN", 1: "GREEK", 2: "GERMAN", 3: "BULGARIAN+DUTCH", 4: "GREEK"}


def _sig(Y, strength, rng):
    return 1 / (1 + np.exp(-(strength * (Y - 0.3) * 3 + rng.normal(0, 1, Y.shape))))


def _write_rows(p: Path, uids, fold, P, Y, extra: dict) -> None:
    df = pd.DataFrame(P, columns=FINDINGS)
    for j, f in enumerate(FINDINGS):
        df[f"y_{f}"] = Y[:, j]
    for k, v in extra.items():
        df.insert(0, k, v)
    df.insert(0, "s2_fold", fold)
    df.insert(0, UID, uids)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def _sessions(root: Path, encs=MAIN, strength=None, env_b=None, seed=5, n=70, fold_strength=None):
    """Two sessions' outputs (a: folds 0-2, b: folds 3, 4) with hand-made inner and test files.
    strength: {arm: signal}; fold_strength: {(fold, arm): signal} overrides it for one fold."""
    from knee.nb import n11

    strength = {"start": 0.8, "head_only_seed_0": 0.9, "head_only_seed_100": 0.9, "b_lite": 1.8, **(strength or {})}
    rng = np.random.default_rng(seed)
    val = {v: (np.array([f"v{v}{i}" for i in range(n)]), (rng.random((n, 12)) < 0.3).astype(float)) for v in set(VSET.values())}
    truth = {}
    for e in encs:
        for k in range(5):
            sess = "a" if k < 3 else "b"
            ud = root / f"knee-n11-{sess}" / "private" / f"n11_{sess}" / f"{e}_f{k}"
            (root / f"knee-n11-{sess}" / "run.json").parent.mkdir(parents=True, exist_ok=True)
            (root / f"knee-n11-{sess}" / "run.json").write_text(json.dumps({"commit": "c0ffee"}))
            uids, Y = val[VSET[k]]
            st = {"encoder": e, "fold": k, "session": sess, "schedule": "full", "inner_validation_set": VSET[k],
                  "environment": dict(ENV if sess == "a" or env_b is None else env_b), "frozen0": {"valid": True},
                  "inner": {"complete": True, "chosen_lr": 2e-5, "keep_rule_arms": {}}, "files_sha16": {}}
            for arm in n11.INNER_ARMS:
                P = _sig(Y, (fold_strength or {}).get((k, arm), strength[arm]), rng)
                _write_rows(ud / f"inner_{arm}.csv", uids, k, P, Y, {"val_set": VSET[k]})
                st["inner"]["keep_rule_arms"]["b_lite_2e-05" if arm == "b_lite" else arm] = round(n11._wmacro(P, Y), 4)
                st["files_sha16"][f"inner_{arm}.csv"] = n11.sha16(ud / f"inner_{arm}.csv")
                truth[(e, k, arm)] = (P, Y)
            tu = np.array([f"t{k}_{i}" for i in range(n)])
            TY = (rng.random((n, 12)) < 0.3).astype(float)
            for arm, sg in [("frozen0", 0.8), ("b_lite", 1.5)] + [(f"head_only_seed_{i}", 1.0) for i in range(5)]:
                P = _sig(TY, sg + 0.2 * MAIN.index(e), rng)
                _write_rows(ud / f"pred_{arm}.csv", tu, k, P, TY, {"resolved": np.arange(n) % 10 != 0})
                st["files_sha16"][f"pred_{arm}.csv"] = n11.sha16(ud / f"pred_{arm}.csv")
                truth[(e, k, "pred_" + arm)] = (P, TY)
            (ud / "state.json").write_text(json.dumps(st))
    return truth


def test_decide_on_hand_made_inner_files(tmp_path, monkeypatch):
    from knee import agg, paths
    from knee.nb import n11

    root = tmp_path / "in"
    truth = _sessions(root)
    opened = []
    real = n11._read_rows
    monkeypatch.setattr(n11, "_read_rows", lambda p: (opened.append(Path(p).name), real(p))[1])
    units = n11.find_units(root)  # two sessions' outputs merged
    assert len(units) == 10 and {ud.parent.name for ud in units.values()} == {"n11_a", "n11_b"}
    rec = n11.decision(units, n_boot=60)
    assert opened and all(n.startswith("inner_") for n in opened)  # no test-fold file is opened
    assert rec["status"] == "decided" and rec["sessions"] == ["a", "b"] and rec["n_null"] == 10
    assert rec["sha256"] == n11._signed({k: v for k, v in rec.items() if k != "sha256"})["sha256"] and rec["written_at_utc"]
    agg._check(rec)
    sc = {key: round(n11._wmacro(*v), 4) for key, v in truth.items() if not key[2].startswith("pred_")}
    null = [sc[(e, k, "head_only_seed_100")] - sc[(e, k, "head_only_seed_0")] for e in MAIN for k in range(5)]
    s = float(np.std(null, ddof=1))
    assert rec["s"] == pytest.approx(s) and rec["threshold"] == round(max(0.005, 2 * s / 5 ** 0.5), 4)
    assert set(rec["validation_sets"]) == {"CROATIAN", "GREEK", "GERMAN", "BULGARIAN+DUTCH"}  # four distinct sets
    for e in MAIN:
        r = rec["encoders"][e]
        means = {a: np.mean([sc[(e, k, n)] for k in range(5)]) for a, n in (("start", "start"), ("head_only", "head_only_seed_0"))}
        base = "start" if round(means["start"], 4) > round(means["head_only"], 4) else "head_only"
        bn = "start" if base == "start" else "head_only_seed_0"
        diffs = [round(sc[(e, k, "b_lite")] - sc[(e, k, bn)], 4) for k in range(5)]
        assert r["baseline"] == base and [r["per_fold"][f"fold_{k}"]["difference"] for k in range(5)] == diffs
        assert r["gain"] == round(float(np.mean(diffs)), 4) and r["folds_up"] == sum(x > 0 for x in diffs) == 5
        assert r["interval"]["lo"] > 0 and r["keep"] is True and all(r["conditions"].values())
        assert r["interval"]["lo"] <= r["gain"] <= r["interval"]["hi"]
    assert set(rec["prediction_files_sha16"]) == {f"{e}_f{k}" for e in MAIN for k in range(5)}
    assert all(len(v) == 7 for v in rec["prediction_files_sha16"].values())
    # the notebook step writes the same record
    monkeypatch.setattr(paths, "WORKING", tmp_path / "wd")
    rec2 = n11.decide(root=root, n_boot=60, quiet=True)
    assert rec2["encoders"] == rec["encoders"] and json.loads((tmp_path / "wd" / "agg_n11_decide.json").read_text()) == rec2
    # one draw per distinct validation set: folds 1 and 4 (both GREEK) get the same weights
    seen = {}
    real_w = n11._wmacro
    monkeypatch.setattr(n11, "_wmacro", lambda P, Y, w=None: (seen.setdefault(len(seen), (Y, w)), real_w(P, Y, w))[1])
    n11.decision({k: v for k, v in units.items() if k[0] == MAIN[1]}, n_boot=2)
    drawn = [(Y, w) for Y, w in seen.values() if w is not None]
    greek = [w for Y, w in drawn[:10] if np.array_equal(Y, truth[(MAIN[1], 1, "b_lite")][1])]
    assert len(greek) == 4 and all(np.array_equal(greek[0], w) for w in greek)  # 2 folds x 2 arms, one draw


def test_decide_rule_cases(tmp_path, monkeypatch):
    from knee import paths
    from knee.nb import n11

    # baseline choice: the starting head is the better baseline when head-only training hurts
    _sessions(tmp_path / "a", strength={"start": 1.2, "head_only_seed_0": 0.6, "head_only_seed_100": 0.6})
    rec = n11.decision(n11.find_units(tmp_path / "a"), n_boot=40)
    assert all(r["baseline"] == "start" for r in rec["encoders"].values())
    # 4 of 5: b-lite below the baseline in two folds is not kept, whatever the mean
    fs = {(k, "b_lite"): 0.2 for k in (0, 2)}
    _sessions(tmp_path / "b", fold_strength=fs)
    rec = n11.decision(n11.find_units(tmp_path / "b"), n_boot=40)
    for r in rec["encoders"].values():
        assert r["folds_up"] == 3 and r["conditions"]["b_at_least_4_of_5_folds_up"] is False and r["keep"] is False
    monkeypatch.setattr(paths, "WORKING", tmp_path / "wd")  # end to end through agg.write with folds_up 3
    rec = n11.decide(root=tmp_path / "b", n_boot=40, quiet=True)
    assert rec["status"] == "decided" and all(r["folds_up"] == 3 for r in rec["encoders"].values())
    assert json.loads((tmp_path / "wd" / "agg_n11_decide.json").read_text())["sha256"] == rec["sha256"]
    # the threshold: a noisy null seed raises it above a small gain
    _sessions(tmp_path / "c", strength={"b_lite": 0.95, "head_only_seed_100": 0.3})
    rec = n11.decision(n11.find_units(tmp_path / "c"), n_boot=40)
    assert rec["threshold"] > 0.005 and rec["threshold"] == round(2 * rec["s"] / 5 ** 0.5, 4)
    assert all(r["keep"] is (r["gain"] > rec["threshold"] and r["folds_up"] >= 4 and r["interval_excludes_0"])
               for r in rec["encoders"].values())
    # ConvNeXt-T only (12.5's trigger): s from its 5 differences, no development Delta1
    _sessions(tmp_path / "d", encs=(MAIN[1],))
    rec = n11.decision(n11.find_units(tmp_path / "d"), n_boot=40)
    assert rec["n_null"] == 5 and rec["encoders_run"] == [MAIN[1]] and rec["development_delta1_reported"] is False
    # the two sessions must share one environment, field by field
    _sessions(tmp_path / "e", env_b={**ENV, "torch": "2.5"})
    rec = n11.decision(n11.find_units(tmp_path / "e"), n_boot=40)
    assert rec["status"] == "env_mismatch" and rec["env_differences"] == ["torch"]
    assert all(r == {"keep": False} for r in rec["encoders"].values())
    # refused: a missing fold, an inner file changed after the run, two units for one fold
    units = n11.find_units(tmp_path / "d")
    with pytest.raises(ValueError):
        n11.decision({k: v for k, v in units.items() if k[1] != 4}, n_boot=10)
    p = units[(MAIN[1], 2)] / "inner_b_lite.csv"
    p.write_text(p.read_text().replace("GERMAN", "german"))
    with pytest.raises(ValueError):
        n11.decision(units, n_boot=10)
    shutil.copytree(tmp_path / "a" / "knee-n11-a", tmp_path / "a" / "knee-n11-a2")
    with pytest.raises(ValueError):
        n11.find_units(tmp_path / "a")


def test_dev_needs_the_decision_and_matching_files(tmp_path, monkeypatch):
    from knee import agg, paths
    from knee.nb import n11

    root = tmp_path / "in"
    truth = _sessions(root)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    with pytest.raises(ValueError):  # no decision attached
        n11.dev(root=root, quiet=True)
    rec = n11.decision(n11.find_units(root), n_boot=30)
    for bad in ({**rec, "status": "env_mismatch"}, {k: v for k, v in rec.items() if k != "status"},
                {**rec, "encoders": {e: {**v, "keep": not v["keep"]} for e, v in rec["encoders"].items()}}):
        with pytest.raises(ValueError):  # no decision, or a record edited after it was written
            n11.dev(bad, root=root, quiet=True)
    (root / "knee-n11-decide").mkdir()
    (root / "knee-n11-decide" / "agg_n11_decide.json").write_text(json.dumps(rec))
    s = n11.dev(root=root, n_boot=20, quiet=True)  # the attached decision file is found
    assert s["status"] == "ok" and s["label"].startswith("development") and s["decision"]["sha256"] == rec["sha256"]
    agg._check(json.loads((tmp_path / "w" / "agg_n11_dev.json").read_text()))
    for e in MAIN:
        r = s["encoders"][e]
        for arm in ("frozen0", "head_only_seed_0", "b_lite"):
            parts = [truth[(e, k, "pred_" + arm)] for k in range(5)]
            P, Y = np.concatenate([p for p, _ in parts]), np.concatenate([y for _, y in parts])
            fold = np.repeat(np.arange(5), [len(p) for p, _ in parts])
            assert r[arm]["mean_of_fold_macros"] == pytest.approx(np.mean([_macro(p, y) for p, y in parts]), abs=6e-5)
            ref = 0.0
            for j in range(12):  # pairs inside one fold, pair-weighted
                num = den = 0.0
                for k in range(5):
                    m = fold == k
                    w = (Y[m, j] == 1).sum() * (Y[m, j] == 0).sum()
                    num += sk.roc_auc_score(Y[m, j], P[m, j]) * w
                    den += w
                ref += num / den / 12
            assert r[arm]["fold_stratified_macro_auc"] == pytest.approx(ref, abs=6e-5)
        assert r["label"] == n11.DEV_LABEL and len(r["head_only_seed_values"]) == 5 and r["decision_keep"] is True
        assert r["b_lite_minus_frozen0"]["mean_of_fold_macros"] == pytest.approx(
            r["b_lite"]["mean_of_fold_macros"] - r["frozen0"]["mean_of_fold_macros"], abs=2e-4)
    d1 = s["delta1"]
    assert d1["label"] == n11.DEV_LABEL and d1["change"]["mean_of_fold_macros"] == pytest.approx(
        d1["b_lite"]["mean_of_fold_macros"] - d1["frozen0"]["mean_of_fold_macros"], abs=2e-4)
    assert set(s["side_discrimination"]) >= {"frozen0", "b_lite"} and s["side_discrimination"]["b_lite"]["label"] == n11.DEV_LABEL
    # a prediction file that is not the one the record lists is refused
    p = n11.find_units(root)[(MAIN[0], 3)] / "pred_b_lite.csv"
    p.write_text(p.read_text().replace("t3_0,", "t3_x,"))
    with pytest.raises(safe_log_failed()):
        n11.dev(rec, root=root, quiet=True)
    # ConvNeXt-T only: no development Delta1
    root2 = tmp_path / "in2"
    _sessions(root2, encs=(MAIN[1],))
    s = n11.dev(n11.decision(n11.find_units(root2), n_boot=20), root=root2, n_boot=20, side_auc=False, quiet=True)
    assert s["status"] == "ok" and s["delta1"]["status"].startswith("not reported") and set(s["encoders"]) == {MAIN[1]}


def safe_log_failed():
    from knee import safe_log

    return safe_log.StepFailed


def test_agg_write_is_atomic_and_writes_the_same_bytes(tmp_path):
    from knee import agg

    payload = {"a": 0.5, "fold": 1, "counts": [10, "<5"], "note": "text"}
    p = agg.write("x", payload, folder=tmp_path)
    assert p.read_text() == json.dumps(payload, indent=1, sort_keys=False, default=agg._json_default)
    assert [q.name for q in tmp_path.iterdir()] == ["agg_x.json"]


# ---------------------------------------------------------------- the real split (needs timm; no download)

@pytest.mark.parametrize("name,lo,hi", [("dinov2_vits14", 5.2e6, 5.4e6), ("convnext_tiny", 4.7e6, 4.9e6)])
def test_timm_split_counts_names_and_forward(name, lo, hi):
    pytest.importorskip("timm")
    from knee.nb import n11

    torch.manual_seed(0)
    sp = n11.TimmSplit(name, "cpu", pretrained=False)
    info = sp.check()
    assert lo <= info["parameters"] <= hi and sp.n_trainable() == info["parameters"]
    assert sum(p.numel() for p in sp.model.parameters() if p.requires_grad) == info["parameters"]
    assert info["drop_path_max"] == 0.0 and info["batch_norm"] is False and not sp.model.training
    named = dict(sp.tail_named_parameters())
    ls = sp.layer_scale_names()
    if name == "dinov2_vits14":
        assert sorted(ls) == sorted(f"blocks.{i}.{k}.gamma" for i in (9, 10, 11) for k in ("ls1", "ls2"))
        assert {"norm.weight", "norm.bias"} <= set(named) and not any(n.startswith("blocks.8.") for n in named)
    else:
        assert ls == ["stages.3.blocks.2.gamma"]
        assert {"head.norm.weight", "head.norm.bias"} <= set(named) and not any(".blocks.1." in n for n in named)
    assert all(sp.no_decay(n, named[n]) for n in ls)
    assert all(sp.no_decay(n, p) == (p.ndim <= 1) for n, p in named.items())  # biases, norm weights, layer-scale
    assert any(not sp.no_decay(n, p) for n, p in named.items())
    with torch.no_grad():  # give layer-scale and norms non-trivial values, then compare the two paths
        for p in sp.model.parameters():
            if p.ndim <= 1:
                p.add_(0.05 * torch.randn_like(p))
    u8 = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
    m = sp.model
    with torch.no_grad():
        x = sp.prep(u8)
        if sp.kind == "vit":  # encoders.Encoder.__call__, in float32
            t = m.forward_features(x)
            ref = torch.cat([t[:, 0], t[:, sp.n_prefix:].mean(1)], dim=1)
        else:
            ref = m.forward_head(m.forward_features(x), pre_logits=True)
        h = sp.trunk(x)
    f = sp.tail(h)  # the tail with a graph, as in training
    assert f.dtype == torch.float32 and f.shape == (3, 768) and f.requires_grad and not h.requires_grad
    assert float((f - ref).abs().max()) < 1e-5 * max(1.0, float(ref.abs().max()))
    f.sum().backward()
    assert all(p.grad is not None for p in named.values())
    assert all(p.grad is None for n, p in m.named_parameters() if n not in named)


# ---------------------------------------------------------------- the notebook entries and their card

def test_n11_cli_entries_match_the_runbook_card():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_notebooks", root / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (root / "notebooks" / "RUNBOOK.md").read_text().split("## Stage B-lite N11 (iteration 3, all five folds)")[1]
    gpu_inputs = [f"u/knee-{u}" for u in ("n0", "n3-r", "n4-sagittal", "n4-coronal", "n4-axial", "n5", "n6a")]
    for name in ("n11-smoke", "n11-a", "n11-b"):
        assert mk.cell2(name).split("schedule=")[0] in card and f"expect={mk.EXPECT}" in mk.cell2(name)
        meta = mk.metadata("u", name)
        assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["kernel_sources"] == gpu_inputs
    assert 'session="a"' in mk.cell2("n11-a") and 'session="b"' in mk.cell2("n11-b") and "smoke=True" in mk.cell2("n11-smoke")
    assert mk.metadata("u", "n11-decide")["kernel_sources"] == ["u/knee-n11-a", "u/knee-n11-b"]
    assert mk.metadata("u", "n11-dev")["kernel_sources"] == ["u/knee-n11-a", "u/knee-n11-b", "u/knee-n11-decide"]
    for name in ("n11-decide", "n11-dev"):
        assert mk.cell2(name) in card and mk.metadata("u", name)["enable_gpu"] is False
    assert "n11" not in mk.NOTEBOOKS
