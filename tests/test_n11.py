"""N11, Stage B-lite (it3_declaration_r3.md), on synthetic data (CPU, no weights downloaded).

A tiny fake encoder pair with the same interface split as the real ones (frozen trunk + trainable tail)
stands in for DINOv2 and ConvNeXt-T. Tested: frozen0 reproduces the head's saved predictions; gradient
accumulation equals one large batch; a rung-1 resume equals an uninterrupted run; the inner refit never
sees validation, test or gold rows; the tie rules; the reading rule's order; head-only with encoder
learning rate 0 leaves the encoder bit-identical; the chosen rate is recorded before a trained model sees
the test fold; the outputs pass agg's checks. With timm: the real split (parameter counts, layer-scale
names, drop-path, the split forward against the model's own feature path)."""
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
        tr, te, _ = d.split(0)
        stats = bank.stats(tr)
        r = heads.fit(bank, cfg, tr, epochs=3, stats=stats, forbidden=np.concatenate([te, d.gold_rows()]), seed=0)
        ud = inp / "n6a" / "private" / "n6a" / "units" / f"{e}_f0"
        ud.mkdir(parents=True)
        rows_out = np.concatenate([te, d.gold_rows(0)])
        pred = pd.DataFrame(heads.predict(bank, r.model, rows_out, cfg, stats), columns=FINDINGS)
        pred.insert(0, "is_gold", d.is_gold[rows_out])
        pred.insert(0, "s2_fold", 0)
        pred.insert(0, UID, d.uids[rows_out])
        pred.to_csv(ud / "predictions.csv", index=False)
        torch.save(r.model.state_dict(), ud / "head.pt")
        torch.save({p: (mu.cpu(), sd.cpu()) for p, (mu, sd) in stats.items()}, ud / "stats.pt")
        (ud / "config.json").write_text(json.dumps({"encoder": e, "fold": 0, "config": GATED, "E": 3}))


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    inp = tmp_path_factory.mktemp("n11") / "input"
    _make_inputs(inp)
    return inp


@pytest.fixture(scope="module")
def e2e(inputs):
    """One full run (both fake encoders, in this process), with spies on the inner refit and the test pass."""
    from knee import heads, paths, stage
    from knee.nb import n11

    work = inputs.parent / "w11"
    seen = {"fit": [], "test": []}
    real_fit, real_test = heads.fit, n11.predict_test

    def spy_fit(bank, cfg, train_idx, *a, **k):
        seen["fit"].append((np.asarray(train_idx).copy(), np.asarray(k.get("forbidden")).copy()))
        return real_fit(bank, cfg, train_idx, *a, **k)

    def spy_test(c, src, head, trained, lr=None):
        seen["test"].append({"encoder": c.name, "trained": trained, "lr": lr,
                             "recorded": lr is not None and n11.chosen_recorded(c.working, c.name, lr),
                             "rows": np.asarray(c.te_s).copy()})
        return real_test(c, src, head, trained, lr)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "WORKING", work)
        mp.setattr(heads, "fit", spy_fit)
        mp.setattr(n11, "predict_test", spy_test)
        s = n11.run(root=inputs, n_boot=40, inner_val_range=None, encoder_factory=fake_factory,
                    min_slices_per_s=1e12, seed1=True, probe_micro=10, budget_h=1.0, quiet=True)
    return SimpleNamespace(s=s, work=work, seen=seen, d=stage.StageData(inputs))


def _task(inputs, work, enc=MAIN[0]):
    return {"root": str(inputs), "expect": None, "working": str(work), "factory": fake_factory, "encoder": enc,
            "device": "cpu", "inner_val_range": None, "eval_studies": 4, "micro": 2}


@pytest.fixture()
def ctx(inputs, tmp_path):
    from knee.nb import n11

    return n11._context(_task(inputs, tmp_path / "w"))


def _macro(P, Y):
    return float(np.mean([sk.roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


# ---------------------------------------------------------------- the full run

def test_run_arms_checks_and_outputs(e2e):
    from knee import agg
    from knee.nb import n11

    s, d = e2e.s, e2e.d
    assert s["status"] == "ok", s.get("errors")
    assert s["schedule"]["name"] == "short" and s["schedule"]["forced"] is False  # the throughput rule fired
    assert s["schedule"]["config"] == {"epochs": 6, "rung0": 2}
    assert s["seed1"]["ran"] is True
    res = s["results"]
    assert res["label"].startswith("development") and res["added_after_results"] is True
    te_s = d.split(0)[2]
    Y = d.eval_y[te_s]
    row_of = {u: i for i, u in enumerate(d.uids)}
    for e in MAIN:
        b = s["encoders"][e]
        # frozen0: this module's forward reproduces the head's saved predictions
        f0 = b["frozen0"]
        assert f0["valid"] is True and f0["check_b"]["passed"] is True
        assert f0["check_b"]["abs_difference"] <= 0.0005 and f0["check_b"]["max_abs_probability_gap"] < 2e-3
        assert f0["check_a"]["max_abs_gap"] < 0.02 and f0["check_a"]["min_cosine"] > 0.999
        ref = pd.read_csv(inputs_unit(e2e, e) / "predictions.csv").astype({UID: str})
        ref = ref[~ref["is_gold"]].set_index(UID).reindex(d.uids[te_s])[FINDINGS].to_numpy()
        mine = pd.read_csv(e2e.work / "private" / "n11" / e / "pred_frozen0.csv").astype({UID: str})
        assert [row_of[u] for u in mine[UID]] == te_s.tolist()  # scored test rows only; no gold row
        assert np.abs(mine[FINDINGS].to_numpy() - ref).max() < 2e-3
        m = res["encoders"][e]["macro_auc"]
        assert m["frozen0"]["estimate"] == pytest.approx(_macro(mine[FINDINGS].to_numpy(), Y), abs=6e-5)
        # the search: three rung-0 runs of 2 epochs, the survivor resumed to epoch 6, every inner score reported
        se = b["search"]
        assert set(se["rung0"]) == {"5e-06", "2e-05", "8e-05"}
        for v in se["rung0"].values():
            assert v["epochs"] == 2 and len(v["curve"]["train_loss"]) == 2 and len(v["curve"]["val_loss"]) == 2
            assert len(v["curve"]["val_macro_auc"]) == 2
        assert se["chosen_lr"] in n11.LRS and n11._lr_key(se["chosen_lr"]) in se["rung1"]
        for k, v in se["rung1"].items():
            assert v["epochs"] == 6 and v["resumed_from_rung0"] is True and len(v["curve"]["val_loss"]) == 6
            assert v["curve"]["val_macro_auc"][:2] == se["rung0"][k]["curve"]["val_macro_auc"]
            assert 1 <= v["val_loss_min"]["epoch"] <= 6
        # final fits: all arms, training loss only
        fin = b["final"]
        assert set(fin["head_only"]) == {f"seed_{k}" for k in range(5)} and set(fin["b_lite"]) == {"seed_0", "seed_1"}
        for v in list(fin["head_only"].values()) + list(fin["b_lite"].values()):
            assert set(v["curve"]) == {"train_loss", "mean_abs_z"} and len(v["curve"]["train_loss"]) == 6
        assert fin["b_lite"]["seed_0"]["encoder_lr"] == se["chosen_lr"]
        assert fin["drift"]["mean_abs_z_frozen0"] > 0 and fin["drift"]["mean_abs_z_b_lite_last_epoch"] > 0
        # scores: every arm, the differences, the reading
        r = res["encoders"][e]
        assert len(r["head_only_seed_macro_aucs"]) == 5 and r["head_only_seed_sd"] is not None
        bl = pd.read_csv(e2e.work / "private" / "n11" / e / "pred_b_lite_seed_0.csv")[FINDINGS].to_numpy()
        bl1 = pd.read_csv(e2e.work / "private" / "n11" / e / "pred_b_lite_seed_1.csv")[FINDINGS].to_numpy()
        assert m["b_lite_seed_0"]["estimate"] == pytest.approx(_macro(bl, Y), abs=6e-5)
        assert m["b_lite"] == pytest.approx((_macro(bl, Y) + _macro(bl1, Y)) / 2, abs=1e-4)
        assert r["b_lite_minus_frozen0"]["estimate"] == pytest.approx(m["b_lite"] - m["frozen0"]["estimate"], abs=2e-4)
        assert r["stricter_baseline"] in ("frozen0", "head_only")
        assert r["effect"] == r["b_lite_minus_" + r["stricter_baseline"]]
        assert r["reading"] in (n11.NOISE, n11.HELPS, n11.LOWERED, n11.SMALL)
        assert r["reading"] == n11.reading(m["b_lite"], m["frozen0"]["estimate"], m["head_only_seed_0"]["estimate"],
                                           r["head_only_seed_sd"], r["b_lite_minus_frozen0"], r["b_lite_minus_head_only"])
    assert res["joint_reading"] == n11.joint_reading(res["readings"])
    d1 = res["delta1_fold0"]
    assert d1["change"]["estimate"] == pytest.approx(d1["b_lite"]["estimate"] - d1["frozen0"]["estimate"], abs=2e-4)
    assert d1["b_lite_is"] == "the mean of seeds 0 and 1"
    assert set(res["side_discrimination"]) >= {"frozen0", "b_lite"}
    for arm in ("frozen0", "b_lite"):
        assert set(res["side_discrimination"][arm]["pairs"]) == {"menisci", "oa"}
    # privacy: aggregates only, through agg's checks; nothing per study at the output root
    for f in ("agg_n11.json", "run.json"):
        text = (e2e.work / f).read_text()
        agg._check(json.loads(text))
        assert not any(u in text for u in d.uids[:50])
    assert sorted(p.name for p in e2e.work.iterdir()) == ["agg_n11.json", "private", "run.json"]


def inputs_unit(e2e, e):
    return e2e.work.parent / "input" / "n6a" / "private" / "n6a" / "units" / f"{e}_f0"


def test_chosen_rate_is_recorded_before_any_trained_test_pass(e2e):
    run_json = json.loads((e2e.work / "run.json").read_text())
    for e in MAIN:
        chosen = e2e.s["encoders"][e]["search"]["chosen_lr"]
        assert run_json["n11_chosen_encoder_lr"][e]["lr"] == chosen
        calls = [c for c in e2e.seen["test"] if c["encoder"] == e]
        assert [c["trained"] for c in calls] == [False] + [True] * 7  # frozen0, head-only x 5, b-lite seeds 0 and 1
        assert all(c["recorded"] and c["lr"] == chosen for c in calls if c["trained"])


def test_inner_refit_never_sees_validation_test_or_gold_rows(e2e, ctx):
    from knee import heads
    from knee.nb import n11

    d = e2e.d
    tr, te, _ = d.split(0)
    itr, iva_all, iva = n11.inner_split(d, tr)
    g = d.g_lang[tr].astype(str)
    assert len(iva_all) == max(np.unique(g, return_counts=True)[1])  # the largest site group of the training set
    assert set(iva.tolist()) <= set(iva_all.tolist()) and not set(itr.tolist()) & set(iva_all.tolist())
    assert len(e2e.seen["fit"]) == 4  # the probe and the search, per encoder
    for train, forb in e2e.seen["fit"]:
        assert set(train.tolist()) == set(itr.tolist())
        assert set(forb.tolist()) >= set(iva_all.tolist()) | set(te.tolist()) | set(d.gold_rows().tolist())
        assert not set(train.tolist()) & set(forb.tolist())
    # the guard raises if a validation, test or gold row would reach the refit
    for bad in (ctx.iva_all[0], ctx.te[0], ctx.gold[0]):
        c2 = copy.copy(ctx)
        c2.itr = np.append(ctx.itr, bad)
        with pytest.raises(heads.LeakageError):
            n11.inner_refit(c2)


def test_invalid_run_stops_and_reads_nothing_else(inputs, tmp_path, monkeypatch, capfd):
    from knee import paths, safe_log
    from knee.nb import n11

    inp = tmp_path / "input"
    shutil.copytree(inputs, inp)
    p = inp / "n6a" / "private" / "n6a" / "units" / "dinov2_vits14_f0" / "predictions.csv"
    df = pd.read_csv(p)
    df[FINDINGS] = df[FINDINGS].to_numpy()[::-1]  # other studies' probabilities: check (b) must fail
    df.to_csv(p, index=False)
    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    monkeypatch.setattr(n11, "inner_refit", lambda *a, **k: pytest.fail("nothing may run after the failed check"))
    monkeypatch.setattr(n11, "new_run", lambda *a, **k: pytest.fail("nothing may run after the failed check"))
    with pytest.raises(safe_log.StepFailed):
        n11.run(root=inp, n_boot=20, inner_val_range=None, encoder_factory=fake_factory, schedule="short",
                blocks=("frozen0", "search", "final", "combine"), quiet=True)
    assert "STEP n11: FAILED - InvalidRun" in capfd.readouterr().out
    a = json.loads((tmp_path / "w" / "agg_n11.json").read_text())
    assert a["status"].startswith("invalid") and "results" not in a
    assert a["encoders"]["dinov2_vits14"]["frozen0"]["check_b"]["passed"] is False
    assert a["encoders"]["convnext_tiny"]["frozen0"]["check_b"]["passed"] is True  # its check is still reported
    assert all(set(v) == {"frozen0"} for v in a["encoders"].values())
    assert set(a["errors"]) == set(MAIN) and all(v.startswith("InvalidRun at n11.py") for v in a["errors"].values())


def test_smoke_reads_no_score(inputs, tmp_path, monkeypatch):
    from knee import paths
    from knee.nb import n11

    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    for name in ("_validate", "predict", "_frozen0", "combine"):
        monkeypatch.setattr(n11, name, lambda *a, **k: pytest.fail("the smoke run must not score anything"))
    s = n11.run(root=inputs, smoke=True, inner_val_range=None, encoder_factory=fake_factory, quiet=True)
    assert s["status"].startswith("ok (smoke") and "results" not in s
    for e in MAIN:
        sm = s["encoders"][e]["smoke"]
        assert sm["optimiser_steps"] == 20 and sm["slices_per_s"] > 0 and sm["reads_no_score"] is True
        assert set(s["encoders"][e]) & {"frozen0", "search", "final"} == set()
    assert (tmp_path / "w" / "agg_n11_smoke.json").exists() and not (tmp_path / "w" / "agg_n11.json").exists()


# ---------------------------------------------------------------- the loop

def test_gradient_accumulation_equals_one_large_batch(ctx):
    from knee.nb import n11

    b = ctx.tr[:16]
    grads = []
    for micro in (2, len(b), 5):
        r = n11.new_run(ctx, ctx.src, ctx.head_state, ctx.tr, 2e-5, 12, 0)
        r.head.eval()  # dropout off: the two passes must see the same function
        n11.batch_backward(ctx, r, b, micro=micro)
        named = ctx.enc.tail_named_parameters() + list(r.head.named_parameters())
        grads.append({n: p.grad.detach().clone() for n, p in named})
    assert all(g.dtype == torch.float32 for g in grads[0].values())
    assert any(float(g.abs().max()) > 1e-4 for n, g in grads[0].items() if n.startswith("block."))
    for other in grads[1:]:
        for n, g in grads[0].items():
            assert float((g - other[n]).abs().max()) < 1e-6, n


def test_rung1_resume_equals_an_uninterrupted_run(ctx, tmp_path):
    from knee.nb import n11

    rows, val = ctx.itr[:48], ctx.iva

    def start():
        return n11.new_run(ctx, ctx.src, ctx.head_state, rows, 1e-3, 3, 0, val_rows=val)

    a = start()
    n11.run_epochs(ctx, a, 2)
    tail_a, head_a = ctx.enc.tail_state(), {k: v.clone() for k, v in a.head.state_dict().items()}
    b = start()
    n11.run_epochs(ctx, b, 1)
    n11.save_checkpoint(ctx, b, tmp_path / "ck.pt")
    other = n11.new_run(ctx, ctx.src, ctx.head_state, rows, 8e-5, 3, 0, val_rows=val)  # another run in between
    n11.run_epochs(ctx, other, 1)
    b2 = start()
    n11.load_checkpoint(ctx, b2, tmp_path / "ck.pt")
    assert b2.epoch == 1 and b2.step == b.step
    n11.run_epochs(ctx, b2, 2)
    assert b2.history == a.history and len(a.history) == 2
    tail_b = ctx.enc.tail_state()
    assert any(not torch.equal(tail_a[n], ctx.enc._tail0[n]) for n in tail_a)  # the encoder did train
    assert all(torch.equal(tail_a[n], tail_b[n]) for n in tail_a)
    assert all(torch.equal(head_a[k], v) for k, v in b2.head.state_dict().items())
    with pytest.raises(AssertionError):  # a checkpoint of another learning rate is refused
        n11.load_checkpoint(ctx, other, tmp_path / "ck.pt")


def test_head_only_with_encoder_rate_0_leaves_the_encoder_bit_identical(ctx):
    from knee.nb import n11

    rows = ctx.tr[:48]
    r = n11.new_run(ctx, ctx.src, ctx.head_state, rows, 0.0, 6, 0)  # the b-lite loop, encoder learning rate 0
    assert len(r.opt.param_groups) == 3 and [g["weight_decay"] for g in r.opt.param_groups][:2] == [0.05, 0.0]
    n11.run_epochs(ctx, r, 2)
    now = ctx.enc.tail_state()
    assert all(torch.equal(now[n], ctx.enc._tail0[n]) for n in now)
    assert any(not torch.equal(v, ctx.head_state[k]) for k, v in r.head.state_dict().items())  # the head trained
    # the arm as it runs: the same loop on the frozen0 features gives the same head
    feat = n11._frozen0(ctx, {})
    r2 = n11.new_run(ctx, feat, ctx.head_state, rows, 0.0, 6, 0, with_encoder=False)
    n11.run_epochs(ctx, r2, 2)
    for k, v in r.head.state_dict().items():
        assert float((v - r2.head.state_dict()[k]).abs().max()) < 1e-5, k
    assert [h["train_loss"] for h in r.history] == pytest.approx([h["train_loss"] for h in r2.history], abs=1e-5)


def test_parameter_groups_schedule_and_modes(ctx):
    from knee import heads
    from knee.nb import n11

    r = n11.new_run(ctx, ctx.src, ctx.head_state, ctx.tr[:32], 2e-5, 12, 0)
    named = dict(ctx.enc.tail_named_parameters())
    no_decay = {n for n, p in named.items() if ctx.enc.no_decay(n, p)}
    assert no_decay == {"block.bias", "ls.gamma", "norm.weight", "norm.bias"}  # biases, layer-scale, norm weights
    assert {id(p) for p in r.opt.param_groups[1]["params"]} == {id(named[n]) for n in no_decay}
    assert r.opt.param_groups[2]["weight_decay"] == ctx.cfg.weight_decay and r.peaks == [2e-5, 2e-5, ctx.cfg.lr]
    assert r.bs == ctx.cfg.batch_size and r.spe == 2
    n11.run_epochs(ctx, r, 1)
    f = heads.schedule_factor(r.step - 1, r.spe, 12)  # the factor of the last step taken
    assert [g["lr"] for g in r.opt.param_groups] == pytest.approx([2e-5 * f, 2e-5 * f, ctx.cfg.lr * f])
    assert not ctx.enc.model.training and r.step == 2
    assert not any(p.requires_grad for n, p in ctx.enc.model.named_parameters() if n not in named)


def test_trained_model_is_refused_on_the_test_fold_until_the_rate_is_recorded(ctx):
    from knee import heads
    from knee.nb import n11

    head = heads.make_head(ctx.cfg, ctx.dim, 3)
    head.load_state_dict(ctx.head_state)
    with pytest.raises(AssertionError):
        n11.predict_test(ctx, ctx.src, head, trained=True, lr=2e-5)
    with pytest.raises(AssertionError):
        n11.predict_test(ctx, ctx.src, head, trained=True)
    n11.record_chosen(ctx.working, ctx.name, {"lr": 2e-5, "reason": "test"})
    with pytest.raises(AssertionError):  # another rate than the recorded one
        n11.predict_test(ctx, ctx.src, head, trained=True, lr=8e-5)
    P = n11.predict_test(ctx, ctx.src, head, trained=True, lr=2e-5)
    assert P.shape == (len(ctx.te_s), 12)
    # flip averaging for unresolved laterality, as heads.predict on the N5 embeddings
    ref = heads.predict(ctx.bank, head, ctx.te_s, ctx.cfg, ctx.stats)
    assert (ctx.d.flip_row[ctx.te_s] >= 0).any() and np.abs(P - ref).max() < 2e-3


# ---------------------------------------------------------------- the rules

def test_tie_rules():
    from knee.nb import n11

    lo, mid, hi = n11.LRS
    # 1. no tie at rung 0: the best continues and is chosen
    sel = n11.select_rung0({lo: 0.70, mid: 0.72, hi: 0.76})
    assert sel["continue"] == [hi] and sel["tied"] is False
    assert n11.select_final(sel["continue"], {hi: 0.60})[0] == hi
    # 2. the top two within 0.005: both continue and the last epoch decides
    sel = n11.select_rung0({lo: 0.70, mid: 0.757, hi: 0.76})
    assert sel["continue"] == [mid, hi] and sel["tied"] is True
    assert n11.select_final(sel["continue"], {mid: 0.78, hi: 0.80})[0] == hi
    assert n11.select_final(sel["continue"], {mid: 0.80, hi: 0.78})[0] == mid
    # 3. still within 0.005: 2e-5 if it is one of the two, else the lower of the two
    assert n11.select_final([mid, hi], {mid: 0.78, hi: 0.785})[0] == mid
    assert n11.select_final([lo, mid], {lo: 0.785, mid: 0.78})[0] == mid
    sel = n11.select_rung0({lo: 0.76, mid: 0.70, hi: 0.758})
    assert sel["continue"] == [lo, hi]
    assert n11.select_final(sel["continue"], {lo: 0.78, hi: 0.784})[0] == lo
    assert n11.select_final(sel["continue"], {lo: 0.78, hi: 0.79})[0] == hi
    # exactly 0.005 apart is a tie; 0.0051 is not (4 decimals)
    assert n11.select_rung0({lo: 0.70, mid: 0.755, hi: 0.76})["tied"] is True
    assert n11.select_rung0({lo: 0.70, mid: 0.7549, hi: 0.76})["tied"] is False
    # the cut "rung 1 for a tied pair": the fallback alone continues
    assert n11.select_rung0({lo: 0.70, mid: 0.757, hi: 0.76}, rung1_tie=False)["continue"] == [mid]
    assert n11.select_rung0({lo: 0.76, mid: 0.70, hi: 0.758}, rung1_tie=False)["continue"] == [lo]
    # no finite score: the registered rate; a diverged run never wins
    assert n11.select_rung0({lo: None, mid: None, hi: None})["continue"] == [mid]
    assert n11.select_rung0({lo: 0.6, mid: 0.7, hi: float("nan")})["continue"] == [mid]
    assert n11.select_final([lo, hi], {lo: 0.7, hi: None})[0] == lo


def test_reading_rule_order_and_joint():
    from knee.nb import n11

    out, inc = {"lo": 0.01, "hi": 0.05}, {"lo": -0.01, "hi": 0.05}
    # 1. noise comes first, even where line 2 would fit
    assert n11.reading(0.78, 0.75, 0.74, 0.02, out, out) == n11.NOISE
    assert n11.reading(0.70, 0.75, 0.74, 0.03, out, out) == n11.NOISE  # a drop inside twice the SD is noise too
    # 2. helps: at least +0.02 over both and both paired intervals exclude 0
    assert n11.reading(0.78, 0.75, 0.74, 0.005, out, out) == n11.HELPS
    assert n11.reading(0.77, 0.75, 0.74, 0.005, out, out) == n11.HELPS  # exactly +0.02 over the stricter baseline
    assert n11.reading(0.78, 0.75, 0.74, 0.005, out, inc) == n11.SMALL  # one interval includes 0
    assert n11.reading(0.78, 0.75, 0.77, 0.004, out, out) == n11.SMALL  # beats only one by 0.02
    # 3. lowered: below both by 0.02 or more (no interval condition)
    assert n11.reading(0.72, 0.75, 0.74, 0.005, inc, inc) == n11.LOWERED
    assert n11.reading(0.73, 0.75, 0.74, 0.001, inc, inc) == n11.SMALL  # below head-only by only 0.01
    # 4. otherwise small or mixed; twice the SD is a strict bound
    assert n11.reading(0.76, 0.75, 0.74, 0.005, out, out) == n11.SMALL
    assert n11.reading(0.76, 0.75, 0.74, 0.0051, out, out) == n11.NOISE
    # joint: helps only if both; any disagreement is mixed
    a, b = MAIN
    assert n11.joint_reading({a: n11.HELPS, b: n11.HELPS}) == n11.HELPS
    assert n11.joint_reading({a: n11.HELPS, b: n11.NOISE}) == "mixed"
    assert n11.joint_reading({a: n11.LOWERED, b: n11.SMALL}) == "mixed"
    assert n11.joint_reading({a: n11.NOISE, b: n11.NOISE}) == n11.NOISE
    assert n11.joint_reading({a: n11.HELPS}) == "not available"


def test_throughput_rule_and_defaults():
    import inspect

    from knee.nb import n11

    assert n11.schedule_for([450.0, 299.9]) == {"name": "short", "epochs": 6, "rung0": 2}
    assert n11.schedule_for([300.0, 620.0]) == {"name": "long", "epochs": 12, "rung0": 3}
    p = inspect.signature(n11.run).parameters
    assert (p["seed1"].default, p["side_auc"].default, p["rung1_tie"].default, p["search"].default) == (None, True, True, True)
    assert p["n_boot"].default == 2000 and p["min_slices_per_s"].default == 300.0 and p["schedule"].default is None
    assert n11.LRS == (5e-6, 2e-5, 8e-5) and n11.ENC_WEIGHT_DECAY == 0.05 and n11.MICRO == 2
    assert n11.HEAD_ONLY_SEEDS == (0, 1, 2, 3, 4) and n11.CHECK_B_TOL == 0.0005 and n11.PROBE_MICRO == 200


def test_cuts_search_and_seed1(inputs, tmp_path, monkeypatch):
    """search=False: the registered 2e-5, its inner run still gives the graph; seed1=False: one b-lite seed."""
    from knee import paths
    from knee.nb import n11

    monkeypatch.setattr(paths, "WORKING", tmp_path / "w")
    s = n11.run(root=inputs, encoders=(MAIN[1],), n_boot=20, inner_val_range=None, encoder_factory=fake_factory,
                schedule="short", blocks=("frozen0", "search", "final", "seed1", "combine"), search=False,
                seed1=False, side_auc=False, quiet=True)
    b = s["encoders"][MAIN[1]]
    assert s["status"] == "ok" and s["cuts"] == {"seed1": True, "side_auc": True, "rung1_for_a_tied_pair": False, "search": True}
    assert b["search"]["chosen_lr"] == 2e-5 and b["search"]["rung0"] == {} and set(b["search"]["rung1"]) == {"2e-05"}
    assert len(b["search"]["rung1"]["2e-05"]["curve"]["val_loss"]) == 6
    assert s["seed1"]["ran"] is False and set(b["final"]["b_lite"]) == {"seed_0"}
    r = s["results"]
    assert r["encoders"][MAIN[1]]["macro_auc"]["b_lite_is"] == "seed 0" and r["joint_reading"] == "not available"
    assert r["delta1_fold0"]["status"].startswith("not available") and r["side_discrimination"]["status"].startswith("cut")


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


# ---------------------------------------------------------------- the notebook entry and its card

def test_n11_cli_entry_matches_the_runbook_card():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_notebooks", root / "scripts" / "make_notebooks.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    card = (root / "notebooks" / "RUNBOOK.md").read_text().split("## Stage B-lite N11 (iteration 3)")[1]
    for name in ("n11", "n11-smoke"):
        assert mk.cell2(name) in card and f"expect={mk.EXPECT}" in mk.cell2(name)
        meta = mk.metadata("u", name)
        assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["competition_sources"] == []
        assert meta["kernel_sources"] == [f"u/knee-{u}" for u in ("n0", "n3-r", "n4-sagittal", "n4-coronal",
                                                                    "n4-axial", "n5", "n6a")]
    assert "smoke=True" in mk.cell2("n11-smoke")
