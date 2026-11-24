"""N10: inference on the competition's test studies and submission.csv (internet OFF; S4).

Inputs: the competition, the offline bundle dataset (private/bundle: wheels, encoder
checkpoints, manifest.json), N0's output (agg_n0.json: the training prevalence for
the per-study fallback) and, except for the canary, the Stage A output (N6a; or the
one-unit smoke): the chosen encoder's S2 fold units (head.pt, stats.pt, config.json).

Canary (canary=True, CPU, Monday): decodes every selected test series with the
training code and fails on purpose if more than 1% of test studies fail to decode
(any plane failed or missing), because a scored or failed canary is the only view of
hidden-test decoding before the real runs. Otherwise it submits the training
prevalence for every study.

Real run, with the same code as training:
  1. study table and series choice for the test split (N0/N4 code);
  2. one uint8 cache per plane (N4's cache.build), under /tmp (not saved);
  3. slice embeddings with the encoder loaded from the bundle checkpoint (its sha256
     is checked against the bundle manifest), plus flipped coronal/axial slices for
     unresolved laterality (N5 code);
  4. the mean over the attached fold heads (flip-averaged for unresolved laterality);
     a study missing any plane gets the training prevalence (per-study fallback).
Writes submission.csv at the output root (Kaggle requires it; Claude never opens it)
and agg_n10.json.

B-lite run (tails="n11"; iteration 3, only if N11's decision keeps the change): steps 1 and 2
as above, then the five fold models of N11's final fits instead of steps 3 and 4. A fold model
is the shared frozen trunk, the fold's fine-tuned tail and the fold's head. Extra inputs: the
outputs of knee-n11-a and knee-n11-b (and of a rerun session), which hold per fold
private/n11_<session>/<encoder>_f<k>/b_lite_seed_0.pt (tail and head) and state.json, and each
session's run.json. knee-n6a stays attached: N11 saves neither the fold's z-score statistics
nor its head config, so both are read from knee-n6a's unit, as N11 itself reads them.
Per batch of studies the trunk runs once and the five tails run on its output; the forward is
N11's own (n11.EncSource, n11.predict: fp16 on a GPU, flip averaging, eval mode).
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import agg, cache, encoders, heads, paths, study_table
from ..labels import FINDINGS, UID
from . import common, n5

PLANES = heads.PLANES
MAX_FAILED_SHARE = 0.01


def _bundle(root: Path | None) -> tuple[Path, dict]:
    hits = paths.glob_outputs("private/bundle/manifest.json", root)
    if not hits:
        raise FileNotFoundError("offline bundle not attached")
    return hits[0].parent, json.loads(hits[0].read_text())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _units(encoder: str, root: Path | None) -> dict:
    out = {}
    for c in paths.glob_outputs(f"private/*/units/{encoder}_f*/config.json", root):
        u = json.loads(c.read_text())
        k = int(u["fold"])
        if k not in out or common.is_rerun(c.parent):  # a rerun replaces the original unit
            out[k] = {"dir": c.parent, "config": heads.config_from(u["config"]),
                      "head_sha16": common.sha16(c.parent / "head.pt")}
    if not out:
        raise FileNotFoundError("Stage A heads not attached")
    return out


def _prevalence(root: Path | None) -> np.ndarray:
    p = paths.find_file("agg_n0.json", root)
    if p is None:
        raise FileNotFoundError("N0's output (training prevalence) not attached")
    prev = json.loads(p.read_text())["labels"]["prevalence_training"]
    return np.array([float(prev[f]) for f in FINDINGS], np.float32)


def series_tiers(st: pd.DataFrame) -> dict:
    """How the series of the test studies were chosen, per plane, as rounded counts (series.choose:
    2 = fluid-sensitive and fat-suppressed, 1 = fluid-sensitive only, 0 = any other series)."""
    names = {2: "fluid_sensitive_fat_suppressed", 1: "fluid_sensitive_only", 0: "any_series"}
    out, top = {}, np.ones(len(st), bool)
    for p in PLANES:
        t = pd.to_numeric(st[f"{p}_tier"], errors="coerce").to_numpy(float)
        out[p] = {**{v: agg.count(int((t == k).sum())) for k, v in names.items()},
                  "no_series": agg.count(int(np.isnan(t).sum()))}
        top &= t == 2
    out["studies_all_planes_top_tier"] = agg.count(int(top.sum()))
    return out


# ---------------------------------------------------------------- B-lite: N11's fold models

TAILS = ("n11",)
BLITE_FILE = "b_lite_seed_0.pt"
N_FOLDS = 5


class TailError(ValueError):
    """The attached N11 outputs are not five complete, matching final fits of the encoder."""


def _state_sha16(sd: dict) -> str:
    """A hash of a state dict's content (names, shapes, dtypes and values, in name order)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        a = sd[k].detach().cpu().numpy()
        h.update(f"{k}|{a.dtype}|{a.shape}|".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _close(a, b) -> bool:
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


def tail_units(encoder: str, root: Path | None = None) -> dict:
    """The five final fits of `encoder` in the attached N11 outputs, or a TailError.

    Refused unless: exactly folds 0-4; every unit complete (frozen0 valid, inner work and final fits
    complete, the weights file present with the hash its state records); one known commit, one schedule
    and one learning-rate rule (rung1_tie and passes of the sessions; every final fit at its unit's
    chosen rate) over the sessions; no session with test overrides. Each fold's z-score statistics and
    head config come from knee-n6a's unit (N11 does not save them); the config must agree with the
    one N11 logged. Returns {"units": {fold: {...}}, "commit", "schedule", "lr_rule"}."""
    from . import n11

    found = {k: ud for (e, k), ud in n11.find_units(root).items() if e == encoder}
    if sorted(found) != list(range(N_FOLDS)):
        raise TailError(f"b-lite needs exactly folds 0-4 of {encoder}; attached: {sorted(found)}")
    units, commits, schedules, rules = {}, set(), set(), set()
    for k, ud in sorted(found.items()):
        b = n11._load_state(ud)
        fin, inner = b.get("final") or {}, b.get("inner") or {}
        wp = ud / BLITE_FILE
        if not (fin.get("complete") and inner.get("complete") and (b.get("frozen0") or {}).get("valid")):
            raise TailError(f"b-lite: the unit of fold {k} is not complete")
        if not wp.exists() or common.sha16(wp) != (b.get("files_sha16") or {}).get(BLITE_FILE):
            raise TailError(f"b-lite: the weights of fold {k} are missing or not the file its state records")
        run = n11._session_run(ud)
        rc = run.get("config") or {}
        if rc.get("test_overrides"):
            raise TailError(f"b-lite: the session of fold {k} ran with test overrides")
        lr = ((fin.get("b_lite") or {}).get("seed_0") or {}).get("encoder_lr")
        if not _close(lr, inner.get("chosen_lr")):
            raise TailError(f"b-lite: the final fit of fold {k} was not run at the unit's chosen learning rate")
        commits.add(str(run.get("commit")))
        schedules.add(str(b.get("schedule")))
        rules.add(json.dumps({"rung1_tie": rc.get("rung1_tie"), "passes": rc.get("passes")}, sort_keys=True))
        sa = paths.glob_outputs(f"private/n6a/units/{encoder}_f{k}/config.json", root)
        if len(sa) != 1:  # knee-n6a only, as n11._context: never an iteration-2 unit or a rerun
            raise TailError(f"b-lite: knee-n6a's unit of fold {k} must be attached exactly once")
        cj = json.loads(sa[0].read_text())
        cfg, logged = heads.config_from(cj["config"]), b.get("config") or {}
        if (cfg.pooling, cfg.planes) != (logged.get("pooling"), logged.get("planes")) or not (
                _close(cfg.dropout, logged.get("dropout")) and _close(cfg.lr, logged.get("head_lr"))
                and _close(cfg.weight_decay, logged.get("head_weight_decay"))):
            raise TailError(f"b-lite: knee-n6a's config of fold {k} is not the one N11 trained with")
        units[k] = {"dir": ud, "weights": wp, "weights_file_sha16": common.sha16(wp), "config": cfg,
                    "stats": sa[0].parent / "stats.pt", "stats_sha16": common.sha16(sa[0].parent / "stats.pt"),
                    "encoder_lr": float(lr), "session": str(b.get("session")),
                    "amp_fp16_in_training": logged.get("amp_fp16")}
    if len(commits) != 1 or "None" in commits:
        raise TailError("b-lite: the N11 sessions must have run at one known commit")
    if len(schedules) != 1 or schedules & {"None"}:
        raise TailError("b-lite: the N11 units ran under different schedules")
    if len(rules) != 1:
        raise TailError("b-lite: the N11 sessions used different learning-rate rules")
    return {"units": units, "commit": commits.pop(), "schedule": schedules.pop(), "lr_rule": json.loads(rules.pop())}


class _Trunk:
    """The shared frozen trunk: one pass per batch of studies (one more, mirrored, for studies of
    unresolved laterality), kept until every fold's tail has read it."""

    def __init__(self, enc, caches: dict, planes, amp: bool):
        from . import n11

        self.planes = tuple(planes)
        self.src = n11.EncSource(enc, caches, self.planes, amp)
        self.kept: dict = {}
        self.slices = 0  # slices that went through the trunk

    def __call__(self, rows, flipped: bool):
        key = (np.asarray(rows).tobytes(), bool(flipped))
        if key not in self.kept:
            x, m, S = self.src.to_device(self.src.read(np.asarray(rows)), flipped)
            self.kept[key] = (self.src.trunk(x), m, S)
            self.slices += int(x.shape[0])
        return self.kept[key]

    def clear(self) -> None:
        self.kept.clear()


class _FoldTail:
    """One fold's feature source for n11.predict: the kept trunk output through this fold's tail."""

    def __init__(self, trunk: _Trunk, planes, tail_state: dict):
        from . import n11

        self.trunk, self.planes, self.tail_state = trunk, tuple(planes), tail_state
        self.src = n11.EncSource(trunk.src.enc, {}, self.planes, trunk.src.amp)  # for its tail() only
        self.at = [trunk.planes.index(p) for p in self.planes]

    def __call__(self, rows, grad: bool = False, flipped: bool = False):
        import torch

        h, m, S = self.trunk(rows, flipped)
        if self.planes != self.trunk.planes:  # a fold whose head reads fewer planes: its planes' slices
            h = torch.cat([h[i * m * S:(i + 1) * m * S] for i in self.at])
        self.trunk.src.enc.load_tail_state(self.tail_state)
        return self.src.tail(h, m, S, False)


def predict_tails(enc, tails: dict, caches: dict, rows, flip_row, eval_studies: int = 4) -> tuple[np.ndarray, dict]:
    """Probabilities [folds, len(rows), 12] of the fold models, and each fold's content hashes.

    enc: an n11.Split holding the frozen trunk's weights. tails: tail_units()'s result. caches:
    {plane: uint8 [studies, S, H, W]}. Every batch of `eval_studies` studies goes through the trunk
    once; n11.predict then runs each fold's tail and head on it (z-scores with the fold's fixed
    Stage A statistics; a study with flip_row >= 0 averages the stored and the mirrored pass)."""
    import torch

    from . import n11

    dev, amp = enc.device, enc.device.type == "cuda"  # fp16 on a GPU, as n11._shared
    names = [n for n, _ in enc.tail_named_parameters()]
    folds, info = [], {}
    for k, u in sorted(tails["units"].items()):
        ck = torch.load(u["weights"], map_location="cpu")
        if ck.get("encoder") != enc.name or int(ck.get("fold", -1)) != k:
            raise TailError(f"b-lite: the weights file of fold {k} belongs to another unit")
        if tuple(ck.get("tail_prefixes") or ()) != tuple(enc.tail_prefixes) or sorted(ck["tail"]) != sorted(names):
            raise TailError(f"b-lite: the tail of fold {k} is not this encoder's trainable tail")
        cfg = u["config"]
        planes = heads.PLANE_SETS[cfg.planes]
        raw = torch.load(u["stats"], map_location="cpu")
        stats = {p: (mu.float().to(dev), sd.float().to(dev)) for p, (mu, sd) in raw.items()}  # as n11._context
        head = heads.make_head(cfg, int(stats[planes[0]][0].shape[-1]), len(planes))
        head.load_state_dict(ck["head"])
        head.to(dev).eval()
        folds.append((k, planes, {n: v.to(dev) for n, v in ck["tail"].items()}, head,
                      SimpleNamespace(device=dev, eval_studies=int(eval_studies), planes=planes,
                                      stats_l=[stats[p] for p in planes], d=SimpleNamespace(flip_row=np.asarray(flip_row)))))
        info[k] = {"tail_sha16": _state_sha16(ck["tail"]), "head_sha16": _state_sha16(ck["head"])}
    used = tuple(p for p in PLANES if any(p in f[1] for f in folds))
    trunk = _Trunk(enc, {p: caches[p] for p in used}, used, amp)
    srcs = [_FoldTail(trunk, planes, tail) for _, planes, tail, _, _ in folds]
    rows = np.asarray(rows)
    out = np.zeros((len(folds), len(rows), len(FINDINGS)), np.float32)
    for i in range(0, len(rows), int(eval_studies)):
        b = rows[i:i + int(eval_studies)]
        for j, (_, _, _, head, c) in enumerate(folds):
            out[j, i:i + len(b)] = n11.predict(c, srcs[j], head, b, use_flip=True)
        trunk.clear()
    if enc.model.training:
        raise AssertionError("the encoder must stay in eval mode")
    return out, {"per_fold": info, "trunk_slices": trunk.slices, "amp_fp16": amp, "planes": list(used)}


def write_submission(sample: pd.DataFrame, uids: np.ndarray, P: np.ndarray, out: Path,
                     fallback: np.ndarray | None = None) -> dict:
    """Fill the competition's sample submission (wide layout: one row per study, 12 probabilities).

    A listed study without a prediction (no series in the test table) gets `fallback`
    (the training prevalence); other layouts are refused until SMOKE has shown the real one."""
    cols = list(sample.columns)
    if UID in cols and all(f in cols for f in FINDINGS):
        sub = sample[[UID]].copy()
        sub[UID] = sub[UID].astype(str)
        pred = pd.DataFrame(P, columns=FINDINGS)
        pred[UID] = [str(u) for u in uids]
        sub = sub.merge(pred, on=UID, how="left")
        missing = sub[FINDINGS].isna().any(axis=1).to_numpy()
        if missing.any():
            if fallback is None:
                raise ValueError("sample submission lists studies without a prediction")
            sub.loc[missing, FINDINGS] = np.tile(np.asarray(fallback, float), (int(missing.sum()), 1))
        sub = sub[cols]
        sub.to_csv(out, index=False)
        return {"layout": "wide", "rows": agg.count(len(sub)), "listed_without_series": agg.count(int(missing.sum()))}
    raise ValueError("unknown submission layout")


def run(encoder: str = "dinov2_vits14", canary: bool = False, root=None, workers: int = 4,
        budget_h: float = 8.0, max_failed_share: float = MAX_FAILED_SHARE, quiet: bool = True,
        tails: str | None = None, encoder_factory=None, eval_studies: int = 4) -> dict:
    """budget_h 8.0: the watchdog ends the step at 8.8 h, inside the 9-hour submission limit.

    tails: None = the fold heads on the frozen encoder's embeddings (the registered path, unchanged);
      "n11" = the five B-lite fold models of the attached N11 outputs (see the module docstring).
    encoder_factory (tests; b-lite only): (name, device) -> an n11.Split, instead of the timm encoder with
      the bundle checkpoint. eval_studies: studies per forward batch of the b-lite path (4, as N11)."""
    if tails is not None and tails not in TAILS:
        raise ValueError('tails: None or "n11"')
    if tails is not None and canary:
        raise ValueError("the canary reads no model: tails must be None")
    if encoder_factory is not None and tails is None:
        raise ValueError("encoder_factory belongs to the b-lite path")
    s: dict = {"canary": canary}
    cfg = {"encoder": encoder, "canary": canary, "workers": workers, "budget_h": budget_h,
           "max_failed_share": max_failed_share}
    if tails is not None:  # the registered path's run.json stays as it was
        cfg.update({"tails": tails, "eval_studies": eval_studies, "test_overrides": encoder_factory is not None})
    with common.notebook("n10", cfg, quiet=quiet, budget_h=budget_h):
        import torch

        t0 = time.time()
        rootp = Path(root) if root else None
        comp = paths.competition_root()
        bdir, man = _bundle(rootp)
        s["bundle_commit"] = (man.get("commit") or "")[:12] or None  # the installed wheel's commit
        prev = _prevalence(rootp)
        tl = tail_units(encoder, rootp) if tails else None  # refused here, before any decoding
        units = {} if canary or tails else _units(encoder, rootp)
        if not canary:  # a wrong bundle fails in seconds, before hours of decoding
            ckpt = bdir / "checkpoints" / f"{encoder}.pth"
            want = man.get("files_sha256", {}).get(ckpt.name)
            if want is None or not ckpt.exists() or _sha256(ckpt) != want:
                raise ValueError("checkpoint does not match the bundle manifest")
        s["mode"] = "canary" if canary else "b-lite" if tails else "heads"
        if tails:
            s["folds"] = sorted(tl["units"])
            s["b_lite"] = {"n11_commit": tl["commit"][:12], "schedule": tl["schedule"], "lr_rule": tl["lr_rule"],
                           "stage_a_unit": "knee-n6a (z-score statistics and head config)",
                           "units": {f"fold_{k}": {"encoder_lr": u["encoder_lr"], "session": u["session"],
                                                   "weights_file_sha16": u["weights_file_sha16"],
                                                   "stage_a_stats_sha16": u["stats_sha16"],
                                                   "amp_fp16_in_training": u["amp_fp16_in_training"]}
                                     for k, u in sorted(tl["units"].items())}}
        else:
            s["folds"] = sorted(units)
            s["heads_sha16"] = {f"fold_{k}": u["head_sha16"] for k, u in sorted(units.items())}
        work = Path(tempfile.mkdtemp(prefix="knee_n10_"))  # caches are large and never saved

        # 1-2. Study table and caches for the test split (training code); half the budget for decoding
        st, _, hdr_err = study_table.build(comp, "test", workers=workers)
        st.to_csv(work / "study_table.csv", index=False)
        n = len(st)
        s["series_tiers"] = series_tiers(st)
        decode_until = t0 + 0.5 * budget_h * 3600
        ok, cache_sum = {}, {}
        for i, p in enumerate(PLANES):
            cd = work / f"cache_{p}"
            left = max(60.0, (decode_until - time.time()) / (len(PLANES) - i))  # an equal share of what is left
            cache_sum[p] = cache.build(st, comp / "test_series", p, cd, workers=workers, budget_s=left)
            per = pd.read_csv(cd / f"cache_{p}_studies.csv").set_index("row")
            ok[p] = per["ok"].reindex(np.arange(n)).fillna(False).astype(bool).to_numpy()
        s["caches"] = {p: {"failed": v["failed"], "missing_series": v["missing_series"],
                           "not_started_budget": v["not_started_budget"]} for p, v in cache_sum.items()}
        s["header_read_errors"] = agg.counts(hdr_err)
        all_ok = ok["sagittal"] & ok["coronal"] & ok["axial"]
        failed_share = float(1 - all_ok.mean()) if n else 0.0
        s["studies"] = agg.count(n)
        s["failed_share"] = agg.share(failed_share)
        s["studies_with_fallback"] = agg.count(int((~all_ok).sum()))
        if failed_share > max_failed_share:
            agg.write("n10", s)
            raise RuntimeError("more than 1% of test studies failed to decode")  # deliberate (canary rule)

        P = np.tile(prev, (n, 1))  # per-study fallback: the training prevalence
        if tails:
            # 3-4 (b-lite). The shared trunk once per batch of studies, then the five fold tails and heads
            from . import n11

            dev = n5._devices(None)[0]
            if encoder_factory is not None:
                enc = encoder_factory(encoder, dev)
            else:
                enc = n11.TimmSplit(encoder, dev, pretrained=False)
                enc.model.load_state_dict(torch.load(ckpt, map_location=dev))
            s["b_lite"]["trainable_tail"] = enc.check()["parameters"]
            caches = {}
            for p in PLANES:
                cd = work / f"cache_{p}"
                cm = json.loads((cd / f"cache_{p}_manifest.json").read_text())
                caches[p] = n11.BlockCache(cd / cm["file"], cm["shape"])
            unresolved = ~st["laterality"].isin(["L", "R"]).to_numpy()
            flip_row = np.full(n, -1)
            flip_row[unresolved] = np.arange(int(unresolved.sum()))
            good = np.flatnonzero(all_ok)
            preds, info = predict_tails(enc, tl, caches, good, flip_row, eval_studies)
            for k, h in info.pop("per_fold").items():
                s["b_lite"]["units"][f"fold_{k}"].update(h)
            s["b_lite"].update({"amp_fp16": info["amp_fp16"], "planes_read": info["planes"],
                                "trunk_passes_per_study": round(info["trunk_slices"] / max(
                                    1, len(good) * len(info["planes"]) * caches[PLANES[0]].shape[1]), 2),
                                "trunk_note": "1.0 = one pass per study; above 1: mirrored passes for unresolved laterality"})
            if len(good):
                P[good] = np.mean(list(preds), axis=0)
            del enc
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        elif not canary:
            # 3. Embeddings with the bundled checkpoint (no download; checked above)
            devs = n5._devices(None)
            encs_by_dev = {}
            for dv in devs:
                e = encoders.Encoder(encoder, dv, pretrained=False)
                e.model.load_state_dict(torch.load(ckpt, map_location=dv))
                encs_by_dev[dv] = {encoder: e}
            unresolved = ~st["laterality"].isin(["L", "R"]).to_numpy()
            flip_row = np.full(n, -1)
            flip_row[unresolved] = np.arange(int(unresolved.sum()))
            emb, flip = {}, {}
            rows = np.flatnonzero(all_ok).tolist()
            for p in PLANES:
                cd = work / f"cache_{p}"
                cm = json.loads((cd / f"cache_{p}_manifest.json").read_text())
                mm = np.memmap(cd / cm["file"], dtype=np.uint8, mode="r", shape=tuple(cm["shape"]))
                emb[p] = np.zeros((n, mm.shape[1], encoders.FEATURE_DIM), np.float16)
                n5._embed(mm, rows, {encoder: emb[p]}, encs_by_dev, False, 8)
                if p in heads.FLIP_PLANES and unresolved.any():
                    flip[p] = np.zeros((int(unresolved.sum()), mm.shape[1], encoders.FEATURE_DIM), np.float16)
                    fr = np.flatnonzero(unresolved & all_ok).tolist()
                    if fr:
                        n5._embed(mm, fr, {encoder: flip[p]}, encs_by_dev, True, 8, out_rows=flip_row[fr].tolist())
            del encs_by_dev
            if devs[0].startswith("cuda"):
                torch.cuda.empty_cache()

            # 4. Fold heads, averaged, for the studies with all three planes
            dummy = np.zeros((n, len(FINDINGS)), np.float32)
            bank = heads.Bank(emb, dummy, dummy, dummy, flip=flip, flip_row=flip_row, device=devs[0])
            good = np.flatnonzero(all_ok)
            preds = []
            for k, u in sorted(units.items()):
                cfg_k = u["config"]
                planes = heads.PLANE_SETS[cfg_k.planes]
                stats = {p: (mu.to(bank.device), sd.to(bank.device))
                         for p, (mu, sd) in torch.load(u["dir"] / "stats.pt", map_location="cpu").items()}
                model = heads.make_head(cfg_k, encoders.FEATURE_DIM, len(planes)).to(bank.device)
                model.load_state_dict(torch.load(u["dir"] / "head.pt", map_location=bank.device))
                preds.append(heads.predict(bank, model, good, cfg_k, stats))
            if len(good):
                P[good] = np.mean(preds, axis=0)
        sample = pd.read_csv(comp / "sample_submission.csv")
        s["submission"] = write_submission(sample, st[UID].astype(str).to_numpy(), P, paths.WORKING / "submission.csv",
                                           fallback=prev)
        s["seconds"] = round(time.time() - t0, 1)
        agg.write("n10", s)
    common.summary({k: s.get(k) for k in ("canary", "submission", "studies", "failed_share", "studies_with_fallback",
                                          "folds", "bundle_commit", "seconds", "mode", "series_tiers")})
    return s
