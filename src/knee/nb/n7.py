"""N7: follow-on settings after Stage A (GPU): S1 seen sites and the G-vendor sensitivity.

Inputs: N0, N3, N5 outputs and the Stage A outputs (N6a, N6b; reruns such as
knee-n6a-r too), whose units each hold config.json (configs and E only).

- Consensus config per encoder (plan, "Configs outside the nested loop"):
  pooling and planes = the most common L1 winner across folds (tie: the
  higher mean inner score); L2 settings = those of the fold whose best inner
  score is the median among the folds with that L1 winner (lower middle for an
  even count); E = that fold's E.
- S1 (seen sites): random folds sized like the S2 folds (N3's s1_fold). S1
  fold j uses the config and E chosen for S2 fold j; refit on the S1 training
  set (z-scored with its own statistics), predict the S1 test fold.
- G-vendor (labelled "not nested"): N3's vendor folds with the consensus
  config and E; refit per vendor fold, predict its test fold.
Writes private/n7/{s1,vendor}_<encoder>.csv and agg_n7.json (the consensus
configs, which also go to the repo as configs/consensus.json for KneeMRI S5).

Secondary, descriptive extras (added 23 Nov after the pilot reviews; they change no config, no
decision and no existing file). They live here because they need the embeddings and the device
Stage A's refits ran on, so a seed-0 refit can be compared with the stored one:
- Seed refits (main encoders): each S2 fold's chosen config and E, refit exactly like Stage A's
  refit (n6._unit) with seeds 0..5 -> private/n7/s2seeds_<encoder>.csv. Seed 0 only checks that
  the refit is reproduced; seeds 1..5 are the sensitivity seeds. Decisions stay on Stage A's seed 0.
- Default heads (every encoder), refit the same way -> private/n7/default_<encoder>.csv (the
  fold's L1 winner at the L2 defaults, E from BO trial 0) and default_fixed_<encoder>.csv (gated
  attention over all planes at the L2 defaults, E from l1.csv). Stage A logged only inner scores
  for these configs, so their S2 test score needs this refit.
- Triage thresholds (plan, Phase 6): per fold, the chosen config is fit on each inner training
  part for E epochs and predicts the inner validation fold; the threshold is the highest one that
  keeps 95% sensitivity for "any finding" on those out-of-fold predictions
  -> private/n7/triage_thresholds.csv (N9 applies it to the test fold).
- Near-duplicate recall probe: counts of study pairs above cosine 0.90, 0.95 and 0.98 (the
  pilot's vectors and code path) and how many cross groups or folds -> agg_n7.json.
Every extra fit passes the test fold and all gold studies as `forbidden`, like Stage A.
Order: pass 1 writes the base outputs and agg_n7.json; pass 2 runs the extras one block at a time
(run(extras=...) picks them), records a failing block under extras_errors and writes agg_n7.json
again. So no extra can cost S1, G-vendor or the consensus.
Device: a GPU when there is one, else the CPU (nothing here needs CUDA). "device" is recorded,
because a CPU refit differs from Stage A's GPU refit like another seed (same weights and batch
order, other dropout draws; see n9x.seed_sensitivity). Seeds 1..5 stay valid sensitivity seeds.
The chain guard (knee.units) runs first: the units' test folds must equal the attached S2 folds.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, encoders, heads, metrics, paths, safe_log, stage
from .. import units as unit_files
from ..labels import FINDINGS, UID
from . import common

SEEDS = (0, 1, 2, 3, 4, 5)  # 0 = Stage A's refit seed (a reproduction check), 1..5 = sensitivity seeds
PROBE_THRESHOLDS = (0.90, 0.95, 0.98)  # 0.98 is the pilot's flag threshold
TRIAGE_SENSITIVITY = 0.95
EXTRAS = ("seeds", "defaults", "triage", "probe")  # the secondary blocks, each skippable and each failing alone


def unit_configs(root: Path | None = None) -> dict:
    """{encoder: {fold: unit summary}} from every attached Stage A unit (a rerun replaces the original)."""
    found: dict = {}
    for p in paths.glob_outputs("private/*/units/*_f*/config.json", root):
        u = json.loads(p.read_text())
        prev = found.setdefault(u["encoder"], {}).get(int(u["fold"]))
        if prev is None or common.is_rerun(p.parent):
            found[u["encoder"]][int(u["fold"])] = u
    return found


def consensus(units: dict) -> dict:
    """The plan's consensus rule for one encoder's units {fold: summary}."""
    wins = Counter((u["l1_winner"]["pooling"], u["l1_winner"]["planes"]) for u in units.values())
    top = max(wins.values())
    tied = [w for w, c in wins.items() if c == top]
    if len(tied) > 1:  # tie: the higher mean inner score
        def mean_score(w):
            return np.mean([u["best_inner_score"] for u in units.values()
                            if (u["l1_winner"]["pooling"], u["l1_winner"]["planes"]) == w])
        tied.sort(key=mean_score, reverse=True)
    win = tied[0]
    cand = sorted([u for u in units.values() if (u["l1_winner"]["pooling"], u["l1_winner"]["planes"]) == win],
                  key=lambda u: u["best_inner_score"])
    med = cand[(len(cand) - 1) // 2]
    won = {}
    for u in units.values():
        won.setdefault(f"{u['l1_winner']['pooling']}|{u['l1_winner']['planes']}", []).append(int(u["fold"]))
    # fold numbers sit under "folds"/"fold" keys (structure, not counts of studies)
    return {"config": med["config"], "E": int(med["E"]), "fold": int(med["fold"]),
            "refit_steps": med.get("refit_steps"),
            "l1_winner_folds": {k: {"folds": sorted(v)} for k, v in won.items()},
            "folds": sorted(int(f) for f in units)}


def _refits(d, bank, scheme: str, cfg_for_fold, seed: int = 0) -> np.ndarray:
    P = np.full((d.n, heads.N_OUT), np.nan, np.float32)
    f = d.folds[f"{scheme}_fold"]
    K = int(f[d.train].max() + 1) if (f[d.train] >= 0).any() else 0
    gold = d.gold_rows()
    for k in range(K):
        tr = np.flatnonzero(d.train & (f >= 0) & (f != k))
        te = np.flatnonzero(d.train & (f == k))
        cfg, E = cfg_for_fold(k)
        stats = bank.stats(tr)
        r = heads.fit(bank, cfg, tr, epochs=E, stats=stats, forbidden=np.concatenate([te, gold]), seed=seed)
        P[te] = heads.predict(bank, r.model, te, cfg, stats)
    return P


def s2_refits(d, bank, spec_for_fold, seed: int, stats_cache: dict | None = None) -> np.ndarray:
    """Stage A's refit again (n6._unit): fit on the outer training set for E epochs with the test fold
    and every gold study forbidden, then predict the test fold and its gold studies.
    spec_for_fold(k) -> (config, E) or None (fold left out)."""
    P = np.full((d.n, heads.N_OUT), np.nan, np.float32)
    gold = d.gold_rows()
    cache = {} if stats_cache is None else stats_cache
    for k in range(d.n_folds("s2")):
        spec = spec_for_fold(k)
        if spec is None:
            continue
        cfg, E = spec
        tr, te, _ = d.split(k)
        if k not in cache:
            cache[k] = bank.stats(tr)
        r = heads.fit(bank, cfg, tr, epochs=E, stats=cache[k], forbidden=np.concatenate([te, gold]), seed=seed)
        rows = np.concatenate([te, d.gold_rows(k)])
        P[rows] = heads.predict(bank, r.model, rows, cfg, cache[k])
    return P


def triage_thresholds(d, bank, spec_for_fold, seed: int = 0, stats_cache: dict | None = None) -> list[dict]:
    """Per S2 fold: the "any finding" threshold for 95% sensitivity on inner out-of-fold predictions.

    The fold's chosen config is fit on each inner training part for E epochs (no stopping rule) and
    predicts that inner validation fold. Only outer training studies are used; the test fold and
    all gold studies are forbidden in every fit."""
    out = []
    gold = d.gold_rows()
    cache = {} if stats_cache is None else stats_cache
    for k in range(d.n_folds("s2")):
        cfg, E = spec_for_fold(k)
        tr, te, _ = d.split(k)
        inner = d.inner(k)
        if k not in cache:
            cache[k] = bank.stats(tr)
        score, pos = [], []
        for j in sorted(set(inner[tr][inner[tr] >= 0].tolist())):
            itr, iva = tr[inner[tr] != j], tr[(inner[tr] == j) & d.scored[tr]]
            r = heads.fit(bank, cfg, itr, epochs=E, stats=cache[k], forbidden=np.concatenate([te, gold, iva]),
                          seed=seed + j)
            score.append(heads.predict(bank, r.model, iva, cfg, cache[k]).max(axis=1))
            pos.append((d.eval_y[iva] == 1).any(axis=1))
        score, pos = np.concatenate(score), np.concatenate(pos)
        thr = metrics.threshold_at_sensitivity(score, pos, TRIAGE_SENSITIVITY)
        out.append({"fold": k, "threshold": thr,
                    "inner_sensitivity": float((score[pos] >= thr).mean()) if pos.any() else float("nan"),
                    "inner_specificity": float((score[~pos] < thr).mean()) if (~pos).any() else float("nan")})
    return out


def near_duplicate_probe(d, vectors: np.ndarray, device: str = "cpu") -> dict:
    """Counts of study pairs above each cosine threshold and how many cross groups or folds.

    The same vectors, studies and code path as the pilot's flag (stage.near_duplicates), run once at
    the lowest threshold. Group counts use every pair: a pair is "across" when the two groups differ
    (a study without a group counts as different). Fold counts use only the pairs whose two studies
    are both training studies (pairs_both_training), because only those can sit on two sides of a
    train-test split; a pair with a gold or test-set study is left out of them.
    Counts only, rounded as usual."""
    pairs = stage.near_duplicates(vectors, np.flatnonzero(d.usable), min(PROBE_THRESHOLDS), device)
    ra, rb = pairs["row_a"].astype(int).to_numpy(), pairs["row_b"].astype(int).to_numpy()
    cos = pairs["cosine"].to_numpy(float)
    both = d.train[ra] & d.train[rb]
    groups = {"across_language_groups": d.g_lang, "across_vendor_groups": d.g_vendor}
    fold_cols = {"across_s2_folds": d.folds["s2_fold"], "across_vendor_folds": d.folds["vendor_fold"],
                 "across_s1_folds": d.folds["s1_fold"]}
    out = {}
    for t in PROBE_THRESHOLDS:
        m = cos > t
        mt = m & both
        out[f"cosine_above_{t:.2f}"] = {
            "pairs": agg.count(int(m.sum())),
            **{name: agg.count(int((col[ra[m]] != col[rb[m]]).sum())) for name, col in groups.items()},
            "pairs_both_training": agg.count(int(mt.sum())),
            **{name: agg.count(int((col[ra[mt]] != col[rb[mt]]).sum())) for name, col in fold_cols.items()}}
    return out


def _save(P: np.ndarray, d, path: Path) -> None:
    out = pd.DataFrame(P, columns=FINDINGS)
    out.insert(0, "s2_fold", d.folds["s2_fold"])
    out.insert(0, UID, d.uids)
    out[np.isfinite(P).any(axis=1)].to_csv(path, index=False)


def _seed_block(d, bank, enc: str, chosen, od: Path, seeds, stats: dict) -> dict:
    """Seed refits of the chosen configs -> s2seeds_<encoder>.csv (written only when every seed is done)."""
    parts = []
    for sd in seeds:
        P = s2_refits(d, bank, chosen, sd, stats)
        df = pd.DataFrame(P, columns=FINDINGS)
        df.insert(0, "seed", sd)
        df.insert(0, "s2_fold", d.folds["s2_fold"])
        df.insert(0, UID, d.uids)
        parts.append(df[np.isfinite(P).any(axis=1)])
    pd.concat(parts).to_csv(od / f"s2seeds_{enc}.csv", index=False)
    return {"seeds": [int(x) for x in seeds]}


def _default_block(d, bank, enc: str, us: dict, dirs: dict, od: Path, seed: int, stats: dict) -> dict:
    """Default-head refits -> default_<encoder>.csv and default_fixed_<encoder>.csv."""
    U = d.pos_cap_upper()
    specs = {k: unit_files.default_specs(dirs[(enc, k)], U) for k in us}
    done = {}
    for name, fname in (("l2_defaults", f"default_{enc}.csv"), ("fixed", f"default_fixed_{enc}.csv")):
        if all(name in specs[k] for k in us):
            _save(s2_refits(d, bank, lambda k, name=name: specs[k][name][:2], seed, stats), d, od / fname)
            done[f"default_{name}"] = {"E": {"folds": sorted(us), "epochs": [int(specs[k][name][1]) for k in sorted(us)]},
                                       "E_source": sorted({specs[k][name][2] for k in us})}
    return done


def _run_extras(s: dict, d, units: dict, dirs: dict, od: Path, names: tuple, encoder_names, seeds, seed: int,
                dev: str) -> None:
    """The second pass: every extra block of every encoder on its own, after the base outputs and
    agg_n7.json are on disk. A block that fails is recorded in s["extras_errors"] (error type and
    place only) and the others still run. A leakage error is recorded too and turns
    leakage_assertion_passed false; the guard fires before a fit starts, so that block wrote nothing."""
    import torch

    errors: dict = {}
    triage_rows: list = []

    def attempt(key: str, fn):
        t = time.time()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 - descriptive extras must not stop each other
            errors[key] = safe_log.describe(type(e), e.__traceback__)
            if isinstance(e, heads.LeakageError):
                s["leakage_assertion_passed"] = False
            return None
        s["extras_seconds"][key] = round(time.time() - t, 1)
        return out

    for enc in encoder_names:
        us = units.get(enc, {})
        if not us or s["missing_units"][enc]["folds"] or not all((enc, k) in dirs for k in us):
            continue
        if not {"seeds", "defaults", "triage"} & set(names):
            continue
        bank = attempt(f"{enc}.bank", lambda: d.bank(enc, dev))
        if bank is None:
            continue
        stats: dict = {}
        done: dict = {}

        def chosen(k):
            return heads.config_from(us[k]["config"]), int(us[k]["E"])

        if "seeds" in names and enc in encoders.MAIN and seeds:
            done.update(attempt(f"{enc}.seeds", lambda: _seed_block(d, bank, enc, chosen, od, seeds, stats)) or {})
        if "defaults" in names:
            done.update(attempt(f"{enc}.defaults", lambda: _default_block(d, bank, enc, us, dirs, od, seed, stats)) or {})
        if "triage" in names:
            rows = attempt(f"{enc}.triage", lambda: triage_thresholds(d, bank, chosen, seed, stats))
            if rows:
                triage_rows += [{"encoder": enc, **r} for r in rows]
                pd.DataFrame(triage_rows).to_csv(od / "triage_thresholds.csv", index=False)  # after each encoder
                done["triage"] = True
        s["extras"][enc] = done
        del bank
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    if "probe" in names and encoders.MAIN[0] in encoder_names:
        probe = attempt("probe", lambda: near_duplicate_probe(d, d.study_vectors(encoders.MAIN[0]), dev))
        if probe:
            s["near_duplicate_probe"] = {"encoder": encoders.MAIN[0], **probe}
    s["extras_errors"] = errors
    s["extras_ok"] = not errors


def run(encoder_names: tuple[str, ...] = ("dinov2_vits14", "convnext_tiny", "vit_s16"), root=None,
        expect: dict | None = None, device: str | None = None, seed: int = 0, budget_h: float = 4.0,
        quiet: bool = True, extras: bool | tuple[str, ...] = True, seeds: tuple[int, ...] = SEEDS) -> dict:
    """extras: True = all of EXTRAS, False = none (S1, G-vendor and the consensus still run), or a
    tuple of names from EXTRAS, e.g. ("seeds", "defaults") to leave out the triage fits and the probe.
    device: None picks cuda:0 when there is a GPU, else the CPU; every step runs on either (the
    device is recorded as "device", because a CPU refit is not bit-equal to Stage A's GPU refit).
    Without N6b attached, pass encoder_names=("dinov2_vits14", "convnext_tiny")."""
    s: dict = {}
    names = EXTRAS if extras is True else tuple(extras or ())
    unknown = sorted(set(names) - set(EXTRAS))
    if unknown:
        raise ValueError(f"unknown extras: {unknown}")
    with common.notebook("n7", {"encoders": list(encoder_names), "seed": seed, "budget_h": budget_h,
                                "extras": list(names), "seeds": list(seeds)},
                         quiet=quiet, budget_h=budget_h):
        import torch

        t0 = time.time()
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        d = stage.StageData(root, expect)
        od = paths.out_dir("private", "n7")
        units = unit_configs(Path(root) if root else None)
        dirs = unit_files.unit_dirs(Path(root) if root else None)
        s["upstream"] = d.upstream
        s["device"] = "cuda" if dev.startswith("cuda") else dev
        s["chain"] = unit_files.chain_summary(d, dirs, Path(root) if root else None)
        if s["chain"]["s2_folds_match_units"] is False:
            agg.write("n7", {**s, "status": "chain mismatch"})
            raise unit_files.ChainError("Stage A units do not match the attached S2 folds")
        s["consensus"], s["missing_units"] = {}, {}
        # Pass 1: the base outputs (consensus, S1, G-vendor). H2 and H1's G-vendor arm need them.
        for enc in encoder_names:
            us = units.get(enc, {})
            K = d.n_folds("s2")
            s["missing_units"][enc] = {"folds": [k for k in range(K) if k not in us]}
            if not us:
                continue
            con = consensus(us)
            s["consensus"][enc] = con
            if s["missing_units"][enc]["folds"]:
                continue  # S1 needs every fold's config
            bank = d.bank(enc, dev)
            s1 = _refits(d, bank, "s1", lambda k: (heads.config_from(us[k]["config"]), int(us[k]["E"])), seed)
            ccfg = heads.config_from(con["config"])
            vd = _refits(d, bank, "vendor", lambda k: (ccfg, con["E"]), seed)
            for name, P, scheme in (("s1", s1, "s1"), ("vendor", vd, "vendor")):
                out = pd.DataFrame(P, columns=FINDINGS)
                out.insert(0, f"{scheme}_fold", d.folds[f"{scheme}_fold"])
                out.insert(0, UID, d.uids)
                out[np.isfinite(P).any(axis=1)].to_csv(od / f"{name}_{enc}.csv", index=False)
            del bank
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        s["leakage_assertion_passed"] = True  # heads.fit raises LeakageError otherwise
        s["base_seconds"] = round(time.time() - t0, 1)
        s["extras_requested"] = list(names)
        s["extras"], s["extras_errors"], s["extras_seconds"], s["extras_ok"] = {}, {}, {}, None
        s["files_sha16"] = {p.name: common.sha16(p) for p in sorted(od.glob("*.csv"))}
        s["seconds"] = s["base_seconds"]
        agg.write("n7", s)  # on disk before any extra runs: a later failure cannot cost the base outputs
        # Pass 2: the secondary extras, block by block
        if names:
            _run_extras(s, d, units, dirs, od, names, encoder_names, seeds, seed, dev)
            s["files_sha16"] = {p.name: common.sha16(p) for p in sorted(od.glob("*.csv"))}
            s["seconds"] = round(time.time() - t0, 1)
            agg.write("n7", s)
    common.summary({"consensus": {k: {"l1": f"{v['config']['pooling']}|{v['config']['planes']}", "E": v["E"]}
                                  for k, v in s["consensus"].items()},
                    "missing_units": s["missing_units"], "chain": s.get("chain"), "device": s.get("device"),
                    "leakage_assertion_passed": s.get("leakage_assertion_passed"),
                    "extras_ok": s.get("extras_ok"), "extras_errors": s.get("extras_errors"),
                    "near_duplicate_probe": s.get("near_duplicate_probe")})
    return s
