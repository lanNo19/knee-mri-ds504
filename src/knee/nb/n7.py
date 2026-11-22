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
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, heads, paths, stage
from ..labels import FINDINGS, UID
from . import common


def unit_configs(root: Path | None = None) -> dict:
    """{encoder: {fold: unit summary}} from every attached Stage A unit (a rerun replaces the original)."""
    base = Path(root) if root else paths.INPUT
    found: dict = {}
    for p in sorted(base.glob("**/units/*_f*/config.json")):
        u = json.loads(p.read_text())
        prev = found.setdefault(u["encoder"], {}).get(int(u["fold"]))
        rerun = "-r" in str(p.parent.parent.parent.parent) or "_r" in p.parent.parent.parent.name
        if prev is None or rerun:
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


def run(encoder_names: tuple[str, ...] = ("dinov2_vits14", "convnext_tiny", "vit_s16"), root=None,
        expect: dict | None = None, device: str | None = None, seed: int = 0, budget_h: float = 4.0,
        quiet: bool = True) -> dict:
    s: dict = {}
    with common.notebook("n7", {"encoders": list(encoder_names), "seed": seed, "budget_h": budget_h},
                         quiet=quiet, budget_h=budget_h):
        import torch

        t0 = time.time()
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        d = stage.StageData(root, expect)
        od = paths.out_dir("private", "n7")
        units = unit_configs(Path(root) if root else None)
        s["upstream"] = d.upstream
        s["consensus"], s["missing_units"] = {}, {}
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
        s["files_sha16"] = {p.name: common.sha16(p) for p in sorted(od.glob("*.csv"))}
        s["seconds"] = round(time.time() - t0, 1)
        agg.write("n7", s)
    common.summary({"consensus": {k: {"l1": f"{v['config']['pooling']}|{v['config']['planes']}", "E": v["E"]}
                                  for k, v in s["consensus"].items()},
                    "missing_units": s["missing_units"]})
    return s
