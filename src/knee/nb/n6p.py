"""N6p: the pilot (GPU T4 x2), before Stage A. Every rule below was fixed in the plan before it runs.

Scoring everywhere uses the plan's rule: uncertain labels count as negative.
The S2 macro is the mean over all 12 findings (fold-stratified, pair-weighted).
1. Positive control: default heads (gated attention over all three planes, the
   L2 defaults), early-stopped on inner fold 0 of each outer fold, must reach an
   S2 macro AUC of at least 0.65 for every encoder. Only pass flags and the
   lowest value are shown; per-encoder values stay private (positive_control.json,
   for Ablay's alignment audit), so the pilot shows nothing about H1's direction.
2. RSNA half-width: the decision interval of Delta1 (DINOv2 - ConvNeXt) on the
   default-head predictions; only widths are reported (no estimate, no bounds).
3. Train-test overlap control (DINOv2): labels permuted across training
   studies; gated attention over all planes; 30 fixed epochs; weight decay
   1e-6. Pass band 0.5 +/- 0.02. Known-answer test: the same with 10% of each
   test fold copied into training must fall outside the band (else the control
   is reported as underpowered; that triggers no audit).
4. Patient overlap: pairs of studies with cosine above 0.98 between z-scored
   DINOv2 study vectors -> private/n6p/pairs.csv (pair_id, uid_a, uid_b,
   cosine), for N3 (flagged pairs stay together in S1) and the montage notebook.
5. Timings, the epoch plateau and the projected Stage A hours.
Both GPUs are used: the control encoder (with the overlap control) on one,
the other encoders on the other.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import agg, encoders, heads, intervals, metrics, paths, stage
from ..labels import FINDINGS
from . import common

POSITIVE_CONTROL_MIN = 0.65
BAND = 0.02
DUP_COSINE = 0.98
KNOWN_ANSWER_SHARE = 0.10
ALL = list(range(len(FINDINGS)))


def _r(x, nd=3):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def _devices(device):
    import torch

    if device:
        return [device]
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(min(n, 2))] or ["cpu"]


def run(encoder_names: tuple[str, ...] = tuple(encoders.SPECS), root=None, expect: dict | None = None,
        device: str | None = None, seed: int = 0, n_boot: int = intervals.N_BOOT, budget_h: float = 3.0,
        control_encoder: str = "dinov2_vits14", quiet: bool = True) -> dict:
    s: dict = {}
    cfg = {"encoders": list(encoder_names), "seed": seed, "n_boot": n_boot, "budget_h": budget_h,
           "control_encoder": control_encoder}
    with common.notebook("n6p", cfg, quiet=quiet, budget_h=budget_h):
        import torch

        devs = _devices(device)
        t0 = time.time()
        d = stage.StageData(root, expect)
        od = paths.out_dir("private", "n6p")
        f2 = d.folds["s2_fold"]
        K = d.n_folds("s2")
        tr_all = np.flatnonzero(d.train & (f2 >= 0))
        ev = tr_all[d.scored[tr_all]]
        Y = d.eval_y
        s["upstream"] = d.upstream
        s["data"] = {"studies": agg.count(d.n), "usable": agg.count(int(d.usable.sum())),
                     "training": agg.count(len(tr_all)), "scored_training": agg.count(len(ev)),
                     "n_folds": K, "n_groups": int(pd.Series(d.g_lang[tr_all]).nunique())}
        s["findings"] = list(FINDINGS)
        # sensitivity set only (labels and folds only): >= 5 positives and 5 negatives in every test fold
        s["sensitivity_5x5_findings"] = [FINDINGS[j] for j in metrics.eligible(Y[ev], f2[ev])] if K >= 2 else []
        if K < 2:
            # Too few groups (SMOKE-ALL's 20 studies): still exercise loading, a one-epoch fit
            # and the near-duplicate pass on the real arrays, then stop.
            s["status"] = "not_testable"
            rows = np.flatnonzero(d.usable & d.train)
            if control_encoder in encoder_names and len(rows) >= 2:
                bank = d.bank(control_encoder, devs[0])
                cfg0 = heads.default_config(20.0)
                r = heads.fit(bank, cfg0, rows, epochs=1, seed=seed)
                p = heads.predict(bank, r.model, rows[:8], cfg0, bank.stats(rows))
                pairs = stage.near_duplicates(d.study_vectors(control_encoder), np.flatnonzero(d.usable),
                                              DUP_COSINE, devs[0])
                s["smoke"] = {"fit_ok": bool(np.isfinite(p).all()), "pairs_ok": bool(len(pairs) >= 0)}
            agg.write("pilot", s)
        else:
            U = d.pos_cap_upper()
            cfg0 = heads.default_config(U)
            s["default_config"] = cfg0.to_dict()
            preds, fits, ctrl, errors = {}, [], {}, []
            lock = threading.Lock()

            def work(dev, encs):
                try:
                    for enc in encs:
                        bank = d.bank(enc, dev)
                        P = np.full((d.n, heads.N_OUT), np.nan, np.float32)
                        for k in range(K):
                            tr, te, _ = d.split(k)
                            inner = d.inner(k)
                            val = tr[(inner[tr] == 0) & d.scored[tr]]
                            fit_tr = tr[inner[tr] != 0]
                            stats = bank.stats(tr)
                            forbid = np.concatenate([te, d.gold_rows()])
                            r = heads.fit(bank, cfg0, fit_tr, val, ALL, stats=stats, forbidden=forbid, seed=seed)
                            rows = np.concatenate([te, d.gold_rows(k)])
                            P[rows] = heads.predict(bank, r.model, rows, cfg0, stats)
                            with lock:
                                fits.append({"encoder": enc, "fold": k, "best_epoch": r.best_epoch,
                                             "epochs_run": r.epochs_run, "seconds": r.seconds,
                                             "per_epoch_s": r.seconds / max(r.epochs_run, 1), "best_inner": r.best_score})
                        np.save(od / f"preds_default_{enc}.npy", P)
                        with lock:
                            preds[enc] = P
                        if enc == control_encoder:
                            ctrl.update(_overlap_control(d, bank, cfg0, K, tr_all, ev, seed))
                        del bank
                        if dev.startswith("cuda"):
                            torch.cuda.empty_cache()
                except BaseException as e:  # noqa: BLE001 - re-raised below
                    errors.append(e)

            first = [e for e in encoder_names if e == control_encoder]
            rest = [e for e in encoder_names if e != control_encoder]
            plan = ({devs[0]: first + rest} if len(devs) == 1 else {devs[0]: first, devs[1]: rest})
            threads = [threading.Thread(target=work, args=(dv, es), daemon=True) for dv, es in plan.items() if es]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            pd.DataFrame(fits).to_csv(od / "fits.csv", index=False)

            # 1. Positive control: flags and the lowest value only (values stay private)
            vals = {enc: metrics.macro(preds[enc][ev], Y[ev], f2[ev], ALL) for enc in encoder_names}
            (od / "positive_control.json").write_text(json.dumps({k: _r(v) for k, v in vals.items()}, indent=1))
            passed = {enc: bool(np.isfinite(v) and v >= POSITIVE_CONTROL_MIN) for enc, v in vals.items()}
            finite = [v for v in vals.values() if np.isfinite(v)]
            s["positive_control"] = {"min": POSITIVE_CONTROL_MIN, "passed": passed, "all_passed": all(passed.values()),
                                     "lowest_s2_macro_auc": _r(min(finite)) if finite else None}

            # 2. Half-width of Delta1 (widths only)
            if all(m in preds for m in encoders.MAIN):
                A, B = preds[encoders.MAIN[0]][ev], preds[encoders.MAIN[1]][ev]
                ye, fe = Y[ev], f2[ev]

                def delta(w):
                    return metrics.macro(A, ye, fe, ALL, w) - metrics.macro(B, ye, fe, ALL, w)

                di = intervals.decision_interval(delta, d.g_lang[ev], n_boot=n_boot, seed=seed)
                bw = (di["bootstrap"]["hi"] - di["bootstrap"]["lo"]) / 2 if di["bootstrap"]["lo"] is not None else None
                jw = (di["jackknife"]["hi"] - di["jackknife"]["lo"]) / 2 if di["jackknife"]["lo"] is not None else None
                s["half_width_delta1"] = {"union": _r(di["half_width"], 4), "bootstrap": _r(bw, 4), "jackknife": _r(jw, 4),
                                          "bootstrap_valid_share": _r(di["bootstrap"]["valid_share"], 2),
                                          "n_groups": di["n_groups"], "approximate": di["approximate"]}
            else:
                s["half_width_delta1"] = None

            # 3. Overlap control
            s["overlap_control"] = ctrl or None

            # 4. Near-duplicate pairs (DINOv2 study vectors)
            pairs = pd.DataFrame(columns=["row_a", "row_b", "cosine"])
            if control_encoder in encoder_names:
                vec = d.study_vectors(control_encoder)
                pairs = stage.near_duplicates(vec, np.flatnonzero(d.usable), DUP_COSINE, devs[0])
            ra, rb = pairs["row_a"].astype(int).to_numpy(), pairs["row_b"].astype(int).to_numpy()
            out = pd.DataFrame({"pair_id": np.arange(len(pairs)), "uid_a": d.uids[ra], "uid_b": d.uids[rb],
                                "cosine": pairs["cosine"].astype(float).round(5).to_numpy()})
            out.to_csv(od / "pairs.csv", index=False)
            s["near_duplicates"] = {
                "threshold": DUP_COSINE, "flagged_pairs": agg.count(len(pairs)),
                "across_language_groups": agg.count(int((d.g_lang[ra] != d.g_lang[rb]).sum())),
                "across_s2_folds": agg.count(int((f2[ra] != f2[rb]).sum())),
                "to_review": agg.count(min(10, len(pairs))),
                "pairs_sha16": common.sha16(od / "pairs.csv"),
            }

            # 5. Plateau and projected Stage A hours
            fdf = pd.DataFrame(fits)
            s["plateau"] = {"best_epoch": agg.quantiles(fdf["best_epoch"]),
                            "share_best_by_20": agg.share((fdf["best_epoch"] <= 20).mean()),
                            "share_ran_30": agg.share((fdf["epochs_run"] >= heads.MAX_EPOCHS).mean())}
            s["projection"] = _projection(d, fdf, K)
            s["leakage_assertion_passed"] = True  # heads.fit raises LeakageError otherwise
            s["seconds"] = round(time.time() - t0, 1)
            agg.write("pilot", s)
    summ = {k: s.get(k) for k in ("positive_control", "half_width_delta1", "overlap_control", "near_duplicates",
                                  "projection", "status")}
    common.summary({k: v for k, v in summ.items() if v is not None})
    return s


def _overlap_control(d, bank, cfg0, K, tr_all, ev, seed) -> dict:
    """Permuted labels: the null (pass band) and the known-answer test (10% of each test fold copied in)."""
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(tr_all)
    soft, weight, binary = d.soft.copy(), d.weight.copy(), d.binary.copy()
    soft[tr_all], weight[tr_all], binary[tr_all] = d.soft[perm], d.weight[perm], d.binary[perm]
    bp = bank.with_labels(soft, weight, binary)
    yp = heads.scoring_labels(binary)
    cfg_c = replace(cfg0, pooling="gated", planes="all", weight_decay=1e-6)
    f2 = d.folds["s2_fold"]
    null = np.full((d.n, heads.N_OUT), np.nan, np.float32)
    known = null.copy()
    for k in range(K):
        tr, te, _ = d.split(k)
        stats = bp.stats(tr)
        r = heads.fit(bp, cfg_c, tr, None, epochs=heads.MAX_EPOCHS, stats=stats,
                      forbidden=np.concatenate([te, d.gold_rows()]), seed=seed)
        null[te] = heads.predict(bp, r.model, te, cfg_c, stats, use_flip=False)
        add = rng.choice(te, size=max(1, int(round(KNOWN_ANSWER_SHARE * len(te)))), replace=False)
        # deliberately leaky (the known answer), so no forbidden set here
        r2 = heads.fit(bp, cfg_c, np.concatenate([tr, add]), None, epochs=heads.MAX_EPOCHS, stats=stats, seed=seed)
        known[te] = heads.predict(bp, r2.model, te, cfg_c, stats, use_flip=False)
    a0 = metrics.macro(null[ev], yp[ev], f2[ev], ALL)
    a1 = metrics.macro(known[ev], yp[ev], f2[ev], ALL)
    passed = bool(np.isfinite(a0) and abs(a0 - 0.5) <= BAND)
    detected = bool(np.isfinite(a1) and abs(a1 - 0.5) > BAND)
    return {"null_macro_auc": _r(a0, 4), "known_answer_macro_auc": _r(a1, 4), "band": [0.5 - BAND, 0.5 + BAND],
            "passed": passed, "known_answer_detected": detected, "underpowered": not detected}


def _projection(d, fdf: pd.DataFrame, K: int) -> dict:
    """Projected Stage A hours from the default fits (gated pooling: an upper bound per epoch).

    N6a runs 4 worker processes on 2 GPUs: the wall time lies between one process's
    total / 4 (CPU-bound) and / 2 (GPU-bound); the upper end is compared with the plan's
    pilot gate (projected N6a above 6 h: simplified Stage A).
    """
    ks = [int(d.inner(k)[d.inner(k) >= 0].max() + 1) for k in range(K) if (d.inner(k) >= 0).any()]
    k_in = int(np.median(ks)) if ks else 4
    out = {"inner_k": k_in}
    unit = {}
    for enc, g in fdf.groupby("encoder"):
        per_epoch = float(g["per_epoch_s"].median())
        avg_epochs = min(heads.MAX_EPOCHS, float(g["best_epoch"].median()) + heads.PATIENCE)
        fits = (len(heads.L1_GRID) + 30) * k_in
        unit[enc] = (fits * avg_epochs + float(g["best_epoch"].median())) * per_epoch / 3600
    out["unit_hours_one_process"] = {k: round(v, 2) for k, v in unit.items()}
    main = [unit[e] for e in encoders.MAIN if e in unit]
    if main:
        one = K * sum(main)
        out["n6a_wall_hours"] = {"low": round(one / 4, 2), "high": round(one / 2, 2)}
        out["n6a_over_6h_gate"] = one / 2 > 6
    if "vit_s16" in unit:
        dm = fdf.loc[fdf["encoder"] == encoders.MAIN[0], "per_epoch_s"]
        demo = 3 * 900 * float(dm.median()) / 3600 if len(dm) else 0.0
        one = K * unit["vit_s16"] + demo
        out["n6b_wall_hours"] = {"low": round(one / 4, 2), "high": round(one / 2, 2)}
    out["corrected"] = _projection_corrected(d, fdf, K)
    return out


BATCH_FACTOR = 1.26  # mean extra optimiser steps of the searched batch sizes (16-128, log) against the default 48


def _projection_corrected(d, fdf: pd.DataFrame, K: int) -> dict:
    """The projection again with the three faults the 23 Nov code review found removed (finding 1).

    The keys above are kept as they were (the pilot gate read them). Here, per unit:
    1. seconds per epoch are scaled from the pilot fit's size (outer training set minus inner
       fold 0) to the mean size of Stage A's inner fits (time per epoch is proportional to steps);
    2. times BATCH_FACTOR for the searched batch sizes;
    3. every search fit is counted at all 30 epochs, i.e. times 30 / (median best epoch + 5):
       low learning rates keep improving, so early stopping often does not fire;
    and the wall time divides one process's total by 2 only (two GPUs; four processes do not give 4x).
    Each unit's refit on the whole outer training set is added. Loading and the sampler are not."""
    size = {}
    for k in range(K):
        tr, _, _ = d.split(k)
        inner = d.inner(k)[tr]
        js = sorted(set(inner[inner >= 0].tolist()))
        pilot_n = int((inner != 0).sum())
        if js and pilot_n:
            size[k] = {"k_in": len(js), "scale": float(np.mean([(inner != j).sum() for j in js])) / pilot_n,
                       "refit_scale": len(tr) / pilot_n}
    fits_per_inner = len(heads.L1_GRID) + 30
    unit: dict = {}
    for enc, g in fdf.groupby("encoder"):
        best = float(g["best_epoch"].median())
        for _, row in g.iterrows():
            z = size.get(int(row["fold"]))
            if z is None:
                continue
            search = fits_per_inner * z["k_in"] * heads.MAX_EPOCHS * row["per_epoch_s"] * z["scale"] * BATCH_FACTOR
            refit = best * row["per_epoch_s"] * z["refit_scale"]
            unit.setdefault(enc, {})[int(row["fold"])] = (search + refit) / 3600
    out = {"unit_hours_one_process": {e: {"folds": sorted(v), "hours": [round(v[k], 2) for k in sorted(v)]}
                                      for e, v in unit.items()},
           "inner_fit_size_factor": {"folds": sorted(size), "factor": [round(size[k]["scale"], 2) for k in sorted(size)]},
           "batch_factor": BATCH_FACTOR,
           "epoch_factor": {e: round(heads.MAX_EPOCHS / min(heads.MAX_EPOCHS, float(g["best_epoch"].median()) + heads.PATIENCE), 2)
                            for e, g in fdf.groupby("encoder")},
           "wall_divisor": 2.0}
    main = [sum(unit[e].values()) for e in encoders.MAIN if e in unit]
    if len(main) == len(encoders.MAIN):
        out["n6a_wall_hours"] = round(sum(main) / 2, 2)
        out["n6a_over_6h"] = bool(sum(main) / 2 > 6)  # information only: the pilot gate used the old key
    if "vit_s16" in unit:  # N6b: the control encoder's units plus the HPO demo (its fits already have the pilot's size)
        dm = fdf.loc[fdf["encoder"] == encoders.MAIN[0], "per_epoch_s"]
        demo = 3 * 900 * float(dm.median()) / 3600 if len(dm) else 0.0
        out["n6b_wall_hours"] = round((sum(unit["vit_s16"].values()) + demo) / 2, 2)
    return out
