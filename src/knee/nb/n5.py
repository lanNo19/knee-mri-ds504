"""N5: slice embeddings for the three encoders (GPU T4 x2).

Inputs: the three N4 outputs (attached notebook outputs), each holding
private/n4_<plane>/cache_<plane>.u8, its manifest, cache_<plane>_studies.csv
and study_table.csv. The three study tables must agree (same studies, same
order, same laterality), because every array below is indexed by that order.

Writes private/n5/:
  index.csv                        row, StudyInstanceUID, laterality, ok_<plane>, usable, flip_row
  emb_<encoder>_<plane>.npy        float16 [studies, 32, 768], rows as index.csv (zeros where not ok)
  emb_<encoder>_<plane>_flip.npy   coronal and axial only: float16 [unresolved studies, 32, 768],
                                   the same slices mirrored left-right, for test-time averaging of
                                   studies whose laterality is unresolved (row = flip_row)
  manifest.json                    shapes, file hashes, checkpoints, normalisation, versions
and agg_n5.json (counts, checks, throughput). Both GPUs are used: each takes
half of the studies with its own copy of every encoder.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import agg, encoders, paths
from . import common

PLANES = ("sagittal", "coronal", "axial")
FLIP_PLANES = ("coronal", "axial")  # sagittal: laterality only reverses the stack, which pooling ignores
UID = "StudyInstanceUID"


def _devices(devices):
    if devices:
        return list(devices)
    try:
        import torch

        n = torch.cuda.device_count()
        if n:
            return [f"cuda:{i}" for i in range(min(n, 2))]
    except Exception:  # noqa: BLE001
        pass
    return ["cpu"]


def _load_plane(plane: str, root: Path | None):
    man_path = paths.find_file(f"cache_{plane}_manifest.json", root)
    if man_path is None:
        raise FileNotFoundError("cache not attached")
    d = man_path.parent
    man = json.loads(man_path.read_text())
    mm = np.memmap(d / man["file"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    per = pd.read_csv(d / f"cache_{plane}_studies.csv")
    st_path, up = common.upstream("study_table.csv", d / "study_table.csv")
    st = pd.read_csv(st_path)
    return mm, per, st, up, man


def _blocks(rows: list[int], size: int):
    """Batches of row numbers; contiguous runs become slices (fast memmap reads)."""
    for i in range(0, len(rows), size):
        b = rows[i:i + size]
        yield (slice(b[0], b[-1] + 1) if b[-1] - b[0] + 1 == len(b) else b), b


def _embed(mm, rows: list[int], outs: dict, encs_by_dev: dict, flip: bool, studies_per_batch: int,
           out_rows: list[int] | None = None, busy: dict | None = None) -> dict:
    """Embed mm[rows] with every encoder; write into outs[name][out_rows].

    Returns the number of slices per encoder; adds each encoder's busy seconds
    (summed over devices) to `busy`.
    """
    import torch

    out_rows = rows if out_rows is None else out_rows
    devs = list(encs_by_dev)
    halves = np.array_split(np.arange(len(rows)), len(devs))
    errors: list[BaseException] = []
    lock = threading.Lock()

    def work(dev, idx):
        try:
            r_in = [rows[i] for i in idx]
            r_out = [out_rows[i] for i in idx]
            for (sel, b), start in zip(_blocks(r_in, studies_per_batch), range(0, len(r_in), studies_per_batch)):
                x = torch.from_numpy(np.ascontiguousarray(mm[sel])).to(dev)
                if flip:
                    x = x.flip(-1)
                n, s, h, w = x.shape
                x = x.reshape(n * s, h, w)
                tgt = r_out[start:start + len(b)]
                for name, enc in encs_by_dev[dev].items():
                    t1 = time.time()
                    f = enc(x).reshape(n, s, -1).cpu().numpy()
                    if busy is not None:
                        with lock:
                            busy[name] = busy.get(name, 0.0) + time.time() - t1
                    outs[name][tgt] = f
        except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
            errors.append(e)

    threads = [threading.Thread(target=work, args=(d, h), daemon=True) for d, h in zip(devs, halves)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return {name: len(rows) * mm.shape[1] for name in next(iter(encs_by_dev.values()))}


def _gpu_agreement(mm, row: int, encs_by_dev: dict) -> dict:
    """The same slices on every device: minimum cosine per encoder (catches a bad device or dtype)."""
    import torch

    if len(encs_by_dev) < 2:
        return {}
    out = {}
    feats = {}
    for dev, encs in encs_by_dev.items():
        x = torch.from_numpy(np.ascontiguousarray(mm[row])).to(dev)
        for name, enc in encs.items():
            feats.setdefault(name, []).append(enc(x).float().cpu())
    for name, fs in feats.items():
        a, b = fs[0], fs[1]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        out[name] = round(float(cos.min()), 4)
    return out


def run(encoder_names: tuple[str, ...] = tuple(encoders.SPECS), root: str | Path | None = None,
        devices: list[str] | None = None, studies_per_batch: int = 8, budget_h: float = 3.0,
        pretrained: bool = True, model_overrides: dict | None = None, quiet: bool = True) -> dict:
    """root: where to look for the N4 outputs (default /kaggle/input). pretrained and
    model_overrides exist for tests (small random encoders on CPU)."""
    s: dict = {}
    cfg = {"encoders": list(encoder_names), "studies_per_batch": studies_per_batch, "budget_h": budget_h,
           "pretrained": pretrained, "overrides": bool(model_overrides)}
    with common.notebook("n5", cfg, quiet=quiet, budget_h=budget_h):
        import torch

        torch.backends.cudnn.benchmark = True
        t0 = time.time()
        root = Path(root) if root else None
        od = paths.out_dir("private", "n5")
        planes = {p: _load_plane(p, root) for p in PLANES}

        # 1. One study order for all three caches
        ref = planes["sagittal"][2]
        same = all(
            planes[p][2][UID].astype(str).tolist() == ref[UID].astype(str).tolist()
            and planes[p][2]["laterality"].fillna("U").astype(str).tolist() == ref["laterality"].fillna("U").astype(str).tolist()
            for p in PLANES
        )
        if not same:
            raise AssertionError("the three caches were built from different study tables")
        n = len(ref)
        for p in PLANES:
            if planes[p][0].shape[0] != n:
                raise AssertionError("cache rows differ from the study table")
        idx = pd.DataFrame({"row": np.arange(n), UID: ref[UID].astype(str), "laterality": ref["laterality"]})
        for p in PLANES:
            per = planes[p][1].set_index("row")
            idx[f"ok_{p}"] = per["ok"].reindex(idx["row"]).fillna(False).astype(bool).to_numpy()
        idx["usable"] = idx[[f"ok_{p}" for p in PLANES]].all(axis=1)
        unresolved = ~idx["laterality"].isin(["L", "R"])
        idx["flip_row"] = -1
        idx.loc[unresolved, "flip_row"] = np.arange(int(unresolved.sum()))

        # 2. Encoders: one copy per device
        devs = _devices(devices)
        encs_by_dev = {d: {name: encoders.Encoder(name, d, pretrained=pretrained,
                                                  overrides=(model_overrides or {}).get(name))
                           for name in encoder_names} for d in devs}
        first_ok = int(idx.index[idx["ok_sagittal"]][0]) if idx["ok_sagittal"].any() else None
        s["gpu_agreement_min_cosine"] = _gpu_agreement(planes["sagittal"][0], first_ok, encs_by_dev) if first_ok is not None else {}

        # 3. Embeddings, plane by plane (each cache is read once for all encoders)
        files, images, seconds, busy = {}, dict.fromkeys(encoder_names, 0), {}, {}
        D, S = encoders.FEATURE_DIM, planes["sagittal"][0].shape[1]
        n_flip = int(unresolved.sum())
        for p in PLANES:
            mm = planes[p][0]
            rows = idx.index[idx[f"ok_{p}"]].tolist()
            outs = {}
            for name in encoder_names:
                f = od / f"emb_{name}_{p}.npy"
                outs[name] = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(n, S, D))
                files[f.name] = f
            tp = time.time()
            done = _embed(mm, rows, outs, encs_by_dev, False, studies_per_batch, busy=busy)
            seconds[p] = round(time.time() - tp, 1)
            for k, v in done.items():
                images[k] += v
            if p in FLIP_PLANES and n_flip:
                fr = idx.index[unresolved & idx[f"ok_{p}"]].tolist()
                fouts = {}
                for name in encoder_names:
                    f = od / f"emb_{name}_{p}_flip.npy"
                    fouts[name] = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(n_flip, S, D))
                    files[f.name] = f
                if fr:
                    done = _embed(mm, fr, fouts, encs_by_dev, True, studies_per_batch,
                                  out_rows=idx.loc[fr, "flip_row"].astype(int).tolist(), busy=busy)
                    for k, v in done.items():
                        images[k] += v
                for a in fouts.values():
                    a.flush()
            for a in outs.values():
                a.flush()
            del outs

        # 4. Checks on what was written
        finite, zeros_ok, spread, norms, raw_spread = True, True, {}, {}, {}
        rng = np.random.default_rng(0)
        for name in encoder_names:
            for p in PLANES:
                a = np.load(od / f"emb_{name}_{p}.npy", mmap_mode="r")
                rows = idx.index[idx[f"ok_{p}"]].to_numpy()
                for i in range(0, len(rows), 256):
                    if not np.isfinite(a[rows[i:i + 256]]).all():
                        finite = False
                samp = np.sort(rng.choice(rows, size=min(len(rows), 512), replace=False)) if len(rows) else rows
                if len(samp):
                    m = np.asarray(a[samp], dtype=np.float32).mean(axis=1)
                    raw_spread[f"{name}_{p}"] = float(m.std(axis=0).mean())
                    spread[f"{name}_{p}"] = round(raw_spread[f"{name}_{p}"], 4)
                    norms[f"{name}_{p}"] = agg.quantiles(np.linalg.norm(np.asarray(a[samp], np.float32), axis=2).ravel())
                not_ok = idx.index[~idx[f"ok_{p}"]].to_numpy()[:64]
                if len(not_ok) and np.abs(np.asarray(a[not_ok], np.float32)).max() > 0:
                    zeros_ok = False  # rows without a cache must stay zero
        agree = s["gpu_agreement_min_cosine"]
        s["checks"] = {
            "study_tables_match": True,
            "all_finite": bool(finite),
            "missing_rows_zero": bool(zeros_ok),
            "features_vary": bool(raw_spread) and all(v > 1e-3 for v in raw_spread.values()),
            "gpu_agreement": all(v >= 0.999 for v in agree.values()) if agree else None,
            "normalisation_as_planned": True,  # Encoder() refuses otherwise
        }
        elapsed = time.time() - t0
        s["studies"] = agg.count(n)
        s["usable_studies"] = agg.count(int(idx["usable"].sum()))
        s["usable_share"] = agg.share(idx["usable"].mean()) if n else None
        s["unresolved_laterality"] = agg.count(n_flip)
        s["missing_by_plane"] = {p: agg.count(int((~idx[f"ok_{p}"]).sum())) for p in PLANES}
        s["devices"] = [torch.cuda.get_device_name(int(d.split(":")[1])) if d.startswith("cuda") else d for d in devs]
        # slices per second for each encoder alone (its busy time, spread over the devices)
        s["slices_per_s"] = {k: round(v / max(busy.get(k, 0.0) / len(devs), 1e-6), 1) for k, v in images.items()}
        s["seconds_by_plane"] = seconds
        s["seconds"] = round(elapsed, 1)
        s["feature_spread"] = spread
        s["feature_norms"] = norms
        idx.to_csv(od / "index.csv", index=False)
        s["upstream_n4"] = {p: planes[p][3] for p in PLANES}
        s["index_sha16"] = common.sha16(od / "index.csv")
        s["files_sha16"] = {k: common.sha16(v) for k, v in sorted(files.items())}
        e0 = next(iter(encs_by_dev.values()))
        manifest = {
            "shape": [n, S, D], "dtype": "float16", "planes": list(PLANES), "flip_planes": list(FLIP_PLANES),
            "encoders": {k: {"timm_id": encoders.SPECS[k].timm_id, "kind": encoders.SPECS[k].kind,
                             "mean_std": e0[k].mean_std, "pretrained": pretrained} for k in encoder_names},
            "files_sha16": s["files_sha16"], "index_sha16": s["index_sha16"],
        }
        (od / "manifest.json").write_text(json.dumps(manifest, indent=1))
        agg.write("n5", s)
    common.summary({"checks": s["checks"], "usable_studies": s["usable_studies"],
                    "slices_per_s": s["slices_per_s"], "gpu_agreement_min_cosine": s["gpu_agreement_min_cosine"],
                    "index_sha16": s["index_sha16"]})
    return s
