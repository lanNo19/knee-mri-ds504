"""Graphs G1 and G2 of the learning-rate extension (it3_lr_extension_r3.md, section 7).

Reads only the aggregate JSON files named on the command line and writes three PNG files. It needs
matplotlib and the standard library, nothing of the knee package, and opens no private file.

    python scripts/plot_lrx.py --it3 agg_n11_a.json agg_n11_b.json --inner agg_n11x_inner.json
        --select agg_n11x_select.json [--refine agg_n11x_refine.json] [--out graphs]

--it3     iteration 3's session aggregates (the DINOv2 units' inner blocks: rung 0, the runs to the end,
          the starting head, the head-only arm, their curves).
--inner   the extension's inner session aggregate(s) (agg_n11x_inner.json; several files are merged).
--select  the signed selection record: each fold's pick, which rates have a five-fold mean by Rule D, the
          best twelfth of each mean curve, Rule D's words. Where an aggregate above is missing, the
          record's own graph data (graphs.g1, graphs.g2) fills in.
--refine  the refinement point's aggregate, if it ran (step 4).

Written to --out:
  g1_steps_to_the_optimum.png  inner validation macro AUC against the encoder learning rate
  g2_fold0.png                 training and validation loss per rate, fold 0's pick and its neighbours
  g2_all_folds.png             the same for every fold and every rate with a curve (appendix)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BLUE, ORANGE = "#2a78d6", "#eb6834"  # the two series colours
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e3e2de"
DINO = "dinov2_vits14"
FOLDS = (0, 1, 2, 3, 4)
TWELFTHS = 12
CAPTION_G1 = "One seed; one site of 260-395 studies per fold; folds 1 and 4 share the Greek site. Added after the results."
CAPTION_G2 = "Training loss with dropout on, validation loss with dropout off. Under each panel: the validation AUC per twelfth."
CAPTION_FOLD0 = ("Fold 0 is the unusual fold (the only one where training lowered the score); "
                 "see the appendix for the others.")


def lr_text(lr: float) -> str:
    m, e = f"{float(lr):e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def lr_key(lr: float) -> str:
    return f"{float(lr):g}"


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _fold_of(unit: str):
    name, _, k = unit.rpartition("_f")
    return int(k) if name == DINO and k.isdigit() else None


def _curve(c: dict | None) -> dict:
    c = c or {}
    return {k: [_num(v) for v in (c.get(k) or [])] for k in ("train_loss", "val_loss", "val_macro_auc")}


def load(it3_files, inner_files, select_file, refine_file) -> dict:
    """Everything the graphs draw, keyed by learning-rate key and fold."""
    d = {"quarter": {}, "end": {}, "curves": {}, "invalid": {}, "start": {}, "head_only": {}, "head_lr": {},
         "steps": {1: set(), 2: set(), 3: set(), 4: set()}}

    def put(what, key, k, v):
        if v is not None:
            d[what].setdefault(key, {})[k] = v

    for f in it3_files or []:
        for unit, b in (json.loads(Path(f).read_text()).get("units") or {}).items():
            k = _fold_of(unit)
            inner = (b or {}).get("inner") or {}
            if k is None or not inner:
                continue
            d["start"][k] = _num((inner.get("start") or {}).get("val_macro_auc"))
            d["head_only"][k] = _num(((inner.get("head_only") or {}).get("seed_0") or {}).get("val_macro_auc"))
            d["head_lr"][k] = _num((b.get("config") or {}).get("head_lr"))
            for key, blk in (inner.get("rung0") or {}).items():
                put("quarter", key, k, _num(blk.get("val_macro_auc")))
                d["steps"][1].add(key)
            for key, blk in (inner.get("to_the_end") or {}).items():
                put("end", key, k, _num(blk.get("val_macro_auc")))
                put("curves", key, k, _curve(blk.get("curve")))
                d["steps"][2].add(key)
    for step, files in ((3, inner_files or []), (4, [refine_file] if refine_file else [])):
        for f in files:
            for key, units in (json.loads(Path(f).read_text()).get("units") or {}).items():
                for unit, b in (units or {}).items():
                    k = _fold_of(unit)
                    run = ((b or {}).get("ext") or {}).get("run") or {}
                    if k is None or not run:
                        continue
                    d["steps"][step].add(key)
                    cv = _curve(run.get("curve"))
                    put("quarter", key, k, _num(run.get("quarter_val_macro_auc")))
                    if run.get("valid"):
                        put("end", key, k, _num(run.get("val_macro_auc")))
                    if any(cv.values()):
                        put("curves", key, k, cv)
                    if run.get("status") == "invalid":
                        seen = [v for v in cv["val_macro_auc"] if v is not None]
                        d["invalid"][(key, k)] = {"k": (run.get("invalid") or {}).get("k"), "y": seen[-1] if seen else None}
                    d["head_lr"].setdefault(k, _num(run.get("head_lr")))
    sel = json.loads(Path(select_file).read_text())
    g1, g2 = (sel.get("graphs") or {}).get("g1") or {}, (sel.get("graphs") or {}).get("g2") or {}
    for what in ("end", "quarter"):  # the record's own graph data where an aggregate was not given
        for key, per in (g1.get(what) or {}).items():
            for f, v in (per or {}).items():
                k = int(f.split("_")[1])
                if k not in d[what].get(key, {}) and (key, k) not in d["invalid"]:
                    put(what, key, k, _num(v))
    for f, blk in (g2.get("folds") or {}).items():
        for key, cv in (blk.get("curves") or {}).items():
            k = int(f.split("_")[1])
            if k not in d["curves"].get(key, {}):
                put("curves", key, k, _curve(cv))
    for name, what in (("start_head", "start"), ("head_only", "head_only")):
        for f, v in (g1.get(name) or {}).items():
            d[what].setdefault(int(f.split("_")[1]), _num(v))
    rd = sel.get("rule_d") or {}
    d["picks"] = {int(f.split("_")[1]): float(v) for f, v in (sel.get("picks") or {}).items()}
    d["means"] = {key: _num(v) for key, v in (rd.get("means") or {}).items() if _num(v) is not None}
    d["best"] = {key: (v.get("best") or {}) for key, v in (rd.get("mean_curve") or {}).items() if v.get("best")}
    d["words"] = rd.get("words_as_shown") or rd.get("words") or ""
    d["words_swap"] = rd.get("words_swap")
    d["status"] = sel.get("status")
    d["panels"] = {int(f.split("_")[1]): list(blk.get("panels") or []) for f, blk in (g2.get("folds") or {}).items()}
    return d


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "legend.frameon": False, "legend.fontsize": 9, "axes.axisbelow": True, "lines.solid_capstyle": "round"})


def draw_g1(d: dict, out: Path, plt) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    keys = sorted({*d["quarter"], *d["end"]}, key=float)
    if not keys:
        raise SystemExit("G1: no score in the files given")
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.08, right=0.84, top=0.80, bottom=0.27)
    ax.set_xscale("log")
    for what, colour in (("quarter", BLUE), ("end", ORANGE)):
        for k in FOLDS:  # thin lines: one fold across the rates it has a score for
            pts = [(float(key), d[what][key][k]) for key in keys if k in d[what].get(key, {})]
            if pts:
                ax.plot(*zip(*pts), color=colour, linewidth=0.8, alpha=0.45, marker="o", markersize=3, zorder=2)
    # thick lines: the five-fold mean, only over rates that have a mean by Rule D
    with_mean = set(d["means"]) | {key for key in d["steps"][1] | d["steps"][2]}
    q = [(float(key), _mean(d["quarter"][key].values())) for key in keys
         if key in with_mean and len(d["quarter"].get(key, {})) == len(FOLDS)]
    e = [(float(key), d["means"][key]) for key in keys if key in d["means"]]
    for pts, colour, name in ((q, BLUE, "quarter mark"), (e, ORANGE, "end of schedule")):
        if pts:
            ax.plot(*zip(*pts), color=colour, linewidth=2.6, marker="o", markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4)
            ax.annotate(name, pts[-1], xytext=(9, 0), textcoords="offset points", color=INK, fontsize=9.5, va="center",
                        fontweight="bold")
    b = [(float(key), _num(v.get("val_macro_auc"))) for key, v in sorted(d["best"].items(), key=lambda kv: float(kv[0]))]
    b = [p for p in b if p[1] is not None]
    if b:
        ax.plot(*zip(*b), color=INK2, linewidth=1.4, linestyle=(0, (4, 3)), marker="D", markersize=4.5, markerfacecolor=SURFACE,
                zorder=3)
        ax.annotate("best twelfth of the mean curve\n(reported, not used)", b[-1], xytext=(9, 14), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="bottom")
    for k, lr in sorted(d["picks"].items()):  # each fold's pick: a filled marker on its thin end-of-schedule line
        y = d["end"].get(lr_key(lr), {}).get(k)
        if y is not None:
            ax.plot([lr], [y], marker="o", markersize=9, color=ORANGE, markeredgecolor=INK, markeredgewidth=1.2, zorder=6,
                    linestyle="none")
            ax.annotate(f"f{k}", (lr, y), xytext=(-9, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5,
                        color=INK, zorder=7)
    lo = min(v for what in ("quarter", "end") for per in d[what].values() for v in per.values())
    for (key, k), inv in sorted(d["invalid"].items()):  # invalid runs: a cross at the last score the run has
        y = inv["y"] if inv["y"] is not None else lo
        ax.plot([float(key)], [y], marker="x", markersize=9, color=INK, markeredgewidth=1.8, linestyle="none", zorder=6)
        ax.annotate(f"f{k} invalid" + (f" at {inv['k']}/12" if inv.get("k") else ""), (float(key), y), xytext=(8, -9),
                    textcoords="offset points", fontsize=8, color=INK2)
    for vals, style, name in ((d["start"], ":", "starting head"), (d["head_only"], "-.", "head-only arm")):
        m = _mean(vals.values()) if len([v for v in vals.values() if v is not None]) == len(FOLDS) else None
        if m is not None:
            ax.axhline(m, color=INK2, linewidth=1.1, linestyle=style, zorder=1)
            ax.annotate(f"{name}, mean {m:.4f}", (0.012, m), xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", fontsize=8.5, color=INK2, va="bottom", ha="left")
    if "5e-06" in d["quarter"] and "5e-06" not in d["end"]:
        y = _mean(d["quarter"]["5e-06"].values())
        ax.annotate("5e-6: dropped at rung 0,\nno end score", (5e-6, y), xytext=(8, -30), textcoords="offset points",
                    fontsize=8, color=INK2, ha="left")
    # the steps, numbered above the graph: which rates each step added
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    names = {1: "rung 0: three rates to the quarter mark", 2: "the survivor and the registered rate to the end",
             3: "the extension: new rates to the end", 4: "the refinement point"}
    row = 0
    for step in (1, 2, 3, 4):
        xs = sorted(float(key) for key in d["steps"][step])
        if not xs:
            continue
        y = 1.04 + 0.055 * row
        row += 1
        ax.plot([xs[0], xs[-1]], [y, y], transform=tr, color=INK2, linewidth=1.2, marker="|", markersize=7, clip_on=False)
        ax.text(xs[-1], y, f"   {step}  {names[step]}", transform=tr, fontsize=8.5, color=INK, va="center", ha="left",
                clip_on=False)
    ax.set_xticks([float(key) for key in keys])
    ax.set_xticklabels([lr_text(float(key)) for key in keys])
    ax.minorticks_off()
    ax.set_xlabel("encoder learning rate (log scale)")
    ax.set_ylabel("inner validation macro AUC")
    fig.suptitle("Steps to the optimum: inner validation macro AUC against the encoder learning rate (DINOv2 B-lite)",
                 x=0.08, ha="left", fontsize=12, fontweight="bold", color=INK)
    handles = [Line2D([], [], color=BLUE, linewidth=2.6, marker="o", label="quarter mark (twelfth 3), mean of five folds"),
               Line2D([], [], color=ORANGE, linewidth=2.6, marker="o", label="end of schedule, mean of five folds"),
               Line2D([], [], color=INK2, linewidth=1.4, linestyle=(0, (4, 3)), marker="D", markerfacecolor=SURFACE,
                      label="best twelfth of the mean curve (reported, not used)"),
               Line2D([], [], color=INK2, linewidth=0.8, alpha=0.6, marker="o", markersize=3, label="thin lines: one fold"),
               Line2D([], [], color=ORANGE, marker="o", markersize=9, markeredgecolor=INK, linestyle="none", label="a fold's pick (f0-f4)"),
               Line2D([], [], color=INK, marker="x", markersize=8, markeredgewidth=1.8, linestyle="none", label="invalid run")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.075, 0.105), ncol=3, columnspacing=1.6, handlelength=2.4)
    picks = ", ".join(f"f{k} {lr_text(lr)}" for k, lr in sorted(d["picks"].items()))
    lines = [f"pick per fold: {picks}", "The mean line is drawn only over rates with a five-fold mean (Rule D). " + CAPTION_G1]
    if d["words"]:
        lines.append("Rule D: " + d["words"] + (f"; {d['words_swap']}" if d.get("words_swap") else ""))
    for i, text in enumerate(lines):
        fig.text(0.08, 0.092 - 0.027 * i, text, fontsize=8.8, color=INK if i == 0 else INK2, ha="left", va="top")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _panel(ax_loss, ax_auc, cv: dict | None, title: str, invalid: dict | None, first: bool, last: bool) -> None:
    ax_loss.set_title(title, loc="left", fontsize=9.5)
    ax_auc.set_xticks(range(1, TWELFTHS + 1, 1))
    ax_auc.set_xlim(0.5, TWELFTHS + 0.5)
    ax_loss.set_xlim(0.5, TWELFTHS + 0.5)
    ax_loss.tick_params(labelbottom=False)
    ax_auc.tick_params(labelsize=7.5)
    if not cv or not any(cv.values()):
        ax_loss.text(0.5, 0.5, "no curve", transform=ax_loss.transAxes, ha="center", va="center", color=INK2, fontsize=9)
        return
    for name, colour, label in (("train_loss", BLUE, "training loss"), ("val_loss", ORANGE, "validation loss")):
        pts = [(i + 1, v) for i, v in enumerate(cv.get(name) or []) if v is not None]
        if not pts:
            continue
        ax_loss.plot(*zip(*pts), color=colour, linewidth=2, marker="o", markersize=3.5, label=label)
        if last:  # direct labels on the last panel of the first row
            ax_loss.annotate(label, pts[-1], xytext=(7, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center",
                             annotation_clip=False)
    auc = [(i + 1, v) for i, v in enumerate(cv.get("val_macro_auc") or []) if v is not None]
    if auc:
        ax_auc.plot(*zip(*auc), color=INK2, linewidth=1.5, marker="o", markersize=3.5)
        top = max(auc, key=lambda p: (p[1], -p[0]))
        close = top != auc[-1] and auc[-1][0] - top[0] < 3  # two labels side by side would collide
        for (x, y), side in {(top, "right" if close else "center"), (auc[-1], "center")}:
            ax_auc.annotate(f"{y:.3f}", (x, y), xytext=(-3 if side == "right" else 0, 5), textcoords="offset points",
                            fontsize=7.5, color=INK, ha=side)
    if invalid:
        ends = [(len(v), v[-1]) for v in (cv.get("train_loss") or [], cv.get("val_loss") or []) if v and v[-1] is not None]
        for x, y in ends[:1]:
            ax_loss.plot([x], [y], marker="x", markersize=9, color=INK, markeredgewidth=1.8, linestyle="none", zorder=5)
        ax_loss.text(0.97, 0.94, f"invalid at twelfth {invalid.get('k') or len(cv.get('train_loss') or [])}",
                     transform=ax_loss.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
    if first:
        ax_loss.set_ylabel("loss")
        ax_auc.set_ylabel("validation\nmacro AUC", fontsize=8.5)


def _limits(curves: list, names) -> tuple | None:
    vals = [v for cv in curves for n in names for v in (cv.get(n) or []) if v is not None]
    if not vals:
        return None
    pad = 0.06 * (max(vals) - min(vals) or 1.0)
    return min(vals) - pad, max(vals) + pad


def _title(d: dict, key: str, k: int) -> str:
    pick = d["picks"].get(k)
    return f"encoder rate {lr_text(float(key))}" + ("  (the fold's pick)" if pick is not None and lr_key(pick) == key else "")


def _grid(d: dict, rows: list, out: Path, plt, title: str, caption: list) -> None:
    """rows: [(fold, [lr keys])]. Per row a loss panel and, under it, the validation AUC panel, per rate.
    One y axis for all loss panels and one for all AUC panels (no second axis anywhere)."""
    from matplotlib.lines import Line2D

    ncol = max(len(keys) for _, keys in rows)
    height = 3.9 * len(rows) + 1.9
    fig = plt.figure(figsize=(max(3.6 * ncol + 1.6, 11.0), height))
    outer = fig.add_gridspec(len(rows), 1, hspace=0.3, left=0.085, right=0.9, top=1 - 1.5 / height,
                             bottom=(0.75 + 0.2 * len(caption)) / height)
    blocks = [g.subgridspec(2, ncol, height_ratios=[3, 1.35], hspace=0.16, wspace=0.12) for g in outer]
    every = [d["curves"].get(key, {}).get(k) or {} for k, keys in rows for key in keys]
    ylim, alim = _limits(every, ("train_loss", "val_loss")), _limits(every, ("val_macro_auc",))
    for i, (k, keys) in enumerate(rows):
        for j in range(ncol):
            a, b = fig.add_subplot(blocks[i][0, j]), fig.add_subplot(blocks[i][1, j])
            if j >= len(keys):
                a.axis("off")
                b.axis("off")
                continue
            key = keys[j]
            _panel(a, b, d["curves"].get(key, {}).get(k), _title(d, key, k), d["invalid"].get((key, k)), j == 0,
                   i == 0 and j == len(keys) - 1)
            if ylim:
                a.set_ylim(*ylim)
            if alim:
                b.set_ylim(alim[0], alim[1] + 0.25 * (alim[1] - alim[0]))
            if j:
                a.tick_params(labelleft=False)
                b.tick_params(labelleft=False)
            if i == len(rows) - 1:
                b.set_xlabel("twelfth of the schedule", fontsize=8.5)
            if j == 0 and len(rows) > 1:  # the appendix line: the head's learning rate beside the encoder's
                head, pick = d["head_lr"].get(k), d["picks"].get(k)
                a.text(-0.22, 1.16, f"fold {k}: head learning rate {lr_text(head) if head else 'n/a'}; "
                       f"encoder pick {lr_text(pick) if pick is not None else 'n/a'}", transform=a.transAxes, fontsize=9.5,
                       fontweight="bold", color=INK, ha="left", va="bottom")
    fig.suptitle(title, x=0.085, y=1 - 0.3 / height, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=BLUE, linewidth=2, marker="o", markersize=3.5, label="training loss (dropout on)"),
                        Line2D([], [], color=ORANGE, linewidth=2, marker="o", markersize=3.5, label="validation loss (dropout off)"),
                        Line2D([], [], color=INK2, linewidth=1.5, marker="o", markersize=3.5, label="validation macro AUC (lower panels)")],
               loc="upper left", bbox_to_anchor=(0.078, 1 - 0.5 / height), ncol=3)
    for i, text in enumerate(caption):
        fig.text(0.085, (0.2 * (len(caption) - i)) / height, text, fontsize=8.8, color=INK2, ha="left", va="center")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fold_panels(d: dict, k: int) -> list:
    """The rates of one fold's G2: its pick and the pick's grid neighbours that have a curve."""
    have = sorted((key for key, per in d["curves"].items() if k in per), key=float)
    listed = [key for key in d["panels"].get(k, []) if key in have]
    if listed:
        return listed
    grid = sorted({*d["quarter"], *d["end"]} - (d["steps"][4] - d["steps"][3]), key=float)
    pick = lr_key(d["picks"][k]) if k in d["picks"] else None
    if pick in grid:
        i = grid.index(pick)
        near = [grid[j] for j in (i - 1, i, i + 1) if 0 <= j < len(grid) and grid[j] in have]
        if near:
            return near
    return have


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--it3", nargs="*", default=[], help="agg_n11_a.json agg_n11_b.json")
    ap.add_argument("--inner", nargs="*", default=[], help="agg_n11x_inner.json (one or more)")
    ap.add_argument("--select", required=True, help="agg_n11x_select.json")
    ap.add_argument("--refine", default=None, help="agg_n11x_refine.json, if the refinement point ran")
    ap.add_argument("--out", default="graphs")
    a = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    d = load(a.it3, a.inner, a.select, a.refine)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "g1_steps_to_the_optimum.png", out / "g2_fold0.png", out / "g2_all_folds.png"]
    draw_g1(d, files[0], plt)
    head = d["head_lr"].get(0)
    _grid(d, [(0, fold_panels(d, 0))], files[1], plt,
          "Fold 0: training and validation loss per twelfth, for the fold's pick and its grid neighbours",
          [CAPTION_G2 + (f" Head learning rate {lr_text(head)}." if head else ""), CAPTION_FOLD0 + " One seed. Added after the results."])
    every = sorted((key for key, per in d["curves"].items() if per), key=float)
    _grid(d, [(k, [key for key in every if k in d["curves"][key]]) for k in FOLDS if any(k in d["curves"][key] for key in every)],
          files[2], plt, "Appendix: training and validation loss per twelfth, every fold and every rate with a curve",
          [CAPTION_G2, "An invalid run is drawn up to its last twelfth. 5e-6 has no curve to the end (dropped at rung 0). "
                       "One seed. Added after the results."])
    for f in files:
        print(f)
    return files


if __name__ == "__main__":
    main()
