"""Figures and the README results table.

Usage:
    python scripts/make_figures.py --run runs/i3d_transformer --baselines results/baselines.json
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tad.data import load_annotations  # noqa: E402

NEW = "#2a78d6"      # the new detector: the one accent colour
OLD = "#9a9994"      # old baselines: recessive grey
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})
TIOUS = ["0.3", "0.4", "0.5", "0.6", "0.7"]


def fig_map_vs_tiou(new, baselines, path, new_name, others=()):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = [float(t) for t in TIOUS]
    label_y, prev = {}, -1e9
    for name in sorted(baselines, key=lambda k: baselines[k]["mAP"][TIOUS[0]]):  # spread labels apart
        prev = max(baselines[name]["mAP"][TIOUS[0]], prev + 4.5)
        label_y[name] = prev
    for name, res in baselines.items():
        y = [res["mAP"][t] for t in TIOUS]
        ax.plot(x, y, color=OLD, lw=1.5, marker="o", ms=4)
        ax.annotate(name + " (v1)", (x[0], y[0]), xytext=(x[0] - 0.015, label_y[name]), ha="right", va="center",
                    color=INK2, fontsize=9)
    for name, res in others:  # other detector variants: same hue, dashed, no value labels
        yo = [res["mAP"][t] for t in TIOUS]
        ax.plot(x, yo, color=NEW, lw=1.5, ls="--", alpha=0.6, marker="o", ms=4)
        ax.annotate(name, (x[-1], yo[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    y = [new["mAP"][t] for t in TIOUS]
    ax.plot(x, y, color=NEW, lw=2.5, marker="o", ms=6)
    ax.annotate(new_name, (x[0], y[0]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                color=INK, fontsize=10, fontweight="bold")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xlim(0.1, 0.75 if not others else 0.95)
    ax.set_xticks(x)
    ax.set_ylim(0, max(100, max(y) + 10))
    ax.set_xlabel("temporal IoU threshold")
    ax.set_ylabel("mAP (%)")
    ax.set_title("Same I3D features, detection framing: mAP on THUMOS-14 test", loc="left", color=INK, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def fig_per_class(new, best_old, best_old_name, path, new_name):
    classes = sorted(new["per_class_AP"], key=lambda c: new["per_class_AP"][c][2])
    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7, 7))
    h = 0.38
    ax.barh(y + h / 2, [new["per_class_AP"][c][2] for c in classes], h, color=NEW, label=new_name)
    if best_old:
        ax.barh(y - h / 2, [best_old["per_class_AP"][c][2] for c in classes], h, color=OLD, label=best_old_name)
    ax.set_yticks(y, classes)
    ax.set_xlabel("AP at tIoU 0.5 (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Per-class AP@0.5, THUMOS-14 test", loc="left", color=INK, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def fig_timeline(preds, ann, classes, video, path, score_thresh=0.3):
    gt = ann[video]
    shown = sorted({c for _, _, c in gt})
    pv = [p for p in preds if p["video"] == video and p["score"] >= score_thresh and p["label"] in shown]
    end = max([e for _, e, _ in gt] + [p["end"] for p in pv]) * 1.03
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.8 * len(shown)))
    for row, c in enumerate(shown):
        for s, e, cc in gt:
            if cc == c:
                ax.broken_barh([(s, e - s)], (row + 0.05, 0.38), color=INK2)
        for p in pv:
            if p["label"] == c:
                ax.broken_barh([(p["start"], p["end"] - p["start"])], (row - 0.43, 0.38), color=NEW,
                               alpha=0.35 + 0.65 * p["score"])
    ax.set_yticks(range(len(shown)), [classes[c] for c in shown])
    ax.set_xlim(0, end)
    ax.set_ylim(-0.6, len(shown) - 0.4)
    ax.set_xlabel("time (s)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{video}: ground truth (grey, top) vs predictions (blue, bottom; opacity = score)",
                 loc="left", color=INK, fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/i3d_transformer"], help="first run = headline model")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--videos", nargs="*", help="videos for timeline plots (default: 3 examples)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = args.names or [os.path.basename(r) for r in args.runs]
    runs = [json.load(open(os.path.join(r, "results.json"))) for r in args.runs]
    new, name = runs[0], names[0]
    others = list(zip(names[1:], runs[1:]))
    preds = json.load(open(os.path.join(args.runs[0], "test_predictions.json")))
    baselines = json.load(open(args.baselines)) if os.path.exists(args.baselines) else {}
    cfg = new["config"]
    classes, ann, _ = load_annotations(cfg["data"]["ann_dirs"], cfg["data"].get("json_file"))

    fig_map_vs_tiou(new, baselines, os.path.join(args.out, "map_vs_tiou.png"), name, others)
    best_name = max(baselines, key=lambda k: baselines[k]["avg_mAP"]) if baselines else None
    fig_per_class(new, baselines.get(best_name), f"{best_name} (old)", os.path.join(args.out, "per_class_ap.png"), name)
    videos = args.videos or sorted({p["video"] for p in preds if p["video"] in ann},
                                   key=lambda v: -len(ann[v]))[3:6]
    for i, v in enumerate(videos):
        fig_timeline(preds, ann, classes, v, os.path.join(args.out, f"timeline_{v}.png"))
        if i == 0:  # the one shown in the README
            fig_timeline(preds, ann, classes, v, os.path.join(args.out, "timeline_example.png"))

    # README table
    rows = [(k + " (v1)", r) for k, r in baselines.items()] + list(zip(names, runs))
    print("| Model | " + " | ".join(f"mAP@{t}" for t in TIOUS) + " | Avg |")
    print("|---|" + "---|" * (len(TIOUS) + 1))
    for name, r in rows:
        bold = "**" if r is new else ""
        print(f"| {bold}{name}{bold} | " + " | ".join(f"{bold}{r['mAP'][t]:.1f}{bold}" for t in TIOUS)
              + f" | {bold}{r['avg_mAP']:.1f}{bold} |")
    print(f"\nfigures written to {args.out}/")


if __name__ == "__main__":
    main()
