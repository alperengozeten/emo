"""Single-task vs joint evolution on covid_deaths — y-axis variant.

Same data, bars, hatching, and legend as plot_single_vs_joint.py, but the
metric labels MASE and wQL are placed on the LEFT y-axis (rotated 90°)
instead of being attached to the bars on the right with curly braces.

Numbers are hard-coded from:
  - covid_deaths_individual/val_test_improvements.md  (single-task)
  - examples/.../output_covid_deaths_joint_bigger_pop/evaluation.json
    (joint, final.pct_change_vs_baseline, signs flipped)

Usage:
    python plot_single_vs_joint_yaxis.py \\
        --out-stem evaluations/_single_vs_joint_bars_yaxis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METRICS = ("MASE", "wQL")                   # rows, top-to-bottom

# % improvement over identity baseline (positive = better).
PCT = {
    "val":  {"Per-series": [12.32, 12.99],
             "Joint":      [10.01, 24.34]},
    "test": {"Per-series": [-1.12,  2.71],
             "Joint":      [13.24, 32.56]},
}

COLORS = {
    "Per-series": "#f4a896",  # coral
    "Joint":      "#a8d8c9",  # mint
}
EDGE = "#1f1f1f"
HATCH = "///"            # applied to validation bars
LABEL_COLOR = "#222222"  # match x-axis tick / label color


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-stem", required=True, type=str)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 19,
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.4,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "xtick.color": "#222222",
        "ytick.color": LABEL_COLOR,
        "xtick.labelsize": 14,
        "ytick.labelsize": 22,
    })

    n_rows = len(METRICS)
    bar_h = 0.18
    # Four bars per row at offsets (top-to-bottom):
    #   val Per-series, val Joint, test Per-series, test Joint
    bar_specs = [
        ("val",  "Per-series", +1.5 * bar_h, HATCH),
        ("val",  "Joint",      +0.5 * bar_h, HATCH),
        ("test", "Per-series", -0.5 * bar_h, None),
        ("test", "Joint",      -1.5 * bar_h, None),
    ]
    y_centers = np.arange(n_rows)[::-1].astype(float) * 0.95  # MASE on top

    fig, ax = plt.subplots(figsize=(9.0, 3.8))

    all_vals = np.concatenate([
        np.asarray(PCT[split][method])
        for split, method, _, _ in bar_specs
    ])
    span = float(all_vals.max() - all_vals.min())
    label_pad = 0.018 * max(span, 1.0)

    for split, method, offset, hatch in bar_specs:
        vals = np.asarray(PCT[split][method], dtype=float)
        ys = y_centers + offset
        ax.barh(
            ys, vals, bar_h * 0.92,
            color=COLORS[method], edgecolor=EDGE, linewidth=1.2,
            hatch=hatch, zorder=3,
        )
        for v, yi in zip(vals, ys):
            x_pos = v + (label_pad if v >= 0 else -label_pad)
            ha = "left" if v >= 0 else "right"
            ax.text(
                x_pos, yi, f"{v:+.2f}%",
                va="center", ha=ha,
                fontsize=12, fontweight="bold", color="#1f1f1f",
                zorder=5,
            )

    # Zero line: highlights the test-MASE regression (-1.12%).
    ax.axvline(0.0, color="#666666", linewidth=1.0, zorder=2)

    x_min = min(0.0, float(all_vals.min())) - 4.5
    x_max = float(all_vals.max()) + 5.0
    ax.set_xlim(x_min, x_max)

    # y-axis: MASE / wQL as tick labels, rotated 90° (read bottom-to-top).
    ax.set_yticks(y_centers)
    ax.set_yticklabels(
        METRICS, rotation=90, va="center", ha="center",
        fontsize=22, fontweight="bold", color=LABEL_COLOR,
    )
    ax.tick_params(axis="y", length=0, pad=14)

    ax.set_xlabel("Improvement over baseline (%)")
    ax.grid(True, axis="x", linestyle=":", color="#bbbbbb",
            linewidth=0.9, alpha=0.8)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v:+.0f}%")
    )

    # Legend: color encodes method; hatch encodes split.
    method_handles = [
        Patch(facecolor=COLORS["Per-series"], edgecolor=EDGE,
              label="Single-task"),
        Patch(facecolor=COLORS["Joint"],      edgecolor=EDGE,
              label="STA Best-Shared (No Adaptation)"),
    ]
    split_handles = [
        Patch(facecolor="white", edgecolor=EDGE, hatch=HATCH, label="Val"),
        Patch(facecolor="white", edgecolor=EDGE,                label="Test"),
    ]
    leg = fig.legend(
        handles=method_handles + split_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.14),
        ncol=2, fontsize=14,
        prop={"weight": "bold", "size": 14},
        handlelength=1.6, handleheight=1.1,
        borderpad=0.5, columnspacing=2.4, labelspacing=0.6,
    )
    leg.set_zorder(7)

    fig.tight_layout()

    stem = Path(args.out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.png')}")
    print(f"wrote {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
