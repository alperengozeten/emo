"""Overlap-aware ARC overfitting and STA recovery plot.

Each model has 20 failed single-task ARC runs. The overfit row shows how many
failed runs overfit the training examples. Method rows are stacked by whether
the recovered cases came from that overfitting subset or from other failures.

Usage:
    python plot_arc_overfitting_overlap_methods.py
    python plot_arc_overfitting_overlap_methods.py --out-stem figures/arc_overlap_methods
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


N_FAILED = 20


@dataclass(frozen=True)
class MethodResult:
    label: str
    solved_overfit: int
    solved_non_overfit: int = 0

    @property
    def total_solved(self) -> int:
        return self.solved_overfit + self.solved_non_overfit


@dataclass(frozen=True)
class ModelResult:
    label: str
    overfit: int
    methods: tuple[MethodResult, ...]


RESULTS = (
    ModelResult(
        label="Gemini 3.1 Pro",
        overfit=19,
        methods=(
            MethodResult("STA Best-Shared", solved_overfit=13),
            MethodResult("STA Warmstart", solved_overfit=13),
            MethodResult("STA Best-Local", solved_overfit=12),
        ),
    ),
    ModelResult(
        label="Opus 4.6",
        overfit=12,
        methods=(
            MethodResult("STA Best-Shared", solved_overfit=8),
            MethodResult("STA Warmstart", solved_overfit=7),
            MethodResult("STA Best-Local", solved_overfit=7, solved_non_overfit=1),
        ),
    ),
)

OVERFIT_COLOR = "#F6C8B8"
SOLVED_OVERFIT_COLOR = "#A9D8C8"
SOLVED_NON_OVERFIT_COLOR = "#C7D7F0"
REMAINING_FACE = "#f8f8f8"
REMAINING_EDGE = "#c9c9c9"
EDGE = "#222222"
GRID = "#d0d0d0"
TEXT = "#222222"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": True,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 1.2,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rate_label(count: int) -> str:
    return f"{count}/{N_FAILED} ({100.0 * count / N_FAILED:.0f}%)"


def annotate_bar_end(ax: plt.Axes, *, x: float, y: float, text: str) -> None:
    ax.text(
        x + 0.35,
        y,
        text,
        ha="left",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=TEXT,
        zorder=5,
    )


def draw_background_bar(ax: plt.Axes, *, y: float, height: float) -> None:
    ax.barh(
        y,
        N_FAILED,
        height=height,
        color=REMAINING_FACE,
        edgecolor=REMAINING_EDGE,
        linewidth=0.8,
        zorder=1,
    )


def make_figure() -> plt.Figure:
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    row_h = 0.31
    row_step = 0.44
    group_gap = 0.45
    header_offset = 0.32
    rows: list[tuple[float, str]] = []
    current_y = 4.85

    for model_idx, model in enumerate(RESULTS):
        header_y = current_y + header_offset

        ax.text(
            0.0,
            header_y,
            model.label,
            ha="left",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color=TEXT,
        )

        overfit_y = current_y
        rows.append((overfit_y, "Overfit failures"))
        draw_background_bar(ax, y=overfit_y, height=row_h)
        ax.barh(
            overfit_y,
            model.overfit,
            height=row_h,
            color=OVERFIT_COLOR,
            edgecolor=EDGE,
            linewidth=1.1,
            zorder=3,
        )
        annotate_bar_end(ax, x=model.overfit, y=overfit_y, text=rate_label(model.overfit))

        current_y -= row_step
        for method in model.methods:
            rows.append((current_y, method.label))
            draw_background_bar(ax, y=current_y, height=row_h)
            ax.barh(
                current_y,
                method.solved_overfit,
                height=row_h,
                color=SOLVED_OVERFIT_COLOR,
                edgecolor=EDGE,
                linewidth=1.1,
                zorder=3,
            )
            if method.solved_non_overfit:
                ax.barh(
                    current_y,
                    method.solved_non_overfit,
                    left=method.solved_overfit,
                    height=row_h,
                    color=SOLVED_NON_OVERFIT_COLOR,
                    edgecolor=EDGE,
                    linewidth=1.1,
                    zorder=4,
                )
                split = f"{method.solved_overfit}+{method.solved_non_overfit}"
                label = f"{rate_label(method.total_solved)} ({split})"
            else:
                label = rate_label(method.total_solved)
            annotate_bar_end(ax, x=method.total_solved, y=current_y, text=label)
            current_y -= row_step

        if model_idx != len(RESULTS) - 1:
            ax.axhline(current_y + 0.12, color="#eeeeee", linewidth=1.1, zorder=0)
            current_y -= group_gap

    y_ticks, y_labels = zip(*rows)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontweight="bold")

    ax.set_xlim(0, 22.4)
    ax.set_ylim(current_y + 0.05, 5.38)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{int(value)}")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlabel("Failed single-task cases")
    ax.grid(True, axis="x", linestyle=":", color=GRID, linewidth=1.0, alpha=0.95)

    handles = [
        Patch(facecolor=OVERFIT_COLOR, edgecolor=EDGE, label="Overfit failures"),
        Patch(facecolor=SOLVED_OVERFIT_COLOR, edgecolor=EDGE, label="Solved overfit failures"),
        Patch(
            facecolor=SOLVED_NON_OVERFIT_COLOR,
            edgecolor=EDGE,
            label="Solved non-overfit failures",
        ),
        Patch(facecolor=REMAINING_FACE, edgecolor=REMAINING_EDGE, label="Remaining failures"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.12),
        ncol=4,
        prop={"weight": "bold", "size": 8.8},
        handlelength=1.05,
        handletextpad=0.42,
        columnspacing=0.95,
        labelspacing=0.25,
    )

    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.17, top=0.80)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-stem",
        default="overfitting_overlap_methods",
        type=str,
        help="Output path without extension. Default: overfitting_overlap_methods.",
    )
    parser.add_argument("--dpi", default=300, type=int, help="PNG DPI. Default: 300.")
    args = parser.parse_args()

    stem = Path(args.out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
