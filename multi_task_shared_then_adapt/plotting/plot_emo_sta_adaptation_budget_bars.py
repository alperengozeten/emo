#!/usr/bin/env python3
"""Create a budget sweep bar plot for EMO-STA adaptation branches."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

PLOT_DIR = Path(__file__).resolve().parent
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plot_budget_labels import (
    build_budget_triplet,
    build_single_task_budget_triplet,
    budget_axis_label,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_task_shared_then_adapt.scripts import report_emo_sta_results as rpt

DEFAULT_MANIFEST = "multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml"
DEFAULT_RESULTS_DIR = "multi_task_shared_then_adapt/results/circle_packing"
DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/"
    "circle_packing_fixed_b30_adaptation_budget_sweep"
)

SETTING_RE = re.compile(r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-")

MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
]

MODEL_DISPLAY = {
    "claude-haiku-4-5": "Haiku-4.5",
    "claude-sonnet-4-5": "Sonnet-4.5",
    "claude-sonnet-4-6": "Sonnet-4.6",
    "claude-opus-4-5": "Opus-4.5",
    "claude-opus-4-6": "Opus-4.6",
}

METHOD_SPECS = [
    ("baseline_mean", "Single-task", "#F6C8B8", ""),
    ("adapt_mean", "Warmstart", "#A9D8C8", "/"),
    ("best_local_mean", "Best-Local", "#A9D8C8", ""),
    ("best_shared_mean", "Best-Shared", "#A9D8C8", "x"),
]
SINGLE_TASK_SPEC = METHOD_SPECS[0]
BUDGET_METHOD_SPECS = METHOD_SPECS[1:]
EDGE_COLOR = "#000000"

EMO_STA_TABLE_SELECTED_BUDGETS = {
    "function_minimization": (40, 15, 25),
    "circle_packing": (60, 15, 30),
    "circle_packing_rectangle": (60, 15, 30),
    "heilbronn_triangle": (60, 15, 30),
    "k_module_problem_balanced": (40, 20, 30),
    "signal_processing": (60, 10, 25),
    "sldbench_3d": (60, 10, 40),
    "rust_adaptive_sort": (60, 10, 25),
}

Y_LIMITS_BY_FAMILY = {
    "circle_packing": (0.80, 1.00),
    "circle_packing_rectangle": (0.80, 1.00),
    "heilbronn_triangle": (0.50, 0.80),
    "function_minimization": (0.80, 1.00),
}

Y_AXIS_LABEL_BY_FAMILY = {
    "function_minimization": "Mean Score Across LLMs",
    "circle_packing": "Mean Normalized Score Across LLMs",
    "circle_packing_rectangle": "Mean Normalized Score Across LLMs",
    "heilbronn_triangle": "Mean Normalized Score Across LLMs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot budget-sweep adaptation bars for one EMO-STA family."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"EMO-STA manifest path. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--fixed-baseline",
        type=int,
        required=True,
        help="Keep only settings with this baseline iteration count.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=(
            "Output path without extension. The script writes .png, .pdf, .svg, and .json. "
            f"Default: {DEFAULT_OUTPUT_STEM}"
        ),
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title. Leave empty to omit the title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI for the PNG. Default: 300.",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Do not draw the Single-task or STA Adaptation Variants legends.",
    )
    return parser.parse_args()


def parse_budget_from_setting_dir(path: Path) -> tuple[int, int, int] | None:
    match = SETTING_RE.match(path.name)
    if not match:
        return None
    return (
        int(match.group("shared")),
        int(match.group("adapt")),
        int(match.group("baseline")),
    )


def family_label_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem
    if stem.endswith("_emo_sta"):
        stem = stem[: -len("_emo_sta")]
    return stem.replace("_", " ")


def family_id_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem
    if stem.endswith("_emo_sta"):
        stem = stem[: -len("_emo_sta")]
    return stem


def sorted_budget_labels(budgets: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return sorted(budgets, key=lambda item: (item[0], item[1], item[2]))


def aggregate_budget_data(
    *,
    repo_root: Path,
    manifest_path: Path,
    results_dir: Path,
    fixed_baseline: int,
) -> list[dict[str, Any]]:
    manifest = rpt.load_manifest(manifest_path)
    task_specs = rpt.family_task_specs(manifest)
    task_count = len(task_specs)
    family_id = family_id_from_manifest(manifest_path)
    canonical_budget = EMO_STA_TABLE_SELECTED_BUDGETS.get(family_id)
    use_canonical_single_task = (
        canonical_budget is not None and canonical_budget[2] == fixed_baseline
    )

    by_budget: dict[tuple[int, int, int], dict[str, dict[str, float]]] = {}

    for setting_dir in sorted(results_dir.iterdir()):
        if not setting_dir.is_dir():
            continue
        budget = parse_budget_from_setting_dir(setting_dir)
        if budget is None or budget[2] != fixed_baseline:
            continue

        run_roots = rpt.discover_run_roots(
            repo_root=repo_root,
            results_dir=setting_dir,
            explicit_run_roots=None,
        )

        comparable_runs = []
        for run_root in run_roots:
            run = rpt.load_run_report(
                run_root,
                repo_root=repo_root,
                manifest_path=manifest_path,
                manifest_family=family_label_from_manifest(manifest_path),
                task_specs=task_specs,
                wandb_entity_override=None,
            )
            if run["health"]["status"] != "ok":
                continue
            macro = run.get("macro", {})
            if (
                macro.get("baseline_mean_score") is None
                or macro.get("adaptation_mean_score") is None
                or macro.get("best_local_mean_score") is None
                or macro.get("best_shared_mean_score") is None
            ):
                continue
            comparable_runs.append(run)

        if not comparable_runs:
            continue

        model_name = comparable_runs[0]["setting"]["model"]
        adapt_values = [run["macro"]["adaptation_mean_score"] for run in comparable_runs]
        best_local_values = [
            run["macro"]["best_local_mean_score"] for run in comparable_runs
        ]
        best_shared_values = [
            run["macro"]["best_shared_mean_score"] for run in comparable_runs
        ]

        by_budget.setdefault(budget, {})[model_name] = {
            "n_runs": float(len(comparable_runs)),
            "setting_baseline_mean": statistics.fmean(
                [run["macro"]["baseline_mean_score"] for run in comparable_runs]
            ),
            "adapt_mean": statistics.fmean(adapt_values),
            "best_local_mean": statistics.fmean(best_local_values),
            "best_shared_mean": statistics.fmean(best_shared_values),
        }

    canonical_single_task_by_model: dict[str, float] = {}
    if use_canonical_single_task and canonical_budget in by_budget:
        canonical_single_task_by_model = {
            model_name: values["setting_baseline_mean"]
            for model_name, values in by_budget[canonical_budget].items()
        }

    summaries: list[dict[str, Any]] = []
    for budget in sorted_budget_labels(list(by_budget)):
        per_model = by_budget[budget]
        model_names = [name for name in MODEL_ORDER if name in per_model]
        if not model_names:
            continue

        adapt_values = [per_model[name]["adapt_mean"] for name in model_names]
        best_local_values = [
            per_model[name]["best_local_mean"] for name in model_names
        ]
        best_shared_values = [
            per_model[name]["best_shared_mean"] for name in model_names
        ]
        baseline_values = [
            canonical_single_task_by_model.get(
                name,
                per_model[name]["setting_baseline_mean"],
            )
            for name in model_names
        ]

        summaries.append(
            {
                "budget": {
                    **build_budget_triplet(
                        shared=budget[0],
                        adapt=budget[1],
                        baseline=budget[2],
                        task_count=task_count,
                    ),
                },
                "models": [
                    {
                        "id": name,
                        "label": MODEL_DISPLAY.get(name, name),
                        "baseline_mean": canonical_single_task_by_model.get(
                            name,
                            per_model[name]["setting_baseline_mean"],
                        ),
                        "baseline_source_budget": (
                            build_budget_triplet(
                                shared=canonical_budget[0],
                                adapt=canonical_budget[1],
                                baseline=canonical_budget[2],
                                task_count=task_count,
                            )
                            if use_canonical_single_task and name in canonical_single_task_by_model
                            else build_budget_triplet(
                                shared=budget[0],
                                adapt=budget[1],
                                baseline=budget[2],
                                task_count=task_count,
                            )
                        ),
                        **per_model[name],
                    }
                    for name in model_names
                ],
                "baseline_mean": statistics.fmean(baseline_values),
                "baseline_std_across_models": (
                    statistics.stdev(baseline_values)
                    if len(baseline_values) > 1
                    else 0.0
                ),
                "adapt_mean": statistics.fmean(adapt_values),
                "adapt_std_across_models": (
                    statistics.stdev(adapt_values) if len(adapt_values) > 1 else 0.0
                ),
                "best_local_mean": statistics.fmean(best_local_values),
                "best_local_std_across_models": (
                    statistics.stdev(best_local_values)
                    if len(best_local_values) > 1
                    else 0.0
                ),
                "best_shared_mean": statistics.fmean(best_shared_values),
                "best_shared_std_across_models": (
                    statistics.stdev(best_shared_values)
                    if len(best_shared_values) > 1
                    else 0.0
                ),
                "model_count": len(model_names),
            }
        )

    return summaries


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14.5,
            "legend.title_fontsize": 14.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "hatch.linewidth": 0.8,
        }
    )


def bold_legend(legend: Any) -> None:
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_title().set_fontweight("bold")


def bold_axis_tick_labels(ax: plt.Axes) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def align_single_task_legend_to_method_row(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    single_task_legend: Any,
    method_legend: Any,
    single_task_anchor: tuple[float, float],
) -> None:
    """Vertically align Single-task with the method labels below the STA title."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    method_texts = method_legend.get_texts()
    single_texts = single_task_legend.get_texts()
    if not method_texts or not single_texts:
        return

    def text_center_y(text: Any) -> float:
        bbox = text.get_window_extent(renderer=renderer)
        return 0.5 * (bbox.y0 + bbox.y1)

    target_y = statistics.mean(text_center_y(text) for text in method_texts)
    current_y = text_center_y(single_texts[0])
    axes_y0 = ax.transAxes.transform((0.0, 0.0))[1]
    axes_y1 = ax.transAxes.transform((0.0, 1.0))[1]
    if axes_y1 == axes_y0:
        return

    delta_axes = (target_y - current_y) / (axes_y1 - axes_y0)
    single_task_legend.set_bbox_to_anchor(
        (single_task_anchor[0], single_task_anchor[1] + delta_axes),
        transform=ax.transAxes,
    )


def annotate_bar_values(
    ax: plt.Axes,
    bars: list[Any],
    values: list[float],
    *,
    bold_index: int | None = None,
) -> None:
    y_min, y_max = ax.get_ylim()
    value_offset = max(0.0025, 0.012 * (y_max - y_min))
    for idx, (bar, value) in enumerate(zip(bars, values)):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        is_bold = idx == bold_index
        ax.text(
            x,
            y + value_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=14 if is_bold else 13,
            fontweight="black" if is_bold else "semibold",
            color="#1F2933",
        )


def plot_budget_bars(
    summaries: list[dict[str, Any]],
    *,
    family_id: str,
    title: str,
    output_stem: Path,
    dpi: int,
    fixed_baseline: int,
    hide_legend: bool,
) -> None:
    apply_plot_style()

    anchor_gap = 0.62
    budget_gap = 0.82
    budget_x = np.concatenate(
        (
            np.array([0.0]),
            anchor_gap + np.arange(len(summaries), dtype=float) * budget_gap,
        )
    )
    width = 0.22
    cluster_center_spacing = 1.00 * width
    offsets = np.array([-cluster_center_spacing, 0.0, cluster_center_spacing])

    fig_width = max(7.5, 2.2 * len(summaries) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    task_count = int(summaries[0]["budget"]["task_count"])
    budget_labels = [
        build_single_task_budget_triplet(
            baseline=fixed_baseline,
            task_count=task_count,
        )["label"]
    ] + [
        (
            item["budget"]["label"]
            if item["model_count"] == 5
            else f"{item['budget']['label']}\n(n={item['model_count']})"
        )
        for item in summaries
    ]

    single_value = summaries[0]["baseline_mean"]
    y_limits = Y_LIMITS_BY_FAMILY.get(family_id)
    if y_limits is None:
        plotted_values = [single_value]
        for item in summaries:
            plotted_values.extend(
                [
                    item["adapt_mean"],
                    item["best_local_mean"],
                    item["best_shared_mean"],
                ]
            )
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        lower_margin = max(0.01, 0.15 * (y_max - y_min))
        upper_margin = max(0.01, 0.20 * (y_max - y_min))
        y_limits = (max(0.0, y_min - lower_margin), min(1.06, y_max + upper_margin))
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    single_bar = ax.bar(
        budget_x[0],
        single_value,
        width,
        color=SINGLE_TASK_SPEC[2],
        edgecolor=EDGE_COLOR,
        linewidth=1.15,
        hatch=SINGLE_TASK_SPEC[3],
    )[0]
    annotate_bar_values(ax, [single_bar], [single_value], bold_index=None)

    for budget_idx, item in enumerate(summaries):
        cluster_values = [
            item[field] for field, _label, _color, _hatch in BUDGET_METHOD_SPECS
        ]
        best_method_idx = max(range(len(cluster_values)), key=cluster_values.__getitem__)
        bars = []
        for method_idx, (field, _label, color, hatch) in enumerate(BUDGET_METHOD_SPECS):
            bar = ax.bar(
                budget_x[budget_idx + 1] + offsets[method_idx],
                item[field],
                width,
                color=color,
                edgecolor=EDGE_COLOR,
                linewidth=1.15,
                hatch=hatch,
            )[0]
            bars.append(bar)
        annotate_bar_values(ax, bars, cluster_values, bold_index=best_method_idx)

    ax.set_xlabel(
        budget_axis_label(),
        fontweight="bold",
    )
    ax.set_ylabel(
        Y_AXIS_LABEL_BY_FAMILY.get(family_id, "Mean Score Across Models"),
        fontweight="bold",
    )
    if title:
        ax.set_title(title, fontweight="bold", pad=16)
    ax.set_xticks(budget_x)
    ax.set_xticklabels(budget_labels)
    bold_axis_tick_labels(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    extra_artists: tuple[Any, ...] = ()
    if not hide_legend:
        single_task_handle = Patch(
            facecolor=SINGLE_TASK_SPEC[2],
            edgecolor=EDGE_COLOR,
            linewidth=0.9,
            hatch=SINGLE_TASK_SPEC[3],
            label=SINGLE_TASK_SPEC[1],
        )
        sts_method_handles = [
            Patch(
                facecolor=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.9,
                hatch=hatch,
                label=label,
            )
            for _field, label, color, hatch in BUDGET_METHOD_SPECS
        ]

        sts_legend = ax.legend(
            handles=sts_method_handles,
            title="STA Adaptation Variants",
            loc="upper center",
            bbox_to_anchor=(0.60, 1.045),
            ncol=3,
            borderaxespad=0.0,
            borderpad=1.0,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="#C7CDD4",
            facecolor="white",
        )
        bold_legend(sts_legend)
        ax.add_artist(sts_legend)
        single_task_anchor = (0.20, 0.978)
        single_task_legend = ax.legend(
            handles=[single_task_handle],
            loc="upper center",
            bbox_to_anchor=single_task_anchor,
            ncol=1,
            borderaxespad=0.0,
            frameon=False,
        )
        bold_legend(single_task_legend)
        extra_artists = (sts_legend, single_task_legend)

    layout_rect = (0.0, 0.0, 1.0, 0.995) if hide_legend else (0.0, 0.0, 1.0, 0.952)
    plt.tight_layout(rect=layout_rect)
    if not hide_legend:
        align_single_task_legend_to_method_row(
            fig,
            ax,
            single_task_legend=single_task_legend,
            method_legend=sts_legend,
            single_task_anchor=single_task_anchor,
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.10,
            "bbox_extra_artists": extra_artists,
        }
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def write_sidecar_json(
    output_stem: Path, summaries: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    manifest_path = Path(args.manifest)
    family_id = family_id_from_manifest(manifest_path)
    canonical_budget = EMO_STA_TABLE_SELECTED_BUDGETS.get(family_id)
    task_count = (
        int(summaries[0]["budget"]["task_count"])
        if summaries
        else len(rpt.family_task_specs(rpt.load_manifest(rpt.resolve_repo_path(REPO_ROOT, args.manifest))))
    )
    payload = {
        "manifest": args.manifest,
        "results_dir": args.results_dir,
        "fixed_baseline": args.fixed_baseline,
        "single_task_source_mode": (
            "emo_sta_table_selected_budget"
            if canonical_budget is not None and canonical_budget[2] == args.fixed_baseline
            else "per_budget"
        ),
        "single_task_source_budget": (
            build_budget_triplet(
                shared=canonical_budget[0],
                adapt=canonical_budget[1],
                baseline=canonical_budget[2],
                task_count=task_count,
            )
            if canonical_budget is not None and canonical_budget[2] == args.fixed_baseline
            else None
        ),
        "y_limits": Y_LIMITS_BY_FAMILY.get(family_id),
        "hide_legend": args.hide_legend,
        "methods": [
            {"field": field, "label": label, "color": color, "hatch": hatch}
            for field, label, color, hatch in METHOD_SPECS
        ],
        "edge_color": EDGE_COLOR,
        "budgets": summaries,
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    repo_root = REPO_ROOT
    manifest_path = rpt.resolve_repo_path(repo_root, args.manifest)
    results_dir = rpt.resolve_repo_path(repo_root, args.results_dir)
    output_stem = rpt.resolve_repo_path(repo_root, args.output_stem)

    summaries = aggregate_budget_data(
        repo_root=repo_root,
        manifest_path=manifest_path,
        results_dir=results_dir,
        fixed_baseline=args.fixed_baseline,
    )
    if not summaries:
        raise SystemExit(
            f"No comparable settings found in {results_dir} for baseline={args.fixed_baseline}."
        )

    plot_budget_bars(
        summaries,
        family_id=family_id_from_manifest(manifest_path),
        title=args.title,
        output_stem=output_stem,
        dpi=args.dpi,
        fixed_baseline=args.fixed_baseline,
        hide_legend=args.hide_legend,
    )
    write_sidecar_json(output_stem, summaries, args)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")
    for item in summaries:
        print(
            f"{item['budget']['label']}: "
            f"single={item['baseline_mean']:.3f}, "
            f"warmstart={item['adapt_mean']:.3f}, "
            f"best-local={item['best_local_mean']:.3f}, "
            f"best-shared={item['best_shared_mean']:.3f}, "
            f"models={item['model_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
