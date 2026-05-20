#!/usr/bin/env python3
"""Create a paper-ready EMO-STA budget sweep bar plot at a fixed baseline."""

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

DEFAULT_MANIFEST = (
    "multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml"
)
DEFAULT_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/k_module_problem_balanced"
)
DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/"
    "k_module_problem_balanced_fixed_b30_budget_sweep"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cross-model EMO-STA averages for a fixed-baseline budget sweep."
        )
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
        default=30,
        help="Keep only settings with this baseline iteration count. Default: 30.",
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
        help=(
            "Optional figure title. Leave empty to omit the title, which is often "
            "better for paper figures."
        ),
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Include a third Shared bar per budget.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI for the PNG. Default: 300.",
    )
    parser.add_argument(
        "--legend-anchor-y",
        type=float,
        default=1.0,
        help=(
            "Optional legend y anchor in axes coordinates. Values above 1.0 "
            "move the legend upward. Default: 1.0."
        ),
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
        clean_runs = []
        for run_root in run_roots:
            run = rpt.load_run_report(
                run_root,
                repo_root=repo_root,
                manifest_path=manifest_path,
                manifest_family=family_label_from_manifest(manifest_path),
                task_specs=task_specs,
                wandb_entity_override=None,
            )
            if run["health"]["status"] == "ok":
                clean_runs.append(run)
        if not clean_runs:
            continue

        model_name = clean_runs[0]["setting"]["model"]
        shared_values = [run["macro"]["shared_best_score"] for run in clean_runs]
        adapt_values = [run["macro"]["adaptation_mean_score"] for run in clean_runs]
        baseline_values = [run["macro"]["baseline_mean_score"] for run in clean_runs]

        by_budget.setdefault(budget, {})[model_name] = {
            "n_runs": float(len(clean_runs)),
            "shared_mean": statistics.fmean(shared_values),
            "adapt_mean": statistics.fmean(adapt_values),
            "baseline_mean": statistics.fmean(baseline_values),
        }

    summaries: list[dict[str, Any]] = []
    for budget in sorted_budget_labels(list(by_budget)):
        per_model = by_budget[budget]
        model_names = [name for name in MODEL_ORDER if name in per_model]
        if not model_names:
            continue

        shared_values = [per_model[name]["shared_mean"] for name in model_names]
        adapt_values = [per_model[name]["adapt_mean"] for name in model_names]
        baseline_values = [per_model[name]["baseline_mean"] for name in model_names]

        shared_mean = statistics.fmean(shared_values)
        adapt_mean = statistics.fmean(adapt_values)
        baseline_mean = statistics.fmean(baseline_values)

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
                        **per_model[name],
                    }
                    for name in model_names
                ],
                "shared_mean": shared_mean,
                "shared_std_across_models": (
                    statistics.stdev(shared_values) if len(shared_values) > 1 else 0.0
                ),
                "adapt_mean": adapt_mean,
                "adapt_std_across_models": (
                    statistics.stdev(adapt_values) if len(adapt_values) > 1 else 0.0
                ),
                "baseline_mean": baseline_mean,
                "baseline_std_across_models": (
                    statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0.0
                ),
                "adapt_minus_baseline_mean": adapt_mean - baseline_mean,
                "model_count": len(model_names),
            }
        )

    return summaries


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def single_task_reference(summary_items: list[dict[str, Any]]) -> tuple[float, float]:
    per_model_values: dict[str, list[float]] = {}
    for item in summary_items:
        for model in item["models"]:
            per_model_values.setdefault(model["id"], []).append(model["baseline_mean"])

    collapsed = [
        statistics.fmean(values)
        for model_id, values in sorted(per_model_values.items())
        if values
    ]
    if not collapsed:
        return 0.0, 0.0
    return (
        statistics.fmean(collapsed),
        statistics.stdev(collapsed) if len(collapsed) > 1 else 0.0,
    )


def annotate_bar_values(
    ax: plt.Axes,
    xs: list[float] | np.ndarray,
    ys: list[float],
    *,
    color: str,
    y_offset: float = 0.015,
    bold_indices: set[int] | None = None,
) -> None:
    bold_indices = bold_indices or set()
    for idx, (x, y) in enumerate(zip(xs, ys)):
        is_bold = idx in bold_indices
        ax.text(
            float(x),
            float(y) + y_offset,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=11.5 if is_bold else 10,
            fontweight="black" if is_bold else "semibold",
            color="#2F3B4A" if is_bold else color,
        )


def plot_budget_bars(
    summaries: list[dict[str, Any]],
    *,
    title: str,
    include_shared: bool,
    output_stem: Path,
    dpi: int,
    legend_anchor_y: float,
) -> None:
    apply_plot_style()

    budget_x = np.arange(1, len(summaries) + 1)
    best_budget_idx = max(
        range(len(summaries)),
        key=lambda idx: summaries[idx]["adapt_mean"],
    )
    task_count = int(summaries[0]["budget"]["task_count"])
    fixed_baseline = int(summaries[0]["budget"]["baseline"])
    budget_labels = [
        (
            item["budget"]["label"]
            if item["model_count"] == 5
            else f"{item['budget']['label']}\n(n={item['model_count']})"
        )
        for item in summaries
    ]
    single_task_mean, single_task_std = single_task_reference(summaries)

    shared_color = "#C7D7F0"
    baseline_color = "#F6C8B8"
    adapt_color = "#A9D8C8"
    edge_color = "#6C7A89"

    if include_shared:
        width = 0.24
        fig, ax = plt.subplots(figsize=(10.4, 5.4))
        ax.bar(
            0,
            single_task_mean,
            width,
            color=baseline_color,
            edgecolor=edge_color,
            linewidth=0.8,
            label="Single-task",
        )
        ax.bar(
            budget_x - width / 2,
            [item["shared_mean"] for item in summaries],
            width,
            color=shared_color,
            edgecolor=edge_color,
            linewidth=0.8,
            label="Shared",
        )
        ax.bar(
            budget_x + width / 2,
            [item["adapt_mean"] for item in summaries],
            width,
            color=adapt_color,
            edgecolor=edge_color,
            linewidth=0.8,
            label="EMO-STA Adapt",
        )
    else:
        width = 0.48
        fig, ax = plt.subplots(figsize=(9.8, 5.2))
        ax.bar(
            0,
            single_task_mean,
            width,
            color=baseline_color,
            edgecolor=edge_color,
            linewidth=0.8,
            label="Single-task",
        )
        ax.bar(
            budget_x,
            [item["adapt_mean"] for item in summaries],
            width,
            color=adapt_color,
            edgecolor=edge_color,
            linewidth=0.8,
            label="EMO-STA Adapt",
        )
    ax.set_xlabel(
        budget_axis_label(),
        fontweight="bold",
    )
    ax.set_ylabel("Mean Score Across Models", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(np.concatenate(([0], budget_x)))
    ax.set_xticklabels(
        [
            build_single_task_budget_triplet(
                baseline=fixed_baseline,
                task_count=task_count,
            )["label"],
            *budget_labels,
        ]
    )
    ax.set_ylim(0.0, 1.03)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        ncol=1 if include_shared else 2,
        bbox_to_anchor=(0.0, legend_anchor_y),
        borderaxespad=0.0,
    )

    annotate_bar_values(ax, [0], [single_task_mean], color=edge_color)
    if include_shared:
        annotate_bar_values(
            ax,
            budget_x - width / 2,
            [item["shared_mean"] for item in summaries],
            color=edge_color,
        )
        annotate_bar_values(
            ax,
            budget_x + width / 2,
            [item["adapt_mean"] for item in summaries],
            color=edge_color,
            bold_indices={best_budget_idx},
        )
    else:
        annotate_bar_values(
            ax,
            budget_x,
            [item["adapt_mean"] for item in summaries],
            color=edge_color,
            bold_indices={best_budget_idx},
        )

    plt.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def write_sidecar_json(output_stem: Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "manifest": args.manifest,
        "results_dir": args.results_dir,
        "fixed_baseline": args.fixed_baseline,
        "include_shared": args.include_shared,
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
            f"No clean settings found in {results_dir} for baseline={args.fixed_baseline}."
        )

    plot_budget_bars(
        summaries,
        title=args.title,
        include_shared=args.include_shared,
        output_stem=output_stem,
        dpi=args.dpi,
        legend_anchor_y=args.legend_anchor_y,
    )
    write_sidecar_json(output_stem, summaries, args)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")
    for item in summaries:
        print(
            f"{item['budget']['label']}: adapt={item['adapt_mean']:.3f}, "
            f"single={item['baseline_mean']:.3f}, delta={item['adapt_minus_baseline_mean']:+.3f}, "
            f"models={item['model_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
