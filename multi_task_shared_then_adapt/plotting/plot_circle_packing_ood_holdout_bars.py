#!/usr/bin/env python3
"""Create a paper-ready OOD holdout bar plot for circle packing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_DIR = "multi_task_shared_then_adapt/results/circle_packing"
DEFAULT_SETTING_PREFIX = "s60-a15-b30"
DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/"
    "circle_packing_s60_a15_b30_ood_holdout_eval"
)

MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot circle-packing holdout performance for a selected EMO-STA budget."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Circle-packing results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--setting-prefix",
        default=DEFAULT_SETTING_PREFIX,
        help=(
            "Budget prefix to aggregate, e.g. s60-a15-b30. "
            f"Default: {DEFAULT_SETTING_PREFIX}"
        ),
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
        help="Optional title. Leave empty for paper figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI for PNG output. Default: 300.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


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


def mean_or_raise(values: list[float], *, context: str) -> float:
    if not values:
        raise ValueError(f"No values available for {context}")
    return statistics.fmean(values)


def collect_model_level_means(
    *,
    results_dir: Path,
    setting_prefix: str,
) -> dict[str, Any]:
    model_results: dict[str, dict[str, Any]] = {}
    task_count: int | None = None

    for setting_dir in sorted(results_dir.glob(f"{setting_prefix}-*")):
        if not setting_dir.is_dir():
            continue
        model_id = "-".join(setting_dir.name.split("-")[3:-1])
        run_paths = sorted(setting_dir.glob("run_*_seed_*/comparison_summary.json"))
        if not run_paths:
            continue

        run_shared: list[float] = []
        run_adapt: list[float] = []
        run_baseline: list[float] = []
        holdout_acc = {
            holdout_id: {"shared": [], "adapt": [], "baseline": []}
            for holdout_id in ("cp_n21", "cp_n23", "cp_n25")
        }

        for summary_path in run_paths:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            holdout = data.get("holdout_evaluation") or {}
            if not holdout.get("enabled"):
                continue
            source_task_count = len(holdout.get("adaptation_by_source_task") or {})
            if source_task_count <= 0:
                raise ValueError(
                    f"Could not infer source task count from {summary_path}"
                )
            if task_count is None:
                task_count = source_task_count
            elif task_count != source_task_count:
                raise ValueError(
                    f"Inconsistent source task count in {summary_path}: "
                    f"expected {task_count}, got {source_task_count}"
                )

            shared = holdout["shared_zero_shot"]
            run_shared.append(shared["average_holdout_score"])
            for holdout_id, result in shared["holdout_task_results"].items():
                holdout_acc[holdout_id]["shared"].append(result["final_task_score"])

            for mode, key in (
                ("adapt", "adaptation_by_source_task"),
                ("baseline", "baseline_by_source_task"),
            ):
                source_entries = holdout[key]
                per_source_overall = []
                per_holdout = {holdout_id: [] for holdout_id in holdout_acc}
                for source_summary in source_entries.values():
                    per_source_overall.append(source_summary["average_holdout_score"])
                    for holdout_id, result in source_summary[
                        "holdout_task_results"
                    ].items():
                        per_holdout[holdout_id].append(result["final_task_score"])
                if mode == "adapt":
                    run_adapt.append(mean_or_raise(per_source_overall, context="adapt"))
                else:
                    run_baseline.append(
                        mean_or_raise(per_source_overall, context="baseline")
                    )
                for holdout_id, values in per_holdout.items():
                    holdout_acc[holdout_id][mode].append(
                        mean_or_raise(values, context=f"{mode}:{holdout_id}")
                    )

        if not run_shared:
            continue

        model_results[model_id] = {
            "shared_mean": statistics.fmean(run_shared),
            "adapt_mean": statistics.fmean(run_adapt),
            "baseline_mean": statistics.fmean(run_baseline),
            "holdouts": {
                holdout_id: {
                    mode: statistics.fmean(values)
                    for mode, values in mode_map.items()
                }
                for holdout_id, mode_map in holdout_acc.items()
            },
            "run_count": len(run_shared),
        }

    ordered_models = [model for model in MODEL_ORDER if model in model_results]
    if not ordered_models:
        raise SystemExit(
            f"No holdout-enabled runs found in {results_dir} for {setting_prefix}."
        )
    assert task_count is not None

    categories = [
        ("cp_n21", "N = 21"),
        ("cp_n23", "N = 23"),
        ("cp_n25", "N = 25"),
        ("average", "Average"),
    ]
    series = {"shared": [], "adapt": [], "baseline": []}

    for holdout_id, _label in categories:
        if holdout_id == "average":
            for mode in series:
                series[mode].append(
                    statistics.fmean(
                        [model_results[model][f"{mode}_mean"] for model in ordered_models]
                    )
                )
            continue
        for mode in series:
            series[mode].append(
                statistics.fmean(
                    [
                        model_results[model]["holdouts"][holdout_id][mode]
                        for model in ordered_models
                    ]
                )
            )

    return {
        "setting_prefix": setting_prefix,
        "results_dir": str(results_dir),
        "model_count": len(ordered_models),
        "task_count": task_count,
        "models": ordered_models,
        "categories": [{"id": cid, "label": label} for cid, label in categories],
        "series": series,
        "per_model": model_results,
    }


def annotate_bar_values(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: list[float],
    *,
    color: str,
    bold_indices: set[int] | None = None,
    y_offset: float = 0.015,
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
            fontsize=11.2 if is_bold else 10,
            fontweight="black" if is_bold else "semibold",
            color="#2F3B4A" if is_bold else color,
        )


def plot(payload: dict[str, Any], *, title: str, output_stem: Path, dpi: int) -> None:
    apply_plot_style()

    labels = [item["label"] for item in payload["categories"]]
    adapt_vals = payload["series"]["adapt"]
    baseline_vals = payload["series"]["baseline"]

    x = np.arange(len(labels))
    width = 0.32

    baseline_color = "#F6C8B8"
    adapt_color = "#A9D8C8"
    edge_color = "#6C7A89"

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        color=baseline_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Single-task",
    )
    ax.bar(
        x + width / 2,
        adapt_vals,
        width,
        color=adapt_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="EMO-STA Adapt",
    )

    ax.set_xlabel("Held-Out Circle Count (N)", fontweight="bold")
    ax.set_ylabel("Mean OOD Normalized Score Across LLMs", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.06)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)

    baseline_best = {
        idx
        for idx, (a, b) in enumerate(zip(adapt_vals, baseline_vals))
        if b >= a
    }
    adapt_best = {
        idx
        for idx, (a, b) in enumerate(zip(adapt_vals, baseline_vals))
        if a > b
    }

    annotate_bar_values(
        ax,
        x - width / 2,
        baseline_vals,
        color=edge_color,
        bold_indices=baseline_best,
    )
    annotate_bar_values(
        ax,
        x + width / 2,
        adapt_vals,
        color=edge_color,
        bold_indices=adapt_best,
    )

    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def write_sidecar_json(output_stem: Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    output_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "results_dir": args.results_dir,
                "setting_prefix": args.setting_prefix,
                **payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_stem = resolve_repo_path(args.output_stem)

    payload = collect_model_level_means(
        results_dir=results_dir,
        setting_prefix=args.setting_prefix,
    )
    plot(payload, title=args.title, output_stem=output_stem, dpi=args.dpi)
    write_sidecar_json(output_stem, payload, args)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")
    for category, adapt, baseline in zip(
        payload["categories"],
        payload["series"]["adapt"],
        payload["series"]["baseline"],
    ):
        print(
            f"{category['label']}: adapt={adapt:.3f}, single={baseline:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
