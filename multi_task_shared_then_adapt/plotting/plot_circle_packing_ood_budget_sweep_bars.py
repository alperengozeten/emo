#!/usr/bin/env python3
"""Create a circle-packing OOD holdout budget-sweep bar plot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = Path(__file__).resolve().parent
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plot_budget_labels import build_budget_triplet, budget_axis_label
from plot_circle_packing_ood_holdout_bars import (
    apply_plot_style,
    collect_model_level_means,
    resolve_repo_path,
)


DEFAULT_RESULTS_DIR = "multi_task_shared_then_adapt/results/circle_packing"
DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/"
    "circle_packing_ood_budget_sweep"
)

SETTING_RE = re.compile(r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot circle-packing OOD holdout performance across EMO-STA budgets."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Circle-packing results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--setting-prefix",
        action="append",
        dest="setting_prefixes",
        help=(
            "Budget prefix to include, e.g. s60-a15-b30. May be passed multiple "
            "times. If omitted, all clean holdout-enabled settings are used."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=(
            "Output path without extension. The script writes .png, .pdf, .svg, "
            f"and .json. Default: {DEFAULT_OUTPUT_STEM}"
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


def parse_budget_prefix(prefix: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)", prefix)
    if not match:
        raise ValueError(f"Invalid budget prefix: {prefix}")
    return (
        int(match.group("shared")),
        int(match.group("adapt")),
        int(match.group("baseline")),
    )


def discover_setting_prefixes(results_dir: Path) -> list[str]:
    prefixes: set[str] = set()
    for setting_dir in results_dir.iterdir():
        if not setting_dir.is_dir():
            continue
        match = SETTING_RE.match(setting_dir.name)
        if not match:
            continue
        prefixes.add(
            f"s{match.group('shared')}-a{match.group('adapt')}-b{match.group('baseline')}"
        )
    return sorted(prefixes, key=parse_budget_prefix)


def average_index(payload: dict[str, Any]) -> int:
    for idx, category in enumerate(payload["categories"]):
        if category["id"] == "average":
            return idx
    raise ValueError("Holdout payload does not contain an average category")


def collect_budget_payloads(
    *,
    results_dir: Path,
    setting_prefixes: list[str],
) -> list[dict[str, Any]]:
    budgets: list[dict[str, Any]] = []
    for prefix in setting_prefixes:
        payload = collect_model_level_means(
            results_dir=results_dir,
            setting_prefix=prefix,
        )
        idx = average_index(payload)
        shared, adapt, baseline = parse_budget_prefix(prefix)
        budgets.append(
            {
                "setting_prefix": prefix,
                "budget": build_budget_triplet(
                    shared=shared,
                    adapt=adapt,
                    baseline=baseline,
                    task_count=int(payload["task_count"]),
                ),
                "model_count": payload["model_count"],
                "models": payload["models"],
                "adapt_mean": payload["series"]["adapt"][idx],
                "baseline_mean": payload["series"]["baseline"][idx],
                "shared_mean": payload["series"]["shared"][idx],
                "adapt_minus_baseline": (
                    payload["series"]["adapt"][idx]
                    - payload["series"]["baseline"][idx]
                ),
                "holdout_payload": payload,
            }
        )
    return budgets


def annotate_bar_values(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: list[float],
    *,
    color: str,
    bold_indices: set[int] | None = None,
    y_offset: float = 0.014,
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
            fontsize=11.0 if is_bold else 9.8,
            fontweight="black" if is_bold else "semibold",
            color="#2F3B4A" if is_bold else color,
        )


def plot_budget_sweep(
    budgets: list[dict[str, Any]],
    *,
    title: str,
    output_stem: Path,
    dpi: int,
) -> None:
    apply_plot_style()

    labels = [
        (
            item["budget"]["label"]
            if item["model_count"] == 5
            else f"{item['budget']['label']}\n(n={item['model_count']})"
        )
        for item in budgets
    ]
    baseline_vals = [item["baseline_mean"] for item in budgets]
    adapt_vals = [item["adapt_mean"] for item in budgets]

    x = np.arange(len(labels))
    width = 0.34

    baseline_color = "#F6C8B8"
    adapt_color = "#A9D8C8"
    edge_color = "#6C7A89"

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
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

    ax.set_xlabel(budget_axis_label(), fontweight="bold")
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
        for idx, (adapt_value, baseline_value) in enumerate(zip(adapt_vals, baseline_vals))
        if baseline_value >= adapt_value
    }
    adapt_best = {
        idx
        for idx, (adapt_value, baseline_value) in enumerate(zip(adapt_vals, baseline_vals))
        if adapt_value > baseline_value
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


def write_sidecar_json(
    output_stem: Path,
    budgets: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "results_dir": args.results_dir,
                "setting_prefixes": [item["setting_prefix"] for item in budgets],
                "holdouts": ["cp_n21", "cp_n23", "cp_n25"],
                "aggregation": (
                    "For each budget, average holdout score is computed per run, "
                    "then averaged over source tasks for adaptation and baseline, "
                    "then averaged over the five model-level means."
                ),
                "budgets": budgets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_stem = resolve_repo_path(args.output_stem)

    setting_prefixes = args.setting_prefixes or discover_setting_prefixes(results_dir)
    budgets = collect_budget_payloads(
        results_dir=results_dir,
        setting_prefixes=setting_prefixes,
    )
    if not budgets:
        raise SystemExit("No holdout-enabled circle-packing budget settings found.")

    plot_budget_sweep(
        budgets,
        title=args.title,
        output_stem=output_stem,
        dpi=args.dpi,
    )
    write_sidecar_json(output_stem, budgets, args)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")
    for item in budgets:
        print(
            f"{item['budget']['label']}: "
            f"adapt={item['adapt_mean']:.3f}, "
            f"single={item['baseline_mean']:.3f}, "
            f"delta={item['adapt_minus_baseline']:+.3f}, "
            f"models={item['model_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
