#!/usr/bin/env python3
"""Create a circle-packing OOD bar plot by holdout N at fixed baseline budget."""

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

from plot_budget_labels import build_budget_triplet, build_single_task_budget_triplet
from plot_circle_packing_ood_holdout_bars import (
    apply_plot_style,
    collect_model_level_means,
    resolve_repo_path,
)


DEFAULT_RESULTS_DIR = "multi_task_shared_then_adapt/results/circle_packing"
DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/"
    "circle_packing_ood_b30_by_holdout"
)

SETTING_RE = re.compile(r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot circle-packing OOD holdout performance by N for fixed-baseline "
            "budget sweeps."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Circle-packing results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=30,
        help="Single-task baseline budget to use. Default: 30.",
    )
    parser.add_argument(
        "--baseline-reference-prefix",
        default="s60-a15-b30",
        help=(
            "Setting prefix whose single-task holdout results are used for the "
            "Single-task bars. Default: s60-a15-b30, matching the original "
            "circle-packing OOD figure."
        ),
    )
    parser.add_argument(
        "--setting-prefix",
        action="append",
        dest="setting_prefixes",
        help=(
            "Budget prefix to include, e.g. s60-a15-b30. May be passed multiple "
            "times. If omitted, all holdout-enabled settings matching "
            "--baseline-budget are used."
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


def discover_setting_prefixes(results_dir: Path, *, baseline_budget: int) -> list[str]:
    prefixes: set[str] = set()
    for setting_dir in results_dir.iterdir():
        if not setting_dir.is_dir():
            continue
        match = SETTING_RE.match(setting_dir.name)
        if not match or int(match.group("baseline")) != baseline_budget:
            continue
        prefixes.add(
            f"s{match.group('shared')}-a{match.group('adapt')}-b{match.group('baseline')}"
        )
    return sorted(prefixes, key=parse_budget_prefix)


def collect_budget_payloads(
    *,
    results_dir: Path,
    setting_prefixes: list[str],
) -> list[dict[str, Any]]:
    budgets: list[dict[str, Any]] = []
    for prefix in setting_prefixes:
        shared, adapt, baseline = parse_budget_prefix(prefix)
        payload = collect_model_level_means(
            results_dir=results_dir,
            setting_prefix=prefix,
        )
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
                "categories": payload["categories"],
                "series": payload["series"],
                "per_model": payload["per_model"],
            }
        )
    return budgets


def category_indices(categories: list[dict[str, str]]) -> list[tuple[int, str, str]]:
    ordered_ids = ["cp_n21", "cp_n23", "cp_n25", "average"]
    labels_by_id = {item["id"]: item["label"] for item in categories}
    indices_by_id = {item["id"]: idx for idx, item in enumerate(categories)}
    return [
        (indices_by_id[category_id], category_id, labels_by_id[category_id])
        for category_id in ordered_ids
        if category_id in indices_by_id
    ]


def build_plot_payload(
    budgets: list[dict[str, Any]],
    *,
    baseline_reference: dict[str, Any],
) -> dict[str, Any]:
    if not budgets:
        raise ValueError("No budget payloads provided")

    category_info = category_indices(budgets[0]["categories"])
    baseline_label = build_single_task_budget_triplet(
        baseline=int(baseline_reference["budget"]["baseline"]),
        task_count=int(baseline_reference["budget"]["task_count"]),
    )["label"]
    baseline_values = [
        baseline_reference["series"]["baseline"][idx]
        for idx, _category_id, _label in category_info
    ]

    adapt_by_budget = []
    for budget in budgets:
        adapt_by_budget.append(
            {
                "setting_prefix": budget["setting_prefix"],
                "budget": budget["budget"],
                "values": [
                    budget["series"]["adapt"][idx]
                    for idx, _category_id, _label in category_info
                ],
                "model_count": budget["model_count"],
                "models": budget["models"],
            }
        )

    return {
        "categories": [
            {"id": category_id, "label": label}
            for _idx, category_id, label in category_info
        ],
        "baseline_reference": {
            "label": f"Single-task {baseline_label}",
            "values": baseline_values,
            "source_setting_prefix": baseline_reference["setting_prefix"],
            "aggregation": (
                "For each holdout N, use the single-task holdout score from the "
                "specified baseline reference setting so this figure is consistent "
                "with the original selected-budget OOD figure."
            ),
        },
        "adapt_by_budget": adapt_by_budget,
        "raw_budgets": budgets,
    }


def annotate_bar_values(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: list[float],
    *,
    color: str,
    bold_indices: set[int] | None = None,
    y_offset: float = 0.012,
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
            fontsize=9.8 if is_bold else 8.8,
            fontweight="black" if is_bold else "semibold",
            color="#2F3B4A" if is_bold else color,
            rotation=0,
        )


def plot(payload: dict[str, Any], *, title: str, output_stem: Path, dpi: int) -> None:
    apply_plot_style()

    labels = [category["label"] for category in payload["categories"]]
    series = [
        {
            "label": payload["baseline_reference"]["label"],
            "values": payload["baseline_reference"]["values"],
            "color": "#F6C8B8",
        },
        *[
            {
                "label": f"EMO-STA {item['budget']['label']}",
                "values": item["values"],
                "color": color,
            }
            for item, color in zip(
                payload["adapt_by_budget"],
                ["#CBE3D2", "#A9D8C8", "#7CC7A8", "#4EA685"],
            )
        ],
    ]

    x = np.arange(len(labels))
    width = min(0.18, 0.76 / len(series))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    edge_color = "#6C7A89"

    fig, ax = plt.subplots(figsize=(11.0, 5.4))

    values_by_category = list(zip(*[item["values"] for item in series]))
    best_indices_by_series: list[set[int]] = [set() for _ in series]
    for category_idx, values in enumerate(values_by_category):
        best_value = max(values)
        for series_idx, value in enumerate(values):
            if abs(value - best_value) <= 1e-12:
                best_indices_by_series[series_idx].add(category_idx)

    for idx, item in enumerate(series):
        xs = x + offsets[idx]
        ax.bar(
            xs,
            item["values"],
            width,
            color=item["color"],
            edgecolor=edge_color,
            linewidth=0.8,
            label=item["label"],
        )
        annotate_bar_values(
            ax,
            xs,
            item["values"],
            color=edge_color,
            bold_indices=best_indices_by_series[idx],
        )

    ax.set_xlabel("OOD Holdout Task", fontweight="bold")
    ax.set_ylabel("Mean OOD Normalized Score Across LLMs", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.06)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)

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
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "results_dir": args.results_dir,
                "baseline_budget": args.baseline_budget,
                "baseline_reference_prefix": args.baseline_reference_prefix,
                "setting_prefixes": [
                    item["setting_prefix"] for item in payload["adapt_by_budget"]
                ],
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

    setting_prefixes = args.setting_prefixes or discover_setting_prefixes(
        results_dir,
        baseline_budget=args.baseline_budget,
    )
    budgets = collect_budget_payloads(
        results_dir=results_dir,
        setting_prefixes=setting_prefixes,
    )
    if not budgets:
        raise SystemExit(
            f"No holdout-enabled circle-packing settings found for b={args.baseline_budget}."
        )

    baseline_references = collect_budget_payloads(
        results_dir=results_dir,
        setting_prefixes=[args.baseline_reference_prefix],
    )
    baseline_reference = baseline_references[0]

    payload = build_plot_payload(
        budgets,
        baseline_reference=baseline_reference,
    )
    plot(payload, title=args.title, output_stem=output_stem, dpi=args.dpi)
    write_sidecar_json(output_stem, payload, args)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")

    print(payload["baseline_reference"]["label"])
    for category, value in zip(
        payload["categories"],
        payload["baseline_reference"]["values"],
    ):
        print(f"  {category['label']}: {value:.3f}")
    for budget in payload["adapt_by_budget"]:
        print(f"EMO-STA {budget['budget']['label']}")
        for category, value in zip(payload["categories"], budget["values"]):
            print(f"  {category['label']}: {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
