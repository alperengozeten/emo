#!/usr/bin/env python3
"""Create holdout-style figures from post-hoc OOD evaluation summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

SERIES = [
    ("baseline", "Single-task", "#F6C8B8"),
    ("adapted", "EMO-STA Adapt", "#A9D8C8"),
    ("shared_best", "Shared best", "#A8C7E8"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot post-hoc unseen-nearby-n OOD results from an ood_summary.json. "
            "Writes *_ood_holdout_eval and *_ood_by_holdout figures."
        )
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to a post-hoc OOD ood_summary.json file.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "Output path prefix without the figure suffix. For example, passing "
            "'figures/foo' writes figures/foo_ood_holdout_eval.* and "
            "figures/foo_ood_by_holdout.*."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("score", "combined_score"),
        default="score",
        help="Metric to plot from each OOD result. Default: score.",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional title prefix. Leave empty for paper-style figures.",
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


def task_n(task_id: str) -> int | None:
    match = re.search(r"_n(?P<n>\d+)$", task_id)
    return int(match.group("n")) if match else None


def task_label(task_id: str, direction: str | None) -> str:
    n_value = task_n(task_id)
    base = f"N = {n_value}" if n_value is not None else task_id
    return f"{base}\n{direction}" if direction else base


def family_label(family: str) -> str:
    return {
        "heilbronn_triangle": "Heilbronn Triangle",
        "circle_packing_rectangle": "Circle Packing Rectangle",
    }.get(family, family.replace("_", " ").title())


def source_task_label(task_id: str | None) -> str:
    if task_id is None:
        return "shared"
    n_value = task_n(task_id)
    return f"N={n_value}" if n_value is not None else task_id


def mean(values: list[float], *, context: str) -> float:
    if not values:
        raise ValueError(f"No values available for {context}")
    return statistics.fmean(values)


def load_summary(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if data.get("algorithm") != "posthoc_ood_evaluation":
        raise ValueError(f"{summary_path} is not a post-hoc OOD summary")
    if not data.get("ood_tasks"):
        raise ValueError(f"{summary_path} does not contain any OOD tasks")
    if not data.get("programs"):
        raise ValueError(f"{summary_path} does not contain any programs")
    return data


def collect_task_rows(summary: dict[str, Any], *, metric: str) -> list[dict[str, Any]]:
    ood_tasks = list(summary["ood_tasks"])
    task_rows: list[dict[str, Any]] = []

    for task_id in ood_tasks:
        by_kind: dict[str, list[dict[str, Any]]] = {
            "shared_best": [],
            "adapted": [],
            "baseline": [],
        }
        direction = None
        target_value = None
        objective_key = None

        for program_label, program in summary["programs"].items():
            result = (program.get("ood_results") or {}).get(task_id)
            if not result:
                continue
            metrics = result.get("metrics") or {}
            if "target_min_area" in metrics:
                target_value = metrics["target_min_area"]
                objective_key = "min_triangle_area"
            elif "target_sum_radii" in metrics:
                target_value = metrics["target_sum_radii"]
                objective_key = "sum_radii"

            source_kind = program.get("source_kind")
            if source_kind not in by_kind:
                continue
            by_kind[source_kind].append(
                {
                    "program_label": program_label,
                    "source_task_id": program.get("source_task_id"),
                    "value": float(result.get(metric, 0.0)),
                    "score": float(result.get("score", 0.0)),
                    "combined_score": float(result.get("combined_score", 0.0)),
                    "validity": float(metrics.get("validity", 0.0)),
                    "objective_value": (
                        metrics.get(objective_key) if objective_key else None
                    ),
                    "target_value": target_value,
                    "error": result.get("error"),
                }
            )

        # Direction is currently carried in the compact CSV, not every JSON result.
        n_value = task_n(task_id)
        train_ns = [
            task_n(training_id)
            for training_id in summary.get("training_task_ids", [])
            if task_n(training_id) is not None
        ]
        if train_ns and n_value is not None:
            if n_value < min(train_ns):
                direction = "backward"
            elif n_value > max(train_ns):
                direction = "forward"

        task_rows.append(
            {
                "task_id": task_id,
                "n": n_value,
                "direction": direction,
                "target_value": target_value,
                "objective_key": objective_key,
                "label": task_label(task_id, direction),
                "by_kind": by_kind,
            }
        )

    return task_rows


def build_aggregate_payload(
    summary: dict[str, Any],
    task_rows: list[dict[str, Any]],
    *,
    metric: str,
    include_average: bool,
) -> dict[str, Any]:
    categories = [
        {
            "id": row["task_id"],
            "label": row["label"],
            "direction": row["direction"],
            "n": row["n"],
            "target_value": row["target_value"],
            "objective_key": row["objective_key"],
        }
        for row in task_rows
    ]
    series_values: dict[str, list[float]] = {series_id: [] for series_id, *_ in SERIES}
    details: dict[str, dict[str, Any]] = {}

    for row in task_rows:
        task_details: dict[str, Any] = {}
        for series_id, _label, _color in SERIES:
            entries = row["by_kind"][series_id]
            value = mean(
                [entry["value"] for entry in entries],
                context=f"{series_id}:{row['task_id']}",
            )
            series_values[series_id].append(value)
            best_entry = max(entries, key=lambda entry: entry["value"])
            task_details[series_id] = {
                "mean": value,
                "count": len(entries),
                "valid_count": sum(1 for entry in entries if entry["validity"] > 0.0),
                "best_program_label": best_entry["program_label"],
                "best_score": best_entry["score"],
                "best_combined_score": best_entry["combined_score"],
                "best_objective_value": best_entry["objective_value"],
                "target_value": best_entry["target_value"],
            }
        details[row["task_id"]] = task_details

    if include_average:
        categories.append(
            {
                "id": "average",
                "label": "Average",
                "direction": "average",
                "n": None,
                "target_value": None,
                "objective_key": None,
            }
        )
        for series_id in series_values:
            series_values[series_id].append(
                mean(series_values[series_id], context=f"{series_id}:average")
            )

    return {
        "family": summary["family"],
        "results_dir": summary.get("results_dir"),
        "summary_json": summary.get("summary_path"),
        "metric": metric,
        "evaluation_regime": summary.get("evaluation_regime"),
        "note": summary.get("note"),
        "selection_note": summary.get("selection_note"),
        "aggregation": (
            "shared_best is the single shared frozen program; EMO-STA Adapt and "
            "Single-task are means across the frozen programs selected by each "
            "in-distribution source task."
        ),
        "categories": categories,
        "series": series_values,
        "details": details,
    }


def annotate_bar_values(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: list[float],
    *,
    color: str,
    bold_indices: set[int] | None = None,
    y_offset: float,
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
            fontsize=10.6 if is_bold else 9.3,
            fontweight="black" if is_bold else "semibold",
            color="#2F3B4A" if is_bold else color,
        )


def plot_grouped_bars(
    payload: dict[str, Any],
    *,
    title: str,
    output_stem: Path,
    ylabel: str,
    dpi: int,
) -> None:
    apply_plot_style()

    labels = [category["label"] for category in payload["categories"]]
    plotted_series = [
        {
            "id": series_id,
            "label": label,
            "values": payload["series"][series_id],
            "color": color,
        }
        for series_id, label, color in SERIES
    ]

    x = np.arange(len(labels))
    width = min(0.22, 0.76 / len(plotted_series))
    offsets = (np.arange(len(plotted_series)) - (len(plotted_series) - 1) / 2.0) * width
    edge_color = "#6C7A89"
    max_value = max(max(item["values"]) for item in plotted_series)
    y_upper = max(1.0, math.ceil(max_value * 120.0) / 100.0)
    y_offset = max(0.01, y_upper * 0.014)

    fig_width = 8.4 if len(labels) <= 4 else 10.2
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    values_by_category = list(zip(*[item["values"] for item in plotted_series]))
    best_indices_by_series: list[set[int]] = [set() for _ in plotted_series]
    for category_idx, values in enumerate(values_by_category):
        best_value = max(values)
        for series_idx, value in enumerate(values):
            if abs(value - best_value) <= 1e-12:
                best_indices_by_series[series_idx].add(category_idx)

    for idx, item in enumerate(plotted_series):
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
            y_offset=y_offset,
        )

    ax.set_xlabel("OOD Holdout Task", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, y_upper + y_offset * 4.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)

    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def write_sidecar_json(output_stem: Path, payload: dict[str, Any]) -> None:
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    summary_path = resolve_repo_path(args.summary_json)
    output_prefix = resolve_repo_path(args.output_prefix)
    summary = load_summary(summary_path)
    summary["summary_path"] = str(summary_path)

    task_rows = collect_task_rows(summary, metric=args.metric)
    family_name = family_label(summary["family"])
    title_prefix = args.title_prefix.strip()
    title_base = f"{title_prefix} {family_name}".strip()
    ylabel = f"Post-hoc OOD {args.metric.replace('_', ' ').title()}"

    holdout_payload = build_aggregate_payload(
        summary,
        task_rows,
        metric=args.metric,
        include_average=True,
    )
    holdout_stem = output_prefix.with_name(f"{output_prefix.name}_ood_holdout_eval")
    plot_grouped_bars(
        holdout_payload,
        title=f"{title_base} OOD Holdout Evaluation" if title_prefix else "",
        output_stem=holdout_stem,
        ylabel=ylabel,
        dpi=args.dpi,
    )
    write_sidecar_json(holdout_stem, holdout_payload)

    by_holdout_payload = build_aggregate_payload(
        summary,
        task_rows,
        metric=args.metric,
        include_average=False,
    )
    by_holdout_stem = output_prefix.with_name(f"{output_prefix.name}_ood_by_holdout")
    plot_grouped_bars(
        by_holdout_payload,
        title=f"{title_base} OOD by Holdout" if title_prefix else "",
        output_stem=by_holdout_stem,
        ylabel=ylabel,
        dpi=args.dpi,
    )
    write_sidecar_json(by_holdout_stem, by_holdout_payload)

    for stem in (holdout_stem, by_holdout_stem):
        print(f"Wrote {stem.with_suffix('.png')}")
        print(f"Wrote {stem.with_suffix('.pdf')}")
        print(f"Wrote {stem.with_suffix('.svg')}")
        print(f"Wrote {stem.with_suffix('.json')}")

    for payload_name, payload in (
        ("ood_holdout_eval", holdout_payload),
        ("ood_by_holdout", by_holdout_payload),
    ):
        print(payload_name)
        for category_idx, category in enumerate(payload["categories"]):
            values = {
                label: payload["series"][series_id][category_idx]
                for series_id, label, _color in SERIES
            }
            print(
                f"  {category['label'].replace(chr(10), ' ')}: "
                + ", ".join(f"{label}={value:.3f}" for label, value in values.items())
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
