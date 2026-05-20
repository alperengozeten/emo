#!/usr/bin/env python3
"""Create fixed-b30 post-hoc OOD figures from completed EMO-STA runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PLOT_DIR = Path(__file__).resolve().parent
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plot_budget_labels import build_budget_triplet, build_single_task_budget_triplet

REPO_ROOT = Path(__file__).resolve().parents[2]

FAMILY_RESULTS_DIRS = {
    "heilbronn_triangle": "multi_task_shared_then_adapt/results/heilbronn_triangle",
    "circle_packing_rectangle": "multi_task_shared_then_adapt/results/circle_packing_rectangle",
}

FAMILY_OUTPUT_STEMS = {
    "heilbronn_triangle": {
        "holdout_eval": (
            "multi_task_shared_then_adapt/figures/"
            "heilbronn_triangle_s60_a15_b30_ood_holdout_eval"
        ),
        "by_holdout": (
            "multi_task_shared_then_adapt/figures/"
            "heilbronn_triangle_ood_b30_by_holdout"
        ),
    },
    "circle_packing_rectangle": {
        "holdout_eval": (
            "multi_task_shared_then_adapt/figures/"
            "circle_packing_rectangle_s60_a15_b30_ood_holdout_eval"
        ),
        "by_holdout": (
            "multi_task_shared_then_adapt/figures/"
            "circle_packing_rectangle_ood_b30_by_holdout"
        ),
    },
}

SETTING_RE = re.compile(
    r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-(?P<model>.+)-full$"
)

SERIES_COLORS = {
    "baseline": "#F6C8B8",
    "adapt": "#A9D8C8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot post-hoc OOD holdout figures for the completed b30 EMO-STA runs."
        )
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        choices=sorted(FAMILY_RESULTS_DIRS),
        help=(
            "Family to plot. May be passed multiple times. If omitted, plots both "
            "heilbronn_triangle and circle_packing_rectangle."
        ),
    )
    parser.add_argument(
        "--selected-setting-prefix",
        default="s60-a15-b30",
        help=(
            "Setting prefix for the selected-budget OOD holdout figure. "
            "Default: s60-a15-b30."
        ),
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=30,
        help="Baseline budget for the by-holdout figure. Default: 30.",
    )
    parser.add_argument(
        "--baseline-reference-prefix",
        default="s60-a15-b30",
        help=(
            "Setting prefix whose single-task OOD results anchor the by-holdout "
            "figure. Default: s60-a15-b30."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI for PNG outputs. Default: 300.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def apply_plot_style(plt_module) -> None:
    plt_module.rcParams.update(
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


def parse_budget_prefix(prefix: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)", prefix)
    if not match:
        raise ValueError(f"Invalid budget prefix: {prefix}")
    return (
        int(match.group("shared")),
        int(match.group("adapt")),
        int(match.group("baseline")),
    )


def family_label(family: str) -> str:
    return {
        "heilbronn_triangle": "Heilbronn Triangle",
        "circle_packing_rectangle": "Circle Packing Rectangle",
    }[family]


def sort_task_ids(task_ids: list[str]) -> list[str]:
    def key(task_id: str) -> tuple[int, str]:
        match = re.search(r"_n(?P<n>\d+)$", task_id)
        if match:
            return (int(match.group("n")), task_id)
        return (10**9, task_id)

    return sorted(task_ids, key=key)


def task_label(task_id: str) -> str:
    match = re.search(r"_n(?P<n>\d+)$", task_id)
    if match:
        return f"N = {int(match.group('n'))}"
    return task_id


def mean_or_raise(values: list[float], *, context: str) -> float:
    if not values:
        raise ValueError(f"No values available for {context}")
    return statistics.fmean(values)


def _summary_entries_by_kind(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped = {"shared_best": {}, "adapted": {}, "baseline": {}}
    for program_label, payload in summary.get("programs", {}).items():
        source_kind = payload.get("source_kind")
        if source_kind in grouped:
            grouped[source_kind][program_label] = payload
    return grouped


def _series_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    grouped = _summary_entries_by_kind(summary)
    ood_tasks = sort_task_ids(list(summary["ood_tasks"]))
    series = {"shared": [], "adapt": [], "baseline": []}

    for task_id in ood_tasks:
        shared_payload = grouped["shared_best"].get("shared_best")
        if shared_payload is None:
            raise ValueError("Missing shared_best program in post-hoc summary")
        shared_result = shared_payload["ood_results"][task_id]
        series["shared"].append(float(shared_result["score"]))

        for mode_name, grouped_key in (("adapt", "adapted"), ("baseline", "baseline")):
            entries = []
            for payload in grouped[grouped_key].values():
                result = payload["ood_results"][task_id]
                entries.append(float(result["score"]))
            series[mode_name].append(
                mean_or_raise(entries, context=f"{grouped_key}:{task_id}")
            )

    for mode_name in series:
        series[mode_name].append(
            mean_or_raise(series[mode_name], context=f"{mode_name}:average")
        )

    return {
        "ood_tasks": ood_tasks,
        "categories": ood_tasks + ["average"],
        "series": series,
    }


def infer_source_task_count_from_posthoc_summary(summary: dict[str, Any]) -> int:
    grouped = _summary_entries_by_kind(summary)
    task_count = len(grouped["adapted"]) or len(grouped["baseline"])
    if task_count <= 0:
        raise ValueError("Could not infer source task count from post-hoc summary")
    return task_count


def collect_model_level_means_from_posthoc(
    *,
    results_dir: Path,
    setting_prefix: str,
) -> dict[str, Any]:
    model_results: dict[str, dict[str, Any]] = {}
    categories: list[str] | None = None
    ood_tasks: list[str] | None = None
    task_count: int | None = None

    for setting_dir in sorted(results_dir.glob(f"{setting_prefix}-*")):
        if not setting_dir.is_dir():
            continue
        match = SETTING_RE.fullmatch(setting_dir.name)
        if match is None:
            continue
        model_id = match.group("model")
        run_paths = sorted(setting_dir.glob("run_*_seed_*/posthoc_ood_all_known/ood_summary.json"))
        if not run_paths:
            continue

        run_shared: list[list[float]] = []
        run_adapt: list[list[float]] = []
        run_baseline: list[list[float]] = []

        for summary_path in run_paths:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_task_count = infer_source_task_count_from_posthoc_summary(summary)
            if task_count is None:
                task_count = source_task_count
            elif task_count != source_task_count:
                raise ValueError(
                    f"Inconsistent source task count for {summary_path}: "
                    f"expected {task_count}, got {source_task_count}"
                )
            try:
                payload = _series_from_summary(summary)
            except ValueError as exc:
                print(
                    f"Skipping malformed post-hoc OOD summary {summary_path}: {exc}",
                    file=sys.stderr,
                )
                continue
            if categories is None:
                categories = list(payload["categories"])
                ood_tasks = list(payload["ood_tasks"])
            elif categories != payload["categories"]:
                print(
                    "Skipping inconsistent post-hoc OOD summary "
                    f"{summary_path}: expected {categories}, got {payload['categories']}",
                    file=sys.stderr,
                )
                continue
            run_shared.append(payload["series"]["shared"])
            run_adapt.append(payload["series"]["adapt"])
            run_baseline.append(payload["series"]["baseline"])

        if not run_shared:
            continue

        model_results[model_id] = {
            "shared_mean": [statistics.fmean(values) for values in zip(*run_shared)],
            "adapt_mean": [statistics.fmean(values) for values in zip(*run_adapt)],
            "baseline_mean": [statistics.fmean(values) for values in zip(*run_baseline)],
            "run_count": len(run_paths),
        }

    ordered_models = sorted(model_results)
    if not ordered_models:
        raise ValueError(
            f"No post-hoc OOD summaries found in {results_dir} for setting {setting_prefix}"
        )
    assert categories is not None
    assert ood_tasks is not None
    assert task_count is not None

    series = {
        "shared": [
            statistics.fmean([model_results[model]["shared_mean"][idx] for model in ordered_models])
            for idx in range(len(categories))
        ],
        "adapt": [
            statistics.fmean([model_results[model]["adapt_mean"][idx] for model in ordered_models])
            for idx in range(len(categories))
        ],
        "baseline": [
            statistics.fmean(
                [model_results[model]["baseline_mean"][idx] for model in ordered_models]
            )
            for idx in range(len(categories))
        ],
    }

    return {
        "setting_prefix": setting_prefix,
        "results_dir": str(results_dir),
        "model_count": len(ordered_models),
        "task_count": task_count,
        "models": ordered_models,
        "categories": [
            {"id": task_id, "label": task_label(task_id)}
            for task_id in ood_tasks
        ]
        + [{"id": "average", "label": "Average"}],
        "series": series,
        "per_model": model_results,
    }


def discover_setting_prefixes(results_dir: Path, *, baseline_budget: int) -> list[str]:
    prefixes: set[str] = set()
    for setting_dir in results_dir.iterdir():
        if not setting_dir.is_dir():
            continue
        match = SETTING_RE.fullmatch(setting_dir.name)
        if match is None:
            continue
        if int(match.group("baseline")) != baseline_budget:
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
        payload = collect_model_level_means_from_posthoc(
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


def build_by_holdout_payload(
    budgets: list[dict[str, Any]],
    *,
    baseline_reference: dict[str, Any],
) -> dict[str, Any]:
    if not budgets:
        raise ValueError("No budget payloads provided")

    categories = budgets[0]["categories"]
    baseline_label = build_single_task_budget_triplet(
        baseline=int(baseline_reference["budget"]["baseline"]),
        task_count=int(baseline_reference["budget"]["task_count"]),
    )["label"]
    baseline_values = baseline_reference["series"]["baseline"]
    adapt_by_budget = []
    for budget in budgets:
        adapt_by_budget.append(
            {
                "setting_prefix": budget["setting_prefix"],
                "budget": budget["budget"],
                "values": budget["series"]["adapt"],
                "model_count": budget["model_count"],
                "models": budget["models"],
            }
        )

    return {
        "categories": categories,
        "baseline_reference": {
            "label": f"Single-task {baseline_label}",
            "values": baseline_values,
            "source_setting_prefix": baseline_reference["setting_prefix"],
            "aggregation": (
                "For each holdout N, use the single-task post-hoc OOD score from the "
                "specified baseline reference setting so this figure mirrors the "
                "fixed-b30 comparison setup."
            ),
        },
        "adapt_by_budget": adapt_by_budget,
        "raw_budgets": budgets,
    }


def annotate_bar_values(
    ax,
    xs,
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
        )


def write_sidecar_json(output_stem: Path, payload: dict[str, Any]) -> None:
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def plot_holdout_eval(
    payload: dict[str, Any],
    *,
    family: str,
    output_stem: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style(plt)

    labels = [item["label"] for item in payload["categories"]]
    adapt_vals = payload["series"]["adapt"]
    baseline_vals = payload["series"]["baseline"]

    x = np.arange(len(labels))
    width = 0.32
    edge_color = "#6C7A89"

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        color=SERIES_COLORS["baseline"],
        edgecolor=edge_color,
        linewidth=0.8,
        label="Single-task",
    )
    ax.bar(
        x + width / 2,
        adapt_vals,
        width,
        color=SERIES_COLORS["adapt"],
        edgecolor=edge_color,
        linewidth=0.8,
        label="EMO-STA Adapt",
    )

    ax.set_xlabel("Held-Out Task Size (N)", fontweight="bold")
    ax.set_ylabel("Mean OOD Normalized Score Across LLMs", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.06)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)

    baseline_best = {
        idx for idx, (adapt, baseline) in enumerate(zip(adapt_vals, baseline_vals)) if baseline >= adapt
    }
    adapt_best = {
        idx for idx, (adapt, baseline) in enumerate(zip(adapt_vals, baseline_vals)) if adapt > baseline
    }
    annotate_bar_values(ax, x - width / 2, baseline_vals, color=edge_color, bold_indices=baseline_best)
    annotate_bar_values(ax, x + width / 2, adapt_vals, color=edge_color, bold_indices=adapt_best)

    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def plot_by_holdout(
    payload: dict[str, Any],
    *,
    family: str,
    output_stem: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style(plt)

    labels = [category["label"] for category in payload["categories"]]
    series = [
        {
            "label": payload["baseline_reference"]["label"],
            "values": payload["baseline_reference"]["values"],
            "color": SERIES_COLORS["baseline"],
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


def main() -> int:
    args = parse_args()
    families = args.families or ["heilbronn_triangle", "circle_packing_rectangle"]

    for family in families:
        results_dir = resolve_repo_path(FAMILY_RESULTS_DIRS[family])
        output_stems = FAMILY_OUTPUT_STEMS[family]

        holdout_payload = collect_model_level_means_from_posthoc(
            results_dir=results_dir,
            setting_prefix=args.selected_setting_prefix,
        )
        holdout_payload["family"] = family
        holdout_payload["aggregation"] = (
            "For each model, average per-run post-hoc OOD scores first, then average "
            "across models. EMO-STA Adapt and Single-task aggregate across the frozen "
            "task-adapted programs from the completed runs."
        )
        holdout_output_stem = resolve_repo_path(output_stems["holdout_eval"])
        plot_holdout_eval(
            holdout_payload,
            family=family,
            output_stem=holdout_output_stem,
            dpi=args.dpi,
        )
        write_sidecar_json(holdout_output_stem, holdout_payload)

        setting_prefixes = discover_setting_prefixes(
            results_dir,
            baseline_budget=args.baseline_budget,
        )
        budgets = collect_budget_payloads(
            results_dir=results_dir,
            setting_prefixes=setting_prefixes,
        )
        baseline_reference = next(
            budget
            for budget in budgets
            if budget["setting_prefix"] == args.baseline_reference_prefix
        )
        by_holdout_payload = build_by_holdout_payload(
            budgets,
            baseline_reference=baseline_reference,
        )
        by_holdout_payload["family"] = family
        by_holdout_output_stem = resolve_repo_path(output_stems["by_holdout"])
        plot_by_holdout(
            by_holdout_payload,
            family=family,
            output_stem=by_holdout_output_stem,
            dpi=args.dpi,
        )
        write_sidecar_json(by_holdout_output_stem, by_holdout_payload)

        print(f"Wrote {holdout_output_stem.with_suffix('.png')}")
        print(f"Wrote {holdout_output_stem.with_suffix('.pdf')}")
        print(f"Wrote {holdout_output_stem.with_suffix('.svg')}")
        print(f"Wrote {holdout_output_stem.with_suffix('.json')}")
        print(f"Wrote {by_holdout_output_stem.with_suffix('.png')}")
        print(f"Wrote {by_holdout_output_stem.with_suffix('.pdf')}")
        print(f"Wrote {by_holdout_output_stem.with_suffix('.svg')}")
        print(f"Wrote {by_holdout_output_stem.with_suffix('.json')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
