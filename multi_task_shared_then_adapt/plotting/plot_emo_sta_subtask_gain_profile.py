#!/usr/bin/env python3
"""Plot subtask-level EMO-STA gains relative to direct single-task baselines."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError:
    plt = None
    Line2D = None


DEFAULT_SUMMARY_JSON = "multi_task_shared_then_adapt/docs/emo_sta_results_summary.json"
DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/emo_sta_subtask_gain_profile"
)
PREFERRED_GAIN_X_RIGHT = 0.2
FAMILY_X_LIMIT_OVERRIDES = {
    "circle_packing": (-0.05, 0.05),
    "circle_packing_rectangle": (-0.05, 0.05),
    "signal_processing": (-0.05, 0.10),
    "sldbench_3d": (-0.05, 0.05),
}
FAMILY_X_TICK_STEPS = {
    "circle_packing": 0.05,
    "circle_packing_rectangle": 0.05,
    "signal_processing": 0.05,
    "sldbench_3d": 0.05,
}

FAMILY_DISPLAY = {
    "circle_packing": "Circle packing",
    "circle_packing_rectangle": "Circle packing rectangles",
    "function_minimization": "Function minimization",
    "heilbronn_triangle": "Heilbronn triangle",
    "k_module_problem_balanced": "K-module",
    "signal_processing": "Signal processing",
    "sldbench_3d": "SLDBench-3D",
    "rust_adaptive_sort": "Rust adaptive sort",
}

FAMILY_COLORS = {
    "circle_packing": "#2f6db5",
    "circle_packing_rectangle": "#8c564b",
    "function_minimization": "#1b9e77",
    "heilbronn_triangle": "#b07aa1",
    "k_module_problem_balanced": "#d95f02",
    "signal_processing": "#4c78a8",
    "sldbench_3d": "#af7aa1",
    "rust_adaptive_sort": "#59a14f",
}

TASK_DISPLAY = {
    "cp_n20": "N=20",
    "cp_n22": "N=22",
    "cp_n24": "N=24",
    "cp_n26": "N=26",
    "cp_rect_n20": "N=20",
    "cp_rect_n21": "N=21",
    "cp_rect_n22": "N=22",
    "cp_rect_n23": "N=23",
    "fm_ackley_2d": "Ackley",
    "fm_rastrigin_2d": "Rastrigin",
    "fm_rosenbrock_2d": "Rosenbrock",
    "fm_sincosxy_2d": "Oscillatory Basin",
    "heil_tri_n9": "N=9",
    "heil_tri_n10": "N=10",
    "heil_tri_n11": "N=11",
    "heil_tri_n12": "N=12",
    "sp_chirp_700_n04": "chirp",
    "sp_multifreq_600_n03": "multi-freq",
    "sp_step_800_n05": "step",
    "sp_trend_sine_500_n02": "trend+sine",
    "kmb_task_a": "task a",
    "kmb_task_b": "task b",
    "kmb_task_c": "task c",
    "kmb_task_d": "task d",
    "data_constrained_scaling_law": "data-constr",
    "vocab_scaling_law": "vocab",
    "ras_duplicates": "duplicates",
    "ras_nearly_sorted": "nearly sorted",
    "ras_random": "random",
    "ras_reverse_sorted": "reverse sorted",
}

SVG_FONT = "DejaVu Sans, Arial, sans-serif"

EMO_STA_TABLE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]

DEFAULT_EMO_STA_TABLE_FAMILIES = {
    "warmstart": [
        "function_minimization",
        "signal_processing",
        "k_module_problem_balanced",
        "sldbench_3d",
        "rust_adaptive_sort",
    ],
    "best_local": [
        "function_minimization",
        "circle_packing",
        "circle_packing_rectangle",
        "heilbronn_triangle",
        "signal_processing",
        "sldbench_3d",
    ],
}

EMO_STA_TABLE_BUDGETS = {
    "circle_packing": (60, 15, 30),
    "circle_packing_rectangle": (60, 15, 30),
    "function_minimization": (40, 15, 25),
    "heilbronn_triangle": (60, 15, 30),
    "signal_processing": (60, 10, 25),
    "k_module_problem_balanced": (40, 20, 30),
    "sldbench_3d": (60, 10, 40),
    "rust_adaptive_sort": (60, 10, 25),
}

PHASE_PAIR_MODES = {
    "warmstart": {
        "figure_name": "emo_sta_subtask_gain_profile",
        "start_key": "spawn",
        "end_key": "adaptation",
        "start_label": "Spawned shared checkpoint",
        "end_label": "After task-specific adaptation",
        "family_mean_label": "mean warmstart improvement",
    },
    "best_local": {
        "figure_name": "emo_sta_subtask_gain_profile_best_local",
        "start_key": "best_shared_spawn",
        "end_key": "best_local_adaptation",
        "start_label": "STA Best-Shared (Before Adaptation)",
        "end_label": "STA Best-Local",
        "family_mean_label": "mean STA Best-Local improvement",
    },
}


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def score_from_metrics(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    score = metrics.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    score = metrics.get("combined_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def load_seed_spawn_score(run_root: Path, task_id: str, *, seed_kind: str) -> float | None:
    root_name = {
        "best_shared": "spawned_checkpoints_best_shared",
        "best_local": "spawned_checkpoints_best_local",
    }.get(seed_kind)
    if root_name is None:
        return None
    best_info = read_json_if_exists(run_root / root_name / task_id / "best_program_info.json")
    if best_info is None:
        return None
    return score_from_metrics(best_info.get("metrics"))


def compare_scores(left: float | None, right: float | None, *, tol: float = 1e-12) -> str | None:
    if left is None or right is None:
        return None
    diff = left - right
    if abs(diff) <= tol:
        return "tie"
    return "win" if diff > 0 else "loss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean per-subtask EMO-STA gains relative to direct single-task baselines."
        )
    )
    parser.add_argument(
        "--summary-json",
        default=DEFAULT_SUMMARY_JSON,
        help=f"Path to EMO-STA summary JSON. Default: {DEFAULT_SUMMARY_JSON}",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=(
            "Output path without extension. The script always writes .svg and .json, "
            "and additionally writes .png and .pdf when matplotlib is available. "
            f"Default: {DEFAULT_OUTPUT_STEM}"
        ),
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title. Leave empty to omit the title.",
    )
    parser.add_argument(
        "--preset",
        choices=["emo_sta_table"],
        default=None,
        help=(
            "Optional setting filter preset. `emo_sta_table` keeps only the model/budget "
            "settings used in emo_sta_table.md."
        ),
    )
    parser.add_argument(
        "--include-family",
        action="append",
        default=None,
        help=(
            "Family id to include. Pass multiple times to both filter and set panel order, "
            "for example: function_minimization, signal_processing, "
            "k_module_problem_balanced, sldbench_3d, rust_adaptive_sort."
        ),
    )
    parser.add_argument(
        "--exclude-family",
        action="append",
        default=None,
        help="Family id to exclude. Applied after --include-family filtering.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI for PNG when matplotlib is available. Default: 300.",
    )
    parser.add_argument(
        "--phase-pair",
        choices=sorted(PHASE_PAIR_MODES),
        default="warmstart",
        help=(
            "Which pair of EMO-STA phases to compare. "
            "Default: warmstart."
        ),
    )
    return parser.parse_args()


def family_display_name(family_id: str) -> str:
    return FAMILY_DISPLAY.get(family_id, family_id.replace("_", " "))


def task_display_name(task_id: str) -> str:
    return TASK_DISPLAY.get(task_id, task_id)


def apply_plot_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def summarize_family(report: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    group_count = len(report.get("groups", []))

    for group in report.get("groups", []):
        per_task = group.get("aggregate", {}).get("per_task", {})
        for task_id, task_metrics in per_task.items():
            baseline = task_metrics["baseline_score"]["mean"]
            spawn = task_metrics["spawn_score"]["mean"]
            adapt = task_metrics["adaptation_score"]["mean"]
            counts = task_metrics.get("adaptation_vs_baseline_counts", {})
            if baseline is None or spawn is None or adapt is None:
                continue

            entry = tasks.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_label": task_display_name(task_id),
                    "spawn_minus_baseline_values": [],
                    "adapt_minus_baseline_values": [],
                    "adapt_minus_spawn_values": [],
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                    "comparable": 0,
                },
            )
            entry["spawn_minus_baseline_values"].append(spawn - baseline)
            entry["adapt_minus_baseline_values"].append(adapt - baseline)
            entry["adapt_minus_spawn_values"].append(adapt - spawn)
            entry["wins"] += counts.get("wins", 0)
            entry["losses"] += counts.get("losses", 0)
            entry["ties"] += counts.get("ties", 0)
            entry["comparable"] += counts.get("comparable", 0)

    task_summaries = []
    for task_id, entry in tasks.items():
        spawn_delta = statistics.fmean(entry["spawn_minus_baseline_values"])
        adapt_delta = statistics.fmean(entry["adapt_minus_baseline_values"])
        adaptation_gain = statistics.fmean(entry["adapt_minus_spawn_values"])
        task_summaries.append(
            {
                "task_id": task_id,
                "task_label": entry["task_label"],
                "spawn_minus_baseline_mean": spawn_delta,
                "adapt_minus_baseline_mean": adapt_delta,
                "adapt_minus_spawn_mean": adaptation_gain,
                "wins": entry["wins"],
                "losses": entry["losses"],
                "ties": entry["ties"],
                "comparable": entry["comparable"],
            }
        )

    task_summaries.sort(
        key=lambda item: (
            item["adapt_minus_baseline_mean"],
            item["adapt_minus_spawn_mean"],
        ),
        reverse=True,
    )

    family_mean = (
        statistics.fmean(item["adapt_minus_baseline_mean"] for item in task_summaries)
        if task_summaries
        else 0.0
    )

    return {
        "family_id": report["family"],
        "family_label": family_display_name(report["family"]),
        "group_count": group_count,
        "family_mean_adapt_minus_baseline": family_mean,
        "tasks": task_summaries,
    }


def setting_matches_preset(
    family_id: str,
    setting: dict[str, Any],
    *,
    preset: str | None,
) -> bool:
    if preset != "emo_sta_table":
        return True

    target_budget = EMO_STA_TABLE_BUDGETS.get(family_id)
    if target_budget is None:
        return False
    if setting.get("model") not in EMO_STA_TABLE_MODELS:
        return False
    budget = (
        setting.get("shared_iterations"),
        setting.get("adaptation_iterations"),
        setting.get("baseline_iterations"),
    )
    if budget != target_budget:
        return False
    if setting.get("edit_mode") != "full":
        return False
    return True


def summarize_family_from_runs(
    report: dict[str, Any],
    *,
    preset: str | None,
    phase_pair: str,
) -> dict[str, Any]:
    mode = PHASE_PAIR_MODES[phase_pair]
    start_key = mode["start_key"]
    end_key = mode["end_key"]
    tasks: dict[str, dict[str, Any]] = {}
    matched_groups = 0
    matched_runs = 0
    matched_setting_labels: set[str] = set()

    for group in report.get("groups", []):
        setting = group.get("setting", {})
        if not setting_matches_preset(report["family"], setting, preset=preset):
            continue
        matched_groups += 1
        matched_setting_labels.add(setting.get("label") or setting.get("id") or "unknown")

        for run in group.get("runs", []):
            matched_runs += 1
            run_tasks = run.get("tasks", {})
            for task_id, task_metrics in run_tasks.items():
                baseline = task_metrics.get("baseline", {}).get("score")
                if phase_pair == "best_local":
                    start_score = load_seed_spawn_score(
                        Path(run.get("run_root", "")),
                        task_id,
                        seed_kind="best_shared",
                    )
                else:
                    start_score = task_metrics.get(start_key, {}).get("score")
                end_score = task_metrics.get(end_key, {}).get("score")
                if baseline is None or start_score is None or end_score is None:
                    continue

                entry = tasks.setdefault(
                    task_id,
                    {
                        "task_id": task_id,
                        "task_label": task_display_name(task_id),
                        "start_minus_baseline_values": [],
                        "end_minus_baseline_values": [],
                        "end_minus_start_values": [],
                        "wins": 0,
                        "losses": 0,
                        "ties": 0,
                        "comparable": 0,
                    },
                )
                entry["start_minus_baseline_values"].append(start_score - baseline)
                entry["end_minus_baseline_values"].append(end_score - baseline)
                entry["end_minus_start_values"].append(end_score - start_score)

                if phase_pair == "warmstart":
                    outcome = task_metrics.get("adaptation_vs_baseline")
                elif phase_pair == "best_local":
                    outcome = compare_scores(end_score, start_score)
                else:
                    outcome = None
                if outcome == "win":
                    entry["wins"] += 1
                    entry["comparable"] += 1
                elif outcome == "loss":
                    entry["losses"] += 1
                    entry["comparable"] += 1
                elif outcome == "tie":
                    entry["ties"] += 1
                    entry["comparable"] += 1

    task_summaries = []
    for task_id, entry in tasks.items():
        start_delta = statistics.fmean(entry["start_minus_baseline_values"])
        end_delta = statistics.fmean(entry["end_minus_baseline_values"])
        pair_gain = statistics.fmean(entry["end_minus_start_values"])
        task_summaries.append(
            {
                "task_id": task_id,
                "task_label": entry["task_label"],
                "start_minus_baseline_mean": start_delta,
                "end_minus_baseline_mean": end_delta,
                "end_minus_start_mean": pair_gain,
                "wins": entry["wins"],
                "losses": entry["losses"],
                "ties": entry["ties"],
                "comparable": entry["comparable"],
            }
        )

    task_summaries.sort(
        key=lambda item: (
            item["end_minus_baseline_mean"],
            item["end_minus_start_mean"],
        ),
        reverse=True,
    )

    family_mean = (
        statistics.fmean(item["end_minus_baseline_mean"] for item in task_summaries)
        if task_summaries
        else 0.0
    )

    return {
        "family_id": report["family"],
        "family_label": family_display_name(report["family"]),
        "group_count": matched_groups,
        "run_count": matched_runs,
        "matched_setting_labels": sorted(matched_setting_labels),
        "family_mean_end_minus_baseline": family_mean,
        "tasks": task_summaries,
    }


def load_plot_data(
    summary_path: Path,
    *,
    include_families: list[str] | None,
    exclude_families: set[str],
    preset: str | None,
    phase_pair: str,
) -> list[dict[str, Any]]:
    data = json.loads(summary_path.read_text())
    all_families = {
        family["family_id"]: family
        for family in (
            summarize_family_from_runs(report, preset=preset, phase_pair=phase_pair)
            for report in data.get("family_reports", [])
        )
        if family["tasks"]
    }

    if include_families:
        families = [
            all_families[family_id]
            for family_id in include_families
            if family_id in all_families and family_id not in exclude_families
        ]
    else:
        families = [
            family
            for family_id, family in all_families.items()
            if family_id not in exclude_families
        ]
        families.sort(key=lambda item: item["family_mean_end_minus_baseline"], reverse=True)
    return families


def compute_x_limits(families: list[dict[str, Any]]) -> tuple[float, float]:
    values = []
    for family in families:
        for task in family["tasks"]:
            values.append(task["start_minus_baseline_mean"])
            values.append(task["end_minus_baseline_mean"])
    if not values:
        return (-0.05, 0.05)

    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    padding = max(0.02, 0.08 * span)
    left = min_value - padding
    right = max_value + padding

    if left > 0:
        left = -0.01
    if right < 0:
        right = 0.01

    step = nice_tick_step(right - left)
    left = math.floor(left / step) * step
    right = math.ceil(right / step) * step
    if max_value <= PREFERRED_GAIN_X_RIGHT:
        right = PREFERRED_GAIN_X_RIGHT

    return (left, right)


def family_x_limits(
    family_id: str,
    default_limits: tuple[float, float],
) -> tuple[float, float]:
    return FAMILY_X_LIMIT_OVERRIDES.get(family_id, default_limits)


def default_include_families(*, preset: str | None, phase_pair: str) -> list[str] | None:
    if preset != "emo_sta_table":
        return None
    return list(DEFAULT_EMO_STA_TABLE_FAMILIES.get(phase_pair, [])) or None


def nice_tick_step(span: float, target_ticks: int = 5) -> float:
    if span <= 0:
        return 0.05
    raw_step = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        multiplier = 1
    elif normalized <= 2:
        multiplier = 2
    elif normalized <= 5:
        multiplier = 5
    else:
        multiplier = 10
    return multiplier * magnitude


def nice_ticks(left: float, right: float) -> list[float]:
    step = nice_tick_step(right - left)
    start = math.floor(left / step) * step
    end = math.ceil(right / step) * step
    ticks = []
    current = start
    while current <= end + 1e-12:
        rounded = round(current, 10)
        if left - 1e-12 <= rounded <= right + 1e-12:
            ticks.append(rounded)
        current += step
    return ticks


def ticks_for_family(
    family_id: str,
    *,
    left: float,
    right: float,
) -> list[float]:
    step = FAMILY_X_TICK_STEPS.get(family_id)
    if step is None:
        return nice_ticks(left, right)

    start = math.ceil((left - 1e-12) / step) * step
    end = math.floor((right + 1e-12) / step) * step
    ticks = []
    current = start
    while current <= end + 1e-12:
        ticks.append(round(current, 10))
        current += step
    return ticks


def x_to_svg(value: float, left: float, right: float, plot_left: float, plot_width: float) -> float:
    if right <= left:
        return plot_left + plot_width / 2
    return plot_left + (value - left) / (right - left) * plot_width


def escape_svg_text(text: str) -> str:
    return html.escape(text, quote=False)


def compute_svg_plot_left_pad(tasks: list[dict[str, Any]]) -> float:
    if not tasks:
        return 77.0
    max_len = max(len(task.get("task_label", "")) for task in tasks)
    return min(109.0, max(71.0, 19.0 + 4.9 * max_len))


def render_svg(
    *,
    families: list[dict[str, Any]],
    x_limits: tuple[float, float],
    output_path: Path,
    title: str,
    phase_pair: str,
) -> None:
    mode = PHASE_PAIR_MODES[phase_pair]
    n_families = len(families)
    n_cols = 3 if n_families > 3 else n_families
    n_rows = math.ceil(n_families / n_cols)

    outer_pad = 5
    title_height = 24 if title else 0
    legend_height = 24
    panel_width = 380
    panel_height = 210
    width = outer_pad * 2 + panel_width * n_cols
    height = outer_pad * 2 + title_height + panel_height * n_rows + legend_height

    plot_right_pad = 16
    plot_top_pad = 54
    plot_bottom_pad = 44
    plot_height = panel_height - plot_top_pad - plot_bottom_pad

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="EMO-STA subtask gain profile">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
    ]

    if title:
        lines.append(
            f'<text x="{width / 2:.1f}" y="{outer_pad + 4:.1f}" text-anchor="middle" '
            f'font-family="{SVG_FONT}" font-size="16" font-weight="600" fill="#222222">'
            f"{escape_svg_text(title)}</text>"
        )

    top_offset = outer_pad + title_height

    for family_idx, family in enumerate(families):
        row = family_idx // n_cols
        col = family_idx % n_cols
        panel_x = outer_pad + col * panel_width
        panel_y = top_offset + row * panel_height

        color = FAMILY_COLORS.get(family["family_id"], "#333333")
        tasks = family["tasks"]
        plot_left_pad = compute_svg_plot_left_pad(tasks)
        plot_width = panel_width - plot_left_pad - plot_right_pad
        row_gap = plot_height / max(len(tasks), 1)
        x_left, x_right = family_x_limits(family["family_id"], x_limits)
        ticks = ticks_for_family(family["family_id"], left=x_left, right=x_right)

        lines.append(
            f'<rect x="{panel_x + 0.5:.1f}" y="{panel_y + 0.5:.1f}" width="{panel_width - 1:.1f}" '
            f'height="{panel_height - 1:.1f}" fill="white" stroke="#dddddd" stroke-width="1" />'
        )

        title_x = panel_x + panel_width / 2
        title_y = panel_y + 22
        subtitle_y = panel_y + 40
        subtitle_text = f"{mode['family_mean_label']} {family['family_mean_end_minus_baseline']:+.3f}"
        lines.append(
            f'<text x="{title_x:.1f}" y="{title_y:.1f}" text-anchor="middle" '
            f'font-family="{SVG_FONT}" font-size="15" font-weight="600" fill="{color}">'
            f"{escape_svg_text(family['family_label'])}</text>"
        )
        lines.append(
            f'<text x="{title_x:.1f}" y="{subtitle_y:.1f}" text-anchor="middle" '
            f'font-family="{SVG_FONT}" font-size="11" fill="#555555">'
            f"{escape_svg_text(subtitle_text)}"
            f"</text>"
        )

        plot_left = panel_x + plot_left_pad
        plot_top = panel_y + plot_top_pad

        for tick in ticks:
            x_tick = x_to_svg(tick, x_left, x_right, plot_left, plot_width)
            is_zero = abs(tick) < 1e-12
            grid_color = "#bbbbbb" if is_zero else "#ececec"
            dash = ' stroke-dasharray="4 4"' if is_zero else ""
            lines.append(
                f'<line x1="{x_tick:.2f}" y1="{plot_top:.2f}" x2="{x_tick:.2f}" '
                f'y2="{plot_top + plot_height:.2f}" stroke="{grid_color}" stroke-width="1"{dash} />'
            )
            lines.append(
                f'<text x="{x_tick:.2f}" y="{plot_top + plot_height + 18:.2f}" text-anchor="middle" '
                f'font-family="{SVG_FONT}" font-size="10" fill="#666666">{tick:+.2f}</text>'
            )

        for task_idx, task in enumerate(tasks):
            y = plot_top + row_gap * (task_idx + 0.5)
            task_label_x = plot_left - 8
            start_delta = task["start_minus_baseline_mean"]
            end_delta = task["end_minus_baseline_mean"]
            start_x = x_to_svg(start_delta, x_left, x_right, plot_left, plot_width)
            end_x = x_to_svg(end_delta, x_left, x_right, plot_left, plot_width)

            lines.append(
                f'<text x="{task_label_x:.2f}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="{SVG_FONT}" font-size="11" fill="#333333">'
                f"{escape_svg_text(task['task_label'])}</text>"
            )
            lines.append(
                f'<line x1="{start_x:.2f}" y1="{y:.2f}" x2="{end_x:.2f}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="2.4" stroke-linecap="round" opacity="0.9" />'
            )
            lines.append(
                f'<circle cx="{start_x:.2f}" cy="{y:.2f}" r="5.2" fill="white" '
                f'stroke="#666666" stroke-width="1.6" />'
            )
            lines.append(
                f'<circle cx="{end_x:.2f}" cy="{y:.2f}" r="5.6" fill="{color}" '
                f'stroke="white" stroke-width="1.0" />'
            )

        axis_label_y = plot_top + plot_height + 34
        lines.append(
            f'<text x="{plot_left + plot_width / 2:.2f}" y="{axis_label_y:.2f}" text-anchor="middle" '
            f'font-family="{SVG_FONT}" font-size="11" fill="#333333">'
            "Improvement vs single-task baseline</text>"
        )

    legend_y = top_offset + panel_height * n_rows + 16
    legend_center_x = width / 2
    legend_gap = 222
    legend_font_size = 13

    start_x = legend_center_x - legend_gap
    end_x = legend_center_x + 76
    lines.append(
        f'<circle cx="{start_x:.2f}" cy="{legend_y:.2f}" r="5.2" fill="white" '
        f'stroke="#666666" stroke-width="1.6" />'
    )
    lines.append(
        f'<text x="{start_x + 14:.2f}" y="{legend_y + 4:.2f}" font-family="{SVG_FONT}" '
        f'font-size="{legend_font_size}" fill="#333333">{escape_svg_text(mode["start_label"])}</text>'
    )
    lines.append(
        f'<circle cx="{end_x:.2f}" cy="{legend_y:.2f}" r="5.6" fill="#333333" '
        f'stroke="white" stroke-width="1.0" />'
    )
    lines.append(
        f'<text x="{end_x + 14:.2f}" y="{legend_y + 4:.2f}" font-family="{SVG_FONT}" '
        f'font-size="{legend_font_size}" fill="#333333">{escape_svg_text(mode["end_label"])}</text>'
    )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


def make_matplotlib_figure(families: list[dict[str, Any]], *, title: str, phase_pair: str):
    if plt is None or Line2D is None:
        return None

    apply_plot_style()
    mode = PHASE_PAIR_MODES[phase_pair]

    n_families = len(families)
    n_cols = 3 if n_families > 3 else n_families
    n_rows = math.ceil(n_families / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 2.25 * n_rows + 0.4),
        sharex=False,
    )

    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    x_limits = compute_x_limits(families)

    for ax, family in zip(axes_list, families):
        color = FAMILY_COLORS.get(family["family_id"], "#333333")
        tasks = family["tasks"]
        y_positions = list(range(len(tasks)))[::-1]

        ax.axvline(0.0, color="#b8b8b8", linewidth=1.0, linestyle="--", zorder=0)

        for y_pos, task in zip(y_positions, tasks):
            start_delta = task["start_minus_baseline_mean"]
            end_delta = task["end_minus_baseline_mean"]
            ax.plot(
                [start_delta, end_delta],
                [y_pos, y_pos],
                color=color,
                linewidth=2.0,
                alpha=0.85,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter(
                start_delta,
                y_pos,
                s=46,
                facecolors="white",
                edgecolors="#666666",
                linewidths=1.4,
                zorder=3,
            )
            ax.scatter(
                end_delta,
                y_pos,
                s=52,
                facecolors=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([task["task_label"] for task in tasks])
        family_limits = family_x_limits(family["family_id"], x_limits)
        ax.set_xlim(*family_limits)
        ax.set_xticks(
            ticks_for_family(
                family["family_id"],
                left=family_limits[0],
                right=family_limits[1],
            )
        )
        ax.grid(axis="x", color="#ececec", linewidth=0.8)
        ax.set_title(
            f"{family['family_label']}  ({mode['family_mean_label']} {family['family_mean_end_minus_baseline']:+.3f})",
            color=color,
        )
        ax.tick_params(axis="y", length=0, pad=1.5)

    for ax in axes_list[n_families:]:
        ax.remove()

    for ax in axes_list[:n_families]:
        ax.set_xlabel("Improvement vs single-task baseline")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#666666",
            markeredgewidth=1.4,
            markersize=7,
            label=mode["start_label"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7,
            label=mode["end_label"],
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.01),
        columnspacing=3.2,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.985))
    return fig


def write_plot_data(
    output_path: Path,
    families: list[dict[str, Any]],
    *,
    summary_path: Path,
    include_families: list[str] | None,
    exclude_families: set[str],
    preset: str | None,
    phase_pair: str,
) -> None:
    mode = PHASE_PAIR_MODES[phase_pair]
    payload = {
        "figure": mode["figure_name"],
        "summary_json": str(summary_path),
        "family_count": len(families),
        "preset": preset,
        "phase_pair": phase_pair,
        "start_label": mode["start_label"],
        "end_label": mode["end_label"],
        "include_families": include_families,
        "exclude_families": sorted(exclude_families),
        "families": families,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    default_output_stem = Path(DEFAULT_OUTPUT_STEM)
    if Path(args.output_stem) == default_output_stem:
        output_stem = default_output_stem.with_name(
            PHASE_PAIR_MODES[args.phase_pair]["figure_name"]
        )
    else:
        output_stem = Path(args.output_stem)
    exclude_families = set(args.exclude_family or [])
    include_families = args.include_family or default_include_families(
        preset=args.preset,
        phase_pair=args.phase_pair,
    )

    families = load_plot_data(
        summary_path,
        include_families=include_families,
        exclude_families=exclude_families,
        preset=args.preset,
        phase_pair=args.phase_pair,
    )
    if not families:
        raise SystemExit(f"No family data found in {summary_path}")

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    json_path = output_stem.with_suffix(".json")

    render_svg(
        families=families,
        x_limits=compute_x_limits(families),
        output_path=svg_path,
        title=args.title,
        phase_pair=args.phase_pair,
    )
    write_plot_data(
        json_path,
        families,
        summary_path=summary_path,
        include_families=include_families,
        exclude_families=exclude_families,
        preset=args.preset,
        phase_pair=args.phase_pair,
    )

    print(f"Wrote {svg_path}")
    print(f"Wrote {json_path}")

    figure = make_matplotlib_figure(families, title=args.title, phase_pair=args.phase_pair)
    if figure is not None:
        figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")
    else:
        print("Skipped PNG/PDF output because matplotlib is not installed.")


if __name__ == "__main__":
    main()
