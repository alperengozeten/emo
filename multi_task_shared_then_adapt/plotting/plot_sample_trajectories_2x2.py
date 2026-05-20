#!/usr/bin/env python3
"""Create 2x2 sample EMO-STA trajectory figures.

Each figure shows task-level trajectories on the left and averages on the right.
The top row shows shared evolution; the bottom row shows task-specific adaptation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.database import Program
from openevolve.multi_task_shared_then_adapt.spawn import (
    _invoke_reevaluate_program_for_task,
)
from openevolve.multi_task_shared_then_adapt.workflow import load_manifest


DEFAULT_OUTPUT_DIR = "multi_task_shared_then_adapt/figures/sample_trajectories"

BEST_EVENT_RE = re.compile(
    r"New best solution found at iteration (?P<iteration>\d+): (?P<program_id>[0-9a-f-]+)"
)
ITERATION_RE = re.compile(r"Iteration (?P<iteration>\d+): Program .* completed")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>-?\d+(?:\.\d+)?)")
SEED_RE = re.compile(r"run_(?P<index>\d+)_seed_(?P<seed>\d+)")

COLORS = ["#1b9e77", "#d95f02", "#4c78a8", "#e15759", "#59a14f", "#af7aa1"]
AVERAGE_GREEN = "#2f6f61"
PANEL_BLUE = "#e8eef8"
AVERAGE_FILL_GREEN = "#1b9e77"
SINGLE_TASK_LABEL = "Single-task"
SINGLE_TASK_BASELINE_LABEL = "Single-task"


@dataclass(frozen=True)
class TraceSpec:
    family: str
    title: str
    model: str
    manifest: Path
    run_root: Path
    method_name: str
    source_score_key: str
    adaptation_score_key: str
    slug: str
    adaptation_dir: str
    source_dir: str
    task_labels: dict[str, str]
    total_budget_label: str = "60 / 15 / 120"


TRACE_SPECS = [
    TraceSpec(
        family="circle_packing",
        title="Circle packing",
        model="Haiku-4.5",
        manifest=Path("multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml"),
        run_root=Path(
            "multi_task_shared_then_adapt/results/circle_packing/"
            "s60-a15-b30-claude-haiku-4-5-full/run_01_seed_42"
        ),
        method_name="STA Best-Local",
        source_score_key="best_local_spawn_score",
        adaptation_score_key="best_local_adaptation_best_score",
        slug="circle_packing_s60_a15_b30_haiku45_seed42_bestlocal",
        adaptation_dir="adaptation_best_local",
        source_dir="spawned_checkpoints_best_local",
        task_labels={
            "cp_n20": "N=20",
            "cp_n22": "N=22",
            "cp_n24": "N=24",
            "cp_n26": "N=26",
        },
    ),
    TraceSpec(
        family="circle_packing_rectangle",
        title="Circle packing rectangles",
        model="Sonnet-4.5",
        manifest=Path("multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml"),
        run_root=Path(
            "multi_task_shared_then_adapt/results/circle_packing_rectangle/"
            "s60-a15-b30-claude-sonnet-4-5-full/run_04_seed_45"
        ),
        method_name="STA Best-Shared",
        source_score_key="best_shared_spawn_score",
        adaptation_score_key="best_shared_adaptation_best_score",
        slug="circle_packing_rectangles_s60_a15_b30_sonnet45_seed45_bestshared",
        adaptation_dir="adaptation_best_shared",
        source_dir="spawned_checkpoints_best_shared",
        task_labels={
            "cp_rect_n20": "N=20",
            "cp_rect_n21": "N=21",
            "cp_rect_n22": "N=22",
            "cp_rect_n23": "N=23",
        },
    ),
    TraceSpec(
        family="heilbronn_triangle",
        title="Heilbronn triangle",
        model="Sonnet-4.6",
        manifest=Path("multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml"),
        run_root=Path(
            "multi_task_shared_then_adapt/results/heilbronn_triangle/"
            "s60-a15-b30-claude-sonnet-4-6-full/run_03_seed_44"
        ),
        method_name="STA Best-Shared",
        source_score_key="best_shared_spawn_score",
        adaptation_score_key="best_shared_adaptation_best_score",
        slug="heilbronn_triangle_s60_a15_b30_sonnet46_seed44_bestshared",
        adaptation_dir="adaptation_best_shared",
        source_dir="spawned_checkpoints_best_shared",
        task_labels={
            "heil_tri_n9": "N=9",
            "heil_tri_n10": "N=10",
            "heil_tri_n11": "N=11",
            "heil_tri_n12": "N=12",
        },
    ),
    TraceSpec(
        family="signal_processing",
        title="Signal processing",
        model="Opus-4.6",
        manifest=Path("multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml"),
        run_root=Path(
            "multi_task_shared_then_adapt/results/signal_processing/"
            "s60-a10-b25-claude-opus-4-6-full/run_01_seed_42"
        ),
        method_name="STA Best-Local",
        source_score_key="best_local_spawn_score",
        adaptation_score_key="best_local_adaptation_best_score",
        slug="signal_processing_s60_a10_b25_opus46_seed42_bestlocal",
        adaptation_dir="adaptation_best_local",
        source_dir="spawned_checkpoints_best_local",
        task_labels={
            "sp_trend_sine_500_n02": "Trend",
            "sp_multifreq_600_n03": "Multifreq",
            "sp_chirp_700_n04": "Chirp",
            "sp_step_800_n05": "Step",
        },
        total_budget_label="60 / 10 / 100",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 2x2 sample EMO-STA trajectory figures.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "font.size": 9.2,
            "axes.labelsize": 9.8,
            "axes.labelweight": "bold",
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_score(info: dict[str, Any]) -> float | None:
    metrics = info.get("best_metrics") or info.get("metrics") or {}
    for key in ("combined_score", "score", "target_ratio"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    value = info.get("best_fitness") or info.get("score")
    return float(value) if value is not None else None


def latest_output_log(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "wandb").glob("run-*/files/output.log"))
    if candidates:
        return candidates[-1]
    candidates = sorted((run_dir / "logs").glob("openevolve_*.log"))
    return candidates[-1] if candidates else None


def score_from_metric_line(line: str) -> float | None:
    metrics = {m.group("key"): float(m.group("value")) for m in METRIC_RE.finditer(line)}
    for key in ("combined_score", "score", "target_ratio"):
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def parse_scores_from_log(log_path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    pending_iteration: int | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            pending_iteration = int(iteration_match.group("iteration"))
            continue
        if pending_iteration is None or "Metrics:" not in line:
            continue
        score = score_from_metric_line(line)
        if score is not None:
            rows.append((pending_iteration, score))
        pending_iteration = None
    return sorted(rows)


def parse_best_events(log_path: Path) -> list[tuple[int, str]]:
    events = [
        (int(match.group("iteration")), match.group("program_id"))
        for match in BEST_EVENT_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace"))
    ]
    unique: dict[tuple[int, str], tuple[int, str]] = {}
    for iteration, program_id in events:
        unique[(iteration, program_id)] = (iteration, program_id)
    return sorted(unique.values())


def running_best(points: list[tuple[int, float]], initial: float | None = None) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    best = initial
    if initial is not None:
        xs.append(0)
        ys.append(initial)
    for iteration, score in sorted(points):
        best = score if best is None else max(best, score)
        xs.append(iteration)
        ys.append(best)
    return xs, ys


def shared_average_trace(run_root: Path) -> tuple[list[int], list[float]]:
    log_path = latest_output_log(run_root / "shared_run")
    if log_path is None:
        summary = load_json(run_root / "shared_run" / "summary.json")
        return [0, int(summary["total_iterations"])], [float(summary["best_fitness"])] * 2

    initial: float | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Evaluated program" in line:
            initial = score_from_metric_line(line)
            if initial is not None:
                break
    return running_best(parse_scores_from_log(log_path), initial)


def latest_shared_checkpoint(run_root: Path) -> Path:
    checkpoints = sorted((run_root / "shared_run" / "checkpoints").glob("checkpoint_*"))
    if not checkpoints:
        raise FileNotFoundError(f"Missing shared checkpoints under {run_root}")
    return checkpoints[-1]


def build_initial_program(initial_program_path: Path) -> Program:
    language = {
        ".py": "python",
        ".r": "r",
        ".rs": "rust",
    }.get(initial_program_path.suffix.lower(), initial_program_path.suffix.lstrip(".") or "python")
    return Program(
        id="INITIAL",
        code=initial_program_path.read_text(encoding="utf-8"),
        language=language,
    )


def evaluate_task_score(
    *,
    program: Program,
    task_id: str,
    family: str,
    evaluation_file: Path,
    initial_program: Path,
) -> float:
    if program.artifacts_json:
        artifacts = json.loads(program.artifacts_json)
        for task_result in artifacts.get("task_results", []):
            if task_result.get("task_id") != task_id:
                continue
            score = task_result.get("final_task_score", task_result.get("case_score"))
            if score is not None:
                return float(score)

    task_result = _invoke_reevaluate_program_for_task(
        program=program,
        task_id=task_id,
        family=family,
        evaluation_file=evaluation_file,
        default_file_suffix=initial_program.suffix or ".py",
        initial_program=initial_program,
    )
    score = task_result.get("final_task_score", task_result.get("case_score"))
    if score is None:
        metrics = task_result.get("metrics") or {}
        score = metrics.get("combined_score", metrics.get("score"))
    if score is None:
        raise ValueError(f"Could not extract score for {family}:{task_id}")
    return float(score)


def shared_task_traces(spec: TraceSpec) -> dict[str, tuple[list[int], list[float]]]:
    log_path = latest_output_log(spec.run_root / "shared_run")
    if log_path is None:
        raise FileNotFoundError(f"Missing shared output log under {spec.run_root}")

    manifest_path = (REPO_ROOT / spec.manifest).resolve()
    manifest = load_manifest(manifest_path)
    evaluation_file = (manifest_path.parent / manifest.evaluation_file).resolve()
    initial_program_path = (manifest_path.parent / manifest.initial_program).resolve()
    initial_program = build_initial_program(initial_program_path)

    checkpoint = latest_shared_checkpoint(spec.run_root)
    programs = {
        path.stem: Program.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in (checkpoint / "programs").glob("*.json")
    }

    points: list[tuple[int, str]] = [(0, "INITIAL")]
    points.extend((iteration, program_id) for iteration, program_id in parse_best_events(log_path) if program_id in programs)
    points = sorted(dict.fromkeys(points))

    score_cache: dict[tuple[str, str], float] = {}

    def score(program_id: str, task_id: str) -> float:
        key = (program_id, task_id)
        if key in score_cache:
            return score_cache[key]
        program = initial_program if program_id == "INITIAL" else programs[program_id]
        score_cache[key] = evaluate_task_score(
            program=program,
            task_id=task_id,
            family=spec.family,
            evaluation_file=evaluation_file,
            initial_program=initial_program_path,
        )
        return score_cache[key]

    event_scores = [
        (iteration, {task_id: score(program_id, task_id) for task_id in spec.task_labels})
        for iteration, program_id in points
    ]

    max_iter = max(shared_average_trace(spec.run_root)[0])
    traces: dict[str, tuple[list[int], list[float]]] = {}
    for task_id in spec.task_labels:
        xs = list(range(max_iter + 1))
        ys: list[float] = []
        event_index = 0
        current = event_scores[0][1][task_id]
        for iteration in xs:
            while event_index + 1 < len(event_scores) and event_scores[event_index + 1][0] <= iteration:
                event_index += 1
                current = event_scores[event_index][1][task_id]
            ys.append(float(current))
        traces[task_id] = (xs, ys)
    return traces


def comparison_task(spec: TraceSpec, task_id: str) -> dict[str, Any]:
    comparison = load_json(spec.run_root / "comparison_summary.json")
    return comparison["tasks"][task_id]


def source_score(spec: TraceSpec, task_id: str) -> float:
    task_summary = comparison_task(spec, task_id)
    score = task_summary.get(spec.source_score_key)
    if score is not None:
        return float(score)

    path = spec.run_root / spec.source_dir / task_id / "best_program_info.json"
    fallback = metric_score(load_json(path))
    if fallback is None:
        raise ValueError(f"Missing source score in {path}")
    return fallback


def adaptation_trace(spec: TraceSpec, task_id: str) -> tuple[list[int], list[float]]:
    initial = source_score(spec, task_id)
    log_path = latest_output_log(spec.run_root / spec.adaptation_dir / task_id)
    if log_path is None:
        summary = load_json(spec.run_root / spec.adaptation_dir / task_id / "summary.json")
        final_score = metric_score(summary)
        return [0, int(summary["total_iterations"])], [initial, final_score or initial]

    xs, ys = running_best(parse_scores_from_log(log_path), initial)
    expected = comparison_task(spec, task_id).get(spec.adaptation_score_key)
    if expected is not None and ys:
        ys[-1] = float(expected)
    return xs, ys


def baseline_score_for_run(run_root: Path, task_id: str) -> float | None:
    comparison_path = run_root / "comparison_summary.json"
    if comparison_path.exists():
        comparison = load_json(comparison_path)
        task_summary = comparison["tasks"][task_id]
        value = task_summary.get("baseline_best_score")
        if value is None:
            value = (task_summary.get("direct_baseline") or {}).get("best_score")
        if value is not None:
            return float(value)

    summary_path = run_root / "baselines" / task_id / "summary.json"
    if summary_path.exists():
        return metric_score(load_json(summary_path))
    return None


def baseline_scores_for_task(spec: TraceSpec, task_id: str) -> list[float]:
    run_roots = sorted(spec.run_root.parent.glob("run_*_seed_*"))
    scores = [baseline_score_for_run(run_root, task_id) for run_root in run_roots]
    return [float(score) for score in scores if score is not None]


def baseline_score(spec: TraceSpec, task_id: str) -> float | None:
    scores = baseline_scores_for_task(spec, task_id)
    if scores:
        return sum(scores) / len(scores)

    task_summary = comparison_task(spec, task_id)
    value = task_summary.get("baseline_best_score")
    if value is None:
        value = (task_summary.get("direct_baseline") or {}).get("best_score")
    if value is not None:
        return float(value)

    summary_path = spec.run_root / "baselines" / task_id / "summary.json"
    if summary_path.exists():
        return float(load_json(summary_path)["best_fitness"])
    return None


def dense_trace(xs: list[int], ys: list[float], end: int) -> list[float]:
    values: list[float] = []
    index = 0
    current = ys[0]
    for iteration in range(end + 1):
        while index + 1 < len(xs) and xs[index + 1] <= iteration:
            index += 1
            current = ys[index]
        values.append(float(current))
    return values


def average_trace(traces: dict[str, tuple[list[int], list[float]]]) -> tuple[list[int], list[float]]:
    end = max(max(xs) for xs, _ys in traces.values())
    dense = [dense_trace(xs, ys, end) for xs, ys in traces.values()]
    avg = [sum(values[i] for values in dense) / len(dense) for i in range(end + 1)]
    return list(range(end + 1)), avg


def run_label(run_root: Path) -> str:
    match = SEED_RE.search(run_root.name)
    if not match:
        return run_root.name
    return f"seed {match.group('seed')}"


def y_limits(series: list[list[float]]) -> tuple[float, float]:
    values = [v for row in series for v in row]
    lo = min(values)
    hi = max(values)
    pad = max((hi - lo) * 0.14, 0.01)
    return max(0.0, lo - pad), min(1.03, hi + pad)


def add_background(ax: plt.Axes, color: str, xmax: int) -> None:
    ax.axvspan(0, xmax, color=color, alpha=0.45, linewidth=0)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.55)


def fill_under_average(ax: plt.Axes, xs: list[int], ys: list[float]) -> None:
    ymin, _ymax = ax.get_ylim()
    ax.fill_between(
        xs,
        [ymin] * len(xs),
        ys,
        color=AVERAGE_FILL_GREEN,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )


def annotate_final_score(ax: plt.Axes, xs: list[int], ys: list[float], color: str) -> None:
    final_x = xs[-1]
    final_y = ys[-1]
    ax.scatter([final_x], [final_y], color=color, s=30, zorder=4)
    ax.annotate(
        f"final {final_y:.3f}",
        xy=(final_x, final_y),
        xytext=(-18, -14),
        textcoords="offset points",
        ha="right",
        va="top",
        color=color,
        fontsize=9.2,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "lw": 1.2,
            "shrinkA": 0,
            "shrinkB": 4,
        },
    )


def bold_axis_text(ax: plt.Axes) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def task_legend(ax: plt.Axes, baseline: bool) -> None:
    handles, labels = ax.get_legend_handles_labels()
    task_items = [
        (h, label)
        for h, label in zip(handles, labels, strict=False)
        if label not in {"single-task baseline", SINGLE_TASK_BASELINE_LABEL}
    ]
    if baseline:
        baseline_item = (
            Line2D([0], [0], color="#8c8c8c", linewidth=1.4, linestyle=(0, (4, 3))),
            SINGLE_TASK_BASELINE_LABEL,
        )
        task_items = task_items[:2] + [baseline_item] + task_items[2:]
    legend_handles = [h for h, _label in task_items]
    legend_labels = [label for _h, label in task_items]
    ax.legend(legend_handles, legend_labels, loc="lower right", ncol=2, handlelength=1.6, columnspacing=0.9)


def plot_spec(spec: TraceSpec, output_dir: Path, dpi: int) -> dict[str, Any]:
    shared_tasks = shared_task_traces(spec)
    shared_avg_x, shared_avg_y = shared_average_trace(spec.run_root)
    adaptation_tasks = {task_id: adaptation_trace(spec, task_id) for task_id in spec.task_labels}
    adaptation_avg_x, adaptation_avg_y = average_trace(adaptation_tasks)
    baseline_run_scores = {task_id: baseline_scores_for_task(spec, task_id) for task_id in spec.task_labels}
    baselines = {
        task_id: (sum(scores) / len(scores) if scores else baseline_score(spec, task_id))
        for task_id, scores in baseline_run_scores.items()
    }
    baseline_values = [value for value in baselines.values() if value is not None]
    baseline_avg = sum(baseline_values) / len(baseline_values) if baseline_values else None

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.4, 6.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    ax_shared_tasks, ax_shared_avg = axes[0]
    ax_adapt_tasks, ax_adapt_avg = axes[1]

    for color, task_id in zip(COLORS, spec.task_labels, strict=False):
        label = spec.task_labels[task_id]
        xs, ys = shared_tasks[task_id]
        ax_shared_tasks.plot(xs, ys, color=color, linewidth=1.8, label=label)
    add_background(ax_shared_tasks, PANEL_BLUE, max(shared_avg_x))
    ax_shared_tasks.set_xlim(0, max(shared_avg_x))
    ax_shared_tasks.set_ylim(*y_limits([ys for _xs, ys in shared_tasks.values()]))
    ax_shared_tasks.set_title("Shared Evolution Task Breakdown")
    ax_shared_tasks.set_ylabel("Normalized score")
    task_legend(ax_shared_tasks, baseline=False)

    add_background(ax_shared_avg, PANEL_BLUE, max(shared_avg_x))
    ax_shared_avg.set_xlim(0, max(shared_avg_x))
    ax_shared_avg.set_ylim(*y_limits([shared_avg_y]))
    fill_under_average(ax_shared_avg, shared_avg_x, shared_avg_y)
    ax_shared_avg.plot(shared_avg_x, shared_avg_y, color=AVERAGE_GREEN, linewidth=2.6, zorder=3)
    ax_shared_avg.set_title("Shared Evolution Average")

    adapt_series = []
    for color, task_id in zip(COLORS, spec.task_labels, strict=False):
        label = spec.task_labels[task_id]
        xs, ys = adaptation_tasks[task_id]
        adapt_series.append(ys)
        ax_adapt_tasks.plot(xs, ys, color=color, linewidth=1.9, marker="o", markersize=3.0, label=label)
        base = baselines[task_id]
        if base is not None:
            adapt_series.append([base])
            ax_adapt_tasks.axhline(base, color=color, linewidth=1.35, linestyle=(0, (4, 3)), alpha=0.42)
    add_background(ax_adapt_tasks, PANEL_BLUE, max(adaptation_avg_x))
    ax_adapt_tasks.set_xlim(0, max(adaptation_avg_x))
    ax_adapt_tasks.set_ylim(*y_limits(adapt_series))
    ax_adapt_tasks.set_title("Task-Specific Adaptation Task Breakdown")
    ax_adapt_tasks.set_xlabel("Iteration")
    ax_adapt_tasks.set_ylabel("Normalized score")
    task_legend(ax_adapt_tasks, baseline=bool(baseline_values))

    avg_series = [adaptation_avg_y]
    if baseline_avg is not None:
        avg_series.append([baseline_avg])
        ax_adapt_avg.axhline(
            baseline_avg,
            color="#8c8c8c",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            label=SINGLE_TASK_LABEL,
        )
    add_background(ax_adapt_avg, PANEL_BLUE, max(adaptation_avg_x))
    ax_adapt_avg.set_xlim(0, max(adaptation_avg_x))
    ax_adapt_avg.set_ylim(*y_limits(avg_series))
    fill_under_average(ax_adapt_avg, adaptation_avg_x, adaptation_avg_y)
    ax_adapt_avg.plot(
        adaptation_avg_x,
        adaptation_avg_y,
        color=AVERAGE_GREEN,
        linewidth=2.6,
        marker="o",
        markersize=3.0,
        label="STA",
        zorder=3,
    )
    ax_adapt_avg.set_title("Task-Specific Adaptation Average")
    ax_adapt_avg.set_xlabel("Iteration")
    ax_adapt_avg.legend(loc="lower right", bbox_to_anchor=(1.0, 0.14), handlelength=1.8)

    for ax in axes.ravel():
        bold_axis_text(ax)

    fig.suptitle(
        f"{spec.title}: {spec.model}, {run_label(spec.run_root)}, {spec.method_name} "
        f"({spec.total_budget_label})",
        fontsize=11.0,
        fontweight="bold",
    )

    output_stem = output_dir / spec.slug
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")

    payload = {
        "family": spec.family,
        "title": spec.title,
        "model": spec.model,
        "run_root": str(spec.run_root),
        "method": spec.method_name,
        "budget": spec.total_budget_label,
        "shared_task_traces": {
            task_id: {"label": spec.task_labels[task_id], "x": xs, "y": ys}
            for task_id, (xs, ys) in shared_tasks.items()
        },
        "shared_average": {"x": shared_avg_x, "y": shared_avg_y},
        "adaptation_task_traces": {
            task_id: {
                "label": spec.task_labels[task_id],
                "baseline": baselines[task_id],
                "baseline_run_scores": baseline_run_scores[task_id],
                "x": xs,
                "y": ys,
                "initial": ys[0],
                "final": ys[-1],
                "gain": ys[-1] - ys[0],
            }
            for task_id, (xs, ys) in adaptation_tasks.items()
        },
        "adaptation_average": {"x": adaptation_avg_x, "y": adaptation_avg_y, "baseline": baseline_avg},
    }
    output_stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)

    return {
        "slug": spec.slug,
        "files": {
            "png": str(output_stem.with_suffix(".png")),
            "pdf": str(output_stem.with_suffix(".pdf")),
            "svg": str(output_stem.with_suffix(".svg")),
            "json": str(output_stem.with_suffix(".json")),
        },
        "payload": payload,
    }


def main() -> None:
    args = parse_args()
    apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"budget": "mixed", "layout": "2x2", "figures": []}
    for spec in TRACE_SPECS:
        result = plot_spec(spec, output_dir, args.dpi)
        manifest["figures"].append(result)
        for file_path in result["files"].values():
            print(file_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
