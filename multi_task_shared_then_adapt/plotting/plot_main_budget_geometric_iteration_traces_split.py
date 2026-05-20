#!/usr/bin/env python3
"""Create separate main-budget EMO-STA trajectory figures for geometry tasks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_OUTPUT_DIR = "multi_task_shared_then_adapt/figures/sample_trajectories"

ITERATION_RE = re.compile(r"Iteration (?P<iteration>\d+): Program .* completed")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>-?\d+(?:\.\d+)?)")
SEED_RE = re.compile(r"run_(?P<index>\d+)_seed_(?P<seed>\d+)")

COLORS = ["#1b9e77", "#d95f02", "#4c78a8", "#e15759", "#59a14f", "#af7aa1"]


@dataclass(frozen=True)
class TraceSpec:
    family: str
    title: str
    model: str
    run_root: Path
    method_name: str
    source_score_key: str
    adaptation_score_key: str
    slug: str
    adaptation_dir: str
    source_dir: str
    task_labels: dict[str, str]


TRACE_SPECS = [
    TraceSpec(
        family="circle_packing",
        title="Circle packing",
        model="Haiku-4.5",
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
            "cp_n20": "n=20",
            "cp_n22": "n=22",
            "cp_n24": "n=24",
            "cp_n26": "n=26",
        },
    ),
    TraceSpec(
        family="circle_packing_rectangle",
        title="Circle packing rectangles",
        model="Sonnet-4.5",
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
            "cp_rect_n20": "n=20",
            "cp_rect_n21": "n=21",
            "cp_rect_n22": "n=22",
            "cp_rect_n23": "n=23",
        },
    ),
    TraceSpec(
        family="heilbronn_triangle",
        title="Heilbronn triangle",
        model="Sonnet-4.6",
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
            "heil_tri_n9": "n=9",
            "heil_tri_n10": "n=10",
            "heil_tri_n11": "n=11",
            "heil_tri_n12": "n=12",
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate shared/adaptation trajectory figures for the exact 60/15/30 geometry budget."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
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


def shared_trace(run_root: Path) -> tuple[list[int], list[float]]:
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


def baseline_score(spec: TraceSpec, task_id: str) -> float | None:
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
    return max(0.0, lo - pad), min(1.02, hi + pad)


def plot_spec(spec: TraceSpec, output_dir: Path, dpi: int) -> dict[str, Any]:
    fig, (shared_ax, adapt_ax) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )

    shared_x, shared_y = shared_trace(spec.run_root)
    shared_ax.axvspan(0, max(shared_x), color="#d8f0df", alpha=0.48, linewidth=0)
    shared_ax.plot(shared_x, shared_y, color="#1b9e77", linewidth=2.3)
    shared_ax.scatter(shared_x[-1], shared_y[-1], color="#1b9e77", s=28, zorder=3)
    shared_ax.set_xlim(0, max(shared_x))
    shared_ax.set_ylim(*y_limits([shared_y]))
    shared_ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.55)
    shared_ax.set_ylabel("Normalized score")
    shared_ax.set_xlabel("Shared iteration")
    shared_ax.set_title("Shared evolution")

    task_payload: dict[str, Any] = {}
    adapt_series: list[list[float]] = []
    all_adapt_traces = {task_id: adaptation_trace(spec, task_id) for task_id in spec.task_labels}
    baselines = {task_id: baseline_score(spec, task_id) for task_id in spec.task_labels}
    max_adapt_x = max(max(xs) for xs, _ys in all_adapt_traces.values())
    adapt_ax.axvspan(0, max_adapt_x, color="#e8eef8", alpha=0.4, linewidth=0)

    baseline_label_added = False
    for color, task_id in zip(COLORS, spec.task_labels, strict=False):
        xs, ys = all_adapt_traces[task_id]
        adapt_series.append(ys)
        label = spec.task_labels[task_id]
        adapt_ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=3.2, label=label)

        base = baselines[task_id]
        if base is not None:
            adapt_series.append([base])
            adapt_ax.axhline(
                base,
                color=color,
                linewidth=1.4,
                linestyle=(0, (4, 3)),
                alpha=0.42,
                label="single-task baseline" if not baseline_label_added else None,
            )
            baseline_label_added = True

        task_payload[task_id] = {
            "label": label,
            "initial": ys[0],
            "final": ys[-1],
            "gain": ys[-1] - ys[0],
            "baseline": base,
            "x": xs,
            "y": ys,
        }

    adapt_ax.set_xlim(0, max_adapt_x)
    adapt_ax.set_ylim(*y_limits(adapt_series))
    adapt_ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.55)
    adapt_ax.set_xlabel("Adaptation iteration")
    adapt_ax.set_title("Task-specific adaptation")
    handles, labels = adapt_ax.get_legend_handles_labels()
    task_items = [(h, label) for h, label in zip(handles, labels, strict=False) if label != "single-task baseline"]
    legend_handles = [h for h, _label in task_items]
    legend_labels = [label for _h, label in task_items]
    if baseline_label_added:
        legend_handles.append(Line2D([0], [0], color="#8c8c8c", linewidth=1.4, linestyle=(0, (4, 3))))
        legend_labels.append("single-task baseline")
    adapt_ax.legend(legend_handles, legend_labels, loc="lower right", ncol=2, handlelength=1.6, columnspacing=0.9)

    fig.suptitle(
        f"{spec.title}: {spec.model}, {run_label(spec.run_root)}, {spec.method_name} "
        r"($60/15/30$)",
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
        "budget": "60 / 15 / 120",
        "shared": {"x": shared_x, "y": shared_y},
        "tasks": task_payload,
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
    manifest = {"budget": "60 / 15 / 120", "figures": []}
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
