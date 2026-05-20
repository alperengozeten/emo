#!/usr/bin/env python3
"""Plot shared-best-program subtask trajectories for representative EMO-STA runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.database import Program
from openevolve.multi_task_shared_then_adapt.spawn import (
    _invoke_reevaluate_program_for_task,
)
from openevolve.multi_task_shared_then_adapt.workflow import (
    family_task_specs,
    load_manifest,
)

DEFAULT_OUTPUT_STEM = (
    "multi_task_shared_then_adapt/figures/"
    "emo_sta_shared_best_subtask_trajectories"
)

FAMILY_CONFIG = {
    "function_minimization": {
        "manifest": "multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml",
        "results_dir": "multi_task_shared_then_adapt/results/function_minimization",
        "shared_iterations": 40,
        "adaptation_iterations": 15,
        "baseline_iterations": 25,
    },
    "signal_processing": {
        "manifest": "multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml",
        "results_dir": "multi_task_shared_then_adapt/results/signal_processing",
        "shared_iterations": 60,
        "adaptation_iterations": 10,
        "baseline_iterations": 25,
    },
    "k_module_problem_balanced": {
        "manifest": (
            "multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml"
        ),
        "results_dir": (
            "multi_task_shared_then_adapt/results/k_module_problem_balanced"
        ),
        "shared_iterations": 40,
        "adaptation_iterations": 20,
        "baseline_iterations": 30,
    },
    "sldbench_3d": {
        "manifest": "multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml",
        "results_dir": "multi_task_shared_then_adapt/results/sldbench_3d",
        "shared_iterations": 60,
        "adaptation_iterations": 10,
        "baseline_iterations": 40,
    },
    "rust_adaptive_sort": {
        "manifest": "multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml",
        "results_dir": "multi_task_shared_then_adapt/results/rust_adaptive_sort",
        "shared_iterations": 60,
        "adaptation_iterations": 10,
        "baseline_iterations": 25,
    },
}

MODEL_ORDER = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

MODEL_DISPLAY = {
    "claude-haiku-4-5": "Haiku-4.5",
    "claude-sonnet-4-5": "Sonnet-4.5",
    "claude-opus-4-5": "Opus-4.5",
    "claude-sonnet-4-6": "Sonnet-4.6",
    "claude-opus-4-6": "Opus-4.6",
}

FAMILY_DISPLAY = {
    "function_minimization": "Function minimization",
    "signal_processing": "Signal processing",
    "k_module_problem_balanced": "K-module",
    "sldbench_3d": "SLDBench-3D",
    "rust_adaptive_sort": "Rust adaptive sort",
}

TASK_DISPLAY = {
    "fm_ackley_2d": "Ackley",
    "fm_rastrigin_2d": "Rastrigin",
    "fm_rosenbrock_2d": "Rosenbrock",
    "fm_sincosxy_2d": "sincosxy",
    "sp_chirp_700_n04": "chirp",
    "sp_multifreq_600_n03": "multi-freq",
    "sp_step_800_n05": "step",
    "sp_trend_sine_500_n02": "trend+sine",
    "kmb_task_a": "task a",
    "kmb_task_b": "task b",
    "kmb_task_c": "task c",
    "kmb_task_d": "task d",
    "data_constrained_scaling_law": "data-constrained",
    "vocab_scaling_law": "vocab",
    "ras_duplicates": "duplicates",
    "ras_nearly_sorted": "nearly sorted",
    "ras_random": "random",
    "ras_reverse_sorted": "reverse sorted",
}

BEST_EVENT_RE = re.compile(
    r"New best solution found at iteration (?P<iteration>\d+): (?P<program_id>[0-9a-f-]+)"
)
RUN_INDEX_RE = re.compile(r"^run_(?P<index>\d+)_seed_(?P<seed>\d+)$")
CHECKPOINT_INDEX_RE = re.compile(r"^checkpoint_(?P<iteration>\d+)$")

COLOR_CYCLE = [
    "#1b9e77",
    "#d95f02",
    "#4c78a8",
    "#e15759",
    "#59a14f",
    "#af7aa1",
]


@dataclass
class BestEvent:
    iteration: int
    program_id: str


@dataclass
class RepresentativeRun:
    family: str
    model: str
    run_root: Path
    run_name: str
    seed: int
    run_index: int
    coverage_ratio: float
    covered_best_events: int
    total_best_events: int
    best_events: list[BestEvent]
    manifest_path: Path
    evaluation_file: Path
    initial_program: Path
    task_ids: list[str]
    shared_iterations: int
    final_checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot representative shared-best-program subtask trajectories for EMO-STA "
            "families using the settings from emo_sta_table.md."
        )
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
        "--include-family",
        action="append",
        default=None,
        help=(
            "Family id to include. Pass multiple times to filter and set panel order. "
            "Defaults to the five table families."
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
    return parser.parse_args()


def family_display_name(family_id: str) -> str:
    return FAMILY_DISPLAY.get(family_id, family_id.replace("_", " "))


def task_display_name(task_id: str) -> str:
    return TASK_DISPLAY.get(task_id, task_id)


def model_display_name(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id, model_id)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_run_index(run_name: str) -> tuple[int, int] | None:
    match = RUN_INDEX_RE.match(run_name)
    if match is None:
        return None
    return int(match.group("index")), int(match.group("seed"))


def latest_checkpoint(run_root: Path) -> Path | None:
    checkpoints_dir = run_root / "shared_run" / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for checkpoint_dir in checkpoints_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        match = CHECKPOINT_INDEX_RE.match(checkpoint_dir.name)
        if match is None:
            continue
        candidates.append((int(match.group("iteration")), checkpoint_dir))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def load_output_log(run_root: Path) -> str:
    matches = sorted((run_root / "shared_run" / "wandb").glob("run-*/files/output.log"))
    if not matches:
        raise FileNotFoundError(f"Missing shared output.log under {run_root}")
    return matches[-1].read_text(encoding="utf-8", errors="replace")


def parse_best_events(log_text: str) -> list[BestEvent]:
    events = [
        BestEvent(
            iteration=int(match.group("iteration")),
            program_id=match.group("program_id"),
        )
        for match in BEST_EVENT_RE.finditer(log_text)
    ]
    unique: dict[tuple[int, str], BestEvent] = {}
    for event in events:
        unique[(event.iteration, event.program_id)] = event
    return sorted(unique.values(), key=lambda item: (item.iteration, item.program_id))


def expected_setting_name(
    *,
    shared_iterations: int,
    adaptation_iterations: int,
    baseline_iterations: int,
    model: str,
) -> str:
    return (
        f"s{shared_iterations}-a{adaptation_iterations}-b{baseline_iterations}-{model}-full"
    )


def choose_representative_run(family: str) -> RepresentativeRun:
    family_config = FAMILY_CONFIG[family]
    manifest_path = (REPO_ROOT / family_config["manifest"]).resolve()
    manifest = load_manifest(manifest_path)
    results_dir = (REPO_ROOT / family_config["results_dir"]).resolve()
    task_ids = [task.task_id for task in family_task_specs(manifest)]

    candidates: list[RepresentativeRun] = []
    model_priority = {model: index for index, model in enumerate(MODEL_ORDER)}

    for model in MODEL_ORDER:
        setting_dir = results_dir / expected_setting_name(
            shared_iterations=family_config["shared_iterations"],
            adaptation_iterations=family_config["adaptation_iterations"],
            baseline_iterations=family_config["baseline_iterations"],
            model=model,
        )
        if not setting_dir.is_dir():
            continue

        indexed_runs: list[tuple[int, int, Path]] = []
        for run_root in setting_dir.iterdir():
            if not run_root.is_dir():
                continue
            parsed = parse_run_index(run_root.name)
            if parsed is None:
                continue
            indexed_runs.append((parsed[0], parsed[1], run_root))
        indexed_runs.sort()
        latest_five = indexed_runs[-5:]

        for run_index, seed, run_root in latest_five:
            checkpoint_dir = latest_checkpoint(run_root)
            if checkpoint_dir is None or not (checkpoint_dir / "programs").is_dir():
                continue

            log_text = load_output_log(run_root)
            best_events = parse_best_events(log_text)
            if not best_events:
                continue

            final_ids = {path.stem for path in (checkpoint_dir / "programs").glob("*.json")}
            covered = sum(1 for event in best_events if event.program_id in final_ids)
            coverage_ratio = covered / len(best_events)

            candidates.append(
                RepresentativeRun(
                    family=family,
                    model=model,
                    run_root=run_root,
                    run_name=run_root.name,
                    seed=seed,
                    run_index=run_index,
                    coverage_ratio=coverage_ratio,
                    covered_best_events=covered,
                    total_best_events=len(best_events),
                    best_events=best_events,
                    manifest_path=manifest_path,
                    evaluation_file=(manifest_path.parent / manifest.evaluation_file).resolve(),
                    initial_program=(manifest_path.parent / manifest.initial_program).resolve(),
                    task_ids=task_ids,
                    shared_iterations=int(family_config["shared_iterations"]),
                    final_checkpoint=checkpoint_dir,
                )
            )

    if not candidates:
        raise FileNotFoundError(f"No representative candidates found for {family}")

    candidates.sort(
        key=lambda item: (
            item.coverage_ratio,
            item.covered_best_events,
            -model_priority.get(item.model, len(MODEL_ORDER)),
            item.run_index,
            item.seed,
        ),
        reverse=True,
    )
    return candidates[0]


def load_programs(checkpoint_dir: Path) -> dict[str, Program]:
    programs_dir = checkpoint_dir / "programs"
    return {
        path.stem: Program.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in programs_dir.glob("*.json")
    }


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


def build_family_trajectory(run: RepresentativeRun) -> dict[str, Any]:
    programs = load_programs(run.final_checkpoint)
    initial_program = build_initial_program(run.initial_program)
    score_cache: dict[tuple[str, str], float] = {}

    def scores_for_program(program_id: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        if program_id == "INITIAL":
            program = initial_program
        else:
            program = programs[program_id]
        for task_id in run.task_ids:
            cache_key = (program_id, task_id)
            if cache_key not in score_cache:
                score_cache[cache_key] = evaluate_task_score(
                    program=program,
                    task_id=task_id,
                    family=run.family,
                    evaluation_file=run.evaluation_file,
                    initial_program=run.initial_program,
                )
            scores[task_id] = score_cache[cache_key]
        return scores

    event_points = [{"iteration": 0, "program_id": "INITIAL"}]
    event_points.extend(
        {
            "iteration": event.iteration,
            "program_id": event.program_id,
        }
        for event in run.best_events
    )
    event_points = sorted(
        event_points,
        key=lambda item: (int(item["iteration"]), str(item["program_id"])),
    )

    deduped_event_points: list[dict[str, Any]] = []
    seen_iterations: set[int] = set()
    for point in event_points:
        iteration = int(point["iteration"])
        if iteration in seen_iterations:
            continue
        seen_iterations.add(iteration)
        point["task_scores"] = scores_for_program(str(point["program_id"]))
        deduped_event_points.append(point)

    iterations = list(range(run.shared_iterations + 1))
    curves = {
        task_id: {
            "task_id": task_id,
            "task_label": task_display_name(task_id),
            "scores": [],
        }
        for task_id in run.task_ids
    }

    event_index = 0
    current_scores = deduped_event_points[0]["task_scores"]
    for iteration in iterations:
        while (
            event_index + 1 < len(deduped_event_points)
            and int(deduped_event_points[event_index + 1]["iteration"]) <= iteration
        ):
            event_index += 1
            current_scores = deduped_event_points[event_index]["task_scores"]
        for task_id in run.task_ids:
            curves[task_id]["scores"].append(float(current_scores[task_id]))

    return {
        "family": run.family,
        "family_label": family_display_name(run.family),
        "model": run.model,
        "model_label": model_display_name(run.model),
        "run_name": run.run_name,
        "run_root": str(run.run_root),
        "seed": run.seed,
        "run_index": run.run_index,
        "shared_iterations": run.shared_iterations,
        "coverage_ratio": run.coverage_ratio,
        "covered_best_events": run.covered_best_events,
        "total_best_events": run.total_best_events,
        "event_points": deduped_event_points,
        "iterations": iterations,
        "task_curves": list(curves.values()),
    }


def plot_trajectories(
    *,
    families: list[dict[str, Any]],
    title: str,
    output_stem: Path,
    dpi: int,
) -> None:
    panel_count = len(families)
    ncols = 3 if panel_count > 3 else panel_count
    nrows = math.ceil(panel_count / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.4 * nrows),
        squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]

    for ax in flat_axes[panel_count:]:
        ax.axis("off")

    for index, family in enumerate(families):
        ax = flat_axes[index]
        iterations = family["iterations"]
        event_iterations = [int(item["iteration"]) for item in family["event_points"]]
        for task_index, task_curve in enumerate(family["task_curves"]):
            color = COLOR_CYCLE[task_index % len(COLOR_CYCLE)]
            scores = task_curve["scores"]
            ax.step(
                iterations,
                scores,
                where="post",
                color=color,
                linewidth=2.0,
                label=task_curve["task_label"],
            )
            event_scores = [scores[min(iteration, len(scores) - 1)] for iteration in event_iterations]
            ax.scatter(
                event_iterations,
                event_scores,
                color=color,
                s=18,
                zorder=3,
            )

        ax.set_title(family["family_label"])
        ax.set_xlim(0, family["shared_iterations"])
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.6)
        if index % ncols == 0:
            ax.set_ylabel("Task score")
        if index >= (nrows - 1) * ncols:
            ax.set_xlabel("Shared iteration")
        ax.text(
            0.02,
            0.04,
            f"{family['model_label']}, run {family['run_index']:02d}, seed {family['seed']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#555555",
        )
        ax.legend(
            loc="lower right",
            ncol=2 if len(family["task_curves"]) > 3 else 1,
            handlelength=2.0,
            columnspacing=0.9,
        )

    if title:
        fig.suptitle(title, y=0.995, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    apply_plot_style()

    families = args.include_family or [
        "function_minimization",
        "signal_processing",
        "k_module_problem_balanced",
        "sldbench_3d",
        "rust_adaptive_sort",
    ]

    trajectories = [build_family_trajectory(choose_representative_run(family)) for family in families]

    output_stem = (REPO_ROOT / args.output_stem).resolve()
    plot_trajectories(
        families=trajectories,
        title=args.title,
        output_stem=output_stem,
        dpi=args.dpi,
    )

    payload = {
        "figure_type": "shared_best_subtask_trajectories",
        "selection": (
            "Representative table-matched run per family, chosen from the latest five runs "
            "with highest recoverable shared-best coverage, then most best-program events."
        ),
        "families": trajectories,
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
