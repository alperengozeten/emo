#!/usr/bin/env python3
"""Summarize EMO-STA results from local run directories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from numbers import Real
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, Iterable, List, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.workflow import (  # noqa: E402
    family_task_specs,
    load_manifest,
    phase_checkpoint_status,
)


CIRCLE_PACKING_MANIFEST = (
    "multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml"
)
DEFAULT_MANIFEST = CIRCLE_PACKING_MANIFEST
CIRCLE_PACKING_RECTANGLE_MANIFEST = (
    "multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml"
)
K_MODULE_BALANCED_MANIFEST = (
    "multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml"
)
FUNCTION_MINIMIZATION_MANIFEST = (
    "multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml"
)
HEILBRONN_TRIANGLE_MANIFEST = (
    "multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml"
)
SIGNAL_PROCESSING_MANIFEST = (
    "multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml"
)
SLDBENCH_3D_MANIFEST = (
    "multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml"
)
RUST_ADAPTIVE_SORT_MANIFEST = (
    "multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml"
)
CIRCLE_PACKING_RESULTS_DIR = "multi_task_shared_then_adapt/results/circle_packing"
DEFAULT_RESULTS_DIR = CIRCLE_PACKING_RESULTS_DIR
CIRCLE_PACKING_RECTANGLE_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/circle_packing_rectangle"
)
K_MODULE_BALANCED_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/k_module_problem_balanced"
)
FUNCTION_MINIMIZATION_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/function_minimization"
)
HEILBRONN_TRIANGLE_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/heilbronn_triangle"
)
SIGNAL_PROCESSING_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/signal_processing"
)
SLDBENCH_3D_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/sldbench_3d"
)
RUST_ADAPTIVE_SORT_RESULTS_DIR = (
    "multi_task_shared_then_adapt/results/rust_adaptive_sort"
)
DEFAULT_MARKDOWN_OUT = "multi_task_shared_then_adapt/docs/emo_sta_results_summary.md"
TOLERANCE = 1.0e-12
MAIN_PAPER_REPORT_TARGETS = (
    (CIRCLE_PACKING_MANIFEST, CIRCLE_PACKING_RESULTS_DIR),
    (CIRCLE_PACKING_RECTANGLE_MANIFEST, CIRCLE_PACKING_RECTANGLE_RESULTS_DIR),
    (K_MODULE_BALANCED_MANIFEST, K_MODULE_BALANCED_RESULTS_DIR),
    (FUNCTION_MINIMIZATION_MANIFEST, FUNCTION_MINIMIZATION_RESULTS_DIR),
    (HEILBRONN_TRIANGLE_MANIFEST, HEILBRONN_TRIANGLE_RESULTS_DIR),
    (SIGNAL_PROCESSING_MANIFEST, SIGNAL_PROCESSING_RESULTS_DIR),
    (SLDBENCH_3D_MANIFEST, SLDBENCH_3D_RESULTS_DIR),
    (RUST_ADAPTIVE_SORT_MANIFEST, RUST_ADAPTIVE_SORT_RESULTS_DIR),
)


def repo_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parents[2]


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local EMO-STA run directories into a markdown report that handles "
            "single runs today and repeat aggregates later. "
            "With no explicit --manifest/--results-dir arguments, the main-paper "
            "EMO-STA families are summarized together when they have local results. "
            f"To focus on one family, pass --manifest {CIRCLE_PACKING_MANIFEST} "
            f"--results-dir {CIRCLE_PACKING_RESULTS_DIR}, --manifest "
            f"{CIRCLE_PACKING_RECTANGLE_MANIFEST} --results-dir "
            f"{CIRCLE_PACKING_RECTANGLE_RESULTS_DIR}, --manifest "
            f"{K_MODULE_BALANCED_MANIFEST} --results-dir {K_MODULE_BALANCED_RESULTS_DIR}, "
            f"or --manifest "
            f"{FUNCTION_MINIMIZATION_MANIFEST} --results-dir "
            f"{FUNCTION_MINIMIZATION_RESULTS_DIR}, or --manifest "
            f"{HEILBRONN_TRIANGLE_MANIFEST} --results-dir "
            f"{HEILBRONN_TRIANGLE_RESULTS_DIR}, or --manifest "
            f"{SIGNAL_PROCESSING_MANIFEST} --results-dir "
            f"{SIGNAL_PROCESSING_RESULTS_DIR}, or --manifest "
            f"{SLDBENCH_3D_MANIFEST} --results-dir "
            f"{SLDBENCH_3D_RESULTS_DIR}, or --manifest "
            f"{RUST_ADAPTIVE_SORT_MANIFEST} --results-dir "
            f"{RUST_ADAPTIVE_SORT_RESULTS_DIR}."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help=(
            "EMO-STA manifest path. May be passed multiple times to build a combined report. "
            "If omitted together with --results-dir, main-paper families with local results are included. "
            f"Unit-square circle packing: {CIRCLE_PACKING_MANIFEST}. "
            f"Rectangle circle packing: {CIRCLE_PACKING_RECTANGLE_MANIFEST}. "
            f"K-module: {K_MODULE_BALANCED_MANIFEST}. "
            f"Function minimization: {FUNCTION_MINIMIZATION_MANIFEST}. "
            f"Heilbronn triangle: {HEILBRONN_TRIANGLE_MANIFEST}. "
            f"Signal processing: {SIGNAL_PROCESSING_MANIFEST}. "
            f"SLDBench 3D: {SLDBENCH_3D_MANIFEST}. "
            f"Rust adaptive sort: {RUST_ADAPTIVE_SORT_MANIFEST}."
        ),
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        default=None,
        help=(
            "Directory containing EMO-STA run directories. May be passed multiple times and "
            "must line up with repeated --manifest values. If omitted for an explicit "
            "manifest, that manifest's output_root is used. "
            f"Unit-square circle packing: {CIRCLE_PACKING_RESULTS_DIR}. "
            f"Rectangle circle packing: {CIRCLE_PACKING_RECTANGLE_RESULTS_DIR}. "
            f"K-module: {K_MODULE_BALANCED_RESULTS_DIR}. "
            f"Function minimization: {FUNCTION_MINIMIZATION_RESULTS_DIR}. "
            f"Heilbronn triangle: {HEILBRONN_TRIANGLE_RESULTS_DIR}. "
            f"Signal processing: {SIGNAL_PROCESSING_RESULTS_DIR}. "
            f"SLDBench 3D: {SLDBENCH_3D_RESULTS_DIR}. "
            f"Rust adaptive sort: {RUST_ADAPTIVE_SORT_RESULTS_DIR}."
        ),
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=None,
        help=(
            "Explicit run root to include. May be passed multiple times. "
            "If omitted, all run directories under --results-dir are scanned."
        ),
    )
    parser.add_argument(
        "--latest-per-setting",
        type=int,
        default=5,
        help=(
            "When scanning a results directory, keep only the latest N runs for each "
            "setting fingerprint. Default: 5."
        ),
    )
    parser.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Ignore --latest-per-setting and include every discovered run.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help=(
            "Optional W&B entity to use when building run URLs. "
            "Useful if configs omitted wandb.entity."
        ),
    )
    parser.add_argument(
        "--markdown-out",
        default=DEFAULT_MARKDOWN_OUT,
        help="Optional path to write the markdown report.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write a machine-readable JSON summary.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if no run directories are found.",
    )
    return parser.parse_args()


def read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def read_yaml_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def extract_scalar(value: Any) -> float | None:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("combined_score", "score", "mean", "max", "min", "last"):
            nested = value.get(key)
            if isinstance(nested, Real) and not isinstance(nested, bool):
                return float(nested)
    return None


def score_from_metrics(metrics: Dict[str, Any] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    combined_score = extract_scalar(metrics.get("combined_score"))
    if combined_score is not None:
        return combined_score
    return extract_scalar(metrics.get("score"))


def mean_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return statistics.fmean(valid)


def summarize(values: Sequence[float | None]) -> Dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "values": [],
        }
    return {
        "count": len(valid),
        "mean": statistics.fmean(valid),
        "std": statistics.stdev(valid) if len(valid) > 1 else 0.0,
        "min": min(valid),
        "max": max(valid),
        "values": valid,
    }


def compare_scores(lhs: float | None, rhs: float | None) -> str | None:
    if lhs is None or rhs is None:
        return None
    delta = lhs - rhs
    if delta > TOLERANCE:
        return "win"
    if delta < -TOLERANCE:
        return "loss"
    return "tie"


def comparison_counts(outcomes: Iterable[str | None]) -> Dict[str, int]:
    counts = {"wins": 0, "ties": 0, "losses": 0, "comparable": 0}
    for outcome in outcomes:
        if outcome is None:
            continue
        counts["comparable"] += 1
        if outcome == "win":
            counts["wins"] += 1
        elif outcome == "loss":
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def summary_count(*summaries: Dict[str, Any]) -> int:
    counts = [int(summary.get("count", 0) or 0) for summary in summaries if isinstance(summary, dict)]
    return max(counts) if counts else 0


def delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def format_counts(counts: Dict[str, int]) -> str:
    comparable = counts.get("comparable", 0)
    if comparable == 0:
        return "N/A"
    return f"{counts['wins']}/{counts['ties']}/{counts['losses']}"


def format_float(value: float | None, *, decimals: int = 4, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    prefix = "+" if signed else ""
    return f"{value:{prefix}.{decimals}f}"


def format_stat(summary: Dict[str, Any], *, decimals: int = 4, signed: bool = False) -> str:
    count = int(summary.get("count", 0) or 0)
    mean = summary.get("mean")
    std = summary.get("std")
    if count == 0 or mean is None:
        return "N/A"
    if count == 1 or std is None:
        return format_float(mean, decimals=decimals, signed=signed)
    prefix = "+" if signed else ""
    return f"{mean:{prefix}.{decimals}f} ± {std:.{decimals}f}"


def display_repo_relative(path: Path, repo_root: Path) -> str:
    lexical_path = path if path.is_absolute() else Path.cwd() / path
    lexical_repo_root = repo_root if repo_root.is_absolute() else Path.cwd() / repo_root
    try:
        return str(lexical_path.relative_to(lexical_repo_root))
    except ValueError:
        pass

    resolved_repo_root = repo_root.resolve()
    try:
        return str(path.resolve().relative_to(resolved_repo_root))
    except ValueError:
        return str(path.resolve())


def serialize_report_path(path: Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    return display_repo_relative(path, repo_root)


def render_report_path(path_value: str | None, repo_root: Path) -> str:
    if not path_value:
        return "N/A"
    path = Path(path_value)
    if path.is_absolute():
        return display_repo_relative(path, repo_root)
    return path_value


def parse_run_time(run_name: str, run_root: Path) -> datetime:
    match = re.search(r"(\d{8}_\d{6})", run_name)
    if match is not None:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(run_root.stat().st_mtime)


def normalize_config_for_setting(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = json.loads(json.dumps(raw or {}))
    normalized.pop("wandb", None)
    normalized.pop("max_iterations", None)
    normalized.pop("checkpoint_interval", None)
    normalized.pop("random_seed", None)
    database = normalized.get("database")
    if isinstance(database, dict):
        database.pop("random_seed", None)
    llm = normalized.get("llm")
    if isinstance(llm, dict):
        normalized["llm"] = normalize_llm_config_for_setting(llm)
    # EMO-STA repeat trials intentionally vary seed, but missing diff_based_evolution
    # still means the runtime default (False), so canonicalize that here.
    normalized["diff_based_evolution"] = bool(normalized.get("diff_based_evolution", False))
    return normalized


def normalize_llm_config_for_setting(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Drop run-local LLM transport details that should not split repeat settings."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            normalized_value = {}
            for key, item in value.items():
                if key in {"api_base", "api_key"}:
                    continue
                normalized_value[key] = _normalize(item)
            return normalized_value
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(raw)


def extract_primary_model(raw: Dict[str, Any] | None) -> str | None:
    if not isinstance(raw, dict):
        return None
    llm = raw.get("llm")
    if isinstance(llm, dict):
        primary_model = llm.get("primary_model")
        if primary_model:
            return str(primary_model)
        models = llm.get("models")
        if isinstance(models, list) and models:
            first_model = models[0]
            if isinstance(first_model, dict) and first_model.get("name"):
                return str(first_model["name"])
    return None


def extract_edit_mode(raw: Dict[str, Any] | None) -> str:
    if not isinstance(raw, dict):
        return "full"
    return "diff" if bool(raw.get("diff_based_evolution", False)) else "full"


def unique_in_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def collect_iteration_budgets(configs_root: Path, prefix: str) -> List[int]:
    budgets: List[int] = []
    for path in sorted(configs_root.glob(f"{prefix}_*.yaml")):
        raw = read_yaml_if_exists(path) or {}
        budget = extract_scalar(raw.get("max_iterations"))
        if budget is not None:
            budgets.append(int(round(budget)))
    return unique_in_order(budgets)


def collect_iteration_budgets_for_prefixes(
    configs_root: Path,
    *prefixes: str,
) -> List[int]:
    budgets: List[int] = []
    for prefix in prefixes:
        budgets.extend(collect_iteration_budgets(configs_root, prefix))
    return unique_in_order(budgets)


def legacy_branch_key(branch: str) -> str:
    return f"{branch}_{'seed'}_adaptation"


def legacy_branch_dir(branch: str) -> str:
    return f"adaptation_{branch}_{'seed'}"


def first_mapping(payload: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def existing_child_dir(parent: Path, canonical_name: str, legacy_name: str) -> Path:
    canonical = parent / canonical_name
    if canonical.is_dir():
        return canonical
    legacy = parent / legacy_name
    if legacy.is_dir():
        return legacy
    return canonical


def format_setting_label(
    setting: Dict[str, Any],
    *,
    best_shared_iterations: Any | None = None,
    best_local_iterations: Any | None = None,
) -> str:
    parts = [
        f"model={setting.get('model') or 'unknown'}",
        f"shared={adaptation_label(setting.get('shared_iterations'))}",
        f"adapt={budget_label(setting.get('adaptation_iterations'))}",
    ]
    if best_shared_iterations is not None:
        parts.append(f"best-shared={budget_label(best_shared_iterations)}")
    if best_local_iterations is not None:
        parts.append(f"best-local={budget_label(best_local_iterations)}")
    parts.extend(
        [
            f"baseline={budget_label(setting.get('baseline_iterations'))}",
            f"edit={setting.get('edit_mode') or 'full'}",
        ]
    )
    return " | ".join(parts)


def load_shared_result(
    shared_output: Path,
    expected_iterations: int | None,
    *,
    repo_root: Path,
) -> Dict[str, Any]:
    summary = read_json_if_exists(shared_output / "summary.json")
    best_info = read_json_if_exists(shared_output / "best" / "best_program_info.json")
    available = shared_output.is_dir() and (summary is not None or best_info is not None)
    metrics = None
    score = None
    total_iterations = None
    best_iteration = None
    wall_clock_time_sec = None
    if summary is not None:
        metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else None
        score = extract_scalar(summary.get("best_fitness"))
        if score is None:
            score = score_from_metrics(metrics)
        total_iterations = extract_scalar(summary.get("total_iterations"))
        best_iteration = extract_scalar(summary.get("best_iteration"))
        wall_clock_time_sec = extract_scalar(summary.get("wall_clock_time_sec"))
    if score is None and best_info is not None:
        metrics = metrics or best_info.get("metrics")
        score = score_from_metrics(metrics)
        total_iterations = (
            total_iterations
            if total_iterations is not None
            else extract_scalar(best_info.get("current_iteration"))
        )
        best_iteration = (
            best_iteration
            if best_iteration is not None
            else extract_scalar(best_info.get("iteration"))
        )
    complete = False
    if expected_iterations is not None and shared_output.is_dir():
        complete = phase_checkpoint_status(
            shared_output,
            expected_iterations,
            require_best_info=False,
        )[0]
    if (
        not complete
        and expected_iterations is not None
        and total_iterations is not None
        and total_iterations >= expected_iterations
    ):
        complete = True
    return {
        "available": available,
        "complete": complete,
        "score": score,
        "metrics": metrics,
        "total_iterations": int(round(total_iterations)) if total_iterations is not None else None,
        "best_iteration": int(round(best_iteration)) if best_iteration is not None else None,
        "wall_clock_time_sec": wall_clock_time_sec,
        "successful_task_count": extract_scalar((metrics or {}).get("successful_task_count")),
        "failed_task_count": extract_scalar((metrics or {}).get("failed_task_count")),
        "summary_path": serialize_report_path(
            shared_output / "summary.json" if summary is not None else None,
            repo_root,
        ),
    }


def load_spawn_result(spawn_dir: Path, *, repo_root: Path) -> Dict[str, Any]:
    best_info = read_json_if_exists(spawn_dir / "best_program_info.json")
    spawn_metadata = read_json_if_exists(spawn_dir / "spawn_metadata.json")
    metrics = best_info.get("metrics") if isinstance(best_info, dict) else None
    score = score_from_metrics(metrics)
    available = spawn_dir.is_dir() and (best_info is not None or spawn_metadata is not None)
    complete = best_info is not None and spawn_metadata is not None
    return {
        "available": available,
        "complete": complete,
        "score": score,
        "metrics": metrics,
        "total_iterations": extract_scalar((spawn_metadata or {}).get("last_iteration")),
        "best_iteration": extract_scalar((best_info or {}).get("iteration")),
        "summary_path": serialize_report_path(
            spawn_dir / "best_program_info.json" if best_info is not None else None,
            repo_root,
        ),
    }


def load_task_phase_result(
    output_dir: Path,
    expected_iterations: int | None,
    *,
    repo_root: Path,
) -> Dict[str, Any]:
    summary = read_json_if_exists(output_dir / "summary.json")
    best_info = read_json_if_exists(output_dir / "best" / "best_program_info.json")
    metrics = None
    score = None
    total_iterations = None
    best_iteration = None
    wall_clock_time_sec = None
    if summary is not None:
        metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else None
        score = extract_scalar(summary.get("best_fitness"))
        if score is None:
            score = score_from_metrics(metrics)
        total_iterations = extract_scalar(summary.get("total_iterations"))
        best_iteration = extract_scalar(summary.get("best_iteration"))
        wall_clock_time_sec = extract_scalar(summary.get("wall_clock_time_sec"))
    if score is None and best_info is not None:
        metrics = metrics or best_info.get("metrics")
        score = score_from_metrics(metrics)
        total_iterations = (
            total_iterations
            if total_iterations is not None
            else extract_scalar(best_info.get("current_iteration"))
        )
        best_iteration = (
            best_iteration
            if best_iteration is not None
            else extract_scalar(best_info.get("iteration"))
        )

    available = output_dir.is_dir() and (summary is not None or best_info is not None)
    complete = False
    if expected_iterations is not None and output_dir.is_dir():
        complete = phase_checkpoint_status(
            output_dir,
            expected_iterations,
            require_best_info=True,
        )[0]
    if (
        not complete
        and expected_iterations is not None
        and total_iterations is not None
        and total_iterations >= expected_iterations
        and score is not None
    ):
        complete = True

    return {
        "available": available,
        "complete": complete,
        "score": score,
        "metrics": metrics,
        "total_iterations": int(round(total_iterations)) if total_iterations is not None else None,
        "best_iteration": int(round(best_iteration)) if best_iteration is not None else None,
        "wall_clock_time_sec": wall_clock_time_sec,
        "summary_path": serialize_report_path(
            output_dir / "summary.json" if summary is not None else None,
            repo_root,
        ),
    }


def _phase_score_from_task_summary(
    task_summary: Dict[str, Any] | None,
    *,
    flat_key: str | Sequence[str],
    nested_key: str | Sequence[str],
) -> float | None:
    if not isinstance(task_summary, dict):
        return None
    flat_keys = (flat_key,) if isinstance(flat_key, str) else tuple(flat_key)
    nested_keys = (nested_key,) if isinstance(nested_key, str) else tuple(nested_key)
    for key in flat_keys:
        direct_score = extract_scalar(task_summary.get(key))
        if direct_score is not None:
            return direct_score
    for key in nested_keys:
        nested = task_summary.get(key)
        if isinstance(nested, dict):
            nested_score = extract_scalar(nested.get("best_score"))
            if nested_score is not None:
                return nested_score
    return None


def progress_label(run: Dict[str, Any], phase: str, total_tasks: int) -> str:
    if phase == "shared":
        shared = run["shared"]
        if shared["complete"]:
            return "complete"
        if shared["available"]:
            return "partial"
        return "missing"

    task_entries = run["tasks"].values()
    available = sum(1 for task in task_entries if task[phase]["available"])
    complete = sum(1 for task in task_entries if task[phase]["complete"])
    if available == 0:
        return f"0/{total_tasks}"
    if complete == total_tasks:
        return f"{total_tasks}/{total_tasks}"
    if complete == available:
        return f"{complete}/{total_tasks}"
    return f"{complete}/{total_tasks} ({available} present)"


def _task_phase_health_reasons(phase_name: str, task_id: str, result: Dict[str, Any]) -> List[str]:
    if not result.get("available"):
        return []
    total_iterations = result.get("total_iterations")
    if total_iterations == 0:
        return [f"{phase_name} {task_id} reported zero iterations"]
    if not result.get("complete"):
        return [f"{phase_name} {task_id} incomplete"]
    return []


def classify_run_health(
    *,
    shared: Dict[str, Any],
    tasks: Dict[str, Any],
    comparison_summary_available: bool,
) -> Dict[str, Any]:
    reasons: List[str] = []
    if shared.get("available"):
        total_iterations = shared.get("total_iterations")
        if total_iterations == 0:
            reasons.append("shared phase reported zero iterations")
        elif not shared.get("complete"):
            reasons.append("shared phase incomplete")

    observed_any_phase = bool(shared.get("available"))
    for task_id, task in tasks.items():
        spawn = task["spawn"]
        if spawn.get("available"):
            observed_any_phase = True
            if not spawn.get("complete"):
                reasons.append(f"spawn {task_id} incomplete")
        for phase_name in (
            "adaptation",
            "best_shared_adaptation",
            "best_local_adaptation",
            "baseline",
        ):
            result = task[phase_name]
            if result.get("available"):
                observed_any_phase = True
            reasons.extend(_task_phase_health_reasons(phase_name, task_id, result))

    if observed_any_phase and not comparison_summary_available:
        reasons.append("missing comparison_summary.json")

    reasons = unique_in_order(reasons)
    has_zero_iteration = any("zero iterations" in reason for reason in reasons)
    has_incomplete = any(
        "incomplete" in reason or "missing comparison_summary.json" == reason
        for reason in reasons
    )
    status = "failed" if has_zero_iteration else "incomplete" if has_incomplete else "ok"
    return {
        "status": status,
        "failed": status == "failed",
        "incomplete": status == "incomplete",
        "ok": status == "ok",
        "reasons": reasons,
    }


def build_wandb_url(wandb: Dict[str, Any]) -> str | None:
    entity = wandb.get("entity")
    project = wandb.get("project")
    run_id = wandb.get("run_id")
    if not entity or not project or not run_id:
        return None
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def run_has_observable_results(run: Dict[str, Any]) -> bool:
    if run["shared"]["available"]:
        return True
    if run["comparison_summary_available"]:
        return True
    for task in run["tasks"].values():
        if (
            task["spawn"]["available"]
            or task["adaptation"]["available"]
            or task["best_shared_adaptation"]["available"]
            or task["best_local_adaptation"]["available"]
            or task["baseline"]["available"]
        ):
            return True
    return False


def load_run_report(
    run_root: Path,
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_family: str,
    task_specs: Sequence[Any],
    wandb_entity_override: str | None,
) -> Dict[str, Any]:
    configs_root = run_root / "configs"
    comparison_summary = read_json_if_exists(run_root / "comparison_summary.json") or {}
    shared_config = read_yaml_if_exists(configs_root / "shared_config.yaml") or {}
    shared_iterations = extract_scalar(shared_config.get("max_iterations"))
    adaptation_budgets = collect_iteration_budgets(configs_root, "adaptation")
    best_shared_budgets = collect_iteration_budgets_for_prefixes(
        configs_root,
        "best_shared_adaptation",
        legacy_branch_key("best_shared"),
    )
    best_local_budgets = collect_iteration_budgets_for_prefixes(
        configs_root,
        "best_local_adaptation",
        legacy_branch_key("best_local"),
    )
    baseline_budgets = collect_iteration_budgets(configs_root, "baseline")
    adaptation_iterations = adaptation_budgets[0] if len(adaptation_budgets) == 1 else adaptation_budgets
    best_shared_iterations: int | List[int] | None = None
    best_local_iterations: int | List[int] | None = None
    baseline_iterations = baseline_budgets[0] if len(baseline_budgets) == 1 else baseline_budgets
    best_shared_summary = first_mapping(
        comparison_summary,
        "best_shared_adaptation",
        legacy_branch_key("best_shared"),
    )
    best_local_summary = first_mapping(
        comparison_summary,
        "best_local_adaptation",
        legacy_branch_key("best_local"),
    )
    best_shared_root = existing_child_dir(
        run_root,
        "adaptation_best_shared",
        legacy_branch_dir("best_shared"),
    )
    best_local_root = existing_child_dir(
        run_root,
        "adaptation_best_local",
        legacy_branch_dir("best_local"),
    )
    best_shared_branch_present = best_shared_root.is_dir()
    best_local_branch_present = best_local_root.is_dir()
    best_shared_summary_iterations = extract_scalar(best_shared_summary.get("iterations"))
    if best_shared_summary_iterations is not None:
        best_shared_iterations = int(round(best_shared_summary_iterations))
    elif len(best_shared_budgets) == 1:
        best_shared_iterations = best_shared_budgets[0]
    elif best_shared_budgets:
        best_shared_iterations = best_shared_budgets
    best_local_summary_iterations = extract_scalar(best_local_summary.get("iterations"))
    if best_local_summary_iterations is not None:
        best_local_iterations = int(round(best_local_summary_iterations))
    elif len(best_local_budgets) == 1:
        best_local_iterations = best_local_budgets[0]
    elif best_local_budgets:
        best_local_iterations = best_local_budgets
    comparison_tasks = (
        comparison_summary.get("tasks")
        if isinstance(comparison_summary.get("tasks"), dict)
        else {}
    )

    normalized_config = normalize_config_for_setting(shared_config)
    setting_payload = {
        "manifest_path": display_repo_relative(manifest_path, repo_root),
        "family": manifest_family,
        "shared_config": normalized_config,
        "shared_iterations": int(round(shared_iterations)) if shared_iterations is not None else None,
        "adaptation_iterations": adaptation_iterations,
        "best_shared_adaptation_iterations": best_shared_iterations,
        "best_local_adaptation_iterations": best_local_iterations,
        "baseline_iterations": baseline_iterations,
        "task_ids": [task.task_id for task in task_specs],
    }
    setting_fingerprint = hashlib.sha1(
        json.dumps(setting_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    model = extract_primary_model(shared_config)
    edit_mode = extract_edit_mode(shared_config)
    label = format_setting_label(
        {
            "model": model,
            "shared_iterations": int(round(shared_iterations)) if shared_iterations is not None else None,
            "adaptation_iterations": adaptation_iterations,
            "baseline_iterations": baseline_iterations,
            "edit_mode": edit_mode,
        }
    )

    shared = load_shared_result(
        run_root / "shared_run",
        int(round(shared_iterations)) if shared_iterations is not None else None,
        repo_root=repo_root,
    )

    wandb_config = shared_config.get("wandb") if isinstance(shared_config.get("wandb"), dict) else {}
    wandb = {
        "project": wandb_config.get("project"),
        "entity": wandb_entity_override or wandb_config.get("entity"),
        "run_id": wandb_config.get("run_id"),
        "name": wandb_config.get("name"),
        "group": wandb_config.get("group"),
    }
    wandb["url"] = build_wandb_url(wandb)

    tasks: Dict[str, Any] = {}
    for task in task_specs:
        task_id = task.task_id
        spawn = load_spawn_result(run_root / "spawned_checkpoints" / task_id, repo_root=repo_root)
        adaptation = load_task_phase_result(
            run_root / "adaptation" / task_id,
            adaptation_iterations if isinstance(adaptation_iterations, int) else None,
            repo_root=repo_root,
        )
        best_shared = load_task_phase_result(
            best_shared_root / task_id,
            best_shared_iterations if isinstance(best_shared_iterations, int) else None,
            repo_root=repo_root,
        )
        best_local = load_task_phase_result(
            best_local_root / task_id,
            best_local_iterations if isinstance(best_local_iterations, int) else None,
            repo_root=repo_root,
        )
        baseline = load_task_phase_result(
            run_root / "baselines" / task_id,
            baseline_iterations if isinstance(baseline_iterations, int) else None,
            repo_root=repo_root,
        )
        task_summary = (
            comparison_tasks.get(task_id)
            if isinstance(comparison_tasks.get(task_id), dict)
            else {}
        )
        if best_shared_branch_present and best_shared["score"] is None:
            best_shared["score"] = _phase_score_from_task_summary(
                task_summary,
                flat_key=(
                    "best_shared_adaptation_best_score",
                    f"{legacy_branch_key('best_shared')}_best_score",
                ),
                nested_key=("best_shared_adaptation", legacy_branch_key("best_shared")),
            )
        if best_local_branch_present and best_local["score"] is None:
            best_local["score"] = _phase_score_from_task_summary(
                task_summary,
                flat_key=(
                    "best_local_adaptation_best_score",
                    f"{legacy_branch_key('best_local')}_best_score",
                ),
                nested_key=("best_local_adaptation", legacy_branch_key("best_local")),
            )
        tasks[task_id] = {
            "task_id": task_id,
            "task_spec": task.to_spec_dict(),
            "spawn": spawn,
            "adaptation": adaptation,
            "best_shared_adaptation": best_shared,
            "best_local_adaptation": best_local,
            "baseline": baseline,
            "adaptation_minus_spawn": delta(adaptation["score"], spawn["score"]),
            "adaptation_minus_best_shared": delta(
                adaptation["score"],
                best_shared["score"],
            ),
            "adaptation_minus_best_local": delta(
                adaptation["score"],
                best_local["score"],
            ),
            "best_local_minus_best_shared": delta(
                best_local["score"],
                best_shared["score"],
            ),
            "adaptation_minus_baseline": delta(adaptation["score"], baseline["score"]),
            "adaptation_vs_spawn": compare_scores(adaptation["score"], spawn["score"]),
            "adaptation_vs_best_shared": compare_scores(
                adaptation["score"],
                best_shared["score"],
            ),
            "adaptation_vs_best_local": compare_scores(
                adaptation["score"],
                best_local["score"],
            ),
            "best_local_vs_best_shared": compare_scores(
                best_local["score"],
                best_shared["score"],
            ),
            "adaptation_vs_baseline": compare_scores(adaptation["score"], baseline["score"]),
        }

    spawn_scores = [task["spawn"]["score"] for task in tasks.values()]
    adaptation_scores = [task["adaptation"]["score"] for task in tasks.values()]
    best_shared_scores = [
        task["best_shared_adaptation"]["score"] for task in tasks.values()
    ]
    best_local_scores = [
        task["best_local_adaptation"]["score"] for task in tasks.values()
    ]
    baseline_scores = [task["baseline"]["score"] for task in tasks.values()]
    adaptation_vs_spawn_outcomes = [task["adaptation_vs_spawn"] for task in tasks.values()]
    adaptation_vs_best_shared_outcomes = [
        task["adaptation_vs_best_shared"] for task in tasks.values()
    ]
    adaptation_vs_best_local_outcomes = [
        task["adaptation_vs_best_local"] for task in tasks.values()
    ]
    best_local_vs_best_shared_outcomes = [
        task["best_local_vs_best_shared"] for task in tasks.values()
    ]
    adaptation_vs_baseline_outcomes = [task["adaptation_vs_baseline"] for task in tasks.values()]
    macro = {
        "shared_best_score": shared["score"],
        "spawn_mean_score": mean_or_none(spawn_scores),
        "adaptation_mean_score": mean_or_none(adaptation_scores),
        "best_shared_mean_score": mean_or_none(best_shared_scores),
        "best_local_mean_score": mean_or_none(best_local_scores),
        "baseline_mean_score": mean_or_none(baseline_scores),
        "adaptation_minus_spawn_mean": mean_or_none(
            task["adaptation_minus_spawn"] for task in tasks.values()
        ),
        "adaptation_minus_best_shared_mean": mean_or_none(
            task["adaptation_minus_best_shared"] for task in tasks.values()
        ),
        "adaptation_minus_best_local_mean": mean_or_none(
            task["adaptation_minus_best_local"] for task in tasks.values()
        ),
        "best_local_minus_best_shared_mean": mean_or_none(
            task["best_local_minus_best_shared"] for task in tasks.values()
        ),
        "adaptation_minus_baseline_mean": mean_or_none(
            task["adaptation_minus_baseline"] for task in tasks.values()
        ),
        "adaptation_vs_spawn_counts": comparison_counts(adaptation_vs_spawn_outcomes),
        "adaptation_vs_best_shared_counts": comparison_counts(
            adaptation_vs_best_shared_outcomes
        ),
        "adaptation_vs_best_local_counts": comparison_counts(
            adaptation_vs_best_local_outcomes
        ),
        "best_local_vs_best_shared_counts": comparison_counts(
            best_local_vs_best_shared_outcomes
        ),
        "adaptation_vs_baseline_counts": comparison_counts(adaptation_vs_baseline_outcomes),
    }

    total_tasks = len(tasks)
    comparison_summary_available = (run_root / "comparison_summary.json").is_file()
    health = classify_run_health(
        shared=shared,
        tasks=tasks,
        comparison_summary_available=comparison_summary_available,
    )
    return {
        "run_name": run_root.name,
        "run_root": serialize_report_path(run_root, repo_root),
        "run_path_display": display_repo_relative(run_root, repo_root),
        "run_started_at": parse_run_time(run_root.name, run_root).isoformat(),
        "setting": {
            "id": setting_fingerprint,
            "family": manifest_family,
            "label": label,
            "model": model,
            "edit_mode": edit_mode,
            "shared_iterations": int(round(shared_iterations)) if shared_iterations is not None else None,
            "adaptation_iterations": adaptation_iterations,
            "best_shared_adaptation_iterations": best_shared_iterations,
            "best_local_adaptation_iterations": best_local_iterations,
            "baseline_iterations": baseline_iterations,
        },
        "wandb": wandb,
        "shared": shared,
        "tasks": tasks,
        "macro": macro,
        "best_shared_adaptation": best_shared_summary,
        "best_local_adaptation": best_local_summary,
        "comparison_summary_available": comparison_summary_available,
        "health": health,
        "phase_presence": {
            "best_shared_adaptation": best_shared_branch_present,
            "best_local_adaptation": best_local_branch_present,
        },
        "status": {
            "shared": progress_label({"shared": shared, "tasks": tasks}, "shared", total_tasks),
            "spawn": progress_label({"shared": shared, "tasks": tasks}, "spawn", total_tasks),
            "adaptation": progress_label({"shared": shared, "tasks": tasks}, "adaptation", total_tasks),
            "best_shared_adaptation": progress_label(
                {"shared": shared, "tasks": tasks},
                "best_shared_adaptation",
                total_tasks,
            ),
            "best_local_adaptation": progress_label(
                {"shared": shared, "tasks": tasks},
                "best_local_adaptation",
                total_tasks,
            ),
            "baseline": progress_label({"shared": shared, "tasks": tasks}, "baseline", total_tasks),
        },
    }


def adaptation_label(value: float | None) -> str:
    if value is None:
        return "N/A"
    return str(int(round(value)))


def budget_label(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        if not value:
            return "N/A"
        return ",".join(str(item) for item in value)
    return str(value)


def discover_run_roots(
    *,
    repo_root: Path,
    results_dir: Path,
    explicit_run_roots: Sequence[str] | None,
) -> List[Path]:
    def is_run_root(path: Path) -> bool:
        return path.is_dir() and (
            (path / "configs" / "shared_config.yaml").is_file()
            or (path / "shared_run").is_dir()
        )

    if explicit_run_roots:
        run_roots = [resolve_repo_path(repo_root, value) for value in explicit_run_roots]
        return [path for path in run_roots if path.is_dir()]
    if not results_dir.is_dir():
        return []
    discovered: List[Path] = []
    for child in sorted(results_dir.iterdir()):
        if is_run_root(child):
            discovered.append(child.resolve())
            continue
        if not child.is_dir():
            continue
        for grandchild in sorted(child.iterdir()):
            if is_run_root(grandchild):
                discovered.append(grandchild.resolve())
    return discovered


def resolve_report_targets(args: argparse.Namespace, *, repo_root: Path) -> List[Dict[str, Any]]:
    manifest_values = list(args.manifest or [])
    results_dir_values = list(args.results_dir or [])

    if not manifest_values and not results_dir_values:
        targets: List[Dict[str, Any]] = []
        for manifest_value, results_dir_value in MAIN_PAPER_REPORT_TARGETS:
            manifest_path = resolve_repo_path(repo_root, manifest_value)
            manifest = load_manifest(manifest_path)
            targets.append(
                {
                    "manifest_path": manifest_path,
                    "manifest": manifest,
                    "results_dir": resolve_repo_path(repo_root, results_dir_value),
                    "explicit": False,
                }
            )
        return targets

    if not manifest_values:
        raise ValueError("--results-dir requires at least one --manifest")

    if results_dir_values and len(results_dir_values) != len(manifest_values):
        raise ValueError(
            "When repeating --manifest, pass --results-dir the same number of times, "
            "or omit --results-dir to use each manifest's output_root."
        )

    targets = []
    for index, manifest_value in enumerate(manifest_values):
        manifest_path = resolve_repo_path(repo_root, manifest_value)
        manifest = load_manifest(manifest_path)
        results_dir = (
            resolve_repo_path(repo_root, results_dir_values[index])
            if results_dir_values
            else manifest.output_root
        )
        targets.append(
            {
                "manifest_path": manifest_path,
                "manifest": manifest,
                "results_dir": results_dir,
                "explicit": True,
            }
        )
    return targets


def select_runs_by_setting(
    run_reports: Sequence[Dict[str, Any]],
    *,
    include_all_runs: bool,
    latest_per_setting: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run_report in run_reports:
        grouped[run_report["setting"]["id"]].append(run_report)

    groups: List[Dict[str, Any]] = []
    for setting_id, members in grouped.items():
        ordered_members = sorted(
            members,
            key=lambda report: (
                report["run_started_at"],
                report["run_name"],
            ),
            reverse=True,
        )
        healthy_members = [
            report for report in ordered_members if report.get("health", {}).get("ok")
        ]
        problematic_members = [
            report for report in ordered_members if not report.get("health", {}).get("ok")
        ]
        if include_all_runs:
            included = healthy_members
            excluded = []
        else:
            limit = max(1, int(latest_per_setting))
            included = healthy_members[:limit]
            excluded = healthy_members[limit:]
        groups.append(
            {
                "setting_id": setting_id,
                "setting": (
                    included[0]["setting"]
                    if included
                    else ordered_members[0]["setting"]
                ),
                "runs": included,
                "problem_runs": [
                    {
                        "run_name": report["run_name"],
                        "run_path_display": report["run_path_display"],
                        "health_status": report["health"]["status"],
                        "health_reasons": list(report["health"]["reasons"]),
                    }
                    for report in problematic_members
                ],
                "excluded_runs": [
                    {"run_name": report["run_name"], "run_path_display": report["run_path_display"]}
                    for report in excluded
                ],
                "all_runs": ordered_members,
            }
        )
    return sorted(groups, key=lambda group: group["setting"]["label"])


def aggregate_group(group: Dict[str, Any]) -> Dict[str, Any]:
    runs = group["runs"]
    reference_runs = runs if runs else group.get("all_runs", [])
    task_ids = list(reference_runs[0]["tasks"].keys()) if reference_runs else []
    overall = {
        "shared_best_score": summarize([run["macro"]["shared_best_score"] for run in runs]),
        "spawn_mean_score": summarize([run["macro"]["spawn_mean_score"] for run in runs]),
        "adaptation_mean_score": summarize([run["macro"]["adaptation_mean_score"] for run in runs]),
        "best_shared_mean_score": summarize(
            [run["macro"]["best_shared_mean_score"] for run in runs]
        ),
        "best_local_mean_score": summarize(
            [run["macro"]["best_local_mean_score"] for run in runs]
        ),
        "baseline_mean_score": summarize([run["macro"]["baseline_mean_score"] for run in runs]),
        "adaptation_minus_spawn_mean": summarize(
            [run["macro"]["adaptation_minus_spawn_mean"] for run in runs]
        ),
        "adaptation_minus_best_shared_mean": summarize(
            [run["macro"]["adaptation_minus_best_shared_mean"] for run in runs]
        ),
        "adaptation_minus_best_local_mean": summarize(
            [run["macro"]["adaptation_minus_best_local_mean"] for run in runs]
        ),
        "best_local_minus_best_shared_mean": summarize(
            [run["macro"]["best_local_minus_best_shared_mean"] for run in runs]
        ),
        "adaptation_minus_baseline_mean": summarize(
            [run["macro"]["adaptation_minus_baseline_mean"] for run in runs]
        ),
        "adaptation_vs_spawn_counts": {
            key: sum(
                run["macro"]["adaptation_vs_spawn_counts"][key]
                for run in runs
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "adaptation_vs_best_shared_counts": {
            key: sum(
                run["macro"]["adaptation_vs_best_shared_counts"][key]
                for run in runs
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "adaptation_vs_best_local_counts": {
            key: sum(
                run["macro"]["adaptation_vs_best_local_counts"][key]
                for run in runs
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "best_local_vs_best_shared_counts": {
            key: sum(
                run["macro"]["best_local_vs_best_shared_counts"][key]
                for run in runs
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "adaptation_vs_baseline_counts": {
            key: sum(
                run["macro"]["adaptation_vs_baseline_counts"][key]
                for run in runs
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
    }

    per_task = {}
    for task_id in task_ids:
        outcomes_vs_spawn = [run["tasks"][task_id]["adaptation_vs_spawn"] for run in runs]
        outcomes_vs_best_shared = [
            run["tasks"][task_id]["adaptation_vs_best_shared"] for run in runs
        ]
        outcomes_vs_best_local = [
            run["tasks"][task_id]["adaptation_vs_best_local"] for run in runs
        ]
        outcomes_best_local_vs_best_shared = [
            run["tasks"][task_id]["best_local_vs_best_shared"] for run in runs
        ]
        outcomes_vs_baseline = [run["tasks"][task_id]["adaptation_vs_baseline"] for run in runs]
        per_task[task_id] = {
            "spawn_score": summarize([run["tasks"][task_id]["spawn"]["score"] for run in runs]),
            "adaptation_score": summarize(
                [run["tasks"][task_id]["adaptation"]["score"] for run in runs]
            ),
            "best_shared_score": summarize(
                [
                    run["tasks"][task_id]["best_shared_adaptation"]["score"]
                    for run in runs
                ]
            ),
            "best_local_score": summarize(
                [
                    run["tasks"][task_id]["best_local_adaptation"]["score"]
                    for run in runs
                ]
            ),
            "baseline_score": summarize([run["tasks"][task_id]["baseline"]["score"] for run in runs]),
            "adaptation_minus_spawn": summarize(
                [run["tasks"][task_id]["adaptation_minus_spawn"] for run in runs]
            ),
            "adaptation_minus_best_shared": summarize(
                [run["tasks"][task_id]["adaptation_minus_best_shared"] for run in runs]
            ),
            "adaptation_minus_best_local": summarize(
                [run["tasks"][task_id]["adaptation_minus_best_local"] for run in runs]
            ),
            "best_local_minus_best_shared": summarize(
                [run["tasks"][task_id]["best_local_minus_best_shared"] for run in runs]
            ),
            "adaptation_minus_baseline": summarize(
                [run["tasks"][task_id]["adaptation_minus_baseline"] for run in runs]
            ),
            "adaptation_vs_spawn_counts": comparison_counts(outcomes_vs_spawn),
            "adaptation_vs_best_shared_counts": comparison_counts(
                outcomes_vs_best_shared
            ),
            "adaptation_vs_best_local_counts": comparison_counts(
                outcomes_vs_best_local
            ),
            "best_local_vs_best_shared_counts": comparison_counts(
                outcomes_best_local_vs_best_shared
            ),
            "adaptation_vs_baseline_counts": comparison_counts(outcomes_vs_baseline),
        }

    group["aggregate"] = {
        "overall": overall,
        "per_task": per_task,
        "phase_completion": {
            "run_count": len(runs),
            "problem_run_count": len(group.get("problem_runs", [])),
            "shared_complete": sum(1 for run in runs if run["shared"]["complete"]),
            "spawn_complete": sum(
                1
                for run in runs
                if all(task["spawn"]["complete"] for task in run["tasks"].values())
            ),
            "adaptation_complete": sum(
                1
                for run in runs
                if all(task["adaptation"]["complete"] for task in run["tasks"].values())
            ),
            "best_shared_adaptation_complete": sum(
                1
                for run in runs
                if all(
                    task["best_shared_adaptation"]["complete"]
                    for task in run["tasks"].values()
                )
            ),
            "best_local_adaptation_complete": sum(
                1
                for run in runs
                if all(
                    task["best_local_adaptation"]["complete"]
                    for task in run["tasks"].values()
                )
            ),
            "baseline_complete": sum(
                1
                for run in runs
                if all(task["baseline"]["complete"] for task in run["tasks"].values())
            ),
            "comparison_ready": sum(
                1 for run in runs if run["comparison_summary_available"]
            ),
        },
    }
    return group


def build_report(
    *,
    manifest_path: Path,
    manifest_family: str,
    results_dir: Path,
    run_roots: Sequence[Path],
    groups: Sequence[Dict[str, Any]],
    latest_per_setting: int,
    include_all_runs: bool,
) -> Dict[str, Any]:
    return {
        "workflow": "multi_task_shared_then_adapt",
        "manifest_path": serialize_report_path(manifest_path, REPO_ROOT),
        "family": manifest_family,
        "results_dir": serialize_report_path(results_dir, REPO_ROOT),
        "scanned_run_roots": [serialize_report_path(path, REPO_ROOT) for path in run_roots],
        "latest_per_setting": latest_per_setting,
        "include_all_runs": include_all_runs,
        "groups": groups,
    }


def build_combined_report(
    *,
    family_reports: Sequence[Dict[str, Any]],
    latest_per_setting: int,
    include_all_runs: bool,
) -> Dict[str, Any]:
    scanned_run_roots = unique_in_order(
        run_root
        for report in family_reports
        for run_root in report.get("scanned_run_roots", [])
    )
    return {
        "workflow": "multi_task_shared_then_adapt",
        "scope": "multi_family",
        "families": [report["family"] for report in family_reports],
        "family_reports": list(family_reports),
        "scanned_run_roots": scanned_run_roots,
        "latest_per_setting": latest_per_setting,
        "include_all_runs": include_all_runs,
    }


def markdown_divider(column_count: int, *, right_aligned_indices: Iterable[int] = ()) -> str:
    right_aligned = set(right_aligned_indices)
    return (
        "| "
        + " | ".join("---:" if index in right_aligned else "---" for index in range(column_count))
        + " |"
    )


def group_reference_runs(group: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = group.get("runs") or []
    if runs:
        return list(runs)
    return list(group.get("all_runs") or [])


def group_seed_column_visibility(group: Dict[str, Any]) -> Dict[str, bool]:
    reference_runs = group_reference_runs(group)
    return {
        "best_shared_adaptation": any(
            run.get("phase_presence", {}).get("best_shared_adaptation", False)
            for run in reference_runs
        ),
        "best_local_adaptation": any(
            run.get("phase_presence", {}).get("best_local_adaptation", False)
            for run in reference_runs
        ),
    }


def group_optional_iteration_budget(
    group: Dict[str, Any],
    *,
    phase_key: str,
    setting_key: str,
) -> Any | None:
    values = unique_in_order(
        run["setting"].get(setting_key)
        for run in group_reference_runs(group)
        if run.get("phase_presence", {}).get(phase_key, False)
        and run["setting"].get(setting_key) is not None
    )
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def group_setting_label(group: Dict[str, Any]) -> str:
    visibility = group_seed_column_visibility(group)
    return format_setting_label(
        group["setting"],
        best_shared_iterations=(
            group_optional_iteration_budget(
                group,
                phase_key="best_shared_adaptation",
                setting_key="best_shared_adaptation_iterations",
            )
            if visibility["best_shared_adaptation"]
            else None
        ),
        best_local_iterations=(
            group_optional_iteration_budget(
                group,
                phase_key="best_local_adaptation",
                setting_key="best_local_adaptation_iterations",
            )
            if visibility["best_local_adaptation"]
            else None
        ),
    )


def status_row(
    run: Dict[str, Any],
    *,
    show_best_shared: bool,
    show_best_local: bool,
) -> str:
    comparison_status = "yes" if run["comparison_summary_available"] else "no"
    wandb_run = run["wandb"].get("run_id") or "N/A"
    cells = [
        run["run_name"],
        run["status"]["shared"],
        run["status"]["spawn"],
        run["status"]["adaptation"],
    ]
    if show_best_shared:
        cells.append(run["status"]["best_shared_adaptation"])
    if show_best_local:
        cells.append(run["status"]["best_local_adaptation"])
    cells.extend(
        [
            run["status"]["baseline"],
            run["health"]["status"],
            comparison_status,
            wandb_run,
        ]
    )
    return (
        "| "
        + " | ".join(cells)
        + " |"
    )


def task_detail_row(
    task: Dict[str, Any],
    *,
    show_best_shared: bool,
    show_best_local: bool,
) -> str:
    cells = [
        task["task_id"],
        format_float(task["spawn"]["score"]),
        format_float(task["adaptation"]["score"]),
    ]
    if show_best_shared:
        cells.append(format_float(task["best_shared_adaptation"]["score"]))
    if show_best_local:
        cells.append(format_float(task["best_local_adaptation"]["score"]))
    cells.extend(
        [
            format_float(task["baseline"]["score"]),
            format_float(task["adaptation_minus_spawn"], signed=True),
        ]
    )
    if show_best_shared:
        cells.append(format_float(task["adaptation_minus_best_shared"], signed=True))
    if show_best_local:
        cells.append(format_float(task["adaptation_minus_best_local"], signed=True))
    if show_best_shared and show_best_local:
        cells.append(format_float(task["best_local_minus_best_shared"], signed=True))
    cells.extend(
        [
            format_float(task["adaptation_minus_baseline"], signed=True),
            task["adaptation_vs_spawn"] or "N/A",
        ]
    )
    if show_best_shared:
        cells.append(task["adaptation_vs_best_shared"] or "N/A")
    if show_best_local:
        cells.append(task["adaptation_vs_best_local"] or "N/A")
    if show_best_shared and show_best_local:
        cells.append(task["best_local_vs_best_shared"] or "N/A")
    cells.append(task["adaptation_vs_baseline"] or "N/A")
    return (
        "| "
        + " | ".join(cells)
        + " |"
    )


def aggregate_task_row(
    task_id: str,
    aggregate: Dict[str, Any],
    *,
    show_best_shared: bool,
    show_best_local: bool,
) -> str:
    count = summary_count(
        aggregate["spawn_score"],
        aggregate["adaptation_score"],
        aggregate["baseline_score"],
        aggregate["adaptation_minus_spawn"],
        aggregate["adaptation_minus_baseline"],
    )
    if show_best_shared:
        count = max(
            count,
            summary_count(
                aggregate["best_shared_score"],
                aggregate["adaptation_minus_best_shared"],
            ),
        )
    if show_best_local:
        count = max(
            count,
            summary_count(
                aggregate["best_local_score"],
                aggregate["adaptation_minus_best_local"],
            ),
        )
    if show_best_shared and show_best_local:
        count = max(
            count,
            summary_count(aggregate["best_local_minus_best_shared"]),
        )
    cells = [
        task_id,
        str(count),
        format_stat(aggregate["spawn_score"]),
        format_stat(aggregate["adaptation_score"]),
    ]
    if show_best_shared:
        cells.append(format_stat(aggregate["best_shared_score"]))
    if show_best_local:
        cells.append(format_stat(aggregate["best_local_score"]))
    cells.extend(
        [
            format_stat(aggregate["baseline_score"]),
            format_stat(aggregate["adaptation_minus_spawn"], signed=True),
        ]
    )
    if show_best_shared:
        cells.append(format_stat(aggregate["adaptation_minus_best_shared"], signed=True))
    if show_best_local:
        cells.append(format_stat(aggregate["adaptation_minus_best_local"], signed=True))
    if show_best_shared and show_best_local:
        cells.append(format_stat(aggregate["best_local_minus_best_shared"], signed=True))
    cells.extend(
        [
            format_stat(aggregate["adaptation_minus_baseline"], signed=True),
            format_counts(aggregate["adaptation_vs_spawn_counts"]),
        ]
    )
    if show_best_shared:
        cells.append(format_counts(aggregate["adaptation_vs_best_shared_counts"]))
    if show_best_local:
        cells.append(format_counts(aggregate["adaptation_vs_best_local_counts"]))
    if show_best_shared and show_best_local:
        cells.append(format_counts(aggregate["best_local_vs_best_shared_counts"]))
    cells.append(format_counts(aggregate["adaptation_vs_baseline_counts"]))
    return (
        "| "
        + " | ".join(cells)
        + " |"
    )


def append_family_report_markdown(
    lines: List[str],
    report: Dict[str, Any],
    *,
    repo_root: Path,
    family_heading: str | None = None,
    setting_heading_level: int = 2,
    run_heading_level: int = 3,
) -> None:
    if family_heading:
        lines.append(family_heading)
        lines.append("")

    lines.extend(
        [
            f"- Manifest: `{render_report_path(report['manifest_path'], repo_root)}`",
            f"- Results dir: `{render_report_path(report['results_dir'], repo_root)}`",
            f"- Scanned run directories: `{len(report['scanned_run_roots'])}`",
            f"- Selected latest runs per setting: `{report['latest_per_setting']}`",
            f"- Include all runs: `{report['include_all_runs']}`",
            "",
        ]
    )

    groups = report.get("groups") or []
    if not groups:
        lines.append("No EMO-STA run directories were found.")
        lines.append("")
        return

    for group in groups:
        aggregate = group["aggregate"]
        visibility = group_seed_column_visibility(group)
        show_best_shared = visibility["best_shared_adaptation"]
        show_best_local = visibility["best_local_adaptation"]
        lines.append(f"{'#' * setting_heading_level} {group_setting_label(group)}")
        lines.append("")
        lines.append(f"- Setting id: `{group['setting_id']}`")
        lines.append(f"- Included healthy runs: `{len(group['runs'])}`")
        if group.get("problem_runs"):
            flagged = ", ".join(
                f"{entry['run_name']} ({entry['health_status']})"
                for entry in group["problem_runs"]
            )
            lines.append(f"- Problematic runs excluded from aggregates: `{flagged}`")
        if group["excluded_runs"]:
            skipped = ", ".join(entry["run_name"] for entry in group["excluded_runs"])
            lines.append(f"- Older matching runs skipped by selection: `{skipped}`")
        phase_completion = aggregate["phase_completion"]
        completion_parts = [
            f"`shared {phase_completion['shared_complete']}/{phase_completion['run_count']}`",
            f"`spawn {phase_completion['spawn_complete']}/{phase_completion['run_count']}`",
            f"`adaptation {phase_completion['adaptation_complete']}/{phase_completion['run_count']}`",
        ]
        if show_best_shared:
            completion_parts.append(
                f"`best-shared "
                f"{phase_completion['best_shared_adaptation_complete']}/{phase_completion['run_count']}`"
            )
        if show_best_local:
            completion_parts.append(
                f"`best-local "
                f"{phase_completion['best_local_adaptation_complete']}/{phase_completion['run_count']}`"
            )
        completion_parts.extend(
            [
                f"`baseline {phase_completion['baseline_complete']}/{phase_completion['run_count']}`",
                f"`comparison {phase_completion['comparison_ready']}/{phase_completion['run_count']}`",
            ]
        )
        lines.append("- Fully complete runs: " + ", ".join(completion_parts))
        if phase_completion["problem_run_count"]:
            lines.append(
                f"- Problematic run count excluded from aggregates: `{phase_completion['problem_run_count']}`"
            )
        lines.append("")
        status_header = ["run", "shared", "spawn", "adaptation"]
        if show_best_shared:
            status_header.append("best-shared")
        if show_best_local:
            status_header.append("best-local")
        status_header.extend(["baselines", "health", "comparison", "W&B run"])
        lines.append("| " + " | ".join(status_header) + " |")
        lines.append(markdown_divider(len(status_header)))
        for run in group["runs"]:
            lines.append(
                status_row(
                    run,
                    show_best_shared=show_best_shared,
                    show_best_local=show_best_local,
                )
            )
        lines.append("")

        overall = aggregate["overall"]
        overall_header = ["n", "shared best", "spawn mean", "adapted mean"]
        if show_best_shared:
            overall_header.append("best-shared mean")
        if show_best_local:
            overall_header.append("best-local mean")
        overall_header.extend(["baseline mean", "adapt - spawn"])
        if show_best_shared:
            overall_header.append("adapt - best-shared")
        if show_best_local:
            overall_header.append("adapt - best-local")
        if show_best_shared and show_best_local:
            overall_header.append("best-local - best-shared")
        overall_header.append("adapt - baseline")
        overall_header.append("adapt>spawn W/T/L")
        if show_best_shared:
            overall_header.append("adapt>best-shared W/T/L")
        if show_best_local:
            overall_header.append("adapt>best-local W/T/L")
        if show_best_shared and show_best_local:
            overall_header.append("best-local>best-shared W/T/L")
        overall_header.append("adapt>baseline W/T/L")
        lines.append("| " + " | ".join(overall_header) + " |")
        lines.append(markdown_divider(len(overall_header), right_aligned_indices=(0,)))
        overall_cells = [
            str(len(group["runs"])),
            format_stat(overall["shared_best_score"]),
            format_stat(overall["spawn_mean_score"]),
            format_stat(overall["adaptation_mean_score"]),
        ]
        if show_best_shared:
            overall_cells.append(format_stat(overall["best_shared_mean_score"]))
        if show_best_local:
            overall_cells.append(format_stat(overall["best_local_mean_score"]))
        overall_cells.extend(
            [
                format_stat(overall["baseline_mean_score"]),
                format_stat(overall["adaptation_minus_spawn_mean"], signed=True),
            ]
        )
        if show_best_shared:
            overall_cells.append(
                format_stat(overall["adaptation_minus_best_shared_mean"], signed=True)
            )
        if show_best_local:
            overall_cells.append(
                format_stat(overall["adaptation_minus_best_local_mean"], signed=True)
            )
        if show_best_shared and show_best_local:
            overall_cells.append(
                format_stat(overall["best_local_minus_best_shared_mean"], signed=True)
            )
        overall_cells.extend(
            [
                format_stat(overall["adaptation_minus_baseline_mean"], signed=True),
                format_counts(overall["adaptation_vs_spawn_counts"]),
            ]
        )
        if show_best_shared:
            overall_cells.append(format_counts(overall["adaptation_vs_best_shared_counts"]))
        if show_best_local:
            overall_cells.append(format_counts(overall["adaptation_vs_best_local_counts"]))
        if show_best_shared and show_best_local:
            overall_cells.append(
                format_counts(overall["best_local_vs_best_shared_counts"])
            )
        overall_cells.append(format_counts(overall["adaptation_vs_baseline_counts"]))
        lines.append("| " + " | ".join(overall_cells) + " |")
        lines.append("")

        task_header = ["task", "n", "spawn", "adapted"]
        if show_best_shared:
            task_header.append("best-shared")
        if show_best_local:
            task_header.append("best-local")
        task_header.extend(["baseline", "adapt - spawn"])
        if show_best_shared:
            task_header.append("adapt - best-shared")
        if show_best_local:
            task_header.append("adapt - best-local")
        if show_best_shared and show_best_local:
            task_header.append("best-local - best-shared")
        task_header.append("adapt - baseline")
        task_header.append("adapt>spawn W/T/L")
        if show_best_shared:
            task_header.append("adapt>best-shared W/T/L")
        if show_best_local:
            task_header.append("adapt>best-local W/T/L")
        if show_best_shared and show_best_local:
            task_header.append("best-local>best-shared W/T/L")
        task_header.append("adapt>baseline W/T/L")
        lines.append("| " + " | ".join(task_header) + " |")
        lines.append(markdown_divider(len(task_header), right_aligned_indices=(1,)))
        for task_id in sorted(aggregate["per_task"]):
            lines.append(
                aggregate_task_row(
                    task_id,
                    aggregate["per_task"][task_id],
                    show_best_shared=show_best_shared,
                    show_best_local=show_best_local,
                )
            )
        lines.append("")

        for run in group["runs"]:
            lines.append(f"{'#' * run_heading_level} Run `{run['run_name']}`")
            lines.append("")
            lines.append(f"- Path: `{run['run_path_display']}`")
            lines.append(f"- Health: `{run['health']['status']}`")
            if run["health"]["reasons"]:
                lines.append("- Health notes:")
                for reason in run["health"]["reasons"]:
                    lines.append(f"  - `{reason}`")
            if run["wandb"].get("url"):
                lines.append(f"- W&B: `{run['wandb']['url']}`")
            elif run["wandb"].get("run_id"):
                lines.append(f"- W&B run id: `{run['wandb']['run_id']}`")
            macro_parts = [
                f"`shared {format_float(run['macro']['shared_best_score'])}`",
                f"`spawn mean {format_float(run['macro']['spawn_mean_score'])}`",
                f"`adapted mean {format_float(run['macro']['adaptation_mean_score'])}`",
            ]
            if show_best_shared:
                macro_parts.append(
                    f"`best-shared mean "
                    f"{format_float(run['macro']['best_shared_mean_score'])}`"
                )
            if show_best_local:
                macro_parts.append(
                    f"`best-local mean "
                    f"{format_float(run['macro']['best_local_mean_score'])}`"
                )
            macro_parts.append(f"`baseline mean {format_float(run['macro']['baseline_mean_score'])}`")
            lines.append("- Macro scores: " + ", ".join(macro_parts))
            lines.append("")
            run_task_header = ["task", "spawn", "adapted"]
            if show_best_shared:
                run_task_header.append("best-shared")
            if show_best_local:
                run_task_header.append("best-local")
            run_task_header.extend(["baseline", "adapt - spawn"])
            if show_best_shared:
                run_task_header.append("adapt - best-shared")
            if show_best_local:
                run_task_header.append("adapt - best-local")
            if show_best_shared and show_best_local:
                run_task_header.append("best-local - best-shared")
            run_task_header.append("adapt - baseline")
            run_task_header.append("adapt>spawn")
            if show_best_shared:
                run_task_header.append("adapt>best-shared")
            if show_best_local:
                run_task_header.append("adapt>best-local")
            if show_best_shared and show_best_local:
                run_task_header.append("best-local>best-shared")
            run_task_header.append("adapt>baseline")
            lines.append("| " + " | ".join(run_task_header) + " |")
            lines.append(markdown_divider(len(run_task_header)))
            for task_id in sorted(run["tasks"]):
                lines.append(
                    task_detail_row(
                        run["tasks"][task_id],
                        show_best_shared=show_best_shared,
                        show_best_local=show_best_local,
                    )
                )
            lines.append("")


def build_markdown_report(report: Dict[str, Any], *, repo_root: Path) -> str:
    lines = ["# EMO-STA Results Summary", ""]

    family_reports = report.get("family_reports")
    if isinstance(family_reports, list):
        lines.extend(
            [
                f"- Families included: `{len(family_reports)}`",
                f"- Total scanned run directories: `{len(report.get('scanned_run_roots', []))}`",
                f"- Selected latest runs per setting: `{report['latest_per_setting']}`",
                f"- Include all runs: `{report['include_all_runs']}`",
                "",
            ]
        )
        if not family_reports:
            lines.append("No EMO-STA run directories were found.")
            return "\n".join(lines).rstrip() + "\n"
        for family_report in family_reports:
            append_family_report_markdown(
                lines,
                family_report,
                repo_root=repo_root,
                family_heading=f"## Family `{family_report['family']}`",
                setting_heading_level=3,
                run_heading_level=4,
            )
    else:
        append_family_report_markdown(
            lines,
            report,
            repo_root=repo_root,
            setting_heading_level=2,
            run_heading_level=3,
        )

    return "\n".join(lines).rstrip() + "\n"


def write_text_if_requested(path_value: str | None, content: str, repo_root: Path) -> None:
    if path_value is None:
        return
    path = resolve_repo_path(repo_root, path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json_if_requested(path_value: str | None, content: Dict[str, Any], repo_root: Path) -> None:
    if path_value is None:
        return
    path = resolve_repo_path(repo_root, path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_file(__file__)
    try:
        targets = resolve_report_targets(args, repo_root=repo_root)
        if args.run_root and len(targets) != 1:
            raise ValueError(
                "--run-root can only be used when reporting a single manifest/results pair"
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    family_reports: List[Dict[str, Any]] = []
    discovered_any = False
    for target in targets:
        manifest = target["manifest"]
        manifest_path = target["manifest_path"]
        results_dir = target["results_dir"]
        task_specs = family_task_specs(manifest)
        run_roots = discover_run_roots(
            repo_root=repo_root,
            results_dir=results_dir,
            explicit_run_roots=args.run_root,
        )
        discovered_any = discovered_any or bool(run_roots)

        run_reports = [
            load_run_report(
                run_root,
                repo_root=repo_root,
                manifest_path=manifest_path,
                manifest_family=manifest.family,
                task_specs=task_specs,
                wandb_entity_override=args.wandb_entity,
            )
            for run_root in run_roots
        ]
        run_reports = [
            run_report for run_report in run_reports if run_has_observable_results(run_report)
        ]
        selected_groups = select_runs_by_setting(
            run_reports,
            include_all_runs=args.include_all_runs,
            latest_per_setting=args.latest_per_setting,
        )
        groups = [aggregate_group(group) for group in selected_groups]
        family_report = build_report(
            manifest_path=manifest_path,
            manifest_family=manifest.family,
            results_dir=results_dir,
            run_roots=run_roots,
            groups=groups,
            latest_per_setting=args.latest_per_setting,
            include_all_runs=args.include_all_runs,
        )
        if target["explicit"] or run_roots or groups:
            family_reports.append(family_report)

    if args.strict and not discovered_any:
        return 1

    report: Dict[str, Any]
    if len(family_reports) == 1:
        report = family_reports[0]
    else:
        report = build_combined_report(
            family_reports=family_reports,
            latest_per_setting=args.latest_per_setting,
            include_all_runs=args.include_all_runs,
        )

    markdown = build_markdown_report(report, repo_root=repo_root)
    sys.stdout.write(markdown)
    write_text_if_requested(args.markdown_out, markdown, repo_root)
    write_json_if_requested(args.json_out, report, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
