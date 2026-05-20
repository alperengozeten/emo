"""Canonical function-minimization task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR = "FUNCTION_MINIMIZATION_TASK_ID"
FUNCTION_MINIMIZATION_SHARED_SELECTOR = "all"
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 5.0

Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]
Objective2D = Callable[[float, float], float]


def objective_sincosxy(x: float, y: float) -> float:
    """Public SinCosXY benchmark used by the EMO-STA family."""
    return math.sin(x) * math.cos(y) + math.sin(x * y) + (x**2 + y**2) / 20.0


def objective_ackley(x: float, y: float) -> float:
    """Standard 2D Ackley benchmark."""
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
        - math.exp(0.5 * (math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)))
        + math.e
        + 20.0
    )


def objective_rastrigin(x: float, y: float) -> float:
    """Standard 2D Rastrigin benchmark."""
    return 20.0 + x**2 + y**2 - 10.0 * (
        math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)
    )


def objective_rosenbrock(x: float, y: float) -> float:
    """Standard 2D Rosenbrock benchmark."""
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def _shifted_objective(
    base_objective: Objective2D,
    *,
    shift_x: float,
    shift_y: float,
) -> Objective2D:
    """Translate a public benchmark so its optimum is hidden from candidates."""

    def shifted(x: float, y: float) -> float:
        return base_objective(x - shift_x, y - shift_y)

    return shifted


@dataclass(frozen=True)
class FunctionMinimizationTaskSpec:
    """Stable task identity for the function-minimization EMO-STA family."""

    task_id: str
    task_index: int
    display_name: str
    bounds: Bounds2D
    optimum_x: float
    optimum_y: float
    optimum_value: float
    search_iterations_full: int
    search_iterations_stage1: int
    evaluation_seeds_full: tuple[int, ...]
    evaluation_seeds_stage1: tuple[int, ...]
    _objective_fn: Objective2D = field(repr=False, compare=False)

    @property
    def objective_fn(self) -> Objective2D:
        return self._objective_fn

    def to_spec_dict(self) -> Dict[str, Any]:
        """Return the prompt-safe task metadata surfaced in public artifacts."""
        return {
            "display_name": self.display_name,
            "bounds": [[float(low), float(high)] for (low, high) in self.bounds],
        }


FUNCTION_MINIMIZATION_TASK_SPECS: tuple[FunctionMinimizationTaskSpec, ...] = (
    FunctionMinimizationTaskSpec(
        task_id="fm_sincosxy_2d",
        task_index=0,
        display_name="SinCosXY",
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        optimum_x=-0.80406466,
        optimum_y=-0.72247960,
        optimum_value=-1.51868584,
        search_iterations_full=200,
        search_iterations_stage1=50,
        evaluation_seeds_full=(0, 1, 2, 3, 4),
        evaluation_seeds_stage1=(0, 1),
        _objective_fn=_shifted_objective(
            objective_sincosxy,
            shift_x=0.9,
            shift_y=-1.4,
        ),
    ),
    FunctionMinimizationTaskSpec(
        task_id="fm_ackley_2d",
        task_index=1,
        display_name="Ackley",
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        optimum_x=1.7,
        optimum_y=-1.3,
        optimum_value=0.0,
        search_iterations_full=200,
        search_iterations_stage1=50,
        evaluation_seeds_full=(0, 1, 2, 3, 4),
        evaluation_seeds_stage1=(0, 1),
        _objective_fn=_shifted_objective(
            objective_ackley,
            shift_x=1.7,
            shift_y=-1.3,
        ),
    ),
    FunctionMinimizationTaskSpec(
        task_id="fm_rastrigin_2d",
        task_index=2,
        display_name="Rastrigin",
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        optimum_x=-2.2,
        optimum_y=1.4,
        optimum_value=0.0,
        search_iterations_full=200,
        search_iterations_stage1=50,
        evaluation_seeds_full=(0, 1, 2, 3, 4),
        evaluation_seeds_stage1=(0, 1),
        _objective_fn=_shifted_objective(
            objective_rastrigin,
            shift_x=-2.2,
            shift_y=1.4,
        ),
    ),
    FunctionMinimizationTaskSpec(
        task_id="fm_rosenbrock_2d",
        task_index=3,
        display_name="Rosenbrock",
        bounds=((-3.0, 3.0), (-3.0, 3.0)),
        optimum_x=-0.4,
        optimum_y=1.7,
        optimum_value=0.0,
        search_iterations_full=200,
        search_iterations_stage1=50,
        evaluation_seeds_full=(0, 1, 2, 3, 4),
        evaluation_seeds_stage1=(0, 1),
        _objective_fn=_shifted_objective(
            objective_rosenbrock,
            shift_x=-1.4,
            shift_y=0.7,
        ),
    ),
)

FUNCTION_MINIMIZATION_TASKS_BY_ID: Dict[str, FunctionMinimizationTaskSpec] = {
    task.task_id: task for task in FUNCTION_MINIMIZATION_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[FunctionMinimizationTaskSpec]:
    """Resolve a task selector into concrete task specs."""
    normalized = (selector or FUNCTION_MINIMIZATION_SHARED_SELECTOR).strip()
    if not normalized or normalized == FUNCTION_MINIMIZATION_SHARED_SELECTOR:
        return list(FUNCTION_MINIMIZATION_TASK_SPECS)
    if normalized not in FUNCTION_MINIMIZATION_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in FUNCTION_MINIMIZATION_TASK_SPECS)
        raise ValueError(
            f"Unknown function-minimization task '{normalized}'. Available: {available}"
        )
    return [FUNCTION_MINIMIZATION_TASKS_BY_ID[normalized]]


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _coerce_non_negative_float(value: Any, default: float) -> float:
    return max(0.0, _coerce_finite_float(value, default))


def _distance_to_optimum(task: FunctionMinimizationTaskSpec, x: float, y: float) -> float:
    return math.sqrt((x - task.optimum_x) ** 2 + (y - task.optimum_y) ** 2)


def empty_task_metrics(
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    del timeout_seconds
    return {
        "best_value": 0.0,
        "value_gap": 0.0,
        "distance_to_optimum": 0.0,
        "value_score": 0.0,
        "distance_score": 0.0,
        "reliability_score": 0.0,
        "avg_eval_time": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, float]:
    """Normalize evaluator outputs into a stable finite metric schema."""
    defaults = empty_task_metrics(timeout_seconds=timeout_seconds)
    raw_metrics = raw_metrics or {}

    metrics = {
        "best_value": _coerce_finite_float(raw_metrics.get("best_value"), defaults["best_value"]),
        "value_gap": _coerce_non_negative_float(
            raw_metrics.get("value_gap"),
            defaults["value_gap"],
        ),
        "distance_to_optimum": _coerce_non_negative_float(
            raw_metrics.get("distance_to_optimum"),
            defaults["distance_to_optimum"],
        ),
        "value_score": _coerce_non_negative_float(
            raw_metrics.get("value_score"),
            defaults["value_score"],
        ),
        "distance_score": _coerce_non_negative_float(
            raw_metrics.get("distance_score"),
            defaults["distance_score"],
        ),
        "reliability_score": _coerce_non_negative_float(
            raw_metrics.get("reliability_score"),
            defaults["reliability_score"],
        ),
        "avg_eval_time": _coerce_non_negative_float(
            raw_metrics.get("avg_eval_time"),
            defaults["avg_eval_time"],
        ),
    }
    score = _coerce_non_negative_float(raw_metrics.get("score"), defaults["score"])
    combined_score = _coerce_non_negative_float(
        raw_metrics.get("combined_score"),
        score if score > 0.0 else defaults["combined_score"],
    )
    metrics["score"] = score if score > 0.0 else combined_score
    metrics["combined_score"] = metrics["score"]
    return metrics


def score_task_metrics(
    *,
    optimum_value: float,
    best_values: Sequence[float],
    distances: Sequence[float],
    eval_times: Sequence[float],
    total_trials: int,
) -> Dict[str, float]:
    """Compute the canonical task-local score from successful trial metrics."""
    if total_trials <= 0 or not best_values or not distances or not eval_times:
        return empty_task_metrics()

    avg_best_value = sum(float(value) for value in best_values) / len(best_values)
    avg_distance = sum(float(distance) for distance in distances) / len(distances)
    avg_eval_time = sum(float(runtime) for runtime in eval_times) / len(eval_times)
    value_gap = max(0.0, avg_best_value - float(optimum_value))
    value_score = 1.0 / (1.0 + value_gap)
    distance_score = 1.0 / (1.0 + avg_distance)
    reliability_score = len(best_values) / float(total_trials)
    score = (
        0.50 * value_score
        + 0.35 * distance_score
        + 0.15 * reliability_score
    )
    return {
        "best_value": avg_best_value,
        "value_gap": value_gap,
        "distance_to_optimum": avg_distance,
        "value_score": value_score,
        "distance_score": distance_score,
        "reliability_score": reliability_score,
        "avg_eval_time": avg_eval_time,
        "score": score,
        "combined_score": score,
    }


def finalize_task_metrics(
    task: FunctionMinimizationTaskSpec,
    metrics: Mapping[str, float],
) -> Dict[str, float]:
    """Recompute canonical derived metrics and score for one task."""
    reliability_score = max(0.0, float(metrics["reliability_score"]))
    avg_eval_time = max(0.0, float(metrics["avg_eval_time"]))
    if reliability_score <= 0.0:
        return {
            **empty_task_metrics(),
            "avg_eval_time": avg_eval_time,
        }

    best_value = float(metrics["best_value"])
    distance_to_optimum = max(0.0, float(metrics["distance_to_optimum"]))
    value_gap = max(0.0, best_value - task.optimum_value)
    value_score = 1.0 / (1.0 + value_gap)
    distance_score = 1.0 / (1.0 + distance_to_optimum)
    score = (
        0.50 * value_score
        + 0.35 * distance_score
        + 0.15 * reliability_score
    )
    return {
        "best_value": best_value,
        "value_gap": value_gap,
        "distance_to_optimum": distance_to_optimum,
        "value_score": value_score,
        "distance_score": distance_score,
        "reliability_score": reliability_score,
        "avg_eval_time": avg_eval_time,
        "score": score,
        "combined_score": score,
    }


def build_task_result(
    task: FunctionMinimizationTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = finalize_task_metrics(
        task,
        normalize_task_metrics(raw_metrics, timeout_seconds=timeout_seconds),
    )
    final_task_score = float(metrics["score"])
    metrics["combined_score"] = final_task_score
    return {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "spec": task.to_spec_dict(),
        "metrics": metrics,
        "case_score": final_task_score,
        "final_task_score": final_task_score,
        "error": error,
    }


def aggregate_task_results(task_results: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregate per-task function-minimization results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    metric_keys = (
        "best_value",
        "value_gap",
        "distance_to_optimum",
        "value_score",
        "distance_score",
        "reliability_score",
        "avg_eval_time",
        "score",
        "combined_score",
    )
    aggregate: Dict[str, float] = {}
    for key in metric_keys:
        aggregate[key] = sum(float(result["metrics"][key]) for result in normalized_results) / len(
            normalized_results
        )
    aggregate["task_count"] = float(len(normalized_results))
    aggregate["successful_task_count"] = float(
        sum(1 for result in normalized_results if not result.get("error"))
    )
    aggregate["failed_task_count"] = (
        float(len(normalized_results)) - aggregate["successful_task_count"]
    )
    return aggregate


def extract_task_result(artifacts: Mapping[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Extract one per-task artifact entry from stored evaluation artifacts."""
    evaluation_stage = artifacts.get("evaluation_stage")
    if isinstance(evaluation_stage, str) and evaluation_stage.strip().lower() != "full":
        return None

    task_results = artifacts.get("task_results")
    if not isinstance(task_results, list):
        return None

    for task_result in task_results:
        if not isinstance(task_result, Mapping):
            continue
        if task_result.get("task_id") != task_id:
            continue

        metrics = task_result.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        required_metric_keys = (
            "best_value",
            "value_gap",
            "distance_to_optimum",
            "value_score",
            "distance_score",
            "reliability_score",
            "avg_eval_time",
            "score",
            "combined_score",
        )
        for key in required_metric_keys:
            if key not in metrics:
                return None
            numeric = _coerce_finite_float(metrics.get(key), float("nan"))
            if not math.isfinite(numeric):
                return None

        task = FUNCTION_MINIMIZATION_TASKS_BY_ID.get(task_id)
        if task is None:
            return dict(task_result)
        return build_task_result(
            task,
            raw_metrics=metrics,
            error=task_result.get("error"),
        )
    return None


def project_task_artifacts(
    artifacts: Mapping[str, Any],
    task_id: str,
    task_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project shared artifacts into the task-local view used by spawned checkpoints."""
    projected: Dict[str, Any] = {
        "task_selector": task_id,
        "task_results": [dict(task_result)],
        "evaluation_mode": "task_specific",
        "evaluation_stage": "full",
    }

    trial_counts = artifacts.get("trial_counts")
    if isinstance(trial_counts, Mapping):
        task_trial_counts = trial_counts.get(task_id)
        if isinstance(task_trial_counts, Mapping):
            projected["trial_counts"] = dict(task_trial_counts)

    convergence_notes = artifacts.get("convergence_notes")
    if isinstance(convergence_notes, Mapping):
        note = convergence_notes.get(task_id)
        if isinstance(note, str):
            projected["convergence_notes"] = note

    return projected


__all__ = [
    "DEFAULT_EVALUATION_TIMEOUT_SECONDS",
    "FUNCTION_MINIMIZATION_SHARED_SELECTOR",
    "FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR",
    "FUNCTION_MINIMIZATION_TASK_SPECS",
    "FUNCTION_MINIMIZATION_TASKS_BY_ID",
    "FunctionMinimizationTaskSpec",
    "_distance_to_optimum",
    "aggregate_task_results",
    "build_task_result",
    "empty_task_metrics",
    "extract_task_result",
    "finalize_task_metrics",
    "normalize_task_metrics",
    "objective_ackley",
    "objective_rastrigin",
    "objective_rosenbrock",
    "objective_sincosxy",
    "project_task_artifacts",
    "resolve_task_specs",
    "score_task_metrics",
]
