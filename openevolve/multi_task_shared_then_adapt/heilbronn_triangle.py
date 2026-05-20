"""Canonical Heilbronn-triangle task family definitions for EMO-STA."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR = "HEILBRONN_TRIANGLE_TASK_ID"
HEILBRONN_TRIANGLE_SHARED_SELECTOR = "all"
DEFAULT_STAGE1_TIMEOUT_SECONDS = 30
DEFAULT_FULL_TIMEOUT_SECONDS = 90

HEILBRONN_TRIANGLE_METRIC_KEYS: tuple[str, ...] = (
    "min_triangle_area",
    "target_min_area",
    "target_ratio",
    "validity",
    "point_spread",
    "boundary_utilization",
    "min_pair_distance",
    "eval_time",
    "score",
    "combined_score",
)


@dataclass(frozen=True)
class HeilbronnTriangleTaskSpec:
    """Stable task identity for the Heilbronn-triangle EMO-STA family."""

    task_id: str
    task_index: int
    display_name: str
    n_points: int
    target_min_area: float
    container_name: str = "canonical_unit_area_triangle"
    timeout_seconds_stage1: int = DEFAULT_STAGE1_TIMEOUT_SECONDS
    timeout_seconds_full: int = DEFAULT_FULL_TIMEOUT_SECONDS

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "n_points": int(self.n_points),
            "container_name": self.container_name,
            "target_min_area": float(self.target_min_area),
        }


HEILBRONN_TRIANGLE_TRAIN_TASK_SPECS: tuple[HeilbronnTriangleTaskSpec, ...] = (
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n9",
        task_index=0,
        display_name="HeilbronnTriangleN9",
        n_points=9,
        target_min_area=0.0548469387755102,
    ),
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n10",
        task_index=1,
        display_name="HeilbronnTriangleN10",
        n_points=10,
        target_min_area=0.04337673349889024,
    ),
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n11",
        task_index=2,
        display_name="HeilbronnTriangleN11",
        n_points=11,
        target_min_area=0.03609267801015405,
    ),
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n12",
        task_index=3,
        display_name="HeilbronnTriangleN12",
        n_points=12,
        target_min_area=0.03100478174352528,
    ),
)

# Backwards-compatible name for the in-distribution training task set.
HEILBRONN_TRIANGLE_TASK_SPECS = HEILBRONN_TRIANGLE_TRAIN_TASK_SPECS

HEILBRONN_TRIANGLE_TASKS_BY_ID: Dict[str, HeilbronnTriangleTaskSpec] = {
    task.task_id: task for task in HEILBRONN_TRIANGLE_TASK_SPECS
}

HEILBRONN_TRIANGLE_OOD_TASK_SPECS: tuple[HeilbronnTriangleTaskSpec, ...] = (
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n8",
        task_index=4,
        display_name="HeilbronnTriangleN8",
        n_points=8,
        target_min_area=0.06778914101959856,
    ),
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n13",
        task_index=5,
        display_name="HeilbronnTriangleN13",
        n_points=13,
        target_min_area=0.02456425934867466,
    ),
)

HEILBRONN_TRIANGLE_ADDITIONAL_OOD_TASK_SPECS: tuple[
    HeilbronnTriangleTaskSpec, ...
] = (
    HeilbronnTriangleTaskSpec(
        task_id="heil_tri_n14",
        task_index=6,
        display_name="HeilbronnTriangleN14",
        n_points=14,
        target_min_area=0.02377577301721215,
    ),
)

HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS: tuple[HeilbronnTriangleTaskSpec, ...] = (
    *HEILBRONN_TRIANGLE_OOD_TASK_SPECS,
    *HEILBRONN_TRIANGLE_ADDITIONAL_OOD_TASK_SPECS,
)

HEILBRONN_TRIANGLE_OOD_TASKS_BY_ID: Dict[str, HeilbronnTriangleTaskSpec] = {
    task.task_id: task for task in HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS
}

HEILBRONN_TRIANGLE_ALL_EVAL_TASK_SPECS: tuple[HeilbronnTriangleTaskSpec, ...] = (
    *HEILBRONN_TRIANGLE_TRAIN_TASK_SPECS,
    *HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS,
)

HEILBRONN_TRIANGLE_ALL_EVAL_TASKS_BY_ID: Dict[str, HeilbronnTriangleTaskSpec] = {
    task.task_id: task for task in HEILBRONN_TRIANGLE_ALL_EVAL_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[HeilbronnTriangleTaskSpec]:
    """Resolve a task selector into concrete Heilbronn-triangle task specs."""
    normalized = (selector or HEILBRONN_TRIANGLE_SHARED_SELECTOR).strip()
    if not normalized or normalized == HEILBRONN_TRIANGLE_SHARED_SELECTOR:
        return list(HEILBRONN_TRIANGLE_TASK_SPECS)
    if normalized not in HEILBRONN_TRIANGLE_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in HEILBRONN_TRIANGLE_TASK_SPECS)
        raise ValueError(
            f"Unknown Heilbronn-triangle task '{normalized}'. Available: {available}"
        )
    return [HEILBRONN_TRIANGLE_TASKS_BY_ID[normalized]]


def resolve_eval_task_specs(selector: Optional[str]) -> List[HeilbronnTriangleTaskSpec]:
    """Resolve evaluator selectors, allowing explicit OOD task IDs.

    The shared selector remains training-only so normal EMO-STA runs keep the
    original in-distribution objective.
    """
    normalized = (selector or HEILBRONN_TRIANGLE_SHARED_SELECTOR).strip()
    if not normalized or normalized == HEILBRONN_TRIANGLE_SHARED_SELECTOR:
        return list(HEILBRONN_TRIANGLE_TRAIN_TASK_SPECS)
    if normalized not in HEILBRONN_TRIANGLE_ALL_EVAL_TASKS_BY_ID:
        available = ", ".join(
            task.task_id for task in HEILBRONN_TRIANGLE_ALL_EVAL_TASK_SPECS
        )
        raise ValueError(
            f"Unknown Heilbronn-triangle evaluation task '{normalized}'. "
            f"Available: {available}"
        )
    return [HEILBRONN_TRIANGLE_ALL_EVAL_TASKS_BY_ID[normalized]]


def resolve_ood_task_specs(selector: Optional[str]) -> List[HeilbronnTriangleTaskSpec]:
    """Resolve post-hoc OOD task selectors for Heilbronn triangle."""
    normalized = (selector or "all_ood").strip()
    if not normalized or normalized in {"all", "all_ood", "ood"}:
        return list(HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS)

    requested_ids = [task_id.strip() for task_id in normalized.split(",") if task_id.strip()]
    if not requested_ids:
        return list(HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS)

    resolved: List[HeilbronnTriangleTaskSpec] = []
    seen_task_ids: set[str] = set()
    for task_id in requested_ids:
        if task_id in HEILBRONN_TRIANGLE_TASKS_BY_ID:
            raise ValueError(
                f"Heilbronn-triangle task '{task_id}' is part of the seen training "
                "family, not the post-hoc OOD set"
            )
        task = HEILBRONN_TRIANGLE_OOD_TASKS_BY_ID.get(task_id)
        if task is None:
            available = ", ".join(
                task.task_id for task in HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS
            )
            raise ValueError(
                f"Unknown Heilbronn-triangle OOD task '{task_id}'. "
                f"Available OOD tasks: {available}"
            )
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        resolved.append(task)
    return resolved


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _coerce_non_negative_float(value: Any, default: float) -> float:
    return max(0.0, _coerce_finite_float(value, default))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def empty_task_metrics(*, target_min_area: float = 0.0) -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "min_triangle_area": 0.0,
        "target_min_area": float(target_min_area),
        "target_ratio": 0.0,
        "validity": 0.0,
        "point_spread": 0.0,
        "boundary_utilization": 0.0,
        "min_pair_distance": 0.0,
        "eval_time": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    target_min_area: float,
) -> Dict[str, float]:
    """Normalize evaluator outputs into the stable Heilbronn metric schema."""
    defaults = empty_task_metrics(target_min_area=target_min_area)
    if not isinstance(raw_metrics, Mapping):
        return defaults

    validity = 1.0 if _coerce_finite_float(raw_metrics.get("validity"), 0.0) >= 0.5 else 0.0
    if validity <= 0.0:
        metrics = dict(defaults)
        metrics["eval_time"] = _coerce_non_negative_float(
            raw_metrics.get("eval_time"),
            defaults["eval_time"],
        )
        return metrics

    min_triangle_area = _coerce_non_negative_float(
        raw_metrics.get("min_triangle_area"),
        defaults["min_triangle_area"],
    )
    target_ratio = (
        min_triangle_area / float(target_min_area) if float(target_min_area) > 0.0 else 0.0
    )
    score = target_ratio * validity
    return {
        "min_triangle_area": min_triangle_area,
        "target_min_area": float(target_min_area),
        "target_ratio": target_ratio,
        "validity": validity,
        "point_spread": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("point_spread"),
                defaults["point_spread"],
            ),
            0.0,
            1.0,
        ),
        "boundary_utilization": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("boundary_utilization"),
                defaults["boundary_utilization"],
            ),
            0.0,
            1.0,
        ),
        "min_pair_distance": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("min_pair_distance"),
                defaults["min_pair_distance"],
            ),
            0.0,
            1.0,
        ),
        "eval_time": _coerce_non_negative_float(
            raw_metrics.get("eval_time"),
            defaults["eval_time"],
        ),
        "score": score,
        "combined_score": score,
    }


def build_task_result(
    task: HeilbronnTriangleTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    validation_summary: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(
        raw_metrics,
        target_min_area=task.target_min_area,
    )
    final_task_score = float(metrics["score"])
    metrics["combined_score"] = final_task_score
    task_result: Dict[str, Any] = {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "spec": task.to_spec_dict(),
        "metrics": metrics,
        "case_score": final_task_score,
        "final_task_score": final_task_score,
        "error": error,
    }
    if isinstance(validation_summary, Mapping):
        task_result["validation_summary"] = dict(validation_summary)
    return task_result


def aggregate_task_results(task_results: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregate per-task Heilbronn results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    aggregate: Dict[str, float] = {}
    for key in HEILBRONN_TRIANGLE_METRIC_KEYS:
        aggregate[key] = sum(
            float(result["metrics"][key]) for result in normalized_results
        ) / len(normalized_results)
    aggregate["task_count"] = float(len(normalized_results))
    aggregate["successful_task_count"] = float(
        sum(1 for result in normalized_results if not result.get("error"))
    )
    aggregate["failed_task_count"] = (
        float(len(normalized_results)) - aggregate["successful_task_count"]
    )
    aggregate["combined_score"] = float(aggregate["score"])
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
            "min_triangle_area",
            "target_ratio",
            "validity",
            "score",
            "combined_score",
        )
        for key in required_metric_keys:
            if key not in metrics:
                return None
            numeric = _coerce_finite_float(metrics.get(key), float("nan"))
            if not math.isfinite(numeric):
                return None

        task = HEILBRONN_TRIANGLE_ALL_EVAL_TASKS_BY_ID.get(task_id)
        if task is None:
            return dict(task_result)
        return build_task_result(
            task,
            raw_metrics=metrics,
            error=task_result.get("error"),
            validation_summary=task_result.get("validation_summary"),
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
        "selected_task_ids": [task_id],
        "evaluation_mode": "task_specific",
        "evaluation_stage": "full",
        "task_results": [dict(task_result)],
    }

    validation_summaries = artifacts.get("validation_summaries")
    if isinstance(validation_summaries, Mapping):
        task_summary = validation_summaries.get(task_id)
        if isinstance(task_summary, Mapping):
            projected["validation_summaries"] = {task_id: dict(task_summary)}

    compact_task_summary = artifacts.get("compact_task_summary")
    if isinstance(compact_task_summary, Mapping):
        task_summary = compact_task_summary.get(task_id)
        if isinstance(task_summary, Mapping):
            projected["compact_task_summary"] = {task_id: dict(task_summary)}

    subprocess_timeout_by_task = artifacts.get("subprocess_timeout_by_task")
    if isinstance(subprocess_timeout_by_task, Mapping) and task_id in subprocess_timeout_by_task:
        projected["subprocess_timeout_by_task"] = {
            task_id: bool(subprocess_timeout_by_task[task_id])
        }

    execution_summary = artifacts.get("execution_summary")
    if isinstance(execution_summary, Mapping):
        projected["execution_summary"] = {
            "selected_task_count": 1,
            "timed_out_task_count": int(
                bool(
                    (artifacts.get("subprocess_timeout_by_task") or {}).get(task_id, False)
                )
            ),
        }

    best_local_summary = artifacts.get("best_local_summary")
    if isinstance(best_local_summary, Mapping) and best_local_summary.get("task_id") == task_id:
        projected["best_local_summary"] = dict(best_local_summary)

    return projected


__all__ = [
    "DEFAULT_FULL_TIMEOUT_SECONDS",
    "DEFAULT_STAGE1_TIMEOUT_SECONDS",
    "HEILBRONN_TRIANGLE_METRIC_KEYS",
    "HEILBRONN_TRIANGLE_SHARED_SELECTOR",
    "HEILBRONN_TRIANGLE_ADDITIONAL_OOD_TASK_SPECS",
    "HEILBRONN_TRIANGLE_ALL_EVAL_TASK_SPECS",
    "HEILBRONN_TRIANGLE_ALL_EVAL_TASKS_BY_ID",
    "HEILBRONN_TRIANGLE_ALL_OOD_TASK_SPECS",
    "HEILBRONN_TRIANGLE_OOD_TASK_SPECS",
    "HEILBRONN_TRIANGLE_OOD_TASKS_BY_ID",
    "HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR",
    "HEILBRONN_TRIANGLE_TASK_SPECS",
    "HEILBRONN_TRIANGLE_TASKS_BY_ID",
    "HEILBRONN_TRIANGLE_TRAIN_TASK_SPECS",
    "HeilbronnTriangleTaskSpec",
    "aggregate_task_results",
    "build_task_result",
    "empty_task_metrics",
    "extract_task_result",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_eval_task_specs",
    "resolve_ood_task_specs",
    "resolve_task_specs",
]
