"""Canonical circle-packing task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR = "CIRCLE_PACKING_TASK_ID"
CIRCLE_PACKING_SHARED_SELECTOR = "all"
DEFAULT_STAGE1_TIMEOUT_SECONDS = 60
DEFAULT_FULL_TIMEOUT_SECONDS = 180

CIRCLE_PACKING_METRIC_KEYS: tuple[str, ...] = (
    "sum_radii",
    "target_sum_radii",
    "target_ratio",
    "validity",
    "radius_variance",
    "spatial_spread",
    "min_radius",
    "max_radius",
    "eval_time",
    "score",
    "combined_score",
)


@dataclass(frozen=True)
class CirclePackingTaskSpec:
    """Stable task identity for the circle-packing EMO-STA family."""

    task_id: str
    task_index: int
    display_name: str
    n_circles: int
    target_sum_radii: float
    container_name: str = "unit_square"
    timeout_seconds_stage1: int = DEFAULT_STAGE1_TIMEOUT_SECONDS
    timeout_seconds_full: int = DEFAULT_FULL_TIMEOUT_SECONDS

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "n_circles": int(self.n_circles),
            "container_name": self.container_name,
            "target_sum_radii": float(self.target_sum_radii),
        }


CIRCLE_PACKING_TASK_SPECS: tuple[CirclePackingTaskSpec, ...] = (
    CirclePackingTaskSpec(
        task_id="cp_n20",
        task_index=0,
        display_name="CirclePackingN20",
        n_circles=20,
        target_sum_radii=2.301,
    ),
    CirclePackingTaskSpec(
        task_id="cp_n22",
        task_index=1,
        display_name="CirclePackingN22",
        n_circles=22,
        target_sum_radii=2.420,
    ),
    CirclePackingTaskSpec(
        task_id="cp_n24",
        task_index=2,
        display_name="CirclePackingN24",
        n_circles=24,
        target_sum_radii=2.530,
    ),
    CirclePackingTaskSpec(
        task_id="cp_n26",
        task_index=3,
        display_name="CirclePackingN26",
        n_circles=26,
        target_sum_radii=2.635,
    ),
)

CIRCLE_PACKING_TASKS_BY_ID: Dict[str, CirclePackingTaskSpec] = {
    task.task_id: task for task in CIRCLE_PACKING_TASK_SPECS
}

CIRCLE_PACKING_HOLDOUT_TASK_SPECS: tuple[CirclePackingTaskSpec, ...] = (
    CirclePackingTaskSpec(
        task_id="cp_n21",
        task_index=4,
        display_name="CirclePackingN21",
        n_circles=21,
        target_sum_radii=2.362,
    ),
    CirclePackingTaskSpec(
        task_id="cp_n23",
        task_index=5,
        display_name="CirclePackingN23",
        n_circles=23,
        target_sum_radii=2.478,
    ),
    CirclePackingTaskSpec(
        task_id="cp_n25",
        task_index=6,
        display_name="CirclePackingN25",
        n_circles=25,
        target_sum_radii=2.587,
    ),
)

CIRCLE_PACKING_HOLDOUT_TASKS_BY_ID: Dict[str, CirclePackingTaskSpec] = {
    task.task_id: task for task in CIRCLE_PACKING_HOLDOUT_TASK_SPECS
}

_ALL_CIRCLE_PACKING_TASKS_BY_ID: Dict[str, CirclePackingTaskSpec] = {
    **CIRCLE_PACKING_TASKS_BY_ID,
    **CIRCLE_PACKING_HOLDOUT_TASKS_BY_ID,
}


def resolve_task_specs(selector: Optional[str]) -> List[CirclePackingTaskSpec]:
    """Resolve a task selector into concrete circle-packing task specs."""
    normalized = (selector or CIRCLE_PACKING_SHARED_SELECTOR).strip()
    if not normalized or normalized == CIRCLE_PACKING_SHARED_SELECTOR:
        return list(CIRCLE_PACKING_TASK_SPECS)
    if normalized not in CIRCLE_PACKING_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in CIRCLE_PACKING_TASK_SPECS)
        raise ValueError(
            f"Unknown circle-packing task '{normalized}'. Available: {available}"
        )
    return [CIRCLE_PACKING_TASKS_BY_ID[normalized]]


def resolve_holdout_task_specs(selector: Optional[str]) -> List[CirclePackingTaskSpec]:
    """Resolve evaluation-only circle-packing holdout task selectors."""
    normalized = (selector or "all").strip()
    if not normalized or normalized in {"all", "all_holdouts"}:
        return list(CIRCLE_PACKING_HOLDOUT_TASK_SPECS)

    requested_ids = [task_id.strip() for task_id in normalized.split(",") if task_id.strip()]
    if not requested_ids:
        return list(CIRCLE_PACKING_HOLDOUT_TASK_SPECS)

    resolved: List[CirclePackingTaskSpec] = []
    seen_task_ids: set[str] = set()
    for task_id in requested_ids:
        if task_id in CIRCLE_PACKING_TASKS_BY_ID:
            raise ValueError(
                f"Circle-packing task '{task_id}' is part of the seen training family, "
                "not the evaluation-only holdout set"
            )
        task = CIRCLE_PACKING_HOLDOUT_TASKS_BY_ID.get(task_id)
        if task is None:
            available = ", ".join(task.task_id for task in CIRCLE_PACKING_HOLDOUT_TASK_SPECS)
            raise ValueError(
                f"Unknown circle-packing holdout task '{task_id}'. Available holdouts: {available}"
            )
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        resolved.append(task)
    return resolved


def all_circle_packing_task_specs() -> List[CirclePackingTaskSpec]:
    """Return the combined seen-task and holdout circle-packing task specs."""
    return [*CIRCLE_PACKING_TASK_SPECS, *CIRCLE_PACKING_HOLDOUT_TASK_SPECS]


def all_circle_packing_holdout_task_ids() -> List[str]:
    """Return the stable ordered list of evaluation-only holdout task ids."""
    return [task.task_id for task in CIRCLE_PACKING_HOLDOUT_TASK_SPECS]


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


def empty_task_metrics(*, target_sum_radii: float = 0.0) -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "sum_radii": 0.0,
        "target_sum_radii": float(target_sum_radii),
        "target_ratio": 0.0,
        "validity": 0.0,
        "radius_variance": 0.0,
        "spatial_spread": 0.0,
        "min_radius": 0.0,
        "max_radius": 0.0,
        "eval_time": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    target_sum_radii: float,
) -> Dict[str, float]:
    """Normalize evaluator outputs into the stable circle-packing metric schema."""
    defaults = empty_task_metrics(target_sum_radii=target_sum_radii)
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

    sum_radii = _coerce_non_negative_float(raw_metrics.get("sum_radii"), defaults["sum_radii"])
    target_ratio = (
        sum_radii / float(target_sum_radii) if float(target_sum_radii) > 0.0 else 0.0
    )
    score = target_ratio * validity
    return {
        "sum_radii": sum_radii,
        "target_sum_radii": float(target_sum_radii),
        "target_ratio": target_ratio,
        "validity": validity,
        "radius_variance": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("radius_variance"),
                defaults["radius_variance"],
            ),
            0.0,
            1.0,
        ),
        "spatial_spread": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("spatial_spread"),
                defaults["spatial_spread"],
            ),
            0.0,
            1.0,
        ),
        "min_radius": _coerce_non_negative_float(
            raw_metrics.get("min_radius"),
            defaults["min_radius"],
        ),
        "max_radius": _coerce_non_negative_float(
            raw_metrics.get("max_radius"),
            defaults["max_radius"],
        ),
        "eval_time": _coerce_non_negative_float(
            raw_metrics.get("eval_time"),
            defaults["eval_time"],
        ),
        "score": score,
        "combined_score": score,
    }


def build_task_result(
    task: CirclePackingTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    validation_summary: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(
        raw_metrics,
        target_sum_radii=task.target_sum_radii,
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
    """Aggregate per-task circle-packing results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    aggregate: Dict[str, float] = {}
    for key in CIRCLE_PACKING_METRIC_KEYS:
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
            "sum_radii",
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

        task = _ALL_CIRCLE_PACKING_TASKS_BY_ID.get(task_id)
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
    "CIRCLE_PACKING_HOLDOUT_TASK_SPECS",
    "CIRCLE_PACKING_HOLDOUT_TASKS_BY_ID",
    "CIRCLE_PACKING_METRIC_KEYS",
    "CIRCLE_PACKING_SHARED_SELECTOR",
    "CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR",
    "CIRCLE_PACKING_TASK_SPECS",
    "CIRCLE_PACKING_TASKS_BY_ID",
    "CirclePackingTaskSpec",
    "all_circle_packing_holdout_task_ids",
    "all_circle_packing_task_specs",
    "aggregate_task_results",
    "build_task_result",
    "empty_task_metrics",
    "extract_task_result",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_holdout_task_specs",
    "resolve_task_specs",
]
