"""Canonical hexagon-packing task family definitions for EMO-STA."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR = "HEXAGON_PACKING_TASK_ID"
HEXAGON_PACKING_SHARED_SELECTOR = "all"
DEFAULT_STAGE1_TIMEOUT_SECONDS = 60
DEFAULT_FULL_TIMEOUT_SECONDS = 180
INNER_HEX_SIDE_LENGTH = 1.0
OUTER_SIDE_LENGTH_TOLERANCE = 1.0e-6

HEXAGON_PACKING_METRIC_KEYS: tuple[str, ...] = (
    "outer_side_length",
    "target_outer_side_length",
    "target_ratio",
    "inv_outer_side_length",
    "validity",
    "center_spread",
    "angle_spread",
    "min_center_distance",
    "eval_time",
    "score",
    "combined_score",
)


@dataclass(frozen=True)
class HexagonPackingTaskSpec:
    """Stable task identity for the hexagon-packing EMO-STA family."""

    task_id: str
    task_index: int
    display_name: str
    n_hexagons: int
    target_outer_side_length: float
    container_name: str = "regular_hexagon"
    inner_side_length: float = 1.0
    timeout_seconds_stage1: int = DEFAULT_STAGE1_TIMEOUT_SECONDS
    timeout_seconds_full: int = DEFAULT_FULL_TIMEOUT_SECONDS

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "n_hexagons": int(self.n_hexagons),
            "container_name": self.container_name,
            "inner_side_length": float(self.inner_side_length),
            "target_outer_side_length": float(self.target_outer_side_length),
        }


HEXAGON_PACKING_TASK_SPECS: tuple[HexagonPackingTaskSpec, ...] = (
    HexagonPackingTaskSpec(
        task_id="hex_pack_n10",
        task_index=0,
        display_name="HexagonPackingN10",
        n_hexagons=10,
        target_outer_side_length=3.7320508075688772,
    ),
    HexagonPackingTaskSpec(
        task_id="hex_pack_n11",
        task_index=1,
        display_name="HexagonPackingN11",
        n_hexagons=11,
        target_outer_side_length=3.9245008972987525,
    ),
    HexagonPackingTaskSpec(
        task_id="hex_pack_n12",
        task_index=2,
        display_name="HexagonPackingN12",
        n_hexagons=12,
        target_outer_side_length=3.94164,
    ),
    HexagonPackingTaskSpec(
        task_id="hex_pack_n13",
        task_index=3,
        display_name="HexagonPackingN13",
        n_hexagons=13,
        target_outer_side_length=4.0,
    ),
)

HEXAGON_PACKING_TASKS_BY_ID: Dict[str, HexagonPackingTaskSpec] = {
    task.task_id: task for task in HEXAGON_PACKING_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[HexagonPackingTaskSpec]:
    """Resolve a task selector into concrete hexagon-packing task specs."""
    normalized = (selector or HEXAGON_PACKING_SHARED_SELECTOR).strip()
    if not normalized or normalized == HEXAGON_PACKING_SHARED_SELECTOR:
        return list(HEXAGON_PACKING_TASK_SPECS)
    if normalized not in HEXAGON_PACKING_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in HEXAGON_PACKING_TASK_SPECS)
        raise ValueError(
            f"Unknown hexagon-packing task '{normalized}'. Available: {available}"
        )
    return [HEXAGON_PACKING_TASKS_BY_ID[normalized]]


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


def minimum_outer_side_length_for_area(
    n_hexagons: int,
    *,
    inner_side_length: float = INNER_HEX_SIDE_LENGTH,
) -> float:
    """Area-based lower bound for packing n unit regular hexagons."""
    n = max(0, int(n_hexagons))
    return math.sqrt(float(n)) * float(inner_side_length)


def empty_task_metrics(*, target_outer_side_length: float = 0.0) -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "outer_side_length": 0.0,
        "target_outer_side_length": float(target_outer_side_length),
        "target_ratio": 0.0,
        "inv_outer_side_length": 0.0,
        "validity": 0.0,
        "center_spread": 0.0,
        "angle_spread": 0.0,
        "min_center_distance": 0.0,
        "eval_time": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    target_outer_side_length: float,
    n_hexagons: int | None = None,
    inner_side_length: float = INNER_HEX_SIDE_LENGTH,
) -> Dict[str, float]:
    """Normalize evaluator outputs into the stable hexagon-packing metric schema."""
    defaults = empty_task_metrics(target_outer_side_length=target_outer_side_length)
    if not isinstance(raw_metrics, Mapping):
        return defaults

    validity = 1.0 if _coerce_finite_float(raw_metrics.get("validity"), 0.0) >= 0.5 else 0.0
    outer_side_length = _coerce_finite_float(
        raw_metrics.get("outer_side_length"),
        float("nan"),
    )
    if validity <= 0.0 or not math.isfinite(outer_side_length) or outer_side_length <= 0.0:
        metrics = dict(defaults)
        metrics["eval_time"] = _coerce_non_negative_float(
            raw_metrics.get("eval_time"),
            defaults["eval_time"],
        )
        return metrics
    if n_hexagons is not None:
        lower_bound = minimum_outer_side_length_for_area(
            int(n_hexagons),
            inner_side_length=float(inner_side_length),
        )
        if outer_side_length + OUTER_SIDE_LENGTH_TOLERANCE < lower_bound:
            metrics = dict(defaults)
            metrics["eval_time"] = _coerce_non_negative_float(
                raw_metrics.get("eval_time"),
                defaults["eval_time"],
            )
            return metrics

    target_ratio = float(target_outer_side_length) / float(outer_side_length)
    inv_outer_side_length = 1.0 / float(outer_side_length)
    score = target_ratio * validity
    return {
        "outer_side_length": float(outer_side_length),
        "target_outer_side_length": float(target_outer_side_length),
        "target_ratio": float(target_ratio),
        "inv_outer_side_length": float(inv_outer_side_length),
        "validity": float(validity),
        "center_spread": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("center_spread"),
                defaults["center_spread"],
            ),
            0.0,
            1.0,
        ),
        "angle_spread": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("angle_spread"),
                defaults["angle_spread"],
            ),
            0.0,
            1.0,
        ),
        "min_center_distance": _clamp(
            _coerce_non_negative_float(
                raw_metrics.get("min_center_distance"),
                defaults["min_center_distance"],
            ),
            0.0,
            1.0,
        ),
        "eval_time": _coerce_non_negative_float(
            raw_metrics.get("eval_time"),
            defaults["eval_time"],
        ),
        "score": float(score),
        "combined_score": float(score),
    }


def build_task_result(
    task: HexagonPackingTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    validation_summary: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(
        raw_metrics,
        target_outer_side_length=task.target_outer_side_length,
        n_hexagons=task.n_hexagons,
        inner_side_length=task.inner_side_length,
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
    """Aggregate per-task hexagon-packing results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    aggregate: Dict[str, float] = {}
    for key in HEXAGON_PACKING_METRIC_KEYS:
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
            "outer_side_length",
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
        if _coerce_finite_float(metrics.get("validity"), 0.0) >= 0.5:
            outer_side_length = _coerce_finite_float(
                metrics.get("outer_side_length"),
                float("nan"),
            )
            if not math.isfinite(outer_side_length) or outer_side_length <= 0.0:
                return None

        task = HEXAGON_PACKING_TASKS_BY_ID.get(task_id)
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
    "HEXAGON_PACKING_METRIC_KEYS",
    "HEXAGON_PACKING_SHARED_SELECTOR",
    "HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR",
    "HEXAGON_PACKING_TASK_SPECS",
    "HEXAGON_PACKING_TASKS_BY_ID",
    "INNER_HEX_SIDE_LENGTH",
    "OUTER_SIDE_LENGTH_TOLERANCE",
    "HexagonPackingTaskSpec",
    "aggregate_task_results",
    "build_task_result",
    "empty_task_metrics",
    "extract_task_result",
    "minimum_outer_side_length_for_area",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_task_specs",
]
