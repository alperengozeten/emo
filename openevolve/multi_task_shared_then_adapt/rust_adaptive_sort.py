"""Canonical Rust adaptive-sort task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR = "RUST_ADAPTIVE_SORT_TASK_ID"
RUST_ADAPTIVE_SORT_SHARED_SELECTOR = "all"
DEFAULT_BUILD_TIMEOUT_SECONDS = 90
DEFAULT_RUN_TIMEOUT_SECONDS = 30

RUST_ADAPTIVE_SORT_METRIC_KEYS: tuple[str, ...] = (
    "correctness_rate",
    "speed_score",
    "consistency_score",
    "mean_speedup",
    "median_speedup",
    "candidate_avg_time",
    "reference_avg_time",
    "dataset_count",
    "successful_dataset_count",
    "score",
    "combined_score",
)


@dataclass(frozen=True)
class RustAdaptiveSortTaskSpec:
    """Stable task identity for the Rust adaptive-sort EMO-STA family."""

    task_id: str
    task_index: int
    display_name: str
    pattern: str
    dataset_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    disorder_rate: Optional[float] = None
    unique_values_by_size: Optional[Mapping[int, int]] = None
    warmup_repetitions: int = 1
    benchmark_repetitions: int = 5
    build_timeout_seconds: int = DEFAULT_BUILD_TIMEOUT_SECONDS
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS

    def to_spec_dict(self) -> Dict[str, Any]:
        spec: Dict[str, Any] = {
            "display_name": self.display_name,
            "pattern": self.pattern,
            "dataset_sizes": [int(size) for size in self.dataset_sizes],
            "seeds": [int(seed) for seed in self.seeds],
            "warmup_repetitions": int(self.warmup_repetitions),
            "benchmark_repetitions": int(self.benchmark_repetitions),
        }
        if self.disorder_rate is not None:
            spec["disorder_rate"] = float(self.disorder_rate)
        if self.unique_values_by_size is not None:
            spec["unique_values_by_size"] = {
                int(size): int(unique_values)
                for size, unique_values in self.unique_values_by_size.items()
            }
        return spec


RUST_ADAPTIVE_SORT_TASK_SPECS: tuple[RustAdaptiveSortTaskSpec, ...] = (
    RustAdaptiveSortTaskSpec(
        task_id="ras_random",
        task_index=0,
        display_name="Random",
        pattern="random",
        dataset_sizes=(1000, 10000),
        seeds=(0, 1, 2),
    ),
    RustAdaptiveSortTaskSpec(
        task_id="ras_nearly_sorted",
        task_index=1,
        display_name="NearlySorted",
        pattern="nearly_sorted",
        dataset_sizes=(1000, 10000),
        seeds=(0, 1, 2),
        disorder_rate=0.05,
    ),
    RustAdaptiveSortTaskSpec(
        task_id="ras_reverse_sorted",
        task_index=2,
        display_name="ReverseSorted",
        pattern="reverse_sorted",
        dataset_sizes=(1000, 10000),
        seeds=(),
    ),
    RustAdaptiveSortTaskSpec(
        task_id="ras_duplicates",
        task_index=3,
        display_name="Duplicates",
        pattern="duplicates",
        dataset_sizes=(1000, 10000),
        seeds=(0, 1, 2),
        unique_values_by_size={1000: 10, 10000: 100},
    ),
)

RUST_ADAPTIVE_SORT_TASKS_BY_ID: Dict[str, RustAdaptiveSortTaskSpec] = {
    task.task_id: task for task in RUST_ADAPTIVE_SORT_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[RustAdaptiveSortTaskSpec]:
    """Resolve a task selector into concrete Rust adaptive-sort task specs."""
    normalized = (selector or RUST_ADAPTIVE_SORT_SHARED_SELECTOR).strip()
    if not normalized or normalized == RUST_ADAPTIVE_SORT_SHARED_SELECTOR:
        return list(RUST_ADAPTIVE_SORT_TASK_SPECS)
    if normalized not in RUST_ADAPTIVE_SORT_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in RUST_ADAPTIVE_SORT_TASK_SPECS)
        raise ValueError(
            f"Unknown Rust adaptive-sort task '{normalized}'. Available: {available}"
        )
    return [RUST_ADAPTIVE_SORT_TASKS_BY_ID[normalized]]


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


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, numeric)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def empty_task_metrics() -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "correctness_rate": 0.0,
        "speed_score": 0.0,
        "consistency_score": 0.0,
        "mean_speedup": 0.0,
        "median_speedup": 0.0,
        "candidate_avg_time": 0.0,
        "reference_avg_time": 0.0,
        "dataset_count": 0.0,
        "successful_dataset_count": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
    }


def _normalize_dataset_summaries(
    dataset_summaries: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for summary in dataset_summaries or ():
        if not isinstance(summary, Mapping):
            continue
        candidate_median_time = _coerce_non_negative_float(
            summary.get("candidate_median_time"),
            0.0,
        )
        reference_median_time = _coerce_non_negative_float(
            summary.get("reference_median_time"),
            0.0,
        )
        speedup_ratio = (
            reference_median_time / candidate_median_time
            if candidate_median_time > 0.0
            else 0.0
        )
        seed_value = summary.get("seed")
        seed: Optional[int]
        if seed_value is None:
            seed = None
        else:
            try:
                seed = int(seed_value)
            except (TypeError, ValueError):
                seed = None
        normalized.append(
            {
                "label": str(summary.get("label") or ""),
                "size": _coerce_non_negative_int(summary.get("size"), 0),
                "seed": seed,
                "candidate_median_time": candidate_median_time,
                "reference_median_time": reference_median_time,
                "speedup_ratio": speedup_ratio,
                "sorted_correctly": bool(summary.get("sorted_correctly", False)),
            }
        )
    return normalized


def _metrics_from_dataset_summaries(
    dataset_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    if not dataset_summaries:
        return empty_task_metrics()

    speedup_ratios = [
        _coerce_non_negative_float(summary.get("speedup_ratio"), 0.0)
        for summary in dataset_summaries
    ]
    candidate_times = [
        _coerce_non_negative_float(summary.get("candidate_median_time"), 0.0)
        for summary in dataset_summaries
    ]
    reference_times = [
        _coerce_non_negative_float(summary.get("reference_median_time"), 0.0)
        for summary in dataset_summaries
    ]
    dataset_count = float(len(dataset_summaries))
    successful_dataset_count = float(
        sum(1 for summary in dataset_summaries if bool(summary.get("sorted_correctly", False)))
    )
    correctness_rate = (
        successful_dataset_count / dataset_count if dataset_count > 0.0 else 0.0
    )
    mean_speedup = statistics.fmean(speedup_ratios) if speedup_ratios else 0.0
    median_speedup = statistics.median(speedup_ratios) if speedup_ratios else 0.0
    speed_score = mean_speedup / (1.0 + mean_speedup) if mean_speedup > 0.0 else 0.0
    speedup_cv = (
        statistics.pstdev(speedup_ratios) / max(mean_speedup, 1e-9)
        if len(speedup_ratios) > 1
        else 0.0
    )
    consistency_score = 1.0 / (1.0 + speedup_cv)
    candidate_avg_time = statistics.fmean(candidate_times) if candidate_times else 0.0
    reference_avg_time = statistics.fmean(reference_times) if reference_times else 0.0
    score = (
        0.0
        if correctness_rate < 1.0
        else 0.8 * speed_score + 0.2 * consistency_score
    )
    return {
        "correctness_rate": correctness_rate,
        "speed_score": speed_score,
        "consistency_score": consistency_score,
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
        "candidate_avg_time": candidate_avg_time,
        "reference_avg_time": reference_avg_time,
        "dataset_count": dataset_count,
        "successful_dataset_count": successful_dataset_count,
        "score": score,
        "combined_score": score,
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    dataset_summaries: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, float]:
    """Normalize evaluator outputs into the stable Rust adaptive-sort metric schema."""
    defaults = empty_task_metrics()
    raw_metrics = raw_metrics or {}

    normalized_dataset_summaries = _normalize_dataset_summaries(
        dataset_summaries
        if dataset_summaries is not None
        else raw_metrics.get("dataset_summaries") or raw_metrics.get("datasets")
    )
    if normalized_dataset_summaries:
        return _metrics_from_dataset_summaries(normalized_dataset_summaries)

    correctness_rate = _clamp(
        _coerce_finite_float(raw_metrics.get("correctness_rate"), defaults["correctness_rate"]),
        0.0,
        1.0,
    )
    mean_speedup = _coerce_non_negative_float(
        raw_metrics.get("mean_speedup"),
        defaults["mean_speedup"],
    )
    median_speedup = _coerce_non_negative_float(
        raw_metrics.get("median_speedup"),
        defaults["median_speedup"],
    )
    speed_score = mean_speedup / (1.0 + mean_speedup) if mean_speedup > 0.0 else 0.0
    consistency_score = _coerce_non_negative_float(
        raw_metrics.get("consistency_score"),
        defaults["consistency_score"],
    )
    dataset_count = _coerce_non_negative_float(
        raw_metrics.get("dataset_count"),
        defaults["dataset_count"],
    )
    successful_dataset_count = _coerce_non_negative_float(
        raw_metrics.get("successful_dataset_count"),
        defaults["successful_dataset_count"],
    )
    candidate_avg_time = _coerce_non_negative_float(
        raw_metrics.get("candidate_avg_time"),
        defaults["candidate_avg_time"],
    )
    reference_avg_time = _coerce_non_negative_float(
        raw_metrics.get("reference_avg_time"),
        defaults["reference_avg_time"],
    )
    score = (
        0.0
        if correctness_rate < 1.0
        else 0.8 * speed_score + 0.2 * consistency_score
    )
    return {
        "correctness_rate": correctness_rate,
        "speed_score": speed_score,
        "consistency_score": consistency_score,
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
        "candidate_avg_time": candidate_avg_time,
        "reference_avg_time": reference_avg_time,
        "dataset_count": dataset_count,
        "successful_dataset_count": successful_dataset_count,
        "score": score,
        "combined_score": score,
    }


def build_task_result(
    task: RustAdaptiveSortTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    dataset_summaries: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    normalized_dataset_summaries = _normalize_dataset_summaries(
        dataset_summaries
        if dataset_summaries is not None
        else (raw_metrics or {}).get("dataset_summaries") or (raw_metrics or {}).get("datasets")
    )
    metrics = normalize_task_metrics(
        raw_metrics,
        dataset_summaries=normalized_dataset_summaries,
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
        "dataset_summaries": normalized_dataset_summaries,
    }


def aggregate_task_results(task_results: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregate per-task Rust adaptive-sort results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    aggregate: Dict[str, float] = {}
    for key in RUST_ADAPTIVE_SORT_METRIC_KEYS:
        aggregate[key] = sum(
            float(result["metrics"][key]) for result in normalized_results
        ) / len(normalized_results)
    successful_task_count = float(
        sum(
            1
            for result in normalized_results
            if not result.get("error")
            and float(result["metrics"]["correctness_rate"]) >= 1.0
        )
    )
    aggregate["task_count"] = float(len(normalized_results))
    aggregate["successful_task_count"] = successful_task_count
    aggregate["failed_task_count"] = float(len(normalized_results)) - successful_task_count
    return aggregate


def extract_task_result(artifacts: Mapping[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Extract one per-task artifact entry from stored evaluation artifacts."""
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
        for key in RUST_ADAPTIVE_SORT_METRIC_KEYS:
            if key not in metrics:
                return None
            numeric = _coerce_finite_float(metrics.get(key), float("nan"))
            if not math.isfinite(numeric):
                return None

        task = RUST_ADAPTIVE_SORT_TASKS_BY_ID.get(task_id)
        if task is None:
            return dict(task_result)
        return build_task_result(
            task,
            raw_metrics=metrics,
            error=task_result.get("error"),
            dataset_summaries=task_result.get("dataset_summaries"),
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
        "task_results": [dict(task_result)],
    }

    for key in (
        "compile_succeeded",
        "binary_name",
        "binary_path_name",
        "status",
    ):
        if key in artifacts:
            projected[key] = artifacts[key]

    compact_task_summary = artifacts.get("compact_task_summary")
    if isinstance(compact_task_summary, Mapping):
        summary = compact_task_summary.get(task_id)
        if isinstance(summary, Mapping):
            projected["compact_task_summary"] = {task_id: dict(summary)}

    task_execution_artifacts = artifacts.get("task_execution_artifacts")
    if isinstance(task_execution_artifacts, Mapping):
        task_execution = task_execution_artifacts.get(task_id)
        if isinstance(task_execution, Mapping):
            projected["task_execution_artifacts"] = {task_id: dict(task_execution)}

    return projected


__all__ = [
    "DEFAULT_BUILD_TIMEOUT_SECONDS",
    "DEFAULT_RUN_TIMEOUT_SECONDS",
    "RUST_ADAPTIVE_SORT_METRIC_KEYS",
    "RUST_ADAPTIVE_SORT_SHARED_SELECTOR",
    "RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR",
    "RUST_ADAPTIVE_SORT_TASK_SPECS",
    "RUST_ADAPTIVE_SORT_TASKS_BY_ID",
    "RustAdaptiveSortTaskSpec",
    "aggregate_task_results",
    "build_task_result",
    "empty_task_metrics",
    "extract_task_result",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_task_specs",
]
