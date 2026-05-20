"""Canonical K-module task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


K_MODULE_TASK_SELECTOR_ENV_VAR = "K_MODULE_TASK_ID"
K_MODULE_SHARED_SELECTOR = "all"
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 30.0
K_MODULE_MODULE_NAMES: tuple[str, ...] = (
    "loader",
    "preprocess",
    "algorithm",
    "formatter",
)
K_MODULE_VALID_OPTIONS: Dict[str, tuple[str, ...]] = {
    "loader": ("csv_reader", "json_reader", "xml_reader", "parquet_reader", "sql_reader"),
    "preprocess": ("normalize", "standardize", "minmax", "scale", "none"),
    "algorithm": ("quicksort", "mergesort", "heapsort", "bubblesort", "insertion"),
    "formatter": ("json", "xml", "csv", "yaml", "protobuf"),
}
K_MODULE_TOTAL_MODULES = len(K_MODULE_MODULE_NAMES)


@dataclass(frozen=True)
class KModuleTaskSpec:
    """Stable task identity for the K-module EMO-STA family."""

    task_id: str
    task_index: int
    _target_config: Dict[str, str] = field(repr=False, compare=False)

    @property
    def target_config(self) -> Dict[str, str]:
        return dict(self._target_config)

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "module_names": list(K_MODULE_MODULE_NAMES),
            "num_modules": K_MODULE_TOTAL_MODULES,
        }


K_MODULE_TASK_SPECS: tuple[KModuleTaskSpec, ...] = (
    KModuleTaskSpec(
        task_id="km_task_a",
        task_index=0,
        _target_config={
            "loader": "csv_reader",
            "preprocess": "normalize",
            "algorithm": "quicksort",
            "formatter": "json",
        },
    ),
    KModuleTaskSpec(
        task_id="km_task_b",
        task_index=1,
        _target_config={
            "loader": "csv_reader",
            "preprocess": "normalize",
            "algorithm": "heapsort",
            "formatter": "json",
        },
    ),
    KModuleTaskSpec(
        task_id="km_task_c",
        task_index=2,
        _target_config={
            "loader": "parquet_reader",
            "preprocess": "normalize",
            "algorithm": "quicksort",
            "formatter": "json",
        },
    ),
    KModuleTaskSpec(
        task_id="km_task_d",
        task_index=3,
        _target_config={
            "loader": "csv_reader",
            "preprocess": "minmax",
            "algorithm": "quicksort",
            "formatter": "yaml",
        },
    ),
)

K_MODULE_TASKS_BY_ID: Dict[str, KModuleTaskSpec] = {
    task.task_id: task for task in K_MODULE_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[KModuleTaskSpec]:
    """Resolve a task selector into concrete task specs."""
    normalized = (selector or K_MODULE_SHARED_SELECTOR).strip()
    if not normalized or normalized == K_MODULE_SHARED_SELECTOR:
        return list(K_MODULE_TASK_SPECS)
    if normalized not in K_MODULE_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in K_MODULE_TASK_SPECS)
        raise ValueError(f"Unknown K-module task '{normalized}'. Available: {available}")
    return [K_MODULE_TASKS_BY_ID[normalized]]


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    if numeric in (float("inf"), float("-inf")):
        return default
    return numeric


def _coerce_bounded_int(value: Any, default: int, *, low: int, high: int) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, numeric))


def empty_task_metrics(
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, float]:
    """Finite failure metrics safe for ranking and logging."""
    return {
        "correct_modules": 0.0,
        "total_modules": float(K_MODULE_TOTAL_MODULES),
        "accuracy": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
        "eval_time": float(timeout_seconds),
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, float]:
    """Normalize evaluator outputs into a stable finite metric schema."""
    defaults = empty_task_metrics(timeout_seconds=timeout_seconds)
    raw_metrics = raw_metrics or {}

    default_correct_modules = int(
        round(
            _coerce_finite_float(raw_metrics.get("accuracy"), defaults["accuracy"])
            * K_MODULE_TOTAL_MODULES
        )
    )
    correct_modules = _coerce_bounded_int(
        raw_metrics.get("correct_modules"),
        default_correct_modules,
        low=0,
        high=K_MODULE_TOTAL_MODULES,
    )
    accuracy = correct_modules / float(K_MODULE_TOTAL_MODULES)
    eval_time = max(
        0.0,
        _coerce_finite_float(raw_metrics.get("eval_time"), defaults["eval_time"]),
    )
    return {
        "correct_modules": float(correct_modules),
        "total_modules": float(K_MODULE_TOTAL_MODULES),
        "accuracy": accuracy,
        "score": accuracy,
        "combined_score": accuracy,
        "eval_time": eval_time,
    }


def build_task_result(
    task: KModuleTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(raw_metrics, timeout_seconds=timeout_seconds)
    final_task_score = float(metrics["score"])
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
    """Aggregate per-task K-module results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    metric_keys = (
        "correct_modules",
        "total_modules",
        "accuracy",
        "score",
        "combined_score",
        "eval_time",
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
            "correct_modules",
            "total_modules",
            "accuracy",
            "score",
            "combined_score",
            "eval_time",
        )
        for key in required_metric_keys:
            if key not in metrics:
                return None
            numeric = _coerce_finite_float(metrics.get(key), float("nan"))
            if numeric != numeric:
                return None

        task = K_MODULE_TASKS_BY_ID.get(task_id)
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
    }

    search_space_size_value = artifacts.get("search_space_size")
    if isinstance(search_space_size_value, (int, float)) and not isinstance(
        search_space_size_value, bool
    ):
        projected["search_space_size"] = int(search_space_size_value)

    candidate_configuration = artifacts.get("candidate_configuration")
    if isinstance(candidate_configuration, str):
        projected["candidate_configuration"] = candidate_configuration

    error = task_result.get("error")
    if isinstance(error, str) and error:
        projected["status"] = "ERROR"
        projected["suggestion"] = "Fix the program so it returns a valid pipeline configuration."
        projected["error"] = error
        return projected

    metrics = task_result.get("metrics")
    if not isinstance(metrics, Mapping):
        return projected

    correct_modules = metrics.get("correct_modules")
    total_modules = metrics.get("total_modules")
    if isinstance(correct_modules, (int, float)) and isinstance(total_modules, (int, float)):
        projected["status"] = f"{int(correct_modules)}/{int(total_modules)} modules correct."
        projected["suggestion"] = "Try different module combinations to improve the score."

    return projected


def count_correct_modules(
    task: KModuleTaskSpec,
    candidate_config: Mapping[str, Any],
) -> int:
    """Count matching modules for one hidden task configuration."""
    return sum(
        1
        for module_name in K_MODULE_MODULE_NAMES
        if candidate_config.get(module_name) == task._target_config[module_name]
    )


def validate_candidate_config(config: Any) -> List[str]:
    """Validate that a candidate configuration has the expected shape and values."""
    errors: List[str] = []
    if not isinstance(config, Mapping):
        return [f"Configuration must be a dict, got {type(config).__name__}"]

    for module_name in K_MODULE_MODULE_NAMES:
        if module_name not in config:
            errors.append(f"Missing required module: '{module_name}'")
            continue
        candidate_value = config[module_name]
        if candidate_value not in K_MODULE_VALID_OPTIONS[module_name]:
            errors.append(
                f"Invalid value for '{module_name}': '{candidate_value}'. "
                f"Valid options: {list(K_MODULE_VALID_OPTIONS[module_name])}"
            )
    return errors


def search_space_size() -> int:
    size = 1
    for options in K_MODULE_VALID_OPTIONS.values():
        size *= len(options)
    return size
