"""Harder balanced K-module task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR = "K_MODULE_BALANCED_TASK_ID"
K_MODULE_BALANCED_SHARED_SELECTOR = "all"
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 30.0

K_MODULE_BALANCED_MODULE_NAMES: tuple[str, ...] = (
    "loader",
    "preprocess",
    "sampler",
    "algorithm",
    "scheduler",
    "formatter",
)
K_MODULE_BALANCED_VALID_OPTIONS: Dict[str, tuple[str, ...]] = {
    "loader": (
        "loader_0",
        "loader_1",
        "loader_2",
        "loader_3",
        "loader_4",
        "loader_5",
    ),
    "preprocess": ("prep_0", "prep_1", "prep_2", "prep_3", "prep_4", "prep_5"),
    "sampler": (
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
        "sample_5",
    ),
    "algorithm": ("algo_0", "algo_1", "algo_2", "algo_3", "algo_4", "algo_5"),
    "scheduler": (
        "sched_0",
        "sched_1",
        "sched_2",
        "sched_3",
        "sched_4",
        "sched_5",
    ),
    "formatter": ("fmt_0", "fmt_1", "fmt_2", "fmt_3", "fmt_4", "fmt_5"),
}
K_MODULE_BALANCED_TOTAL_MODULES = len(K_MODULE_BALANCED_MODULE_NAMES)


def _public_option_counts() -> Dict[str, int]:
    return {
        module_name: len(K_MODULE_BALANCED_VALID_OPTIONS[module_name])
        for module_name in K_MODULE_BALANCED_MODULE_NAMES
    }


@dataclass(frozen=True)
class KModuleBalancedTaskSpec:
    """Stable task identity for the harder balanced K-module EMO-STA family."""

    task_id: str
    task_index: int
    _target_config: Dict[str, str] = field(repr=False, compare=False)

    @property
    def target_config(self) -> Dict[str, str]:
        return dict(self._target_config)

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "module_names": list(K_MODULE_BALANCED_MODULE_NAMES),
            "num_modules": K_MODULE_BALANCED_TOTAL_MODULES,
            "option_counts": _public_option_counts(),
        }


K_MODULE_BALANCED_TASK_SPECS: tuple[KModuleBalancedTaskSpec, ...] = (
    KModuleBalancedTaskSpec(
        task_id="kmb_task_a",
        task_index=0,
        _target_config={
            "loader": "loader_0",
            "preprocess": "prep_0",
            "sampler": "sample_0",
            "algorithm": "algo_0",
            "scheduler": "sched_4",
            "formatter": "fmt_0",
        },
    ),
    KModuleBalancedTaskSpec(
        task_id="kmb_task_b",
        task_index=1,
        _target_config={
            "loader": "loader_0",
            "preprocess": "prep_3",
            "sampler": "sample_5",
            "algorithm": "algo_3",
            "scheduler": "sched_2",
            "formatter": "fmt_1",
        },
    ),
    KModuleBalancedTaskSpec(
        task_id="kmb_task_c",
        task_index=2,
        _target_config={
            "loader": "loader_2",
            "preprocess": "prep_0",
            "sampler": "sample_4",
            "algorithm": "algo_3",
            "scheduler": "sched_3",
            "formatter": "fmt_3",
        },
    ),
    KModuleBalancedTaskSpec(
        task_id="kmb_task_d",
        task_index=3,
        _target_config={
            "loader": "loader_4",
            "preprocess": "prep_2",
            "sampler": "sample_0",
            "algorithm": "algo_5",
            "scheduler": "sched_2",
            "formatter": "fmt_3",
        },
    ),
)

K_MODULE_BALANCED_TASKS_BY_ID: Dict[str, KModuleBalancedTaskSpec] = {
    task.task_id: task for task in K_MODULE_BALANCED_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[KModuleBalancedTaskSpec]:
    """Resolve a task selector into concrete task specs."""
    normalized = (selector or K_MODULE_BALANCED_SHARED_SELECTOR).strip()
    if not normalized or normalized == K_MODULE_BALANCED_SHARED_SELECTOR:
        return list(K_MODULE_BALANCED_TASK_SPECS)
    if normalized not in K_MODULE_BALANCED_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in K_MODULE_BALANCED_TASK_SPECS)
        raise ValueError(
            f"Unknown balanced K-module task '{normalized}'. Available: {available}"
        )
    return [K_MODULE_BALANCED_TASKS_BY_ID[normalized]]


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
        "total_modules": float(K_MODULE_BALANCED_TOTAL_MODULES),
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
            * K_MODULE_BALANCED_TOTAL_MODULES
        )
    )
    correct_modules = _coerce_bounded_int(
        raw_metrics.get("correct_modules"),
        default_correct_modules,
        low=0,
        high=K_MODULE_BALANCED_TOTAL_MODULES,
    )
    accuracy = correct_modules / float(K_MODULE_BALANCED_TOTAL_MODULES)
    eval_time = max(
        0.0,
        _coerce_finite_float(raw_metrics.get("eval_time"), defaults["eval_time"]),
    )
    return {
        "correct_modules": float(correct_modules),
        "total_modules": float(K_MODULE_BALANCED_TOTAL_MODULES),
        "accuracy": accuracy,
        "score": accuracy,
        "combined_score": accuracy,
        "eval_time": eval_time,
    }


def build_task_result(
    task: KModuleBalancedTaskSpec,
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
    """Aggregate per-task balanced K-module results into one shared metric payload."""
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

        task = K_MODULE_BALANCED_TASKS_BY_ID.get(task_id)
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
    """Project shared artifacts into a safe task-local view used by spawned checkpoints."""
    projected: Dict[str, Any] = {
        "task_selector": task_id,
        "task_results": [dict(task_result)],
        "evaluation_mode": "task_specific",
    }

    search_space_size_value = artifacts.get("search_space_size")
    if isinstance(search_space_size_value, (int, float)) and not isinstance(
        search_space_size_value, bool
    ):
        projected["search_space_size"] = int(search_space_size_value)
    else:
        projected["search_space_size"] = search_space_size()

    error = task_result.get("error")
    if isinstance(error, str) and error:
        projected["status"] = "error"
        projected["error"] = error
        return projected

    projected["status"] = "task_evaluation_complete"
    return projected


def count_correct_modules(
    task: KModuleBalancedTaskSpec,
    candidate_config: Mapping[str, Any],
) -> int:
    """Count matching modules for one hidden task configuration."""
    return sum(
        1
        for module_name in K_MODULE_BALANCED_MODULE_NAMES
        if candidate_config.get(module_name) == task._target_config[module_name]
    )


def validate_candidate_config(config: Any) -> List[str]:
    """Validate that a candidate configuration has the exact expected shape and values."""
    errors: List[str] = []
    if not isinstance(config, Mapping):
        return [f"Configuration must be a dict, got {type(config).__name__}"]

    expected_keys = set(K_MODULE_BALANCED_MODULE_NAMES)
    provided_keys = set(config.keys())

    missing_keys = [name for name in K_MODULE_BALANCED_MODULE_NAMES if name not in config]
    for module_name in missing_keys:
        errors.append(f"Missing required module: '{module_name}'")

    unexpected_keys = sorted(str(key) for key in provided_keys - expected_keys)
    if unexpected_keys:
        errors.append(f"Unexpected modules: {unexpected_keys}")

    for module_name in K_MODULE_BALANCED_MODULE_NAMES:
        if module_name not in config:
            continue
        candidate_value = config[module_name]
        if candidate_value not in K_MODULE_BALANCED_VALID_OPTIONS[module_name]:
            errors.append(
                f"Invalid value for '{module_name}': '{candidate_value}'. "
                f"Valid options: {list(K_MODULE_BALANCED_VALID_OPTIONS[module_name])}"
            )
    return errors


def search_space_size() -> int:
    size = 1
    for options in K_MODULE_BALANCED_VALID_OPTIONS.values():
        size *= len(options)
    return size


__all__ = [
    "DEFAULT_EVALUATION_TIMEOUT_SECONDS",
    "K_MODULE_BALANCED_MODULE_NAMES",
    "K_MODULE_BALANCED_SHARED_SELECTOR",
    "K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR",
    "K_MODULE_BALANCED_TASK_SPECS",
    "K_MODULE_BALANCED_TASKS_BY_ID",
    "K_MODULE_BALANCED_TOTAL_MODULES",
    "K_MODULE_BALANCED_VALID_OPTIONS",
    "KModuleBalancedTaskSpec",
    "aggregate_task_results",
    "build_task_result",
    "count_correct_modules",
    "empty_task_metrics",
    "extract_task_result",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_task_specs",
    "search_space_size",
    "validate_candidate_config",
]
