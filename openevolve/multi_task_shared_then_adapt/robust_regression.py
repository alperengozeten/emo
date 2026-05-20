"""Canonical robust-regression task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


ROBUST_REGRESSION_TASK_SELECTOR_ENV_VAR = "R_ROBUST_TASK_ID"
ROBUST_REGRESSION_SHARED_SELECTOR = "all"
STAGE1_SEEDS: tuple[int, ...] = (0, 1)
FULL_EVAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_STAGE1_TIMEOUT_SECONDS = 20.0
DEFAULT_FULL_TIMEOUT_SECONDS = 30.0
DEFAULT_EVALUATION_TIMEOUT_SECONDS = DEFAULT_FULL_TIMEOUT_SECONDS

ROBUST_REGRESSION_REQUIRED_TASK_METRICS: tuple[str, ...] = (
    "nmse_signal_test",
    "nmse_noisy_test",
    "coef_rel_error",
    "signal_score",
    "noisy_score",
    "coef_score",
    "r2_signal_test",
    "r2_noisy_test",
    "success_rate",
    "seed_count",
    "successful_seed_count",
    "avg_exec_time",
    "score",
    "combined_score",
)

ROBUST_REGRESSION_SUCCESSFUL_SEED_METRIC_KEYS: tuple[str, ...] = (
    "mse_signal_test",
    "mae_signal_test",
    "r2_signal_test",
    "nmse_signal_test",
    "signal_score",
    "mse_noisy_test",
    "mae_noisy_test",
    "r2_noisy_test",
    "nmse_noisy_test",
    "noisy_score",
    "coef_rel_error",
    "coef_score",
)


@dataclass(frozen=True)
class RobustRegressionTaskSpec:
    """Stable task identity for the robust-regression family."""

    task_id: str
    task_index: int
    n_train: int
    n_test: int
    n_features: int
    noise_std: float
    rho: float
    vertical_outlier_fraction_train: float
    leverage_outlier_fraction_train: float
    hetero_strength: float
    evaluation_seeds_full: tuple[int, ...] = FULL_EVAL_SEEDS
    evaluation_seeds_stage1: tuple[int, ...] = STAGE1_SEEDS
    trial_timeout_seconds_full: float = DEFAULT_FULL_TIMEOUT_SECONDS
    trial_timeout_seconds_stage1: float = DEFAULT_STAGE1_TIMEOUT_SECONDS

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "noise_std": self.noise_std,
            "rho": self.rho,
            "vertical_outlier_fraction_train": self.vertical_outlier_fraction_train,
            "leverage_outlier_fraction_train": self.leverage_outlier_fraction_train,
            "hetero_strength": self.hetero_strength,
        }


@dataclass(frozen=True)
class RobustRegressionDataset:
    """Synthetic train/test regression case for one task and seed."""

    task_id: str
    task_index: int
    base_seed: int
    derived_seed: int
    true_coefficients: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test_clean_signal: np.ndarray
    y_test_noisy: np.ndarray


ROBUST_REGRESSION_TASK_SPECS: tuple[RobustRegressionTaskSpec, ...] = (
    RobustRegressionTaskSpec(
        task_id="rr_outliers10_100x3",
        task_index=0,
        n_train=100,
        n_test=200,
        n_features=3,
        noise_std=0.10,
        rho=0.0,
        vertical_outlier_fraction_train=0.10,
        leverage_outlier_fraction_train=0.0,
        hetero_strength=0.0,
    ),
    RobustRegressionTaskSpec(
        task_id="rr_outliers20_100x3",
        task_index=1,
        n_train=100,
        n_test=200,
        n_features=3,
        noise_std=0.10,
        rho=0.0,
        vertical_outlier_fraction_train=0.20,
        leverage_outlier_fraction_train=0.0,
        hetero_strength=0.0,
    ),
    RobustRegressionTaskSpec(
        task_id="rr_leverage10_100x3",
        task_index=2,
        n_train=100,
        n_test=200,
        n_features=3,
        noise_std=0.10,
        rho=0.0,
        vertical_outlier_fraction_train=0.0,
        leverage_outlier_fraction_train=0.10,
        hetero_strength=0.0,
    ),
    RobustRegressionTaskSpec(
        task_id="rr_hard_120x8",
        task_index=3,
        n_train=120,
        n_test=400,
        n_features=8,
        noise_std=0.20,
        rho=0.85,
        vertical_outlier_fraction_train=0.20,
        leverage_outlier_fraction_train=0.15,
        hetero_strength=1.00,
    ),
)

ROBUST_REGRESSION_TASKS_BY_ID: Dict[str, RobustRegressionTaskSpec] = {
    task.task_id: task for task in ROBUST_REGRESSION_TASK_SPECS
}


def resolve_task_specs(selector: Optional[str]) -> List[RobustRegressionTaskSpec]:
    """Resolve a task selector into concrete robust-regression task specs."""
    normalized = (selector or ROBUST_REGRESSION_SHARED_SELECTOR).strip()
    if not normalized or normalized == ROBUST_REGRESSION_SHARED_SELECTOR:
        return list(ROBUST_REGRESSION_TASK_SPECS)
    if normalized not in ROBUST_REGRESSION_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in ROBUST_REGRESSION_TASK_SPECS)
        raise ValueError(
            f"Unknown robust-regression task '{normalized}'. Available: {available}"
        )
    return [ROBUST_REGRESSION_TASKS_BY_ID[normalized]]


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


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, numeric)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def derived_seed_for_task(task: RobustRegressionTaskSpec, base_seed: int) -> int:
    """Return the deterministic dataset seed for one task/seed pair."""
    return 10000 + 1000 * int(task.task_index) + int(base_seed)


def build_covariance_matrix(n_features: int, rho: float) -> np.ndarray:
    """Construct the Toeplitz covariance matrix for one task."""
    if float(rho) == 0.0:
        return np.eye(int(n_features), dtype=float)
    return np.asarray(
        [[float(rho) ** abs(i - j) for j in range(int(n_features))] for i in range(int(n_features))],
        dtype=float,
    )


def generate_regression_dataset(
    task: RobustRegressionTaskSpec,
    base_seed: int,
) -> RobustRegressionDataset:
    """Generate one deterministic synthetic train/test regression dataset."""
    derived_seed = derived_seed_for_task(task, base_seed)
    rng = np.random.default_rng(derived_seed)

    beta0 = float(rng.normal(0.0, 1.0))
    beta = np.asarray(rng.normal(0.0, 1.0, size=int(task.n_features)), dtype=float)
    covariance = build_covariance_matrix(task.n_features, task.rho)

    X_train = np.asarray(
        rng.multivariate_normal(
            mean=np.zeros(int(task.n_features), dtype=float),
            cov=covariance,
            size=int(task.n_train),
        ),
        dtype=float,
    )
    X_test = np.asarray(
        rng.multivariate_normal(
            mean=np.zeros(int(task.n_features), dtype=float),
            cov=covariance,
            size=int(task.n_test),
        ),
        dtype=float,
    )

    signal_train = beta0 + X_train @ beta
    signal_test = beta0 + X_test @ beta

    if float(task.hetero_strength) == 0.0:
        sigma_train = np.full(int(task.n_train), float(task.noise_std), dtype=float)
        sigma_test = np.full(int(task.n_test), float(task.noise_std), dtype=float)
    else:
        sigma_train = float(task.noise_std) * (
            1.0 + float(task.hetero_strength) * np.abs(X_train[:, 0])
        )
        sigma_test = float(task.noise_std) * (
            1.0 + float(task.hetero_strength) * np.abs(X_test[:, 0])
        )

    epsilon_train = np.asarray(rng.normal(0.0, sigma_train), dtype=float)
    epsilon_test = np.asarray(rng.normal(0.0, sigma_test), dtype=float)

    y_train_clean = signal_train + epsilon_train
    y_train = np.asarray(y_train_clean, dtype=float).copy()
    y_test_noisy = signal_test + epsilon_test

    vertical_outliers = math.floor(
        float(task.vertical_outlier_fraction_train) * float(task.n_train)
    )
    if vertical_outliers > 0:
        indices = rng.choice(int(task.n_train), size=vertical_outliers, replace=False)
        y_scale = float(np.std(y_train_clean)) + 1e-8
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=vertical_outliers)
        y_train[indices] += signs * 6.0 * y_scale

    leverage_outliers = math.floor(
        float(task.leverage_outlier_fraction_train) * float(task.n_train)
    )
    if leverage_outliers > 0:
        indices = rng.choice(int(task.n_train), size=leverage_outliers, replace=False)
        for column_index in range(min(2, int(task.n_features))):
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=leverage_outliers)
            X_train[indices, column_index] += signs * 6.0

    return RobustRegressionDataset(
        task_id=task.task_id,
        task_index=task.task_index,
        base_seed=int(base_seed),
        derived_seed=int(derived_seed),
        true_coefficients=np.asarray([beta0, *beta.tolist()], dtype=float),
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_test_clean_signal=np.asarray(signal_test, dtype=float),
        y_test_noisy=np.asarray(y_test_noisy, dtype=float),
    )


def empty_task_metrics(
    *,
    seed_count: int = len(FULL_EVAL_SEEDS),
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, float | int]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "nmse_signal_test": 1.0,
        "nmse_noisy_test": 1.0,
        "coef_rel_error": 1.0,
        "signal_score": 0.0,
        "noisy_score": 0.0,
        "coef_score": 0.0,
        "r2_signal_test": 0.0,
        "r2_noisy_test": 0.0,
        "success_rate": 0.0,
        "seed_count": int(seed_count),
        "successful_seed_count": 0,
        "avg_exec_time": max(0.0, _coerce_finite_float(timeout_seconds, DEFAULT_FULL_TIMEOUT_SECONDS)),
        "score": 0.0,
        "combined_score": 0.0,
    }


def aggregate_seed_results(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    seed_count: int,
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, float | int]:
    """Aggregate per-seed benchmark results into one task-level metric payload."""
    successful = [result for result in seed_results if bool(result.get("success"))]
    if not successful:
        return empty_task_metrics(seed_count=seed_count, timeout_seconds=timeout_seconds)

    metrics = {
        "nmse_signal_test": float(
            np.mean([float(result["nmse_signal_test"]) for result in successful])
        ),
        "nmse_noisy_test": float(
            np.mean([float(result["nmse_noisy_test"]) for result in successful])
        ),
        "coef_rel_error": float(
            np.mean([float(result["coef_rel_error"]) for result in successful])
        ),
        "signal_score": float(
            np.mean([float(result["signal_score"]) for result in successful])
        ),
        "noisy_score": float(
            np.mean([float(result["noisy_score"]) for result in successful])
        ),
        "coef_score": float(np.mean([float(result["coef_score"]) for result in successful])),
        "r2_signal_test": float(
            np.mean([float(result["r2_signal_test"]) for result in successful])
        ),
        "r2_noisy_test": float(
            np.mean([float(result["r2_noisy_test"]) for result in successful])
        ),
        "success_rate": float(len(successful)) / float(max(1, seed_count)),
        "seed_count": int(seed_count),
        "successful_seed_count": int(len(successful)),
        "avg_exec_time": float(np.mean([float(result["runtime"]) for result in successful])),
    }
    quality_score = (
        0.55 * metrics["signal_score"]
        + 0.25 * metrics["noisy_score"]
        + 0.20 * metrics["coef_score"]
    )
    metrics["score"] = metrics["success_rate"] * quality_score
    metrics["combined_score"] = metrics["score"]
    return metrics


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    seed_count: Optional[int] = None,
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, float | int]:
    """Normalize evaluator outputs into the stable robust-regression metric schema."""
    defaults = empty_task_metrics(
        seed_count=int(seed_count) if seed_count is not None else len(FULL_EVAL_SEEDS),
        timeout_seconds=timeout_seconds,
    )
    raw_metrics = raw_metrics or {}

    normalized_seed_count = _coerce_non_negative_int(
        raw_metrics.get("seed_count"),
        int(defaults["seed_count"]),
    )
    successful_seed_count = _coerce_non_negative_int(
        raw_metrics.get("successful_seed_count"),
        int(defaults["successful_seed_count"]),
    )
    if normalized_seed_count <= 0:
        normalized_seed_count = int(defaults["seed_count"])
    successful_seed_count = min(successful_seed_count, normalized_seed_count)
    success_rate = (
        float(successful_seed_count) / float(normalized_seed_count)
        if normalized_seed_count > 0
        else 0.0
    )

    metrics: Dict[str, float | int] = {
        "nmse_signal_test": _coerce_non_negative_float(
            raw_metrics.get("nmse_signal_test"),
            float(defaults["nmse_signal_test"]),
        ),
        "nmse_noisy_test": _coerce_non_negative_float(
            raw_metrics.get("nmse_noisy_test"),
            float(defaults["nmse_noisy_test"]),
        ),
        "coef_rel_error": _coerce_non_negative_float(
            raw_metrics.get("coef_rel_error"),
            float(defaults["coef_rel_error"]),
        ),
        "signal_score": _clamp(
            _coerce_finite_float(raw_metrics.get("signal_score"), float(defaults["signal_score"])),
            0.0,
            1.0,
        ),
        "noisy_score": _clamp(
            _coerce_finite_float(raw_metrics.get("noisy_score"), float(defaults["noisy_score"])),
            0.0,
            1.0,
        ),
        "coef_score": _clamp(
            _coerce_finite_float(raw_metrics.get("coef_score"), float(defaults["coef_score"])),
            0.0,
            1.0,
        ),
        "r2_signal_test": _coerce_finite_float(
            raw_metrics.get("r2_signal_test"),
            float(defaults["r2_signal_test"]),
        ),
        "r2_noisy_test": _coerce_finite_float(
            raw_metrics.get("r2_noisy_test"),
            float(defaults["r2_noisy_test"]),
        ),
        "success_rate": _clamp(success_rate, 0.0, 1.0),
        "seed_count": int(normalized_seed_count),
        "successful_seed_count": int(successful_seed_count),
        "avg_exec_time": _coerce_non_negative_float(
            raw_metrics.get("avg_exec_time"),
            float(defaults["avg_exec_time"]),
        ),
    }

    if metrics["successful_seed_count"] <= 0:
        metrics.update(
            {
                "signal_score": 0.0,
                "noisy_score": 0.0,
                "coef_score": 0.0,
                "r2_signal_test": 0.0,
                "r2_noisy_test": 0.0,
                "score": 0.0,
                "combined_score": 0.0,
            }
        )
        return metrics

    quality_score = (
        0.55 * float(metrics["signal_score"])
        + 0.25 * float(metrics["noisy_score"])
        + 0.20 * float(metrics["coef_score"])
    )
    score = float(metrics["success_rate"]) * quality_score
    metrics["score"] = score
    metrics["combined_score"] = score
    return metrics


def build_task_result(
    task: RobustRegressionTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    seed_results: Optional[Sequence[Mapping[str, Any]]] = None,
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(raw_metrics, timeout_seconds=timeout_seconds)
    final_task_score = float(metrics["score"])
    resolved_error = error
    if resolved_error is None and int(metrics["successful_seed_count"]) <= 0:
        resolved_error = "All evaluation seeds failed"
    normalized_seed_results = [dict(result) for result in (seed_results or [])]
    return {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "spec": task.to_spec_dict(),
        "metrics": metrics,
        "case_score": final_task_score,
        "final_task_score": final_task_score,
        "seed_results": normalized_seed_results,
        "error": resolved_error,
    }


def aggregate_task_results(task_results: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregate per-task robust-regression results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        empty = empty_task_metrics(seed_count=0, timeout_seconds=DEFAULT_FULL_TIMEOUT_SECONDS)
        return {
            "nmse_signal_test": float(empty["nmse_signal_test"]),
            "nmse_noisy_test": float(empty["nmse_noisy_test"]),
            "coef_rel_error": float(empty["coef_rel_error"]),
            "signal_score": float(empty["signal_score"]),
            "noisy_score": float(empty["noisy_score"]),
            "coef_score": float(empty["coef_score"]),
            "r2_signal_test": float(empty["r2_signal_test"]),
            "r2_noisy_test": float(empty["r2_noisy_test"]),
            "success_rate": float(empty["success_rate"]),
            "avg_exec_time": float(empty["avg_exec_time"]),
            "score": 0.0,
            "combined_score": 0.0,
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    metric_keys = (
        "nmse_signal_test",
        "nmse_noisy_test",
        "coef_rel_error",
        "signal_score",
        "noisy_score",
        "coef_score",
        "r2_signal_test",
        "r2_noisy_test",
        "success_rate",
        "avg_exec_time",
        "score",
    )
    aggregate: Dict[str, float] = {}
    for key in metric_keys:
        aggregate[key] = sum(float(result["metrics"][key]) for result in normalized_results) / float(
            len(normalized_results)
        )
    aggregate["combined_score"] = aggregate["score"]
    aggregate["task_count"] = float(len(normalized_results))
    aggregate["successful_task_count"] = float(
        sum(1 for result in normalized_results if float(result["metrics"]["success_rate"]) > 0.0)
    )
    aggregate["failed_task_count"] = (
        float(len(normalized_results)) - aggregate["successful_task_count"]
    )
    return aggregate


def _validate_seed_results(
    seed_results: Any,
    *,
    expected_seed_count: int,
    expected_successful_seed_count: int,
) -> bool:
    if not isinstance(seed_results, list):
        return False
    if len(seed_results) != int(expected_seed_count):
        return False
    observed_successful_seed_count = 0
    for seed_result in seed_results:
        if not isinstance(seed_result, Mapping):
            return False
        if "seed" not in seed_result or "runtime" not in seed_result or "success" not in seed_result:
            return False
        if not math.isfinite(_coerce_finite_float(seed_result.get("runtime"), float("nan"))):
            return False
        if bool(seed_result.get("success")):
            observed_successful_seed_count += 1
            for key in ROBUST_REGRESSION_SUCCESSFUL_SEED_METRIC_KEYS:
                if key not in seed_result:
                    return False
                value = _coerce_finite_float(seed_result.get(key), float("nan"))
                if not math.isfinite(value):
                    return False
    return observed_successful_seed_count == int(expected_successful_seed_count)


def extract_task_result(artifacts: Mapping[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Extract one per-task artifact entry from stored evaluation artifacts."""
    evaluation_stage = artifacts.get("evaluation_stage")
    if isinstance(evaluation_stage, str) and evaluation_stage.strip().lower() != "full":
        return None

    task_results = artifacts.get("task_results")
    if not isinstance(task_results, list):
        return None

    task = ROBUST_REGRESSION_TASKS_BY_ID.get(task_id)
    for task_result in task_results:
        if not isinstance(task_result, Mapping):
            continue
        if task_result.get("task_id") != task_id:
            continue

        metrics = task_result.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        for key in ROBUST_REGRESSION_REQUIRED_TASK_METRICS:
            if key not in metrics:
                return None
            value = _coerce_finite_float(metrics.get(key), float("nan"))
            if not math.isfinite(value):
                return None

        seed_results = task_result.get("seed_results")
        if not _validate_seed_results(
            seed_results,
            expected_seed_count=int(metrics["seed_count"]),
            expected_successful_seed_count=int(metrics["successful_seed_count"]),
        ):
            return None

        if task is None:
            return dict(task_result)
        return build_task_result(
            task,
            raw_metrics=metrics,
            error=task_result.get("error"),
            seed_results=seed_results,
            timeout_seconds=task.trial_timeout_seconds_full,
        )
    return None


def project_task_artifacts(
    artifacts: Mapping[str, Any],
    task_id: str,
    task_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project shared artifacts into the task-local view used by spawned checkpoints."""
    del artifacts
    return {
        "task_selector": task_id,
        "evaluation_stage": "full",
        "task_results": [dict(task_result)],
    }
