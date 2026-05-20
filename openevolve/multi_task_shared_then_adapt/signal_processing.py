"""Canonical signal-processing task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR = "SIGNAL_PROCESSING_TASK_ID"
SIGNAL_PROCESSING_SHARED_SELECTOR = "all"
DEFAULT_STAGE1_TIMEOUT_SECONDS = 2.0
DEFAULT_FULL_TIMEOUT_SECONDS = 5.0
DEFAULT_EVALUATION_TIMEOUT_SECONDS = DEFAULT_FULL_TIMEOUT_SECONDS


@dataclass(frozen=True)
class SignalProcessingTaskSpec:
    """Stable task identity for the signal-processing EMO-STA family."""

    task_id: str
    task_index: int
    display_name: str
    signal_family: str
    length: int
    noise_level: float
    window_size: int
    t_start: float
    t_end: float
    evaluation_seeds_full: tuple[int, ...]
    evaluation_seeds_stage1: tuple[int, ...]
    trial_timeout_seconds_full: float = DEFAULT_FULL_TIMEOUT_SECONDS
    trial_timeout_seconds_stage1: float = DEFAULT_STAGE1_TIMEOUT_SECONDS

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "signal_family": self.signal_family,
            "length": self.length,
            "noise_level": self.noise_level,
            "window_size": self.window_size,
            "t_start": self.t_start,
            "t_end": self.t_end,
        }


SIGNAL_PROCESSING_TASK_SPECS: tuple[SignalProcessingTaskSpec, ...] = (
    SignalProcessingTaskSpec(
        task_id="sp_trend_sine_500_n02",
        task_index=0,
        display_name="TrendSine",
        signal_family="trend_sine",
        length=500,
        noise_level=0.2,
        window_size=20,
        t_start=0.0,
        t_end=10.0,
        evaluation_seeds_full=(0, 1, 2),
        evaluation_seeds_stage1=(0,),
    ),
    SignalProcessingTaskSpec(
        task_id="sp_multifreq_600_n03",
        task_index=1,
        display_name="MultiFrequency",
        signal_family="multifrequency",
        length=600,
        noise_level=0.3,
        window_size=20,
        t_start=0.0,
        t_end=10.0,
        evaluation_seeds_full=(0, 1, 2),
        evaluation_seeds_stage1=(0,),
    ),
    SignalProcessingTaskSpec(
        task_id="sp_chirp_700_n04",
        task_index=2,
        display_name="Chirp",
        signal_family="chirp",
        length=700,
        noise_level=0.4,
        window_size=20,
        t_start=0.0,
        t_end=10.0,
        evaluation_seeds_full=(0, 1, 2),
        evaluation_seeds_stage1=(0,),
    ),
    SignalProcessingTaskSpec(
        task_id="sp_step_800_n05",
        task_index=3,
        display_name="StepChanges",
        signal_family="step_changes",
        length=800,
        noise_level=0.5,
        window_size=20,
        t_start=0.0,
        t_end=10.0,
        evaluation_seeds_full=(0, 1, 2),
        evaluation_seeds_stage1=(0,),
    ),
)

SIGNAL_PROCESSING_TASKS_BY_ID: Dict[str, SignalProcessingTaskSpec] = {
    task.task_id: task for task in SIGNAL_PROCESSING_TASK_SPECS
}

SIGNAL_PROCESSING_METRIC_KEYS: tuple[str, ...] = (
    "composite_score",
    "overall_score",
    "slope_changes",
    "lag_error",
    "avg_error",
    "false_reversals",
    "correlation",
    "noise_reduction",
    "smoothness_score",
    "responsiveness_score",
    "accuracy_score",
    "efficiency_score",
    "success_rate",
    "execution_time",
    "score",
    "combined_score",
)


def resolve_task_specs(selector: Optional[str]) -> List[SignalProcessingTaskSpec]:
    """Resolve a task selector into concrete signal-processing task specs."""
    normalized = (selector or SIGNAL_PROCESSING_SHARED_SELECTOR).strip()
    if not normalized or normalized == SIGNAL_PROCESSING_SHARED_SELECTOR:
        return list(SIGNAL_PROCESSING_TASK_SPECS)
    if normalized not in SIGNAL_PROCESSING_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in SIGNAL_PROCESSING_TASK_SPECS)
        raise ValueError(
            f"Unknown signal-processing task '{normalized}'. Available: {available}"
        )
    return [SIGNAL_PROCESSING_TASKS_BY_ID[normalized]]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a scalar-like value to a finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_clean_signal(task: SignalProcessingTaskSpec) -> np.ndarray:
    """Generate the clean deterministic signal for one task."""
    length = int(task.length)
    t = np.linspace(float(task.t_start), float(task.t_end), length)

    if task.signal_family == "trend_sine":
        clean = 2.0 * np.sin(2.0 * np.pi * 0.5 * t) + 0.1 * t
    elif task.signal_family == "multifrequency":
        clean = (
            np.sin(2.0 * np.pi * 0.5 * t)
            + 0.5 * np.sin(2.0 * np.pi * 2.0 * t)
            + 0.2 * np.sin(2.0 * np.pi * 5.0 * t)
        )
    elif task.signal_family == "chirp":
        clean = np.sin(2.0 * np.pi * (0.5 + 0.2 * t) * t)
    elif task.signal_family == "step_changes":
        clean = np.concatenate(
            [
                np.ones(length // 3),
                2.0 * np.ones(length // 3),
                0.5 * np.ones(length - 2 * (length // 3)),
            ]
        )
    else:
        raise ValueError(f"Unsupported signal family: {task.signal_family}")

    return np.asarray(clean, dtype=float)


def generate_noisy_signal(task: SignalProcessingTaskSpec, seed: int) -> np.ndarray:
    """Generate the noisy observed signal for one task and seed."""
    clean_signal = generate_clean_signal(task)
    rng = np.random.default_rng(int(seed))
    return clean_signal + rng.normal(0.0, float(task.noise_level), int(task.length))


def generate_signal_pair(task: SignalProcessingTaskSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic noisy/clean signal pairs for evaluation."""
    clean_signal = generate_clean_signal(task)
    rng = np.random.default_rng(int(seed))
    noisy_signal = clean_signal + rng.normal(0.0, float(task.noise_level), int(task.length))
    return np.asarray(noisy_signal, dtype=float), np.asarray(clean_signal, dtype=float)


def calculate_slope_changes(signal_data: Sequence[float]) -> int:
    """Count directional reversals in a 1D filtered signal."""
    signal_array = np.asarray(signal_data, dtype=float)
    if signal_array.ndim != 1 or signal_array.size < 3:
        return 0

    diffs = np.diff(signal_array)
    sign_changes = 0
    for index in range(1, len(diffs)):
        if np.sign(diffs[index]) != np.sign(diffs[index - 1]) and diffs[index - 1] != 0:
            sign_changes += 1
    return int(sign_changes)


def calculate_lag_error(
    filtered_signal: Sequence[float],
    noisy_signal: Sequence[float],
    window_size: int,
) -> float:
    """Calculate the instantaneous lag error against the aligned noisy signal."""
    filtered_array = np.asarray(filtered_signal, dtype=float)
    noisy_array = np.asarray(noisy_signal, dtype=float)
    if filtered_array.ndim != 1 or noisy_array.ndim != 1 or filtered_array.size == 0:
        return 1.0

    delay = int(window_size) - 1
    if noisy_array.size <= delay:
        return 1.0

    recent_filtered = filtered_array[-1]
    recent_noisy = noisy_array[delay + filtered_array.size - 1]
    return abs(float(recent_filtered) - float(recent_noisy))


def calculate_average_tracking_error(
    filtered_signal: Sequence[float],
    noisy_signal: Sequence[float],
    window_size: int,
) -> float:
    """Calculate the mean absolute aligned error against the noisy signal."""
    filtered_array = np.asarray(filtered_signal, dtype=float)
    noisy_array = np.asarray(noisy_signal, dtype=float)
    if filtered_array.ndim != 1 or noisy_array.ndim != 1 or filtered_array.size == 0:
        return 1.0

    delay = int(window_size) - 1
    if noisy_array.size <= delay:
        return 1.0

    aligned_noisy = noisy_array[delay : delay + filtered_array.size]
    min_length = min(filtered_array.size, aligned_noisy.size)
    if min_length <= 0:
        return 1.0
    return float(np.mean(np.abs(filtered_array[:min_length] - aligned_noisy[:min_length])))


def calculate_false_reversal_penalty(
    filtered_signal: Sequence[float],
    clean_signal: Sequence[float],
    window_size: int,
) -> int:
    """Count filtered trend reversals that do not exist in the aligned clean signal."""
    filtered_array = np.asarray(filtered_signal, dtype=float)
    clean_array = np.asarray(clean_signal, dtype=float)
    if filtered_array.ndim != 1 or clean_array.ndim != 1:
        return 0
    if filtered_array.size < 3 or clean_array.size < 3:
        return 0

    delay = int(window_size) - 1
    if clean_array.size <= delay:
        return 1

    aligned_clean = clean_array[delay : delay + filtered_array.size]
    min_length = min(filtered_array.size, aligned_clean.size)
    if min_length < 3:
        return 0

    filtered_diffs = np.diff(filtered_array[:min_length])
    clean_diffs = np.diff(aligned_clean[:min_length])

    false_reversals = 0
    for index in range(1, len(filtered_diffs)):
        filtered_change = (
            np.sign(filtered_diffs[index]) != np.sign(filtered_diffs[index - 1])
            and filtered_diffs[index - 1] != 0
        )
        clean_change = (
            np.sign(clean_diffs[index]) != np.sign(clean_diffs[index - 1])
            and clean_diffs[index - 1] != 0
        )
        if filtered_change and not clean_change:
            false_reversals += 1
    return int(false_reversals)


def calculate_composite_score(
    S: float,
    L_recent: float,
    L_avg: float,
    R: float,
    alpha: tuple[float, float, float, float] = (0.3, 0.2, 0.2, 0.3),
) -> float:
    """Compute the normalized composite score for one trial."""
    S_norm = min(max(0.0, safe_float(S)) / 50.0, 2.0)
    L_recent_norm = min(max(0.0, safe_float(L_recent)), 2.0)
    L_avg_norm = min(max(0.0, safe_float(L_avg)), 2.0)
    R_norm = min(max(0.0, safe_float(R)) / 25.0, 2.0)
    penalty = (
        alpha[0] * S_norm
        + alpha[1] * L_recent_norm
        + alpha[2] * L_avg_norm
        + alpha[3] * R_norm
    )
    return 1.0 / (1.0 + penalty)


def empty_task_metrics(
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "composite_score": 0.0,
        "overall_score": 0.0,
        "slope_changes": 50.0,
        "lag_error": 2.0,
        "avg_error": 2.0,
        "false_reversals": 25.0,
        "correlation": 0.0,
        "noise_reduction": 0.0,
        "smoothness_score": 0.0,
        "responsiveness_score": 0.0,
        "accuracy_score": 0.0,
        "efficiency_score": 0.0,
        "success_rate": 0.0,
        "execution_time": max(0.0, safe_float(timeout_seconds, DEFAULT_FULL_TIMEOUT_SECONDS)),
        "score": 0.0,
        "combined_score": 0.0,
    }


def normalize_task_metrics(
    raw_metrics: Mapping[str, Any] | None,
    *,
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, float]:
    """Normalize evaluator outputs into the stable task-level metric schema."""
    defaults = empty_task_metrics(timeout_seconds=timeout_seconds)
    raw_metrics = raw_metrics or {}

    composite_score = _clamp(
        safe_float(raw_metrics.get("composite_score"), defaults["composite_score"]),
        0.0,
        1.0,
    )
    slope_changes = max(
        0.0,
        safe_float(raw_metrics.get("slope_changes"), defaults["slope_changes"]),
    )
    lag_error = max(0.0, safe_float(raw_metrics.get("lag_error"), defaults["lag_error"]))
    avg_error = max(0.0, safe_float(raw_metrics.get("avg_error"), defaults["avg_error"]))
    false_reversals = max(
        0.0,
        safe_float(raw_metrics.get("false_reversals"), defaults["false_reversals"]),
    )
    correlation = _clamp(
        safe_float(raw_metrics.get("correlation"), defaults["correlation"]),
        -1.0,
        1.0,
    )
    noise_reduction = _clamp(
        max(0.0, safe_float(raw_metrics.get("noise_reduction"), defaults["noise_reduction"])),
        0.0,
        1.0,
    )
    success_rate = _clamp(
        safe_float(raw_metrics.get("success_rate"), defaults["success_rate"]),
        0.0,
        1.0,
    )
    execution_time = max(
        0.0,
        safe_float(raw_metrics.get("execution_time"), defaults["execution_time"]),
    )

    if success_rate <= 0.0:
        return {
            **defaults,
            "execution_time": execution_time,
        }

    smoothness_score = 1.0 / (1.0 + slope_changes / 20.0)
    responsiveness_score = 1.0 / (1.0 + lag_error)
    accuracy_score = _clamp(max(0.0, correlation), 0.0, 1.0)
    efficiency_score = _clamp(
        min(1.0, 1.0 / max(0.001, execution_time)),
        0.0,
        1.0,
    )
    overall_score = _clamp(
        0.4 * composite_score
        + 0.2 * smoothness_score
        + 0.2 * accuracy_score
        + 0.1 * noise_reduction
        + 0.1 * success_rate,
        0.0,
        1.0,
    )

    return {
        "composite_score": composite_score,
        "overall_score": overall_score,
        "slope_changes": slope_changes,
        "lag_error": lag_error,
        "avg_error": avg_error,
        "false_reversals": false_reversals,
        "correlation": correlation,
        "noise_reduction": noise_reduction,
        "smoothness_score": smoothness_score,
        "responsiveness_score": responsiveness_score,
        "accuracy_score": accuracy_score,
        "efficiency_score": efficiency_score,
        "success_rate": success_rate,
        "execution_time": execution_time,
        "score": overall_score,
        "combined_score": overall_score,
    }


def build_task_result(
    task: SignalProcessingTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
    timeout_seconds: float = DEFAULT_FULL_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(raw_metrics, timeout_seconds=timeout_seconds)
    final_task_score = float(metrics["overall_score"])
    metrics["score"] = final_task_score
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
    """Aggregate per-task signal-processing results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    aggregate: Dict[str, float] = {}
    for key in SIGNAL_PROCESSING_METRIC_KEYS:
        if key in {"score", "combined_score"}:
            continue
        aggregate[key] = sum(float(result["metrics"][key]) for result in normalized_results) / len(
            normalized_results
        )

    aggregate["score"] = sum(
        float(result["final_task_score"]) for result in normalized_results
    ) / len(normalized_results)
    aggregate["combined_score"] = aggregate["score"]
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
        for key in SIGNAL_PROCESSING_METRIC_KEYS:
            if key not in metrics:
                return None
            numeric = safe_float(metrics.get(key), float("nan"))
            if not math.isfinite(numeric):
                return None

        task = SIGNAL_PROCESSING_TASKS_BY_ID.get(task_id)
        if task is None:
            return dict(task_result)
        return build_task_result(
            task,
            raw_metrics=metrics,
            error=task_result.get("error"),
            timeout_seconds=task.trial_timeout_seconds_full,
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

    trial_counts = artifacts.get("trial_counts")
    if isinstance(trial_counts, Mapping):
        task_trial_counts = trial_counts.get(task_id)
        if isinstance(task_trial_counts, Mapping):
            projected["trial_counts"] = dict(task_trial_counts)

    best_observed_per_task = artifacts.get("best_observed_per_task")
    if isinstance(best_observed_per_task, Mapping):
        best_summary = best_observed_per_task.get(task_id)
        if isinstance(best_summary, Mapping):
            projected["best_observed_per_task"] = {task_id: dict(best_summary)}

    compact_task_summary = artifacts.get("compact_task_summary")
    if isinstance(compact_task_summary, Mapping):
        task_summary = compact_task_summary.get(task_id)
        if isinstance(task_summary, Mapping):
            projected["compact_task_summary"] = {task_id: dict(task_summary)}

    convergence_notes = artifacts.get("convergence_notes")
    if isinstance(convergence_notes, Mapping):
        task_note = convergence_notes.get(task_id)
        if isinstance(task_note, str):
            projected["convergence_notes"] = {task_id: task_note}

    return projected


__all__ = [
    "DEFAULT_EVALUATION_TIMEOUT_SECONDS",
    "DEFAULT_FULL_TIMEOUT_SECONDS",
    "DEFAULT_STAGE1_TIMEOUT_SECONDS",
    "SIGNAL_PROCESSING_METRIC_KEYS",
    "SIGNAL_PROCESSING_SHARED_SELECTOR",
    "SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR",
    "SIGNAL_PROCESSING_TASK_SPECS",
    "SIGNAL_PROCESSING_TASKS_BY_ID",
    "SignalProcessingTaskSpec",
    "aggregate_task_results",
    "build_task_result",
    "calculate_average_tracking_error",
    "calculate_composite_score",
    "calculate_false_reversal_penalty",
    "calculate_lag_error",
    "calculate_slope_changes",
    "empty_task_metrics",
    "extract_task_result",
    "generate_clean_signal",
    "generate_noisy_signal",
    "generate_signal_pair",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_task_specs",
    "safe_float",
]
