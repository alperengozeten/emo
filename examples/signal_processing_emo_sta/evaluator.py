"""Evaluator for the EMO-STA signal-processing family."""

from __future__ import annotations

import concurrent.futures
import importlib.util
from pathlib import Path
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.signal_processing import (
    SIGNAL_PROCESSING_SHARED_SELECTOR,
    SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR,
    SignalProcessingTaskSpec,
    aggregate_task_results,
    build_task_result,
    calculate_average_tracking_error,
    calculate_composite_score,
    calculate_false_reversal_penalty,
    calculate_lag_error,
    calculate_slope_changes,
    generate_signal_pair,
    resolve_task_specs,
    safe_float,
)


class TrialTimeoutError(TimeoutError):
    """Raised when one filtering trial exceeds the evaluator budget."""


def run_with_timeout(
    func: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    timeout_seconds: float,
) -> Any:
    """Run a callable in a worker thread with a timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **(dict(kwargs or {})))
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TrialTimeoutError(
                f"Trial timed out after {timeout_seconds:.1f} seconds"
            ) from exc


def _load_program_module(program_path: str):
    module_name = f"signal_processing_emo_sta_program_{hash(Path(program_path).resolve())}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_candidate_runner(module: Any) -> Callable[..., Any]:
    for candidate_name in ("run_signal_processing", "process_signal"):
        candidate = getattr(module, candidate_name, None)
        if callable(candidate):
            return candidate
    raise AttributeError("Program must define run_signal_processing() or process_signal()")


def _normalize_filtered_signal(result: Any, *, expected_length: int) -> np.ndarray:
    if isinstance(result, Mapping):
        if "filtered_signal" not in result:
            raise ValueError("Result dict must contain filtered_signal")
        candidate_output = result["filtered_signal"]
    else:
        candidate_output = result

    filtered_signal = np.asarray(candidate_output, dtype=float)
    if filtered_signal.ndim != 1:
        raise ValueError("Filtered signal must be a 1D numeric array")
    if filtered_signal.size != int(expected_length):
        raise ValueError(
            f"Filtered signal length must equal {expected_length}, got {filtered_signal.size}"
        )
    if not np.all(np.isfinite(filtered_signal)):
        raise ValueError("Filtered signal must contain only finite numeric values")
    return filtered_signal


def _trial_error_message(seed: int, exc: Exception) -> str:
    return f"seed={seed}: {type(exc).__name__}: {exc}"


def _calculate_correlation(filtered_signal: np.ndarray, aligned_clean: np.ndarray) -> float:
    if filtered_signal.size < 2 or aligned_clean.size < 2:
        return 0.0
    if np.std(filtered_signal) <= 0.0 or np.std(aligned_clean) <= 0.0:
        return 0.0
    try:
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1]
    except Exception:
        return 0.0
    return safe_float(correlation, 0.0)


def _calculate_noise_reduction(
    filtered_signal: np.ndarray,
    aligned_noisy: np.ndarray,
    aligned_clean: np.ndarray,
) -> float:
    noise_before = safe_float(np.var(aligned_noisy - aligned_clean), 0.0)
    noise_after = safe_float(np.var(filtered_signal - aligned_clean), 0.0)
    if noise_before <= 0.0:
        return 0.0
    reduction = (noise_before - noise_after) / noise_before
    if not np.isfinite(reduction):
        return 0.0
    return max(0.0, min(1.0, float(reduction)))


def _aggregate_successful_trials(
    successful_trials: Iterable[Mapping[str, float]],
    *,
    total_trials: int,
) -> Dict[str, float]:
    successful_list = list(successful_trials)
    if total_trials <= 0 or not successful_list:
        return {
            "composite_score": 0.0,
            "slope_changes": 50.0,
            "lag_error": 2.0,
            "avg_error": 2.0,
            "false_reversals": 25.0,
            "correlation": 0.0,
            "noise_reduction": 0.0,
            "execution_time": 0.0,
            "success_rate": 0.0,
        }

    def _mean(key: str) -> float:
        return sum(float(trial[key]) for trial in successful_list) / len(successful_list)

    return {
        "composite_score": _mean("composite_score"),
        "slope_changes": _mean("slope_changes"),
        "lag_error": _mean("lag_error"),
        "avg_error": _mean("avg_error"),
        "false_reversals": _mean("false_reversals"),
        "correlation": _mean("correlation"),
        "noise_reduction": _mean("noise_reduction"),
        "execution_time": _mean("execution_time"),
        "success_rate": len(successful_list) / float(total_trials),
    }


def evaluate_one_task(
    candidate_runner: Callable[..., Any],
    task: SignalProcessingTaskSpec,
    *,
    stage1: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = task.evaluation_seeds_stage1 if stage1 else task.evaluation_seeds_full
    timeout_seconds = (
        task.trial_timeout_seconds_stage1
        if stage1
        else task.trial_timeout_seconds_full
    )

    successful_trials: list[dict[str, float]] = []
    failed_trials: list[str] = []
    best_trial_summary: dict[str, Any] | None = None

    for seed in seeds:
        noisy_signal, clean_signal = generate_signal_pair(task, seed)
        original_noisy_signal = np.asarray(noisy_signal, dtype=float).copy()
        candidate_input_signal = original_noisy_signal.copy()
        expected_length = len(original_noisy_signal) - task.window_size + 1
        try:
            start_time = time.perf_counter()
            raw_result = run_with_timeout(
                candidate_runner,
                args=(candidate_input_signal,),
                kwargs={"window_size": task.window_size},
                timeout_seconds=timeout_seconds,
            )
            execution_time = time.perf_counter() - start_time

            filtered_signal = _normalize_filtered_signal(
                raw_result,
                expected_length=expected_length,
            )

            delay = task.window_size - 1
            aligned_clean = clean_signal[delay : delay + filtered_signal.size]
            aligned_noisy = original_noisy_signal[delay : delay + filtered_signal.size]

            slope_changes = calculate_slope_changes(filtered_signal)
            lag_error = calculate_lag_error(
                filtered_signal,
                original_noisy_signal,
                task.window_size,
            )
            avg_error = calculate_average_tracking_error(
                filtered_signal,
                original_noisy_signal,
                task.window_size,
            )
            false_reversals = calculate_false_reversal_penalty(
                filtered_signal,
                clean_signal,
                task.window_size,
            )
            composite_score = calculate_composite_score(
                slope_changes,
                lag_error,
                avg_error,
                false_reversals,
            )
            correlation = _calculate_correlation(filtered_signal, aligned_clean)
            noise_reduction = _calculate_noise_reduction(
                filtered_signal,
                aligned_noisy,
                aligned_clean,
            )

            trial_metrics = {
                "slope_changes": safe_float(slope_changes, 0.0),
                "lag_error": safe_float(lag_error, 0.0),
                "avg_error": safe_float(avg_error, 0.0),
                "false_reversals": safe_float(false_reversals, 0.0),
                "composite_score": safe_float(composite_score, 0.0),
                "correlation": safe_float(correlation, 0.0),
                "noise_reduction": safe_float(noise_reduction, 0.0),
                "execution_time": safe_float(execution_time, timeout_seconds),
            }
            successful_trials.append(trial_metrics)

            if best_trial_summary is None or (
                trial_metrics["composite_score"] > best_trial_summary["composite_score"]
            ):
                best_trial_summary = {
                    "seed": int(seed),
                    "composite_score": trial_metrics["composite_score"],
                    "correlation": trial_metrics["correlation"],
                    "noise_reduction": trial_metrics["noise_reduction"],
                    "execution_time": trial_metrics["execution_time"],
                }
        except Exception as exc:
            failed_trials.append(_trial_error_message(seed, exc))

    raw_metrics = _aggregate_successful_trials(
        successful_trials,
        total_trials=len(seeds),
    )
    error = None if successful_trials else f"All {len(seeds)} trials failed"
    task_result = build_task_result(
        task,
        raw_metrics=raw_metrics if successful_trials else None,
        error=error,
        timeout_seconds=timeout_seconds,
    )

    task_artifacts = {
        "task_id": task.task_id,
        "trial_count": len(seeds),
        "successful_trials": len(successful_trials),
        "failed_trials": len(seeds) - len(successful_trials),
        "failed_trial_messages": failed_trials,
        "best_trial_summary": best_trial_summary,
        "compact_task_summary": {
            "score": float(task_result["final_task_score"]),
            "best_composite_score": (
                float(best_trial_summary["composite_score"])
                if isinstance(best_trial_summary, Mapping)
                else 0.0
            ),
            "best_runtime": (
                float(best_trial_summary["execution_time"])
                if isinstance(best_trial_summary, Mapping)
                else 0.0
            ),
        },
        "convergence_notes": (
            f"Best composite score over {len(successful_trials)}/{len(seeds)} successful trials."
            if successful_trials
            else f"No successful trials over {len(seeds)} attempts."
        ),
    }
    return task_result, task_artifacts


def _public_artifacts(
    *,
    selector: str,
    stage_name: str,
    task_results: list[dict[str, Any]],
    per_task_artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    task_artifacts_list = list(per_task_artifacts)
    artifacts: dict[str, Any] = {
        "task_selector": selector,
        "selected_task_ids": [task_result["task_id"] for task_result in task_results],
        "evaluation_mode": "shared" if len(task_results) > 1 else "task_specific",
        "evaluation_stage": stage_name,
        "task_results": task_results,
        "trial_counts": {},
        "best_observed_per_task": {},
        "compact_task_summary": {},
        "convergence_notes": {},
    }

    for task_artifacts in task_artifacts_list:
        task_id = task_artifacts["task_id"]
        artifacts["trial_counts"][task_id] = {
            "total": int(task_artifacts["trial_count"]),
            "successful": int(task_artifacts["successful_trials"]),
            "failed": int(task_artifacts["failed_trials"]),
        }
        artifacts["convergence_notes"][task_id] = task_artifacts["convergence_notes"]
        artifacts["compact_task_summary"][task_id] = dict(task_artifacts["compact_task_summary"])
        best_trial_summary = task_artifacts.get("best_trial_summary")
        if isinstance(best_trial_summary, Mapping):
            artifacts["best_observed_per_task"][task_id] = dict(best_trial_summary)
        if task_artifacts.get("failed_trial_messages"):
            artifacts.setdefault("failed_trial_messages", {})[task_id] = list(
                task_artifacts["failed_trial_messages"]
            )

    return artifacts


def _error_evaluation_result(
    *,
    selected_tasks: list[SignalProcessingTaskSpec],
    selector: str,
    stage_name: str,
    error: str,
) -> EvaluationResult:
    task_results = [
        build_task_result(
            task,
            raw_metrics=None,
            error=error,
            timeout_seconds=(
                task.trial_timeout_seconds_stage1
                if stage_name == "stage1"
                else task.trial_timeout_seconds_full
            ),
        )
        for task in selected_tasks
    ]
    artifacts = _public_artifacts(
        selector=selector,
        stage_name=stage_name,
        task_results=task_results,
        per_task_artifacts=[
            {
                "task_id": task.task_id,
                "trial_count": len(
                    task.evaluation_seeds_stage1
                    if stage_name == "stage1"
                    else task.evaluation_seeds_full
                ),
                "successful_trials": 0,
                "failed_trials": len(
                    task.evaluation_seeds_stage1
                    if stage_name == "stage1"
                    else task.evaluation_seeds_full
                ),
                "failed_trial_messages": [error],
                "best_trial_summary": None,
                "compact_task_summary": {"score": 0.0, "best_composite_score": 0.0, "best_runtime": 0.0},
                "convergence_notes": error,
            }
            for task in selected_tasks
        ],
    )
    artifacts["error"] = error
    artifacts["status"] = "ERROR"
    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
    return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)


def _evaluate(program_path: str, *, stage1: bool) -> EvaluationResult:
    selector = os.environ.get(
        SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR,
        SIGNAL_PROCESSING_SHARED_SELECTOR,
    )
    selected_tasks = resolve_task_specs(selector)
    stage_name = "stage1" if stage1 else "full"

    try:
        module = _load_program_module(program_path)
        candidate_runner = _resolve_candidate_runner(module)
    except Exception as exc:
        return _error_evaluation_result(
            selected_tasks=selected_tasks,
            selector=selector,
            stage_name=stage_name,
            error=str(exc),
        )

    task_results: list[dict[str, Any]] = []
    task_artifacts: list[dict[str, Any]] = []
    for task in selected_tasks:
        task_result, per_task_artifacts = evaluate_one_task(
            candidate_runner,
            task,
            stage1=stage1,
        )
        task_results.append(task_result)
        task_artifacts.append(per_task_artifacts)

    artifacts = _public_artifacts(
        selector=selector,
        stage_name=stage_name,
        task_results=task_results,
        per_task_artifacts=task_artifacts,
    )
    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
    return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate either one task or the shared signal-processing family."""
    return _evaluate(program_path, stage1=False)


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Cheaper cascade stage that uses fewer trials and shorter timeouts."""
    return _evaluate(program_path, stage1=True)


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation for cascade mode."""
    return evaluate(program_path)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise SystemExit("Usage: python evaluator.py <program_path>")

    try:
        evaluation_result = evaluate(sys.argv[1])
    except Exception:
        print(traceback.format_exc())
        raise

    print(f"Score: {evaluation_result.metrics['score']:.6f}")
    print(f"Combined Score: {evaluation_result.metrics['combined_score']:.6f}")
    if "task_count" in evaluation_result.metrics:
        print(f"Task Count: {int(evaluation_result.metrics['task_count'])}")
