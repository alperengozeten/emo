"""Evaluator for the EMO-STA function-minimization family."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from itertools import count
import math
from pathlib import Path
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.function_minimization import (
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    FUNCTION_MINIMIZATION_SHARED_SELECTOR,
    FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR,
    FunctionMinimizationTaskSpec,
    aggregate_task_results,
    build_task_result,
    resolve_task_specs,
    score_task_metrics,
)


class TrialTimeoutError(TimeoutError):
    """Raised when a single search trial exceeds the evaluator budget."""


_OPAQUE_OBJECTIVE_REGISTRY: dict[int, Callable[[float, float], float]] = {}
_OPAQUE_OBJECTIVE_TOKEN_COUNTER = count(1)


class _OpaqueObjective:
    """Callable view of the objective that hides task identity from candidates."""

    __slots__ = ("_token",)

    def __init__(self, token: int):
        object.__setattr__(self, "_token", int(token))

    def __call__(self, x: float, y: float) -> float:
        token = object.__getattribute__(self, "_token")
        return float(_OPAQUE_OBJECTIVE_REGISTRY[token](x, y))

    def __dir__(self) -> list[str]:
        return ["__call__"]

    def __getattribute__(self, name: str) -> Any:
        if name in {"__call__", "__class__", "__dir__", "__repr__"}:
            return object.__getattribute__(self, name)
        raise AttributeError(f"{type(self).__name__} does not expose {name!r}")

    def __repr__(self) -> str:
        return "OpaqueObjective2D()"


@contextmanager
def _opaque_objective(fn: Callable[[float, float], float]):
    token = next(_OPAQUE_OBJECTIVE_TOKEN_COUNTER)
    _OPAQUE_OBJECTIVE_REGISTRY[token] = fn
    try:
        yield _OpaqueObjective(token)
    finally:
        _OPAQUE_OBJECTIVE_REGISTRY.pop(token, None)


def _load_program_module(program_path: str):
    module_name = f"function_minimization_emo_sta_program_{hash(Path(program_path).resolve())}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_candidate_runner(module: Any) -> Callable[..., Any]:
    for candidate_name in ("run_search", "search_algorithm"):
        candidate = getattr(module, candidate_name, None)
        if callable(candidate):
            return candidate
    raise AttributeError("Program must define run_search() or search_algorithm()")


@contextmanager
def _time_limit(timeout_seconds: float):
    if timeout_seconds <= 0.0:
        yield
        return

    # OpenEvolve runs function-minimization evaluations inside executor worker
    # threads during cascade evaluation. Installing a SIGALRM handler from a
    # non-main thread raises ValueError, which otherwise makes every trial fail
    # and collapses shared-mode scores to zero. In that threaded path, rely on
    # OpenEvolve's outer evaluator timeout instead.
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):
        del signum, frame
        raise TrialTimeoutError(f"Trial timed out after {timeout_seconds:.0f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _coerce_finite_float(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {numeric}")
    return numeric


def _normalize_result(result: Any) -> tuple[float, float, float | None]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise ValueError(f"Expected a tuple/list result, got {type(result).__name__}")
    if len(result) == 2:
        x, y = result
        return x, y, None
    if len(result) == 3:
        x, y, value = result
        return x, y, value
    raise ValueError(f"Expected 2 or 3 return values, got {len(result)}")


def _point_within_bounds(x: float, y: float, bounds: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    (x_min, x_max), (y_min, y_max) = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def _trial_error_message(seed: int, exc: Exception) -> str:
    return f"seed={seed}: {type(exc).__name__}: {exc}"


def _evaluate_trial(
    candidate_runner: Callable[..., Any],
    task: FunctionMinimizationTaskSpec,
    *,
    iterations: int,
    seed: int,
    timeout_seconds: float,
) -> dict[str, float]:
    start_time = time.perf_counter()
    with _opaque_objective(task.objective_fn) as objective_fn:
        with _time_limit(timeout_seconds):
            raw_result = candidate_runner(
                objective_fn,
                task.bounds,
                iterations=int(iterations),
                seed=int(seed),
            )
    runtime = time.perf_counter() - start_time

    x, y, reported_value = _normalize_result(raw_result)
    x_value = _coerce_finite_float(x, name="x")
    y_value = _coerce_finite_float(y, name="y")
    if reported_value is not None:
        _coerce_finite_float(reported_value, name="value")
    if not _point_within_bounds(x_value, y_value, task.bounds):
        raise ValueError(f"Returned point ({x_value}, {y_value}) is outside task bounds")

    best_value = float(task.objective_fn(x_value, y_value))
    distance = math.sqrt((x_value - task.optimum_x) ** 2 + (y_value - task.optimum_y) ** 2)
    return {
        "x": x_value,
        "y": y_value,
        "best_value": best_value,
        "distance_to_optimum": distance,
        "runtime": runtime,
    }


def evaluate_one_task(
    candidate_runner: Callable[..., Any],
    task: FunctionMinimizationTaskSpec,
    *,
    stage1: bool = False,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = task.evaluation_seeds_stage1 if stage1 else task.evaluation_seeds_full
    iterations = task.search_iterations_stage1 if stage1 else task.search_iterations_full

    successful_trials: list[dict[str, float]] = []
    failed_trials: list[str] = []
    for seed in seeds:
        try:
            successful_trials.append(
                _evaluate_trial(
                    candidate_runner,
                    task,
                    iterations=iterations,
                    seed=seed,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:
            failed_trials.append(_trial_error_message(seed, exc))

    raw_metrics = score_task_metrics(
        optimum_value=task.optimum_value,
        best_values=[trial["best_value"] for trial in successful_trials],
        distances=[trial["distance_to_optimum"] for trial in successful_trials],
        eval_times=[trial["runtime"] for trial in successful_trials],
        total_trials=len(seeds),
    )
    error = None if successful_trials else f"All {len(seeds)} trials failed"
    task_result = build_task_result(
        task,
        raw_metrics=raw_metrics,
        error=error,
        timeout_seconds=timeout_seconds,
    )

    task_artifacts = {
        "task_id": task.task_id,
        "trial_count": len(seeds),
        "successful_trials": len(successful_trials),
        "failed_trials": len(seeds) - len(successful_trials),
        "failed_trial_messages": failed_trials,
        "convergence_notes": (
            f"Best value over {len(successful_trials)}/{len(seeds)} successful trials."
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
        "evaluation_stage": stage_name,
        "evaluation_mode": "shared" if len(task_results) > 1 else "task_specific",
        "selected_task_ids": [task_result["task_id"] for task_result in task_results],
        "task_results": task_results,
        "trial_counts": {},
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
        if task_artifacts.get("failed_trial_messages"):
            artifacts.setdefault("failed_trial_messages", {})[task_id] = list(
                task_artifacts["failed_trial_messages"]
            )

    return artifacts


def _error_evaluation_result(
    *,
    selected_tasks: list[FunctionMinimizationTaskSpec],
    selector: str,
    stage_name: str,
    error: str,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> EvaluationResult:
    task_results = [
        build_task_result(
            task,
            raw_metrics=None,
            error=error,
            timeout_seconds=timeout_seconds,
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
                "trial_count": len(task.evaluation_seeds_stage1 if stage_name == "stage1" else task.evaluation_seeds_full),
                "successful_trials": 0,
                "failed_trials": len(
                    task.evaluation_seeds_stage1 if stage_name == "stage1" else task.evaluation_seeds_full
                ),
                "failed_trial_messages": [error],
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
        FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR,
        FUNCTION_MINIMIZATION_SHARED_SELECTOR,
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
            timeout_seconds=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
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
    """Evaluate either one task or the shared function-minimization family."""
    return _evaluate(program_path, stage1=False)


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Cheaper cascade stage that uses fewer iterations and seeds per task."""
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
