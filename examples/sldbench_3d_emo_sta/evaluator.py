"""Evaluator for the EMO-STA SLDBench 3D family."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.sldbench_3d import (
    SLDBENCH_3D_SHARED_SELECTOR,
    SLDBENCH_3D_TASK_SELECTOR_ENV_VAR,
    SLDBench3DTaskSpec,
    aggregate_task_results,
    build_task_result,
    resolve_task_specs,
)


def _load_local_module(module_filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, THIS_DIR / module_filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DATA_LOADER_MODULE = _load_local_module(
    "data_loader.py",
    "sldbench_3d_emo_sta_data_loader",
)
get_loader_mode = _DATA_LOADER_MODULE.get_loader_mode
load_grouped_data = _DATA_LOADER_MODULE.load_grouped_data


DEFAULT_CALL_TIMEOUT_SECONDS = 120.0


class EvaluationTimeoutError(TimeoutError):
    """Raised when one fit or prediction call exceeds the per-call timeout."""


def run_with_timeout(
    func: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
) -> Any:
    """Run a callable in a worker thread with a timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **(dict(kwargs or {})))
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise EvaluationTimeoutError(
                f"Call timed out after {timeout_seconds:.1f} seconds"
            ) from exc


def _load_program_module(program_path: str):
    module_name = f"sldbench_3d_emo_sta_program_{hash(Path(program_path).resolve())}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_candidate_functions(module: Any) -> tuple[Callable[..., Any], Callable[..., Any]]:
    fit_scaling_law = getattr(module, "fit_scaling_law", None)
    scaling_law_func = getattr(module, "scaling_law_func", None)
    if not callable(fit_scaling_law):
        raise AttributeError("Program must define fit_scaling_law(data_points, loss_values)")
    if not callable(scaling_law_func):
        raise AttributeError("Program must define scaling_law_func(data_points, params)")
    return fit_scaling_law, scaling_law_func


def _normalize_fitted_params(params: Any, task: SLDBench3DTaskSpec) -> np.ndarray:
    params_array = np.asarray(params, dtype=float)
    if params_array.ndim == 2 and params_array.shape[0] == 1:
        params_array = params_array.reshape(-1)
    if params_array.ndim != 1:
        raise ValueError(
            f"fit_scaling_law for task '{task.task_id}' must return shape (P,) or (1, P)"
        )
    if params_array.size < 1 or params_array.size > int(task.param_budget):
        raise ValueError(
            f"fit_scaling_law for task '{task.task_id}' must return between 1 and "
            f"{task.param_budget} parameters, got {params_array.size}"
        )
    if not np.all(np.isfinite(params_array)):
        raise ValueError(f"fit_scaling_law for task '{task.task_id}' returned non-finite params")
    return params_array.astype(float, copy=False)


def _normalize_predictions(
    predictions: Any,
    *,
    expected_length: int,
    task_id: str,
) -> np.ndarray:
    prediction_array = np.asarray(predictions, dtype=float)
    if prediction_array.ndim == 2 and prediction_array.shape == (expected_length, 1):
        prediction_array = prediction_array[:, 0]
    if prediction_array.ndim != 1 or prediction_array.size != int(expected_length):
        raise ValueError(
            f"Predictions for task '{task_id}' must have shape ({expected_length},) or "
            f"({expected_length}, 1), got {prediction_array.shape}"
        )
    if not np.all(np.isfinite(prediction_array)):
        raise ValueError(f"Predictions for task '{task_id}' must be finite")
    return prediction_array.astype(float, copy=False)


def _calculate_scalar_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> Dict[str, float]:
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    true = np.asarray(true_values, dtype=float).reshape(-1)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError(
            f"Prediction/target shape mismatch: predictions {pred.shape}, targets {true.shape}"
        )
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
        raise ValueError("Predictions and targets must be finite")

    variance = float(np.var(true))
    mean_abs_deviation = float(np.mean(np.abs(true - np.mean(true))))
    if variance <= 1.0e-12 or mean_abs_deviation <= 1.0e-12:
        raise ValueError("Test targets must have non-zero variance and mean absolute deviation")

    mse = float(np.mean((true - pred) ** 2))
    mae = float(np.mean(np.abs(true - pred)))
    nmse = mse / variance
    nmae = mae / mean_abs_deviation
    r2 = 1.0 - nmse
    if not all(np.isfinite([nmse, nmae, r2])):
        raise ValueError("Computed metrics must be finite")
    score = 1.0 / (1.0 + nmse)
    return {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "score": float(score),
        "combined_score": float(score),
    }


def _task_summary(task_result: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = task_result["metrics"]
    return {
        "score": float(metrics["score"]),
        "nmse": float(metrics["nmse"]),
        "nmae": float(metrics["nmae"]),
        "r2": float(metrics["r2"]),
        "train_group_count": int(summary["train_group_count"]),
        "test_group_count": int(summary["test_group_count"]),
        "shared_group_count": int(summary["shared_group_count"]),
        "fit_group_count": int(summary["fit_group_count"]),
        "eval_group_count": int(summary["eval_group_count"]),
        "successful_group_count": int(summary["successful_group_count"]),
        "failed_group_count": int(summary["failed_group_count"]),
    }


def evaluate_one_task(
    fit_scaling_law: Callable[..., Any],
    scaling_law_func: Callable[..., Any],
    task: SLDBench3DTaskSpec,
    *,
    timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_data = load_grouped_data(task.task_id, "train")
    test_data = load_grouped_data(task.task_id, "test")
    shared_group_keys = sorted(set(train_data) & set(test_data))
    train_only_group_keys = sorted(set(train_data) - set(test_data))
    test_only_group_keys = sorted(set(test_data) - set(train_data))

    fitted_params_by_group: dict[str, np.ndarray] = {}
    fit_failure_messages: list[str] = []
    for group_key, (X_train, y_train) in train_data.items():
        try:
            raw_params = run_with_timeout(
                fit_scaling_law,
                args=(np.asarray(X_train, dtype=float).copy(), np.asarray(y_train, dtype=float).copy()),
                timeout_seconds=timeout_seconds,
            )
            fitted_params_by_group[group_key] = _normalize_fitted_params(raw_params, task)
        except Exception as exc:
            fit_failure_messages.append(
                f"{group_key}: {type(exc).__name__}: {exc}"
            )

    successful_predictions: list[np.ndarray] = []
    successful_targets: list[np.ndarray] = []
    evaluation_failure_messages: list[str] = []
    successful_group_count = 0
    for group_key in shared_group_keys:
        params = fitted_params_by_group.get(group_key)
        if params is None:
            evaluation_failure_messages.append(
                f"{group_key}: missing fitted params for shared test group"
            )
            continue

        X_test, y_test = test_data[group_key]
        try:
            raw_predictions = run_with_timeout(
                scaling_law_func,
                args=(np.asarray(X_test, dtype=float).copy(), params.copy()),
                timeout_seconds=timeout_seconds,
            )
            predictions = _normalize_predictions(
                raw_predictions,
                expected_length=int(np.asarray(y_test).reshape(-1).size),
                task_id=task.task_id,
            )
        except Exception as exc:
            evaluation_failure_messages.append(
                f"{group_key}: {type(exc).__name__}: {exc}"
            )
            continue

        successful_predictions.append(predictions)
        successful_targets.append(np.asarray(y_test, dtype=float).reshape(-1))
        successful_group_count += 1

    summary = {
        "train_group_count": len(train_data),
        "test_group_count": len(test_data),
        "shared_group_count": len(shared_group_keys),
        "train_only_group_count": len(train_only_group_keys),
        "test_only_group_count": len(test_only_group_keys),
        "fit_group_count": len(fitted_params_by_group),
        "eval_group_count": len(shared_group_keys),
        "successful_group_count": successful_group_count,
        "failed_group_count": max(0, len(shared_group_keys) - successful_group_count),
        "fit_failure_examples": fit_failure_messages[:5],
        "evaluation_failure_examples": evaluation_failure_messages[:5],
    }

    if not successful_predictions:
        return (
            build_task_result(
                task,
                raw_metrics=None,
                error="No successful evaluation groups",
            ),
            summary,
        )

    try:
        metrics = _calculate_scalar_metrics(
            np.concatenate(successful_predictions),
            np.concatenate(successful_targets),
        )
    except Exception as exc:
        return (
            build_task_result(
                task,
                raw_metrics=None,
                error=f"{type(exc).__name__}: {exc}",
            ),
            summary,
        )

    metrics.update(
        {
            "fit_group_count": float(summary["fit_group_count"]),
            "eval_group_count": float(summary["eval_group_count"]),
            "successful_group_count": float(summary["successful_group_count"]),
            "failed_group_count": float(summary["failed_group_count"]),
        }
    )
    return build_task_result(task, raw_metrics=metrics), summary


def _build_artifacts(
    *,
    selector: str,
    evaluation_mode: str,
    loader_mode: str,
    task_results: list[dict[str, Any]],
    task_summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    compact_task_summary = {
        task_id: _task_summary(task_results_by_id, summary)
        for task_id, task_results_by_id, summary in (
            (task_result["task_id"], task_result, task_summaries[task_result["task_id"]])
            for task_result in task_results
        )
    }
    has_task_failures = any(task_result.get("error") for task_result in task_results)
    return {
        "task_selector": selector,
        "selected_task_ids": [task_result["task_id"] for task_result in task_results],
        "evaluation_mode": evaluation_mode,
        "loader_mode": loader_mode,
        "task_results": task_results,
        "task_summaries": {
            task_id: dict(summary) for task_id, summary in task_summaries.items()
        },
        "compact_task_summary": compact_task_summary,
        "status": "completed_with_task_failures" if has_task_failures else "ok",
    }


def evaluate(program_path: str) -> EvaluationResult:
    selector = os.getenv(SLDBENCH_3D_TASK_SELECTOR_ENV_VAR, SLDBENCH_3D_SHARED_SELECTOR)
    selected_tasks = resolve_task_specs(selector)
    evaluation_mode = (
        "shared"
        if selector == SLDBENCH_3D_SHARED_SELECTOR or len(selected_tasks) > 1
        else "task_specific"
    )
    loader_mode = get_loader_mode()
    timeout_seconds = float(
        os.getenv("SLDBENCH_3D_CALL_TIMEOUT_SECONDS", str(DEFAULT_CALL_TIMEOUT_SECONDS))
    )

    try:
        module = _load_program_module(program_path)
        fit_scaling_law, scaling_law_func = _resolve_candidate_functions(module)
    except Exception as exc:
        task_results = [
            build_task_result(task, raw_metrics=None, error=f"{type(exc).__name__}: {exc}")
            for task in selected_tasks
        ]
        artifacts = _build_artifacts(
            selector=selector,
            evaluation_mode=evaluation_mode,
            loader_mode=loader_mode,
            task_results=task_results,
            task_summaries={
                task.task_id: {
                    "train_group_count": 0,
                    "test_group_count": 0,
                    "shared_group_count": 0,
                    "train_only_group_count": 0,
                    "test_only_group_count": 0,
                    "fit_group_count": 0,
                    "eval_group_count": 0,
                    "successful_group_count": 0,
                    "failed_group_count": 0,
                    "fit_failure_examples": [],
                    "evaluation_failure_examples": [],
                }
                for task in selected_tasks
            },
        )
        if len(task_results) == 1:
            return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
        return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)

    task_results: list[dict[str, Any]] = []
    task_summaries: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        try:
            task_result, summary = evaluate_one_task(
                fit_scaling_law,
                scaling_law_func,
                task,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - defensive path.
            task_result = build_task_result(
                task,
                raw_metrics=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            summary = {
                "train_group_count": 0,
                "test_group_count": 0,
                "shared_group_count": 0,
                "train_only_group_count": 0,
                "test_only_group_count": 0,
                "fit_group_count": 0,
                "eval_group_count": 0,
                "successful_group_count": 0,
                "failed_group_count": 0,
                "fit_failure_examples": [],
                "evaluation_failure_examples": [
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(limit=5),
                ],
            }
        task_results.append(task_result)
        task_summaries[task.task_id] = summary

    artifacts = _build_artifacts(
        selector=selector,
        evaluation_mode=evaluation_mode,
        loader_mode=loader_mode,
        task_results=task_results,
        task_summaries=task_summaries,
    )
    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
    return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an EMO-STA SLDBench 3D candidate.")
    parser.add_argument("program_path", type=str, help="Path to the candidate Python program.")
    args = parser.parse_args()

    result = evaluate(args.program_path)
    print(
        json.dumps(
            {
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    all_failed = all(task_result.get("error") for task_result in result.artifacts["task_results"])
    return 1 if all_failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
