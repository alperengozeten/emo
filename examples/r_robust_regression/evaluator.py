"""Evaluator for the robust-regression task family."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.robust_regression import (
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    FULL_EVAL_SEEDS,
    ROBUST_REGRESSION_SHARED_SELECTOR,
    ROBUST_REGRESSION_TASK_SELECTOR_ENV_VAR,
    STAGE1_SEEDS,
    RobustRegressionDataset,
    RobustRegressionTaskSpec,
    aggregate_seed_results,
    aggregate_task_results,
    build_task_result,
    generate_regression_dataset,
    resolve_task_specs,
)


def _build_r_runner_script(
    program_path: str,
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    results_path: str,
) -> str:
    return f"""
tryCatch({{
  source({json.dumps(program_path)})
  X_train <- as.matrix(read.csv({json.dumps(x_train_path)}, header=FALSE, check.names=FALSE))
  y_train <- as.numeric(read.csv({json.dumps(y_train_path)}, header=FALSE, check.names=FALSE)[[1]])
  X_test <- as.matrix(read.csv({json.dumps(x_test_path)}, header=FALSE, check.names=FALSE))
  result <- main(X_train, y_train, X_test)
  write(
    jsonlite::toJSON(result, auto_unbox=TRUE, null="null", digits=NA),
    {json.dumps(results_path)}
  )
}}, error=function(e) {{
  message(conditionMessage(e))
  quit(save="no", status=1)
}})
"""


def run_r_program_on_dataset(
    program_path: str,
    task: RobustRegressionTaskSpec,
    dataset: RobustRegressionDataset,
    *,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> tuple[Dict[str, Any] | None, str | None, float]:
    """Run the candidate R program on one synthetic train/test dataset."""
    resolved_program_path = str(Path(program_path).resolve())
    start_time = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"{task.task_id}_{dataset.base_seed}_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            x_train_path = temp_dir_path / "X_train.csv"
            y_train_path = temp_dir_path / "y_train.csv"
            x_test_path = temp_dir_path / "X_test.csv"
            results_path = temp_dir_path / "results.json"
            script_path = temp_dir_path / "run_eval.r"

            np.savetxt(x_train_path, dataset.X_train, delimiter=",", fmt="%.17g")
            np.savetxt(y_train_path, dataset.y_train.reshape(-1, 1), delimiter=",", fmt="%.17g")
            np.savetxt(x_test_path, dataset.X_test, delimiter=",", fmt="%.17g")
            script_path.write_text(
                _build_r_runner_script(
                    resolved_program_path,
                    str(x_train_path),
                    str(y_train_path),
                    str(x_test_path),
                    str(results_path),
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                ["Rscript", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=temp_dir,
            )
            runtime = time.perf_counter() - start_time
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                error_message = stderr or stdout or "empty R output"
                return None, f"R execution failed: {error_message}", runtime
            if not results_path.is_file():
                return None, "No results file produced", runtime

            raw_result = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(raw_result, dict):
                return None, "Malformed JSON payload from candidate", runtime
            return raw_result, None, runtime
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout_seconds:.0f}s", float(timeout_seconds)
    except Exception as exc:
        runtime = time.perf_counter() - start_time
        return None, str(exc), runtime


def _coerce_numeric_vector(value: Any, *, field_name: str) -> tuple[np.ndarray | None, str | None]:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None, f"{field_name} must be numeric"
    if array.size == 0:
        return None, f"{field_name} must be non-empty"
    if not np.all(np.isfinite(array)):
        return None, f"{field_name} must contain only finite values"
    return array.astype(float, copy=False), None


def validate_candidate_outputs(
    raw_result: Mapping[str, Any],
    task: RobustRegressionTaskSpec,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Validate candidate outputs for one task/seed evaluation."""
    if "predictions" not in raw_result:
        return None, None, "Candidate did not return predictions"
    if "coefficients" not in raw_result:
        return None, None, "Candidate did not return coefficients"

    predictions, prediction_error = _coerce_numeric_vector(
        raw_result.get("predictions"),
        field_name="predictions",
    )
    if prediction_error is not None:
        return None, None, prediction_error
    coefficients, coefficient_error = _coerce_numeric_vector(
        raw_result.get("coefficients"),
        field_name="coefficients",
    )
    if coefficient_error is not None:
        return None, None, coefficient_error

    if int(predictions.size) != int(task.n_test):
        return None, None, (
            f"predictions length {predictions.size} does not match n_test={task.n_test}"
        )
    expected_coef_count = int(task.n_features) + 1
    if int(coefficients.size) != expected_coef_count:
        return None, None, (
            "coefficients length "
            f"{coefficients.size} does not match n_features + 1 = {expected_coef_count}"
        )
    return predictions, coefficients, None


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_sum_of_squares = float(np.sum((y_true - y_pred) ** 2))
    total_sum_of_squares = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if total_sum_of_squares <= 1e-8:
        return 1.0 if residual_sum_of_squares <= 1e-8 else 0.0
    return 1.0 - (residual_sum_of_squares / total_sum_of_squares)


def compute_seed_metrics(
    predictions: np.ndarray,
    coefficients: np.ndarray,
    dataset: RobustRegressionDataset,
    *,
    runtime: float,
) -> Dict[str, float | int | bool]:
    """Compute all benchmark metrics in Python for one seed."""
    y_signal = np.asarray(dataset.y_test_clean_signal, dtype=float)
    y_noisy = np.asarray(dataset.y_test_noisy, dtype=float)
    true_coefficients = np.asarray(dataset.true_coefficients, dtype=float)

    mse_signal_test = float(np.mean((predictions - y_signal) ** 2))
    mae_signal_test = float(np.mean(np.abs(predictions - y_signal)))
    r2_signal_test = _r2_score(y_signal, predictions)
    nmse_signal_test = mse_signal_test / max(float(np.var(y_signal)), 1e-8)
    signal_score = 1.0 / (1.0 + nmse_signal_test)

    mse_noisy_test = float(np.mean((predictions - y_noisy) ** 2))
    mae_noisy_test = float(np.mean(np.abs(predictions - y_noisy)))
    r2_noisy_test = _r2_score(y_noisy, predictions)
    nmse_noisy_test = mse_noisy_test / max(float(np.var(y_noisy)), 1e-8)
    noisy_score = 1.0 / (1.0 + nmse_noisy_test)

    coef_rel_error = float(
        np.linalg.norm(coefficients - true_coefficients)
        / max(float(np.linalg.norm(true_coefficients)), 1e-8)
    )
    coef_score = 1.0 / (1.0 + coef_rel_error)

    return {
        "seed": int(dataset.base_seed),
        "derived_seed": int(dataset.derived_seed),
        "mse_signal_test": mse_signal_test,
        "mae_signal_test": mae_signal_test,
        "r2_signal_test": r2_signal_test,
        "nmse_signal_test": nmse_signal_test,
        "signal_score": signal_score,
        "mse_noisy_test": mse_noisy_test,
        "mae_noisy_test": mae_noisy_test,
        "r2_noisy_test": r2_noisy_test,
        "nmse_noisy_test": nmse_noisy_test,
        "noisy_score": noisy_score,
        "coef_rel_error": coef_rel_error,
        "coef_score": coef_score,
        "runtime": float(runtime),
        "success": True,
    }


def evaluate_one_task(
    program_path: str,
    task: RobustRegressionTaskSpec,
    *,
    base_seeds: Sequence[int],
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Evaluate one task across a deterministic seed bank."""
    seed_results: list[Dict[str, Any]] = []
    failure_messages: list[str] = []

    for base_seed in base_seeds:
        dataset = generate_regression_dataset(task, int(base_seed))
        raw_result, error, runtime = run_r_program_on_dataset(
            program_path,
            task,
            dataset,
            timeout_seconds=timeout_seconds,
        )
        if error is not None:
            seed_results.append(
                {
                    "seed": int(base_seed),
                    "derived_seed": int(dataset.derived_seed),
                    "runtime": float(runtime),
                    "success": False,
                    "error": error,
                }
            )
            failure_messages.append(f"seed {base_seed}: {error}")
            continue

        predictions, coefficients, validation_error = validate_candidate_outputs(raw_result, task)
        if validation_error is not None:
            seed_results.append(
                {
                    "seed": int(base_seed),
                    "derived_seed": int(dataset.derived_seed),
                    "runtime": float(runtime),
                    "success": False,
                    "error": validation_error,
                }
            )
            failure_messages.append(f"seed {base_seed}: {validation_error}")
            continue

        seed_results.append(
            compute_seed_metrics(
                predictions,
                coefficients,
                dataset,
                runtime=float(runtime),
            )
        )

    aggregated_metrics = aggregate_seed_results(
        seed_results,
        seed_count=len(tuple(base_seeds)),
        timeout_seconds=timeout_seconds,
    )
    task_error = None
    if int(aggregated_metrics["successful_seed_count"]) <= 0:
        task_error = "; ".join(failure_messages[:3]) or "All evaluation seeds failed"
    return build_task_result(
        task,
        raw_metrics=aggregated_metrics,
        error=task_error,
        seed_results=seed_results,
        timeout_seconds=timeout_seconds,
    )


def _evaluate_for_seed_bank(
    program_path: str,
    *,
    stage_name: str,
    seed_bank: Sequence[int],
) -> EvaluationResult:
    selector = os.environ.get(
        ROBUST_REGRESSION_TASK_SELECTOR_ENV_VAR,
        ROBUST_REGRESSION_SHARED_SELECTOR,
    )
    selected_tasks = resolve_task_specs(selector)
    task_results = []
    for task in selected_tasks:
        timeout_seconds = (
            task.trial_timeout_seconds_stage1
            if stage_name == "stage1"
            else task.trial_timeout_seconds_full
        )
        task_seed_bank = (
            task.evaluation_seeds_stage1 if stage_name == "stage1" else task.evaluation_seeds_full
        )
        if tuple(task_seed_bank) != tuple(seed_bank):
            task_seed_bank = tuple(seed_bank)
        task_results.append(
            evaluate_one_task(
                program_path,
                task,
                base_seeds=task_seed_bank,
                timeout_seconds=timeout_seconds,
            )
        )

    artifacts = {
        "task_selector": selector,
        "evaluation_stage": stage_name,
        "task_results": task_results,
    }

    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)

    return EvaluationResult(
        metrics=aggregate_task_results(task_results),
        artifacts=artifacts,
    )


def evaluate(program_path: str) -> EvaluationResult:
    """Full evaluation using the full deterministic seed bank."""
    return _evaluate_for_seed_bank(
        program_path,
        stage_name="full",
        seed_bank=FULL_EVAL_SEEDS,
    )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Stage-1 evaluation using the reduced deterministic seed bank."""
    return _evaluate_for_seed_bank(
        program_path,
        stage_name="stage1",
        seed_bank=STAGE1_SEEDS,
    )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Stage-2 cascade entrypoint for full evaluation."""
    return evaluate(program_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/r_robust_regression/evaluator.py <program.r>")

    evaluation_result = evaluate(sys.argv[1])
    print(json.dumps(evaluation_result.metrics, indent=2, sort_keys=True))
