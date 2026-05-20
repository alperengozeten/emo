"""Evaluator for the K-module EMO-STA family."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.k_module_problem_balanced import (
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    K_MODULE_BALANCED_SHARED_SELECTOR,
    K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR,
    KModuleBalancedTaskSpec,
    aggregate_task_results,
    build_task_result,
    count_correct_modules,
    resolve_task_specs,
    search_space_size,
    validate_candidate_config,
)


def _load_program_module(program_path: str):
    module_name = (
        "k_module_problem_emo_sta_program_"
        f"{hash(Path(program_path).resolve())}"
    )
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_candidate_config(program_path: str) -> Mapping[str, Any]:
    module = _load_program_module(program_path)
    if hasattr(module, "run_pipeline"):
        config = module.run_pipeline()
    elif hasattr(module, "configure_pipeline"):
        config = module.configure_pipeline()
    else:
        raise AttributeError("Program must define run_pipeline() or configure_pipeline()")

    validation_errors = validate_candidate_config(config)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return config


def evaluate_one_task(
    candidate_config: Mapping[str, Any],
    task: KModuleBalancedTaskSpec,
    *,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    start_time = time.time()
    try:
        correct_modules = count_correct_modules(task, candidate_config)
        eval_time = time.time() - start_time
        return build_task_result(
            task,
            raw_metrics={
                "correct_modules": correct_modules,
                "total_modules": 6,
                "accuracy": correct_modules / 6.0,
                "score": correct_modules / 6.0,
                "combined_score": correct_modules / 6.0,
                "eval_time": eval_time,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return build_task_result(
            task,
            raw_metrics=None,
            error=str(exc),
            timeout_seconds=timeout_seconds,
        )


def _public_artifacts(
    *,
    selector: str,
    task_results: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "task_selector": selector,
        "task_results": task_results,
        "search_space_size": search_space_size(),
        "evaluation_mode": "shared" if len(task_results) > 1 else "task_specific",
    }
    if error:
        artifacts["status"] = "error"
        artifacts["error"] = error
        return artifacts

    artifacts["status"] = (
        "shared_evaluation_complete" if len(task_results) > 1 else "task_evaluation_complete"
    )
    return artifacts


def _error_evaluation_result(
    *,
    selected_tasks: list[KModuleBalancedTaskSpec],
    selector: str,
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
        task_results=task_results,
        error=error,
    )
    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
    return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate either one task or the full shared K-module EMO-STA family."""
    selector = os.environ.get(
        K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR,
        K_MODULE_BALANCED_SHARED_SELECTOR,
    )
    selected_tasks = resolve_task_specs(selector)

    try:
        candidate_config = _load_candidate_config(program_path)
    except Exception as exc:
        return _error_evaluation_result(
            selected_tasks=selected_tasks,
            selector=selector,
            error=str(exc),
        )

    task_results = [
        evaluate_one_task(
            candidate_config,
            task,
            timeout_seconds=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
        )
        for task in selected_tasks
    ]
    artifacts = _public_artifacts(
        selector=selector,
        task_results=task_results,
    )
    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
    return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise SystemExit("Usage: python evaluator.py <program_path>")

    try:
        evaluation_result = evaluate(sys.argv[1])
    except Exception:
        print(traceback.format_exc())
        raise

    print(f"Score: {evaluation_result.metrics['score']:.4f}")
    print(f"Combined Score: {evaluation_result.metrics['combined_score']:.4f}")
    if "task_count" in evaluation_result.metrics:
        print(f"Task Count: {int(evaluation_result.metrics['task_count'])}")
    else:
        print(
            "Correct Modules: "
            f"{int(evaluation_result.metrics['correct_modules'])}/"
            f"{int(evaluation_result.metrics['total_modules'])}"
        )
