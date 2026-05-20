"""Shared evaluator for the synthetic ARC multitask prototype."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from openevolve.evaluation_result import EvaluationResult

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from synthetic_tasks import ArcExample, get_task_spec  # noqa: E402

ARC_SYNTHETIC_TASK_ID = "ARC_SYNTHETIC_TASK_ID"


def _get_task_id() -> str:
    return os.getenv(ARC_SYNTHETIC_TASK_ID, "rotate_90_cw")


def _coerce_prediction(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("Candidate must return a numpy.ndarray.")
    return value.astype(np.int32, copy=False)


def _load_program_module(program_path: str):
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import candidate program from '{program_path}'.")
    program_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program_module)
    return program_module


def pass_at_2_accuracy_single(
    attempts: List[np.ndarray],
    ground_truth: np.ndarray,
) -> Tuple[int, Dict[int, Dict[str, Any]]]:
    """Compute pass@2 diagnostics for one example."""
    if len(attempts) != 2:
        raise ValueError("Expected exactly two attempts for pass@2 evaluation.")

    diagnostics: Dict[int, Dict[str, Any]] = {}
    passed = False

    for attempt_index, pred in enumerate(attempts):
        attempt_info: Dict[str, Any] = {}
        if pred.shape != ground_truth.shape:
            attempt_info["size_match"] = False
            attempt_info["pred_shape"] = list(pred.shape)
            attempt_info["gt_shape"] = list(ground_truth.shape)
            attempt_info["incorrect_indices"] = None
            attempt_info["num_incorrect"] = None
            attempt_passed = False
        else:
            incorrect_mask = pred != ground_truth
            incorrect_indices = np.argwhere(incorrect_mask)
            attempt_info["size_match"] = True
            attempt_info["pred_shape"] = list(pred.shape)
            attempt_info["gt_shape"] = list(ground_truth.shape)
            attempt_info["incorrect_indices"] = incorrect_indices.tolist()
            attempt_info["num_incorrect"] = int(incorrect_mask.sum())
            attempt_passed = bool(incorrect_mask.sum() == 0)

        attempt_info["perfect_match"] = attempt_passed
        diagnostics[attempt_index] = attempt_info
        passed = passed or attempt_passed

    return (1 if passed else 0), diagnostics


def _evaluate_split(
    program_module,
    *,
    examples: List[ArcExample],
    split_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    split_passes: List[int] = []

    for example_index, example in enumerate(examples):
        attempts = []
        for attempt_index in (1, 2):
            transform = getattr(program_module, f"transform_grid_attempt_{attempt_index}")
            prediction = _coerce_prediction(transform(example.input_grid.copy()))
            attempts.append(prediction)

        passed, diagnostics = pass_at_2_accuracy_single(attempts, example.output_grid)
        split_passes.append(passed)
        prefix = f"{split_name}_example_{example_index}"
        metrics[f"{prefix}_pass_at_2"] = passed

        for attempt_index, attempt_diagnostics in diagnostics.items():
            metrics[f"{prefix}_attempt_{attempt_index}"] = bool(
                attempt_diagnostics["perfect_match"]
            )
            if not attempt_diagnostics["perfect_match"]:
                artifacts[f"{prefix}_attempt_{attempt_index}_diagnostics"] = attempt_diagnostics

    score = float(sum(split_passes) / len(split_passes)) if split_passes else 0.0
    metrics[f"{split_name}_combined_score"] = score
    return metrics, artifacts


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate on the selected ARC synthetic task."""
    task_id = _get_task_id()
    task_spec = get_task_spec(task_id)

    try:
        program_module = _load_program_module(program_path)
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0,
                "combined_score": 0.0,
                "task_id": task_id,
                "error": f"Program import failed: {exc}",
            },
            artifacts={"error_type": "ImportFailure", "task_id": task_id},
        )

    missing_functions = [
        function_name
        for function_name in ("transform_grid_attempt_1", "transform_grid_attempt_2")
        if not hasattr(program_module, function_name)
    ]
    if missing_functions:
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0,
                "combined_score": 0.0,
                "task_id": task_id,
                "error": f"Missing required functions: {', '.join(missing_functions)}",
            },
            artifacts={
                "error_type": "MissingFunction",
                "task_id": task_id,
                "missing_functions": missing_functions,
            },
        )

    try:
        train_metrics, train_artifacts = _evaluate_split(
            program_module,
            examples=task_spec.train_examples,
            split_name="train",
        )
        heldout_metrics, heldout_artifacts = _evaluate_split(
            program_module,
            examples=task_spec.heldout_examples,
            split_name="heldout",
        )
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0,
                "combined_score": 0.0,
                "task_id": task_id,
                "error": f"Evaluation failed: {exc}",
            },
            artifacts={"error_type": "RuntimeFailure", "task_id": task_id},
        )

    metrics: Dict[str, Any] = {
        "runs_successfully": 1.0,
        "combined_score": train_metrics["train_combined_score"],
        "heldout_score": heldout_metrics["heldout_combined_score"],
        "task_id": task_id,
    }
    metrics.update(train_metrics)
    metrics.update(heldout_metrics)

    artifacts: Dict[str, Any] = {
        "task_id": task_id,
        "task_title": task_spec.title,
        "task_transformation_summary": task_spec.transformation_summary,
    }
    artifacts.update(train_artifacts)
    artifacts.update(heldout_artifacts)

    return EvaluationResult(metrics=metrics, artifacts=artifacts)
