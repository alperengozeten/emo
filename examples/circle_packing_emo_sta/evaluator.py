"""Evaluator for the EMO-STA circle-packing family."""

from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.circle_packing import (
    CIRCLE_PACKING_SHARED_SELECTOR,
    CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR,
    CirclePackingTaskSpec,
    aggregate_task_results,
    build_task_result,
    resolve_holdout_task_specs,
    resolve_task_specs,
)


BOUNDARY_TOLERANCE = 1.0e-6
MAX_VALIDATION_MESSAGES = 5
MAX_TEXT_EXCERPT = 400
OOD_TIMEOUT_OVERRIDE_ENV_VAR = "MT_STS_OOD_TIMEOUT_OVERRIDE_SECONDS"


def _truncate_text(value: str | None, *, limit: int = MAX_TEXT_EXCERPT) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _apply_ood_timeout_override(timeout_seconds: float) -> float:
    raw_override = os.environ.get(OOD_TIMEOUT_OVERRIDE_ENV_VAR)
    if raw_override is None:
        return float(timeout_seconds)
    try:
        override_seconds = float(raw_override)
    except (TypeError, ValueError):
        return float(timeout_seconds)
    if override_seconds <= 0.0:
        return float(timeout_seconds)
    return max(float(timeout_seconds), override_seconds)


def _append_validation_message(validation_details: dict[str, Any], message: str) -> None:
    example_messages = validation_details.setdefault("example_messages", [])
    if len(example_messages) < MAX_VALIDATION_MESSAGES:
        example_messages.append(str(message))


def _empty_validation_details() -> dict[str, Any]:
    return {
        "shape_message": None,
        "boundary_violations": 0,
        "overlap_violations": 0,
        "negative_radius_violations": 0,
        "non_finite_value_violations": 0,
        "example_messages": [],
    }


def _validation_summary(
    validation_details: Mapping[str, Any],
    *,
    sum_mismatch: bool,
) -> dict[str, Any]:
    summary = {
        "boundary_violations": int(validation_details.get("boundary_violations", 0) or 0),
        "overlap_violations": int(validation_details.get("overlap_violations", 0) or 0),
        "negative_radius_violations": int(
            validation_details.get("negative_radius_violations", 0) or 0
        ),
        "non_finite_value_violations": int(
            validation_details.get("non_finite_value_violations", 0) or 0
        ),
        "sum_mismatch": bool(sum_mismatch),
    }
    shape_message = validation_details.get("shape_message")
    if isinstance(shape_message, str) and shape_message:
        summary["shape_message"] = shape_message
    example_messages = validation_details.get("example_messages")
    if isinstance(example_messages, Sequence) and not isinstance(example_messages, (str, bytes)):
        compact_messages = [str(message) for message in example_messages[:MAX_VALIDATION_MESSAGES]]
        if compact_messages:
            summary["example_messages"] = compact_messages
    return summary


def validate_packing(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    n_expected: int,
) -> tuple[bool, dict[str, Any]]:
    """Validate that circles stay inside the unit square and do not overlap."""
    validation_details = _empty_validation_details()

    if centers.shape != (n_expected, 2):
        validation_details["shape_message"] = (
            f"centers must have shape ({n_expected}, 2), got {tuple(centers.shape)}"
        )
        _append_validation_message(validation_details, validation_details["shape_message"])
        return False, validation_details
    if radii.shape != (n_expected,):
        validation_details["shape_message"] = (
            f"radii must have shape ({n_expected},), got {tuple(radii.shape)}"
        )
        _append_validation_message(validation_details, validation_details["shape_message"])
        return False, validation_details

    non_finite_values = int(np.size(centers) - int(np.isfinite(centers).sum()))
    non_finite_values += int(np.size(radii) - int(np.isfinite(radii).sum()))
    validation_details["non_finite_value_violations"] = non_finite_values
    if non_finite_values > 0:
        _append_validation_message(
            validation_details,
            f"Packing contains {non_finite_values} non-finite center/radius values",
        )
        return False, validation_details

    negative_count = int(np.sum(radii < 0.0))
    validation_details["negative_radius_violations"] = negative_count
    if negative_count > 0:
        _append_validation_message(
            validation_details,
            f"Packing contains {negative_count} negative radii",
        )

    for index, (center, radius) in enumerate(zip(centers, radii)):
        x, y = float(center[0]), float(center[1])
        r = float(radius)
        if (
            x - r < -BOUNDARY_TOLERANCE
            or x + r > 1.0 + BOUNDARY_TOLERANCE
            or y - r < -BOUNDARY_TOLERANCE
            or y + r > 1.0 + BOUNDARY_TOLERANCE
        ):
            validation_details["boundary_violations"] += 1
            _append_validation_message(
                validation_details,
                (
                    f"Circle {index} at ({x:.6f}, {y:.6f}) with radius {r:.6f} "
                    "violates the unit-square boundary"
                ),
            )

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            min_distance = float(radii[i] + radii[j])
            if distance < min_distance - BOUNDARY_TOLERANCE:
                validation_details["overlap_violations"] += 1
                _append_validation_message(
                    validation_details,
                    (
                        f"Circles {i} and {j} overlap: distance={distance:.6f}, "
                        f"required={min_distance:.6f}"
                    ),
                )

    is_valid = (
        validation_details["shape_message"] is None
        and validation_details["negative_radius_violations"] == 0
        and validation_details["non_finite_value_violations"] == 0
        and validation_details["boundary_violations"] == 0
        and validation_details["overlap_violations"] == 0
    )
    return is_valid, validation_details


def _subprocess_runner_script(
    *,
    program_path: str,
    result_path: str,
    n_circles: int,
) -> str:
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import os
        import sys
        import traceback
        import numpy as np

        PROGRAM_PATH = {program_path!r}
        RESULT_PATH = {result_path!r}
        N_CIRCLES = {int(n_circles)}

        payload = None
        try:
            sys.path.insert(0, os.path.dirname(PROGRAM_PATH))
            module_name = "circle_packing_emo_sta_candidate"
            spec = importlib.util.spec_from_file_location(module_name, PROGRAM_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load program from {{PROGRAM_PATH}}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            runner = getattr(module, "run_packing", None)
            if not callable(runner):
                runner = getattr(module, "construct_packing", None)
            if not callable(runner):
                raise AttributeError(
                    "Program must define run_packing(n) or construct_packing(n)"
                )

            result = runner(N_CIRCLES)
            if not isinstance(result, (tuple, list)):
                raise TypeError(
                    f"Expected tuple/list return value, got {{type(result).__name__}}"
                )

            if len(result) == 2:
                centers, radii = result
                reported_sum_radii = None
            elif len(result) == 3:
                centers, radii, reported_sum_radii = result
                if reported_sum_radii is not None:
                    reported_sum_radii = float(reported_sum_radii)
            else:
                raise ValueError(
                    f"Expected (centers, radii) or (centers, radii, sum_radii); got {{len(result)}} values"
                )

            centers_array = np.asarray(centers, dtype=float)
            radii_array = np.asarray(radii, dtype=float)
            payload = {{
                "ok": True,
                "centers": centers_array.tolist(),
                "radii": radii_array.tolist(),
                "reported_sum_radii": reported_sum_radii,
            }}
        except Exception as exc:
            payload = {{
                "ok": False,
                "error": f"{{type(exc).__name__}}: {{exc}}",
                "traceback": traceback.format_exc(),
            }}

        with open(RESULT_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        """
    ).strip()


def _run_program_in_subprocess(
    program_path: str,
    task: CirclePackingTaskSpec,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="circle_packing_emo_sta_") as temp_dir:
        temp_root = Path(temp_dir)
        runner_path = temp_root / "runner.py"
        result_path = temp_root / "result.pkl"
        runner_path.write_text(
            _subprocess_runner_script(
                program_path=str(Path(program_path).resolve()),
                result_path=str(result_path),
                n_circles=task.n_circles,
            ),
            encoding="utf-8",
        )

        process = subprocess.Popen(
            [sys.executable, str(runner_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()

        eval_time = time.perf_counter() - start_time
        payload = None
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None

    return {
        "timed_out": timed_out,
        "exit_code": process.returncode,
        "stdout_excerpt": _truncate_text(stdout),
        "stderr_excerpt": _truncate_text(stderr),
        "eval_time": float(eval_time),
        "payload": payload,
    }


def _task_metrics(
    task: CirclePackingTaskSpec,
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    eval_time: float,
    valid: bool,
) -> dict[str, float]:
    if not valid:
        return {
            "sum_radii": 0.0,
            "target_sum_radii": float(task.target_sum_radii),
            "target_ratio": 0.0,
            "validity": 0.0,
            "radius_variance": 0.0,
            "spatial_spread": 0.0,
            "min_radius": 0.0,
            "max_radius": 0.0,
            "eval_time": float(max(0.0, eval_time)),
            "score": 0.0,
            "combined_score": 0.0,
        }

    sum_radii = float(np.sum(radii))
    target_ratio = sum_radii / float(task.target_sum_radii)
    centroid = np.mean(centers, axis=0)
    distances_from_centroid = np.linalg.norm(centers - centroid, axis=1)
    radius_variance = float(np.var(radii) / 0.0625)
    spatial_spread = float(np.std(distances_from_centroid) / (0.5 * np.sqrt(2.0)))
    score = target_ratio
    return {
        "sum_radii": sum_radii,
        "target_sum_radii": float(task.target_sum_radii),
        "target_ratio": target_ratio,
        "validity": 1.0,
        "radius_variance": float(np.clip(radius_variance, 0.0, 1.0)),
        "spatial_spread": float(np.clip(spatial_spread, 0.0, 1.0)),
        "min_radius": float(np.min(radii)),
        "max_radius": float(np.max(radii)),
        "eval_time": float(max(0.0, eval_time)),
        "score": score,
        "combined_score": score,
    }


def evaluate_one_task(
    program_path: str,
    task: CirclePackingTaskSpec,
    *,
    stage1: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timeout_seconds = (
        float(task.timeout_seconds_stage1) if stage1 else float(task.timeout_seconds_full)
    )
    timeout_seconds = _apply_ood_timeout_override(timeout_seconds)
    execution = _run_program_in_subprocess(
        program_path,
        task,
        timeout_seconds=timeout_seconds,
    )
    validation_details = _empty_validation_details()
    sum_mismatch = False
    error = None

    if execution["timed_out"]:
        error = f"Timed out after {timeout_seconds:.0f}s"
        metrics = _task_metrics(
            task,
            centers=np.zeros((task.n_circles, 2), dtype=float),
            radii=np.zeros(task.n_circles, dtype=float),
            eval_time=float(execution["eval_time"]),
            valid=False,
        )
    else:
        payload = execution.get("payload")
        if not isinstance(payload, Mapping):
            error = "Missing subprocess result payload"
            metrics = _task_metrics(
                task,
                centers=np.zeros((task.n_circles, 2), dtype=float),
                radii=np.zeros(task.n_circles, dtype=float),
                eval_time=float(execution["eval_time"]),
                valid=False,
            )
        elif not bool(payload.get("ok", False)):
            error = str(payload.get("error") or "Candidate execution failed")
            metrics = _task_metrics(
                task,
                centers=np.zeros((task.n_circles, 2), dtype=float),
                radii=np.zeros(task.n_circles, dtype=float),
                eval_time=float(execution["eval_time"]),
                valid=False,
            )
        else:
            try:
                centers = np.asarray(payload.get("centers"), dtype=float)
                radii = np.asarray(payload.get("radii"), dtype=float)
            except Exception as exc:
                validation_details["shape_message"] = (
                    f"Could not convert centers/radii to numeric arrays: {exc}"
                )
                _append_validation_message(
                    validation_details,
                    validation_details["shape_message"],
                )
                centers = np.zeros((task.n_circles, 2), dtype=float)
                radii = np.zeros(task.n_circles, dtype=float)
                error = validation_details["shape_message"]
                metrics = _task_metrics(
                    task,
                    centers=centers,
                    radii=radii,
                    eval_time=float(execution["eval_time"]),
                    valid=False,
                )
            else:
                valid, validation_details = validate_packing(
                    centers,
                    radii,
                    n_expected=task.n_circles,
                )
                metrics = _task_metrics(
                    task,
                    centers=centers,
                    radii=radii,
                    eval_time=float(execution["eval_time"]),
                    valid=valid,
                )
                reported_sum_radii = payload.get("reported_sum_radii")
                if reported_sum_radii is not None and valid:
                    try:
                        reported_sum_radii = float(reported_sum_radii)
                    except (TypeError, ValueError):
                        sum_mismatch = True
                    else:
                        sum_mismatch = abs(reported_sum_radii - float(np.sum(radii))) > 1.0e-9

                if not valid:
                    error = str(validation_details.get("shape_message") or "")
                    if not error:
                        example_messages = validation_details.get("example_messages") or []
                        if example_messages:
                            error = str(example_messages[0])
                    if not error:
                        error = "Packing validation failed"

    validation_summary = _validation_summary(
        validation_details,
        sum_mismatch=sum_mismatch,
    )
    task_result = build_task_result(
        task,
        raw_metrics=metrics,
        error=error,
        validation_summary=validation_summary,
    )

    task_artifacts = {
        "task_id": task.task_id,
        "timed_out": bool(execution["timed_out"]),
        "validation_summary": validation_summary,
        "execution_summary": {
            "status": "success" if error is None else "error",
            "timed_out": bool(execution["timed_out"]),
            "exit_code": execution.get("exit_code"),
            "eval_time": float(execution["eval_time"]),
        },
    }
    if error is not None:
        stdout_excerpt = str(execution.get("stdout_excerpt") or "")
        stderr_excerpt = str(execution.get("stderr_excerpt") or "")
        if stdout_excerpt:
            task_artifacts["execution_summary"]["stdout_excerpt"] = stdout_excerpt
        if stderr_excerpt:
            task_artifacts["execution_summary"]["stderr_excerpt"] = stderr_excerpt
        payload = execution.get("payload")
        if isinstance(payload, Mapping):
            traceback_text = _truncate_text(str(payload.get("traceback") or ""))
            if traceback_text:
                task_artifacts["execution_summary"]["traceback_excerpt"] = traceback_text
    task_artifacts["compact_task_summary"] = {
        "score": float(task_result["final_task_score"]),
        "sum_radii": float(task_result["metrics"]["sum_radii"]),
        "target_ratio": float(task_result["metrics"]["target_ratio"]),
        "validity": float(task_result["metrics"]["validity"]),
        "timed_out": bool(execution["timed_out"]),
    }
    return task_result, task_artifacts


def _public_artifacts(
    *,
    selector: str,
    stage_name: str,
    task_results: list[dict[str, Any]],
    per_task_artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    task_artifact_list = list(per_task_artifacts)
    artifacts: dict[str, Any] = {
        "task_selector": selector,
        "selected_task_ids": [task_result["task_id"] for task_result in task_results],
        "evaluation_mode": "shared" if len(task_results) > 1 else "task_specific",
        "evaluation_stage": stage_name,
        "task_results": task_results,
        "subprocess_timeout_by_task": {},
        "validation_summaries": {},
        "compact_task_summary": {},
        "execution_summary": {
            "selected_task_count": len(task_results),
            "timed_out_task_count": 0,
            "successful_task_count": int(sum(1 for result in task_results if not result.get("error"))),
            "failed_task_count": int(sum(1 for result in task_results if result.get("error"))),
        },
    }

    for task_artifacts in task_artifact_list:
        task_id = str(task_artifacts["task_id"])
        timed_out = bool(task_artifacts.get("timed_out", False))
        artifacts["subprocess_timeout_by_task"][task_id] = timed_out
        artifacts["validation_summaries"][task_id] = dict(task_artifacts["validation_summary"])
        artifacts["compact_task_summary"][task_id] = dict(task_artifacts["compact_task_summary"])
        if timed_out:
            artifacts["execution_summary"]["timed_out_task_count"] += 1

    if task_results:
        best_local_result = max(
            task_results,
            key=lambda result: float(result["final_task_score"]),
        )
        artifacts["best_local_summary"] = {
            "task_id": best_local_result["task_id"],
            "score": float(best_local_result["final_task_score"]),
            "sum_radii": float(best_local_result["metrics"]["sum_radii"]),
            "target_ratio": float(best_local_result["metrics"]["target_ratio"]),
        }

    return artifacts


def _resolve_selected_tasks(selector: str) -> list[CirclePackingTaskSpec]:
    normalized = (selector or CIRCLE_PACKING_SHARED_SELECTOR).strip()
    if not normalized or normalized == CIRCLE_PACKING_SHARED_SELECTOR:
        return resolve_task_specs(normalized)
    try:
        return resolve_task_specs(normalized)
    except ValueError:
        return resolve_holdout_task_specs(normalized)


def _evaluate(program_path: str, *, stage1: bool) -> EvaluationResult:
    selector = os.environ.get(
        CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR,
        CIRCLE_PACKING_SHARED_SELECTOR,
    )
    selected_tasks = _resolve_selected_tasks(selector)
    stage_name = "stage1" if stage1 else "full"

    task_results: list[dict[str, Any]] = []
    task_artifacts: list[dict[str, Any]] = []
    for task in selected_tasks:
        task_result, per_task_artifacts = evaluate_one_task(
            program_path,
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
    """Evaluate one task or the shared circle-packing family."""
    return _evaluate(program_path, stage1=False)


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Cheaper cascade stage that uses the lighter per-task timeout."""
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
    for task_result in evaluation_result.artifacts.get("task_results", []):
        print(
            f"{task_result['task_id']}: "
            f"score={task_result['final_task_score']:.6f} "
            f"sum_radii={task_result['metrics']['sum_radii']:.6f}"
        )
