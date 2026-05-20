"""Evaluator for the EMO-STA Heilbronn-triangle family."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from typing import Any, Dict, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult

try:
    from openevolve.multi_task_shared_then_adapt.heilbronn_triangle import (
        HEILBRONN_TRIANGLE_SHARED_SELECTOR,
        HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR,
        HeilbronnTriangleTaskSpec,
        aggregate_task_results,
        build_task_result,
        resolve_eval_task_specs,
    )
except ImportError:
    # Older long-lived worker processes may still expose a family module shape
    # without the newer evaluator-only selector helper. For in-distribution
    # reevaluation, falling back to resolve_task_specs is equivalent.
    from openevolve.multi_task_shared_then_adapt.heilbronn_triangle import (
        HEILBRONN_TRIANGLE_SHARED_SELECTOR,
        HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR,
        HeilbronnTriangleTaskSpec,
        aggregate_task_results,
        build_task_result,
        resolve_task_specs as resolve_eval_task_specs,
    )


BOUNDARY_TOLERANCE = 1.0e-9
CANONICAL_TRIANGLE_BBOX_DIAMETER = math.sqrt(5.0)
MAX_TEXT_EXCERPT = 400
REPORTED_MIN_AREA_ABS_TOL = 1.0e-7
REPORTED_MIN_AREA_REL_TOL = 1.0e-6
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


def _empty_validation_summary() -> dict[str, Any]:
    return {
        "boundary_violations": 0,
        "nonfinite_points": 0,
        "shape_valid": False,
        "finite_values_valid": False,
        "containment_valid": False,
        "reported_min_area_mismatch": False,
    }


def _reported_min_area_mismatch(
    reported_min_area: float,
    evaluator_min_area: float,
) -> bool:
    if not math.isfinite(reported_min_area):
        return True
    return not bool(
        np.isclose(
            reported_min_area,
            evaluator_min_area,
            atol=REPORTED_MIN_AREA_ABS_TOL,
            rtol=REPORTED_MIN_AREA_REL_TOL,
        )
    )


def validate_point_set(
    points: np.ndarray,
    *,
    n_expected: int,
    tol: float = BOUNDARY_TOLERANCE,
) -> tuple[bool, dict[str, Any], np.ndarray | None]:
    """Validate shape, finiteness, and containment in the canonical triangle."""
    validation_summary = _empty_validation_summary()

    if points.shape != (n_expected, 2):
        validation_summary["shape_valid"] = False
        validation_summary["shape_message"] = (
            f"points must have shape ({n_expected}, 2), got {tuple(points.shape)}"
        )
        return False, validation_summary, None
    validation_summary["shape_valid"] = True

    nonfinite_points = int(np.size(points) - int(np.isfinite(points).sum()))
    validation_summary["nonfinite_points"] = nonfinite_points
    if nonfinite_points > 0:
        return False, validation_summary, None
    validation_summary["finite_values_valid"] = True

    lambda_b = points[:, 0] / 2.0
    lambda_c = points[:, 1]
    lambda_a = 1.0 - lambda_b - lambda_c
    barycentric_coords = np.stack((lambda_a, lambda_b, lambda_c), axis=1)

    boundary_violations = int(
        np.sum(
            (points[:, 0] < -tol)
            | (points[:, 1] < -tol)
            | ((points[:, 0] / 2.0 + points[:, 1]) > 1.0 + tol)
        )
    )
    validation_summary["boundary_violations"] = boundary_violations
    if boundary_violations > 0:
        return False, validation_summary, barycentric_coords
    validation_summary["containment_valid"] = True

    return True, validation_summary, barycentric_coords


def compute_min_triangle_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    if n_points < 3:
        return 0.0

    min_area = float("inf")
    for i in range(n_points - 2):
        for j in range(i + 1, n_points - 1):
            edge = points[j] - points[i]
            for k in range(j + 1, n_points):
                area = 0.5 * abs(
                    edge[0] * (points[k, 1] - points[i, 1])
                    - edge[1] * (points[k, 0] - points[i, 0])
                )
                if area < min_area:
                    min_area = float(area)
    return 0.0 if not math.isfinite(min_area) else float(min_area)


def _compute_min_pair_distance(points: np.ndarray) -> float:
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0
    min_distance = float("inf")
    for i in range(n_points - 1):
        for j in range(i + 1, n_points):
            distance = float(np.linalg.norm(points[i] - points[j]))
            if distance < min_distance:
                min_distance = distance
    return 0.0 if not math.isfinite(min_distance) else float(min_distance)


def _task_metrics(
    task: HeilbronnTriangleTaskSpec,
    *,
    points: np.ndarray,
    barycentric_coords: np.ndarray | None,
    eval_time: float,
    valid: bool,
) -> dict[str, float]:
    if not valid or barycentric_coords is None:
        return {
            "min_triangle_area": 0.0,
            "target_min_area": float(task.target_min_area),
            "target_ratio": 0.0,
            "validity": 0.0,
            "point_spread": 0.0,
            "boundary_utilization": 0.0,
            "min_pair_distance": 0.0,
            "eval_time": float(max(0.0, eval_time)),
            "score": 0.0,
            "combined_score": 0.0,
        }

    min_triangle_area = compute_min_triangle_area(points)
    target_ratio = min_triangle_area / float(task.target_min_area)
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    point_spread = float(np.clip(np.std(distances) / CANONICAL_TRIANGLE_BBOX_DIAMETER, 0.0, 1.0))
    min_lambda = np.min(barycentric_coords, axis=1)
    boundary_utilization = float(np.clip(1.0 - 3.0 * np.mean(min_lambda), 0.0, 1.0))
    min_pair_distance = float(
        np.clip(
            _compute_min_pair_distance(points) / CANONICAL_TRIANGLE_BBOX_DIAMETER,
            0.0,
            1.0,
        )
    )
    score = target_ratio
    return {
        "min_triangle_area": float(min_triangle_area),
        "target_min_area": float(task.target_min_area),
        "target_ratio": float(target_ratio),
        "validity": 1.0,
        "point_spread": point_spread,
        "boundary_utilization": boundary_utilization,
        "min_pair_distance": min_pair_distance,
        "eval_time": float(max(0.0, eval_time)),
        "score": float(score),
        "combined_score": float(score),
    }


def _subprocess_runner_script(
    *,
    program_path: str,
    result_path: str,
    n_points: int,
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
        N_POINTS = {int(n_points)}

        def _is_scalar_like(value):
            try:
                array = np.asarray(value, dtype=float)
            except Exception:
                return False
            return array.ndim == 0 or array.size == 1

        payload = None
        try:
            sys.path.insert(0, os.path.dirname(PROGRAM_PATH))
            module_name = "heilbronn_triangle_emo_sta_candidate"
            spec = importlib.util.spec_from_file_location(module_name, PROGRAM_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load program from {{PROGRAM_PATH}}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            runner = getattr(module, "run_heilbronn", None)
            if not callable(runner):
                runner = getattr(module, "construct_points", None)
            if not callable(runner):
                raise AttributeError(
                    "Program must define run_heilbronn(n) or construct_points(n)"
                )

            result = runner(N_POINTS)
            points_payload = result
            reported_min_area = None
            if isinstance(result, (tuple, list)) and len(result) == 2:
                maybe_points, maybe_min_area = result
                maybe_points_array = np.asarray(maybe_points, dtype=float)
                if (
                    maybe_points_array.ndim == 2
                    and maybe_points_array.shape[1] == 2
                    and _is_scalar_like(maybe_min_area)
                ):
                    points_payload = maybe_points_array
                    reported_min_area = float(np.asarray(maybe_min_area, dtype=float).reshape(-1)[0])

            points_array = np.asarray(points_payload, dtype=float)
            payload = {{
                "ok": True,
                "points": points_array.tolist(),
                "reported_min_area": reported_min_area,
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
    task: HeilbronnTriangleTaskSpec,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="heilbronn_triangle_emo_sta_") as temp_dir:
        temp_root = Path(temp_dir)
        runner_path = temp_root / "runner.py"
        result_path = temp_root / "result.json"
        runner_path.write_text(
            _subprocess_runner_script(
                program_path=str(Path(program_path).resolve()),
                result_path=str(result_path),
                n_points=task.n_points,
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


def evaluate_one_task(
    program_path: str,
    task: HeilbronnTriangleTaskSpec,
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
    validation_summary = _empty_validation_summary()
    error = None

    if execution["timed_out"]:
        error = f"Timed out after {timeout_seconds:.0f}s"
        metrics = _task_metrics(
            task,
            points=np.zeros((task.n_points, 2), dtype=float),
            barycentric_coords=None,
            eval_time=float(execution["eval_time"]),
            valid=False,
        )
    else:
        payload = execution.get("payload")
        if not isinstance(payload, Mapping):
            error = "Missing subprocess result payload"
            metrics = _task_metrics(
                task,
                points=np.zeros((task.n_points, 2), dtype=float),
                barycentric_coords=None,
                eval_time=float(execution["eval_time"]),
                valid=False,
            )
        elif not bool(payload.get("ok", False)):
            error = str(payload.get("error") or "Candidate execution failed")
            metrics = _task_metrics(
                task,
                points=np.zeros((task.n_points, 2), dtype=float),
                barycentric_coords=None,
                eval_time=float(execution["eval_time"]),
                valid=False,
            )
        else:
            try:
                points = np.asarray(payload.get("points"), dtype=float)
            except Exception as exc:
                validation_summary["shape_message"] = (
                    f"Could not convert points to a numeric array: {exc}"
                )
                points = np.zeros((task.n_points, 2), dtype=float)
                error = validation_summary["shape_message"]
                metrics = _task_metrics(
                    task,
                    points=points,
                    barycentric_coords=None,
                    eval_time=float(execution["eval_time"]),
                    valid=False,
                )
            else:
                valid, validation_summary, barycentric_coords = validate_point_set(
                    points,
                    n_expected=task.n_points,
                )
                metrics = _task_metrics(
                    task,
                    points=points,
                    barycentric_coords=barycentric_coords,
                    eval_time=float(execution["eval_time"]),
                    valid=valid,
                )
                reported_min_area = payload.get("reported_min_area")
                if reported_min_area is not None and valid:
                    try:
                        reported_value = float(reported_min_area)
                    except (TypeError, ValueError):
                        validation_summary["reported_min_area_mismatch"] = True
                    else:
                        validation_summary["reported_min_area_mismatch"] = _reported_min_area_mismatch(
                            reported_value,
                            float(metrics["min_triangle_area"]),
                        )

                if not valid:
                    error = str(validation_summary.get("shape_message") or "")
                    if not error and validation_summary.get("nonfinite_points", 0) > 0:
                        error = "Point set contains non-finite values"
                    if not error and validation_summary.get("boundary_violations", 0) > 0:
                        error = "Point set violates the canonical triangle containment rule"
                    if not error:
                        error = "Point set validation failed"

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
        "compact_task_summary": {
            "score": float(task_result["final_task_score"]),
            "min_triangle_area": float(task_result["metrics"]["min_triangle_area"]),
            "target_ratio": float(task_result["metrics"]["target_ratio"]),
            "validity": float(task_result["metrics"]["validity"]),
            "timed_out": bool(execution["timed_out"]),
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
            "min_triangle_area": float(best_local_result["metrics"]["min_triangle_area"]),
            "target_ratio": float(best_local_result["metrics"]["target_ratio"]),
        }

    return artifacts


def _evaluate(program_path: str, *, stage1: bool) -> EvaluationResult:
    selector = os.environ.get(
        HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR,
        HEILBRONN_TRIANGLE_SHARED_SELECTOR,
    )
    selected_tasks = resolve_eval_task_specs(selector)
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
    """Evaluate one task or the shared Heilbronn-triangle family."""
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
            f"min_triangle_area={task_result['metrics']['min_triangle_area']:.8f}"
        )
