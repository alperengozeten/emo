"""Evaluator for the EMO-STA hexagon-packing family."""

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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.hexagon_packing import (
    HEXAGON_PACKING_SHARED_SELECTOR,
    HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR,
    HexagonPackingTaskSpec,
    aggregate_task_results,
    build_task_result,
    minimum_outer_side_length_for_area,
    resolve_task_specs,
)


BOUNDARY_TOLERANCE = 1.0e-6
MAX_VALIDATION_MESSAGES = 5
MAX_TEXT_EXCERPT = 400
ANGLE_PERIOD_DEGREES = 60.0


def _truncate_text(value: str | None, *, limit: int = MAX_TEXT_EXCERPT) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _append_validation_message(validation_summary: dict[str, Any], message: str) -> None:
    example_messages = validation_summary.setdefault("example_messages", [])
    if len(example_messages) < MAX_VALIDATION_MESSAGES:
        example_messages.append(str(message))


def _empty_validation_summary(expected_n: int) -> dict[str, Any]:
    return {
        "shape_valid": False,
        "finite_valid": False,
        "outer_side_valid": False,
        "outer_side_lower_bound": None,
        "containment_violations": 0,
        "overlap_violations": 0,
        "nonfinite_value_violations": 0,
        "reported_n": None,
        "expected_n": int(expected_n),
        "example_messages": [],
    }


def hexagon_vertices(
    center_x: float,
    center_y: float,
    side_length: float,
    angle_degrees: float,
) -> np.ndarray:
    angle0 = math.radians(float(angle_degrees))
    return np.asarray(
        [
            (
                float(center_x) + float(side_length) * math.cos(angle0 + 2.0 * math.pi * k / 6.0),
                float(center_y) + float(side_length) * math.sin(angle0 + 2.0 * math.pi * k / 6.0),
            )
            for k in range(6)
        ],
        dtype=float,
    )


def _polygon_edge_normals(vertices: np.ndarray) -> list[np.ndarray]:
    normals: list[np.ndarray] = []
    for index in range(len(vertices)):
        p1 = vertices[index]
        p2 = vertices[(index + 1) % len(vertices)]
        edge = p2 - p1
        normal = np.asarray((-edge[1], edge[0]), dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            continue
        normals.append(normal / norm)
    return normals


def _project_polygon(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    projections = np.asarray(vertices, dtype=float) @ np.asarray(axis, dtype=float)
    return float(np.min(projections)), float(np.max(projections))


def polygons_overlap_positive_area(
    vertices1: np.ndarray,
    vertices2: np.ndarray,
    *,
    tol: float = BOUNDARY_TOLERANCE,
) -> bool:
    axes = _polygon_edge_normals(vertices1) + _polygon_edge_normals(vertices2)
    for axis in axes:
        min1, max1 = _project_polygon(vertices1, axis)
        min2, max2 = _project_polygon(vertices2, axis)
        if max1 <= min2 + float(tol) or max2 <= min1 + float(tol):
            return False
    return True


def point_inside_or_on_convex_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
    *,
    tol: float = BOUNDARY_TOLERANCE,
) -> bool:
    px, py = float(point[0]), float(point[1])
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        edge_length = math.hypot(float(x2 - x1), float(y2 - y1))
        if edge_length <= 0.0:
            return False
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        signed_distance = cross / edge_length
        if signed_distance < -float(tol):
            return False
    return True


def _compute_min_center_distance(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return 0.0
    min_distance = float("inf")
    for i in range(centers.shape[0] - 1):
        for j in range(i + 1, centers.shape[0]):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            if distance < min_distance:
                min_distance = distance
    return 0.0 if not math.isfinite(min_distance) else float(min_distance)


def _periodic_angle_spread(
    angles_degrees: np.ndarray,
    *,
    period_degrees: float = ANGLE_PERIOD_DEGREES,
) -> float:
    angles = np.asarray(angles_degrees, dtype=float)
    if angles.size == 0:
        return 0.0
    angles_mod_period = np.mod(angles, float(period_degrees))
    theta = 2.0 * math.pi * (angles_mod_period / float(period_degrees))
    resultant_x = float(np.mean(np.cos(theta)))
    resultant_y = float(np.mean(np.sin(theta)))
    resultant_length = math.sqrt(resultant_x**2 + resultant_y**2)
    return float(np.clip(1.0 - resultant_length, 0.0, 1.0))


def validate_hexagon_packing(
    inner_hex_data: np.ndarray,
    outer_hex_data: np.ndarray,
    outer_hex_side_length: float,
    n: int,
    *,
    tol: float = BOUNDARY_TOLERANCE,
) -> tuple[bool, dict[str, Any]]:
    """Validate hexagon-packing outputs."""
    validation_summary = _empty_validation_summary(n)
    validation_summary["reported_n"] = (
        int(inner_hex_data.shape[0])
        if isinstance(inner_hex_data, np.ndarray) and inner_hex_data.ndim >= 1
        else None
    )

    if inner_hex_data.shape != (n, 3):
        validation_summary["shape_message"] = (
            f"inner_hex_data must have shape ({n}, 3), got {tuple(inner_hex_data.shape)}"
        )
        _append_validation_message(validation_summary, validation_summary["shape_message"])
        return False, validation_summary
    if outer_hex_data.shape != (3,):
        validation_summary["shape_message"] = (
            f"outer_hex_data must have shape (3,), got {tuple(outer_hex_data.shape)}"
        )
        _append_validation_message(validation_summary, validation_summary["shape_message"])
        return False, validation_summary

    validation_summary["shape_valid"] = True

    nonfinite_values = int(np.size(inner_hex_data) - int(np.isfinite(inner_hex_data).sum()))
    nonfinite_values += int(np.size(outer_hex_data) - int(np.isfinite(outer_hex_data).sum()))
    nonfinite_values += 0 if math.isfinite(float(outer_hex_side_length)) else 1
    validation_summary["nonfinite_value_violations"] = nonfinite_values
    if nonfinite_values > 0:
        _append_validation_message(
            validation_summary,
            f"Construction contains {nonfinite_values} non-finite values",
        )
        return False, validation_summary

    validation_summary["finite_valid"] = True

    outer_side_length = float(outer_hex_side_length)
    if outer_side_length <= 0.0:
        validation_summary["outer_side_message"] = (
            "outer_hex_side_length must be positive and finite"
        )
        _append_validation_message(
            validation_summary,
            validation_summary["outer_side_message"],
        )
        return False, validation_summary

    outer_side_lower_bound = minimum_outer_side_length_for_area(n)
    validation_summary["outer_side_lower_bound"] = float(outer_side_lower_bound)
    if outer_side_length + float(tol) < outer_side_lower_bound:
        validation_summary["outer_side_message"] = (
            "outer_hex_side_length violates the area lower bound "
            f"sqrt(n)={outer_side_lower_bound:.12g}"
        )
        _append_validation_message(
            validation_summary,
            validation_summary["outer_side_message"],
        )
        return False, validation_summary

    validation_summary["outer_side_valid"] = True

    outer_polygon = hexagon_vertices(
        float(outer_hex_data[0]),
        float(outer_hex_data[1]),
        outer_side_length,
        float(outer_hex_data[2]),
    )
    inner_polygons: list[np.ndarray] = []
    for index, (center_x, center_y, angle_degrees) in enumerate(inner_hex_data):
        polygon = hexagon_vertices(
            float(center_x),
            float(center_y),
            1.0,
            float(angle_degrees),
        )
        inner_polygons.append(polygon)
        for vertex in polygon:
            if not point_inside_or_on_convex_polygon(vertex, outer_polygon, tol=tol):
                validation_summary["containment_violations"] += 1
                _append_validation_message(
                    validation_summary,
                    f"Inner hexagon {index} has a vertex outside the outer hexagon",
                )

    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_positive_area(inner_polygons[i], inner_polygons[j], tol=tol):
                validation_summary["overlap_violations"] += 1
                _append_validation_message(
                    validation_summary,
                    f"Inner hexagons {i} and {j} overlap in positive area",
                )

    is_valid = (
        validation_summary["shape_valid"]
        and validation_summary["finite_valid"]
        and validation_summary["outer_side_valid"]
        and validation_summary["containment_violations"] == 0
        and validation_summary["overlap_violations"] == 0
    )
    return is_valid, validation_summary


def _task_metrics(
    task: HexagonPackingTaskSpec,
    *,
    inner_hex_data: np.ndarray,
    outer_hex_side_length: float,
    eval_time: float,
    valid: bool,
) -> dict[str, float]:
    if not valid:
        return {
            "outer_side_length": 0.0,
            "target_outer_side_length": float(task.target_outer_side_length),
            "target_ratio": 0.0,
            "inv_outer_side_length": 0.0,
            "validity": 0.0,
            "center_spread": 0.0,
            "angle_spread": 0.0,
            "min_center_distance": 0.0,
            "eval_time": float(max(0.0, eval_time)),
            "score": 0.0,
            "combined_score": 0.0,
        }

    centers = np.asarray(inner_hex_data[:, :2], dtype=float)
    centroid = np.mean(centers, axis=0)
    distances = np.linalg.norm(centers - centroid, axis=1)
    target = max(float(task.target_outer_side_length), 1.0e-9)
    center_spread = float(np.clip(np.std(distances) / target, 0.0, 1.0))
    angle_spread = _periodic_angle_spread(
        np.asarray(inner_hex_data[:, 2], dtype=float),
        period_degrees=ANGLE_PERIOD_DEGREES,
    )
    min_center_distance = 0.0
    if task.n_hexagons >= 2:
        min_center_distance = float(
            np.clip(_compute_min_center_distance(centers) / target, 0.0, 1.0)
        )
    outer_side_length = float(outer_hex_side_length)
    target_ratio = float(task.target_outer_side_length) / outer_side_length
    inv_outer_side_length = 1.0 / outer_side_length
    score = target_ratio
    return {
        "outer_side_length": outer_side_length,
        "target_outer_side_length": float(task.target_outer_side_length),
        "target_ratio": float(target_ratio),
        "inv_outer_side_length": float(inv_outer_side_length),
        "validity": 1.0,
        "center_spread": center_spread,
        "angle_spread": angle_spread,
        "min_center_distance": min_center_distance,
        "eval_time": float(max(0.0, eval_time)),
        "score": float(score),
        "combined_score": float(score),
    }


def _subprocess_runner_script(
    *,
    program_path: str,
    result_path: str,
    n_hexagons: int,
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
        N_HEXAGONS = {int(n_hexagons)}

        def _coerce_scalar(value):
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                return float(array)
            if array.size == 1:
                return float(array.reshape(-1)[0])
            raise TypeError(
                f"outer_hex_side_length must be scalar-like, got shape {{array.shape}}"
            )

        payload = None
        try:
            sys.path.insert(0, os.path.dirname(PROGRAM_PATH))
            module_name = "hexagon_packing_emo_sta_candidate"
            spec = importlib.util.spec_from_file_location(module_name, PROGRAM_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load program from {{PROGRAM_PATH}}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            runner = getattr(module, "run_hexagon_packing", None)
            if not callable(runner):
                runner = getattr(module, "construct_hexagon_packing", None)
            if not callable(runner):
                raise AttributeError(
                    "Program must define run_hexagon_packing(n) or construct_hexagon_packing(n)"
                )

            result = runner(N_HEXAGONS)
            if isinstance(result, dict):
                inner_hex_data = result["inner_hex_data"]
                outer_hex_data = result["outer_hex_data"]
                outer_hex_side_length = result["outer_hex_side_length"]
            elif isinstance(result, (tuple, list)) and len(result) == 3:
                inner_hex_data, outer_hex_data, outer_hex_side_length = result
            else:
                raise TypeError(
                    "Program must return either a dict with inner_hex_data/outer_hex_data/"
                    "outer_hex_side_length or a 3-tuple"
                )

            payload = {{
                "ok": True,
                "inner_hex_data": np.asarray(inner_hex_data, dtype=float).tolist(),
                "outer_hex_data": np.asarray(outer_hex_data, dtype=float).tolist(),
                "outer_hex_side_length": _coerce_scalar(outer_hex_side_length),
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
    task: HexagonPackingTaskSpec,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hexagon_packing_emo_sta_") as temp_dir:
        temp_root = Path(temp_dir)
        runner_path = temp_root / "runner.py"
        result_path = temp_root / "result.json"
        runner_path.write_text(
            _subprocess_runner_script(
                program_path=str(Path(program_path).resolve()),
                result_path=str(result_path),
                n_hexagons=task.n_hexagons,
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
    task: HexagonPackingTaskSpec,
    *,
    stage1: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timeout_seconds = (
        float(task.timeout_seconds_stage1) if stage1 else float(task.timeout_seconds_full)
    )
    execution = _run_program_in_subprocess(
        program_path,
        task,
        timeout_seconds=timeout_seconds,
    )
    validation_summary = _empty_validation_summary(task.n_hexagons)
    error = None

    if execution["timed_out"]:
        error = f"Timed out after {timeout_seconds:.0f}s"
        metrics = _task_metrics(
            task,
            inner_hex_data=np.zeros((task.n_hexagons, 3), dtype=float),
            outer_hex_side_length=float("inf"),
            eval_time=float(execution["eval_time"]),
            valid=False,
        )
    else:
        payload = execution.get("payload")
        if not isinstance(payload, Mapping):
            error = "Missing subprocess result payload"
            metrics = _task_metrics(
                task,
                inner_hex_data=np.zeros((task.n_hexagons, 3), dtype=float),
                outer_hex_side_length=float("inf"),
                eval_time=float(execution["eval_time"]),
                valid=False,
            )
        elif not bool(payload.get("ok", False)):
            error = str(payload.get("error") or "Candidate execution failed")
            metrics = _task_metrics(
                task,
                inner_hex_data=np.zeros((task.n_hexagons, 3), dtype=float),
                outer_hex_side_length=float("inf"),
                eval_time=float(execution["eval_time"]),
                valid=False,
            )
        else:
            try:
                inner_hex_data = np.asarray(payload.get("inner_hex_data"), dtype=float)
                outer_hex_data = np.asarray(payload.get("outer_hex_data"), dtype=float)
                outer_hex_side_length = float(payload.get("outer_hex_side_length"))
            except Exception as exc:
                validation_summary["shape_message"] = (
                    f"Could not convert candidate outputs to numeric arrays/scalars: {exc}"
                )
                _append_validation_message(
                    validation_summary,
                    validation_summary["shape_message"],
                )
                inner_hex_data = np.zeros((task.n_hexagons, 3), dtype=float)
                outer_hex_data = np.zeros(3, dtype=float)
                outer_hex_side_length = float("nan")
                error = validation_summary["shape_message"]
                metrics = _task_metrics(
                    task,
                    inner_hex_data=inner_hex_data,
                    outer_hex_side_length=outer_hex_side_length,
                    eval_time=float(execution["eval_time"]),
                    valid=False,
                )
            else:
                valid, validation_summary = validate_hexagon_packing(
                    inner_hex_data,
                    outer_hex_data,
                    outer_hex_side_length,
                    task.n_hexagons,
                )
                metrics = _task_metrics(
                    task,
                    inner_hex_data=inner_hex_data,
                    outer_hex_side_length=outer_hex_side_length,
                    eval_time=float(execution["eval_time"]),
                    valid=valid,
                )
                if not valid:
                    error = str(validation_summary.get("shape_message") or "")
                    if not error and validation_summary.get("nonfinite_value_violations", 0) > 0:
                        error = "Construction contains non-finite values"
                    if not error and not validation_summary.get("outer_side_valid", False):
                        error = str(
                            validation_summary.get("outer_side_message")
                            or "outer_hex_side_length must be positive and finite"
                        )
                    if not error and validation_summary.get("containment_violations", 0) > 0:
                        error = "At least one inner hexagon is not contained in the outer hexagon"
                    if not error and validation_summary.get("overlap_violations", 0) > 0:
                        error = "At least one pair of inner hexagons overlaps"
                    if not error:
                        example_messages = validation_summary.get("example_messages") or []
                        if example_messages:
                            error = str(example_messages[0])
                    if not error:
                        error = "Hexagon packing validation failed"

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
            "outer_side_length": float(task_result["metrics"]["outer_side_length"]),
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
            "outer_side_length": float(best_local_result["metrics"]["outer_side_length"]),
            "target_ratio": float(best_local_result["metrics"]["target_ratio"]),
        }

    return artifacts


def _evaluate(program_path: str, *, stage1: bool) -> EvaluationResult:
    selector = os.environ.get(
        HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR,
        HEXAGON_PACKING_SHARED_SELECTOR,
    )
    selected_tasks = resolve_task_specs(selector)
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
    """Evaluate one task or the shared hexagon-packing family."""
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
            f"outer_side_length={task_result['metrics']['outer_side_length']:.6f}"
        )
