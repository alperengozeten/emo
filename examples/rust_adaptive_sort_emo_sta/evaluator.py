"""Evaluator for the EMO-STA Rust adaptive-sort family."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult
from openevolve.multi_task_shared_then_adapt.rust_adaptive_sort import (
    DEFAULT_BUILD_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    RUST_ADAPTIVE_SORT_SHARED_SELECTOR,
    RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR,
    RustAdaptiveSortTaskSpec,
    aggregate_task_results,
    build_task_result,
    resolve_task_specs,
)


THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_RUNNER_DIR = THIS_DIR / "benchmark_runner"
BINARY_NAME = "rust_adaptive_sort_emo_sta_benchmark"


def _trim_text(value: str, *, limit: int = 32768) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n... truncated {omitted} characters ..."


def _binary_path(project_dir: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return project_dir / "target" / "release" / f"{BINARY_NAME}{suffix}"


def _prepare_project(program_path: Path, project_dir: Path) -> None:
    shutil.copytree(BENCHMARK_RUNNER_DIR, project_dir, dirs_exist_ok=True)
    lib_path = project_dir / "src" / "lib.rs"
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(program_path.read_text(encoding="utf-8"), encoding="utf-8")


def _build_failure_task_results(
    tasks: Iterable[RustAdaptiveSortTaskSpec],
    *,
    error: str,
) -> list[dict[str, Any]]:
    return [build_task_result(task, raw_metrics=None, error=error) for task in tasks]


def _task_execution_summary(task_result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = task_result["metrics"]
    return {
        "score": float(metrics["score"]),
        "correctness_rate": float(metrics["correctness_rate"]),
        "dataset_count": float(metrics["dataset_count"]),
        "successful_dataset_count": float(metrics["successful_dataset_count"]),
    }


def _public_artifacts(
    *,
    selector: str,
    evaluation_mode: str,
    compile_succeeded: bool,
    task_results: list[dict[str, Any]],
    binary_path: Path | None = None,
    build_stdout: str | None = None,
    build_stderr: str | None = None,
    task_execution_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    has_task_failures = any(task_result.get("error") for task_result in task_results)
    compact_task_summary = {
        task_result["task_id"]: _task_execution_summary(task_result)
        for task_result in task_results
    }
    artifacts: dict[str, Any] = {
        "task_selector": selector,
        "selected_task_ids": [task_result["task_id"] for task_result in task_results],
        "evaluation_mode": evaluation_mode,
        "compile_succeeded": compile_succeeded,
        "task_results": task_results,
        "compact_task_summary": compact_task_summary,
        "status": (
            "ok"
            if compile_succeeded and not error and not has_task_failures
            else "completed_with_task_failures"
            if compile_succeeded and not error and has_task_failures
            else "error"
        ),
    }
    if binary_path is not None:
        artifacts["binary_name"] = binary_path.name
        artifacts["binary_path_name"] = str(binary_path.name)
    if build_stdout:
        artifacts["build_stdout"] = _trim_text(build_stdout)
    if build_stderr:
        artifacts["build_stderr"] = _trim_text(build_stderr)
    if task_execution_artifacts:
        artifacts["task_execution_artifacts"] = {
            task_id: dict(details) for task_id, details in task_execution_artifacts.items()
        }
    if error:
        artifacts["error"] = error
    return artifacts


def _build_failure_result(
    *,
    tasks: list[RustAdaptiveSortTaskSpec],
    selector: str,
    evaluation_mode: str,
    error: str,
    build_stdout: str | None = None,
    build_stderr: str | None = None,
) -> EvaluationResult:
    task_results = _build_failure_task_results(tasks, error=error)
    artifacts = _public_artifacts(
        selector=selector,
        evaluation_mode=evaluation_mode,
        compile_succeeded=False,
        task_results=task_results,
        build_stdout=build_stdout,
        build_stderr=build_stderr,
        error=error,
    )
    if len(task_results) == 1:
        return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
    return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)


def _run_task(
    *,
    binary_path: Path,
    task: RustAdaptiveSortTaskSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [str(binary_path), "--task", task.task_id]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, int(task.run_timeout_seconds or DEFAULT_RUN_TIMEOUT_SECONDS)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        error = f"Task run timed out after {task.run_timeout_seconds} seconds"
        return (
            build_task_result(task, raw_metrics=None, error=error),
            {
                "status": "timeout",
                "command": command,
                "stdout": _trim_text(exc.stdout or ""),
                "stderr": _trim_text(exc.stderr or ""),
            },
        )

    if completed.returncode != 0:
        error = f"Task run failed with exit code {completed.returncode}"
        return (
            build_task_result(task, raw_metrics=None, error=error),
            {
                "status": "runtime_failure",
                "command": command,
                "returncode": completed.returncode,
                "stdout": _trim_text(completed.stdout),
                "stderr": _trim_text(completed.stderr),
            },
        )

    try:
        raw_payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        error = f"Task output was not valid JSON: {exc}"
        return (
            build_task_result(task, raw_metrics=None, error=error),
            {
                "status": "malformed_json",
                "command": command,
                "stdout": _trim_text(completed.stdout),
                "stderr": _trim_text(completed.stderr),
            },
        )

    if not isinstance(raw_payload, Mapping):
        error = f"Task output must be a JSON object, got {type(raw_payload).__name__}"
        return (
            build_task_result(task, raw_metrics=None, error=error),
            {
                "status": "malformed_json_object",
                "command": command,
                "stdout": _trim_text(completed.stdout),
                "stderr": _trim_text(completed.stderr),
            },
        )

    if raw_payload.get("task_id") != task.task_id:
        error = (
            f"Task output task_id mismatch: expected {task.task_id}, "
            f"got {raw_payload.get('task_id')!r}"
        )
        return (
            build_task_result(task, raw_metrics=None, error=error),
            {
                "status": "task_id_mismatch",
                "command": command,
                "stdout": _trim_text(completed.stdout),
                "stderr": _trim_text(completed.stderr),
            },
        )

    task_result = build_task_result(
        task,
        raw_metrics=raw_payload,
        dataset_summaries=raw_payload.get("datasets"),
    )
    return (
        task_result,
        {
            "status": "ok",
            "command": command,
            "returncode": completed.returncode,
        },
    )


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate one task or the full shared Rust adaptive-sort EMO-STA family."""
    selector = os.environ.get(
        RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR,
        RUST_ADAPTIVE_SORT_SHARED_SELECTOR,
    )
    selected_tasks = resolve_task_specs(selector)
    evaluation_mode = "shared" if len(selected_tasks) > 1 else "task_specific"

    program_file = Path(program_path).resolve()
    build_timeout_seconds = max(
        int(task.build_timeout_seconds or DEFAULT_BUILD_TIMEOUT_SECONDS)
        for task in selected_tasks
    )

    try:
        with tempfile.TemporaryDirectory(prefix="rust_adaptive_sort_emo_sta_") as temp_dir:
            project_dir = Path(temp_dir) / BINARY_NAME
            _prepare_project(program_file, project_dir)

            build_env = dict(os.environ)
            build_env["CARGO_TERM_COLOR"] = "never"
            try:
                build_completed = subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=build_timeout_seconds,
                    check=False,
                    env=build_env,
                )
            except subprocess.TimeoutExpired as exc:
                return _build_failure_result(
                    tasks=selected_tasks,
                    selector=selector,
                    evaluation_mode=evaluation_mode,
                    error=f"Cargo build timed out after {build_timeout_seconds} seconds",
                    build_stdout=exc.stdout or "",
                    build_stderr=exc.stderr or "",
                )

            if build_completed.returncode != 0:
                return _build_failure_result(
                    tasks=selected_tasks,
                    selector=selector,
                    evaluation_mode=evaluation_mode,
                    error=f"Cargo build failed with exit code {build_completed.returncode}",
                    build_stdout=build_completed.stdout,
                    build_stderr=build_completed.stderr,
                )

            binary_path = _binary_path(project_dir)
            if not binary_path.is_file():
                return _build_failure_result(
                    tasks=selected_tasks,
                    selector=selector,
                    evaluation_mode=evaluation_mode,
                    error=f"Built binary not found at {binary_path}",
                    build_stdout=build_completed.stdout,
                    build_stderr=build_completed.stderr,
                )
            task_results: list[dict[str, Any]] = []
            task_execution_artifacts: dict[str, dict[str, Any]] = {}
            for task in selected_tasks:
                task_result, task_execution = _run_task(binary_path=binary_path, task=task)
                task_results.append(task_result)
                task_execution_artifacts[task.task_id] = task_execution

            artifacts = _public_artifacts(
                selector=selector,
                evaluation_mode=evaluation_mode,
                compile_succeeded=True,
                task_results=task_results,
                binary_path=binary_path,
                task_execution_artifacts=task_execution_artifacts,
            )
            if len(task_results) == 1:
                return EvaluationResult(metrics=dict(task_results[0]["metrics"]), artifacts=artifacts)
            return EvaluationResult(metrics=aggregate_task_results(task_results), artifacts=artifacts)
    except FileNotFoundError as exc:
        return _build_failure_result(
            tasks=selected_tasks,
            selector=selector,
            evaluation_mode=evaluation_mode,
            error=str(exc),
        )
    except Exception as exc:
        return _build_failure_result(
            tasks=selected_tasks,
            selector=selector,
            evaluation_mode=evaluation_mode,
            error=f"{type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise SystemExit("Usage: python evaluator.py <program_path>")

    evaluation_result = evaluate(sys.argv[1])
    print(
        json.dumps(
            {
                "metrics": evaluation_result.metrics,
                "artifacts": evaluation_result.artifacts,
            },
            indent=2,
            default=str,
        )
    )
