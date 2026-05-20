"""Post-hoc evaluation-only holdout helpers for EMO-STA families."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .circle_packing import (
    CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR,
    CirclePackingTaskSpec,
    aggregate_task_results,
    build_task_result,
    resolve_holdout_task_specs,
)
from .runner import write_json
from .workflow import SharedThenAdaptManifest, family_task_specs


def _load_evaluation_module(evaluation_file: Path):
    spec = importlib.util.spec_from_file_location(
        f"emo_sta_holdout_eval_{hash(str(evaluation_file.resolve()))}",
        str(evaluation_file),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluation module from {evaluation_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TemporaryEnv:
    def __init__(self, env: Mapping[str, str]):
        self._env = dict(env)
        self._previous: Dict[str, Optional[str]] = {}

    def __enter__(self):
        self._previous = {key: os.environ.get(key) for key in self._env}
        for key, value in self._env.items():
            os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _extract_evaluation_artifacts(evaluation_result: Any) -> Mapping[str, Any]:
    if isinstance(evaluation_result, Mapping):
        artifacts = evaluation_result.get("artifacts")
        return artifacts if isinstance(artifacts, Mapping) else {}
    artifacts = getattr(evaluation_result, "artifacts", None)
    return artifacts if isinstance(artifacts, Mapping) else {}


def _extract_evaluation_metrics(evaluation_result: Any) -> Mapping[str, Any]:
    if isinstance(evaluation_result, Mapping):
        metrics = evaluation_result.get("metrics")
        if isinstance(metrics, Mapping):
            return metrics
        return {
            key: value for key, value in evaluation_result.items() if key != "artifacts"
        }
    metrics = getattr(evaluation_result, "metrics", None)
    return metrics if isinstance(metrics, Mapping) else {}


def _read_json_if_present(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _extract_task_result(
    evaluation_result: Any,
    task: CirclePackingTaskSpec,
) -> Dict[str, Any]:
    artifacts = _extract_evaluation_artifacts(evaluation_result)
    task_results = artifacts.get("task_results")
    if isinstance(task_results, list):
        for task_result in task_results:
            if isinstance(task_result, Mapping) and task_result.get("task_id") == task.task_id:
                return dict(task_result)

    metrics = _extract_evaluation_metrics(evaluation_result)
    return build_task_result(
        task,
        raw_metrics=metrics if isinstance(metrics, Mapping) else None,
        error="Holdout evaluation did not return a per-task artifact entry",
    )


def _validity_count(task_results: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for task_result in task_results:
        metrics = task_result.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        try:
            if float(metrics.get("validity", 0.0)) >= 0.5:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def unavailable_holdout_evaluation_result(
    *,
    family: str,
    holdout_task_ids: Sequence[str],
    error: str,
    program_path: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_program_path = None
    if program_path is not None:
        resolved_program_path = str(Path(program_path).resolve())
    return {
        "available": False,
        "family": family,
        "program_path": resolved_program_path,
        "holdout_task_ids": list(holdout_task_ids),
        "average_holdout_score": None,
        "average_holdout_target_ratio": None,
        "average_holdout_eval_time": None,
        "evaluated_task_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "holdout_task_results": {},
        "error": str(error),
    }


def resolve_best_program_path(
    base_dir: str | Path,
    *,
    initial_program: str | Path | None = None,
    checkpoint_layout: bool,
) -> Optional[Path]:
    base_path = Path(base_dir).resolve()
    search_root = base_path if checkpoint_layout else base_path / "best"
    if not search_root.is_dir():
        return None

    candidate_suffixes: list[str] = []
    if initial_program is not None:
        initial_suffix = Path(initial_program).suffix
        if initial_suffix:
            candidate_suffixes.append(initial_suffix)
    candidate_suffixes.extend([".py", ".r", ".rs"])

    seen_suffixes: set[str] = set()
    for suffix in candidate_suffixes:
        if not suffix or suffix in seen_suffixes:
            continue
        seen_suffixes.add(suffix)
        candidate = search_root / f"best_program{suffix}"
        if candidate.is_file():
            return candidate.resolve()

    extensionless_candidate = search_root / "best_program"
    if extensionless_candidate.is_file():
        return extensionless_candidate.resolve()

    wildcard_matches = sorted(search_root.glob("best_program.*"))
    if wildcard_matches:
        return wildcard_matches[0].resolve()
    return None


def evaluate_best_program_on_holdouts(
    *,
    program_path: Path,
    family: str,
    holdout_task_specs: Sequence[CirclePackingTaskSpec],
    evaluation_file: Path,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    if family != "circle_packing":
        raise ValueError(
            "evaluate_best_program_on_holdouts currently only supports family='circle_packing'"
        )

    holdout_tasks = list(holdout_task_specs)
    if not holdout_tasks:
        raise ValueError("holdout_task_specs must contain at least one holdout task")

    resolved_program_path = Path(program_path).resolve()
    if not resolved_program_path.is_file():
        raise FileNotFoundError(f"Missing program file for holdout evaluation: {resolved_program_path}")

    evaluation_path = Path(evaluation_file).resolve()
    evaluator_module = _load_evaluation_module(evaluation_path)
    evaluate_fn = getattr(evaluator_module, "evaluate", None)
    if not callable(evaluate_fn):
        raise AttributeError(f"{evaluation_path} does not define evaluate()")

    env_updates = dict(env or {})
    task_results: list[Dict[str, Any]] = []
    results_by_task: Dict[str, Dict[str, Any]] = {}

    for task in holdout_tasks:
        with _TemporaryEnv(
            {
                **env_updates,
                CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR: task.task_id,
            }
        ):
            raw_result = evaluate_fn(str(resolved_program_path))
            if inspect.isawaitable(raw_result):
                evaluation_result = asyncio.run(raw_result)
            else:
                evaluation_result = raw_result

        task_result = _extract_task_result(evaluation_result, task)
        task_results.append(task_result)
        results_by_task[task.task_id] = task_result

    aggregate = aggregate_task_results(task_results)
    valid_count = _validity_count(task_results)
    holdout_task_ids = [task.task_id for task in holdout_tasks]
    return {
        "available": True,
        "family": family,
        "program_path": str(resolved_program_path),
        "holdout_task_ids": holdout_task_ids,
        "average_holdout_score": float(aggregate["score"]),
        "average_holdout_target_ratio": float(aggregate["target_ratio"]),
        "average_holdout_eval_time": float(aggregate["eval_time"]),
        "evaluated_task_count": len(task_results),
        "valid_count": valid_count,
        "invalid_count": len(task_results) - valid_count,
        "holdout_task_results": results_by_task,
    }


def evaluate_source_programs_on_holdouts(
    *,
    program_paths: Mapping[str, str | Path | None],
    family: str,
    holdout_task_specs: Sequence[CirclePackingTaskSpec],
    evaluation_file: Path,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    holdout_task_ids = [task.task_id for task in holdout_task_specs]
    results: Dict[str, Dict[str, Any]] = {}
    for source_task_id, program_path in program_paths.items():
        if program_path is None:
            results[source_task_id] = unavailable_holdout_evaluation_result(
                family=family,
                holdout_task_ids=holdout_task_ids,
                error=f"Missing program path for source task '{source_task_id}'",
            )
            continue

        resolved_program_path = Path(program_path).resolve()
        if not resolved_program_path.is_file():
            results[source_task_id] = unavailable_holdout_evaluation_result(
                family=family,
                holdout_task_ids=holdout_task_ids,
                error=f"Missing program file: {resolved_program_path}",
                program_path=resolved_program_path,
            )
            continue

        results[source_task_id] = evaluate_best_program_on_holdouts(
            program_path=resolved_program_path,
            family=family,
            holdout_task_specs=holdout_task_specs,
            evaluation_file=evaluation_file,
            env=env,
        )
    return results


def run_circle_packing_holdout_evaluation(
    *,
    family: str,
    run_root: Path,
    holdout_selector: Optional[str],
    skip_holdouts: bool,
    shared_program_path: str | Path | None,
    adaptation_program_paths: Mapping[str, str | Path | None],
    baseline_program_paths: Mapping[str, str | Path | None],
    evaluation_file: Path,
    env: Mapping[str, str] | None = None,
) -> Optional[Dict[str, Any]]:
    if family != "circle_packing":
        return None

    if skip_holdouts:
        return {
            "enabled": False,
            "holdout_task_ids": [],
            "shared_zero_shot": None,
            "adaptation_by_source_task": {},
            "baseline_by_source_task": {},
            "reason": "skipped_by_flag",
        }

    holdout_task_specs = resolve_holdout_task_specs(holdout_selector)
    holdout_task_ids = [task.task_id for task in holdout_task_specs]

    if shared_program_path is None:
        shared_result = unavailable_holdout_evaluation_result(
            family=family,
            holdout_task_ids=holdout_task_ids,
            error="Missing shared best-program path",
        )
    else:
        resolved_shared_program_path = Path(shared_program_path).resolve()
        if resolved_shared_program_path.is_file():
            shared_result = evaluate_best_program_on_holdouts(
                program_path=resolved_shared_program_path,
                family=family,
                holdout_task_specs=holdout_task_specs,
                evaluation_file=evaluation_file,
                env=env,
            )
        else:
            shared_result = unavailable_holdout_evaluation_result(
                family=family,
                holdout_task_ids=holdout_task_ids,
                error=f"Missing shared best program file: {resolved_shared_program_path}",
                program_path=resolved_shared_program_path,
            )

    adaptation_results = evaluate_source_programs_on_holdouts(
        program_paths=adaptation_program_paths,
        family=family,
        holdout_task_specs=holdout_task_specs,
        evaluation_file=evaluation_file,
        env=env,
    )
    baseline_results = evaluate_source_programs_on_holdouts(
        program_paths=baseline_program_paths,
        family=family,
        holdout_task_specs=holdout_task_specs,
        evaluation_file=evaluation_file,
        env=env,
    )

    holdout_output_root = Path(run_root).resolve() / "holdout_evaluation"
    summary = {
        "enabled": True,
        "holdout_task_ids": holdout_task_ids,
        "shared_zero_shot": shared_result,
        "adaptation_by_source_task": adaptation_results,
        "baseline_by_source_task": baseline_results,
    }

    write_json(
        holdout_output_root / "shared_holdouts.json",
        {
            "enabled": True,
            "holdout_task_ids": holdout_task_ids,
            "result": shared_result,
        },
    )
    write_json(
        holdout_output_root / "adaptation_holdouts.json",
        {
            "enabled": True,
            "holdout_task_ids": holdout_task_ids,
            "results_by_source_task": adaptation_results,
        },
    )
    write_json(
        holdout_output_root / "baseline_holdouts.json",
        {
            "enabled": True,
            "holdout_task_ids": holdout_task_ids,
            "results_by_source_task": baseline_results,
        },
    )
    write_json(holdout_output_root / "holdout_summary.json", summary)
    return summary


def _adaptation_branch_task_dir(
    results_path: Path,
    branch: str,
    source_task_id: str,
) -> Path:
    canonical = results_path / f"adaptation_{branch}" / source_task_id
    if canonical.is_dir():
        return canonical
    legacy = results_path / f"adaptation_{branch}_{'seed'}" / source_task_id
    if legacy.is_dir():
        return legacy
    return canonical


def run_circle_packing_adaptation_holdout_update(
    *,
    manifest: SharedThenAdaptManifest,
    results_dir: str | Path,
    holdout_selector: Optional[str] = "all",
    include_best_shared: bool = True,
    include_best_local: bool = True,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Merge optional adaptation holdout results into the run-local holdout JSON."""
    if manifest.family != "circle_packing":
        raise ValueError(
            "run_circle_packing_adaptation_holdout_update only supports family='circle_packing'"
        )

    results_path = Path(results_dir).resolve()
    if not results_path.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_path}")

    holdout_output_root = results_path / "holdout_evaluation"
    summary_path = holdout_output_root / "holdout_summary.json"
    existing_summary = _read_json_if_present(summary_path)
    if not existing_summary:
        comparison_summary = _read_json_if_present(results_path / "comparison_summary.json")
        holdout_payload = comparison_summary.get("holdout_evaluation")
        if isinstance(holdout_payload, Mapping):
            existing_summary = dict(holdout_payload)

    holdout_task_specs = resolve_holdout_task_specs(holdout_selector)
    holdout_task_ids = [task.task_id for task in holdout_task_specs]
    source_task_ids = [task.task_id for task in family_task_specs(manifest)]

    summary: Dict[str, Any] = dict(existing_summary)
    summary["enabled"] = True
    summary["holdout_task_ids"] = holdout_task_ids
    summary["include_best_shared"] = bool(
        existing_summary.get("include_best_shared", False)
    ) or bool(include_best_shared)
    summary["include_best_local"] = bool(
        existing_summary.get("include_best_local", False)
    ) or bool(include_best_local)

    if include_best_shared:
        best_shared_program_paths = {
            source_task_id: resolve_best_program_path(
                _adaptation_branch_task_dir(
                    results_path,
                    "best_shared",
                    source_task_id,
                ),
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
            )
            for source_task_id in source_task_ids
        }
        best_shared_results = evaluate_source_programs_on_holdouts(
            program_paths=best_shared_program_paths,
            family="circle_packing",
            holdout_task_specs=holdout_task_specs,
            evaluation_file=manifest.evaluation_file,
            env=env,
        )
        summary["best_shared_adaptation_by_source_task"] = best_shared_results
        write_json(
            holdout_output_root / "best_shared_adaptation_holdouts.json",
            {
                "enabled": True,
                "holdout_task_ids": holdout_task_ids,
                "results_by_source_task": best_shared_results,
            },
        )

    if include_best_local:
        best_local_program_paths = {
            source_task_id: resolve_best_program_path(
                _adaptation_branch_task_dir(
                    results_path,
                    "best_local",
                    source_task_id,
                ),
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
            )
            for source_task_id in source_task_ids
        }
        best_local_results = evaluate_source_programs_on_holdouts(
            program_paths=best_local_program_paths,
            family="circle_packing",
            holdout_task_specs=holdout_task_specs,
            evaluation_file=manifest.evaluation_file,
            env=env,
        )
        summary["best_local_adaptation_by_source_task"] = best_local_results
        write_json(
            holdout_output_root / "best_local_adaptation_holdouts.json",
            {
                "enabled": True,
                "holdout_task_ids": holdout_task_ids,
                "results_by_source_task": best_local_results,
            },
        )

    write_json(summary_path, summary)
    return summary


__all__ = [
    "evaluate_best_program_on_holdouts",
    "evaluate_source_programs_on_holdouts",
    "resolve_best_program_path",
    "run_circle_packing_holdout_evaluation",
    "run_circle_packing_adaptation_holdout_update",
    "unavailable_holdout_evaluation_result",
]
