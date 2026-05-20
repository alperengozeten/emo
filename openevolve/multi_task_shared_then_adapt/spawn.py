"""Checkpoint spawning for multi-task shared-then-adapt workflows."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.multi_task_shared_then_adapt.registry import (
    SharedThenAdaptFamily,
    get_family_definition,
    infer_family_from_task_ids,
)


_EVALUATION_FAMILY_MODULES: Dict[str, str] = {
    "heilbronn_triangle_emo_sta": (
        "openevolve.multi_task_shared_then_adapt.heilbronn_triangle"
    ),
    "hexagon_packing_emo_sta": (
        "openevolve.multi_task_shared_then_adapt.hexagon_packing"
    ),
    "circle_packing_rectangle_emo_sta": (
        "openevolve.multi_task_shared_then_adapt.circle_packing_rectangle"
    ),
}


def _family_module_for_evaluation_file(evaluation_file: Path) -> Optional[str]:
    return _EVALUATION_FAMILY_MODULES.get(evaluation_file.parent.name)


def _reset_stale_family_module_for_evaluator(evaluation_file: Path) -> None:
    family_module_name = _family_module_for_evaluation_file(evaluation_file)
    if family_module_name is None:
        return

    module = sys.modules.get(family_module_name)
    if module is None:
        return

    # Older detached EMO-STA workers can hold onto a family module object that
    # predates resolve_eval_task_specs. Drop it before loading the evaluator so
    # the import rehydrates from current source instead of a stale module.
    if (
        getattr(module, "resolve_eval_task_specs", None) is None
        and getattr(module, "resolve_task_specs", None) is not None
    ):
        sys.modules.pop(family_module_name, None)


def _load_evaluation_module(evaluation_file: Path):
    importlib.invalidate_caches()
    _reset_stale_family_module_for_evaluator(evaluation_file)
    spec = importlib.util.spec_from_file_location(
        f"emo_sta_spawn_evaluator_{hash(str(evaluation_file.resolve()))}",
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


def _normalize_checkpoint_name(path: Path) -> str:
    return path.resolve().name


def _sorted_programs(programs: Iterable[Program]) -> List[Program]:
    return sorted(
        programs,
        key=lambda program: (
            int(program.iteration_found),
            int(program.generation),
            float(program.timestamp),
            program.id,
        ),
    )


def _write_best_program_info(database: ProgramDatabase, checkpoint_path: Path, suffix: str) -> None:
    best_program = database.get_best_program()
    if best_program is None:
        return

    best_program_path = checkpoint_path / f"best_program{suffix}"
    best_program_path.write_text(best_program.code, encoding="utf-8")
    best_program_info_path = checkpoint_path / "best_program_info.json"
    best_program_info_path.write_text(
        json.dumps(
            {
                "id": best_program.id,
                "generation": best_program.generation,
                "iteration": best_program.iteration_found,
                "current_iteration": 0,
                "metrics": best_program.metrics,
                "language": best_program.language,
                "timestamp": best_program.timestamp,
                "saved_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_shared_database(shared_checkpoint_path: Path, base_config_path: Path) -> Tuple[Config, ProgramDatabase]:
    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    database = ProgramDatabase(config.database)
    database.load(str(shared_checkpoint_path))
    return config, database


def _resolve_spawn_family(
    family: Optional[str],
    task_ids: Optional[Iterable[str]],
) -> SharedThenAdaptFamily:
    if family:
        return get_family_definition(family)
    inferred_family = infer_family_from_task_ids(task_ids or ())
    return get_family_definition(inferred_family or "circle_packing")


def _program_file_suffix(
    *,
    program_language: Optional[str],
    default_file_suffix: str,
    initial_program: Optional[Path],
) -> str:
    normalized_language = (program_language or "").strip().lower()
    language_map = {
        "python": ".py",
        "py": ".py",
        "r": ".r",
        "rscript": ".r",
        "rust": ".rs",
        "rs": ".rs",
    }
    if normalized_language in language_map:
        return language_map[normalized_language]
    if normalized_language.startswith(".") and len(normalized_language) > 1:
        return normalized_language

    suffix_candidate = Path(program_language).suffix if program_language else ""
    if suffix_candidate:
        return suffix_candidate
    if initial_program is not None and initial_program.suffix:
        return initial_program.suffix
    if default_file_suffix.startswith(".") and len(default_file_suffix) > 1:
        return default_file_suffix
    fallback = language_map.get(default_file_suffix.strip().lower())
    if fallback is not None:
        return fallback
    if default_file_suffix and not default_file_suffix.startswith("."):
        return f".{default_file_suffix}"
    return ".py"


def _normalize_evaluation_result(
    evaluation_result: Any,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
    if isinstance(evaluation_result, Mapping):
        artifacts = evaluation_result.get("artifacts")
        if not isinstance(artifacts, Mapping):
            artifacts = {}
        nested_metrics = evaluation_result.get("metrics")
        if isinstance(nested_metrics, Mapping):
            return nested_metrics, artifacts
        metrics = {
            key: value
            for key, value in evaluation_result.items()
            if key != "artifacts"
        }
        return metrics, artifacts

    metrics = getattr(evaluation_result, "metrics", None)
    artifacts = getattr(evaluation_result, "artifacts", {}) or {}
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    return metrics if isinstance(metrics, Mapping) else None, artifacts


def _invoke_reevaluate_program_for_task(**kwargs) -> Dict[str, Any]:
    reevaluate_fn = _reevaluate_program_for_task
    signature = inspect.signature(reevaluate_fn)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return reevaluate_fn(**kwargs)
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return reevaluate_fn(**filtered_kwargs)


def _reevaluate_program_for_task(
    *,
    program: Program,
    task_id: str,
    family: str = "circle_packing",
    evaluation_file: Path,
    default_file_suffix: str = ".py",
    initial_program: Optional[Path] = None,
) -> Dict[str, Any]:
    family_definition = get_family_definition(family)
    task = family_definition.tasks_by_id[task_id]
    evaluator_module = _load_evaluation_module(evaluation_file)
    if not hasattr(evaluator_module, "evaluate"):
        raise AttributeError(f"{evaluation_file} does not define evaluate()")

    suffix = _program_file_suffix(
        program_language=program.language,
        default_file_suffix=default_file_suffix,
        initial_program=initial_program,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as handle:
        handle.write(program.code)
        program_path = Path(handle.name)

    try:
        with _TemporaryEnv({family_definition.task_selector_env_var: task_id}):
            raw_result = evaluator_module.evaluate(str(program_path))
            if inspect.isawaitable(raw_result):
                evaluation_result = asyncio.run(raw_result)
            else:
                evaluation_result = raw_result
        metrics, artifacts = _normalize_evaluation_result(evaluation_result)
        task_result = family_definition.extract_task_result(artifacts, task_id)
        if task_result is not None:
            return task_result
        return family_definition.build_task_result(
            task,
            raw_metrics=metrics if isinstance(metrics, Mapping) else None,
        )
    finally:
        program_path.unlink(missing_ok=True)


def _clone_program_for_task(
    source_program: Program,
    *,
    target_metrics: Mapping[str, Any],
    task_artifacts: Mapping[str, Any],
    shared_checkpoint_path: Path,
    shared_last_iteration: int,
    task_id: str,
    source_selection_mode: Optional[str] = None,
) -> Program:
    metadata = deepcopy(source_program.metadata) if isinstance(source_program.metadata, dict) else {}
    metadata.update(
        {
            "sts_warmstarted": True,
            "sts_target_task_id": task_id,
            "sts_source_shared_checkpoint": str(shared_checkpoint_path.resolve()),
            "sts_source_shared_iteration": int(shared_last_iteration),
            "sts_source_shared_program_id": source_program.id,
            "sts_source_shared_metrics": deepcopy(source_program.metrics),
        }
    )
    if source_selection_mode is not None:
        metadata["sts_selection_mode"] = source_selection_mode
    return Program(
        id=source_program.id,
        code=source_program.code,
        changes_description=source_program.changes_description,
        language=source_program.language,
        parent_id=source_program.parent_id,
        generation=source_program.generation,
        timestamp=source_program.timestamp,
        iteration_found=source_program.iteration_found,
        metrics=dict(target_metrics),
        complexity=source_program.complexity,
        diversity=source_program.diversity,
        metadata=metadata,
        prompts=deepcopy(source_program.prompts),
        artifacts_json=json.dumps(task_artifacts),
        artifact_dir=None,
        embedding=deepcopy(source_program.embedding),
    )


def _task_result_score(task_result: Mapping[str, Any]) -> float:
    metrics = task_result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Task result is missing a valid metrics mapping")

    for key in ("final_task_score", "combined_score", "score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise ValueError("Task result metrics are missing final_task_score/combined_score/score")


def _collect_task_entries(
    *,
    source_programs: Iterable[Program],
    shared_database: ProgramDatabase,
    family_definition: SharedThenAdaptFamily,
    task_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task_entries: List[Dict[str, Any]] = []
    reevaluated_entries: List[Dict[str, Any]] = []
    for source_program in source_programs:
        artifacts = shared_database.get_artifacts(source_program.id)
        task_result = family_definition.extract_task_result(artifacts, task_id)
        entry = {
            "source_program": source_program,
            "artifacts": artifacts,
            "task_result": task_result,
            "needs_reevaluation": task_result is None,
        }
        task_entries.append(entry)
        if task_result is None:
            reevaluated_entries.append(entry)
    return task_entries, reevaluated_entries


def _reevaluate_missing_task_entries(
    *,
    task_entries: List[Dict[str, Any]],
    reevaluated_entries: List[Dict[str, Any]],
    task_id: str,
    family_definition: SharedThenAdaptFamily,
    evaluation_path: Path,
    config: Config,
    initial_program_path: Optional[Path],
) -> List[str]:
    if not reevaluated_entries:
        return []

    # Keep spawn-time reevaluation single-threaded. This path dynamically loads
    # evaluator modules and temporarily mutates process-wide task selector env
    # vars, which made threaded reevaluation nondeterministically fail with
    # partially initialized family imports in long-running EMO-STA runs.
    worker_count = 1
    print(
        f"  {len(reevaluated_entries)} program(s) for '{task_id}' are missing full "
        f"task-local artifacts; reevaluating with {worker_count} worker(s)."
    )

    def reevaluate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        source_program = entry["source_program"]
        return _invoke_reevaluate_program_for_task(
            program=source_program,
            task_id=task_id,
            family=family_definition.family,
            evaluation_file=evaluation_path,
            default_file_suffix=config.file_suffix,
            initial_program=initial_program_path,
        )

    if worker_count == 1:
        for idx, entry in enumerate(reevaluated_entries, start=1):
            entry["task_result"] = reevaluate_entry(entry)
            if idx == len(reevaluated_entries) or idx % 5 == 0:
                print(
                    f"    Reevaluated {idx}/{len(reevaluated_entries)} "
                    f"program(s) for '{task_id}'"
                )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_entry = {
                executor.submit(reevaluate_entry, entry): entry
                for entry in reevaluated_entries
            }
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                entry["task_result"] = future.result()
                completed += 1
                if completed == len(reevaluated_entries) or completed % 5 == 0:
                    print(
                        f"    Reevaluated {completed}/{len(reevaluated_entries)} "
                        f"program(s) for '{task_id}'"
                    )

    return [
        entry["source_program"].id
        for entry in task_entries
        if bool(entry["needs_reevaluation"])
    ]


def _select_best_local_entry(task_entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    best_entry: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for entry in task_entries:
        task_result = entry.get("task_result")
        if not isinstance(task_result, Mapping):
            raise RuntimeError("Task-local best-seed selection is missing task_result payload")
        score = _task_result_score(task_result)
        if best_entry is None or score > float(best_score):
            best_entry = entry
            best_score = score
    if best_entry is None:
        raise RuntimeError("Could not select a best local from an empty checkpoint")
    return best_entry


def _write_task_checkpoint(
    *,
    selected_entries: List[Dict[str, Any]],
    task_id: str,
    task_checkpoint_path: Path,
    config: Config,
    family_definition: SharedThenAdaptFamily,
    shared_checkpoint: Path,
    shared_database: ProgramDatabase,
    reevaluated_program_ids: List[str],
    source_selection_mode: Optional[str],
) -> Dict[str, Any]:
    if task_checkpoint_path.exists():
        shutil.rmtree(task_checkpoint_path)
    task_checkpoint_path.mkdir(parents=True, exist_ok=True)

    task_config = deepcopy(config)
    task_config.database.db_path = None
    task_database = ProgramDatabase(task_config.database)

    for entry in selected_entries:
        source_program = entry["source_program"]
        artifacts = entry["artifacts"]
        task_result = entry["task_result"]
        if task_result is None:
            raise RuntimeError(
                f"Task spawn for '{task_id}' did not produce task-local metrics "
                f"for shared program {source_program.id}"
            )

        task_artifacts = family_definition.project_task_artifacts(
            artifacts,
            task_id,
            task_result,
        )
        cloned_program = _clone_program_for_task(
            source_program,
            target_metrics=task_result["metrics"],
            task_artifacts=task_artifacts,
            shared_checkpoint_path=shared_checkpoint,
            shared_last_iteration=shared_database.last_iteration,
            task_id=task_id,
            source_selection_mode=source_selection_mode,
        )
        target_island = None
        if isinstance(source_program.metadata, dict):
            candidate_island = source_program.metadata.get("island")
            if isinstance(candidate_island, int):
                target_island = candidate_island
        task_database.add(cloned_program, target_island=target_island)

    task_database.config.db_path = str(task_checkpoint_path)
    task_database.save(str(task_checkpoint_path), iteration=0)
    _write_best_program_info(task_database, task_checkpoint_path, suffix=config.file_suffix)

    spawn_metadata = {
        "family": family_definition.family,
        "shared_checkpoint_path": str(shared_checkpoint),
        "shared_checkpoint_name": _normalize_checkpoint_name(shared_checkpoint),
        "shared_last_iteration": shared_database.last_iteration,
        "target_task_id": task_id,
        "spawned_program_count": len(task_database.programs),
        "reevaluated_program_ids": reevaluated_program_ids,
        "last_iteration": 0,
    }
    if source_selection_mode is not None and selected_entries:
        selected_program = selected_entries[0]["source_program"]
        spawn_metadata.update(
            {
                "selection_mode": source_selection_mode,
                "source_shared_program_id": selected_program.id,
                "source_shared_metrics": deepcopy(selected_program.metrics),
            }
        )
    (task_checkpoint_path / "spawn_metadata.json").write_text(
        json.dumps(spawn_metadata, indent=2),
        encoding="utf-8",
    )

    result = {
        "checkpoint_path": str(task_checkpoint_path),
        "spawn_metadata_path": str(task_checkpoint_path / "spawn_metadata.json"),
        "best_program_info_path": str(task_checkpoint_path / "best_program_info.json"),
        "reevaluated_program_ids": reevaluated_program_ids,
    }
    if source_selection_mode is not None and selected_entries:
        result.update(
            {
                "selection_mode": source_selection_mode,
                "source_shared_program_id": selected_entries[0]["source_program"].id,
            }
        )
    return result


def _spawn_task_checkpoints(
    *,
    shared_checkpoint_path: str | Path,
    output_root: str | Path,
    base_config_path: str | Path,
    evaluation_file: str | Path,
    family: Optional[str] = None,
    task_ids: Optional[Iterable[str]] = None,
    initial_program: str | Path | None = None,
    selection_mode: str = "all",
) -> Dict[str, Dict[str, Any]]:
    """Spawn one task-specific checkpoint per task from a shared checkpoint."""
    shared_checkpoint = Path(shared_checkpoint_path).resolve()
    output_root_path = Path(output_root).resolve()
    base_config = Path(base_config_path).resolve()
    evaluation_path = Path(evaluation_file).resolve()
    initial_program_path = Path(initial_program).resolve() if initial_program else None
    family_definition = _resolve_spawn_family(family, task_ids)

    config, shared_database = _load_shared_database(shared_checkpoint, base_config)
    output_root_path.mkdir(parents=True, exist_ok=True)

    selected_task_ids = list(task_ids or family_definition.tasks_by_id.keys())
    results: Dict[str, Dict[str, Any]] = {}

    source_programs = _sorted_programs(shared_database.programs.values())
    if not source_programs:
        raise RuntimeError(f"Shared checkpoint {shared_checkpoint} does not contain any programs")

    shared_best_program = None
    if selection_mode == "best_shared":
        shared_best_program = shared_database.get_best_program()
        if shared_best_program is None:
            raise RuntimeError(
                f"Shared checkpoint {shared_checkpoint} does not have a resolvable best program"
            )

    for task_id in selected_task_ids:
        if task_id not in family_definition.tasks_by_id:
            raise ValueError(f"Unknown {family_definition.family} task: {task_id}")

        task_source_programs = source_programs
        source_selection_mode = None
        mode_description = "shared programs"
        if selection_mode == "best_shared":
            task_source_programs = [shared_best_program]
            source_selection_mode = "best_shared"
            mode_description = "best shared"
        elif selection_mode == "best_local":
            source_selection_mode = "best_local"
            mode_description = "best local candidate pool"
        elif selection_mode != "all":
            raise ValueError(f"Unsupported EMO-STA spawn selection mode '{selection_mode}'")

        print(
            f"Spawning EMO-STA checkpoint for task '{task_id}' from "
            f"{shared_checkpoint.name} ({len(task_source_programs)} {mode_description})"
        )
        task_entries, reevaluated_entries = _collect_task_entries(
            source_programs=task_source_programs,
            shared_database=shared_database,
            family_definition=family_definition,
            task_id=task_id,
        )
        reevaluated_programs = _reevaluate_missing_task_entries(
            task_entries=task_entries,
            reevaluated_entries=reevaluated_entries,
            task_id=task_id,
            family_definition=family_definition,
            evaluation_path=evaluation_path,
            config=config,
            initial_program_path=initial_program_path,
        )

        selected_entries = task_entries
        if selection_mode == "best_local":
            selected_entries = [_select_best_local_entry(task_entries)]

        results[task_id] = _write_task_checkpoint(
            selected_entries=selected_entries,
            task_id=task_id,
            task_checkpoint_path=output_root_path / task_id,
            config=config,
            family_definition=family_definition,
            shared_checkpoint=shared_checkpoint,
            shared_database=shared_database,
            reevaluated_program_ids=reevaluated_programs,
            source_selection_mode=source_selection_mode,
        )
        selected_count = len(selected_entries)
        selected_label = (
            f", source {selected_entries[0]['source_program'].id}"
            if source_selection_mode is not None and selected_entries
            else ""
        )
        print(
            f"Finished EMO-STA checkpoint for '{task_id}': "
            f"{selected_count} program(s){selected_label}, "
            f"{len(reevaluated_programs)} re-evaluated."
        )

    return results


def spawn_task_checkpoints(
    *,
    shared_checkpoint_path: str | Path,
    output_root: str | Path,
    base_config_path: str | Path,
    evaluation_file: str | Path,
    family: Optional[str] = None,
    task_ids: Optional[Iterable[str]] = None,
    initial_program: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    return _spawn_task_checkpoints(
        shared_checkpoint_path=shared_checkpoint_path,
        output_root=output_root,
        base_config_path=base_config_path,
        evaluation_file=evaluation_file,
        family=family,
        task_ids=task_ids,
        initial_program=initial_program,
        selection_mode="all",
    )


def spawn_best_shared_checkpoints(
    *,
    shared_checkpoint_path: str | Path,
    output_root: str | Path,
    base_config_path: str | Path,
    evaluation_file: str | Path,
    family: Optional[str] = None,
    task_ids: Optional[Iterable[str]] = None,
    initial_program: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    return _spawn_task_checkpoints(
        shared_checkpoint_path=shared_checkpoint_path,
        output_root=output_root,
        base_config_path=base_config_path,
        evaluation_file=evaluation_file,
        family=family,
        task_ids=task_ids,
        initial_program=initial_program,
        selection_mode="best_shared",
    )


def spawn_best_local_checkpoints(
    *,
    shared_checkpoint_path: str | Path,
    output_root: str | Path,
    base_config_path: str | Path,
    evaluation_file: str | Path,
    family: Optional[str] = None,
    task_ids: Optional[Iterable[str]] = None,
    initial_program: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    return _spawn_task_checkpoints(
        shared_checkpoint_path=shared_checkpoint_path,
        output_root=output_root,
        base_config_path=base_config_path,
        evaluation_file=evaluation_file,
        family=family,
        task_ids=task_ids,
        initial_program=initial_program,
        selection_mode="best_local",
    )
