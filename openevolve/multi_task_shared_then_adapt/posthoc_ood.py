"""Post-hoc OOD evaluation for finished EMO-STA runs."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from openevolve.multi_task_shared_then_adapt import (
    circle_packing_rectangle,
    heilbronn_triangle,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.runner import write_json
from openevolve.multi_task_shared_then_adapt.workflow import (
    SharedThenAdaptManifest,
    family_task_specs,
)


POSTHOC_OOD_EVALUATION_REGIME = "posthoc_unseen_nearby_n"
POSTHOC_OOD_SELECTION_NOTE = (
    "Programs were selected by completed in-distribution runs, not by OOD score."
)
POSTHOC_OOD_NOTE = (
    "This is post-hoc OOD evaluation on frozen finished programs; evolution was "
    "not rerun with OOD tasks included."
)


@dataclass(frozen=True)
class OodFamilySupport:
    family: str
    resolve_ood_task_specs: Callable[[Optional[str]], list[Any]]
    metric_keys: tuple[str, ...]
    objective_metric_key: str
    target_metric_key: str
    directions: Mapping[str, str]


@dataclass(frozen=True)
class DiscoveredProgram:
    label: str
    source_kind: str
    source_task_id: Optional[str]
    program_path: Path
    discovery_source: str


SUPPORTED_OOD_FAMILIES: Dict[str, OodFamilySupport] = {
    "heilbronn_triangle": OodFamilySupport(
        family="heilbronn_triangle",
        resolve_ood_task_specs=heilbronn_triangle.resolve_ood_task_specs,
        metric_keys=(
            "min_triangle_area",
            "target_min_area",
            "target_ratio",
            "validity",
        ),
        objective_metric_key="min_triangle_area",
        target_metric_key="target_min_area",
        directions={
            "heil_tri_n8": "backward",
            "heil_tri_n13": "forward",
            "heil_tri_n14": "forward",
        },
    ),
    "circle_packing_rectangle": OodFamilySupport(
        family="circle_packing_rectangle",
        resolve_ood_task_specs=circle_packing_rectangle.resolve_ood_task_specs,
        metric_keys=(
            "sum_radii",
            "target_sum_radii",
            "target_ratio",
            "validity",
            "alpha",
        ),
        objective_metric_key="sum_radii",
        target_metric_key="target_sum_radii",
        directions={
            "cp_rect_n19": "backward",
            "cp_rect_n24": "forward",
            "cp_rect_n25": "forward",
        },
    ),
}


def _load_evaluation_module(evaluation_file: Path):
    spec = importlib.util.spec_from_file_location(
        f"emo_sta_posthoc_ood_eval_{hash(str(evaluation_file.resolve()))}",
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


def _read_json_if_present(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _suffix_from_language(language: Any) -> Optional[str]:
    normalized = str(language or "").strip().lower()
    if not normalized:
        return None
    language_map = {
        "python": ".py",
        "py": ".py",
        "r": ".r",
        "rscript": ".r",
        "rust": ".rs",
        "rs": ".rs",
    }
    if normalized in language_map:
        return language_map[normalized]
    if normalized.startswith(".") and len(normalized) > 1:
        return normalized
    suffix = Path(normalized).suffix
    return suffix or None


def _candidate_suffixes(
    *,
    initial_program: str | Path | None,
    best_program_info: Mapping[str, Any],
) -> list[str]:
    suffixes: list[str] = []
    if initial_program is not None:
        initial_suffix = Path(initial_program).suffix
        if initial_suffix:
            suffixes.append(initial_suffix)
    language_suffix = _suffix_from_language(best_program_info.get("language"))
    if language_suffix:
        suffixes.append(language_suffix)
    suffixes.extend([".py", ".r", ".rs"])

    deduped: list[str] = []
    seen: set[str] = set()
    for suffix in suffixes:
        if suffix and suffix not in seen:
            seen.add(suffix)
            deduped.append(suffix)
    return deduped


def resolve_best_program_path(
    base_dir: str | Path,
    *,
    initial_program: str | Path | None = None,
    checkpoint_layout: bool | None = None,
) -> Optional[Path]:
    """Resolve an OpenEvolve best-program artifact path under a run/checkpoint dir."""
    base_path = Path(base_dir).resolve()
    if base_path.is_file():
        return base_path
    if not base_path.is_dir():
        return None

    if checkpoint_layout is True:
        search_roots = [base_path]
    elif checkpoint_layout is False:
        search_roots = [base_path / "best"]
    else:
        search_roots = [base_path / "best", base_path]

    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        info = _read_json_if_present(search_root / "best_program_info.json")

        for key in ("program_path", "best_program_path", "path"):
            raw_path = info.get(key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = search_root / candidate
            if candidate.is_file():
                return candidate.resolve()

        for suffix in _candidate_suffixes(
            initial_program=initial_program,
            best_program_info=info,
        ):
            candidate = search_root / f"best_program{suffix}"
            if candidate.is_file():
                return candidate.resolve()

        extensionless_candidate = search_root / "best_program"
        if extensionless_candidate.is_file():
            return extensionless_candidate.resolve()

        wildcard_matches = sorted(search_root.glob("best_program.*"))
        for candidate in wildcard_matches:
            if candidate.name == "best_program_info.json":
                continue
            if candidate.is_file():
                return candidate.resolve()

    return None


def _load_comparison_summary(results_dir: Path) -> Mapping[str, Any]:
    return _read_json_if_present(results_dir / "comparison_summary.json")


def _add_program_if_present(
    programs: dict[str, DiscoveredProgram],
    *,
    label: str,
    source_kind: str,
    source_task_id: Optional[str],
    base_dir: str | Path | None,
    initial_program: Path,
    checkpoint_layout: bool | None,
    discovery_source: str,
) -> None:
    if label in programs or base_dir is None:
        return
    program_path = resolve_best_program_path(
        base_dir,
        initial_program=initial_program,
        checkpoint_layout=checkpoint_layout,
    )
    if program_path is None:
        return
    programs[label] = DiscoveredProgram(
        label=label,
        source_kind=source_kind,
        source_task_id=source_task_id,
        program_path=program_path,
        discovery_source=discovery_source,
    )


def _legacy_branch_key(branch: str) -> str:
    return f"{branch}_{'seed'}_adaptation"


def _adaptation_branch_dir(results_path: Path, branch: str, task_id: str) -> Path:
    canonical = results_path / f"adaptation_{branch}" / task_id
    if canonical.is_dir():
        return canonical
    legacy = results_path / f"adaptation_{branch}_{'seed'}" / task_id
    if legacy.is_dir():
        return legacy
    return canonical


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def discover_finished_programs(
    *,
    manifest: SharedThenAdaptManifest,
    results_dir: str | Path,
    include_shared: bool = True,
    include_adapted: bool = True,
    include_baselines: bool = True,
    include_best_shared: bool = False,
    include_best_local: bool = False,
) -> list[DiscoveredProgram]:
    """Discover frozen best programs from a completed static EMO-STA run."""
    results_path = Path(results_dir).resolve()
    task_ids = [task.task_id for task in family_task_specs(manifest)]
    summary = _load_comparison_summary(results_path)
    programs: dict[str, DiscoveredProgram] = {}

    if summary:
        shared_run = summary.get("shared_run")
        if include_shared and isinstance(shared_run, Mapping):
            _add_program_if_present(
                programs,
                label="shared_best",
                source_kind="shared_best",
                source_task_id=None,
                base_dir=shared_run.get("output_dir"),
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
                discovery_source="comparison_summary.shared_run.output_dir",
            )
            _add_program_if_present(
                programs,
                label="shared_best",
                source_kind="shared_best",
                source_task_id=None,
                base_dir=shared_run.get("checkpoint_used"),
                initial_program=manifest.initial_program,
                checkpoint_layout=True,
                discovery_source="comparison_summary.shared_run.checkpoint_used",
            )

        task_summaries = summary.get("tasks")
        if isinstance(task_summaries, Mapping):
            for task_id in task_ids:
                task_summary = task_summaries.get(task_id)
                if not isinstance(task_summary, Mapping):
                    continue
                if include_adapted:
                    _add_program_if_present(
                        programs,
                        label=f"adapted__{task_id}",
                        source_kind="adapted",
                        source_task_id=task_id,
                        base_dir=task_summary.get("adaptation_output_dir"),
                        initial_program=manifest.initial_program,
                        checkpoint_layout=False,
                        discovery_source=(
                            f"comparison_summary.tasks.{task_id}.adaptation_output_dir"
                        ),
                    )
                if include_baselines:
                    _add_program_if_present(
                        programs,
                        label=f"baseline__{task_id}",
                        source_kind="baseline",
                        source_task_id=task_id,
                        base_dir=task_summary.get("baseline_output_dir"),
                        initial_program=manifest.initial_program,
                        checkpoint_layout=False,
                        discovery_source=(
                            f"comparison_summary.tasks.{task_id}.baseline_output_dir"
                        ),
                    )
                if include_best_shared:
                    branch_key = _legacy_branch_key("best_shared")
                    _add_program_if_present(
                        programs,
                        label=f"best_shared__{task_id}",
                        source_kind="best_shared",
                        source_task_id=task_id,
                        base_dir=_first_value(
                            task_summary,
                            "best_shared_adaptation_output_dir",
                            f"{branch_key}_output_dir",
                        ),
                        initial_program=manifest.initial_program,
                        checkpoint_layout=False,
                        discovery_source=(
                            f"comparison_summary.tasks.{task_id}.best_shared_adaptation_output_dir"
                        ),
                    )
                if include_best_local:
                    branch_key = _legacy_branch_key("best_local")
                    _add_program_if_present(
                        programs,
                        label=f"best_local__{task_id}",
                        source_kind="best_local",
                        source_task_id=task_id,
                        base_dir=_first_value(
                            task_summary,
                            "best_local_adaptation_output_dir",
                            f"{branch_key}_output_dir",
                        ),
                        initial_program=manifest.initial_program,
                        checkpoint_layout=False,
                        discovery_source=(
                            f"comparison_summary.tasks.{task_id}.best_local_adaptation_output_dir"
                        ),
                    )

    if include_shared:
        _add_program_if_present(
            programs,
            label="shared_best",
            source_kind="shared_best",
            source_task_id=None,
            base_dir=results_path / "shared_run",
            initial_program=manifest.initial_program,
            checkpoint_layout=False,
            discovery_source="standard_layout.shared_run.best",
        )

    for task_id in task_ids:
        if include_adapted:
            _add_program_if_present(
                programs,
                label=f"adapted__{task_id}",
                source_kind="adapted",
                source_task_id=task_id,
                base_dir=results_path / "adaptation" / task_id,
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
                discovery_source="standard_layout.adaptation",
            )
        if include_baselines:
            _add_program_if_present(
                programs,
                label=f"baseline__{task_id}",
                source_kind="baseline",
                source_task_id=task_id,
                base_dir=results_path / "baselines" / task_id,
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
                discovery_source="standard_layout.baselines",
            )
        if include_best_shared:
            _add_program_if_present(
                programs,
                label=f"best_shared__{task_id}",
                source_kind="best_shared",
                source_task_id=task_id,
                base_dir=_adaptation_branch_dir(results_path, "best_shared", task_id),
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
                discovery_source="standard_layout.adaptation_best_shared",
            )
        if include_best_local:
            _add_program_if_present(
                programs,
                label=f"best_local__{task_id}",
                source_kind="best_local",
                source_task_id=task_id,
                base_dir=_adaptation_branch_dir(results_path, "best_local", task_id),
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
                discovery_source="standard_layout.adaptation_best_local",
            )

    ordered_labels: list[str] = []
    if include_shared:
        ordered_labels.append("shared_best")
    if include_adapted:
        ordered_labels.extend(f"adapted__{task_id}" for task_id in task_ids)
    if include_baselines:
        ordered_labels.extend(f"baseline__{task_id}" for task_id in task_ids)
    if include_best_shared:
        ordered_labels.extend(f"best_shared__{task_id}" for task_id in task_ids)
    if include_best_local:
        ordered_labels.extend(f"best_local__{task_id}" for task_id in task_ids)
    return [programs[label] for label in ordered_labels if label in programs]


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


def _compact_metrics(
    metrics: Mapping[str, Any],
    *,
    family_support: OodFamilySupport,
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in family_support.metric_keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
    return compact


def _score_from_metrics(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return None


def _failure_task_result(
    *,
    family: str,
    task: Any,
    error: str,
) -> Dict[str, Any]:
    if family == "heilbronn_triangle":
        return heilbronn_triangle.build_task_result(
            task,
            raw_metrics=None,
            error=error,
        )
    if family == "circle_packing_rectangle":
        return circle_packing_rectangle.build_task_result(
            task,
            raw_metrics=None,
            error=error,
        )
    raise ValueError(f"Unsupported post-hoc OOD family: {family}")


def _extract_task_result(
    *,
    family: str,
    task: Any,
    evaluation_result: Any,
) -> Dict[str, Any]:
    family_definition = get_family_definition(family)
    artifacts = _extract_evaluation_artifacts(evaluation_result)
    task_result = family_definition.extract_task_result(artifacts, task.task_id)
    if task_result is not None:
        return dict(task_result)

    metrics = _extract_evaluation_metrics(evaluation_result)
    if family == "heilbronn_triangle":
        return heilbronn_triangle.build_task_result(
            task,
            raw_metrics=metrics,
            error="OOD evaluation did not return a per-task artifact entry",
        )
    if family == "circle_packing_rectangle":
        return circle_packing_rectangle.build_task_result(
            task,
            raw_metrics=metrics,
            error="OOD evaluation did not return a per-task artifact entry",
        )
    raise ValueError(f"Unsupported post-hoc OOD family: {family}")


def evaluate_program_on_ood_tasks(
    *,
    program_path: str | Path,
    family: str,
    ood_task_specs: Sequence[Any],
    evaluation_file: str | Path,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate one frozen program on explicit OOD task specs."""
    family_support = SUPPORTED_OOD_FAMILIES.get(family)
    if family_support is None:
        supported = ", ".join(sorted(SUPPORTED_OOD_FAMILIES))
        raise ValueError(f"Unsupported post-hoc OOD family '{family}'. Supported: {supported}")

    family_definition = get_family_definition(family)
    resolved_program_path = Path(program_path).resolve()
    if not resolved_program_path.is_file():
        raise FileNotFoundError(f"Missing program file: {resolved_program_path}")

    evaluation_path = Path(evaluation_file).resolve()
    evaluator_module = _load_evaluation_module(evaluation_path)
    evaluate_fn = getattr(evaluator_module, "evaluate", None)
    if not callable(evaluate_fn):
        raise AttributeError(f"{evaluation_path} does not define evaluate()")

    env_updates = dict(env or {})
    results: Dict[str, Dict[str, Any]] = {}
    for task in ood_task_specs:
        try:
            with _TemporaryEnv(
                {
                    **env_updates,
                    family_definition.task_selector_env_var: task.task_id,
                }
            ):
                raw_result = evaluate_fn(str(resolved_program_path))
                if inspect.isawaitable(raw_result):
                    evaluation_result = asyncio.run(raw_result)
                else:
                    evaluation_result = raw_result
            task_result = _extract_task_result(
                family=family,
                task=task,
                evaluation_result=evaluation_result,
            )
        except Exception as exc:
            task_result = _failure_task_result(
                family=family,
                task=task,
                error=f"{type(exc).__name__}: {exc}",
            )

        metrics = task_result.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        result = {
            "score": _score_from_metrics(metrics, "score"),
            "combined_score": _score_from_metrics(metrics, "combined_score"),
            "metrics": _compact_metrics(metrics, family_support=family_support),
        }
        error = task_result.get("error")
        if error:
            result["error"] = str(error)
        results[task.task_id] = result
    return results


def _csv_rows_from_summary(
    summary: Mapping[str, Any],
    *,
    family_support: OodFamilySupport,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    programs = summary.get("programs")
    if not isinstance(programs, Mapping):
        return rows

    for program_label, program_payload in programs.items():
        if not isinstance(program_payload, Mapping):
            continue
        ood_results = program_payload.get("ood_results")
        if not isinstance(ood_results, Mapping):
            continue
        for task_id, task_result in ood_results.items():
            if not isinstance(task_result, Mapping):
                continue
            metrics = task_result.get("metrics")
            if not isinstance(metrics, Mapping):
                metrics = {}
            row: Dict[str, Any] = {
                "program_label": program_label,
                "source_kind": program_payload.get("source_kind"),
                "source_task_id": program_payload.get("source_task_id"),
                "ood_task_id": task_id,
                "direction": family_support.directions.get(str(task_id), ""),
                "score": task_result.get("score"),
                "combined_score": task_result.get("combined_score"),
                "validity": metrics.get("validity"),
                "objective_value": metrics.get(family_support.objective_metric_key),
                "target_value": metrics.get(family_support.target_metric_key),
                "error": task_result.get("error"),
            }
            for key in family_support.metric_keys:
                row[key] = metrics.get(key)
            rows.append(row)
    return rows


def write_ood_csv(
    *,
    csv_path: str | Path,
    summary: Mapping[str, Any],
    family_support: OodFamilySupport,
) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldname_candidates = [
        "program_label",
        "source_kind",
        "source_task_id",
        "ood_task_id",
        "direction",
        "score",
        "combined_score",
        "validity",
        "objective_value",
        "target_value",
        *family_support.metric_keys,
        "error",
    ]
    fieldnames: list[str] = []
    for fieldname in fieldname_candidates:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)
    rows = _csv_rows_from_summary(summary, family_support=family_support)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_posthoc_ood_evaluation(
    *,
    manifest: SharedThenAdaptManifest,
    results_dir: str | Path,
    ood_task_ids: Optional[str] = None,
    include_shared: bool = True,
    include_adapted: bool = True,
    include_baselines: bool = True,
    include_best_shared: bool = False,
    include_best_local: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    merge_existing: bool = False,
    write_csv: bool = True,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Evaluate frozen finished EMO-STA best programs on family OOD tasks."""
    family_support = SUPPORTED_OOD_FAMILIES.get(manifest.family)
    if family_support is None:
        supported = ", ".join(sorted(SUPPORTED_OOD_FAMILIES))
        raise ValueError(
            f"Unsupported post-hoc OOD family '{manifest.family}'. Supported: {supported}"
        )

    results_path = Path(results_dir).resolve()
    if not results_path.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_path}")

    output_path = Path(output_dir).resolve() if output_dir else results_path / "posthoc_ood"
    summary_path = output_path / "ood_summary.json"
    csv_path = output_path / "ood_summary.csv"
    if (
        not overwrite
        and not merge_existing
        and (summary_path.exists() or csv_path.exists())
    ):
        raise FileExistsError(
            f"Post-hoc OOD outputs already exist under {output_path}; use --overwrite"
        )

    existing_summary: Dict[str, Any] = {}
    existing_programs: Dict[str, Dict[str, Any]] = {}
    if merge_existing and summary_path.is_file():
        loaded_summary = _read_json_if_present(summary_path)
        if loaded_summary:
            existing_summary = dict(loaded_summary)
            loaded_programs = existing_summary.get("programs")
            if isinstance(loaded_programs, Mapping):
                existing_programs = {
                    str(label): dict(payload)
                    for label, payload in loaded_programs.items()
                    if isinstance(payload, Mapping)
                }

    ood_task_specs = family_support.resolve_ood_task_specs(ood_task_ids)
    discovered_programs = discover_finished_programs(
        manifest=manifest,
        results_dir=results_path,
        include_shared=include_shared,
        include_adapted=include_adapted,
        include_baselines=include_baselines,
        include_best_shared=include_best_shared,
        include_best_local=include_best_local,
    )
    if merge_existing and existing_programs:
        discovered_programs = [
            program
            for program in discovered_programs
            if overwrite or program.label not in existing_programs
        ]

    if not discovered_programs and not existing_programs:
        raise FileNotFoundError(
            f"No finished best programs discovered under {results_path}"
        )

    training_task_ids = [task.task_id for task in family_task_specs(manifest)]
    programs_payload: Dict[str, Dict[str, Any]] = dict(existing_programs)
    for program in discovered_programs:
        ood_results = evaluate_program_on_ood_tasks(
            program_path=program.program_path,
            family=manifest.family,
            ood_task_specs=ood_task_specs,
            evaluation_file=manifest.evaluation_file,
            env=env,
        )
        programs_payload[program.label] = {
            "program_path": str(program.program_path),
            "source_kind": program.source_kind,
            "source_task_id": program.source_task_id,
            "discovery_source": program.discovery_source,
            "ood_results": ood_results,
        }

    summary: Dict[str, Any] = {
        "algorithm": "posthoc_ood_evaluation",
        "family": manifest.family,
        "manifest_path": str(manifest.manifest_path),
        "results_dir": str(results_path),
        "output_dir": str(output_path),
        "training_task_ids": training_task_ids,
        "ood_tasks": [task.task_id for task in ood_task_specs],
        "evaluation_regime": POSTHOC_OOD_EVALUATION_REGIME,
        "selection_note": POSTHOC_OOD_SELECTION_NOTE,
        "note": POSTHOC_OOD_NOTE,
        "include_shared": bool(existing_summary.get("include_shared", False))
        or bool(include_shared),
        "include_adapted": bool(existing_summary.get("include_adapted", False))
        or bool(include_adapted),
        "include_baselines": bool(existing_summary.get("include_baselines", False))
        or bool(include_baselines),
        "include_best_shared": bool(
            existing_summary.get("include_best_shared", False)
        )
        or bool(include_best_shared),
        "include_best_local": bool(
            existing_summary.get("include_best_local", False)
        )
        or bool(include_best_local),
        "program_count": len(programs_payload),
        "programs": programs_payload,
    }
    for key, value in existing_summary.items():
        summary.setdefault(key, value)
    summary["summary_path"] = str(summary_path)
    summary["csv_path"] = str(csv_path)

    write_json(summary_path, summary)
    if write_csv:
        write_ood_csv(
            csv_path=csv_path,
            summary=summary,
            family_support=family_support,
        )
    return summary


def supported_posthoc_ood_families() -> list[str]:
    return sorted(SUPPORTED_OOD_FAMILIES)


__all__ = [
    "DiscoveredProgram",
    "OodFamilySupport",
    "POSTHOC_OOD_EVALUATION_REGIME",
    "POSTHOC_OOD_NOTE",
    "POSTHOC_OOD_SELECTION_NOTE",
    "SUPPORTED_OOD_FAMILIES",
    "discover_finished_programs",
    "evaluate_program_on_ood_tasks",
    "resolve_best_program_path",
    "run_posthoc_ood_evaluation",
    "supported_posthoc_ood_families",
    "write_ood_csv",
]
