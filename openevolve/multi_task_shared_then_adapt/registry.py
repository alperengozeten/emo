"""Family registry for multi-task shared-then-adapt workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from . import circle_packing
from . import circle_packing_rectangle
from . import function_minimization
from . import heilbronn_triangle
from . import hexagon_packing
from . import k_module_problem_balanced
from . import sldbench_3d
from . import rust_adaptive_sort
from . import signal_processing


@dataclass(frozen=True)
class SharedThenAdaptFamily:
    """Minimal family-specific dispatch surface for EMO-STA."""

    family: str
    task_selector_env_var: str
    shared_selector: str
    task_specs: tuple[Any, ...]
    tasks_by_id: Mapping[str, Any]
    resolve_task_specs: Callable[[Optional[str]], list[Any]]
    extract_task_result: Callable[[Mapping[str, Any], str], Optional[Dict[str, Any]]]
    build_task_result: Callable[..., Dict[str, Any]]
    project_task_artifacts: Callable[[Mapping[str, Any], str, Mapping[str, Any]], Dict[str, Any]]
    aggregate_task_results: Callable[[Iterable[Mapping[str, Any]]], Dict[str, float]]


FAMILY_REGISTRY: Dict[str, SharedThenAdaptFamily] = {
    "circle_packing": SharedThenAdaptFamily(
        family="circle_packing",
        task_selector_env_var=circle_packing.CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR,
        shared_selector=circle_packing.CIRCLE_PACKING_SHARED_SELECTOR,
        task_specs=circle_packing.CIRCLE_PACKING_TASK_SPECS,
        tasks_by_id=circle_packing.CIRCLE_PACKING_TASKS_BY_ID,
        resolve_task_specs=circle_packing.resolve_task_specs,
        extract_task_result=circle_packing.extract_task_result,
        build_task_result=circle_packing.build_task_result,
        project_task_artifacts=circle_packing.project_task_artifacts,
        aggregate_task_results=circle_packing.aggregate_task_results,
    ),
    "circle_packing_rectangle": SharedThenAdaptFamily(
        family="circle_packing_rectangle",
        task_selector_env_var=(
            circle_packing_rectangle.CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR
        ),
        shared_selector=circle_packing_rectangle.CIRCLE_PACKING_RECTANGLE_SHARED_SELECTOR,
        task_specs=circle_packing_rectangle.CIRCLE_PACKING_RECTANGLE_TASK_SPECS,
        tasks_by_id=circle_packing_rectangle.CIRCLE_PACKING_RECTANGLE_TASKS_BY_ID,
        resolve_task_specs=circle_packing_rectangle.resolve_task_specs,
        extract_task_result=circle_packing_rectangle.extract_task_result,
        build_task_result=circle_packing_rectangle.build_task_result,
        project_task_artifacts=circle_packing_rectangle.project_task_artifacts,
        aggregate_task_results=circle_packing_rectangle.aggregate_task_results,
    ),
    "heilbronn_triangle": SharedThenAdaptFamily(
        family="heilbronn_triangle",
        task_selector_env_var=(
            heilbronn_triangle.HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR
        ),
        shared_selector=heilbronn_triangle.HEILBRONN_TRIANGLE_SHARED_SELECTOR,
        task_specs=heilbronn_triangle.HEILBRONN_TRIANGLE_TASK_SPECS,
        tasks_by_id=heilbronn_triangle.HEILBRONN_TRIANGLE_TASKS_BY_ID,
        resolve_task_specs=heilbronn_triangle.resolve_task_specs,
        extract_task_result=heilbronn_triangle.extract_task_result,
        build_task_result=heilbronn_triangle.build_task_result,
        project_task_artifacts=heilbronn_triangle.project_task_artifacts,
        aggregate_task_results=heilbronn_triangle.aggregate_task_results,
    ),
    "hexagon_packing": SharedThenAdaptFamily(
        family="hexagon_packing",
        task_selector_env_var=hexagon_packing.HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR,
        shared_selector=hexagon_packing.HEXAGON_PACKING_SHARED_SELECTOR,
        task_specs=hexagon_packing.HEXAGON_PACKING_TASK_SPECS,
        tasks_by_id=hexagon_packing.HEXAGON_PACKING_TASKS_BY_ID,
        resolve_task_specs=hexagon_packing.resolve_task_specs,
        extract_task_result=hexagon_packing.extract_task_result,
        build_task_result=hexagon_packing.build_task_result,
        project_task_artifacts=hexagon_packing.project_task_artifacts,
        aggregate_task_results=hexagon_packing.aggregate_task_results,
    ),
    "k_module_problem_balanced": SharedThenAdaptFamily(
        family="k_module_problem_balanced",
        task_selector_env_var=(
            k_module_problem_balanced.K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR
        ),
        shared_selector=k_module_problem_balanced.K_MODULE_BALANCED_SHARED_SELECTOR,
        task_specs=k_module_problem_balanced.K_MODULE_BALANCED_TASK_SPECS,
        tasks_by_id=k_module_problem_balanced.K_MODULE_BALANCED_TASKS_BY_ID,
        resolve_task_specs=k_module_problem_balanced.resolve_task_specs,
        extract_task_result=k_module_problem_balanced.extract_task_result,
        build_task_result=k_module_problem_balanced.build_task_result,
        project_task_artifacts=k_module_problem_balanced.project_task_artifacts,
        aggregate_task_results=k_module_problem_balanced.aggregate_task_results,
    ),
    "function_minimization": SharedThenAdaptFamily(
        family="function_minimization",
        task_selector_env_var=function_minimization.FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR,
        shared_selector=function_minimization.FUNCTION_MINIMIZATION_SHARED_SELECTOR,
        task_specs=function_minimization.FUNCTION_MINIMIZATION_TASK_SPECS,
        tasks_by_id=function_minimization.FUNCTION_MINIMIZATION_TASKS_BY_ID,
        resolve_task_specs=function_minimization.resolve_task_specs,
        extract_task_result=function_minimization.extract_task_result,
        build_task_result=function_minimization.build_task_result,
        project_task_artifacts=function_minimization.project_task_artifacts,
        aggregate_task_results=function_minimization.aggregate_task_results,
    ),
    "sldbench_3d": SharedThenAdaptFamily(
        family="sldbench_3d",
        task_selector_env_var=sldbench_3d.SLDBENCH_3D_TASK_SELECTOR_ENV_VAR,
        shared_selector=sldbench_3d.SLDBENCH_3D_SHARED_SELECTOR,
        task_specs=sldbench_3d.SLDBENCH_3D_TASK_SPECS,
        tasks_by_id=sldbench_3d.SLDBENCH_3D_TASKS_BY_ID,
        resolve_task_specs=sldbench_3d.resolve_task_specs,
        extract_task_result=sldbench_3d.extract_task_result,
        build_task_result=sldbench_3d.build_task_result,
        project_task_artifacts=sldbench_3d.project_task_artifacts,
        aggregate_task_results=sldbench_3d.aggregate_task_results,
    ),
    "signal_processing": SharedThenAdaptFamily(
        family="signal_processing",
        task_selector_env_var=signal_processing.SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR,
        shared_selector=signal_processing.SIGNAL_PROCESSING_SHARED_SELECTOR,
        task_specs=signal_processing.SIGNAL_PROCESSING_TASK_SPECS,
        tasks_by_id=signal_processing.SIGNAL_PROCESSING_TASKS_BY_ID,
        resolve_task_specs=signal_processing.resolve_task_specs,
        extract_task_result=signal_processing.extract_task_result,
        build_task_result=signal_processing.build_task_result,
        project_task_artifacts=signal_processing.project_task_artifacts,
        aggregate_task_results=signal_processing.aggregate_task_results,
    ),
    "rust_adaptive_sort": SharedThenAdaptFamily(
        family="rust_adaptive_sort",
        task_selector_env_var=rust_adaptive_sort.RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR,
        shared_selector=rust_adaptive_sort.RUST_ADAPTIVE_SORT_SHARED_SELECTOR,
        task_specs=rust_adaptive_sort.RUST_ADAPTIVE_SORT_TASK_SPECS,
        tasks_by_id=rust_adaptive_sort.RUST_ADAPTIVE_SORT_TASKS_BY_ID,
        resolve_task_specs=rust_adaptive_sort.resolve_task_specs,
        extract_task_result=rust_adaptive_sort.extract_task_result,
        build_task_result=rust_adaptive_sort.build_task_result,
        project_task_artifacts=rust_adaptive_sort.project_task_artifacts,
        aggregate_task_results=rust_adaptive_sort.aggregate_task_results,
    ),
}


def get_family_definition(family: str) -> SharedThenAdaptFamily:
    try:
        return FAMILY_REGISTRY[family]
    except KeyError as exc:
        available = ", ".join(sorted(FAMILY_REGISTRY))
        raise ValueError(f"Unsupported EMO-STA family '{family}'. Available: {available}") from exc


def infer_family_from_task_ids(task_ids: Iterable[str]) -> Optional[str]:
    normalized_task_ids = list(task_ids)
    if not normalized_task_ids:
        return None
    inference_task_ids: Dict[str, Mapping[str, Any]] = {}
    for family_name, family in FAMILY_REGISTRY.items():
        if family_name == "circle_packing":
            inference_task_ids[family_name] = {
                **family.tasks_by_id,
                **circle_packing.CIRCLE_PACKING_HOLDOUT_TASKS_BY_ID,
            }
        else:
            inference_task_ids[family_name] = family.tasks_by_id
    matching = [
        family_name
        for family_name, family in FAMILY_REGISTRY.items()
        if all(task_id in inference_task_ids[family_name] for task_id in normalized_task_ids)
    ]
    if len(matching) == 1:
        return matching[0]
    return None
