"""Canonical SLDBench 3D task family definitions for multi-task STS."""

from __future__ import annotations

from dataclasses import dataclass
import math
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Mapping, Optional


SLDBENCH_3D_TASK_SELECTOR_ENV_VAR = "SLDBENCH_3D_TASK_ID"
SLDBENCH_3D_SHARED_SELECTOR = "all"
SLDBENCH_DATASET_REPO_ID = "pkuHaowei/sldbench"
CANONICAL_FEATURE_NAMES = (
    "model_size_like",
    "diversity_like",
    "total_data_like",
)
SLDBENCH_3D_METRIC_KEYS: tuple[str, ...] = (
    "nmse",
    "nmae",
    "r2",
    "fit_group_count",
    "eval_group_count",
    "successful_group_count",
    "failed_group_count",
    "score",
    "combined_score",
)


@dataclass(frozen=True)
class SLDBench3DTaskSpec:
    """Stable task identity for the EMO-STA SLDBench 3D family."""

    task_id: str
    task_index: int
    display_name: str
    original_feature_names: tuple[str, str, str]
    canonical_source_feature_names: tuple[str, str, str]
    target_name: str
    canonical_feature_names: tuple[str, str, str] = CANONICAL_FEATURE_NAMES
    input_dim: int = 3
    output_dim: int = 1
    param_budget: int = 7
    dataset_repo_id: str = SLDBENCH_DATASET_REPO_ID
    hf_config_name: Optional[str] = None

    def to_spec_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "original_feature_names": list(self.original_feature_names),
            "canonical_feature_names": list(self.canonical_feature_names),
            "target_name": self.target_name,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "param_budget": int(self.param_budget),
        }


SLDBENCH_3D_TASK_SPECS: tuple[SLDBench3DTaskSpec, ...] = (
    SLDBench3DTaskSpec(
        task_id="vocab_scaling_law",
        task_index=0,
        display_name="VocabularyScalingLaw",
        original_feature_names=(
            "non_vocab_parameters",
            "vocab_size",
            "num_characters",
        ),
        canonical_source_feature_names=(
            "non_vocab_parameters",
            "vocab_size",
            "num_characters",
        ),
        target_name="unigram_normalized_loss",
        hf_config_name="vocab_scaling_law",
    ),
    SLDBench3DTaskSpec(
        task_id="data_constrained_scaling_law",
        task_index=1,
        display_name="DataConstrainedScalingLaw",
        original_feature_names=(
            "unique_tokens",
            "params",
            "tokens",
        ),
        canonical_source_feature_names=(
            "params",
            "unique_tokens",
            "tokens",
        ),
        target_name="loss",
        hf_config_name="data_constrained_scaling_law",
    ),
)

SLDBENCH_3D_TASKS_BY_ID: Dict[str, SLDBench3DTaskSpec] = {
    task.task_id: task for task in SLDBENCH_3D_TASK_SPECS
}


def build_generic_system_prompt_for_sldbench_3d() -> str:
    """Return the EMO-STA prompt for the SLDBench 3D family."""
    return dedent(
        """
        You are improving a generic 3D scaling-law family for two related extrapolation tasks.

        The evolving code must preserve these exact function signatures:

        def scaling_law_func(data_points, params):
            ...

        def fit_scaling_law(data_points, loss_values):
            ...

        In this EMO-STA family, data_points always has shape (N, 3) with canonical columns:
        [model_size_like, diversity_like, total_data_like]

        Task-specific raw columns are canonicalized before they reach your code:
        - vocab_scaling_law:
          [non_vocab_parameters, vocab_size, num_characters]
          -> [model_size_like, diversity_like, total_data_like]
        - data_constrained_scaling_law:
          [unique_tokens, params, tokens]
          -> [model_size_like, diversity_like, total_data_like]
          where params becomes model_size_like.

        Constraints:
        - Use no more than 7 parameters.
        - Do not hardcode task IDs or separate formulas by task name.
        - Do not rely on input-dependent global statistics inside scaling_law_func
          such as min/max/median/quantiles over the current batch.
        - Keep predictions finite and numerically stable.
        - Fit deterministically.
        - Focus on extrapolation quality on held-out larger-scale settings.
        - Favor simple, stable, parameter-efficient law forms.
        - Log-domain reasoning, bounded exponents, positive reparameterizations,
          clipping, and numerically stable optimization are encouraged.

        The goal is to improve both:
        - the symbolic law structure in scaling_law_func
        - the fitting routine in fit_scaling_law

        Write all improvements only between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.
        """
    ).strip()


def get_sldbench_3d_system_prompt(task_id: Optional[str] = None) -> str:
    """Return the EMO-STA prompt for the SLDBench 3D family."""
    return build_generic_system_prompt_for_sldbench_3d()


def resolve_task_specs(selector: Optional[str]) -> List[SLDBench3DTaskSpec]:
    """Resolve a task selector into concrete SLDBench 3D task specs."""
    normalized = (selector or SLDBENCH_3D_SHARED_SELECTOR).strip()
    if not normalized or normalized == SLDBENCH_3D_SHARED_SELECTOR:
        return list(SLDBENCH_3D_TASK_SPECS)
    if normalized not in SLDBENCH_3D_TASKS_BY_ID:
        available = ", ".join(task.task_id for task in SLDBENCH_3D_TASK_SPECS)
        raise ValueError(f"Unknown SLDBench 3D task '{normalized}'. Available: {available}")
    return [SLDBENCH_3D_TASKS_BY_ID[normalized]]


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _coerce_non_negative_float(value: Any, default: float) -> float:
    return max(0.0, _coerce_finite_float(value, default))


def empty_task_metrics() -> Dict[str, float]:
    """Finite failure metrics safe for ranking, logging, and checkpoint spawning."""
    return {
        "nmse": 100000.0,
        "nmae": 100000.0,
        "r2": -1.0,
        "fit_group_count": 0.0,
        "eval_group_count": 0.0,
        "successful_group_count": 0.0,
        "failed_group_count": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
    }


def normalize_task_metrics(raw_metrics: Mapping[str, Any] | None) -> Dict[str, float]:
    """Normalize evaluator outputs into the stable SLDBench 3D metric schema."""
    defaults = empty_task_metrics()
    if not isinstance(raw_metrics, Mapping):
        return defaults

    nmse = _coerce_non_negative_float(raw_metrics.get("nmse"), float("nan"))
    nmae = _coerce_non_negative_float(raw_metrics.get("nmae"), float("nan"))
    r2 = _coerce_finite_float(raw_metrics.get("r2"), float("nan"))
    if not (math.isfinite(nmse) and math.isfinite(nmae) and math.isfinite(r2)):
        return defaults

    score = 1.0 / (1.0 + nmse)
    return {
        "nmse": nmse,
        "nmae": nmae,
        "r2": r2,
        "fit_group_count": _coerce_non_negative_float(
            raw_metrics.get("fit_group_count"),
            defaults["fit_group_count"],
        ),
        "eval_group_count": _coerce_non_negative_float(
            raw_metrics.get("eval_group_count"),
            defaults["eval_group_count"],
        ),
        "successful_group_count": _coerce_non_negative_float(
            raw_metrics.get("successful_group_count"),
            defaults["successful_group_count"],
        ),
        "failed_group_count": _coerce_non_negative_float(
            raw_metrics.get("failed_group_count"),
            defaults["failed_group_count"],
        ),
        "score": score,
        "combined_score": score,
    }


def build_task_result(
    task: SLDBench3DTaskSpec,
    *,
    raw_metrics: Mapping[str, Any] | None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one stable per-task artifact entry."""
    metrics = normalize_task_metrics(raw_metrics)
    final_task_score = float(metrics["score"])
    metrics["combined_score"] = final_task_score
    return {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "spec": task.to_spec_dict(),
        "metrics": metrics,
        "case_score": final_task_score,
        "final_task_score": final_task_score,
        "error": error,
    }


def aggregate_task_results(task_results: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregate per-task SLDBench 3D results into one shared metric payload."""
    normalized_results = list(task_results)
    if not normalized_results:
        return {
            **empty_task_metrics(),
            "task_count": 0.0,
            "successful_task_count": 0.0,
            "failed_task_count": 0.0,
        }

    aggregate: Dict[str, float] = {}
    for key in SLDBENCH_3D_METRIC_KEYS:
        aggregate[key] = sum(
            float(result["metrics"][key]) for result in normalized_results
        ) / len(normalized_results)
    aggregate["task_count"] = float(len(normalized_results))
    aggregate["successful_task_count"] = float(
        sum(1 for result in normalized_results if not result.get("error"))
    )
    aggregate["failed_task_count"] = (
        float(len(normalized_results)) - aggregate["successful_task_count"]
    )
    aggregate["combined_score"] = float(aggregate["score"])
    return aggregate


def extract_task_result(artifacts: Mapping[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Extract one per-task artifact entry from stored evaluation artifacts."""
    task_results = artifacts.get("task_results")
    if not isinstance(task_results, list):
        return None

    for task_result in task_results:
        if not isinstance(task_result, Mapping):
            continue
        if task_result.get("task_id") != task_id:
            continue

        metrics = task_result.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        for key in SLDBENCH_3D_METRIC_KEYS:
            if key not in metrics:
                return None
            numeric = _coerce_finite_float(metrics.get(key), float("nan"))
            if not math.isfinite(numeric):
                return None

        task = SLDBENCH_3D_TASKS_BY_ID.get(task_id)
        if task is None:
            return dict(task_result)
        return build_task_result(
            task,
            raw_metrics=metrics,
            error=task_result.get("error"),
        )
    return None


def project_task_artifacts(
    artifacts: Mapping[str, Any],
    task_id: str,
    task_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project shared artifacts into the task-local view used by spawned checkpoints."""
    projected: Dict[str, Any] = {
        "task_selector": task_id,
        "selected_task_ids": [task_id],
        "evaluation_mode": "task_specific",
        "task_results": [dict(task_result)],
    }

    loader_mode = artifacts.get("loader_mode")
    if isinstance(loader_mode, str):
        projected["loader_mode"] = loader_mode

    task_summaries = artifacts.get("task_summaries")
    if isinstance(task_summaries, Mapping):
        task_summary = task_summaries.get(task_id)
        if isinstance(task_summary, Mapping):
            projected["task_summaries"] = {task_id: dict(task_summary)}

    compact_task_summary = artifacts.get("compact_task_summary")
    if isinstance(compact_task_summary, Mapping):
        task_summary = compact_task_summary.get(task_id)
        if isinstance(task_summary, Mapping):
            projected["compact_task_summary"] = {task_id: dict(task_summary)}

    return projected


__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "SLDBENCH_3D_METRIC_KEYS",
    "SLDBENCH_3D_SHARED_SELECTOR",
    "SLDBENCH_3D_TASK_SELECTOR_ENV_VAR",
    "SLDBENCH_3D_TASK_SPECS",
    "SLDBENCH_3D_TASKS_BY_ID",
    "SLDBENCH_DATASET_REPO_ID",
    "SLDBench3DTaskSpec",
    "aggregate_task_results",
    "build_generic_system_prompt_for_sldbench_3d",
    "build_task_result",
    "empty_task_metrics",
    "extract_task_result",
    "get_sldbench_3d_system_prompt",
    "normalize_task_metrics",
    "project_task_artifacts",
    "resolve_task_specs",
]
