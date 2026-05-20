#!/usr/bin/env python3
"""Create b30 OOD-by-holdout figures with four adaptation methods."""

from __future__ import annotations

import argparse
import json
import re
import signal
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

PLOT_DIR = Path(__file__).resolve().parent
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plot_budget_labels import build_budget_triplet, build_single_task_budget_triplet

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.holdout_eval import (
    evaluate_source_programs_on_holdouts,
    resolve_best_program_path,
    resolve_holdout_task_specs,
)
from openevolve.multi_task_shared_then_adapt.posthoc_ood import (
    SUPPORTED_OOD_FAMILIES,
    evaluate_program_on_ood_tasks,
)
from openevolve.multi_task_shared_then_adapt.workflow import (
    family_task_specs,
    load_manifest,
)


SETTING_RE = re.compile(
    r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-(?P<model>.+)-full$"
)
N_SUFFIX_RE = re.compile(r"_n(?P<n>\d+)$")

MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
]

BUDGET_COLORS = ["#CBE3D2", "#A9D8C8", "#7CC7A8", "#4EA685"]
SINGLE_TASK_COLOR = "#F6C8B8"
SHARED_COLOR = "#A8C7E8"
METHOD_COLORS_SINGLE_BUDGET = {
    "shared": SHARED_COLOR,
    "baseline": SINGLE_TASK_COLOR,
    "adapt": "#A9D8C8",
    "best_local": "#A9D8C8",
    "best_shared": "#4EA685",
}
SINGLE_BUDGET_Y_LIMITS = {
    "circle_packing": (0.8, 1.0),
    "circle_packing_rectangle": (0.75, 1.0),
    "heilbronn_triangle": (0.2, 0.8),
}
SINGLE_BUDGET_Y_TICK_SPACING = {
    "heilbronn_triangle": 0.1,
}
MULTI_BUDGET_Y_LIMITS = {
    "circle_packing": (0.8, 1.0),
    "circle_packing_rectangle": (0.75, 1.0),
    "heilbronn_triangle": (0.3, 0.75),
}
MULTI_BUDGET_Y_TICK_SPACING = {
    "heilbronn_triangle": 0.05,
}

BASE_METHOD_SPECS = [
    ("baseline", "Single-task", ""),
    ("adapt", "STA Warmstart", "//"),
    ("best_local", "STA Best-Local Program", ""),
    ("best_shared", "STA Best-Shared Program", "xx"),
]
SHARED_METHOD_SPEC = ("shared", "STA-Shared", "")
LEGEND_HATCH_OVERRIDES = {
    "adapt": "//",
    "best_shared": "xx",
}


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    manifest: str
    results_dir: str
    output_stem: str
    evaluation_kind: str


FAMILY_CONFIGS = {
    "circle_packing": FamilyConfig(
        family="circle_packing",
        manifest="multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml",
        results_dir="multi_task_shared_then_adapt/results/circle_packing",
        output_stem=(
            "multi_task_shared_then_adapt/figures/"
            "circle_packing_ood_b30_by_holdout_adaptation_methods"
        ),
        evaluation_kind="circle_packing",
    ),
    "circle_packing_rectangle": FamilyConfig(
        family="circle_packing_rectangle",
        manifest="multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml",
        results_dir="multi_task_shared_then_adapt/results/circle_packing_rectangle",
        output_stem=(
            "multi_task_shared_then_adapt/figures/"
            "circle_packing_rectangle_ood_b30_by_holdout_adaptation_methods"
        ),
        evaluation_kind="posthoc",
    ),
    "heilbronn_triangle": FamilyConfig(
        family="heilbronn_triangle",
        manifest="multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml",
        results_dir="multi_task_shared_then_adapt/results/heilbronn_triangle",
        output_stem=(
            "multi_task_shared_then_adapt/figures/"
            "heilbronn_triangle_ood_b30_by_holdout_adaptation_methods"
        ),
        evaluation_kind="posthoc",
    ),
}


class RunEvaluationTimeout(RuntimeError):
    """Raised when one run-level OOD evaluation exceeds the timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-b30 OOD-by-holdout figures that compare single-task, "
            "warmstart, best-local, and best-shared adaptation."
        )
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        choices=sorted(FAMILY_CONFIGS),
        help=(
            "Family to plot. May be passed multiple times. If omitted, plots "
            "circle_packing, circle_packing_rectangle, and heilbronn_triangle."
        ),
    )
    parser.add_argument(
        "--setting-prefix",
        action="append",
        dest="setting_prefixes",
        help=(
            "Exact setting prefix to include, e.g. s60-a15-b30. May be passed "
            "multiple times. If omitted, all settings matching --baseline-budget "
            "are included."
        ),
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=30,
        help="Only include settings with this baseline budget. Default: 30.",
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Include the persisted shared-program OOD series in the figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI for PNG output. Default: 300.",
    )
    parser.add_argument(
        "--single-budget-unified-sts-color",
        action="store_true",
        help=(
            "For single-budget figures only, render STA Warmstart, "
            "STA Best-Local Program, and STA Best-Shared Program with the "
            "same fill color as STA Best-Local Program."
        ),
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Do not draw legends in the generated figures.",
    )
    parser.add_argument(
        "--output-stem-suffix",
        default="",
        help=(
            "Optional suffix appended to each family output stem before the file "
            "extension, e.g. _s60_a15_b30_shared."
        ),
    )
    parser.add_argument(
        "--baseline-reference-prefix",
        default="s60-a15-b30",
        help=(
            "Setting prefix whose single-task OOD results are used as the one-time "
            "baseline series in multi-budget figures. Default: s60-a15-b30."
        ),
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=30.0,
        help=(
            "Skip a run if its branch OOD evaluation exceeds this timeout. "
            "Default: 30 seconds."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_setting_prefix(prefix: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)", prefix)
    if match is None:
        raise ValueError(f"Invalid setting prefix: {prefix}")
    return (
        int(match.group("shared")),
        int(match.group("adapt")),
        int(match.group("baseline")),
    )


def build_method_specs(include_shared: bool) -> list[tuple[str, str, str]]:
    if include_shared:
        return [SHARED_METHOD_SPEC, *BASE_METHOD_SPECS]
    return list(BASE_METHOD_SPECS)


def legend_hatch_for_method(method_key: str, bar_hatch: str) -> str:
    return LEGEND_HATCH_OVERRIDES.get(method_key, bar_hatch)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15.5,
            "legend.title_fontsize": 15.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "hatch.linewidth": 0.8,
        }
    )


def bold_axis_tick_labels(ax: plt.Axes) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def bold_legend(legend: Any) -> None:
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_title().set_fontweight("bold")


def parse_setting_dir_name(name: str) -> tuple[tuple[int, int, int], str] | None:
    match = SETTING_RE.fullmatch(name)
    if match is None:
        return None
    budget = (
        int(match.group("shared")),
        int(match.group("adapt")),
        int(match.group("baseline")),
    )
    return budget, match.group("model")


def sort_task_ids(task_ids: Sequence[str]) -> list[str]:
    def sort_key(task_id: str) -> tuple[int, str]:
        match = N_SUFFIX_RE.search(task_id)
        if match:
            return (int(match.group("n")), task_id)
        return (10**9, task_id)

    return sorted(task_ids, key=sort_key)


def task_label(task_id: str) -> str:
    match = N_SUFFIX_RE.search(task_id)
    if match:
        return f"N = {int(match.group('n'))}"
    return task_id


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise RunEvaluationTimeout("run-level OOD evaluation timed out")


def run_with_timeout(timeout_seconds: float | None, callback):
    if timeout_seconds is None or timeout_seconds <= 0:
        return callback()

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def extract_circle_packing_source_results(
    results_by_source_task: Mapping[str, Any],
) -> dict[str, dict[str, float]] | None:
    extracted: dict[str, dict[str, float]] = {}
    for source_task_id, payload in results_by_source_task.items():
        if not isinstance(payload, Mapping) or not payload.get("available", False):
            return None
        holdout_results = payload.get("holdout_task_results")
        if not isinstance(holdout_results, Mapping):
            return None
        source_scores: dict[str, float] = {}
        for holdout_id, task_result in holdout_results.items():
            if not isinstance(task_result, Mapping):
                return None
            score = task_result.get("final_task_score")
            if not is_number(score):
                return None
            source_scores[str(holdout_id)] = float(score)
        extracted[str(source_task_id)] = source_scores
    return extracted


def extract_circle_packing_shared_series(
    holdout_summary: Mapping[str, Any],
    *,
    category_ids: Sequence[str],
) -> list[float] | None:
    payload = holdout_summary.get("shared_zero_shot")
    if not isinstance(payload, Mapping) or not payload.get("available", False):
        return None
    holdout_results = payload.get("holdout_task_results")
    if not isinstance(holdout_results, Mapping):
        return None

    values: list[float] = []
    for category_id in category_ids:
        result = holdout_results.get(category_id)
        if not isinstance(result, Mapping):
            return None
        score = result.get("final_task_score")
        if not is_number(score):
            return None
        values.append(float(score))
    values.append(statistics.fmean(values))
    return values


def extract_posthoc_source_results(
    programs: Mapping[str, Any],
    *,
    source_kind: str,
) -> dict[str, dict[str, float]] | None:
    extracted: dict[str, dict[str, float]] = {}
    for payload in programs.values():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("source_kind") != source_kind:
            continue
        source_task_id = payload.get("source_task_id")
        if not isinstance(source_task_id, str) or not source_task_id:
            return None
        ood_results = payload.get("ood_results")
        if not isinstance(ood_results, Mapping):
            return None
        scores: dict[str, float] = {}
        for ood_task_id, result in ood_results.items():
            if not isinstance(result, Mapping):
                return None
            score = result.get("score")
            if not is_number(score):
                return None
            scores[str(ood_task_id)] = float(score)
        extracted[source_task_id] = scores
    return extracted or None


def extract_posthoc_shared_series(
    programs: Mapping[str, Any],
    *,
    category_ids: Sequence[str],
) -> list[float] | None:
    shared_payloads = [
        payload
        for payload in programs.values()
        if isinstance(payload, Mapping) and payload.get("source_kind") == "shared_best"
    ]
    if len(shared_payloads) != 1:
        return None
    ood_results = shared_payloads[0].get("ood_results")
    if not isinstance(ood_results, Mapping):
        return None

    values: list[float] = []
    for category_id in category_ids:
        result = ood_results.get(category_id)
        if not isinstance(result, Mapping):
            return None
        score = result.get("score")
        if not is_number(score):
            return None
        values.append(float(score))
    values.append(statistics.fmean(values))
    return values


def summarize_method_by_category(
    *,
    source_results: Mapping[str, Mapping[str, float]],
    source_task_ids: Sequence[str],
    category_ids: Sequence[str],
) -> list[float] | None:
    if set(source_results) != set(source_task_ids):
        return None
    category_means: list[float] = []
    for category_id in category_ids:
        values: list[float] = []
        for source_task_id in source_task_ids:
            task_results = source_results.get(source_task_id)
            if task_results is None:
                return None
            value = task_results.get(category_id)
            if not is_number(value):
                return None
            values.append(float(value))
        category_means.append(statistics.fmean(values))
    category_means.append(statistics.fmean(category_means))
    return category_means


def persisted_circle_packing_seed_results(
    *,
    holdout_summary: Mapping[str, Any],
    key: str,
    source_task_ids: Sequence[str],
    category_ids: Sequence[str],
) -> dict[str, dict[str, float]] | None:
    payload = holdout_summary.get(key)
    if not isinstance(payload, Mapping):
        return None
    extracted = extract_circle_packing_source_results(payload)
    if extracted is None:
        return None
    if (
        summarize_method_by_category(
            source_results=extracted,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        )
        is None
    ):
        return None
    return extracted


def persisted_posthoc_seed_results(
    *,
    programs: Mapping[str, Any],
    source_kind: str,
    source_task_ids: Sequence[str],
    category_ids: Sequence[str],
) -> dict[str, dict[str, float]] | None:
    extracted = extract_posthoc_source_results(programs, source_kind=source_kind)
    if extracted is None:
        return None
    if (
        summarize_method_by_category(
            source_results=extracted,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        )
        is None
    ):
        return None
    return extracted


def evaluate_circle_packing_seed_branch(
    *,
    run_root: Path,
    branch_root_name: str,
    source_task_ids: Sequence[str],
    holdout_task_specs: Sequence[Any],
    evaluation_file: Path,
) -> dict[str, dict[str, float]] | None:
    branch_root = run_root / branch_root_name
    if not branch_root.is_dir():
        return None
    program_paths: dict[str, str | Path | None] = {}
    for source_task_id in source_task_ids:
        program_paths[source_task_id] = resolve_best_program_path(
            branch_root / source_task_id,
            checkpoint_layout=False,
        )
    results = evaluate_source_programs_on_holdouts(
        program_paths=program_paths,
        family="circle_packing",
        holdout_task_specs=holdout_task_specs,
        evaluation_file=evaluation_file,
    )
    return extract_circle_packing_source_results(results)


def evaluate_posthoc_seed_branch(
    *,
    run_root: Path,
    branch_root_name: str,
    family: str,
    source_task_ids: Sequence[str],
    ood_task_specs: Sequence[Any],
    evaluation_file: Path,
) -> dict[str, dict[str, float]] | None:
    branch_root = run_root / branch_root_name
    if not branch_root.is_dir():
        return None

    extracted: dict[str, dict[str, float]] = {}
    for source_task_id in source_task_ids:
        program_path = resolve_best_program_path(
            branch_root / source_task_id,
            checkpoint_layout=False,
        )
        if program_path is None:
            return None
        results = evaluate_program_on_ood_tasks(
            program_path=program_path,
            family=family,
            ood_task_specs=ood_task_specs,
            evaluation_file=evaluation_file,
        )
        source_scores: dict[str, float] = {}
        for ood_task_id, payload in results.items():
            score = payload.get("score")
            if not is_number(score):
                return None
            source_scores[str(ood_task_id)] = float(score)
        extracted[source_task_id] = source_scores
    return extracted


def collect_circle_packing_run_payload(
    *,
    run_root: Path,
    source_task_ids: Sequence[str],
    category_ids: Sequence[str],
    holdout_task_specs: Sequence[Any],
    evaluation_file: Path,
    include_shared: bool,
) -> dict[str, list[float]] | None:
    holdout_summary_path = run_root / "holdout_evaluation" / "holdout_summary.json"
    if holdout_summary_path.is_file():
        holdout_summary = load_json(holdout_summary_path)
        holdout_enabled = bool(holdout_summary.get("enabled"))
    else:
        comparison_summary = load_json(run_root / "comparison_summary.json")
        holdout_summary = comparison_summary.get("holdout_evaluation")
        holdout_enabled = isinstance(holdout_summary, Mapping) and bool(
            holdout_summary.get("enabled")
        )
    if not isinstance(holdout_summary, Mapping) or not holdout_enabled:
        return None

    warmstart_results = holdout_summary.get("adaptation_by_source_task")
    if not isinstance(warmstart_results, Mapping):
        return None
    warmstart_sources = extract_circle_packing_source_results(warmstart_results)
    if warmstart_sources is None:
        return None

    baseline_results = holdout_summary.get("baseline_by_source_task")
    if not isinstance(baseline_results, Mapping):
        return None
    baseline_sources = extract_circle_packing_source_results(baseline_results)
    if baseline_sources is None:
        return None

    best_shared_sources = persisted_circle_packing_seed_results(
        holdout_summary=holdout_summary,
        key="best_shared_adaptation_by_source_task",
        source_task_ids=source_task_ids,
        category_ids=category_ids,
    )
    if best_shared_sources is None:
        return None

    best_local_sources = persisted_circle_packing_seed_results(
        holdout_summary=holdout_summary,
        key="best_local_adaptation_by_source_task",
        source_task_ids=source_task_ids,
        category_ids=category_ids,
    )
    if best_local_sources is None:
        return None

    methods = {
        "baseline": summarize_method_by_category(
            source_results=baseline_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
        "adapt": summarize_method_by_category(
            source_results=warmstart_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
        "best_shared": summarize_method_by_category(
            source_results=best_shared_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
        "best_local": summarize_method_by_category(
            source_results=best_local_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
    }
    if include_shared:
        methods["shared"] = extract_circle_packing_shared_series(
            holdout_summary,
            category_ids=category_ids,
        )
    if any(series is None for series in methods.values()):
        return None
    return {key: value for key, value in methods.items() if value is not None}


def collect_posthoc_run_payload(
    *,
    family: str,
    run_root: Path,
    source_task_ids: Sequence[str],
    category_ids: Sequence[str],
    ood_task_specs: Sequence[Any],
    evaluation_file: Path,
    include_shared: bool,
) -> dict[str, list[float]] | None:
    summary_path = run_root / "posthoc_ood_all_known" / "ood_summary.json"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    programs = summary.get("programs")
    if not isinstance(programs, Mapping):
        return None

    warmstart_sources = extract_posthoc_source_results(programs, source_kind="adapted")
    if warmstart_sources is None:
        return None

    baseline_sources = extract_posthoc_source_results(programs, source_kind="baseline")
    if baseline_sources is None:
        return None

    best_shared_sources = persisted_posthoc_seed_results(
        programs=programs,
        source_kind="best_shared",
        source_task_ids=source_task_ids,
        category_ids=category_ids,
    )
    if best_shared_sources is None:
        return None

    best_local_sources = persisted_posthoc_seed_results(
        programs=programs,
        source_kind="best_local",
        source_task_ids=source_task_ids,
        category_ids=category_ids,
    )
    if best_local_sources is None:
        return None

    methods = {
        "baseline": summarize_method_by_category(
            source_results=baseline_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
        "adapt": summarize_method_by_category(
            source_results=warmstart_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
        "best_shared": summarize_method_by_category(
            source_results=best_shared_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
        "best_local": summarize_method_by_category(
            source_results=best_local_sources,
            source_task_ids=source_task_ids,
            category_ids=category_ids,
        ),
    }
    if include_shared:
        methods["shared"] = extract_posthoc_shared_series(
            programs,
            category_ids=category_ids,
        )
    if any(series is None for series in methods.values()):
        return None
    return {key: value for key, value in methods.items() if value is not None}


def collect_family_payload(
    *,
    config: FamilyConfig,
    baseline_budget: int,
    baseline_reference_prefix: str,
    run_timeout_seconds: float | None,
    setting_prefixes: set[tuple[int, int, int]] | None,
    include_shared: bool,
) -> dict[str, Any]:
    manifest_path = resolve_repo_path(config.manifest)
    manifest = load_manifest(manifest_path)
    results_dir = resolve_repo_path(config.results_dir)
    source_task_ids = [task.task_id for task in family_task_specs(manifest)]
    task_count = len(source_task_ids)

    if config.evaluation_kind == "circle_packing":
        holdout_task_specs = resolve_holdout_task_specs("all")
        category_ids = [task.task_id for task in holdout_task_specs]
        evaluation_context: dict[str, Any] = {
            "holdout_task_specs": holdout_task_specs,
        }
    else:
        family_support = SUPPORTED_OOD_FAMILIES[config.family]
        ood_task_specs = family_support.resolve_ood_task_specs(None)
        category_ids = [task.task_id for task in ood_task_specs]
        evaluation_context = {"ood_task_specs": ood_task_specs}

    method_specs = build_method_specs(include_shared)
    per_budget: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}

    for setting_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        parsed = parse_setting_dir_name(setting_dir.name)
        if parsed is None:
            continue
        budget, model_id = parsed
        if budget[2] != baseline_budget:
            continue
        if setting_prefixes is not None and budget not in setting_prefixes:
            continue

        comparable_runs: list[dict[str, Any]] = []
        for run_root in sorted(
            path for path in setting_dir.iterdir() if path.is_dir() and path.name.startswith("run_")
        ):
            comparison_path = run_root / "comparison_summary.json"
            if not comparison_path.is_file():
                continue
            print(
                f"[{config.family}] evaluating {setting_dir.name}/{run_root.name}",
                flush=True,
            )
            try:
                if config.evaluation_kind == "circle_packing":
                    run_payload = run_with_timeout(
                        run_timeout_seconds,
                        lambda: collect_circle_packing_run_payload(
                            run_root=run_root,
                            source_task_ids=source_task_ids,
                            category_ids=category_ids,
                            holdout_task_specs=evaluation_context["holdout_task_specs"],
                            evaluation_file=manifest.evaluation_file,
                            include_shared=include_shared,
                        ),
                    )
                else:
                    run_payload = run_with_timeout(
                        run_timeout_seconds,
                        lambda: collect_posthoc_run_payload(
                            family=config.family,
                            run_root=run_root,
                            source_task_ids=source_task_ids,
                            category_ids=category_ids,
                            ood_task_specs=evaluation_context["ood_task_specs"],
                            evaluation_file=manifest.evaluation_file,
                            include_shared=include_shared,
                        ),
                    )
            except RunEvaluationTimeout:
                print(
                    f"[{config.family}] skipping {setting_dir.name}/{run_root.name}: "
                    f"timed out after {run_timeout_seconds:.1f}s",
                    flush=True,
                )
                continue
            if run_payload is None:
                print(
                    f"[{config.family}] skipping {setting_dir.name}/{run_root.name}: "
                    "incomplete comparable methods",
                    flush=True,
                )
                continue
            print(
                f"[{config.family}] keeping {setting_dir.name}/{run_root.name}",
                flush=True,
            )
            comparable_runs.append(
                {
                    "run_name": run_root.name,
                    "methods": run_payload,
                }
            )

        if not comparable_runs:
            continue

        method_means = {
            method_key: [
                statistics.fmean(
                    run_payload["methods"][method_key][idx]
                    for run_payload in comparable_runs
                )
                for idx in range(len(category_ids) + 1)
            ]
            for method_key, _label, _hatch in method_specs
        }

        per_budget.setdefault(budget, {})[model_id] = {
            "run_count": len(comparable_runs),
            "runs": [payload["run_name"] for payload in comparable_runs],
            "series": method_means,
        }

    budgets_payload: list[dict[str, Any]] = []
    for budget in sorted(per_budget):
        model_payloads = per_budget[budget]
        ordered_models = [model for model in MODEL_ORDER if model in model_payloads]
        if not ordered_models:
            continue

        budget_series = {
            method_key: [
                statistics.fmean(
                    model_payloads[model]["series"][method_key][idx]
                    for model in ordered_models
                )
                for idx in range(len(category_ids) + 1)
            ]
            for method_key, _label, _hatch in method_specs
        }

        budgets_payload.append(
            {
                "setting_prefix": f"s{budget[0]}-a{budget[1]}-b{budget[2]}",
                "budget": build_budget_triplet(
                    shared=budget[0],
                    adapt=budget[1],
                    baseline=budget[2],
                    task_count=task_count,
                ),
                "model_count": len(ordered_models),
                "models": {
                    model: model_payloads[model]
                    for model in ordered_models
                },
                "series": budget_series,
            }
        )

    if not budgets_payload:
        raise SystemExit(
            f"No comparable adaptation OOD runs found for family='{config.family}' "
            f"with baseline budget {baseline_budget}."
        )

    baseline_reference = next(
        (
            budget_payload
            for budget_payload in budgets_payload
            if budget_payload["setting_prefix"] == baseline_reference_prefix
        ),
        None,
    )
    if baseline_reference is None:
        if len(budgets_payload) == 1:
            baseline_reference = budgets_payload[0]
        else:
            raise SystemExit(
                f"Baseline reference setting '{baseline_reference_prefix}' was not found "
                f"among comparable OOD runs for family='{config.family}'."
            )

    return {
        "family": config.family,
        "manifest": config.manifest,
        "results_dir": config.results_dir,
        "baseline_budget": baseline_budget,
        "baseline_reference_prefix": baseline_reference["setting_prefix"],
        "task_count": task_count,
        "categories": [
            {"id": category_id, "label": task_label(category_id)}
            for category_id in category_ids
        ]
        + [{"id": "average", "label": "Average"}],
        "methods": [
            {
                "id": method_key,
                "label": label,
                "hatch": hatch,
                "legend_hatch": legend_hatch_for_method(method_key, hatch),
            }
            for method_key, label, hatch in method_specs
        ],
        "baseline_reference": {
            "setting_prefix": baseline_reference["setting_prefix"],
            "budget": baseline_reference["budget"],
            "label": build_single_task_budget_triplet(
                baseline=int(baseline_reference["budget"]["baseline"]),
                task_count=int(baseline_reference["budget"]["task_count"]),
            )["label"],
            "values": list(baseline_reference["series"]["baseline"]),
        },
        "budgets": budgets_payload,
        "aggregation": (
            "Within a run, each method averages OOD scores across source tasks for "
            "each holdout N, except Shared, which uses the single persisted shared "
            "program directly. The figure keeps only runs where all requested methods "
            "have complete OOD results. It then averages comparable runs within each "
            "model and averages the resulting model means within each budget."
        ),
    }


def plot_family_payload(
    payload: dict[str, Any],
    *,
    output_stem: Path,
    dpi: int,
    single_budget_unified_sts_color: bool,
    hide_legend: bool,
) -> None:
    apply_plot_style()

    categories = payload["categories"]
    budgets = payload["budgets"]
    method_specs = [
        (
            item["id"],
            item["label"],
            item["hatch"],
            item.get("legend_hatch", item["hatch"]),
        )
        for item in payload["methods"]
    ]
    labels = [category["label"] for category in categories]
    x = np.arange(len(categories), dtype=float)
    family = payload["family"]

    n_budgets = len(budgets)
    n_methods = len(method_specs)
    cluster_width = 0.82
    per_budget_width = cluster_width / max(n_budgets, 1)
    bar_width = min(0.22, per_budget_width / (n_methods + 0.45))

    fig_width = max(8.6, 1.9 * len(categories) + 2.2 * n_budgets)
    fig_height = 6.6 if n_budgets > 1 else 6.35
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if n_budgets == 1:
        budget_payload = budgets[0]
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width
        for method_idx, (method_key, _label, hatch, _legend_hatch) in enumerate(method_specs):
            xs = x + offsets[method_idx]
            ys = budget_payload["series"][method_key]
            facecolor = METHOD_COLORS_SINGLE_BUDGET.get(method_key, "#D9D9D9")
            if (
                single_budget_unified_sts_color
                and method_key in {"adapt", "best_local", "best_shared"}
            ):
                facecolor = METHOD_COLORS_SINGLE_BUDGET["best_local"]
            ax.bar(
                xs,
                ys,
                bar_width,
                color=facecolor,
                edgecolor="#000000",
                linewidth=0.8,
                hatch=hatch,
            )

        ax.set_xlabel("Held-Out Task Size (N)", fontweight="bold")
        ax.set_ylabel(
            "Mean OOD Normalized Score Across LLMs",
            fontweight="bold",
            fontsize=16.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(*SINGLE_BUDGET_Y_LIMITS.get(family, (0.0, 1.06)))
        ax.yaxis.set_major_locator(
            MultipleLocator(SINGLE_BUDGET_Y_TICK_SPACING.get(family, 0.05))
        )
        bold_axis_tick_labels(ax)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)
        if not hide_legend:
            legend = ax.legend(
                handles=[
                    Patch(
                        facecolor=(
                            METHOD_COLORS_SINGLE_BUDGET["best_local"]
                            if (
                                single_budget_unified_sts_color
                                and method_key
                                in {"adapt", "best_local", "best_shared"}
                            )
                            else METHOD_COLORS_SINGLE_BUDGET.get(method_key, "#D9D9D9")
                        ),
                        edgecolor="#000000",
                        hatch=legend_hatch,
                        label=label,
                    )
                    for method_key, label, _hatch, legend_hatch in method_specs
                ],
                loc="upper left",
                bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0.0,
                ncol=min(3, max(1, len(method_specs))),
            )
            bold_legend(legend)

        plt.tight_layout()
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf", ".svg"):
            save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.10}
            if suffix == ".png":
                save_kwargs["dpi"] = dpi
            fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
        plt.close(fig)
        return

    baseline_reference = payload["baseline_reference"]
    nonbaseline_method_specs = [
        (method_key, label, hatch, legend_hatch)
        for method_key, label, hatch, legend_hatch in method_specs
        if method_key != "baseline"
    ]
    total_slots = 1 + n_budgets * len(nonbaseline_method_specs)
    bar_width = min(0.12, cluster_width / (total_slots + 0.65))
    slot_offsets = (
        np.arange(total_slots, dtype=float) - (total_slots - 1) / 2.0
    ) * bar_width

    ax.bar(
        x + slot_offsets[0],
        baseline_reference["values"],
        bar_width,
        color=SINGLE_TASK_COLOR,
        edgecolor="#000000",
        linewidth=0.8,
    )

    slot_idx = 1
    for budget_idx, budget_payload in enumerate(budgets):
        color = BUDGET_COLORS[budget_idx % len(BUDGET_COLORS)]

        for method_key, _label, hatch, _legend_hatch in nonbaseline_method_specs:
            xs = x + slot_offsets[slot_idx]
            ys = budget_payload["series"][method_key]
            ax.bar(
                xs,
                ys,
                bar_width,
                color=color,
                edgecolor="#000000",
                linewidth=0.8,
                hatch=hatch,
            )
            slot_idx += 1

    ax.set_xlabel("Held-Out Task Size (N)", fontweight="bold")
    ax.set_ylabel(
        "Mean OOD Normalized Score Across LLMs",
        fontweight="bold",
        fontsize=16.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*MULTI_BUDGET_Y_LIMITS.get(family, (0.7, 1.0)))
    ax.yaxis.set_major_locator(
        MultipleLocator(MULTI_BUDGET_Y_TICK_SPACING.get(family, 0.05))
    )
    bold_axis_tick_labels(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    budget_handles = [
        Patch(
            facecolor=SINGLE_TASK_COLOR,
            edgecolor="#000000",
            label="Single-task",
        )
    ]
    for budget_idx, budget_payload in enumerate(budgets):
        label = budget_payload["budget"]["label"]
        if budget_payload["model_count"] < 5:
            label = f"{label} (n={budget_payload['model_count']})"
        budget_handles.append(
            Patch(
                facecolor=BUDGET_COLORS[budget_idx % len(BUDGET_COLORS)],
                edgecolor="#000000",
                label=label,
            )
        )
    method_handles = [
        Patch(
            facecolor="#D9D9D9",
            edgecolor="#000000",
            hatch=legend_hatch,
            label=label,
        )
        for method_key, label, _hatch, legend_hatch in method_specs
        if method_key != "baseline"
    ]

    if not hide_legend:
        budget_legend = ax.legend(
            handles=budget_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.05),
            borderaxespad=0.0,
            ncol=min(3, max(1, len(budget_handles))),
        )
        bold_legend(budget_legend)
        ax.add_artist(budget_legend)
        method_legend = ax.legend(
            handles=method_handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.05),
            borderaxespad=0.0,
            ncol=1,
        )
        bold_legend(method_legend)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.10}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_stem.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def write_sidecar_json(output_stem: Path, payload: dict[str, Any]) -> None:
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    families = args.families or [
        "circle_packing",
        "circle_packing_rectangle",
        "heilbronn_triangle",
    ]
    setting_prefixes = (
        {parse_setting_prefix(prefix) for prefix in args.setting_prefixes}
        if args.setting_prefixes
        else None
    )

    for family in families:
        config = FAMILY_CONFIGS[family]
        print(f"[{family}] collecting comparable OOD adaptation runs...")
        payload = collect_family_payload(
            config=config,
            baseline_budget=args.baseline_budget,
            baseline_reference_prefix=args.baseline_reference_prefix,
            run_timeout_seconds=args.run_timeout_seconds,
            setting_prefixes=setting_prefixes,
            include_shared=args.include_shared,
        )
        output_stem = resolve_repo_path(f"{config.output_stem}{args.output_stem_suffix}")
        plot_family_payload(
            payload,
            output_stem=output_stem,
            dpi=args.dpi,
            single_budget_unified_sts_color=args.single_budget_unified_sts_color,
            hide_legend=args.hide_legend,
        )
        write_sidecar_json(output_stem, payload)
        print(f"[{family}] wrote {output_stem.with_suffix('.png')}")
        print(f"[{family}] wrote {output_stem.with_suffix('.pdf')}")
        print(f"[{family}] wrote {output_stem.with_suffix('.svg')}")
        print(f"[{family}] wrote {output_stem.with_suffix('.json')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
