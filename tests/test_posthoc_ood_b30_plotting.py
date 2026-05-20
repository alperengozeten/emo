import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "multi_task_shared_then_adapt"
    / "plotting"
    / "plot_posthoc_ood_b30_figures.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("posthoc_ood_b30_figures", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(
    run_dir: Path,
    *,
    family: str,
    ood_tasks: list[str],
    shared: list[float],
    adapt_rows: list[list[float]],
    baseline_rows: list[list[float]],
) -> None:
    source_task_ids = ["task_a", "task_b", "task_c", "task_d"]
    assert len(adapt_rows) == len(source_task_ids)
    assert len(baseline_rows) == len(source_task_ids)
    payload = {
        "algorithm": "posthoc_ood_evaluation",
        "family": family,
        "training_task_ids": source_task_ids,
        "ood_tasks": ood_tasks,
        "programs": {
            "shared_best": {
                "source_kind": "shared_best",
                "source_task_id": None,
                "ood_results": {
                    task_id: {"score": score}
                    for task_id, score in zip(ood_tasks, shared)
                },
            },
        },
    }
    for source_task_id, scores in zip(source_task_ids, adapt_rows):
        payload["programs"][f"adapted__{source_task_id}"] = {
            "source_kind": "adapted",
            "source_task_id": source_task_id,
            "ood_results": {
                task_id: {"score": score}
                for task_id, score in zip(ood_tasks, scores)
            },
        }
    for source_task_id, scores in zip(source_task_ids, baseline_rows):
        payload["programs"][f"baseline__{source_task_id}"] = {
            "source_kind": "baseline",
            "source_task_id": source_task_id,
            "ood_results": {
                task_id: {"score": score}
                for task_id, score in zip(ood_tasks, scores)
            },
        }
    output_dir = run_dir / "posthoc_ood_all_known"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ood_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_model_level_means_from_posthoc_and_build_b30_payloads(tmp_path):
    plotter = _load_module()
    results_dir = tmp_path / "results" / "heilbronn_triangle"
    ood_tasks = ["heil_tri_n8", "heil_tri_n13"]

    _write_summary(
        results_dir / "s20-a25-b30-model-a-full" / "run_01_seed_42",
        family="heilbronn_triangle",
        ood_tasks=ood_tasks,
        shared=[0.4, 0.5],
        adapt_rows=[[0.6, 0.7], [0.8, 0.9], [0.6, 0.7], [0.8, 0.9]],
        baseline_rows=[[0.2, 0.3], [0.4, 0.5], [0.2, 0.3], [0.4, 0.5]],
    )
    _write_summary(
        results_dir / "s20-a25-b30-model-b-full" / "run_01_seed_42",
        family="heilbronn_triangle",
        ood_tasks=ood_tasks,
        shared=[0.2, 0.3],
        adapt_rows=[[0.4, 0.5], [0.6, 0.7], [0.4, 0.5], [0.6, 0.7]],
        baseline_rows=[[0.1, 0.2], [0.3, 0.4], [0.1, 0.2], [0.3, 0.4]],
    )
    _write_summary(
        results_dir / "s60-a15-b30-model-a-full" / "run_01_seed_42",
        family="heilbronn_triangle",
        ood_tasks=ood_tasks,
        shared=[0.5, 0.6],
        adapt_rows=[[0.7, 0.8], [0.9, 1.0], [0.7, 0.8], [0.9, 1.0]],
        baseline_rows=[[0.3, 0.4], [0.5, 0.6], [0.3, 0.4], [0.5, 0.6]],
    )
    _write_summary(
        results_dir / "s40-a15-b25-model-a-full" / "run_01_seed_42",
        family="heilbronn_triangle",
        ood_tasks=ood_tasks,
        shared=[0.9, 0.9],
        adapt_rows=[[0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]],
        baseline_rows=[[0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]],
    )

    prefixes = plotter.discover_setting_prefixes(results_dir, baseline_budget=30)
    assert prefixes == ["s20-a25-b30", "s60-a15-b30"]

    selected = plotter.collect_model_level_means_from_posthoc(
        results_dir=results_dir,
        setting_prefix="s20-a25-b30",
    )
    assert [item["label"] for item in selected["categories"]] == ["N = 8", "N = 13", "Average"]
    assert selected["series"]["shared"] == pytest.approx([0.3, 0.4, 0.35])
    assert selected["series"]["adapt"] == pytest.approx([0.6, 0.7, 0.65])
    assert selected["series"]["baseline"] == pytest.approx([0.25, 0.35, 0.30])

    budgets = plotter.collect_budget_payloads(
        results_dir=results_dir,
        setting_prefixes=prefixes,
    )
    assert [item["setting_prefix"] for item in budgets] == ["s20-a25-b30", "s60-a15-b30"]
    by_holdout = plotter.build_by_holdout_payload(
        budgets,
        baseline_reference=budgets[1],
    )

    assert by_holdout["baseline_reference"]["source_setting_prefix"] == "s60-a15-b30"
    assert by_holdout["baseline_reference"]["values"] == pytest.approx([0.4, 0.5, 0.45])
    assert [item["budget"]["label"] for item in by_holdout["adapt_by_budget"]] == [
        "20 / 25 / 120",
        "60 / 15 / 120",
    ]
    assert by_holdout["adapt_by_budget"][0]["values"] == pytest.approx([0.6, 0.7, 0.65])
    assert by_holdout["adapt_by_budget"][1]["values"] == pytest.approx([0.8, 0.9, 0.85])
