import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.rust_adaptive_sort import (
    RUST_ADAPTIVE_SORT_SHARED_SELECTOR,
    RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR,
    RUST_ADAPTIVE_SORT_TASK_SPECS,
    RUST_ADAPTIVE_SORT_TASKS_BY_ID,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    resolve_task_specs,
)
from openevolve.multi_task_shared_then_adapt.spawn import (
    _program_file_suffix,
    spawn_task_checkpoints,
)


CARGO_AVAILABLE = shutil.which("cargo") is not None


def _load_rust_evaluator_module():
    evaluator_path = REPO_ROOT / "examples" / "rust_adaptive_sort_emo_sta" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("rust_adaptive_sort_emo_sta_eval_test", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rust_evaluator = _load_rust_evaluator_module()


def _sample_task_payload(task_id: str) -> dict[str, object]:
    task = RUST_ADAPTIVE_SORT_TASKS_BY_ID[task_id]
    datasets = [
        {
            "label": f"{task.pattern}_size1000_seed0",
            "size": 1000,
            "seed": 0 if task.seeds else None,
            "candidate_median_time": 0.002,
            "reference_median_time": 0.0015,
            "speedup_ratio": 0.75,
            "sorted_correctly": True,
        },
        {
            "label": f"{task.pattern}_size10000_seed1",
            "size": 10000,
            "seed": 1 if task.seeds else None,
            "candidate_median_time": 0.020,
            "reference_median_time": 0.018,
            "speedup_ratio": 0.9,
            "sorted_correctly": True,
        },
    ]
    return {
        "task_id": task.task_id,
        "display_name": task.display_name,
        "pattern": task.pattern,
        "datasets": datasets,
    }


def test_family_registry_resolves_rust_adaptive_sort():
    family = get_family_definition("rust_adaptive_sort")
    assert family.family == "rust_adaptive_sort"
    assert family.task_selector_env_var == RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR
    assert family.shared_selector == RUST_ADAPTIVE_SORT_SHARED_SELECTOR
    assert [task.task_id for task in family.task_specs] == [
        "ras_random",
        "ras_nearly_sorted",
        "ras_reverse_sorted",
        "ras_duplicates",
    ]


def test_resolve_task_specs_all_returns_expected_tasks():
    resolved = resolve_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "ras_random",
        "ras_nearly_sorted",
        "ras_reverse_sorted",
        "ras_duplicates",
    ]


def test_shared_and_task_artifact_extraction_round_trip():
    task = RUST_ADAPTIVE_SORT_TASKS_BY_ID["ras_random"]
    task_result = build_task_result(task, raw_metrics=_sample_task_payload(task.task_id))
    aggregate = aggregate_task_results([task_result])

    assert task_result["metrics"]["combined_score"] == pytest.approx(
        task_result["metrics"]["score"]
    )
    assert aggregate["combined_score"] == pytest.approx(aggregate["score"])

    extracted = extract_task_result(
        {
            "task_results": [task_result],
        },
        task.task_id,
    )

    assert extracted is not None
    assert extracted["task_id"] == task.task_id
    assert extracted["metrics"]["combined_score"] == pytest.approx(
        extracted["metrics"]["score"]
    )
    assert len(extracted["dataset_summaries"]) == 2


def test_extract_task_result_returns_none_on_malformed_stored_metrics():
    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "ras_random",
                    "metrics": "bad",
                }
            ]
        },
        "ras_random",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "ras_random",
                    "metrics": {
                        "correctness_rate": 1.0,
                        "speed_score": 0.4,
                    },
                }
            ]
        },
        "ras_random",
    ) is None


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "rust_adaptive_sort_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "rust_adaptive_sort_emo_sta" / "evaluator.py"
    initial_program = REPO_ROOT / "examples" / "rust_adaptive_sort_emo_sta" / "initial_program.rs"

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)

    source_metrics_by_program: dict[str, dict[str, dict[str, float]]] = {}
    for program_index in range(2):
        task_results = [
            build_task_result(task, raw_metrics=_sample_task_payload(task.task_id))
            for task in RUST_ADAPTIVE_SORT_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"rust_program_{program_index}",
            code="pub fn adaptive_sort<T: Ord + Clone>(arr: &mut [T]) { arr.sort(); }\n",
            changes_description=f"rust program {program_index}",
            language="rust",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "evaluation_mode": "shared",
                    "compile_succeeded": True,
                    "task_results": task_results,
                }
            ),
        )
        shared_database.add(program, target_island=program.metadata["island"])
        source_metrics_by_program[program.id] = {
            task_result["task_id"]: task_result["metrics"] for task_result in task_results
        }

    shared_checkpoint = tmp_path / "shared_checkpoint"
    shared_database.save(str(shared_checkpoint), iteration=4)

    def fail_if_reevaluated(**kwargs):
        raise AssertionError("Spawn should use stored task_results artifacts instead of reevaluation")

    monkeypatch.setattr(
        "openevolve.multi_task_shared_then_adapt.spawn._reevaluate_program_for_task",
        fail_if_reevaluated,
    )

    spawned_root = tmp_path / "spawned"
    spawn_results = spawn_task_checkpoints(
        shared_checkpoint_path=shared_checkpoint,
        output_root=spawned_root,
        base_config_path=base_config_path,
        evaluation_file=evaluation_file,
        family="rust_adaptive_sort",
        task_ids=["ras_random"],
        initial_program=initial_program,
    )

    assert "ras_random" in spawn_results
    spawned_checkpoint = spawned_root / "ras_random"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["ras_random"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "ras_random"

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "ras_random"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "ras_random"


def test_rust_suffix_mapping_returns_rs():
    assert _program_file_suffix(
        program_language="rust",
        default_file_suffix=".py",
        initial_program=None,
    ) == ".rs"
    assert _program_file_suffix(
        program_language="rs",
        default_file_suffix=".py",
        initial_program=None,
    ) == ".rs"


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="cargo not available")
def test_shared_mode_evaluator_smoke(monkeypatch):
    monkeypatch.setenv(RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR, "all")
    result = rust_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "rust_adaptive_sort_emo_sta" / "initial_program.rs")
    )

    assert result.artifacts["compile_succeeded"] is True
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert len(result.artifacts["task_results"]) == 4
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.metrics["correctness_rate"] == pytest.approx(1.0)


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="cargo not available")
def test_task_specific_evaluator_smoke(monkeypatch):
    monkeypatch.setenv(RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR, "ras_random")
    result = rust_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "rust_adaptive_sort_emo_sta" / "initial_program.rs")
    )

    assert result.artifacts["compile_succeeded"] is True
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == "ras_random"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == "ras_random"
    assert result.metrics["correctness_rate"] == pytest.approx(1.0)


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="cargo not available")
def test_compile_failure_returns_finite_zero_metrics(tmp_path, monkeypatch):
    bad_program = tmp_path / "bad.rs"
    bad_program.write_text(
        "pub fn adaptive_sort<T: Ord + Clone>(arr: &mut [T]) { let x = ; }\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(RUST_ADAPTIVE_SORT_TASK_SELECTOR_ENV_VAR, "ras_random")

    result = rust_evaluator.evaluate(str(bad_program))

    assert result.artifacts["compile_succeeded"] is False
    assert result.metrics["score"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    assert all(math.isfinite(float(value)) for value in result.metrics.values())
    assert result.artifacts["task_results"][0]["error"] is not None
