import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.multi_task_shared_then_adapt.k_module_problem_balanced import (
    K_MODULE_BALANCED_MODULE_NAMES,
    K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR,
    K_MODULE_BALANCED_TASK_SPECS,
    K_MODULE_BALANCED_TOTAL_MODULES,
    K_MODULE_BALANCED_VALID_OPTIONS,
    aggregate_task_results,
    build_task_result,
    count_correct_modules,
    extract_task_result,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_balanced_k_module_evaluator_module():
    evaluator_path = (
        REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "evaluator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "k_module_problem_emo_sta_test_module",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k_module_balanced_evaluator = _load_balanced_k_module_evaluator_module()


EXPECTED_TASK_TARGETS = {
    "kmb_task_a": {
        "loader": "loader_0",
        "preprocess": "prep_0",
        "sampler": "sample_0",
        "algorithm": "algo_0",
        "scheduler": "sched_4",
        "formatter": "fmt_0",
    },
    "kmb_task_b": {
        "loader": "loader_0",
        "preprocess": "prep_3",
        "sampler": "sample_5",
        "algorithm": "algo_3",
        "scheduler": "sched_2",
        "formatter": "fmt_1",
    },
    "kmb_task_c": {
        "loader": "loader_2",
        "preprocess": "prep_0",
        "sampler": "sample_4",
        "algorithm": "algo_3",
        "scheduler": "sched_3",
        "formatter": "fmt_3",
    },
    "kmb_task_d": {
        "loader": "loader_4",
        "preprocess": "prep_2",
        "sampler": "sample_0",
        "algorithm": "algo_5",
        "scheduler": "sched_2",
        "formatter": "fmt_3",
    },
}
EXPECTED_CONSENSUS = {
    "loader": "loader_0",
    "preprocess": "prep_0",
    "sampler": "sample_0",
    "algorithm": "algo_3",
    "scheduler": "sched_2",
    "formatter": "fmt_3",
}


def _build_candidate_program_code(config: dict[str, str]) -> str:
    serialized = json.dumps(config, sort_keys=True, indent=4)
    return (
        "# EVOLVE-BLOCK-START\n"
        "def configure_pipeline():\n"
        f"    return {serialized}\n\n"
        "def run_pipeline():\n"
        "    return configure_pipeline()\n"
        "# EVOLVE-BLOCK-END\n"
    )


def _task_metrics_for_config(
    task,
    candidate_config: dict[str, str],
    *,
    eval_time: float,
) -> dict[str, float]:
    correct_modules = count_correct_modules(task, candidate_config)
    accuracy = correct_modules / float(K_MODULE_BALANCED_TOTAL_MODULES)
    return {
        "correct_modules": correct_modules,
        "total_modules": K_MODULE_BALANCED_TOTAL_MODULES,
        "accuracy": accuracy,
        "score": accuracy,
        "combined_score": accuracy,
        "eval_time": eval_time,
    }


def _consensus_from_targets() -> dict[str, str]:
    consensus: dict[str, str] = {}
    for module_name in K_MODULE_BALANCED_MODULE_NAMES:
        counts: dict[str, int] = {}
        for task in K_MODULE_BALANCED_TASK_SPECS:
            value = task.target_config[module_name]
            counts[value] = counts.get(value, 0) + 1
        consensus[module_name] = max(counts.items(), key=lambda item: item[1])[0]
    return consensus


def test_hidden_tasks_are_defined_exactly_as_requested():
    assert [task.task_id for task in K_MODULE_BALANCED_TASK_SPECS] == [
        "kmb_task_a",
        "kmb_task_b",
        "kmb_task_c",
        "kmb_task_d",
    ]
    assert {
        task.task_id: task.target_config for task in K_MODULE_BALANCED_TASK_SPECS
    } == EXPECTED_TASK_TARGETS


def test_each_module_has_exactly_six_expected_options():
    assert tuple(K_MODULE_BALANCED_MODULE_NAMES) == (
        "loader",
        "preprocess",
        "sampler",
        "algorithm",
        "scheduler",
        "formatter",
    )
    for module_name in K_MODULE_BALANCED_MODULE_NAMES:
        options = K_MODULE_BALANCED_VALID_OPTIONS[module_name]
        assert len(options) == 6
        assert len(set(options)) == 6


def test_balanced_family_has_unique_2_of_4_majority_in_each_slot():
    for module_name in K_MODULE_BALANCED_MODULE_NAMES:
        counts: dict[str, int] = {}
        for task in K_MODULE_BALANCED_TASK_SPECS:
            value = task.target_config[module_name]
            counts[value] = counts.get(value, 0) + 1

        assert max(counts.values()) == 2
        assert 3 not in counts.values()
        assert sum(1 for count in counts.values() if count == 2) == 1


def test_unique_consensus_is_exact_and_no_task_equals_it():
    consensus = _consensus_from_targets()
    assert consensus == EXPECTED_CONSENSUS
    for task in K_MODULE_BALANCED_TASK_SPECS:
        assert task.target_config != consensus


def test_each_task_differs_from_consensus_in_exactly_three_modules():
    consensus = _consensus_from_targets()
    for task in K_MODULE_BALANCED_TASK_SPECS:
        differing_modules = [
            module_name
            for module_name in K_MODULE_BALANCED_MODULE_NAMES
            if task.target_config[module_name] != consensus[module_name]
        ]
        assert len(differing_modules) == 3


def test_each_pair_of_tasks_shares_exactly_one_module():
    for lhs, rhs in itertools.combinations(K_MODULE_BALANCED_TASK_SPECS, 2):
        shared_modules = [
            module_name
            for module_name in K_MODULE_BALANCED_MODULE_NAMES
            if lhs.target_config[module_name] == rhs.target_config[module_name]
        ]
        assert len(shared_modules) == 1


def test_shared_mode_returns_aggregate_metrics_and_four_task_results(monkeypatch):
    monkeypatch.setenv(K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR, "all")

    result = k_module_balanced_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "initial_program.py")
    )

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.metrics["successful_task_count"] == pytest.approx(4.0)
    assert result.metrics["failed_task_count"] == pytest.approx(0.0)
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in K_MODULE_BALANCED_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        assert task_result["metrics"]["combined_score"] == pytest.approx(
            task_result["metrics"]["score"]
        )
        assert task_result["spec"] == {
            "module_names": [
                "loader",
                "preprocess",
                "sampler",
                "algorithm",
                "scheduler",
                "formatter",
            ],
            "num_modules": 6,
            "option_counts": {
                "loader": 6,
                "preprocess": 6,
                "sampler": 6,
                "algorithm": 6,
                "scheduler": 6,
                "formatter": 6,
            },
        }


def test_task_specific_mode_returns_exactly_one_task_result(monkeypatch):
    selected_task_id = "kmb_task_c"
    monkeypatch.setenv(K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR, selected_task_id)

    result = k_module_balanced_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "initial_program.py")
    )

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == selected_task_id
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == selected_task_id


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = (
        REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "config.yaml"
    )
    evaluation_file = (
        REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "evaluator.py"
    )
    initial_program = (
        REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "initial_program.py"
    )

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)

    candidate_configs = [
        {
            "loader": "loader_0",
            "preprocess": "prep_0",
            "sampler": "sample_0",
            "algorithm": "algo_3",
            "scheduler": "sched_1",
            "formatter": "fmt_5",
        },
        {
            "loader": "loader_4",
            "preprocess": "prep_2",
            "sampler": "sample_0",
            "algorithm": "algo_5",
            "scheduler": "sched_2",
            "formatter": "fmt_3",
        },
    ]

    source_metrics_by_program: dict[str, dict[str, dict[str, float]]] = {}
    for program_index, candidate_config in enumerate(candidate_configs):
        task_results = [
            build_task_result(
                task,
                raw_metrics=_task_metrics_for_config(
                    task,
                    candidate_config,
                    eval_time=0.01 + 0.01 * program_index + 0.001 * task.task_index,
                ),
            )
            for task in K_MODULE_BALANCED_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"balanced_program_{program_index}",
            code=_build_candidate_program_code(candidate_config),
            changes_description=f"balanced program {program_index}",
            language="python",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "task_results": task_results,
                    "search_space_size": 46656,
                    "evaluation_mode": "shared",
                    "status": "shared_evaluation_complete",
                }
            ),
        )
        shared_database.add(program, target_island=program.metadata["island"])
        source_metrics_by_program[program.id] = {
            task_result["task_id"]: task_result["metrics"] for task_result in task_results
        }

    shared_checkpoint = tmp_path / "shared_checkpoint"
    shared_database.save(str(shared_checkpoint), iteration=5)

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
        family="k_module_problem_balanced",
        task_ids=["kmb_task_c"],
        initial_program=initial_program,
    )

    assert "kmb_task_c" in spawn_results
    spawned_checkpoint = spawned_root / "kmb_task_c"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2
    assert spawned_database.best_program_id is not None

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["kmb_task_c"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "kmb_task_c"
        assert program.artifact_dir is None

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "kmb_task_c"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "kmb_task_c"
        assert task_artifacts["search_space_size"] == 46656
        assert task_artifacts["evaluation_mode"] == "task_specific"
        assert task_artifacts["status"] == "task_evaluation_complete"
        assert "candidate_configuration" not in task_artifacts


def test_extract_task_result_returns_none_for_malformed_stored_metrics():
    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "kmb_task_a",
                    "metrics": "bad",
                }
            ]
        },
        "kmb_task_a",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "kmb_task_a",
                    "metrics": {
                        "correct_modules": 3,
                        "total_modules": 6,
                        "accuracy": 0.5,
                        "score": 0.5,
                        "combined_score": 0.5,
                    },
                }
            ]
        },
        "kmb_task_a",
    ) is None


def test_hidden_target_configs_do_not_leak_into_public_specs_or_artifacts(monkeypatch):
    public_specs_serialized = json.dumps(
        [task.to_spec_dict() for task in K_MODULE_BALANCED_TASK_SPECS],
        sort_keys=True,
    )

    monkeypatch.setenv(K_MODULE_BALANCED_TASK_SELECTOR_ENV_VAR, "all")
    result = k_module_balanced_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "initial_program.py")
    )
    artifact_serialized = json.dumps(result.artifacts, sort_keys=True)

    hidden_values = sorted(
        {
            value
            for target in EXPECTED_TASK_TARGETS.values()
            for value in target.values()
        }
    )
    for hidden_value in hidden_values:
        assert hidden_value not in public_specs_serialized
        assert hidden_value not in artifact_serialized


def test_balanced_k_module_family_registry_resolves():
    balanced_family = get_family_definition("k_module_problem_balanced")
    assert balanced_family.family == "k_module_problem_balanced"
    assert [task.task_id for task in balanced_family.task_specs] == [
        task.task_id for task in K_MODULE_BALANCED_TASK_SPECS
    ]
