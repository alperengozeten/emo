import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.sldbench_3d import (
    CANONICAL_FEATURE_NAMES,
    SLDBENCH_3D_SHARED_SELECTOR,
    SLDBENCH_3D_TASK_SELECTOR_ENV_VAR,
    SLDBENCH_3D_TASK_SPECS,
    SLDBENCH_3D_TASKS_BY_ID,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    resolve_task_specs,
)
import openevolve.multi_task_shared_then_adapt.spawn as emo_sta_spawn
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sldbench_data_loader = _load_module(
    REPO_ROOT / "examples" / "sldbench_3d_emo_sta" / "data_loader.py",
    "sldbench_3d_emo_sta_data_loader_test",
)
sldbench_evaluator = _load_module(
    REPO_ROOT / "examples" / "sldbench_3d_emo_sta" / "evaluator.py",
    "sldbench_3d_emo_sta_evaluator_test",
)


def _initial_program_path() -> Path:
    return REPO_ROOT / "examples" / "sldbench_3d_emo_sta" / "initial_program.py"


def _fake_raw_metrics(task_index: int, program_index: int) -> dict[str, float]:
    nmse = 0.20 + 0.05 * task_index + 0.02 * program_index
    score = 1.0 / (1.0 + nmse)
    return {
        "nmse": nmse,
        "nmae": 0.35 + 0.04 * task_index + 0.01 * program_index,
        "r2": 1.0 - nmse,
        "fit_group_count": 2.0,
        "eval_group_count": 2.0,
        "successful_group_count": 2.0,
        "failed_group_count": 0.0,
        "score": score,
        "combined_score": score,
    }


def test_family_registry_resolves_sldbench_3d():
    family = get_family_definition("sldbench_3d")
    assert family.family == "sldbench_3d"
    assert family.task_selector_env_var == SLDBENCH_3D_TASK_SELECTOR_ENV_VAR
    assert family.shared_selector == SLDBENCH_3D_SHARED_SELECTOR
    assert [task.task_id for task in family.task_specs] == [
        "vocab_scaling_law",
        "data_constrained_scaling_law",
    ]


def test_resolve_task_specs_all_returns_expected_tasks():
    resolved = resolve_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "vocab_scaling_law",
        "data_constrained_scaling_law",
    ]


def test_canonicalization_maps_vocab_columns_to_canonical_order():
    canonical = sldbench_data_loader.canonicalize_feature_columns(
        "vocab_scaling_law",
        {
            "non_vocab_parameters": [1.0, 2.0],
            "vocab_size": [3.0, 4.0],
            "num_characters": [5.0, 6.0],
        },
    )

    assert CANONICAL_FEATURE_NAMES == (
        "model_size_like",
        "diversity_like",
        "total_data_like",
    )
    assert np.array_equal(
        canonical,
        np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=float),
    )


def test_canonicalization_maps_data_constrained_columns_to_canonical_order():
    canonical = sldbench_data_loader.canonicalize_feature_columns(
        "data_constrained_scaling_law",
        {
            "unique_tokens": [11.0, 12.0],
            "params": [21.0, 22.0],
            "tokens": [31.0, 32.0],
        },
    )

    assert np.array_equal(
        canonical,
        np.asarray([[21.0, 11.0, 31.0], [22.0, 12.0, 32.0]], dtype=float),
    )


def test_shared_mode_evaluator_returns_aggregate_metrics_and_task_artifacts(monkeypatch):
    monkeypatch.setenv(sldbench_data_loader.SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR, "1")
    monkeypatch.setenv(SLDBENCH_3D_TASK_SELECTOR_ENV_VAR, "all")

    result = sldbench_evaluator.evaluate(str(_initial_program_path()))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(2.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert result.artifacts["loader_mode"] == sldbench_data_loader.SYNTHETIC_FIXTURE_MODE
    assert len(result.artifacts["task_results"]) == 2
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in SLDBENCH_3D_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        metrics = task_result["metrics"]
        assert task_result["error"] is None
        assert metrics["combined_score"] == pytest.approx(metrics["score"])
        assert metrics["score"] == pytest.approx(1.0 / (1.0 + metrics["nmse"]))


def test_task_specific_mode_returns_one_task_and_score_formula(monkeypatch):
    monkeypatch.setenv(sldbench_data_loader.SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR, "1")
    monkeypatch.setenv(SLDBENCH_3D_TASK_SELECTOR_ENV_VAR, "vocab_scaling_law")

    result = sldbench_evaluator.evaluate(str(_initial_program_path()))

    assert result.artifacts["task_selector"] == "vocab_scaling_law"
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == "vocab_scaling_law"
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["score"] == pytest.approx(1.0 / (1.0 + result.metrics["nmse"]))


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "sldbench_3d_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "sldbench_3d_emo_sta" / "evaluator.py"
    initial_program = _initial_program_path()

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)

    source_metrics_by_program: dict[str, dict[str, dict[str, float]]] = {}
    for program_index in range(2):
        task_results = [
            build_task_result(
                task,
                raw_metrics=_fake_raw_metrics(task.task_index, program_index),
            )
            for task in SLDBENCH_3D_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"sldbench_program_{program_index}",
            code=_initial_program_path().read_text(encoding="utf-8"),
            changes_description=f"sldbench program {program_index}",
            language="python",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "evaluation_mode": "shared",
                    "loader_mode": "synthetic_fixture",
                    "task_results": task_results,
                }
            ),
        )
        shared_database.add(program, target_island=program.metadata["island"])
        source_metrics_by_program[program.id] = {
            task_result["task_id"]: task_result["metrics"] for task_result in task_results
        }

    shared_checkpoint = tmp_path / "shared_checkpoint"
    shared_database.save(str(shared_checkpoint), iteration=6)

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
        family="sldbench_3d",
        task_ids=["vocab_scaling_law"],
        initial_program=initial_program,
    )

    assert "vocab_scaling_law" in spawn_results
    spawned_checkpoint = spawned_root / "vocab_scaling_law"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["vocab_scaling_law"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "vocab_scaling_law"

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "vocab_scaling_law"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "vocab_scaling_law"


def test_extract_task_result_returns_none_for_malformed_stored_metrics():
    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "vocab_scaling_law",
                    "metrics": "bad",
                }
            ]
        },
        "vocab_scaling_law",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "vocab_scaling_law",
                    "metrics": {
                        "nmse": 0.2,
                        "nmae": 0.3,
                        "r2": 0.8,
                        "score": 0.83,
                    },
                }
            ]
        },
        "vocab_scaling_law",
    ) is None


def test_reevaluate_program_for_task_supports_plain_dict_results(tmp_path, monkeypatch):
    task_results = [
        build_task_result(
            task,
            raw_metrics=_fake_raw_metrics(task.task_index, 0),
        )
        for task in SLDBENCH_3D_TASK_SPECS
    ]
    evaluation_result = {
        "metrics": aggregate_task_results(task_results),
        "artifacts": {
            "task_selector": "all",
            "evaluation_mode": "shared",
            "loader_mode": "synthetic_fixture",
            "task_results": task_results,
        },
    }

    monkeypatch.setattr(
        emo_sta_spawn,
        "_load_evaluation_module",
        lambda _: SimpleNamespace(evaluate=lambda _: evaluation_result),
    )

    program = Program(
        id="program_plain_dict_eval",
        code=_initial_program_path().read_text(encoding="utf-8"),
        changes_description="plain dict evaluator compatibility",
        language="python",
        metrics={"score": 0.0, "combined_score": 0.0},
    )

    task_result = emo_sta_spawn._reevaluate_program_for_task(
        program=program,
        task_id="data_constrained_scaling_law",
        family="sldbench_3d",
        evaluation_file=tmp_path / "unused_evaluator.py",
    )

    assert task_result["task_id"] == "data_constrained_scaling_law"
    assert task_result["metrics"]["combined_score"] == pytest.approx(
        task_result["metrics"]["score"]
    )
    assert task_result["metrics"]["score"] == pytest.approx(
        1.0 / (1.0 + task_result["metrics"]["nmse"])
    )


def test_synthetic_fixture_mode_works_without_network(monkeypatch):
    monkeypatch.setenv(sldbench_data_loader.SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR, "1")
    monkeypatch.setenv(SLDBENCH_3D_TASK_SELECTOR_ENV_VAR, "all")
    sldbench_data_loader._GROUPED_DATA_CACHE.clear()
    sldbench_evaluator._DATA_LOADER_MODULE._GROUPED_DATA_CACHE.clear()
    monkeypatch.setattr(
        sldbench_data_loader,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network"))),
    )
    monkeypatch.setattr(
        sldbench_evaluator._DATA_LOADER_MODULE,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network"))),
    )

    train_data = sldbench_data_loader.load_grouped_data("vocab_scaling_law", "train")
    assert len(train_data) >= 2

    result = sldbench_evaluator.evaluate(str(_initial_program_path()))
    assert result.artifacts["loader_mode"] == sldbench_data_loader.SYNTHETIC_FIXTURE_MODE
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert len(result.artifacts["task_results"]) == 2
