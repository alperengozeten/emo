import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.multi_task_shared_then_adapt.function_minimization import (
    FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR,
    FUNCTION_MINIMIZATION_TASK_SPECS,
    FUNCTION_MINIMIZATION_TASKS_BY_ID,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    objective_ackley,
    objective_rastrigin,
    objective_rosenbrock,
    objective_sincosxy,
)
import openevolve.multi_task_shared_then_adapt.spawn as emo_sta_spawn
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_fm_evaluator_module():
    evaluator_path = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("fm_emo_sta_test_module", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fm_evaluator = _load_fm_evaluator_module()


def _candidate_program_code() -> str:
    return (
        "# EVOLVE-BLOCK-START\n"
        "import numpy as np\n\n"
        "def search_algorithm(objective_fn, bounds, iterations=1000, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    (x_min, x_max), (y_min, y_max) = bounds\n"
        "    x = float(rng.uniform(x_min, x_max))\n"
        "    y = float(rng.uniform(y_min, y_max))\n"
        "    return x, y, float(objective_fn(x, y))\n\n"
        "def run_search(objective_fn, bounds, iterations=1000, seed=0):\n"
        "    return search_algorithm(objective_fn, bounds, iterations=iterations, seed=seed)\n"
        "# EVOLVE-BLOCK-END\n"
    )


def _fake_raw_metrics(task, program_index: int) -> dict[str, float]:
    value_gap = 0.05 * (task.task_index + 1) + 0.02 * program_index
    distance = 0.20 * (task.task_index + 1) + 0.05 * program_index
    reliability = max(0.0, 1.0 - 0.1 * program_index)
    best_value = task.optimum_value + value_gap
    value_score = 1.0 / (1.0 + value_gap)
    distance_score = 1.0 / (1.0 + distance)
    score = 0.50 * value_score + 0.35 * distance_score + 0.15 * reliability
    return {
        "best_value": best_value,
        "value_gap": value_gap,
        "distance_to_optimum": distance,
        "value_score": value_score,
        "distance_score": distance_score,
        "reliability_score": reliability,
        "avg_eval_time": 0.01 + 0.01 * task.task_index + 0.02 * program_index,
        "score": score,
        "combined_score": score,
    }


def test_shared_mode_returns_aggregate_metrics_and_task_artifacts(monkeypatch):
    monkeypatch.setenv(FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR, "all")

    result = fm_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "function_minimization_emo_sta" / "initial_program.py")
    )

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in FUNCTION_MINIMIZATION_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        assert task_result["metrics"]["combined_score"] == pytest.approx(
            task_result["metrics"]["score"]
        )
        assert task_result["spec"] == {
            "display_name": task_result["spec"]["display_name"],
            "bounds": task_result["spec"]["bounds"],
        }
    assert "best_observed_points" not in result.artifacts


def test_hidden_optima_are_not_leaked_in_public_artifacts(monkeypatch):
    monkeypatch.setenv(FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR, "all")

    result = fm_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "function_minimization_emo_sta" / "initial_program.py")
    )
    serialized = json.dumps(result.artifacts, sort_keys=True)

    for hidden_field in (
        "formula_name",
        "optimum_x",
        "optimum_y",
        "optimum_value",
        "best_observed_point",
        "best_observed_points",
    ):
        assert hidden_field not in serialized


def test_task_specific_mode_returns_one_task(monkeypatch):
    selected_task_id = "fm_ackley_2d"
    monkeypatch.setenv(FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR, selected_task_id)

    result = fm_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "function_minimization_emo_sta" / "initial_program.py")
    )

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == selected_task_id
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == selected_task_id


def test_objective_formulas_match_known_optima():
    assert objective_ackley(0.0, 0.0) == pytest.approx(0.0, abs=1.0e-12)
    assert objective_rastrigin(0.0, 0.0) == pytest.approx(0.0, abs=1.0e-12)
    assert objective_rosenbrock(1.0, 1.0) == pytest.approx(0.0, abs=1.0e-12)
    assert objective_sincosxy(-1.70406466, 0.67752040) == pytest.approx(
        -1.51868584,
        abs=1.0e-6,
    )


def test_task_registry_uses_shifted_hidden_optima():
    expected_optima = {
        "fm_sincosxy_2d": (-0.80406466, -0.72247960, -1.51868584),
        "fm_ackley_2d": (1.7, -1.3, 0.0),
        "fm_rastrigin_2d": (-2.2, 1.4, 0.0),
        "fm_rosenbrock_2d": (-0.4, 1.7, 0.0),
    }

    for task_id, (expected_x, expected_y, expected_value) in expected_optima.items():
        task = FUNCTION_MINIMIZATION_TASKS_BY_ID[task_id]
        assert task.optimum_x == pytest.approx(expected_x, abs=1.0e-12)
        assert task.optimum_y == pytest.approx(expected_y, abs=1.0e-12)
        assert task.objective_fn(task.optimum_x, task.optimum_y) == pytest.approx(
            expected_value,
            abs=1.0e-6,
        )


def test_evaluator_accepts_candidate_returning_xy_and_computes_value(monkeypatch, tmp_path):
    task = FUNCTION_MINIMIZATION_TASKS_BY_ID["fm_ackley_2d"]
    candidate_path = tmp_path / "candidate_xy_only.py"
    candidate_path.write_text(
        (
            "def run_search(objective_fn, bounds, iterations=1000, seed=0):\n"
            f"    return {task.optimum_x}, {task.optimum_y}\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR, "fm_ackley_2d")

    result = fm_evaluator.evaluate(str(candidate_path))

    assert result.metrics["best_value"] == pytest.approx(task.optimum_value, abs=1.0e-12)
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluate_trial_skips_signal_timer_off_main_thread(monkeypatch):
    task = FUNCTION_MINIMIZATION_TASKS_BY_ID["fm_ackley_2d"]
    actual_main_thread = threading.main_thread()

    class _WorkerThread:
        pass

    monkeypatch.setattr(fm_evaluator.threading, "current_thread", lambda: _WorkerThread())
    monkeypatch.setattr(fm_evaluator.threading, "main_thread", lambda: actual_main_thread)

    result = fm_evaluator._evaluate_trial(
        lambda objective_fn, bounds, iterations=1000, seed=0: (
            task.optimum_x,
            task.optimum_y,
            objective_fn(task.optimum_x, task.optimum_y),
        ),
        task,
        iterations=task.search_iterations_stage1,
        seed=0,
        timeout_seconds=0.1,
    )

    assert result["best_value"] == pytest.approx(task.optimum_value, abs=1.0e-12)
    assert result["distance_to_optimum"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["runtime"] >= 0.0


def test_evaluator_hides_raw_objective_identity_from_candidate(monkeypatch, tmp_path):
    candidate_path = tmp_path / "candidate_name_cheat.py"
    candidate_path.write_text(
        (
            "def run_search(objective_fn, bounds, iterations=1000, seed=0):\n"
            "    leaked_name = getattr(objective_fn, '__name__', '')\n"
            "    if leaked_name == 'objective_sincosxy':\n"
            "        return -1.70406466, 0.67752040, -1.51868584\n"
            "    if leaked_name == 'objective_ackley':\n"
            "        return 0.0, 0.0, 0.0\n"
            "    if leaked_name == 'objective_rastrigin':\n"
            "        return 0.0, 0.0, 0.0\n"
            "    if leaked_name == 'objective_rosenbrock':\n"
            "        return 1.0, 1.0, 0.0\n"
            "    return 2.0, 2.0, float(objective_fn(2.0, 2.0))\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR, "all")

    result = fm_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] < 0.95
    assert all(task_result["error"] is None for task_result in result.artifacts["task_results"])


def test_evaluator_blocks_textbook_optimum_cheat(monkeypatch, tmp_path):
    candidate_path = tmp_path / "candidate_textbook_optima.py"
    candidate_path.write_text(
        (
            "def run_search(objective_fn, bounds, iterations=1000, seed=0):\n"
            "    candidates = [(-1.70406466, 0.67752040), (0.0, 0.0), (1.0, 1.0)]\n"
            "    (x_min, x_max), (y_min, y_max) = bounds\n"
            "    best = None\n"
            "    for x, y in candidates:\n"
            "        if not (x_min <= x <= x_max and y_min <= y <= y_max):\n"
            "            continue\n"
            "        value = float(objective_fn(x, y))\n"
            "        if best is None or value < best[2]:\n"
            "            best = (x, y, value)\n"
            "    return best\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(FUNCTION_MINIMIZATION_TASK_SELECTOR_ENV_VAR, "all")

    result = fm_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] < 0.99
    assert any(
        task_result["metrics"]["combined_score"] < 0.99
        for task_result in result.artifacts["task_results"]
    )


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "evaluator.py"
    initial_program = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "initial_program.py"

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)

    source_metrics_by_program: dict[str, dict[str, dict[str, float]]] = {}
    for program_index in range(2):
        task_results = [
            build_task_result(
                task,
                raw_metrics=_fake_raw_metrics(task, program_index),
            )
            for task in FUNCTION_MINIMIZATION_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"program_{program_index}",
            code=_candidate_program_code(),
            changes_description=f"program {program_index}",
            language="python",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "task_results": task_results,
                    "trial_counts": {
                        task.task_id: {"total": 5, "successful": 5, "failed": 0}
                        for task in FUNCTION_MINIMIZATION_TASK_SPECS
                    },
                    "convergence_notes": {
                        task.task_id: "Shared evaluation complete."
                        for task in FUNCTION_MINIMIZATION_TASK_SPECS
                    },
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
        family="function_minimization",
        task_ids=["fm_ackley_2d"],
        initial_program=initial_program,
    )

    assert "fm_ackley_2d" in spawn_results
    spawned_checkpoint = spawned_root / "fm_ackley_2d"
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
        expected_metrics = source_metrics_by_program[program_id]["fm_ackley_2d"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "fm_ackley_2d"
        assert program.artifact_dir is None

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "fm_ackley_2d"
        assert task_artifacts["evaluation_mode"] == "task_specific"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "fm_ackley_2d"
        assert "formula_name" not in json.dumps(task_artifacts, sort_keys=True)
        assert "optimum_x" not in json.dumps(task_artifacts, sort_keys=True)
        assert "best_observed_point" not in task_artifacts
        assert task_artifacts["task_results"][0]["metrics"]["combined_score"] == pytest.approx(
            expected_metrics["combined_score"]
        )


def test_extract_task_result_returns_none_for_malformed_stored_metrics():
    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "fm_ackley_2d",
                    "metrics": "bad",
                }
            ]
        },
        "fm_ackley_2d",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "fm_ackley_2d",
                    "metrics": {
                        "best_value": 0.1,
                        "value_gap": 0.1,
                        "distance_to_optimum": 0.1,
                        "value_score": 0.9,
                        "distance_score": 0.9,
                        "reliability_score": 1.0,
                        "avg_eval_time": 0.01,
                        "score": 0.9,
                    },
                }
            ]
        },
        "fm_ackley_2d",
    ) is None


def test_extract_task_result_returns_none_for_stage1_artifacts():
    task = FUNCTION_MINIMIZATION_TASKS_BY_ID["fm_ackley_2d"]
    task_result = build_task_result(task, raw_metrics=_fake_raw_metrics(task, 0))

    assert extract_task_result(
        {
            "evaluation_stage": "stage1",
            "task_results": [task_result],
        },
        "fm_ackley_2d",
    ) is None


def test_reevaluate_program_for_task_supports_plain_dict_results(tmp_path, monkeypatch):
    task = FUNCTION_MINIMIZATION_TASKS_BY_ID["fm_ackley_2d"]
    raw_metrics = _fake_raw_metrics(task, 0)

    monkeypatch.setattr(
        emo_sta_spawn,
        "_load_evaluation_module",
        lambda _: SimpleNamespace(evaluate=lambda _: {**raw_metrics, "artifacts": {}}),
    )

    program = Program(
        id="program_plain_dict_eval",
        code=_candidate_program_code(),
        changes_description="plain dict evaluator compatibility",
        language="python",
        metrics={"score": 0.0, "combined_score": 0.0},
    )

    task_result = emo_sta_spawn._reevaluate_program_for_task(
        program=program,
        task_id="fm_ackley_2d",
        family="function_minimization",
        evaluation_file=tmp_path / "unused_evaluator.py",
    )

    assert task_result["task_id"] == "fm_ackley_2d"
    assert task_result["metrics"]["combined_score"] == pytest.approx(
        task_result["metrics"]["score"]
    )
    assert task_result["metrics"]["combined_score"] > 0.0


def test_spawn_reevaluates_when_stored_task_metrics_are_only_stage1(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "evaluator.py"
    initial_program = REPO_ROOT / "examples" / "function_minimization_emo_sta" / "initial_program.py"

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)
    task = FUNCTION_MINIMIZATION_TASKS_BY_ID["fm_ackley_2d"]

    shared_database.add(
        Program(
            id="program_stage1_artifact",
            code=_candidate_program_code(),
            changes_description="stage1 artifact source",
            language="python",
            generation=0,
            iteration_found=0,
            metrics=aggregate_task_results(
                [build_task_result(task, raw_metrics=_fake_raw_metrics(task, 0))]
            ),
            metadata={"island": 0},
            artifacts_json=json.dumps(
                {
                    "evaluation_stage": "stage1",
                    "task_selector": "all",
                    "task_results": [
                        build_task_result(task, raw_metrics=_fake_raw_metrics(task, 0))
                    ],
                }
            ),
        ),
        target_island=0,
    )

    shared_checkpoint = tmp_path / "shared_checkpoint"
    shared_database.save(str(shared_checkpoint), iteration=5)

    reevaluated_task_result = build_task_result(
        task,
        raw_metrics=_fake_raw_metrics(task, 3),
    )
    reevaluation_calls: list[str] = []

    def fake_reevaluate_program_for_task(*, program, task_id, evaluation_file):
        del evaluation_file
        reevaluation_calls.append(program.id)
        assert task_id == "fm_ackley_2d"
        return reevaluated_task_result

    monkeypatch.setattr(
        "openevolve.multi_task_shared_then_adapt.spawn._reevaluate_program_for_task",
        fake_reevaluate_program_for_task,
    )

    spawned_root = tmp_path / "spawned"
    spawn_results = spawn_task_checkpoints(
        shared_checkpoint_path=shared_checkpoint,
        output_root=spawned_root,
        base_config_path=base_config_path,
        evaluation_file=evaluation_file,
        family="function_minimization",
        task_ids=["fm_ackley_2d"],
        initial_program=initial_program,
    )

    assert reevaluation_calls == ["program_stage1_artifact"]
    assert spawn_results["fm_ackley_2d"]["reevaluated_program_ids"] == ["program_stage1_artifact"]

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_root / "fm_ackley_2d"))

    spawned_program = spawned_database.programs["program_stage1_artifact"]
    assert spawned_program.metrics["combined_score"] == pytest.approx(
        reevaluated_task_result["metrics"]["combined_score"]
    )
    task_artifacts = spawned_database.get_artifacts("program_stage1_artifact")
    assert task_artifacts["evaluation_stage"] == "full"
