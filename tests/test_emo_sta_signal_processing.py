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
from openevolve.multi_task_shared_then_adapt.signal_processing import (
    SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR,
    SIGNAL_PROCESSING_TASK_SPECS,
    SIGNAL_PROCESSING_TASKS_BY_ID,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    generate_signal_pair,
)
import openevolve.multi_task_shared_then_adapt.spawn as emo_sta_spawn
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_sp_evaluator_module():
    evaluator_path = REPO_ROOT / "examples" / "signal_processing_emo_sta" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("sp_emo_sta_test_module", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp_evaluator = _load_sp_evaluator_module()


def _fake_raw_metrics(task_index: int, program_index: int) -> dict[str, float]:
    base = 0.04 * task_index + 0.02 * program_index
    return {
        "composite_score": max(0.0, 0.72 - base),
        "slope_changes": 8.0 + 2.0 * task_index + program_index,
        "lag_error": 0.15 + 0.03 * task_index + 0.01 * program_index,
        "avg_error": 0.12 + 0.02 * task_index + 0.01 * program_index,
        "false_reversals": 2.0 + task_index + program_index,
        "correlation": 0.90 - 0.05 * task_index - 0.02 * program_index,
        "noise_reduction": max(0.0, 0.45 - 0.05 * task_index - 0.01 * program_index),
        "execution_time": 0.02 + 0.01 * task_index + 0.01 * program_index,
        "success_rate": 1.0,
    }


def test_shared_mode_returns_aggregate_metrics_and_task_artifacts(monkeypatch):
    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, "all")

    result = sp_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "signal_processing_emo_sta" / "initial_program.py")
    )

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["overall_score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert result.artifacts["evaluation_stage"] == "full"
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in SIGNAL_PROCESSING_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        metrics = task_result["metrics"]
        assert metrics["combined_score"] == pytest.approx(metrics["score"])
        assert metrics["combined_score"] == pytest.approx(metrics["overall_score"])
        assert "spec" in task_result


def test_task_specific_mode_returns_one_task(monkeypatch):
    selected_task_id = "sp_chirp_700_n04"
    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, selected_task_id)

    result = sp_evaluator.evaluate(
        str(REPO_ROOT / "examples" / "signal_processing_emo_sta" / "initial_program.py")
    )

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["overall_score"])
    assert result.artifacts["task_selector"] == selected_task_id
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == selected_task_id


def test_task_signal_generation_is_deterministic_for_fixed_seed():
    task = SIGNAL_PROCESSING_TASKS_BY_ID["sp_multifreq_600_n03"]

    noisy_a, clean_a = generate_signal_pair(task, seed=7)
    noisy_b, clean_b = generate_signal_pair(task, seed=7)

    assert np.array_equal(clean_a, clean_b)
    assert np.array_equal(noisy_a, noisy_b)


def test_different_seeds_produce_different_noisy_signals():
    task = SIGNAL_PROCESSING_TASKS_BY_ID["sp_multifreq_600_n03"]

    noisy_a, clean_a = generate_signal_pair(task, seed=0)
    noisy_b, clean_b = generate_signal_pair(task, seed=1)

    assert np.array_equal(clean_a, clean_b)
    assert not np.array_equal(noisy_a, noisy_b)


def test_evaluator_passes_actual_noisy_signal_into_candidate(monkeypatch, tmp_path):
    selected_task_id = "sp_trend_sine_500_n02"
    custom_noisy = np.linspace(10.0, 20.0, 30, dtype=float)
    custom_clean = np.zeros_like(custom_noisy)

    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, selected_task_id)
    monkeypatch.setattr(
        sp_evaluator,
        "generate_signal_pair",
        lambda task, seed: (custom_noisy.copy(), custom_clean.copy()),
    )

    candidate_path = tmp_path / "candidate_checks_input.py"
    candidate_path.write_text(
        (
            "import numpy as np\n"
            f"EXPECTED = np.asarray({custom_noisy.tolist()}, dtype=float)\n\n"
            "def process_signal(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    if arr.shape != EXPECTED.shape or not np.allclose(arr, EXPECTED):\n"
            "        raise RuntimeError('unexpected_input')\n"
            "    return arr[window_size - 1:]\n"
        ),
        encoding="utf-8",
    )

    result = sp_evaluator.evaluate(str(candidate_path))

    assert result.metrics["success_rate"] == pytest.approx(1.0)
    assert result.artifacts["task_results"][0]["error"] is None


def test_mutating_candidate_cannot_change_metrics_without_changing_output(monkeypatch, tmp_path):
    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, "sp_trend_sine_500_n02")

    baseline_path = tmp_path / "candidate_non_mutating.py"
    baseline_path.write_text(
        (
            "import numpy as np\n\n"
            "def run_signal_processing(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    filtered = arr[window_size - 1:].copy()\n"
            "    return {'filtered_signal': filtered}\n"
        ),
        encoding="utf-8",
    )

    mutating_path = tmp_path / "candidate_mutating.py"
    mutating_path.write_text(
        (
            "import numpy as np\n\n"
            "def run_signal_processing(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    filtered = arr[window_size - 1:].copy()\n"
            "    noisy_signal[window_size - 1:] *= 100.0\n"
            "    return {'filtered_signal': filtered}\n"
        ),
        encoding="utf-8",
    )

    baseline_result = sp_evaluator.evaluate(str(baseline_path))
    mutating_result = sp_evaluator.evaluate(str(mutating_path))

    for key in ("combined_score", "score", "overall_score", "lag_error", "avg_error", "noise_reduction"):
        assert mutating_result.metrics[key] == pytest.approx(baseline_result.metrics[key])


def test_evaluator_accepts_candidate_returning_raw_array(monkeypatch, tmp_path):
    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, "sp_step_800_n05")
    candidate_path = tmp_path / "candidate_raw_array.py"
    candidate_path.write_text(
        (
            "import numpy as np\n\n"
            "def process_signal(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    return arr[window_size - 1:]\n"
        ),
        encoding="utf-8",
    )

    result = sp_evaluator.evaluate(str(candidate_path))

    assert result.metrics["success_rate"] == pytest.approx(1.0)
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_accepts_candidate_returning_dict(monkeypatch, tmp_path):
    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, "sp_trend_sine_500_n02")
    candidate_path = tmp_path / "candidate_dict.py"
    candidate_path.write_text(
        (
            "import numpy as np\n\n"
            "def run_signal_processing(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    return {'filtered_signal': arr[window_size - 1:]}\n"
        ),
        encoding="utf-8",
    )

    result = sp_evaluator.evaluate(str(candidate_path))

    assert result.metrics["success_rate"] == pytest.approx(1.0)
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_results"][0]["error"] is None


def test_invalid_output_length_causes_failed_trial(monkeypatch, tmp_path):
    monkeypatch.setenv(SIGNAL_PROCESSING_TASK_SELECTOR_ENV_VAR, "sp_chirp_700_n04")
    candidate_path = tmp_path / "candidate_bad_length.py"
    candidate_path.write_text(
        (
            "import numpy as np\n\n"
            "def run_signal_processing(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    return {'filtered_signal': arr}\n"
        ),
        encoding="utf-8",
    )

    result = sp_evaluator.evaluate(str(candidate_path))

    assert result.metrics["success_rate"] == pytest.approx(0.0)
    assert result.metrics["score"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["error"] == "All 3 trials failed"


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "signal_processing_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "signal_processing_emo_sta" / "evaluator.py"

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
            for task in SIGNAL_PROCESSING_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"program_{program_index}",
            code=(
                "import numpy as np\n\n"
                "def run_signal_processing(noisy_signal, window_size=20):\n"
                "    arr = np.asarray(noisy_signal, dtype=float)\n"
                "    return {'filtered_signal': arr[window_size - 1:]}\n"
            ),
            changes_description=f"program {program_index}",
            language="python",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "evaluation_stage": "full",
                    "task_results": task_results,
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
        family="signal_processing",
        task_ids=["sp_trend_sine_500_n02"],
        initial_program=REPO_ROOT / "examples" / "signal_processing_emo_sta" / "initial_program.py",
    )

    assert "sp_trend_sine_500_n02" in spawn_results
    spawned_checkpoint = spawned_root / "sp_trend_sine_500_n02"
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
        expected_metrics = source_metrics_by_program[program_id]["sp_trend_sine_500_n02"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["overall_score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "sp_trend_sine_500_n02"
        assert program.artifact_dir is None

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "sp_trend_sine_500_n02"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "sp_trend_sine_500_n02"
        assert task_artifacts["task_results"][0]["metrics"]["combined_score"] == pytest.approx(
            expected_metrics["combined_score"]
        )


def test_extract_task_result_returns_none_for_malformed_stored_metrics():
    assert extract_task_result(
        {
            "evaluation_stage": "full",
            "task_results": [
                {
                    "task_id": "sp_trend_sine_500_n02",
                    "metrics": "bad",
                }
            ],
        },
        "sp_trend_sine_500_n02",
    ) is None

    assert extract_task_result(
        {
            "evaluation_stage": "full",
            "task_results": [
                {
                    "task_id": "sp_trend_sine_500_n02",
                    "metrics": {
                        "composite_score": 0.5,
                        "overall_score": 0.6,
                        "slope_changes": 10.0,
                    },
                }
            ],
        },
        "sp_trend_sine_500_n02",
    ) is None


def test_reevaluate_program_for_task_supports_plain_dict_evaluate(tmp_path, monkeypatch):
    task_results = [
        build_task_result(
            task,
            raw_metrics=_fake_raw_metrics(task.task_index, 0),
        )
        for task in SIGNAL_PROCESSING_TASK_SPECS
    ]
    evaluation_result = {
        "metrics": aggregate_task_results(task_results),
        "artifacts": {
            "task_selector": "all",
            "evaluation_stage": "full",
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
        code=(
            "import numpy as np\n\n"
            "def process_signal(noisy_signal, window_size=20):\n"
            "    arr = np.asarray(noisy_signal, dtype=float)\n"
            "    return arr[window_size - 1:]\n"
        ),
        changes_description="plain dict evaluator compatibility",
        language="python",
        metrics={"score": 0.0, "combined_score": 0.0},
    )

    task_result = emo_sta_spawn._reevaluate_program_for_task(
        program=program,
        task_id="sp_trend_sine_500_n02",
        family="signal_processing",
        evaluation_file=tmp_path / "unused_evaluator.py",
    )

    assert task_result["task_id"] == "sp_trend_sine_500_n02"
    assert task_result["metrics"]["combined_score"] == pytest.approx(
        task_result["metrics"]["score"]
    )
    assert task_result["metrics"]["combined_score"] == pytest.approx(
        task_result["metrics"]["overall_score"]
    )
