import importlib.util
import json
import math
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
from openevolve.multi_task_shared_then_adapt.heilbronn_triangle import (
    HEILBRONN_TRIANGLE_SHARED_SELECTOR,
    HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR,
    HEILBRONN_TRIANGLE_TASK_SPECS,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    resolve_task_specs,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_heilbronn_triangle_evaluator_module():
    evaluator_path = REPO_ROOT / "examples" / "heilbronn_triangle_emo_sta" / "evaluator.py"
    spec = importlib.util.spec_from_file_location(
        "heilbronn_triangle_emo_sta_eval_test",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heilbronn_triangle_evaluator = _load_heilbronn_triangle_evaluator_module()

_VALID_BARY = [
    [0.72, 0.18, 0.10],
    [0.18, 0.70, 0.12],
    [0.16, 0.14, 0.70],
    [0.38, 0.32, 0.30],
    [0.55, 0.27, 0.18],
    [0.24, 0.52, 0.24],
    [0.29, 0.19, 0.52],
    [0.44, 0.11, 0.45],
    [0.12, 0.46, 0.42],
    [0.41, 0.43, 0.16],
    [0.25, 0.33, 0.42],
    [0.33, 0.49, 0.18],
]


def _write_candidate(tmp_path: Path, code: str, name: str = "candidate.py") -> Path:
    candidate_path = tmp_path / name
    candidate_path.write_text(code, encoding="utf-8")
    return candidate_path


def _valid_points_array(n: int):
    import numpy as np

    bary = np.asarray(_VALID_BARY[:n], dtype=float)
    return np.column_stack((2.0 * bary[:, 1], bary[:, 2]))


def _valid_candidate_code(*, include_min_area: bool, reported_min_area: str | None = None) -> str:
    points_expr = "points" if include_min_area else "points.tolist()"
    if include_min_area:
        reported_expr = reported_min_area or "0.0"
        return (
            "import numpy as np\n\n"
            "_BARY = np.asarray([\n"
            "    [0.72, 0.18, 0.10],\n"
            "    [0.18, 0.70, 0.12],\n"
            "    [0.16, 0.14, 0.70],\n"
            "    [0.38, 0.32, 0.30],\n"
            "    [0.55, 0.27, 0.18],\n"
            "    [0.24, 0.52, 0.24],\n"
            "    [0.29, 0.19, 0.52],\n"
            "    [0.44, 0.11, 0.45],\n"
            "    [0.12, 0.46, 0.42],\n"
            "    [0.41, 0.43, 0.16],\n"
            "    [0.25, 0.33, 0.42],\n"
            "    [0.33, 0.49, 0.18],\n"
            "], dtype=float)\n\n"
            "def _points(n):\n"
            "    bary = _BARY[:n]\n"
            "    return np.column_stack((2.0 * bary[:, 1], bary[:, 2]))\n\n"
            "def construct_points(n):\n"
            "    points = _points(n)\n"
            f"    return {points_expr}, {reported_expr}\n\n"
            "def run_heilbronn(n):\n"
            "    return construct_points(n)\n"
        )
    return (
        "import numpy as np\n\n"
        "_BARY = np.asarray([\n"
        "    [0.72, 0.18, 0.10],\n"
        "    [0.18, 0.70, 0.12],\n"
        "    [0.16, 0.14, 0.70],\n"
        "    [0.38, 0.32, 0.30],\n"
        "    [0.55, 0.27, 0.18],\n"
        "    [0.24, 0.52, 0.24],\n"
        "    [0.29, 0.19, 0.52],\n"
        "    [0.44, 0.11, 0.45],\n"
        "    [0.12, 0.46, 0.42],\n"
        "    [0.41, 0.43, 0.16],\n"
        "    [0.25, 0.33, 0.42],\n"
        "    [0.33, 0.49, 0.18],\n"
        "], dtype=float)\n\n"
        "def _points(n):\n"
        "    bary = _BARY[:n]\n"
        "    return np.column_stack((2.0 * bary[:, 1], bary[:, 2]))\n\n"
        "def construct_points(n):\n"
        "    points = _points(n)\n"
        "    return points.tolist()\n\n"
        "def run_heilbronn(n):\n"
        "    return construct_points(n)\n"
    )


def _fake_raw_metrics(task, program_index: int) -> dict[str, float]:
    ratio = 0.55 + 0.04 * task.task_index + 0.03 * program_index
    return {
        "min_triangle_area": task.target_min_area * ratio,
        "validity": 1.0,
        "point_spread": 0.20 + 0.05 * task.task_index,
        "boundary_utilization": 0.55 + 0.04 * task.task_index,
        "min_pair_distance": 0.10 + 0.03 * task.task_index,
        "eval_time": 0.05 + 0.01 * task.task_index + 0.01 * program_index,
    }


def test_family_registry_resolves_heilbronn_triangle():
    family = get_family_definition("heilbronn_triangle")
    assert family.family == "heilbronn_triangle"
    assert family.task_selector_env_var == HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR
    assert family.shared_selector == HEILBRONN_TRIANGLE_SHARED_SELECTOR
    assert [task.task_id for task in family.task_specs] == [
        "heil_tri_n9",
        "heil_tri_n10",
        "heil_tri_n11",
        "heil_tri_n12",
    ]


def test_resolve_task_specs_all_returns_expected_tasks():
    resolved = resolve_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "heil_tri_n9",
        "heil_tri_n10",
        "heil_tri_n11",
        "heil_tri_n12",
    ]


def test_shared_mode_returns_aggregate_metrics_and_task_artifacts(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_min_area=True))
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "all")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert result.artifacts["evaluation_stage"] == "full"
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in HEILBRONN_TRIANGLE_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        metrics = task_result["metrics"]
        assert metrics["validity"] == pytest.approx(1.0)
        assert metrics["score"] == pytest.approx(metrics["target_ratio"])
        assert metrics["combined_score"] == pytest.approx(metrics["score"])
        assert "points" not in task_result


def test_task_specific_mode_returns_one_task(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_min_area=True))
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n11")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == "heil_tri_n11"
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == "heil_tri_n11"


def test_successful_valid_evaluation_uses_target_ratio_as_score(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_min_area=True))
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n9")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["score"] == pytest.approx(result.metrics["target_ratio"])
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["target_ratio"])


def test_evaluator_accepts_candidate_returning_points_only(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_min_area=False))
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n10")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["min_triangle_area"] > 0.0
    assert result.artifacts["task_results"][0]["error"] is None
    assert result.artifacts["task_results"][0]["validation_summary"]["shape_valid"] is True
    assert result.artifacts["task_results"][0]["validation_summary"]["finite_values_valid"] is True
    assert result.artifacts["task_results"][0]["validation_summary"]["containment_valid"] is True


def test_evaluator_accepts_candidate_returning_points_and_min_area(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_min_area=True))
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n12")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["min_triangle_area"] > 0.0
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_rejects_invalid_shape_outputs(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_heilbronn(n):\n"
            "    return np.zeros((n - 1, 2), dtype=float)\n"
        ),
        name="candidate_bad_shape.py",
    )
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n11")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["shape_valid"] is False
    assert result.artifacts["task_results"][0]["validation_summary"]["finite_values_valid"] is False
    assert result.artifacts["task_results"][0]["validation_summary"]["containment_valid"] is False


def test_evaluator_rejects_points_outside_canonical_triangle(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_heilbronn(n):\n"
            "    points = np.zeros((n, 2), dtype=float)\n"
            "    for i in range(n):\n"
            "        points[i] = np.array([0.1 * i, 0.05], dtype=float)\n"
            "    points[0] = np.array([1.9, 0.2], dtype=float)\n"
            "    return points\n"
        ),
        name="candidate_outside.py",
    )
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n10")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["boundary_violations"] > 0
    assert result.artifacts["task_results"][0]["validation_summary"]["shape_valid"] is True
    assert result.artifacts["task_results"][0]["validation_summary"]["finite_values_valid"] is True
    assert result.artifacts["task_results"][0]["validation_summary"]["containment_valid"] is False


def test_duplicate_or_collinear_point_sets_are_valid_but_zero_score(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_heilbronn(n):\n"
            "    xs = np.linspace(0.05, 1.15, n)\n"
            "    points = np.column_stack((xs, np.full(n, 0.1, dtype=float)))\n"
            "    return points\n"
        ),
        name="candidate_collinear.py",
    )
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n12")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["min_triangle_area"] == pytest.approx(0.0)
    assert result.metrics["score"] == pytest.approx(0.0)


def test_evaluator_uses_evaluator_computed_min_area_when_reported_value_is_inconsistent(
    monkeypatch,
    tmp_path,
):
    candidate_path = _write_candidate(
        tmp_path,
        _valid_candidate_code(include_min_area=True, reported_min_area="123.456"),
        name="candidate_bad_min_area.py",
    )
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n9")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["min_triangle_area"] < 1.0
    assert result.artifacts["task_results"][0]["validation_summary"][
        "reported_min_area_mismatch"
    ] is True


def test_small_reported_min_area_roundoff_does_not_trigger_mismatch(monkeypatch, tmp_path):
    actual_min_area = heilbronn_triangle_evaluator.compute_min_triangle_area(_valid_points_array(9))
    candidate_path = _write_candidate(
        tmp_path,
        _valid_candidate_code(
            include_min_area=True,
            reported_min_area=repr(actual_min_area + 5.0e-8),
        ),
        name="candidate_nearly_equal_min_area.py",
    )
    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "heil_tri_n9")

    result = heilbronn_triangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.artifacts["task_results"][0]["validation_summary"][
        "reported_min_area_mismatch"
    ] is False


def test_extract_task_result_returns_none_on_malformed_stored_metrics():
    assert extract_task_result(
        {"task_results": [{"task_id": "heil_tri_n9", "metrics": "bad"}]},
        "heil_tri_n9",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "heil_tri_n9",
                    "metrics": {
                        "min_triangle_area": 0.01,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "score": float("nan"),
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "heil_tri_n9",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "heil_tri_n9",
                    "metrics": {
                        "min_triangle_area": 0.01,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "heil_tri_n9",
    ) is None


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "heilbronn_triangle_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "heilbronn_triangle_emo_sta" / "evaluator.py"
    initial_program = REPO_ROOT / "examples" / "heilbronn_triangle_emo_sta" / "initial_program.py"

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)

    source_metrics_by_program: dict[str, dict[str, dict[str, float]]] = {}
    for program_index in range(2):
        task_results = [
            build_task_result(
                task,
                raw_metrics=_fake_raw_metrics(task, program_index),
                validation_summary={
                    "boundary_violations": 0,
                    "nonfinite_points": 0,
                    "shape_valid": True,
                    "finite_values_valid": True,
                    "containment_valid": True,
                    "reported_min_area_mismatch": False,
                },
            )
            for task in HEILBRONN_TRIANGLE_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"heil_program_{program_index}",
            code=_valid_candidate_code(include_min_area=True),
            changes_description=f"heil program {program_index}",
            language="python",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "evaluation_mode": "shared",
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
    shared_database.save(str(shared_checkpoint), iteration=4)

    def fail_if_reevaluated(**kwargs):
        raise AssertionError("Spawn should use stored task_results instead of reevaluation")

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
        family="heilbronn_triangle",
        task_ids=["heil_tri_n11"],
        initial_program=initial_program,
    )

    assert "heil_tri_n11" in spawn_results
    spawned_checkpoint = spawned_root / "heil_tri_n11"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["heil_tri_n11"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "heil_tri_n11"

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "heil_tri_n11"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "heil_tri_n11"
        assert "points" not in task_artifacts["task_results"][0]
