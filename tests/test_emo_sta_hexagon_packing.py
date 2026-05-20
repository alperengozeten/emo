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
from openevolve.multi_task_shared_then_adapt.hexagon_packing import (
    HEXAGON_PACKING_SHARED_SELECTOR,
    HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR,
    HEXAGON_PACKING_TASK_SPECS,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    minimum_outer_side_length_for_area,
    resolve_task_specs,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_hexagon_packing_evaluator_module():
    evaluator_path = REPO_ROOT / "examples" / "hexagon_packing_emo_sta" / "evaluator.py"
    spec = importlib.util.spec_from_file_location(
        "hexagon_packing_emo_sta_eval_test",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hexagon_packing_evaluator = _load_hexagon_packing_evaluator_module()


def _write_candidate(tmp_path: Path, code: str, name: str = "candidate.py") -> Path:
    candidate_path = tmp_path / name
    candidate_path.write_text(code, encoding="utf-8")
    return candidate_path


def _valid_candidate_code(
    *,
    return_mode: str = "tuple",
    outer_side_length_expr: str = "12.0",
) -> str:
    if return_mode not in {"tuple", "dict"}:
        raise ValueError(f"Unsupported return_mode {return_mode!r}")

    return_statement = (
        "    return {\n"
        '        "inner_hex_data": inner_hex_data,\n'
        '        "outer_hex_data": outer_hex_data,\n'
        '        "outer_hex_side_length": outer_hex_side_length,\n'
        "    }\n"
        if return_mode == "dict"
        else "    return inner_hex_data, outer_hex_data, outer_hex_side_length\n"
    )
    return (
        "import math\n"
        "import numpy as np\n\n"
        "def _axial_to_cartesian(q, r):\n"
        "    return np.asarray((1.5 * float(q), math.sqrt(3.0) * (float(r) + 0.5 * float(q))), dtype=float)\n\n"
        "def _centers(n):\n"
        "    ring = max(2, int(np.ceil(np.sqrt(max(1, n)))) + 2)\n"
        "    candidates = []\n"
        "    for q in range(-ring, ring + 1):\n"
        "        for r in range(-ring, ring + 1):\n"
        "            s = -q - r\n"
        "            if max(abs(q), abs(r), abs(s)) > ring:\n"
        "                continue\n"
        "            center = _axial_to_cartesian(q, r)\n"
        "            candidates.append((float(np.linalg.norm(center)), abs(q) + abs(r) + abs(s), abs(q), abs(r), center))\n"
        "    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], float(item[4][0]), float(item[4][1])))\n"
        "    return np.asarray([item[-1] for item in candidates[:n]], dtype=float)\n\n"
        "def construct_hexagon_packing(n):\n"
        "    n = int(n)\n"
        "    centers = _centers(n)\n"
        "    inner_hex_data = np.column_stack((centers, np.zeros(n, dtype=float)))\n"
        "    outer_hex_data = np.asarray((0.0, 0.0, 0.0), dtype=float)\n"
        f"    outer_hex_side_length = float({outer_side_length_expr})\n"
        f"{return_statement}\n"
        "def run_hexagon_packing(n):\n"
        "    return construct_hexagon_packing(n)\n"
    )


def _fake_raw_metrics(task, program_index: int) -> dict[str, float]:
    ratio = 0.70 + 0.05 * task.task_index + 0.02 * program_index
    outer_side_length = task.target_outer_side_length / ratio
    return {
        "outer_side_length": outer_side_length,
        "validity": 1.0,
        "center_spread": 0.10 + 0.05 * task.task_index,
        "angle_spread": 0.0,
        "min_center_distance": 0.35 + 0.05 * task.task_index,
        "eval_time": 0.05 + 0.01 * task.task_index + 0.01 * program_index,
    }


def test_family_registry_resolves_hexagon_packing():
    family = get_family_definition("hexagon_packing")
    assert family.family == "hexagon_packing"
    assert family.task_selector_env_var == HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR
    assert family.shared_selector == HEXAGON_PACKING_SHARED_SELECTOR
    assert [task.task_id for task in family.task_specs] == [
        "hex_pack_n10",
        "hex_pack_n11",
        "hex_pack_n12",
        "hex_pack_n13",
    ]


def test_resolve_task_specs_all_returns_expected_tasks():
    resolved = resolve_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "hex_pack_n10",
        "hex_pack_n11",
        "hex_pack_n12",
        "hex_pack_n13",
    ]


def test_shared_mode_returns_aggregate_metrics_and_task_artifacts(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(return_mode="tuple"))
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "all")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert result.artifacts["evaluation_stage"] == "full"
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in HEXAGON_PACKING_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        metrics = task_result["metrics"]
        assert metrics["validity"] == pytest.approx(1.0)
        assert metrics["score"] == pytest.approx(metrics["target_ratio"])
        assert metrics["combined_score"] == pytest.approx(metrics["score"])
        assert "inner_hex_data" not in task_result
        assert "outer_hex_data" not in task_result


def test_task_specific_mode_returns_one_task(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(return_mode="tuple"))
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n11")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == "hex_pack_n11"
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == "hex_pack_n11"


def test_successful_valid_evaluation_uses_target_ratio_as_score(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(return_mode="tuple"))
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n10")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["score"] == pytest.approx(result.metrics["target_ratio"])
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["target_ratio"])


def test_evaluator_accepts_candidate_returning_tuple(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(return_mode="tuple"))
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n12")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["outer_side_length"] == pytest.approx(12.0)
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_accepts_candidate_returning_dict(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(return_mode="dict"))
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n13")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["outer_side_length"] == pytest.approx(12.0)
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_rejects_invalid_shape_outputs(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_hexagon_packing(n):\n"
            "    inner_hex_data = np.zeros((n - 1, 3), dtype=float)\n"
            "    outer_hex_data = np.zeros(3, dtype=float)\n"
            "    return inner_hex_data, outer_hex_data, 12.0\n"
        ),
        name="candidate_bad_shape.py",
    )
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n10")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["shape_valid"] is False


def test_evaluator_rejects_nonfinite_outputs(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_hexagon_packing(n):\n"
            "    inner_hex_data = np.zeros((n, 3), dtype=float)\n"
            "    inner_hex_data[0, 0] = np.nan\n"
            "    outer_hex_data = np.zeros(3, dtype=float)\n"
            "    return inner_hex_data, outer_hex_data, 12.0\n"
        ),
        name="candidate_nonfinite.py",
    )
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n11")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["finite_valid"] is False


def test_evaluator_rejects_nonpositive_outer_side_length(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        _valid_candidate_code(return_mode="tuple", outer_side_length_expr="0.0"),
        name="candidate_bad_outer_side.py",
    )
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n12")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["outer_side_valid"] is False


def test_evaluator_rejects_overlapping_inner_hexagons(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_hexagon_packing(n):\n"
            "    inner_hex_data = np.zeros((n, 3), dtype=float)\n"
            "    outer_hex_data = np.zeros(3, dtype=float)\n"
            "    return inner_hex_data, outer_hex_data, 12.0\n"
        ),
        name="candidate_overlap.py",
    )
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n10")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["overlap_violations"] > 0


def test_evaluator_rejects_inner_hexagons_outside_outer_hexagon(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        _valid_candidate_code(return_mode="tuple", outer_side_length_expr="3.7"),
        name="candidate_outside.py",
    )
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n13")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["containment_violations"] > 0


def test_evaluator_rejects_tiny_outer_side_length_reward_hack(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_hexagon_packing(n):\n"
            "    centers = np.column_stack((np.arange(n, dtype=float) * 10.0, np.zeros(n, dtype=float)))\n"
            "    inner_hex_data = np.column_stack((centers, np.zeros(n, dtype=float)))\n"
            "    outer_hex_data = np.zeros(3, dtype=float)\n"
            "    return inner_hex_data, outer_hex_data, 1.0e-16\n"
        ),
        name="candidate_tiny_outer_side.py",
    )
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n10")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    validation_summary = result.artifacts["task_results"][0]["validation_summary"]
    assert validation_summary["outer_side_valid"] is False
    assert validation_summary["outer_side_lower_bound"] == pytest.approx(math.sqrt(10.0))


def test_metric_normalization_rejects_area_lower_bound_violation():
    task = HEXAGON_PACKING_TASK_SPECS[0]

    result = build_task_result(
        task,
        raw_metrics={
            "outer_side_length": minimum_outer_side_length_for_area(task.n_hexagons) / 2.0,
            "validity": 1.0,
            "center_spread": 1.0,
            "angle_spread": 0.0,
            "min_center_distance": 1.0,
            "eval_time": 0.1,
        },
    )

    assert result["metrics"]["validity"] == pytest.approx(0.0)
    assert result["metrics"]["score"] == pytest.approx(0.0)
    assert result["final_task_score"] == pytest.approx(0.0)


def test_evaluator_allows_boundary_contact_within_tolerance(monkeypatch):
    candidate_path = REPO_ROOT / "examples" / "hexagon_packing_emo_sta" / "initial_program.py"
    monkeypatch.setenv(HEXAGON_PACKING_TASK_SELECTOR_ENV_VAR, "hex_pack_n10")

    result = hexagon_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.artifacts["task_results"][0]["error"] is None


def test_periodic_angle_spread_treats_wraparound_rotations_as_close():
    spread = hexagon_packing_evaluator._periodic_angle_spread(
        [0.0, 59.9],
        period_degrees=60.0,
    )

    assert spread < 0.01


def test_extract_task_result_returns_none_on_malformed_stored_metrics():
    assert extract_task_result(
        {"task_results": [{"task_id": "hex_pack_n10", "metrics": "bad"}]},
        "hex_pack_n10",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "hex_pack_n10",
                    "metrics": {
                        "outer_side_length": 4.5,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "score": float("nan"),
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "hex_pack_n10",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "hex_pack_n10",
                    "metrics": {
                        "outer_side_length": 4.5,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "hex_pack_n10",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "hex_pack_n10",
                    "metrics": {
                        "outer_side_length": 0.0,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "score": 0.4,
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "hex_pack_n10",
    ) is None


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "hexagon_packing_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "hexagon_packing_emo_sta" / "evaluator.py"
    initial_program = REPO_ROOT / "examples" / "hexagon_packing_emo_sta" / "initial_program.py"

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
                    "shape_valid": True,
                    "finite_valid": True,
                    "outer_side_valid": True,
                    "containment_violations": 0,
                    "overlap_violations": 0,
                    "reported_n": task.n_hexagons,
                    "expected_n": task.n_hexagons,
                },
            )
            for task in HEXAGON_PACKING_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"hex_program_{program_index}",
            code=_valid_candidate_code(return_mode="tuple"),
            changes_description=f"hex program {program_index}",
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
        family="hexagon_packing",
        task_ids=["hex_pack_n12"],
        initial_program=initial_program,
    )

    assert "hex_pack_n12" in spawn_results
    spawned_checkpoint = spawned_root / "hex_pack_n12"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["hex_pack_n12"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "hex_pack_n12"

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "hex_pack_n12"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "hex_pack_n12"
        assert "inner_hex_data" not in task_artifacts["task_results"][0]
        assert "outer_hex_data" not in task_artifacts["task_results"][0]
