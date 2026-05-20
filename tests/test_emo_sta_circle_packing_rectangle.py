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
from openevolve.multi_task_shared_then_adapt.circle_packing_rectangle import (
    CIRCLE_PACKING_RECTANGLE_SHARED_SELECTOR,
    CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR,
    CIRCLE_PACKING_RECTANGLE_TASK_SPECS,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    resolve_task_specs,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints
from openevolve.multi_task_shared_then_adapt.workflow import (
    fair_emo_sta_baseline_iterations,
    load_manifest,
    validate_emo_sta_iteration_budget,
)


def _load_circle_packing_rectangle_evaluator_module():
    evaluator_path = (
        REPO_ROOT / "examples" / "circle_packing_rectangle_emo_sta" / "evaluator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "circle_packing_rectangle_emo_sta_eval_test",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


circle_packing_rectangle_evaluator = _load_circle_packing_rectangle_evaluator_module()


def _write_candidate(tmp_path: Path, code: str, name: str = "candidate.py") -> Path:
    candidate_path = tmp_path / name
    candidate_path.write_text(code, encoding="utf-8")
    return candidate_path


def _valid_rectangle_candidate_code(
    *,
    include_sum: bool,
    reported_sum: str | None = None,
    alpha_expr: str = "0.9",
) -> str:
    if include_sum:
        reported_sum_expr = reported_sum or "float(np.sum(radii))"
        return (
            "import numpy as np\n\n"
            "def construct_packing(n):\n"
            f"    alpha = float({alpha_expr})\n"
            "    height = 2.0 - alpha\n"
            "    cols = max(1, int(np.ceil(np.sqrt(max(1, n) * alpha / height))))\n"
            "    rows = int(np.ceil(n / cols))\n"
            "    x_margin = 0.05 * alpha\n"
            "    y_margin = 0.05 * height\n"
            "    xs = np.linspace(x_margin, alpha - x_margin, cols) if cols > 1 else np.array([0.5 * alpha])\n"
            "    ys = np.linspace(y_margin, height - y_margin, rows) if rows > 1 else np.array([0.5 * height])\n"
            "    centers = []\n"
            "    for y in ys:\n"
            "        for x in xs:\n"
            "            centers.append((float(x), float(y)))\n"
            "    centers = np.asarray(centers[:n], dtype=float)\n"
            "    radii = np.full(n, 0.01, dtype=float)\n"
            f"    return centers, radii, alpha, {reported_sum_expr}\n\n"
            "def run_packing(n):\n"
            "    return construct_packing(n)\n"
        )
    return (
        "import numpy as np\n\n"
        "def construct_packing(n):\n"
        f"    alpha = float({alpha_expr})\n"
        "    height = 2.0 - alpha\n"
        "    cols = max(1, int(np.ceil(np.sqrt(max(1, n) * alpha / height))))\n"
        "    rows = int(np.ceil(n / cols))\n"
        "    x_margin = 0.05 * alpha\n"
        "    y_margin = 0.05 * height\n"
        "    xs = np.linspace(x_margin, alpha - x_margin, cols) if cols > 1 else np.array([0.5 * alpha])\n"
        "    ys = np.linspace(y_margin, height - y_margin, rows) if rows > 1 else np.array([0.5 * height])\n"
        "    centers = []\n"
        "    for y in ys:\n"
        "        for x in xs:\n"
        "            centers.append((float(x), float(y)))\n"
        "    centers = np.asarray(centers[:n], dtype=float)\n"
        "    radii = np.full(n, 0.01, dtype=float)\n"
        "    return centers, radii, alpha\n\n"
        "def run_packing(n):\n"
        "    return construct_packing(n)\n"
    )


def _fake_raw_metrics(task, program_index: int) -> dict[str, float]:
    return {
        "sum_radii": 1.50 + 0.04 * task.task_index + 0.02 * program_index,
        "validity": 1.0,
        "alpha": 0.82 + 0.03 * task.task_index,
        "radius_variance": 0.20 + 0.05 * task.task_index,
        "spatial_spread": 0.30 + 0.04 * task.task_index,
        "min_radius": 0.01 + 0.002 * task.task_index,
        "max_radius": 0.08 + 0.004 * task.task_index,
        "eval_time": 0.05 + 0.01 * task.task_index + 0.01 * program_index,
    }


def test_family_registry_resolves_circle_packing_rectangle():
    family = get_family_definition("circle_packing_rectangle")
    assert family.family == "circle_packing_rectangle"
    assert family.task_selector_env_var == CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR
    assert family.shared_selector == CIRCLE_PACKING_RECTANGLE_SHARED_SELECTOR
    assert [task.task_id for task in family.task_specs] == [
        "cp_rect_n20",
        "cp_rect_n21",
        "cp_rect_n22",
        "cp_rect_n23",
    ]


def test_resolve_task_specs_all_returns_expected_tasks():
    resolved = resolve_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "cp_rect_n20",
        "cp_rect_n21",
        "cp_rect_n22",
        "cp_rect_n23",
    ]


def test_rectangle_manifest_defaults_are_iteration_fair():
    manifest = load_manifest(
        REPO_ROOT / "multi_task_shared_then_adapt" / "configs" / "circle_packing_rectangle_emo_sta.yaml"
    )
    task_count = len(resolve_task_specs("all"))

    assert fair_emo_sta_baseline_iterations(
        task_count=task_count,
        shared_iterations=manifest.default_shared_iterations,
        adaptation_iterations=manifest.default_adaptation_iterations,
    ) == manifest.default_baseline_iterations

    validate_emo_sta_iteration_budget(
        task_count=task_count,
        shared_iterations=manifest.default_shared_iterations,
        adaptation_iterations=manifest.default_adaptation_iterations,
        baseline_iterations=manifest.default_baseline_iterations,
    )


def test_shared_mode_returns_aggregate_metrics_and_task_artifacts(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_rectangle_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "all")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert result.artifacts["evaluation_stage"] == "full"
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in CIRCLE_PACKING_RECTANGLE_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        metrics = task_result["metrics"]
        assert metrics["validity"] == pytest.approx(1.0)
        assert metrics["score"] == pytest.approx(metrics["target_ratio"])
        assert metrics["combined_score"] == pytest.approx(metrics["score"])
        assert "centers" not in task_result
        assert "radii" not in task_result


def test_task_specific_mode_returns_one_task(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_rectangle_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n22")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == "cp_rect_n22"
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == "cp_rect_n22"


def test_successful_valid_evaluation_uses_target_ratio_as_score(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_rectangle_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n20")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["score"] == pytest.approx(result.metrics["target_ratio"])
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["target_ratio"])


def test_evaluator_accepts_candidate_returning_centers_radii_alpha_only(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_rectangle_candidate_code(include_sum=False))
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n21")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["sum_radii"] == pytest.approx(0.21)
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_accepts_candidate_returning_explicit_sum(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_rectangle_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n23")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["sum_radii"] == pytest.approx(0.23)
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_rejects_invalid_shape_outputs(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_packing(n):\n"
            "    alpha = 0.9\n"
            "    centers = np.zeros((n - 1, 2), dtype=float)\n"
            "    radii = np.full(n, 0.01, dtype=float)\n"
            "    return centers, radii, alpha\n"
        ),
        name="candidate_bad_shape.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n23")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    assert "shape_message" in result.artifacts["task_results"][0]["validation_summary"]


def test_evaluator_rejects_invalid_alpha_values(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        _valid_rectangle_candidate_code(include_sum=False, alpha_expr="1.2"),
        name="candidate_bad_alpha.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n20")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["alpha_valid"] is False


def test_evaluator_rejects_overlapping_circles(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_packing(n):\n"
            "    alpha = 0.9\n"
            "    centers = np.full((n, 2), [0.45, 0.55], dtype=float)\n"
            "    radii = np.full(n, 0.05, dtype=float)\n"
            "    return centers, radii, alpha\n"
        ),
        name="candidate_overlap.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n20")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["overlap_violations"] > 0


def test_evaluator_rejects_circles_outside_rectangle(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_packing(n):\n"
            "    alpha = 0.9\n"
            "    height = 2.0 - alpha\n"
            "    centers = np.tile(np.array([[0.45, 0.55]], dtype=float), (n, 1))\n"
            "    centers[0] = np.array([alpha - 0.005, height - 0.005], dtype=float)\n"
            "    radii = np.full(n, 0.03, dtype=float)\n"
            "    return centers, radii, alpha\n"
        ),
        name="candidate_outside.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n21")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["boundary_violations"] > 0


def test_evaluator_uses_evaluator_computed_sum_when_reported_sum_is_inconsistent(
    monkeypatch,
    tmp_path,
):
    candidate_path = _write_candidate(
        tmp_path,
        _valid_rectangle_candidate_code(include_sum=True, reported_sum="123.456"),
        name="candidate_bad_sum.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "cp_rect_n20")

    result = circle_packing_rectangle_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["sum_radii"] == pytest.approx(0.20)
    assert result.artifacts["task_results"][0]["validation_summary"]["sum_mismatch"] is True


def test_extract_task_result_returns_none_on_malformed_stored_metrics():
    assert extract_task_result(
        {"task_results": [{"task_id": "cp_rect_n20", "metrics": "bad"}]},
        "cp_rect_n20",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "cp_rect_n20",
                    "metrics": {
                        "sum_radii": 1.0,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "score": 0.4,
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "cp_rect_n20",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "cp_rect_n20",
                    "metrics": {
                        "sum_radii": 1.0,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "alpha": float("nan"),
                        "score": 0.4,
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "cp_rect_n20",
    ) is None


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "circle_packing_rectangle_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "circle_packing_rectangle_emo_sta" / "evaluator.py"
    initial_program = (
        REPO_ROOT / "examples" / "circle_packing_rectangle_emo_sta" / "initial_program.py"
    )

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
                    "alpha_valid": True,
                    "boundary_violations": 0,
                    "overlap_violations": 0,
                    "sum_mismatch": False,
                },
            )
            for task in CIRCLE_PACKING_RECTANGLE_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"circle_rect_program_{program_index}",
            code=_valid_rectangle_candidate_code(include_sum=True),
            changes_description=f"rectangle circle program {program_index}",
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
        family="circle_packing_rectangle",
        task_ids=["cp_rect_n22"],
        initial_program=initial_program,
    )

    assert "cp_rect_n22" in spawn_results
    spawned_checkpoint = spawned_root / "cp_rect_n22"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["cp_rect_n22"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "cp_rect_n22"

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "cp_rect_n22"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "cp_rect_n22"
        assert "centers" not in task_artifacts["task_results"][0]
        assert "radii" not in task_artifacts["task_results"][0]
