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
from openevolve.multi_task_shared_then_adapt.circle_packing import (
    CIRCLE_PACKING_HOLDOUT_TASK_SPECS,
    CIRCLE_PACKING_SHARED_SELECTOR,
    CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR,
    CIRCLE_PACKING_TASK_SPECS,
    CIRCLE_PACKING_TASKS_BY_ID,
    all_circle_packing_holdout_task_ids,
    aggregate_task_results,
    build_task_result,
    extract_task_result,
    resolve_holdout_task_specs,
    resolve_task_specs,
)
from openevolve.multi_task_shared_then_adapt.holdout_eval import (
    evaluate_best_program_on_holdouts,
    resolve_best_program_path,
    run_circle_packing_holdout_evaluation,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints


def _load_circle_packing_evaluator_module():
    evaluator_path = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("circle_packing_emo_sta_eval_test", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


circle_packing_evaluator = _load_circle_packing_evaluator_module()


def _write_candidate(tmp_path: Path, code: str, name: str = "candidate.py") -> Path:
    candidate_path = tmp_path / name
    candidate_path.write_text(code, encoding="utf-8")
    return candidate_path


def _valid_candidate_code(*, include_sum: bool, reported_sum: str | None = None) -> str:
    if include_sum:
        reported_sum_expr = reported_sum or "float(np.sum(radii))"
        return (
            "import numpy as np\n\n"
            "def construct_packing(n):\n"
            "    cols = int(np.ceil(np.sqrt(n)))\n"
            "    rows = int(np.ceil(n / cols))\n"
            "    xs = np.linspace(0.15, 0.85, cols)\n"
            "    ys = np.linspace(0.15, 0.85, rows)\n"
            "    centers = []\n"
            "    spacing = xs[1] - xs[0] if cols > 1 else 0.0\n"
            "    for row_index, y in enumerate(ys):\n"
            "        offset = 0.5 * spacing if row_index % 2 else 0.0\n"
            "        for x in xs:\n"
            "            centers.append((float(np.clip(x + offset, 0.15, 0.85)), float(y)))\n"
            "    centers = np.asarray(centers[:n], dtype=float)\n"
            "    radii = np.full(n, 0.01, dtype=float)\n"
            f"    return centers, radii, {reported_sum_expr}\n\n"
            "def run_packing(n):\n"
            "    return construct_packing(n)\n"
        )
    return (
        "import numpy as np\n\n"
        "def construct_packing(n):\n"
        "    cols = int(np.ceil(np.sqrt(n)))\n"
        "    rows = int(np.ceil(n / cols))\n"
        "    xs = np.linspace(0.15, 0.85, cols)\n"
        "    ys = np.linspace(0.15, 0.85, rows)\n"
        "    centers = []\n"
        "    spacing = xs[1] - xs[0] if cols > 1 else 0.0\n"
        "    for row_index, y in enumerate(ys):\n"
        "        offset = 0.5 * spacing if row_index % 2 else 0.0\n"
        "        for x in xs:\n"
        "            centers.append((float(np.clip(x + offset, 0.15, 0.85)), float(y)))\n"
        "    centers = np.asarray(centers[:n], dtype=float)\n"
        "    radii = np.full(n, 0.01, dtype=float)\n"
        "    return centers, radii\n\n"
        "def run_packing(n):\n"
        "    return construct_packing(n)\n"
    )


def _fake_raw_metrics(task, program_index: int) -> dict[str, float]:
    return {
        "sum_radii": 1.50 + 0.04 * task.task_index + 0.02 * program_index,
        "validity": 1.0,
        "radius_variance": 0.20 + 0.05 * task.task_index,
        "spatial_spread": 0.30 + 0.04 * task.task_index,
        "min_radius": 0.01 + 0.002 * task.task_index,
        "max_radius": 0.08 + 0.004 * task.task_index,
        "eval_time": 0.05 + 0.01 * task.task_index + 0.01 * program_index,
    }


def test_family_registry_resolves_circle_packing():
    family = get_family_definition("circle_packing")
    assert family.family == "circle_packing"
    assert family.task_selector_env_var == CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR
    assert family.shared_selector == CIRCLE_PACKING_SHARED_SELECTOR
    assert [task.task_id for task in family.task_specs] == [
        "cp_n20",
        "cp_n22",
        "cp_n24",
        "cp_n26",
    ]


def test_family_inference_accepts_holdout_task_ids():
    from openevolve.multi_task_shared_then_adapt.registry import (
        infer_family_from_task_ids,
    )

    assert infer_family_from_task_ids(["cp_n21"]) == "circle_packing"
    assert infer_family_from_task_ids(["cp_n21", "cp_n23"]) == "circle_packing"


def test_resolve_task_specs_all_returns_expected_tasks():
    resolved = resolve_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "cp_n20",
        "cp_n22",
        "cp_n24",
        "cp_n26",
    ]


def test_resolve_holdout_task_specs_all_returns_expected_tasks():
    resolved = resolve_holdout_task_specs("all")
    assert [task.task_id for task in resolved] == [
        "cp_n21",
        "cp_n23",
        "cp_n25",
    ]
    assert all_circle_packing_holdout_task_ids() == [
        "cp_n21",
        "cp_n23",
        "cp_n25",
    ]


def test_holdout_tasks_are_not_part_of_training_family():
    training_task_ids = [task.task_id for task in resolve_task_specs("all")]
    holdout_task_ids = {task.task_id for task in CIRCLE_PACKING_HOLDOUT_TASK_SPECS}

    assert holdout_task_ids.isdisjoint(training_task_ids)
    assert [task.task_id for task in get_family_definition("circle_packing").task_specs] == (
        training_task_ids
    )

    with pytest.raises(ValueError, match="seen training family"):
        resolve_holdout_task_specs("cp_n20")


def test_shared_mode_returns_aggregate_metrics_and_task_artifacts(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "all")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.metrics["task_count"] == pytest.approx(4.0)
    assert result.artifacts["evaluation_mode"] == "shared"
    assert result.artifacts["evaluation_stage"] == "full"
    assert len(result.artifacts["task_results"]) == 4
    assert {task_result["task_id"] for task_result in result.artifacts["task_results"]} == {
        task.task_id for task in CIRCLE_PACKING_TASK_SPECS
    }
    for task_result in result.artifacts["task_results"]:
        metrics = task_result["metrics"]
        assert metrics["validity"] == pytest.approx(1.0)
        assert metrics["score"] == pytest.approx(metrics["target_ratio"])
        assert metrics["combined_score"] == pytest.approx(metrics["score"])
        assert "centers" not in task_result
        assert "radii" not in task_result


def test_task_specific_mode_returns_one_task(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n24")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
    assert result.artifacts["task_selector"] == "cp_n24"
    assert result.artifacts["evaluation_mode"] == "task_specific"
    assert len(result.artifacts["task_results"]) == 1
    assert result.artifacts["task_results"][0]["task_id"] == "cp_n24"


def test_successful_valid_evaluation_uses_target_ratio_as_score(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n20")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["score"] == pytest.approx(result.metrics["target_ratio"])
    assert result.metrics["combined_score"] == pytest.approx(result.metrics["target_ratio"])


def test_evaluator_accepts_candidate_returning_centers_and_radii_only(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_sum=False))
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n22")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["sum_radii"] == pytest.approx(0.22)
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_accepts_candidate_returning_explicit_sum(monkeypatch, tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_sum=True))
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n26")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["sum_radii"] == pytest.approx(0.26)
    assert result.artifacts["task_results"][0]["error"] is None


def test_evaluator_rejects_invalid_shape_outputs(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_packing(n):\n"
            "    centers = np.zeros((n - 1, 2), dtype=float)\n"
            "    radii = np.full(n, 0.01, dtype=float)\n"
            "    return centers, radii\n"
        ),
        name="candidate_bad_shape.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n24")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.metrics["combined_score"] == pytest.approx(0.0)
    assert "shape_message" in result.artifacts["task_results"][0]["validation_summary"]


def test_evaluator_rejects_overlapping_circles(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_packing(n):\n"
            "    centers = np.full((n, 2), 0.5, dtype=float)\n"
            "    radii = np.full(n, 0.05, dtype=float)\n"
            "    return centers, radii\n"
        ),
        name="candidate_overlap.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n20")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["overlap_violations"] > 0


def test_evaluator_rejects_circles_outside_unit_square(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        (
            "import numpy as np\n\n"
            "def run_packing(n):\n"
            "    centers = np.tile(np.array([[0.5, 0.5]], dtype=float), (n, 1))\n"
            "    centers[0] = np.array([0.99, 0.99], dtype=float)\n"
            "    radii = np.full(n, 0.03, dtype=float)\n"
            "    return centers, radii\n"
        ),
        name="candidate_outside.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n22")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(0.0)
    assert result.artifacts["task_results"][0]["validation_summary"]["boundary_violations"] > 0


def test_evaluator_uses_evaluator_computed_sum_when_reported_sum_is_inconsistent(monkeypatch, tmp_path):
    candidate_path = _write_candidate(
        tmp_path,
        _valid_candidate_code(include_sum=True, reported_sum="123.456"),
        name="candidate_bad_sum.py",
    )
    monkeypatch.setenv(CIRCLE_PACKING_TASK_SELECTOR_ENV_VAR, "cp_n20")

    result = circle_packing_evaluator.evaluate(str(candidate_path))

    assert result.metrics["validity"] == pytest.approx(1.0)
    assert result.metrics["sum_radii"] == pytest.approx(0.20)
    assert result.artifacts["task_results"][0]["validation_summary"]["sum_mismatch"] is True


def test_extract_task_result_returns_none_on_malformed_stored_metrics():
    assert extract_task_result(
        {"task_results": [{"task_id": "cp_n20", "metrics": "bad"}]},
        "cp_n20",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "cp_n20",
                    "metrics": {
                        "sum_radii": 1.0,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "score": float("nan"),
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "cp_n20",
    ) is None

    assert extract_task_result(
        {
            "task_results": [
                {
                    "task_id": "cp_n20",
                    "metrics": {
                        "sum_radii": 1.0,
                        "target_ratio": 0.4,
                        "validity": 1.0,
                        "combined_score": 0.4,
                    },
                }
            ]
        },
        "cp_n20",
    ) is None


def test_spawn_builds_loadable_task_checkpoint_without_reevaluation(tmp_path, monkeypatch):
    base_config_path = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "config.yaml"
    evaluation_file = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "evaluator.py"
    initial_program = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "initial_program.py"

    config = Config.from_yaml(base_config_path)
    config.database.db_path = None
    shared_database = ProgramDatabase(config.database)

    source_metrics_by_program: dict[str, dict[str, dict[str, float]]] = {}
    for program_index in range(2):
        task_results = [
            build_task_result(
                task,
                raw_metrics=_fake_raw_metrics(task, program_index),
                validation_summary={"boundary_violations": 0, "overlap_violations": 0, "sum_mismatch": False},
            )
            for task in CIRCLE_PACKING_TASK_SPECS
        ]
        metrics = aggregate_task_results(task_results)
        program = Program(
            id=f"circle_program_{program_index}",
            code=_valid_candidate_code(include_sum=True),
            changes_description=f"circle program {program_index}",
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
        family="circle_packing",
        task_ids=["cp_n24"],
        initial_program=initial_program,
    )

    assert "cp_n24" in spawn_results
    spawned_checkpoint = spawned_root / "cp_n24"
    assert (spawned_checkpoint / "metadata.json").is_file()
    assert (spawned_checkpoint / "best_program_info.json").is_file()

    spawned_config = Config.from_yaml(base_config_path)
    spawned_config.database.db_path = None
    spawned_database = ProgramDatabase(spawned_config.database)
    spawned_database.load(str(spawned_checkpoint))

    assert spawned_database.last_iteration == 0
    assert len(spawned_database.programs) == 2

    for program_id, program in spawned_database.programs.items():
        expected_metrics = source_metrics_by_program[program_id]["cp_n24"]
        assert program.metrics["combined_score"] == pytest.approx(program.metrics["score"])
        assert program.metrics["combined_score"] == pytest.approx(expected_metrics["combined_score"])
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_target_task_id"] == "cp_n24"

        task_artifacts = spawned_database.get_artifacts(program_id)
        assert task_artifacts["task_selector"] == "cp_n24"
        assert len(task_artifacts["task_results"]) == 1
        assert task_artifacts["task_results"][0]["task_id"] == "cp_n24"
        assert "centers" not in task_artifacts["task_results"][0]
        assert "radii" not in task_artifacts["task_results"][0]


def test_holdout_evaluation_helper_evaluates_valid_candidate(tmp_path):
    candidate_path = _write_candidate(tmp_path, _valid_candidate_code(include_sum=True))
    evaluation_file = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "evaluator.py"

    result = evaluate_best_program_on_holdouts(
        program_path=candidate_path,
        family="circle_packing",
        holdout_task_specs=resolve_holdout_task_specs("cp_n21,cp_n25"),
        evaluation_file=evaluation_file,
    )

    assert result["available"] is True
    assert result["holdout_task_ids"] == ["cp_n21", "cp_n25"]
    assert result["evaluated_task_count"] == 2
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 0
    assert math.isfinite(result["average_holdout_score"])
    assert math.isfinite(result["average_holdout_target_ratio"])
    for task_id in ("cp_n21", "cp_n25"):
        task_result = result["holdout_task_results"][task_id]
        metrics = task_result["metrics"]
        assert metrics["validity"] == pytest.approx(1.0)
        assert math.isfinite(metrics["score"])
        assert math.isfinite(metrics["target_ratio"])
        assert math.isfinite(metrics["eval_time"])


def test_circle_packing_holdout_summary_writes_files_and_returns_section(tmp_path):
    candidate_code = _valid_candidate_code(include_sum=True)
    initial_program = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "initial_program.py"
    evaluation_file = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "evaluator.py"
    run_root = tmp_path / "emo_sta_circle_packing"

    shared_checkpoint = run_root / "shared_run" / "checkpoints" / "checkpoint_1"
    shared_checkpoint.mkdir(parents=True, exist_ok=True)
    (shared_checkpoint / "best_program.py").write_text(candidate_code, encoding="utf-8")

    adaptation_program_paths = {}
    baseline_program_paths = {}
    for task in CIRCLE_PACKING_TASK_SPECS:
        adaptation_dir = run_root / "adaptation" / task.task_id / "best"
        adaptation_dir.mkdir(parents=True, exist_ok=True)
        (adaptation_dir / "best_program.py").write_text(candidate_code, encoding="utf-8")

        baseline_dir = run_root / "baselines" / task.task_id / "best"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "best_program.py").write_text(candidate_code, encoding="utf-8")

        adaptation_program_paths[task.task_id] = resolve_best_program_path(
            adaptation_dir.parent,
            initial_program=initial_program,
            checkpoint_layout=False,
        )
        baseline_program_paths[task.task_id] = resolve_best_program_path(
            baseline_dir.parent,
            initial_program=initial_program,
            checkpoint_layout=False,
        )

    holdout_summary = run_circle_packing_holdout_evaluation(
        family="circle_packing",
        run_root=run_root,
        holdout_selector="cp_n21",
        skip_holdouts=False,
        shared_program_path=resolve_best_program_path(
            shared_checkpoint,
            initial_program=initial_program,
            checkpoint_layout=True,
        ),
        adaptation_program_paths=adaptation_program_paths,
        baseline_program_paths=baseline_program_paths,
        evaluation_file=evaluation_file,
    )

    assert holdout_summary is not None
    assert holdout_summary["enabled"] is True
    assert holdout_summary["holdout_task_ids"] == ["cp_n21"]
    assert math.isfinite(holdout_summary["shared_zero_shot"]["average_holdout_score"])
    assert set(holdout_summary["adaptation_by_source_task"]) == {
        "cp_n20",
        "cp_n22",
        "cp_n24",
        "cp_n26",
    }
    assert set(holdout_summary["baseline_by_source_task"]) == {
        "cp_n20",
        "cp_n22",
        "cp_n24",
        "cp_n26",
    }
    assert (run_root / "holdout_evaluation" / "shared_holdouts.json").is_file()
    assert (run_root / "holdout_evaluation" / "adaptation_holdouts.json").is_file()
    assert (run_root / "holdout_evaluation" / "baseline_holdouts.json").is_file()
    assert (run_root / "holdout_evaluation" / "holdout_summary.json").is_file()

    comparison_summary_path = run_root / "comparison_summary.json"
    comparison_summary_path.write_text(
        json.dumps({"holdout_evaluation": holdout_summary}, indent=2),
        encoding="utf-8",
    )
    loaded_summary = json.loads(comparison_summary_path.read_text(encoding="utf-8"))
    assert loaded_summary["holdout_evaluation"]["enabled"] is True
    assert loaded_summary["holdout_evaluation"]["holdout_task_ids"] == ["cp_n21"]


def test_skip_holdouts_disables_circle_packing_holdout_summary(tmp_path):
    evaluation_file = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "evaluator.py"

    holdout_summary = run_circle_packing_holdout_evaluation(
        family="circle_packing",
        run_root=tmp_path / "run",
        holdout_selector="all",
        skip_holdouts=True,
        shared_program_path=None,
        adaptation_program_paths={},
        baseline_program_paths={},
        evaluation_file=evaluation_file,
    )

    assert holdout_summary == {
        "enabled": False,
        "holdout_task_ids": [],
        "shared_zero_shot": None,
        "adaptation_by_source_task": {},
        "baseline_by_source_task": {},
        "reason": "skipped_by_flag",
    }
    assert not (tmp_path / "run" / "holdout_evaluation").exists()


def test_non_circle_families_do_not_run_circle_packing_holdouts(tmp_path):
    evaluation_file = REPO_ROOT / "examples" / "circle_packing_emo_sta" / "evaluator.py"

    assert (
        run_circle_packing_holdout_evaluation(
            family="function_minimization",
            run_root=tmp_path / "other_family_run",
            holdout_selector="all",
            skip_holdouts=False,
            shared_program_path=None,
            adaptation_program_paths={},
            baseline_program_paths={},
            evaluation_file=evaluation_file,
        )
        is None
    )
