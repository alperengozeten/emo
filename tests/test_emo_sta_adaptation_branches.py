import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.multi_task_shared_then_adapt.adaptation_branches import (
    collect_shared_projected_task_scores,
    resolve_best_shared_adaptation_iterations,
    resolve_best_local_adaptation_iterations,
)
from openevolve.multi_task_shared_then_adapt.k_module_problem_balanced import (
    K_MODULE_BALANCED_TASK_SPECS as K_MODULE_TASK_SPECS,
    K_MODULE_BALANCED_TOTAL_MODULES,
    aggregate_task_results as aggregate_k_module_task_results,
    build_task_result as build_k_module_task_result,
)
from openevolve.multi_task_shared_then_adapt.spawn import (
    spawn_best_shared_checkpoints,
    spawn_best_local_checkpoints,
)
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition
from openevolve.multi_task_shared_then_adapt.workflow import (
    family_task_specs,
    load_manifest,
)

DEFAULT_MANIFEST_PATH = (
    REPO_ROOT / "multi_task_shared_then_adapt" / "configs" / "function_minimization_emo_sta.yaml"
)
K_MODULE_BASE_CONFIG_PATH = (
    REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "config.yaml"
)
K_MODULE_EVALUATION_FILE = (
    REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "evaluator.py"
)
K_MODULE_INITIAL_PROGRAM = (
    REPO_ROOT / "examples" / "k_module_problem_emo_sta" / "initial_program.py"
)


def _load_emo_sta_runner_module():
    runner_path = (
        REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_multi_task_shared_then_adapt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "emo_sta_adaptation_branch_runner_test",
        runner_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load EMO-STA runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


emo_sta_runner = _load_emo_sta_runner_module()


def _build_k_module_program_code(label: str) -> str:
    return (
        "# EVOLVE-BLOCK-START\n"
        "def configure_pipeline():\n"
        f"    return {{'label': {label!r}}}\n\n"
        "def run_pipeline():\n"
        "    return configure_pipeline()\n"
        "# EVOLVE-BLOCK-END\n"
    )


def _k_module_task_result(task, score: float) -> dict[str, object]:
    return build_k_module_task_result(
        task,
        raw_metrics={
            "correct_modules": int(round(float(score) * K_MODULE_BALANCED_TOTAL_MODULES)),
            "total_modules": K_MODULE_BALANCED_TOTAL_MODULES,
            "accuracy": float(score),
            "score": float(score),
            "combined_score": float(score),
            "eval_time": 0.01 + 0.001 * task.task_index,
        },
    )


def _create_k_module_shared_checkpoint(
    tmp_path: Path,
    *,
    program_task_scores: list[dict[str, object]],
) -> Path:
    config = Config.from_yaml(K_MODULE_BASE_CONFIG_PATH)
    config.database.db_path = None
    database = ProgramDatabase(config.database)

    for program_index, spec in enumerate(program_task_scores):
        task_scores = spec["task_scores"]
        task_results = [
            _k_module_task_result(task, float(task_scores[task.task_id]))
            for task in K_MODULE_TASK_SPECS
        ]
        metrics = aggregate_k_module_task_results(task_results)
        program = Program(
            id=str(spec["id"]),
            code=_build_k_module_program_code(str(spec["id"])),
            changes_description=f"source program {spec['id']}",
            language="python",
            generation=program_index,
            iteration_found=program_index,
            metrics=metrics,
            metadata={"island": program_index % config.database.num_islands},
            artifacts_json=json.dumps(
                {
                    "task_selector": "all",
                    "task_results": task_results,
                    "status": "Shared evaluation complete across 4 hidden tasks.",
                }
            ),
        )
        database.add(program, target_island=program.metadata["island"])

    shared_checkpoint = tmp_path / "shared_checkpoint"
    database.save(str(shared_checkpoint), iteration=6)
    return shared_checkpoint


def _write_best_program_info(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": {"combined_score": float(score), "score": float(score)},
                "language": "python",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_checkpoint(checkpoint_dir: Path, score: float) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_best_program_info(checkpoint_dir / "best_program_info.json", score)


def _run_emo_sta_runner(
    *,
    tmp_path: Path,
    monkeypatch,
    run_best_shared_adaptation: bool = False,
    best_shared_adaptation_iterations: int | None = None,
    run_best_local_adaptation: bool = False,
    best_local_adaptation_iterations: int | None = None,
    run_warmstart_adaptation: bool = True,
    skip_baselines: bool = False,
    skip_adaptation: bool = False,
    preexisting_baselines: bool = False,
):
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    task_selector_env_var = get_family_definition(manifest.family).task_selector_env_var
    task_specs = family_task_specs(manifest)
    task_ids = [task.task_id for task in task_specs]
    run_name = "adaptation_branch_smoke"
    run_root = tmp_path / run_name
    shared_checkpoint = tmp_path / "existing_shared_checkpoint"
    shared_score = 0.11
    _write_checkpoint(shared_checkpoint, shared_score)

    projected_scores = {task_id: 0.30 + (idx * 0.05) for idx, task_id in enumerate(task_ids)}
    warmstarted_scores = {
        task_id: projected_scores[task_id] + 0.10 for task_id in task_ids
    }
    best_shared_source_program_id = "shared_source_program"
    best_local_source_program_ids = {
        task_id: f"task_source_program_{idx}" for idx, task_id in enumerate(task_ids)
    }
    best_shared_scores = {
        task_id: projected_scores[task_id] + 0.05 for task_id in task_ids
    }
    best_local_scores = {
        task_id: projected_scores[task_id] + 0.08 for task_id in task_ids
    }
    baseline_scores = {task_id: projected_scores[task_id] + 0.01 for task_id in task_ids}
    calls: list[dict[str, object]] = []

    if preexisting_baselines:
        for task_id in task_ids:
            _write_best_program_info(
                run_root / "baselines" / task_id / "best" / "best_program_info.json",
                baseline_scores[task_id],
            )

    def fake_spawn_task_checkpoints(
        *,
        shared_checkpoint_path,
        output_root,
        base_config_path,
        evaluation_file,
        family,
        task_ids,
        initial_program,
    ):
        del base_config_path, evaluation_file, family, initial_program
        assert Path(shared_checkpoint_path).resolve() == shared_checkpoint.resolve()
        output_root = Path(output_root).resolve()
        results = {}
        for task_id in task_ids:
            checkpoint_dir = output_root / task_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _write_best_program_info(
                checkpoint_dir / "best_program_info.json",
                projected_scores[task_id],
            )
            (checkpoint_dir / "spawn_metadata.json").write_text(
                json.dumps(
                    {
                        "shared_checkpoint_path": str(shared_checkpoint.resolve()),
                        "target_task_id": task_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            results[task_id] = {"checkpoint_path": str(checkpoint_dir)}
        return results

    def fake_spawn_best_shared_checkpoints(
        *,
        shared_checkpoint_path,
        output_root,
        base_config_path,
        evaluation_file,
        family,
        task_ids,
        initial_program,
    ):
        del base_config_path, evaluation_file, family, initial_program
        assert Path(shared_checkpoint_path).resolve() == shared_checkpoint.resolve()
        output_root = Path(output_root).resolve()
        results = {}
        for task_id in task_ids:
            checkpoint_dir = output_root / task_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _write_best_program_info(
                checkpoint_dir / "best_program_info.json",
                projected_scores[task_id] + 0.02,
            )
            (checkpoint_dir / "spawn_metadata.json").write_text(
                json.dumps(
                    {
                        "shared_checkpoint_path": str(shared_checkpoint.resolve()),
                        "target_task_id": task_id,
                        "selection_mode": "best_shared",
                        "source_shared_program_id": best_shared_source_program_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            results[task_id] = {
                "checkpoint_path": str(checkpoint_dir),
                "selection_mode": "best_shared",
                "source_shared_program_id": best_shared_source_program_id,
            }
        return results

    def fake_spawn_best_local_checkpoints(
        *,
        shared_checkpoint_path,
        output_root,
        base_config_path,
        evaluation_file,
        family,
        task_ids,
        initial_program,
    ):
        del base_config_path, evaluation_file, family, initial_program
        assert Path(shared_checkpoint_path).resolve() == shared_checkpoint.resolve()
        output_root = Path(output_root).resolve()
        results = {}
        for task_id in task_ids:
            checkpoint_dir = output_root / task_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _write_best_program_info(
                checkpoint_dir / "best_program_info.json",
                projected_scores[task_id] + 0.04,
            )
            (checkpoint_dir / "spawn_metadata.json").write_text(
                json.dumps(
                    {
                        "shared_checkpoint_path": str(shared_checkpoint.resolve()),
                        "target_task_id": task_id,
                        "selection_mode": "best_local",
                        "source_shared_program_id": best_local_source_program_ids[task_id],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            results[task_id] = {
                "checkpoint_path": str(checkpoint_dir),
                "selection_mode": "best_local",
                "source_shared_program_id": best_local_source_program_ids[task_id],
            }
        return results

    def fake_run_command(command, *, env):
        output_dir = Path(command[command.index("--output") + 1]).resolve()
        iterations = int(command[command.index("--iterations") + 1])
        checkpoint_path = None
        if "--checkpoint" in command:
            checkpoint_path = Path(command[command.index("--checkpoint") + 1]).resolve()

        task_id = output_dir.name
        phase = output_dir.parent.name
        score_by_phase = {
            "adaptation": warmstarted_scores,
            "adaptation_best_shared": best_shared_scores,
            "adaptation_best_local": best_local_scores,
            "baselines": baseline_scores,
        }
        if phase not in score_by_phase:
            raise AssertionError(f"Unexpected phase output dir: {output_dir}")
        score = score_by_phase[phase][task_id]

        calls.append(
            {
                "phase": phase,
                "task_id": task_id,
                "checkpoint_path": checkpoint_path,
                "iterations": iterations,
                "selector_value": env.get(task_selector_env_var),
                "output_dir": output_dir,
            }
        )
        _write_best_program_info(output_dir / "best" / "best_program_info.json", score)
        _write_checkpoint(output_dir / "checkpoints" / f"checkpoint_{iterations}", score)

    monkeypatch.setattr(emo_sta_runner, "spawn_task_checkpoints", fake_spawn_task_checkpoints)
    monkeypatch.setattr(
        emo_sta_runner,
        "spawn_best_shared_checkpoints",
        fake_spawn_best_shared_checkpoints,
    )
    monkeypatch.setattr(
        emo_sta_runner,
        "spawn_best_local_checkpoints",
        fake_spawn_best_local_checkpoints,
    )
    monkeypatch.setattr(emo_sta_runner, "run_command", fake_run_command)

    argv = [
        "run_multi_task_shared_then_adapt.py",
        "--manifest",
        str(DEFAULT_MANIFEST_PATH),
        "--output-root",
        str(tmp_path),
        "--run-name",
        run_name,
        "--shared-checkpoint",
        str(shared_checkpoint),
        "--shared-iterations",
        "4",
        "--adaptation-iterations",
        "2",
        "--baseline-iterations",
        "3",
    ]
    if run_best_shared_adaptation:
        argv.append("--run-best-shared-adaptation")
    if best_shared_adaptation_iterations is not None:
        argv.extend(
            [
                "--best-shared-adaptation-iterations",
                str(best_shared_adaptation_iterations),
            ]
        )
    if run_best_local_adaptation:
        argv.append("--run-best-local-adaptation")
    if best_local_adaptation_iterations is not None:
        argv.extend(
            [
                "--best-local-adaptation-iterations",
                str(best_local_adaptation_iterations),
            ]
        )
    if not run_warmstart_adaptation:
        argv.append("--skip-warmstart-adaptation")
    if skip_baselines:
        argv.append("--skip-baselines")
    if skip_adaptation:
        argv.append("--skip-adaptation")

    monkeypatch.setattr(sys, "argv", argv)
    exit_code = emo_sta_runner.main()

    summary_path = run_root / "comparison_summary.json"
    csv_path = run_root / "comparison_summary.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    return {
        "exit_code": exit_code,
        "summary": summary,
        "csv_rows": csv_rows,
        "calls": calls,
        "run_root": run_root,
        "task_ids": task_ids,
        "shared_score": shared_score,
        "projected_scores": projected_scores,
        "warmstarted_scores": warmstarted_scores,
        "best_shared_scores": best_shared_scores,
        "best_local_scores": best_local_scores,
        "best_shared_source_program_id": best_shared_source_program_id,
        "best_local_source_program_ids": best_local_source_program_ids,
        "baseline_scores": baseline_scores,
    }


def test_parse_args_defaults_disable_adaptation_branches(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multi_task_shared_then_adapt.py",
            "--manifest",
            str(DEFAULT_MANIFEST_PATH),
        ],
    )
    args = emo_sta_runner.parse_args()

    assert args.run_warmstart_adaptation is True
    assert args.run_best_shared_adaptation is False
    assert args.best_shared_adaptation_iterations is None
    assert args.run_best_local_adaptation is False
    assert args.best_local_adaptation_iterations is None


def test_adaptation_branch_iterations_default_to_adaptation_budget():
    best_shared_iterations, best_shared_defaulted = (
        resolve_best_shared_adaptation_iterations(
            adaptation_iterations=7,
            best_shared_adaptation_iterations=None,
        )
    )
    best_local_iterations, best_local_defaulted = resolve_best_local_adaptation_iterations(
        adaptation_iterations=7,
        best_local_adaptation_iterations=None,
    )

    assert best_shared_iterations == 7
    assert best_shared_defaulted is True
    assert best_local_iterations == 7
    assert best_local_defaulted is True

    explicit_best_shared, explicit_best_shared_defaulted = (
        resolve_best_shared_adaptation_iterations(
            adaptation_iterations=7,
            best_shared_adaptation_iterations=4,
        )
    )
    explicit_best_local, explicit_best_local_defaulted = (
        resolve_best_local_adaptation_iterations(
            adaptation_iterations=7,
            best_local_adaptation_iterations=3,
        )
    )

    assert explicit_best_shared == 4
    assert explicit_best_shared_defaulted is False
    assert explicit_best_local == 3
    assert explicit_best_local_defaulted is False


def test_best_shared_spawn_uses_same_source_program_for_all_tasks(tmp_path, monkeypatch):
    shared_checkpoint = _create_k_module_shared_checkpoint(
        tmp_path,
        program_task_scores=[
            {
                "id": "shared_best_program",
                "task_scores": {
                    "kmb_task_a": 5 / 6,
                    "kmb_task_b": 0.50,
                    "kmb_task_c": 1.00,
                    "kmb_task_d": 5 / 6,
                },
            },
            {
                "id": "task_a_specialist",
                "task_scores": {
                    "kmb_task_a": 1.00,
                    "kmb_task_b": 0.50,
                    "kmb_task_c": 0.50,
                    "kmb_task_d": 0.50,
                },
            },
            {
                "id": "task_b_specialist",
                "task_scores": {
                    "kmb_task_a": 0.50,
                    "kmb_task_b": 1.00,
                    "kmb_task_c": 0.50,
                    "kmb_task_d": 0.50,
                },
            },
        ],
    )

    def fail_if_reevaluated(**kwargs):
        raise AssertionError(f"Best-shared branch spawn should use stored task_results: {kwargs}")

    monkeypatch.setattr(
        "openevolve.multi_task_shared_then_adapt.spawn._reevaluate_program_for_task",
        fail_if_reevaluated,
    )

    spawned_root = tmp_path / "best_shared"
    spawn_results = spawn_best_shared_checkpoints(
        shared_checkpoint_path=shared_checkpoint,
        output_root=spawned_root,
        base_config_path=K_MODULE_BASE_CONFIG_PATH,
        evaluation_file=K_MODULE_EVALUATION_FILE,
        family="k_module_problem_balanced",
        task_ids=["kmb_task_a", "kmb_task_b"],
        initial_program=K_MODULE_INITIAL_PROGRAM,
    )

    assert spawn_results["kmb_task_a"]["source_shared_program_id"] == "shared_best_program"
    assert spawn_results["kmb_task_b"]["source_shared_program_id"] == "shared_best_program"

    loaded_config = Config.from_yaml(K_MODULE_BASE_CONFIG_PATH)
    loaded_config.database.db_path = None
    for task_id, expected_score in {"kmb_task_a": 5 / 6, "kmb_task_b": 0.50}.items():
        checkpoint_dir = spawned_root / task_id
        database = ProgramDatabase(loaded_config.database)
        database.load(str(checkpoint_dir))

        assert database.last_iteration == 0
        assert len(database.programs) == 1

        program = next(iter(database.programs.values()))
        assert program.id == "shared_best_program"
        assert program.metrics["combined_score"] == pytest.approx(expected_score)
        assert program.metadata["sts_warmstarted"] is True
        assert program.metadata["sts_selection_mode"] == "best_shared"
        assert program.metadata["sts_target_task_id"] == task_id
        assert program.metadata["sts_source_shared_program_id"] == "shared_best_program"
        if task_id == "kmb_task_b":
            assert program.metrics["combined_score"] != pytest.approx(
                program.metadata["sts_source_shared_metrics"]["combined_score"]
            )

        spawn_metadata = json.loads(
            (checkpoint_dir / "spawn_metadata.json").read_text(encoding="utf-8")
        )
        assert spawn_metadata["selection_mode"] == "best_shared"
        assert spawn_metadata["source_shared_program_id"] == "shared_best_program"


def test_best_local_spawn_selects_task_specific_source_programs(tmp_path, monkeypatch):
    shared_checkpoint = _create_k_module_shared_checkpoint(
        tmp_path,
        program_task_scores=[
            {
                "id": "shared_best_program",
                "task_scores": {
                    "kmb_task_a": 5 / 6,
                    "kmb_task_b": 0.50,
                    "kmb_task_c": 1.00,
                    "kmb_task_d": 5 / 6,
                },
            },
            {
                "id": "task_a_specialist",
                "task_scores": {
                    "kmb_task_a": 1.00,
                    "kmb_task_b": 0.50,
                    "kmb_task_c": 0.50,
                    "kmb_task_d": 0.50,
                },
            },
            {
                "id": "task_b_specialist",
                "task_scores": {
                    "kmb_task_a": 0.50,
                    "kmb_task_b": 1.00,
                    "kmb_task_c": 0.50,
                    "kmb_task_d": 0.50,
                },
            },
        ],
    )

    def fail_if_reevaluated(**kwargs):
        raise AssertionError(f"Best-local branch spawn should use stored task_results: {kwargs}")

    monkeypatch.setattr(
        "openevolve.multi_task_shared_then_adapt.spawn._reevaluate_program_for_task",
        fail_if_reevaluated,
    )

    spawned_root = tmp_path / "best_local"
    spawn_results = spawn_best_local_checkpoints(
        shared_checkpoint_path=shared_checkpoint,
        output_root=spawned_root,
        base_config_path=K_MODULE_BASE_CONFIG_PATH,
        evaluation_file=K_MODULE_EVALUATION_FILE,
        family="k_module_problem_balanced",
        task_ids=["kmb_task_a", "kmb_task_b"],
        initial_program=K_MODULE_INITIAL_PROGRAM,
    )

    assert spawn_results["kmb_task_a"]["source_shared_program_id"] == "task_a_specialist"
    assert spawn_results["kmb_task_b"]["source_shared_program_id"] == "task_b_specialist"

    loaded_config = Config.from_yaml(K_MODULE_BASE_CONFIG_PATH)
    loaded_config.database.db_path = None
    expected = {
        "kmb_task_a": ("task_a_specialist", 1.00),
        "kmb_task_b": ("task_b_specialist", 1.00),
    }
    for task_id, (expected_program_id, expected_score) in expected.items():
        checkpoint_dir = spawned_root / task_id
        database = ProgramDatabase(loaded_config.database)
        database.load(str(checkpoint_dir))

        assert database.last_iteration == 0
        assert len(database.programs) == 1

        program = next(iter(database.programs.values()))
        assert program.id == expected_program_id
        assert program.metrics["combined_score"] == pytest.approx(expected_score)
        assert program.metadata["sts_selection_mode"] == "best_local"
        assert program.metadata["sts_source_shared_program_id"] == expected_program_id

        spawn_metadata = json.loads(
            (checkpoint_dir / "spawn_metadata.json").read_text(encoding="utf-8")
        )
        assert spawn_metadata["selection_mode"] == "best_local"
        assert spawn_metadata["source_shared_program_id"] == expected_program_id


def test_runner_enables_adaptation_branches_with_distinct_checkpoint_semantics(
    monkeypatch,
    tmp_path,
):
    result = _run_emo_sta_runner(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        run_best_shared_adaptation=True,
        run_best_local_adaptation=True,
    )

    assert result["exit_code"] == 0
    task_ids = result["task_ids"]
    summary = result["summary"]
    calls = result["calls"]

    warmstarted_calls = [call for call in calls if call["phase"] == "adaptation"]
    best_shared_calls = [
        call for call in calls if call["phase"] == "adaptation_best_shared"
    ]
    best_local_calls = [
        call for call in calls if call["phase"] == "adaptation_best_local"
    ]
    baseline_calls = [call for call in calls if call["phase"] == "baselines"]

    assert len(warmstarted_calls) == len(task_ids)
    assert len(best_shared_calls) == len(task_ids)
    assert len(best_local_calls) == len(task_ids)
    assert len(baseline_calls) == len(task_ids)
    assert set(summary) == {
        "workflow",
        "family",
        "manifest_path",
        "run_root",
        "shared_iterations",
        "adaptation_iterations",
        "baseline_iterations",
        "best_shared_adaptation",
        "best_local_adaptation",
        "wandb",
        "shared_run",
        "tasks",
    }

    for task_id in task_ids:
        warmstarted_call = next(call for call in warmstarted_calls if call["task_id"] == task_id)
        best_shared_call = next(call for call in best_shared_calls if call["task_id"] == task_id)
        best_local_call = next(call for call in best_local_calls if call["task_id"] == task_id)
        baseline_call = next(call for call in baseline_calls if call["task_id"] == task_id)

        assert warmstarted_call["selector_value"] == task_id
        assert best_shared_call["selector_value"] == task_id
        assert best_local_call["selector_value"] == task_id
        assert baseline_call["selector_value"] == task_id
        assert warmstarted_call["checkpoint_path"] == (
            result["run_root"] / "spawned_checkpoints" / task_id
        ).resolve()
        assert best_shared_call["checkpoint_path"] == (
            result["run_root"] / "spawned_checkpoints_best_shared" / task_id
        ).resolve()
        assert best_local_call["checkpoint_path"] == (
            result["run_root"] / "spawned_checkpoints_best_local" / task_id
        ).resolve()
        assert baseline_call["checkpoint_path"] is None

    assert summary["best_shared_adaptation"] == {
        "requested": True,
        "enabled": True,
        "iterations": 2,
        "defaulted_to_adaptation_iterations": True,
        "output_root": str(result["run_root"] / "adaptation_best_shared"),
        "spawned_checkpoint_root": str(
            result["run_root"] / "spawned_checkpoints_best_shared"
        ),
    }
    assert summary["best_local_adaptation"] == {
        "requested": True,
        "enabled": True,
        "iterations": 2,
        "defaulted_to_adaptation_iterations": True,
        "output_root": str(result["run_root"] / "adaptation_best_local"),
        "spawned_checkpoint_root": str(
            result["run_root"] / "spawned_checkpoints_best_local"
        ),
    }

    first_task_id = task_ids[0]
    first_task_summary = summary["tasks"][first_task_id]
    assert set(first_task_summary) == {
        "task_spec",
        "spawn_checkpoint",
        "spawn_best_score",
        "spawn_best_metrics",
        "adaptation_output_dir",
        "adapted_best_score",
        "adapted_best_metrics",
        "best_shared_adaptation_output_dir",
        "best_shared_adaptation_best_score",
        "best_shared_adaptation_best_metrics",
        "best_local_adaptation_output_dir",
        "best_local_adaptation_best_score",
        "best_local_adaptation_best_metrics",
        "baseline_output_dir",
        "baseline_best_score",
        "baseline_best_metrics",
        "shared_projected",
        "warmstarted_adaptation",
        "best_shared_adaptation",
        "best_local_adaptation",
        "direct_baseline",
        "deltas",
    }
    assert first_task_summary["shared_projected"]["best_score"] == pytest.approx(
        result["projected_scores"][first_task_id]
    )
    assert first_task_summary["warmstarted_adaptation"]["best_score"] == pytest.approx(
        result["warmstarted_scores"][first_task_id]
    )
    assert first_task_summary["best_shared_adaptation"]["best_score"] == pytest.approx(
        result["best_shared_scores"][first_task_id]
    )
    assert first_task_summary["best_local_adaptation"]["best_score"] == pytest.approx(
        result["best_local_scores"][first_task_id]
    )
    assert first_task_summary["direct_baseline"]["best_score"] == pytest.approx(
        result["baseline_scores"][first_task_id]
    )
    assert first_task_summary["best_shared_adaptation"]["source_shared_program_id"] == (
        result["best_shared_source_program_id"]
    )
    assert first_task_summary["best_local_adaptation"]["source_shared_program_id"] == (
        result["best_local_source_program_ids"][first_task_id]
    )
    assert first_task_summary["deltas"]["warmstart_minus_best_shared"] == pytest.approx(0.05)
    assert first_task_summary["deltas"]["warmstart_minus_best_local"] == pytest.approx(0.02)
    assert (
        first_task_summary["deltas"]["best_local_minus_best_shared"]
        == pytest.approx(0.03)
    )
    assert first_task_summary["deltas"]["warmstart_minus_baseline"] == pytest.approx(0.09)

    first_csv_row = result["csv_rows"][0]
    assert set(first_csv_row) == {
        "task_id",
        "shared_projected_score",
        "warmstarted_adaptation_score",
        "best_shared_adaptation_score",
        "best_local_adaptation_score",
        "baseline_score",
        "warmstart_minus_best_shared",
        "warmstart_minus_best_local",
        "best_local_minus_best_shared",
        "warmstart_minus_shared_projected",
        "warmstart_minus_baseline",
        "shared_projected_best_program_info_path",
        "warmstarted_adaptation_best_program_info_path",
        "best_shared_adaptation_best_program_info_path",
        "best_local_adaptation_best_program_info_path",
        "baseline_best_program_info_path",
    }
    assert float(first_csv_row["best_shared_adaptation_score"]) == pytest.approx(
        result["best_shared_scores"][first_task_id]
    )
    assert float(first_csv_row["best_local_adaptation_score"]) == pytest.approx(
        result["best_local_scores"][first_task_id]
    )
    assert float(first_csv_row["warmstart_minus_best_shared"]) == pytest.approx(0.05)
    assert float(first_csv_row["warmstart_minus_best_local"]) == pytest.approx(0.02)
    assert float(first_csv_row["best_local_minus_best_shared"]) == pytest.approx(0.03)


def test_runner_default_behavior_skips_adaptation_branches(monkeypatch, tmp_path):
    result = _run_emo_sta_runner(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        skip_baselines=True,
    )

    best_shared_calls = [
        call
        for call in result["calls"]
        if call["phase"] == "adaptation_best_shared"
    ]
    best_local_calls = [
        call
        for call in result["calls"]
        if call["phase"] == "adaptation_best_local"
    ]
    assert best_shared_calls == []
    assert best_local_calls == []

    assert result["summary"]["best_shared_adaptation"] == {
        "requested": False,
        "enabled": False,
        "iterations": None,
        "defaulted_to_adaptation_iterations": False,
        "output_root": str(result["run_root"] / "adaptation_best_shared"),
        "spawned_checkpoint_root": str(
            result["run_root"] / "spawned_checkpoints_best_shared"
        ),
    }
    assert result["summary"]["best_local_adaptation"] == {
        "requested": False,
        "enabled": False,
        "iterations": None,
        "defaulted_to_adaptation_iterations": False,
        "output_root": str(result["run_root"] / "adaptation_best_local"),
        "spawned_checkpoint_root": str(
            result["run_root"] / "spawned_checkpoints_best_local"
        ),
    }

    first_task_id = result["task_ids"][0]
    first_task_summary = result["summary"]["tasks"][first_task_id]
    assert first_task_summary["best_shared_adaptation"]["executed"] is False
    assert first_task_summary["best_shared_adaptation"]["iterations"] is None
    assert first_task_summary["best_shared_adaptation"]["best_score"] is None
    assert first_task_summary["best_local_adaptation"]["executed"] is False
    assert first_task_summary["best_local_adaptation"]["iterations"] is None
    assert first_task_summary["best_local_adaptation"]["best_score"] is None
    assert first_task_summary["direct_baseline"]["executed"] is False
    assert first_task_summary["direct_baseline"]["iterations"] is None


def test_runner_can_disable_only_warmstart_adaptation_branch(monkeypatch, tmp_path):
    result = _run_emo_sta_runner(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        run_warmstart_adaptation=False,
        run_best_shared_adaptation=True,
        run_best_local_adaptation=True,
    )

    warmstarted_calls = [
        call for call in result["calls"] if call["phase"] == "adaptation"
    ]
    best_shared_calls = [
        call for call in result["calls"] if call["phase"] == "adaptation_best_shared"
    ]
    best_local_calls = [
        call for call in result["calls"] if call["phase"] == "adaptation_best_local"
    ]
    baseline_calls = [call for call in result["calls"] if call["phase"] == "baselines"]

    assert warmstarted_calls == []
    assert len(best_shared_calls) == len(result["task_ids"])
    assert len(best_local_calls) == len(result["task_ids"])
    assert len(baseline_calls) == len(result["task_ids"])

    first_task_id = result["task_ids"][0]
    first_task_summary = result["summary"]["tasks"][first_task_id]
    assert first_task_summary["warmstarted_adaptation"]["executed"] is False
    assert first_task_summary["warmstarted_adaptation"]["iterations"] is None
    assert first_task_summary["warmstarted_adaptation"]["best_score"] is None
    assert first_task_summary["best_shared_adaptation"]["executed"] is True
    assert first_task_summary["best_shared_adaptation"]["iterations"] == 2
    assert first_task_summary["best_local_adaptation"]["executed"] is True
    assert first_task_summary["best_local_adaptation"]["iterations"] == 2


def test_runner_reuses_existing_baseline_results_when_baselines_are_skipped(
    monkeypatch,
    tmp_path,
):
    result = _run_emo_sta_runner(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        skip_baselines=True,
        preexisting_baselines=True,
    )

    baseline_calls = [call for call in result["calls"] if call["phase"] == "baselines"]
    assert baseline_calls == []
    assert result["summary"]["baseline_iterations"] == 3

    first_task_id = result["task_ids"][0]
    first_task_summary = result["summary"]["tasks"][first_task_id]
    assert first_task_summary["direct_baseline"]["executed"] is False
    assert first_task_summary["direct_baseline"]["reused_existing"] is True
    assert first_task_summary["direct_baseline"]["iterations"] == 3
    assert first_task_summary["direct_baseline"]["best_score"] == pytest.approx(
        result["baseline_scores"][first_task_id]
    )
    assert first_task_summary["baseline_best_score"] == pytest.approx(
        result["baseline_scores"][first_task_id]
    )
    assert float(result["csv_rows"][0]["baseline_score"]) == pytest.approx(
        result["baseline_scores"][first_task_id]
    )


def test_collect_shared_projected_task_scores_reads_spawned_checkpoint_best_scores(tmp_path):
    spawned_root = tmp_path / "spawned_checkpoints"
    _write_checkpoint(spawned_root / "task_a", 0.91)
    _write_checkpoint(spawned_root / "task_b", 0.37)

    projected = collect_shared_projected_task_scores(
        spawned_root=spawned_root,
        task_ids=["task_a", "task_b"],
    )

    assert projected["task_a"]["best_score"] == pytest.approx(0.91)
    assert projected["task_a"]["checkpoint_path"] == str((spawned_root / "task_a").resolve())
    assert projected["task_a"]["best_program_info_path"] == str(
        (spawned_root / "task_a" / "best_program_info.json").resolve()
    )
    assert projected["task_b"]["best_score"] == pytest.approx(0.37)


def test_runner_marks_adaptation_branches_unexecuted_when_adaptation_is_skipped(
    monkeypatch,
    tmp_path,
):
    result = _run_emo_sta_runner(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        run_best_shared_adaptation=True,
        run_best_local_adaptation=True,
        skip_adaptation=True,
    )

    warmstarted_calls = [call for call in result["calls"] if call["phase"] == "adaptation"]
    best_shared_calls = [
        call for call in result["calls"] if call["phase"] == "adaptation_best_shared"
    ]
    best_local_calls = [
        call for call in result["calls"] if call["phase"] == "adaptation_best_local"
    ]
    baseline_calls = [call for call in result["calls"] if call["phase"] == "baselines"]

    assert warmstarted_calls == []
    assert best_shared_calls == []
    assert best_local_calls == []
    assert len(baseline_calls) == len(result["task_ids"])
    assert result["summary"]["best_shared_adaptation"]["enabled"] is False
    assert result["summary"]["best_local_adaptation"]["enabled"] is False

    first_task_id = result["task_ids"][0]
    first_task_summary = result["summary"]["tasks"][first_task_id]
    assert first_task_summary["warmstarted_adaptation"]["executed"] is False
    assert first_task_summary["warmstarted_adaptation"]["iterations"] is None
    assert first_task_summary["warmstarted_adaptation"]["best_score"] is None
    assert first_task_summary["best_shared_adaptation"]["executed"] is False
    assert first_task_summary["best_shared_adaptation"]["iterations"] is None
    assert first_task_summary["best_shared_adaptation"]["best_score"] is None
    assert first_task_summary["best_local_adaptation"]["executed"] is False
    assert first_task_summary["best_local_adaptation"]["iterations"] is None
    assert first_task_summary["best_local_adaptation"]["best_score"] is None
    assert first_task_summary["direct_baseline"]["executed"] is True
    assert first_task_summary["direct_baseline"]["iterations"] == 3
