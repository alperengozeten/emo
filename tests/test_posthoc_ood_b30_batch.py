import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "multi_task_shared_then_adapt"
    / "scripts"
    / "run_posthoc_ood_b30_batch.py"
)


def _load_batch_module():
    spec = importlib.util.spec_from_file_location("posthoc_ood_b30_batch", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_finished_run(results_root: Path, family: str, setting: str, run: str) -> Path:
    run_dir = results_root / family / setting / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison_summary.json").write_text(
        json.dumps({"workflow": "multi_task_shared_then_adapt", "family": family}),
        encoding="utf-8",
    )
    return run_dir


def test_discover_jobs_finds_finished_b30_runs_and_skips_existing_outputs(tmp_path):
    batch = _load_batch_module()
    results_root = tmp_path / "results"

    heil_run = _write_finished_run(
        results_root,
        "heilbronn_triangle",
        "s20-a25-b30-claude-haiku-4-5-full",
        "run_01_seed_42",
    )
    _write_finished_run(
        results_root,
        "heilbronn_triangle",
        "s40-a15-b25-claude-haiku-4-5-full",
        "run_01_seed_42",
    )
    rect_run = _write_finished_run(
        results_root,
        "circle_packing_rectangle",
        "s60-a15-b30-claude-sonnet-4-5-full",
        "run_02_seed_43",
    )
    existing_output = rect_run / "posthoc_ood_all_known"
    existing_output.mkdir(parents=True)
    (existing_output / "ood_summary.json").write_text("{}", encoding="utf-8")

    jobs = batch.discover_jobs(
        results_root=results_root,
        families=["heilbronn_triangle", "circle_packing_rectangle"],
        baseline_budget=30,
        output_subdir="posthoc_ood_all_known",
        overwrite=False,
    )

    assert [(job.family, job.setting_prefix, job.run_name) for job in jobs] == [
        ("circle_packing_rectangle", "s60-a15-b30", "run_02_seed_43"),
        ("heilbronn_triangle", "s20-a25-b30", "run_01_seed_42"),
    ]
    by_results_dir = {job.results_dir: job for job in jobs}
    assert by_results_dir[heil_run].skip_reason is None
    assert by_results_dir[rect_run].skip_reason == "existing_outputs"
    assert by_results_dir[heil_run].ood_task_ids == (
        "heil_tri_n8",
        "heil_tri_n13",
        "heil_tri_n14",
    )
    assert by_results_dir[rect_run].ood_task_ids == (
        "cp_rect_n19",
        "cp_rect_n24",
        "cp_rect_n25",
    )


def test_build_job_command_uses_all_known_ood_tasks_and_output_subdir(tmp_path):
    batch = _load_batch_module()
    results_root = tmp_path / "results"
    run_dir = _write_finished_run(
        results_root,
        "heilbronn_triangle",
        "s40-a20-b30-claude-opus-4-6-full",
        "run_05_seed_46",
    )
    job = batch.discover_jobs(
        results_root=results_root,
        families=["heilbronn_triangle"],
        baseline_budget=30,
        output_subdir="posthoc_ood_all_known",
        overwrite=True,
    )[0]

    command = batch.build_job_command(
        job,
        python_executable="/usr/bin/python-test",
        overwrite=True,
    )

    assert command[0] == "/usr/bin/python-test"
    assert "--manifest" in command
    assert str(batch.FAMILY_CONFIGS["heilbronn_triangle"].manifest) in command
    assert command[command.index("--results-dir") + 1] == str(run_dir)
    assert command[command.index("--ood-task-ids") + 1] == (
        "heil_tri_n8,heil_tri_n13,heil_tri_n14"
    )
    assert command[command.index("--output-dir") + 1] == str(
        run_dir / "posthoc_ood_all_known"
    )
    assert "--overwrite" in command


def test_execute_jobs_dry_run_and_fake_parallel_runner(tmp_path):
    batch = _load_batch_module()
    results_root = tmp_path / "results"
    _write_finished_run(
        results_root,
        "circle_packing_rectangle",
        "s20-a25-b30-claude-haiku-4-5-full",
        "run_01_seed_42",
    )
    _write_finished_run(
        results_root,
        "circle_packing_rectangle",
        "s20-a25-b30-claude-haiku-4-5-full",
        "run_02_seed_43",
    )
    jobs = batch.discover_jobs(
        results_root=results_root,
        families=["circle_packing_rectangle"],
        baseline_budget=30,
        output_subdir="posthoc_ood_all_known",
    )

    dry_results = batch.execute_jobs(
        jobs,
        python_executable=sys.executable,
        overwrite=False,
        max_workers=2,
        timeout_seconds=None,
        dry_run=True,
    )
    assert [result["status"] for result in dry_results] == ["dry_run", "dry_run"]

    calls = []

    def fake_runner(job, python_executable, overwrite, timeout_seconds):
        calls.append((job.run_name, python_executable, overwrite, timeout_seconds))
        return {
            **batch.job_payload(job),
            "status": "succeeded",
            "returncode": 0,
            "elapsed_seconds": 0.01,
            "command": batch.build_job_command(
                job,
                python_executable=python_executable,
                overwrite=overwrite,
            ),
        }

    run_results = batch.execute_jobs(
        jobs,
        python_executable="/usr/bin/python-test",
        overwrite=False,
        max_workers=2,
        timeout_seconds=123.0,
        dry_run=False,
        runner=fake_runner,
    )

    assert [result["status"] for result in run_results] == ["succeeded", "succeeded"]
    assert sorted(calls) == [
        ("run_01_seed_42", "/usr/bin/python-test", False, 123.0),
        ("run_02_seed_43", "/usr/bin/python-test", False, 123.0),
    ]


def test_main_dry_run_writes_batch_summary(tmp_path, capsys):
    batch = _load_batch_module()
    results_root = tmp_path / "results"
    summary_path = tmp_path / "summary.json"
    _write_finished_run(
        results_root,
        "heilbronn_triangle",
        "s80-a10-b30-claude-sonnet-4-6-full",
        "run_03_seed_44",
    )

    exit_code = batch.main(
        [
            "--families",
            "heilbronn_triangle",
            "--results-root",
            str(results_root),
            "--summary-path",
            str(summary_path),
            "--dry-run",
            "--max-workers",
            "2",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Dry run only" in captured.out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "posthoc_ood_b30_batch"
    assert payload["baseline_budget"] == 30
    assert payload["families"] == ["heilbronn_triangle"]
    assert payload["status_counts"] == {"dry_run": 1}
    assert payload["jobs"][0]["ood_task_ids"] == [
        "heil_tri_n8",
        "heil_tri_n13",
        "heil_tri_n14",
    ]


def test_build_detach_command_removes_detach_flag():
    batch = _load_batch_module()
    command = batch.build_detach_command(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--families",
            "all",
            "--detach",
            "--max-workers",
            "4",
        ]
    )

    assert "--detach" not in command
    assert command[:2] == [sys.executable, str(SCRIPT_PATH)]
    assert command[-2:] == ["--max-workers", "4"]
