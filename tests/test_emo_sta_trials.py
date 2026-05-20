import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.trials import (
    build_setting_output_dir_name,
    build_shell_command,
    build_trial_run_name,
    existing_trial_numbers,
    load_launcher_defaults,
    load_trial_metrics,
    next_trial_number,
    parse_api_base_host_port,
    read_log_tail,
    resolve_litellm_command,
    summarize_trial_rows,
    write_seeded_trial_manifest,
)
from multi_task_shared_then_adapt.scripts.run_multi_task_shared_then_adapt_trials import (
    _prepare_managed_litellm_env,
)
import multi_task_shared_then_adapt.scripts.run_multi_task_shared_then_adapt_trials as emo_sta_trials_runner


def test_load_launcher_defaults_reads_manifest_defaults():
    defaults = load_launcher_defaults(
        REPO_ROOT / "multi_task_shared_then_adapt" / "configs" / "circle_packing_emo_sta.yaml"
    )

    assert defaults.modules == ()
    assert defaults.litellm_mode == "auto"
    assert defaults.litellm_command == "litellm"
    assert defaults.litellm_preferred_env is None
    assert defaults.litellm_host == "127.0.0.1"
    assert defaults.litellm_port == 4000
    assert defaults.litellm_config is None


def test_write_seeded_trial_manifest_makes_absolute_paths_and_overrides_seed(tmp_path):
    initial_program = tmp_path / "initial_program.py"
    evaluation_file = tmp_path / "evaluator.py"
    base_config = tmp_path / "config.yaml"
    output_root = tmp_path / "results"

    initial_program.write_text("# initial\n", encoding="utf-8")
    evaluation_file.write_text("def evaluate(*args, **kwargs):\n    return {}\n", encoding="utf-8")
    base_config.write_text(
        yaml.safe_dump(
            {
                "random_seed": 42,
                "database": {"random_seed": 42},
                "llm": {"api_base": "http://127.0.0.1:4000"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "circle_packing_emo_sta.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "family": "circle_packing",
                "initial_program": "initial_program.py",
                "evaluation_file": "evaluator.py",
                "base_config": "config.yaml",
                "output_root": "results",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    temp_manifest, temp_base_config = write_seeded_trial_manifest(
        manifest_path=manifest_path,
        seed=123,
        temp_dir=tmp_path / "temp",
        output_root=output_root,
    )

    raw_manifest = yaml.safe_load(temp_manifest.read_text(encoding="utf-8"))
    raw_config = yaml.safe_load(temp_base_config.read_text(encoding="utf-8"))

    assert Path(raw_manifest["initial_program"]).is_absolute()
    assert Path(raw_manifest["evaluation_file"]).is_absolute()
    assert Path(raw_manifest["base_config"]).resolve() == temp_base_config.resolve()
    assert Path(raw_manifest["output_root"]).resolve() == output_root.resolve()
    assert raw_manifest["manifest_label"] == "circle_packing_emo_sta"
    assert raw_config["random_seed"] == 123
    assert raw_config["database"]["random_seed"] == 123


def test_build_shell_command_wraps_modules_setup_and_exec():
    command = build_shell_command(
        ["python", "script.py", "--flag", "value with spaces"],
        modules=["R/4.5.1"],
        setup_commands=["export FOO=bar"],
    )

    assert command[:2] == ["bash", "-lc"]
    assert "module load R/4.5.1" in command[2]
    assert "export FOO=bar" in command[2]
    assert "exec python script.py --flag 'value with spaces'" in command[2]


def test_build_shell_command_preserves_parent_conda_path(monkeypatch):
    monkeypatch.setenv("PATH", "/tmp/conda-env/bin:/usr/bin")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda-env")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "in_context_reasoning_env")
    monkeypatch.setenv("CARGO_HOME", "/tmp/conda-env")
    monkeypatch.setenv("RUSTUP_HOME", "/tmp/conda-env/.rustup")

    command = build_shell_command(["python", "script.py"])

    assert "export PATH=/tmp/conda-env/bin:/usr/bin" in command[2]
    assert "export CONDA_PREFIX=/tmp/conda-env" in command[2]
    assert "export CONDA_DEFAULT_ENV=in_context_reasoning_env" in command[2]
    assert "export CARGO_HOME=/tmp/conda-env" in command[2]
    assert "export RUSTUP_HOME=/tmp/conda-env/.rustup" in command[2]
    assert command[:2] == ["bash", "-lc"]


def test_build_trial_run_name_uses_stable_seeded_format_without_timestamp():
    assert build_trial_run_name(trial_idx=0, seed=42, prefix="run") == "run_01_seed_42"
    assert build_trial_run_name(trial_idx=4, seed=99, prefix="emo_sta") == "emo_sta_05_seed_99"


def test_existing_trial_numbers_and_next_trial_number_skip_nonmatching_entries(tmp_path):
    (tmp_path / "run_01_seed_42").mkdir()
    (tmp_path / "run_03_seed_44").mkdir()
    (tmp_path / "trial_logs").mkdir()
    (tmp_path / "run_abc_seed_99").mkdir()
    (tmp_path / "other_02_seed_43").mkdir()

    assert existing_trial_numbers(tmp_path, "run") == [1, 3]
    assert next_trial_number(tmp_path, "run") == 4
    assert next_trial_number(tmp_path, "other") == 3
    assert next_trial_number(tmp_path, "fresh") == 1


def test_build_setting_output_dir_name_uses_setting_and_model_components():
    assert build_setting_output_dir_name(
        shared_iterations=60,
        adaptation_iterations=10,
        baseline_iterations=25,
        primary_model="claude-sonnet-4-6",
        edit_mode="full",
    ) == "s60-a10-b25-claude-sonnet-4-6-full"


def test_parse_api_base_host_port_handles_local_http_urls():
    assert parse_api_base_host_port("http://127.0.0.1:4000") == ("127.0.0.1", 4000)
    assert parse_api_base_host_port("http://localhost:8080/v1") == ("localhost", 8080)
    assert parse_api_base_host_port("https://example.com") == ("example.com", 443)


def test_resolve_litellm_command_prefers_requested_conda_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "conda" / "miniconda" / "envs"
    preferred = root / "preferred_env" / "bin"
    other = root / "other_env" / "bin"
    preferred.mkdir(parents=True, exist_ok=True)
    other.mkdir(parents=True, exist_ok=True)
    preferred_cmd = preferred / "litellm"
    other_cmd = other / "litellm"
    preferred_cmd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    other_cmd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    preferred_cmd.chmod(0o755)
    other_cmd.chmod(0o755)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(part for part in os.getenv("PATH", "").split(os.pathsep) if part),
    )

    assert resolve_litellm_command("litellm", preferred_env="preferred_env") == str(
        preferred_cmd.resolve()
    )


def test_prepare_managed_litellm_env_strips_debug_and_profile_env_vars(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token-value")
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.setenv("DEBUG", "release")
    monkeypatch.setenv("DETAILED_DEBUG", "verbose")
    monkeypatch.setenv("AWS_PROFILE", "some-profile")
    monkeypatch.setenv("AWS_DEFAULT_PROFILE", "other-profile")

    env = _prepare_managed_litellm_env(
        litellm_config=(REPO_ROOT / "configs" / "litellm_proxy.yaml").resolve()
    )

    assert env["AWS_BEARER_TOKEN_BEDROCK"] == "token-value"
    assert env["AWS_REGION_NAME"] == "us-west-2"
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert "DEBUG" not in env
    assert "DETAILED_DEBUG" not in env
    assert "AWS_PROFILE" not in env
    assert "AWS_DEFAULT_PROFILE" not in env


def test_read_log_tail_returns_recent_lines(tmp_path):
    log_path = tmp_path / "litellm.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    assert read_log_tail(log_path, max_lines=2) == "line2\nline3"
    assert read_log_tail(tmp_path / "missing.log") == ""


def test_load_trial_metrics_and_summarize_trial_rows(tmp_path):
    run_root = tmp_path / "emo_sta_run"
    run_root.mkdir(parents=True, exist_ok=True)
    comparison_summary = {
        "shared_run": {
            "best_program_info": {
                "metrics": {"score": 0.8, "combined_score": 0.8},
            }
        },
        "tasks": {
            "task_a": {
                "spawn_best_score": 0.70,
                "adapted_best_score": 0.75,
                "best_shared_adaptation_best_score": 0.71,
                "best_local_adaptation_best_score": 0.74,
                "baseline_best_score": 0.72,
            },
            "task_b": {
                "spawn_best_score": 0.60,
                "adapted_best_score": 0.58,
                "best_shared_adaptation_best_score": 0.56,
                "best_local_adaptation_best_score": 0.57,
                "baseline_best_score": 0.55,
            },
        },
    }
    (run_root / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2),
        encoding="utf-8",
    )

    row = load_trial_metrics(run_root)
    assert row["shared_best_score"] == 0.8
    assert row["spawn_mean_score"] == pytest.approx(0.65)
    assert row["adapted_mean_score"] == pytest.approx(0.665)
    assert row["best_shared_mean_score"] == pytest.approx(0.635)
    assert row["best_local_mean_score"] == pytest.approx(0.655)
    assert row["baseline_mean_score"] == pytest.approx(0.635)
    assert row["adapted_vs_spawn_counts"] == {
        "wins": 1,
        "ties": 0,
        "losses": 1,
        "comparable": 2,
    }
    assert row["adapted_vs_best_shared_counts"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
        "comparable": 2,
    }
    assert row["adapted_vs_best_local_counts"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
        "comparable": 2,
    }
    assert row["best_local_vs_best_shared_counts"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
        "comparable": 2,
    }
    assert row["adapted_vs_baseline_counts"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
        "comparable": 2,
    }

    summary = summarize_trial_rows([row, row])
    assert summary["shared_best_score"]["count"] == 2
    assert summary["shared_best_score"]["mean"] == 0.8
    assert summary["tasks"]["task_a"]["adapted_best_score"]["mean"] == 0.75
    assert (
        summary["tasks"]["task_a"]["best_shared_adaptation_best_score"]["mean"]
        == 0.71
    )
    assert (
        summary["tasks"]["task_a"]["best_local_adaptation_best_score"]["mean"]
        == 0.74
    )
    assert summary["best_shared_mean_score"]["mean"] == pytest.approx(0.635)
    assert summary["best_local_mean_score"]["mean"] == pytest.approx(0.655)
    assert summary["adapted_vs_spawn_counts"]["wins"] == 2
    assert summary["adapted_vs_best_shared_counts"]["wins"] == 4
    assert summary["adapted_vs_best_local_counts"]["wins"] == 4
    assert summary["best_local_vs_best_shared_counts"]["wins"] == 4


def test_trials_main_rejects_unsafe_iteration_budget(monkeypatch, tmp_path):
    manifest = SimpleNamespace(
        family="dummy_family",
        output_root=tmp_path,
        default_shared_iterations=20,
        default_adaptation_iterations=20,
        default_baseline_iterations=25,
        base_config=tmp_path / "config.yaml",
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "parse_args",
        lambda: SimpleNamespace(
            trials=1,
            parallel_trials=1,
            launch_delay_sec=0.0,
            litellm_wait_sec=30.0,
            litellm_port_search_limit=None,
            start_trial_number=None,
            manifest="dummy.yaml",
            output_root=None,
            shared_iterations=60,
            adaptation_iterations=20,
            baseline_iterations=31,
            skip_adaptation=False,
            skip_baselines=False,
            allow_unsafe_iterations=False,
        ),
    )
    monkeypatch.setattr(emo_sta_trials_runner, "resolve_repo_path", lambda root, path: Path(path))
    monkeypatch.setattr(emo_sta_trials_runner, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "family_task_specs",
        lambda manifest: [SimpleNamespace(task_id=f"task_{i}") for i in range(4)],
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "load_launcher_defaults",
        lambda path: SimpleNamespace(
            modules=(),
            setup_commands=(),
            litellm_mode="skip",
            litellm_command="litellm",
            litellm_preferred_env=None,
            litellm_config=None,
            litellm_host="127.0.0.1",
            litellm_port=4000,
        ),
    )

    with pytest.raises(SystemExit, match="Unsafe EMO-STA iteration setting"):
        emo_sta_trials_runner.main()


def test_trials_main_runs_family_preflight_before_launcher_setup(monkeypatch, tmp_path):
    manifest = SimpleNamespace(
        family="circle_packing",
        output_root=tmp_path,
        default_shared_iterations=40,
        default_adaptation_iterations=15,
        default_baseline_iterations=25,
        base_config=tmp_path / "config.yaml",
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "parse_args",
        lambda: SimpleNamespace(
            trials=1,
            parallel_trials=1,
            launch_delay_sec=0.0,
            litellm_wait_sec=30.0,
            litellm_port_search_limit=None,
            start_trial_number=None,
            manifest="dummy.yaml",
            shared_iterations=None,
            adaptation_iterations=None,
            baseline_iterations=None,
            skip_adaptation=False,
            skip_baselines=False,
            allow_unsafe_iterations=False,
        ),
    )
    monkeypatch.setattr(emo_sta_trials_runner, "resolve_repo_path", lambda root, path: Path(path))
    monkeypatch.setattr(emo_sta_trials_runner, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "family_task_specs",
        lambda manifest: [SimpleNamespace(task_id=f"task_{i}") for i in range(4)],
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "run_emo_sta_family_preflight",
        lambda manifest, task_specs: (_ for _ in ()).throw(RuntimeError("assets missing")),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "load_launcher_defaults",
        lambda path: (_ for _ in ()).throw(
            AssertionError("launcher defaults should not be loaded after preflight failure")
        ),
    )

    with pytest.raises(SystemExit, match="assets missing"):
        emo_sta_trials_runner.main()


def test_trials_main_writes_clean_config_without_removed_scratch_flags(monkeypatch, tmp_path):
    manifest = SimpleNamespace(
        family="heilbronn_triangle",
        output_root=tmp_path / "manifest_results",
        default_shared_iterations=60,
        default_adaptation_iterations=15,
        default_baseline_iterations=30,
        base_config=tmp_path / "config.yaml",
    )
    trial_manifest = tmp_path / "trial_manifest.yaml"
    trial_manifest.write_text("family: heilbronn_triangle\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("llm: {}\n", encoding="utf-8")

    args = SimpleNamespace(
        trials=1,
        parallel_trials=1,
        launch_delay_sec=0.0,
        litellm_wait_sec=30.0,
        litellm_port_search_limit=None,
        start_trial_number=1,
        manifest="dummy.yaml",
        output_root=str(tmp_path / "results"),
        shared_iterations=60,
        adaptation_iterations=15,
        baseline_iterations=30,
        run_warmstart_adaptation=True,
        run_best_shared_adaptation=True,
        best_shared_adaptation_iterations=15,
        run_best_local_adaptation=True,
        best_local_adaptation_iterations=15,
        run_name_prefix="run",
        model=None,
        primary_model="claude-sonnet-4-6",
        secondary_model=None,
        api_base=None,
        base_seed=42,
        seed_step=1,
        python=sys.executable,
        skip_adaptation=False,
        skip_baselines=False,
        shared_checkpoint=None,
        force=False,
        allow_unsafe_iterations=False,
        module=None,
        setup_command=None,
        litellm="skip",
        litellm_per_trial=None,
        litellm_command=None,
        litellm_config=None,
        litellm_host=None,
        litellm_port=None,
        leave_litellm_running=False,
        trial_log_dir=None,
        litellm_log_file=None,
        log_file=None,
        summary_json_out=None,
        refresh_report=False,
        report_markdown_out=None,
        report_json_out=None,
        report_latest_per_setting=None,
        foreground=True,
        dry_run=False,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(emo_sta_trials_runner, "parse_args", lambda: args)
    monkeypatch.setattr(emo_sta_trials_runner, "resolve_repo_path", lambda root, path: Path(path))
    monkeypatch.setattr(emo_sta_trials_runner, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "family_task_specs",
        lambda manifest: [SimpleNamespace(task_id=f"task_{i}") for i in range(4)],
    )
    monkeypatch.setattr(
        emo_sta_trials_runner, "validate_emo_sta_iteration_budget", lambda **kwargs: None
    )
    monkeypatch.setattr(
        emo_sta_trials_runner, "run_emo_sta_family_preflight", lambda manifest, task_specs: None
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "load_launcher_defaults",
        lambda path: SimpleNamespace(
            modules=(),
            setup_commands=(),
            litellm_mode="skip",
            litellm_per_trial=False,
            litellm_command="litellm",
            litellm_preferred_env=None,
            litellm_config=None,
            litellm_host="127.0.0.1",
            litellm_port=4000,
            litellm_port_search_limit=200,
        ),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner, "build_setting_output_dir_name", lambda **kwargs: "test-setting"
    )
    monkeypatch.setattr(emo_sta_trials_runner, "read_base_config_edit_mode", lambda path: "full")
    monkeypatch.setattr(emo_sta_trials_runner, "read_base_config_api_base", lambda path: None)
    monkeypatch.setattr(
        emo_sta_trials_runner, "write_seeded_trial_manifest", lambda **kwargs: (trial_manifest, None)
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "_run_trial_command",
        lambda **kwargs: captured.setdefault("command", list(kwargs["command"])),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "load_trial_metrics",
        lambda run_root: {
            "shared_best_score": 0.8,
            "spawn_mean_score": 0.7,
            "adapted_mean_score": 0.75,
            "best_shared_mean_score": 0.68,
            "best_local_mean_score": 0.71,
            "baseline_mean_score": 0.66,
            "adapted_minus_spawn_mean": 0.05,
            "adapted_minus_best_shared_mean": 0.07,
            "adapted_minus_best_local_mean": 0.04,
            "best_local_minus_best_shared_mean": 0.03,
            "adapted_minus_baseline_mean": 0.09,
            "adapted_vs_spawn_counts": {
                "wins": 1,
                "ties": 0,
                "losses": 0,
                "comparable": 1,
            },
            "adapted_vs_best_shared_counts": {
                "wins": 1,
                "ties": 0,
                "losses": 0,
                "comparable": 1,
            },
            "adapted_vs_best_local_counts": {
                "wins": 1,
                "ties": 0,
                "losses": 0,
                "comparable": 1,
            },
            "best_local_vs_best_shared_counts": {
                "wins": 1,
                "ties": 0,
                "losses": 0,
                "comparable": 1,
            },
            "adapted_vs_baseline_counts": {
                "wins": 1,
                "ties": 0,
                "losses": 0,
                "comparable": 1,
            },
            "tasks": {},
        },
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "run_trial_workers",
        lambda trials, parallel_trials, launch_delay_sec, worker: [worker(0)],
    )

    assert emo_sta_trials_runner.main() == 0

    command = captured["command"]
    assert command[:8] == [
        sys.executable,
        str(REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_multi_task_shared_then_adapt.py"),
        "--manifest",
        str(trial_manifest),
        "--run-name",
        "run_01_seed_42",
        "--shared-iterations",
        "60",
    ]
    assert "--run-best-shared-adaptation" in command
    assert "--run-best-local-adaptation" in command
    assert "--best-shared-adaptation-iterations" in command
    assert "--best-local-adaptation-iterations" in command
    assert command[command.index("--best-shared-adaptation-iterations") + 1] == "15"
    assert command[command.index("--best-local-adaptation-iterations") + 1] == "15"

    summary_path = tmp_path / "results" / "test-setting" / "trial_summary.json"
    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert raw_summary["config"]["shared_iterations"] == 60
    assert raw_summary["config"]["adaptation_iterations"] == 15
    assert raw_summary["config"]["baseline_iterations"] == 30
    assert raw_summary["config"]["run_warmstart_adaptation"] is True
    assert raw_summary["config"]["run_best_shared_adaptation"] is True
    assert raw_summary["config"]["best_shared_adaptation_iterations"] == 15
    assert raw_summary["config"]["run_best_local_adaptation"] is True
    assert raw_summary["config"]["best_local_adaptation_iterations"] == 15
    assert raw_summary["config"]["model"] == "claude-sonnet-4-6"


def test_trials_main_dry_run_writes_commands_without_launching(monkeypatch, tmp_path):
    manifest = SimpleNamespace(
        family="heilbronn_triangle",
        output_root=tmp_path / "manifest_results",
        default_shared_iterations=60,
        default_adaptation_iterations=15,
        default_baseline_iterations=30,
        base_config=tmp_path / "config.yaml",
    )
    trial_manifest = tmp_path / "trial_manifest.yaml"
    trial_base_config = tmp_path / "trial_base_config.yaml"
    args = SimpleNamespace(
        trials=1,
        parallel_trials=1,
        launch_delay_sec=0.0,
        litellm_wait_sec=30.0,
        litellm_port_search_limit=None,
        start_trial_number=1,
        manifest="dummy.yaml",
        output_root=str(tmp_path / "results"),
        shared_iterations=60,
        adaptation_iterations=15,
        baseline_iterations=30,
        run_warmstart_adaptation=True,
        run_best_shared_adaptation=True,
        best_shared_adaptation_iterations=15,
        run_best_local_adaptation=True,
        best_local_adaptation_iterations=15,
        run_name_prefix="run",
        model=None,
        primary_model="claude-sonnet-4-6",
        secondary_model=None,
        api_base=None,
        base_seed=42,
        seed_step=1,
        python=sys.executable,
        skip_adaptation=False,
        skip_baselines=False,
        shared_checkpoint=None,
        force=False,
        allow_unsafe_iterations=False,
        module=None,
        setup_command=None,
        litellm="skip",
        litellm_per_trial=None,
        litellm_command=None,
        litellm_config=None,
        litellm_host=None,
        litellm_port=None,
        leave_litellm_running=False,
        trial_log_dir=None,
        litellm_log_file=None,
        log_file=None,
        summary_json_out=None,
        refresh_report=True,
        report_markdown_out=None,
        report_json_out=None,
        report_latest_per_setting=None,
        foreground=False,
        dry_run=True,
    )

    monkeypatch.setattr(emo_sta_trials_runner, "parse_args", lambda: args)
    monkeypatch.setattr(emo_sta_trials_runner, "resolve_repo_path", lambda root, path: Path(path))
    monkeypatch.setattr(emo_sta_trials_runner, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "family_task_specs",
        lambda manifest: [SimpleNamespace(task_id=f"task_{i}") for i in range(4)],
    )
    monkeypatch.setattr(
        emo_sta_trials_runner, "validate_emo_sta_iteration_budget", lambda **kwargs: None
    )
    monkeypatch.setattr(
        emo_sta_trials_runner, "run_emo_sta_family_preflight", lambda manifest, task_specs: None
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "load_launcher_defaults",
        lambda path: SimpleNamespace(
            modules=(),
            setup_commands=(),
            litellm_mode="skip",
            litellm_per_trial=False,
            litellm_command="litellm",
            litellm_preferred_env=None,
            litellm_config=None,
            litellm_host="127.0.0.1",
            litellm_port=4000,
            litellm_port_search_limit=200,
        ),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner, "build_setting_output_dir_name", lambda **kwargs: "test-setting"
    )
    monkeypatch.setattr(emo_sta_trials_runner, "read_base_config_edit_mode", lambda path: "full")
    monkeypatch.setattr(emo_sta_trials_runner, "read_base_config_api_base", lambda path: None)
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "write_seeded_trial_manifest",
        lambda **kwargs: (trial_manifest, trial_base_config),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "launch_detached",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run should not detach")
        ),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "_run_trial_command",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run should not launch trials")
        ),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "run_trial_workers",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run should not dispatch workers")
        ),
    )
    monkeypatch.setattr(
        emo_sta_trials_runner,
        "_refresh_report",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run should not refresh reports")
        ),
    )

    assert emo_sta_trials_runner.main() == 0

    summary_path = tmp_path / "results" / "test-setting" / "trial_summary.json"
    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert raw_summary["dry_run"] is True
    assert raw_summary["config"]["trial_log_dir"] == str(
        tmp_path / "results" / "test-setting" / "trial_logs"
    )
    command = raw_summary["trials"][0]["command"]
    assert command[:6] == [
        sys.executable,
        str(REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_multi_task_shared_then_adapt.py"),
        "--manifest",
        str(trial_manifest),
        "--run-name",
        "run_01_seed_42",
    ]
    assert "--run-warmstart-adaptation" in command
    assert "--run-best-shared-adaptation" in command
    assert "--run-best-local-adaptation" in command
