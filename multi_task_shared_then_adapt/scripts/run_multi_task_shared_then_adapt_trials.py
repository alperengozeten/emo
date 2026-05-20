#!/usr/bin/env python3
"""Run repeated EMO-STA trials with optional LiteLLM and shell setup automation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CIRCLE_PACKING_MANIFEST = (
    "multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml"
)
DEFAULT_MANIFEST = CIRCLE_PACKING_MANIFEST
CIRCLE_PACKING_RECTANGLE_MANIFEST = (
    "multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml"
)
K_MODULE_BALANCED_MANIFEST = (
    "multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml"
)
FUNCTION_MINIMIZATION_MANIFEST = (
    "multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml"
)
HEILBRONN_TRIANGLE_MANIFEST = (
    "multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml"
)
SIGNAL_PROCESSING_MANIFEST = (
    "multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml"
)
SLDBENCH_3D_MANIFEST = (
    "multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml"
)
RUST_ADAPTIVE_SORT_MANIFEST = (
    "multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml"
)

from openevolve.multi_task_shared_then_adapt.trials import (
    DEFAULT_LITELLM_COMMAND,
    DEFAULT_LITELLM_CONFIG,
    DEFAULT_LITELLM_HOST,
    DEFAULT_LITELLM_MODE,
    DEFAULT_LITELLM_PORT,
    DEFAULT_LITELLM_PORT_SEARCH_LIMIT,
    DEFAULT_REPORT_JSON,
    DEFAULT_REPORT_MARKDOWN,
    build_setting_output_dir_name,
    build_litellm_command,
    build_shell_command,
    build_trial_run_name,
    launch_detached,
    load_launcher_defaults,
    load_trial_metrics,
    next_trial_number,
    parse_api_base_host_port,
    read_log_tail,
    read_base_config_api_base,
    read_base_config_edit_mode,
    read_base_config_primary_model,
    release_reserved_tcp_port,
    reserve_available_tcp_port,
    resolve_litellm_command,
    resolve_repo_path,
    run_trial_workers,
    summarize_trial_rows,
    terminate_process_tree,
    trial_seed,
    wait_for_tcp_ready,
    write_json,
    write_seeded_trial_manifest,
)
from openevolve.multi_task_shared_then_adapt.workflow import load_manifest
from openevolve.multi_task_shared_then_adapt.workflow import (
    family_task_specs,
    run_emo_sta_family_preflight,
    validate_emo_sta_iteration_budget,
)


DEFAULT_LITELLM_WAIT_SEC = 120.0
BEDROCK_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"
DEFAULT_BEDROCK_API_KEY_REGION = "us-east-1"
AWS_PROFILE_ENV_VARS = ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated EMO-STA trials. This launcher can optionally start LiteLLM, "
            "run shell setup like 'module load R/4.5.1', detach to a launcher log, "
            "and write one per-trial log file. "
            f"Defaults to unit-square circle packing; use --manifest {CIRCLE_PACKING_RECTANGLE_MANIFEST} "
            f"for rectangle circle packing, --manifest {K_MODULE_BALANCED_MANIFEST} "
            f"for K-module, or --manifest "
            f"{FUNCTION_MINIMIZATION_MANIFEST} for function minimization, or "
            f"--manifest {HEILBRONN_TRIANGLE_MANIFEST} for Heilbronn triangle, or "
            f"--manifest {SIGNAL_PROCESSING_MANIFEST} for signal processing, or "
            f"--manifest {SLDBENCH_3D_MANIFEST} for SLDBench 3D scaling laws, or "
            f"--manifest {RUST_ADAPTIVE_SORT_MANIFEST} for Rust adaptive sort."
        )
    )
    parser.add_argument(
        "--manifest",
        "--benchmark-config",
        dest="manifest",
        default=DEFAULT_MANIFEST,
        help=(
            "Path to the EMO-STA benchmark configuration file. "
            f"Default: {DEFAULT_MANIFEST}. "
            f"Unit-square circle packing: {CIRCLE_PACKING_MANIFEST}. "
            f"Rectangle circle packing: {CIRCLE_PACKING_RECTANGLE_MANIFEST}. "
            f"K-module: {K_MODULE_BALANCED_MANIFEST}. "
            f"Function minimization: {FUNCTION_MINIMIZATION_MANIFEST}. "
            f"Heilbronn triangle: {HEILBRONN_TRIANGLE_MANIFEST}. "
            f"Signal processing: {SIGNAL_PROCESSING_MANIFEST}. "
            f"SLDBench 3D: {SLDBENCH_3D_MANIFEST}. "
            f"Rust adaptive sort: {RUST_ADAPTIVE_SORT_MANIFEST}."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of repeated runs.",
    )
    parser.add_argument(
        "--shared-iterations",
        type=int,
        default=None,
        help="Override the manifest shared-phase iteration count.",
    )
    parser.add_argument(
        "--adaptation-iterations",
        type=int,
        default=None,
        help="Override the manifest adaptation-phase iteration count.",
    )
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=None,
        help="Override the manifest baseline-phase iteration count.",
    )
    parser.set_defaults(
        run_warmstart_adaptation=True,
        run_best_shared_adaptation=False,
        run_best_local_adaptation=False,
    )
    parser.add_argument(
        "--run-warmstart-adaptation",
        dest="run_warmstart_adaptation",
        action="store_true",
        help=(
            "Pass through to the EMO-STA runner to enable the standard "
            "archive-warmstart adaptation branch. This is enabled by default "
            "for backward compatibility."
        ),
    )
    parser.add_argument(
        "--skip-warmstart-adaptation",
        dest="run_warmstart_adaptation",
        action="store_false",
        help=(
            "Pass through to the EMO-STA runner to disable only the standard "
            "archive-warmstart adaptation branch."
        ),
    )
    parser.add_argument(
        "--run-best-shared-adaptation",
        dest="run_best_shared_adaptation",
        action="store_true",
        help=(
            "Pass through to the EMO-STA runner to enable the optional one-program "
            "best-shared adaptation branch."
        ),
    )
    parser.add_argument(
        "--skip-best-shared-adaptation",
        dest="run_best_shared_adaptation",
        action="store_false",
        help=(
            "Pass through to the EMO-STA runner to explicitly disable the optional "
            "best-shared adaptation branch."
        ),
    )
    parser.add_argument(
        "--best-shared-adaptation-iterations",
        type=int,
        default=None,
        help=(
            "Optional iteration override for the best-shared adaptation branch. "
            "Defaults to --adaptation-iterations in the EMO-STA runner."
        ),
    )
    parser.add_argument(
        "--run-best-local-adaptation",
        dest="run_best_local_adaptation",
        action="store_true",
        help=(
            "Pass through to the EMO-STA runner to enable the optional one-program "
            "best-local adaptation branch."
        ),
    )
    parser.add_argument(
        "--skip-best-local-adaptation",
        dest="run_best_local_adaptation",
        action="store_false",
        help=(
            "Pass through to the EMO-STA runner to explicitly disable the optional "
            "best-local adaptation branch."
        ),
    )
    parser.add_argument(
        "--best-local-adaptation-iterations",
        type=int,
        default=None,
        help=(
            "Optional iteration override for the best-local adaptation branch. "
            "Defaults to --adaptation-iterations in the EMO-STA runner."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Base directory under which setting-specific output directories are written.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="run",
        help="Prefix for generated trial run names.",
    )
    parser.add_argument(
        "--start-trial-number",
        type=int,
        default=None,
        help=(
            "First displayed trial number used in generated run directory names. "
            "If omitted, the launcher continues after the highest existing run number "
            "for this setting and prefix."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Alias for --primary-model.",
    )
    parser.add_argument(
        "--primary-model",
        default=None,
        help="Forwarded OpenEvolve primary model override.",
    )
    parser.add_argument(
        "--secondary-model",
        default=None,
        help="Forwarded OpenEvolve secondary model override.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "Forwarded OpenEvolve API base. If omitted, uses the base config api_base "
            "or the managed LiteLLM endpoint."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="First random seed to use for trial base_config snapshots.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment between trial seeds.",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help="Maximum number of EMO-STA runs to launch at the same time.",
    )
    parser.add_argument(
        "--launch-delay-sec",
        type=float,
        default=15.0,
        help=(
            "Optional delay between parallel trial launches. "
            "Useful when several runs hit W&B and LiteLLM at once."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for launched EMO-STA runs.",
    )
    parser.add_argument(
        "--skip-adaptation",
        action="store_true",
        help="Pass through to the EMO-STA runner.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Pass through to the EMO-STA runner.",
    )
    parser.add_argument(
        "--shared-checkpoint",
        default=None,
        help="Optional shared checkpoint path reused by every launched trial.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass through to the EMO-STA runner.",
    )
    parser.add_argument(
        "--allow-unsafe-iterations",
        action="store_true",
        help=(
            "Allow iteration-unsafe EMO-STA budgets where "
            "shared + task_count * adaptation != task_count * baseline."
        ),
    )
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help=(
            "Shell module to load before each launched command. May be passed multiple times. "
            "If omitted, launcher defaults from the manifest are used."
        ),
    )
    parser.add_argument(
        "--setup-command",
        action="append",
        default=None,
        help=(
            "Arbitrary shell command to run before each launched command. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--litellm",
        choices=("auto", "start", "skip"),
        default=None,
        help=(
            "LiteLLM management mode. 'auto' reuses an existing local endpoint if present, "
            "otherwise starts one. 'start' requires launching a fresh server. "
            "'skip' never manages LiteLLM."
        ),
    )
    parser.add_argument(
        "--litellm-per-trial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Start a fresh managed LiteLLM proxy for each trial on an available local port, "
            "then shut it down after that trial finishes. Defaults to the manifest launcher "
            "setting or enabled."
        ),
    )
    parser.add_argument(
        "--litellm-command",
        default=None,
        help="Command used to start LiteLLM. Defaults to the manifest launcher section or 'litellm'.",
    )
    parser.add_argument(
        "--litellm-config",
        default=None,
        help="LiteLLM config path. Defaults to the manifest launcher section or configs/litellm_proxy.yaml.",
    )
    parser.add_argument(
        "--litellm-host",
        default=None,
        help="Host passed to the managed LiteLLM process.",
    )
    parser.add_argument(
        "--litellm-port",
        type=int,
        default=None,
        help="Port passed to the managed LiteLLM process.",
    )
    parser.add_argument(
        "--litellm-port-search-limit",
        type=int,
        default=None,
        help=(
            "When --litellm-per-trial is enabled, scan this many ports starting from "
            "--litellm-port to find a free local port."
        ),
    )
    parser.add_argument(
        "--litellm-wait-sec",
        type=float,
        default=DEFAULT_LITELLM_WAIT_SEC,
        help="How long to wait for a managed LiteLLM process to become reachable.",
    )
    parser.add_argument(
        "--leave-litellm-running",
        action="store_true",
        help="If this launcher starts LiteLLM, leave it running after the trials finish.",
    )
    parser.add_argument(
        "--trial-log-dir",
        default=None,
        help="Directory for per-trial stdout/stderr logs.",
    )
    parser.add_argument(
        "--litellm-log-file",
        default=None,
        help="Optional log file path for the managed LiteLLM process.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path for the detached launcher log file.",
    )
    parser.add_argument(
        "--summary-json-out",
        default=None,
        help="Optional path for the repeated-trial summary JSON.",
    )
    parser.add_argument(
        "--refresh-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh the EMO-STA markdown/JSON report after the trials finish.",
    )
    parser.add_argument(
        "--report-markdown-out",
        default=None,
        help="Optional output path for the refreshed EMO-STA markdown report.",
    )
    parser.add_argument(
        "--report-json-out",
        default=None,
        help="Optional output path for the refreshed EMO-STA JSON report.",
    )
    parser.add_argument(
        "--report-latest-per-setting",
        type=int,
        default=None,
        help="latest-per-setting value passed to the EMO-STA report refresher. Defaults to --trials.",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in the foreground instead of detaching to a .nohup.log file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate launcher setup and write the per-trial runner commands without "
            "starting LiteLLM or running EMO-STA trials."
        ),
    )
    return parser.parse_args()


def _default_report_paths(
    *,
    repo_root: Path,
    output_root: Path,
    manifest_output_root: Path,
    explicit_markdown: str | None,
    explicit_json: str | None,
) -> tuple[Path, Path]:
    if explicit_markdown is not None:
        markdown_path = resolve_repo_path(repo_root, explicit_markdown)
    elif output_root.resolve() == manifest_output_root.resolve():
        markdown_path = resolve_repo_path(repo_root, DEFAULT_REPORT_MARKDOWN)
    else:
        markdown_path = output_root / "emo_sta_results_summary.md"

    if explicit_json is not None:
        json_path = resolve_repo_path(repo_root, explicit_json)
    elif output_root.resolve() == manifest_output_root.resolve():
        json_path = resolve_repo_path(repo_root, DEFAULT_REPORT_JSON)
    else:
        json_path = output_root / "emo_sta_results_summary.json"

    return markdown_path, json_path


def _load_litellm_proxy_config(litellm_config: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(litellm_config.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _uses_env_bedrock_api_key_auth(litellm_config: Path) -> bool:
    raw = _load_litellm_proxy_config(litellm_config)
    model_list = raw.get("model_list") or []
    if not isinstance(model_list, list):
        return False

    expected_api_key = f"os.environ/{BEDROCK_BEARER_TOKEN_ENV}"
    for entry in model_list:
        if not isinstance(entry, dict):
            continue
        litellm_params = entry.get("litellm_params") or {}
        if not isinstance(litellm_params, dict):
            continue
        model = str(litellm_params.get("model") or "").strip()
        api_key = str(litellm_params.get("api_key") or "").strip()
        if model.startswith("bedrock/") and api_key == expected_api_key:
            return True
    return False


def _prepare_managed_litellm_env(*, litellm_config: Path) -> Dict[str, str]:
    env = dict(os.environ)
    # LiteLLM's proxy CLI binds generic env vars like DEBUG/DETAILED_DEBUG to
    # boolean click flags. Shells that export values like DEBUG=release cause
    # the CLI to fail before the proxy binds its port.
    for key in ("DEBUG", "DETAILED_DEBUG"):
        env.pop(key, None)
    if not _uses_env_bedrock_api_key_auth(litellm_config):
        return env

    bearer_token = env.get(BEDROCK_BEARER_TOKEN_ENV, "").strip()
    if not bearer_token:
        raise SystemExit(
            f"{BEDROCK_BEARER_TOKEN_ENV} must be set before starting managed LiteLLM."
        )

    region = (
        env.get("AWS_REGION_NAME", "").strip()
        or env.get("AWS_DEFAULT_REGION", "").strip()
        or DEFAULT_BEDROCK_API_KEY_REGION
    )
    env[BEDROCK_BEARER_TOKEN_ENV] = bearer_token
    env["AWS_REGION_NAME"] = region
    env["AWS_DEFAULT_REGION"] = region

    # Force token-backed Bedrock auth for the managed LiteLLM process instead
    # of inheriting any shell profile selection from `aws login`.
    for key in AWS_PROFILE_ENV_VARS:
        env.pop(key, None)
    return env


def _log_indicates_litellm_port_conflict(log_tail: str) -> bool:
    lowered = log_tail.lower()
    return any(
        marker in lowered
        for marker in (
            "address already in use",
            "port is already in use",
            "errno 98",
            "errno 48",
        )
    )


def _probe_http_endpoint(url: str, *, timeout_sec: float) -> bool:
    request = Request(url, headers={"accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return 200 <= int(getattr(response, "status", 200)) < 500
    except HTTPError as exc:
        return 200 <= int(exc.code) < 500
    except (URLError, OSError):
        return False


def _ensure_litellm_healthy(api_base: str, *, timeout_sec: float) -> None:
    base = str(api_base).rstrip("/")
    probe_paths = ("/v1/models", "/models", "/health/liveliness", "/health")
    for path in probe_paths:
        if _probe_http_endpoint(base + path, timeout_sec=timeout_sec):
            return
    raise RuntimeError(
        f"LiteLLM health probe failed for {api_base}. Tried: "
        + ", ".join(base + path for path in probe_paths)
    )


def _start_managed_litellm(
    *,
    repo_root: Path,
    modules: List[str],
    setup_commands: List[str],
    litellm_command: str,
    litellm_preferred_env: str | None,
    litellm_config: Path,
    litellm_host: str,
    litellm_port: int,
    litellm_log_path: Path,
    wait_sec: float,
    managed_env: Dict[str, str],
) -> subprocess.Popen[str]:
    litellm_log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_litellm_command = resolve_litellm_command(
        litellm_command,
        preferred_env=litellm_preferred_env,
    )
    command = build_shell_command(
        build_litellm_command(
            litellm_command=resolved_litellm_command,
            litellm_config=litellm_config,
            host=litellm_host,
            port=litellm_port,
        ),
        modules=modules,
        setup_commands=setup_commands,
    )

    with litellm_log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=managed_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    try:
        wait_for_tcp_ready(
            litellm_host,
            litellm_port,
            timeout_sec=wait_sec,
            process=process,
        )
        if process.poll() is not None:
            raise RuntimeError(
                f"Managed LiteLLM process exited during startup for {litellm_host}:{litellm_port} "
                f"with code {process.returncode}."
            )
        _ensure_litellm_healthy(
            f"http://{litellm_host}:{litellm_port}",
            timeout_sec=min(max(wait_sec / 2.0, 1.0), 5.0),
        )
        if process.poll() is not None:
            raise RuntimeError(
                f"Managed LiteLLM process exited after readiness for {litellm_host}:{litellm_port} "
                f"with code {process.returncode}."
            )
    except Exception as exc:
        if process.poll() is None:
            terminate_process_tree(process)
        tail = read_log_tail(litellm_log_path)
        message = (
            "Managed LiteLLM failed to start. "
            f"See {litellm_log_path}."
        )
        if tail:
            message += f"\nRecent LiteLLM log output:\n{tail}"
        raise RuntimeError(message) from exc
    return process


def _run_trial_command(
    *,
    command: List[str],
    cwd: Path,
    modules: List[str],
    setup_commands: List[str],
    log_path: Path,
) -> None:
    wrapped_command = build_shell_command(
        command,
        modules=modules,
        setup_commands=setup_commands,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {' '.join(wrapped_command)}\n")
        log_handle.flush()
        subprocess.run(
            wrapped_command,
            cwd=cwd,
            env=dict(os.environ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def _build_runner_command(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    runner_script: Path,
    trial_manifest: Path,
    run_name: str,
    trial_api_base: str | None,
    primary_model: str | None,
) -> List[str]:
    command = [
        args.python,
        str(runner_script),
        "--manifest",
        str(trial_manifest),
        "--run-name",
        run_name,
    ]
    if args.shared_iterations is not None:
        command.extend(["--shared-iterations", str(args.shared_iterations)])
    if args.adaptation_iterations is not None:
        command.extend(["--adaptation-iterations", str(args.adaptation_iterations)])
    if args.baseline_iterations is not None:
        command.extend(["--baseline-iterations", str(args.baseline_iterations)])
    if args.allow_unsafe_iterations:
        command.append("--allow-unsafe-iterations")
    if args.run_warmstart_adaptation:
        command.append("--run-warmstart-adaptation")
    else:
        command.append("--skip-warmstart-adaptation")
    if args.run_best_shared_adaptation:
        command.append("--run-best-shared-adaptation")
    if args.best_shared_adaptation_iterations is not None:
        command.extend(
            [
                "--best-shared-adaptation-iterations",
                str(args.best_shared_adaptation_iterations),
            ]
        )
    if args.run_best_local_adaptation:
        command.append("--run-best-local-adaptation")
    if args.best_local_adaptation_iterations is not None:
        command.extend(
            [
                "--best-local-adaptation-iterations",
                str(args.best_local_adaptation_iterations),
            ]
        )
    if trial_api_base:
        command.extend(["--api-base", str(trial_api_base)])
    if primary_model:
        command.extend(["--primary-model", primary_model])
    if args.secondary_model:
        command.extend(["--secondary-model", args.secondary_model])
    if args.skip_adaptation:
        command.append("--skip-adaptation")
    if args.skip_baselines:
        command.append("--skip-baselines")
    if args.shared_checkpoint:
        command.extend(
            [
                "--shared-checkpoint",
                str(resolve_repo_path(repo_root, args.shared_checkpoint)),
            ]
        )
    if args.force:
        command.append("--force")
    return command


def _refresh_report(
    *,
    repo_root: Path,
    manifest_path: Path,
    results_dir: Path,
    latest_per_setting: int,
    markdown_out: Path,
    json_out: Path,
) -> None:
    report_script = (
        repo_root
        / "multi_task_shared_then_adapt"
        / "scripts"
        / "report_emo_sta_results.py"
    )
    command = [
        sys.executable,
        str(report_script),
        "--manifest",
        str(manifest_path),
        "--results-dir",
        str(results_dir),
        "--latest-per-setting",
        str(int(latest_per_setting)),
        "--markdown-out",
        str(markdown_out),
        "--json-out",
        str(json_out),
    ]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def main() -> int:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.parallel_trials < 1:
        raise SystemExit("--parallel-trials must be at least 1.")
    if args.launch_delay_sec < 0:
        raise SystemExit("--launch-delay-sec must be non-negative.")
    if args.litellm_wait_sec <= 0:
        raise SystemExit("--litellm-wait-sec must be positive.")
    if args.litellm_port_search_limit is not None and args.litellm_port_search_limit < 1:
        raise SystemExit("--litellm-port-search-limit must be at least 1.")
    if args.start_trial_number is not None and args.start_trial_number < 1:
        raise SystemExit("--start-trial-number must be at least 1.")

    manifest_path = resolve_repo_path(REPO_ROOT, args.manifest)
    manifest = load_manifest(manifest_path)
    task_specs = family_task_specs(manifest)

    shared_iterations = (
        int(args.shared_iterations)
        if args.shared_iterations is not None
        else manifest.default_shared_iterations
    )
    adaptation_iterations = (
        int(args.adaptation_iterations)
        if args.adaptation_iterations is not None
        else manifest.default_adaptation_iterations
    )
    baseline_iterations = (
        int(args.baseline_iterations)
        if args.baseline_iterations is not None
        else manifest.default_baseline_iterations
    )
    try:
        validate_emo_sta_iteration_budget(
            task_count=len(task_specs),
            shared_iterations=shared_iterations,
            adaptation_iterations=adaptation_iterations,
            baseline_iterations=baseline_iterations,
            skip_adaptation=args.skip_adaptation,
            skip_baselines=args.skip_baselines,
            allow_unsafe_iterations=args.allow_unsafe_iterations,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        run_emo_sta_family_preflight(manifest, task_specs=task_specs)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    launcher_defaults = load_launcher_defaults(manifest_path)

    base_output_root = Path(args.output_root).resolve() if args.output_root else manifest.output_root
    base_output_root.mkdir(parents=True, exist_ok=True)
    primary_model = (
        args.primary_model
        or args.model
        or read_base_config_primary_model(manifest.base_config)
    )
    edit_mode = read_base_config_edit_mode(manifest.base_config)
    setting_dir_name = build_setting_output_dir_name(
        shared_iterations=shared_iterations,
        adaptation_iterations=adaptation_iterations,
        baseline_iterations=baseline_iterations,
        primary_model=primary_model,
        edit_mode=edit_mode,
    )
    output_root = base_output_root / setting_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    start_trial_number = (
        int(args.start_trial_number)
        if args.start_trial_number is not None
        else next_trial_number(output_root, args.run_name_prefix)
    )
    trial_log_dir = (
        resolve_repo_path(REPO_ROOT, args.trial_log_dir)
        if args.trial_log_dir
        else output_root / "trial_logs"
    )
    log_path = (
        resolve_repo_path(REPO_ROOT, args.log_file)
        if args.log_file
        else output_root / f"{manifest_path.stem}.nohup.log"
    )
    summary_json_path = (
        resolve_repo_path(REPO_ROOT, args.summary_json_out)
        if args.summary_json_out
        else output_root / "trial_summary.json"
    )
    report_markdown_path, report_json_path = _default_report_paths(
        repo_root=REPO_ROOT,
        output_root=output_root,
        manifest_output_root=base_output_root,
        explicit_markdown=args.report_markdown_out,
        explicit_json=args.report_json_out,
    )

    if not args.foreground and not args.dry_run:
        forwarded_args = list(sys.argv[1:])
        if args.log_file is None:
            forwarded_args.extend(["--log-file", str(log_path)])
        forwarded_args.append("--foreground")
        pid = launch_detached(
            script_path=resolve_repo_path(REPO_ROOT, __file__),
            forwarded_args=forwarded_args,
            cwd=REPO_ROOT,
            log_path=log_path,
        )
        print("Detached EMO-STA launcher started.")
        print(f"PID: {pid}")
        print(f"Log: {log_path}")
        return 0

    if os.getenv("OPENEVOLVE_DETACHED") == "1":
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except AttributeError:
            pass
        print(f"Logging to {log_path}")

    modules = list(args.module) if args.module is not None else list(launcher_defaults.modules)
    setup_commands = (
        list(args.setup_command)
        if args.setup_command is not None
        else list(launcher_defaults.setup_commands)
    )

    litellm_mode = args.litellm or launcher_defaults.litellm_mode or DEFAULT_LITELLM_MODE
    litellm_per_trial = (
        args.litellm_per_trial
        if args.litellm_per_trial is not None
        else launcher_defaults.litellm_per_trial
    )
    litellm_command = args.litellm_command or launcher_defaults.litellm_command or DEFAULT_LITELLM_COMMAND
    litellm_preferred_env = launcher_defaults.litellm_preferred_env
    litellm_config = (
        resolve_repo_path(REPO_ROOT, args.litellm_config)
        if args.litellm_config
        else launcher_defaults.litellm_config
        or resolve_repo_path(REPO_ROOT, DEFAULT_LITELLM_CONFIG)
    )
    litellm_host = args.litellm_host or launcher_defaults.litellm_host or DEFAULT_LITELLM_HOST
    litellm_port = int(args.litellm_port or launcher_defaults.litellm_port or DEFAULT_LITELLM_PORT)
    litellm_port_search_limit = int(
        args.litellm_port_search_limit
        if args.litellm_port_search_limit is not None
        else launcher_defaults.litellm_port_search_limit or DEFAULT_LITELLM_PORT_SEARCH_LIMIT
    )
    litellm_log_path = (
        resolve_repo_path(REPO_ROOT, args.litellm_log_file)
        if args.litellm_log_file
        else output_root / "litellm.log"
    )
    uses_bedrock_api_key_auth = _uses_env_bedrock_api_key_auth(litellm_config)
    if litellm_port_search_limit < 1:
        raise SystemExit("--litellm-port-search-limit must be at least 1.")
    if litellm_per_trial and args.leave_litellm_running:
        raise SystemExit(
            "--leave-litellm-running is incompatible with --litellm-per-trial because "
            "per-trial mode always tears down its trial-scoped proxy."
        )

    base_config_api_base = read_base_config_api_base(manifest.base_config)
    local_litellm_api_base = f"http://{litellm_host}:{litellm_port}"
    effective_api_base = args.api_base or base_config_api_base or local_litellm_api_base
    api_endpoint = parse_api_base_host_port(effective_api_base)
    local_litellm_endpoint = parse_api_base_host_port(local_litellm_api_base)

    def build_summary_config(*, litellm_log_file: str | None) -> Dict[str, Any]:
        return {
            "manifest": str(manifest_path),
            "base_output_root": str(base_output_root),
            "output_root": str(output_root),
            "setting_dir_name": setting_dir_name,
            "trials": args.trials,
            "shared_iterations": shared_iterations,
            "adaptation_iterations": adaptation_iterations,
            "baseline_iterations": baseline_iterations,
            "run_warmstart_adaptation": args.run_warmstart_adaptation,
            "run_best_shared_adaptation": (
                args.run_best_shared_adaptation
            ),
            "best_shared_adaptation_iterations": (
                args.best_shared_adaptation_iterations
            ),
            "run_best_local_adaptation": (
                args.run_best_local_adaptation
            ),
            "best_local_adaptation_iterations": (
                args.best_local_adaptation_iterations
            ),
            "model": primary_model,
            "edit_mode": edit_mode,
            "secondary_model": args.secondary_model,
            "api_base": effective_api_base,
            "base_seed": args.base_seed,
            "seed_step": args.seed_step,
            "parallel_trials": args.parallel_trials,
            "launch_delay_sec": args.launch_delay_sec,
            "start_trial_number": start_trial_number,
            "modules": modules,
            "setup_commands": setup_commands,
            "litellm_mode": litellm_mode,
            "litellm_per_trial": litellm_per_trial,
            "litellm_command": litellm_command if litellm_mode != "skip" else None,
            "litellm_config": str(litellm_config) if litellm_mode != "skip" else None,
            "litellm_host": litellm_host if litellm_mode != "skip" else None,
            "litellm_port": litellm_port if litellm_mode != "skip" else None,
            "litellm_port_search_limit": (
                litellm_port_search_limit if litellm_mode != "skip" else None
            ),
            "litellm_log_file": litellm_log_file,
            "skip_adaptation": args.skip_adaptation,
            "skip_baselines": args.skip_baselines,
            "shared_checkpoint": args.shared_checkpoint,
            "force": args.force,
            "allow_unsafe_iterations": args.allow_unsafe_iterations,
            "trial_log_dir": str(trial_log_dir),
        }

    managed_litellm: subprocess.Popen[str] | None = None
    try:
        should_manage_litellm = (
            litellm_mode != "skip"
            and api_endpoint is not None
            and local_litellm_endpoint is not None
            and api_endpoint == local_litellm_endpoint
        )
        managed_litellm_env: Dict[str, str] | None = None

        if litellm_mode != "skip" and not should_manage_litellm:
            print(
                "Skipping LiteLLM management because the effective api_base "
                f"{effective_api_base!r} does not target the configured local endpoint "
                f"{local_litellm_api_base!r}."
            )

        runner_script = REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_multi_task_shared_then_adapt.py"
        if args.dry_run:
            dry_run_manifest_dir = output_root / "dry_run_manifests"
            dry_run_manifest_dir.mkdir(parents=True, exist_ok=True)
            dry_run_rows: List[Dict[str, Any]] = []

            print("Dry run only: no LiteLLM process or EMO-STA trial will be started.")
            print(f"Output root: {output_root}")
            print(f"Trial logs would be written under: {trial_log_dir}")

            for trial_idx in range(args.trials):
                seed = trial_seed(args.base_seed, args.seed_step, trial_idx)
                display_trial_idx = start_trial_number - 1 + trial_idx
                run_name = build_trial_run_name(
                    trial_idx=display_trial_idx,
                    seed=seed,
                    prefix=args.run_name_prefix,
                )
                run_root = output_root / run_name
                trial_log_path = trial_log_dir / f"{run_name}.log"
                temp_dir = dry_run_manifest_dir / run_name
                temp_dir.mkdir(parents=True, exist_ok=True)
                strip_trial_config_api_base = should_manage_litellm and litellm_per_trial
                trial_manifest, trial_base_config = write_seeded_trial_manifest(
                    manifest_path=manifest_path,
                    seed=seed,
                    temp_dir=temp_dir,
                    output_root=output_root,
                    strip_api_base=strip_trial_config_api_base,
                )
                command = _build_runner_command(
                    args=args,
                    repo_root=REPO_ROOT,
                    runner_script=runner_script,
                    trial_manifest=trial_manifest,
                    run_name=run_name,
                    trial_api_base=effective_api_base,
                    primary_model=primary_model,
                )
                wrapped_command = build_shell_command(
                    command,
                    modules=modules,
                    setup_commands=setup_commands,
                )
                dry_run_rows.append(
                    {
                        "trial": display_trial_idx + 1,
                        "seed": seed,
                        "run_name": run_name,
                        "run_root": str(run_root),
                        "trial_log_path": str(trial_log_path),
                        "trial_manifest": str(trial_manifest),
                        "trial_base_config": (
                            str(trial_base_config)
                            if trial_base_config is not None
                            else None
                        ),
                        "command": command,
                        "wrapped_command": wrapped_command,
                        "litellm_api_base": (
                            effective_api_base if should_manage_litellm else None
                        ),
                        "per_trial_litellm_port_selected_at_runtime": (
                            bool(should_manage_litellm and litellm_per_trial)
                        ),
                    }
                )
                print(f"\nTrial {trial_idx + 1}/{args.trials}: {run_name}")
                print(f"Manifest: {trial_manifest}")
                print("$ " + shlex.join(command))

            dry_run_summary = {
                "dry_run": True,
                "config": build_summary_config(litellm_log_file=None),
                "trials": dry_run_rows,
            }
            summary_path = write_json(summary_json_path, dry_run_summary)
            print(f"\nDry-run summary written to {summary_path}")
            return 0

        if should_manage_litellm and litellm_per_trial:
            print(
                "Using per-trial managed LiteLLM. Each trial will start its own proxy on the "
                f"first available port beginning at {litellm_port} and shut it down afterward."
            )
        elif should_manage_litellm:
            if api_endpoint is None:
                raise SystemExit(
                    "Managed LiteLLM requires a parseable local --api-base or base config api_base."
                )
            else:
                endpoint_host, endpoint_port = api_endpoint
                endpoint_ready = False
                try:
                    wait_for_tcp_ready(endpoint_host, endpoint_port, timeout_sec=1.0)
                    endpoint_ready = True
                except TimeoutError:
                    endpoint_ready = False

                if endpoint_ready:
                    if litellm_mode == "start":
                        raise SystemExit(
                            "LiteLLM is already reachable at "
                            f"{effective_api_base}, but --litellm start requires a fresh server. "
                            "Stop the existing server first or switch to --litellm auto/skip."
                        )
                    if uses_bedrock_api_key_auth:
                        print(
                            "Reusing existing LiteLLM server at "
                            f"{effective_api_base} with {BEDROCK_BEARER_TOKEN_ENV}-backed "
                            "config. Assuming the proxy is already managed intentionally."
                        )
                    else:
                        print(f"Reusing existing LiteLLM server at {effective_api_base}")
                else:
                    if managed_litellm_env is None:
                        managed_litellm_env = _prepare_managed_litellm_env(
                            litellm_config=litellm_config
                        )
                    print(
                        f"Starting managed LiteLLM with config {litellm_config} "
                        f"on {litellm_host}:{litellm_port}"
                    )
                    managed_litellm = _start_managed_litellm(
                        repo_root=REPO_ROOT,
                        modules=modules,
                        setup_commands=setup_commands,
                        litellm_command=litellm_command,
                        litellm_preferred_env=litellm_preferred_env,
                        litellm_config=litellm_config,
                        litellm_host=litellm_host,
                        litellm_port=litellm_port,
                        litellm_log_path=litellm_log_path,
                        wait_sec=args.litellm_wait_sec,
                        managed_env=managed_litellm_env,
                    )
                    effective_api_base = local_litellm_api_base
                    print(f"Managed LiteLLM ready at {effective_api_base}")

        def run_trial(trial_idx: int) -> Dict[str, Any]:
            nonlocal managed_litellm_env
            seed = trial_seed(args.base_seed, args.seed_step, trial_idx)
            display_trial_idx = start_trial_number - 1 + trial_idx
            run_name = build_trial_run_name(
                trial_idx=display_trial_idx,
                seed=seed,
                prefix=args.run_name_prefix,
            )
            run_root = output_root / run_name
            trial_log_path = trial_log_dir / f"{run_name}.log"
            trial_api_base = effective_api_base
            trial_litellm: subprocess.Popen[str] | None = None
            trial_litellm_log_path: Path | None = None
            trial_litellm_port: int | None = None

            print(f"\n{'=' * 72}")
            print(
                f"EMO-STA trial {trial_idx + 1}/{args.trials} "
                f"| run={display_trial_idx + 1} | seed={seed}"
            )
            print(f"Run root: {run_root}")
            print(f"Trial log: {trial_log_path}")
            print(f"{'=' * 72}")

            try:
                if should_manage_litellm and litellm_per_trial:
                    trial_litellm_log_path = trial_log_dir / f"{run_name}.litellm.log"
                    if trial_litellm_log_path.exists():
                        trial_litellm_log_path.unlink()
                    search_start = litellm_port
                    search_stop = litellm_port + litellm_port_search_limit
                    last_port_conflict: Exception | None = None

                    while search_start < search_stop:
                        reserved_port = reserve_available_tcp_port(
                            litellm_host,
                            start_port=search_start,
                            search_limit=search_stop - search_start,
                        )
                        trial_litellm_port = reserved_port
                        try:
                            trial_managed_litellm_env = managed_litellm_env
                            if trial_managed_litellm_env is None:
                                trial_managed_litellm_env = _prepare_managed_litellm_env(
                                    litellm_config=litellm_config
                                )
                                managed_litellm_env = trial_managed_litellm_env
                            trial_litellm = _start_managed_litellm(
                                repo_root=REPO_ROOT,
                                modules=modules,
                                setup_commands=setup_commands,
                                litellm_command=litellm_command,
                                litellm_preferred_env=litellm_preferred_env,
                                litellm_config=litellm_config,
                                litellm_host=litellm_host,
                                litellm_port=reserved_port,
                                litellm_log_path=trial_litellm_log_path,
                                wait_sec=args.litellm_wait_sec,
                                managed_env=trial_managed_litellm_env,
                            )
                        except RuntimeError as exc:
                            tail = read_log_tail(trial_litellm_log_path)
                            release_reserved_tcp_port(litellm_host, reserved_port)
                            trial_litellm_port = None
                            if _log_indicates_litellm_port_conflict(tail):
                                last_port_conflict = exc
                                search_start = reserved_port + 1
                                continue
                            raise

                        trial_api_base = f"http://{litellm_host}:{reserved_port}"
                        print(f"Trial-scoped LiteLLM ready at {trial_api_base}")
                        break

                    if trial_litellm is None:
                        raise RuntimeError(
                            "Could not start a per-trial LiteLLM server on any local port in "
                            f"range {litellm_port}-{search_stop - 1}."
                        ) from last_port_conflict

                with tempfile.TemporaryDirectory(prefix=f"emo_sta_seed_{seed}_") as temp_dir_str:
                    temp_dir = Path(temp_dir_str)
                    strip_trial_config_api_base = should_manage_litellm and litellm_per_trial
                    trial_manifest, _ = write_seeded_trial_manifest(
                        manifest_path=manifest_path,
                        seed=seed,
                        temp_dir=temp_dir,
                        output_root=output_root,
                        strip_api_base=strip_trial_config_api_base,
                    )
                    command = _build_runner_command(
                        args=args,
                        repo_root=REPO_ROOT,
                        runner_script=runner_script,
                        trial_manifest=trial_manifest,
                        run_name=run_name,
                        trial_api_base=trial_api_base,
                        primary_model=primary_model,
                    )
                    if trial_api_base and strip_trial_config_api_base:
                        print(
                            "Using matched LiteLLM endpoint for this trial: "
                            f"{trial_api_base} "
                            "(temp base config api_base stripped)"
                        )

                    if should_manage_litellm and trial_api_base:
                        _ensure_litellm_healthy(
                            trial_api_base,
                            timeout_sec=min(max(args.litellm_wait_sec / 2.0, 1.0), 5.0),
                        )

                    _run_trial_command(
                        command=command,
                        cwd=REPO_ROOT,
                        modules=modules,
                        setup_commands=setup_commands,
                        log_path=trial_log_path,
                    )

                metrics = load_trial_metrics(run_root)
                metrics.update(
                    {
                        "trial": display_trial_idx + 1,
                        "seed": seed,
                        "run_name": run_name,
                        "run_root": str(run_root),
                        "trial_log_path": str(trial_log_path),
                        "litellm_api_base": trial_api_base if should_manage_litellm else None,
                        "litellm_log_path": (
                            str(trial_litellm_log_path) if trial_litellm_log_path is not None else None
                        ),
                    }
                )
                print(
                    "Trial summary: "
                    f"shared={metrics['shared_best_score']}, "
                    f"adapted_mean={metrics['adapted_mean_score']}, "
                    f"baseline_mean={metrics['baseline_mean_score']}"
                )
                return metrics
            finally:
                if trial_litellm is not None:
                    terminate_process_tree(trial_litellm)
                if trial_litellm_port is not None:
                    release_reserved_tcp_port(litellm_host, trial_litellm_port)

        if args.parallel_trials > 1:
            print(f"Launching up to {min(args.parallel_trials, args.trials)} EMO-STA trials at once.")
            if args.launch_delay_sec > 0:
                print(f"Staggering trial launches by {args.launch_delay_sec:.1f}s.")

        trial_rows = run_trial_workers(
            trials=args.trials,
            parallel_trials=args.parallel_trials,
            launch_delay_sec=args.launch_delay_sec,
            worker=run_trial,
        )
        summary = {
            "config": build_summary_config(
                litellm_log_file=str(litellm_log_path)
                if managed_litellm is not None
                else None
            ),
            "summary": summarize_trial_rows(trial_rows),
            "trials": trial_rows,
        }
        summary_path = write_json(summary_json_path, summary)
        print(f"\nTrial summary written to {summary_path}")

        if args.refresh_report:
            latest_per_setting = (
                args.report_latest_per_setting
                if args.report_latest_per_setting is not None
                else (start_trial_number - 1 + args.trials)
            )
            _refresh_report(
                repo_root=REPO_ROOT,
                manifest_path=manifest_path,
                results_dir=output_root,
                latest_per_setting=latest_per_setting,
                markdown_out=report_markdown_path,
                json_out=report_json_path,
            )
            print(f"Refreshed report: {report_markdown_path}")
            print(f"Refreshed report JSON: {report_json_path}")
        return 0
    finally:
        if managed_litellm is not None and not args.leave_litellm_running:
            terminate_process_tree(managed_litellm)


if __name__ == "__main__":
    raise SystemExit(main())
