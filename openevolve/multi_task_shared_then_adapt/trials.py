"""Helpers for repeated EMO-STA launches with optional environment setup."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar
from urllib.parse import urlparse

import yaml

from openevolve.multi_task_shared_then_adapt.workflow import build_emo_sta_setting_slug


DEFAULT_LITELLM_COMMAND = "litellm"
DEFAULT_LITELLM_CONFIG = "configs/litellm_proxy.yaml"
DEFAULT_LITELLM_HOST = "127.0.0.1"
DEFAULT_LITELLM_PORT = 4000
DEFAULT_LITELLM_MODE = "auto"
DEFAULT_LITELLM_PER_TRIAL = True
DEFAULT_LITELLM_PORT_SEARCH_LIMIT = 200
DEFAULT_REPORT_MARKDOWN = "multi_task_shared_then_adapt/docs/emo_sta_results_summary.md"
DEFAULT_REPORT_JSON = "multi_task_shared_then_adapt/docs/emo_sta_results_summary.json"
T = TypeVar("T")
_TOLERANCE = 1.0e-12
_RESERVED_TCP_PORTS: set[tuple[str, int]] = set()
_RESERVED_TCP_PORTS_LOCK = threading.Lock()
_RESERVED_TCP_PORT_LOCK_FILES: dict[tuple[str, int], Path] = {}


@dataclass(frozen=True)
class LauncherDefaults:
    modules: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    litellm_mode: str = DEFAULT_LITELLM_MODE
    litellm_per_trial: bool = DEFAULT_LITELLM_PER_TRIAL
    litellm_command: str = DEFAULT_LITELLM_COMMAND
    litellm_preferred_env: Optional[str] = None
    litellm_config: Optional[Path] = None
    litellm_host: str = DEFAULT_LITELLM_HOST
    litellm_port: int = DEFAULT_LITELLM_PORT
    litellm_port_search_limit: int = DEFAULT_LITELLM_PORT_SEARCH_LIMIT


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def load_launcher_defaults(manifest_path: Path) -> LauncherDefaults:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    launcher = raw.get("launcher") or {}
    litellm = launcher.get("litellm") or {}
    manifest_dir = manifest_path.parent.resolve()

    litellm_config = litellm.get("config")
    resolved_litellm_config = None
    if isinstance(litellm_config, str) and litellm_config:
        resolved_litellm_config = (
            Path(litellm_config).resolve()
            if Path(litellm_config).is_absolute()
            else (manifest_dir / litellm_config).resolve()
        )

    return LauncherDefaults(
        modules=tuple(str(module) for module in (launcher.get("modules") or [])),
        setup_commands=tuple(str(cmd) for cmd in (launcher.get("setup_commands") or [])),
        litellm_mode=str(litellm.get("mode") or DEFAULT_LITELLM_MODE),
        litellm_per_trial=bool(litellm.get("per_trial", DEFAULT_LITELLM_PER_TRIAL)),
        litellm_command=str(litellm.get("command") or DEFAULT_LITELLM_COMMAND),
        litellm_preferred_env=(
            str(litellm.get("preferred_env")) if litellm.get("preferred_env") else None
        ),
        litellm_config=resolved_litellm_config,
        litellm_host=str(litellm.get("host") or DEFAULT_LITELLM_HOST),
        litellm_port=int(litellm.get("port") or DEFAULT_LITELLM_PORT),
        litellm_port_search_limit=int(
            litellm.get("port_search_limit") or DEFAULT_LITELLM_PORT_SEARCH_LIMIT
        ),
    )


def read_base_config_api_base(base_config_path: Path) -> Optional[str]:
    raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        return None
    api_base = llm.get("api_base")
    return str(api_base) if api_base else None


def read_base_config_primary_model(base_config_path: Path) -> Optional[str]:
    raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        return None
    primary_model = llm.get("primary_model")
    return str(primary_model) if primary_model else None


def read_base_config_edit_mode(base_config_path: Path) -> str:
    raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    diff_based_evolution = raw.get("diff_based_evolution", False)
    return "diff" if bool(diff_based_evolution) else "full"


def trial_seed(base_seed: int, seed_step: int, trial_idx: int) -> int:
    return base_seed + trial_idx * seed_step


def existing_trial_numbers(output_root: Path, prefix: str) -> List[int]:
    normalized_prefix = str(prefix).strip() or "run"
    pattern = re.compile(rf"^{re.escape(normalized_prefix)}_(\d+)_seed_")
    numbers: List[int] = []
    if not output_root.is_dir():
        return numbers
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        try:
            numbers.append(int(match.group(1)))
        except ValueError:
            continue
    return sorted(set(numbers))


def next_trial_number(output_root: Path, prefix: str) -> int:
    numbers = existing_trial_numbers(output_root, prefix)
    if not numbers:
        return 1
    return max(numbers) + 1


def launch_detached(
    *,
    script_path: Path,
    forwarded_args: Sequence[str],
    cwd: Path,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script_path), *forwarded_args]
    env = dict(os.environ)
    env["OPENEVOLVE_DETACHED"] = "1"

    with open(os.devnull, "r", encoding="utf-8") as devnull, log_path.open(
        "a", encoding="utf-8"
    ) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=devnull,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return process.pid


def run_trial_workers(
    *,
    trials: int,
    parallel_trials: int,
    launch_delay_sec: float,
    worker: Callable[[int], T],
) -> List[T]:
    effective_parallel = max(1, min(parallel_trials, trials))
    if effective_parallel == 1:
        return [worker(trial_idx) for trial_idx in range(trials)]

    results: Dict[int, T] = {}
    next_trial_idx = 0
    last_launch_at: float | None = None

    def maybe_wait_before_launch() -> None:
        nonlocal last_launch_at
        if launch_delay_sec <= 0 or last_launch_at is None:
            return
        remaining = launch_delay_sec - (time.monotonic() - last_launch_at)
        if remaining > 0:
            time.sleep(remaining)

    def submit_trial(executor: ThreadPoolExecutor, trial_idx: int):
        nonlocal last_launch_at
        maybe_wait_before_launch()
        future = executor.submit(worker, trial_idx)
        last_launch_at = time.monotonic()
        return future

    with ThreadPoolExecutor(max_workers=effective_parallel) as executor:
        future_to_trial_idx = {}

        while next_trial_idx < effective_parallel:
            future = submit_trial(executor, next_trial_idx)
            future_to_trial_idx[future] = next_trial_idx
            next_trial_idx += 1

        while future_to_trial_idx:
            done, _ = wait(future_to_trial_idx, return_when=FIRST_COMPLETED)
            for future in done:
                trial_idx = future_to_trial_idx.pop(future)
                results[trial_idx] = future.result()
                if next_trial_idx < trials:
                    next_future = submit_trial(executor, next_trial_idx)
                    future_to_trial_idx[next_future] = next_trial_idx
                    next_trial_idx += 1

    return [results[trial_idx] for trial_idx in range(trials)]


def build_shell_command(
    command: Sequence[str],
    *,
    modules: Sequence[str] = (),
    setup_commands: Sequence[str] = (),
) -> List[str]:
    script_parts = ["set -euo pipefail"]
    # `bash -lc` can source user shell startup files that overwrite PATH and
    # conda-related variables, which drops tools like `cargo` even when the
    # launcher was started from an activated env. Re-export the parent env's
    # key variables inside the login shell before loading modules or running
    # setup commands.
    for key in (
        "PATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "CONDA_SHLVL",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
    ):
        value = os.environ.get(key)
        if value:
            script_parts.append(f"export {key}={shlex.quote(value)}")
    for module in modules:
        script_parts.append(f"module load {shlex.quote(str(module))}")
    script_parts.extend(str(cmd) for cmd in setup_commands)
    script_parts.append("exec " + " ".join(shlex.quote(part) for part in command))
    return ["bash", "-lc", "; ".join(script_parts)]


def build_litellm_command(
    *,
    litellm_command: str,
    litellm_config: Path,
    host: str,
    port: int,
) -> List[str]:
    return [
        litellm_command,
        "--config",
        str(litellm_config),
        "--host",
        host,
        "--port",
        str(int(port)),
    ]


def _conda_root_candidates() -> List[Path]:
    roots: List[Path] = []
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        roots.append(Path(conda_prefix).expanduser().resolve())

    for candidate in (
        Path.home() / "conda" / "miniconda",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path("/opt/conda"),
    ):
        expanded = candidate.expanduser()
        if expanded.exists():
            roots.append(expanded.resolve())

    unique: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def _litellm_candidate_paths(command_name: str) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        add(Path(conda_prefix) / "bin" / command_name)

    add(Path.home() / ".local" / "bin" / command_name)
    add(Path.home() / "bin" / command_name)

    for root in _conda_root_candidates():
        add(root / "bin" / command_name)
        envs_dir = root / "envs"
        if envs_dir.is_dir():
            for env_bin in sorted(envs_dir.glob(f"*/bin/{command_name}")):
                add(env_bin)

    return [path for path in candidates if path.is_file() and os.access(path, os.X_OK)]


def resolve_litellm_command(
    litellm_command: str,
    *,
    preferred_env: Optional[str] = None,
) -> str:
    command = str(litellm_command).strip()
    if not command:
        raise ValueError("LiteLLM command must not be empty.")

    if any(sep and sep in command for sep in (os.sep, os.altsep)):
        path = Path(command).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        raise FileNotFoundError(f"LiteLLM command path is not executable: {path}")

    resolved = shutil.which(command)
    if resolved:
        return resolved

    candidates = _litellm_candidate_paths(command)
    if preferred_env:
        preferred_fragment = f"{os.sep}envs{os.sep}{preferred_env}{os.sep}bin{os.sep}{command}"
        for candidate in candidates:
            if preferred_fragment in str(candidate):
                return str(candidate)

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        f"Could not locate executable '{command}'. "
        "Set launcher.litellm.command to an absolute path or install litellm on PATH."
    )


def sanitize_path_component(value: Optional[str], *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return text or default


def build_setting_output_dir_name(
    *,
    shared_iterations: Optional[int],
    adaptation_iterations: Optional[int],
    baseline_iterations: Optional[int],
    primary_model: Optional[str],
    edit_mode: Optional[str],
) -> str:
    setting_slug = build_emo_sta_setting_slug(
        shared_iterations=shared_iterations,
        adaptation_iterations=adaptation_iterations,
        baseline_iterations=baseline_iterations,
    )
    parts = [
        sanitize_path_component(setting_slug, default="setting"),
        sanitize_path_component(primary_model, default="unknown-model"),
        sanitize_path_component(edit_mode, default="unknown-edit"),
    ]
    return "-".join(parts)


def build_trial_run_name(
    *,
    trial_idx: int,
    seed: int,
    prefix: str,
) -> str:
    normalized_prefix = str(prefix).strip()
    if not normalized_prefix:
        normalized_prefix = "run"
    return f"{normalized_prefix}_{trial_idx + 1:02d}_seed_{seed}"


def write_seeded_trial_manifest(
    *,
    manifest_path: Path,
    seed: int,
    temp_dir: Path,
    output_root: Optional[Path] = None,
    strip_api_base: bool = False,
    override_api_base: Optional[str] = None,
) -> Tuple[Path, Path]:
    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest_dir = manifest_path.parent.resolve()

    def resolve_manifest_path(key: str) -> Path:
        value = raw_manifest.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Manifest is missing required path field '{key}'")
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (manifest_dir / candidate).resolve()

    initial_program = resolve_manifest_path("initial_program")
    evaluation_file = resolve_manifest_path("evaluation_file")
    base_config_path = resolve_manifest_path("base_config")
    manifest_output_root = output_root or resolve_manifest_path("output_root")

    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    base_config["random_seed"] = int(seed)
    database_config = base_config.setdefault("database", {})
    if isinstance(database_config, dict):
        database_config["random_seed"] = int(seed)
    llm_config = base_config.get("llm")
    if not isinstance(llm_config, dict):
        llm_config = {}
        base_config["llm"] = llm_config
    if strip_api_base:
        llm_config.pop("api_base", None)
    if override_api_base is not None:
        llm_config["api_base"] = str(override_api_base)

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_base_config = temp_dir / f"base_config_seed_{seed}.yaml"
    temp_base_config.write_text(
        yaml.safe_dump(base_config, sort_keys=False),
        encoding="utf-8",
    )

    trial_manifest = dict(raw_manifest)
    trial_manifest["manifest_label"] = str(
        raw_manifest.get("manifest_label") or manifest_path.stem
    )
    trial_manifest["initial_program"] = str(initial_program)
    trial_manifest["evaluation_file"] = str(evaluation_file)
    trial_manifest["base_config"] = str(temp_base_config.resolve())
    trial_manifest["output_root"] = str(manifest_output_root.resolve())

    temp_manifest = temp_dir / f"manifest_seed_{seed}.yaml"
    temp_manifest.write_text(
        yaml.safe_dump(trial_manifest, sort_keys=False),
        encoding="utf-8",
    )
    return temp_manifest, temp_base_config


def parse_api_base_host_port(api_base: str | None) -> Optional[Tuple[str, int]]:
    if not api_base:
        return None
    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.hostname:
        return None
    host = parsed.hostname
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = parsed.port
    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
        else:
            return None
    return host, int(port)


def is_tcp_ready(host: str, port: int, *, timeout_sec: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _port_lock_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "openevolve_litellm_port_locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _port_lock_path(host: str, port: int) -> Path:
    normalized_host = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(host).strip() or DEFAULT_LITELLM_HOST)
    return _port_lock_dir() / f"{normalized_host}_{int(port)}.lock"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_acquire_port_lock(host: str, port: int) -> Optional[Path]:
    lock_path = _port_lock_path(host, port)
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                payload = lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                payload = ""
            try:
                existing_pid = int(payload)
            except (TypeError, ValueError):
                existing_pid = -1
            if existing_pid > 0 and _pid_is_running(existing_pid):
                return None
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return None
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
        except OSError:
            try:
                lock_path.unlink()
            except OSError:
                pass
            return None
        return lock_path
    return None


def reserve_available_tcp_port(
    host: str,
    *,
    start_port: int,
    search_limit: int,
) -> int:
    if search_limit < 1:
        raise ValueError("search_limit must be at least 1")

    normalized_host = str(host).strip() or DEFAULT_LITELLM_HOST
    with _RESERVED_TCP_PORTS_LOCK:
        for port in range(int(start_port), int(start_port) + int(search_limit)):
            reservation = (normalized_host, port)
            if reservation in _RESERVED_TCP_PORTS:
                continue
            if is_tcp_ready(normalized_host, port, timeout_sec=0.25):
                continue
            lock_path = _try_acquire_port_lock(normalized_host, port)
            if lock_path is None:
                continue
            if is_tcp_ready(normalized_host, port, timeout_sec=0.25):
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            _RESERVED_TCP_PORTS.add(reservation)
            _RESERVED_TCP_PORT_LOCK_FILES[reservation] = lock_path
            return port

    raise RuntimeError(
        f"Could not reserve a free TCP port on {normalized_host} in range "
        f"{int(start_port)}-{int(start_port) + int(search_limit) - 1}."
    )


def release_reserved_tcp_port(host: str, port: int) -> None:
    normalized_host = str(host).strip() or DEFAULT_LITELLM_HOST
    with _RESERVED_TCP_PORTS_LOCK:
        reservation = (normalized_host, int(port))
        _RESERVED_TCP_PORTS.discard(reservation)
        lock_path = _RESERVED_TCP_PORT_LOCK_FILES.pop(reservation, None)
        if lock_path is not None:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def wait_for_tcp_ready(
    host: str,
    port: int,
    *,
    timeout_sec: float,
    process: Optional[subprocess.Popen[str]] = None,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if is_tcp_ready(host, port, timeout_sec=1.0):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Process exited before service became ready on {host}:{port}."
            )
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for TCP service on {host}:{port}")


def terminate_process_tree(process: subprocess.Popen[str], *, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def read_log_tail(path: Path, *, max_lines: int = 40) -> str:
    if max_lines <= 0 or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def summarize_optional(values: Iterable[float | None]) -> Dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "values": [],
        }
    return {
        "count": len(valid),
        "mean": statistics.fmean(valid),
        "stdev": statistics.stdev(valid) if len(valid) > 1 else 0.0,
        "min": min(valid),
        "max": max(valid),
        "values": valid,
    }


def mean_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return statistics.fmean(valid)


def compare_scores(lhs: float | None, rhs: float | None) -> Optional[str]:
    if lhs is None or rhs is None:
        return None
    delta = lhs - rhs
    if delta > _TOLERANCE:
        return "win"
    if delta < -_TOLERANCE:
        return "loss"
    return "tie"


def comparison_counts(outcomes: Iterable[Optional[str]]) -> Dict[str, int]:
    counts = {"wins": 0, "ties": 0, "losses": 0, "comparable": 0}
    for outcome in outcomes:
        if outcome is None:
            continue
        counts["comparable"] += 1
        if outcome == "win":
            counts["wins"] += 1
        elif outcome == "loss":
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _legacy_branch_key(branch: str) -> str:
    return f"{branch}_{'seed'}_adaptation"


def _branch_score_from_task_payload(
    task_payload: Dict[str, Any],
    *,
    branch: str,
) -> float | None:
    canonical_key = f"{branch}_adaptation"
    legacy_key = _legacy_branch_key(branch)
    score = _coerce_float(task_payload.get(f"{canonical_key}_best_score"))
    if score is None:
        score = _coerce_float(task_payload.get(f"{legacy_key}_best_score"))
    if score is None:
        nested = task_payload.get(canonical_key)
        if not isinstance(nested, dict):
            nested = task_payload.get(legacy_key)
        if isinstance(nested, dict):
            score = _coerce_float(nested.get("best_score"))
    return score


def load_trial_metrics(run_root: Path) -> Dict[str, Any]:
    summary_path = run_root / "comparison_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing comparison summary: {summary_path}")

    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    shared_info = ((raw.get("shared_run") or {}).get("best_program_info") or {})
    shared_metrics = shared_info.get("metrics") if isinstance(shared_info.get("metrics"), dict) else {}
    shared_best_score = _coerce_float(shared_metrics.get("combined_score"))
    if shared_best_score is None:
        shared_best_score = _coerce_float(shared_metrics.get("score"))

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, dict):
        raise ValueError(f"Invalid tasks payload in {summary_path}")

    tasks: Dict[str, Dict[str, Any]] = {}
    for task_id, task_payload in tasks_raw.items():
        if not isinstance(task_payload, dict):
            continue
        spawn_score = _coerce_float(task_payload.get("spawn_best_score"))
        adapted_score = _coerce_float(task_payload.get("adapted_best_score"))
        best_shared_score = _branch_score_from_task_payload(
            task_payload,
            branch="best_shared",
        )
        best_local_score = _branch_score_from_task_payload(
            task_payload,
            branch="best_local",
        )
        baseline_score = _coerce_float(task_payload.get("baseline_best_score"))
        tasks[str(task_id)] = {
            "spawn_best_score": spawn_score,
            "adapted_best_score": adapted_score,
            "best_shared_adaptation_best_score": best_shared_score,
            "best_local_adaptation_best_score": best_local_score,
            "baseline_best_score": baseline_score,
            "adapted_minus_spawn": delta(adapted_score, spawn_score),
            "adapted_minus_best_shared": delta(adapted_score, best_shared_score),
            "adapted_minus_best_local": delta(adapted_score, best_local_score),
            "best_local_minus_best_shared": delta(
                best_local_score,
                best_shared_score,
            ),
            "adapted_minus_baseline": delta(adapted_score, baseline_score),
            "adapted_vs_spawn": compare_scores(adapted_score, spawn_score),
            "adapted_vs_best_shared": compare_scores(
                adapted_score,
                best_shared_score,
            ),
            "adapted_vs_best_local": compare_scores(adapted_score, best_local_score),
            "best_local_vs_best_shared": compare_scores(
                best_local_score,
                best_shared_score,
            ),
            "adapted_vs_baseline": compare_scores(adapted_score, baseline_score),
        }

    spawn_scores = [task["spawn_best_score"] for task in tasks.values()]
    adapted_scores = [task["adapted_best_score"] for task in tasks.values()]
    best_shared_scores = [
        task["best_shared_adaptation_best_score"] for task in tasks.values()
    ]
    best_local_scores = [
        task["best_local_adaptation_best_score"] for task in tasks.values()
    ]
    baseline_scores = [task["baseline_best_score"] for task in tasks.values()]
    return {
        "shared_best_score": shared_best_score,
        "spawn_mean_score": mean_or_none(spawn_scores),
        "adapted_mean_score": mean_or_none(adapted_scores),
        "best_shared_mean_score": mean_or_none(best_shared_scores),
        "best_local_mean_score": mean_or_none(best_local_scores),
        "baseline_mean_score": mean_or_none(baseline_scores),
        "adapted_minus_spawn_mean": mean_or_none(task["adapted_minus_spawn"] for task in tasks.values()),
        "adapted_minus_best_shared_mean": mean_or_none(
            task["adapted_minus_best_shared"] for task in tasks.values()
        ),
        "adapted_minus_best_local_mean": mean_or_none(
            task["adapted_minus_best_local"] for task in tasks.values()
        ),
        "best_local_minus_best_shared_mean": mean_or_none(
            task["best_local_minus_best_shared"] for task in tasks.values()
        ),
        "adapted_minus_baseline_mean": mean_or_none(
            task["adapted_minus_baseline"] for task in tasks.values()
        ),
        "adapted_vs_spawn_counts": comparison_counts(
            task["adapted_vs_spawn"] for task in tasks.values()
        ),
        "adapted_vs_best_shared_counts": comparison_counts(
            task["adapted_vs_best_shared"] for task in tasks.values()
        ),
        "adapted_vs_best_local_counts": comparison_counts(
            task["adapted_vs_best_local"] for task in tasks.values()
        ),
        "best_local_vs_best_shared_counts": comparison_counts(
            task["best_local_vs_best_shared"] for task in tasks.values()
        ),
        "adapted_vs_baseline_counts": comparison_counts(
            task["adapted_vs_baseline"] for task in tasks.values()
        ),
        "tasks": tasks,
    }


def summarize_trial_rows(trial_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    task_ids = sorted(
        {
            task_id
            for row in trial_rows
            for task_id in (row.get("tasks") or {}).keys()
        }
    )
    summary = {
        "shared_best_score": summarize_optional(
            row.get("shared_best_score") for row in trial_rows
        ),
        "spawn_mean_score": summarize_optional(
            row.get("spawn_mean_score") for row in trial_rows
        ),
        "adapted_mean_score": summarize_optional(
            row.get("adapted_mean_score") for row in trial_rows
        ),
        "best_shared_mean_score": summarize_optional(
            row.get("best_shared_mean_score") for row in trial_rows
        ),
        "best_local_mean_score": summarize_optional(
            row.get("best_local_mean_score") for row in trial_rows
        ),
        "baseline_mean_score": summarize_optional(
            row.get("baseline_mean_score") for row in trial_rows
        ),
        "adapted_minus_spawn_mean": summarize_optional(
            row.get("adapted_minus_spawn_mean") for row in trial_rows
        ),
        "adapted_minus_best_shared_mean": summarize_optional(
            row.get("adapted_minus_best_shared_mean") for row in trial_rows
        ),
        "adapted_minus_best_local_mean": summarize_optional(
            row.get("adapted_minus_best_local_mean") for row in trial_rows
        ),
        "best_local_minus_best_shared_mean": summarize_optional(
            row.get("best_local_minus_best_shared_mean")
            for row in trial_rows
        ),
        "adapted_minus_baseline_mean": summarize_optional(
            row.get("adapted_minus_baseline_mean") for row in trial_rows
        ),
        "adapted_vs_spawn_counts": {
            key: sum(row.get("adapted_vs_spawn_counts", {}).get(key, 0) for row in trial_rows)
            for key in ("wins", "ties", "losses", "comparable")
        },
        "adapted_vs_best_shared_counts": {
            key: sum(
                row.get("adapted_vs_best_shared_counts", {}).get(key, 0)
                for row in trial_rows
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "adapted_vs_best_local_counts": {
            key: sum(
                row.get("adapted_vs_best_local_counts", {}).get(key, 0)
                for row in trial_rows
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "best_local_vs_best_shared_counts": {
            key: sum(
                row.get("best_local_vs_best_shared_counts", {}).get(key, 0)
                for row in trial_rows
            )
            for key in ("wins", "ties", "losses", "comparable")
        },
        "adapted_vs_baseline_counts": {
            key: sum(row.get("adapted_vs_baseline_counts", {}).get(key, 0) for row in trial_rows)
            for key in ("wins", "ties", "losses", "comparable")
        },
        "tasks": {},
    }

    for task_id in task_ids:
        summary["tasks"][task_id] = {
            "spawn_best_score": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("spawn_best_score")
                for row in trial_rows
            ),
            "adapted_best_score": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("adapted_best_score")
                for row in trial_rows
            ),
            "best_shared_adaptation_best_score": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get(
                    "best_shared_adaptation_best_score"
                )
                for row in trial_rows
            ),
            "best_local_adaptation_best_score": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get(
                    "best_local_adaptation_best_score"
                )
                for row in trial_rows
            ),
            "baseline_best_score": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("baseline_best_score")
                for row in trial_rows
            ),
            "adapted_minus_spawn": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("adapted_minus_spawn")
                for row in trial_rows
            ),
            "adapted_minus_best_shared": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("adapted_minus_best_shared")
                for row in trial_rows
            ),
            "adapted_minus_best_local": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("adapted_minus_best_local")
                for row in trial_rows
            ),
            "best_local_minus_best_shared": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get(
                    "best_local_minus_best_shared"
                )
                for row in trial_rows
            ),
            "adapted_minus_baseline": summarize_optional(
                row.get("tasks", {}).get(task_id, {}).get("adapted_minus_baseline")
                for row in trial_rows
            ),
            "adapted_vs_spawn_counts": comparison_counts(
                row.get("tasks", {}).get(task_id, {}).get("adapted_vs_spawn")
                for row in trial_rows
            ),
            "adapted_vs_best_shared_counts": comparison_counts(
                row.get("tasks", {}).get(task_id, {}).get("adapted_vs_best_shared")
                for row in trial_rows
            ),
            "adapted_vs_best_local_counts": comparison_counts(
                row.get("tasks", {}).get(task_id, {}).get("adapted_vs_best_local")
                for row in trial_rows
            ),
            "best_local_vs_best_shared_counts": comparison_counts(
                row.get("tasks", {}).get(task_id, {}).get(
                    "best_local_vs_best_shared"
                )
                for row in trial_rows
            ),
            "adapted_vs_baseline_counts": comparison_counts(
                row.get("tasks", {}).get(task_id, {}).get("adapted_vs_baseline")
                for row in trial_rows
            ),
        }
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
