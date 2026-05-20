"""Shared subprocess helpers for multi-task STS workflow scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from openevolve.multi_task_shared_then_adapt.workflow import repo_root


def default_run_name(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def ensure_pythonpath(env: Dict[str, str], repo_root_path: Path) -> Dict[str, str]:
    updated = dict(env)
    existing = updated.get("PYTHONPATH")
    if existing:
        updated["PYTHONPATH"] = f"{repo_root_path}{os.pathsep}{existing}"
    else:
        updated["PYTHONPATH"] = str(repo_root_path)
    return updated


def build_openevolve_command(
    *,
    initial_program: Path,
    evaluation_file: Path,
    config_path: Path,
    output_dir: Path,
    iterations: int,
    checkpoint_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    primary_model: Optional[str] = None,
    secondary_model: Optional[str] = None,
) -> list[str]:
    command = [
        sys.executable,
        "openevolve-run.py",
        str(initial_program),
        str(evaluation_file),
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
        "--iterations",
        str(int(iterations)),
    ]
    if checkpoint_path is not None:
        command.extend(["--checkpoint", str(checkpoint_path)])
    if api_base:
        command.extend(["--api-base", api_base])
    if primary_model:
        command.extend(["--primary-model", primary_model])
    if secondary_model:
        command.extend(["--secondary-model", secondary_model])
    return command


def run_command(command: list[str], *, env: Dict[str, str]) -> None:
    cwd = repo_root()
    effective_env = ensure_pythonpath(env, cwd)
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=effective_env, check=True)


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
