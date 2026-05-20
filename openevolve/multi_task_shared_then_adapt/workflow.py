"""Helpers for the multi-task shared-then-adapt workflow scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from openevolve.multi_task_shared_then_adapt import sldbench_3d as sldbench_3d_family
from openevolve.multi_task_shared_then_adapt.registry import get_family_definition


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


@dataclass(frozen=True)
class SharedThenAdaptManifest:
    """Top-level manifest for one multi-task shared-then-adapt family."""

    family: str
    initial_program: Path
    evaluation_file: Path
    base_config: Path
    output_root: Path
    default_shared_iterations: int
    default_adaptation_iterations: int
    default_baseline_iterations: int
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_mode: Optional[str]
    wandb_single_run: bool
    wandb_log_best_program_artifact: bool
    wandb_log_checkpoint_artifact: bool
    wandb_log_code: bool
    manifest_label: str
    manifest_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(path: str | Path) -> SharedThenAdaptManifest:
    manifest_path = Path(path).resolve()
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    base_dir = manifest_path.parent

    family = raw.get("family")
    if not isinstance(family, str) or not family.strip():
        raise ValueError("EMO-STA manifest is missing a non-empty family field")
    get_family_definition(family)
    wandb = raw.get("wandb") or {}

    return SharedThenAdaptManifest(
        family=family,
        initial_program=_resolve_path(raw["initial_program"], base_dir),
        evaluation_file=_resolve_path(raw["evaluation_file"], base_dir),
        base_config=_resolve_path(raw["base_config"], base_dir),
        output_root=_resolve_path(raw["output_root"], base_dir),
        default_shared_iterations=int(raw.get("default_shared_iterations", 20)),
        default_adaptation_iterations=int(raw.get("default_adaptation_iterations", 20)),
        default_baseline_iterations=int(raw.get("default_baseline_iterations", 20)),
        wandb_enabled=bool(wandb.get("enabled", True)),
        wandb_project=str(wandb.get("project", "openevolve-emo-sta")),
        wandb_entity=wandb.get("entity"),
        wandb_mode=wandb.get("mode"),
        wandb_single_run=bool(wandb.get("single_run", True)),
        wandb_log_best_program_artifact=bool(
            wandb.get("log_best_program_artifact", True)
        ),
        wandb_log_checkpoint_artifact=bool(
            wandb.get("log_checkpoint_artifact", False)
        ),
        wandb_log_code=bool(wandb.get("log_code", False)),
        manifest_label=str(raw.get("manifest_label") or manifest_path.stem),
        manifest_path=manifest_path,
    )


def family_task_specs(manifest: SharedThenAdaptManifest):
    return list(get_family_definition(manifest.family).task_specs)


def run_emo_sta_family_preflight(
    manifest: SharedThenAdaptManifest,
    *,
    task_specs: Optional[List[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run family-specific preflight validation before launching EMO-STA phases."""
    return None


def fair_emo_sta_baseline_iterations(
    *,
    task_count: int,
    shared_iterations: int,
    adaptation_iterations: int,
) -> int | None:
    """Return the iteration-fair baseline budget when one exists.

    EMO-STA spends `shared_iterations` once across the whole family and then
    `adaptation_iterations` once per task. A direct single-task baseline spends
    `baseline_iterations` once per task, so iteration fairness requires:

        shared_iterations + task_count * adaptation_iterations
        == task_count * baseline_iterations
    """
    if task_count < 1:
        raise ValueError("task_count must be at least 1")

    total_iterations = int(shared_iterations) + int(task_count) * int(adaptation_iterations)
    if total_iterations % int(task_count) != 0:
        return None
    return total_iterations // int(task_count)


def validate_emo_sta_iteration_budget(
    *,
    task_count: int,
    shared_iterations: int,
    adaptation_iterations: int,
    baseline_iterations: int,
    skip_adaptation: bool = False,
    skip_baselines: bool = False,
    allow_unsafe_iterations: bool = False,
) -> None:
    """Reject iteration-unsafe EMO-STA settings unless explicitly allowed."""
    if allow_unsafe_iterations or skip_adaptation or skip_baselines:
        return

    expected_baseline_iterations = fair_emo_sta_baseline_iterations(
        task_count=task_count,
        shared_iterations=shared_iterations,
        adaptation_iterations=adaptation_iterations,
    )
    if expected_baseline_iterations is None:
        raise ValueError(
            "Unsafe EMO-STA iteration setting: "
            f"shared_iterations + task_count * adaptation_iterations must be divisible "
            f"by task_count. Got shared_iterations={shared_iterations}, "
            f"adaptation_iterations={adaptation_iterations}, task_count={task_count}. "
            "Use --allow-unsafe-iterations to override."
        )
    if int(baseline_iterations) != int(expected_baseline_iterations):
        raise ValueError(
            "Unsafe EMO-STA iteration setting: "
            f"for task_count={task_count}, fairness requires "
            f"shared_iterations + task_count * adaptation_iterations "
            f"= task_count * baseline_iterations. "
            f"Got shared_iterations={shared_iterations}, "
            f"adaptation_iterations={adaptation_iterations}, "
            f"baseline_iterations={baseline_iterations}; expected "
            f"baseline_iterations={expected_baseline_iterations}. "
            "Use --allow-unsafe-iterations to override."
        )


def write_phase_config(
    *,
    base_config_path: Path,
    output_config_path: Path,
    iterations: int,
    wandb_config: Optional[Dict[str, Any]] = None,
    api_base: Optional[str] = None,
    primary_model: Optional[str] = None,
    secondary_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Path:
    """Write a per-phase config snapshot with a final-checkpoint-friendly interval."""
    raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    raw["max_iterations"] = int(iterations)
    raw["checkpoint_interval"] = max(1, int(iterations or 1))
    llm_config = raw.get("llm")
    if not isinstance(llm_config, dict):
        llm_config = {}
        raw["llm"] = llm_config
    if api_base is not None:
        llm_config["api_base"] = api_base
    if primary_model is not None:
        llm_config["primary_model"] = primary_model
    if secondary_model is not None:
        llm_config["secondary_model"] = secondary_model
    if system_prompt is not None:
        prompt_config = raw.get("prompt")
        if not isinstance(prompt_config, dict):
            prompt_config = {}
            raw["prompt"] = prompt_config
        prompt_config["system_message"] = str(system_prompt)
    if wandb_config is not None:
        existing_wandb = raw.get("wandb") or {}
        merged_wandb = dict(existing_wandb)
        merged_wandb.update(wandb_config)
        raw["wandb"] = merged_wandb
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output_config_path


def build_emo_sta_wandb_run_id(run_root: Path) -> str:
    run_root_digest = hashlib.sha1(
        str(run_root.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    return f"emosta-{run_root_digest}"


def _emo_sta_wandb_run_id_state_path(run_root: Path) -> Path:
    return run_root / "wandb_run_id.json"


def resolve_emo_sta_wandb_run_id(run_root: Path, *, force_new: bool = False) -> str:
    """Resolve one stable W&B run id for a workflow invocation rooted at run_root.

    The resolved id is persisted under run_root so the shared, adaptation, and
    baseline phases all reconnect to the same W&B run. When force_new is true,
    rotate the persisted id so explicit reruns avoid collisions with deleted or
    otherwise unusable historical W&B runs for the same path.
    """
    state_path = _emo_sta_wandb_run_id_state_path(run_root)
    if not force_new and state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return run_id

    run_root.mkdir(parents=True, exist_ok=True)
    prefix = build_emo_sta_wandb_run_id(run_root)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    run_id = f"{prefix}-{timestamp}-{uuid4().hex[:8]}"
    state_payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_root": str(run_root.resolve()),
    }
    state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
    return run_id


def build_emo_sta_wandb_namespace(*, phase: str, task_id: Optional[str] = None) -> str:
    return f"{phase}/{task_id}" if task_id else phase


def build_emo_sta_setting_slug(
    *,
    shared_iterations: Optional[int] = None,
    adaptation_iterations: Optional[int] = None,
    baseline_iterations: Optional[int] = None,
) -> Optional[str]:
    parts: list[str] = []
    if shared_iterations is not None:
        parts.append(f"s{int(shared_iterations)}")
    if adaptation_iterations is not None:
        parts.append(f"a{int(adaptation_iterations)}")
    if baseline_iterations is not None:
        parts.append(f"b{int(baseline_iterations)}")
    return "-".join(parts) if parts else None


def default_emo_sta_run_prefix(
    *,
    base_prefix: str,
    manifest: SharedThenAdaptManifest,
) -> str:
    return base_prefix


def resolve_phase_system_prompt(
    manifest: SharedThenAdaptManifest,
    *,
    phase: str,
    task_id: Optional[str] = None,
) -> Optional[str]:
    if manifest.family != "sldbench_3d":
        return None

    return sldbench_3d_family.get_sldbench_3d_system_prompt(task_id=task_id)


def build_phase_wandb_config(
    manifest: SharedThenAdaptManifest,
    *,
    run_name: str,
    run_root: Optional[Path] = None,
    wandb_run_id: Optional[str] = None,
    phase: str,
    task_id: Optional[str] = None,
    run_label: Optional[str] = None,
    shared_iterations: Optional[int] = None,
    adaptation_iterations: Optional[int] = None,
    baseline_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a dedicated W&B config for one EMO-STA phase run."""
    base_tags = ["emo-sta", "multi-task", "archive-sharing", manifest.family]
    config_label = manifest.manifest_label
    setting_slug = build_emo_sta_setting_slug(
        shared_iterations=shared_iterations,
        adaptation_iterations=adaptation_iterations,
        baseline_iterations=baseline_iterations,
    )
    group_parts = ["emo_sta", config_label]
    if setting_slug:
        group_parts.append(setting_slug)

    if manifest.wandb_single_run:
        if run_root is None:
            raise ValueError("run_root is required when wandb.single_run=true")
        name_parts = ["emo_sta", config_label]
        if setting_slug:
            name_parts.append(setting_slug)
        if run_label:
            name_parts.append(run_label)
        name_parts.extend(["{model}", "{edit_mode}"])
        tags = base_tags + ["single-run", config_label]
        if setting_slug:
            tags.append(setting_slug)
        return {
            "enabled": manifest.wandb_enabled,
            "project": manifest.wandb_project,
            "entity": manifest.wandb_entity,
            "run_id": wandb_run_id or build_emo_sta_wandb_run_id(run_root),
            "resume": "allow",
            "allow_val_change": True,
            "name": "-".join(name_parts),
            "group": "/".join([*group_parts, "{model}"]),
            "job_type": "emo_sta_workflow",
            "tags": tags,
            "mode": manifest.wandb_mode,
            "namespace": build_emo_sta_wandb_namespace(phase=phase, task_id=task_id),
            "log_best_program_artifact": manifest.wandb_log_best_program_artifact,
            "log_checkpoint_artifact": manifest.wandb_log_checkpoint_artifact,
            "log_code": manifest.wandb_log_code,
            "notes": (
                "EMO-STA archive-sharing multi-task workflow. Metrics are namespaced by "
                "phase/task inside a single W&B run."
            ),
        }

    tags = base_tags + [phase, config_label]
    if setting_slug:
        tags.append(setting_slug)
    if task_id:
        tags.append(task_id)

    name_parts = ["emo_sta", config_label]
    if setting_slug:
        name_parts.append(setting_slug)
    if run_label:
        name_parts.append(run_label)
    name_parts.append(phase)
    if task_id:
        name_parts.append(task_id)
    name_parts.extend(["{model}", "{edit_mode}"])

    return {
        "enabled": manifest.wandb_enabled,
        "project": manifest.wandb_project,
        "entity": manifest.wandb_entity,
        "name": "-".join(name_parts),
        "group": "/".join([*group_parts, "{model}"]),
        "job_type": f"emo_sta_{phase}",
        "tags": tags,
        "mode": manifest.wandb_mode,
        "log_best_program_artifact": manifest.wandb_log_best_program_artifact,
        "log_checkpoint_artifact": manifest.wandb_log_checkpoint_artifact,
        "log_code": manifest.wandb_log_code,
        "notes": (
            "EMO-STA archive-sharing multi-task workflow. Distinct from "
            "multi_task_evolve prompt-only inspiration transfer."
        ),
    }


def find_latest_single_task_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None

    candidates: List[Path] = []
    for candidate in checkpoints_dir.iterdir():
        if not candidate.is_dir():
            continue
        if not candidate.name.startswith("checkpoint_"):
            continue
        suffix = candidate.name.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))[-1]


def checkpoint_iteration_from_path(checkpoint_path: str | Path | None) -> Optional[int]:
    if checkpoint_path is None:
        return None
    checkpoint = Path(checkpoint_path)
    if not checkpoint.name.startswith("checkpoint_"):
        return None
    suffix = checkpoint.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def phase_checkpoint_status(
    output_dir: Path,
    requested_iterations: int,
    *,
    require_best_info: bool,
) -> tuple[bool, Optional[Path]]:
    latest_checkpoint = find_latest_single_task_checkpoint(output_dir)
    if latest_checkpoint is None:
        return False, None

    latest_iteration = checkpoint_iteration_from_path(latest_checkpoint)
    if latest_iteration is None:
        return False, None

    is_complete = latest_iteration >= int(requested_iterations)
    if require_best_info:
        is_complete = is_complete and (output_dir / "best" / "best_program_info.json").is_file()
    return is_complete, latest_checkpoint


def load_best_program_info(run_dir: Path) -> Dict[str, Any]:
    info_path = run_dir / "best" / "best_program_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing best program info: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_checkpoint_best_program_info(checkpoint_dir: Path) -> Dict[str, Any]:
    info_path = checkpoint_dir / "best_program_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint best program info: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def score_from_best_program_info(info: Dict[str, Any]) -> float:
    metrics = info.get("metrics") or {}
    if "combined_score" in metrics:
        return float(metrics["combined_score"])
    if "score" in metrics:
        return float(metrics["score"])
    raise ValueError("best_program_info.json is missing both combined_score and score")


def build_task_env(task_id: str, family: str = "circle_packing") -> Dict[str, str]:
    family_definition = get_family_definition(family)
    return {family_definition.task_selector_env_var: task_id}
