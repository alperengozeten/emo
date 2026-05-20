"""Helpers for EMO-STA comparison summaries and adaptation branches.

This module supports the post-shared projected summary plus the remaining
adaptation comparison branches:
- best shared adaptation
- best local adaptation
- direct baseline comparison
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from openevolve.multi_task_shared_then_adapt.workflow import (
    load_best_program_info,
    load_checkpoint_best_program_info,
    score_from_best_program_info,
)


def resolve_optional_branch_iterations(
    *,
    adaptation_iterations: int,
    branch_iterations: Optional[int],
) -> tuple[int, bool]:
    """Resolve an adaptation-like branch budget and whether it defaulted."""
    if branch_iterations is None:
        return int(adaptation_iterations), True
    return int(branch_iterations), False


def resolve_best_shared_adaptation_iterations(
    *,
    adaptation_iterations: int,
    best_shared_adaptation_iterations: Optional[int],
) -> tuple[int, bool]:
    return resolve_optional_branch_iterations(
        adaptation_iterations=adaptation_iterations,
        branch_iterations=best_shared_adaptation_iterations,
    )


def resolve_best_local_adaptation_iterations(
    *,
    adaptation_iterations: int,
    best_local_adaptation_iterations: Optional[int],
) -> tuple[int, bool]:
    return resolve_optional_branch_iterations(
        adaptation_iterations=adaptation_iterations,
        branch_iterations=best_local_adaptation_iterations,
    )


def _best_program_info_path(*, root: Path, checkpoint_layout: bool) -> Path:
    if checkpoint_layout:
        return root / "best_program_info.json"
    return root / "best" / "best_program_info.json"


def load_optional_best_program_result(
    *,
    root: Path,
    checkpoint_layout: bool,
    iterations: Optional[int],
    executed: bool = True,
    output_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    reuse_existing_if_not_executed: bool = False,
) -> Dict[str, Any]:
    """Return a normalized best-program summary for a phase when present."""
    info_path = _best_program_info_path(root=root, checkpoint_layout=checkpoint_layout)

    if not executed:
        if reuse_existing_if_not_executed and info_path.is_file():
            info = (
                load_checkpoint_best_program_info(root)
                if checkpoint_layout
                else load_best_program_info(root)
            )
            return {
                "executed": False,
                "reused_existing": True,
                "iterations": iterations,
                "output_dir": str((output_dir or root).resolve()),
                "checkpoint_path": (
                    str(checkpoint_path.resolve())
                    if checkpoint_path is not None
                    else None
                ),
                "best_score": score_from_best_program_info(info),
                "best_metrics": info.get("metrics"),
                "best_program_info_path": str(info_path.resolve()),
            }
        return {
            "executed": False,
            "reused_existing": False,
            "iterations": None,
            "output_dir": str((output_dir or root).resolve()),
            "checkpoint_path": (
                str(checkpoint_path.resolve())
                if checkpoint_path is not None
                else None
            ),
            "best_score": None,
            "best_metrics": None,
            "best_program_info_path": None,
        }

    if info_path.is_file():
        info = (
            load_checkpoint_best_program_info(root)
            if checkpoint_layout
            else load_best_program_info(root)
        )
        return {
            "executed": bool(executed),
            "reused_existing": False,
            "iterations": iterations if executed else None,
            "output_dir": str((output_dir or root).resolve()),
            "checkpoint_path": (
                str(checkpoint_path.resolve())
                if checkpoint_path is not None
                else None
            ),
            "best_score": score_from_best_program_info(info),
            "best_metrics": info.get("metrics"),
            "best_program_info_path": str(info_path.resolve()),
        }

    return {
        "executed": bool(executed),
        "reused_existing": False,
        "iterations": iterations if executed else None,
        "output_dir": str((output_dir or root).resolve()),
        "checkpoint_path": (
            str(checkpoint_path.resolve())
            if checkpoint_path is not None
            else None
        ),
        "best_score": None,
        "best_metrics": None,
        "best_program_info_path": None,
    }


def _load_spawn_metadata(checkpoint_root: Path) -> Dict[str, Any]:
    metadata_path = checkpoint_root / "spawn_metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_optional_adaptation_branch_result(
    *,
    root: Path,
    iterations: Optional[int],
    executed: bool,
    output_dir: Optional[Path],
    source_checkpoint_root: Path,
) -> Dict[str, Any]:
    """Return a normalized optional-branch summary with source metadata."""
    result = load_optional_best_program_result(
        root=root,
        checkpoint_layout=False,
        iterations=iterations,
        executed=executed,
        output_dir=output_dir,
        checkpoint_path=source_checkpoint_root,
    )
    spawn_metadata = _load_spawn_metadata(source_checkpoint_root) if executed else {}
    legacy_mode_key = "seed" + "_mode"
    result.update(
        {
            "source_checkpoint_path": str(source_checkpoint_root.resolve()),
            "source_shared_program_id": spawn_metadata.get("source_shared_program_id"),
            "source_shared_metrics": spawn_metadata.get("source_shared_metrics"),
            "selection_mode": spawn_metadata.get("selection_mode")
            or spawn_metadata.get(legacy_mode_key),
        }
    )
    return result


def collect_shared_projected_task_scores(
    *,
    spawned_root: Path,
    task_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Load projected shared-task scores from spawned task checkpoints.

    The spawned checkpoint for each task is the shared checkpoint archive
    reprojected into task-local scores before any adaptation iterations run.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for task_id in task_ids:
        checkpoint_dir = (spawned_root / task_id).resolve()
        results[task_id] = load_optional_best_program_result(
            root=checkpoint_dir,
            checkpoint_layout=True,
            iterations=None,
            checkpoint_path=checkpoint_dir,
        )
    return results


def compute_score_delta(lhs: Any, rhs: Any) -> Optional[float]:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def build_task_comparison_summary(
    *,
    task_spec: Mapping[str, Any],
    spawn_checkpoint: Path,
    shared_projected: Mapping[str, Any],
    warmstarted_adaptation: Mapping[str, Any],
    best_shared_adaptation: Mapping[str, Any],
    best_local_adaptation: Mapping[str, Any],
    direct_baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble one task record while preserving legacy flat keys."""
    task_summary = {
        "task_spec": dict(task_spec),
        "spawn_checkpoint": str(spawn_checkpoint.resolve()),
        "spawn_best_score": shared_projected.get("best_score"),
        "spawn_best_metrics": shared_projected.get("best_metrics"),
        "adaptation_output_dir": warmstarted_adaptation.get("output_dir"),
        "adapted_best_score": warmstarted_adaptation.get("best_score"),
        "adapted_best_metrics": warmstarted_adaptation.get("best_metrics"),
        "best_shared_adaptation_output_dir": best_shared_adaptation.get("output_dir"),
        "best_shared_adaptation_best_score": best_shared_adaptation.get("best_score"),
        "best_shared_adaptation_best_metrics": best_shared_adaptation.get(
            "best_metrics"
        ),
        "best_local_adaptation_output_dir": best_local_adaptation.get("output_dir"),
        "best_local_adaptation_best_score": best_local_adaptation.get("best_score"),
        "best_local_adaptation_best_metrics": best_local_adaptation.get(
            "best_metrics"
        ),
        "baseline_output_dir": direct_baseline.get("output_dir"),
        "baseline_best_score": direct_baseline.get("best_score"),
        "baseline_best_metrics": direct_baseline.get("best_metrics"),
        "shared_projected": dict(shared_projected),
        "warmstarted_adaptation": dict(warmstarted_adaptation),
        "best_shared_adaptation": dict(best_shared_adaptation),
        "best_local_adaptation": dict(best_local_adaptation),
        "direct_baseline": dict(direct_baseline),
    }
    task_summary["deltas"] = {
        "warmstart_minus_best_shared": compute_score_delta(
            warmstarted_adaptation.get("best_score"),
            best_shared_adaptation.get("best_score"),
        ),
        "warmstart_minus_best_local": compute_score_delta(
            warmstarted_adaptation.get("best_score"),
            best_local_adaptation.get("best_score"),
        ),
        "best_local_minus_best_shared": compute_score_delta(
            best_local_adaptation.get("best_score"),
            best_shared_adaptation.get("best_score"),
        ),
        "warmstart_minus_shared_projected": compute_score_delta(
            warmstarted_adaptation.get("best_score"),
            shared_projected.get("best_score"),
        ),
        "warmstart_minus_baseline": compute_score_delta(
            warmstarted_adaptation.get("best_score"),
            direct_baseline.get("best_score"),
        ),
    }
    return task_summary


def write_comparison_summary_csv(
    *,
    csv_path: Path,
    task_summaries: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write a compact flat task comparison table beside the JSON summary."""
    fieldnames = [
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
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task_id, payload in task_summaries.items():
            shared_projected = payload.get("shared_projected") or {}
            warmstarted_adaptation = payload.get("warmstarted_adaptation") or {}
            best_shared_adaptation = payload.get("best_shared_adaptation") or {}
            best_local_adaptation = payload.get("best_local_adaptation") or {}
            direct_baseline = payload.get("direct_baseline") or {}
            deltas = payload.get("deltas") or {}
            writer.writerow(
                {
                    "task_id": task_id,
                    "shared_projected_score": shared_projected.get("best_score"),
                    "warmstarted_adaptation_score": warmstarted_adaptation.get(
                        "best_score"
                    ),
                    "best_shared_adaptation_score": best_shared_adaptation.get(
                        "best_score"
                    ),
                    "best_local_adaptation_score": best_local_adaptation.get(
                        "best_score"
                    ),
                    "baseline_score": direct_baseline.get("best_score"),
                    "warmstart_minus_best_shared": deltas.get(
                        "warmstart_minus_best_shared"
                    ),
                    "warmstart_minus_best_local": deltas.get(
                        "warmstart_minus_best_local"
                    ),
                    "best_local_minus_best_shared": deltas.get(
                        "best_local_minus_best_shared"
                    ),
                    "warmstart_minus_shared_projected": deltas.get(
                        "warmstart_minus_shared_projected"
                    ),
                    "warmstart_minus_baseline": deltas.get("warmstart_minus_baseline"),
                    "shared_projected_best_program_info_path": shared_projected.get(
                        "best_program_info_path"
                    ),
                    "warmstarted_adaptation_best_program_info_path": warmstarted_adaptation.get(
                        "best_program_info_path"
                    ),
                    "best_shared_adaptation_best_program_info_path": (
                        best_shared_adaptation.get("best_program_info_path")
                    ),
                    "best_local_adaptation_best_program_info_path": (
                        best_local_adaptation.get("best_program_info_path")
                    ),
                    "baseline_best_program_info_path": direct_baseline.get(
                        "best_program_info_path"
                    ),
                }
            )
    return csv_path
