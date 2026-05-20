#!/usr/bin/env python3
"""Spawn task-local checkpoints from an EMO-STA shared checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
HEXAGON_PACKING_MANIFEST = (
    "multi_task_shared_then_adapt/configs/hexagon_packing_emo_sta.yaml"
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

from openevolve.multi_task_shared_then_adapt.runner import write_json
from openevolve.multi_task_shared_then_adapt.spawn import spawn_task_checkpoints
from openevolve.multi_task_shared_then_adapt.workflow import family_task_specs, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn task checkpoints from a shared EMO-STA checkpoint. "
            f"Defaults to unit-square circle packing; use --manifest {CIRCLE_PACKING_RECTANGLE_MANIFEST} "
            f"for rectangle circle packing, --manifest {K_MODULE_BALANCED_MANIFEST} "
            f"for K-module, or --manifest "
            f"{FUNCTION_MINIMIZATION_MANIFEST} for function minimization, or "
            f"--manifest {HEILBRONN_TRIANGLE_MANIFEST} for Heilbronn triangle, or "
            f"--manifest {HEXAGON_PACKING_MANIFEST} for hexagon packing, or "
            f"--manifest {SIGNAL_PROCESSING_MANIFEST} for signal processing, or "
            f"--manifest {SLDBENCH_3D_MANIFEST} for SLDBench 3D scaling laws, or "
            f"--manifest {RUST_ADAPTIVE_SORT_MANIFEST} for Rust adaptive sort."
        )
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=(
            "Path to the EMO-STA manifest. "
            f"Default: {DEFAULT_MANIFEST}. "
            f"Unit-square circle packing: {CIRCLE_PACKING_MANIFEST}. "
            f"Rectangle circle packing: {CIRCLE_PACKING_RECTANGLE_MANIFEST}. "
            f"K-module: {K_MODULE_BALANCED_MANIFEST}. "
            f"Function minimization: {FUNCTION_MINIMIZATION_MANIFEST}. "
            f"Heilbronn triangle: {HEILBRONN_TRIANGLE_MANIFEST}. "
            f"Hexagon packing: {HEXAGON_PACKING_MANIFEST}. "
            f"Signal processing: {SIGNAL_PROCESSING_MANIFEST}. "
            f"SLDBench 3D: {SLDBENCH_3D_MANIFEST}. "
            f"Rust adaptive sort: {RUST_ADAPTIVE_SORT_MANIFEST}."
        ),
    )
    parser.add_argument(
        "--shared-checkpoint",
        required=True,
        help="Path to the shared checkpoint directory",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where spawned task checkpoints should be written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    task_ids = [task.task_id for task in family_task_specs(manifest)]
    output_root = Path(args.output_root).resolve()

    spawn_results = spawn_task_checkpoints(
        shared_checkpoint_path=args.shared_checkpoint,
        output_root=output_root,
        base_config_path=manifest.base_config,
        evaluation_file=manifest.evaluation_file,
        family=manifest.family,
        task_ids=task_ids,
        initial_program=manifest.initial_program,
    )

    summary_path = write_json(output_root / "spawn_summary.json", spawn_results)
    print(f"Spawned checkpoints written under {output_root}")
    print(f"Spawn summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
