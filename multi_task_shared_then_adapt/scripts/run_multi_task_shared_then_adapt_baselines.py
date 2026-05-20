#!/usr/bin/env python3
"""Run direct single-task baselines for an EMO-STA family."""

from __future__ import annotations

import argparse
import os
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

from openevolve.multi_task_shared_then_adapt.runner import (
    build_openevolve_command,
    default_run_name,
    run_command,
    write_json,
)
from openevolve.multi_task_shared_then_adapt.workflow import (
    build_phase_wandb_config,
    build_task_env,
    default_emo_sta_run_prefix,
    family_task_specs,
    load_best_program_info,
    load_manifest,
    phase_checkpoint_status,
    resolve_phase_system_prompt,
    resolve_emo_sta_wandb_run_id,
    run_emo_sta_family_preflight,
    score_from_best_program_info,
    write_phase_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct single-task baselines for EMO-STA comparisons. "
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
    parser.add_argument("--run-name", default=None, help="Run directory name under output_root")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the manifest output_root parent directory",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the manifest baseline iteration count",
    )
    parser.add_argument("--api-base", default=None, help="Forwarded OpenEvolve API base")
    parser.add_argument("--model", default=None, help="Alias for --primary-model")
    parser.add_argument("--primary-model", default=None, help="Forwarded OpenEvolve primary model")
    parser.add_argument(
        "--secondary-model",
        default=None,
        help="Forwarded OpenEvolve secondary model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun baseline tasks even if outputs already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    task_specs = family_task_specs(manifest)
    try:
        run_emo_sta_family_preflight(manifest, task_specs=task_specs)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    output_parent = Path(args.output_root).resolve() if args.output_root else manifest.output_root
    run_name = args.run_name or default_run_name(
        default_emo_sta_run_prefix(base_prefix="emo_sta_baselines", manifest=manifest)
    )
    wandb_run_label = args.run_name
    run_root = output_parent / run_name
    baselines_root = run_root / "baselines"
    configs_root = run_root / "configs"
    wandb_run_id = (
        resolve_emo_sta_wandb_run_id(run_root, force_new=args.force)
        if manifest.wandb_single_run
        else None
    )

    primary_model = args.primary_model or args.model
    iterations = (
        int(args.iterations)
        if args.iterations is not None
        else manifest.default_baseline_iterations
    )
    summary = {
        "workflow": "direct_single_task_baselines",
        "family": manifest.family,
        "manifest_path": str(manifest.manifest_path),
        "run_root": str(run_root),
        "baseline_iterations": iterations,
        "wandb": {
            "enabled": manifest.wandb_enabled,
            "project": manifest.wandb_project,
            "single_run": manifest.wandb_single_run,
            "run_id": wandb_run_id,
        },
        "tasks": {},
    }

    for task in task_specs:
        task_output = baselines_root / task.task_id
        baseline_config_path = write_phase_config(
            base_config_path=manifest.base_config,
            output_config_path=configs_root / f"baseline_{task.task_id}.yaml",
            iterations=iterations,
            api_base=args.api_base,
            primary_model=primary_model,
            secondary_model=args.secondary_model,
            system_prompt=resolve_phase_system_prompt(
                manifest,
                phase="baseline",
                task_id=task.task_id,
            ),
            wandb_config=build_phase_wandb_config(
                manifest,
                run_name=run_name,
                run_root=run_root,
                wandb_run_id=wandb_run_id,
                phase="baseline",
                task_id=task.task_id,
                run_label=wandb_run_label,
                baseline_iterations=iterations,
            ),
        )
        is_complete = False
        resume_checkpoint = None
        if not args.force:
            is_complete, resume_checkpoint = phase_checkpoint_status(
                task_output,
                iterations,
                require_best_info=True,
            )
        if args.force or not is_complete:
            env = dict(os.environ)
            env.update(build_task_env(task.task_id, family=manifest.family))
            command = build_openevolve_command(
                initial_program=manifest.initial_program,
                evaluation_file=manifest.evaluation_file,
                config_path=baseline_config_path,
                output_dir=task_output,
                iterations=iterations,
                checkpoint_path=resume_checkpoint,
                api_base=args.api_base,
                primary_model=primary_model,
                secondary_model=args.secondary_model,
            )
            run_command(command, env=env)

        info = load_best_program_info(task_output)
        summary["tasks"][task.task_id] = {
            "task_spec": task.to_spec_dict(),
            "output_dir": str(task_output),
            "best_score": score_from_best_program_info(info),
            "best_metrics": info["metrics"],
        }

    summary_path = write_json(run_root / "baseline_summary.json", summary)
    print(f"\nBaseline summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
