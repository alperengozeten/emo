#!/usr/bin/env python3
"""Single-start multi-task shared-then-adapt orchestrator."""

from __future__ import annotations

import argparse
import json
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
from openevolve.multi_task_shared_then_adapt.adaptation_branches import (
    build_task_comparison_summary,
    collect_shared_projected_task_scores,
    load_optional_adaptation_branch_result,
    load_optional_best_program_result,
    resolve_best_shared_adaptation_iterations,
    resolve_best_local_adaptation_iterations,
    write_comparison_summary_csv,
)
from openevolve.multi_task_shared_then_adapt.holdout_eval import (
    resolve_best_program_path,
    run_circle_packing_holdout_evaluation,
)
from openevolve.multi_task_shared_then_adapt.spawn import (
    spawn_best_shared_checkpoints,
    spawn_best_local_checkpoints,
    spawn_task_checkpoints,
)
from openevolve.multi_task_shared_then_adapt.workflow import (
    build_phase_wandb_config,
    build_task_env,
    default_emo_sta_run_prefix,
    family_task_specs,
    find_latest_single_task_checkpoint,
    load_checkpoint_best_program_info,
    load_manifest,
    phase_checkpoint_status,
    resolve_phase_system_prompt,
    resolve_emo_sta_wandb_run_id,
    run_emo_sta_family_preflight,
    validate_emo_sta_iteration_budget,
    write_phase_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-task shared-then-adapt (EMO-STA) workflow end to end. "
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
    parser.add_argument("--run-name", default=None, help="Run directory name under output_root")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the manifest output_root parent directory",
    )
    parser.add_argument(
        "--shared-iterations",
        type=int,
        default=None,
        help="Override the manifest shared-phase iteration count",
    )
    parser.add_argument(
        "--adaptation-iterations",
        type=int,
        default=None,
        help="Override the manifest adaptation iteration count",
    )
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=None,
        help="Override the manifest baseline iteration count",
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
            "Run the standard archive-warmstart adaptation branch initialized "
            "from the projected shared checkpoint population. This is enabled "
            "by default for backward compatibility."
        ),
    )
    parser.add_argument(
        "--skip-warmstart-adaptation",
        dest="run_warmstart_adaptation",
        action="store_false",
        help=(
            "Disable only the standard archive-warmstart adaptation branch. "
            "Other requested adaptation branches can still run unless "
            "--skip-adaptation is also set."
        ),
    )
    parser.add_argument(
        "--run-best-shared-adaptation",
        dest="run_best_shared_adaptation",
        action="store_true",
        help=(
            "Run a one-program task-local adaptation branch initialized from the globally "
            "best shared-average program from the shared checkpoint."
        ),
    )
    parser.add_argument(
        "--skip-best-shared-adaptation",
        dest="run_best_shared_adaptation",
        action="store_false",
        help="Explicitly disable the optional best-shared adaptation branch.",
    )
    parser.add_argument(
        "--best-shared-adaptation-iterations",
        type=int,
        default=None,
        help=(
            "Optional iteration override for the best-shared adaptation branch. "
            "Defaults to --adaptation-iterations."
        ),
    )
    parser.add_argument(
        "--run-best-local-adaptation",
        dest="run_best_local_adaptation",
        action="store_true",
        help=(
            "Run a one-program task-local adaptation branch initialized from the locally "
            "best program retained in the shared checkpoint for each task."
        ),
    )
    parser.add_argument(
        "--skip-best-local-adaptation",
        dest="run_best_local_adaptation",
        action="store_false",
        help="Explicitly disable the optional best-local adaptation branch.",
    )
    parser.add_argument(
        "--best-local-adaptation-iterations",
        type=int,
        default=None,
        help=(
            "Optional iteration override for the best-local adaptation branch. "
            "Defaults to --adaptation-iterations."
        ),
    )
    parser.add_argument(
        "--shared-checkpoint",
        default=None,
        help="Reuse an existing shared checkpoint instead of launching Phase A",
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
        "--skip-adaptation",
        action="store_true",
        help="Stop after shared phase and checkpoint spawning",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip direct single-task baselines",
    )
    parser.add_argument(
        "--skip-holdouts",
        action="store_true",
        help="Skip circle-packing post-hoc evaluation-only holdout evaluation",
    )
    parser.add_argument(
        "--holdout-selector",
        default="all",
        help=(
            "Circle-packing holdout selector for the post-hoc holdout phase. "
            "Valid values: all, all_holdouts, cp_n21, cp_n23, cp_n25, "
            "or a comma-separated subset such as cp_n21,cp_n25."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun phases even if outputs already exist",
    )
    parser.add_argument(
        "--allow-unsafe-iterations",
        action="store_true",
        help=(
            "Allow iteration-unsafe EMO-STA budgets where "
            "shared + task_count * adaptation != task_count * baseline."
        ),
    )
    return parser.parse_args()


def _spawn_outputs_match(
    spawned_root: Path,
    task_ids: list[str],
    shared_checkpoint: Path,
    *,
    selection_mode: str | None = None,
) -> bool:
    legacy_mode_key = "seed" + "_mode"
    for task_id in task_ids:
        metadata_path = spawned_root / task_id / "spawn_metadata.json"
        if not metadata_path.is_file():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if Path(metadata.get("shared_checkpoint_path", "")).resolve() != shared_checkpoint.resolve():
            return False
        recorded_selection_mode = metadata.get("selection_mode") or metadata.get(
            legacy_mode_key
        )
        if selection_mode is not None and recorded_selection_mode != selection_mode:
            return False
    return True


def _maybe_run_single_task(
    *,
    output_dir: Path,
    checkpoint_path: Path | None,
    env: dict[str, str],
    manifest,
    config_path: Path,
    iterations: int,
    api_base: str | None,
    primary_model: str | None,
    secondary_model: str | None,
    force: bool,
) -> None:
    effective_checkpoint_path = checkpoint_path
    if not force:
        is_complete, latest_output_checkpoint = phase_checkpoint_status(
            output_dir,
            iterations,
            require_best_info=True,
        )
        if is_complete:
            return
        if latest_output_checkpoint is not None:
            effective_checkpoint_path = latest_output_checkpoint

    command = build_openevolve_command(
        initial_program=manifest.initial_program,
        evaluation_file=manifest.evaluation_file,
        config_path=config_path,
        output_dir=output_dir,
        iterations=iterations,
        checkpoint_path=effective_checkpoint_path,
        api_base=api_base,
        primary_model=primary_model,
        secondary_model=secondary_model,
    )
    run_command(command, env=env)


def _run_task_phase_for_all_tasks(
    *,
    task_specs,
    output_root: Path,
    configs_root: Path,
    config_prefix: str,
    phase: str,
    iterations: int,
    checkpoint_root: Path | None,
    manifest,
    run_name: str,
    run_root: Path,
    wandb_run_id: str | None,
    wandb_run_label: str | None,
    shared_iterations: int,
    adaptation_iterations: int,
    baseline_iterations: int,
    api_base: str | None,
    primary_model: str | None,
    secondary_model: str | None,
    force: bool,
) -> None:
    for task in task_specs:
        output_dir = output_root / task.task_id
        config_path = write_phase_config(
            base_config_path=manifest.base_config,
            output_config_path=configs_root / f"{config_prefix}_{task.task_id}.yaml",
            iterations=iterations,
            api_base=api_base,
            primary_model=primary_model,
            secondary_model=secondary_model,
            system_prompt=resolve_phase_system_prompt(
                manifest,
                phase=phase,
                task_id=task.task_id,
            ),
            wandb_config=build_phase_wandb_config(
                manifest,
                run_name=run_name,
                run_root=run_root,
                wandb_run_id=wandb_run_id,
                phase=phase,
                task_id=task.task_id,
                run_label=wandb_run_label,
                shared_iterations=shared_iterations,
                adaptation_iterations=adaptation_iterations,
                baseline_iterations=baseline_iterations,
            ),
        )
        _maybe_run_single_task(
            output_dir=output_dir,
            checkpoint_path=(
                checkpoint_root / task.task_id
                if checkpoint_root is not None
                else None
            ),
            env={**os.environ, **build_task_env(task.task_id, family=manifest.family)},
            manifest=manifest,
            config_path=config_path,
            iterations=iterations,
            api_base=api_base,
            primary_model=primary_model,
            secondary_model=secondary_model,
            force=force,
        )


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    task_specs = family_task_specs(manifest)
    task_ids = [task.task_id for task in task_specs]
    try:
        run_emo_sta_family_preflight(manifest, task_specs=task_specs)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    primary_model = args.primary_model or args.model
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

    output_parent = Path(args.output_root).resolve() if args.output_root else manifest.output_root
    run_name = args.run_name or default_run_name(
        default_emo_sta_run_prefix(base_prefix="emo_sta", manifest=manifest)
    )
    wandb_run_label = args.run_name
    run_root = output_parent / run_name
    configs_root = run_root / "configs"
    shared_output = run_root / "shared_run"
    spawned_root = run_root / "spawned_checkpoints"
    best_shared_spawned_root = run_root / "spawned_checkpoints_best_shared"
    best_local_spawned_root = run_root / "spawned_checkpoints_best_local"
    adaptation_root = run_root / "adaptation"
    best_shared_adaptation_root = run_root / "adaptation_best_shared"
    best_local_adaptation_root = run_root / "adaptation_best_local"
    baselines_root = run_root / "baselines"
    wandb_run_id = (
        resolve_emo_sta_wandb_run_id(run_root, force_new=args.force)
        if manifest.wandb_single_run
        else None
    )
    best_shared_adaptation_iterations = None
    best_shared_branch_defaulted = False
    if args.run_best_shared_adaptation:
        (
            best_shared_adaptation_iterations,
            best_shared_branch_defaulted,
        ) = resolve_best_shared_adaptation_iterations(
            adaptation_iterations=adaptation_iterations,
            best_shared_adaptation_iterations=args.best_shared_adaptation_iterations,
        )
    best_local_adaptation_iterations = None
    best_local_branch_defaulted = False
    if args.run_best_local_adaptation:
        (
            best_local_adaptation_iterations,
            best_local_branch_defaulted,
        ) = resolve_best_local_adaptation_iterations(
            adaptation_iterations=adaptation_iterations,
            best_local_adaptation_iterations=args.best_local_adaptation_iterations,
        )

    shared_config_path = write_phase_config(
        base_config_path=manifest.base_config,
        output_config_path=configs_root / "shared_config.yaml",
        iterations=shared_iterations,
        api_base=args.api_base,
        primary_model=primary_model,
        secondary_model=args.secondary_model,
        system_prompt=resolve_phase_system_prompt(manifest, phase="shared"),
        wandb_config=build_phase_wandb_config(
            manifest,
            run_name=run_name,
            run_root=run_root,
            wandb_run_id=wandb_run_id,
            phase="shared",
            run_label=wandb_run_label,
            shared_iterations=shared_iterations,
            adaptation_iterations=adaptation_iterations,
            baseline_iterations=baseline_iterations,
        ),
    )

    if args.shared_checkpoint:
        shared_checkpoint = Path(args.shared_checkpoint).resolve()
    else:
        shared_phase_complete = False
        existing_shared_checkpoint = None
        if not args.force:
            shared_phase_complete, existing_shared_checkpoint = phase_checkpoint_status(
                shared_output,
                shared_iterations,
                require_best_info=False,
            )

        if shared_phase_complete and existing_shared_checkpoint is not None:
            shared_checkpoint = existing_shared_checkpoint
        else:
            shared_env = dict(os.environ)
            shared_env.update(build_task_env("all", family=manifest.family))
            shared_command = build_openevolve_command(
                initial_program=manifest.initial_program,
                evaluation_file=manifest.evaluation_file,
                config_path=shared_config_path,
                output_dir=shared_output,
                iterations=shared_iterations,
                checkpoint_path=existing_shared_checkpoint,
                api_base=args.api_base,
                primary_model=primary_model,
                secondary_model=args.secondary_model,
            )
            run_command(shared_command, env=shared_env)
            shared_checkpoint = find_latest_single_task_checkpoint(shared_output)
            if shared_checkpoint is None:
                raise FileNotFoundError(
                    f"No shared checkpoint found under {shared_output / 'checkpoints'} after Phase A"
                )
            shared_iteration = phase_checkpoint_status(
                shared_output,
                shared_iterations,
                require_best_info=False,
            )[0]
            if not shared_iteration:
                raise RuntimeError(
                    "Shared phase did not reach the requested iteration budget; "
                    f"latest checkpoint under {shared_output / 'checkpoints'} is incomplete"
                )

    if args.force or not _spawn_outputs_match(spawned_root, task_ids, shared_checkpoint):
        spawn_task_checkpoints(
            shared_checkpoint_path=shared_checkpoint,
            output_root=spawned_root,
            base_config_path=manifest.base_config,
            evaluation_file=manifest.evaluation_file,
            family=manifest.family,
            task_ids=task_ids,
            initial_program=manifest.initial_program,
        )

    run_warmstarted_adaptation = (
        not args.skip_adaptation
        and args.run_warmstart_adaptation
    )
    run_best_shared_adaptation = (
        not args.skip_adaptation
        and args.run_best_shared_adaptation
        and best_shared_adaptation_iterations is not None
    )
    run_best_local_adaptation = (
        not args.skip_adaptation
        and args.run_best_local_adaptation
        and best_local_adaptation_iterations is not None
    )
    run_baselines = not args.skip_baselines

    if run_best_shared_adaptation and (
        args.force
        or not _spawn_outputs_match(
            best_shared_spawned_root,
            task_ids,
            shared_checkpoint,
            selection_mode="best_shared",
        )
    ):
        spawn_best_shared_checkpoints(
            shared_checkpoint_path=shared_checkpoint,
            output_root=best_shared_spawned_root,
            base_config_path=manifest.base_config,
            evaluation_file=manifest.evaluation_file,
            family=manifest.family,
            task_ids=task_ids,
            initial_program=manifest.initial_program,
        )

    if run_best_local_adaptation and (
        args.force
        or not _spawn_outputs_match(
            best_local_spawned_root,
            task_ids,
            shared_checkpoint,
            selection_mode="best_local",
        )
    ):
        spawn_best_local_checkpoints(
            shared_checkpoint_path=shared_checkpoint,
            output_root=best_local_spawned_root,
            base_config_path=manifest.base_config,
            evaluation_file=manifest.evaluation_file,
            family=manifest.family,
            task_ids=task_ids,
            initial_program=manifest.initial_program,
        )

    if run_warmstarted_adaptation:
        _run_task_phase_for_all_tasks(
            task_specs=task_specs,
            output_root=adaptation_root,
            configs_root=configs_root,
            config_prefix="adaptation",
            phase="adaptation",
            iterations=adaptation_iterations,
            checkpoint_root=spawned_root,
            manifest=manifest,
            run_name=run_name,
            run_root=run_root,
            wandb_run_id=wandb_run_id,
            wandb_run_label=wandb_run_label,
            shared_iterations=shared_iterations,
            adaptation_iterations=adaptation_iterations,
            baseline_iterations=baseline_iterations,
            api_base=args.api_base,
            primary_model=primary_model,
            secondary_model=args.secondary_model,
            force=args.force,
        )

    if run_best_shared_adaptation:
        _run_task_phase_for_all_tasks(
            task_specs=task_specs,
            output_root=best_shared_adaptation_root,
            configs_root=configs_root,
            config_prefix="best_shared_adaptation",
            phase="best_shared_adaptation",
            iterations=best_shared_adaptation_iterations,
            checkpoint_root=best_shared_spawned_root,
            manifest=manifest,
            run_name=run_name,
            run_root=run_root,
            wandb_run_id=wandb_run_id,
            wandb_run_label=wandb_run_label,
            shared_iterations=shared_iterations,
            adaptation_iterations=adaptation_iterations,
            baseline_iterations=baseline_iterations,
            api_base=args.api_base,
            primary_model=primary_model,
            secondary_model=args.secondary_model,
            force=args.force,
        )

    if run_best_local_adaptation:
        _run_task_phase_for_all_tasks(
            task_specs=task_specs,
            output_root=best_local_adaptation_root,
            configs_root=configs_root,
            config_prefix="best_local_adaptation",
            phase="best_local_adaptation",
            iterations=best_local_adaptation_iterations,
            checkpoint_root=best_local_spawned_root,
            manifest=manifest,
            run_name=run_name,
            run_root=run_root,
            wandb_run_id=wandb_run_id,
            wandb_run_label=wandb_run_label,
            shared_iterations=shared_iterations,
            adaptation_iterations=adaptation_iterations,
            baseline_iterations=baseline_iterations,
            api_base=args.api_base,
            primary_model=primary_model,
            secondary_model=args.secondary_model,
            force=args.force,
        )

    if run_baselines:
        _run_task_phase_for_all_tasks(
            task_specs=task_specs,
            output_root=baselines_root,
            configs_root=configs_root,
            config_prefix="baseline",
            phase="baseline",
            iterations=baseline_iterations,
            checkpoint_root=None,
            manifest=manifest,
            run_name=run_name,
            run_root=run_root,
            wandb_run_id=wandb_run_id,
            wandb_run_label=wandb_run_label,
            shared_iterations=shared_iterations,
            adaptation_iterations=adaptation_iterations,
            baseline_iterations=baseline_iterations,
            api_base=args.api_base,
            primary_model=primary_model,
            secondary_model=args.secondary_model,
            force=args.force,
        )

    holdout_summary = None
    if manifest.family != "circle_packing":
        if args.skip_holdouts or (args.holdout_selector or "").strip() not in {"", "all"}:
            print(
                "Holdout evaluation is currently only implemented for "
                "family='circle_packing'; ignoring holdout flags."
            )
    else:
        shared_program_path = resolve_best_program_path(
            shared_checkpoint,
            initial_program=manifest.initial_program,
            checkpoint_layout=True,
        )
        adaptation_program_paths = {
            task.task_id: resolve_best_program_path(
                adaptation_root / task.task_id,
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
            )
            for task in task_specs
        }
        baseline_program_paths = {
            task.task_id: resolve_best_program_path(
                baselines_root / task.task_id,
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
            )
            for task in task_specs
        }
        try:
            holdout_summary = run_circle_packing_holdout_evaluation(
                family=manifest.family,
                run_root=run_root,
                holdout_selector=args.holdout_selector,
                skip_holdouts=args.skip_holdouts,
                shared_program_path=shared_program_path,
                adaptation_program_paths=adaptation_program_paths,
                baseline_program_paths=baseline_program_paths,
                evaluation_file=manifest.evaluation_file,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    summary = {
        "workflow": "multi_task_shared_then_adapt",
        "family": manifest.family,
        "manifest_path": str(manifest.manifest_path),
        "run_root": str(run_root),
        "shared_iterations": shared_iterations,
        "adaptation_iterations": adaptation_iterations,
        "baseline_iterations": baseline_iterations if run_baselines else None,
        "best_shared_adaptation": {
            "requested": bool(args.run_best_shared_adaptation),
            "enabled": bool(
                args.run_best_shared_adaptation
                and not args.skip_adaptation
            ),
            "iterations": best_shared_adaptation_iterations,
            "defaulted_to_adaptation_iterations": best_shared_branch_defaulted,
            "output_root": str(best_shared_adaptation_root),
            "spawned_checkpoint_root": str(best_shared_spawned_root),
        },
        "best_local_adaptation": {
            "requested": bool(args.run_best_local_adaptation),
            "enabled": bool(
                args.run_best_local_adaptation
                and not args.skip_adaptation
            ),
            "iterations": best_local_adaptation_iterations,
            "defaulted_to_adaptation_iterations": best_local_branch_defaulted,
            "output_root": str(best_local_adaptation_root),
            "spawned_checkpoint_root": str(best_local_spawned_root),
        },
        "wandb": {
            "enabled": manifest.wandb_enabled,
            "project": manifest.wandb_project,
            "single_run": manifest.wandb_single_run,
            "run_id": wandb_run_id,
        },
        "shared_run": {
            "output_dir": str(shared_output),
            "checkpoint_used": str(shared_checkpoint),
            "best_program_info": load_checkpoint_best_program_info(shared_checkpoint),
        },
        "tasks": {},
    }
    projected_shared_scores = collect_shared_projected_task_scores(
        spawned_root=spawned_root,
        task_ids=task_ids,
    )
    warmstarted_adaptation_executed = run_warmstarted_adaptation
    best_shared_adaptation_executed = run_best_shared_adaptation
    best_local_adaptation_executed = run_best_local_adaptation
    baseline_executed = run_baselines

    for task in task_specs:
        adaptation_output = adaptation_root / task.task_id
        best_shared_output = best_shared_adaptation_root / task.task_id
        best_local_output = best_local_adaptation_root / task.task_id
        baseline_output = baselines_root / task.task_id
        summary["tasks"][task.task_id] = build_task_comparison_summary(
            task_spec=task.to_spec_dict(),
            spawn_checkpoint=spawned_root / task.task_id,
            shared_projected=projected_shared_scores[task.task_id],
            warmstarted_adaptation=load_optional_best_program_result(
                root=adaptation_output,
                checkpoint_layout=False,
                iterations=(
                    adaptation_iterations if warmstarted_adaptation_executed else None
                ),
                executed=warmstarted_adaptation_executed,
                output_dir=adaptation_output,
            ),
            best_shared_adaptation=load_optional_adaptation_branch_result(
                root=best_shared_output,
                iterations=(
                    best_shared_adaptation_iterations
                    if best_shared_adaptation_executed
                    else None
                ),
                executed=best_shared_adaptation_executed,
                output_dir=best_shared_output,
                source_checkpoint_root=best_shared_spawned_root / task.task_id,
            ),
            best_local_adaptation=load_optional_adaptation_branch_result(
                root=best_local_output,
                iterations=(
                    best_local_adaptation_iterations
                    if best_local_adaptation_executed
                    else None
                ),
                executed=best_local_adaptation_executed,
                output_dir=best_local_output,
                source_checkpoint_root=best_local_spawned_root / task.task_id,
            ),
            direct_baseline=load_optional_best_program_result(
                root=baseline_output,
                checkpoint_layout=False,
                iterations=baseline_iterations,
                executed=baseline_executed,
                output_dir=baseline_output,
                reuse_existing_if_not_executed=True,
            ),
        )

    if not run_baselines:
        has_existing_baseline_results = any(
            (task_summary.get("direct_baseline") or {}).get("best_program_info_path")
            for task_summary in summary["tasks"].values()
        )
        if has_existing_baseline_results:
            summary["baseline_iterations"] = baseline_iterations

    if holdout_summary is not None:
        summary["holdout_evaluation"] = holdout_summary

    summary_path = write_json(run_root / "comparison_summary.json", summary)
    csv_path = write_comparison_summary_csv(
        csv_path=run_root / "comparison_summary.csv",
        task_summaries=summary["tasks"],
    )
    print(f"\nEMO-STA comparison summary written to {summary_path}")
    print(f"EMO-STA comparison CSV written to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
