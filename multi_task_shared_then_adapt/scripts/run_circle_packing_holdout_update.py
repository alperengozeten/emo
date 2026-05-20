#!/usr/bin/env python3
"""Recompute persisted circle-packing holdout OOD summaries for one finished run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.holdout_eval import (  # noqa: E402
    resolve_best_program_path,
    run_circle_packing_holdout_evaluation,
    run_circle_packing_adaptation_holdout_update,
)
from openevolve.multi_task_shared_then_adapt.workflow import (  # noqa: E402
    family_task_specs,
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute persisted circle-packing holdout OOD summaries for one "
            "finished EMO-STA run. This script updates shared, warmstart, and "
            "single-task results, and can also refresh the two adaptation-branch "
            "holdout branches."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the circle_packing EMO-STA manifest.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Completed EMO-STA run directory to update in place.",
    )
    parser.add_argument(
        "--holdout-selector",
        default="all",
        help="Circle-packing holdout selector. Default: all.",
    )
    parser.add_argument(
        "--skip-base-holdout",
        action="store_true",
        help="Do not recompute shared/warmstart/single-task holdout results.",
    )
    parser.add_argument(
        "--skip-best-shared",
        action="store_true",
        help="Do not refresh the best-shared holdout branch.",
    )
    parser.add_argument(
        "--skip-best-local",
        action="store_true",
        help="Do not refresh the best-local holdout branch.",
    )
    parser.add_argument(
        "--timeout-override-seconds",
        type=float,
        default=None,
        help=(
            "Optional OOD-only evaluator timeout override in seconds. When set, "
            "the family evaluator uses max(default_timeout, override)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if manifest.family != "circle_packing":
        raise SystemExit(
            "run_circle_packing_holdout_update.py only supports family='circle_packing'"
        )

    include_best_shared = not args.skip_best_shared
    include_best_local = not args.skip_best_local
    env = None
    if args.timeout_override_seconds is not None:
        env = {
            "MT_STS_OOD_TIMEOUT_OVERRIDE_SECONDS": str(
                float(args.timeout_override_seconds)
            )
        }

    results_path = Path(args.results_dir).resolve()
    if not results_path.is_dir():
        raise SystemExit(f"Missing results directory: {results_path}")

    if not args.skip_base_holdout:
        comparison_summary_path = results_path / "comparison_summary.json"
        comparison_summary = {}
        if comparison_summary_path.is_file():
            try:
                loaded = json.loads(comparison_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                comparison_summary = loaded

        shared_run_summary = comparison_summary.get("shared_run")
        shared_checkpoint = None
        if isinstance(shared_run_summary, dict):
            checkpoint_used = shared_run_summary.get("checkpoint_used")
            if isinstance(checkpoint_used, str) and checkpoint_used.strip():
                shared_checkpoint = Path(checkpoint_used).resolve()
        if shared_checkpoint is None:
            shared_checkpoint = results_path / "shared_run"

        task_specs = family_task_specs(manifest)
        shared_program_path = resolve_best_program_path(
            shared_checkpoint,
            initial_program=manifest.initial_program,
            checkpoint_layout=True,
        )
        adaptation_program_paths = {
            task.task_id: resolve_best_program_path(
                results_path / "adaptation" / task.task_id,
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
            )
            for task in task_specs
        }
        baseline_program_paths = {
            task.task_id: resolve_best_program_path(
                results_path / "baselines" / task.task_id,
                initial_program=manifest.initial_program,
                checkpoint_layout=False,
            )
            for task in task_specs
        }
        run_circle_packing_holdout_evaluation(
            family="circle_packing",
            run_root=results_path,
            holdout_selector=args.holdout_selector,
            skip_holdouts=False,
            shared_program_path=shared_program_path,
            adaptation_program_paths=adaptation_program_paths,
            baseline_program_paths=baseline_program_paths,
            evaluation_file=manifest.evaluation_file,
            env=env,
        )

    if include_best_shared or include_best_local:
        run_circle_packing_adaptation_holdout_update(
            manifest=manifest,
            results_dir=results_path,
            holdout_selector=args.holdout_selector,
            include_best_shared=include_best_shared,
            include_best_local=include_best_local,
            env=env,
        )

    summary_path = results_path / "holdout_evaluation" / "holdout_summary.json"
    print(f"Updated holdout summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
