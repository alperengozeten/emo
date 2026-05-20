#!/usr/bin/env python3
"""Run post-hoc OOD evaluation for finished EMO-STA runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.posthoc_ood import (  # noqa: E402
    run_posthoc_ood_evaluation,
    supported_posthoc_ood_families,
)
from openevolve.multi_task_shared_then_adapt.workflow import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    supported = ", ".join(supported_posthoc_ood_families())
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen best programs from an already-finished EMO-STA run on "
            f"post-hoc OOD tasks. Supported families: {supported}."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the EMO-STA manifest used to identify the family/evaluator.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Completed EMO-STA run directory containing comparison_summary.json.",
    )
    parser.add_argument(
        "--ood-task-ids",
        default=None,
        help=(
            "Comma-separated OOD task IDs. Defaults to the family registered OOD "
            "tasks."
        ),
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help=(
            "Include the shared best program. If no include flags are supplied, "
            "shared, adapted, and baselines are all included."
        ),
    )
    parser.add_argument(
        "--include-adapted",
        action="store_true",
        help=(
            "Include final adapted task-specific best programs. If no include "
            "flags are supplied, shared, adapted, and baselines are all included."
        ),
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help=(
            "Include direct single-task baseline best programs. If no include "
            "flags are supplied, shared, adapted, and baselines are all included."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <results-dir>/posthoc_ood/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing post-hoc OOD summary outputs.",
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

    include_filter_used = (
        args.include_shared or args.include_adapted or args.include_baselines
    )
    include_shared = args.include_shared if include_filter_used else True
    include_adapted = args.include_adapted if include_filter_used else True
    include_baselines = args.include_baselines if include_filter_used else True
    env = None
    if args.timeout_override_seconds is not None:
        env = {
            "MT_STS_OOD_TIMEOUT_OVERRIDE_SECONDS": str(
                float(args.timeout_override_seconds)
            )
        }

    try:
        summary = run_posthoc_ood_evaluation(
            manifest=manifest,
            results_dir=args.results_dir,
            ood_task_ids=args.ood_task_ids,
            include_shared=include_shared,
            include_adapted=include_adapted,
            include_baselines=include_baselines,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            env=env,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Post-hoc OOD summary written to {summary['summary_path']}")
    print(f"Post-hoc OOD CSV written to {summary['csv_path']}")
    print(
        f"Evaluated {summary['program_count']} frozen program(s) on "
        f"{len(summary['ood_tasks'])} OOD task(s): {', '.join(summary['ood_tasks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
