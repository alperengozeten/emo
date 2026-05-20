#!/usr/bin/env python3
"""Persist adaptation OOD evaluations into existing run-local JSON summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.holdout_eval import (  # noqa: E402
    run_circle_packing_adaptation_holdout_update,
)
from openevolve.multi_task_shared_then_adapt.posthoc_ood import (  # noqa: E402
    run_posthoc_ood_evaluation,
    supported_posthoc_ood_families,
)
from openevolve.multi_task_shared_then_adapt.workflow import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add best-shared and best-local OOD results to existing "
            "run-local OOD JSON summaries without recomputing or overwriting the "
            "existing shared/adapted/baseline payloads."
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
        help="Completed EMO-STA run directory to update in place.",
    )
    parser.add_argument(
        "--skip-best-shared",
        action="store_true",
        help="Do not evaluate the best-shared adaptation branch.",
    )
    parser.add_argument(
        "--skip-best-local",
        action="store_true",
        help="Do not evaluate the best-local adaptation branch.",
    )
    parser.add_argument(
        "--holdout-selector",
        default="all",
        help=(
            "Circle-packing holdout selector. Default: all. Ignored for post-hoc OOD "
            "families."
        ),
    )
    parser.add_argument(
        "--ood-task-ids",
        default=None,
        help=(
            "Comma-separated explicit post-hoc OOD task IDs. Ignored for circle_packing. "
            "If omitted, the family resolver default is used."
        ),
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

    include_best_shared = not args.skip_best_shared
    include_best_local = not args.skip_best_local
    if not include_best_shared and not include_best_local:
        raise SystemExit("At least one adaptation branch must be selected for evaluation.")
    env = None
    if args.timeout_override_seconds is not None:
        env = {
            "MT_STS_OOD_TIMEOUT_OVERRIDE_SECONDS": str(
                float(args.timeout_override_seconds)
            )
        }

    if manifest.family == "circle_packing":
        summary = run_circle_packing_adaptation_holdout_update(
            manifest=manifest,
            results_dir=args.results_dir,
            holdout_selector=args.holdout_selector,
            include_best_shared=include_best_shared,
            include_best_local=include_best_local,
            env=env,
        )
        summary_path = Path(args.results_dir).resolve() / "holdout_evaluation" / "holdout_summary.json"
        print(f"Updated holdout summary written to {summary_path}")
        print(f"Holdout keys now include: {', '.join(sorted(summary.keys()))}")
        return 0

    if manifest.family in supported_posthoc_ood_families():
        output_dir = Path(args.results_dir).resolve() / "posthoc_ood_all_known"
        summary = run_posthoc_ood_evaluation(
            manifest=manifest,
            results_dir=args.results_dir,
            ood_task_ids=args.ood_task_ids,
            include_shared=False,
            include_adapted=False,
            include_baselines=False,
            include_best_shared=include_best_shared,
            include_best_local=include_best_local,
            output_dir=output_dir,
            merge_existing=True,
            write_csv=False,
            env=env,
        )
        print(f"Updated post-hoc OOD summary written to {summary['summary_path']}")
        print(f"Program count is now {summary['program_count']}")
        return 0

    supported = ", ".join(["circle_packing", *supported_posthoc_ood_families()])
    raise SystemExit(
        f"Unsupported family '{manifest.family}'. Supported for this updater: {supported}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
