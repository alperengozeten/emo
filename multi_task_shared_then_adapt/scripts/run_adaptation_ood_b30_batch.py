#!/usr/bin/env python3
"""Persist adaptation OOD results for completed b30 EMO-STA runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_RUN_SCRIPT = (
    REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_adaptation_ood_update.py"
)
RESULTS_ROOT = REPO_ROOT / "multi_task_shared_then_adapt" / "results"

SETTING_RE = re.compile(
    r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-(?P<model>.+)-full$"
)


@dataclass(frozen=True)
class BatchJob:
    family: str
    setting_name: str
    run_name: str
    manifest: Path
    results_dir: Path
    ood_task_ids: tuple[str, ...] | None = None


FAMILY_OOD_TASK_IDS: dict[str, tuple[str, ...] | None] = {
    "circle_packing": None,
    "circle_packing_rectangle": ("cp_rect_n19", "cp_rect_n24", "cp_rect_n25"),
    "heilbronn_triangle": ("heil_tri_n8", "heil_tri_n13", "heil_tri_n14"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add adaptation OOD results to existing run-local JSON summaries for "
            "the selected families and baseline budget."
        )
    )
    parser.add_argument(
        "--families",
        default="circle_packing,circle_packing_rectangle,heilbronn_triangle",
        help="Comma-separated families, or 'all'. Default: circle_packing,circle_packing_rectangle,heilbronn_triangle.",
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=30,
        help="Baseline budget to match in setting names. Default: 30.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of run-level updates to execute concurrently. Default: 2.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for per-run updater subprocesses.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on discovered jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the jobs that would run without executing them.",
    )
    return parser.parse_args(argv)


def parse_families(raw: str) -> list[str]:
    supported = ["circle_packing", "circle_packing_rectangle", "heilbronn_triangle"]
    if raw.strip().lower() == "all":
        return supported
    families = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(families) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported family/families {unknown}; supported: {supported}")
    return families


def task_complete(task_dir: Path) -> bool:
    if not task_dir.is_dir():
        return False
    direct = [
        "best/best_program_info.json",
        "best/best_program.py",
        "best/best_program.r",
        "best/best_program.rs",
        "best_program_info.json",
        "best_program.py",
        "best_program.r",
        "best_program.rs",
    ]
    for rel in direct:
        if (task_dir / rel).exists():
            return True
    if (task_dir / "best").is_dir() and any((task_dir / "best").glob("best_program*")):
        return True
    if any(task_dir.glob("checkpoint_*/best_program_info.json")):
        return True
    return False


def completed_branch_tasks(run_root: Path, branch_root_name: str, task_ids: Sequence[str]) -> bool:
    branch_root = run_root / branch_root_name
    if not branch_root.is_dir():
        return False
    return all(task_complete(branch_root / task_id) for task_id in task_ids)


def already_updated(run_root: Path, family: str, task_ids: Sequence[str]) -> bool:
    if family == "circle_packing":
        summary_path = run_root / "holdout_evaluation" / "holdout_summary.json"
        if not summary_path.is_file():
            return False
        import json

        payload = json.loads(summary_path.read_text())
        return (
            isinstance(payload.get("best_shared_adaptation_by_source_task"), dict)
            and isinstance(payload.get("best_local_adaptation_by_source_task"), dict)
        )

    summary_path = run_root / "posthoc_ood_all_known" / "ood_summary.json"
    if not summary_path.is_file():
        return False
    import json

    payload = json.loads(summary_path.read_text())
    programs = payload.get("programs", {})
    return all(f"best_shared__{task_id}" in programs for task_id in task_ids) and all(
        f"best_local__{task_id}" in programs for task_id in task_ids
    )


def discover_jobs(
    *,
    families: Iterable[str],
    baseline_budget: int,
) -> list[BatchJob]:
    from openevolve.multi_task_shared_then_adapt.workflow import load_manifest, family_task_specs

    jobs: list[BatchJob] = []
    for family in families:
        manifest = REPO_ROOT / "multi_task_shared_then_adapt" / "configs" / f"{family}_emo_sta.yaml"
        manifest_obj = load_manifest(manifest)
        task_ids = [task.task_id for task in family_task_specs(manifest_obj)]
        family_root = RESULTS_ROOT / family
        if not family_root.is_dir():
            continue

        for setting_dir in sorted(path for path in family_root.iterdir() if path.is_dir()):
            match = SETTING_RE.fullmatch(setting_dir.name)
            if match is None or int(match.group("baseline")) != baseline_budget:
                continue

            for run_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir() and path.name.startswith("run_")):
                if not (run_dir / "comparison_summary.json").is_file():
                    continue
                if not completed_branch_tasks(run_dir, "adaptation_best_shared", task_ids):
                    continue
                if not completed_branch_tasks(run_dir, "adaptation_best_local", task_ids):
                    continue
                if family == "circle_packing":
                    if not (run_dir / "holdout_evaluation" / "holdout_summary.json").is_file():
                        continue
                else:
                    if not (run_dir / "posthoc_ood_all_known" / "ood_summary.json").is_file():
                        continue
                if already_updated(run_dir, family, task_ids):
                    continue
                jobs.append(
                    BatchJob(
                        family=family,
                        setting_name=setting_dir.name,
                        run_name=run_dir.name,
                        manifest=manifest,
                        results_dir=run_dir,
                        ood_task_ids=FAMILY_OOD_TASK_IDS.get(family),
                    )
                )
    return jobs


def run_job(job: BatchJob, *, python_executable: str) -> tuple[BatchJob, int, str]:
    command = [
        python_executable,
        str(SINGLE_RUN_SCRIPT),
        "--manifest",
        str(job.manifest),
        "--results-dir",
        str(job.results_dir),
    ]
    if job.ood_task_ids:
        command.extend(["--ood-task-ids", ",".join(job.ood_task_ids)])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    output = completed.stdout if completed.returncode == 0 else completed.stderr or completed.stdout
    return job, completed.returncode, output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = discover_jobs(
        families=parse_families(args.families),
        baseline_budget=args.baseline_budget,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.dry_run:
        for job in jobs:
            print(f"{job.family}/{job.setting_name}/{job.run_name}")
        print(f"Discovered {len(jobs)} job(s).")
        return 0

    if not jobs:
        print("No eligible completed adaptation-branch runs found to update.")
        return 0

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(run_job, job, python_executable=args.python)
            for job in jobs
        ]
        for future in as_completed(futures):
            job, returncode, output = future.result()
            prefix = f"{job.family}/{job.setting_name}/{job.run_name}"
            if returncode == 0:
                print(f"[ok] {prefix}")
            else:
                failures += 1
                print(f"[failed] {prefix}")
            if output.strip():
                print(output.strip())
                print()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
