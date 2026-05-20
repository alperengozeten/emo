#!/usr/bin/env python3
"""Rerun only persisted EMO-STA OOD runs that currently contain OOD errors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RESULTS_ROOT = REPO_ROOT / "multi_task_shared_then_adapt" / "results"
CP_HOLDOUT_SCRIPT = (
    REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_circle_packing_holdout_update.py"
)
POSTHOC_SCRIPT = (
    REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_posthoc_ood_evaluation.py"
)
SEED_UPDATE_SCRIPT = (
    REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_adaptation_ood_update.py"
)
DEFAULT_SUMMARY_PATH = RESULTS_ROOT / "rerun_problematic_ood_timeouts_summary.json"
SETTING_RE = re.compile(
    r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-(?P<model>.+)-full$"
)


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    manifest: Path
    results_root: Path
    summary_kind: str
    ood_task_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class OodRepairJob:
    family: str
    setting_name: str
    run_name: str
    manifest: Path
    results_dir: Path
    commands: tuple[tuple[str, ...], ...]
    base_problem: bool
    branch_problem: bool
    branches_complete: bool


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "circle_packing": FamilyConfig(
        family="circle_packing",
        manifest=REPO_ROOT / "multi_task_shared_then_adapt" / "configs" / "circle_packing_emo_sta.yaml",
        results_root=RESULTS_ROOT / "circle_packing",
        summary_kind="holdout",
        ood_task_ids=None,
    ),
    "circle_packing_rectangle": FamilyConfig(
        family="circle_packing_rectangle",
        manifest=REPO_ROOT
        / "multi_task_shared_then_adapt"
        / "configs"
        / "circle_packing_rectangle_emo_sta.yaml",
        results_root=RESULTS_ROOT / "circle_packing_rectangle",
        summary_kind="posthoc",
        ood_task_ids=("cp_rect_n19", "cp_rect_n24", "cp_rect_n25"),
    ),
    "heilbronn_triangle": FamilyConfig(
        family="heilbronn_triangle",
        manifest=REPO_ROOT
        / "multi_task_shared_then_adapt"
        / "configs"
        / "heilbronn_triangle_emo_sta.yaml",
        results_root=RESULTS_ROOT / "heilbronn_triangle",
        summary_kind="posthoc",
        ood_task_ids=("heil_tri_n8", "heil_tri_n13", "heil_tri_n14"),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only b30 EMO-STA runs whose persisted OOD summaries still contain "
            "errors or missing OOD payloads."
        )
    )
    parser.add_argument(
        "--families",
        default="circle_packing,circle_packing_rectangle,heilbronn_triangle",
        help=(
            "Comma-separated families, or 'all'. Default: "
            "circle_packing,circle_packing_rectangle,heilbronn_triangle."
        ),
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=30,
        help="Baseline budget to match in setting names. Default: 30.",
    )
    parser.add_argument(
        "--timeout-override-seconds",
        type=float,
        default=600.0,
        help=(
            "OOD-only evaluator timeout override in seconds. Default: 600."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of run-level repair jobs to execute concurrently. Default: 2.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for repair subprocesses.",
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
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help=(
            "Batch summary JSON path. Default: "
            "multi_task_shared_then_adapt/results/rerun_problematic_ood_timeouts_summary.json."
        ),
    )
    return parser.parse_args(argv)


def parse_families(raw: str) -> list[str]:
    supported = sorted(FAMILY_CONFIGS)
    if raw.strip().lower() == "all":
        return supported
    families = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(families) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported family/families {unknown}; supported: {supported}")
    return families


def _read_json_if_present(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


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


def circle_packing_result_has_error(result: Mapping[str, Any], holdout_task_ids: Sequence[str]) -> bool:
    if result.get("error"):
        return True
    task_results = result.get("holdout_task_results")
    if not isinstance(task_results, Mapping):
        return True
    for holdout_task_id in holdout_task_ids:
        task_result = task_results.get(holdout_task_id)
        if not isinstance(task_result, Mapping):
            return True
        if task_result.get("error"):
            return True
    return False


def circle_packing_mapping_has_error(
    mapping: Mapping[str, Any] | None,
    source_task_ids: Sequence[str],
    holdout_task_ids: Sequence[str],
) -> bool:
    if not isinstance(mapping, Mapping):
        return True
    for source_task_id in source_task_ids:
        result = mapping.get(source_task_id)
        if not isinstance(result, Mapping):
            return True
        if circle_packing_result_has_error(result, holdout_task_ids):
            return True
    return False


def posthoc_result_has_error(result: Mapping[str, Any]) -> bool:
    return bool(result.get("error"))


def posthoc_program_has_errors(
    program_payload: Mapping[str, Any],
    ood_task_ids: Sequence[str],
) -> bool:
    ood_results = program_payload.get("ood_results")
    if not isinstance(ood_results, Mapping):
        return True
    for ood_task_id in ood_task_ids:
        task_result = ood_results.get(ood_task_id)
        if not isinstance(task_result, Mapping):
            return True
        if posthoc_result_has_error(task_result):
            return True
    return False


def circle_packing_problem_flags(
    *,
    run_root: Path,
    source_task_ids: Sequence[str],
    branches_complete: bool,
) -> tuple[bool, bool]:
    summary = _read_json_if_present(
        run_root / "holdout_evaluation" / "holdout_summary.json"
    )
    if not summary:
        return True, bool(branches_complete)

    holdout_task_ids = summary.get("holdout_task_ids")
    if not isinstance(holdout_task_ids, list) or not holdout_task_ids:
        return True, bool(branches_complete)

    shared_result = summary.get("shared_zero_shot")
    base_problem = not isinstance(shared_result, Mapping) or circle_packing_result_has_error(
        shared_result, holdout_task_ids
    )
    if not base_problem:
        base_problem = circle_packing_mapping_has_error(
            summary.get("adaptation_by_source_task"),
            source_task_ids,
            holdout_task_ids,
        ) or circle_packing_mapping_has_error(
            summary.get("baseline_by_source_task"),
            source_task_ids,
            holdout_task_ids,
        )

    branch_problem = False
    if branches_complete:
        branch_problem = circle_packing_mapping_has_error(
            summary.get("best_shared_adaptation_by_source_task"),
            source_task_ids,
            holdout_task_ids,
        ) or circle_packing_mapping_has_error(
            summary.get("best_local_adaptation_by_source_task"),
            source_task_ids,
            holdout_task_ids,
        )
    return base_problem, branch_problem


def posthoc_problem_flags(
    *,
    run_root: Path,
    source_task_ids: Sequence[str],
    ood_task_ids: Sequence[str],
    branches_complete: bool,
) -> tuple[bool, bool]:
    summary = _read_json_if_present(run_root / "posthoc_ood_all_known" / "ood_summary.json")
    if not summary:
        return True, bool(branches_complete)

    programs = summary.get("programs")
    if not isinstance(programs, Mapping):
        return True, bool(branches_complete)

    base_labels = ["shared_best"]
    base_labels.extend(f"adapted__{task_id}" for task_id in source_task_ids)
    base_labels.extend(f"baseline__{task_id}" for task_id in source_task_ids)

    base_problem = False
    for label in base_labels:
        payload = programs.get(label)
        if not isinstance(payload, Mapping) or posthoc_program_has_errors(payload, ood_task_ids):
            base_problem = True
            break

    branch_problem = False
    if branches_complete:
        branch_labels = [f"best_shared__{task_id}" for task_id in source_task_ids]
        branch_labels.extend(f"best_local__{task_id}" for task_id in source_task_ids)
        for label in branch_labels:
            payload = programs.get(label)
            if not isinstance(payload, Mapping) or posthoc_program_has_errors(
                payload, ood_task_ids
            ):
                branch_problem = True
                break

    return base_problem, branch_problem


def build_circle_packing_command(
    *,
    config: FamilyConfig,
    run_root: Path,
    python_executable: str,
    timeout_override_seconds: float,
    rerun_base: bool,
    branches_complete: bool,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(CP_HOLDOUT_SCRIPT),
        "--manifest",
        str(config.manifest),
        "--results-dir",
        str(run_root),
        "--holdout-selector",
        "all",
        "--timeout-override-seconds",
        str(float(timeout_override_seconds)),
    ]
    if not rerun_base:
        command.append("--skip-base-holdout")
    if not branches_complete:
        command.extend(["--skip-best-shared", "--skip-best-local"])
    return tuple(command)


def build_posthoc_commands(
    *,
    config: FamilyConfig,
    run_root: Path,
    python_executable: str,
    timeout_override_seconds: float,
    rerun_base: bool,
    branches_complete: bool,
) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    if rerun_base:
        commands.append(
            (
                python_executable,
                str(POSTHOC_SCRIPT),
                "--manifest",
                str(config.manifest),
                "--results-dir",
                str(run_root),
                "--ood-task-ids",
                ",".join(config.ood_task_ids or ()),
                "--output-dir",
                str(run_root / "posthoc_ood_all_known"),
                "--overwrite",
                "--timeout-override-seconds",
                str(float(timeout_override_seconds)),
            )
        )
    if branches_complete:
        commands.append(
            (
                python_executable,
                str(SEED_UPDATE_SCRIPT),
                "--manifest",
                str(config.manifest),
                "--results-dir",
                str(run_root),
                "--ood-task-ids",
                ",".join(config.ood_task_ids or ()),
                "--timeout-override-seconds",
                str(float(timeout_override_seconds)),
            )
        )
    return tuple(commands)


def discover_jobs(
    *,
    families: Iterable[str],
    baseline_budget: int,
    python_executable: str,
    timeout_override_seconds: float,
) -> list[OodRepairJob]:
    from openevolve.multi_task_shared_then_adapt.workflow import load_manifest, family_task_specs

    jobs: list[OodRepairJob] = []
    for family in families:
        config = FAMILY_CONFIGS[family]
        manifest = load_manifest(config.manifest)
        source_task_ids = [task.task_id for task in family_task_specs(manifest)]

        if not config.results_root.is_dir():
            continue
        for setting_dir in sorted(path for path in config.results_root.iterdir() if path.is_dir()):
            match = SETTING_RE.fullmatch(setting_dir.name)
            if match is None or int(match.group("baseline")) != baseline_budget:
                continue

            for run_dir in sorted(
                path for path in setting_dir.iterdir() if path.is_dir() and path.name.startswith("run_")
            ):
                if not (run_dir / "comparison_summary.json").is_file():
                    continue

                branches_complete = completed_branch_tasks(
                    run_dir,
                    "adaptation_best_shared",
                    source_task_ids,
                ) and completed_branch_tasks(
                    run_dir,
                    "adaptation_best_local",
                    source_task_ids,
                )

                if config.summary_kind == "holdout":
                    base_problem, branch_problem = circle_packing_problem_flags(
                        run_root=run_dir,
                        source_task_ids=source_task_ids,
                        branches_complete=branches_complete,
                    )
                    if not base_problem and not branch_problem:
                        continue
                    commands = (
                        build_circle_packing_command(
                            config=config,
                            run_root=run_dir,
                            python_executable=python_executable,
                            timeout_override_seconds=timeout_override_seconds,
                            rerun_base=base_problem,
                            branches_complete=branches_complete,
                        ),
                    )
                else:
                    base_problem, branch_problem = posthoc_problem_flags(
                        run_root=run_dir,
                        source_task_ids=source_task_ids,
                        ood_task_ids=config.ood_task_ids or (),
                        branches_complete=branches_complete,
                    )
                    if not base_problem and not branch_problem:
                        continue
                    commands = build_posthoc_commands(
                        config=config,
                        run_root=run_dir,
                        python_executable=python_executable,
                        timeout_override_seconds=timeout_override_seconds,
                        rerun_base=base_problem,
                        branches_complete=branches_complete,
                    )
                if not commands:
                    continue
                jobs.append(
                    OodRepairJob(
                        family=family,
                        setting_name=setting_dir.name,
                        run_name=run_dir.name,
                        manifest=config.manifest,
                        results_dir=run_dir,
                        commands=commands,
                        base_problem=base_problem,
                        branch_problem=branch_problem,
                        branches_complete=branches_complete,
                    )
                )
    return jobs


def job_payload(job: OodRepairJob) -> dict[str, object]:
    payload = asdict(job)
    payload["manifest"] = str(job.manifest)
    payload["results_dir"] = str(job.results_dir)
    payload["commands"] = [list(command) for command in job.commands]
    payload["command_texts"] = [shlex.join(command) for command in job.commands]
    return payload


def run_job(job: OodRepairJob) -> dict[str, object]:
    output_root = job.results_dir / "ood_timeout_repair"
    output_root.mkdir(parents=True, exist_ok=True)
    combined_output: list[str] = []

    for index, command in enumerate(job.commands, start=1):
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        output = completed.stdout or completed.stderr or ""
        (output_root / f"step_{index}.log").write_text(output, encoding="utf-8")
        combined_output.append(
            f"$ {shlex.join(command)}\n{output}".strip()
        )
        if completed.returncode != 0:
            return {
                **job_payload(job),
                "status": "failed",
                "returncode": completed.returncode,
                "log_dir": str(output_root),
                "output_tail": "\n\n".join(combined_output)[-8000:],
            }

    return {
        **job_payload(job),
        "status": "succeeded",
        "returncode": 0,
        "log_dir": str(output_root),
        "output_tail": "\n\n".join(combined_output)[-8000:],
    }


def build_summary(
    *,
    args: argparse.Namespace,
    jobs: Sequence[OodRepairJob],
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "algorithm": "rerun_problematic_ood_timeouts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "families": parse_families(args.families),
        "baseline_budget": args.baseline_budget,
        "timeout_override_seconds": float(args.timeout_override_seconds),
        "max_workers": int(args.max_workers),
        "job_count": len(jobs),
        "status_counts": status_counts,
        "jobs": list(results),
    }


def write_summary(path_str: str | Path, summary: Mapping[str, Any]) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = discover_jobs(
        families=parse_families(args.families),
        baseline_budget=args.baseline_budget,
        python_executable=args.python,
        timeout_override_seconds=args.timeout_override_seconds,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.dry_run:
        for job in jobs:
            print(
                f"{job.family}/{job.setting_name}/{job.run_name} "
                f"(base_problem={job.base_problem}, branch_problem={job.branch_problem}, "
                f"branches_complete={job.branches_complete})"
            )
            for command in job.commands:
                print(f"  {shlex.join(command)}")
        print(f"Discovered {len(jobs)} job(s).")
        return 0

    if not jobs:
        print("No problematic OOD runs found.")
        summary_path = write_summary(
            args.summary_path,
            build_summary(args=args, jobs=jobs, results=[]),
        )
        print(f"Summary written to {summary_path}")
        return 0

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            prefix = (
                f"{result['family']}/{result['setting_name']}/{result['run_name']}"
            )
            if result["status"] == "succeeded":
                print(f"[ok] {prefix}")
            else:
                print(f"[failed] {prefix}")
            output_tail = str(result.get("output_tail") or "").strip()
            if output_tail:
                print(output_tail)
                print()

    summary = build_summary(args=args, jobs=jobs, results=results)
    summary_path = write_summary(args.summary_path, summary)
    print(f"Summary written to {summary_path}")
    return 0 if summary["status_counts"].get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
