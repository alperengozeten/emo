#!/usr/bin/env python3
"""Launch post-hoc OOD evaluation for every finished b30 EMO-STA run."""

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
import time
from typing import Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_RUN_SCRIPT = REPO_ROOT / "multi_task_shared_then_adapt" / "scripts" / "run_posthoc_ood_evaluation.py"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "multi_task_shared_then_adapt" / "results"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_ROOT / "posthoc_ood_b30_batch_summary.json"
DEFAULT_OUTPUT_SUBDIR = "posthoc_ood_all_known"

SETTING_RE = re.compile(
    r"^s(?P<shared>\d+)-a(?P<adapted>\d+)-b(?P<baseline>\d+)-(?P<model>.+)-full$"
)


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    results_name: str
    manifest: Path
    ood_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class OodBatchJob:
    family: str
    setting_name: str
    setting_prefix: str
    model: str
    run_name: str
    manifest: Path
    results_dir: Path
    output_dir: Path
    ood_task_ids: tuple[str, ...]
    skip_reason: str | None = None


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "heilbronn_triangle": FamilyConfig(
        family="heilbronn_triangle",
        results_name="heilbronn_triangle",
        manifest=REPO_ROOT
        / "multi_task_shared_then_adapt"
        / "configs"
        / "heilbronn_triangle_emo_sta.yaml",
        ood_task_ids=("heil_tri_n8", "heil_tri_n13", "heil_tri_n14"),
    ),
    "circle_packing_rectangle": FamilyConfig(
        family="circle_packing_rectangle",
        results_name="circle_packing_rectangle",
        manifest=REPO_ROOT
        / "multi_task_shared_then_adapt"
        / "configs"
        / "circle_packing_rectangle_emo_sta.yaml",
        ood_task_ids=("cp_rect_n19", "cp_rect_n24", "cp_rect_n25"),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-hoc all-known OOD evaluation for every finished EMO-STA run "
            "whose setting directory has the selected baseline budget, default b30."
        )
    )
    parser.add_argument(
        "--families",
        default="heilbronn_triangle,circle_packing_rectangle",
        help=(
            "Comma-separated supported families, or 'all'. Default: "
            "heilbronn_triangle,circle_packing_rectangle."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=(
            "Root containing family result directories. Default: "
            f"{DEFAULT_RESULTS_ROOT.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--baseline-budget",
        type=int,
        default=30,
        help="Baseline budget to match in setting directory names. Default: 30.",
    )
    parser.add_argument(
        "--setting-prefix",
        action="append",
        dest="setting_prefixes",
        help=(
            "Optional setting prefix to include, e.g. s20-a25-b30. May be passed "
            "multiple times. If omitted, all settings matching --baseline-budget are used."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help=(
            "Per-run output subdirectory for OOD summaries. Default: "
            f"{DEFAULT_OUTPUT_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of run-level evaluations to execute concurrently. Default: 2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of discovered jobs to execute, useful for smoke tests.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for per-run evaluator subprocesses.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help=(
            "Batch summary JSON path. Default: "
            f"{DEFAULT_SUMMARY_PATH.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--job-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for each run-level evaluator subprocess.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run jobs even if the per-run OOD summary already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and summarize the jobs that would be launched without running them.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Start this batch in the background with stdout/stderr redirected to "
            "--log-file, then exit."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Log file used by --detach. Defaults to "
            "multi_task_shared_then_adapt/results/posthoc_ood_b30_batch_<timestamp>.log."
        ),
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def parse_families(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return sorted(FAMILY_CONFIGS)
    families = [item.strip() for item in raw.split(",") if item.strip()]
    if not families:
        raise ValueError("At least one family must be selected")
    unknown = sorted(set(families) - set(FAMILY_CONFIGS))
    if unknown:
        supported = ", ".join(sorted(FAMILY_CONFIGS))
        raise ValueError(f"Unsupported family/families {unknown}; supported: {supported}")
    return families


def parse_setting_name(setting_name: str) -> dict[str, object] | None:
    match = SETTING_RE.fullmatch(setting_name)
    if not match:
        return None
    return {
        "shared": int(match.group("shared")),
        "adapted": int(match.group("adapted")),
        "baseline": int(match.group("baseline")),
        "model": match.group("model"),
        "setting_prefix": (
            f"s{match.group('shared')}-a{match.group('adapted')}-b{match.group('baseline')}"
        ),
    }


def discover_jobs(
    *,
    results_root: str | Path,
    families: Iterable[str],
    baseline_budget: int = 30,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    setting_prefixes: Sequence[str] | None = None,
    overwrite: bool = False,
) -> list[OodBatchJob]:
    root = resolve_repo_path(results_root)
    selected_prefixes = set(setting_prefixes or [])
    jobs: list[OodBatchJob] = []

    for family in families:
        config = FAMILY_CONFIGS[family]
        family_root = root / config.results_name
        if not family_root.is_dir():
            continue

        for setting_dir in sorted(path for path in family_root.iterdir() if path.is_dir()):
            parsed = parse_setting_name(setting_dir.name)
            if parsed is None:
                continue
            if parsed["baseline"] != baseline_budget:
                continue
            setting_prefix = str(parsed["setting_prefix"])
            if selected_prefixes and setting_prefix not in selected_prefixes:
                continue

            for run_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir()):
                if not (run_dir / "comparison_summary.json").is_file():
                    continue
                output_dir = run_dir / output_subdir
                summary_path = output_dir / "ood_summary.json"
                csv_path = output_dir / "ood_summary.csv"
                skip_reason = None
                if not overwrite and (summary_path.exists() or csv_path.exists()):
                    skip_reason = "existing_outputs"
                jobs.append(
                    OodBatchJob(
                        family=family,
                        setting_name=setting_dir.name,
                        setting_prefix=setting_prefix,
                        model=str(parsed["model"]),
                        run_name=run_dir.name,
                        manifest=config.manifest,
                        results_dir=run_dir,
                        output_dir=output_dir,
                        ood_task_ids=config.ood_task_ids,
                        skip_reason=skip_reason,
                    )
                )

    return sorted(jobs, key=job_sort_key)


def job_sort_key(job: OodBatchJob) -> tuple[str, str, str]:
    return (job.family, job.setting_name, job.run_name)


def build_job_command(
    job: OodBatchJob,
    *,
    python_executable: str,
    overwrite: bool,
) -> list[str]:
    command = [
        python_executable,
        str(SINGLE_RUN_SCRIPT),
        "--manifest",
        str(job.manifest),
        "--results-dir",
        str(job.results_dir),
        "--ood-task-ids",
        ",".join(job.ood_task_ids),
        "--output-dir",
        str(job.output_dir),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _tail(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_job_subprocess(
    job: OodBatchJob,
    *,
    python_executable: str,
    overwrite: bool,
    timeout_seconds: float | None,
) -> dict[str, object]:
    command = build_job_command(
        job,
        python_executable=python_executable,
        overwrite=overwrite,
    )
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "batch_eval.log"
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.monotonic() - start
        output = completed.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        return {
            **job_payload(job),
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "command": command,
            "command_text": shlex.join(command),
            "log_path": str(log_path),
            "output_tail": _tail(output),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        message = f"Timed out after {timeout_seconds} seconds\n{output}"
        log_path.write_text(message, encoding="utf-8")
        return {
            **job_payload(job),
            "status": "failed",
            "returncode": None,
            "elapsed_seconds": elapsed,
            "command": command,
            "command_text": shlex.join(command),
            "log_path": str(log_path),
            "output_tail": _tail(message),
        }


def job_payload(job: OodBatchJob) -> dict[str, object]:
    payload = asdict(job)
    for key in ("manifest", "results_dir", "output_dir"):
        payload[key] = str(payload[key])
    return payload


Runner = Callable[
    [OodBatchJob, str, bool, float | None],
    dict[str, object],
]


def execute_jobs(
    jobs: Sequence[OodBatchJob],
    *,
    python_executable: str,
    overwrite: bool,
    max_workers: int,
    timeout_seconds: float | None,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> list[dict[str, object]]:
    if max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    results: list[dict[str, object]] = []
    runnable_jobs: list[OodBatchJob] = []

    for job in jobs:
        command = build_job_command(
            job,
            python_executable=python_executable,
            overwrite=overwrite,
        )
        if job.skip_reason:
            results.append(
                {
                    **job_payload(job),
                    "status": "skipped",
                    "command": command,
                    "command_text": shlex.join(command),
                }
            )
        elif dry_run:
            results.append(
                {
                    **job_payload(job),
                    "status": "dry_run",
                    "command": command,
                    "command_text": shlex.join(command),
                }
            )
        else:
            runnable_jobs.append(job)

    if not runnable_jobs:
        return sorted(results, key=result_sort_key)

    actual_runner = runner or (
        lambda job, python, ow, timeout: run_job_subprocess(
            job,
            python_executable=python,
            overwrite=ow,
            timeout_seconds=timeout,
        )
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                actual_runner,
                job,
                python_executable,
                overwrite,
                timeout_seconds,
            ): job
            for job in runnable_jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard.
                result = {
                    **job_payload(job),
                    "status": "failed",
                    "returncode": None,
                    "elapsed_seconds": None,
                    "command": build_job_command(
                        job,
                        python_executable=python_executable,
                        overwrite=overwrite,
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                result["command_text"] = shlex.join(result["command"])
            results.append(result)

    return sorted(results, key=result_sort_key)


def result_sort_key(result: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(result.get("family") or ""),
        str(result.get("setting_name") or ""),
        str(result.get("run_name") or ""),
        str(result.get("status") or ""),
    )


def build_batch_summary(
    *,
    jobs: Sequence[OodBatchJob],
    results: Sequence[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "algorithm": "posthoc_ood_b30_batch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_budget": args.baseline_budget,
        "families": parse_families(args.families),
        "results_root": str(resolve_repo_path(args.results_root)),
        "output_subdir": args.output_subdir,
        "max_workers": args.max_workers,
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "job_count": len(jobs),
        "status_counts": status_counts,
        "jobs": list(results),
    }


def write_summary(summary_path: str | Path, summary: dict[str, object]) -> Path:
    path = resolve_repo_path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def default_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RESULTS_ROOT / f"posthoc_ood_b30_batch_{timestamp}.log"


def build_detach_command(argv: Sequence[str]) -> list[str]:
    command: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--detach":
            continue
        command.append(item)
    return command


def detach_self(argv: Sequence[str], *, log_file: str | Path | None) -> int:
    command = build_detach_command(argv)
    log_path = resolve_repo_path(log_file) if log_file else default_log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    print(f"Started detached post-hoc OOD b30 batch: pid={process.pid}")
    print(f"Log file: {log_path}")
    print(f"Summary path will be written by the child process.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.detach:
        detach_argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            *(list(argv) if argv is not None else sys.argv[1:]),
        ]
        return detach_self(detach_argv, log_file=args.log_file)

    try:
        families = parse_families(args.families)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    jobs = discover_jobs(
        results_root=args.results_root,
        families=families,
        baseline_budget=args.baseline_budget,
        output_subdir=args.output_subdir,
        setting_prefixes=args.setting_prefixes,
        overwrite=args.overwrite,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(
        f"Discovered {len(jobs)} finished run(s) for b{args.baseline_budget} "
        f"across {', '.join(families)}."
    )
    skipped = sum(1 for job in jobs if job.skip_reason)
    if skipped:
        print(f"Skipping {skipped} run(s) with existing OOD outputs.")
    if args.dry_run:
        print("Dry run only; no evaluations will be launched.")

    for job in jobs:
        status = f"skip:{job.skip_reason}" if job.skip_reason else "run"
        print(
            f"[{status}] {job.family} {job.setting_name}/{job.run_name} -> "
            f"{job.output_dir}"
        )

    results = execute_jobs(
        jobs,
        python_executable=args.python,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        timeout_seconds=args.job_timeout_seconds,
        dry_run=args.dry_run,
    )

    summary = build_batch_summary(jobs=jobs, results=results, args=args)
    summary_path = write_summary(args.summary_path, summary)

    print(f"Batch summary written to {summary_path}")
    print(
        "Status counts: "
        + ", ".join(
            f"{status}={count}" for status, count in sorted(summary["status_counts"].items())
        )
    )

    failed = summary["status_counts"].get("failed", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
