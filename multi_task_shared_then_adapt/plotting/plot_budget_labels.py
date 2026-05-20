"""Shared helpers for budget labels used in EMO-STA figures."""

from __future__ import annotations


def compute_total_budget(
    *,
    shared: int,
    adapt: int,
    baseline: int,
    task_count: int,
) -> int:
    if task_count <= 0:
        raise ValueError(f"task_count must be positive, got {task_count}")

    emo_sta_total = shared + task_count * adapt
    single_task_total = task_count * baseline
    if emo_sta_total != single_task_total:
        raise ValueError(
            "Inconsistent EMO-STA budget: "
            f"shared={shared}, adapt={adapt}, baseline={baseline}, "
            f"task_count={task_count} gives shared+K*adapt={emo_sta_total} "
            f"but K*baseline={single_task_total}"
        )
    return emo_sta_total


def build_budget_triplet(
    *,
    shared: int,
    adapt: int,
    baseline: int,
    task_count: int,
) -> dict[str, int | str]:
    total = compute_total_budget(
        shared=shared,
        adapt=adapt,
        baseline=baseline,
        task_count=task_count,
    )
    return {
        "shared": shared,
        "adapt": adapt,
        "baseline": baseline,
        "task_count": task_count,
        "total": total,
        "label": f"{shared} / {adapt} / {total}",
    }


def build_single_task_budget_triplet(
    *,
    baseline: int,
    task_count: int,
) -> dict[str, int | str]:
    total = task_count * baseline
    return {
        "shared": 0,
        "adapt": baseline,
        "baseline": baseline,
        "task_count": task_count,
        "total": total,
        "label": f"0 / {baseline} / {total}",
    }


def budget_axis_label() -> str:
    return "Shared / Per-Task Adaptation / Total Iterations"
