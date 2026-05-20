#!/usr/bin/env python3
"""Plot Best-Local source-task transfer on circle-packing OOD holdouts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "multi_task_shared_then_adapt" / "results" / "circle_packing"
)
DEFAULT_OUTPUT_STEM = (
    REPO_ROOT
    / "multi_task_shared_then_adapt"
    / "figures"
    / "circle_packing_s60_a15_b30_best_local_ood_transfer_heatmap"
)
SETTING_RE = re.compile(
    r"^s(?P<shared>\d+)-a(?P<adapt>\d+)-b(?P<baseline>\d+)-(?P<model>.+)-full$"
)
N_SUFFIX_RE = re.compile(r"_n(?P<n>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a source-task x OOD-holdout heatmap for circle-packing "
            "STA Best-Local adaptation transfer."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Circle-packing results directory.",
    )
    parser.add_argument(
        "--setting-prefix",
        default="s60-a15-b30",
        help="Budget setting prefix to include. Default: s60-a15-b30.",
    )
    parser.add_argument(
        "--output-stem",
        default=str(DEFAULT_OUTPUT_STEM),
        help="Output stem for .png/.pdf/.svg/.json files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG DPI. Default: 300.",
    )
    return parser.parse_args()


def task_n(task_id: str) -> int:
    match = N_SUFFIX_RE.search(task_id)
    if not match:
        raise ValueError(f"Could not parse N from task id: {task_id}")
    return int(match.group("n"))


def task_label(task_id: str) -> str:
    return f"N={task_n(task_id)}"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def load_best_local_scores(
    *,
    results_dir: Path,
    setting_prefix: str,
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, int]],
    list[dict[str, str]],
]:
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    error_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skipped: list[dict[str, str]] = []

    for setting_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        match = SETTING_RE.fullmatch(setting_dir.name)
        if match is None:
            continue
        current_prefix = (
            f"s{match.group('shared')}-a{match.group('adapt')}-b{match.group('baseline')}"
        )
        if current_prefix != setting_prefix:
            continue

        for run_dir in sorted(
            path for path in setting_dir.iterdir() if path.is_dir() and path.name.startswith("run_")
        ):
            summary_path = run_dir / "holdout_evaluation" / "holdout_summary.json"
            if not summary_path.is_file():
                skipped.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "missing holdout_evaluation/holdout_summary.json",
                    }
                )
                continue

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            branch = payload.get("best_local_adaptation_by_source_task")
            if not isinstance(branch, Mapping):
                skipped.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "missing best_local_adaptation_by_source_task",
                    }
                )
                continue

            for source_task_id, source_payload in branch.items():
                if not isinstance(source_payload, Mapping):
                    continue
                holdout_results = source_payload.get("holdout_task_results")
                if not isinstance(holdout_results, Mapping):
                    continue
                for holdout_task_id, result in holdout_results.items():
                    if not isinstance(result, Mapping):
                        continue
                    score = result.get("final_task_score")
                    if is_number(score):
                        scores[str(holdout_task_id)][str(source_task_id)].append(float(score))
                    if result.get("error"):
                        error_counts[str(holdout_task_id)][str(source_task_id)] += 1

    return scores, error_counts, skipped


def summarize_scores(
    scores: Mapping[str, Mapping[str, list[float]]],
    error_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    holdout_ids = sorted(scores, key=task_n)
    source_ids = sorted(
        {source_id for holdout_scores in scores.values() for source_id in holdout_scores},
        key=task_n,
    )
    if not holdout_ids or not source_ids:
        raise ValueError("No Best-Local OOD scores were found.")

    cells: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for holdout_id in holdout_ids:
        cells[holdout_id] = {}
        for source_id in source_ids:
            values = list(scores.get(holdout_id, {}).get(source_id, []))
            cells[holdout_id][source_id] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
                "error_count": int(error_counts.get(holdout_id, {}).get(source_id, 0)),
            }

    return {
        "holdout_task_ids": holdout_ids,
        "source_task_ids": source_ids,
        "cells": cells,
    }


def plot_heatmap(summary: Mapping[str, Any], *, output_stem: Path, dpi: int) -> None:
    holdout_ids = list(summary["holdout_task_ids"])
    source_ids = list(summary["source_task_ids"])
    cells = summary["cells"]

    matrix = np.array(
        [
            [cells[holdout_id][source_id]["mean"] for source_id in source_ids]
            for holdout_id in holdout_ids
        ],
        dtype=float,
    )

    cmap = LinearSegmentedColormap.from_list(
        "sta_transfer_green",
        ["#F4F0E7", "#CFE4D1", "#8CC9A8", "#3F8D72"],
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.88, vmax=0.97, aspect="auto")

    ax.set_xticks(np.arange(len(source_ids)))
    ax.set_xticklabels([task_label(task_id) for task_id in source_ids], fontsize=10)
    ax.set_yticks(np.arange(len(holdout_ids)))
    ax.set_yticklabels([task_label(task_id) for task_id in holdout_ids], fontsize=10)

    ax.set_xlabel(
        "Adaptation Source Task Size",
        fontsize=11.2,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Held-Out Task Size",
        fontsize=11.2,
        fontweight="bold",
        labelpad=8,
    )

    for row_idx, holdout_id in enumerate(holdout_ids):
        row = matrix[row_idx]
        best_col = int(np.nanargmax(row))
        for col_idx, source_id in enumerate(source_ids):
            value = matrix[row_idx, col_idx]
            is_best = col_idx == best_col
            text_color = "white" if is_best else "#173A32"
            weight = "bold" if is_best else "normal"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11.0 if is_best else 10.7,
                weight=weight,
            )
        ax.add_patch(
            Rectangle(
                (best_col - 0.5, row_idx - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#1E4E3F",
                linewidth=2.0,
            )
        )

    ax.set_xticks(np.arange(-0.5, len(source_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(holdout_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(
        "Mean OOD Normalized Score Across LLMs",
        fontsize=9.5,
        fontweight="bold",
        labelpad=8,
    )
    cbar.ax.tick_params(labelsize=9)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.13, right=0.91, bottom=0.17, top=0.98)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_stem = Path(args.output_stem).resolve()

    scores, error_counts, skipped = load_best_local_scores(
        results_dir=results_dir,
        setting_prefix=args.setting_prefix,
    )
    summary = summarize_scores(scores, error_counts)
    summary.update(
        {
            "family": "circle_packing",
            "method": "STA Best-Local",
            "setting_prefix": args.setting_prefix,
            "results_dir": str(results_dir),
            "output_stem": str(output_stem),
            "skipped_runs": skipped,
        }
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_heatmap(summary, output_stem=output_stem, dpi=args.dpi)
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
