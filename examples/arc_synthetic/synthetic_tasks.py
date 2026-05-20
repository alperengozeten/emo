"""Deterministic synthetic ARC-style tasks used by the multitask prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


GridTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ArcExample:
    """One input-output grid pair."""

    input_grid: np.ndarray
    output_grid: np.ndarray


@dataclass(frozen=True)
class ArcSyntheticTask:
    """A synthetic ARC-style task family member."""

    task_id: str
    title: str
    transformation_summary: str
    train_examples: List[ArcExample]
    heldout_examples: List[ArcExample]
    transform: GridTransform


def _grid(rows: List[List[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int32)


def rotate_90_cw(grid: np.ndarray) -> np.ndarray:
    """Rotate a grid 90 degrees clockwise."""
    return np.rot90(np.asarray(grid, dtype=np.int32), -1).astype(np.int32)


def shift_right_wrap(grid: np.ndarray) -> np.ndarray:
    """Shift every row right by one cell with wraparound."""
    return np.roll(np.asarray(grid, dtype=np.int32), 1, axis=1).astype(np.int32)


def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Mirror a grid left-to-right."""
    return np.fliplr(np.asarray(grid, dtype=np.int32)).astype(np.int32)


def _build_examples(
    transform: GridTransform,
    raw_inputs: List[List[List[int]]],
) -> List[ArcExample]:
    return [
        ArcExample(input_grid=_grid(rows), output_grid=transform(_grid(rows)))
        for rows in raw_inputs
    ]


ROTATE_TRAIN_INPUTS = [
    [[1, 2, 0], [3, 4, 5]],
    [[6, 0, 2], [1, 7, 3], [4, 5, 8]],
    [[9, 1, 2, 3], [4, 0, 5, 6], [7, 8, 2, 1]],
]
ROTATE_HELDOUT_INPUTS = [
    [[2, 4], [5, 7], [8, 1], [3, 6]],
]

SHIFT_TRAIN_INPUTS = [
    [[1, 2, 3, 4], [5, 6, 7, 8]],
    [[9, 0, 1], [2, 3, 4], [5, 6, 7]],
    [[8, 1, 4, 2, 0], [3, 5, 7, 9, 6]],
]
SHIFT_HELDOUT_INPUTS = [
    [[4, 2, 8], [1, 9, 3], [7, 5, 6], [0, 4, 2]],
]

FLIP_TRAIN_INPUTS = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 0, 2, 9], [3, 8, 4, 1]],
    [[5, 1], [2, 7], [9, 3]],
]
FLIP_HELDOUT_INPUTS = [
    [[6, 4, 2, 0, 8], [1, 3, 5, 7, 9]],
]


TASK_SPECS: Dict[str, ArcSyntheticTask] = {
    "rotate_90_cw": ArcSyntheticTask(
        task_id="rotate_90_cw",
        title="ARC Synthetic Rotate 90 CW",
        transformation_summary="Rotate the entire grid 90 degrees clockwise.",
        train_examples=_build_examples(rotate_90_cw, ROTATE_TRAIN_INPUTS),
        heldout_examples=_build_examples(rotate_90_cw, ROTATE_HELDOUT_INPUTS),
        transform=rotate_90_cw,
    ),
    "shift_right_wrap": ArcSyntheticTask(
        task_id="shift_right_wrap",
        title="ARC Synthetic Shift Right Wrap",
        transformation_summary="Shift every row one cell to the right with wraparound.",
        train_examples=_build_examples(shift_right_wrap, SHIFT_TRAIN_INPUTS),
        heldout_examples=_build_examples(shift_right_wrap, SHIFT_HELDOUT_INPUTS),
        transform=shift_right_wrap,
    ),
    "flip_horizontal": ArcSyntheticTask(
        task_id="flip_horizontal",
        title="ARC Synthetic Flip Horizontal",
        transformation_summary="Mirror the grid left-to-right.",
        train_examples=_build_examples(flip_horizontal, FLIP_TRAIN_INPUTS),
        heldout_examples=_build_examples(flip_horizontal, FLIP_HELDOUT_INPUTS),
        transform=flip_horizontal,
    ),
}


def get_task_spec(task_id: str) -> ArcSyntheticTask:
    """Return the task spec for one synthetic ARC task."""
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_SPECS))
        raise KeyError(f"Unknown ARC synthetic task '{task_id}'. Known tasks: {known}") from exc


def list_task_ids() -> List[str]:
    """Return task ids in stable order."""
    return list(TASK_SPECS.keys())
