"""Data loading utilities for the EMO-STA SLDBench 3D family."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

try:
    import datasets
except ImportError:  # pragma: no cover - exercised through synthetic fixture mode.
    datasets = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt.sldbench_3d import (
    SLDBENCH_3D_TASKS_BY_ID,
    SLDBENCH_DATASET_REPO_ID,
    SLDBench3DTaskSpec,
)


REAL_HF_DATASET_MODE = "real_hf_dataset"
SYNTHETIC_FIXTURE_MODE = "synthetic_fixture"
SLDBENCH_CACHE_DIR_ENV_VAR = "SLDBENCH_CACHE_DIR"
SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR = "SLDBENCH_3D_USE_SYNTHETIC_FIXTURE"

GroupedData = Dict[str, Tuple[np.ndarray, np.ndarray]]
_GROUPED_DATA_CACHE: dict[tuple[str, str, str | None, str], GroupedData] = {}


def get_loader_mode() -> str:
    return (
        SYNTHETIC_FIXTURE_MODE
        if os.getenv(SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR) == "1"
        else REAL_HF_DATASET_MODE
    )


def _require_task(task_id: str) -> SLDBench3DTaskSpec:
    try:
        return SLDBENCH_3D_TASKS_BY_ID[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(SLDBENCH_3D_TASKS_BY_ID))
        raise ValueError(f"Unsupported SLDBench 3D task '{task_id}'. Available: {available}") from exc


def _copy_grouped_data(grouped_data: GroupedData) -> GroupedData:
    return {
        str(group_key): (np.asarray(X, dtype=float).copy(), np.asarray(y, dtype=float).copy())
        for group_key, (X, y) in grouped_data.items()
    }


def canonicalize_feature_columns(
    task_id: str,
    feature_columns: Mapping[str, Sequence[Any]],
) -> np.ndarray:
    """Return canonical `(N, 3)` feature arrays for one supported task."""
    task = _require_task(task_id)
    columns: list[np.ndarray] = []
    expected_length: int | None = None
    for source_name in task.canonical_source_feature_names:
        if source_name not in feature_columns:
            raise KeyError(
                f"Missing feature column '{source_name}' for task '{task_id}'. "
                f"Expected canonical sources: {task.canonical_source_feature_names}"
            )
        column = np.asarray(feature_columns[source_name], dtype=float).reshape(-1)
        if expected_length is None:
            expected_length = int(column.size)
        elif int(column.size) != expected_length:
            raise ValueError(
                f"Inconsistent feature lengths for task '{task_id}': "
                f"expected {expected_length}, got {column.size} for '{source_name}'"
            )
        if not np.all(np.isfinite(column)):
            raise ValueError(
                f"Feature column '{source_name}' for task '{task_id}' must be finite"
            )
        columns.append(column)
    if expected_length is None:
        return np.zeros((0, 3), dtype=float)
    return np.column_stack(columns).astype(float, copy=False)


def _synthetic_scaling_law(data_points: np.ndarray, params: Sequence[float]) -> np.ndarray:
    X = np.maximum(np.asarray(data_points, dtype=float), 1.0)
    params_array = np.asarray(params, dtype=float).reshape(-1)
    coeffs = params_array[:3]
    exponents = np.clip(params_array[3:6], -2.0, 2.0)
    bias = float(params_array[6])
    log_x = np.log(X)
    powered = np.exp(np.clip(-exponents[None, :] * log_x, -60.0, 60.0))
    return bias + np.sum(coeffs[None, :] * powered, axis=1)


def _synthetic_grouped_points(task_id: str) -> dict[str, dict[str, np.ndarray]]:
    if task_id == "vocab_scaling_law":
        return {
            "vocab_group_a": {
                "train": np.asarray(
                    [
                        [3.3e7, 4.1e3, 1.0e8],
                        [6.0e7, 8.2e3, 2.0e8],
                        [1.1e8, 1.6e4, 5.0e8],
                        [2.0e8, 3.2e4, 1.2e9],
                        [3.8e8, 6.5e4, 3.0e9],
                        [6.5e8, 9.6e4, 8.0e9],
                    ],
                    dtype=float,
                ),
                "test": np.asarray(
                    [
                        [8.0e8, 9.6e4, 1.6e10],
                        [9.0e8, 6.5e4, 3.2e10],
                        [1.0e9, 3.2e4, 6.4e10],
                    ],
                    dtype=float,
                ),
                "params": np.asarray([2.0, -1.1, 1.4, 0.08, 0.05, 0.10, 1.35], dtype=float),
            },
            "vocab_group_b": {
                "train": np.asarray(
                    [
                        [4.0e7, 4.1e3, 1.5e8],
                        [7.5e7, 8.2e3, 3.0e8],
                        [1.4e8, 1.6e4, 7.5e8],
                        [2.7e8, 3.2e4, 1.6e9],
                        [5.2e8, 6.5e4, 4.5e9],
                        [8.6e8, 9.6e4, 9.0e9],
                    ],
                    dtype=float,
                ),
                "test": np.asarray(
                    [
                        [9.8e8, 9.6e4, 1.8e10],
                        [1.05e9, 6.5e4, 3.6e10],
                        [1.10e9, 3.2e4, 7.2e10],
                    ],
                    dtype=float,
                ),
                "params": np.asarray([1.7, -0.8, 1.0, 0.07, 0.06, 0.11, 1.55], dtype=float),
            },
        }

    if task_id == "data_constrained_scaling_law":
        return {
            "dc_group_a": {
                "train": np.asarray(
                    [
                        [1.1e8, 1.0e7, 1.0e9],
                        [1.8e8, 2.0e7, 2.0e9],
                        [2.8e8, 4.0e7, 5.0e9],
                        [4.5e8, 8.0e7, 1.2e10],
                        [7.0e8, 1.6e8, 3.0e10],
                        [1.0e9, 3.2e8, 8.0e10],
                    ],
                    dtype=float,
                ),
                "test": np.asarray(
                    [
                        [1.05e9, 3.8e8, 1.6e11],
                        [1.08e9, 4.4e8, 3.2e11],
                        [1.10e9, 5.0e8, 6.4e11],
                    ],
                    dtype=float,
                ),
                "params": np.asarray([2.2, -0.9, 0.8, 0.09, 0.05, 0.08, 1.95], dtype=float),
            },
            "dc_group_b": {
                "train": np.asarray(
                    [
                        [1.2e8, 1.4e7, 1.5e9],
                        [2.0e8, 2.8e7, 3.0e9],
                        [3.2e8, 5.6e7, 7.0e9],
                        [5.0e8, 1.1e8, 1.5e10],
                        [7.8e8, 2.2e8, 3.5e10],
                        [1.05e9, 4.0e8, 8.5e10],
                    ],
                    dtype=float,
                ),
                "test": np.asarray(
                    [
                        [1.08e9, 4.3e8, 1.7e11],
                        [1.10e9, 4.6e8, 3.4e11],
                        [1.12e9, 4.9e8, 6.8e11],
                    ],
                    dtype=float,
                ),
                "params": np.asarray([1.9, -0.7, 1.0, 0.08, 0.04, 0.09, 2.10], dtype=float),
            },
        }

    raise ValueError(f"Unsupported SLDBench 3D task '{task_id}' for synthetic fixture")


def _build_synthetic_grouped_data(task_id: str, split: str) -> GroupedData:
    grouped_points = _synthetic_grouped_points(task_id)
    grouped_data: GroupedData = {}
    for group_key, group_payload in grouped_points.items():
        X = np.asarray(group_payload[split], dtype=float)
        y = _synthetic_scaling_law(X, group_payload["params"])
        grouped_data[str(group_key)] = (X, np.asarray(y, dtype=float).reshape(-1))
    return grouped_data


def _load_real_grouped_data(task: SLDBench3DTaskSpec, split: str, cache_dir: str | None) -> GroupedData:
    if datasets is None:
        raise RuntimeError(
            "The 'datasets' package is required to load the real SLDBench dataset. "
            f"Install it or set {SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR}=1 for local smoke tests."
        )

    try:
        dataset = datasets.load_dataset(
            task.dataset_repo_id,
            name=task.hf_config_name or task.task_id,
            split=split,
            cache_dir=cache_dir,
        )
    except Exception as exc:  # pragma: no cover - depends on external environment.
        cache_msg = f" cache_dir={cache_dir!r}" if cache_dir else ""
        raise RuntimeError(
            f"Failed to load split '{split}' for SLDBench task '{task.task_id}' from "
            f"'{task.dataset_repo_id}'.{cache_msg} Set {SLDBENCH_CACHE_DIR_ENV_VAR} to a "
            "writable directory if the default Hugging Face cache is not writable, or set "
            f"{SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR}=1 for offline smoke tests."
        ) from exc

    required_columns = set(task.original_feature_names) | {task.target_name, "group"}
    missing_columns = sorted(required_columns - set(dataset.column_names))
    if missing_columns:
        raise RuntimeError(
            f"Dataset for task '{task.task_id}' is missing required columns: {missing_columns}"
        )

    grouped_feature_columns: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    grouped_targets: dict[str, list[Any]] = defaultdict(list)

    groups = dataset["group"]
    target_values = dataset[task.target_name]
    original_feature_columns = {
        feature_name: dataset[feature_name] for feature_name in task.original_feature_names
    }

    for index, raw_group in enumerate(groups):
        group_key = str(raw_group)
        for feature_name, values in original_feature_columns.items():
            grouped_feature_columns[group_key][feature_name].append(values[index])
        grouped_targets[group_key].append(target_values[index])

    grouped_data: GroupedData = {}
    for group_key in sorted(grouped_feature_columns):
        X = canonicalize_feature_columns(task.task_id, grouped_feature_columns[group_key])
        y = np.asarray(grouped_targets[group_key], dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"Task '{task.task_id}' group '{group_key}' has mismatched feature/target rows: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )
        if not np.all(np.isfinite(y)):
            raise RuntimeError(
                f"Task '{task.task_id}' group '{group_key}' contains non-finite targets"
            )
        grouped_data[group_key] = (X, y)
    return grouped_data


def load_grouped_data(task_id: str, split: str) -> GroupedData:
    """Load grouped `(X, y)` arrays for one EMO-STA SLDBench 3D task and split."""
    normalized_split = str(split).strip().lower()
    if normalized_split not in {"train", "test"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")

    task = _require_task(task_id)
    cache_dir = os.getenv(SLDBENCH_CACHE_DIR_ENV_VAR)
    loader_mode = get_loader_mode()
    cache_key = (task.task_id, normalized_split, cache_dir, loader_mode)

    if cache_key not in _GROUPED_DATA_CACHE:
        if loader_mode == SYNTHETIC_FIXTURE_MODE:
            grouped_data = _build_synthetic_grouped_data(task.task_id, normalized_split)
        else:
            grouped_data = _load_real_grouped_data(task, normalized_split, cache_dir)
        _GROUPED_DATA_CACHE[cache_key] = grouped_data

    return _copy_grouped_data(_GROUPED_DATA_CACHE[cache_key])


__all__ = [
    "REAL_HF_DATASET_MODE",
    "SLDBENCH_CACHE_DIR_ENV_VAR",
    "SLDBENCH_SYNTHETIC_FIXTURE_ENV_VAR",
    "SYNTHETIC_FIXTURE_MODE",
    "canonicalize_feature_columns",
    "get_loader_mode",
    "load_grouped_data",
]
