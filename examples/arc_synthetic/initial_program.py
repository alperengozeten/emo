# EVOLVE-BLOCK-START

import numpy as np


def transform_grid_attempt_1(grid):
    """
    Baseline attempt: preserve the input grid exactly.
    """
    arr = _validate_grid(grid)
    return arr.copy()


def transform_grid_attempt_2(grid):
    """
    Baseline attempt: reverse the row order.
    """
    arr = _validate_grid(grid)
    return np.flipud(arr).astype(np.int32)


# EVOLVE-BLOCK-END


def _validate_grid(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)
    if arr.size and (arr.min() < 0 or arr.max() > 9):
        raise ValueError("Grid values must stay in the inclusive range [0, 9].")
    return arr.astype(np.int32)
