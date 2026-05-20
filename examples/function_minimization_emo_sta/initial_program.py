# EVOLVE-BLOCK-START
"""Generic derivative-free 2D minimization baseline for EMO-STA."""

import numpy as np


def search_algorithm(objective_fn, bounds, iterations=1000, seed=0):
    """
    Search for a low-value point for a generic 2D objective.

    Args:
        objective_fn: Callable with signature objective_fn(x, y) -> float.
        bounds: ((x_min, x_max), (y_min, y_max)) search box.
        iterations: Number of objective evaluations to spend.
        seed: Deterministic random seed for the search trajectory.

    Returns:
        Tuple of (best_x, best_y, best_value).
    """
    rng = np.random.default_rng(seed)
    (x_min, x_max), (y_min, y_max) = bounds
    total_iterations = max(1, int(iterations))

    best_x = float(rng.uniform(x_min, x_max))
    best_y = float(rng.uniform(y_min, y_max))
    best_value = float(objective_fn(best_x, best_y))

    for _ in range(total_iterations - 1):
        candidate_x = float(rng.uniform(x_min, x_max))
        candidate_y = float(rng.uniform(y_min, y_max))
        candidate_value = float(objective_fn(candidate_x, candidate_y))
        if candidate_value < best_value:
            best_x = candidate_x
            best_y = candidate_y
            best_value = candidate_value

    return best_x, best_y, best_value


def run_search(objective_fn, bounds, iterations=1000, seed=0):
    """Entry point preferred by the evaluator."""
    return search_algorithm(objective_fn, bounds, iterations=iterations, seed=seed)


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from openevolve.multi_task_shared_then_adapt.function_minimization import (
        objective_sincosxy,
    )

    result = run_search(objective_sincosxy, ((-5.0, 5.0), (-5.0, 5.0)))
    print(result)
