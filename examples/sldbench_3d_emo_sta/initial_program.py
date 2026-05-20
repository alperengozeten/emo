# EVOLVE-BLOCK-START
"""
Generic 3D scaling-law seed for the EMO-STA SLDBench family.

The evaluator always passes `data_points` with shape `(N, 3)` and canonical
columns:
    [model_size_like, diversity_like, total_data_like]

Both tasks are scalar-output, and the parameter budget is at most 7.
"""
import numpy as np
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


def _as_feature_matrix(data_points):
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("data_points must have shape (N, 3)")
    if not np.all(np.isfinite(X)):
        raise ValueError("data_points must be finite")
    return np.maximum(X, 1.0)


def _as_parameter_vector(params):
    p = np.asarray(params, dtype=float)
    if p.ndim == 2:
        if p.shape[0] != 1:
            raise ValueError("params must have shape (P,) or (1, P)")
        p = p.reshape(-1)
    if p.ndim != 1 or p.size != 7:
        raise ValueError("params must contain exactly 7 values")
    if not np.all(np.isfinite(p)):
        raise ValueError("params must be finite")
    return p


def scaling_law_func(data_points, params):
    X = _as_feature_matrix(data_points)
    p = _as_parameter_vector(params)

    coeffs = p[:3]
    exponents = np.clip(p[3:6], -1.0, 2.0)
    bias = float(p[6])

    log_x = np.log(X)
    powered = np.exp(np.clip(-exponents[None, :] * log_x, -60.0, 60.0))
    predictions = bias + np.sum(coeffs[None, :] * powered, axis=1)
    return predictions


def fit_scaling_law(data_points, loss_values):
    X = _as_feature_matrix(data_points)
    y = np.asarray(loss_values, dtype=float).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("loss_values length must match data_points rows")
    if not np.all(np.isfinite(y)):
        raise ValueError("loss_values must be finite")

    base_exponents = np.full(3, 0.1, dtype=float)
    log_x = np.log(X)
    design = np.column_stack(
        [
            np.exp(np.clip(-base_exponents[0] * log_x[:, 0], -60.0, 60.0)),
            np.exp(np.clip(-base_exponents[1] * log_x[:, 1], -60.0, 60.0)),
            np.exp(np.clip(-base_exponents[2] * log_x[:, 2], -60.0, 60.0)),
            np.ones(X.shape[0], dtype=float),
        ]
    )
    linear_solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    init = np.concatenate([linear_solution[:3], base_exponents, [linear_solution[3]]])

    y_scale = max(1.0, float(np.max(np.abs(y))))
    bounds = [
        (-5.0 * y_scale, 5.0 * y_scale),
        (-5.0 * y_scale, 5.0 * y_scale),
        (-5.0 * y_scale, 5.0 * y_scale),
        (-1.0, 2.0),
        (-1.0, 2.0),
        (-1.0, 2.0),
        (float(np.min(y)) - 2.0 * y_scale, float(np.max(y)) + 2.0 * y_scale),
    ]

    def objective(param_vector):
        predictions = scaling_law_func(X, param_vector)
        residual = predictions - y
        mse = np.mean(residual ** 2)
        regularizer = 1.0e-6 * np.sum(np.square(param_vector[:6]))
        return float(mse + regularizer)

    if minimize is not None:
        result = minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1.0e-12},
        )
        if result.success and np.all(np.isfinite(result.x)):
            return np.asarray(result.x, dtype=float)

    best = init.copy()
    best_value = objective(best)
    step_sizes = np.asarray(
        [max(0.1 * (upper - lower), 0.05) for (lower, upper) in bounds],
        dtype=float,
    )
    for _ in range(8):
        improved = False
        for index, (lower, upper) in enumerate(bounds):
            for direction in (-1.0, 1.0):
                candidate = best.copy()
                candidate[index] = np.clip(
                    candidate[index] + direction * step_sizes[index],
                    lower,
                    upper,
                )
                candidate_value = objective(candidate)
                if candidate_value + 1.0e-12 < best_value:
                    best = candidate
                    best_value = candidate_value
                    improved = True
        if not improved:
            step_sizes *= 0.5
        if np.max(step_sizes) < 1.0e-6:
            break
    if np.all(np.isfinite(best)):
        return best
    return init
# EVOLVE-BLOCK-END
