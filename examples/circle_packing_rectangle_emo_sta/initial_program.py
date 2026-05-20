# EVOLVE-BLOCK-START
"""Generic constructor-style rectangle circle packing for EMO-STA tasks."""

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - optional runtime acceleration
    linprog = None


_ALPHA_CANDIDATES = (0.72, 0.80, 0.88, 0.96, 1.00)
_CENTER_EPS = 1.0e-6


def repair_centers_into_rectangle(centers, alpha, height):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (n, 2)")

    n = centers.shape[0]
    if n == 0:
        return centers.copy()

    x_margin = min(0.08 * alpha, 0.18 * alpha)
    y_margin = min(0.08 * height, 0.18 * height)
    lower = np.array([max(_CENTER_EPS, x_margin), max(_CENTER_EPS, y_margin)], dtype=float)
    upper = np.array(
        [
            max(lower[0], alpha - lower[0]),
            max(lower[1], height - lower[1]),
        ],
        dtype=float,
    )

    repaired = np.clip(centers, lower, upper)
    if upper[0] <= lower[0]:
        repaired[:, 0] = 0.5 * alpha
    if upper[1] <= lower[1]:
        repaired[:, 1] = 0.5 * height
    return repaired


def _append_unique_points(points, candidates, n):
    seen = {
        (round(float(point[0]), 12), round(float(point[1]), 12))
        for point in points
    }
    for candidate in candidates:
        key = (round(float(candidate[0]), 12), round(float(candidate[1]), 12))
        if key in seen:
            continue
        seen.add(key)
        points.append(np.asarray(key, dtype=float))
        if len(points) >= n:
            break


def make_staggered_grid_centers(n, alpha, height):
    if n <= 0:
        raise ValueError("n must be positive")

    aspect = alpha / max(height, _CENTER_EPS)
    cols = max(1, int(np.ceil(np.sqrt(max(1, n) * aspect))))
    rows = max(1, int(np.ceil(n / cols)))

    x_margin = min(0.08 * alpha, 0.18 * alpha)
    y_margin = min(0.08 * height, 0.12 * height)
    xs = (
        np.linspace(x_margin, alpha - x_margin, cols)
        if cols > 1
        else np.array([0.5 * alpha], dtype=float)
    )
    ys = (
        np.linspace(y_margin, height - y_margin, rows)
        if rows > 1
        else np.array([0.5 * height], dtype=float)
    )

    points = []
    step_x = float(xs[1] - xs[0]) if cols > 1 else 0.0
    for row_index, y in enumerate(ys):
        if cols > 1 and row_index % 2 == 1:
            row_xs = np.linspace(
                x_margin + 0.5 * step_x,
                alpha - x_margin - 0.5 * step_x,
                cols,
            )
        else:
            row_xs = xs
        for x in row_xs:
            points.append((float(x), float(y)))

    centers = np.asarray(points[:n], dtype=float)
    if centers.shape[0] < n:
        raise RuntimeError(f"Could not build {n} staggered-grid centers")
    return repair_centers_into_rectangle(centers, alpha, height)


def make_ring_grid_hybrid_centers(n, alpha, height):
    center = np.array([0.5 * alpha, 0.5 * height], dtype=float)
    points = [center]

    ring_count = min(n - 1, max(6, int(np.ceil(0.4 * n)))) if n > 1 else 0
    ring_radius_x = 0.30 * alpha
    ring_radius_y = 0.24 * height
    for index in range(ring_count):
        angle = 2.0 * np.pi * index / max(1, ring_count)
        points.append(
            np.asarray(
                [
                    center[0] + ring_radius_x * np.cos(angle),
                    center[1] + ring_radius_y * np.sin(angle),
                ],
                dtype=float,
            )
        )

    grid_points = make_staggered_grid_centers(max(n, 2 * ring_count + 1), alpha, height)
    _append_unique_points(points, grid_points, n)
    centers = np.asarray(points[:n], dtype=float)
    return repair_centers_into_rectangle(centers, alpha, height)


def make_corner_biased_centers(n, alpha, height):
    points = []
    x0 = 0.16 * alpha
    x1 = alpha - x0
    y0 = 0.10 * height
    y1 = height - y0
    points.extend(
        [
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
            (0.5 * alpha, y0),
            (0.5 * alpha, y1),
            (x0, 0.5 * height),
            (x1, 0.5 * height),
            (0.5 * alpha, 0.5 * height),
        ]
    )
    grid_points = make_staggered_grid_centers(max(n, 2 * len(points)), alpha, height)
    _append_unique_points(points, grid_points, n)
    centers = np.asarray(points[:n], dtype=float)
    return repair_centers_into_rectangle(centers, alpha, height)


def compute_max_radii_rect(centers, alpha):
    centers = np.asarray(centers, dtype=float)
    alpha = float(alpha)
    height = 2.0 - alpha
    n = centers.shape[0]

    if centers.shape != (n, 2):
        raise ValueError("centers must have shape (n, 2)")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if height <= 0.0:
        raise ValueError("height must be positive")

    centers = repair_centers_into_rectangle(centers, alpha, height)
    boundary_limits = np.minimum.reduce(
        [
            centers[:, 0],
            alpha - centers[:, 0],
            centers[:, 1],
            height - centers[:, 1],
        ]
    )
    boundary_limits = np.maximum(boundary_limits, 0.0)

    if n == 0:
        return np.zeros(0, dtype=float)

    constraint_rows = []
    constraint_rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            constraint_rows.append(row)
            constraint_rhs.append(float(np.linalg.norm(centers[i] - centers[j])))

    result = None
    if linprog is not None:
        try:
            result = linprog(
                c=-np.ones(n, dtype=float),
                A_ub=np.asarray(constraint_rows, dtype=float) if constraint_rows else None,
                b_ub=np.asarray(constraint_rhs, dtype=float) if constraint_rhs else None,
                bounds=[(0.0, float(limit)) for limit in boundary_limits],
                method="highs",
            )
        except Exception:
            result = None

    if result is not None and result.success and result.x is not None:
        radii = np.asarray(result.x, dtype=float)
        return np.maximum(radii, 0.0)

    radii = boundary_limits.copy()
    for _ in range(4):
        for i in range(n):
            radii[i] = min(
                radii[i],
                centers[i, 0],
                alpha - centers[i, 0],
                centers[i, 1],
                height - centers[i, 1],
            )
        for i in range(n):
            for j in range(i + 1, n):
                distance = float(np.linalg.norm(centers[i] - centers[j]))
                total_radius = float(radii[i] + radii[j])
                if distance <= 0.0:
                    radii[i] = 0.0
                    radii[j] = 0.0
                    continue
                if total_radius > distance:
                    scale = distance / total_radius
                    radii[i] *= scale
                    radii[j] *= scale
    return np.maximum(radii, 0.0)


def _candidate_layouts(n, alpha, height):
    return (
        make_staggered_grid_centers(n, alpha, height),
        make_ring_grid_hybrid_centers(n, alpha, height),
        make_corner_biased_centers(n, alpha, height),
    )


def construct_packing(n: int):
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    best = None
    best_key = None
    for alpha in _ALPHA_CANDIDATES:
        height = 2.0 - alpha
        for centers in _candidate_layouts(n, alpha, height):
            repaired_centers = repair_centers_into_rectangle(centers, alpha, height)
            radii = compute_max_radii_rect(repaired_centers, alpha)
            sum_radii = float(np.sum(radii))
            key = (sum_radii, -abs(alpha - 0.9))
            if best is None or key > best_key:
                best = (
                    repaired_centers,
                    radii,
                    float(alpha),
                    sum_radii,
                )
                best_key = key

    if best is None:
        raise RuntimeError("Could not construct a rectangle circle packing")
    return best


def run_packing(n: int):
    return construct_packing(n)


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    default_n = 23
    centers, radii, alpha, sum_radii = run_packing(default_n)
    print(
        f"n={default_n} alpha={alpha:.6f} height={2.0 - alpha:.6f} "
        f"sum_radii={sum_radii:.6f}"
    )
