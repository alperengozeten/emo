# EVOLVE-BLOCK-START
"""Generic constructor-style circle packing for EMO-STA circle-packing tasks."""

import numpy as np


def _unique_points(points):
    seen = set()
    unique = []
    for point in points:
        key = (round(float(point[0]), 12), round(float(point[1]), 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(np.asarray(key, dtype=float))
    return unique


def _candidate_pool(n):
    margin = 0.08
    points = []

    points.append((0.5, 0.5))

    for radius, count in ((0.24, 6), (0.37, 12)):
        for index in range(count):
            angle = 2.0 * np.pi * index / count
            points.append(
                (
                    0.5 + radius * np.cos(angle),
                    0.5 + radius * np.sin(angle),
                )
            )

    points.extend(
        [
            (0.15, 0.15),
            (0.85, 0.15),
            (0.15, 0.85),
            (0.85, 0.85),
            (0.50, 0.12),
            (0.50, 0.88),
            (0.12, 0.50),
            (0.88, 0.50),
        ]
    )

    grid_side = max(6, int(np.ceil(np.sqrt(2 * max(1, n)))) + 2)
    xs = np.linspace(margin, 1.0 - margin, grid_side)
    ys = np.linspace(margin, 1.0 - margin, grid_side)
    spacing = xs[1] - xs[0] if grid_side > 1 else 0.0
    for row_index, y in enumerate(ys):
        offset = 0.5 * spacing if row_index % 2 else 0.0
        for x in xs:
            shifted_x = np.clip(x + offset, margin, 1.0 - margin)
            points.append((shifted_x, y))

    return _unique_points(points)


def _initialize_centers(n):
    if n <= 0:
        raise ValueError("n must be positive")

    selected = [np.asarray((0.5, 0.5), dtype=float)]
    first_ring_count = min(6, max(0, n - 1))
    first_ring_radius = 0.24
    for index in range(first_ring_count):
        angle = 2.0 * np.pi * index / max(1, first_ring_count)
        selected.append(
            np.asarray(
                (
                    0.5 + first_ring_radius * np.cos(angle),
                    0.5 + first_ring_radius * np.sin(angle),
                ),
                dtype=float,
            )
        )

    if len(selected) >= n:
        return np.asarray(selected[:n], dtype=float)

    pool = _candidate_pool(n)
    while len(selected) < n:
        selected_array = np.asarray(selected, dtype=float)
        best_point = None
        best_score = -1.0
        for point in pool:
            distances = np.linalg.norm(selected_array - point, axis=1)
            min_distance = float(np.min(distances))
            if min_distance < 1.0e-9:
                continue
            border_margin = float(
                min(point[0], point[1], 1.0 - point[0], 1.0 - point[1])
            )
            score = min_distance + 0.15 * border_margin
            if score > best_score:
                best_score = score
                best_point = point

        if best_point is None:
            break
        selected.append(np.asarray(best_point, dtype=float))

    if len(selected) < n:
        raise RuntimeError(f"Could not initialize {n} distinct circle centers")

    return np.asarray(selected[:n], dtype=float)


def compute_max_radii(centers):
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    radii = np.zeros(n, dtype=float)

    for index in range(n):
        x, y = centers[index]
        radii[index] = min(x, y, 1.0 - x, 1.0 - y)

    for _ in range(2):
        for i in range(n):
            radii[i] = min(
                radii[i],
                centers[i, 0],
                centers[i, 1],
                1.0 - centers[i, 0],
                1.0 - centers[i, 1],
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


def construct_packing(n: int):
    n = int(n)
    centers = _initialize_centers(n)
    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def run_packing(n: int):
    return construct_packing(n)


# EVOLVE-BLOCK-END


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    for index, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.45)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(index), ha="center", va="center", fontsize=8)

    ax.set_title(f"Circle Packing (n={len(centers)}, sum={float(np.sum(radii)):.6f})")
    plt.show()


if __name__ == "__main__":
    default_n = 26
    centers, radii, sum_radii = run_packing(default_n)
    print(f"n={default_n} sum_radii={sum_radii:.6f}")
