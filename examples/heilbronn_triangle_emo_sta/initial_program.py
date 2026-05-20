# EVOLVE-BLOCK-START
"""Generic constructor-style Heilbronn triangle baseline for EMO-STA tasks."""

import numpy as np


def triangle_vertices():
    return np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def barycentric_to_cartesian(l1, l2, l3):
    bary = np.asarray([l1, l2, l3], dtype=float)
    total = float(np.sum(bary))
    if total <= 0.0:
        raise ValueError("Barycentric coordinates must have positive sum")
    bary = bary / total
    vertices = triangle_vertices()
    return bary[0] * vertices[0] + bary[1] * vertices[1] + bary[2] * vertices[2]


def compute_min_triangle_area(points):
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    if n_points < 3:
        return 0.0

    min_area = float("inf")
    for i in range(n_points - 2):
        for j in range(i + 1, n_points - 1):
            edge = points[j] - points[i]
            for k in range(j + 1, n_points):
                area = 0.5 * abs(
                    edge[0] * (points[k, 1] - points[i, 1])
                    - edge[1] * (points[k, 0] - points[i, 0])
                )
                if area < min_area:
                    min_area = float(area)
    return 0.0 if not np.isfinite(min_area) else float(min_area)


def _min_pair_distance(points):
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0
    min_distance = float("inf")
    for i in range(n_points - 1):
        for j in range(i + 1, n_points):
            distance = float(np.linalg.norm(points[i] - points[j]))
            if distance < min_distance:
                min_distance = distance
    return 0.0 if not np.isfinite(min_distance) else float(min_distance)


def _unique_barycentric_rows(rows):
    seen = set()
    unique = []
    for row in rows:
        bary = np.asarray(row, dtype=float)
        bary = bary / float(np.sum(bary))
        key = tuple(round(float(value), 12) for value in bary)
        if key in seen:
            continue
        seen.add(key)
        unique.append(bary)
    return unique


def _candidate_barycentric_pool(n):
    samples = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ]

    edge_sample_count = max(5, n)
    for epsilon in (0.03, 0.05, 0.08):
        for t in np.linspace(0.12, 0.88, edge_sample_count):
            a = (1.0 - epsilon) * t
            b = (1.0 - epsilon) * (1.0 - t)
            samples.append((epsilon, a, b))
            samples.append((a, epsilon, b))
            samples.append((a, b, epsilon))

    for denominator in range(4, 10):
        for i in range(1, denominator - 1):
            for j in range(1, denominator - i):
                k = denominator - i - j
                if k <= 0:
                    continue
                bary = np.asarray((i, j, k), dtype=float) / float(denominator)
                samples.append(tuple(bary))
                samples.append((bary[1], bary[2], bary[0]))
                samples.append((bary[2], bary[0], bary[1]))

    return _unique_barycentric_rows(samples)


def initial_barycentric_pattern(n):
    n = int(n)
    if n < 3:
        raise ValueError("n must be at least 3")

    pool = _candidate_barycentric_pool(n)
    selected = [
        np.asarray((1.0, 0.0, 0.0), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
        np.asarray((0.0, 0.0, 1.0), dtype=float),
    ]

    while len(selected) < n:
        selected_points = np.asarray(
            [barycentric_to_cartesian(*bary) for bary in selected],
            dtype=float,
        )
        best_candidate = None
        best_score = None
        for candidate in pool:
            candidate_point = barycentric_to_cartesian(*candidate)
            min_distance = float(
                np.min(np.linalg.norm(selected_points - candidate_point, axis=1))
            )
            if min_distance <= 1.0e-9:
                continue
            boundary_bonus = 1.0 - 3.0 * float(np.min(candidate))
            centroid_distance = float(
                np.linalg.norm(candidate_point - np.mean(selected_points, axis=0))
            )
            score = min_distance + 0.08 * boundary_bonus + 0.03 * centroid_distance
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = np.asarray(candidate, dtype=float)

        if best_candidate is None:
            break
        selected.append(best_candidate)

    if len(selected) < n:
        raise RuntimeError(f"Could not initialize {n} Heilbronn points")
    return np.asarray(selected[:n], dtype=float)


def _local_barycentric_moves(bary):
    bary = np.asarray(bary, dtype=float)
    proposals = []
    for delta in (0.02, 0.04, 0.06):
        for source_index in range(3):
            for target_index in range(3):
                if source_index == target_index or bary[source_index] <= delta:
                    continue
                moved = bary.copy()
                moved[source_index] -= delta
                moved[target_index] += delta
                if np.min(moved) >= 0.0:
                    proposals.append(moved / float(np.sum(moved)))
    return _unique_barycentric_rows(proposals)


def _improve_points(bary_points, passes=2):
    bary_points = np.asarray(bary_points, dtype=float)
    pool = _candidate_barycentric_pool(len(bary_points))
    points = np.asarray([barycentric_to_cartesian(*bary) for bary in bary_points], dtype=float)
    best_area = compute_min_triangle_area(points)
    best_pair_distance = _min_pair_distance(points)

    for _ in range(max(0, int(passes))):
        improved = False
        for index in range(len(points)):
            local_candidates = pool + _local_barycentric_moves(bary_points[index])
            best_local_bary = None
            best_local_point = None
            best_local_area = best_area
            best_local_pair_distance = best_pair_distance
            for candidate_bary in local_candidates:
                candidate_point = barycentric_to_cartesian(*candidate_bary)
                if any(
                    np.linalg.norm(candidate_point - points[other_index]) <= 1.0e-9
                    for other_index in range(len(points))
                    if other_index != index
                ):
                    continue

                trial_points = points.copy()
                trial_points[index] = candidate_point
                trial_area = compute_min_triangle_area(trial_points)
                trial_pair_distance = _min_pair_distance(trial_points)
                improved_area = trial_area > best_local_area + 1.0e-12
                improved_pair_distance = (
                    abs(trial_area - best_local_area) <= 1.0e-12
                    and trial_pair_distance > best_local_pair_distance + 1.0e-12
                )
                if improved_area or improved_pair_distance:
                    best_local_bary = np.asarray(candidate_bary, dtype=float)
                    best_local_point = candidate_point
                    best_local_area = float(trial_area)
                    best_local_pair_distance = float(trial_pair_distance)

            if best_local_bary is not None and best_local_point is not None:
                bary_points[index] = best_local_bary
                points[index] = best_local_point
                best_area = best_local_area
                best_pair_distance = best_local_pair_distance
                improved = True
        if not improved:
            break

    return points, float(best_area)


def construct_points(n: int):
    n = int(n)
    bary_points = initial_barycentric_pattern(n)
    points, min_area = _improve_points(bary_points, passes=3)
    return np.asarray(points, dtype=float), float(min_area)


def run_heilbronn(n: int):
    return construct_points(n)


# EVOLVE-BLOCK-END


def visualize(points):
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=float)
    vertices = triangle_vertices()

    fig, ax = plt.subplots(figsize=(8, 5))
    polygon = np.vstack([vertices, vertices[0]])
    ax.plot(polygon[:, 0], polygon[:, 1], color="black")
    ax.scatter(points[:, 0], points[:, 1], color="#c13b22", zorder=3)
    for index, point in enumerate(points):
        ax.text(point[0], point[1], str(index), fontsize=8, ha="left", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Heilbronn Triangle (n={len(points)})")
    plt.show()


if __name__ == "__main__":
    default_n = 12
    points, min_area = run_heilbronn(default_n)
    print(f"n={default_n} min_triangle_area={min_area:.8f}")
