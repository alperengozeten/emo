# EVOLVE-BLOCK-START
"""Generic constructor-style hexagon packing baseline for EMO-STA tasks."""

import math

import numpy as np


def hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    angle0 = math.radians(float(angle_degrees))
    return np.asarray(
        [
            (
                float(center_x) + float(side_length) * math.cos(angle0 + 2.0 * math.pi * k / 6.0),
                float(center_y) + float(side_length) * math.sin(angle0 + 2.0 * math.pi * k / 6.0),
            )
            for k in range(6)
        ],
        dtype=float,
    )


def _point_inside_hexagon(point, center_x, center_y, side_length, angle_degrees, tol=1.0e-9):
    polygon = hexagon_vertices(center_x, center_y, side_length, angle_degrees)
    px, py = float(point[0]), float(point[1])
    for index in range(6):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % 6]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -float(tol):
            return False
    return True


def _axial_to_cartesian(q, r):
    q = float(q)
    r = float(r)
    return np.asarray(
        (
            1.5 * q,
            math.sqrt(3.0) * (r + 0.5 * q),
        ),
        dtype=float,
    )


def make_hex_lattice_centers(n):
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    ring_limit = max(2, int(math.ceil(math.sqrt(max(1, n)))) + 2)
    candidates = []
    for q in range(-ring_limit, ring_limit + 1):
        for r in range(-ring_limit, ring_limit + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) > ring_limit:
                continue
            center = _axial_to_cartesian(q, r)
            candidates.append(
                (
                    float(np.linalg.norm(center)),
                    abs(q) + abs(r) + abs(s),
                    abs(q),
                    abs(r),
                    center,
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4][0], item[4][1]))
    selected = [item[-1] for item in candidates[:n]]
    return np.asarray(selected, dtype=float)


def estimate_safe_outer_side_length(
    inner_hex_data,
    *,
    outer_center=(0.0, 0.0),
    outer_angle_degrees=0.0,
):
    inner_hex_data = np.asarray(inner_hex_data, dtype=float)
    all_vertices = []
    for center_x, center_y, angle_degrees in inner_hex_data:
        all_vertices.append(
            hexagon_vertices(
                center_x,
                center_y,
                1.0,
                angle_degrees,
            )
        )
    all_vertices = np.vstack(all_vertices)

    center_x = float(outer_center[0])
    center_y = float(outer_center[1])
    outer_angle_degrees = float(outer_angle_degrees)

    high = 1.0
    while True:
        if all(
            _point_inside_hexagon(
                vertex,
                center_x,
                center_y,
                high,
                outer_angle_degrees,
                tol=1.0e-9,
            )
            for vertex in all_vertices
        ):
            break
        high *= 2.0

    low = 0.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if all(
            _point_inside_hexagon(
                vertex,
                center_x,
                center_y,
                mid,
                outer_angle_degrees,
                tol=1.0e-9,
            )
            for vertex in all_vertices
        ):
            high = mid
        else:
            low = mid
    return float(high + 1.0e-6)


def construct_initial_lattice_packing(n):
    n = int(n)
    centers = make_hex_lattice_centers(n)
    angles = np.zeros((n, 1), dtype=float)
    inner_hex_data = np.hstack((centers, angles))
    outer_hex_data = np.asarray((0.0, 0.0, 0.0), dtype=float)
    outer_hex_side_length = estimate_safe_outer_side_length(
        inner_hex_data,
        outer_center=outer_hex_data[:2],
        outer_angle_degrees=outer_hex_data[2],
    )
    return inner_hex_data, outer_hex_data, float(outer_hex_side_length)


def construct_hexagon_packing(n: int):
    return construct_initial_lattice_packing(int(n))


def run_hexagon_packing(n: int):
    return construct_hexagon_packing(n)


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    default_n = 11
    _, _, outer_side_length = run_hexagon_packing(default_n)
    print(f"n={default_n} outer_side_length={outer_side_length:.6f}")
