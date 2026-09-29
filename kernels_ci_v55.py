"""Compiled and optimized point-by-point construction kernel for CI."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from operator import index as integer_index
from time import perf_counter
from execution_v55 import BackendExecutionError, current_runtime

try:
    import numba
    from numba import njit
    from numba.core.registry import CPUDispatcher
except Exception as exc:
    # Dependency errors must stop startup, never expose a Python-loop fallback.
    raise BackendExecutionError(
        f"CI_NUMBA_IMPORT_FAILED:{type(exc).__name__}:{exc}"
    ) from exc

NUMBA_AVAILABLE = True
if numba.config.DISABLE_JIT:
    raise BackendExecutionError("CI_NUMBA_JIT_DISABLED")


def _unit_vec(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Thực thi _unit_vec."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    denom = max(n, eps)
    return np.array([v[0] / denom, v[1] / denom, v[2] / denom], dtype=np.float64)


@njit(cache=True, nogil=True, fastmath=False, parallel=False, error_model="numpy")
def point_by_point_numba_core(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    seed_index: int,
    eps_parallel: float,
    eps_unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thực thi point_by_point_numba_core."""
    n_rays = len(starts)
    used = np.zeros(n_rays, dtype=np.bool_)
    order = np.empty(n_rays, dtype=np.int64)
    points = np.empty((n_rays, 3), dtype=np.float64)
    normals = np.empty((n_rays, 3), dtype=np.float64)
    parent = np.full(n_rays, -1, dtype=np.int64)
    distance = np.full(n_rays, np.nan, dtype=np.float64)

    current_ray = int(seed_index)
    points[0, 0] = P0[current_ray, 0]
    points[0, 1] = P0[current_ray, 1]
    points[0, 2] = P0[current_ray, 2]

    # inc = unit(points[0] - starts[current_ray])
    inc_v0 = points[0, 0] - starts[current_ray, 0]
    inc_v1 = points[0, 1] - starts[current_ray, 1]
    inc_v2 = points[0, 2] - starts[current_ray, 2]
    inc_n = math.sqrt(inc_v0 * inc_v0 + inc_v1 * inc_v1 + inc_v2 * inc_v2)
    inc_d = max(inc_n, eps_unit)
    inc_0 = inc_v0 / inc_d
    inc_1 = inc_v1 / inc_d
    inc_2 = inc_v2 / inc_d

    # out = unit(targets[current_ray] - points[0])
    out_v0 = targets[current_ray, 0] - points[0, 0]
    out_v1 = targets[current_ray, 1] - points[0, 1]
    out_v2 = targets[current_ray, 2] - points[0, 2]
    out_n = math.sqrt(out_v0 * out_v0 + out_v1 * out_v1 + out_v2 * out_v2)
    out_d = max(out_n, eps_unit)
    out_0 = out_v0 / out_d
    out_1 = out_v1 / out_d
    out_2 = out_v2 / out_d

    # normals[0] = unit(inc - out)
    diff_0 = inc_0 - out_0
    diff_1 = inc_1 - out_1
    diff_2 = inc_2 - out_2
    diff_n = math.sqrt(diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
    diff_d = max(diff_n, eps_unit)
    normals[0, 0] = diff_0 / diff_d
    normals[0, 1] = diff_1 / diff_d
    normals[0, 2] = diff_2 / diff_d

    order[0] = current_ray
    used[current_ray] = True

    for k in range(1, n_rays):
        ncur_0 = normals[k - 1, 0]
        ncur_1 = normals[k - 1, 1]
        ncur_2 = normals[k - 1, 2]

        pcur_0 = points[k - 1, 0]
        pcur_1 = points[k - 1, 1]
        pcur_2 = points[k - 1, 2]

        # Scan remaining rays
        best_d = math.inf
        pick_local_ray = -1
        best_q0 = 0.0
        best_q1 = 0.0
        best_q2 = 0.0
        best_parent = k - 1

        first_remaining = -1

        for r in range(n_rays):
            if used[r]:
                continue
            if first_remaining < 0:
                first_remaining = r

            dr_0 = directions[r, 0]
            dr_1 = directions[r, 1]
            dr_2 = directions[r, 2]

            den = dr_0 * ncur_0 + dr_1 * ncur_1 + dr_2 * ncur_2
            if abs(den) > eps_parallel:
                st_0 = starts[r, 0]
                st_1 = starts[r, 1]
                st_2 = starts[r, 2]

                num = (pcur_0 - st_0) * ncur_0 + (pcur_1 - st_1) * ncur_1 + (pcur_2 - st_2) * ncur_2
                t = num / den
                if t > 1e-6 and math.isfinite(t):
                    qr_0 = st_0 + t * dr_0
                    qr_1 = st_1 + t * dr_1
                    qr_2 = st_2 + t * dr_2

                    dist_sq = (qr_0 - pcur_0) ** 2 + (qr_1 - pcur_1) ** 2 + (qr_2 - pcur_2) ** 2
                    d = math.sqrt(dist_sq)
                    if d < best_d:
                        best_d = d
                        pick_local_ray = r
                        best_q0 = qr_0
                        best_q1 = qr_1
                        best_q2 = qr_2

        ray = pick_local_ray
        # Match NumPy argmin(all-inf): still try the first remaining ray
        # against older tangent planes before using the seed fallback.
        if ray < 0:
            ray = first_remaining

        if k >= 2 and ray >= 0:
            dir_ray_0 = directions[ray, 0]
            dir_ray_1 = directions[ray, 1]
            dir_ray_2 = directions[ray, 2]

            st_ray_0 = starts[ray, 0]
            st_ray_1 = starts[ray, 1]
            st_ray_2 = starts[ray, 2]

            for j in range(k - 1):
                nj_0 = normals[j, 0]
                nj_1 = normals[j, 1]
                nj_2 = normals[j, 2]

                den_old = nj_0 * dir_ray_0 + nj_1 * dir_ray_1 + nj_2 * dir_ray_2
                if abs(den_old) > eps_parallel:
                    pj_0 = points[j, 0]
                    pj_1 = points[j, 1]
                    pj_2 = points[j, 2]

                    num_old = (pj_0 - st_ray_0) * nj_0 + (pj_1 - st_ray_1) * nj_1 + (pj_2 - st_ray_2) * nj_2
                    t_old = num_old / den_old
                    if t_old > 1e-6 and math.isfinite(t_old):
                        qo_0 = st_ray_0 + t_old * dir_ray_0
                        qo_1 = st_ray_1 + t_old * dir_ray_1
                        qo_2 = st_ray_2 + t_old * dir_ray_2

                        d_old = math.sqrt((qo_0 - pj_0) ** 2 + (qo_1 - pj_1) ** 2 + (qo_2 - pj_2) ** 2)
                        if d_old < best_d:
                            best_d = d_old
                            best_q0 = qo_0
                            best_q1 = qo_1
                            best_q2 = qo_2
                            best_parent = j

        if not math.isfinite(best_d) or ray < 0:
            ray = first_remaining
            best_q0 = P0[ray, 0]
            best_q1 = P0[ray, 1]
            best_q2 = P0[ray, 2]
            best_d = math.inf
            best_parent = -2

        points[k, 0] = best_q0
        points[k, 1] = best_q1
        points[k, 2] = best_q2

        inc_v0 = points[k, 0] - starts[ray, 0]
        inc_v1 = points[k, 1] - starts[ray, 1]
        inc_v2 = points[k, 2] - starts[ray, 2]
        inc_n = math.sqrt(inc_v0 * inc_v0 + inc_v1 * inc_v1 + inc_v2 * inc_v2)
        inc_d = max(inc_n, eps_unit)
        inc_0 = inc_v0 / inc_d
        inc_1 = inc_v1 / inc_d
        inc_2 = inc_v2 / inc_d

        out_v0 = targets[ray, 0] - points[k, 0]
        out_v1 = targets[ray, 1] - points[k, 1]
        out_v2 = targets[ray, 2] - points[k, 2]
        out_n = math.sqrt(out_v0 * out_v0 + out_v1 * out_v1 + out_v2 * out_v2)
        out_d = max(out_n, eps_unit)
        out_0 = out_v0 / out_d
        out_1 = out_v1 / out_d
        out_2 = out_v2 / out_d

        diff_0 = inc_0 - out_0
        diff_1 = inc_1 - out_1
        diff_2 = inc_2 - out_2
        diff_n = math.sqrt(diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
        diff_d = max(diff_n, eps_unit)
        n0 = diff_0 / diff_d
        n1 = diff_1 / diff_d
        n2 = diff_2 / diff_d

        if (n0 * normals[k - 1, 0] + n1 * normals[k - 1, 1] + n2 * normals[k - 1, 2]) < 0.0:
            n0 = -n0
            n1 = -n1
            n2 = -n2

        normals[k, 0] = n0
        normals[k, 1] = n1
        normals[k, 2] = n2

        order[k] = ray
        parent[k] = best_parent
        distance[k] = best_d
        used[ray] = True

    return points, normals, order, parent, distance


@njit(
    cache=True,
    nogil=True,
    fastmath=False,
    error_model="numpy",
)
def _ci_pair_surface_compatible(
    q0: float,
    q1: float,
    q2: float,
    nq0: float,
    nq1: float,
    nq2: float,
    p0: float,
    p1: float,
    p2: float,
    nr0: float,
    nr1: float,
    nr2: float,
    normal_angle_floor_rad: float,
    normal_angle_per_mm_rad: float,
    normal_angle_cap_rad: float,
    edge_height_residual_cap_mm: float,
    maximum_separation_mm: float,
    enforce_maximum_separation: bool,
) -> tuple[bool, float, float, float]:
    """Kiểm tra tính tương thích hình học bề mặt giữa hai điểm và hai pháp tuyến theo góc và độ lệch biên."""
    normal_dot = (
        nq0 * nr0
        + nq1 * nr1
        + nq2 * nr2
    )

    if normal_dot < 0.0:
        nr0 = -nr0
        nr1 = -nr1
        nr2 = -nr2
        normal_dot = -normal_dot

    if normal_dot > 1.0:
        normal_dot = 1.0
    elif normal_dot < -1.0:
        normal_dot = -1.0

    dx = q0 - p0
    dy = q1 - p1
    dz = q2 - p2

    separation = math.sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    )

    allowed_angle = (
        normal_angle_floor_rad
        + normal_angle_per_mm_rad
        * separation
    )

    if allowed_angle > normal_angle_cap_rad:
        allowed_angle = normal_angle_cap_rad

    edge_residual = abs(
        0.5 * (
            dx * (nq0 + nr0)
            + dy * (nq1 + nr1)
            + dz * (nq2 + nr2)
        )
    )

    compatible = (
        normal_dot >= math.cos(allowed_angle)
        and
        edge_residual
        <= edge_height_residual_cap_mm
    )

    if enforce_maximum_separation:
        compatible = (
            compatible
            and separation <= maximum_separation_mm
        )

    return (
        compatible,
        math.degrees(math.acos(normal_dot)),
        edge_residual,
        separation,
    )


@njit(
    cache=True,
    nogil=True,
    fastmath=False,
    parallel=False,
    error_model="numpy",
)
def point_by_point_compatible_numba_core(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    N0: np.ndarray,
    seed_index: int,
    eps_parallel: float,
    eps_unit: float,
    minimum_candidate_radius_mm: float,
    candidate_distance_factor: float,
    maximum_candidate_radius_mm: float,
    normal_angle_floor_rad: float,
    normal_angle_per_mm_rad: float,
    normal_angle_cap_rad: float,
    edge_height_residual_cap_mm: float,
    secondary_neighbor_gate_enabled: bool,
):
    """Thực thi point_by_point_compatible_numba_core."""
    n_rays = len(starts)

    used = np.zeros(
        n_rays,
        dtype=np.bool_,
    )

    order = np.empty(
        n_rays,
        dtype=np.int64,
    )

    points = np.empty(
        (n_rays, 3),
        dtype=np.float64,
    )

    normals = np.empty(
        (n_rays, 3),
        dtype=np.float64,
    )

    parent = np.full(
        n_rays,
        -1,
        dtype=np.int64,
    )

    distance = np.full(
        n_rays,
        np.nan,
        dtype=np.float64,
    )

    front_id = np.full(
        n_rays,
        -1,
        dtype=np.int64,
    )

    restart_reason_code = np.zeros(
        n_rays,
        dtype=np.int64,
    )

    parent_normal_angle_deg = np.full(
        n_rays,
        np.nan,
        dtype=np.float64,
    )

    local_candidate_count = np.zeros(
        n_rays,
        dtype=np.int64,
    )

    compatible_candidate_count = np.zeros(
        n_rays,
        dtype=np.int64,
    )

    candidate_t = np.full(
        n_rays,
        np.nan,
        dtype=np.float64,
    )

    candidate_distance = np.full(
        n_rays,
        np.inf,
        dtype=np.float64,
    )

    current_ray = int(seed_index)

    points[0, 0] = P0[current_ray, 0]
    points[0, 1] = P0[current_ray, 1]
    points[0, 2] = P0[current_ray, 2]

    inc_v0 = points[0, 0] - starts[current_ray, 0]
    inc_v1 = points[0, 1] - starts[current_ray, 1]
    inc_v2 = points[0, 2] - starts[current_ray, 2]
    inc_n = math.sqrt(inc_v0 * inc_v0 + inc_v1 * inc_v1 + inc_v2 * inc_v2)
    inc_d = max(inc_n, eps_unit)
    inc_0 = inc_v0 / inc_d
    inc_1 = inc_v1 / inc_d
    inc_2 = inc_v2 / inc_d

    out_v0 = targets[current_ray, 0] - points[0, 0]
    out_v1 = targets[current_ray, 1] - points[0, 1]
    out_v2 = targets[current_ray, 2] - points[0, 2]
    out_n = math.sqrt(out_v0 * out_v0 + out_v1 * out_v1 + out_v2 * out_v2)
    out_d = max(out_n, eps_unit)
    out_0 = out_v0 / out_d
    out_1 = out_v1 / out_d
    out_2 = out_v2 / out_d

    diff_0 = inc_0 - out_0
    diff_1 = inc_1 - out_1
    diff_2 = inc_2 - out_2
    diff_n = math.sqrt(diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
    diff_d = max(diff_n, eps_unit)
    n0 = diff_0 / diff_d
    n1 = diff_1 / diff_d
    n2 = diff_2 / diff_d

    seed_dot = (
        n0 * N0[current_ray, 0]
        + n1 * N0[current_ray, 1]
        + n2 * N0[current_ray, 2]
    )

    if seed_dot < 0.0:
        n0 = -n0
        n1 = -n1
        n2 = -n2

    normals[0, 0] = n0
    normals[0, 1] = n1
    normals[0, 2] = n2

    order[0] = current_ray
    used[current_ray] = True
    front_id[0] = 0

    active_front_id = 0
    front_start_order = 0

    for k in range(1, n_rays):
        ncur_0 = normals[k - 1, 0]
        ncur_1 = normals[k - 1, 1]
        ncur_2 = normals[k - 1, 2]

        pcur_0 = points[k - 1, 0]
        pcur_1 = points[k - 1, 1]
        pcur_2 = points[k - 1, 2]

        current_ray = order[k - 1]

        minimum_distance = math.inf
        first_remaining = -1

        restart_ray = -1
        restart_seed_distance = math.inf

        # PASS A - Geometry only
        for r in range(n_rays):
            if used[r]:
                candidate_t[r] = math.nan
                candidate_distance[r] = math.inf
                continue

            if first_remaining < 0:
                first_remaining = r

            dx0 = P0[r, 0] - P0[current_ray, 0]
            dy0 = P0[r, 1] - P0[current_ray, 1]
            dz0 = P0[r, 2] - P0[current_ray, 2]
            seed_distance = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
            if seed_distance < restart_seed_distance:
                restart_seed_distance = seed_distance
                restart_ray = r

            dr_0 = directions[r, 0]
            dr_1 = directions[r, 1]
            dr_2 = directions[r, 2]

            den = dr_0 * ncur_0 + dr_1 * ncur_1 + dr_2 * ncur_2
            if abs(den) > eps_parallel:
                st_0 = starts[r, 0]
                st_1 = starts[r, 1]
                st_2 = starts[r, 2]

                num = (pcur_0 - st_0) * ncur_0 + (pcur_1 - st_1) * ncur_1 + (pcur_2 - st_2) * ncur_2
                t = num / den
                if t > 1e-6 and math.isfinite(t):
                    qr_0 = st_0 + t * dr_0
                    qr_1 = st_1 + t * dr_1
                    qr_2 = st_2 + t * dr_2

                    dist_sq = (qr_0 - pcur_0) ** 2 + (qr_1 - pcur_1) ** 2 + (qr_2 - pcur_2) ** 2
                    d = math.sqrt(dist_sq)

                    candidate_t[r] = t
                    candidate_distance[r] = d
                    if d < minimum_distance:
                        minimum_distance = d
                else:
                    candidate_t[r] = math.nan
                    candidate_distance[r] = math.inf
            else:
                candidate_t[r] = math.nan
                candidate_distance[r] = math.inf

        # Window determination
        if math.isfinite(minimum_distance):
            local_radius = max(
                minimum_candidate_radius_mm,
                candidate_distance_factor * minimum_distance,
            )
            local_radius = min(
                maximum_candidate_radius_mm,
                local_radius,
            )
        else:
            local_radius = minimum_candidate_radius_mm

        # PASS B - Normal compatibility only for local candidates
        best_compatible_ray = -1
        best_compatible_distance = math.inf
        best_q0 = 0.0
        best_q1 = 0.0
        best_q2 = 0.0
        best_n0 = 0.0
        best_n1 = 0.0
        best_n2 = 0.0

        for r in range(n_rays):
            if used[r]:
                continue
            cand_d = candidate_distance[r]
            if not math.isfinite(cand_d) or cand_d > local_radius:
                continue

            local_candidate_count[k] += 1
            t = candidate_t[r]
            qr_0 = starts[r, 0] + t * directions[r, 0]
            qr_1 = starts[r, 1] + t * directions[r, 1]
            qr_2 = starts[r, 2] + t * directions[r, 2]

            inc_v0 = qr_0 - starts[r, 0]
            inc_v1 = qr_1 - starts[r, 1]
            inc_v2 = qr_2 - starts[r, 2]
            inc_n = math.sqrt(inc_v0 * inc_v0 + inc_v1 * inc_v1 + inc_v2 * inc_v2)
            inc_d = max(inc_n, eps_unit)
            inc_0 = inc_v0 / inc_d
            inc_1 = inc_v1 / inc_d
            inc_2 = inc_v2 / inc_d

            out_v0 = targets[r, 0] - qr_0
            out_v1 = targets[r, 1] - qr_1
            out_v2 = targets[r, 2] - qr_2
            out_n = math.sqrt(out_v0 * out_v0 + out_v1 * out_v1 + out_v2 * out_v2)
            out_d = max(out_n, eps_unit)
            out_0 = out_v0 / out_d
            out_1 = out_v1 / out_d
            out_2 = out_v2 / out_d

            diff_0 = inc_0 - out_0
            diff_1 = inc_1 - out_1
            diff_2 = inc_2 - out_2
            diff_n = math.sqrt(diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
            diff_d = max(diff_n, eps_unit)
            nreq_0 = diff_0 / diff_d
            nreq_1 = diff_1 / diff_d
            nreq_2 = diff_2 / diff_d

            normal_dot = (
                nreq_0 * ncur_0
                + nreq_1 * ncur_1
                + nreq_2 * ncur_2
            )
            if normal_dot < 0.0:
                nreq_0 = -nreq_0
                nreq_1 = -nreq_1
                nreq_2 = -nreq_2
                normal_dot = -normal_dot

            primary_ok, primary_angle_deg, primary_edge_residual, primary_separation = (
                _ci_pair_surface_compatible(
                    qr_0,
                    qr_1,
                    qr_2,
                    nreq_0,
                    nreq_1,
                    nreq_2,
                    pcur_0,
                    pcur_1,
                    pcur_2,
                    ncur_0,
                    ncur_1,
                    ncur_2,
                    normal_angle_floor_rad,
                    normal_angle_per_mm_rad,
                    normal_angle_cap_rad,
                    edge_height_residual_cap_mm,
                    maximum_candidate_radius_mm,
                    False,
                )
            )

            secondary_ok = True
            if secondary_neighbor_gate_enabled and (k - front_start_order) >= 2:
                secondary_index = k - 2
                p2_0 = points[secondary_index, 0]
                p2_1 = points[secondary_index, 1]
                p2_2 = points[secondary_index, 2]

                n2_0 = normals[secondary_index, 0]
                n2_1 = normals[secondary_index, 1]
                n2_2 = normals[secondary_index, 2]

                secondary_ok, _, _, _ = (
                    _ci_pair_surface_compatible(
                        qr_0,
                        qr_1,
                        qr_2,
                        nreq_0,
                        nreq_1,
                        nreq_2,
                        p2_0,
                        p2_1,
                        p2_2,
                        n2_0,
                        n2_1,
                        n2_2,
                        normal_angle_floor_rad,
                        normal_angle_per_mm_rad,
                        normal_angle_cap_rad,
                        edge_height_residual_cap_mm,
                        maximum_candidate_radius_mm,
                        True,
                    )
                )

            compatible = (
                primary_ok
                and secondary_ok
            )

            if compatible:
                compatible_candidate_count[k] += 1
                if cand_d < best_compatible_distance:
                    best_compatible_distance = cand_d
                    best_compatible_ray = r
                    best_q0 = qr_0
                    best_q1 = qr_1
                    best_q2 = qr_2
                    best_n0 = nreq_0
                    best_n1 = nreq_1
                    best_n2 = nreq_2

        if best_compatible_ray >= 0:
            ray = best_compatible_ray
            best_parent = k - 1
            best_distance = best_compatible_distance

            # Old-tangent recovery only in current front
            if k - 1 > front_start_order:
                dir_ray_0 = directions[ray, 0]
                dir_ray_1 = directions[ray, 1]
                dir_ray_2 = directions[ray, 2]
                st_ray_0 = starts[ray, 0]
                st_ray_1 = starts[ray, 1]
                st_ray_2 = starts[ray, 2]

                for j in range(front_start_order, k - 1):
                    nj_0 = normals[j, 0]
                    nj_1 = normals[j, 1]
                    nj_2 = normals[j, 2]

                    den_old = dir_ray_0 * nj_0 + dir_ray_1 * nj_1 + dir_ray_2 * nj_2
                    if abs(den_old) > eps_parallel:
                        pj_0 = points[j, 0]
                        pj_1 = points[j, 1]
                        pj_2 = points[j, 2]

                        num_old = (pj_0 - st_ray_0) * nj_0 + (pj_1 - st_ray_1) * nj_1 + (pj_2 - st_ray_2) * nj_2
                        t_old = num_old / den_old
                        if t_old > 1e-6 and math.isfinite(t_old):
                            qo_0 = st_ray_0 + t_old * dir_ray_0
                            qo_1 = st_ray_1 + t_old * dir_ray_1
                            qo_2 = st_ray_2 + t_old * dir_ray_2

                            dist_sq_old = (qo_0 - pj_0) ** 2 + (qo_1 - pj_1) ** 2 + (qo_2 - pj_2) ** 2
                            d_old = math.sqrt(dist_sq_old)

                            if d_old < best_distance:
                                inc_v0 = qo_0 - st_ray_0
                                inc_v1 = qo_1 - st_ray_1
                                inc_v2 = qo_2 - st_ray_2
                                inc_n = math.sqrt(inc_v0 * inc_v0 + inc_v1 * inc_v1 + inc_v2 * inc_v2)
                                inc_d = max(inc_n, eps_unit)
                                inc_0 = inc_v0 / inc_d
                                inc_1 = inc_v1 / inc_d
                                inc_2 = inc_v2 / inc_d

                                out_v0 = targets[ray, 0] - qo_0
                                out_v1 = targets[ray, 1] - qo_1
                                out_v2 = targets[ray, 2] - qo_2
                                out_n = math.sqrt(out_v0 * out_v0 + out_v1 * out_v1 + out_v2 * out_v2)
                                out_d = max(out_n, eps_unit)
                                out_0 = out_v0 / out_d
                                out_1 = out_v1 / out_d
                                out_2 = out_v2 / out_d

                                diff_0 = inc_0 - out_0
                                diff_1 = inc_1 - out_1
                                diff_2 = inc_2 - out_2
                                diff_n = math.sqrt(diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
                                diff_d = max(diff_n, eps_unit)
                                nreq_old_0 = diff_0 / diff_d
                                nreq_old_1 = diff_1 / diff_d
                                nreq_old_2 = diff_2 / diff_d

                                normal_dot_old = (
                                    nreq_old_0 * nj_0
                                    + nreq_old_1 * nj_1
                                    + nreq_old_2 * nj_2
                                )
                                if normal_dot_old < 0.0:
                                    nreq_old_0 = -nreq_old_0
                                    nreq_old_1 = -nreq_old_1
                                    nreq_old_2 = -nreq_old_2

                                old_parent_ok, _, _, _ = _ci_pair_surface_compatible(
                                    qo_0,
                                    qo_1,
                                    qo_2,
                                    nreq_old_0,
                                    nreq_old_1,
                                    nreq_old_2,
                                    pj_0,
                                    pj_1,
                                    pj_2,
                                    nj_0,
                                    nj_1,
                                    nj_2,
                                    normal_angle_floor_rad,
                                    normal_angle_per_mm_rad,
                                    normal_angle_cap_rad,
                                    edge_height_residual_cap_mm,
                                    maximum_candidate_radius_mm,
                                    False,
                                )

                                if secondary_neighbor_gate_enabled:
                                    current_continuity_ok, _, _, _ = _ci_pair_surface_compatible(
                                        qo_0,
                                        qo_1,
                                        qo_2,
                                        nreq_old_0,
                                        nreq_old_1,
                                        nreq_old_2,
                                        pcur_0,
                                        pcur_1,
                                        pcur_2,
                                        ncur_0,
                                        ncur_1,
                                        ncur_2,
                                        normal_angle_floor_rad,
                                        normal_angle_per_mm_rad,
                                        normal_angle_cap_rad,
                                        edge_height_residual_cap_mm,
                                        maximum_candidate_radius_mm,
                                        False,
                                    )
                                else:
                                    current_continuity_ok = True

                                if secondary_neighbor_gate_enabled and (k - front_start_order) >= 2:
                                    p2_0 = points[k - 2, 0]
                                    p2_1 = points[k - 2, 1]
                                    p2_2 = points[k - 2, 2]
                                    n2_0 = normals[k - 2, 0]
                                    n2_1 = normals[k - 2, 1]
                                    n2_2 = normals[k - 2, 2]
                                    secondary_ok, _, _, _ = _ci_pair_surface_compatible(
                                        qo_0,
                                        qo_1,
                                        qo_2,
                                        nreq_old_0,
                                        nreq_old_1,
                                        nreq_old_2,
                                        p2_0,
                                        p2_1,
                                        p2_2,
                                        n2_0,
                                        n2_1,
                                        n2_2,
                                        normal_angle_floor_rad,
                                        normal_angle_per_mm_rad,
                                        normal_angle_cap_rad,
                                        edge_height_residual_cap_mm,
                                        maximum_candidate_radius_mm,
                                        True,
                                    )
                                else:
                                    secondary_ok = True

                                if old_parent_ok and current_continuity_ok and secondary_ok:
                                    best_parent = j
                                    best_distance = d_old
                                    best_q0 = qo_0
                                    best_q1 = qo_1
                                    best_q2 = qo_2
                                    best_n0 = nreq_old_0
                                    best_n1 = nreq_old_1
                                    best_n2 = nreq_old_2

            points[k, 0] = best_q0
            points[k, 1] = best_q1
            points[k, 2] = best_q2

            parent_n0 = normals[best_parent, 0]
            parent_n1 = normals[best_parent, 1]
            parent_n2 = normals[best_parent, 2]
            if (best_n0 * parent_n0 + best_n1 * parent_n1 + best_n2 * parent_n2) < 0.0:
                best_n0 = -best_n0
                best_n1 = -best_n1
                best_n2 = -best_n2

            normals[k, 0] = best_n0
            normals[k, 1] = best_n1
            normals[k, 2] = best_n2

            order[k] = ray
            parent[k] = best_parent
            distance[k] = best_distance
            front_id[k] = active_front_id
            used[ray] = True

            d_angle = abs(best_n0 * parent_n0 + best_n1 * parent_n1 + best_n2 * parent_n2)
            if d_angle > 1.0:
                d_angle = 1.0
            parent_normal_angle_deg[k] = math.degrees(math.acos(d_angle))

        else:
            if math.isfinite(minimum_distance):
                restart_reason_code[k] = 1
            else:
                restart_reason_code[k] = 2

            ray = restart_ray
            if ray < 0:
                ray = first_remaining

            points[k, 0] = P0[ray, 0]
            points[k, 1] = P0[ray, 1]
            points[k, 2] = P0[ray, 2]

            parent[k] = -3
            distance[k] = np.nan
            active_front_id += 1
            front_start_order = k
            front_id[k] = active_front_id

            inc_v0 = points[k, 0] - starts[ray, 0]
            inc_v1 = points[k, 1] - starts[ray, 1]
            inc_v2 = points[k, 2] - starts[ray, 2]
            inc_n = math.sqrt(inc_v0 * inc_v0 + inc_v1 * inc_v1 + inc_v2 * inc_v2)
            inc_d = max(inc_n, eps_unit)
            inc_0 = inc_v0 / inc_d
            inc_1 = inc_v1 / inc_d
            inc_2 = inc_v2 / inc_d

            out_v0 = targets[ray, 0] - points[k, 0]
            out_v1 = targets[ray, 1] - points[k, 1]
            out_v2 = targets[ray, 2] - points[k, 2]
            out_n = math.sqrt(out_v0 * out_v0 + out_v1 * out_v1 + out_v2 * out_v2)
            out_d = max(out_n, eps_unit)
            out_0 = out_v0 / out_d
            out_1 = out_v1 / out_d
            out_2 = out_v2 / out_d

            diff_0 = inc_0 - out_0
            diff_1 = inc_1 - out_1
            diff_2 = inc_2 - out_2
            diff_n = math.sqrt(diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
            diff_d = max(diff_n, eps_unit)
            n0 = diff_0 / diff_d
            n1 = diff_1 / diff_d
            n2 = diff_2 / diff_d

            seed_dot = n0 * N0[ray, 0] + n1 * N0[ray, 1] + n2 * N0[ray, 2]
            if seed_dot < 0.0:
                n0 = -n0
                n1 = -n1
                n2 = -n2

            normals[k, 0] = n0
            normals[k, 1] = n1
            normals[k, 2] = n2

            order[k] = ray
            used[ray] = True

    return (
        points,
        normals,
        order,
        parent,
        distance,
        front_id,
        restart_reason_code,
        parent_normal_angle_deg,
        local_candidate_count,
        compatible_candidate_count,
    )


def require_ci_dispatcher() -> CPUDispatcher:
    """Refuse disabled JIT or a plain Python callable before any CI loop."""
    if numba.config.DISABLE_JIT:
        raise BackendExecutionError("CI_NUMBA_JIT_DISABLED")
    kernel = point_by_point_numba_core
    if not isinstance(kernel, CPUDispatcher):
        raise BackendExecutionError("CI_KERNEL_NOT_CPU_DISPATCHER")
    return kernel


def execute_compiled_point_by_point(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    seed_index: int,
    eps_parallel: float = 1e-10,
    eps_unit: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run only a verified nopython CPU specialization; never call py_func."""
    kernel = require_ci_dispatcher()
    prepare_start = perf_counter()
    arrays = tuple(np.ascontiguousarray(a, dtype=np.float64)
                   for a in (starts, directions, targets, P0))
    n_rays = len(arrays[0]) if arrays[0].ndim > 0 else 0
    if n_rays < 1 or any(a.shape != (n_rays, 3) for a in arrays):
        raise ValueError("CI_INPUT_SHAPE_REQUIRES_NONEMPTY_N_BY_3")
    if isinstance(seed_index, (bool, np.bool_)):
        raise ValueError("CI_SEED_INDEX_MUST_BE_INTEGER")
    try:
        seed = integer_index(seed_index)
    except TypeError as exc:
        raise ValueError("CI_SEED_INDEX_MUST_BE_INTEGER") from exc
    if not 0 <= seed < n_rays:
        raise ValueError("CI_SEED_INDEX_OUT_OF_RANGE")
    ep, eu = float(eps_parallel), float(eps_unit)
    if not math.isfinite(ep) or ep < 0.0 or not math.isfinite(eu) or eu <= 0.0:
        raise ValueError("CI_INVALID_EPSILON")
    # Do not normalize inputs or sanitize nonfinite values here.
    # Optical gates and the original geometric fallback remain unchanged.
    args = (*arrays, int(seed), ep, eu)
    signature = tuple(numba.typeof(a) for a in args)
    prepare_seconds = perf_counter() - prepare_start

    compile_seconds = 0.0
    if signature not in kernel.overloads:
        started = perf_counter()
        try:
            kernel.compile(signature)
        except Exception as exc:
            raise BackendExecutionError(
                f"CI_NUMBA_COMPILE_FAILED:{type(exc).__name__}:{exc}"
            ) from exc
        compile_seconds = perf_counter() - started

    specialization = kernel.overloads.get(signature)
    if (specialization is None or specialization.objectmode
            or not kernel.nopython_signatures):
        raise BackendExecutionError("CI_NO_MATCHING_NOPYTHON_SPECIALIZATION")

    started = perf_counter()
    try:
        result = kernel(*args)
    except Exception as exc:
        raise BackendExecutionError(
            f"CI_NUMBA_EXECUTION_FAILED:{type(exc).__name__}:{exc}"
        ) from exc
    kernel_seconds = perf_counter() - started
    runtime = current_runtime()
    if runtime is not None:
        runtime.record(
            "CI_KERNEL_COMPLETED", kernel="point_by_point_ci",
            actual_backend="cpu", implementation="numba_nopython",
            ray_count=n_rays, seed_index=int(seed), dtype="float64",
            prepare_seconds=prepare_seconds,
            compile_or_cache_load_seconds=compile_seconds,
            kernel_call_seconds=kernel_seconds,
            nopython_signature=str(specialization.signature),
        )
    return result


def warmup_ci_numba() -> dict[str, Any]:
    """Compile/load and actually execute two rays before production steps."""
    started = perf_counter()
    starts = np.array([[0., 0., 0.], [5., 0., 0.]])
    directions = np.array([[0., 0., 1.], [0., 0., 1.]])
    targets = np.array([[0., 10., 50.], [5., 10., 50.]])
    P0 = np.array([[0., 0., 20.], [5., 0., 20.]])
    result = execute_compiled_point_by_point(
        starts, directions, targets, P0, 0, 1e-10, 1e-9
    )
    if not (np.all(np.isfinite(result[0]))
            and np.all(np.isfinite(result[1]))
            and np.array_equal(result[2], [0, 1])):
        raise BackendExecutionError("CI_JIT_WARMUP_OUTPUT_INVALID")
    return {
        "actual_backend": "cpu", "implementation": "numba_nopython",
        "numba_version": numba.__version__, "jit_enabled": True,
        "warmup_ray_count": 2, "warmup_wall_seconds": perf_counter() - started,
        "nopython_signatures": [str(s) for s in
                                require_ci_dispatcher().nopython_signatures],
    }


def require_ci_compatible_dispatcher() -> CPUDispatcher:
    """Refuse disabled JIT or a plain Python callable before any CI loop."""
    if numba.config.DISABLE_JIT:
        raise BackendExecutionError("CI_NUMBA_JIT_DISABLED")
    kernel = point_by_point_compatible_numba_core
    if not isinstance(kernel, CPUDispatcher):
        raise BackendExecutionError("CI_COMPATIBLE_KERNEL_NOT_CPU_DISPATCHER")
    return kernel


def _execute_compiled_point_by_point_compatible_common(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    N0: np.ndarray,
    seed_index: int,
    eps_parallel: float = 1e-10,
    eps_unit: float = 1e-9,
    minimum_candidate_radius_mm: float = 0.75,
    candidate_distance_factor: float = 8.0,
    maximum_candidate_radius_mm: float = 2.0,
    normal_angle_floor_deg: float = 0.1,
    normal_angle_per_mm_deg: float = 1.0,
    normal_angle_cap_deg: float = 5.0,
    *,
    edge_height_residual_cap_mm: float = math.inf,
    secondary_neighbor_gate_enabled: bool = False,
    construction_policy: str = "SURFACE_COMPATIBLE_FRONTIER_V1",
    implementation_name: str = "numba_nopython_surface_compatible_frontier_v1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run verified nopython CPU specialization for compatible frontier; common implementation for V1, V2, and V3."""
    kernel = require_ci_compatible_dispatcher()
    prepare_start = perf_counter()
    arrays = tuple(np.ascontiguousarray(a, dtype=np.float64)
                   for a in (starts, directions, targets, P0, N0))
    n_rays = len(arrays[0]) if arrays[0].ndim > 0 else 0
    if n_rays < 1 or any(a.shape != (n_rays, 3) for a in arrays):
        raise ValueError("CI_INPUT_SHAPE_REQUIRES_NONEMPTY_N_BY_3")
    if not np.all(np.isfinite(arrays[3])):
        raise RuntimeError("CI_COMPATIBLE_P0_NONFINITE")
    if not np.all(np.isfinite(arrays[4])):
        raise RuntimeError("CI_COMPATIBLE_N0_NONFINITE")
    if isinstance(seed_index, (bool, np.bool_)):
        raise ValueError("CI_SEED_INDEX_MUST_BE_INTEGER")
    try:
        seed = integer_index(seed_index)
    except TypeError as exc:
        raise ValueError("CI_SEED_INDEX_MUST_BE_INTEGER") from exc
    if not 0 <= seed < n_rays:
        raise ValueError("CI_SEED_INDEX_OUT_OF_RANGE")
    ep, eu = float(eps_parallel), float(eps_unit)
    if not math.isfinite(ep) or ep < 0.0 or not math.isfinite(eu) or eu <= 0.0:
        raise ValueError("CI_INVALID_EPSILON")

    valid_policies = (
        "SURFACE_COMPATIBLE_FRONTIER_V1",
        "SURFACE_COMPATIBLE_FRONTIER_V2",
        "SURFACE_COMPATIBLE_PATCH_V3",
    )
    if construction_policy not in valid_policies:
        raise ValueError(f"CI_CONSTRUCTION_POLICY_INVALID:{construction_policy}")

    edge_cap = float(edge_height_residual_cap_mm)
    if construction_policy in (
        "SURFACE_COMPATIBLE_FRONTIER_V2",
        "SURFACE_COMPATIBLE_PATCH_V3",
    ):
        if not math.isfinite(edge_cap) or edge_cap <= 0.0:
            raise ValueError("CI_EDGE_HEIGHT_CAP_INVALID")

    min_r = float(minimum_candidate_radius_mm)
    dist_fac = float(candidate_distance_factor)
    max_r = float(maximum_candidate_radius_mm)
    floor_rad = math.radians(float(normal_angle_floor_deg))
    rate_rad = math.radians(float(normal_angle_per_mm_deg))
    cap_rad = math.radians(float(normal_angle_cap_deg))

    args = (
        *arrays,
        int(seed),
        ep,
        eu,
        min_r,
        dist_fac,
        max_r,
        floor_rad,
        rate_rad,
        cap_rad,
        edge_cap,
        bool(secondary_neighbor_gate_enabled),
    )
    signature = tuple(numba.typeof(a) for a in args)
    prepare_seconds = perf_counter() - prepare_start

    compile_seconds = 0.0
    if signature not in kernel.overloads:
        started = perf_counter()
        try:
            kernel.compile(signature)
        except Exception as exc:
            raise BackendExecutionError(
                f"CI_NUMBA_COMPILE_FAILED:{type(exc).__name__}:{exc}"
            ) from exc
        compile_seconds = perf_counter() - started

    specialization = kernel.overloads.get(signature)
    if (specialization is None or specialization.objectmode
            or not kernel.nopython_signatures):
        raise BackendExecutionError("CI_NO_MATCHING_NOPYTHON_SPECIALIZATION")

    started = perf_counter()
    try:
        result = kernel(*args)
    except Exception as exc:
        raise BackendExecutionError(
            f"CI_NUMBA_EXECUTION_FAILED:{type(exc).__name__}:{exc}"
        ) from exc
    kernel_seconds = perf_counter() - started
    runtime = current_runtime()
    if runtime is not None:
        runtime.record(
            "CI_KERNEL_COMPLETED",
            kernel="point_by_point_ci",
            actual_backend="cpu",
            implementation=implementation_name,
            construction_policy=construction_policy,
            ray_count=n_rays,
            seed_index=int(seed),
            dtype="float64",
            prepare_seconds=prepare_seconds,
            compile_or_cache_load_seconds=compile_seconds,
            kernel_call_seconds=kernel_seconds,
            nopython_signature=str(specialization.signature),
        )
    return result


def execute_compiled_point_by_point_compatible(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    N0: np.ndarray,
    seed_index: int,
    eps_parallel: float = 1e-10,
    eps_unit: float = 1e-9,
    minimum_candidate_radius_mm: float = 0.75,
    candidate_distance_factor: float = 8.0,
    maximum_candidate_radius_mm: float = 2.0,
    normal_angle_floor_deg: float = 0.1,
    normal_angle_per_mm_deg: float = 1.0,
    normal_angle_cap_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run verified nopython CPU specialization for SURFACE_COMPATIBLE_FRONTIER_V1."""
    return _execute_compiled_point_by_point_compatible_common(
        starts,
        directions,
        targets,
        P0,
        N0,
        seed_index,
        eps_parallel=eps_parallel,
        eps_unit=eps_unit,
        minimum_candidate_radius_mm=minimum_candidate_radius_mm,
        candidate_distance_factor=candidate_distance_factor,
        maximum_candidate_radius_mm=maximum_candidate_radius_mm,
        normal_angle_floor_deg=normal_angle_floor_deg,
        normal_angle_per_mm_deg=normal_angle_per_mm_deg,
        normal_angle_cap_deg=normal_angle_cap_deg,
        edge_height_residual_cap_mm=math.inf,
        secondary_neighbor_gate_enabled=False,
        construction_policy="SURFACE_COMPATIBLE_FRONTIER_V1",
        implementation_name="numba_nopython_surface_compatible_frontier_v1",
    )


def execute_compiled_point_by_point_compatible_v2(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    N0: np.ndarray,
    seed_index: int,
    eps_parallel: float = 1e-10,
    eps_unit: float = 1e-9,
    minimum_candidate_radius_mm: float = 0.75,
    candidate_distance_factor: float = 8.0,
    maximum_candidate_radius_mm: float = 2.0,
    normal_angle_floor_deg: float = 0.1,
    normal_angle_per_mm_deg: float = 1.0,
    normal_angle_cap_deg: float = 5.0,
    edge_height_residual_cap_mm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run verified nopython CPU specialization for SURFACE_COMPATIBLE_FRONTIER_V2."""
    edge_cap = float(edge_height_residual_cap_mm)
    if not math.isfinite(edge_cap) or edge_cap <= 0.0:
        raise ValueError("CI_EDGE_HEIGHT_CAP_INVALID")

    return _execute_compiled_point_by_point_compatible_common(
        starts,
        directions,
        targets,
        P0,
        N0,
        seed_index,
        eps_parallel=eps_parallel,
        eps_unit=eps_unit,
        minimum_candidate_radius_mm=minimum_candidate_radius_mm,
        candidate_distance_factor=candidate_distance_factor,
        maximum_candidate_radius_mm=maximum_candidate_radius_mm,
        normal_angle_floor_deg=normal_angle_floor_deg,
        normal_angle_per_mm_deg=normal_angle_per_mm_deg,
        normal_angle_cap_deg=normal_angle_cap_deg,
        edge_height_residual_cap_mm=edge_cap,
        secondary_neighbor_gate_enabled=False,
        construction_policy="SURFACE_COMPATIBLE_FRONTIER_V2",
        implementation_name="numba_nopython_surface_compatible_frontier_v2",
    )


def execute_compiled_point_by_point_compatible_v3(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    P0: np.ndarray,
    N0: np.ndarray,
    seed_index: int,
    eps_parallel: float = 1e-10,
    eps_unit: float = 1e-9,
    minimum_candidate_radius_mm: float = 0.75,
    candidate_distance_factor: float = 8.0,
    maximum_candidate_radius_mm: float = 2.0,
    normal_angle_floor_deg: float = 0.1,
    normal_angle_per_mm_deg: float = 1.0,
    normal_angle_cap_deg: float = 5.0,
    edge_height_residual_cap_mm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run verified nopython CPU specialization for SURFACE_COMPATIBLE_PATCH_V3."""
    edge_cap = float(edge_height_residual_cap_mm)
    if not math.isfinite(edge_cap) or edge_cap <= 0.0:
        raise ValueError("CI_EDGE_HEIGHT_CAP_INVALID")

    return _execute_compiled_point_by_point_compatible_common(
        starts,
        directions,
        targets,
        P0,
        N0,
        seed_index,
        eps_parallel=eps_parallel,
        eps_unit=eps_unit,
        minimum_candidate_radius_mm=minimum_candidate_radius_mm,
        candidate_distance_factor=candidate_distance_factor,
        maximum_candidate_radius_mm=maximum_candidate_radius_mm,
        normal_angle_floor_deg=normal_angle_floor_deg,
        normal_angle_per_mm_deg=normal_angle_per_mm_deg,
        normal_angle_cap_deg=normal_angle_cap_deg,
        edge_height_residual_cap_mm=edge_cap,
        secondary_neighbor_gate_enabled=True,
        construction_policy="SURFACE_COMPATIBLE_PATCH_V3",
        implementation_name="numba_nopython_surface_compatible_patch_v3",
    )


def warmup_ci_compatible_numba(
    edge_height_residual_cap_mm: float = math.inf,
    construction_policy: str = "SURFACE_COMPATIBLE_FRONTIER_V1",
) -> dict[str, Any]:
    """Compile/load and execute 3 rays with compatible kernel before production steps."""
    started = perf_counter()
    starts = np.array([[0., 0., 0.], [5., 0., 0.], [10., 0., 0.]])
    directions = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    targets = np.array([[0., 10., 50.], [5., 10., 50.], [10., 10., 50.]])
    P0 = np.array([[0., 0., 20.], [5., 0., 20.], [10., 0., 20.]])
    N0 = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])

    if construction_policy == "SURFACE_COMPATIBLE_PATCH_V3":
        impl_name = "numba_nopython_surface_compatible_patch_v3"
    elif construction_policy == "SURFACE_COMPATIBLE_FRONTIER_V2":
        impl_name = "numba_nopython_surface_compatible_frontier_v2"
    else:
        impl_name = "numba_nopython_surface_compatible_frontier_v1"

    result = _execute_compiled_point_by_point_compatible_common(
        starts, directions, targets, P0, N0, 0, 1e-10, 1e-9,
        0.75, 8.0, 20.0, 0.1, 1.0, 5.0,
        edge_height_residual_cap_mm=edge_height_residual_cap_mm,
        secondary_neighbor_gate_enabled=(
            construction_policy == "SURFACE_COMPATIBLE_PATCH_V3"
        ),
        construction_policy=construction_policy,
        implementation_name=impl_name,
    )
    if not (np.all(np.isfinite(result[0]))
            and np.all(np.isfinite(result[1]))
            and np.array_equal(np.sort(result[2]), [0, 1, 2])):
        raise BackendExecutionError("CI_COMPATIBLE_JIT_WARMUP_OUTPUT_INVALID")
    return {
        "actual_backend": "cpu",
        "implementation": impl_name,
        "construction_policy": construction_policy,
        "numba_version": numba.__version__,
        "jit_enabled": True,
        "warmup_ray_count": 3,
        "warmup_wall_seconds": perf_counter() - started,
        "nopython_signatures": [str(s) for s in
                                require_ci_compatible_dispatcher().nopython_signatures],
    }
