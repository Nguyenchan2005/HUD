"""Các phép toán hình học, ray-trace, merit function và tối ưu lõi của HUD Fan v5.5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
import numpy as np
import pandas as pd
import scipy
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import Polynomial
from scipy.linalg import solve
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull, cKDTree

from live_monitor_v55 import progress_indices
from execution_v55 import selected_backend, BackendExecutionError, current_runtime, is_compute_worker
from kernel_dispatch_v55 import dispatch_kernel
from kernels_cpu_v55 import FixedDesignLeastSquares
from kernels_ci_v55 import (
    execute_compiled_point_by_point,
    execute_compiled_point_by_point_compatible,
    execute_compiled_point_by_point_compatible_v2,
    execute_compiled_point_by_point_compatible_v3,
)


EPS = 1.0e-9


INTERSECTION_RESIDUAL_TOLERANCE_MM = 1.0e-7
APERTURE_BOUNDARY_TOLERANCE_MM = 1.0e-8
CONIC_DOMAIN_TOLERANCE = 1.0e-12


def normalize_aperture_polygon(polygon: np.ndarray) -> np.ndarray:
    """Validate finite vertices, simplify collinear points, and normalize to counterclockwise."""
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("CLEAR_APERTURE_POLYGON_MUST_BE_N_BY_2_WITH_AT_LEAST_THREE_VERTICES")
    if not np.all(np.isfinite(poly)):
        raise ValueError("CLEAR_APERTURE_POLYGON_NONFINITE")
    if np.linalg.norm(poly[0] - poly[-1]) <= APERTURE_BOUNDARY_TOLERANCE_MM:
        poly = poly[:-1]
    if len(poly) < 3:
        raise ValueError("CLEAR_APERTURE_POLYGON_DEGENERATE")

    # Loai bo cac diem thang hang trung gian doc theo canh da giac
    while len(poly) >= 3:
        prev_edges = poly - np.roll(poly, 1, axis=0)
        next_edges = np.roll(poly, -1, axis=0) - poly
        scale = max(float(np.max(np.linalg.norm(next_edges, axis=1))), 1.0)
        turns = prev_edges[:, 0] * next_edges[:, 1] - prev_edges[:, 1] * next_edges[:, 0]
        collinear = np.abs(turns) <= 1e-8 * scale * np.linalg.norm(next_edges, axis=1)
        if not np.any(collinear) or np.sum(~collinear) < 3:
            break
        poly = poly[~collinear]

    if len(poly) < 3:
        raise ValueError("CLEAR_APERTURE_POLYGON_DEGENERATE")

    edges = np.roll(poly, -1, axis=0) - poly
    area2 = float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                         - poly[:, 1] * np.roll(poly[:, 0], -1)))
    scale = max(float(np.max(np.ptp(poly, axis=0))), 1.0)
    if abs(area2) <= 1e-12 * scale * scale or np.any(np.linalg.norm(edges, axis=1) <= 1e-12 * scale):
        raise ValueError("CLEAR_APERTURE_POLYGON_DEGENERATE")
    turns = (edges[:, 0] * np.roll(edges[:, 1], -1)
             - edges[:, 1] * np.roll(edges[:, 0], -1))
    significant = turns[np.abs(turns) > 1e-12 * scale * scale]
    if len(significant) != len(turns) or (np.any(significant > 0.0) and np.any(significant < 0.0)):
        raise ValueError("CLEAR_APERTURE_POLYGON_MUST_BE_STRICTLY_CONVEX")
    return poly.copy() if area2 > 0.0 else poly[::-1].copy()


def aperture_bbox(polygon: np.ndarray) -> np.ndarray:
    """Return the symmetric half-size bounding box of a validated polygon."""
    poly = normalize_aperture_polygon(polygon)
    return np.max(np.abs(poly), axis=0)


def aperture_contains_xy(half_aperture: np.ndarray, polygon: np.ndarray | None,
                         x: np.ndarray, y: np.ndarray,
                         tolerance_mm: float = APERTURE_BOUNDARY_TOLERANCE_MM) -> np.ndarray:
    """Classify local points against the single authoritative clear aperture."""
    xv, yv = np.asarray(x, float), np.asarray(y, float)
    if polygon is None:
        half = np.asarray(half_aperture, float)
        return ((np.abs(xv) <= half[0]+tolerance_mm)
                & (np.abs(yv) <= half[1]+tolerance_mm))
    poly = normalize_aperture_polygon(polygon)
    edges = np.roll(poly, -1, axis=0)-poly
    cross = (edges[None, :, 0]*(yv.reshape(-1, 1)-poly[None, :, 1])
             - edges[None, :, 1]*(xv.reshape(-1, 1)-poly[None, :, 0]))
    edge_scale = np.maximum(np.linalg.norm(edges, axis=1), 1.0)
    inside = np.all(cross >= -tolerance_mm*edge_scale[None, :], axis=1)
    return inside.reshape(xv.shape)


def aperture_boundary_xy(half_aperture: np.ndarray, polygon: np.ndarray | None) -> np.ndarray:
    """Return exact local boundary vertices used by sanity, export and rendering."""
    if polygon is not None:
        return normalize_aperture_polygon(polygon)
    hx, hy = np.asarray(half_aperture, float)
    if not np.all(np.isfinite([hx, hy])) or hx <= 0.0 or hy <= 0.0:
        raise ValueError("CLEAR_APERTURE_HALF_SIZE_INVALID")
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], float)


def _offset_convex_polygon_edges(polygon: np.ndarray, margin_mm: float) -> np.ndarray:
    """Offset every supporting edge of a convex CCW polygon outward by a true normal distance."""
    poly = normalize_aperture_polygon(polygon)
    margin = float(margin_mm)
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("CLEAR_APERTURE_MARGIN_INVALID")
    if margin == 0.0:
        return poly.copy()

    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("CLEAR_APERTURE_OFFSET_EDGE_INVALID")

    # normalize_aperture_polygon() guarantees CCW winding; the right-hand
    # normal of each directed edge therefore points outward.
    outward = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    shifted = poly + margin * outward
    expanded = np.empty_like(poly)

    for index in range(len(poly)):
        previous = (index - 1) % len(poly)
        d_previous = edges[previous]
        d_current = edges[index]
        denominator = (
            d_previous[0] * d_current[1]
            - d_previous[1] * d_current[0]
        )
        denominator_scale = max(
            float(lengths[previous] * lengths[index]),
            1.0,
        )
        if abs(denominator) <= 64.0 * np.finfo(float).eps * denominator_scale:
            raise ValueError("CLEAR_APERTURE_OFFSET_ADJACENT_EDGES_PARALLEL")

        delta = shifted[index] - shifted[previous]
        parameter = (
            delta[0] * d_current[1]
            - delta[1] * d_current[0]
        ) / denominator
        expanded[index] = shifted[previous] + parameter * d_previous

    if not np.all(np.isfinite(expanded)):
        raise ValueError("CLEAR_APERTURE_OFFSET_NONFINITE")

    return normalize_aperture_polygon(expanded)


def convex_aperture_from_points(points_xy: np.ndarray, margin_mm: float = 0.5,
                                floor: tuple[float, float] = (1.0, 1.0)
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Build one deterministic convex clear aperture and its consistent bbox."""
    xy = np.asarray(points_xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.all(np.isfinite(xy)):
        raise ValueError("CLEAR_APERTURE_FOOTPRINT_INCOMPLETE_OR_NONFINITE")
    try:
        hull = ConvexHull(xy)
    except Exception as exc:
        raise ValueError("CLEAR_APERTURE_FOOTPRINT_DEGENERATE") from exc
    poly = normalize_aperture_polygon(xy[hull.vertices])
    if margin_mm < 0.0 or not np.isfinite(margin_mm):
        raise ValueError("CLEAR_APERTURE_MARGIN_INVALID")
    if margin_mm:
        poly = _offset_convex_polygon_edges(poly, margin_mm)
    half = np.maximum(aperture_bbox(poly), np.asarray(floor, float))
    return poly, half


def rescale_surface_polynomial(surface: "PolySurface", new_scale: np.ndarray) -> None:
    """Change normalized XY scale while preserving polynomial sag exactly."""
    new = np.asarray(new_scale, float)
    old = np.asarray(surface.scale, float)
    if new.shape != (2,) or np.any(~np.isfinite(new)) or np.any(new <= 0.0):
        raise ValueError("SURFACE_SCALE_INVALID")
    if old.shape != (2,) or np.any(~np.isfinite(old)) or np.any(old <= 0.0):
        raise ValueError("SURFACE_EXISTING_SCALE_INVALID")
    surface.coeff = np.asarray([
        value*(new[0]/old[0])**i*(new[1]/old[1])**j
        for (i, j), value in zip(surface.terms, surface.coeff)
    ], float)
    surface.scale = new.copy()


def _normalized_ray_batch(origins: np.ndarray, directions: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each ray without converting zero/non-finite directions into usable rays."""
    O = np.atleast_2d(np.asarray(origins, float))
    raw = np.atleast_2d(np.asarray(directions, float))
    if O.shape != raw.shape or O.ndim != 2 or O.shape[1] != 3:
        raise ValueError("RAY_ORIGIN_DIRECTION_SHAPE_MISMATCH")
    finite = np.all(np.isfinite(O), axis=1) & np.all(np.isfinite(raw), axis=1)
    norm = np.linalg.norm(raw, axis=1)
    usable = finite & (norm > EPS)
    D = np.full_like(raw, np.nan)
    D[usable] = raw[usable]/norm[usable, None]
    return O, D, usable, finite


def _polynomial_real_roots(
    poly: Polynomial,
) -> tuple[np.ndarray, str]:
    """Return numerically real roots using the legacy raw-coefficient reduction."""
    coef = np.asarray(poly.coef, dtype=float)

    if not np.all(np.isfinite(coef)):
        return (
            np.empty(0),
            "UNRESOLVED_NONFINITE_POLYNOMIAL",
        )

    scale = max(
        float(np.max(np.abs(coef))),
        1.0,
    )

    nz = np.flatnonzero(
        np.abs(coef) > 1e-13 * scale
    )

    if len(nz) == 0:
        return (
            np.empty(0),
            "DEGENERATE_RAY_ON_SURFACE",
        )

    reduced = Polynomial(
        coef[:int(nz[-1]) + 1]
    )

    if reduced.degree() == 0:
        return (
            np.empty(0),
            "NO_INTERSECTION_IN_SEARCH_DOMAIN",
        )

    try:
        roots = reduced.roots()
    except Exception:
        return (
            np.empty(0),
            "UNRESOLVED_NUMERIC_ROOT_ISOLATION",
        )

    real = np.asarray(
        [
            float(root.real)
            for root in roots
            if np.isfinite(root.real)
            and np.isfinite(root.imag)
            and abs(root.imag)
            <= 2e-7 * (1.0 + abs(root.real))
        ],
        dtype=float,
    )

    if len(real):
        real.sort()

    return (
        real,
        "ROOTS_ISOLATED_BY_REDUCED_POLYNOMIAL",
    )


def _linear_power_coefficients(
    constant: float,
    slope: float,
    power: int,
) -> np.ndarray:
    """Khai triển nhị thức (a + b*t)^p thành mảng hệ số đơn thức theo t."""
    return np.asarray([
        math.comb(power, k)
        * (constant ** (power - k))
        * (slope ** k)
        for k in range(power + 1)
    ], dtype=float)


def _poly_coeff_convolve(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Tích chập hai mảng hệ số đa thức."""
    return np.convolve(a, b)


def _poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cộng hai mảng hệ số đa thức khác độ dài."""
    la = len(a)
    lb = len(b)
    if la < lb:
        res = b.copy()
        res[:la] += a
        return res
    else:
        res = a.copy()
        res[:lb] += b
        return res


def _distance_scaled_real_roots_from_coefficients(
    coefficients: np.ndarray,
    t_reference: float,
) -> tuple[np.ndarray, str]:
    """Isolate physically relevant roots after scaling the ray-distance variable directly from coefficient array.

    The importance of coefficient a_k is evaluated through
    a_k * t_scale**k, not through raw |a_k|.
    """
    coef = np.asarray(
        coefficients,
        dtype=float,
    )

    if (
        coef.ndim != 1
        or coef.size == 0
    ):
        return (
            np.empty(0),
            "UNRESOLVED_INVALID_POLYNOMIAL_COEFFICIENTS",
        )

    if not np.all(np.isfinite(coef)):
        return (
            np.empty(0),
            "UNRESOLVED_NONFINITE_POLYNOMIAL",
        )

    reference = float(t_reference)

    if not np.isfinite(reference):
        reference = 1.0

    t_scale = max(
        1.0,
        2.0 * abs(reference),
    )

    degree = len(coef) - 1

    powers = np.power(
        t_scale,
        np.arange(degree + 1, dtype=float),
    )

    scaled_coef = coef * powers

    if not np.all(np.isfinite(scaled_coef)):
        return (
            np.empty(0),
            "UNRESOLVED_DISTANCE_SCALING_OVERFLOW",
        )

    contribution_scale = max(
        float(np.max(np.abs(scaled_coef))),
        1.0,
    )

    significant = np.flatnonzero(
        np.abs(scaled_coef)
        > 1e-13 * contribution_scale
    )

    if len(significant) == 0:
        return (
            np.empty(0),
            "DEGENERATE_RAY_ON_SURFACE",
        )

    last = int(significant[-1])

    working_coef = scaled_coef[:last + 1].copy()

    if len(working_coef) == 1:
        return (
            np.empty(0),
            "NO_INTERSECTION_IN_SEARCH_DOMAIN",
        )

    amplitude = float(
        np.max(np.abs(working_coef))
    )

    if (
        not np.isfinite(amplitude)
        or amplitude <= 0.0
    ):
        return (
            np.empty(0),
            "UNRESOLVED_NUMERIC_ROOT_ISOLATION",
        )

    try:
        roots_u = np.polynomial.polynomial.polyroots(
            working_coef / amplitude
        )
    except Exception:
        return (
            np.empty(0),
            "UNRESOLVED_NUMERIC_ROOT_ISOLATION",
        )

    roots_t = []

    for root in roots_u:
        if (
            not np.isfinite(root.real)
            or not np.isfinite(root.imag)
        ):
            continue

        if (
            abs(root.imag)
            > 2e-7 * (1.0 + abs(root.real))
        ):
            continue

        t_value = (
            float(root.real) * t_scale
        )

        if np.isfinite(t_value):
            roots_t.append(t_value)

    real = np.asarray(
        roots_t,
        dtype=float,
    )

    if len(real):
        real.sort()

    return (
        real,
        "ROOTS_ISOLATED_BY_DISTANCE_SCALED_POLYNOMIAL",
    )


def _distance_scaled_polynomial_real_roots(
    poly: Polynomial,
    t_reference: float,
) -> tuple[np.ndarray, str]:
    """Compatibility wrapper around _distance_scaled_real_roots_from_coefficients."""
    return _distance_scaled_real_roots_from_coefficients(
        np.asarray(
            poly.coef,
            dtype=float,
        ),
        t_reference,
    )


def unit(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Chuẩn hóa vectơ về độ dài 1; vectơ gần 0 được giữ an toàn."""
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, EPS)


def as_jsonable(value: Any) -> Any:
    """Đổi kiểu NumPy và cấu trúc lồng nhau thành dữ liệu ghi được bằng JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    """Ghi JSON UTF-8 ổn định để kiểm tra và truy vết."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Ghi các bản ghi ra CSV và bảo toàn toàn bộ tên cột."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            w.writeheader()
            w.writerows([{k: as_jsonable(v) for k, v in r.items()} for r in rows])


def _score_value(value: Any, unit: str = "") -> str:
    """Định dạng một giá trị scorecard ổn định cho CSV, JSON và ảnh."""
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).upper()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "N/A"
    suffix = f" {unit}" if unit else ""
    return f"{number:.6g}{suffix}"


def build_final_scorecard(config: dict[str, Any], numerical: dict[str, Any]) -> list[dict[str, Any]]:
    """Lập scorecard có bằng chứng động từ đúng config và kết quả STEP 24.

    Mỗi dòng lưu actual/operator/threshold/reason để hàm vẽ không phải
    hard-code lại giới hạn. Thay giới hạn trong config rồi chạy lại pipeline sẽ
    đồng thời đổi quyết định, CSV, JSON và nội dung ảnh STEP 25.
    """
    n = numerical
    dist = n["distortion"]
    rows: list[dict[str, Any]] = []

    def add(item: str, result: Any, operator: str | None, threshold: Any,
            unit: str, criterion: str, status: str, actual_display: str,
            rule_display: str, reason: str, config_source: str) -> None:
        """Thêm một dòng với cùng schema truy vết cho mọi loại gate."""
        rows.append({
            "item": item, "result": result, "operator": operator,
            "threshold": threshold, "unit": unit, "criterion": criterion,
            "status": status, "actual_display": actual_display,
            "rule_display": rule_display, "reason": reason,
            "config_source": config_source,
        })

    # USER distortion: nếu bundle chưa đủ thì partial Dmax chỉ là diagnostic,
    # không được dùng để biến một kết quả chưa đánh giá được thành PASS.
    distortion_limit = float(config["distortion"]["hard_limit_percent"])
    distortion_actual = dist.get("D_max_percent")
    distortion_complete = (dist.get("evaluation_status") == "COMPLETE"
                           and distortion_actual is not None
                           and np.isfinite(float(distortion_actual)))
    distortion_pass = bool(distortion_complete and float(distortion_actual) < distortion_limit)
    distortion_status = "PASS" if distortion_pass else "FAIL"
    distortion_rule = f"Dmax < {distortion_limit:g} %"
    if distortion_complete:
        observed = "<" if distortion_pass else ">="
        distortion_reason = (f"{distortion_status}: Dmax {_score_value(distortion_actual, '%')} "
                             f"{observed} {_score_value(distortion_limit, '%')}")
        distortion_display = f"Dmax={_score_value(distortion_actual, '%')}"
    else:
        valid = int(dist.get("valid_bundle_count", 0))
        required = int(dist.get("required_bundle_count", 0))
        missing = max(0, required - valid)
        evaluated_pairs = int(dist.get("evaluated_noncentral_field_pupil_count", 0))
        required_pairs = int(dist.get("required_count", 0))
        partial = dist.get("D_partial_max_percent_diagnostic")
        partial_text = (f"; partial Dmax={_score_value(partial, '%')} is diagnostic only"
                        if partial is not None else "")
        distortion_display = f"Dmax=N/A; complete bundles={valid}/{required}"
        distortion_reason = (f"FAIL: Dmax not evaluable; complete bundles {valid}/{required} "
                             f"(missing {missing}), graded pairs {evaluated_pairs}/{required_pairs}"
                             f"{partial_text}")
    add("USER_FIXED_GRID_VECTOR_DISTORTION", distortion_actual, "<", distortion_limit,
        "%", f"<{distortion_limit:g} percent", distortion_status, distortion_display,
        distortion_rule, distortion_reason, "distortion.hard_limit_percent")

    # Packaging is genuinely disabled when P1-P8 is absent.
    packaging_enabled = bool(n.get("packaging_constraint_enabled", False))
    packaging_actual = n.get("packaging_lambda")
    packaging_limit = float(config["packaging_lambda_max"])
    if packaging_enabled:
        packaging_pass = bool(packaging_actual is not None
                              and float(packaging_actual) <= packaging_limit)
        packaging_status = "PASS" if packaging_pass else "FAIL"
        observed = "<=" if packaging_pass else ">"
        packaging_display = f"lambda={_score_value(packaging_actual)}"
        packaging_rule = f"lambda <= {packaging_limit:g}"
        packaging_reason = (f"{packaging_status}: lambda {_score_value(packaging_actual)} "
                            f"{observed} {_score_value(packaging_limit)}")
        packaging_criterion = f"<={packaging_limit:g}"
        packaging_operator: str | None = "<="
        packaging_threshold: Any = packaging_limit
    else:
        packaging_status = "NOT_CONSTRAINED"
        packaging_display = "lambda=N/A"
        packaging_rule = "gate enabled only when P1-P8 is declared"
        packaging_reason = "NOT CONSTRAINED: packaging_vertices_mm is null/[]/missing"
        packaging_criterion = "P1-P8_NOT_DECLARED"
        packaging_operator = None
        packaging_threshold = None
    add("PACKAGING_LAMBDA", packaging_actual, packaging_operator, packaging_threshold,
        "ratio", packaging_criterion, packaging_status, packaging_display,
        packaging_rule, packaging_reason,
        "hud_geometry_authority.packaging_vertices_mm + packaging_lambda_max")

    # Visor footprint uses the configured full active size, not a number embedded in the plot.
    footprint = n.get("visor_footprint", {})
    visor_full = np.asarray(config["visor_active_mm"], float)
    visor_half = visor_full / 2.0
    def maximum_absolute(low: Any, high: Any) -> float | None:
        """Lấy trị tuyệt đối lớn nhất của hai biên hữu hạn."""
        values = []
        for value in (low, high):
            if value is not None and np.isfinite(float(value)):
                values.append(abs(float(value)))
        return max(values) if values else None
    max_u = maximum_absolute(footprint.get("converged_VISOR_u_min_mm"),
                             footprint.get("converged_VISOR_u_max_mm"))
    max_v = maximum_absolute(footprint.get("converged_VISOR_v_min_mm"),
                             footprint.get("converged_VISOR_v_max_mm"))
    visor_pass = bool(n.get("visor_footprint_inside_66x42", False))
    visor_status = "PASS" if visor_pass else "FAIL"
    visor_display = f"max|u|={_score_value(max_u, 'mm')}; max|v|={_score_value(max_v, 'mm')}"
    visor_rule = (f"max|u| <= {visor_half[0]:g} mm AND "
                  f"max|v| <= {visor_half[1]:g} mm")
    if max_u is None or max_v is None:
        visor_reason = "FAIL: no finite converged-ray footprint is available"
    else:
        failed_axes = []
        if max_u > visor_half[0]:
            failed_axes.append(f"|u| exceeds by {max_u - visor_half[0]:.6g} mm")
        if max_v > visor_half[1]:
            failed_axes.append(f"|v| exceeds by {max_v - visor_half[1]:.6g} mm")
        visor_reason = ("PASS: both footprint axes are within the active visor"
                        if visor_pass else "FAIL: " + "; ".join(failed_axes or ["footprint gate is false"]))
    add("VISOR_FOOTPRINT_CONVERGED_RAYS", visor_pass, "<=", visor_half.tolist(),
        "mm", visor_rule, visor_status, visor_display, visor_rule, visor_reason,
        "visor_active_mm")

    required_rays = int(n["ray_count"])
    converged_rays = int(n["forward_converged"])
    physical_rays = int(n.get("forward_physical_valid", converged_rays))
    forward_pass = bool(n.get("physical_sequential_validity", False))
    forward_status = "PASS" if forward_pass else "FAIL"
    forward_display = (f"converged={converged_rays}/{required_rays}; "
                       f"physical-order={physical_rays}/{required_rays}")
    forward_rule = f"converged rays == {required_rays}/{required_rays}"
    if forward_pass:
        forward_reason = f"PASS: every one of {required_rays} rays completed the physical sequence"
    else:
        forward_reason = (f"FAIL: missing {max(0, required_rays-converged_rays)} converged ray(s); "
                          f"missing {max(0, required_rays-physical_rays)} physical-order ray(s)")
    add("PHYSICAL_SEQUENTIAL_FORWARD_TRACE", converged_rays, "==", required_rays,
        "rays", f"{required_rays}/{required_rays}", forward_status, forward_display,
        forward_rule, forward_reason, "ray_count / physical_sequential_validity")

    fermat_history = list(n.get("fermat_convergence_history", []))
    fermat_pass = bool(n.get("fermat_all_converged", False))
    fermat_status = "PASS" if fermat_pass else "FAIL"
    solve_total = len(fermat_history)
    solve_pass = sum(bool(x.get("all_converged", False)) for x in fermat_history)
    worst_success = min((int(x.get("success_count", 0)) for x in fermat_history),
                        default=required_rays if fermat_pass else 0)
    worst_gradient = min((int(x.get("gradient_pass_count", 0)) for x in fermat_history),
                         default=required_rays if fermat_pass else 0)
    fermat_display = (f"accepted solves={solve_pass}/{solve_total}; "
                      f"worst solver/gradient={worst_success}/{worst_gradient} of {required_rays}")
    fermat_rule = f"every accepted solve: solver and gradient counts == {required_rays}"
    if fermat_pass:
        fermat_reason = f"PASS: all {solve_total} accepted Fermat solve(s) converged"
    else:
        failed = next((str(x.get("solve", "unnamed solve")) for x in fermat_history
                       if not bool(x.get("all_converged", False))), "accepted-path convergence flag is false")
        fermat_reason = f"FAIL: {failed} did not fully converge"
    add("FERMAT_STATIONARY_POINT_CONVERGENCE", fermat_pass, "==", required_rays,
        "rays/solve", f"{required_rays}/{required_rays} for every solve on accepted construction path",
        fermat_status, fermat_display, fermat_rule, fermat_reason,
        "fermat_all_converged + fermat_convergence_history")

    if config["mtf_requested"]:
        mtf = n["MTF"]
        mtf_actual = mtf.get("minimum_MTF_at_max_frequency")
        mtf_limit = float(config["mtf"]["minimum_at_max_frequency"])
        if mtf_limit > 0.0:
            mtf_pass = bool(mtf.get("complete", False) and mtf_actual is not None
                            and float(mtf_actual) >= mtf_limit)
            mtf_status = "PASS" if mtf_pass else "FAIL"
            observed = ">=" if mtf_pass else "<"
            mtf_rule = f"minimum MTF >= {mtf_limit:g}"
            mtf_reason = (f"{mtf_status}: minimum MTF {_score_value(mtf_actual)} "
                          f"{observed} {_score_value(mtf_limit)}")
            mtf_criterion = f">={mtf_limit:g}"
            mtf_operator: str | None = ">="
            mtf_threshold: Any = mtf_limit
        else:
            mtf_status = "UNGRADED"
            mtf_rule = "no pass/fail threshold (configured value is 0)"
            mtf_reason = "UNGRADED: set mtf.minimum_at_max_frequency > 0 to grade this metric"
            mtf_criterion = "TARGET_REPORTED_BUT_NO_USER_TOLERANCE"
            mtf_operator = None
            mtf_threshold = None
        add("NUMERICAL_MTF_AT_MAX_FREQUENCY", mtf_actual, mtf_operator, mtf_threshold,
            "MTF", mtf_criterion, mtf_status,
            f"minimum MTF={_score_value(mtf_actual)}", mtf_rule, mtf_reason,
            "mtf.minimum_at_max_frequency")

    # MF1 is the all-characteristic-ray chief-centered image-quality merit.
    # Pupil chief spread and field bias are reported separately so a small
    # within-pupil spot cannot hide eyebox image shift or mapping bias.
    final_mf1 = n.get(
        "final_mf1",
        {},
    )

    if final_mf1:
        reverse_valid = int(
            final_mf1.get(
                "physical_valid_ray_count",
                0,
            )
        )

        reverse_total = int(
            final_mf1.get(
                "active_ray_count",
                required_rays,
            )
        )

        certified_mf1 = (
            final_mf1.get(
                "certified_physical_MF1_Fan"
            )
        )

        mf1_display = (
            f"chief-centered all-ray MF1="
            f"{_score_value(final_mf1.get('MF1_Fan'))}; "
            f"physical-valid={reverse_valid}/{reverse_total}; "
            f"valid-only chief RMS="
            f"{_score_value(final_mf1.get('physical_valid_global_RMS_diagnostic_mm'), 'mm')}; "
            f"chief-pupil spread RMS="
            f"{_score_value(final_mf1.get('chief_pupil_spread_RMS_mm'), 'mm')}; "
            f"chief-field bias RMS="
            f"{_score_value(final_mf1.get('chief_field_bias_RMS_mm'), 'mm')}"
        )

        mf1_reason = (
            "UNGRADED QUALITY: reverse first-hit coverage is complete; "
            "chief-centered all-ray MF1 is physically certified"
            if certified_mf1 is not None
            else
            "UNGRADED QUALITY: chief-centered all-ray MF1 is numerical only "
            "because reverse first-hit coverage is incomplete"
        )

        add(
            "MF1_FAN_IMAGE_QUALITY",
            final_mf1.get(
                "MF1_Fan"
            ),
            None,
            None,
            "merit",
            "CHIEF_CENTERED_QUALITY_REPORTED_WITH_SEPARATE_PUPIL_SHIFT_FIELD_BIAS_AND_PHYSICAL_GATES",
            "UNGRADED",
            mf1_display,
            "no user chief-centered MF1 quality threshold; never drop invalid rays to create a pass",
            mf1_reason,
            "fan_weights.omega1 + final reverse physical-valid mask",
        )

    # Re-evaluate the signed obscuration geometry on the final surfaces.  This
    # closes the certification gap where an obscured incumbent could otherwise
    # survive a no-improvement optimization and remain absent from STEP 25.
    final_mf2 = n.get("final_mf2")
    signed_area = None if not final_mf2 else final_mf2.get("S_AQP_signed_mm2")
    mf2_evaluable = signed_area is not None and np.isfinite(float(signed_area))
    mf2_pass = bool(mf2_evaluable and float(signed_area) >= 0.0)
    mf2_status = "PASS" if mf2_pass else "FAIL"
    mf2_display = f"S_AQP={_score_value(signed_area, 'mm^2')}"
    if not mf2_evaluable:
        mf2_reason = "FAIL: final signed A-Q-P obscuration area is not finite/evaluable"
    elif mf2_pass:
        mf2_reason = f"PASS: S_AQP {_score_value(signed_area, 'mm^2')} >= 0 mm^2"
    else:
        mf2_reason = (f"FAIL: S_AQP {_score_value(signed_area, 'mm^2')} < 0 mm^2; "
                      f"obscuration violation={_score_value(-float(signed_area), 'mm^2')}")
    add("SIGNED_MF2_OBSCURATION", signed_area, ">=", 0.0, "mm^2", ">=0",
        mf2_status, mf2_display, "S_AQP >= 0 mm^2", mf2_reason,
        "final reverse trace + fan_weights.omega2")

    optics = n["optics"]
    achieved = optics["achieved_mean"]
    optics_keys = {
        "FOV_H_deg": ("FOV_H_deg", "fov_h_deg"),
        "FOV_V_deg": ("FOV_V_deg", "fov_v_deg"),
        "VID_mm": ("VID_center_mm", "vid_mm"),
        "D6_deg": ("D6_center_deg", "d6_deg"),
        "azimuth_deg": ("azimuth_center_deg", "azimuth_deg"),
    }
    tolerances = config["hard_tolerances"]
    for item, (achieved_key, target_key) in optics_keys.items():
        actual = achieved.get(achieved_key)
        target = float(config[target_key])
        tolerance = tolerances.get(target_key)
        unit = "mm" if item == "VID_mm" else "deg"
        config_source = f"hard_tolerances.{target_key}"
        if tolerance is None:
            add(item, actual, None, None, unit,
                "TARGET_REPORTED_BUT_NO_USER_TOLERANCE", "UNGRADED",
                f"actual={_score_value(actual, unit)}; target={_score_value(target, unit)}",
                "no declared tolerance",
                f"UNGRADED: {config_source} is null; metric is reported only",
                config_source)
            continue
        tolerance = float(tolerance)
        actual_finite = actual is not None and np.isfinite(float(actual))
        if actual_finite:
            raw_delta = float(actual) - target
            # Circular shortest-distance error avoids a false 359-degree error
            # for azimuths on opposite sides of the +/-180-degree boundary.
            delta = ((raw_delta + 180.0) % 360.0 - 180.0
                     if item == "azimuth_deg" else raw_delta)
            error = abs(delta)
            tolerance_pass = error <= tolerance
        else:
            error = None
            tolerance_pass = False
        tolerance_status = "PASS" if tolerance_pass else "FAIL"
        actual_display = (f"actual={_score_value(actual, unit)}; target={_score_value(target, unit)}; "
                          f"|error|={_score_value(error, unit)}")
        rule_display = f"|actual - target| <= {_score_value(tolerance, unit)}"
        if not actual_finite:
            reason = f"FAIL: actual {item} is not evaluable from complete forward bundles"
        else:
            observed = "<=" if tolerance_pass else ">"
            reason = (f"{tolerance_status}: |error| {_score_value(error, unit)} {observed} "
                      f"tolerance {_score_value(tolerance, unit)}")
        add(item, error, "<=", tolerance, unit,
            f"absolute error <= {tolerance:g} {unit}", tolerance_status,
            actual_display, rule_display, reason, config_source)
    return rows


def centroided_geometric_spot_rms(landing: np.ndarray, valid: np.ndarray,
                                  field_index: np.ndarray, pupil_index: np.ndarray,
                                  display_frame: np.ndarray) -> dict[str, Any]:
    """Tính RMS spot hình học quanh centroid của từng bundle field/pupil.

    Đây là metric spot độc lập với Fan reference: mỗi bundle có centroid riêng,
    rồi gộp bình phương bán kính của toàn bộ ray hợp lệ. Nó phù hợp để đối chiếu
    planar image quality kiểu Fig. 8(a) của Fan hơn MF1 mapping residual.
    """
    landing = np.asarray(landing, float)
    valid = np.asarray(valid, bool)
    field_index = np.asarray(field_index, int)
    pupil_index = np.asarray(pupil_index, int)
    frame = np.asarray(display_frame, float)
    if (landing.ndim != 2 or landing.shape[1] != 3 or valid.shape != (len(landing),)
            or field_index.shape != (len(landing),) or pupil_index.shape != (len(landing),)
            or frame.shape != (3, 3)):
        raise ValueError("CENTROIDED_SPOT_RMS_INPUT_SHAPE_MISMATCH")
    local_xy = (landing @ frame)[:, :2]
    finite = np.all(np.isfinite(local_xy), axis=1)
    usable = valid & finite
    bundle_rows: list[dict[str, Any]] = []
    squared_radius: list[np.ndarray] = []
    for field in range(int(np.max(field_index)) + 1 if len(field_index) else 0):
        for pupil in range(int(np.max(pupil_index)) + 1 if len(pupil_index) else 0):
            bundle = (field_index == field) & (pupil_index == pupil)
            good = bundle & usable
            if not np.any(good):
                bundle_rows.append({"field_index": field, "pupil_index": pupil,
                                    "ray_count": int(np.sum(bundle)), "valid_count": 0,
                                    "centroid_local_mm": None,
                                    "RMS_spot_radius_mm": None,
                                    "max_spot_radius_mm": None})
                continue
            centroid = np.mean(local_xy[good], axis=0)
            radius2 = np.sum((local_xy[good] - centroid) ** 2, axis=1)
            squared_radius.append(radius2)
            bundle_rows.append({"field_index": field, "pupil_index": pupil,
                                "ray_count": int(np.sum(bundle)), "valid_count": int(np.sum(good)),
                                "centroid_local_mm": centroid,
                                "RMS_spot_radius_mm": float(np.sqrt(np.mean(radius2))),
                                "max_spot_radius_mm": float(np.sqrt(np.max(radius2)))})
    pooled = np.concatenate(squared_radius) if squared_radius else np.empty(0, float)
    return {
        "metric_name": "CENTROIDED_GEOMETRIC_SPOT_RMS_RADIUS_ALL_FIELD_PUPIL_RAYS",
        "aggregation": ("sqrt(mean_over_all_valid_rays(||landing_display_xy - "
                        "centroid_of_its_field_pupil_bundle||^2))"),
        "unit": "mm", "ray_count": int(len(landing)),
        "valid_ray_count": int(np.sum(usable)),
        "field_count": int(np.max(field_index)) + 1 if len(field_index) else 0,
        "pupil_count": int(np.max(pupil_index)) + 1 if len(pupil_index) else 0,
        "RMS_spot_radius_mm": float(np.sqrt(np.mean(pooled))) if len(pooled) else None,
        "max_spot_radius_mm": float(np.sqrt(np.max(pooled))) if len(pooled) else None,
        "bundle_rows": bundle_rows,
    }


def chief_centered_geometric_spot_rms(landing: np.ndarray, valid: np.ndarray,
                                      field_index: np.ndarray, pupil_index: np.ndarray,
                                      chief: np.ndarray, display_frame: np.ndarray) -> dict[str, Any]:
    """Tính RMS spot hình học quanh chief ray của từng bundle field/pupil."""
    landing = np.asarray(landing, float)
    valid = np.asarray(valid, bool)
    field_index = np.asarray(field_index, int)
    pupil_index = np.asarray(pupil_index, int)
    chief = np.asarray(chief, bool)
    frame = np.asarray(display_frame, float)
    if (landing.ndim != 2 or landing.shape[1] != 3 or valid.shape != (len(landing),)
            or field_index.shape != (len(landing),) or pupil_index.shape != (len(landing),)
            or chief.shape != (len(landing),) or frame.shape != (3, 3)):
        raise ValueError("CHIEF_CENTERED_SPOT_RMS_INPUT_SHAPE_MISMATCH")

    local_xy = (landing @ frame)[:, :2]
    finite = np.all(np.isfinite(local_xy), axis=1)
    usable = valid & finite

    bundle_rows: list[dict[str, Any]] = []
    squared_radius: list[np.ndarray] = []

    field_count = int(np.max(field_index)) + 1 if len(field_index) else 0
    pupil_count = int(np.max(pupil_index)) + 1 if len(pupil_index) else 0

    for field in range(field_count):
        for pupil in range(pupil_count):
            bundle = (field_index == field) & (pupil_index == pupil)
            chief_indices = np.where(bundle & chief)[0]

            if len(chief_indices) != 1:
                raise RuntimeError(
                    "CHIEF_CENTERED_SPOT_CHIEF_SELECTION_MISMATCH"
                )

            chief_index = int(chief_indices[0])
            good = bundle & usable

            if not usable[chief_index] or not np.any(good):
                bundle_rows.append({
                    "field_index": field,
                    "pupil_index": pupil,
                    "ray_count": int(np.sum(bundle)),
                    "valid_count": int(np.sum(good)),
                    "chief_ray_index": chief_index,
                    "chief_valid": bool(usable[chief_index]),
                    "RMS_to_chief_mm": None,
                    "max_radius_to_chief_mm": None,
                })
                continue

            radius2 = np.sum(
                (local_xy[good] - local_xy[chief_index]) ** 2,
                axis=1,
            )

            squared_radius.append(radius2)

            bundle_rows.append({
                "field_index": field,
                "pupil_index": pupil,
                "ray_count": int(np.sum(bundle)),
                "valid_count": int(np.sum(good)),
                "chief_ray_index": chief_index,
                "chief_valid": True,
                "RMS_to_chief_mm": float(
                    np.sqrt(np.mean(radius2))
                ),
                "max_radius_to_chief_mm": float(
                    np.sqrt(np.max(radius2))
                ),
            })

    pooled = (
        np.concatenate(squared_radius)
        if squared_radius
        else np.empty(0, float)
    )

    return {
        "metric_name":
            "CHIEF_CENTERED_GEOMETRIC_SPOT_RMS_ALL_FIELD_PUPIL_RAYS",

        "aggregation": (
            "sqrt(mean_over_all_valid_rays("
            "||landing_display_xy - "
            "chief_landing_xy_of_its_field_pupil_bundle||^2))"
        ),

        "unit": "mm",
        "ray_count": int(len(landing)),
        "valid_ray_count": int(np.sum(usable)),
        "field_count": field_count,
        "pupil_count": pupil_count,

        "RMS_spot_radius_mm": (
            float(np.sqrt(np.mean(pooled)))
            if len(pooled)
            else None
        ),

        "max_spot_radius_mm": (
            float(np.sqrt(np.max(pooled)))
            if len(pooled)
            else None
        ),

        "bundle_rows": bundle_rows,
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Tính SHA-256 của tệp để khóa chính xác đầu vào."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_json(data: Any) -> str:
    """Tính SHA-256 ổn định của đối tượng JSON đã sắp khóa."""
    raw = json.dumps(as_jsonable(data), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def versions() -> dict[str, str]:
    """Ghi phiên bản Python và các thư viện số của lần chạy."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def repair_text(s: str) -> str:
    """Sửa lỗi chuỗi UTF-8 từng bị đọc nhầm theo cp1252."""
    if not isinstance(s, str):
        return s
    if any(mark in s for mark in ("Ã", "Â", "Ä", "Æ", "á»")):
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def load_inputs(input_dir: Path, geometry_authority: dict[str, Any]) -> dict[str, Any]:
    """Nạp đúng hai file visor sạch; hình học HUD còn lại lấy từ config có truy vết."""
    canonical_path = input_dir / "Visor2_016_ULTRA_SMOOTH_CLEAN_CANONICAL.json"
    verification_path = input_dir / "Visor2_016_ULTRA_SMOOTH_CLEAN_VERIFICATION.json"
    if not canonical_path.is_file():
        local_parent = Path(__file__).resolve().parent.parent / "Visor2_016_ULTRA_SMOOTH_CLEAN_CANONICAL.json"
        local_same = Path(__file__).resolve().parent / "Visor2_016_ULTRA_SMOOTH_CLEAN_CANONICAL.json"
        if local_parent.is_file():
            canonical_path = local_parent
            verification_path = local_parent.parent / "Visor2_016_ULTRA_SMOOTH_CLEAN_VERIFICATION.json"
        elif local_same.is_file():
            canonical_path = local_same
            verification_path = local_same.parent / "Visor2_016_ULTRA_SMOOTH_CLEAN_VERIFICATION.json"
    files = [canonical_path, verification_path]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f"STOP_MISSING_INPUT: required clean visor files missing: {missing}")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (verification.get("status") != "PASS"
            or verification.get("authoritative_dataset") != canonical_path.name):
        raise RuntimeError("STOP_INVALID_INPUT: CLEAN_VERIFICATION does not certify CLEAN_CANONICAL")
    paired = json.loads(canonical_path.read_text(encoding="utf-8"))
    if not isinstance(paired, list) or len(paired) != int(verification.get("point_count", -1)):
        raise RuntimeError("STOP_INVALID_INPUT: canonical point count disagrees with verification")
    required_fields = {
        "u_mm", "v_mm", "INNER_X_mm", "INNER_Y_mm", "INNER_Z_mm",
        "INNER_NX", "INNER_NY", "INNER_NZ", "OUTER_X_mm", "OUTER_Y_mm",
        "OUTER_Z_mm", "OUTER_NX", "OUTER_NY", "OUTER_NZ", "thickness_normal_mm",
    }
    if not paired or any(not required_fields.issubset(row) for row in paired):
        raise RuntimeError("STOP_INVALID_INPUT: canonical visor rows do not contain all required fields")

    # P1-P8 là bao cơ khí tùy chọn.  null, [] hoặc bỏ khóa này nghĩa là
    # không áp constraint packaging; tuyệt đối không tự nạp lại tám điểm cũ.
    packaging_raw = geometry_authority.get("packaging_vertices_mm")
    packaging = (None if packaging_raw is None or packaging_raw == []
                 else np.asarray(packaging_raw, dtype=float))
    center = np.asarray(geometry_authority["visor_center_mm"], dtype=float)
    U = unit(np.asarray(geometry_authority["visor_U"], dtype=float))
    V = unit(np.asarray(geometry_authority["visor_V"], dtype=float))
    N = unit(np.asarray(geometry_authority["visor_N"], dtype=float))
    if (packaging is not None and packaging.shape != (8, 3)) or center.shape != (3,):
        raise RuntimeError("STOP_INVALID_INPUT: HUD geometry authority has an invalid shape")
    if (abs(float(np.dot(U, V))) > 1e-5 or abs(float(np.dot(U, N))) > 1e-5
            or abs(float(np.dot(V, N))) > 1e-5 or float(np.dot(np.cross(U, V), N)) < 0.99999):
        raise RuntimeError("STOP_INVALID_INPUT: visor U/V/N frame is not right-handed orthonormal")

    uv = np.array([[r["u_mm"], r["v_mm"]] for r in paired], dtype=float)
    inner = np.array([[r["INNER_X_mm"], r["INNER_Y_mm"], r["INNER_Z_mm"]] for r in paired], dtype=float)
    inner_n = np.array([[r["INNER_NX"], r["INNER_NY"], r["INNER_NZ"]] for r in paired], dtype=float)
    outer = np.array([[r["OUTER_X_mm"], r["OUTER_Y_mm"], r["OUTER_Z_mm"]] for r in paired], dtype=float)
    outer_n = np.array([[r["OUTER_NX"], r["OUTER_NY"], r["OUTER_NZ"]] for r in paired], dtype=float)
    thickness = np.array([r["thickness_normal_mm"] for r in paired], dtype=float)
    nu, nv = len(np.unique(uv[:, 0])), len(np.unique(uv[:, 1]))
    if nu * nv != len(uv):
        raise RuntimeError("STOP_MISSING_INPUT: paired visor samples are not a complete regular grid")

    return {
        "files": files,
        "hashes": {p.name: sha256_file(p) for p in files},
        "clean_verification": verification,
        "geometry_authority_source": geometry_authority["source"],
        "packaging_constraint_enabled": packaging is not None,
        "packaging_vertices": packaging,
        "visor_center": center,
        "visor_U": U,
        "visor_V": V,
        "visor_N": N,
        "visor_uv": uv,
        "visor_inner": inner,
        "visor_inner_normals": inner_n,
        "visor_outer": outer,
        "visor_outer_normals": outer_n,
        "visor_thickness": thickness,
        "visor_grid_shape": (nv, nu),
    }


def local_frame_from_normal(normal: np.ndarray, preferred_x: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Dựng hệ trục tiếp tuyến trực chuẩn từ pháp tuyến bề mặt."""
    n = unit(normal)
    x = preferred_x - np.dot(preferred_x, n) * n
    if np.linalg.norm(x) < 1e-6:
        preferred_x = np.array([0.0, 0.0, 1.0])
        x = preferred_x - np.dot(preferred_x, n) * n
    x = unit(x)
    y = unit(np.cross(n, x))
    x = unit(np.cross(y, n))
    return np.column_stack([x, y, n])


@dataclass
class ChebVisor:
    """Mô hình visor Chebyshev được fit từ point cloud authority."""
    center: np.ndarray
    frame: np.ndarray
    coeff: np.ndarray
    scale: tuple[float, float]
    bounds: tuple[float, float, float, float]

    @classmethod
    def fit(cls, center: np.ndarray, U: np.ndarray, V: np.ndarray, N: np.ndarray,
            xyz: np.ndarray, degree: int, scale: tuple[float, float]) -> tuple["ChebVisor", dict[str, Any], np.ndarray]:
        """Fit hệ số bề mặt từ dữ liệu bằng bình phương tối thiểu."""
        frame = np.column_stack([unit(U), unit(V), unit(N)])
        handed = float(np.linalg.det(frame))
        if handed < 0:
            frame[:, 2] *= -1
        loc = (xyz - center) @ frame
        x, y, z = loc.T
        sx, sy = scale
        A = cheb.chebvander2d(x / sx, y / sy, [degree, degree]).reshape(len(x), -1)
        coeff = np.linalg.lstsq(A, z, rcond=1e-13)[0].reshape(degree + 1, degree + 1)
        model = cls(np.asarray(center), frame, coeff, (sx, sy), (x.min(), x.max(), y.min(), y.max()))
        fit_z = model.sag(x, y)
        stats = {
            "representation": "GLOBAL_CHEBYSHEV_GRAPH_C2",
            "degree_axis_x": degree,
            "degree_axis_y": degree,
            "sag_rms_mm": float(np.sqrt(np.mean((fit_z - z) ** 2))),
            "sag_max_mm": float(np.max(np.abs(fit_z - z))),
            "handedness_determinant": handed,
            "orthogonality_max_abs": float(np.max(np.abs(frame.T @ frame - np.eye(3)))),
        }
        return model, stats, loc

    def _dcoeff(self, axis: int, order: int = 1) -> np.ndarray:
        """Tạo hệ số đạo hàm Chebyshev theo trục yêu cầu."""
        c = cheb.chebder(self.coeff, m=order, axis=axis)
        return c / (self.scale[axis] ** order)

    def sag(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tính độ võng bề mặt tại tọa độ cục bộ."""
        return cheb.chebval2d(np.asarray(x) / self.scale[0], np.asarray(y) / self.scale[1], self.coeff)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tính hai đạo hàm riêng bậc nhất của độ võng."""
        xn = np.asarray(x) / self.scale[0]
        yn = np.asarray(y) / self.scale[1]
        return (
            cheb.chebval2d(xn, yn, self._dcoeff(0, 1)),
            cheb.chebval2d(xn, yn, self._dcoeff(1, 1)),
        )

    def derivatives(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        """Tính hai đạo hàm riêng của độ võng."""
        xn, yn = np.asarray(x) / self.scale[0], np.asarray(y) / self.scale[1]
        gx = cheb.chebval2d(xn, yn, self._dcoeff(0, 1))
        gy = cheb.chebval2d(xn, yn, self._dcoeff(1, 1))
        gxx = cheb.chebval2d(xn, yn, self._dcoeff(0, 2))
        gxy = cheb.chebval2d(xn, yn, cheb.chebder(cheb.chebder(self.coeff, axis=0), axis=1) / (self.scale[0] * self.scale[1]))
        gyy = cheb.chebval2d(xn, yn, self._dcoeff(1, 2))
        return gx, gy, gxx, gxy, gyy

    def point(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Đổi tọa độ cục bộ thành điểm 3D trên bề mặt."""
        local = np.column_stack([x, y, self.sag(x, y)])
        return self.center + local @ self.frame.T

    def normal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tính pháp tuyến đơn vị tại điểm trên bề mặt."""
        gx, gy = self.gradient(x, y)
        nloc = unit(np.column_stack([-gx, -gy, np.ones_like(gx)]))
        return nloc @ self.frame.T

    def intersect(self, origins: np.ndarray, directions: np.ndarray, finite: bool = True,
                   t_min: float = 1e-6, iterations: int = 12) -> dict[str, np.ndarray]:
        """Return the nearest positive root of the exact Chebyshev ray polynomial.

        ``iterations`` remains in the compatible signature but is not used: along a
        straight ray a tensor Chebyshev graph is a finite univariate polynomial, so
        all roots (including tangencies and close pairs) can be isolated together.
        """
        del iterations
        O, D, usable, finite_input = _normalized_ray_batch(origins, directions)
        parallel_result = _parallel_cheb_visor_intersect(
            self,
            O,
            D,
            finite,
            t_min,
        )
        if parallel_result is not None:
            return parallel_result
        ol = (O-self.center)@self.frame
        dl = D@self.frame
        count = len(O)
        t = np.full(count, np.nan); x = np.full(count, np.nan); y = np.full(count, np.nan)
        residual = np.full(count, np.nan); valid = np.zeros(count, bool)
        resolved = np.zeros(count, bool); candidate_count = np.zeros(count, int)
        outside_count = np.zeros(count, int)
        status = np.full(count, "NONFINITE_INPUT", dtype=object)
        status[finite_input & ~usable] = "ZERO_DIRECTION"
        xmin, xmax, ymin, ymax = map(float, self.bounds)

        equation_rows = None
        usable_ids = np.flatnonzero(usable)

        if (
            len(usable_ids)
            and selected_backend("cheb_ray_equations") != "reference"
        ):
            equation_rows = dispatch_kernel(
                "cheb_ray_equations",
                {
                    "coeff": np.asarray(self.coeff, dtype=float),
                    "scale": np.asarray(self.scale, dtype=float),
                    "ol": ol[usable_ids],
                    "dl": dl[usable_ids],
                },
            )
            if equation_rows.shape != (
                len(usable_ids),
                max(2, self.coeff.shape[0] + self.coeff.shape[1] - 1),
            ):
                raise BackendExecutionError("CHEB_OUTPUT_SHAPE_MISMATCH")

        for local_index, index in enumerate(progress_indices(
            usable_ids,
            "ChebVisor.intersect",
            batch_total=len(O),
        )):
            if equation_rows is None:
                tx = Polynomial([ol[index, 0]/self.scale[0], dl[index, 0]/self.scale[0]])
                ty = Polynomial([ol[index, 1]/self.scale[1], dl[index, 1]/self.scale[1]])
                cheb_x = [Polynomial([1.0]), tx]
                cheb_y = [Polynomial([1.0]), ty]
                for degree in range(2, self.coeff.shape[0]):
                    cheb_x.append(2.0*tx*cheb_x[-1]-cheb_x[-2])
                for degree in range(2, self.coeff.shape[1]):
                    cheb_y.append(2.0*ty*cheb_y[-1]-cheb_y[-2])
                sag_poly = Polynomial([0.0])
                for ix in range(self.coeff.shape[0]):
                    for iy in range(self.coeff.shape[1]):
                        sag_poly += float(self.coeff[ix, iy])*cheb_x[ix]*cheb_y[iy]
                equation = Polynomial([ol[index, 2], dl[index, 2]])-sag_poly
            else:
                equation = Polynomial(equation_rows[local_index])
            roots, root_status = _polynomial_real_roots(equation)
            if root_status.startswith("UNRESOLVED") or root_status.startswith("DEGENERATE"):
                status[index] = root_status
                continue
            positive = roots[roots > float(t_min)]
            original_candidates: list[tuple[float, float, float, float]] = []
            for root in positive:
                t_val = float(root)
                xr = float(ol[index, 0] + t_val * dl[index, 0])
                yr = float(ol[index, 1] + t_val * dl[index, 1])
                if abs(xr / self.scale[0]) > 2.0 or abs(yr / self.scale[1]) > 2.0:
                    continue
                zr = float(ol[index, 2] + t_val * dl[index, 2])
                sz = float(self.sag(np.array([xr]), np.array([yr]))[0])
                rr = zr - sz
                for _ in range(5):
                    if abs(rr) <= INTERSECTION_RESIDUAL_TOLERANCE_MM:
                        break
                    gx, gy = self.gradient(np.array([xr]), np.array([yr]))
                    f_prime = float(dl[index, 2] - gx[0] * dl[index, 0] - gy[0] * dl[index, 1])
                    if abs(f_prime) < 1e-12:
                        break
                    step = rr / f_prime
                    if abs(step) > 2.0:
                        break
                    t_new = t_val - step
                    xr_new = float(ol[index, 0] + t_new * dl[index, 0])
                    yr_new = float(ol[index, 1] + t_new * dl[index, 1])
                    zr_new = float(ol[index, 2] + t_new * dl[index, 2])
                    sz_new = float(self.sag(np.array([xr_new]), np.array([yr_new]))[0])
                    rr_new = zr_new - sz_new
                    if abs(rr_new) < abs(rr):
                        t_val, xr, yr, zr, rr = t_new, xr_new, yr_new, zr_new, rr_new
                    else:
                        break

                if not np.isfinite(rr) or abs(rr) > 1e-4:
                    continue
                candidate_count[index] += 1
                in_aperture = (xmin - APERTURE_BOUNDARY_TOLERANCE_MM <= xr
                               <= xmax + APERTURE_BOUNDARY_TOLERANCE_MM
                               and ymin - APERTURE_BOUNDARY_TOLERANCE_MM <= yr
                               <= ymax + APERTURE_BOUNDARY_TOLERANCE_MM)
                if finite and not in_aperture:
                    outside_count[index] += 1
                    continue
                original_candidates.append((t_val, xr, yr, rr))
            resolved[index] = True
            if original_candidates:
                chosen = min(original_candidates, key=lambda row: row[0])
                t[index], x[index], y[index], residual[index] = chosen
                valid[index] = True; status[index] = "HIT"
            elif outside_count[index]:
                status[index] = "OUTSIDE_CLEAR_APERTURE"
            else:
                status[index] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"

        point = np.full((count, 3), np.nan); normal = np.full((count, 3), np.nan)
        point[valid] = O[valid]+t[valid, None]*D[valid]
        if np.any(valid):
            normal[valid] = self.normal(x[valid], y[valid])
        return {"t": t, "point": point, "normal": normal, "x": x, "y": y,
                "residual": residual, "valid": valid, "resolved": resolved,
                "status": status, "candidate_count": candidate_count,
                "outside_aperture_candidate_count": outside_count,
                "root_method": "COMPLETE_CHEBYSHEV_RAY_POLYNOMIAL"}

    def to_dict(self) -> dict[str, Any]:
        """Tuần tự hóa mô hình bề mặt để lưu checkpoint."""
        return {"type": "GLOBAL_CHEBYSHEV_GRAPH_C2", "center_mm": self.center, "frame_columns": self.frame,
                "coefficients": self.coeff, "scale_mm": self.scale, "bounds_mm": self.bounds}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ChebVisor":
        """Thực thi from_dict."""
        if data.get("type") != "GLOBAL_CHEBYSHEV_GRAPH_C2":
            raise ValueError(
                "INVALID_CHEB_VISOR_TYPE"
            )

        return cls(
            np.asarray(
                data["center_mm"],
                dtype=float,
            ),
            np.asarray(
                data["frame_columns"],
                dtype=float,
            ),
            np.asarray(
                data["coefficients"],
                dtype=float,
            ),
            tuple(
                map(
                    float,
                    data["scale_mm"],
                )
            ),
            tuple(
                map(
                    float,
                    data["bounds_mm"],
                )
            ),
        )


def _parallel_cheb_visor_intersect(
    visor: "ChebVisor",
    O: np.ndarray,
    D: np.ndarray,
    finite: bool,
    t_min: float,
) -> dict[str, np.ndarray] | None:
    """Thực thi song song phân đoạn chùm tia trên ChebVisor nếu đủ điều kiện."""
    runtime = current_runtime()
    if runtime is None or is_compute_worker():
        return None
    executor = getattr(runtime, "_ray_trace_executor", None)
    if executor is None:
        return None

    ray_count = len(O)
    min_rays = int(getattr(runtime, "_ray_trace_min_rays", 512))
    if ray_count < min_rays:
        return None

    max_inflight = int(getattr(runtime, "_ray_trace_max_inflight", 1))
    from execution_v55 import compute_dynamic_ray_chunk_size
    chunk_size = compute_dynamic_ray_chunk_size(
        ray_count,
        max_inflight,
        runtime.config,
    )

    from execution_workers_v55 import ordered_bounded_map, cheb_visor_intersect_worker

    visor_dict = visor.to_dict()
    jobs = []
    chunk_idx = 0
    for start in range(0, ray_count, chunk_size):
        end = min(start + chunk_size, ray_count)
        jobs.append({
            "chunk_index": chunk_idx,
            "start": start,
            "end": end,
            "visor": visor_dict,
            "origins": O[start:end],
            "directions": D[start:end],
            "finite": bool(finite),
            "t_min": float(t_min),
        })
        chunk_idx += 1

    runtime.record(
        "PARALLEL_CHEB_INTERSECT_STARTED",
        ray_count=int(ray_count),
        chunk_count=int(len(jobs)),
        chunk_size=int(chunk_size),
        dynamic_chunk_size=int(chunk_size),
        worker_count=int(max_inflight),
        target_job_count=int(len(jobs)),
        max_inflight=int(max_inflight),
        finite=bool(finite),
    )

    replies = ordered_bounded_map(
        executor,
        cheb_visor_intersect_worker,
        jobs,
        max_inflight,
    )

    if not replies:
        return None

    runtime.record(
        "PARALLEL_CHEB_INTERSECT_FINISHED",
        ray_count=int(ray_count),
        chunk_count=int(len(jobs)),
    )

    t = np.concatenate([r["result"]["t"] for r in replies])
    point = np.concatenate([r["result"]["point"] for r in replies])
    normal = np.concatenate([r["result"]["normal"] for r in replies])
    x = np.concatenate([r["result"]["x"] for r in replies])
    y = np.concatenate([r["result"]["y"] for r in replies])
    residual = np.concatenate([r["result"]["residual"] for r in replies])
    valid = np.concatenate([r["result"]["valid"] for r in replies])
    resolved = np.concatenate([r["result"]["resolved"] for r in replies])
    status = np.concatenate([r["result"]["status"] for r in replies])
    candidate_count = np.concatenate([r["result"]["candidate_count"] for r in replies])
    outside_count = np.concatenate([r["result"]["outside_aperture_candidate_count"] for r in replies])
    root_method = replies[0]["result"].get("root_method", "COMPLETE_CHEBYSHEV_RAY_POLYNOMIAL")

    return {
        "t": t,
        "point": point,
        "normal": normal,
        "x": x,
        "y": y,
        "residual": residual,
        "valid": valid,
        "resolved": resolved,
        "status": status,
        "candidate_count": candidate_count,
        "outside_aperture_candidate_count": outside_count,
        "root_method": root_method,
    }


def monomial_terms(order: int, axis_order2: bool = False, include_constant: bool = True,
                   preserve_fan_axis_order2: bool = False) -> list[tuple[int, int]]:
    """Liệt kê các số mũ đơn thức XY đến bậc đã chọn."""
    if axis_order2:
        terms = [(i, j) for i in range(3) for j in range(3)]
    else:
        terms = [(i, j) for d in range(order + 1) for i in range(d + 1) for j in [d - i]]
        if preserve_fan_axis_order2:
            # Fan Eq.(1) order-2 is an axis-wise tensor basis and contains
            # A22*x^2*y^2 (total degree four).  Keep its exact union with the
            # next total-degree level so CI never drops an active old term.
            fan_order2 = [(i, j) for i in range(3) for j in range(3)]
            terms = fan_order2 + [term for term in terms if term not in fan_order2]
    if not include_constant:
        terms = [t for t in terms if t != (0, 0)]
    return terms


def _parallel_poly_surface_intersect(
    surface: "PolySurface",
    O: np.ndarray,
    D: np.ndarray,
    finite: bool,
    t_min: float,
) -> dict[str, np.ndarray] | None:
    """Thực thi song song phân đoạn chùm tia trên PolySurface nếu đủ điều kiện."""
    runtime = current_runtime()
    if runtime is None or is_compute_worker():
        return None
    executor = getattr(runtime, "_ray_trace_executor", None)
    if executor is None:
        return None

    ray_count = len(O)
    min_rays = int(getattr(runtime, "_ray_trace_min_rays", 512))
    if ray_count < min_rays:
        return None

    max_inflight = int(getattr(runtime, "_ray_trace_max_inflight", 1))
    from execution_v55 import compute_dynamic_ray_chunk_size
    chunk_size = compute_dynamic_ray_chunk_size(
        ray_count,
        max_inflight,
        runtime.config,
    )

    from execution_workers_v55 import ordered_bounded_map, poly_surface_intersect_worker

    surf_dict = surface.to_dict()
    jobs = []
    chunk_idx = 0
    for start in range(0, ray_count, chunk_size):
        end = min(start + chunk_size, ray_count)
        jobs.append({
            "chunk_index": chunk_idx,
            "start": start,
            "end": end,
            "surface": surf_dict,
            "origins": O[start:end],
            "directions": D[start:end],
            "finite": bool(finite),
            "t_min": float(t_min),
        })
        chunk_idx += 1

    runtime.record(
        "PARALLEL_POLY_INTERSECT_STARTED",
        surface=str(surface.name),
        ray_count=int(ray_count),
        chunk_count=int(len(jobs)),
        chunk_size=int(chunk_size),
        dynamic_chunk_size=int(chunk_size),
        worker_count=int(max_inflight),
        target_job_count=int(len(jobs)),
        max_inflight=int(max_inflight),
        finite=bool(finite),
    )

    replies = ordered_bounded_map(
        executor,
        poly_surface_intersect_worker,
        jobs,
        max_inflight,
    )

    if not replies:
        return None

    runtime.record(
        "PARALLEL_POLY_INTERSECT_FINISHED",
        surface=str(surface.name),
        ray_count=int(ray_count),
        chunk_count=int(len(jobs)),
    )

    first_method = replies[0]["result"].get("root_method")
    for rep in replies[1:]:
        if rep["result"].get("root_method") != first_method:
            raise BackendExecutionError("PARALLEL_ROOT_METHOD_MISMATCH")

    t = np.concatenate([r["result"]["t"] for r in replies])
    point = np.concatenate([r["result"]["point"] for r in replies])
    normal = np.concatenate([r["result"]["normal"] for r in replies])
    x = np.concatenate([r["result"]["x"] for r in replies])
    y = np.concatenate([r["result"]["y"] for r in replies])
    residual = np.concatenate([r["result"]["residual"] for r in replies])
    valid = np.concatenate([r["result"]["valid"] for r in replies])
    resolved = np.concatenate([r["result"]["resolved"] for r in replies])
    status = np.concatenate([r["result"]["status"] for r in replies])
    candidate_count = np.concatenate([r["result"]["candidate_count"] for r in replies])
    outside_count = np.concatenate([r["result"]["outside_aperture_candidate_count"] for r in replies])
    conic_invalid_count = np.concatenate([r["result"]["conic_invalid_candidate_count"] for r in replies])
    raw_conic_argument = np.concatenate([r["result"]["raw_conic_argument"] for r in replies])

    return {
        "t": t,
        "point": point,
        "normal": normal,
        "x": x,
        "y": y,
        "residual": residual,
        "valid": valid,
        "resolved": resolved,
        "status": status,
        "candidate_count": candidate_count,
        "outside_aperture_candidate_count": outside_count,
        "conic_invalid_candidate_count": conic_invalid_count,
        "raw_conic_argument": raw_conic_argument,
        "root_method": first_method,
    }


@dataclass
class PolySurface:
    """Mô hình gương gồm conic nền và các hệ số freeform XY."""
    name: str
    center: np.ndarray
    frame: np.ndarray
    half_aperture: np.ndarray
    terms: list[tuple[int, int]]
    coeff: np.ndarray
    scale: np.ndarray
    curvature: float = 0.0
    conic: float = 0.0
    pure_xy20: bool = False
    aperture_polygon: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalize arrays and keep polygon/bounding-box aperture data synchronized."""
        self.center = np.asarray(self.center, float)
        self.frame = np.asarray(self.frame, float)
        self.half_aperture = np.asarray(self.half_aperture, float)
        self.coeff = np.asarray(self.coeff, float)
        self.scale = np.asarray(self.scale, float)
        if self.aperture_polygon is not None:
            self.aperture_polygon = normalize_aperture_polygon(self.aperture_polygon)
            self.half_aperture = aperture_bbox(self.aperture_polygon)
        if (self.half_aperture.shape != (2,) or np.any(~np.isfinite(self.half_aperture))
                or np.any(self.half_aperture <= 0.0)):
            raise ValueError("CLEAR_APERTURE_HALF_SIZE_INVALID")

    @classmethod
    def plane(cls, name: str, center: np.ndarray, normal: np.ndarray, half_aperture=(30.0, 30.0),
              preferred_x=np.array([0.0, 1.0, 0.0])) -> "PolySurface":
        """Khởi tạo mặt phẳng từ tâm, hai trục tiếp tuyến và kích thước chuẩn hóa."""
        return cls(name, np.asarray(center, float), local_frame_from_normal(normal, preferred_x),
                   np.asarray(half_aperture, float), [(0, 0)], np.zeros(1), np.asarray(half_aperture, float))

    @classmethod
    def biconic(
        cls,
        name: str,
        center: np.ndarray,
        frame: np.ndarray,
        half_aperture: np.ndarray,
        rx: float,
        ry: float,
        kx: float = 0.0,
        ky: float = 0.0,
        curvature_sign: float = 1.0,
        terms: list[tuple[int, int]] | None = None,
        scale: np.ndarray | None = None,
        aperture_polygon: np.ndarray | None = None,
    ) -> "PolySurface":
        """Khởi tạo PolySurface dạng biconic/spherical mượt chuẩn (KG > 0)."""
        rx_f = float(rx)
        ry_f = float(ry)
        sign_x = -1.0 if rx_f < 0.0 else 1.0
        sign_y = -1.0 if ry_f < 0.0 else 1.0
        sign_c = -1.0 if float(curvature_sign) < 0.0 else 1.0
        eff_sign_x = sign_x * sign_c
        eff_sign_y = sign_y * sign_c
        rx_val = max(abs(rx_f), 1e-6)
        ry_val = max(abs(ry_f), 1e-6)
        c = eff_sign_x / rx_val
        if terms is None:
            terms = [(i, j) for i in range(3) for j in range(3)]
        coeff = np.zeros(len(terms), dtype=float)
        if abs(rx_f - ry_f) > 1e-9:
            c_y = eff_sign_y / ry_val
            delta_c = 0.5 * (c_y - c)
            if (0, 2) in terms:
                s_y = float(scale[1]) if scale is not None else float(half_aperture[1])
                idx_02 = terms.index((0, 2))
                coeff[idx_02] = delta_c * (s_y ** 2)
        scale_val = np.asarray(half_aperture if scale is None else scale, dtype=float)
        return cls(
            name,
            np.asarray(center, float),
            np.asarray(frame, float),
            np.asarray(half_aperture, float),
            list(terms),
            coeff,
            scale_val,
            curvature=c,
            conic=float(kx),
            pure_xy20=False,
            aperture_polygon=None if aperture_polygon is None else np.asarray(aperture_polygon, float),
        )

    def copy(self) -> "PolySurface":
        """Tạo bản sao độc lập của bề mặt để tối ưu an toàn."""
        return PolySurface(self.name, self.center.copy(), self.frame.copy(), self.half_aperture.copy(),
                           list(self.terms), self.coeff.copy(), self.scale.copy(), self.curvature, self.conic,
                           self.pure_xy20, None if self.aperture_polygon is None else self.aperture_polygon.copy())

    def _conic(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tính sag và đạo hàm của phần conic nền."""
        c, k = self.curvature, self.conic
        r2 = x * x + y * y
        arg = 1.0 - (1.0 + k) * c * c * r2
        arg = np.maximum(arg, 1e-12)
        root = np.sqrt(arg)
        den = 1.0 + root
        z = c * r2 / den
        # Stable derivative of the exact conic expression.
        dden_dr2 = -(1.0 + k) * c * c / (2.0 * root)
        dz_dr2 = c / den - c * r2 * dden_dr2 / (den * den)
        return z, 2.0 * x * dz_dr2, 2.0 * y * dz_dr2

    def basis(self, x: np.ndarray, y: np.ndarray, derivative: str | None = None) -> np.ndarray:
        """Tính đơn thức XY và đạo hàm bậc một/hai dùng cho freeform."""
        X, Y = np.asarray(x) / self.scale[0], np.asarray(y) / self.scale[1]
        cols = []
        for i, j in self.terms:
            if derivative == "x":
                cols.append(np.zeros_like(X) if i == 0 else i * X ** (i - 1) * Y ** j / self.scale[0])
            elif derivative == "y":
                cols.append(np.zeros_like(X) if j == 0 else j * X ** i * Y ** (j - 1) / self.scale[1])
            elif derivative == "xx":
                cols.append(
                    np.zeros_like(X)
                    if i < 2
                    else i * (i - 1) * X ** (i - 2) * Y ** j / (self.scale[0] ** 2)
                )
            elif derivative == "xy":
                cols.append(
                    np.zeros_like(X)
                    if i == 0 or j == 0
                    else i * j * X ** (i - 1) * Y ** (j - 1) / (self.scale[0] * self.scale[1])
                )
            elif derivative == "yy":
                cols.append(
                    np.zeros_like(X)
                    if j < 2
                    else j * (j - 1) * X ** i * Y ** (j - 2) / (self.scale[1] ** 2)
                )
            elif derivative is None:
                cols.append(X ** i * Y ** j)
            else:
                raise ValueError(f"POLY_SURFACE_BASIS_DERIVATIVE_INVALID:{derivative}")
        if not cols:
            return np.zeros((len(X), 0), float)
        return np.column_stack(cols)

    def _conic_hessian(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tính Hessian chính xác của sag conic theo tọa độ local x/y."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        c = float(self.curvature)
        k = float(self.conic)
        r2 = x * x + y * y
        a = (1.0 + k) * c * c
        arg = np.maximum(1.0 - a * r2, 1e-12)
        root = np.sqrt(arg)
        den = 1.0 + root
        dden_dr2 = -a / (2.0 * root)
        dz_dr2 = c / den - c * r2 * dden_dr2 / (den * den)
        d2z_dr22 = (
            c * a / (root * den * den)
            + c * a * a * r2 / (4.0 * root ** 3 * den * den)
            + c * a * a * r2 / (2.0 * root * root * den ** 3)
        )
        gxx = 2.0 * dz_dr2 + 4.0 * x * x * d2z_dr22
        gxy = 4.0 * x * y * d2z_dr22
        gyy = 2.0 * dz_dr2 + 4.0 * y * y * d2z_dr22
        return gxx, gxy, gyy

    def sag_slopes(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tính tổng sag và độ dốc của conic cùng freeform."""
        zc, gxc, gyc = self._conic(np.asarray(x), np.asarray(y))
        return (zc + self.basis(x, y) @ self.coeff,
                gxc + self.basis(x, y, "x") @ self.coeff,
                gyc + self.basis(x, y, "y") @ self.coeff)

    def sag_slopes_hessian(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tính sag, slope và Hessian của toàn surface conic + polynomial."""
        z, gx, gy = self.sag_slopes(x, y)
        gxxc, gxyc, gyyc = self._conic_hessian(x, y)
        gxx = gxxc + self.basis(x, y, "xx") @ self.coeff
        gxy = gxyc + self.basis(x, y, "xy") @ self.coeff
        gyy = gyyc + self.basis(x, y, "yy") @ self.coeff
        return z, gx, gy, gxx, gxy, gyy

    def point(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Đổi tọa độ cục bộ thành điểm 3D trên bề mặt."""
        z, _, _ = self.sag_slopes(x, y)
        return self.center + np.column_stack([x, y, z]) @ self.frame.T

    def normal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tính pháp tuyến đơn vị tại điểm trên bề mặt."""
        _, gx, gy = self.sag_slopes(x, y)
        return unit(np.column_stack([-gx, -gy, np.ones_like(gx)])) @ self.frame.T

    def intersect(self, origins: np.ndarray, directions: np.ndarray, finite: bool = True,
                   t_min: float = 1e-6, iterations: int = 12) -> dict[str, np.ndarray]:
        """Tìm nghiệm dương nhỏ nhất trên nhánh conic+polynomial thực sự thuộc clear aperture.

        Bao gồm fast-path giải tích cho mặt phẳng và conic thuần; với freeform tổng quát,
        các nghiệm đại số được cô lập, lọc theo miền xác định conic, thay ngược vào phương trình
        sag gốc để loại nghiệm ngoại lai và chọn nghiệm gần nhất trong clear aperture.
        """
        del iterations
        O, D, usable, finite_input = _normalized_ray_batch(origins, directions)
        ol = (O - self.center) @ self.frame
        dl = D @ self.frame
        count = len(O)
        t = np.full(count, np.nan); x = np.full(count, np.nan); y = np.full(count, np.nan)
        residual = np.full(count, np.nan); conic_argument = np.full(count, np.nan)
        valid = np.zeros(count, bool); resolved = np.zeros(count, bool)
        candidate_count = np.zeros(count, int); outside_count = np.zeros(count, int)
        conic_invalid_count = np.zeros(count, int)
        status = np.full(count, "NONFINITE_INPUT", dtype=object)
        status[finite_input & ~usable] = "ZERO_DIRECTION"

        # Đường nhanh giải tích cho mặt phẳng (curvature=0 và chỉ có hệ số hằng số)
        is_plane = abs(float(self.curvature)) <= 1e-14 and (
            len(self.terms) == 0 or (len(self.terms) == 1 and self.terms[0] == (0, 0))
        )
        # Đường nhanh giải tích cho conic thuần (curvature!=0 và không có hệ số freeform)
        is_pure_conic = (not is_plane) and abs(float(self.curvature)) > 1e-14 and (
            len(self.coeff) == 0 or np.all(self.coeff == 0.0)
        )

        if is_plane:
            if selected_backend("plane_batch") == "reference":
                c_val = float(self.coeff[0]) if len(self.coeff) else 0.0
                for index in progress_indices(
                    np.where(usable)[0],
                    f"PolySurface.intersect:{self.name}",
                    batch_total=len(O),
                ):
                    dlz = float(dl[index, 2])
                    if abs(dlz) <= 1e-12:
                        if abs(c_val - ol[index, 2]) <= 1e-12:
                            status[index] = "DEGENERATE_RAY_ON_SURFACE"
                            resolved[index] = False
                        else:
                            status[index] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"
                            resolved[index] = True
                        continue
                    root = float((c_val - ol[index, 2]) / dlz)
                    if root <= float(t_min) or not np.isfinite(root):
                        status[index] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"
                        resolved[index] = True
                        continue
                    xr = float(ol[index, 0] + root * dl[index, 0])
                    yr = float(ol[index, 1] + root * dl[index, 1])
                    candidate_count[index] = 1
                    in_aperture = bool(aperture_contains_xy(
                        self.half_aperture, self.aperture_polygon,
                        np.array([xr]), np.array([yr]))[0])
                    if finite and not in_aperture:
                        outside_count[index] = 1
                        status[index] = "OUTSIDE_CLEAR_APERTURE"
                        resolved[index] = True
                    else:
                        t[index] = root; x[index] = xr; y[index] = yr; residual[index] = 0.0
                        conic_argument[index] = 1.0
                        valid[index] = True; status[index] = "HIT"; resolved[index] = True
            else:
                c_val = float(self.coeff[0]) if len(self.coeff) else 0.0

                parallel = usable & (np.abs(dl[:, 2]) <= 1e-12)
                on_plane = parallel & (np.abs(c_val - ol[:, 2]) <= 1e-12)
                parallel_miss = parallel & ~on_plane

                status[on_plane] = "DEGENERATE_RAY_ON_SURFACE"
                resolved[on_plane] = False

                status[parallel_miss] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"
                resolved[parallel_miss] = True

                moving = np.flatnonzero(usable & ~parallel)
                roots = (c_val - ol[moving, 2]) / dl[moving, 2]

                positive = np.isfinite(roots) & (roots > float(t_min))
                miss_ids = moving[~positive]
                status[miss_ids] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"
                resolved[miss_ids] = True

                ids = moving[positive]
                roots = roots[positive]

                if len(ids):
                    xr = ol[ids, 0] + roots * dl[ids, 0]
                    yr = ol[ids, 1] + roots * dl[ids, 1]

                    candidate_count[ids] = 1

                    in_aperture = aperture_contains_xy(
                        self.half_aperture,
                        self.aperture_polygon,
                        xr,
                        yr,
                    )

                    accepted = (
                        in_aperture
                        if finite
                        else np.ones(len(ids), dtype=bool)
                    )

                    outside_ids = ids[~accepted]
                    outside_count[outside_ids] = 1
                    status[outside_ids] = "OUTSIDE_CLEAR_APERTURE"
                    resolved[outside_ids] = True

                    hit_ids = ids[accepted]
                    t[hit_ids] = roots[accepted]
                    x[hit_ids] = xr[accepted]
                    y[hit_ids] = yr[accepted]
                    residual[hit_ids] = 0.0
                    conic_argument[hit_ids] = 1.0
                    valid[hit_ids] = True
                    resolved[hit_ids] = True
                    status[hit_ids] = "HIT"
        elif is_pure_conic:
            c_curv = float(self.curvature)
            conic_k = float(self.conic)
            for index in progress_indices(
                np.where(usable)[0],
                f"PolySurface.intersect:{self.name}",
                batch_total=len(O),
            ):
                dl0, dl1, dl2 = float(dl[index, 0]), float(dl[index, 1]), float(dl[index, 2])
                ol0, ol1, ol2 = float(ol[index, 0]), float(ol[index, 1]), float(ol[index, 2])
                A_quad = (dl0 * dl0 + dl1 * dl1) + (1.0 + conic_k) * (dl2 * dl2)
                B_quad = 2.0 * (ol0 * dl0 + ol1 * dl1) - 2.0 * dl2 / c_curv + 2.0 * (1.0 + conic_k) * ol2 * dl2
                C_quad = (ol0 * ol0 + ol1 * ol1) - 2.0 * ol2 / c_curv + (1.0 + conic_k) * (ol2 * ol2)
                roots: list[float] = []
                if abs(A_quad) <= 1e-14:
                    if abs(B_quad) > 1e-14:
                        roots = [-C_quad / B_quad]
                else:
                    disc = B_quad * B_quad - 4.0 * A_quad * C_quad
                    if disc >= -1e-12:
                        disc_sqrt = math.sqrt(max(disc, 0.0))
                        roots = [(-B_quad - disc_sqrt) / (2.0 * A_quad),
                                 (-B_quad + disc_sqrt) / (2.0 * A_quad)]
                roots = sorted([r for r in roots if np.isfinite(r) and r > float(t_min)])
                original_candidates: list[tuple[float, float, float, float, float]] = []
                for root in roots:
                    xr = float(ol0 + root * dl0)
                    yr = float(ol1 + root * dl1)
                    raw_argument = float(1.0 - (1.0 + conic_k) * c_curv * c_curv * (xr * xr + yr * yr))
                    if not np.isfinite(raw_argument) or raw_argument < -CONIC_DOMAIN_TOLERANCE:
                        conic_invalid_count[index] += 1
                        continue
                    root_argument = math.sqrt(max(raw_argument, 0.0))
                    conic_sag = c_curv * (xr * xr + yr * yr) / (1.0 + root_argument)
                    rr = float(ol2 + root * dl2 - conic_sag)
                    tolerance = INTERSECTION_RESIDUAL_TOLERANCE_MM * (1.0 + abs(ol2))
                    if not np.isfinite(rr) or abs(rr) > tolerance:
                        continue
                    candidate_count[index] += 1
                    in_aperture = bool(aperture_contains_xy(
                        self.half_aperture, self.aperture_polygon,
                        np.array([xr]), np.array([yr]))[0])
                    if finite and not in_aperture:
                        outside_count[index] += 1
                        continue
                    original_candidates.append((float(root), xr, yr, rr, raw_argument))
                resolved[index] = True
                if original_candidates:
                    chosen = min(original_candidates, key=lambda row: row[0])
                    t[index], x[index], y[index], residual[index], conic_argument[index] = chosen
                    valid[index] = True; status[index] = "HIT"
                elif outside_count[index]:
                    status[index] = "OUTSIDE_CLEAR_APERTURE"
                elif conic_invalid_count[index]:
                    status[index] = "CONIC_DOMAIN_INVALID"
                else:
                    status[index] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"
        else:
            parallel_res = _parallel_poly_surface_intersect(self, O, D, finite, t_min)
            if parallel_res is not None:
                return parallel_res

            for index in progress_indices(
                np.where(usable)[0],
                f"PolySurface.intersect:{self.name}",
                batch_total=len(O),
            ):
                x0 = float(ol[index, 0])
                dx = float(dl[index, 0])
                y0 = float(ol[index, 1])
                dy = float(dl[index, 1])
                z0 = float(ol[index, 2])
                dz = float(dl[index, 2])
                sx = float(self.scale[0])
                sy = float(self.scale[1])

                freeform_coeff = np.array([0.0], dtype=float)
                for (power_x, power_y), coefficient in zip(self.terms, self.coeff):
                    coeff_val = float(coefficient)
                    if coeff_val == 0.0:
                        continue
                    x_power = _linear_power_coefficients(x0 / sx, dx / sx, power_x)
                    y_power = _linear_power_coefficients(y0 / sy, dy / sy, power_y)
                    term_coeff = coeff_val * _poly_coeff_convolve(x_power, y_power)
                    freeform_coeff = _poly_add(freeform_coeff, term_coeff)

                z_coeff = np.array([z0, dz], dtype=float)
                q_coeff = _poly_add(z_coeff, -freeform_coeff)

                if abs(float(self.curvature)) <= 1e-14:
                    equation_coeff = q_coeff
                else:
                    xp_coeff = np.array([x0, dx], dtype=float)
                    yp_coeff = np.array([y0, dy], dtype=float)
                    radial_coeff = _poly_add(
                        _poly_coeff_convolve(xp_coeff, xp_coeff),
                        _poly_coeff_convolve(yp_coeff, yp_coeff),
                    )
                    q_sq = _poly_coeff_convolve(q_coeff, q_coeff)
                    equation_coeff = _poly_add(
                        radial_coeff,
                        _poly_add(-2.0 * q_coeff / float(self.curvature), (1.0 + float(self.conic)) * q_sq),
                    )

                if abs(float(dl[index, 2])) > 1e-12:
                    t_reference = (
                        -float(ol[index, 2])
                        / float(dl[index, 2])
                    )
                else:
                    t_reference = 1.0

                roots, root_status = (
                    _distance_scaled_real_roots_from_coefficients(
                        equation_coeff,
                        t_reference,
                    )
                )
                if root_status.startswith("UNRESOLVED") or root_status.startswith("DEGENERATE"):
                    status[index] = root_status
                    continue
                positive = roots[roots > float(t_min)]
                original_candidates: list[tuple[float, float, float, float, float]] = []
                for root in positive:
                    xr = float(ol[index, 0] + root * dl[index, 0])
                    yr = float(ol[index, 1] + root * dl[index, 1])
                    raw_argument = float(1.0 - (1.0 + self.conic) * self.curvature * self.curvature
                                         * (xr * xr + yr * yr))
                    if not np.isfinite(raw_argument) or raw_argument < -CONIC_DOMAIN_TOLERANCE:
                        conic_invalid_count[index] += 1
                        continue
                    root_argument = math.sqrt(max(raw_argument, 0.0))
                    conic_sag = (0.0 if abs(self.curvature) <= 1e-14 else
                                 self.curvature * (xr * xr + yr * yr) / (1.0 + root_argument))
                    polynomial_sag = float(np.polynomial.polynomial.polyval(root, freeform_coeff))
                    rr = float(ol[index, 2] + root * dl[index, 2] - conic_sag - polynomial_sag)
                    tolerance = INTERSECTION_RESIDUAL_TOLERANCE_MM * (1.0 + abs(float(ol[index, 2])))
                    if not np.isfinite(rr) or abs(rr) > tolerance:
                        continue
                    candidate_count[index] += 1
                    in_aperture = bool(aperture_contains_xy(
                        self.half_aperture, self.aperture_polygon,
                        np.array([xr]), np.array([yr]))[0])
                    if finite and not in_aperture:
                        outside_count[index] += 1
                        continue
                    original_candidates.append((float(root), xr, yr, rr, raw_argument))
                resolved[index] = True
                if original_candidates:
                    chosen = min(original_candidates, key=lambda row: row[0])
                    t[index], x[index], y[index], residual[index], conic_argument[index] = chosen
                    valid[index] = True; status[index] = "HIT"
                elif outside_count[index]:
                    status[index] = "OUTSIDE_CLEAR_APERTURE"
                elif conic_invalid_count[index]:
                    status[index] = "CONIC_DOMAIN_INVALID"
                else:
                    status[index] = "NO_INTERSECTION_IN_SEARCH_DOMAIN"

        point = np.full((count, 3), np.nan); normal = np.full((count, 3), np.nan)
        point[valid] = O[valid] + t[valid, None] * D[valid]
        if np.any(valid):
            normal[valid] = self.normal(x[valid], y[valid])
        return {"t": t, "point": point, "normal": normal, "x": x, "y": y,
                "residual": residual, "valid": valid, "resolved": resolved,
                "status": status, "candidate_count": candidate_count,
                "outside_aperture_candidate_count": outside_count,
                "conic_invalid_candidate_count": conic_invalid_count,
                "raw_conic_argument": conic_argument,
                "root_method": ("ANALYTIC_PLANE_INTERSECT" if is_plane else
                                ("ANALYTIC_PURE_CONIC_QUADRATIC" if is_pure_conic else
                                 "COMPLETE_CONIC_POLYNOMIAL_WITH_ORIGINAL_BRANCH_SUBSTITUTION"))}

    def to_dict(self) -> dict[str, Any]:
        """Tuần tự hóa mô hình bề mặt để lưu checkpoint."""
        physical = []
        for (i, j), a in zip(self.terms, self.coeff):
            physical.append({"i": i, "j": j, "normalized_coefficient_mm": a,
                             "physical_coefficient_mm_power": a / (self.scale[0] ** i * self.scale[1] ** j)})
        return {"name": self.name, "center_mm": self.center, "frame_columns": self.frame,
                "half_aperture_mm": self.half_aperture, "scale_mm": self.scale,
                "curvature_per_mm": self.curvature, "conic_constant": self.conic,
                "pure_xy20": self.pure_xy20, "aperture_polygon_local_mm": self.aperture_polygon,
                "terms": physical}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolySurface":
        """Khôi phục mô hình bề mặt từ checkpoint JSON."""
        terms = [(int(t["i"]), int(t["j"])) for t in d["terms"]]
        coeff = np.array([t["normalized_coefficient_mm"] for t in d["terms"]])
        return cls(d["name"], np.array(d["center_mm"]), np.array(d["frame_columns"]),
                   np.array(d["half_aperture_mm"]), terms, coeff, np.array(d["scale_mm"]),
                   float(d.get("curvature_per_mm", 0.0)), float(d.get("conic_constant", 0.0)),
                   bool(d.get("pure_xy20", False)),
                   None if d.get("aperture_polygon_local_mm") is None else np.array(d["aperture_polygon_local_mm"]))


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Phản xạ tia theo định luật phản xạ trên pháp tuyến đơn vị."""
    d, n = unit(direction), unit(normal)
    return unit(d - 2.0 * np.sum(d * n, axis=1)[:, None] * n)


def make_virtual_image(config: dict[str, Any]) -> dict[str, Any]:
    """Sinh field grid ảnh ảo từ FOV, VID, look-down và config."""
    d6, az = np.deg2rad(config["d6_deg"]), np.deg2rad(config["azimuth_deg"])
    central = unit(np.array([math.cos(d6) * math.cos(az), math.cos(d6) * math.sin(az), math.sin(d6)]))
    horizontal = unit(np.array([-math.sin(az), math.cos(az), 0.0]))
    vertical = unit(np.cross(central, horizontal))
    center = config["vid_mm"] * central
    grid = config["field_grid"]
    nh = int(grid["horizontal_count"]); nv = int(grid["vertical_count"])
    horizontal_samples = np.linspace(-float(config["fov_h_deg"]) / 2.0,
                                     float(config["fov_h_deg"]) / 2.0, nh)
    vertical_samples = np.linspace(-float(config["fov_v_deg"]) / 2.0,
                                   float(config["fov_v_deg"]) / 2.0, nv)
    center_h, center_v = nh // 2, nv // 2
    fields = []
    for iv, vdeg in enumerate(vertical_samples):
        for ih, hdeg in enumerate(horizontal_samples):
            d = unit(central + math.tan(math.radians(hdeg)) * horizontal + math.tan(math.radians(vdeg)) * vertical)
            p = d * (config["vid_mm"] / np.dot(d, central))
            fields.append({"field_id": f"F{iv:02d}_{ih:02d}", "h_deg": hdeg, "v_deg": vdeg,
                           "horizontal_index": ih, "vertical_index": iv,
                           "is_central_field": bool(ih == center_h and iv == center_v),
                           "direction": d, "vi_point": p})
    central_field_index = center_v * nh + center_h
    return {"center": center, "normal": central, "horizontal": horizontal, "vertical": vertical,
            "fields": fields, "central_field_index": central_field_index,
            "field_grid": {"horizontal_count": nh, "vertical_count": nv,
                           "field_count": nh * nv,
                           "horizontal_samples_deg": horizontal_samples,
                           "vertical_samples_deg": vertical_samples}}


def polar_pattern(kind: int, radius: float = 4.0, phase_deg: float = 0.0) -> list[dict[str, Any]]:
    """Sinh mẫu pupil gồm chief ray và các vòng cực có trọng số."""
    if kind == 9:
        rings, azimuths = [radius], 8
    elif kind == 25:
        rings, azimuths = [radius / 3, 2 * radius / 3, radius], 8
    elif kind == 49:
        rings, azimuths = [radius / 3, 2 * radius / 3, radius], 16
    elif kind == 81:
        rings, azimuths = [radius * k / 5 for k in range(1, 6)], 16
    else:
        raise ValueError(kind)
    rows = [{"sample_id": "C", "ring": 0, "azimuth_deg": 0.0, "dy_mm": 0.0, "dz_mm": 0.0,
             "chief": True}]
    for ir, r in enumerate(rings, 1):
        for m in range(azimuths):
            a = math.radians(phase_deg + m * 360.0 / azimuths)
            rows.append({"sample_id": f"R{ir}A{m:02d}", "ring": ir,
                         "azimuth_deg": (phase_deg + m * 360.0 / azimuths) % 360.0,
                         "dy_mm": r * math.cos(a), "dz_mm": r * math.sin(a), "chief": False})
    return rows


def primary_pupils(y_bounds: tuple[float, float] = (-8.0, 8.0),
                   z_bounds: tuple[float, float] = (-5.0, 5.0),
                   grid: tuple[int, int] = (3, 3)) -> list[dict[str, Any]]:
    """Sinh lưới tâm pupil dùng trong Fan core, phủ đều tâm eye-box."""
    y0, y1 = map(float, y_bounds); z0, z1 = map(float, z_bounds)
    ny, nz = map(int, grid)
    y_locations = np.linspace(y0, y1, ny)
    z_locations = np.linspace(z1, z0, nz)
    pupils = []
    for iz, z in enumerate(z_locations):
        for iy, y in enumerate(y_locations):
            pupil_id = "C" if (iy == ny // 2 and iz == nz // 2) else f"P{iz:02d}_{iy:02d}"
            pupils.append({"pupil_id": pupil_id, "x_mm": 0.0, "y_mm": float(y), "z_mm": float(z)})
    return pupils


def dense_pupils(y_bounds: tuple[float, float] = (-8.0, 8.0),
                 z_bounds: tuple[float, float] = (-5.0, 5.0)) -> list[dict[str, Any]]:
    """Sinh lưới pupil dày để đánh giá phủ eyebox."""
    y0, y1 = map(float, y_bounds); z0, z1 = map(float, z_bounds)
    yc, zc = 0.5*(y0+y1), 0.5*(z0+z1)
    return [{"pupil_id": f"P{iz}{iy}", "x_mm": 0.0, "y_mm": y, "z_mm": z}
            for iz, z in enumerate((z1, zc, z0)) for iy, y in enumerate((y0, yc, y1))]


def holdout_pupils(y_bounds: tuple[float, float] = (-8.0, 8.0),
                   z_bounds: tuple[float, float] = (-5.0, 5.0)) -> list[dict[str, Any]]:
    # Normalized centers never overlap the {-1,0,+1} optimizer grid.
    """Sinh pupil kiểm tra độc lập, không tham gia dựng mặt."""
    y0, y1 = map(float, y_bounds); z0, z1 = map(float, z_bounds)
    yc, zc = 0.5*(y0+y1), 0.5*(z0+z1); hy, hz = 0.5*(y1-y0), 0.5*(z1-z0)
    normalized = [(-.5,.3),(.5,.3),(-.5,-.3),(.5,-.3),(-.875,.5),(.875,.5),
                  (-.875,-.5),(.875,-.5),(-.25,.8),(.25,.8),(-.25,-.8),(.25,-.8),(.125,.2)]
    return [{"pupil_id": f"H{i:02d}", "x_mm": 0.0, "y_mm": yc+y*hy, "z_mm": zc+z*hz}
            for i, (y, z) in enumerate(normalized)]


def build_rays(fields: list[dict[str, Any]], pupils: list[dict[str, Any]], pattern: list[dict[str, Any]]) -> dict[str, Any]:
    """Ghép field, pupil và mẫu khẩu độ thành characteristic rays."""
    rows, origins, directions = [], [], []
    for fi, f in enumerate(fields):
        vp = np.asarray(f["vi_point"])
        for pi, p in enumerate(pupils):
            for ri, s in enumerate(pattern):
                o = np.array([p["x_mm"], p["y_mm"] + s["dy_mm"], p["z_mm"] + s["dz_mm"]])
                d = unit(vp - o)
                ray_id = f"{f['field_id']}_{p['pupil_id']}_{s['sample_id']}"
                rows.append({"ray_id": ray_id, "field_id": f["field_id"], "field_index": fi,
                             "is_central_field": bool(f.get("is_central_field", False)),
                             "pupil_id": p["pupil_id"], "pupil_index": pi, "sample_id": s["sample_id"],
                             "sample_index": ri, "chief_preseed": bool(s["chief"]),
                             "origin_x": o[0], "origin_y": o[1], "origin_z": o[2],
                             "dir_x": d[0], "dir_y": d[1], "dir_z": d[2]})
                origins.append(o); directions.append(d)
    centers = [i for i, f in enumerate(fields) if bool(f.get("is_central_field", False))]
    if len(centers) != 1:
        raise RuntimeError("FIELD_GRID_REQUIRES_EXACTLY_ONE_CENTRAL_FIELD")
    center_pupils = [i for i, p in enumerate(pupils) if p.get("pupil_id") == "C"]
    if len(center_pupils) != 1:
        raise RuntimeError("PRIMARY_PUPIL_GRID_REQUIRES_EXACTLY_ONE_CENTRAL_PUPIL")
    return {"rows": rows, "origins": np.array(origins), "directions": np.array(directions),
            "field_index": np.array([r["field_index"] for r in rows], int),
            "pupil_index": np.array([r["pupil_index"] for r in rows], int),
            "sample_index": np.array([r["sample_index"] for r in rows], int),
            "chief": np.array([r["chief_preseed"] for r in rows], bool),
            "field_count": len(fields), "pupil_count": len(pupils),
            "samples_per_pupil": len(pattern), "central_field_index": centers[0],
            "central_pupil_index": center_pupils[0]}


def ray_plane_intersection(origins: np.ndarray, directions: np.ndarray, point: np.ndarray,
                           normal: np.ndarray, t_min: float = 1e-6) -> dict[str, np.ndarray]:
    """Tính tham số và điểm giao tia–mặt phẳng."""
    den = directions @ normal
    t = ((point - origins) @ normal) / np.where(np.abs(den) > EPS, den, np.nan)
    return {"t": t, "point": origins + t[:, None] * directions,
            "valid": (t > t_min) & np.isfinite(t)}


def first_hit(origins: np.ndarray, directions: np.ndarray, surfaces: dict[str, Any], t_min=1e-5) -> dict[str, Any]:
    """Select the nearest solved hit without treating unresolved surfaces as misses."""
    names = list(surfaces)
    all_hits = [surfaces[n].intersect(origins, directions, finite=True, t_min=t_min) for n in names]
    ts = np.column_stack([np.where(h["valid"], h["t"], np.inf) for h in all_hits])
    idx = np.argmin(ts, axis=1)
    t = ts[np.arange(len(ts)), idx]
    solved_hit = np.isfinite(t)
    resolved_matrix = np.column_stack([
        np.asarray(h.get("resolved", np.ones(len(t), bool)), bool) for h in all_hits])
    unresolved = ~np.all(resolved_matrix, axis=1)
    # Without a certified lower bound for an unresolved graph, it may precede the
    # selected hit.  The conservative result is unresolved, never physical-valid.
    valid = solved_hit & ~unresolved
    hit_name = np.array([names[i] for i in idx], dtype=object)
    hit_name[~solved_hit & ~unresolved] = "MISS"
    hit_name[unresolved] = "UNRESOLVED"
    point = np.full((len(t), 3), np.nan); normal = np.full((len(t), 3), np.nan)
    for surface_index, hit in enumerate(all_hits):
        selected = valid & (idx == surface_index)
        point[selected] = hit["point"][selected]
        normal[selected] = hit["normal"][selected]
    status = np.where(unresolved, "UNRESOLVED_POTENTIAL_FIRST_HIT",
                      np.where(solved_hit, "HIT", "MISS"))
    return {"name": hit_name, "t": t, "point": point, "normal": normal,
            "valid": valid, "resolved": ~unresolved, "status": status,
            "unresolved_surface_count": np.sum(~resolved_matrix, axis=1),
            "all": dict(zip(names, all_hits))}


def trace_reverse(rays: dict[str, Any], visor: ChebVisor, m1: PolySurface, m2: PolySurface,
                  display: PolySurface, physical_first_hit: bool = True,
                  sequential_aperture_mode: str = "FINITE") -> dict[str, Any]:
    """Ray-trace ngược từ mắt qua visor, M1, M2 đến display."""
    if sequential_aperture_mode not in ("FINITE", "CONSTRUCTION_FOOTPRINT"):
        raise ValueError(f"INVALID_SEQUENTIAL_APERTURE_MODE:{sequential_aperture_mode}")
    if physical_first_hit and sequential_aperture_mode != "FINITE":
        raise ValueError("PHYSICAL_FIRST_HIT_REQUIRES_FINITE_APERTURES")

    O, D = rays["origins"], rays["directions"]
    surfaces = {"VISOR": visor, "M1": m1, "M2": m2, "DISPLAY": display}
    logs, points, dirs = [], [], []
    expected = ["VISOR", "M1", "M2", "DISPLAY"]
    current_o, current_d = O.copy(), D.copy()
    valid = np.ones(len(O), dtype=bool)
    resolved = np.ones(len(O), dtype=bool)
    first_names = []
    for si, name in enumerate(expected):
        if physical_first_hit:
            fh = first_hit(current_o, current_d, surfaces)
            hit = fh["all"][name]
            is_expected = fh["name"] == name
            stage_resolved = np.asarray(fh["resolved"], bool)
            first_names.append(fh["name"].copy())
        else:
            finite_flag = True if name == "VISOR" else (sequential_aperture_mode == "FINITE")
            hit = surfaces[name].intersect(current_o, current_d, finite=finite_flag)
            is_expected = hit["valid"]
            stage_resolved = np.asarray(hit.get("resolved", np.ones(len(O), bool)), bool)
            first_names.append(np.where(is_expected, name, "MISS"))
        resolved &= stage_resolved
        valid &= resolved & is_expected & hit["valid"]
        p = hit["point"]
        points.append(p)
        if name != "DISPLAY":
            current_d = reflect(current_d, hit["normal"])
            current_o = p + 1e-4 * current_d
        dirs.append(current_d.copy())
        logs.append({"surface": name, "expected_count": int(np.sum(is_expected)),
                     "resolved_count": int(np.sum(stage_resolved)),
                     "unresolved_count": int(np.sum(~stage_resolved)),
                     "valid_cumulative": int(np.sum(valid))})
    return {"valid": valid, "resolved": resolved, "points": points, "directions": dirs, "first_names": first_names,
            "summary": logs, "landing": points[-1], "display_local": (points[-1] - display.center) @ display.frame,
            "display_frame": display.frame, "display_center": display.center}


def point_by_point_engine(
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    seed_surface: PolySurface,
    seed_index: int,
    eps_parallel=1e-10,
    construction_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dựng M1 theo CI và kiểm physical order cùng unobscured."""
    if selected_backend("point_by_point_ci") != "cpu":
        raise BackendExecutionError("CI_REQUIRES_QUALIFIED_CPU_BACKEND")
    base = seed_surface.intersect(
        starts,
        directions,
        finite=False,
    )
    P0 = base["point"]
    n_rays = len(starts)

    options = (
        {"mode": "LEGACY_NEAREST"}
        if construction_options is None
        else construction_options
    )
    mode = str(options.get("mode", "LEGACY_NEAREST"))

    if mode in (
        "SURFACE_COMPATIBLE_FRONTIER_V1",
        "SURFACE_COMPATIBLE_FRONTIER_V2",
        "SURFACE_COMPATIBLE_PATCH_V3",
    ):
        if "normal" not in base:
            raise RuntimeError(
                "CI_COMPATIBLE_SEED_NORMAL_MISSING"
            )
        N0 = np.asarray(
            base["normal"],
            dtype=float,
        )

    if mode == "LEGACY_NEAREST":
        points, normals, order, parent, distance = execute_compiled_point_by_point(
            starts, directions, targets, P0, seed_index, float(eps_parallel), float(EPS)
        )

        inverse = np.empty(n_rays, int)
        inverse[order] = np.arange(n_rays)
        return {
            "points_ordered": points,
            "normals_ordered": normals,
            "ray_order": order,
            "parent_order_index": parent,
            "nearest_distance_mm": distance,
            "points_by_ray": points[inverse],
            "normals_by_ray": normals[inverse],
            "starts_by_ray": np.asarray(starts, float).copy(),
            "directions_by_ray": np.asarray(directions, float).copy(),
            "targets_by_ray": np.asarray(targets, float).copy(),
            "fallback_count": int(np.sum(parent == -2)),
        }

    if mode in (
        "SURFACE_COMPATIBLE_FRONTIER_V1",
        "SURFACE_COMPATIBLE_FRONTIER_V2",
        "SURFACE_COMPATIBLE_PATCH_V3",
    ):
        frontier = options["compatible_frontier"]

        if mode == "SURFACE_COMPATIBLE_FRONTIER_V1":
            (
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
            ) = execute_compiled_point_by_point_compatible(
                starts,
                directions,
                targets,
                P0,
                N0,
                seed_index,
                float(eps_parallel),
                float(EPS),
                float(frontier["minimum_candidate_radius_mm"]),
                float(frontier["candidate_distance_factor"]),
                float(frontier["maximum_candidate_radius_mm"]),
                float(frontier["normal_angle_floor_deg"]),
                float(frontier["normal_angle_per_mm_deg"]),
                float(frontier["normal_angle_cap_deg"]),
            )
        elif mode == "SURFACE_COMPATIBLE_FRONTIER_V2":
            edge_cap = float(frontier["edge_height_residual_cap_mm"])
            (
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
            ) = execute_compiled_point_by_point_compatible_v2(
                starts,
                directions,
                targets,
                P0,
                N0,
                seed_index,
                float(eps_parallel),
                float(EPS),
                float(frontier["minimum_candidate_radius_mm"]),
                float(frontier["candidate_distance_factor"]),
                float(frontier["maximum_candidate_radius_mm"]),
                float(frontier["normal_angle_floor_deg"]),
                float(frontier["normal_angle_per_mm_deg"]),
                float(frontier["normal_angle_cap_deg"]),
                edge_height_residual_cap_mm=edge_cap,
            )
        elif mode == "SURFACE_COMPATIBLE_PATCH_V3":
            edge_cap = float(frontier["edge_height_residual_cap_mm"])
            (
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
            ) = execute_compiled_point_by_point_compatible_v3(
                starts,
                directions,
                targets,
                P0,
                N0,
                seed_index,
                float(eps_parallel),
                float(EPS),
                float(frontier["minimum_candidate_radius_mm"]),
                float(frontier["candidate_distance_factor"]),
                float(frontier["maximum_candidate_radius_mm"]),
                float(frontier["normal_angle_floor_deg"]),
                float(frontier["normal_angle_per_mm_deg"]),
                float(frontier["normal_angle_cap_deg"]),
                edge_height_residual_cap_mm=edge_cap,
            )

        if not np.array_equal(np.sort(order), np.arange(n_rays)):
            raise RuntimeError("CI_COMPATIBLE_RAY_ORDER_NOT_PERMUTATION")

        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
            raise RuntimeError("CI_COMPATIBLE_NONFINITE_OUTPUT")

        valid_parents = (parent == -1) | (parent == -3) | (parent >= 0)
        if not np.all(valid_parents):
            raise RuntimeError("CI_COMPATIBLE_PARENT_CODE_INVALID")

        inverse = np.empty(n_rays, int)
        inverse[order] = np.arange(n_rays)

        return {
            "points_ordered": points,
            "normals_ordered": normals,
            "ray_order": order,
            "parent_order_index": parent,
            "nearest_distance_mm": distance,
            "points_by_ray": points[inverse],
            "normals_by_ray": normals[inverse],
            "starts_by_ray": np.asarray(starts, float).copy(),
            "directions_by_ray": np.asarray(directions, float).copy(),
            "targets_by_ray": np.asarray(targets, float).copy(),
            "fallback_count": int(np.sum(parent == -2)),
            "construction_policy": mode,
            "front_id": front_id,
            "restart_reason_code": restart_reason_code,
            "parent_normal_angle_deg": parent_normal_angle_deg,
            "local_candidate_count": local_candidate_count,
            "compatible_candidate_count": compatible_candidate_count,
            "front_restart_count": int(np.sum(parent == -3)),
            "front_count": int(np.max(front_id) + 1),
        }

    raise ValueError(f"CI_CONSTRUCTION_MODE_INVALID:{mode}")


def _finite_summary(values: np.ndarray, operation: str) -> float | None:
    """Tóm tắt một mảng diagnostic nhưng không biến NaN thành kết luận giả."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    if operation == "rms":
        return float(np.sqrt(np.mean(finite * finite)))
    if operation == "max":
        return float(np.max(finite))
    if operation.startswith("p"):
        return float(np.percentile(finite, float(operation[1:])))
    raise ValueError(f"UNKNOWN_DIAGNOSTIC_SUMMARY:{operation}")


def _single_cloud_diagnostics(points: np.ndarray, normals: np.ndarray, frame: np.ndarray,
                              neighbors: int, max_samples: int,
                              include_rows: bool) -> dict[str, Any]:
    """Đo graph, local patch, curl và loop circulation cho đúng một point cloud liên thông."""
    count = len(points)
    if count < 4:
        return {"status": "NOT_EVALUABLE_TOO_FEW_POINTS", "point_count": count,
                "sample_count": 0, "sample_rows": []}
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    frame = np.asarray(frame, dtype=float)

    if not np.all(np.isfinite(points)):
        raise ValueError(
            "POINT_NORMAL_DIAGNOSTIC_NONFINITE_POINTS"
        )

    if not np.all(np.isfinite(normals)):
        raise ValueError(
            "POINT_NORMAL_DIAGNOSTIC_NONFINITE_NORMALS"
        )

    if not np.all(np.isfinite(frame)):
        raise ValueError(
            "POINT_NORMAL_DIAGNOSTIC_NONFINITE_FRAME"
        )

    local = (points - np.mean(points, axis=0)) @ frame
    nlocal = unit(normals) @ frame
    nlocal[nlocal[:, 2] < 0.0] *= -1.0
    abs_nz = np.abs(nlocal[:, 2])
    slopes = np.column_stack([-nlocal[:, 0] / np.maximum(abs_nz, 1e-12),
                              -nlocal[:, 1] / np.maximum(abs_nz, 1e-12)])
    sample_count = min(max(int(max_samples), 1), count)
    sample_index = np.unique(np.linspace(0, count - 1, sample_count, dtype=int))
    k = min(max(int(neighbors), 3), count - 1)
    tree = cKDTree(local[:, :2])
    distances, indices = tree.query(local[sample_index, :2], k=k + 1)
    if np.ndim(indices) == 1:
        indices = np.asarray(indices)[:, None]
        distances = np.asarray(distances)[:, None]
    curl = np.full(len(sample_index), np.nan)
    normal_jump = np.full(len(sample_index), np.nan)
    patch_normal_angle = np.full(len(sample_index), np.nan)
    nearest_xy = distances[:, 1]
    nearest_dz = np.full(len(sample_index), np.nan)
    nearest_normal_angle = np.full(len(sample_index), np.nan)
    loop_circulation: list[float] = []
    loop_circulation_perimeter: list[float] = []
    edge_height_residual: list[float] = []
    for row, (source, queried) in enumerate(zip(sample_index, indices)):
        near = np.asarray(queried[1:], int)
        delta = local[near, :2] - local[source, :2]
        predicted_dz = np.sum(0.5*(slopes[near]+slopes[source])*delta, axis=1)
        observed_dz = local[near, 2]-local[source, 2]
        edge_height_residual.extend(np.abs(observed_dz-predicted_dz).tolist())
        derivative_design = delta
        if len(near) >= 2 and np.linalg.matrix_rank(derivative_design) >= 2:
            dp = np.linalg.lstsq(derivative_design,
                                 slopes[near, 0] - slopes[source, 0], rcond=1e-10)[0]
            dq = np.linalg.lstsq(derivative_design,
                                 slopes[near, 1] - slopes[source, 1], rcond=1e-10)[0]
            curl[row] = float(dp[1] - dq[0])
        patch_index = np.r_[source, near]
        patch_delta = local[patch_index, :2] - local[source, :2]
        patch_design = np.column_stack([
            np.ones(len(patch_index)), patch_delta[:, 0], patch_delta[:, 1],
            patch_delta[:, 0] ** 2, patch_delta[:, 0] * patch_delta[:, 1],
            patch_delta[:, 1] ** 2,
        ])
        if len(patch_index) >= 6 and np.linalg.matrix_rank(patch_design) >= 6:
            patch = np.linalg.lstsq(patch_design, local[patch_index, 2], rcond=1e-10)[0]
            geometry_normal = unit(np.array([-patch[1], -patch[2], 1.0]))
            patch_normal_angle[row] = float(np.degrees(np.arccos(np.clip(
                np.dot(geometry_normal, nlocal[source]), -1.0, 1.0))))
        angles = np.degrees(np.arccos(np.clip(nlocal[near] @ nlocal[source], -1.0, 1.0)))
        normal_jump[row] = float(np.max(angles))
        nearest = int(near[0])
        nearest_dz[row] = float(abs(local[nearest, 2] - local[source, 2]))
        nearest_normal_angle[row] = float(angles[0])
        # One deterministic non-collinear k-NN triangle per sampled point is a
        # cheap closed-loop path-dependence diagnostic suitable for every CI cycle.
        second = None
        for candidate in near[1:]:
            first_edge = local[near[0], :2]-local[source, :2]
            second_edge = local[candidate, :2]-local[source, :2]
            area2 = abs(float(first_edge[0]*second_edge[1]-first_edge[1]*second_edge[0]))
            if area2 > 1e-12:
                second = int(candidate); break
        if second is not None:
            triangle = [int(source), int(near[0]), second]
            circulation = 0.0; perimeter = 0.0
            for edge in range(3):
                a = triangle[edge]; b = triangle[(edge + 1) % 3]
                edge_delta = local[b, :2]-local[a, :2]
                circulation += float(0.5*np.dot(slopes[a]+slopes[b], edge_delta))
                perimeter += float(np.linalg.norm(edge_delta))
            loop_circulation.append(abs(circulation))
            loop_circulation_perimeter.append(abs(circulation)/max(perimeter, 1e-12))

    rows = []
    if include_rows:
        rows = [{"point_index": int(source), "nearest_xy_mm": float(nearest_xy[row]),
                 "nearest_abs_dz_mm": float(nearest_dz[row]),
                 "nearest_normal_angle_deg": float(nearest_normal_angle[row]),
                 "knn_max_normal_jump_deg": float(normal_jump[row]),
                 "local_geometry_normal_angle_deg": (float(patch_normal_angle[row])
                                                       if np.isfinite(patch_normal_angle[row]) else None),
                 "local_slope_curl_per_mm": (float(curl[row])
                                              if np.isfinite(curl[row]) else None)}
                for row, source in enumerate(sample_index)]
    return {
        "status": "EVALUATED_DISCRETE_DIAGNOSTIC_NOT_A_MATHEMATICAL_PROOF",
        "point_count": count, "sample_count": len(sample_index), "neighbors": k,
        "abs_nz_min": float(np.min(abs_nz)),
        "abs_nz_p01": float(np.percentile(abs_nz, 1)),
        "abs_nz_p05": float(np.percentile(abs_nz, 5)),
        "slope_magnitude_max": float(np.max(np.linalg.norm(slopes, axis=1))),
        "knn_abs_curl_rms_per_mm": _finite_summary(curl, "rms"),
        "knn_abs_curl_p95_per_mm": _finite_summary(np.abs(curl), "p95"),
        "local_geometry_normal_rms_deg": _finite_summary(patch_normal_angle, "rms"),
        "local_geometry_normal_p95_deg": _finite_summary(patch_normal_angle, "p95"),
        "knn_max_normal_jump_p95_deg": _finite_summary(normal_jump, "p95"),
        "knn_max_normal_jump_max_deg": _finite_summary(normal_jump, "max"),
        "nearest_xy_p01_mm": _finite_summary(nearest_xy, "p01"),
        "nearest_abs_dz_p99_mm": _finite_summary(nearest_dz, "p99"),
        "nearest_normal_angle_p99_deg": _finite_summary(nearest_normal_angle, "p99"),
        "local_closed_loop_count": len(loop_circulation),
        "loop_abs_circulation_rms_mm": _finite_summary(np.asarray(loop_circulation), "rms"),
        "loop_abs_circulation_p95_mm": _finite_summary(np.asarray(loop_circulation), "p95"),
        "loop_abs_circulation_perimeter_p95": _finite_summary(
            np.asarray(loop_circulation_perimeter), "p95"),
        "edge_gradient_height_residual_rms_mm": _finite_summary(
            np.asarray(edge_height_residual), "rms"),
        "edge_gradient_height_residual_p95_mm": _finite_summary(
            np.asarray(edge_height_residual), "p95"),
        "sample_rows": rows,
    }


def point_normal_cloud_diagnostics(points: np.ndarray, normals: np.ndarray, frame: np.ndarray,
                                   neighbors: int = 8, max_samples: int = 2000,
                                   group_ids: np.ndarray | None = None,
                                   group_max_samples: int | None = None) -> dict[str, Any]:
    """Đo ba lớp consistency toàn cloud và theo từng field/pupil bundle độc lập."""
    points = np.asarray(points, float)
    normals = unit(np.asarray(normals, float))
    frame = np.asarray(frame, float)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or frame.shape != (3, 3):
        raise ValueError("POINT_NORMAL_DIAGNOSTIC_INPUT_SHAPE_MISMATCH")
    global_result = _single_cloud_diagnostics(
        points, normals, frame, neighbors, max_samples, include_rows=True)
    if group_ids is None:
        return global_result | {
            "grouping": "NONE_GLOBAL_SCATTERED_CLOUD",
            "interpretation": "GLOBAL_KNN_MAY_MIX_DISTINCT_FIELD_PUPIL_FOOTPRINTS",
        }
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) != len(points):
        raise ValueError("POINT_NORMAL_DIAGNOSTIC_GROUP_ID_SHAPE_MISMATCH")
    bundle_rows: list[dict[str, Any]] = []
    for group in np.unique(groups):
        selected = np.where(groups == group)[0]
        per_group_limit = (len(selected) if group_max_samples is None
                           else max(int(group_max_samples), 1))
        result = _single_cloud_diagnostics(
            points[selected], normals[selected], frame, neighbors,
            min(per_group_limit, len(selected)), include_rows=False)
        bundle_rows.append({"group_id": as_jsonable(group), **result})

    def bundle_values(key: str) -> np.ndarray:
        """Lấy các scalar hữu hạn cùng tên từ mọi bundle để lập thống kê cấp hệ thống."""
        return np.asarray([row[key] for row in bundle_rows
                           if row.get(key) is not None and np.isfinite(float(row[key]))], float)

    bundle_summary = {
        "bundle_count": len(bundle_rows),
        "evaluable_bundle_count": sum(row["status"].startswith("EVALUATED") for row in bundle_rows),
        "bundle_curl_rms_median_per_mm": _finite_summary(
            bundle_values("knn_abs_curl_rms_per_mm"), "p50"),
        "bundle_curl_rms_p95_per_mm": _finite_summary(
            bundle_values("knn_abs_curl_rms_per_mm"), "p95"),
        "bundle_local_geometry_normal_rms_p95_deg": _finite_summary(
            bundle_values("local_geometry_normal_rms_deg"), "p95"),
        "bundle_loop_abs_circulation_p95_of_bundle_p95_mm": _finite_summary(
            bundle_values("loop_abs_circulation_p95_mm"), "p95"),
        "bundle_edge_gradient_height_residual_p95_of_bundle_p95_mm": _finite_summary(
            bundle_values("edge_gradient_height_residual_p95_mm"), "p95"),
        "bundle_nearest_abs_dz_p99_of_bundle_p99_mm": _finite_summary(
            bundle_values("nearest_abs_dz_p99_mm"), "p99"),
        "bundle_nearest_normal_angle_p99_of_bundle_p99_deg": _finite_summary(
            bundle_values("nearest_normal_angle_p99_deg"), "p99"),
    }
    return global_result | {
        "status": "EVALUATED_PER_FIELD_PUPIL_BUNDLE_AND_GLOBAL_MIXED_CLOUD",
        "grouping": "FIELD_PUPIL_BUNDLE",
        "global_mixed_cloud_diagnostic": {k: v for k, v in global_result.items()
                                           if k != "sample_rows"},
        "bundle_summary": bundle_summary,
        "bundle_rows": bundle_rows,
        "interpretation": (
            "BUNDLE_METRICS_ARE_PRIMARY; GLOBAL_MIXED_CLOUD_METRICS_ARE_DIAGNOSTIC_ONLY; "
            "FINITE_DISCRETE_TESTS_DO_NOT_PROVE_CONTINUOUS_INTEGRABILITY"),
    }


def fit_surface(points: np.ndarray, target_normals: np.ndarray, seed: PolySurface, order: int,
                axis_order2: bool, fit_weights: dict[str, float], vertex_index: int,
                fit_options: dict[str, Any] | None = None,
                incident_starts: np.ndarray | None = None,
                outgoing_targets: np.ndarray | None = None) -> tuple[PolySurface, dict[str, Any]]:
    """Fit sphere-conic-freeform nhat quan, co variable projection cho c, K va Aij."""
    if not 0 <= int(vertex_index) < len(points):
        raise ValueError("CI_CENTRAL_CHIEF_VERTEX_INDEX_INVALID")

    points = np.asarray(points, float)
    target_normals = np.asarray(target_normals, float)

    if target_normals.shape != points.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("CI_SURFACE_FIT_POINT_NORMAL_SHAPE_MISMATCH")

    if not np.all(np.isfinite(points)):
        raise RuntimeError("CI_SURFACE_FIT_NONFINITE_POINTS")

    if not np.all(np.isfinite(target_normals)):
        raise RuntimeError("CI_SURFACE_FIT_NONFINITE_NORMALS")

    target_normals = unit(target_normals)

    options = {} if fit_options is None else fit_options
    sphere_mode = options.get("sphere_mode", "CHIEF_VERTEX_CONSTRAINED_GEOMETRIC_SPHERE")
    if sphere_mode != "CHIEF_VERTEX_CONSTRAINED_GEOMETRIC_SPHERE":
        raise ValueError("SURFACE_FIT_SPHERE_MODE_UNSUPPORTED")

    # Algebraic LS gives a robust axis.  Its mathematically consistent radius
    # uses the solved constant d; this value is retained as an audit reference.
    sphere_A = np.column_stack([2.0 * points, np.ones(len(points))])
    sphere_b = np.sum(points * points, axis=1)
    sphere_sol = np.linalg.lstsq(sphere_A, sphere_b, rcond=1e-12)[0]
    algebraic_center = sphere_sol[:3]
    radius_squared = float(np.dot(algebraic_center, algebraic_center) + sphere_sol[3])
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        raise RuntimeError("CI_BASE_SPHERE_ALGEBRAIC_RADIUS_INVALID")
    algebraic_radius = math.sqrt(radius_squared)
    vertex_idx = int(vertex_index)
    vertex = points[vertex_idx]
    former_hybrid_radius = max(float(np.linalg.norm(algebraic_center-vertex)), 1e-6)

    # Refit the center while defining R=||C-Vchief|| inside the objective.  This
    # makes center, radius and chief vertex members of one geometric sphere.  A
    # weak center prior and explicit radius interval prevent the familiar
    # almost-planar infinite-radius escape on shallow freeform patches.
    radius_factors = np.asarray(options.get("sphere_radius_factor_bounds", [0.25, 4.0]), float)
    if radius_factors.shape != (2,) or radius_factors[0] <= 0.0 or radius_factors[0] >= radius_factors[1]:
        raise ValueError("SURFACE_FIT_SPHERE_RADIUS_FACTOR_BOUNDS_INVALID")
    radius_min = max(float(radius_factors[0] * algebraic_radius), 1e-6)
    radius_max = max(float(radius_factors[1] * algebraic_radius), radius_min * (1.0 + 1e-6))
    sphere_scale = max(float(np.std(np.linalg.norm(points-algebraic_center, axis=1))), 1e-4)
    center_regularization = float(options.get("sphere_center_regularization", 1e-4))

    # Toi uu q = R / R_alg va hai goc, khong toi uu Cx/Cy/Cz truc tiep.
    # He truc quay theo huong tam ban dau de tranh diem cuc tai khoi tao.
    sphere_axis = np.asarray(algebraic_center - vertex, float)
    axis_norm = float(np.linalg.norm(sphere_axis))
    if axis_norm < 1e-12:
        sphere_axis = np.asarray(seed.frame[:, 2], float).copy()
        axis_norm = float(np.linalg.norm(sphere_axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-12:
        raise RuntimeError("CI_CHIEF_VERTEX_SPHERE_INITIAL_AXIS_INVALID")
    sphere_axis /= axis_norm
    helper_axis = np.eye(3)[int(np.argmin(np.abs(sphere_axis)))]
    sphere_u = np.cross(sphere_axis, helper_axis)
    sphere_u /= np.linalg.norm(sphere_u)
    sphere_v = np.cross(sphere_axis, sphere_u)

    def sphere_from_parameters(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        """Doi q va hai goc thanh tam, ban kinh; sphere luon di qua chief."""
        q, alpha, beta = np.asarray(parameters, float)
        direction = (
            math.cos(beta) * (
                math.cos(alpha) * sphere_axis + math.sin(alpha) * sphere_u
            )
            + math.sin(beta) * sphere_v
        )
        direction /= np.linalg.norm(direction)
        candidate_radius = float(algebraic_radius * q)
        center_value = vertex + candidate_radius * direction
        return center_value, candidate_radius

    def constrained_sphere_residual(parameters: np.ndarray) -> np.ndarray:
        """Giu residual radial va prior cu; bounds thay cho phat ban kinh."""
        center_value, candidate_radius = sphere_from_parameters(parameters)
        radial = (
            np.linalg.norm(points - center_value, axis=1) - candidate_radius
        ) / sphere_scale
        prior = math.sqrt(max(center_regularization, 0.0)) * (
            center_value - algebraic_center
        ) / max(algebraic_radius, 1e-6)
        return np.concatenate([radial, prior])

    q_lower = radius_min / algebraic_radius
    q_upper = radius_max / algebraic_radius
    # Chi dua DIEM KHOI TAO vao mien hop le; khong clip nghiem sau toi uu.
    q_initial = float(np.clip(
        former_hybrid_radius / algebraic_radius, q_lower, q_upper
    ))
    constrained_sphere = least_squares(
        constrained_sphere_residual,
        np.array([q_initial, 0.0, 0.0]),
        bounds=(
            np.array([q_lower, -np.inf, -np.inf]),
            np.array([q_upper, np.inf, np.inf]),
        ),
        method="trf", jac="3-point",
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=int(options.get("sphere_max_nfev", 200)),
        x_scale=np.ones(3),
    )
    if not constrained_sphere.success or not np.all(np.isfinite(constrained_sphere.x)):
        raise RuntimeError(
            "CI_CHIEF_VERTEX_CONSTRAINED_SPHERE_FIT_FAILED: "
            f"status={constrained_sphere.status}; "
            f"nfev={constrained_sphere.nfev}; "
            f"message={constrained_sphere.message}"
        )
    sphere_center, radius = sphere_from_parameters(constrained_sphere.x)
    if (not np.all(np.isfinite(sphere_center)) or not np.isfinite(radius)
            or not radius_min*(1.0-1e-6) <= radius <= radius_max*(1.0+1e-6)):
        raise RuntimeError(
            "CI_CHIEF_VERTEX_CONSTRAINED_SPHERE_RADIUS_OUT_OF_BOUNDS: "
            f"R={radius:.12g}; Rmin={radius_min:.12g}; Rmax={radius_max:.12g}"
        )
    zaxis = unit(sphere_center - vertex)
    sphere_curvature_sign = 1.0
    if np.dot(zaxis, seed.frame[:, 2]) < 0.0:
        zaxis *= -1.0
        sphere_curvature_sign = -1.0

    xaxis = seed.frame[:, 0] - np.dot(seed.frame[:, 0], zaxis) * zaxis
    if np.linalg.norm(xaxis) < 1e-8:
        xaxis = seed.frame[:, 1] - np.dot(seed.frame[:, 1], zaxis) * zaxis
    if np.linalg.norm(xaxis) < 1e-8:
        fallback_axis = np.eye(3)[int(np.argmin(np.abs(zaxis)))]
        xaxis = fallback_axis - np.dot(fallback_axis, zaxis) * zaxis
    xaxis = unit(xaxis)
    yaxis = unit(np.cross(zaxis, xaxis))
    xaxis = unit(np.cross(yaxis, zaxis))
    frame = np.column_stack([xaxis, yaxis, zaxis])
    loc = (points - vertex) @ frame
    x, y, z = loc.T
    nloc = target_normals @ frame
    nloc[nloc[:, 2] < 0.0] *= -1.0
    abs_nz = np.abs(nloc[:, 2])

    # Tạo candidate clear aperture ngay từ đầu để làm miền ràng buộc giải tích conic
    candidate_poly, candidate_half = convex_aperture_from_points(
        loc[:, :2], margin_mm=0.5, floor=(1.0, 1.0))
    candidate_r2_max = float(np.max(np.sum(candidate_poly ** 2, axis=1)))

    scale = np.maximum(np.percentile(np.abs(loc[:, :2]), 99.5, axis=0), 1.0)
    c0 = float(np.clip(sphere_curvature_sign / max(radius, 1e-9), -0.2, 0.2))
    ws = math.sqrt(fit_weights["sag"]) / fit_weights["sag_scale_mm"]
    normal_angle_scale_deg = float(fit_weights.get(
        "normal_angle_scale_deg",
        math.degrees(float(fit_weights.get("slope_scale", 0.01)))))
    normal_angle_scale_rad = math.radians(normal_angle_scale_deg)
    if not normal_angle_scale_rad > 0.0:
        raise ValueError("SURFACE_FIT_NORMAL_ANGLE_SCALE_MUST_BE_POSITIVE")
    wn = math.sqrt(fit_weights["normal"]) / normal_angle_scale_rad
    preserve_fan_basis = bool(not axis_order2 and order >= 3)
    terms = monomial_terms(order, axis_order2=axis_order2, include_constant=True,
                           preserve_fan_axis_order2=preserve_fan_basis)
    model = PolySurface(seed.name, vertex.copy(), frame.copy(), candidate_half.copy(),
                        terms, np.zeros(len(terms)), scale, c0, 0.0, False,
                        aperture_polygon=candidate_poly.copy())
    A0 = model.basis(x, y)
    Ax = model.basis(x, y, "x")
    Ay = model.basis(x, y, "y")

    parent_trust_enabled = bool(
        options.get("parent_trust_enabled", False)
    )
    parent_trust_grid_samples = int(
        options.get("parent_trust_grid_samples", 15)
    )
    parent_trust_sag_lambda = float(
        options.get("parent_trust_sag_lambda", 0.0)
    )
    parent_trust_normal_lambda = float(
        options.get("parent_trust_normal_lambda", 0.0)
    )
    parent_trust_balance_enabled = bool(
        options.get(
            "parent_trust_balance_by_sample_count",
            True,
        )
    )

    trust_x = np.empty(0, dtype=float)
    trust_y = np.empty(0, dtype=float)
    trust_z = np.empty(0, dtype=float)
    trust_nloc = np.empty((0, 3), dtype=float)
    trust_A0 = np.empty((0, len(terms)), dtype=float)
    trust_Ax = np.empty((0, len(terms)), dtype=float)
    trust_Ay = np.empty((0, len(terms)), dtype=float)
    trust_r2 = np.empty(0, dtype=float)
    trust_sample_balance = 1.0

    if parent_trust_enabled:
        parent_trust_points = aperture_points(
            seed,
            margin=0.0,
            samples=parent_trust_grid_samples,
        )

        if (
            parent_trust_points.ndim != 2
            or parent_trust_points.shape[1] != 3
            or len(parent_trust_points) < 3
            or not np.all(np.isfinite(parent_trust_points))
        ):
            raise RuntimeError(
                "SURFACE_FIT_PARENT_TRUST_POINTS_INVALID"
            )

        parent_local = (
            parent_trust_points - seed.center
        ) @ seed.frame

        parent_normals_global = seed.normal(
            parent_local[:, 0],
            parent_local[:, 1],
        )

        trust_local = (
            parent_trust_points - vertex
        ) @ frame

        trust_x = trust_local[:, 0]
        trust_y = trust_local[:, 1]
        trust_z = trust_local[:, 2]

        trust_nloc = parent_normals_global @ frame
        trust_nloc[
            trust_nloc[:, 2] < 0.0
        ] *= -1.0

        if (
            not np.all(np.isfinite(trust_local))
            or not np.all(np.isfinite(trust_nloc))
        ):
            raise RuntimeError(
                "SURFACE_FIT_PARENT_TRUST_GEOMETRY_NONFINITE"
            )

        trust_A0 = model.basis(trust_x, trust_y)
        trust_Ax = model.basis(
            trust_x,
            trust_y,
            "x",
        )
        trust_Ay = model.basis(
            trust_x,
            trust_y,
            "y",
        )
        trust_r2 = trust_x * trust_x + trust_y * trust_y

        if parent_trust_balance_enabled:
            trust_sample_balance = math.sqrt(
                len(points) / max(len(trust_x), 1)
            )

    # Exact gauge: the chief point remains the analytic vertex (A00=0), while
    # the rotationally symmetric quadratic power belongs to the base conic
    # (A20+A02=0).  Astigmatic quadratic departure remains free.
    gauge_mode = options.get(
        "polynomial_gauge_mode", "FIX_CHIEF_PISTON_AND_REMOVE_SYMMETRIC_QUADRATIC")
    if gauge_mode != "FIX_CHIEF_PISTON_AND_REMOVE_SYMMETRIC_QUADRATIC":
        raise ValueError("SURFACE_FIT_POLYNOMIAL_GAUGE_MODE_UNSUPPORTED")
    gauge_rows: list[np.ndarray] = []
    gauge_labels: list[str] = []
    if (0, 0) in terms:
        row = np.zeros(len(terms)); row[terms.index((0, 0))] = 1.0
        gauge_rows.append(row); gauge_labels.append("A00=0_CHIEF_VERTEX_ON_ANALYTIC_SURFACE")
    if (2, 0) in terms and (0, 2) in terms:
        row = np.zeros(len(terms)); row[terms.index((2, 0))] = 1.0; row[terms.index((0, 2))] = 1.0
        gauge_rows.append(row); gauge_labels.append("A20+A02=0_BASE_CONIC_OWNS_SYMMETRIC_QUADRATIC_POWER")
    gauge_matrix = np.vstack(gauge_rows) if gauge_rows else np.zeros((0, len(terms)))
    if len(gauge_matrix):
        _, gauge_singular, gauge_vh = np.linalg.svd(gauge_matrix, full_matrices=True)
        gauge_rank = int(np.sum(gauge_singular > 1e-12))
        coefficient_nullspace = gauge_vh[gauge_rank:].T
    else:
        gauge_rank = 0
        coefficient_nullspace = np.eye(len(terms))
    if coefficient_nullspace.shape[1] == 0:
        raise RuntimeError("SURFACE_FIT_POLYNOMIAL_GAUGE_REMOVED_ALL_DEGREES_OF_FREEDOM")

    # n_target dot tangent_x/y = 0 is a unit-normal/tangent-plane residual.
    # It preserves the linear inner solve while avoiding the non-uniform raw
    # (p,q) Euclidean metric at steep slopes.
    trust_ws = (
        math.sqrt(parent_trust_sag_lambda)
        * ws
        * trust_sample_balance
    )
    trust_wn = (
        math.sqrt(parent_trust_normal_lambda)
        * wn
        * trust_sample_balance
    )

    cloud_design_rows = [
        ws * A0,
        wn * nloc[:, 2, None] * Ax,
        wn * nloc[:, 2, None] * Ay,
    ]

    trust_design_rows: list[np.ndarray] = []

    if (
        parent_trust_enabled
        and parent_trust_sag_lambda > 0.0
    ):
        trust_design_rows.append(
            trust_ws * trust_A0
        )

    if (
        parent_trust_enabled
        and parent_trust_normal_lambda > 0.0
    ):
        trust_design_rows.extend([
            trust_wn
            * trust_nloc[:, 2, None]
            * trust_Ax,
            trust_wn
            * trust_nloc[:, 2, None]
            * trust_Ay,
        ])

    design_data_full = np.vstack(
        cloud_design_rows + trust_design_rows
    )
    design_data = design_data_full @ coefficient_nullspace
    data_singular = np.linalg.svd(design_data, compute_uv=False)
    data_design_condition = (float(data_singular[0]/data_singular[-1])
                             if len(data_singular) and data_singular[-1] > 1e-15 else None)
    polynomial_regularization = float(
        options.get(
            "polynomial_departure_regularization",
            1e-6,
        )
    )
    if (
        not np.isfinite(polynomial_regularization)
        or polynomial_regularization < 0.0
    ):
        raise ValueError(
            "SURFACE_FIT_POLYNOMIAL_REGULARIZATION_MUST_BE_NONNEGATIVE"
        )

    degree_weight = np.asarray(
        [
            1.0 + i + j
            for i, j in terms
        ],
        float,
    )
    regularization_design = (
        math.sqrt(polynomial_regularization)
        * degree_weight[:, None]
        * coefficient_nullspace
    )

    o4_degree_regularization = float(
        options.get(
            "o4_degree_regularization",
            0.0,
        )
    )
    if (
        not np.isfinite(o4_degree_regularization)
        or o4_degree_regularization < 0.0
    ):
        raise ValueError(
            "SURFACE_FIT_O4_DEGREE_REGULARIZATION_MUST_BE_NONNEGATIVE"
        )

    o4_degree_mask = np.asarray(
        [
            i + j == 4
            for i, j in terms
        ],
        dtype=bool,
    )
    o4_degree_regularization_active = bool(
        o4_degree_regularization > 0.0
        and np.any(o4_degree_mask)
    )

    o4_regularization_design = (
        math.sqrt(o4_degree_regularization)
        * coefficient_nullspace[
            o4_degree_mask,
            :,
        ]
        if o4_degree_regularization_active
        else np.empty(
            (
                0,
                coefficient_nullspace.shape[1],
            ),
            dtype=float,
        )
    )

    design = np.vstack([
        design_data,
        regularization_design,
        o4_regularization_design,
    ])
    fixed_design_solver = None
    if selected_backend("fixed_design_lstsq") == "cpu":
        fixed_design_solver = FixedDesignLeastSquares(design, rcond=1e-11)
        singular = fixed_design_solver.singular_values
    else:
        singular = np.linalg.svd(design, compute_uv=False)
    regularized_design_condition = (float(singular[0]/singular[-1])
                                    if len(singular) and singular[-1] > 1e-15 else None)

    configured_k = options.get("conic_bounds", [-50.0, 50.0])
    if len(configured_k) != 2:
        raise ValueError("SURFACE_FIT_CONIC_BOUNDS_MUST_HAVE_TWO_VALUES")
    lower_k, configured_upper_k = map(float, configured_k)
    max_r2_values = [
        float(np.max(x * x + y * y)),
        candidate_r2_max,
    ]

    if parent_trust_enabled and len(trust_r2):
        max_r2_values.append(
            float(np.max(trust_r2))
        )

    max_r2 = max(max_r2_values)
    domain_upper_k = (1.0-1e-8) / max(c0*c0*max_r2, 1e-15) - 1.0
    initial_upper_k = min(configured_upper_k, domain_upper_k-1e-8)
    if not initial_upper_k > lower_k:
        raise RuntimeError("CI_CONIC_DOMAIN_INVALID")

    def conic_initial_residual(k_value: np.ndarray) -> np.ndarray:
        """Fit a sag-only K initializer before the joint sag/slope solve."""
        model.curvature = c0
        model.conic = float(k_value[0])
        zc, _, _ = model._conic(x, y)
        return ws*(zc-z)

    k_initial = float(np.clip(seed.conic, lower_k+1e-8, initial_upper_k-1e-8))
    conic_fit = least_squares(
        conic_initial_residual, np.array([k_initial]),
        bounds=(np.array([lower_k]), np.array([initial_upper_k])), method="trf",
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300, x_scale="jac")
    if not conic_fit.success or not np.all(np.isfinite(conic_fit.x)):
        raise RuntimeError("CI_BASE_CONIC_FIT_FAILED")
    k_fit = float(conic_fit.x[0])

    def projected_coefficients(curvature: float, conic: float
                               ) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
        """Eliminate the linear Aij coefficients for a proposed curvature/conic pair."""
        model.curvature = float(curvature)
        model.conic = float(conic)
        zc, gxc, gyc = model._conic(x, y)
        if parent_trust_enabled:
            (
                trust_zc,
                trust_gxc,
                trust_gyc,
            ) = model._conic(
                trust_x,
                trust_y,
            )
        else:
            trust_zc = np.empty(0, dtype=float)
            trust_gxc = np.empty(0, dtype=float)
            trust_gyc = np.empty(0, dtype=float)

        cloud_rhs_data = np.concatenate([
            ws * (z - zc),
            wn * (
                -nloc[:, 0]
                - nloc[:, 2] * gxc
            ),
            wn * (
                -nloc[:, 1]
                - nloc[:, 2] * gyc
            ),
        ])

        trust_rhs_parts: list[np.ndarray] = []

        if (
            parent_trust_enabled
            and parent_trust_sag_lambda > 0.0
        ):
            trust_rhs_parts.append(
                trust_ws
                * (trust_z - trust_zc)
            )

        if (
            parent_trust_enabled
            and parent_trust_normal_lambda > 0.0
        ):
            trust_rhs_parts.extend([
                trust_wn * (
                    -trust_nloc[:, 0]
                    - trust_nloc[:, 2]
                    * trust_gxc
                ),
                trust_wn * (
                    -trust_nloc[:, 1]
                    - trust_nloc[:, 2]
                    * trust_gyc
                ),
            ])

        trust_rhs_data = (
            np.concatenate(trust_rhs_parts)
            if trust_rhs_parts
            else np.empty(0, dtype=float)
        )

        rhs = np.concatenate([
            cloud_rhs_data,
            trust_rhs_data,
            np.zeros(
                len(terms),
                dtype=float,
            ),
            np.zeros(
                len(o4_regularization_design),
                dtype=float,
            ),
        ])

        if fixed_design_solver is None:
            reduced = np.linalg.lstsq(design, rhs, rcond=1e-11)[0]
        else:
            reduced = fixed_design_solver.solve(rhs)
        coeff_value = coefficient_nullspace @ reduced
        sag_residual = ws*(zc+A0@coeff_value-z)
        tangent_x_residual = wn*(nloc[:, 0]+nloc[:, 2]*(gxc+Ax@coeff_value))
        tangent_y_residual = wn*(nloc[:, 1]+nloc[:, 2]*(gyc+Ay@coeff_value))
        regularization_residual = (
            math.sqrt(polynomial_regularization)
            * degree_weight
            * coeff_value
        )

        o4_regularization_residual = (
            math.sqrt(o4_degree_regularization)
            * coeff_value[o4_degree_mask]
            if o4_degree_regularization_active
            else np.empty(
                0,
                dtype=float,
            )
        )

        trust_residual_parts: list[np.ndarray] = []

        if (
            parent_trust_enabled
            and parent_trust_sag_lambda > 0.0
        ):
            trust_residual_parts.append(
                trust_ws * (
                    trust_zc
                    + trust_A0 @ coeff_value
                    - trust_z
                )
            )

        if (
            parent_trust_enabled
            and parent_trust_normal_lambda > 0.0
        ):
            trust_residual_parts.extend([
                trust_wn * (
                    trust_nloc[:, 0]
                    + trust_nloc[:, 2] * (
                        trust_gxc
                        + trust_Ax @ coeff_value
                    )
                ),
                trust_wn * (
                    trust_nloc[:, 1]
                    + trust_nloc[:, 2] * (
                        trust_gyc
                        + trust_Ay @ coeff_value
                    )
                ),
            ])

        trust_residual = (
            np.concatenate(trust_residual_parts)
            if trust_residual_parts
            else np.empty(0, dtype=float)
        )

        residual_value = np.concatenate([
            sag_residual,
            tangent_x_residual,
            tangent_y_residual,
            trust_residual,
            regularization_residual,
            o4_regularization_residual,
        ])

        domain_cloud = 1.0 - (1.0 + conic) * curvature * curvature * (x * x + y * y)
        domain_aperture = 1.0 - (1.0 + conic) * curvature * curvature * candidate_r2_max
        domain_trust = (
            1.0
            - (1.0 + conic)
            * curvature
            * curvature
            * trust_r2
            if parent_trust_enabled
            and len(trust_r2)
            else np.empty(0, dtype=float)
        )

        domain_candidates = [
            float(np.min(domain_cloud))
        ]

        if conic > -1.0:
            domain_candidates.append(
                float(domain_aperture)
            )

        if len(domain_trust):
            domain_candidates.append(
                float(np.min(domain_trust))
            )

        min_domain = min(domain_candidates)

        components = {
            "sag_weighted_rms": float(
                np.sqrt(
                    np.mean(
                        sag_residual**2
                    )
                )
            ),
            "unit_normal_tangent_weighted_rms": float(
                np.sqrt(
                    np.mean(
                        np.r_[
                            tangent_x_residual,
                            tangent_y_residual,
                        ]**2
                    )
                )
            ),
            "polynomial_regularization_norm": float(
                np.linalg.norm(
                    regularization_residual
                )
            ),
            "o4_degree_regularization_norm": float(
                np.linalg.norm(
                    o4_regularization_residual
                )
            ),
        }
        return coeff_value, residual_value, min_domain, components

    joint_enabled = bool(options.get("joint_variable_projection_enabled", True))
    relative = float(options.get("curvature_relative_bound", 0.35))
    absolute_max = float(options.get("curvature_absolute_max_per_mm", 0.2))
    raw_c_bounds = [c0*(1.0-relative), c0*(1.0+relative)]
    c_bounds = [max(-absolute_max, min(raw_c_bounds)), min(absolute_max, max(raw_c_bounds))]
    if c_bounds[1]-c_bounds[0] <= 1e-12:
        joint_enabled = False
    joint_result = None
    joint_accepted = False
    model.curvature, model.conic = c0, k_fit
    if joint_enabled:
        regularization = float(options.get("nonlinear_regularization", 1e-3))

        def variable_projection_residual(values: np.ndarray) -> np.ndarray:
            """Evaluate the reduced nonlinear residual after solving all Aij terms."""
            curvature_value, conic_value = map(float, values)
            _, residual_value, minimum_domain, _ = projected_coefficients(curvature_value, conic_value)
            extra = np.array([
                regularization*(curvature_value-c0)/max(abs(c0), 1e-6),
                regularization*(conic_value-k_fit)/max(abs(k_fit), 1.0),
                1e6*max(0.0, 1e-8-minimum_domain),
            ])
            return np.concatenate([residual_value, extra])

        joint_result = least_squares(
            variable_projection_residual, np.array([c0, k_fit]),
            bounds=(np.array([c_bounds[0], lower_k]),
                    np.array([c_bounds[1], configured_upper_k])), method="trf",
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=int(options.get("joint_max_nfev", 80)),
            x_scale=np.array([max(abs(c0), 1e-4), max(abs(k_fit), 1.0)]))
        if joint_result.success and np.all(np.isfinite(joint_result.x)):
            candidate_curvature, candidate_conic = map(float, joint_result.x)
            candidate_cloud_domain = float(np.min(
                1.0-(1.0+candidate_conic)*candidate_curvature*candidate_curvature*(x*x+y*y)))
            candidate_aperture_domain = float(
                1.0-(1.0+candidate_conic)*candidate_curvature*candidate_curvature*candidate_r2_max)
            domain_candidates = [candidate_cloud_domain]
            if candidate_conic > -1.0:
                domain_candidates.append(candidate_aperture_domain)
            if parent_trust_enabled and len(trust_r2):
                candidate_trust_domain = float(np.min(
                    1.0 - (1.0 + candidate_conic) * candidate_curvature * candidate_curvature * trust_r2
                ))
                domain_candidates.append(candidate_trust_domain)
            candidate_min_domain = min(domain_candidates)
            if candidate_min_domain > 1e-8:
                model.curvature, model.conic = candidate_curvature, candidate_conic
                joint_accepted = True
            else:
                model.curvature, model.conic = c0, k_fit
        else:
            model.curvature, model.conic = c0, k_fit
    coeff, _, domain_margin, objective_components = projected_coefficients(
        model.curvature, model.conic)
    model.coeff = coeff

    parent_trust_stats = {
        "enabled": parent_trust_enabled,
        "sample_count": int(len(trust_x)),
        "grid_samples": parent_trust_grid_samples,
        "sag_lambda": parent_trust_sag_lambda,
        "normal_lambda": parent_trust_normal_lambda,
        "sample_balance": trust_sample_balance,
        "sag_rms_mm": None,
        "sag_max_mm": None,
        "normal_rms_deg": None,
        "normal_max_deg": None,
    }

    if parent_trust_enabled and len(trust_x):
        (
            trust_fit_z,
            trust_fit_gx,
            trust_fit_gy,
        ) = model.sag_slopes(
            trust_x,
            trust_y,
        )

        trust_sag_delta = (
            trust_fit_z - trust_z
        )

        trust_fit_normal = unit(
            np.column_stack([
                -trust_fit_gx,
                -trust_fit_gy,
                np.ones_like(trust_fit_gx),
            ])
        )

        trust_normal_angle = np.degrees(
            np.arccos(
                np.clip(
                    np.sum(
                        trust_fit_normal
                        * trust_nloc,
                        axis=1,
                    ),
                    -1.0,
                    1.0,
                )
            )
        )

        parent_trust_stats.update({
            "sag_rms_mm": float(
                np.sqrt(
                    np.mean(
                        trust_sag_delta
                        * trust_sag_delta
                    )
                )
            ),
            "sag_max_mm": float(
                np.max(
                    np.abs(trust_sag_delta)
                )
            ),
            "normal_rms_deg": float(
                np.sqrt(
                    np.mean(
                        trust_normal_angle
                        * trust_normal_angle
                    )
                )
            ),
            "normal_max_deg": float(
                np.max(trust_normal_angle)
            ),
        })

    fit_z, fit_px, fit_py = model.sag_slopes(x, y)
    nfit = unit(np.column_stack([-fit_px, -fit_py, np.ones_like(fit_px)]))
    angle = np.degrees(np.arccos(np.clip(np.sum(nfit*nloc, axis=1), -1.0, 1.0)))
    sag_err = fit_z-z
    model.aperture_polygon = candidate_poly
    model.half_aperture = candidate_half

    reflection_stats: dict[str, Any] = {"evaluated": False}
    if incident_starts is not None and outgoing_targets is not None:
        starts = np.asarray(incident_starts, float)
        targets = np.asarray(outgoing_targets, float)
        if starts.shape != points.shape or targets.shape != points.shape:
            raise ValueError("SURFACE_FIT_REFLECTION_DIAGNOSTIC_SHAPE_MISMATCH")
        fitted_points = model.point(x, y)
        actual_out = reflect(unit(fitted_points-starts), model.normal(x, y))
        required_out = unit(targets-fitted_points)
        reflection_angle = np.degrees(np.arccos(
            np.clip(np.sum(actual_out*required_out, axis=1), -1.0, 1.0)))
        reflection_stats = {
            "evaluated": True,
            "rms_deg": float(np.sqrt(np.mean(reflection_angle**2))),
            "p95_deg": float(np.percentile(reflection_angle, 95)),
            "max_deg": float(np.max(reflection_angle)),
        }

    # Profile K with c fixed at the selected value and refit every allowed Aij.
    # This distinguishes an isolated boundary hit from a monotone decomposition
    # drift without mechanically widening the physical K interval.
    profile_enabled = bool(options.get("k_profile_enabled", True))
    profile_count = max(int(options.get("k_profile_sample_count", 11)), 3)
    profile_values = (np.linspace(lower_k, configured_upper_k, profile_count)
                      if profile_enabled else np.asarray([], float))
    if profile_enabled and np.all(np.abs(profile_values-model.conic) > 1e-10):
        profile_values = np.sort(np.r_[profile_values, model.conic])
    k_profile = []
    selected_coeff = model.coeff.copy()
    selected_conic = float(model.conic)
    for profile_k in profile_values:
        profile_coeff, _, profile_domain, profile_components = projected_coefficients(
            model.curvature, float(profile_k))
        if profile_domain <= 1e-8:
            k_profile.append({"K": float(profile_k), "domain_valid": False,
                              "minimum_conic_domain_argument": profile_domain})
            continue
        model.conic = float(profile_k); model.coeff = profile_coeff
        profile_z, profile_gx, profile_gy = model.sag_slopes(x, y)
        profile_normal = unit(np.column_stack([-profile_gx, -profile_gy, np.ones_like(profile_gx)]))
        profile_angle = np.degrees(np.arccos(np.clip(
            np.sum(profile_normal*nloc, axis=1), -1.0, 1.0)))
        profile_row = {
            "K": float(profile_k), "domain_valid": True,
            "minimum_conic_domain_argument": profile_domain,
            "sag_rms_mm": float(np.sqrt(np.mean((profile_z-z)**2))),
            "normal_rms_deg": float(np.sqrt(np.mean(profile_angle**2))),
            **profile_components,
        }
        if incident_starts is not None and outgoing_targets is not None:
            profile_points = model.point(x, y)
            profile_actual_out = reflect(unit(profile_points-starts), model.normal(x, y))
            profile_required_out = unit(targets-profile_points)
            profile_reflection = np.degrees(np.arccos(np.clip(
                np.sum(profile_actual_out*profile_required_out, axis=1), -1.0, 1.0)))
            profile_row["reflection_rms_deg"] = float(np.sqrt(np.mean(profile_reflection**2)))
        k_profile.append(profile_row)
    model.conic = selected_conic; model.coeff = selected_coeff

    unconstrained_radial = np.linalg.norm(points-algebraic_center, axis=1)-algebraic_radius
    base_sphere_radial = np.linalg.norm(points-sphere_center, axis=1)-radius
    k_span = max(configured_upper_k-lower_k, 1.0)
    k_bound_hit = bool(abs(model.conic-lower_k) <= 1e-4*k_span
                       or abs(model.conic-configured_upper_k) <= 1e-4*k_span)
    c_span = max(c_bounds[1]-c_bounds[0], 1e-12)
    curvature_bound_hit = bool(abs(model.curvature-c_bounds[0]) <= 1e-4*c_span
                               or abs(model.curvature-c_bounds[1]) <= 1e-4*c_span)
    stats = {
        "basis_type": ("FAN_EQ1_AXIS_ORDER_2" if axis_order2 else
                       f"FAN_AXIS_ORDER2_UNION_TOTAL_XY_ORDER_{order}"
                       if order == 3 else f"CONIC_PLUS_TOTAL_XY_ORDER_{order}"),
        "basis_nested_over_fan_axis_order2": bool(axis_order2 or preserve_fan_basis),
        "FAN_EQ1_TOTAL_DEGREE_TRUNCATION": False if axis_order2 else None,
        "curvature_per_mm": model.curvature, "conic_constant": model.conic,
        "conic_fit_status": ("JOINT_CURVATURE_CONIC_WITH_GAUGED_LINEAR_AIJ_VARIABLE_PROJECTION"
                             if joint_accepted else "CHIEF_CONSTRAINED_SPHERE_THEN_GAUGED_K_AND_AIJ"),
        "curvature_locked_from_base_sphere": not joint_accepted,
        "central_chief_vertex_index": vertex_idx,
        "base_sphere_mode": sphere_mode,
        "base_sphere_center_mm": sphere_center, "base_sphere_radius_mm": radius,
        "base_sphere_unconstrained_center_mm": algebraic_center,
        "base_sphere_unconstrained_radius_mm": algebraic_radius,
        "former_hybrid_vertex_radius_mm": former_hybrid_radius,
        "chief_vertex_radial_residual_to_geometric_sphere_mm": float(
            np.linalg.norm(sphere_center-vertex)-radius),
        "unconstrained_sphere_radial_rms_mm": float(np.sqrt(np.mean(unconstrained_radial**2))),
        "base_sphere_radial_rms_mm": float(np.sqrt(np.mean(base_sphere_radial**2))),
        "constrained_sphere_fit": {
            "success": bool(constrained_sphere.success),
            "status": int(constrained_sphere.status),
            "message": str(constrained_sphere.message),
            "nfev": int(constrained_sphere.nfev),
            "cost": float(constrained_sphere.cost),
            "optimality": float(constrained_sphere.optimality),
            "parameterization": "C=Vchief+(q*Ralg)*u(alpha,beta)",
            "parameter_order": ["radius_factor", "alpha_rad", "beta_rad"],
            "bound_enforcement": "SCIPY_LEAST_SQUARES_BOUNDS",
            "radius_mm": radius,
            "radius_factor": float(radius / algebraic_radius),
            "radius_bounds_mm": [radius_min, radius_max],
            "radius_bound_active": int(constrained_sphere.active_mask[0]),
            "radius_within_bounds": bool(
                radius_min*(1.0-1e-6) <= radius <= radius_max*(1.0+1e-6)
            ),
            "center_regularization": center_regularization,
            "chief_vertex_constraint": "R=norm(C-Vchief)_BY_PARAMETERIZATION",
            "signed_curvature_orientation": sphere_curvature_sign,
        },
        "local_vertex_mm": vertex, "local_frame_columns": frame,
        "base_conic_fit": {"success": bool(conic_fit.success), "nfev": int(conic_fit.nfev),
                           "cost": float(conic_fit.cost), "lower_k": lower_k,
                           "upper_k": initial_upper_k,
                           "objective": "SAG_INITIALIZATION_BEFORE_JOINT_VARIABLE_PROJECTION"},
        "joint_variable_projection": {
            "enabled": bool(joint_enabled),
            "accepted": bool(joint_accepted),
            "success": None if joint_result is None else bool(joint_result.success),
            "nfev": None if joint_result is None else int(joint_result.nfev),
            "curvature_bounds_per_mm": c_bounds,
            "conic_bounds": [lower_k, configured_upper_k],
            "minimum_conic_domain_argument": domain_margin,
            "objective_components": objective_components,
        },
        "polynomial_gauge": {
            "mode": gauge_mode,
            "constraints": gauge_labels,
            "constraint_rank": gauge_rank,
            "free_coefficient_dimension": int(
                coefficient_nullspace.shape[1]
            ),
            "maximum_constraint_residual": (
                float(
                    np.max(
                        np.abs(
                            gauge_matrix @ model.coeff
                        )
                    )
                )
                if len(gauge_matrix)
                else 0.0
            ),
            "departure_regularization":
                polynomial_regularization,
            "o4_degree_regularization":
                o4_degree_regularization,
            "o4_degree_regularization_active":
                o4_degree_regularization_active,
            "o4_regularized_terms": [
                [
                    int(i),
                    int(j),
                ]
                for i, j in terms
                if i + j == 4
            ],
        },
        "normal_fit_metric": {
            "mode": "TARGET_UNIT_NORMAL_DOT_SURFACE_TANGENTS",
            "scale_deg": normal_angle_scale_deg,
            "raw_slope_euclidean_objective_used": False,
            "reported_unit_normal_angle_role": "POST_FIT_GEOMETRIC_VALIDATION",
        },
        "k_profile_scan": {
            "enabled": profile_enabled,
            "curvature_fixed_per_mm": float(model.curvature),
            "Aij_refit_at_every_K": True,
            "selected_K": float(model.conic),
            "samples": k_profile,
        },
        "k_bound_hit": k_bound_hit,
        "curvature_bound_hit": curvature_bound_hit,
        "linear_design_condition": data_design_condition,
        "regularized_reduced_design_condition": regularized_design_condition,
        "abs_local_normal_z_min": float(np.min(abs_nz)),
        "abs_local_normal_z_p01": float(np.percentile(abs_nz, 1)),
        "abs_local_normal_z_p05": float(np.percentile(abs_nz, 5)),
        "sag_rms_mm": float(np.sqrt(np.mean(sag_err**2))),
        "sag_p95_mm": float(np.percentile(np.abs(sag_err), 95)),
        "sag_max_mm": float(np.max(np.abs(sag_err))),
        "normal_rms_deg": float(np.sqrt(np.mean(angle**2))),
        "normal_p95_deg": float(np.percentile(angle, 95)),
        "normal_max_deg": float(np.max(angle)),
        "reflection_direction_error": reflection_stats,
        "fit_weights": fit_weights, "terms": terms, "half_aperture_mm": model.half_aperture,
    }
    stats["parent_trust"] = parent_trust_stats
    return model, stats


def symmetric_conjugate(points: np.ndarray, plane: PolySurface) -> np.ndarray:
    # Mirror a point through the planar M2 surface; exact only in Fan Step One.
    """Tính điểm liên hợp đối xứng quanh mặt phẳng chuẩn."""
    n = plane.frame[:, 2]
    signed = (points - plane.center) @ n
    return points - 2.0 * signed[:, None] * n


def solve_fermat_m2(q1: np.ndarray, targets: np.ndarray, m2: PolySurface, initial: np.ndarray,
                    max_iter: int, grad_tol: float, construction_domain_factor: float = 3.0,
                    reflection_tolerance: float = 1e-4, *,
                    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
                    progress_interval_seconds: float = 30.0) -> dict[str, Any]:
    """Giải điều kiện đường quang dừng và kiểm tra nhánh phản xạ hợp lệ để dựng mục tiêu M2."""
    solver_started = time.perf_counter()

    def report_progress(event: str, **details: Any) -> None:
        """Forward optional progress without changing the numerical path."""
        if progress_callback is not None:
            progress_callback(event, details)

    loc = (initial - m2.center) @ m2.frame
    xy = loc[:, :2].copy()
    op0 = None
    converged = np.zeros(len(q1), bool)
    last_step = np.full(len(q1), np.inf)

    def eval_g(
        xyv: np.ndarray,
        *,
        q1_values: np.ndarray | None = None,
        target_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tính hệ stationarity, điểm giao, chiều dài đường quang và residual phản xạ."""
        q1_eval = q1 if q1_values is None else q1_values
        targets_eval = targets if target_values is None else target_values

        xyv = np.asarray(xyv, dtype=float)
        q1_eval = np.asarray(q1_eval, dtype=float)
        targets_eval = np.asarray(targets_eval, dtype=float)

        if xyv.ndim != 2 or xyv.shape[1] != 2:
            raise ValueError("FERMAT_XY_SHAPE_MISMATCH")
        if (
            q1_eval.shape != (len(xyv), 3)
            or targets_eval.shape != (len(xyv), 3)
        ):
            raise ValueError("FERMAT_ENDPOINT_SHAPE_MISMATCH")

        fermat_backend = selected_backend(
            "fermat_eval"
        )
        use_dispatched_kernel = (
            fermat_backend != "reference"
        )

        if fermat_backend == "cuda":
            runtime = current_runtime()
            gpu_config = (
                runtime.config.get("gpu", {})
                if runtime is not None
                else {}
            )
            min_cuda_rows = int(
                gpu_config.get(
                    "min_fermat_rows", 256
                )
            )

            use_dispatched_kernel = (
                len(xyv) >= min_cuda_rows
            )

        if use_dispatched_kernel:
            return dispatch_kernel(
                "fermat_eval",
                {
                    "xy": xyv,
                    "q1": q1_eval,
                    "targets": targets_eval,
                    "center": m2.center,
                    "frame": m2.frame,
                    "scale": m2.scale,
                    "terms": list(m2.terms),
                    "coeff": m2.coeff,
                    "curvature": float(
                        m2.curvature
                    ),
                    "conic": float(m2.conic),
                    "eps": float(EPS),
                },
            )

        p = m2.point(xyv[:, 0], xyv[:, 1])
        _, gx, gy = m2.sag_slopes(xyv[:, 0], xyv[:, 1])
        tx = m2.frame[:, 0] + gx[:, None] * m2.frame[:, 2]
        ty = m2.frame[:, 1] + gy[:, None] * m2.frame[:, 2]
        nloc = unit(np.column_stack([-gx, -gy, np.ones_like(gx)]))
        n = nloc @ m2.frame.T
        din = unit(p - q1_eval)
        dout = unit(targets_eval - p)
        s = din - dout
        g = np.column_stack([np.sum(s * tx, axis=1), np.sum(s * ty, axis=1)])
        op = np.linalg.norm(p - q1_eval, axis=1) + np.linalg.norm(targets_eval - p, axis=1)
        cos_in = np.sum(din * n, axis=1)[:, None]
        d_reflect = unit(din - 2.0 * cos_in * n)
        refl_res = np.linalg.norm(dout - d_reflect, axis=1)
        return g, p, op, refl_res

    fermat_backend = selected_backend("fermat_eval")
    runtime = current_runtime()
    if fermat_backend == "reference":
        for it in range(max_iter):
            iteration_started = time.perf_counter()
            g, p, op, refl_res = eval_g(xy)
            if op0 is None: op0 = op.copy()
            gn = np.linalg.norm(g, axis=1)
            refl_pass = refl_res <= float(reflection_tolerance)
            converged = (gn <= grad_tol) & refl_pass
            active_before = len(xy)
            active_after = int(np.count_nonzero(~converged))
            if np.all(converged):
                report_progress(
                    "newton_iteration",
                    iteration=it + 1,
                    max_iterations=max_iter,
                    active_before=active_before,
                    active_after=active_after,
                    backend=fermat_backend,
                    iteration_seconds=time.perf_counter() - iteration_started,
                    elapsed_seconds=time.perf_counter() - solver_started,
                )
                break
            h = np.maximum(1e-5, 1e-6 * np.maximum(np.linalg.norm(xy, axis=1), 1.0))
            xp, xm, yp, ym = xy.copy(), xy.copy(), xy.copy(), xy.copy()
            xp[:, 0] += h; xm[:, 0] -= h; yp[:, 1] += h; ym[:, 1] -= h
            gx1, _, _, _ = eval_g(xp); gx0, _, _, _ = eval_g(xm)
            gy1, _, _, _ = eval_g(yp); gy0, _, _, _ = eval_g(ym)
            J = np.empty((len(xy), 2, 2))
            J[:, :, 0] = (gx1 - gx0) / (2 * h[:, None])
            J[:, :, 1] = (gy1 - gy0) / (2 * h[:, None])
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            good = np.abs(det) > 1e-12
            step = np.zeros_like(xy)
            step[good, 0] = (-g[good, 0] * J[good, 1, 1] + g[good, 1] * J[good, 0, 1]) / det[good]
            step[good, 1] = (-J[good, 0, 0] * g[good, 1] + J[good, 1, 0] * g[good, 0]) / det[good]
            step = np.clip(step, -2.0, 2.0)
            xy += 0.7 * step
            last_step = np.linalg.norm(step, axis=1)
            report_progress(
                "newton_iteration",
                iteration=it + 1,
                max_iterations=max_iter,
                active_before=active_before,
                active_after=active_after,
                backend=fermat_backend,
                iteration_seconds=time.perf_counter() - iteration_started,
                elapsed_seconds=time.perf_counter() - solver_started,
            )
    else:
        ray_count = len(q1)
        op0 = np.full(ray_count, np.nan, dtype=float)
        last_step = np.full(ray_count, np.inf, dtype=float)
        active = np.ones(ray_count, dtype=bool)

        for it in range(max_iter):
            iteration_started = time.perf_counter()
            active_ids = np.flatnonzero(active)
            if len(active_ids) == 0:
                break

            xy_active = xy[active_ids]
            h = np.maximum(
                1e-5,
                1e-6 * np.maximum(np.linalg.norm(xy_active, axis=1), 1.0),
            )

            base_xy = xy_active.copy()
            xp = xy_active.copy()
            xm = xy_active.copy()
            yp = xy_active.copy()
            ym = xy_active.copy()

            xp[:, 0] += h
            xm[:, 0] -= h
            yp[:, 1] += h
            ym[:, 1] -= h

            stacked_xy = np.concatenate((base_xy, xp, xm, yp, ym), axis=0)
            q1_active = q1[active_ids]
            targets_active = targets[active_ids]

            stacked_q1 = np.concatenate((q1_active, q1_active, q1_active, q1_active, q1_active), axis=0)
            stacked_targets = np.concatenate((targets_active, targets_active, targets_active, targets_active, targets_active), axis=0)

            g_all, p_all, op_all, refl_all = eval_g(
                stacked_xy,
                q1_values=stacked_q1,
                target_values=stacked_targets,
            )

            n = len(active_ids)
            g = g_all[0:n]
            p = p_all[0:n]
            op = op_all[0:n]
            refl_res = refl_all[0:n]

            gx1 = g_all[n:2*n]
            gx0 = g_all[2*n:3*n]
            gy1 = g_all[3*n:4*n]
            gy0 = g_all[4*n:5*n]

            missing_op0 = np.isnan(op0[active_ids])
            op0[active_ids[missing_op0]] = op[missing_op0]

            gn = np.linalg.norm(g, axis=1)
            reflection_ok = refl_res <= float(reflection_tolerance)
            converged_local = (gn <= grad_tol) & reflection_ok

            active[active_ids[converged_local]] = False

            J = np.empty((n, 2, 2), dtype=float)
            J[:, :, 0] = (gx1 - gx0) / (2.0 * h[:, None])
            J[:, :, 1] = (gy1 - gy0) / (2.0 * h[:, None])

            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            solve_local = ~converged_local & (np.abs(det) > 1e-12)

            solve_ids = active_ids[solve_local]
            if len(solve_ids) > 0:
                g0 = g[solve_local, 0]
                g1 = g[solve_local, 1]
                J00 = J[solve_local, 0, 0]
                J01 = J[solve_local, 0, 1]
                J10 = J[solve_local, 1, 0]
                J11 = J[solve_local, 1, 1]
                det_s = det[solve_local]

                step_local = np.empty((len(solve_ids), 2), dtype=float)
                step_local[:, 0] = (-g0 * J11 + g1 * J01) / det_s
                step_local[:, 1] = (-J00 * g1 + J10 * g0) / det_s

                step_local = np.clip(step_local, -2.0, 2.0)
                xy[solve_ids] += 0.7 * step_local
                last_step[solve_ids] = np.linalg.norm(step_local, axis=1)

            if runtime is not None:
                runtime.record(
                    "FERMAT_ITERATION",
                    iteration=it,
                    active_rays_before=int(n),
                    active_rays_after=int(np.count_nonzero(active)),
                    rows_sent_to_fermat_eval=int(len(stacked_xy)),
                    backend=fermat_backend,
                )

            report_progress(
                "newton_iteration",
                iteration=it + 1,
                max_iterations=max_iter,
                active_before=int(n),
                active_after=int(np.count_nonzero(active)),
                backend=fermat_backend,
                iteration_seconds=time.perf_counter() - iteration_started,
                elapsed_seconds=time.perf_counter() - solver_started,
            )

    g, p, op, refl_res = eval_g(xy)
    fallback_used = np.zeros(len(xy), bool)
    unconverged_idx = np.where(~((np.linalg.norm(g, axis=1) <= grad_tol) & (refl_res <= float(reflection_tolerance))))[0]
    fallback_total = int(len(unconverged_idx))
    fallback_started = time.perf_counter()
    fallback_last_report = fallback_started
    fallback_accepted = 0
    fallback_numeric_failure = np.zeros(len(xy), bool)
    fallback_invalid_start_count = 0
    fallback_solver_error_count = 0
    report_progress(
        "fallback_started",
        total=fallback_total,
        elapsed_seconds=fallback_started - solver_started,
    )
    if runtime is not None:
        runtime.record(
            "FERMAT_FALLBACK_STARTED",
            unresolved_rays=fallback_total,
        )

    construction_bound = (
        max(float(construction_domain_factor), 1.0)
        * np.asarray(m2.half_aperture, dtype=float)
    )
    if (
        construction_bound.shape != (2,)
        or not np.all(np.isfinite(construction_bound))
        or np.any(construction_bound <= 0.0)
    ):
        raise RuntimeError("FERMAT_CONSTRUCTION_DOMAIN_BOUNDS_INVALID")

    for fallback_position, ridx in enumerate(unconverged_idx, start=1):
        q1_one = q1[ridx:ridx + 1]
        target_one = targets[ridx:ridx + 1]
        fallback_used[ridx] = True

        def fun(v: np.ndarray) -> np.ndarray:
            """Trả residual đã chuẩn hóa cho bộ giải số."""
            xv, yv = np.array([v[0]]), np.array([v[1]])
            gv, _, _, _ = eval_g(
                np.column_stack([xv, yv]),
                q1_values=q1_one,
                target_values=target_one,
            )
            return gv[0]

        def optical_path(v: np.ndarray) -> float:
            """Tính tổng chiều dài đường quang của nhánh tia."""
            pv = m2.point(np.array([v[0]]), np.array([v[1]]))[0]
            return float(np.linalg.norm(pv - q1[ridx]) + np.linalg.norm(targets[ridx] - pv))

        bound = construction_bound
        initial_xy = ((initial[ridx] - m2.center) @ m2.frame)[:2]
        starts = [xy[ridx], initial_xy, np.zeros(2), 0.5 * (xy[ridx] + initial_xy)]
        candidates = []
        reference_xy = None
        for start in starts:
            start_value = np.asarray(start, dtype=float)
            if start_value.shape != (2,) or not np.all(np.isfinite(start_value)):
                fallback_invalid_start_count += 1
                continue

            x0 = np.clip(start_value, -0.999 * bound, 0.999 * bound)
            if not np.all(np.isfinite(x0)):
                fallback_invalid_start_count += 1
                continue
            if reference_xy is None:
                reference_xy = x0.copy()

            trials = [x0]
            try:
                op_opt = minimize(
                    optical_path,
                    x0,
                    method="Powell",
                    bounds=list(zip(-bound, bound)),
                    options={"maxiter": 500, "xtol": 1e-11, "ftol": 1e-13},
                )
                op_trial = np.asarray(op_opt.x, dtype=float)
                if op_trial.shape == (2,) and np.all(np.isfinite(op_trial)):
                    trials.append(op_trial)
                else:
                    fallback_invalid_start_count += 1
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                fallback_solver_error_count += 1

            for trial in trials:
                trial_start = np.clip(
                    np.asarray(trial, dtype=float),
                    -0.999 * bound,
                    0.999 * bound,
                )
                if (
                    trial_start.shape != (2,)
                    or not np.all(np.isfinite(trial_start))
                ):
                    fallback_invalid_start_count += 1
                    continue
                try:
                    ls = least_squares(
                        fun,
                        trial_start,
                        bounds=(-bound, bound),
                        method="trf",
                        xtol=1e-13,
                        ftol=1e-13,
                        gtol=1e-13,
                        max_nfev=800,
                        x_scale="jac",
                    )
                    ls_xy = np.asarray(ls.x, dtype=float)
                    if ls_xy.shape != (2,) or not np.all(np.isfinite(ls_xy)):
                        fallback_invalid_start_count += 1
                        continue
                    trial_xy = ls_xy.reshape(1, 2)
                    g_t, p_t, op_t, refl_t = eval_g(
                        trial_xy,
                        q1_values=q1_one,
                        target_values=target_one,
                    )
                    gn_t = float(np.linalg.norm(g_t[0]))
                    refl_val = float(refl_t[0])
                    op_value = float(op_t[0])
                    conic_arg = float(
                        1.0
                        - (1.0 + m2.conic)
                        * m2.curvature ** 2
                        * np.sum(trial_xy ** 2)
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    fallback_solver_error_count += 1
                    continue

                if not np.all(np.isfinite(
                    [gn_t, refl_val, op_value, conic_arg]
                )):
                    fallback_invalid_start_count += 1
                    continue
                is_admissible = bool(
                    refl_val <= float(reflection_tolerance)
                    and gn_t <= grad_tol
                    and (
                        conic_arg >= -CONIC_DOMAIN_TOLERANCE
                        or m2.conic <= -1.0
                    )
                )
                candidates.append(
                    (is_admissible, refl_val, gn_t, op_value, ls_xy)
                )

        if candidates:
            best = min(
                candidates,
                key=lambda z: (not z[0], z[1], z[2], z[3]),
            )
            xy[ridx] = best[4]
            last_step[ridx] = float(
                np.linalg.norm(best[4] - reference_xy)
            )
            fallback_accepted += int(bool(best[0]))
        else:
            fallback_numeric_failure[ridx] = True
            xy[ridx] = (
                reference_xy
                if reference_xy is not None
                else np.zeros(2, dtype=float)
            )
            last_step[ridx] = float("inf")

        now = time.perf_counter()
        should_report = (
            fallback_position == 1
            or fallback_position == fallback_total
            or now - fallback_last_report >= max(float(progress_interval_seconds), 0.1)
        )
        if should_report:
            fallback_elapsed = now - fallback_started
            rate = fallback_position / max(fallback_elapsed, 1e-12)
            remaining = fallback_total - fallback_position
            eta_seconds = remaining / rate if rate > 0.0 else float("inf")
            fallback_details = {
                "processed": int(fallback_position),
                "total": fallback_total,
                "accepted": int(fallback_accepted),
                "numeric_failed": int(np.count_nonzero(fallback_numeric_failure)),
                "elapsed_seconds": fallback_elapsed,
                "eta_seconds": eta_seconds,
                "rate_rays_per_second": rate,
            }
            report_progress("fallback_progress", **fallback_details)
            if runtime is not None:
                runtime.record("FERMAT_FALLBACK_PROGRESS", **fallback_details)
            fallback_last_report = now

    fallback_elapsed = time.perf_counter() - fallback_started
    report_progress(
        "fallback_complete",
        processed=fallback_total,
        total=fallback_total,
        accepted=int(fallback_accepted),
        numeric_failed=int(np.count_nonzero(fallback_numeric_failure)),
        elapsed_seconds=fallback_elapsed,
    )
    if runtime is not None:
        runtime.record(
            "FERMAT_FALLBACK_COMPLETED",
            processed_rays=fallback_total,
            accepted_rays=int(fallback_accepted),
            numeric_failed_rays=int(np.count_nonzero(fallback_numeric_failure)),
            invalid_start_count=int(fallback_invalid_start_count),
            solver_error_count=int(fallback_solver_error_count),
            wall_seconds=fallback_elapsed,
        )

    g, p, op, refl_res = eval_g(xy)
    gn = np.linalg.norm(g, axis=1)
    in_ap = aperture_contains_xy(m2.half_aperture, m2.aperture_polygon, xy[:, 0], xy[:, 1])
    gradient_pass = gn <= grad_tol
    reflection_pass = refl_res <= float(reflection_tolerance)
    finite_pass = np.isfinite(op) & np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(xy), axis=1)
    conic_domain = 1.0 - (1.0 + m2.conic) * m2.curvature ** 2 * np.sum(xy ** 2, axis=1)
    conic_domain_pass = (conic_domain >= -CONIC_DOMAIN_TOLERANCE) | (m2.conic <= -1.0)
    construction_half = (
        max(float(construction_domain_factor), 1.0)
        * np.asarray(m2.half_aperture, dtype=float)
    )
    in_construction_domain = (
        np.all(np.isfinite(xy), axis=1)
        & np.all(np.abs(xy) <= construction_half[None, :], axis=1)
    )
    success = (
        gradient_pass
        & reflection_pass
        & finite_pass
        & conic_domain_pass
        & ~fallback_numeric_failure
    )
    report_progress(
        "solver_complete",
        converged=int(np.count_nonzero(success)),
        ray_count=int(len(success)),
        fallback_count=int(np.count_nonzero(fallback_used)),
        elapsed_seconds=time.perf_counter() - solver_started,
    )
    termination_reason = np.where(
        success,
        "FERMAT_STATIONARY_GRADIENT_AND_REFLECTION_PASS",
        np.where(
            fallback_numeric_failure,
            "FERMAT_FALLBACK_NUMERIC_FAILURE",
            np.where(
                gradient_pass & ~reflection_pass,
                "WRONG_REFLECTION_BRANCH",
                "NO_STATIONARY_ROOT_IN_CONSTRUCTION_DOMAIN",
            ),
        ),
    )
    return {"target_points": p, "xy": xy, "gradient": g, "gradient_norm": gn, "success": success,
            "iterations": it + 1, "step_norm": last_step, "op_initial": op0, "op_final": op,
            "fallback_used": fallback_used,
            "fallback_numeric_failure": fallback_numeric_failure,
            "fallback_numeric_failure_count": int(np.count_nonzero(fallback_numeric_failure)),
            "fallback_invalid_start_count": int(fallback_invalid_start_count),
            "fallback_solver_error_count": int(fallback_solver_error_count),
            "gradient_pass": gradient_pass,
            "reflection_pass": reflection_pass, "reflection_residual": refl_res,
            "wrong_reflection_branch": gradient_pass & ~reflection_pass,
            "conic_domain_pass": conic_domain_pass, "in_aperture": in_ap,
            "finite_pass": finite_pass, "in_construction_domain": in_construction_domain,
            "construction_domain_factor": float(construction_domain_factor),
            "reflection_tolerance": float(reflection_tolerance),
            "termination_reason": termination_reason,
            "failure_reasons": termination_reason}


def reference_grid(display: PolySurface, trace: dict[str, Any], rays: dict[str, Any], fields: list[dict[str, Any]],
                   mx: float, my: float) -> tuple[np.ndarray, dict[str, Any]]:
    # Central field comes from the configured odd field grid, never a fixed index.
    """Dựng Fan dynamic reference từ central chief ray và magnification."""
    center_field = int(rays["central_field_index"])
    center_pupil = int(rays["central_pupil_index"])
    mask = ((rays["field_index"] == center_field)
            & (rays["pupil_index"] == center_pupil) & rays["chief"])
    idx = int(np.where(mask)[0][0])
    anchor_global = trace["landing"][idx]
    aloc = (anchor_global - display.center) @ display.frame
    refs = []
    for f in fields:
        du = float(f["u_vi_mm"])
        dv = float(f["v_vi_mm"])
        loc = np.array([aloc[0] + mx * du, aloc[1] + my * dv, 0.0])
        refs.append(display.center + loc @ display.frame.T)
    return np.array(refs), {"anchor_ray_index": idx, "central_field_index": center_field,
                            "anchor_global_mm": anchor_global,
                            "M_x": mx, "M_y": my, "M_x_source": "HARD_TARGET",
                            "M_y_source": "HARD_TARGET", "target_chasing": False}


def mapping_metrics(trace: dict[str, Any], rays: dict[str, Any], refs: np.ndarray,
                    weights: dict[str, float], distortion_limit: float = 7.0,
                    epsilon_D: float = 1e-12) -> dict[str, Any]:
    """Tính sai số ánh xạ theo field tại display."""
    land = trace["landing"]
    ref = refs[rays["field_index"]]
    e3 = land - ref
    err = np.linalg.norm(e3, axis=1)
    display_frame = trace.get("display_frame")
    if display_frame is None:
        # The display normal component is numerically zero; obtain a stable 2-D basis from landing cloud.
        _, _, vh = np.linalg.svd(land - np.mean(land, axis=0), full_matrices=False)
        display_frame = vh[:2].T
    local_error = e3 @ display_frame[:, :2]
    n_pupils = int(np.max(rays["pupil_index"])) + 1
    n_fields = int(np.max(rays["field_index"])) + 1
    per_pupil = []
    for p in range(n_pupils):
        chief = (rays["pupil_index"] == p) & rays["chief"]
        per_pupil.append(float(np.sqrt(np.mean(err[chief] ** 2))))
    mf1_fan = weights["omega1"] * float(np.sum(per_pupil))
    mf_dense = weights["omega_dense"] * float(np.sqrt(np.mean(err ** 2)))
    means_fp = np.zeros((n_fields, n_pupils, 3))
    means_f = np.zeros((n_fields, 3))
    spot_sq = []
    for f in range(n_fields):
        for p in range(n_pupils):
            m = (rays["field_index"] == f) & (rays["pupil_index"] == p)
            means_fp[f, p] = np.mean(land[m], axis=0)
            spot_sq.extend(np.sum((land[m] - means_fp[f, p]) ** 2, axis=1))
        means_f[f] = np.mean(means_fp[f], axis=0)
    pupil_raw = float(np.mean(np.sum((means_fp - means_f[:, None, :]) ** 2, axis=2)))
    mf_pupil = weights["omega_pupil"] * pupil_raw
    spot_sq = np.asarray(spot_sq)
    distortion = distortion_metrics(trace, rays, refs, weights.get("omega_dist", 0.0), distortion_limit, epsilon_D)
    terms = {"MF1_Fan": mf1_fan, "MF1_DENSE_EXTENSION": mf_dense, "MF2": 0.0,
             "MF_pupil": mf_pupil, "MF3_Fan": None, "MF_dist": distortion["MF_dist"],
             "MF_surface": 0.0, "MF_geometry": 0.0}
    total = sum(v for v in terms.values() if v is not None)
    return {"valid_count": int(np.sum(trace["valid"])), "ray_count": len(err),
            "RMS_mapping": float(np.sqrt(np.mean(err ** 2))), "Max_mapping": float(np.max(err)),
            "RMS_u": float(np.sqrt(np.mean(local_error[:, 0] ** 2))),
            "RMS_v": float(np.sqrt(np.mean(local_error[:, 1] ** 2))),
            "RMS_pupil": math.sqrt(max(pupil_raw, 0.0)), "RMS_spot": float(np.sqrt(np.mean(spot_sq))),
            "Max_spot_radius": float(np.sqrt(np.max(spot_sq))), "per_pupil_R": per_pupil,
            "distortion": distortion, "terms": terms, "MF_total": float(total), "exactly_one_weight": True}


def distortion_metrics(trace: dict[str, Any], rays: dict[str, Any], refs: np.ndarray,
                       omega_dist: float, hard_limit_percent: float = 7.0,
                       epsilon_D: float = 1e-12) -> dict[str, Any]:
    """Tính méo fixed-grid từ centroid của từng bundle hữu hạn."""
    actual = np.asarray(trace["display_local"])[:, :2]
    frame = np.asarray(trace["display_frame"])
    ref_local = (np.asarray(refs) - np.asarray(trace["display_center"])) @ frame
    center_field = int(rays["central_field_index"])
    ref_delta = (np.asarray(refs) - np.asarray(refs)[center_field]) @ frame[:, :2]
    nf = int(np.max(rays["field_index"])) + 1
    npup = int(np.max(rays["pupil_index"])) + 1
    centroids = np.full((nf, npup, 2), np.nan)
    chiefs = np.full((nf, npup, 2), np.nan)
    for f in range(nf):
        for p in range(npup):
            mask = (rays["field_index"] == f) & (rays["pupil_index"] == p)
            centroids[f, p] = np.mean(actual[mask], axis=0)
            chief_mask = mask & rays["chief"]
            if np.sum(chief_mask) != 1:
                raise RuntimeError("DISTORTION_CHIEF_SELECTION_MISMATCH")
            chiefs[f, p] = actual[chief_mask][0]
    rows = []
    field_ids = [next(r["field_id"] for r in rays["rows"] if r["field_index"] == f) for f in range(nf)]
    pupil_ids = [next(r["pupil_id"] for r in rays["rows"] if r["pupil_index"] == p) for p in range(npup)]
    centroid_values, chief_values = [], []
    for f in range(nf):
        for p in range(npup):
            if f == center_field:
                rows.append({"field_id": field_ids[f], "pupil_id": pupil_ids[p],
                             "field_index": f, "pupil_index": p, "is_central_field": True,
                             "D_grid_centroid_percent": None, "D_grid_chief_percent": None,
                             "D_parallel_percent": None, "D_perp_percent": None,
                             "central_anchor_error_mm": float(np.linalg.norm(
                                 centroids[center_field, p] - ref_local[center_field, :2])),
                             "actual_R_u_mm": 0.0, "actual_R_v_mm": 0.0,
                             "ideal_R_u_mm": 0.0, "ideal_R_v_mm": 0.0})
                continue
            rref = ref_delta[f]
            ract = centroids[f, p] - centroids[center_field, p]
            rchief = chiefs[f, p] - chiefs[center_field, p]
            den = max(float(np.linalg.norm(rref)), epsilon_D)
            dgrid = 100.0 * float(np.linalg.norm(ract-rref)) / den
            dchief = 100.0 * float(np.linalg.norm(rchief-rref)) / den
            den2 = max(float(np.dot(rref, rref)), epsilon_D**2)
            dparallel = 100.0 * (float(np.dot(ract, rref))/den2 - 1.0)
            dperp = 100.0 * float(ract[0]*rref[1]-ract[1]*rref[0]) / den2
            centroid_values.append(dgrid); chief_values.append(dchief)
            rows.append({"field_id": field_ids[f], "pupil_id": pupil_ids[p],
                         "field_index": f, "pupil_index": p, "is_central_field": False,
                         "D_grid_centroid_percent": dgrid, "D_grid_chief_percent": dchief,
                         "D_parallel_percent": dparallel, "D_perp_percent": dperp,
                          "central_anchor_error_mm": float(np.linalg.norm(
                              centroids[center_field, p]-ref_local[center_field, :2])),
                         "actual_R_u_mm": ract[0], "actual_R_v_mm": ract[1],
                         "ideal_R_u_mm": rref[0], "ideal_R_v_mm": rref[1]})
    vals = np.asarray(centroid_values)
    violations = np.maximum(0.0, (vals-hard_limit_percent)/hard_limit_percent)
    mf_dist = omega_dist * float(np.mean(violations**2)) if len(vals) else 0.0
    worst_flat = int(np.argmax(vals))
    noncentral_rows = [r for r in rows if not r["is_central_field"]]
    worst = noncentral_rows[worst_flat]
    dmax = float(np.max(vals))
    return {"convention": "V5_2_FROZEN_CENTROID_GRID_DISPLACEMENT",
            "hard_metric_uses": "finite-pupil centroid; chief ray reported separately",
            "hard_limit_percent": hard_limit_percent, "strict_less_than": True,
            "D_max_abs": dmax, "D_max_abs_chief_diagnostic": float(np.max(chief_values)),
            "worst_field_index": int(worst["field_index"]), "worst_pupil_index": int(worst["pupil_index"]),
            "violation_count": int(np.sum(vals >= hard_limit_percent)),
            "feasible": bool(dmax < hard_limit_percent), "MF_dist": mf_dist,
            "omega_dist": omega_dist, "epsilon_D_mm": epsilon_D, "rows": rows}


def mf2_geometry(visor_hits: np.ndarray, m1_hits: np.ndarray, m2_hits: np.ndarray,
                 m2: PolySurface, omega2: float, obs_orientation: np.ndarray | None = None) -> dict[str, Any]:
    # Closest line to O over the finite Visor->M1 segments.
    """Tính MF2 bằng diện tích tam giác có dấu để phạt obscuration."""
    seg = m1_hits - visor_hits
    l2 = np.sum(seg * seg, axis=1)
    tau = np.clip(np.sum((m2.center - visor_hits) * seg, axis=1) / np.maximum(l2, EPS), 0.0, 1.0)
    feet = visor_hits + tau[:, None] * seg
    dist = np.linalg.norm(feet - m2.center, axis=1)
    i = int(np.argmin(dist)); A, Q = visor_hits[i], feet[i]
    ray = unit(seg[i])
    tp = np.sum((m2_hits - A) * ray, axis=1)
    qp = A + tp[:, None] * ray
    pd = np.linalg.norm(m2_hits - qp, axis=1)
    j = int(np.argmin(pd)); P = m2_hits[j]
    # Project into the oriented plane with normal chosen from M2 local +y; retain sign.
    orient = unit(m2.frame[:, 1] if obs_orientation is None else np.asarray(obs_orientation, dtype=float))
    signed_area = 0.5 * float(np.dot(np.cross(Q - A, P - A), orient))
    mf2 = 0.0 if signed_area >= 0 else omega2 * abs(signed_area)
    return {"ray_I_index": i, "point_P_index": j, "A": A, "Q": Q, "P": P,
            "L_min_mm": float(dist[i]), "D_min_mm": float(pd[j]), "projection_orientation": orient,
            "S_AQP_signed_mm2": signed_area, "MF2": mf2,
            "logic": "MF2=0 for S_AQP>=0 else omega2*abs(S_AQP)"}


def fan_imaging_metrics(trace: dict[str, Any], rays: dict[str, Any], refs: np.ndarray,
                        omega1: float) -> dict[str, Any]:
    """Tính MF1 chief-centered và diagnostic pupil-shift / field-bias trên mọi ray."""
    landing = np.asarray(trace["landing"], float)
    valid = np.asarray(trace["valid"], bool)
    field_index = np.asarray(rays["field_index"], int)
    pupil_index = np.asarray(rays["pupil_index"], int)
    chief = np.asarray(rays["chief"], bool)
    refs = np.asarray(refs, float)

    if (landing.ndim != 2 or landing.shape[1] != 3
            or valid.shape != (len(landing),)
            or field_index.shape != (len(landing),)
            or pupil_index.shape != (len(landing),)
            or chief.shape != (len(landing),)):
        raise ValueError(
            "FAN_IMAGING_INPUT_SHAPE_MISMATCH"
        )

    pupils = (
        int(np.max(pupil_index)) + 1
        if len(pupil_index)
        else 0
    )

    fields = (
        int(np.max(field_index)) + 1
        if len(field_index)
        else 0
    )

    if refs.shape != (fields, 3):
        raise ValueError(
            "FAN_IMAGING_REFERENCE_SHAPE_MISMATCH"
        )

    chief_points = np.empty(
        (fields, pupils, 3),
        float,
    )

    chief_indices = np.empty(
        (fields, pupils),
        int,
    )

    bundle_rows: list[dict[str, Any]] = []

    error_vectors = np.empty_like(
        landing
    )

    for f in range(fields):
        for p in range(pupils):
            bundle = (
                (field_index == f)
                & (pupil_index == p)
            )

            selected = np.where(
                bundle & chief
            )[0]

            if len(selected) != 1:
                raise RuntimeError(
                    "MF1_CHIEF_SELECTION_MISMATCH"
                )

            chief_index = int(
                selected[0]
            )

            chief_indices[f, p] = (
                chief_index
            )

            chief_points[f, p] = (
                landing[chief_index]
            )

            error_vectors[bundle] = (
                landing[bundle]
                - landing[chief_index]
            )

    error = np.linalg.norm(
        error_vectors,
        axis=1,
    )

    per_pupil = []
    per_pupil_valid: list[
        float | None
    ] = []

    counts = []
    valid_counts = []

    for p in range(pupils):
        mask = (
            pupil_index == p
        )

        per_pupil.append(
            float(
                np.sqrt(
                    np.mean(
                        error[mask] ** 2
                    )
                )
            )
        )

        counts.append(
            int(np.sum(mask))
        )

        physical_mask = (
            mask & valid
        )

        valid_counts.append(
            int(
                np.sum(
                    physical_mask
                )
            )
        )

        per_pupil_valid.append(
            float(
                np.sqrt(
                    np.mean(
                        error[
                            physical_mask
                        ] ** 2
                    )
                )
            )
            if np.any(
                physical_mask
            )
            else None
        )

    for f in range(fields):
        for p in range(pupils):
            bundle = (
                (field_index == f)
                & (pupil_index == p)
            )

            bundle_error = (
                error[bundle]
            )

            bundle_valid = (
                bundle & valid
            )

            bundle_rows.append({
                "field_index":
                    f,

                "pupil_index":
                    p,

                "chief_ray_index":
                    int(
                        chief_indices[
                            f,
                            p,
                        ]
                    ),

                "ray_count":
                    int(
                        np.sum(
                            bundle
                        )
                    ),

                "physical_valid_count":
                    int(
                        np.sum(
                            bundle_valid
                        )
                    ),

                "RMS_to_chief_mm":
                    float(
                        np.sqrt(
                            np.mean(
                                bundle_error
                                ** 2
                            )
                        )
                    ),

                "max_radius_to_chief_mm":
                    float(
                        np.max(
                            bundle_error
                        )
                    ),
            })

    chief_field_mean = np.mean(
        chief_points,
        axis=1,
    )

    pupil_shift = (
        chief_points
        - chief_field_mean[
            :,
            None,
            :,
        ]
    )

    pupil_shift_radius = (
        np.linalg.norm(
            pupil_shift,
            axis=2,
        )
    )

    per_field_pupil_spread = (
        np.sqrt(
            np.mean(
                pupil_shift_radius
                ** 2,
                axis=1,
            )
        )
    )

    field_bias_vectors = (
        chief_field_mean
        - refs
    )

    field_bias_radius = (
        np.linalg.norm(
            field_bias_vectors,
            axis=1,
        )
    )

    e1 = float(
        np.sum(
            per_pupil
        )
    )

    all_physical = bool(
        np.all(
            valid
        )
    )

    valid_error = (
        error[valid]
    )

    valid_e1_diagnostic = (
        float(
            sum(
                x
                for x in
                per_pupil_valid
                if x is not None
            )
        )
        if len(
            valid_error
        )
        else None
    )

    return {
        "metric_name":
            "CHIEF_CENTERED_MF1_WITH_CHIEF_PUPIL_SHIFT_AND_FIELD_BIAS_DIAGNOSTICS",

        "E1_mm":
            e1,

        "MF1_Fan":
            float(
                omega1
                * e1
            ),

        "per_pupil_RMS_mm":
            per_pupil,

        "bundle_rows":
            bundle_rows,

        "chief_pupil_spread_RMS_mm":
            float(
                np.sqrt(
                    np.mean(
                        pupil_shift_radius
                        ** 2
                    )
                )
            ),

        "chief_pupil_spread_max_mm":
            float(
                np.max(
                    pupil_shift_radius
                )
            ),

        "chief_pupil_spread_per_field_mm":
            per_field_pupil_spread.tolist(),

        "chief_field_bias_RMS_mm":
            float(
                np.sqrt(
                    np.mean(
                        field_bias_radius
                        ** 2
                    )
                )
            ),

        "chief_field_bias_max_mm":
            float(
                np.max(
                    field_bias_radius
                )
            ),

        "chief_field_bias_per_field_mm":
            field_bias_radius.tolist(),

        "rays_per_pupil_term":
            counts,

        "active_ray_count":
            len(error),

        "failed_ray_count":
            int(
                np.sum(
                    ~valid
                )
            ),

        "physical_valid_ray_count":
            int(
                np.sum(
                    valid
                )
            ),

        "physical_valid_ray_fraction":
            float(
                np.mean(
                    valid
                )
            ),

        "per_pupil_physical_valid_count":
            valid_counts,

        "per_pupil_physical_valid_RMS_diagnostic_mm":
            per_pupil_valid,

        "physical_valid_E1_diagnostic_mm":
            valid_e1_diagnostic,

        "physical_valid_MF1_diagnostic":
            (
                None
                if valid_e1_diagnostic
                is None
                else float(
                    omega1
                    * valid_e1_diagnostic
                )
            ),

        "physical_valid_global_RMS_diagnostic_mm":
            (
                float(
                    np.sqrt(
                        np.mean(
                            valid_error
                            ** 2
                        )
                    )
                )
                if len(
                    valid_error
                )
                else None
            ),

        "certified_physical_MF1_Fan":
            (
                float(
                    omega1
                    * e1
                )
                if all_physical
                else None
            ),

        "physical_certification_status":
            (
                "CERTIFIED_ALL_REVERSE_FIRST_HITS_PHYSICAL"
                if all_physical
                else
                "NOT_CERTIFIED_INCOMPLETE_REVERSE_FIRST_HIT_COVERAGE"
            ),

        "invalid_ray_policy":
            (
                "ALL_RAYS_RETAINED_IN_DESIGN_MERIT; "
                "VALID_ONLY_DIAGNOSTIC_REPORTED_SEPARATELY; "
                "INVALID_RAYS_NEVER_DROPPED_TO_CREATE_A_PASS"
            ),

        "global_RMS_diagnostic_mm":
            float(
                np.sqrt(
                    np.mean(
                        error
                        ** 2
                    )
                )
            ),

        "max_chief_centered_error_diagnostic_mm":
            float(
                np.max(
                    error
                )
            ),

        "objective_interpretation":
            (
                "omega1 times the outer sum over sampled pupils "
                "of the RMS 3-D ray deviation from the chief ray "
                "landing of each field/pupil bundle"
            ),

        "pupil_shift_interpretation":
            (
                "RMS spread of the sampled pupil chief landings "
                "around their per-field chief mean"
            ),

        "field_bias_interpretation":
            (
                "RMS distance from the per-field mean chief landing "
                "to the dynamic field reference"
            ),

        "normalization":
            (
                "N_pupil=field_count*rays_per_field "
                "for each outer-pupil term"
            ),
    }


def ray_based_mtf(trace: dict[str, Any], rays: dict[str, Any], pattern: list[dict[str, Any]],
                  spatial_frequencies_lpmm: list[float], wavelengths_nm: list[float],
                  wavelength_weights: list[float]) -> dict[str, Any]:
    """Ước lượng MTF số từ ray intercept và OTF pupil tròn."""
    frequencies = np.asarray(spatial_frequencies_lpmm, dtype=float)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    spectral_weights = np.asarray(wavelength_weights, dtype=float)
    if len(wavelengths) == 0 or len(wavelengths) != len(spectral_weights):
        raise ValueError("MTF_WAVELENGTH_WEIGHT_MISMATCH")
    spectral_weights = spectral_weights / np.sum(spectral_weights)

    radii = np.asarray([math.hypot(float(p["dy_mm"]), float(p["dz_mm"])) for p in pattern])
    unique = np.unique(np.round(radii, 12))
    bounds = np.zeros(len(unique) + 1)
    if len(unique) > 1:
        bounds[1:-1] = 0.5 * (unique[:-1] + unique[1:])
    bounds[-1] = max(float(unique[-1]), EPS)
    sample_weights = np.zeros(len(pattern))
    for k, radius in enumerate(unique):
        members = np.isclose(radii, radius, atol=1e-10)
        annular_area = max(bounds[k + 1] ** 2 - bounds[k] ** 2, 0.0)
        sample_weights[members] = annular_area / max(int(np.sum(members)), 1)
    sample_weights /= np.sum(sample_weights)

    local = np.asarray(trace["display_local"])[:, :2]
    image_directions = unit(np.asarray(trace["directions"][-1]))
    rows: list[dict[str, Any]] = []
    n_fields = int(np.max(rays["field_index"])) + 1
    n_pupils = int(np.max(rays["pupil_index"])) + 1
    for field_index in range(n_fields):
        for pupil_index in range(n_pupils):
            mask = (rays["field_index"] == field_index) & (rays["pupil_index"] == pupil_index)
            indices = np.where(mask)[0]
            if len(indices) != len(pattern) or not np.all(np.asarray(trace["valid"])[indices]):
                for frequency in frequencies:
                    rows.append({"field_index": field_index, "pupil_index": pupil_index,
                                 "frequency_lpmm": float(frequency), "MTF_u": None, "MTF_v": None,
                                 "geometric_MTF_u": None, "geometric_MTF_v": None,
                                 "diffraction_MTF": None, "NA_estimate": None})
                continue
            order = np.argsort(rays["sample_index"][indices])
            indices = indices[order]
            weights = sample_weights[rays["sample_index"][indices]]
            weights = weights / np.sum(weights)
            points = local[indices]
            centroid = np.sum(weights[:, None] * points, axis=0)
            offset = points - centroid
            chief = image_directions[indices[np.where(rays["chief"][indices])[0][0]]]
            cosang = np.clip(np.abs(image_directions[indices] @ chief), 0.0, 1.0)
            na = float(np.max(np.sqrt(np.maximum(0.0, 1.0 - cosang * cosang))))
            for frequency in frequencies:
                phase_u = 2.0 * math.pi * frequency * offset[:, 0]
                phase_v = 2.0 * math.pi * frequency * offset[:, 1]
                geo_u = float(abs(np.sum(weights * np.exp(-1j * phase_u))))
                geo_v = float(abs(np.sum(weights * np.exp(-1j * phase_v))))
                diffraction = 0.0
                for wavelength_nm, sw in zip(wavelengths, spectral_weights):
                    cutoff = 2.0 * na / max(wavelength_nm * 1e-6, EPS)
                    nu = frequency / max(cutoff, EPS)
                    if nu < 1.0:
                        diffraction += float(sw) * (2.0 / math.pi) * (
                            math.acos(nu) - nu * math.sqrt(max(0.0, 1.0 - nu * nu)))
                rows.append({"field_index": field_index, "pupil_index": pupil_index,
                             "frequency_lpmm": float(frequency),
                             "MTF_u": geo_u * diffraction, "MTF_v": geo_v * diffraction,
                             "geometric_MTF_u": geo_u, "geometric_MTF_v": geo_v,
                             "diffraction_MTF": diffraction, "NA_estimate": na})
    finite_at_max = [min(float(row["MTF_u"]), float(row["MTF_v"])) for row in rows
                     if row["frequency_lpmm"] == float(np.max(frequencies))
                     and row["MTF_u"] is not None and row["MTF_v"] is not None]
    complete = len(finite_at_max) == n_fields * n_pupils
    return {"method": "POLAR_AREA_WEIGHTED_GEOMETRIC_RAY_OTF_TIMES_CIRCULAR_PUPIL_DIFFRACTION_OTF",
            "claim": "NUMERICAL_APPROXIMATION_NOT_COMMERCIAL_CODE_DIFFRACTION_MTF",
            "spatial_frequencies_lpmm": frequencies, "wavelengths_nm": wavelengths,
            "wavelength_weights": spectral_weights, "rows": rows, "complete": complete,
            "minimum_MTF_at_max_frequency": float(min(finite_at_max)) if complete else None}


def forward_shoot(display_sources: np.ndarray, target_eye: np.ndarray, initial_directions: np.ndarray,
                  display: PolySurface, m2: PolySurface, m1: PolySurface, visor: ChebVisor,
                  max_iter: int, tolerance_mm: float, fd_angle: float, damping: float) -> dict[str, Any]:
    """Bắn tia thuận Display→M2→M1→Visor→mắt bằng nghiệm hai tham số."""
    source = np.asarray(display_sources, float); target = np.asarray(target_eye, float)
    base = unit(np.asarray(initial_directions, float))
    e1 = np.tile(display.frame[:, 0], (len(source), 1)); e2 = np.tile(display.frame[:, 1], (len(source), 1))
    params = np.zeros((len(source), 2)); iteration_count = np.zeros(len(source), int)

    def trace(
        params_value: np.ndarray,
        finite: bool = False,
        *,
        indices: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Tính một lượt đường tia cho tham số shooting hiện tại."""
        if indices is None:
            source_value = source
            target_value = target
            base_value = base
            e1_value = e1
            e2_value = e2
        else:
            indices = np.asarray(indices, dtype=int)
            if indices.ndim != 1:
                raise ValueError("FORWARD_SUBSET_INDEX_SHAPE")
            source_value = source[indices]
            target_value = target[indices]
            base_value = base[indices]
            e1_value = e1[indices]
            e2_value = e2[indices]

        if params_value.shape != (len(source_value), 2):
            raise ValueError("FORWARD_SUBSET_PARAMETER_SHAPE")

        direction = unit(
            base_value
            + params_value[:, 0, None] * e1_value
            + params_value[:, 1, None] * e2_value
        )

        h2 = m2.intersect(
            source_value + 1e-4 * direction, direction, finite
        )
        d2 = reflect(direction, h2["normal"])
        h1 = m1.intersect(
            h2["point"] + 1e-4 * d2, d2, finite
        )
        d1 = reflect(d2, h1["normal"])
        hv = visor.intersect(
            h1["point"] + 1e-4 * d1, d1, finite
        )
        dv = reflect(d1, hv["normal"])

        t = -hv["point"][:, 0] / np.where(
            np.abs(dv[:, 0]) > 1e-12, dv[:, 0], np.nan
        )
        eye = hv["point"] + t[:, None] * dv
        valid = (
            h2["valid"] & h1["valid"] & hv["valid"]
            & (t > 0)
            & np.all(np.isfinite(eye), axis=1)
        )
        return {
            "direction": direction,
            "direction_after_m2": d2,
            "direction_after_m1": d1,
            "m2": h2["point"],
            "m1": h1["point"],
            "visor": hv["point"],
            "arrive_direction": dv,
            "eye_hit": eye,
            "valid": valid,
            "residual_yz": eye[:, 1:3] - target_value[:, 1:3],
        }

    if selected_backend("forward_active_fd") == "reference":
        for it in range(max_iter):
            current = trace(params); residual = current["residual_yz"]
            norm = np.linalg.norm(residual, axis=1); active = (norm > tolerance_mm) & current["valid"]
            if not np.any(active): break
            J = np.empty((len(source), 2, 2))
            for axis in range(2):
                pp = params.copy(); pm = params.copy(); pp[:, axis] += fd_angle; pm[:, axis] -= fd_angle
                J[:, :, axis] = (trace(pp)["residual_yz"] - trace(pm)["residual_yz"]) / (2 * fd_angle)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            good = active & (np.abs(det) > 1e-12); step = np.zeros_like(params)
            step[good, 0] = (-residual[good, 0] * J[good, 1, 1] + residual[good, 1] * J[good, 0, 1]) / det[good]
            step[good, 1] = (-J[good, 0, 0] * residual[good, 1] + J[good, 1, 0] * residual[good, 0]) / det[good]
            params[good] += damping * np.clip(step[good], -0.2, 0.2); iteration_count[good] = it + 1
    else:
        active_ids = np.arange(len(source), dtype=int)
        for it in range(max_iter):
            if len(active_ids) == 0:
                break
            current = trace(params[active_ids], indices=active_ids)
            residual = current["residual_yz"]
            norm = np.linalg.norm(residual, axis=1)
            active_local = (norm > tolerance_mm) & current["valid"]
            solve_ids = active_ids[active_local]
            if len(solve_ids) == 0:
                break

            p = params[solve_ids].copy()
            px_plus = p.copy()
            px_minus = p.copy()
            py_plus = p.copy()
            py_minus = p.copy()

            px_plus[:, 0] += fd_angle
            px_minus[:, 0] -= fd_angle
            py_plus[:, 1] += fd_angle
            py_minus[:, 1] -= fd_angle

            fd_params = np.concatenate((px_plus, px_minus, py_plus, py_minus), axis=0)
            fd_indices = np.concatenate((solve_ids, solve_ids, solve_ids, solve_ids), axis=0)

            fd_result = trace(fd_params, indices=fd_indices)
            fd_res = fd_result["residual_yz"]
            ns = len(solve_ids)
            rx1 = fd_res[0:ns]
            rx0 = fd_res[ns:2*ns]
            ry1 = fd_res[2*ns:3*ns]
            ry0 = fd_res[3*ns:4*ns]

            Jx = (rx1 - rx0) / (2.0 * fd_angle)
            Jy = (ry1 - ry0) / (2.0 * fd_angle)

            J = np.empty((ns, 2, 2), dtype=float)
            J[:, :, 0] = Jx
            J[:, :, 1] = Jy

            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            good_local = np.abs(det) > 1e-12
            good_ids = solve_ids[good_local]

            if len(good_ids) > 0:
                res_good = residual[active_local][good_local]
                J_good = J[good_local]
                det_good = det[good_local]

                step = np.empty((len(good_ids), 2), dtype=float)
                step[:, 0] = (-res_good[:, 0] * J_good[:, 1, 1] + res_good[:, 1] * J_good[:, 0, 1]) / det_good
                step[:, 1] = (-J_good[:, 0, 0] * res_good[:, 1] + J_good[:, 1, 0] * res_good[:, 0]) / det_good

                params[good_ids] += damping * np.clip(step, -0.2, 0.2)
                iteration_count[good_ids] = it + 1

            active_ids = solve_ids
    unconstrained = trace(params, False)
    rn = np.linalg.norm(unconstrained["residual_yz"], axis=1)
    final = trace(params, True)
    rf = np.linalg.norm(final["residual_yz"], axis=1)
    # Giu nguyen duong tia va residual tinh tu finite trace de dam bao tinh nhat quan toan he.
    final.update({
        "parameters": params,
        "iterations": iteration_count,
        "residual_mm": rf,
        "converged": final["valid"] & (rf <= tolerance_mm),
        "tolerance_mm": tolerance_mm,
        "unconstrained_diagnostic": {
            "parameters": params,
            "direction": unconstrained["direction"],
            "direction_after_m2": unconstrained["direction_after_m2"],
            "direction_after_m1": unconstrained["direction_after_m1"],
            "m2": unconstrained["m2"],
            "m1": unconstrained["m1"],
            "visor": unconstrained["visor"],
            "arrive_direction": unconstrained["arrive_direction"],
            "eye_hit": unconstrained["eye_hit"],
            "valid": unconstrained["valid"],
            "residual_yz": unconstrained["residual_yz"],
            "residual_mm": rn,
            "converged": unconstrained["valid"] & (rn <= tolerance_mm),
        }
    })
    return final


def reconstruct_virtual_points(forward: dict[str, Any], rays: dict[str, Any]) -> dict[str, Any]:
    """Dựng ảnh ảo chỉ từ bundle forward đầy đủ, hội tụ và nằm trước mắt."""
    apparent = -unit(forward["arrive_direction"])
    nf=int(np.max(rays["field_index"]))+1;npup=int(np.max(rays["pupil_index"]))+1
    points=np.full((nf,npup,3),np.nan)
    diagnostic_points=np.full((nf,npup,3),np.nan)
    diagnostic_bundle_valid=np.zeros((nf,npup),bool)
    bundle_valid=np.zeros((nf,npup),bool);rows=[]
    eye=forward["eye_hit"]
    for f in range(nf):
        for p in range(npup):
            group=(rays["field_index"]==f)&(rays["pupil_index"]==p)
            mask=group&forward["converged"]
            required_count=int(np.sum(group));converged_count=int(np.sum(mask))
            aa=apparent[mask];ee=eye[mask]
            A=np.sum(np.eye(3)[None,:,:]-aa[:,:,None]*aa[:,None,:],axis=0)
            b=np.sum(np.einsum('nij,nj->ni',np.eye(3)[None,:,:]-aa[:,:,None]*aa[:,None,:],ee),axis=0)
            rank=int(np.linalg.matrix_rank(A));cond=float(np.linalg.cond(A))
            x=np.linalg.lstsq(A,b,rcond=1e-12)[0] if np.sum(mask)>=2 and rank>=2 else np.full(3,np.nan)
            diagnostic_points[f,p]=x
            residual=float(np.sqrt(np.mean(np.sum(np.cross(x-ee,aa)**2,axis=1)))) if np.all(np.isfinite(x)) else float('nan')
            forward_parameters=np.einsum("ij,ij->i", x - ee, aa, optimize=False) if np.all(np.isfinite(x)) and len(aa) else np.array([])
            in_front=bool(len(forward_parameters) and np.all(forward_parameters>0.0))
            complete=bool(required_count>0 and converged_count==required_count)
            diagnostic_valid=bool(converged_count>=2 and rank>=2 and np.isfinite(cond)
                                  and np.all(np.isfinite(x)) and in_front)
            valid=bool(complete and diagnostic_valid)
            diagnostic_bundle_valid[f,p]=diagnostic_valid
            bundle_valid[f,p]=valid
            if valid: points[f,p]=x
            rows.append({"field_index":f,"pupil_index":p,"required_ray_count":required_count,
                         "converged_ray_count":converged_count,"complete_bundle":complete,
                         "rank":rank,"condition":cond,"RMS_line_residual_mm":residual,
                         "minimum_forward_parameter_mm":float(np.min(forward_parameters)) if len(forward_parameters) else None,
                         "virtual_point_in_front_of_every_eye_ray":in_front,
                         "diagnostic_bundle_valid":diagnostic_valid,"bundle_valid":valid,
                         "x_mm":x[0],"y_mm":x[1],"z_mm":x[2]})
    return {"points":points,"diagnostic_points":diagnostic_points,
            "diagnostic_bundle_valid":diagnostic_bundle_valid,"bundle_valid":bundle_valid,
            "diagnostic_valid_bundle_count":int(np.sum(diagnostic_bundle_valid)),
            "valid_bundle_count":int(np.sum(bundle_valid)),"required_bundle_count":int(nf*npup),
            "rows":rows,"apparent_directions":apparent}


def normalized_dlsq_optimize(m1: PolySurface, m2: PolySurface, evaluate: Any,
                             variable_indices: list[tuple[str, int]], iterations: int,
                             damping: float, fd_sag: float, plateau_relative: float,
                             engineering_worsening_relative: float = 0.0,
                             evaluate_engineering: Any | None = None,
                             engineering_absolute_tolerance: float = 1e-9,
                             restoration_fan_worsening_relative: float = 1e-3,
                             restoration_engineering_improvement_relative: float = 1e-4,
                             restoration_component_worsening_relative: float = 2e-2,
                             restoration_trial_limit: int = 8,
                             restoration_poll_variable_count: int = 4,
                             restoration_poll_step_mm: float = 1e-2,
                             damping_multipliers: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
                             line_search_alphas: tuple[float, ...] =
                             (1.0, .5, .25, .1, .05, .02, .01, .005, .002, .001),
                             *,
                             evaluate_columns=None,
                             evaluate_trials=None,
                             ) -> tuple[PolySurface, PolySurface, list[dict[str, Any]],
                                        list[dict[str, Any]], dict[str, Any]]:
    """Tối ưu Fan bằng filter DLSQ và nhánh phục hồi engineering có giới hạn."""
    m1, m2 = m1.copy(), m2.copy()
    history: list[dict[str, Any]] = []
    trial_history: list[dict[str, Any]] = []
    termination: dict[str, Any] = {"status": "MAXIMUM_ITERATIONS_REACHED"}
    component_keys = (
        "forward_distortion_monitor_percent",
        "forward_invalid_ray_fraction",
        "forward_invalid_bundle_fraction",
        "packaging_lambda_constraint_value",
    )

    if (engineering_absolute_tolerance < 0.0
            or restoration_fan_worsening_relative < 0.0
            or restoration_engineering_improvement_relative < 0.0
            or restoration_component_worsening_relative < 0.0
            or restoration_trial_limit < 1
            or restoration_poll_variable_count < 1
            or restoration_poll_step_mm <= 0.0):
        raise ValueError("DLSQ_FILTER_POLICY_INVALID")
    if (not damping_multipliers or any(value <= 0.0 for value in damping_multipliers)
            or not line_search_alphas
            or any(value <= 0.0 or value > 1.0 for value in line_search_alphas)):
        raise ValueError("DLSQ_TRUST_REGION_SCHEDULE_INVALID")

    def apply_step(step: np.ndarray) -> None:
        """Cộng vector coefficient đã chuẩn hóa vào hai mặt."""
        for value, (which, index) in zip(step, variable_indices):
            (m1 if which == "M1" else m2).coeff[index] += float(value)

    def engineering_checks(base: dict[str, Any], candidate: dict[str, Any],
                           relative_limit: float) -> tuple[bool, list[str]]:
        """Kiểm tra tổng engineering, từng thành phần và coverage mask."""
        reasons: list[str] = []
        candidate_total = float(candidate["J_engineering_dimensionless"])
        total_limit = (float(base["J_engineering_dimensionless"])
                       * (1.0 + relative_limit) + engineering_absolute_tolerance)
        if not np.isfinite(candidate_total):
            reasons.append("ENGINEERING_TOTAL_NONFINITE")
        elif candidate_total > total_limit:
            reasons.append("ENGINEERING_TOTAL_WORSENED")
        for key in component_keys:
            if key not in base or key not in candidate:
                continue
            component_limit = (float(base[key]) * (1.0 + relative_limit)
                               + engineering_absolute_tolerance)
            candidate_value = float(candidate[key])
            if not np.isfinite(candidate_value):
                reasons.append(f"{key.upper()}_NONFINITE")
            elif candidate_value > component_limit:
                reasons.append(f"{key.upper()}_WORSENED")
        mask_key = "_forward_distortion_evaluated_pair_mask"
        if mask_key in base and mask_key in candidate:
            base_mask = np.asarray(base[mask_key], bool)
            candidate_mask = np.asarray(candidate[mask_key], bool)
            if (base_mask.shape != candidate_mask.shape
                    or not bool(np.all(candidate_mask[base_mask]))):
                reasons.append("FORWARD_DISTORTION_PAIR_COVERAGE_REGRESSED")
        return not reasons, reasons

    def attach_engineering_metrics(record: dict[str, Any], base: dict[str, Any],
                                   candidate: dict[str, Any]) -> None:
        """Ghi giá trị engineering trước/sau để truy nguyên từng trial."""
        record["forward_engineering_evaluated"] = True
        record["J_engineering_before"] = float(base["J_engineering_dimensionless"])
        record["J_engineering_candidate"] = float(candidate["J_engineering_dimensionless"])
        for key in component_keys:
            if key in base:
                record[f"{key}_before"] = float(base[key])
            if key in candidate:
                record[f"{key}_candidate"] = float(candidate[key])
        mask_key = "_forward_distortion_evaluated_pair_mask"
        if mask_key in base:
            record["forward_pair_count_before"] = int(np.sum(np.asarray(base[mask_key], bool)))
        if mask_key in candidate:
            record["forward_pair_count_candidate"] = int(
                np.sum(np.asarray(candidate[mask_key], bool)))

    for it in range(1, iterations + 1):
        r0, rec0 = evaluate(m1, m2)
        r0 = np.asarray(r0, dtype=float)
        if not np.all(np.isfinite(r0)):
            raise BackendExecutionError("BASE_RESIDUAL_NONFINITE")
        if evaluate_engineering is not None:
            rec0 = {**rec0, **evaluate_engineering(m1, m2)}

        if evaluate_columns is None:
            perturbed_residuals = []
            for which, index in variable_indices:
                a = m1.copy()
                b = m2.copy()
                surface = a if which == "M1" else b
                surface.coeff[index] += fd_sag
                rp, _ = evaluate(a, b)
                perturbed_residuals.append(np.asarray(rp, float))
        else:
            perturbed_residuals = evaluate_columns(
                m1.copy(),
                m2.copy(),
                tuple(variable_indices),
                float(fd_sag),
            )

        if len(perturbed_residuals) != len(variable_indices):
            raise BackendExecutionError("JACOBIAN_COLUMN_COUNT_MISMATCH")

        J = np.empty((len(r0), len(variable_indices)), dtype=float)
        for column, rp in enumerate(perturbed_residuals):
            rp = np.asarray(rp, float)
            if rp.shape != r0.shape:
                raise BackendExecutionError("JACOBIAN_RESIDUAL_SHAPE_MISMATCH")
            if not np.all(np.isfinite(rp)):
                raise BackendExecutionError("JACOBIAN_RESIDUAL_NONFINITE")
            J[:, column] = (rp - r0) / fd_sag
        jtj = J.T@J
        diag = np.maximum(np.diag(jtj), 1e-12)
        rhs = -(J.T@r0)
        accepted = False
        accepted_mode = "NONE"
        best = rec0
        alpha_used = 0.0
        applied_step = np.zeros(len(variable_indices))
        iteration_trials: list[dict[str, Any]] = []
        restoration_candidates: list[dict[str, Any]] = []
        base_fan = float(rec0["J_Fan_dimensionless"])
        base_engineering = float(rec0.get("J_engineering_dimensionless", 0.0))
        if not np.isfinite(base_fan) or not np.isfinite(base_engineering):
            raise RuntimeError("DLSQ_BASE_OBJECTIVE_OR_ENGINEERING_NONFINITE")

        candidate_specs: list[dict[str, Any]] = []
        for multiplier in damping_multipliers:
            damp = float(damping * multiplier)
            delta = np.clip(solve(jtj+damp*np.diag(diag), rhs, assume_a="sym"), -.25, .25)
            for alpha in line_search_alphas:
                step = float(alpha) * delta
                candidate_specs.append({
                    "multiplier": multiplier,
                    "damping": damp,
                    "alpha": alpha,
                    "step": step.copy(),
                })

        if evaluate_trials is not None:
            trial_outputs = evaluate_trials(
                m1.copy(),
                m2.copy(),
                tuple(variable_indices),
                [spec["step"] for spec in candidate_specs],
            )
        else:
            trial_outputs = None

        for idx, spec in enumerate(candidate_specs):
            step = spec["step"]
            multiplier = spec["multiplier"]
            damp = spec["damping"]
            alpha = spec["alpha"]

            if trial_outputs is not None:
                candidate = trial_outputs[idx]["record"]
            else:
                apply_step(step)
                _, candidate = evaluate(m1, m2)
                apply_step(-step)

            candidate_fan = float(candidate["J_Fan_dimensionless"])
            finite_fan = bool(np.isfinite(candidate_fan))
            hard_ok = bool(candidate.get("hard_physical_surface_valid", False))
            fan_better = bool(finite_fan and candidate_fan < base_fan)
            predicted_residual = r0 + J@step
            predicted_fan = float(np.dot(predicted_residual, predicted_residual))
            predicted_reduction = float(np.dot(r0, r0)-predicted_fan)
            actual_fan_reduction = base_fan-candidate_fan
            record: dict[str, Any] = {
                "candidate_family": "FAN_LM_LINE_SEARCH",
                "iteration": it, "damping": damp, "damping_multiplier": float(multiplier),
                "alpha": float(alpha), "step_norm": float(np.linalg.norm(step)),
                "J_Fan_before": base_fan, "J_Fan_candidate": candidate_fan,
                "predicted_residual_norm_squared": predicted_fan,
                "predicted_reduction": predicted_reduction,
                "actual_fan_reduction": actual_fan_reduction,
                "actual_to_predicted_reduction_ratio": (
                    actual_fan_reduction/predicted_reduction
                    if abs(predicted_reduction) > 1e-15 else None),
                "finite_fan": finite_fan, "fan_better": fan_better,
                "hard_physical_surface_valid": hard_ok,
                "forward_engineering_evaluated": False,
                "engineering_nonworsening": None,
                "restoration_admissible": False,
                "accepted": False, "acceptance_mode": "REJECTED",
                "rejection_reasons": "",
            }
            reasons: list[str] = []
            if not finite_fan:
                reasons.append("NONFINITE_FAN_OBJECTIVE")
            if not hard_ok:
                reasons.append("HARD_PHYSICAL_OR_SURFACE_GATE_FAILED")
                if candidate.get("M1_surface_reasons"):
                    reasons.extend([f"M1_{r}" for r in candidate["M1_surface_reasons"]])
                if candidate.get("M2_surface_reasons"):
                    reasons.extend([f"M2_{r}" for r in candidate["M2_surface_reasons"]])
                if candidate.get("invalid_fraction", 0.0) > 0:
                    reasons.append(f"REVERSE_INVALID_RAYS_{candidate.get('invalid_fraction', 0.0):.1%}")
                if candidate.get("S_AQP_signed_mm2", 0.0) < 0:
                    reasons.append("OBSCURATION_VIOLATION_S_AQP_NEGATIVE")

            forward_evaluated = bool(finite_fan and hard_ok and fan_better
                                     and evaluate_engineering is not None)
            if finite_fan and hard_ok and fan_better and evaluate_engineering is None:
                apply_step(step)
                record.update({"accepted": True,
                               "acceptance_mode": "FAN_OBJECTIVE_STEP_NO_ENGINEERING_CALLBACK",
                               "rejection_reasons": ""})
                accepted = True
                accepted_mode = "FAN_OBJECTIVE_STEP_NO_ENGINEERING_CALLBACK"
                best = candidate
                alpha_used = float(alpha)
                applied_step = step.copy()
                iteration_trials.append(record)
                trial_history.append(record)
                break
            if forward_evaluated:
                apply_step(step)
                candidate = {**candidate, **evaluate_engineering(m1, m2)}
                attach_engineering_metrics(record, rec0, candidate)
                engineering_ok, engineering_reasons = engineering_checks(
                    rec0, candidate, engineering_worsening_relative)
                record["engineering_nonworsening"] = engineering_ok
                reasons.extend(engineering_reasons)
                if fan_better and engineering_ok:
                    record.update({"accepted": True,
                                   "acceptance_mode": "FAN_OBJECTIVE_FILTER_STEP",
                                   "rejection_reasons": ""})
                    accepted = True
                    accepted_mode = "FAN_OBJECTIVE_FILTER_STEP"
                    best = candidate
                    alpha_used = float(alpha)
                    applied_step = step.copy()
                    iteration_trials.append(record)
                    trial_history.append(record)
                    break
                else:
                    apply_step(-step)

            fan_restoration_limit = (base_fan*(1.0+restoration_fan_worsening_relative)
                                     + engineering_absolute_tolerance)
            if (finite_fan and hard_ok and evaluate_engineering is not None
                    and candidate_fan <= fan_restoration_limit):
                restoration_candidates.append({
                    "step": step.copy(), "candidate": candidate,
                    "record": record, "forward_evaluated": forward_evaluated,
                })
            if not fan_better:
                reasons.append("FAN_OBJECTIVE_NOT_IMPROVED")
            if not forward_evaluated and finite_fan and hard_ok:
                reasons.append("FORWARD_ENGINEERING_DEFERRED_TO_RESTORATION_SHORTLIST")
            record["rejection_reasons"] = "|".join(dict.fromkeys(reasons))
            iteration_trials.append(record)
            trial_history.append(record)

        if not accepted and evaluate_engineering is not None and variable_indices:
            # A pure Fan LM family can point outside the engineering-feasible cone.
            # Probe a small deterministic coordinate mesh as a genuine feasibility-
            # restoration search.  Low Fan-sensitivity coordinates are tried first,
            # and every poll still obeys the hard gate and bounded Fan worsening.
            column_sensitivity = np.linalg.norm(J, axis=0)
            poll_count = min(int(restoration_poll_variable_count), len(variable_indices))
            poll_indices = np.argsort(column_sensitivity, kind="stable")[:poll_count]
            fan_restoration_limit = (base_fan*(1.0+restoration_fan_worsening_relative)
                                     + engineering_absolute_tolerance)
            poll_specs: list[dict[str, Any]] = []
            for variable_index in poll_indices:
                for sign in (-1.0, 1.0):
                    step = np.zeros(len(variable_indices), float)
                    step[int(variable_index)] = sign*restoration_poll_step_mm
                    poll_specs.append({
                        "variable_index": variable_index,
                        "sign": sign,
                        "step": step,
                    })

            if evaluate_trials is not None:
                poll_outputs = evaluate_trials(
                    m1.copy(),
                    m2.copy(),
                    tuple(variable_indices),
                    [p["step"] for p in poll_specs],
                )
            else:
                poll_outputs = None

            for idx, p_spec in enumerate(poll_specs):
                variable_index = p_spec["variable_index"]
                sign = p_spec["sign"]
                step = p_spec["step"]

                if poll_outputs is not None:
                    candidate = poll_outputs[idx]["record"]
                else:
                    apply_step(step)
                    _, candidate = evaluate(m1, m2)
                    apply_step(-step)

                candidate_fan = float(candidate["J_Fan_dimensionless"])
                finite_fan = bool(np.isfinite(candidate_fan))
                hard_ok = bool(candidate.get("hard_physical_surface_valid", False))
                predicted_residual = r0 + J@step
                predicted_fan = float(np.dot(predicted_residual, predicted_residual))
                predicted_reduction = float(np.dot(r0, r0)-predicted_fan)
                actual_fan_reduction = base_fan-candidate_fan
                record = {
                    "candidate_family": "ENGINEERING_COORDINATE_POLL",
                    "iteration": it, "damping": None, "damping_multiplier": None,
                    "alpha": None, "step_norm": float(np.linalg.norm(step)),
                    "polled_surface": variable_indices[int(variable_index)][0],
                    "polled_coefficient_index": variable_indices[int(variable_index)][1],
                    "polled_sign": sign,
                    "J_Fan_before": base_fan, "J_Fan_candidate": candidate_fan,
                    "predicted_residual_norm_squared": predicted_fan,
                    "predicted_reduction": predicted_reduction,
                    "actual_fan_reduction": actual_fan_reduction,
                    "actual_to_predicted_reduction_ratio": (
                        actual_fan_reduction/predicted_reduction
                        if abs(predicted_reduction) > 1e-15 else None),
                    "finite_fan": finite_fan,
                    "fan_better": bool(finite_fan and candidate_fan < base_fan),
                    "hard_physical_surface_valid": hard_ok,
                    "forward_engineering_evaluated": False,
                    "engineering_nonworsening": None,
                    "restoration_admissible": False,
                    "accepted": False, "acceptance_mode": "REJECTED",
                    "rejection_reasons": "",
                }
                reasons: list[str] = []
                within_fan_bound = bool(finite_fan and candidate_fan <= fan_restoration_limit)
                if not finite_fan:
                    reasons.append("NONFINITE_FAN_OBJECTIVE")
                if not hard_ok:
                    reasons.append("HARD_PHYSICAL_OR_SURFACE_GATE_FAILED")
                    if candidate.get("M1_surface_reasons"):
                        reasons.extend([f"M1_{r}" for r in candidate["M1_surface_reasons"]])
                    if candidate.get("M2_surface_reasons"):
                        reasons.extend([f"M2_{r}" for r in candidate["M2_surface_reasons"]])
                    if candidate.get("invalid_fraction", 0.0) > 0:
                        reasons.append(f"REVERSE_INVALID_RAYS_{candidate.get('invalid_fraction', 0.0):.1%}")
                    if candidate.get("S_AQP_signed_mm2", 0.0) < 0:
                        reasons.append("OBSCURATION_VIOLATION_S_AQP_NEGATIVE")
                if finite_fan and not within_fan_bound:
                    reasons.append("RESTORATION_FAN_WORSENING_BOUND_EXCEEDED")
                if finite_fan and hard_ok and within_fan_bound:
                    apply_step(step)
                    candidate = {**candidate, **evaluate_engineering(m1, m2)}
                    apply_step(-step)
                    attach_engineering_metrics(record, rec0, candidate)
                    restoration_candidates.append({
                        "step": step.copy(), "candidate": candidate,
                        "record": record, "forward_evaluated": True,
                    })
                record["rejection_reasons"] = "|".join(reasons)
                iteration_trials.append(record)
                trial_history.append(record)

        if not accepted and restoration_candidates:
            # Only the most promising bounded-Fan candidates receive the expensive
            # forward evaluation.  This turns engineering into an active restoration
            # filter without multiplying every coefficient Jacobian evaluation.
            restoration_candidates.sort(key=lambda item: (
                float(item["candidate"]["J_Fan_dimensionless"]),
                float(item["record"]["step_norm"])))
            already_evaluated = [item for item in restoration_candidates
                                 if item["forward_evaluated"]]
            deferred = [item for item in restoration_candidates
                        if not item["forward_evaluated"]]
            shortlisted = already_evaluated + deferred[:restoration_trial_limit]
            admissible_restoration: list[dict[str, Any]] = []
            for item in shortlisted:
                candidate = item["candidate"]
                record = item["record"]
                if not item["forward_evaluated"]:
                    apply_step(item["step"])
                    candidate = {**candidate, **evaluate_engineering(m1, m2)}
                    apply_step(-item["step"])
                    item["candidate"] = candidate
                    item["forward_evaluated"] = True
                    attach_engineering_metrics(record, rec0, candidate)
                    old_reasons = [reason for reason in
                                   str(record["rejection_reasons"]).split("|")
                                   if reason and reason !=
                                   "FORWARD_ENGINEERING_DEFERRED_TO_RESTORATION_SHORTLIST"]
                    record["rejection_reasons"] = "|".join(old_reasons)
                candidate_engineering = float(candidate["J_engineering_dimensionless"])
                required_drop = max(
                    engineering_absolute_tolerance,
                    restoration_engineering_improvement_relative
                    * max(abs(base_engineering), 1.0))
                engineering_improved = candidate_engineering <= base_engineering-required_drop
                components_ok, component_reasons = engineering_checks(
                    rec0, candidate, restoration_component_worsening_relative)
                # Restoration may trade a small amount of distortion/packaging, but
                # it must never reduce forward ray or bundle validity.
                for validity_key in ("forward_invalid_ray_fraction",
                                     "forward_invalid_bundle_fraction"):
                    if (validity_key in rec0 and validity_key in candidate
                            and float(candidate[validity_key])
                            > float(rec0[validity_key])+engineering_absolute_tolerance):
                        components_ok = False
                        component_reasons.append(
                            f"RESTORATION_{validity_key.upper()}_WORSENED")
                restoration_ok = bool(engineering_improved and components_ok)
                record["restoration_admissible"] = restoration_ok
                record["engineering_restoration_improved"] = engineering_improved
                record["engineering_restoration_required_drop"] = required_drop
                if restoration_ok:
                    admissible_restoration.append(item)
                else:
                    old_reasons = [reason for reason in
                                   str(record["rejection_reasons"]).split("|") if reason]
                    old_reasons.extend(component_reasons)
                    if not engineering_improved:
                        old_reasons.append("ENGINEERING_RESTORATION_IMPROVEMENT_INSUFFICIENT")
                    record["rejection_reasons"] = "|".join(dict.fromkeys(old_reasons))
            for item in deferred[restoration_trial_limit:]:
                old_reasons = [reason for reason in
                               str(item["record"]["rejection_reasons"]).split("|") if reason]
                old_reasons.append("RESTORATION_SHORTLIST_LIMIT")
                item["record"]["rejection_reasons"] = "|".join(dict.fromkeys(old_reasons))
            if admissible_restoration:
                selected = min(admissible_restoration, key=lambda item: (
                    float(item["candidate"]["J_engineering_dimensionless"]),
                    float(item["candidate"]["J_Fan_dimensionless"])))
                apply_step(selected["step"])
                best = selected["candidate"]
                accepted = True
                accepted_mode = (
                    "ENGINEERING_COORDINATE_POLL_RESTORATION_STEP"
                    if selected["record"].get("candidate_family") ==
                    "ENGINEERING_COORDINATE_POLL"
                    else "ENGINEERING_FILTER_RESTORATION_STEP")
                alpha_value = selected["record"].get("alpha")
                alpha_used = 1.0 if alpha_value is None else float(alpha_value)
                applied_step = selected["step"].copy()
                selected["record"].update({
                    "accepted": True,
                    "acceptance_mode": accepted_mode,
                    "rejection_reasons": "",
                })

        relative = ((base_fan-float(best["J_Fan_dimensionless"]))
                    / max(abs(base_fan), 1e-12))
        rejected_reason_counts: dict[str, int] = {}
        for trial in iteration_trials:
            if bool(trial["accepted"]):
                continue
            for reason in str(trial.get("rejection_reasons", "")).split("|"):
                if reason:
                    rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0)+1
        iteration_status = ("ACCEPTED_FAN_OBJECTIVE_STEP"
                            if accepted_mode in ("FAN_OBJECTIVE_FILTER_STEP",
                                                 "FAN_OBJECTIVE_STEP_NO_ENGINEERING_CALLBACK")
                            else "ACCEPTED_ENGINEERING_RESTORATION_STEP"
                            if accepted_mode in ("ENGINEERING_FILTER_RESTORATION_STEP",
                                                 "ENGINEERING_COORDINATE_POLL_RESTORATION_STEP")
                            else "BLOCKED_NO_ADMISSIBLE_TRIAL")
        history.append({
            "iteration": it, "iteration_status": iteration_status,
            "J_Fan_before": base_fan, "J_Fan_after": best["J_Fan_dimensionless"],
            "J_engineering_before": base_engineering,
            "J_engineering_after": best.get("J_engineering_dimensionless", base_engineering),
            "relative_improvement": relative, "accepted": accepted,
            "acceptance_mode": accepted_mode, "alpha": alpha_used,
            "step_norm": float(np.linalg.norm(applied_step)),
            "trial_count": len(iteration_trials),
            "rejected_reason_counts": rejected_reason_counts,
            **{key: value for key, value in best.items()
               if key not in ("J_Fan_dimensionless", "J_engineering_dimensionless")
               and not key.startswith("_")},
        })
        if not accepted:
            termination = {
                "status": "BLOCKED_NO_ADMISSIBLE_DLSQ_TRIAL",
                "iteration": it,
                "trial_count": len(iteration_trials),
                "rejected_reason_counts": rejected_reason_counts,
                "mislabelled_as_converged": False,
            }
            break
        if (accepted_mode in ("FAN_OBJECTIVE_FILTER_STEP",
                              "FAN_OBJECTIVE_STEP_NO_ENGINEERING_CALLBACK")
                and 0.0 <= relative <= plateau_relative):
            termination = {
                "status": "CONVERGED_FAN_PLATEAU_AFTER_ADMISSIBLE_STEP",
                "iteration": it,
                "relative_improvement": relative,
            }
            break
    else:
        termination = {"status": "MAXIMUM_ITERATIONS_REACHED",
                       "iteration": max(iterations-1, 0)}
    return m1, m2, history, trial_history, termination


def packaging_lambda(vertices: np.ndarray, points: np.ndarray) -> float:
    """Tính hệ số co packaging nhỏ nhất chứa các điểm gương."""
    hull = ConvexHull(vertices)
    c = np.mean(vertices, axis=0)
    vals = []
    for eq in hull.equations:
        n, b = eq[:3], eq[3]
        denom = -(np.dot(n, c) + b)
        vals.append((points @ n + b + denom) / max(denom, EPS))
    return float(max(1.0, np.max(np.column_stack(vals))))


def aperture_points(surface: PolySurface, margin: float = 0.0, samples: int = 21) -> np.ndarray:
    """Lấy bao lồi và biên clear aperture từ hit points."""
    hx, hy = surface.half_aperture + margin
    x = np.linspace(-hx, hx, samples); y = np.linspace(-hy, hy, samples)
    X, Y = np.meshgrid(x, y); xv, yv = X.ravel(), Y.ravel()
    if surface.aperture_polygon is not None:
        poly = surface.aperture_polygon
        if margin:
            ctr=np.mean(poly,axis=0);poly=poly+margin*unit(poly-ctr)
        edges=np.roll(poly,-1,axis=0)-poly
        cross=edges[None,:,0]*(yv[:,None]-poly[None,:,1])-edges[None,:,1]*(xv[:,None]-poly[None,:,0])
        area=0.5*np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))
        keep=np.all(cross*np.sign(area)>=-1e-8,axis=1);xv,yv=xv[keep],yv[keep]
        xv=np.concatenate([xv,poly[:,0]]);yv=np.concatenate([yv,poly[:,1]])
    return surface.point(xv, yv)


def surface_sanity(surface: PolySurface, samples: int = 41) -> dict[str, Any]:
    """Kiem sag, phap tuyen va tinh huu han tren clear aperture."""
    x = np.linspace(-surface.half_aperture[0], surface.half_aperture[0], samples)
    y = np.linspace(-surface.half_aperture[1], surface.half_aperture[1], samples)
    X, Y = np.meshgrid(x, y)
    xv, yv = X.ravel(), Y.ravel()
    domain = "HALF_APERTURE_RECTANGLE"
    if surface.aperture_polygon is not None:
        poly = np.asarray(surface.aperture_polygon, dtype=float)
        edges = np.roll(poly, -1, axis=0) - poly
        cross = (edges[None, :, 0] * (yv[:, None] - poly[None, :, 1])
                 - edges[None, :, 1] * (xv[:, None] - poly[None, :, 0]))
        area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                            - poly[:, 1] * np.roll(poly[:, 0], -1))
        keep = np.all(cross * np.sign(area) >= -1e-9, axis=1)
        xv = np.concatenate([xv[keep], poly[:, 0]])
        yv = np.concatenate([yv[keep], poly[:, 1]])
        domain = "CONVEX_CLEAR_APERTURE_POLYGON_PLUS_BOUNDARY_VERTICES"
    z, gx, gy = surface.sag_slopes(xv, yv)
    slope = np.sqrt(gx * gx + gy * gy)
    finite = bool(np.all(np.isfinite(z)) and np.all(np.isfinite(slope)))
    conic_domain = 1.0 - (1.0 + surface.conic) * surface.curvature * surface.curvature * (xv * xv + yv * yv)
    conic_domain_min = float(np.min(conic_domain)) if len(conic_domain) > 0 and np.all(np.isfinite(conic_domain)) else -np.inf
    conic_domain_valid = bool(np.all(np.isfinite(conic_domain)) and conic_domain_min > 1e-10)

    max_half = float(max(surface.half_aperture))
    spike_detected = bool(np.max(np.abs(z)) > 0.5 * max_half) if np.all(np.isfinite(z)) else True
    passed = bool(finite and conic_domain_valid and (not spike_detected))

    failure_reasons: list[str] = []
    if not finite:
        failure_reasons.append("NONFINITE_VALUES")
    if not conic_domain_valid:
        failure_reasons.append("CONIC_DOMAIN_INVALID")
    if spike_detected:
        failure_reasons.append("SAG_SPIKE_DETECTED")

    max_z_idx = int(np.argmax(np.abs(z))) if np.all(np.isfinite(z)) else 0
    min_conic_idx = int(np.argmin(conic_domain)) if np.all(np.isfinite(conic_domain)) else 0

    return {
        "surface": surface.name,
        "sag_min_mm": float(np.min(z)) if np.all(np.isfinite(z)) else float("nan"),
        "sag_max_mm": float(np.max(z)) if np.all(np.isfinite(z)) else float("nan"),
        "sag_pv_mm": float(np.ptp(z)) if np.all(np.isfinite(z)) else float("nan"),
        "max_slope": float(np.max(slope)) if np.all(np.isfinite(slope)) else float("nan"),
        "sample_domain": domain,
        "sample_count": int(len(xv)),
        "minimum_conic_domain_argument": conic_domain_min,
        "conic_domain_valid": conic_domain_valid,
        "finite": finite,
        "spike_detected": spike_detected,
        "pass": passed,
        "failure_reasons": failure_reasons,
        "checks": {
            "finite": finite,
            "conic_domain": conic_domain_valid,
            "sag_spike": not spike_detected,
            "aperture_valid": bool(surface.half_aperture is not None and np.all(np.asarray(surface.half_aperture) > 0))
        },
        "violation_details": {
            "max_sag_violation": {
                "value": float(np.abs(z[max_z_idx])) if np.all(np.isfinite(z)) else float("nan"),
                "threshold": 0.5 * max_half,
                "coord_local_mm": [float(xv[max_z_idx]), float(yv[max_z_idx]), float(z[max_z_idx])] if np.all(np.isfinite(z)) else [0.0, 0.0, 0.0]
            },
            "min_conic_domain": {
                "value": conic_domain_min,
                "threshold": 1e-10,
                "coord_local_mm": [float(xv[min_conic_idx]), float(yv[min_conic_idx])] if np.all(np.isfinite(conic_domain)) else [0.0, 0.0]
            }
        },
        "conic_is_hyperbolic_or_parabolic": bool(surface.conic <= -1.0)
    }


def mirror_shape_gate(
    surface: PolySurface,
    policy: dict[str, float],
    orientation_sign: float | None = None,
    samples: int = 41,
) -> dict[str, Any]:
    """Hard-gate single-bowl mirror shape independently from optical merit."""
    if samples < 5:
        raise ValueError("MIRROR_SHAPE_GATE_SAMPLES_MUST_BE_AT_LEAST_FIVE")

    required_policy = {
        "maximum_freeform_departure_mm",
        "maximum_normal_departure_deg",
        "maximum_principal_curvature_per_mm",
    }
    if set(policy) != required_policy:
        raise ValueError(
            f"MIRROR_SHAPE_GATE_POLICY_KEYS_INVALID:{sorted(set(policy) ^ required_policy)}"
        )
    if any(
        not np.isfinite(float(policy[key])) or float(policy[key]) <= 0.0
        for key in required_policy
    ):
        raise ValueError("MIRROR_SHAPE_GATE_POLICY_LIMIT_INVALID")

    x = np.linspace(-surface.half_aperture[0], surface.half_aperture[0], samples)
    y = np.linspace(-surface.half_aperture[1], surface.half_aperture[1], samples)
    X, Y = np.meshgrid(x, y)
    xv, yv = X.ravel(), Y.ravel()
    sample_domain = "HALF_APERTURE_RECTANGLE"
    if surface.aperture_polygon is not None:
        poly = np.asarray(surface.aperture_polygon, dtype=float)
        edges = np.roll(poly, -1, axis=0) - poly
        cross = (
            edges[None, :, 0] * (yv[:, None] - poly[None, :, 1])
            - edges[None, :, 1] * (xv[:, None] - poly[None, :, 0])
        )
        area = 0.5 * np.sum(
            poly[:, 0] * np.roll(poly[:, 1], -1)
            - poly[:, 1] * np.roll(poly[:, 0], -1)
        )
        keep = np.all(cross * np.sign(area) >= -1e-9, axis=1)
        xv = np.concatenate([xv[keep], poly[:, 0]])
        yv = np.concatenate([yv[keep], poly[:, 1]])
        sample_domain = "CONVEX_CLEAR_APERTURE_POLYGON_PLUS_BOUNDARY_VERTICES"

    basic = surface_sanity(surface, samples=samples)
    z, gx, gy, gxx, gxy, gyy = surface.sag_slopes_hessian(xv, yv)
    zc, gxc, gyc = surface._conic(xv, yv)

    finite = bool(
        np.all(np.isfinite(z))
        and np.all(np.isfinite(gx))
        and np.all(np.isfinite(gy))
        and np.all(np.isfinite(gxx))
        and np.all(np.isfinite(gxy))
        and np.all(np.isfinite(gyy))
    )

    departure = z - zc
    freeform_rms = (
        float(np.sqrt(np.mean(departure * departure)))
        if np.all(np.isfinite(departure))
        else float("inf")
    )
    freeform_max = (
        float(np.max(np.abs(departure)))
        if np.all(np.isfinite(departure))
        else float("inf")
    )

    normal = unit(np.column_stack([-gx, -gy, np.ones_like(gx)]))
    conic_normal = unit(np.column_stack([-gxc, -gyc, np.ones_like(gxc)]))
    normal_dot = np.sum(normal * conic_normal, axis=1)
    normal_departure_deg = np.degrees(
        np.arccos(np.clip(normal_dot, -1.0, 1.0))
    )
    normal_departure_rms = (
        float(np.sqrt(np.mean(normal_departure_deg * normal_departure_deg)))
        if np.all(np.isfinite(normal_departure_deg))
        else float("inf")
    )
    normal_departure_max = (
        float(np.max(normal_departure_deg))
        if np.all(np.isfinite(normal_departure_deg))
        else float("inf")
    )

    denominator = 1.0 + gx * gx + gy * gy
    mean_curvature = (
        (1.0 + gy * gy) * gxx
        - 2.0 * gx * gy * gxy
        + (1.0 + gx * gx) * gyy
    ) / (2.0 * denominator ** 1.5)
    gaussian_curvature = (gxx * gyy - gxy * gxy) / (denominator * denominator)
    discriminant = np.maximum(mean_curvature * mean_curvature - gaussian_curvature, 0.0)
    root = np.sqrt(discriminant)
    k1 = mean_curvature + root
    k2 = mean_curvature - root

    finite_curvature = bool(
        np.all(np.isfinite(mean_curvature))
        and np.all(np.isfinite(gaussian_curvature))
        and np.all(np.isfinite(k1))
        and np.all(np.isfinite(k2))
    )

    if orientation_sign is None:
        median_h = float(np.median(mean_curvature[np.isfinite(mean_curvature)]))
        orientation = float(np.sign(median_h))
        if orientation == 0.0:
            orientation = float(np.sign(surface.curvature))
        if orientation == 0.0:
            orientation = 1.0
    else:
        orientation = float(np.sign(float(orientation_sign)))
        if orientation == 0.0:
            raise ValueError("MIRROR_SHAPE_GATE_ORIENTATION_SIGN_ZERO")

    oriented_h = orientation * mean_curvature
    sign_flip = (gaussian_curvature <= 0.0) | (oriented_h <= 0.0)
    sign_flip_count = int(np.count_nonzero(sign_flip))

    xy = np.column_stack([xv, yv])
    gradient_k1: list[np.ndarray] = []
    gradient_k2: list[np.ndarray] = []
    if len(xy) >= 2 and finite_curvature:
        tree = cKDTree(xy)
        neighbor_count = min(5, len(xy))
        distances, indices = tree.query(xy, k=neighbor_count)
        if neighbor_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for column in range(1, neighbor_count):
            distance = np.asarray(distances[:, column], float)
            neighbor = np.asarray(indices[:, column], int)
            good = np.isfinite(distance) & (distance > 1e-12)
            if np.any(good):
                gradient_k1.append(np.abs(k1[good] - k1[neighbor[good]]) / distance[good])
                gradient_k2.append(np.abs(k2[good] - k2[neighbor[good]]) / distance[good])

    g1 = np.concatenate(gradient_k1) if gradient_k1 else np.asarray([], float)
    g2 = np.concatenate(gradient_k2) if gradient_k2 else np.asarray([], float)
    gradient_p95 = max(
        float(np.percentile(g1, 95)) if len(g1) else 0.0,
        float(np.percentile(g2, 95)) if len(g2) else 0.0,
    )
    gradient_max = max(
        float(np.max(g1)) if len(g1) else 0.0,
        float(np.max(g2)) if len(g2) else 0.0,
    )

    max_principal = float(policy["maximum_principal_curvature_per_mm"])
    characteristic_length = max(float(np.max(surface.half_aperture)), 1e-9)
    gradient_limit = max_principal / characteristic_length

    # Numerical acceptance policy for nearly flat Gaussian curvature.
    # Keep the calculated KG values unchanged for diagnostics.
    gaussian_curvature_tolerance_per_mm2 = 1e-7
    minimum_accepted_gaussian_curvature_per_mm2 = (
        -gaussian_curvature_tolerance_per_mm2
    )

    checks = {
        "basic_surface_sanity": bool(basic["pass"]),
        "freeform_departure": bool(
            np.isfinite(freeform_max)
            and freeform_max <= float(policy["maximum_freeform_departure_mm"])
        ),
        "normal_departure": bool(
            np.isfinite(normal_departure_max)
            and normal_departure_max <= float(policy["maximum_normal_departure_deg"])
        ),
        "single_bowl_gaussian_curvature": bool(
            finite_curvature
            and float(np.min(gaussian_curvature))
            >= minimum_accepted_gaussian_curvature_per_mm2
        ),
        "single_bowl_mean_curvature_orientation": bool(
            finite_curvature and float(np.min(oriented_h)) > 0.0
        ),
        "principal_curvature_magnitude": bool(
            finite_curvature
            and max(float(np.max(np.abs(k1))), float(np.max(np.abs(k2)))) <= max_principal
        ),
        "curvature_gradient_p95": bool(
            np.isfinite(gradient_p95) and gradient_p95 <= gradient_limit
        ),
        "curvature_gradient_emergency": bool(
            np.isfinite(gradient_max) and gradient_max <= gradient_limit
        ),
    }
    failure_reasons = [name for name, passed in checks.items() if not passed]

    return {
        "surface": surface.name,
        "sample_domain": sample_domain,
        "sample_count": int(len(xv)),
        "finite": bool(finite and finite_curvature),
        "conic_domain_min": float(basic["minimum_conic_domain_argument"]),
        "freeform_departure_rms_mm": freeform_rms,
        "freeform_departure_max_mm": freeform_max,
        "normal_departure_rms_deg": normal_departure_rms,
        "normal_departure_max_deg": normal_departure_max,
        "H_min_per_mm": float(np.min(mean_curvature)) if finite_curvature else float("nan"),
        "H_max_per_mm": float(np.max(mean_curvature)) if finite_curvature else float("nan"),
        "KG_min_per_mm2": float(np.min(gaussian_curvature)) if finite_curvature else float("nan"),
        "KG_max_per_mm2": float(np.max(gaussian_curvature)) if finite_curvature else float("nan"),
        "k1_min_per_mm": float(np.min(k1)) if finite_curvature else float("nan"),
        "k1_max_per_mm": float(np.max(k1)) if finite_curvature else float("nan"),
        "k2_min_per_mm": float(np.min(k2)) if finite_curvature else float("nan"),
        "k2_max_per_mm": float(np.max(k2)) if finite_curvature else float("nan"),
        "orientation_sign": orientation,
        "oriented_H_min_per_mm": float(np.min(oriented_h)) if finite_curvature else float("nan"),
        "curvature_sign_flip_count": sign_flip_count,
        "curvature_gradient_p95_per_mm2": gradient_p95,
        "curvature_gradient_max_per_mm2": gradient_max,
        "curvature_gradient_limit_per_mm2": gradient_limit,
        "policy": {key: float(value) for key, value in policy.items()},
        "checks": checks,
        "pass": bool(all(checks.values())),
        "failure_reasons": failure_reasons,
        "basic_surface_sanity": basic,
    }


def pure_xy20_fit(surface: PolySurface, fit_weights: dict[str, float]) -> tuple[PolySurface, dict[str, Any]]:
    """Fit bề mặt thuần XY order 20 để đối chiếu bản xuất."""
    x = np.linspace(-surface.half_aperture[0], surface.half_aperture[0], 41)
    y = np.linspace(-surface.half_aperture[1], surface.half_aperture[1], 41)
    X, Y = np.meshgrid(x, y); xv, yv = X.ravel(), Y.ravel()
    z, gx, gy = surface.sag_slopes(xv, yv)
    terms = monomial_terms(5, include_constant=False)
    out = PolySurface(surface.name, surface.center.copy(), surface.frame.copy(), surface.half_aperture.copy(),
                      terms, np.zeros(20), surface.scale.copy(), 0.0, 0.0, True,
                      None if surface.aperture_polygon is None else surface.aperture_polygon.copy())
    A0, Ax, Ay = out.basis(xv, yv), out.basis(xv, yv, "x"), out.basis(xv, yv, "y")
    ws = math.sqrt(fit_weights["sag"]) / fit_weights["sag_scale_mm"]
    normal_scale = math.radians(float(fit_weights.get(
        "normal_angle_scale_deg", math.degrees(float(fit_weights.get("slope_scale", 0.01))))))
    wn = math.sqrt(fit_weights["normal"]) / normal_scale
    n1 = unit(np.column_stack([-gx, -gy, np.ones_like(gx)]))
    A = np.vstack([ws * A0, wn * n1[:, 2, None] * Ax, wn * n1[:, 2, None] * Ay])
    b = np.concatenate([ws * z, -wn * n1[:, 0], -wn * n1[:, 1]])
    out.coeff = np.linalg.lstsq(A, b, rcond=1e-12)[0]
    z2, gx2, gy2 = out.sag_slopes(xv, yv)
    n2 = unit(np.column_stack([-gx2, -gy2, np.ones_like(gx2)]))
    ang = np.degrees(np.arccos(np.clip(np.sum(n1 * n2, axis=1), -1, 1)))
    return out, {"sag_rms_mm": float(np.sqrt(np.mean((z2-z)**2))), "sag_max_mm": float(np.max(np.abs(z2-z))),
                 "normal_rms_deg": float(np.sqrt(np.mean(ang**2))), "normal_max_deg": float(np.max(ang)),
                 "term_count": len(terms), "terms": terms}


def dlsq_optimize(m1: PolySurface, m2: PolySurface, rays: dict[str, Any], visor: ChebVisor,
                  display: PolySurface, fields: list[dict[str, Any]], mx: float, my: float,
                  weights: dict[str, float], iterations: int, damping: float, fd_sag: float,
                  variable_indices: list[tuple[str, int]], plateau_relative: float = 0.0
                  ) -> tuple[PolySurface, PolySurface, list[dict[str, Any]]]:
    """Tối ưu DLSQ tương thích cũ cho tập hệ số chọn trước."""
    m1, m2 = m1.copy(), m2.copy()
    history = []

    def evaluate() -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
        """Tính residual và diagnostics tại một vectơ hệ số."""
        tr = trace_reverse(rays, visor, m1, m2, display, physical_first_hit=False)
        refs, rr = reference_grid(display, tr, rays, fields, mx, my)
        met = mapping_metrics(tr, rays, refs, weights)
        local_mapping = (tr["landing"] - refs[rays["field_index"]]) @ display.frame[:, :2]
        rms_mapping = max(float(np.sqrt(np.mean(np.sum(local_mapping**2, axis=1)))), 1e-12)
        # Scale the LS surrogate so its squared norm equals the already-weighted
        # dense RMS term, instead of accidentally multiplying mapping influence
        # by the ray count and drowning the v5.2 distortion steering residual.
        mapping_scale = math.sqrt(weights["omega_dense"] / (len(local_mapping) * rms_mapping))
        residual = (mapping_scale * local_mapping).ravel()
        dvals = np.array([row["D_grid_centroid_percent"] for row in met["distortion"]["rows"]
                          if not row["is_central_field"]], dtype=float)
        violation = np.maximum(0.0, (dvals-met["distortion"]["hard_limit_percent"])/met["distortion"]["hard_limit_percent"])
        dres = math.sqrt(weights.get("omega_dist", 0.0)/max(len(violation),1))*violation
        residual = np.concatenate([residual, dres])
        return residual, met, refs, tr

    for it in range(iterations):
        r0, met0, refs, tr0 = evaluate()
        J = np.empty((len(r0), len(variable_indices)))
        for j, (which, idx) in enumerate(variable_indices):
            surf = m1 if which == "M1" else m2
            surf.coeff[idx] += fd_sag
            rp, _, _, _ = evaluate()
            surf.coeff[idx] -= fd_sag
            J[:, j] = (rp - r0) / fd_sag
        JTJ = J.T @ J
        rhs = -(J.T @ r0)
        diag = np.maximum(np.diag(JTJ), 1e-12)
        accepted = False; best_met = met0; alpha_used = 0.0; damping_used = damping; delta = np.zeros(len(variable_indices))
        for damp_try in (damping, damping*10.0, damping*100.0):
            delta_try = solve(JTJ + damp_try * np.diag(diag), rhs, assume_a="sym")
            delta_try = np.clip(delta_try, -0.25, 0.25)
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
                for d, (which, idx) in zip(delta_try, variable_indices):
                    (m1 if which == "M1" else m2).coeff[idx] += alpha * d
                _, met, _, tr = evaluate()
                base_d, cand_d = best_met["distortion"], met["distortion"]
                if cand_d["feasible"] and not base_d["feasible"]:
                    better = True
                elif cand_d["feasible"] and base_d["feasible"]:
                    better = met["MF_total"] < best_met["MF_total"]
                elif not cand_d["feasible"] and not base_d["feasible"]:
                    better = (cand_d["D_max_abs"] < base_d["D_max_abs"]-1e-10 or
                              (abs(cand_d["D_max_abs"]-base_d["D_max_abs"]) <= 1e-10 and met["MF_total"] < best_met["MF_total"]))
                else:
                    better = False
                if better and np.all(tr["valid"]):
                    accepted = True; best_met = met; alpha_used = alpha; damping_used = damp_try; delta = delta_try
                    break
                for d, (which, idx) in zip(delta_try, variable_indices):
                    (m1 if which == "M1" else m2).coeff[idx] -= alpha * d
            if accepted:
                break
            delta = delta_try
        history.append({"iteration": it, "MF_before": met0["MF_total"], "MF_after": best_met["MF_total"],
                        "RMS_before_mm": met0["RMS_mapping"], "RMS_after_mm": best_met["RMS_mapping"],
                        "D_max_abs_before_percent":met0["distortion"]["D_max_abs"],
                        "D_max_abs_after_percent":best_met["distortion"]["D_max_abs"],
                        "distortion_feasible_before":met0["distortion"]["feasible"],
                        "distortion_feasible_after":best_met["distortion"]["feasible"],
                        "distortion_violation_count_after":best_met["distortion"]["violation_count"],
                        "step_norm": float(np.linalg.norm(delta)), "alpha": alpha_used, "damping": damping_used,
                        "accepted": accepted, "valid_count": best_met["valid_count"],
                        "termination": "accepted_dlsq" if accepted else "no_improving_physical_step"})
        if not accepted:
            break
        relative_improvement = (met0["MF_total"]-best_met["MF_total"])/max(abs(met0["MF_total"]), 1e-12)
        if (met0["distortion"]["feasible"] and best_met["distortion"]["feasible"] and
                0.0 <= relative_improvement <= plateau_relative):
            history[-1]["termination"] = "objective_plateau"
            history[-1]["relative_improvement"] = relative_improvement
            break
    return m1, m2, history


def triangle_intersect(origin: np.ndarray, direction: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
    """Tính giao tia–tam giác bằng tọa độ barycentric."""
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2); det = np.dot(e1, p)
    if abs(det) < 1e-12: return math.inf, np.full(3, np.nan)
    inv = 1.0 / det; tvec = origin - a; u = np.dot(tvec, p) * inv
    if u < 0 or u > 1: return math.inf, np.full(3, np.nan)
    q = np.cross(tvec, e1); v = np.dot(direction, q) * inv
    if v < 0 or u + v > 1: return math.inf, np.full(3, np.nan)
    t = np.dot(e2, q) * inv
    if t <= 1e-6: return math.inf, np.full(3, np.nan)
    return t, unit(np.cross(e1, e2))


def authority_visor_intersections(origins: np.ndarray, directions: np.ndarray, visor: ChebVisor,
                                  uv: np.ndarray, xyz: np.ndarray, grid_shape: tuple[int, int], radius_cells: int = 4) -> dict[str, Any]:
    """Đối chiếu hit visor với tam giác từ point cloud authority."""
    guess = visor.intersect(origins, directions, finite=True)
    gloc = (guess["point"] - visor.center) @ visor.frame
    # File u is opposite current HARD local U; file v closely follows local V.
    ug = -gloc[:, 0]; vg = gloc[:, 1]
    nv, nu = grid_shape; uvals = np.unique(uv[:, 0]); vvals = np.unique(uv[:, 1])
    grid = xyz.reshape(nv, nu, 3)
    tout = np.full(len(origins), np.inf); pout = np.full_like(origins, np.nan); nout = np.full_like(origins, np.nan)
    for r in range(len(origins)):
        iu = int(np.clip(np.searchsorted(uvals, ug[r]) - 1, 0, nu - 2))
        iv = int(np.clip(np.searchsorted(vvals, vg[r]) - 1, 0, nv - 2))
        best = math.inf; best_n = np.full(3, np.nan)
        for j in range(max(0, iv-radius_cells), min(nv-1, iv+radius_cells+1)):
            for i in range(max(0, iu-radius_cells), min(nu-1, iu+radius_cells+1)):
                p00, p10, p01, p11 = grid[j, i], grid[j, i+1], grid[j+1, i], grid[j+1, i+1]
                for tri in ((p00, p10, p11), (p00, p11, p01)):
                    t, n = triangle_intersect(origins[r], directions[r], *tri)
                    if t < best: best, best_n = t, n
        if np.isfinite(best):
            if np.dot(best_n, visor.frame[:, 2]) < 0: best_n *= -1
            tout[r] = best; pout[r] = origins[r] + best * unit(directions[r]); nout[r] = best_n
    valid = np.isfinite(tout)
    return {"t": tout, "point": pout, "normal": nout, "valid": valid, "surrogate_guess": guess["point"],
            "hit_delta_mm": np.linalg.norm(pout - guess["point"], axis=1)}
