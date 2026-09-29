"""Bo kiem thu toan dien cho cac ban va nen tang HUD Fan v5.5.

Kiem chung:
1. Giao tia PolySurface/ChebVisor, nearest root trong aperture, first_hit, unresolved handling.
2. Khau do va scale: tinh nhat quan polygon/bbox, rescale khong doi sag/phap tuyen.
3. Fermat M2: loai bo nhanh truyen tiep (transmission), chi chap nhan phan xa that su.
4. Fit tren candidate aperture va kiem soat conic domain toan khau do.
5. CI incumbent archive: luu giu nghiem fully physical qua beam pruning va restoration cycles.
6. Forward shoot: dong nhat quy dao finite, khong ghi de sai lech.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from core import (
    ChebVisor, PolySurface, aperture_boundary_xy, aperture_contains_xy,
    convex_aperture_from_points, first_hit, fit_surface, forward_shoot,
    normalize_aperture_polygon, rescale_surface_polynomial, solve_fermat_m2,
    surface_sanity, trace_reverse, unit, reflect,
    _polynomial_real_roots, _distance_scaled_polynomial_real_roots,
    point_normal_cloud_diagnostics,
)
from pipeline_v55 import (
    _apply_rho_state, _ci_candidate_improves, _fermat_audit, _resize,
    _rho_sort_key, _rho_state, _select_ci_frontier, Context,
    _rho_o2_search_sort_key,
    _step16_refinement_grid,
    _step16_spot_tail_metrics,
    _step16_full_hard_feasible,
    _promote_surface_order_without_shape_change,
    _rebuild_post_fit_apertures, _surface_shape_snapshot, _assert_surface_shape_unchanged,
    _step7_auto_seed_descriptors, _step7_packaging_pass, _planar_seed_rank_key,
    _step12_actual_o2_gate, _step12_apply_o2_vector,
    _step12_o2_variable_descriptors, _step14_fermat_diagnostics,
    _mirror_topology_gate, validate_config,
)
from typing import Any
from run_multistart_v55 import _select_step7_deep_run_schedule


def test_ray_intersection_and_first_hit():
    """9.1: Kiem tra giao tia va first_hit."""
    # A. Mat phang z = 0, tam tai goc toa do
    plane = PolySurface.plane("TEST_PLANE", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (10.0, 10.0))
    
    # Giao truoc
    h_front = plane.intersect(np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]), finite=True)
    assert h_front["valid"][0]
    assert abs(h_front["t"][0] - 5.0) < 1e-12
    assert np.allclose(h_front["point"][0], [0.0, 0.0, 0.0])
    
    # Giao sau (t < 0 khong tinh voi finite=True)
    h_back = plane.intersect(np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, 1.0]]), finite=True)
    assert not h_back["valid"][0]
    
    # Song song (khong giao)
    h_parallel = plane.intersect(np.array([[0.0, 0.0, 5.0]]), np.array([[1.0, 0.0, 0.0]]), finite=True)
    assert not h_parallel["valid"][0]
    
    # Diem sat bien (inside tolerance)
    h_edge = plane.intersect(np.array([[9.99, 0.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]), finite=True)
    assert h_edge["valid"][0]
    
    # Diem ngoai aperture
    h_outside = plane.intersect(np.array([[15.0, 0.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]), finite=True)
    assert not h_outside["valid"][0]
    assert h_outside["status"][0] == "OUTSIDE_CLEAR_APERTURE"

    # Huong chua chuan hoa (phai duoc tu chuan hoa hoac xu ly dung)
    h_unnorm = plane.intersect(np.array([[0.0, 0.0, 10.0]]), np.array([[0.0, 0.0, -2.0]]), finite=True)
    assert h_unnorm["valid"][0]
    assert abs(h_unnorm["t"][0] - 10.0) < 1e-12

    # Vector 0 hoac khong huu han (phai bi reject voi status phu hop)
    h_zero = plane.intersect(np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, 0.0]]), finite=True)
    assert not h_zero["valid"][0]
    assert h_zero["status"][0] in ("ZERO_DIRECTION", "INVALID_RAY_DIRECTION", "NONFINITE_INPUT")

    # B. Parabola z = 0.1 * x^2
    # origin = (-4, 0, 1), direction = unit((1, 0, -0.1))
    # Nghiem giao voi z = 0.1 * x^2:
    # 1 - 0.1*s = 0.1*(-4 + s)^2 => 1 - 0.1*s = 0.1*(16 - 8s + s^2) => 10 - s = 16 - 8s + s^2 => s^2 - 7s + 6 = 0
    # s1 = 1, s2 = 6.
    # Voi direction = unit((1, 0, -0.1)), norm = sqrt(1 + 0.01) = sqrt(1.01)
    # t1 = 1 * sqrt(1.01) = sqrt(1.01) (t_near)
    # t2 = 6 * sqrt(1.01) (t_far)
    parab = PolySurface("PARABOLA", np.zeros(3), np.eye(3), (10.0, 10.0),
                        [(2, 0)], np.array([0.1]), np.array([1.0, 1.0]), 0.0, 0.0, False)
    raw_dir = np.array([1.0, 0.0, -0.1])
    d_parab = unit(raw_dir)
    o_parab = np.array([-4.0, 0.0, 1.0])
    h_parab = parab.intersect(o_parab.reshape(1, 3), d_parab.reshape(1, 3), finite=True)
    
    t_expected_near = math.sqrt(1.01)
    assert h_parab["valid"][0]
    assert abs(h_parab["t"][0] - t_expected_near) < 1e-8
    assert abs(h_parab["point"][0, 0] - (-3.0)) < 1e-8

    # Them mat phang z = 0.6 tai s = 4 => t_plane = 4 * sqrt(1.01)
    plane_mid = PolySurface("MID_PLANE", np.array([0.0, 0.0, 0.6]), np.eye(3), (10.0, 10.0),
                            [], np.zeros(0), np.array([1.0, 1.0]), 0.0, 0.0, False)
    
    surfaces = {"PARABOLA": parab, "MID_PLANE": plane_mid}
    fh = first_hit(o_parab.reshape(1, 3), d_parab.reshape(1, 3), surfaces)
    # first_hit phai chon PARABOLA vi t_near = sqrt(1.01) < t_plane = 4*sqrt(1.01)
    assert fh["name"][0] == "PARABOLA"
    assert abs(fh["t"][0] - t_expected_near) < 1e-8


def test_aperture_and_scale_invariance():
    """9.2: Kiem tra khau do, da giac, va scale."""
    # A. Winding CW va CCW cho ket qua tuong duong
    pts_ccw = np.array([[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]])
    pts_cw = pts_ccw[::-1].copy()
    norm_ccw = normalize_aperture_polygon(pts_ccw)
    norm_cw = normalize_aperture_polygon(pts_cw)
    assert np.allclose(norm_ccw, norm_cw)

    # B. Polygon suy bien (thang hang) bi phat hien
    pts_collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        normalize_aperture_polygon(pts_collinear)

    # C. Dong bo giua aperture_polygon va half_aperture khi resize
    surf = PolySurface("M1_TEST", np.zeros(3), np.eye(3), (20.0, 20.0),
                       [(2, 0)], np.array([0.001]), np.array([20.0, 20.0]), 0.0, 0.0, False)
    footprint = np.array([
        [-15.0, -10.0, 0.0],
        [15.0, -10.0, 0.0],
        [15.0, 10.0, 0.0],
        [-15.0, 10.0, 0.0]
    ])
    _resize(surf, footprint)
    assert surf.aperture_polygon is not None
    assert surf.half_aperture[0] >= 15.0
    assert surf.half_aperture[1] >= 10.0

    # D. Rescale khong lam doi sag hoac phap tuyen
    terms = [(2, 0), (0, 2), (1, 1), (3, 0), (0, 3)]
    coeffs = np.array([0.05, 0.03, 0.01, -0.002, 0.001])
    scale_old = np.array([15.0, 20.0])
    scale_new = np.array([30.0, 10.0])
    surf_rescale = PolySurface("RESCALE_TEST", np.zeros(3), np.eye(3), (40.0, 40.0),
                               terms, coeffs.copy(), scale_old.copy(), 0.02, -0.5, False)
    test_x = np.linspace(-10.0, 10.0, 15)
    test_y = np.linspace(-8.0, 8.0, 15)
    sag_before, gx_before, gy_before = surf_rescale.sag_slopes(test_x, test_y)
    
    rescale_surface_polynomial(surf_rescale, scale_new)
    sag_after, gx_after, gy_after = surf_rescale.sag_slopes(test_x, test_y)

    assert np.allclose(sag_before, sag_after, atol=1e-12)
    assert np.allclose(gx_before, gx_after, atol=1e-12)
    assert np.allclose(gy_before, gy_after, atol=1e-12)


def test_dense_convex_aperture_margin_uses_edge_offset_regression():
    """Dense convex hull must remain convex after a true 0.5 mm supporting-edge offset."""
    count = 73
    u = np.linspace(0.0, 1.0, count, endpoint=False)
    theta = 2.0 * np.pi * u + 0.8 * np.sin(2.0 * np.pi * u)
    footprint_xy = np.column_stack((10.0 * np.cos(theta), np.sin(theta)))
    footprint = np.column_stack((footprint_xy, np.zeros(count)))

    base_poly, _ = convex_aperture_from_points(
        footprint_xy, margin_mm=0.0, floor=(1.0, 1.0)
    )
    assert len(base_poly) == count

    # Regression witness: the removed centroid-radial rule makes this otherwise
    # convex 73-vertex footprint non-convex after a 0.5 mm expansion.
    centroid = np.mean(base_poly, axis=0)
    radial = base_poly - centroid
    radial_poly = base_poly + 0.5 * radial / np.linalg.norm(radial, axis=1)[:, None]
    radial_edges = np.roll(radial_poly, -1, axis=0) - radial_poly
    radial_turns = (
        radial_edges[:, 0] * np.roll(radial_edges[:, 1], -1)
        - radial_edges[:, 1] * np.roll(radial_edges[:, 0], -1)
    )
    assert np.any(radial_turns < 0.0)

    poly, half = convex_aperture_from_points(
        footprint_xy, margin_mm=0.5, floor=(1.0, 1.0)
    )
    poly = normalize_aperture_polygon(poly)
    assert len(poly) == count
    edges = np.roll(poly, -1, axis=0) - poly
    turns = (
        edges[:, 0] * np.roll(edges[:, 1], -1)
        - edges[:, 1] * np.roll(edges[:, 0], -1)
    )
    assert np.all(turns > 0.0)

    cross = (
        edges[None, :, 0] * (base_poly[:, None, 1] - poly[None, :, 1])
        - edges[None, :, 1] * (base_poly[:, None, 0] - poly[None, :, 0])
    )
    support_distance = cross / np.linalg.norm(edges, axis=1)[None, :]
    assert float(np.min(support_distance)) >= 0.5 - 1e-10
    assert np.all(half >= np.max(np.abs(poly), axis=0))

    surf = PolySurface.plane(
        "M1", np.zeros(3), np.array([0.0, 0.0, 1.0]), (20.0, 20.0)
    )
    _resize(surf, footprint)
    normalize_aperture_polygon(surf.aperture_polygon)


def test_fermat_reflection_branch_validation():
    """9.3: Fermat tren M2 phai chon dung nhanh phan xa va reject truyen tiep."""
    # Guong phang M2 tai z = 0, phap tuyen huong +z
    m2_plane = PolySurface.plane("M2_MIRROR", np.array([0.0, 0.0, 0.0]),
                                 np.array([0.0, 0.0, 1.0]), (20.0, 20.0))
    
    # Truong hop 1: Phan xa that su
    # Q1 = (-1, 0, 2), target = (1, 0, 2)
    # Nghiem phan xa dung tai Q2 = (0, 0, 0)
    q1 = np.array([[-1.0, 0.0, 2.0]])
    target_refl = np.array([[1.0, 0.0, 2.0]])
    initial_guess = np.array([[0.1, 0.0, 0.0]])
    
    sol_refl = solve_fermat_m2(q1, target_refl, m2_plane, initial_guess,
                               max_iter=30, grad_tol=1e-7, reflection_tolerance=1e-4)
    assert sol_refl["success"][0]
    assert sol_refl["reflection_pass"][0]
    assert abs(sol_refl["target_points"][0, 0]) < 1e-5
    assert abs(sol_refl["reflection_residual"][0]) < 1e-4

    # Truong hop 2: Nhanh truyen tiep (truyen thang qua guong)
    # Q1 = (-1, 0, 2), target = (1, 0, -2) (nam ben duoi guong z=0!)
    # Doan thang Q1 -> target cat mat tai Q2 = (0, 0, 0)
    # Stationarity: dOP/dx = 0, dOP/dy = 0 dat chuan xac!
    # Nhung day la tia truyen tiep (transmission), KHONG PHAI phan xa!
    target_trans = np.array([[1.0, 0.0, -2.0]])
    sol_trans = solve_fermat_m2(q1, target_trans, m2_plane, initial_guess,
                                max_iter=30, grad_tol=1e-7, reflection_tolerance=1e-4)
    # Gradient dat nhung reflection phai FAIL!
    assert sol_trans["gradient_pass"][0]
    assert not sol_trans["reflection_pass"][0]
    assert not sol_trans["success"][0]
    assert sol_trans["failure_reasons"][0] in ("WRONG_REFLECTION_BRANCH", "FERMAT_WRONG_REFLECTION_BRANCH")
    assert sol_trans["reflection_residual"][0] > 1.0

    # Audit phai bao branch_may_continue = False
    audit_trans = _fermat_audit(
        m2_plane,
        sol_trans,
        "TEST_TRANS",
        expected_ray_count=1,
        gradient_tolerance=1e-7,
        reflection_tolerance=1e-4,
    )
    assert not audit_trans["branch_may_continue"]
    assert audit_trans["wrong_reflection_branch_count"] == 1


def test_step14_fermat_diagnostic_hard_fails_outside_construction_domain():
    """STEP14 phải fail construction-domain dù core solver success hiện vẫn True."""
    m2 = PolySurface.plane(
        "M2",
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        (10.0, 10.0),
    )

    n = 2
    q1 = np.tile(
        np.array([[0.0, 0.0, 10.0]]),
        (n, 1),
    )
    q2_current = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    q2_fermat = np.array([
        [1.0, 0.0, 0.0],
        [31.0, 0.0, 0.0],
    ])

    rays = {
        "rows": [
            {
                "ray_id": "R0",
                "field_id": "F0",
                "field_index": 0,
                "pupil_id": "P0",
                "pupil_index": 0,
                "sample_id": "S0",
                "sample_index": 0,
                "chief_preseed": True,
            },
            {
                "ray_id": "R1",
                "field_id": "F0",
                "field_index": 0,
                "pupil_id": "P0",
                "pupil_index": 0,
                "sample_id": "S1",
                "sample_index": 1,
                "chief_preseed": False,
            },
        ]
    }

    trace = {
        "points": [
            np.zeros((n, 3)),
            q1,
            q2_current,
        ]
    }

    sol = {
        "target_points":
            q2_fermat,
        "xy":
            q2_fermat[:, :2].copy(),
        "gradient":
            np.zeros((n, 2)),
        "gradient_norm":
            np.zeros(n),
        "reflection_residual":
            np.zeros(n),
        "success":
            np.ones(n, dtype=bool),
        "gradient_pass":
            np.ones(n, dtype=bool),
        "reflection_pass":
            np.ones(n, dtype=bool),
        "finite_pass":
            np.ones(n, dtype=bool),
        "conic_domain_pass":
            np.ones(n, dtype=bool),
        "in_aperture":
            np.array(
                [True, False],
                dtype=bool,
            ),
        "in_construction_domain":
            np.array(
                [True, False],
                dtype=bool,
            ),
        "fallback_used":
            np.zeros(n, dtype=bool),
        "step_norm":
            np.zeros(n),
        "op_initial":
            np.ones(n),
        "op_final":
            np.ones(n),
        "iterations":
            1,
        "construction_domain_factor":
            3.0,
        "failure_reasons":
            np.array(
                [
                    "FERMAT_STATIONARY_GRADIENT_AND_REFLECTION_PASS",
                    "FERMAT_STATIONARY_GRADIENT_AND_REFLECTION_PASS",
                ],
                dtype=object,
            ),
    }

    diagnostic = _step14_fermat_diagnostics(
        rays,
        trace,
        m2,
        sol,
        gradient_tolerance=1e-7,
        reflection_tolerance=1e-4,
        neighbor_count=1,
    )

    assert diagnostic["summary"]["hard_pass_count"] == 1
    assert diagnostic["summary"]["hard_fail_count"] == 1
    assert diagnostic["summary"]["hard_fail_counts"]["construction_domain"] == 1
    assert diagnostic["summary"]["all_hard_pass"] is False

    failed = diagnostic["failed_rows"]
    assert len(failed) == 1
    assert failed[0]["ray_id"] == "R1"
    assert failed[0]["step14_hard_fail_reason"] == "CONSTRUCTION_DOMAIN_FAIL"
    assert failed[0]["construction_limit_x_mm"] == 30.0
    assert failed[0]["construction_excess_x_mm"] == 1.0
    assert failed[0]["solver_success"] is True
    assert failed[0]["step14_hard_pass"] is False


def test_step14_continuity_suspect_is_diagnostic_only():
    """Delta-Q2 outlier phải được note nhưng không biến thành physical hard fail."""
    m2 = PolySurface.plane(
        "M2",
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        (10.0, 10.0),
    )

    n = 6
    q1 = np.tile(
        np.array([[0.0, 0.0, 10.0]]),
        (n, 1),
    )
    q2_current = np.column_stack([
        np.arange(n, dtype=float),
        np.zeros(n),
        np.zeros(n),
    ])
    q2_fermat = q2_current.copy()
    q2_fermat[-1, 0] = 25.0

    rows = []
    for index in range(n):
        rows.append({
            "ray_id": f"R{index}",
            "field_id": "F0",
            "field_index": 0,
            "pupil_id": "P0",
            "pupil_index": 0,
            "sample_id": f"S{index}",
            "sample_index": index,
            "chief_preseed": bool(index == 0),
        })

    rays = {
        "rows":
            rows
    }
    trace = {
        "points": [
            np.zeros((n, 3)),
            q1,
            q2_current,
        ]
    }

    sol = {
        "target_points":
            q2_fermat,
        "xy":
            q2_fermat[:, :2].copy(),
        "gradient":
            np.zeros((n, 2)),
        "gradient_norm":
            np.zeros(n),
        "reflection_residual":
            np.zeros(n),
        "success":
            np.ones(n, dtype=bool),
        "gradient_pass":
            np.ones(n, dtype=bool),
        "reflection_pass":
            np.ones(n, dtype=bool),
        "finite_pass":
            np.ones(n, dtype=bool),
        "conic_domain_pass":
            np.ones(n, dtype=bool),
        "in_aperture":
            np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
                dtype=bool,
            ),
        "in_construction_domain":
            np.ones(n, dtype=bool),
        "fallback_used":
            np.zeros(n, dtype=bool),
        "step_norm":
            np.zeros(n),
        "op_initial":
            np.ones(n),
        "op_final":
            np.ones(n),
        "iterations":
            1,
        "construction_domain_factor":
            3.0,
        "failure_reasons":
            np.full(
                n,
                "FERMAT_STATIONARY_GRADIENT_AND_REFLECTION_PASS",
                dtype=object,
            ),
    }

    diagnostic = _step14_fermat_diagnostics(
        rays,
        trace,
        m2,
        sol,
        gradient_tolerance=1e-7,
        reflection_tolerance=1e-4,
        neighbor_count=2,
    )

    assert diagnostic["summary"]["hard_fail_count"] == 0
    assert diagnostic["summary"]["all_hard_pass"] is True
    assert diagnostic["summary"]["continuity_suspect_count"] >= 1
    assert diagnostic["summary"]["continuity_is_hard_gate"] is False

    last = diagnostic["rows"][-1]
    assert last["step14_hard_pass"] is True
    assert last["continuity_suspect"] is True
    assert last["Delta_Q2_mm"] == 20.0
    assert last["in_current_aperture"] is False
    assert last["construction_domain_pass"] is True


def test_surface_fit_and_conic_aperture_sanity():
    """9.4: Fit tren candidate aperture va rang buoc conic domain toan aperture."""
    # A. Cloud co Delta hop le nhung expanded aperture co Delta khong hop le
    # Khi K > -1, kiem tra (1+K)*c^2*r_max^2 < 1
    c = 0.05
    K = 1.0 # (1+K) = 2.0. (1+K)*c^2 = 2 * 0.0025 = 0.005
    # r_max = 15 => r_max^2 = 225 => (1+K)*c^2*r_max^2 = 1.125 > 1 => CONIC DOMAIN INVALID!
    surf_bad_conic = PolySurface("BAD_CONIC", np.zeros(3), np.eye(3), (15.0, 15.0),
                                 [], np.zeros(0), np.array([15.0, 15.0]), c, K, False)
    sanity_bad = surface_sanity(surf_bad_conic)
    assert not sanity_bad["conic_domain_valid"]
    assert not sanity_bad["pass"]
    assert "CONIC_DOMAIN_INVALID" in sanity_bad["failure_reasons"]

    # B. K <= -1 (hyperbolic / parabolic) khong bi coi la conic domain fail vi 1 - (1+K)c^2 r^2 >= 1
    surf_hyperbolic = PolySurface("HYPERBOLIC", np.zeros(3), np.eye(3), (15.0, 15.0),
                                  [], np.zeros(0), np.array([15.0, 15.0]), c, -2.5, False)
    sanity_hyp = surface_sanity(surf_hyperbolic)
    assert sanity_hyp["conic_domain_valid"]
    assert sanity_hyp["conic_is_hyperbolic_or_parabolic"]


def test_ci_incumbent_and_state_machine():
    """9.5: Bao toan nghiem fully-physical incumbent doc lap voi beam pruning va restoration."""
    ctx = Context(source_dir=Path("."), config_path=Path("config_v55.json"),
                  config={}, run_dir=Path("."), data={})
    
    # Tao 1 candidate O3 fully physical tot
    cand_good = {
        "Fan_core_merit": 0.05,
        "physical_first_hit_complete": True,
        "physical_valid_count": 33075,
        "chief_centered_spot_RMS_mm": 0.03
    }
    state_good = {"Fan_core_merit": 0.05, "reconstruct_out": cand_good, "rho_sequence": [0.5]}
    
    # Tao 1 candidate O3 restoration (thieu tia, e.g. 33070 tia) co Fan merit tot hon ti xiu (0.048)
    cand_resto = {
        "Fan_core_merit": 0.048,
        "physical_first_hit_complete": False,
        "physical_valid_count": 33070,
        "chief_centered_spot_RMS_mm": 0.029,
        "restoration_branch": True
    }
    state_resto = {"Fan_core_merit": 0.048, "reconstruct_out": cand_resto, "rho_sequence": [0.25]}

    solver_cfg = {"rho_beam_width": 1, "restoration_max_branches": 1}

    # Frontier pruning chon candidate tot nhat theo tieu chi
    # Nho _rho_sort_key: state_good (-33075) phai dung truoc state_resto (-33070)
    frontier = _select_ci_frontier([state_good, state_resto], solver_cfg)
    assert frontier[0] == state_good
    assert frontier[0]["reconstruct_out"]["physical_first_hit_complete"] is True

    # Archive incumbents
    incumbents = {"feasible": {3: copy.deepcopy(state_good)}, "admissible": {3: copy.deepcopy(state_resto)}}
    # Khi cycle tiep theo bi failed, incumbent phai duoc lay lai tu archive:
    recovered = incumbents["feasible"][3]
    assert recovered["reconstruct_out"]["physical_first_hit_complete"] is True
    assert recovered["Fan_core_merit"] == 0.05


def test_step16_o2_rho_search_ranking_and_refinement_grid():
    """STEP16 coarse-to-fine phải hard-gate physics rồi mới rank image quality."""
    full_better_tail = {
        "Fan_core_merit": 0.50,
        "rho_sequence": [0.20],
        "reconstruct_out": {
            "step16_hard_feasible": True,
            "step16_spot_tail": {
                "bundle_RMS_P95_mm": 0.64,
                "worst_bundle_RMS_mm": 0.78,
                "global_RMS_mm": 0.51,
                "max_spot_radius_mm": 0.90,
            },
        },
    }

    full_worse_fan = {
        "Fan_core_merit": 0.52,
        "rho_sequence": [0.175],
        "reconstruct_out": {
            "step16_hard_feasible": True,
            "step16_spot_tail": {
                "bundle_RMS_P95_mm": 0.60,
                "worst_bundle_RMS_mm": 0.70,
                "global_RMS_mm": 0.48,
                "max_spot_radius_mm": 0.80,
            },
        },
    }

    restoration = {
        "Fan_core_merit": 0.10,
        "rho_sequence": [0.25],
        "reconstruct_out": {
            "step16_hard_feasible": False,
            "step16_spot_tail": {
                "bundle_RMS_P95_mm": 0.20,
                "worst_bundle_RMS_mm": 0.30,
                "global_RMS_mm": 0.20,
                "max_spot_radius_mm": 0.30,
            },
        },
    }

    ordered = sorted(
        [
            restoration,
            full_worse_fan,
            full_better_tail,
        ],
        key=_rho_o2_search_sort_key,
    )

    assert ordered[0] is full_better_tail
    assert ordered[1] is full_worse_fan
    assert ordered[2] is restoration

    grid, step = _step16_refinement_grid(
        0.20,
        0.025,
        9,
        {round(0.20, 15)},
    )

    assert np.isclose(step, 0.00625)
    assert len(grid) == 8
    assert np.isclose(min(grid), 0.175)
    assert np.isclose(max(grid), 0.225)
    assert all(value > 0.0 for value in grid)
    assert not any(
        np.isclose(value, 0.20)
        for value in grid
    )


def test_step16_spot_tail_and_hard_feasibility():
    """P95/worst bundle phải lấy từ bundle_rows; restoration không được coi là best-feasible."""
    out = {
        "ray_count": 4,
        "physical_valid_count": 4,
        "physical_first_hit_complete": True,
        "unobscured": True,
        "integrability_quality": {
            "status": "PASS",
        },
        "M1_sanity": {
            "pass": True,
        },
        "M2_sanity": {
            "pass": True,
        },
        "chief_centered_spot": {
            "RMS_spot_radius_mm": 0.50,
            "max_spot_radius_mm": 1.20,
            "bundle_rows": [
                {
                    "RMS_to_chief_mm": 0.30,
                },
                {
                    "RMS_to_chief_mm": 0.40,
                },
                {
                    "RMS_to_chief_mm": 0.80,
                },
                {
                    "RMS_to_chief_mm": 1.00,
                },
            ],
        },
    }

    tail = _step16_spot_tail_metrics(out)

    assert tail["bundle_count"] == 4
    assert tail["finite_bundle_count"] == 4
    assert np.isclose(
        tail["worst_bundle_RMS_mm"],
        1.0,
    )
    assert np.isclose(
        tail["global_RMS_mm"],
        0.50,
    )
    assert np.isclose(
        tail["max_spot_radius_mm"],
        1.20,
    )
    assert _step16_full_hard_feasible(out)

    incomplete = copy.deepcopy(out)
    incomplete["physical_valid_count"] = 3
    incomplete["physical_first_hit_complete"] = False

    assert not _step16_full_hard_feasible(
        incomplete
    )


def test_forward_shoot_trajectory_consistency():
    """9.6: Kiem tra forward_shoot giu nguyen tinh nhat quan cua finite trace."""
    # Tao he quang don gian: Display phang, M2 phang, M1 phang, Visor Chebyshev
    display = PolySurface.plane("DISPLAY", np.array([0.0, 0.0, -100.0]), np.array([0.0, 0.0, 1.0]), (10.0, 10.0))
    m2 = PolySurface.plane("M2", np.array([0.0, 0.0, -50.0]), np.array([0.0, 0.0, 1.0]), (20.0, 20.0))
    m1 = PolySurface.plane("M1", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (30.0, 30.0))
    # Visor: center, frame, coeff, scale, bounds
    visor = ChebVisor(np.array([0.0, 0.0, 50.0]), np.eye(3), np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))
    
    src = np.array([[0.0, 0.0, -100.0]])
    target_eye = np.array([[0.0, 0.0, 100.0]])
    init_dir = np.array([[0.0, 0.0, 1.0]])
    
    fwd = forward_shoot(src, target_eye, init_dir, display, m2, m1, visor,
                        max_iter=5, tolerance_mm=1.0, fd_angle=1e-4, damping=0.5)
    
    # Kiem tra tat ca cac truong co mat va thuoc cung mot trace
    assert "residual_mm" in fwd
    assert "converged" in fwd
    assert "unconstrained_diagnostic" in fwd
    # residual_mm phai bang norm cua residual_yz tu chinh finite trace
    expected_norm = np.linalg.norm(fwd["residual_yz"], axis=1)
    assert np.allclose(fwd["residual_mm"], expected_norm, equal_nan=True)


def test_polynomial_root_solver_preserves_small_high_order_terms():
    """High-order coefficients that are small in raw magnitude may still change roots at large t."""
    coef = np.array([
        3.8898e4,
        -2.227e3,
        3.122e1,
        -1.626e-2,
        2.282e-5,
        9.595e-8,
        -2.097e-11,
        4.153e-14,
        8.636e-17,
    ], dtype=float)

    poly = Polynomial(coef)

    roots, status = _distance_scaled_polynomial_real_roots(
        poly,
        t_reference=43.88,
    )

    assert status == "ROOTS_ISOLATED_BY_DISTANCE_SCALED_POLYNOMIAL"
    assert len(roots) >= 1
    assert np.all(np.isfinite(roots))

    expected = 43.842059142671175

    chosen = float(
        roots[np.argmin(np.abs(roots - expected))]
    )

    assert abs(chosen - expected) < 1e-4

    residual = float(poly(chosen))

    assert abs(residual) < 1e-5


def test_nearly_planar_visor_roundoff_does_not_destroy_linear_root():
    """Thực thi test_nearly_planar_visor_roundoff_does_not_destroy_linear_root."""
    coef = np.array([
        -5.836483531432799e1,
        9.48172965e-1,
        -5.83226068e-18,
        1.53702706e-19,
        -1.55540616e-21,
        5.38534206e-24,
        4.95796058e-59,
        -2.03269584e-79,
        -1.40421059e-99,
        1.31030107e-120,
        6.79661754e-141,
    ])

    roots, status = _polynomial_real_roots(
        Polynomial(coef)
    )

    assert len(roots) >= 1

    expected = (
        -coef[0] / coef[1]
    )

    nearest = roots[
        np.argmin(np.abs(roots - expected))
    ]

    assert abs(nearest - expected) < 1e-6


def test_point_normal_diagnostic_rejects_nonfinite_before_kdtree():
    """Thực thi test_point_normal_diagnostic_rejects_nonfinite_before_kdtree."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [np.nan, 1.0, 0.0],
    ])

    normals = np.tile(
        np.array([[0.0, 0.0, 1.0]]),
        (4, 1),
    )

    with pytest.raises(
        ValueError,
        match="POINT_NORMAL_DIAGNOSTIC_NONFINITE_POINTS",
    ):
        point_normal_cloud_diagnostics(
            points,
            normals,
            np.eye(3),
            neighbors=3,
            max_samples=4,
        )


def test_construction_trace_ignores_m1_m2_display_aperture_only():
    """1. Construction trace bỏ qua clear aperture của M1, M2 và DISPLAY."""
    nv = unit(np.array([1.0, 0.0, 1.0]))
    uv = unit(np.array([0.0, 1.0, 0.0]))
    vv = unit(np.cross(nv, uv))
    frame_v = np.column_stack([uv, vv, nv])
    visor = ChebVisor(np.array([0.0, 0.0, 100.0]), frame_v, np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    n1 = unit(np.array([-1.0, 0.0, 1.0]))
    m1 = PolySurface.plane("M1", np.array([50.0, 0.0, 100.0]), n1, (1.0, 1.0))
    n2 = unit(np.array([0.0, -1.0, -1.0]))
    m2 = PolySurface.plane("M2", np.array([50.0, 0.0, 150.0]), n2, (1.0, 1.0))
    display = PolySurface.plane("DISPLAY", np.array([50.0, -50.0, 150.0]), np.array([0.0, 1.0, 0.0]), (1.0, 1.0))

    rays = {
        "origins": np.array([[0.0, 2.0, 150.0]]),
        "directions": np.array([[0.0, 0.0, -1.0]]),
    }

    finite_tr = trace_reverse(rays, visor, m1, m2, display, physical_first_hit=False, sequential_aperture_mode="FINITE")
    assert not finite_tr["valid"][0]

    constr_tr = trace_reverse(rays, visor, m1, m2, display, physical_first_hit=False, sequential_aperture_mode="CONSTRUCTION_FOOTPRINT")
    assert constr_tr["valid"][0]
    assert np.all(np.isfinite(constr_tr["landing"]))


def test_construction_trace_does_not_ignore_visor_bounds():
    """2. Construction trace KHÔNG bỏ qua giới hạn của VISOR."""
    nv = unit(np.array([1.0, 0.0, 1.0]))
    uv = unit(np.array([0.0, 1.0, 0.0]))
    vv = unit(np.cross(nv, uv))
    frame_v = np.column_stack([uv, vv, nv])
    visor = ChebVisor(np.array([0.0, 0.0, 100.0]), frame_v, np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    n1 = unit(np.array([-1.0, 0.0, 1.0]))
    m1 = PolySurface.plane("M1", np.array([50.0, 0.0, 100.0]), n1, (100.0, 100.0))
    n2 = unit(np.array([0.0, -1.0, -1.0]))
    m2 = PolySurface.plane("M2", np.array([50.0, 0.0, 150.0]), n2, (100.0, 100.0))
    display = PolySurface.plane("DISPLAY", np.array([50.0, -50.0, 150.0]), np.array([0.0, 1.0, 0.0]), (100.0, 100.0))

    rays = {
        "origins": np.array([[0.0, 100.0, 150.0]]),
        "directions": np.array([[0.0, 0.0, -1.0]]),
    }
    constr_tr = trace_reverse(rays, visor, m1, m2, display, physical_first_hit=False, sequential_aperture_mode="CONSTRUCTION_FOOTPRINT")
    assert not constr_tr["valid"][0]


def test_physical_first_hit_cannot_use_construction_mode():
    """3. physical_first_hit=True không được dùng sequential_aperture_mode khác FINITE."""
    nv = unit(np.array([1.0, 0.0, 1.0]))
    uv = unit(np.array([0.0, 1.0, 0.0]))
    vv = unit(np.cross(nv, uv))
    frame_v = np.column_stack([uv, vv, nv])
    visor = ChebVisor(np.array([0.0, 0.0, 100.0]), frame_v, np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    m1 = PolySurface.plane("M1", np.array([50.0, 0.0, 100.0]), unit(np.array([-1.0, 0.0, 1.0])), (50.0, 50.0))
    m2 = PolySurface.plane("M2", np.array([50.0, 0.0, 150.0]), unit(np.array([0.0, -1.0, -1.0])), (50.0, 50.0))
    display = PolySurface.plane("DISPLAY", np.array([50.0, -50.0, 150.0]), np.array([0.0, 1.0, 0.0]), (50.0, 50.0))
    rays = {"origins": np.array([[0.0, 0.0, 150.0]]), "directions": np.array([[0.0, 0.0, -1.0]])}

    with pytest.raises(ValueError, match="PHYSICAL_FIRST_HIT_REQUIRES_FINITE_APERTURES"):
        trace_reverse(rays, visor, m1, m2, display, physical_first_hit=True, sequential_aperture_mode="CONSTRUCTION_FOOTPRINT")


def test_stale_aperture_edge_ray_regression():
    """4. Khắc phục lỗi edge ray bị aperture cũ cắt: old finite FAIL -> construction PASS -> rebuild -> new finite PASS."""
    nv = unit(np.array([1.0, 0.0, 1.0]))
    uv = unit(np.array([0.0, 1.0, 0.0]))
    vv = unit(np.cross(nv, uv))
    frame_v = np.column_stack([uv, vv, nv])
    visor = ChebVisor(np.array([0.0, 0.0, 100.0]), frame_v, np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    n1 = unit(np.array([-1.0, 0.0, 1.0]))
    m1 = PolySurface.plane("M1", np.array([50.0, 0.0, 100.0]), n1, (1.0, 1.0))
    n2 = unit(np.array([0.0, -1.0, -1.0]))
    m2 = PolySurface.plane("M2", np.array([50.0, 0.0, 150.0]), n2, (1.0, 1.0))
    display = PolySurface.plane("DISPLAY", np.array([50.0, -50.0, 150.0]), np.array([0.0, 1.0, 0.0]), (1.0, 1.0))

    rays = {
        "origins": np.array([
            [0.0, 0.0, 150.0],
            [0.0, 2.0, 150.0],
            [0.0, -2.0, 150.0],
            [2.0, 0.0, 150.0],
            [-2.0, 0.0, 150.0],
        ]),
        "directions": np.tile(np.array([[0.0, 0.0, -1.0]]), (5, 1)),
    }
    ctx = Context(source_dir=Path("."), config_path=Path("config_v55.json"), config={"display_seed_mm": [20.0, 20.0]}, run_dir=Path("."))
    ctx.data["rays"] = rays
    ctx.data["visor"] = visor

    old_finite = trace_reverse(rays, visor, m1, m2, display, physical_first_hit=False, sequential_aperture_mode="FINITE")
    assert not np.all(old_finite["valid"])

    constr, new_finite, audit = _rebuild_post_fit_apertures(ctx, m1, m2, display, "TEST_REGRESSION")
    assert np.all(constr["valid"])
    assert np.all(new_finite["valid"])
    assert np.all(np.isfinite(new_finite["landing"]))


def test_true_no_root_ray_remains_fail():
    """5. Tia thực sự không có giao điểm vẫn phải FAIL trong cả construction footprint mode."""
    nv = unit(np.array([1.0, 0.0, 1.0]))
    uv = unit(np.array([0.0, 1.0, 0.0]))
    vv = unit(np.cross(nv, uv))
    frame_v = np.column_stack([uv, vv, nv])
    visor = ChebVisor(np.array([0.0, 0.0, 100.0]), frame_v, np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    m1 = PolySurface.plane("M1", np.array([50.0, 0.0, 100.0]), unit(np.array([-1.0, 0.0, 1.0])), (50.0, 50.0))
    m2 = PolySurface.plane("M2", np.array([50.0, 0.0, 150.0]), unit(np.array([0.0, -1.0, -1.0])), (50.0, 50.0))
    display = PolySurface.plane("DISPLAY", np.array([50.0, -50.0, 150.0]), np.array([0.0, 1.0, 0.0]), (50.0, 50.0))

    rays = {"origins": np.array([[0.0, 0.0, 150.0]]), "directions": np.array([[0.0, 0.0, 1.0]])}
    constr = trace_reverse(rays, visor, m1, m2, display, physical_first_hit=False, sequential_aperture_mode="CONSTRUCTION_FOOTPRINT")
    assert not constr["valid"][0]


def test_rebuild_aperture_does_not_change_surface_parameters():
    """6. Rebuild aperture không làm thay đổi các tham số hình học."""
    nv = unit(np.array([1.0, 0.0, 1.0]))
    uv = unit(np.array([0.0, 1.0, 0.0]))
    vv = unit(np.cross(nv, uv))
    frame_v = np.column_stack([uv, vv, nv])
    visor = ChebVisor(np.array([0.0, 0.0, 100.0]), frame_v, np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    n1 = unit(np.array([-1.0, 0.0, 1.0]))
    u1 = unit(np.array([0.0, 1.0, 0.0]))
    v1 = unit(np.cross(n1, u1))
    frame1 = np.column_stack([u1, v1, n1])

    n2 = unit(np.array([0.0, -1.0, -1.0]))
    u2 = unit(np.array([1.0, 0.0, 0.0]))
    v2 = unit(np.cross(n2, u2))
    frame2 = np.column_stack([u2, v2, n2])

    terms = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)]
    coeff = np.array([0.0, 0.001, -0.001, 0.0001, -0.0001])
    m1 = PolySurface("M1", np.array([50.0, 0.0, 100.0]), frame1, np.array([30.0, 30.0]), terms, coeff, np.array([30.0, 30.0]), 0.0005, -1.0)
    m2 = PolySurface("M2", np.array([50.0, 0.0, 150.0]), frame2, np.array([30.0, 30.0]), terms, coeff, np.array([30.0, 30.0]), -0.0005, 0.5)
    display = PolySurface.plane("DISPLAY", np.array([50.0, -50.0, 150.0]), np.array([0.0, 1.0, 0.0]), (20.0, 20.0))

    snap_m1 = _surface_shape_snapshot(m1)
    snap_m2 = _surface_shape_snapshot(m2)
    snap_disp = _surface_shape_snapshot(display)

    rays = {
        "origins": np.array([
            [0.0, 0.0, 150.0],
            [0.0, 2.0, 150.0],
            [0.0, -2.0, 150.0],
            [2.0, 0.0, 150.0],
            [-2.0, 0.0, 150.0],
        ]),
        "directions": np.tile(np.array([[0.0, 0.0, -1.0]]), (5, 1)),
    }
    ctx = Context(source_dir=Path("."), config_path=Path("config_v55.json"), config={"display_seed_mm": [20.0, 20.0]}, run_dir=Path("."))
    ctx.data["rays"] = rays
    ctx.data["visor"] = visor

    constr, finite_tr, audit = _rebuild_post_fit_apertures(ctx, m1, m2, display, "SHAPE_UNCHANGED_TEST")
    _assert_surface_shape_unchanged(snap_m1, m1, "ASSERT_M1")
    _assert_surface_shape_unchanged(snap_m2, m2, "ASSERT_M2")
    _assert_surface_shape_unchanged(snap_disp, display, "ASSERT_DISPLAY")
    assert audit["surface_shape_changed"] is False


def test_parallel_poly_surface_parity_all_arrays():
    """7, 8, 9, 10, 11: Đối chiếu đầy đủ 100% parity giữa serial và worker PolySurface.intersect."""
    from execution_workers_v55 import poly_surface_intersect_worker

    terms = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
    coeff = np.array([0.0, 0.01, -0.01, 0.002, 0.003, -0.002, 0.0001, -0.0001, 0.00005])
    surf = PolySurface("M_TEST", np.array([0.0, 0.0, 0.0]), np.eye(3), np.array([50.0, 50.0]), terms, coeff, np.array([50.0, 50.0]), 0.002, -1.0)

    rng = np.random.default_rng(42)
    n_rays = 2500
    origins = np.zeros((n_rays, 3))
    origins[:, 0] = rng.uniform(-40.0, 40.0, n_rays)
    origins[:, 1] = rng.uniform(-40.0, 40.0, n_rays)
    origins[:, 2] = 50.0
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (n_rays, 1))

    serial_res = surf.intersect(origins, directions, finite=True)

    job = {
        "chunk_index": 0, "start": 0, "end": n_rays,
        "surface": surf.to_dict(), "origins": origins, "directions": directions,
        "finite": True, "t_min": 1e-6
    }
    worker_res = poly_surface_intersect_worker(job)["result"]

    for key in ("t", "point", "normal", "x", "y", "residual", "raw_conic_argument"):
        assert np.allclose(serial_res[key], worker_res[key], rtol=1e-12, atol=1e-12, equal_nan=True), f"Mismatch in {key}"

    assert np.array_equal(serial_res["valid"], worker_res["valid"])
    assert np.array_equal(serial_res["status"], worker_res["status"])
    assert np.array_equal(serial_res["resolved"], worker_res["resolved"])

    for key in ("candidate_count", "outside_aperture_candidate_count", "conic_invalid_candidate_count"):
        assert np.array_equal(serial_res[key], worker_res[key]), f"Mismatch in count {key}"


def test_small_batches_remain_serial():
    """12. Các batch nhỏ hơn ray_trace_min_rays (2048) được chạy tuần tự."""
    from core import _parallel_poly_surface_intersect
    from execution_v55 import ExecutionRuntime, _CURRENT_RUNTIME

    surf = PolySurface.plane("M1", np.zeros(3), np.eye(3), (50.0, 50.0))
    O = np.zeros((10, 3))
    D = np.tile(np.array([0.0, 0.0, -1.0]), (10, 1))

    rt = ExecutionRuntime(mode="accelerated", config={"parallel_ray_trace": True, "ray_trace_min_rays": 2048})
    token = _CURRENT_RUNTIME.set(rt)
    try:
        assert _parallel_poly_surface_intersect(surf, O, D, True, 1e-6) is None
    finally:
        _CURRENT_RUNTIME.reset(token)


def test_worker_cannot_nested_pool():
    """13. Worker con không được phép mở compute pool lồng."""
    import execution_v55

    old_worker = execution_v55._IS_WORKER_PROCESS
    old_runtime = execution_v55._CURRENT_RUNTIME.get()
    try:
        policy = {"execution_config": {"mode": "accelerated"}, "backend_plan": {}}
        execution_v55.mark_as_compute_worker(policy)
        assert execution_v55.is_compute_worker() is True

        with pytest.raises(RuntimeError, match="WORKER_CANNOT_OPEN_COMPUTE_POOL"):
            with execution_v55.managed_compute_pool(None, purpose="NESTED_TEST"):
                pass
    finally:
        execution_v55._IS_WORKER_PROCESS = old_worker
        execution_v55._CURRENT_RUNTIME.set(old_runtime)


def test_forward_active_fd_parity():
    """14. Kiểm tra parity của forward_active_fd kernel."""
    from run_qualification_v55 import qualify_forward_active_fd
    passed, passed_count, failed_count, _, scope = qualify_forward_active_fd()
    assert passed is True
    assert failed_count == 0
    assert passed_count == 1


def test_fit_surface_rejects_nonfinite_points_before_linear_algebra():
    """B16 defense: NaN point must be rejected before LAPACK/SVD is called."""
    seed = PolySurface.plane(
        "M1",
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        (5.0, 5.0),
    )

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [np.nan, 1.0, 0.0],
    ])

    normals = np.tile(
        np.array([[0.0, 0.0, 1.0]]),
        (4, 1),
    )

    with pytest.raises(
        RuntimeError,
        match="CI_SURFACE_FIT_NONFINITE_POINTS",
    ):
        fit_surface(
            points,
            normals,
            seed,
            2,
            True,
            {},
            0,
            {},
        )


def test_fit_surface_rejects_nonfinite_normals_before_linear_algebra():
    """B16 defense: NaN normal must be rejected before normalization and SVD."""
    seed = PolySurface.plane(
        "M1",
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        (5.0, 5.0),
    )

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    normals = np.tile(
        np.array([[0.0, 0.0, 1.0]]),
        (4, 1),
    )

    normals[2, 0] = np.nan

    with pytest.raises(
        RuntimeError,
        match="CI_SURFACE_FIT_NONFINITE_NORMALS",
    ):
        fit_surface(
            points,
            normals,
            seed,
            2,
            True,
            {},
            0,
            {},
        )


def test_reconstruct_translates_linalg_error_to_candidate_runtime_rejection(
    monkeypatch,
):
    """A numerical linear-algebra failure is a branch rejection, not an untyped STEP abort."""
    import pipeline_v55 as pipeline
    from types import SimpleNamespace

    ctx = SimpleNamespace(data={})

    def fail_impl(
        ctx_arg,
        order,
        axis_order2,
        allow_restoration,
        *,
        evidence,
    ):
        """Thực thi fail_impl."""
        evidence["phase"] = "FIT_M2"
        raise np.linalg.LinAlgError(
            "synthetic_svd_failure"
        )

    monkeypatch.setattr(
        pipeline,
        "_reconstruct_impl",
        fail_impl,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"ORDER_2_FIT_M2_"
            r"LINEAR_ALGEBRA_FAILURE:"
            r"synthetic_svd_failure"
        ),
    ):
        pipeline._reconstruct(
            ctx,
            2,
            True,
            allow_restoration=True,
        )

    evidence = ctx.data[
        "_last_candidate_evidence"
    ]

    assert (
        evidence["status"]
        == "REJECTED_OR_ABORTED"
    )

    assert (
        evidence["exception_type"]
        == "RuntimeError"
    )

    assert (
        evidence["root_exception_type"]
        == "LinAlgError"
    )


def test_parallel_poly_surface_process_pool_parity(
    tmp_path,
):
    """Exercise the real ProcessPool chunk path and compare all optical outputs with serial."""
    import execution_v55

    from execution_v55 import (
        ExecutionRuntime,
        ray_trace_acceleration,
    )

    terms = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
    ]

    coeff = np.array([
        0.0,
        0.01,
        -0.01,
        0.002,
        0.003,
        -0.002,
        0.0001,
        -0.0001,
        0.00005,
    ])

    surf = PolySurface(
        "M_TEST_POOL",
        np.zeros(3),
        np.eye(3),
        np.array([50.0, 50.0]),
        terms,
        coeff,
        np.array([50.0, 50.0]),
        0.002,
        -1.0,
    )

    rng = np.random.default_rng(123)

    n_rays = 257

    origins = np.zeros(
        (n_rays, 3)
    )

    origins[:, 0] = rng.uniform(
        -40.0,
        40.0,
        n_rays,
    )

    origins[:, 1] = rng.uniform(
        -40.0,
        40.0,
        n_rays,
    )

    origins[:, 2] = 50.0

    directions = np.tile(
        np.array([
            0.0,
            0.0,
            -1.0,
        ]),
        (n_rays, 1),
    )

    serial = surf.intersect(
        origins,
        directions,
        finite=True,
    )

    runtime = ExecutionRuntime(
        mode="accelerated",
        config={
            "parallel_ray_trace": True,
            "ray_trace_min_rays": 1,
            "ray_trace_chunk_size": 64,
            "cpu_workers": 2,
            "max_inflight_cpu_jobs": 2,
            "blas_threads_per_worker": 1,
            "host_memory_budget_gib": 64.0,
            "host_available_reserve_gib": 0.1,
            "worker_memory_estimate_gib": 0.1,
        },
        run_dir=tmp_path,
        backend_plan={},
        session_id="TEST_PARALLEL_POOL",
    )

    token = (
        execution_v55
        ._CURRENT_RUNTIME
        .set(runtime)
    )

    try:
        with ray_trace_acceleration(
            "TEST_PARALLEL_POOL"
        ):
            parallel = surf.intersect(
                origins,
                directions,
                finite=True,
            )
    finally:
        execution_v55._CURRENT_RUNTIME.reset(
            token
        )

    for key in (
        "t",
        "point",
        "normal",
        "x",
        "y",
        "residual",
        "raw_conic_argument",
    ):
        assert np.allclose(
            serial[key],
            parallel[key],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ), key

    for key in (
        "valid",
        "status",
        "resolved",
        "candidate_count",
        "outside_aperture_candidate_count",
        "conic_invalid_candidate_count",
    ):
        assert np.array_equal(
            serial[key],
            parallel[key],
        ), key


def test_reconstruct_m2_input_gate_rejects_nonfinite_geometry_before_dynamic_refs(
    monkeypatch,
):
    """B16 gate must reject NaN M1/M2 geometry before dynamic Fan refs or M2 fit."""
    import pipeline_v55 as pipeline
    from types import SimpleNamespace

    class DummySurface:
        """Thực thi DummySurface."""
        def copy(self):
            """Thực thi copy."""
            return self

    n = 3

    finite = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
    ])

    tr0 = {
        "points": [
            finite.copy(),
            finite.copy(),
            finite.copy(),
            finite.copy(),
        ],
        "landing": finite.copy(),
        "valid": np.ones(n, bool),
    }

    bad_p2 = finite.copy()
    bad_p2[1, 0] = np.nan

    tr1 = {
        "points": [
            finite.copy(),
            finite.copy(),
            bad_p2,
            finite.copy(),
        ],
        "landing": finite.copy(),
        "valid": np.array([
            True,
            False,
            True,
        ]),
    }

    traces = iter([
        tr0,
        tr1,
    ])

    monkeypatch.setattr(
        pipeline,
        "trace_reverse",
        lambda *args, **kwargs: next(traces),
    )

    monkeypatch.setattr(
        pipeline,
        "_live_model_preview",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        pipeline,
        "fit_surface",
        lambda *args, **kwargs: (
            DummySurface(),
            {"sag_rms_mm": 0.0},
        ),
    )

    def dynamic_refs_must_not_run(
        *args,
        **kwargs,
    ):
        """Thực thi dynamic_refs_must_not_run."""
        raise AssertionError(
            "DYNAMIC_REFS_SHOULD_NOT_RUN_"
            "AFTER_BAD_M2_GEOMETRY"
        )

    monkeypatch.setattr(
        pipeline,
        "_dynamic_refs",
        dynamic_refs_must_not_run,
    )

    rays = {
        "rows": [
            {"ray_id": f"R{i}"}
            for i in range(n)
        ],
        "field_index": np.zeros(
            n,
            int,
        ),
        "pupil_index": np.zeros(
            n,
            int,
        ),
        "chief": np.array([
            True,
            False,
            False,
        ]),
        "central_field_index": 0,
        "central_pupil_index": 0,
    }

    ctx = SimpleNamespace(
        data={
            "rays": rays,
            "visor": object(),
            "display": DummySurface(),
            "m1": DummySurface(),
            "m2": DummySurface(),
            "fermat": {
                "target_points": finite.copy(),
            },
            "rho": 0.25,
        },
        config={
            "surface_fit": {},
            "fit_weights": {},
        },
    )

    evidence = {
        "phase": "START"
    }

    with pytest.raises(
        RuntimeError,
        match=(
            r"ORDER_2_"
            r"M2_CONSTRUCTION_INPUT_"
            r"INCOMPLETE_2_OF_3"
        ),
    ):
        pipeline._reconstruct_impl(
            ctx,
            2,
            True,
            allow_restoration=True,
            evidence=evidence,
        )

    assert (
        evidence["phase"]
        == "M2_CONSTRUCTION_INPUT_GATE"
    )

    assert (
        evidence[
            "m2_construction_input_gate"
        ][
            "uses_cumulative_trace_valid"
        ]
        is False
    )


def test_reconstruct_m2_input_gate_does_not_reject_display_only_invalid(
    monkeypatch,
):
    """Cumulative tr1.valid=false alone must not reject finite M1/M2 construction geometry."""
    import pipeline_v55 as pipeline
    from types import SimpleNamespace

    class DummySurface:
        """Thực thi DummySurface."""
        def copy(self):
            """Thực thi copy."""
            return self

    class ReachedDynamicRefs(Exception):
        """Thực thi ReachedDynamicRefs."""
        pass

    n = 3

    finite = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
    ])

    tr0 = {
        "points": [
            finite.copy(),
            finite.copy(),
            finite.copy(),
            finite.copy(),
        ],
        "landing": finite.copy(),
        "valid": np.ones(n, bool),
    }

    tr1 = {
        "points": [
            finite.copy(),
            finite.copy(),
            finite.copy(),
            finite.copy(),
        ],
        "landing": finite.copy(),

        # Mô phỏng failure downstream DISPLAY.
        # M1/M2 geometry vẫn finite.
        "valid": np.zeros(n, bool),
    }

    traces = iter([
        tr0,
        tr1,
    ])

    monkeypatch.setattr(
        pipeline,
        "trace_reverse",
        lambda *args, **kwargs: next(traces),
    )

    monkeypatch.setattr(
        pipeline,
        "_live_model_preview",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        pipeline,
        "fit_surface",
        lambda *args, **kwargs: (
            DummySurface(),
            {"sag_rms_mm": 0.0},
        ),
    )

    def reached_dynamic_refs(
        *args,
        **kwargs,
    ):
        """Thực thi reached_dynamic_refs."""
        raise ReachedDynamicRefs

    monkeypatch.setattr(
        pipeline,
        "_dynamic_refs",
        reached_dynamic_refs,
    )

    rays = {
        "rows": [
            {"ray_id": f"R{i}"}
            for i in range(n)
        ],
        "field_index": np.zeros(
            n,
            int,
        ),
        "pupil_index": np.zeros(
            n,
            int,
        ),
        "chief": np.array([
            True,
            False,
            False,
        ]),
        "central_field_index": 0,
        "central_pupil_index": 0,
    }

    ctx = SimpleNamespace(
        data={
            "rays": rays,
            "visor": object(),
            "display": DummySurface(),
            "m1": DummySurface(),
            "m2": DummySurface(),
            "fermat": {
                "target_points": finite.copy(),
            },
            "rho": 0.25,
        },
        config={
            "surface_fit": {},
            "fit_weights": {},
        },
    )

    evidence = {
        "phase": "START"
    }

    with pytest.raises(
        ReachedDynamicRefs
    ):
        pipeline._reconstruct_impl(
            ctx,
            2,
            True,
            allow_restoration=True,
            evidence=evidence,
        )

    assert (
        evidence[
            "m2_construction_input_gate"
        ][
            "valid_geometry_count"
        ]
        == n
    )


def test_step24_uses_reverse_sources_rule_from_forward_evaluation(tmp_path, monkeypatch):
    """Regression test: xác nhận step_24 lấy reverse, sources, reference_rule từ _run_forward_evaluation mà không ném NameError."""
    import pipeline_v55 as pipe
    from core import PolySurface

    n_rays = 2
    r_rows = [{"ray_id": i} for i in range(n_rays)]
    rays = {
        "rows": r_rows,
        "field_count": 1,
        "pupil_count": 1,
        "field_index": np.zeros(n_rays, dtype=int),
        "pupil_index": np.zeros(n_rays, dtype=int),
        "samples_per_pupil": 2,
    }

    dummy_m1 = PolySurface.plane("M1", np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 1.0]), (10.0, 10.0))
    dummy_m2 = PolySurface.plane("M2", np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0]), (10.0, 10.0))
    dummy_disp = PolySurface.plane("DISPLAY", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (10.0, 10.0))

    class DummyVisor:
        """Thực thi DummyVisor."""
        center = np.array([0.0, 0.0, 30.0])
        frame = np.eye(3)

    ctx = pipe.Context(
        root=tmp_path,
        data={
            "rays": rays,
            "m1": dummy_m1,
            "m2": dummy_m2,
            "display": dummy_disp,
            "visor": DummyVisor(),
        },
        config={
            "visor_active_mm": [20.0, 20.0],
            "packaging_lambda_max": 1.0,
            "mtf_requested": False,
        },
    )

    fake_eval = {
        "reverse": {
            "display_local": np.zeros((n_rays, 3)),
            "points": [np.zeros((n_rays, 3)), np.zeros((n_rays, 3)), np.zeros((n_rays, 3))],
        },
        "sources": np.zeros((1, 3)),
        "reference_rule": "CHIEF_RAY_AT_DISPLAY",
        "forward": {
            "converged": np.ones(n_rays, dtype=bool),
            "physical_valid": np.ones(n_rays, dtype=bool),
            "valid": np.ones(n_rays, dtype=bool),
            "visor": np.zeros((n_rays, 3)),
            "m1": np.zeros((n_rays, 3)),
            "m2": np.zeros((n_rays, 3)),
            "residual_mm": np.zeros(n_rays),
            "iterations": np.ones(n_rays, dtype=int),
        },
        "virtual": {
            "points": np.zeros((n_rays, 3)),
            "bundle_valid": np.ones(1, dtype=bool),
            "rows": [],
        },
        "distortion": {
            "rows": [],
        },
        "first_hit_m2": {"name": np.array(["M2"] * n_rays)},
        "first_hit_m1": {"name": np.array(["M1"] * n_rays)},
        "first_hit_visor": {"name": np.array(["VISOR"] * n_rays)},
        "first_order": np.ones(n_rays, dtype=bool),
    }

    monkeypatch.setattr(pipe, "_run_forward_evaluation", lambda *args, **kwargs: fake_eval)
    monkeypatch.setattr(pipe, "_publish_trace_debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipe, "_final_reverse_trace_fingerprint", lambda *args, **kwargs: "dummy_fp")
    monkeypatch.setattr(pipe, "_actual_optics", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipe, "_packaging", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(pipe, "write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipe, "write_json", lambda *args, **kwargs: None)

    # step_24 must not raise NameError for reverse, sources, rule
    res = pipe.step_24(ctx)
    assert ctx.data["fan_reference_rule"] == "CHIEF_RAY_AT_DISPLAY"
    assert ctx.data["final_reverse_trace"] is fake_eval["reverse"]
    assert ctx.data["fan_refs"] is fake_eval["sources"]


def test_step7_auto_geometry_is_deterministic():
    """STEP 07 phải sinh cùng geometry khi config và sequence_seed không đổi."""
    search = {
        "mode": "AUTO_CHIEF_RAY_FOLD_LATIN_HYPERCUBE",
        "candidate_count": 32,
        "sequence_seed": 55,
        "m1_distance_scale_bounds": [0.75, 1.75],
        "m1_to_m2_distance_scale_bounds": [0.5, 1.5],
        "m1_to_m2_turn_deg_bounds": [60.0, 120.0],
        "m2_to_display_distance_scale_bounds": [0.25, 1.25],
        "m2_to_display_turn_deg_bounds": [-130.0, -60.0],
    }

    config = {
        "display_seed_mm": [30.0, 18.0],
        "visor_active_mm": [66.0, 42.0],
        "eyebox_y_mm": [-8.0, 8.0],
        "eyebox_z_mm": [-5.0, 5.0],
        "pupil_diameter_mm": 7.0,
        "solver": {
            "step7_geometry_search": search,
        },
    }

    data = {
        "rays": {
            "central_field_index": 0,
            "central_pupil_index": 0,
            "field_index": np.asarray([0], dtype=int),
            "pupil_index": np.asarray([0], dtype=int),
            "chief": np.asarray([True], dtype=bool),
        },
        "visor_hit": {
            "point": np.asarray(
                [[60.0, 0.0, 10.0]],
                dtype=float,
            ),
        },
        "post_visor": np.asarray(
            [[-0.8, 0.0, 0.6]],
            dtype=float,
        ),
        "inputs": {
            "visor_U": np.asarray([0.0, 1.0, 0.0], dtype=float),
            "visor_V": np.asarray([0.2, 0.0, 0.98], dtype=float),
            "visor_N": np.asarray([0.98, 0.0, -0.2], dtype=float),
            "packaging_constraint_enabled": False,
            "packaging_vertices": None,
        },
    }

    first = _step7_auto_seed_descriptors(
        Context(
            config=copy.deepcopy(config),
            data=copy.deepcopy(data),
        )
    )
    second = _step7_auto_seed_descriptors(
        Context(
            config=copy.deepcopy(config),
            data=copy.deepcopy(data),
        )
    )

    assert len(first) == 32
    assert len(second) == 32

    for left, right in zip(first, second):
        assert left["candidate_index"] == right["candidate_index"]
        assert left["d"] == pytest.approx(right["d"])
        assert np.allclose(left["c2"], right["c2"])
        assert np.allclose(left["cd"], right["cd"])
        assert (
            left["geometry_search_parameters"]
            == right["geometry_search_parameters"]
        )


def test_step7_packaging_switch_uses_lambda_limit_only_when_enabled():
    """Packaging OFF bỏ gate; ON dùng inclusive lambda <= limit."""
    ctx = Context(
        config={
            "packaging_lambda_max": 1.2,
        },
        data={
            "inputs": {
                "packaging_constraint_enabled": False,
            },
        },
    )

    assert _step7_packaging_pass(ctx, None)

    ctx.data["inputs"]["packaging_constraint_enabled"] = True

    assert _step7_packaging_pass(ctx, 1.0)
    assert _step7_packaging_pass(ctx, 1.2)
    assert not _step7_packaging_pass(ctx, 1.2000001)
    assert not _step7_packaging_pass(ctx, None)


def test_step7_compactness_precedes_planar_rms():
    """Geometry nhỏ hơn phải thắng dù planar RMS diagnostic lớn hơn."""
    def candidate(
        volume: float,
        diagonal: float,
        optic_diagonal: float,
        area: float,
        path: float,
        rms: float,
    ) -> dict[str, Any]:
        """Tạo cấu trúc candidate giả lập phục vụ kiểm thử độ xếp hạng compact."""
        return {
            "eligible": True,
            "geometry_metrics": {
                "geometry_bbox_volume_mm3": volume,
                "geometry_bbox_diagonal_mm": diagonal,
                "max_clear_aperture_diagonal_mm": optic_diagonal,
                "total_clear_aperture_area_mm2": area,
                "chief_path_length_mm": path,
            },
            "packaging_lambda": None,
            "planar_paper_spot": {
                "RMS_spot_radius_mm": rms,
            },
        }

    compact = candidate(
        100000.0,
        100.0,
        40.0,
        2000.0,
        120.0,
        9.0,
    )
    low_rms_but_larger = candidate(
        110000.0,
        90.0,
        30.0,
        1500.0,
        100.0,
        0.01,
    )

    assert (
        _planar_seed_rank_key(compact)
        <
        _planar_seed_rank_key(low_rms_but_larger)
    )


def test_multistart_deep_runs_only_step7_rank_one_geometry():
    """Nhiều STEP7 geometry có thể eligible nhưng deep-run schedule phải chỉ có rank #1."""
    def candidate(
        volume: float,
        rms: float,
        eligible: bool = True,
    ) -> dict[str, Any]:
        """Tạo cấu trúc candidate giả lập phục vụ kiểm thử lọc duy nhất rank một."""
        return {
            "eligible": eligible,
            "geometry_metrics": {
                "geometry_bbox_volume_mm3": volume,
                "geometry_bbox_diagonal_mm": 100.0,
                "max_clear_aperture_diagonal_mm": 40.0,
                "total_clear_aperture_area_mm2": 2000.0,
                "chief_path_length_mm": 120.0,
            },
            "packaging_lambda": None,
            "planar_paper_spot": {
                "RMS_spot_radius_mm": rms,
            },
        }

    candidates = [
        candidate(
            volume=120000.0,
            rms=0.1,
        ),
        candidate(
            volume=90000.0,
            rms=8.0,
        ),
        candidate(
            volume=100000.0,
            rms=1.0,
        ),
        candidate(
            volume=50000.0,
            rms=0.01,
            eligible=False,
        ),
    ]

    eligible, scheduled = _select_step7_deep_run_schedule(
        candidates
    )

    assert [
        candidate_index
        for candidate_index, seed in eligible
    ] == [
        2,
        3,
        1,
    ]

    assert len(scheduled) == 1
    assert scheduled[0][0] == 2


def test_step12_o2_refinement_preserves_order2_gauge():
    """STEP12 refinement không được mở A00 hoặc phá A20+A02 gauge của fit_surface."""
    terms = [
        (i, j)
        for i in range(3)
        for j in range(3)
    ]

    m1 = PolySurface(
        "M1",
        np.zeros(3),
        np.eye(3),
        (20.0, 20.0),
        terms,
        np.zeros(len(terms)),
        np.array([20.0, 20.0]),
        0.01,
        0.0,
        False,
    )

    m2 = PolySurface(
        "M2",
        np.array([0.0, 0.0, 40.0]),
        np.eye(3),
        (20.0, 20.0),
        terms,
        np.zeros(len(terms)),
        np.array([20.0, 20.0]),
        0.02,
        1.0,
        False,
    )

    config = json.loads(
        (
            Path(__file__).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    descriptors = (
        _step12_o2_variable_descriptors(
            m1,
            m2,
        )
    )

    base = {
        "M1":
            m1.copy(),

        "M2":
            m2.copy(),
    }

    u = np.full(
        len(descriptors),
        0.5,
        dtype=float,
    )

    trial_m1, trial_m2 = (
        _step12_apply_o2_vector(
            base,
            descriptors,
            u,
            config[
                "surface_fit"
            ][
                "step12_o2_refinement"
            ],
            config[
                "surface_fit"
            ],
        )
    )

    for original, trial in (
        (m1, trial_m1),
        (m2, trial_m2),
    ):
        index_00 = trial.terms.index(
            (0, 0)
        )

        index_20 = trial.terms.index(
            (2, 0)
        )

        index_02 = trial.terms.index(
            (0, 2)
        )

        assert (
            trial.coeff[
                index_00
            ]
            ==
            original.coeff[
                index_00
            ]
        )

        assert abs(
            (
                trial.coeff[
                    index_20
                ]
                +
                trial.coeff[
                    index_02
                ]
            )
            -
            (
                original.coeff[
                    index_20
                ]
                +
                original.coeff[
                    index_02
                ]
            )
        ) < 1e-12


def test_step12_actual_o2_gate_uses_surface_metrics_not_cloud_metrics():
    """STEP12 final gate phải phụ thuộc physical + actual target direction, không nhận CI diagnostic."""
    quality = {
        "enforcement":
            "HARD",

        "minimum_physical_fraction":
            0.75,

        "maximum_sag_rms_mm":
            1.0,

        "maximum_normal_rms_deg":
            15.0,

        "maximum_normal_max_deg":
            45.0,

        "maximum_reflection_rms_deg":
            30.0,

        "reject_k_bound_hit":
            False,
    }

    passing = {
        "physical_fraction":
            1.0,

        "display_target_rms_mm":
            0.5,

        "display_target_p95_mm":
            0.8,

        "target_direction_rms_deg":
            5.0,

        "target_direction_p95_deg":
            9.0,
    }

    gate = _step12_actual_o2_gate(
        passing,
        quality,
    )

    assert gate[
        "authority"
    ] == (
        "ACTUAL_O2_SURFACE_RAY_TRACE"
    )

    assert gate[
        "status"
    ] == "PASS"

    failing = copy.deepcopy(
        passing
    )

    failing[
        "target_direction_rms_deg"
    ] = 31.0

    failed_gate = (
        _step12_actual_o2_gate(
            failing,
            quality,
        )
    )

    assert failed_gate[
        "status"
    ] == "FAIL"


def test_step12_o2_refinement_config_validation():
    """Config STEP12 refinement phải fail-fast nếu bound hoặc schedule sai."""
    config = json.loads(
        (
            Path(__file__).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    validate_config(
        config
    )

    hard_topology = copy.deepcopy(
        config
    )
    hard_topology[
        "surface_fit"
    ][
        "mirror_topology_authority"
    ][
        "enforcement"
    ] = {
        "M1": "HARD",
        "M2": "HARD",
    }
    validate_config(
        hard_topology
    )

    bad_topology_enforcement = copy.deepcopy(
        config
    )
    bad_topology_enforcement[
        "surface_fit"
    ][
        "mirror_topology_authority"
    ][
        "enforcement"
    ][
        "M2"
    ] = "BYPASS"
    with pytest.raises(
        ValueError,
        match="MIRROR_TOPOLOGY_AUTHORITY_ENFORCEMENT_INVALID",
    ):
        validate_config(
            bad_topology_enforcement
        )

    missing_topology_enforcement = copy.deepcopy(
        config
    )
    del missing_topology_enforcement[
        "surface_fit"
    ][
        "mirror_topology_authority"
    ][
        "enforcement"
    ][
        "M2"
    ]
    with pytest.raises(
        ValueError,
        match="MIRROR_TOPOLOGY_AUTHORITY_ENFORCEMENT_INVALID",
    ):
        validate_config(
            missing_topology_enforcement
        )

    bad = copy.deepcopy(
        config
    )

    bad[
        "surface_fit"
    ][
        "step12_o2_refinement"
    ][
        "max_step_normalized"
    ] = 1.5

    with pytest.raises(
        ValueError,
        match=
            "STEP12_O2_REFINEMENT_"
            "MAX_STEP_INVALID",
    ):
        validate_config(
            bad
        )

    bad_topology = copy.deepcopy(
        config
    )
    bad_topology[
        "surface_fit"
    ][
        "mirror_topology_authority"
    ][
        "M1"
    ] = "CONVEX_RAY_FACING"

    with pytest.raises(
        ValueError,
        match=
            "MIRROR_TOPOLOGY_"
            "AUTHORITY_MISMATCH",
    ):
        validate_config(
            bad_topology
        )

    bad_surface_refinement = copy.deepcopy(
        config
    )
    bad_surface_refinement[
        "surface_parameter_refinement"
    ][
        "curvature_relative_bound"
    ] = 1.0

    with pytest.raises(
        ValueError,
        match=
            "SURFACE_PARAMETER_REFINEMENT_"
            "CURVATURE_BOUND_INVALID",
    ):
        validate_config(
            bad_surface_refinement
        )


def test_step11_topology_warn_admission_bypasses_only_curvature_failures():
    """WARN admits KG/H failures in STEP11 but never bypasses basic sanity."""
    from pipeline_v55 import _step11_topology_admission

    curvature_failure = {
        "topology_pass": False,
        "topology_checks": {
            "basic_surface_sanity": True,
            "single_bowl_gaussian_curvature": True,
            "single_bowl_mean_curvature_orientation": False,
        },
        "topology_failure_reasons": [
            "single_bowl_mean_curvature_orientation",
        ],
    }
    invalid_surface = {
        "topology_pass": False,
        "topology_checks": {
            "basic_surface_sanity": False,
            "single_bowl_gaussian_curvature": False,
            "single_bowl_mean_curvature_orientation": False,
        },
        "topology_failure_reasons": [
            "basic_surface_sanity",
        ],
    }

    hard = _step11_topology_admission(
        curvature_failure,
        "HARD",
    )
    warn = _step11_topology_admission(
        curvature_failure,
        "WARN",
    )
    invalid_warn = _step11_topology_admission(
        invalid_surface,
        "WARN",
    )

    assert hard["status"] == "FAIL"
    assert hard["admitted"] is False
    assert warn["status"] == "WARN_ADMITTED"
    assert warn["admitted"] is True
    assert warn["warning_reasons"] == [
        "single_bowl_mean_curvature_orientation",
    ]
    assert invalid_warn["status"] == "FAIL"
    assert invalid_warn["admitted"] is False

    with pytest.raises(
        ValueError,
        match="STEP11_TOPOLOGY_ENFORCEMENT_INVALID",
    ):
        _step11_topology_admission(
            curvature_failure,
            "BYPASS",
        )


def test_mirror_shape_gate_gaussian_curvature_tolerance_keeps_orientation_strict():
    """KG near zero may pass, while real saddle and wrong orientation must fail."""
    from core import mirror_shape_gate

    policy = {
        "maximum_freeform_departure_mm": 1.0,
        "maximum_normal_departure_deg": 45.0,
        "maximum_principal_curvature_per_mm": 0.2,
    }
    terms = [
        (2, 0),
        (0, 2),
    ]

    tolerated_surface = PolySurface(
        "TOLERATED_KG",
        np.zeros(3),
        np.eye(3),
        np.array([5.0, 5.0]),
        terms,
        np.array([0.01, -0.00078125]),
        np.array([5.0, 5.0]),
    )
    rejected_surface = PolySurface(
        "REJECTED_KG",
        np.zeros(3),
        np.eye(3),
        np.array([5.0, 5.0]),
        terms,
        np.array([0.01, -0.003125]),
        np.array([5.0, 5.0]),
    )

    tolerated_gate = mirror_shape_gate(
        tolerated_surface,
        policy,
        orientation_sign=1.0,
    )
    rejected_gate = mirror_shape_gate(
        rejected_surface,
        policy,
        orientation_sign=1.0,
    )
    wrong_orientation_gate = mirror_shape_gate(
        tolerated_surface,
        policy,
        orientation_sign=-1.0,
    )

    assert -1e-7 <= tolerated_gate["KG_min_per_mm2"] < 0.0
    assert tolerated_gate["oriented_H_min_per_mm"] > 0.0
    assert (
        tolerated_gate["checks"]["single_bowl_gaussian_curvature"]
        is True
    )
    assert (
        tolerated_gate["checks"]["single_bowl_mean_curvature_orientation"]
        is True
    )
    assert tolerated_gate["pass"] is True
    assert (
        "single_bowl_gaussian_curvature"
        not in tolerated_gate["failure_reasons"]
    )

    # Raw diagnostic must still report mathematically negative KG samples.
    assert (
        tolerated_gate["curvature_sign_flip_count"]
        == tolerated_gate["sample_count"]
    )

    assert rejected_gate["KG_min_per_mm2"] < -1e-7
    assert rejected_gate["oriented_H_min_per_mm"] > 0.0
    assert (
        rejected_gate["checks"]["single_bowl_gaussian_curvature"]
        is False
    )
    assert (
        rejected_gate["checks"]["single_bowl_mean_curvature_orientation"]
        is True
    )
    assert rejected_gate["pass"] is False
    assert (
        "single_bowl_gaussian_curvature"
        in rejected_gate["failure_reasons"]
    )

    assert (
        wrong_orientation_gate["checks"]["single_bowl_gaussian_curvature"]
        is True
    )
    assert (
        wrong_orientation_gate["checks"][
            "single_bowl_mean_curvature_orientation"
        ]
        is False
    )
    assert wrong_orientation_gate["pass"] is False


def test_ray_facing_mirror_topology_gate_is_frame_invariant():
    """Ray-facing topology phải giữ nghĩa vật lý khi local frame bị đảo."""
    config = json.loads(
        (
            Path(__file__).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    rays = {
        "central_field_index":
            0,
        "central_pupil_index":
            0,
        "field_index":
            np.array([0], int),
        "pupil_index":
            np.array([0], int),
        "chief":
            np.array([True]),
    }

    ctx = SimpleNamespace(
        config=config,
        data={
            "rays":
                rays,
        },
    )

    terms = [
        (2, 0),
        (0, 2),
    ]

    concave = PolySurface(
        "M1",
        np.zeros(3),
        np.eye(3),
        np.array([5.0, 5.0]),
        terms,
        np.array([0.02, 0.02]),
        np.array([5.0, 5.0]),
    )

    convex = PolySurface(
        "M2",
        np.zeros(3),
        np.eye(3),
        np.array([5.0, 5.0]),
        terms,
        np.array([-0.02, -0.02]),
        np.array([5.0, 5.0]),
    )

    wrong_m1 = PolySurface(
        "M1",
        np.zeros(3),
        np.eye(3),
        np.array([5.0, 5.0]),
        terms,
        np.array([-0.02, -0.02]),
        np.array([5.0, 5.0]),
    )

    flipped_concave = PolySurface(
        "M1",
        np.zeros(3),
        np.diag([1.0, -1.0, -1.0]),
        np.array([5.0, 5.0]),
        terms,
        np.array([-0.02, -0.02]),
        np.array([5.0, 5.0]),
    )

    zero_points = np.array([
        [0.0, 0.0, 0.0],
    ])
    incoming = np.array([
        [0.0, 0.0, -1.0],
    ])

    trace = {
        "points": [
            zero_points.copy(),
            zero_points.copy(),
            zero_points.copy(),
            zero_points.copy(),
        ],
        "directions": [
            incoming.copy(),
            incoming.copy(),
            incoming.copy(),
            incoming.copy(),
        ],
    }

    concave_gate = _mirror_topology_gate(
        ctx,
        concave,
        trace,
        "M1",
    )
    convex_gate = _mirror_topology_gate(
        ctx,
        convex,
        trace,
        "M2",
    )
    wrong_gate = _mirror_topology_gate(
        ctx,
        wrong_m1,
        trace,
        "M1",
    )
    flipped_gate = _mirror_topology_gate(
        ctx,
        flipped_concave,
        trace,
        "M1",
    )

    assert concave_gate["pass"] is True
    assert concave_gate[
        "topology_authority"
    ] == "CONCAVE_RAY_FACING"
    assert concave_gate[
        "topology_enforcement"
    ] == "HARD"
    assert concave_gate[
        "required_orientation_sign"
    ] == 1.0

    assert convex_gate["pass"] is True
    assert convex_gate[
        "topology_authority"
    ] == "CONVEX_RAY_FACING"
    assert convex_gate[
        "topology_enforcement"
    ] == "WARN"
    assert convex_gate[
        "required_orientation_sign"
    ] == -1.0

    assert wrong_gate["pass"] is False
    assert (
        "single_bowl_mean_curvature_orientation"
        in wrong_gate["failure_reasons"]
    )

    assert flipped_gate["pass"] is True
    assert flipped_gate[
        "required_orientation_sign"
    ] == -1.0


def test_step11_shape_curvature_record_fields():
    """Kiểm tra helper trích xuất đúng 10 curvature scalar cho M1 và M2, bảo toàn độ chính xác và dấu âm."""
    from pipeline_v55 import (
        _step11_shape_curvature_record_fields,
        _step11_curvature_history_fields_from_record,
        _STEP11_SHAPE_CURVATURE_FLOAT_KEYS,
        _STEP11_SHAPE_CURVATURE_INT_KEYS,
    )

    gate = {
        "H_min_per_mm": -0.004,
        "H_max_per_mm": -0.001,
        "KG_min_per_mm2": -1e-12,
        "KG_max_per_mm2": 2.5e-5,
        "k1_min_per_mm": -0.006,
        "k1_max_per_mm": -0.002,
        "k2_min_per_mm": -0.005,
        "k2_max_per_mm": 0.001,
        "oriented_H_min_per_mm": -1e-8,
        "curvature_sign_flip_count": 17,
    }
    gate_copy = copy.deepcopy(gate)

    # 1. Test M2 prefix
    m2_fields = _step11_shape_curvature_record_fields("M2", gate)
    assert len(m2_fields) == 10
    assert m2_fields["M2_H_min_per_mm"] == -0.004
    assert m2_fields["M2_H_max_per_mm"] == -0.001
    assert m2_fields["M2_KG_min_per_mm2"] == -1e-12
    assert m2_fields["M2_KG_max_per_mm2"] == 2.5e-5
    assert m2_fields["M2_k1_min_per_mm"] == -0.006
    assert m2_fields["M2_k1_max_per_mm"] == -0.002
    assert m2_fields["M2_k2_min_per_mm"] == -0.005
    assert m2_fields["M2_k2_max_per_mm"] == 0.001
    assert m2_fields["M2_oriented_H_min_per_mm"] == -1e-8
    assert m2_fields["M2_curvature_sign_flip_count"] == 17
    assert isinstance(m2_fields["M2_curvature_sign_flip_count"], int)

    # 2. Gate input không bị mutate
    assert gate == gate_copy

    # 3. Test M1 prefix
    m1_fields = _step11_shape_curvature_record_fields("M1", gate)
    assert len(m1_fields) == 10
    assert m1_fields["M1_KG_min_per_mm2"] == -1e-12
    assert m1_fields["M1_curvature_sign_flip_count"] == 17

    # 4. Invalid surface name raise ValueError
    with pytest.raises(ValueError, match="STEP11_CURVATURE_DIAGNOSTIC_SURFACE_INVALID"):
        _step11_shape_curvature_record_fields("M3", gate)

    # 5. Missing key trong gate raise KeyError rõ ràng
    bad_gate = copy.deepcopy(gate)
    del bad_gate["KG_min_per_mm2"]
    with pytest.raises(KeyError):
        _step11_shape_curvature_record_fields("M1", bad_gate)


def test_step11_search_history_curvature_export_and_parsing(tmp_path: Path):
    """Kiểm tra export 20 trường curvature ra CSV và đọc lại bảo toàn số âm, ô trống, NaN."""
    import csv
    from pipeline_v55 import _step11_curvature_history_fields_from_record
    from render_step11_diagnostic_v55 import parse_step11_search_history

    record1 = {
        "candidate_id": "CANDIDATE_0001",
        "cycle": 1,
        "trust_scale": 1.0,
        "move": "M1_c_+",
        "feasible": False,
        "rejection_stage": "M2_MIRROR_TOPOLOGY_GATE",
        "rejection_reason": "single_bowl_gaussian_curvature",
        "M1_shape_pass": True,
        "M1_ci_trust_pass": True,
        "M1_H_min_per_mm": -0.0035,
        "M1_H_max_per_mm": -0.0034,
        "M1_KG_min_per_mm2": 1.22e-5,
        "M1_KG_max_per_mm2": 1.25e-5,
        "M1_k1_min_per_mm": -0.0036,
        "M1_k1_max_per_mm": -0.0034,
        "M1_k2_min_per_mm": -0.0036,
        "M1_k2_max_per_mm": -0.0034,
        "M1_oriented_H_min_per_mm": 0.0035,
        "M1_curvature_sign_flip_count": 0,
        "M2_shape_pass": False,
        "M2_H_min_per_mm": -0.002,
        "M2_H_max_per_mm": 0.001,
        "M2_KG_min_per_mm2": -1e-12,
        "M2_KG_max_per_mm2": 5e-6,
        "M2_k1_min_per_mm": -0.004,
        "M2_k1_max_per_mm": 0.002,
        "M2_k2_min_per_mm": -0.003,
        "M2_k2_max_per_mm": 0.001,
        "M2_oriented_H_min_per_mm": -1e-9,
        "M2_curvature_sign_flip_count": 5,
    }

    # Record 2 bị loại sớm ở M1 retrace, M2 hoàn toàn trống (None)
    record2 = {
        "candidate_id": "CANDIDATE_0002",
        "cycle": 1,
        "trust_scale": 1.0,
        "move": "M1_c_-",
        "feasible": False,
        "rejection_stage": "M1_RETRACE_RAYS_INVALID",
        "rejection_reason": "invalid_count>0",
        "M1_shape_pass": True,
        "M1_ci_trust_pass": True,
        "M1_H_min_per_mm": -0.0035,
        "M1_H_max_per_mm": -0.0034,
        "M1_KG_min_per_mm2": 1.22e-5,
        "M1_KG_max_per_mm2": 1.25e-5,
        "M1_k1_min_per_mm": -0.0036,
        "M1_k1_max_per_mm": -0.0034,
        "M1_k2_min_per_mm": -0.0036,
        "M1_k2_max_per_mm": -0.0034,
        "M1_oriented_H_min_per_mm": 0.0035,
        "M1_curvature_sign_flip_count": 0,
        "M2_shape_pass": None,
    }

    csv_file = tmp_path / "11_O2_SEARCH_HISTORY.csv"
    fieldnames = list(record1.keys())

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(record1)
        writer.writerow(record2)

    parsed = parse_step11_search_history(csv_file)
    assert len(parsed) == 2

    # Row 1: M2 đầy đủ giá trị, giữ nguyên dấu âm và -1e-12 không bị đổi thành 0
    r1 = parsed[0]
    assert r1["M2_KG_min_per_mm2"] == -1e-12
    assert r1["M2_oriented_H_min_per_mm"] == -1e-9
    assert r1["M2_curvature_sign_flip_count"] == 5
    assert isinstance(r1["M2_curvature_sign_flip_count"], int)
    assert r1["M1_KG_min_per_mm2"] == 1.22e-5

    # Row 2: M2 chưa chạy gate -> None (không phải 0 hay 0.0)
    r2 = parsed[1]
    assert r2["M2_KG_min_per_mm2"] is None
    assert r2["M2_oriented_H_min_per_mm"] is None
    assert r2["M2_curvature_sign_flip_count"] is None
    assert r2["M1_KG_min_per_mm2"] == 1.22e-5


def test_step11_renderer_backward_compatibility_old_csv(tmp_path: Path):
    """Kiểm tra parse_step11_search_history với CSV cũ không có 20 cột curvature mới."""
    import csv
    from render_step11_diagnostic_v55 import parse_step11_search_history

    old_row = {
        "candidate_id": "CANDIDATE_0001",
        "cycle": "1",
        "trust_scale": "1.0",
        "move": "BASELINE",
        "feasible": "False",
        "rejection_stage": "M2_MIRROR_TOPOLOGY_GATE",
        "rejection_reason": "single_bowl_gaussian_curvature",
        "M1_shape_pass": "True",
        "M1_ci_trust_pass": "True",
        "M2_shape_pass": "False",
        "physical_valid_count": "",
        "physical_fraction": "",
        "optical_objective": "",
        "mapping_rms_mm": "",
        "spot_rms_mm": "",
        "direction_rms_deg": "",
        "representation_score": "",
    }

    old_csv = tmp_path / "old_11_O2_SEARCH_HISTORY.csv"
    with open(old_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(old_row.keys()))
        writer.writeheader()
        writer.writerow(old_row)

    parsed = parse_step11_search_history(old_csv)
    assert len(parsed) == 1
    p0 = parsed[0]

    # Các trường cũ vẫn đọc đúng
    assert p0["candidate_id"] == "CANDIDATE_0001"
    assert p0["cycle"] == 1
    assert p0["feasible"] is False
    assert p0["M1_shape_pass"] is True
    assert p0["M2_shape_pass"] is False

    # Toàn bộ 20 trường curvature mới đều là None, không gây crash
    assert p0["M1_KG_min_per_mm2"] is None
    assert p0["M1_H_min_per_mm"] is None
    assert p0["M1_curvature_sign_flip_count"] is None
    assert p0["M2_KG_min_per_mm2"] is None
    assert p0["M2_H_min_per_mm"] is None
    assert p0["M2_curvature_sign_flip_count"] is None


def test_step11_execution_config_validation():
    """Kiểm tra xác thực config execution dành riêng cho STEP11 candidate multiprocessing."""
    import pytest
    from execution_v55 import validate_execution_config

    base_cfg = {
        "mode": "reference",
        "cpu_workers": 10,
        "max_inflight_cpu_jobs": 10,
        "host_memory_budget_gib": 10.0,
        "worker_memory_estimate_gib": 2.0,
        "parallel_step11_candidates": True,
        "step11_candidate_workers": 4,
        "step11_worker_memory_estimate_gib": 1.5,
    }
    # 1. Config hợp lệ
    validate_execution_config(base_cfg)

    # 2. parallel_step11_candidates không phải bool
    bad_bool = dict(base_cfg, parallel_step11_candidates=1)
    with pytest.raises(ValueError, match="PARALLEL_STEP11_CANDIDATES_MUST_BE_BOOL"):
        validate_execution_config(bad_bool)

    # 3. step11_candidate_workers không hợp lệ
    for bad_workers in (True, 0, 11, 15):
        cfg = dict(base_cfg, step11_candidate_workers=bad_workers)
        with pytest.raises(ValueError, match="STEP11_CANDIDATE_WORKERS_MUST_BE_INT_BETWEEN_1_AND_GLOBAL_LIMITS"):
            validate_execution_config(cfg)

    # 4. step11_worker_memory_estimate_gib không hợp lệ
    for bad_mem in (True, 0, -1.0, float("nan"), float("inf")):
        cfg = dict(base_cfg, step11_worker_memory_estimate_gib=bad_mem)
        with pytest.raises(ValueError, match="STEP11_WORKER_MEMORY_ESTIMATE_GIB_MUST_BE_POSITIVE_FINITE"):
            validate_execution_config(cfg)


def test_step11_preserves_ci_and_fit_diagnostics_in_record_and_history():
    """Bao dam refactor worker khong lam roi diagnostic CI va M2 fit cua STEP11."""
    import inspect
    from pipeline_v55 import evaluate_step11_candidate_job, step_11

    evaluator_source = inspect.getsource(evaluate_step11_candidate_job)
    history_source = inspect.getsource(step_11)

    assert 'record[f"M2_CI_{check_name}_actual"]' in evaluator_source
    assert 'record[f"M2_CI_{check_name}_limit"]' in evaluator_source

    dynamic_ci_history_fields = (
        "M2_CI_bundle_curl_rms_p95_actual",
        "M2_CI_bundle_curl_rms_p95_limit",
        "M2_CI_bundle_local_geometry_normal_rms_p95_actual",
        "M2_CI_bundle_local_geometry_normal_rms_p95_limit",
        "M2_CI_bundle_loop_circulation_p95_actual",
        "M2_CI_bundle_loop_circulation_p95_limit",
        "M2_CI_bundle_edge_gradient_height_residual_p95_actual",
        "M2_CI_bundle_edge_gradient_height_residual_p95_limit",
        "M2_CI_bundle_nearest_normal_angle_p99_actual",
        "M2_CI_bundle_nearest_normal_angle_p99_limit",
    )
    fixed_fields = (
        "M2_CI_bundle_count",
        "M2_CI_evaluable_bundle_count",
        "M2_fit_curvature_per_mm",
        "M2_fit_conic_constant",
        "M2_fit_sag_rms_mm",
        "M2_fit_normal_rms_deg",
        "M2_fit_normal_max_deg",
        "M2_fit_k_bound_hit",
        "M2_fit_curvature_bound_hit",
    )
    topology_admission_fields = (
        "M1_topology_enforcement",
        "M1_topology_admitted",
        "M1_topology_admission_status",
        "M1_topology_warning_reasons",
        "M2_topology_enforcement",
        "M2_topology_admitted",
        "M2_topology_admission_status",
        "M2_topology_warning_reasons",
        "candidate_surface_admitted",
        "actual_M1_topology_pass",
        "actual_M1_topology_admitted",
        "actual_M1_topology_admission_status",
        "actual_M1_topology_warning_reasons",
        "actual_M2_topology_pass",
        "actual_M2_topology_admitted",
        "actual_M2_topology_admission_status",
        "actual_M2_topology_warning_reasons",
    )

    for field in dynamic_ci_history_fields:
        assert f'"{field}"' in history_source

    for field in fixed_fields:
        assert f'"{field}"' in evaluator_source
        assert f'"{field}"' in history_source

    for field in topology_admission_fields:
        assert f'"{field}"' in evaluator_source
        assert f'"{field}"' in history_source

    assert (
        "physical_hard_pass = bool("
        in evaluator_source
    )
    assert (
        "unobscured"
        in evaluator_source
    )


def test_step11_memory_admission_override():
    """Kiểm tra memory admission riêng cho STEP11 không ảnh hưởng STEP7 và caller khác."""
    from execution_v55 import ExecutionRuntime, _admitted_worker_count

    cfg = {
        "cpu_workers": 10,
        "max_inflight_cpu_jobs": 10,
        "host_memory_budget_gib": 9.0,
        "worker_memory_estimate_gib": 2.0,
        "step11_candidate_workers": 4,
        "step11_worker_memory_estimate_gib": 1.5,
    }
    runtime = ExecutionRuntime(
        mode="reference",
        config=cfg,
        run_dir=Path("."),
        is_worker=False,
        implementation_fingerprint="test",
        backend_plan={},
        session_id="test_session",
    )

    # 1. Không truyền override: hành vi cũ (9.0 / 2.0 = 4)
    w_default = _admitted_worker_count(runtime, "STEP7")
    assert w_default == 4

    # 2. Truyền override cho STEP11: host_budget=9.0, estimate=1.5 -> limit 6, worker_limit=4 -> 4
    w_step11 = _admitted_worker_count(
        runtime,
        "STEP11_CANDIDATES",
        worker_limit=4,
        worker_memory_estimate_gib=1.5,
    )
    assert w_step11 == 4

    # 3. Khi budget không đủ
    w_zero = _admitted_worker_count(
        runtime,
        "STEP11_CANDIDATES",
        worker_limit=4,
        worker_memory_estimate_gib=15.0,
    )
    assert w_zero == 0


def test_step11_not_in_parallel_ray_trace_steps():
    """Xác nhận STEP11 không thuộc PARALLEL_RAY_TRACE_STEPS để tránh nested process pool."""
    from pipeline_v55 import PARALLEL_RAY_TRACE_STEPS

    assert 11 not in PARALLEL_RAY_TRACE_STEPS
    assert PARALLEL_RAY_TRACE_STEPS == {12, 14, 17, 20, 24}


def test_step11_restoration_fd_job_ordering():
    """FD jobs phải deterministic: column rồi sign -/+."""

    descriptors = [
        {
            "surface": "M1",
            "kind": "CURVATURE",
        },
        {
            "surface": "M2",
            "kind": "CURVATURE",
        },
    ]

    u = np.zeros(
        2,
        dtype=float,
    )

    h = 0.02

    jobs = []

    for column in range(
        len(descriptors)
    ):

        for sign in (
            -1.0,
            +1.0,
        ):

            probe = u.copy()

            probe[
                column
            ] += (
                sign
                *
                h
            )

            jobs.append({
                "column":
                    column,

                "sign":
                    sign,

                "search_vector":
                    probe,
            })

    assert [
        (
            row[
                "column"
            ],
            row[
                "sign"
            ],
        )
        for row in jobs
    ] == [
        (0, -1.0),
        (0, +1.0),
        (1, -1.0),
        (1, +1.0),
    ]


def test_step11_worker_reply_schema():
    """Kiểm tra schema kiểm tra đầu vào và định dạng phản hồi của step11_candidate_worker."""
    import pytest
    from execution_workers_v55 import step11_candidate_worker

    # Chưa khởi tạo _BASE_CONTEXT thì worker phải raise lỗi
    invalid_job = {"schema": "INVALID_SCHEMA"}
    with pytest.raises(RuntimeError, match="STEP11_WORKER_BASE_CONTEXT_NOT_INITIALIZED"):
        step11_candidate_worker(invalid_job)


def test_step17_o4_degree_regularization_config_validation():
    """STEP17 O4-only degree regularization must be finite and nonnegative."""
    config = json.loads(
        (
            Path(__file__).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    validate_config(config)

    assert config["surface_fit"][
        "step17_o4_degree_regularization"
    ] == pytest.approx(
        1.0e-4
    )

    legacy_config = copy.deepcopy(config)
    legacy_config["surface_fit"].pop(
        "step17_o4_degree_regularization"
    )

    validate_config(
        legacy_config
    )

    for invalid_value in (
        True,
        -1.0e-6,
        float("nan"),
        float("inf"),
    ):
        invalid_config = copy.deepcopy(
            config
        )
        invalid_config["surface_fit"][
            "step17_o4_degree_regularization"
        ] = invalid_value

        with pytest.raises(
            ValueError,
            match=(
                "STEP17_O4_DEGREE_"
                "REGULARIZATION_INVALID"
            ),
        ):
            validate_config(
                invalid_config
            )


def test_step11_macro_micro_conditioning_config_validation():
    """Validate configurable gamma schedules and bounded alternating projection."""
    config = json.loads(
        (
            Path(__file__).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    validate_config(config)

    conditioning = config["surface_fit"][
        "step11_macro_micro_conditioning"
    ]
    assert conditioning["enabled"] is True
    assert conditioning["fallback_gamma_candidates"] == [
        0.70,
        0.85,
        0.95,
        1.0,
        1.05,
        1.15,
        1.30,
    ]
    assert (
        conditioning["maximum_alternating_projection_iterations"]
        == 5
    )

    for valid_gamma_values in (
        [0.70],
        [0.70, 1.0, 1.30],
        [1.30, 1.0, 0.70],
    ):
        custom_gamma = copy.deepcopy(config)
        custom_gamma["surface_fit"][
            "step11_macro_micro_conditioning"
        ]["fallback_gamma_candidates"] = valid_gamma_values
        validate_config(custom_gamma)
        assert custom_gamma["surface_fit"][
            "step11_macro_micro_conditioning"
        ]["fallback_gamma_candidates"] == valid_gamma_values

    for invalid_gamma_values in (
        [],
        [0.0],
        [-0.70],
        [0.70, 0.70],
        [float("nan")],
        [float("inf")],
        [True],
    ):
        bad_gamma = copy.deepcopy(config)
        bad_gamma["surface_fit"][
            "step11_macro_micro_conditioning"
        ]["fallback_gamma_candidates"] = invalid_gamma_values
        with pytest.raises(
            ValueError,
            match="STEP11_FALLBACK_GAMMA_CANDIDATES_INVALID",
        ):
            validate_config(bad_gamma)

    bad_iterations = copy.deepcopy(config)
    bad_iterations["surface_fit"][
        "step11_macro_micro_conditioning"
    ]["maximum_alternating_projection_iterations"] = 0
    with pytest.raises(
        ValueError,
        match="STEP11_ALTERNATING_PROJECTION_ITERATIONS_INVALID",
    ):
        validate_config(bad_iterations)


def test_step11_gamma_anchor_rank_rule():
    """Integrability PASS precedes STEP11 rank, then gamma distance and order."""
    from pipeline_v55 import _step11_gamma_anchor_rank_key

    warn_payload = {
        "M2_integrability_gate": {"status": "WARN"},
        "rank_key": (0, 0.1, 0.2, 0.3),
    }
    pass_payload = {
        "M2_integrability_gate": {"status": "PASS"},
        "rank_key": (1, 100.0, 100.0, 100.0),
    }

    assert (
        _step11_gamma_anchor_rank_key(
            pass_payload,
            0.85,
            0,
        )
        <
        _step11_gamma_anchor_rank_key(
            warn_payload,
            1.0,
            2,
        )
    )

    tied_payload = {
        "M2_integrability_gate": {"status": "PASS"},
        "rank_key": (0, 1.0, 2.0, 3.0),
    }
    assert (
        _step11_gamma_anchor_rank_key(
            tied_payload,
            0.95,
            1,
        )
        <
        _step11_gamma_anchor_rank_key(
            tied_payload,
            0.85,
            0,
        )
    )
    assert (
        _step11_gamma_anchor_rank_key(
            tied_payload,
            0.95,
            1,
        )
        <
        _step11_gamma_anchor_rank_key(
            tied_payload,
            1.05,
            3,
        )
    )


def test_step11_alternating_projection_preserves_reflection_law(
    monkeypatch,
):
    """Conditioned normals must reflect each incoming ray toward its target."""
    import pipeline_v55 as pipeline

    starts = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    directions = np.asarray([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    targets = np.asarray([
        [1.0, 0.0, 3.0],
        [1.0, 1.0, 3.0],
    ])
    ray_order = np.asarray([1, 0], int)
    raw_points = np.asarray([
        [0.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
    ])
    raw_normals = np.asarray([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ])

    raw_ci = {
        "points_ordered": raw_points[ray_order].copy(),
        "normals_ordered": raw_normals[ray_order].copy(),
        "ray_order": ray_order.copy(),
        "parent_order_index": np.asarray([-1, 0], int),
        "nearest_distance_mm": np.asarray([0.0, 1.0]),
        "points_by_ray": raw_points.copy(),
        "normals_by_ray": raw_normals.copy(),
        "starts_by_ray": starts.copy(),
        "directions_by_ray": directions.copy(),
        "targets_by_ray": targets.copy(),
        "fallback_count": 0,
    }

    class TemporarySurface:
        """Temporary test surface stub."""
        frame = np.eye(3)

        def intersect(
            self,
            starts_input,
            directions_input,
            finite=False,
        ):
            """Intersect stub returning ray offset points."""
            points = (
                np.asarray(starts_input, float)
                +
                np.asarray(directions_input, float)
            )
            return {
                "valid": np.ones(len(points), bool),
                "point": points,
                "normal": np.tile(
                    np.asarray([0.0, 0.0, 1.0]),
                    (len(points), 1),
                ),
                "t": np.ones(len(points), float),
            }

    seed_surface = TemporarySurface()

    def fake_point_by_point_engine(
        starts_input,
        directions_input,
        targets_input,
        seed_input,
        seed_index,
        construction_options=None,
    ):
        """Mock point by point engine."""
        return copy.deepcopy(raw_ci)

    def fake_construction_diagnostics(
        ctx_input,
        ci_input,
        frame_input,
        label,
    ):
        """Mock construction diagnostics."""
        iteration = int(
            ci_input.get("conditioning_iteration", 0)
        )
        return {
            "label": label,
            "iteration": iteration,
            "cloud": {
                "bundle_summary": {
                    "bundle_count": 2,
                    "evaluable_bundle_count": 2,
                }
            },
        }

    def fake_integrability_gate(
        diagnostic,
        policy,
        surface,
    ):
        """Mock integrability gate."""
        passed = int(diagnostic["iteration"]) >= 1
        status = "PASS" if passed else "WARN"
        return {
            "surface": surface,
            "status": status,
            "enforcement": "WARN",
            "checks": [{
                "surface": surface,
                "check":
                    "bundle_local_geometry_normal_rms_p95",
                "actual": 10.0 if passed else 30.0,
                "operator": "<=",
                "limit": 15.0,
                "unit": "deg",
                "status": status,
                "enforcement": "WARN",
                "reason": "test",
            }],
        }

    def fake_fit_surface(
        points,
        normals,
        seed,
        order,
        axis_order2,
        fit_weights,
        vertex_index,
        fit_options=None,
        incident_starts=None,
        outgoing_targets=None,
    ):
        """Mock fit surface."""
        return TemporarySurface(), {
            "curvature_per_mm": 0.01,
            "conic_constant": 0.0,
            "sag_rms_mm": 0.01,
            "normal_rms_deg": 1.0,
            "normal_max_deg": 2.0,
        }

    monkeypatch.setattr(
        pipeline,
        "point_by_point_engine",
        fake_point_by_point_engine,
    )
    monkeypatch.setattr(
        pipeline,
        "_construction_diagnostics",
        fake_construction_diagnostics,
    )
    monkeypatch.setattr(
        pipeline,
        "_integrability_quality_gate",
        fake_integrability_gate,
    )
    monkeypatch.setattr(
        pipeline,
        "fit_surface",
        fake_fit_surface,
    )

    ctx = SimpleNamespace(
        config={
            "fit_weights": {},
            "ci_construction": {
                "mode": "SURFACE_COMPATIBLE_PATCH_V3",
                "compatible_frontier": {},
            },
            "surface_fit": {
                "step11_macro_micro_conditioning": {
                    "enabled": True,
                    "maximum_alternating_projection_iterations": 5,
                },
            },
        },
        data={},
    )
    progress_messages: list[str] = []

    (
        conditioned_ci,
        conditioned_diagnostic,
        conditioned_gate,
        audit,
    ) = pipeline._step11_condition_m2_ci(
        ctx,
        starts,
        directions,
        targets,
        seed_surface,
        0,
        ctx.config["surface_fit"],
        {"enforcement": "WARN"},
        progress_callback=progress_messages.append,
    )

    conditioned_points = np.asarray(
        conditioned_ci["points_by_ray"],
        float,
    )
    conditioned_normals = np.asarray(
        conditioned_ci["normals_by_ray"],
        float,
    )
    desired_outgoing = unit(
        targets - conditioned_points
    )
    actual_outgoing = reflect(
        directions,
        conditioned_normals,
    )

    assert np.allclose(
        actual_outgoing,
        desired_outgoing,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.array_equal(
        conditioned_ci["points_ordered"],
        conditioned_points[ray_order],
    )
    assert np.array_equal(
        conditioned_ci["normals_ordered"],
        conditioned_normals[ray_order],
    )
    assert conditioned_diagnostic["iteration"] == 1
    assert conditioned_gate["status"] == "PASS"
    assert audit["selected_iteration"] == 1
    assert (
        audit["status"]
        == "PASS_AFTER_ALTERNATING_PROJECTION"
    )
    assert audit["direct_normal_averaging_used"] is False
    assert any("[AP RAW][START]" in row for row in progress_messages)
    assert any("[AP 0/5][DONE]" in row for row in progress_messages)
    assert any("[AP 1/5][START]" in row for row in progress_messages)
    assert any("[AP 1/5][DONE]" in row for row in progress_messages)
    assert any("[AP FINAL]" in row for row in progress_messages)
    assert np.array_equal(
        raw_ci["points_by_ray"],
        raw_points,
    )


def test_step11_terminal_progress_markers_present():
    """Tất cả heartbeat và milestone progress strings phải có mặt trong source."""
    import inspect
    from pipeline_v55 import _step11_condition_m2_ci, step_11, _step11_run_joint_topology_restoration

    conditioning_source = inspect.getsource(
        _step11_condition_m2_ci
    )
    step11_source = inspect.getsource(step_11)
    restore_source = inspect.getsource(_step11_run_joint_topology_restoration)

    for marker in (
        "[AP RAW][START]",
        "[AP 0/",
        "[AP FINAL]",
    ):
        assert marker in conditioning_source

    for marker in (
        "[GAMMA SWEEP][START]",
        "[GAMMA SWEEP][DONE]",
        "JOINT M1/M2 TOPOLOGY RESTORATION",
    ):
        assert marker in step11_source

    for marker in (
        "[RESTORE][START]",
        "[RESTORE][ITER ",
    ):
        assert marker in restore_source


def test_step11_candidate_snapshot_roundtrip_and_render(
    tmp_path: Path,
):
    """Snapshot giữ full arrays, surfaces và render được mà không ray-trace lại."""
    from pipeline_v55 import (
        _step11_archive_candidate_snapshot,
    )
    from render_multistart_saved_v55 import (
        render_step11_candidate_sequence,
    )

    ray_count = 6
    y = np.linspace(
        -1.0,
        1.0,
        ray_count,
    )
    z = np.linspace(
        -0.5,
        0.5,
        ray_count,
    )

    origins = np.column_stack([
        np.zeros(ray_count),
        y,
        z,
    ])
    visor_points = np.column_stack([
        np.ones(ray_count),
        y,
        z,
    ])
    m1_points = np.column_stack([
        np.full(ray_count, 2.0),
        y,
        z,
    ])
    m2_points = np.column_stack([
        np.full(ray_count, 3.0),
        y,
        z,
    ])
    display_points = np.column_stack([
        np.full(ray_count, 4.0),
        y,
        z,
    ])
    directions = np.tile(
        np.array([
            1.0,
            0.0,
            0.0,
        ]),
        (ray_count, 1),
    )
    normals = np.tile(
        np.array([
            -1.0,
            0.0,
            0.0,
        ]),
        (ray_count, 1),
    )
    valid = np.ones(
        ray_count,
        dtype=bool,
    )

    m1 = PolySurface.plane(
        "M1",
        np.array([2.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        half_aperture=(2.0, 2.0),
    )
    m2 = PolySurface.plane(
        "M2",
        np.array([3.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        half_aperture=(2.0, 2.0),
    )
    display = PolySurface.plane(
        "DISPLAY",
        np.array([4.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        half_aperture=(2.0, 2.0),
    )

    ci_m2 = {
        "points_by_ray": m2_points,
        "normals_by_ray": normals,
        "starts_by_ray": m1_points,
        "directions_by_ray": directions,
        "targets_by_ray": display_points,
    }
    physical_trace = {
        "points": [
            visor_points,
            m1_points,
            m2_points,
            display_points,
        ],
        "directions": [
            directions,
            directions,
            directions,
            directions,
        ],
        "valid": valid,
        "resolved": valid,
        "landing": display_points,
        "display_local": display_points[:, 1:],
    }

    record = {
        "candidate_id": "CANDIDATE_0001",
        "cycle": 1,
        "trust_scale": 1.0,
        "move": "coefficient:+1",
        "feasible": True,
        "rejection_stage": None,
        "rejection_reason": None,
        "M2_CI_bundle_count": 3,
        "M2_fit_sag_rms_mm": 0.01,
        "_visualization_payload": {
            "search_vector":
                np.array([0.25, -0.5]),
            "m1": m1,
            "m1_hit_points": m1_points,
            "m1_hit_normals": normals,
            "m1_reflected_directions":
                directions,
            "m1_valid": valid,
            "ci_m2": ci_m2,
            "m2": m2,
            "physical_trace": physical_trace,
        },
    }
    step_dir = tmp_path / "STEP_11"
    snapshot_root = (
        step_dir
        / "11_CANDIDATE_SNAPSHOTS"
    )
    job = {
        "candidate_id": "CANDIDATE_0001",
        "snapshot": {
            "enabled": True,
            "root": str(snapshot_root),
            "archive_index": 1,
            "candidate_class": "COORDINATE",
        },
    }

    summary = (
        _step11_archive_candidate_snapshot(
            record,
            "CANDIDATE/CANDIDATE_0001/COMPLETE",
            job,
        )
    )

    assert summary is not None
    assert summary["status"] == "SAVED"
    assert "_visualization_payload" not in record

    candidate_dir = (
        snapshot_root
        / summary["directory"]
    )
    metadata = json.loads(
        (
            candidate_dir
            / "metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["surfaces"]["M1"] is not None
    assert metadata["surfaces"]["M2"] is not None
    assert metadata["full_precision_saved"] is True
    assert (
        metadata["candidate_record"][
            "M2_CI_bundle_count"
        ]
        == 3
    )

    with np.load(
        candidate_dir / "arrays.npz",
        allow_pickle=False,
    ) as archive:
        assert np.array_equal(
            archive["m1_hit_points"],
            m1_points,
        )
        assert np.array_equal(
            archive["ci_m2_points_by_ray"],
            m2_points,
        )
        assert np.array_equal(
            archive["physical_point_03"],
            display_points,
        )
        assert archive[
            "m1_hit_points"
        ].dtype == m1_points.dtype

    ctx = SimpleNamespace(
        data={
            "rays": {
                "origins": origins,
                "field_index":
                    np.arange(ray_count) % 3,
            },
            "visor_hit": {
                "point": visor_points,
            },
            "display": display,
        }
    )
    render_result = (
        render_step11_candidate_sequence(
            ctx,
            step_dir,
        )
    )
    assert render_result["status"] == "RENDERED"
    assert render_result["snapshot_count"] == 1
    assert render_result["rendered_count"] == 1
    assert (
        render_result["warning_count"]
        == 0
    )
    assert Path(
        render_result["rows"][0]["image"]
    ).is_file()


def test_step11_restoration_probes_do_not_archive_snapshots():
    """Restoration probe jobs do not write candidate snapshots."""
    import inspect
    import execution_workers_v55
    import pipeline_v55

    probe_source = inspect.getsource(
        pipeline_v55.
        evaluate_step11_restoration_probe_job
    )

    worker_source = inspect.getsource(
        execution_workers_v55.
        step11_restoration_probe_worker
    )

    assert (
        "_step11_archive_candidate_snapshot"
        not in probe_source
    )

    assert (
        "_step11_archive_candidate_snapshot"
        not in worker_source
    )

    assert (
        "_step11_build_parallel_jacobian"
        in inspect.getsource(
            pipeline_v55.
            _step11_run_joint_topology_restoration
        )
    )


def test_step11_fixed_residual_grid_is_deterministic():
    """Fixed residual grid is deterministic."""
    from pipeline_v55 import (
        _step11_fixed_residual_grid,
    )

    surface = PolySurface(
        "M2",
        np.zeros(3),
        np.eye(3),
        np.array([
            10.0,
            5.0,
        ]),
        [
            (2, 0),
            (0, 2),
        ],
        np.array([
            -0.01,
            -0.01,
        ]),
        np.array([
            10.0,
            5.0,
        ]),
    )

    a = _step11_fixed_residual_grid(
        surface,
        31,
    )

    b = _step11_fixed_residual_grid(
        surface,
        31,
    )

    assert np.array_equal(
        a["xy"],
        b["xy"],
    )

    assert np.array_equal(
        a["edge_i"],
        b["edge_i"],
    )

    assert np.array_equal(
        a["edge_j"],
        b["edge_j"],
    )


def test_step11_balanced_residual_is_sample_count_invariant():
    """Balanced residual blocks scale invariantly with sample count."""
    from pipeline_v55 import (
        _step11_balanced_block,
    )

    small = (
        _step11_balanced_block(
            np.ones(10),
            1.0,
            1.0,
        )
    )

    large = (
        _step11_balanced_block(
            np.ones(100),
            1.0,
            1.0,
        )
    )

    assert np.isclose(
        np.sum(
            small * small
        ),
        np.sum(
            large * large
        ),
    )


def test_step11_restoration_kg_target_is_inside_hard_tolerance():
    """Restoration target KG is strictly inside the hard tolerance."""
    config = json.loads(
        (
            Path(
                __file__
            ).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    target = float(
        config[
            "surface_fit"
        ][
            "step11_topology_restoration"
        ][
            "kg_target_per_mm2"
        ]
    )

    # mirror_shape_gate hard acceptance:
    hard_floor = -1e-7

    assert target > hard_floor
    assert target <= 0.0


def test_step11_restoration_rejects_curvature_bound_instead_of_clipping():
    """Restoration rejects parameter moves exceeding curvature limits."""
    from pipeline_v55 import (
        _step11_apply_joint_restoration_vector,
    )

    config = json.loads(
        (
            Path(
                __file__
            ).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    cfg = (
        config[
            "surface_fit"
        ][
            "step11_topology_restoration"
        ]
    )

    surface_fit_cfg = (
        config[
            "surface_fit"
        ]
    )

    limit = float(
        surface_fit_cfg[
            "curvature_absolute_max_per_mm"
        ]
    )

    m1 = PolySurface(
        "M1",
        np.zeros(3),
        np.eye(3),
        np.array([
            5.0,
            5.0,
        ]),
        [],
        np.empty(
            0,
            dtype=float,
        ),
        np.array([
            5.0,
            5.0,
        ]),
        curvature=
            limit * 0.999,
    )

    m2 = m1.copy()
    m2.name = "M2"

    base = {
        "M1": m1,
        "M2": m2,
    }

    descriptors = [
        {
            "surface": "M1",
            "kind": "CURVATURE",
        }
    ]

    u = np.asarray([
        1.0
    ])

    a, b, state = (
        _step11_apply_joint_restoration_vector(
            base,
            descriptors,
            u,
            cfg,
            surface_fit_cfg,
        )
    )

    assert a is None
    assert b is None

    assert (
        "CURVATURE_BOUND"
        in
        state[
            "reason"
        ]
    )


def test_step11_restoration_worker_requires_base_context():
    """Restoration probe worker fails if base context is uninitialized."""
    from execution_workers_v55 import (
        step11_restoration_probe_worker,
    )

    with pytest.raises(
        RuntimeError,
        match=
            "STEP11_RESTORATION_"
            "WORKER_BASE_CONTEXT_NOT_INITIALIZED",
    ):

        step11_restoration_probe_worker({
            "schema":
                "HUD_FAN_V5_5_"
                "STEP11_RESTORATION_PROBE_JOB_V1",
        })


def test_step11_parallel_jacobian_recovers_one_sided_fd(
    monkeypatch,
):
    """Một phía invalid phải fallback sang one-sided derivative."""

    import pipeline_v55

    config = json.loads(
        (
            Path(
                __file__
            ).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    cfg = copy.deepcopy(
        config[
            "surface_fit"
        ][
            "step11_topology_restoration"
        ]
    )

    common = {
        "restoration_cfg":
            cfg,

        "descriptors": [
            {
                "surface":
                    "M2",

                "kind":
                    "CURVATURE",
            }
        ],
    }

    def fake_evaluate(
        common_arg,
        normalized,
        *,
        include_payload,
    ):
        """Mock evaluation for one-sided FD recovery test."""
        u = np.asarray(
            normalized,
            float,
        )

        # Negative side intentionally invalid.
        if u[0] < 0.0:
            return {
                "valid": False,
                "reason": "SYNTHETIC_NEGATIVE_BOUND",
            }

        return {
            "valid": True,
            "reason": None,
            "residual":
                np.asarray([
                    u[0]
                ]),
            "objective":
                float(
                    u[0]
                    *
                    u[0]
                ),
            "meta": {},
            "bound_state": {
                "valid": True,
            },
        }

    monkeypatch.setattr(
        pipeline_v55,
        "_step11_evaluate_restoration_u",
        fake_evaluate,
    )

    jacobian, active, rows = (
        pipeline_v55.
        _step11_build_parallel_jacobian(
            common,
            np.zeros(
                1,
                dtype=float,
            ),
            np.zeros(
                1,
                dtype=float,
            ),
            1,
            1.0,
            None,
            1,
        )
    )

    assert active == [
        0
    ]

    assert np.isclose(
        jacobian[
            0,
            0
        ],
        1.0,
        atol=1e-9,
    )

    assert any(
        row[
            "fd_mode"
        ]
        ==
        "FORWARD"
        for row in rows
    )


def test_step11_joint_restoration_solver_converges_end_to_end(
    monkeypatch,
):
    """Jacobian + LM + rho acceptance phải thật sự di chuyển tới basin feasible."""

    import pipeline_v55
    from pipeline_v55 import Context

    config = json.loads(
        (
            Path(
                __file__
            ).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    cfg = copy.deepcopy(
        config[
            "surface_fit"
        ][
            "step11_topology_restoration"
        ]
    )

    cfg[
        "maximum_iterations"
    ] = 10

    ctx = Context(
        config={
            "execution": {
                "parallel_step11_candidates":
                    False,

                "step11_candidate_workers":
                    1,

                "step11_worker_memory_estimate_gib":
                    1.0,
            }
        }
    )

    target = np.asarray([
        0.20,
        -0.12,
    ])

    descriptors = [
        {
            "surface":
                "M1",

            "kind":
                "CURVATURE",
        },

        {
            "surface":
                "M2",

            "kind":
                "CURVATURE",
        },
    ]

    common = {
        "ctx":
            ctx,

        "restoration_cfg":
            cfg,

        "descriptors":
            descriptors,
    }

    def fake_evaluate(
        common_arg,
        normalized,
        *,
        include_payload,
    ):
        """Mock evaluation for convergence test."""

        u = np.asarray(
            normalized,
            float,
        )

        residual = (
            u
            -
            target
        )

        distance = float(
            np.linalg.norm(
                residual
            )
        )

        hard_pass = bool(
            distance < 0.03
        )

        meta = {
            "M1_raw_topology_pass":
                True,

            "M2_raw_topology_pass":
                hard_pass,

            "unobscured":
                hard_pass,

            "S_AQP_signed_mm2":
                (
                    1.0
                    if hard_pass
                    else
                    -distance
                ),

            "M2_oriented_H_min_per_mm":
                (
                    1e-4
                    if hard_pass
                    else
                    -distance
                ),

            "M2_KG_min_per_mm2":
                (
                    -5e-8
                    if hard_pass
                    else
                    -distance
                    *
                    1e-5
                ),
        }

        result = {
            "valid":
                True,

            "reason":
                None,

            "residual":
                residual,

            "objective":
                float(
                    np.mean(
                        residual
                        *
                        residual
                    )
                ),

            "meta":
                meta,

            "bound_state": {
                "valid":
                    True,
            },
        }

        if include_payload:

            result.update({
                "m1":
                    "SYNTHETIC_M1",

                "m2":
                    "SYNTHETIC_M2",

                "evaluation":
                    {},
            })

        return result

    monkeypatch.setattr(
        pipeline_v55,
        "_step11_evaluate_restoration_u",
        fake_evaluate,
    )

    restoration = (
        pipeline_v55.
        _step11_run_joint_topology_restoration(
            common,
            lambda message:
                None,
        )
    )

    assert restoration[
        "success"
    ]

    assert restoration[
        "failure_message"
    ] is None

    assert (
        np.linalg.norm(
            np.asarray(
                restoration[
                    "u"
                ],
                float,
            )
            -
            target
        )
        <
        0.05
    )

    assert any(
        bool(
            row.get(
                "accepted",
                False,
            )
        )
        for row
        in restoration[
            "history"
        ]
    )


def test_step11_joint_restoration_nonconvergence_returns_diagnostics(
    monkeypatch,
):
    """Không hội tụ phải trả history thay vì raise bên trong solver."""

    import pipeline_v55
    from pipeline_v55 import Context

    config = json.loads(
        (
            Path(
                __file__
            ).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    cfg = copy.deepcopy(
        config[
            "surface_fit"
        ][
            "step11_topology_restoration"
        ]
    )

    cfg[
        "maximum_iterations"
    ] = 2

    ctx = Context(
        config={
            "execution": {
                "parallel_step11_candidates":
                    False,

                "step11_candidate_workers":
                    1,

                "step11_worker_memory_estimate_gib":
                    1.0,
            }
        }
    )

    common = {
        "ctx":
            ctx,

        "restoration_cfg":
            cfg,

        "descriptors": [
            {
                "surface":
                    "M2",

                "kind":
                    "CURVATURE",
            }
        ],
    }

    def fake_constant_evaluate(
        common_arg,
        normalized,
        *,
        include_payload,
    ):
        """Mock evaluation for nonconvergence diagnostics test."""

        result = {
            "valid":
                True,

            "reason":
                None,

            # Constant residual -> Jacobian zero.
            "residual":
                np.asarray([
                    1.0
                ]),

            "objective":
                1.0,

            "meta": {
                "M1_raw_topology_pass":
                    True,

                "M2_raw_topology_pass":
                    False,

                "unobscured":
                    False,

                "S_AQP_signed_mm2":
                    -10.0,

                "M2_oriented_H_min_per_mm":
                    -0.01,

                "M2_KG_min_per_mm2":
                    -1e-4,
            },

            "bound_state": {
                "valid":
                    True,
            },
        }

        if include_payload:

            result.update({
                "m1":
                    "M1",

                "m2":
                    "M2",

                "evaluation":
                    {},
            })

        return result

    monkeypatch.setattr(
        pipeline_v55,
        "_step11_evaluate_restoration_u",
        fake_constant_evaluate,
    )

    restoration = (
        pipeline_v55.
        _step11_run_joint_topology_restoration(
            common,
            lambda message:
                None,
        )
    )

    assert not restoration[
        "success"
    ]

    assert restoration[
        "failure_message"
    ] is not None

    assert restoration[
        "stop_reason"
    ] in {
        "MAXIMUM_ITERATIONS",
        "JACOBIAN_RANK_ZERO",
    }

    assert len(
        restoration[
            "jacobian_history"
        ]
    ) > 0


def test_step11_unobscuration_violation_enters_restoration_residual(
    monkeypatch,
):
    """Signed area âm phải tạo residual; signed area dương phải bằng zero."""

    import pipeline_v55

    dummy_field = {
        "z_mm":
            np.zeros(1),

        "normal":
            np.asarray([
                [
                    0.0,
                    0.0,
                    1.0,
                ]
            ]),

        "H_per_mm":
            np.zeros(1),

        "KG_per_mm2":
            np.zeros(1),

        "k1_per_mm":
            np.zeros(1),

        "k2_per_mm":
            np.zeros(1),

        "oriented_H_per_mm":
            np.zeros(1),

        "gradient_k1_per_mm2":
            np.zeros(0),

        "gradient_k2_per_mm2":
            np.zeros(0),
    }

    monkeypatch.setattr(
        pipeline_v55,
        "_step11_curvature_field",
        lambda *args, **kwargs:
            dummy_field,
    )

    monkeypatch.setattr(
        pipeline_v55,
        "_step11_surface_restoration_blocks",
        lambda *args, **kwargs:
            [],
    )

    state = {
        "cfg": {
            "physical_fraction_scale":
                0.1,

            "weights": {
                "physical_fraction":
                    0.0,

                "unobscuration":
                    1.0,

                "optical":
                    0.0,
            },
        },

        "surface_fit_cfg": {
            "curvature_absolute_max_per_mm":
                1.0,
        },

        "quality_cfg": {
            "minimum_physical_fraction":
                0.0,
        },

        "grids": {
            "M1": {},
            "M2": {},
        },

        "anchors": {
            "M1": {},
            "M2": {},
        },

        "orientation_signs": {
            "M1": 1.0,
            "M2": 1.0,
        },

        "optical_indices":
            np.empty(
                0,
                dtype=int,
            ),

        "unobscuration_scale_mm2":
            2.0,
    }

    evaluation_bad = {
        "optical_residual":
            np.empty(
                0,
                dtype=float,
            ),

        "metrics": {
            "physical_fraction":
                1.0,
        },

        "M1_gate": {
            "topology_pass":
                True,
        },

        "M2_gate": {
            "topology_pass":
                True,
        },

        "S_AQP_signed_mm2":
            -4.0,
    }

    residual_bad, meta_bad = (
        pipeline_v55.
        _step11_build_restoration_residual(
            object(),
            object(),
            evaluation_bad,
            state,
        )
    )

    evaluation_good = dict(
        evaluation_bad
    )

    evaluation_good[
        "S_AQP_signed_mm2"
    ] = 4.0

    residual_good, meta_good = (
        pipeline_v55.
        _step11_build_restoration_residual(
            object(),
            object(),
            evaluation_good,
            state,
        )
    )

    assert residual_bad[
        -1
    ] > 0.0

    assert np.isclose(
        residual_bad[
            -1
        ],
        2.0,
    )

    assert np.isclose(
        residual_good[
            -1
        ],
        0.0,
    )

    assert not meta_bad[
        "unobscured"
    ]

    assert meta_good[
        "unobscured"
    ]


def test_step11_gamma_restoration_seed_prefers_less_bad_m2():
    """Gamma seed phải dùng topology distance thay vì đòi M2 feasible."""

    from pipeline_v55 import (
        _step11_gamma_restoration_seed_rank_key,
    )

    config = json.loads(
        (
            Path(
                __file__
            ).resolve().parent
            /
            "config_v55.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    cfg = (
        config[
            "surface_fit"
        ][
            "step11_topology_restoration"
        ]
    )

    worse = {
        "M2_basic_surface_sanity_pass":
            True,

        "M2_integrability_status":
            "WARN",

        "M2_oriented_H_min_per_mm":
            -0.008,

        "M2_KG_min_per_mm2":
            -2e-5,
    }

    better = {
        "M2_basic_surface_sanity_pass":
            True,

        "M2_integrability_status":
            "WARN",

        "M2_oriented_H_min_per_mm":
            -0.001,

        "M2_KG_min_per_mm2":
            -2e-6,
    }

    key_worse = (
        _step11_gamma_restoration_seed_rank_key(
            worse,
            1.0,
            0,
            cfg,
        )
    )

    key_better = (
        _step11_gamma_restoration_seed_rank_key(
            better,
            0.85,
            1,
            cfg,
        )
    )

    assert (
        key_better
        <
        key_worse
    )


def test_step11_disabled_restoration_only_evaluates_seed(
    monkeypatch,
):
    """Disabled mode không được dựng Jacobian hay chạy LM."""

    import pipeline_v55

    common = {
        "descriptors": [
            {
                "surface":
                    "M1",

                "kind":
                    "CURVATURE",
            }
        ],
    }

    calls = {
        "evaluate":
            0,
    }

    def fake_evaluate(
        common_arg,
        normalized,
        *,
        include_payload,
    ):
        """Mock evaluation for disabled restoration test."""

        calls[
            "evaluate"
        ] += 1

        return {
            "valid":
                True,

            "objective":
                0.0,

            "meta": {
                "M1_raw_topology_pass":
                    True,

                "M2_raw_topology_pass":
                    True,

                "unobscured":
                    True,

                "S_AQP_signed_mm2":
                    1.0,

                "M2_oriented_H_min_per_mm":
                    0.001,

                "M2_KG_min_per_mm2":
                    0.0,
            },

            "m1":
                "M1",

            "m2":
                "M2",

            "evaluation":
                {},
        }

    monkeypatch.setattr(
        pipeline_v55,
        "_step11_evaluate_restoration_u",
        fake_evaluate,
    )

    result = (
        pipeline_v55.
        _step11_disabled_restoration_result(
            common,
            lambda message:
                None,
        )
    )

    assert result[
        "success"
    ]

    assert calls[
        "evaluate"
    ] == 1

    assert result[
        "effective_workers"
    ] == 0

    assert (
        result[
            "stop_reason"
        ]
        ==
        "RESTORATION_DISABLED_SEED_HARD_FEASIBLE"
    )
