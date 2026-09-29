"""Tool to run qualification tests on compute kernels and generate qualified_backend.json."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from execution_v55 import compute_software_fingerprint, KNOWN_KERNELS


def qualify_plane_batch() -> tuple[bool, int, int, int, str]:
    """Kiểm tra kernel plane_batch."""
    from core import PolySurface
    plane = PolySurface.plane("M1_PLANE", [0, 0, 0], [0, 0, 1], half_aperture=(40, 40))
    origins = np.array([
        [0.0, 0.0, -10.0],
        [10.0, 10.0, -10.0],
        [50.0, 50.0, -10.0],
        [0.0, 0.0, 10.0],
    ])
    directions = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    hit = plane.intersect(origins, directions, finite=True)
    passed = bool(hit["valid"][0] and hit["valid"][1] and not hit["valid"][2] and not hit["valid"][3])
    return passed, 1 if passed else 0, 0 if passed else 1, 0, "POLYSURFACE_PLANE_INTERSECT_BATCH_PARITY"


def qualify_fixed_design_lstsq() -> tuple[bool, int, int, int, str]:
    """Kiểm tra kernel fixed_design_lstsq."""
    from kernels_cpu_v55 import FixedDesignLeastSquares
    np.random.seed(42)
    A = np.random.randn(20, 5)
    b = np.random.randn(20)
    solver = FixedDesignLeastSquares(A, rcond=1e-11)
    x_fixed = solver.solve(b)
    x_numpy, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-11)
    passed = bool(np.allclose(x_fixed, x_numpy, rtol=1e-10, atol=1e-10))
    return passed, 1 if passed else 0, 0 if passed else 1, 0, "FIXED_DESIGN_LSTSQ_NUMPY_EQUIVALENCE"


def qualify_point_by_point_ci() -> tuple[bool, int, int, int, str]:
    """Check native execution and discrete/numeric CI parity with legacy oracle, V1, V2, and V3 compatible frontier."""
    from ci_validation_v55 import (
        run_ci_parity_suite,
        run_ci_compatible_suite,
        run_ci_surface_compatible_v2_suite,
        run_ci_surface_compatible_v3_suite,
    )
    legacy = run_ci_parity_suite()
    v1 = run_ci_compatible_suite()
    v2 = run_ci_surface_compatible_v2_suite()
    v3 = run_ci_surface_compatible_v3_suite()
    passed = bool(legacy["passed"] and v1["passed"] and v2["passed"] and v3["passed"])
    cases_passed = int(legacy["cases_passed"] + v1["cases_passed"] + v2["cases_passed"] + v3["cases_passed"])
    cases_failed = int(legacy["cases_failed"] + v1["cases_failed"] + v2["cases_failed"] + v3["cases_failed"])
    return (
        passed,
        cases_passed,
        cases_failed,
        0,
        "NUMBA_LEGACY_FROZEN_PARITY_PLUS_SURFACE_COMPATIBLE_FRONTIER_V1_PLUS_V2_PLUS_PATCH_V3_SYNTHETIC",
    )


def qualify_cheb_ray_equations() -> tuple[bool, int, int, int, str]:
    """Kiểm tra kernel cheb_ray_equations trên CPU."""
    from kernels_array_v55 import cheb_ray_equations
    coeff = np.array([[1.0, 0.5], [0.2, 0.1]], dtype=float)
    scale = np.array([50.0, 50.0], dtype=float)
    ol = np.array([[0.0, 0.0, 10.0]], dtype=float)
    dl = np.array([[0.0, 0.0, -1.0]], dtype=float)

    eq = cheb_ray_equations(np, coeff, scale, ol, dl)
    passed = bool(eq.shape == (1, 3) and np.all(np.isfinite(eq)))
    return passed, 1 if passed else 0, 0 if passed else 1, 0, "CHEB_RAY_EQUATIONS_NUMPY_EVALUATION"


def _fermat_eval_fixture(
    row_count: int = 257,
) -> dict[str, Any]:
    """Fixture Fermat phi-trivial de qualification CPU/CUDA."""
    t = np.linspace(-1.0, 1.0, row_count)

    xy = np.column_stack((
        12.0 * t,
        7.0 * np.sin(np.pi * t),
    ))

    q1 = np.column_stack((
        -30.0 + 2.0 * t,
        5.0 * t,
        np.full(row_count, 35.0),
    ))

    targets = np.column_stack((
        28.0 - 3.0 * t,
        -4.0 * t,
        np.full(row_count, 30.0),
    ))

    return {
        "xy": xy.astype(np.float64),
        "q1": q1.astype(np.float64),
        "targets": targets.astype(np.float64),
        "center": np.array(
            [1.0, -2.0, 0.5], dtype=np.float64
        ),
        "frame": np.eye(3, dtype=np.float64),
        "scale": np.array(
            [50.0, 50.0], dtype=np.float64
        ),
        "terms": [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 0),
            (1, 1),
            (0, 2),
        ],
        "coeff": np.array(
            [
                0.1,
                0.02,
                -0.015,
                0.003,
                -0.002,
                0.001,
            ],
            dtype=np.float64,
        ),
        "curvature": 0.002,
        "conic": -0.5,
        "eps": 1e-9,
    }


def qualify_fermat_eval() -> tuple[
    bool, int, int, int, str
]:
    """Qualification Fermat NumPy CPU tren fixture phi-trivial."""
    from kernels_array_v55 import fermat_eval

    payload = _fermat_eval_fixture()

    result = fermat_eval(
        np,
        **payload,
    )

    grad, pts, op, refl = result
    row_count = len(payload["xy"])

    passed = bool(
        grad.shape == (row_count, 2)
        and pts.shape == (row_count, 3)
        and op.shape == (row_count,)
        and refl.shape == (row_count,)
        and np.all(np.isfinite(grad))
        and np.all(np.isfinite(pts))
        and np.all(np.isfinite(op))
        and np.all(np.isfinite(refl))
    )

    return (
        passed,
        1 if passed else 0,
        0 if passed else 1,
        0,
        "FERMAT_EVAL_NUMPY_NONTRIVIAL_FIXTURE",
    )


def qualify_fermat_eval_cuda(
    source_dir: Path,
) -> tuple[bool, int, int, int, str]:
    """So sanh CUDA Fermat voi NumPy reference tren cung input."""
    from kernels_array_v55 import fermat_eval
    from kernel_dispatch_v55 import dispatch_kernel
    from execution_v55 import execution_session

    payload = _fermat_eval_fixture()
    expected = fermat_eval(
        np,
        **payload,
    )

    config_path = source_dir / "config_v55.json"
    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)

    config.setdefault("execution", {})
    config["execution"].setdefault("gpu", {})
    config["execution"]["gpu"]["enabled"] = True

    with tempfile.TemporaryDirectory(
        prefix="hud_v55_fermat_cuda_qualification_"
    ) as temp_dir:
        with execution_session(
            config,
            Path(temp_dir),
            purpose="qualification",
            qualification_plan={
                "fermat_eval": "cuda",
            },
        ):
            actual = dispatch_kernel(
                "fermat_eval",
                payload,
            )

    if (
        not isinstance(actual, tuple)
        or len(actual) != 4
    ):
        return (
            False,
            0,
            1,
            0,
            "FERMAT_CUDA_RESULT_SCHEMA",
        )

    passed = True

    for ref, got in zip(expected, actual):
        if ref.shape != got.shape:
            passed = False
            break

        if not np.allclose(
            got,
            ref,
            rtol=1e-11,
            atol=1e-10,
            equal_nan=True,
        ):
            passed = False
            break

    return (
        passed,
        1 if passed else 0,
        0 if passed else 1,
        0,
        "FERMAT_EVAL_CUDA_VS_NUMPY_PARITY",
    )


def qualify_forward_active_fd() -> tuple[bool, int, int, int, str]:
    """Kiểm tra parity của kernel forward_active_fd giữa reference và active-only CPU."""
    from core import PolySurface, ChebVisor, forward_shoot
    from execution_v55 import ExecutionRuntime, _CURRENT_RUNTIME

    display = PolySurface.plane("DISPLAY", np.array([0.0, 0.0, -100.0]), np.array([0.0, 0.0, 1.0]), (10.0, 10.0))
    m2 = PolySurface.plane("M2", np.array([0.0, 0.0, -50.0]), np.array([0.0, 0.0, 1.0]), (20.0, 20.0))
    m1 = PolySurface.plane("M1", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (30.0, 30.0))
    visor = ChebVisor(np.array([0.0, 0.0, 50.0]), np.eye(3), np.zeros((6, 6)), (50.0, 50.0), (-40.0, 40.0, -40.0, 40.0))

    src = np.array([[0.0, 0.0, -100.0], [1.0, 1.0, -100.0], [-1.0, 0.5, -100.0]])
    target_eye = np.array([[0.0, 0.0, 100.0], [1.0, 1.0, 100.0], [-1.0, 0.5, 100.0]])
    init_dir = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    # 1. Chạy đường dẫn reference
    rt_ref = ExecutionRuntime(mode="accelerated", backend_plan={"forward_active_fd": "reference"})
    token = _CURRENT_RUNTIME.set(rt_ref)
    try:
        ref_res = forward_shoot(src, target_eye, init_dir, display, m2, m1, visor,
                                max_iter=8, tolerance_mm=0.01, fd_angle=1e-4, damping=0.5)
    finally:
        _CURRENT_RUNTIME.reset(token)

    # 2. Chạy đường dẫn active-only CPU
    rt_cpu = ExecutionRuntime(mode="accelerated", backend_plan={"forward_active_fd": "cpu"})
    token = _CURRENT_RUNTIME.set(rt_cpu)
    try:
        cpu_res = forward_shoot(src, target_eye, init_dir, display, m2, m1, visor,
                                max_iter=8, tolerance_mm=0.01, fd_angle=1e-4, damping=0.5)
    finally:
        _CURRENT_RUNTIME.reset(token)

    bool_pass = (
        np.array_equal(ref_res["valid"], cpu_res["valid"])
        and np.array_equal(ref_res["converged"], cpu_res["converged"])
    )

    float_keys = [
        "parameters", "direction", "direction_after_m2", "direction_after_m1",
        "m2", "m1", "visor", "arrive_direction", "eye_hit", "residual_yz", "residual_mm"
    ]
    float_pass = True
    for k in float_keys:
        if not np.allclose(ref_res[k], cpu_res[k], rtol=1e-11, atol=1e-10, equal_nan=True):
            float_pass = False
            break

    passed = bool(bool_pass and float_pass)
    return passed, 1 if passed else 0, 0 if passed else 1, 0, "FORWARD_ACTIVE_FD_PARITY_CPU"


def qualify_fermat_solver_active() -> tuple[bool, int, int, int, str]:
    """Kiểm tra parity của Fermat solver giữa reference và active/fused loop (CPU và CUDA nếu có)."""
    from core import PolySurface, solve_fermat_m2
    from execution_v55 import ExecutionRuntime, _CURRENT_RUNTIME

    # 1. Planar fixture
    m2_plane = PolySurface.plane("M2_PLANE", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (100.0, 100.0))
    q1_plane = np.array([
        [-1.0, 0.0, 2.0],
        [-2.0, 1.0, 3.0],
        [-1.0, 0.0, 2.0],  # transmission / wrong reflection branch
    ])
    target_plane = np.array([
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 3.0],
        [1.0, 0.0, -2.0],  # wrong branch
    ])
    init_plane = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
    ])

    # 2. Curved PolySurface fixture with early/late converging rays
    m2_curved = PolySurface("M2_CURVED", np.array([0.0, 0.0, 0.0]), np.eye(3), (50.0, 50.0),
                            terms=[(0, 0), (2, 0), (0, 2), (1, 1), (3, 0)],
                            coeff=np.array([0.0, 0.001, 0.001, 0.0002, 0.00005]),
                            scale=np.array([50.0, 50.0]),
                            curvature=0.002, conic=-0.5)
    t = np.linspace(-0.8, 0.8, 10)
    q1_curved = np.column_stack((-30.0 + 2.0 * t, 5.0 * t, np.full(10, 35.0)))
    target_curved = np.column_stack((28.0 - 3.0 * t, -4.0 * t, np.full(10, 30.0)))
    init_curved = np.column_stack((5.0 * t, 2.0 * t, np.zeros(10)))

    fixtures = [
        ("planar", m2_plane, q1_plane, target_plane, init_plane),
        ("curved", m2_curved, q1_curved, target_curved, init_curved),
    ]

    bool_str_keys = [
        "success", "gradient_pass", "reflection_pass",
        "wrong_reflection_branch", "conic_domain_pass",
        "finite_pass", "termination_reason"
    ]
    float_keys = [
        "target_points", "xy", "gradient", "gradient_norm", "reflection_residual"
    ]

    all_passed = True
    cases_passed = 0
    cases_failed = 0

    for name, m2, q1, targets, init in fixtures:
        # Reference execution
        rt_ref = ExecutionRuntime(mode="accelerated", backend_plan={"fermat_eval": "reference"})
        token = _CURRENT_RUNTIME.set(rt_ref)
        try:
            ref_sol = solve_fermat_m2(q1, targets, m2, init, max_iter=30, grad_tol=1e-7, reflection_tolerance=1e-4)
        finally:
            _CURRENT_RUNTIME.reset(token)

        # CPU accelerated execution
        rt_cpu = ExecutionRuntime(mode="accelerated", backend_plan={"fermat_eval": "cpu"})
        token = _CURRENT_RUNTIME.set(rt_cpu)
        try:
            cpu_sol = solve_fermat_m2(q1, targets, m2, init, max_iter=30, grad_tol=1e-7, reflection_tolerance=1e-4)
        finally:
            _CURRENT_RUNTIME.reset(token)

        passed = True
        for k in bool_str_keys:
            if k in ref_sol and k in cpu_sol:
                if not np.array_equal(ref_sol[k], cpu_sol[k]):
                    passed = False
                    break
        for k in float_keys:
            if k in ref_sol and k in cpu_sol:
                if not np.allclose(ref_sol[k], cpu_sol[k], rtol=1e-11, atol=1e-10, equal_nan=True):
                    passed = False
                    break

        if passed:
            cases_passed += 1
        else:
            cases_failed += 1
            all_passed = False

    return (
        all_passed,
        cases_passed,
        cases_failed,
        0,
        "FERMAT_SOLVER_ACTIVE_FUSED_PARITY",
    )


def run_all_qualifications(source_dir: Path) -> dict[str, Any]:
    """Chạy toàn bộ bài test qualification và ghi hồ sơ."""
    fingerprint = compute_software_fingerprint(source_dir)
    report_folder = source_dir / "performance_profiles"
    report_folder.mkdir(parents=True, exist_ok=True)

    test_funcs = {
        "plane_batch": ("cpu", qualify_plane_batch),
        "fixed_design_lstsq": ("cpu", qualify_fixed_design_lstsq),
        "point_by_point_ci": ("cpu", qualify_point_by_point_ci),
        "cheb_ray_equations": ("cpu", qualify_cheb_ray_equations),
        "fermat_eval": ("cpu", qualify_fermat_eval),
        "fermat_solver_active": ("cpu", qualify_fermat_solver_active),
        "forward_active_fd": ("cpu", qualify_forward_active_fd),
    }

    test_report: dict[str, Any] = {
        "schema": "HUD_QUALIFICATION_TEST_REPORT_V1",
        "created_utc": time.time(),
        "tests": {},
    }
    kernel_results: dict[str, Any] = {}

    for k, (backend, fn) in test_funcs.items():
        try:
            passed, cases_passed, cases_failed, cases_skipped, scope = fn()
            test_report["tests"][k] = {
                "passed": passed,
                "cases_passed": cases_passed,
                "cases_failed": cases_failed,
                "cases_skipped": cases_skipped,
                "tested_scope": scope,
            }
            kernel_results[k] = {
                "backend": backend,
                "passed": passed,
                "cases_passed": cases_passed,
                "cases_failed": cases_failed,
                "cases_skipped": cases_skipped,
                "tested_scope": scope,
            }
        except Exception as exc:
            test_report["tests"][k] = {
                "passed": False,
                "error": str(exc),
            }
            kernel_results[k] = {
                "backend": backend,
                "passed": False,
                "cases_passed": 0,
                "cases_failed": 1,
                "cases_skipped": 0,
                "tested_scope": f"FAILED_{exc}",
            }

    fermat_cpu = kernel_results.get(
        "fermat_eval", {}
    )

    if fermat_cpu.get("passed", False):
        try:
            (
                cuda_passed,
                cuda_cases_passed,
                cuda_cases_failed,
                cuda_cases_skipped,
                cuda_scope,
            ) = qualify_fermat_eval_cuda(source_dir)

            test_report["tests"]["fermat_eval_cuda"] = {
                "passed": cuda_passed,
                "cases_passed": cuda_cases_passed,
                "cases_failed": cuda_cases_failed,
                "cases_skipped": cuda_cases_skipped,
                "tested_scope": cuda_scope,
            }

            if cuda_passed:
                kernel_results["fermat_eval"] = {
                    "backend": "cuda",
                    "passed": True,
                    "cases_passed": cuda_cases_passed,
                    "cases_failed": 0,
                    "cases_skipped": cuda_cases_skipped,
                    "tested_scope": cuda_scope,
                }

        except Exception as exc:
            test_report["tests"]["fermat_eval_cuda"] = {
                "passed": False,
                "cases_passed": 0,
                "cases_failed": 0,
                "cases_skipped": 1,
                "tested_scope":
                    "CUDA_UNAVAILABLE_OR_QUALIFICATION_FAILED",
                "error":
                    f"{type(exc).__name__}:{exc}",
            }

    report_path = report_folder / "qualification_test_report.json"
    report_bytes = json.dumps(test_report, indent=2).encode("utf-8")
    report_path.write_bytes(report_bytes)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()

    cuda_identity = {
        "cupy_installed": False,
        "cuda_device_count": 0,
        "notes": "CUDA_NOT_AVAILABLE_TESTS_SKIPPED_QUALIFIED_AS_CPU_ONLY",
    }
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        cuda_identity["cupy_installed"] = True
        cuda_identity["cuda_device_count"] = cp.cuda.runtime.getDeviceCount()
        cuda_identity["device_name"] = str(props.get("name", ""))
    except Exception:
        pass

    qualification_doc = {
        "schema": "HUD_BACKEND_QUALIFICATION_V1",
        "software_fingerprint": fingerprint,
        "test_report_path": str(report_path.resolve()),
        "test_report_sha256": report_sha256,
        "kernel_results": kernel_results,
        "cuda_identity": cuda_identity,
    }

    profile_path = report_folder / "qualified_backend.json"
    profile_path.write_text(json.dumps(qualification_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"QUALIFICATION COMPLETE -> {profile_path}")
    return qualification_doc


if __name__ == "__main__":
    document = run_all_qualifications(Path(__file__).resolve().parent)
    if not document["kernel_results"]["point_by_point_ci"]["passed"]:
        raise SystemExit(1)
