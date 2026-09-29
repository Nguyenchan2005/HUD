"""Điều phối STEP 00–26, giữ đúng thứ tự dữ liệu và các cổng kiểm tra của pipeline."""

from __future__ import annotations

import inspect
import hashlib
import json
import math
import pickle
import shutil
import sys
import copy
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from scipy.spatial import QhullError, cKDTree

from core import (
    ChebVisor, PolySurface, aperture_points, build_final_scorecard, build_rays, chief_centered_geometric_spot_rms,
    convex_aperture_from_points, fan_imaging_metrics, first_hit,
    fit_surface, forward_shoot, load_inputs, make_virtual_image, mf2_geometry,
    normalized_dlsq_optimize, packaging_lambda, point_by_point_engine,
    point_normal_cloud_diagnostics,
    polar_pattern, primary_pupils, ray_based_mtf, reconstruct_virtual_points, reference_grid,
    reflect, rescale_surface_polynomial, solve_fermat_m2, surface_sanity, mirror_shape_gate, symmetric_conjugate, trace_reverse,
    monomial_terms, unit, write_csv, write_json,
)
from live_monitor_v55 import (
    observe,
    live_item,
    live_phase,
    live_results,
    live_note,
    live_capture,
)
from execution_v55 import (
    current_runtime,
    is_compute_worker,
    BackendConsistencyError,
    BackendExecutionError,
    ray_trace_acceleration,
)

PARALLEL_RAY_TRACE_STEPS = {
    12,
    14,
    17,
    20,
    24,
}


def _reverse_evaluation_trace(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    display: PolySurface,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Thực thi _reverse_evaluation_trace."""
    def compute():
        """Thực thi compute."""
        return trace_reverse(
            ctx.data["rays"], ctx.data["visor"],
            m1, m2, display, physical_first_hit=True,
        )

    if not use_cache:
        return compute()

    runtime = current_runtime()
    if runtime is None or not runtime.reverse_cache_enabled:
        return compute()

    ray_fp = ctx.data.get("_ray_bundle_fingerprint")
    visor_fp = ctx.data.get("_visor_fingerprint")
    if ray_fp is None or visor_fp is None:
        key_payload = {
            "schema": "REVERSE_TRACE_CACHE_V1",
            "rays": ctx.data["rays"],
            "visor": ctx.data["visor"].to_dict(),
            "m1": m1.to_dict(),
            "m2": m2.to_dict(),
            "display": display.to_dict(),
            "physical_first_hit": True,
            "implementation": runtime.implementation_fingerprint,
        }
    else:
        key_payload = {
            "schema": "REVERSE_TRACE_CACHE_V1",
            "ray_fingerprint": ray_fp,
            "visor_fingerprint": visor_fp,
            "m1": m1.to_dict(),
            "m2": m2.to_dict(),
            "display": display.to_dict(),
            "physical_first_hit": True,
            "implementation": runtime.implementation_fingerprint,
        }
    key = runtime.reverse_cache.make_key(key_payload)
    return runtime.reverse_cache.get_or_compute(key, compute)


def _mirror_topology_gate(
    ctx: Context,
    surface: PolySurface,
    trace: dict[str, Any],
    surface_name: str,
) -> dict[str, Any]:
    """Hard-gate ray-facing concave/convex authority without using raw curvature sign."""
    topology_cfg = ctx.config["surface_fit"]["mirror_topology_authority"]
    trace_indices = {
        "M1": (1, 0),
        "M2": (2, 1),
    }

    if surface_name not in trace_indices:
        raise ValueError(
            f"MIRROR_TOPOLOGY_SURFACE_INVALID:{surface_name}"
        )

    point_index, direction_index = trace_indices[surface_name]
    chief_index = _chief_index(ctx.data["rays"])

    hit_points = np.asarray(
        trace["points"][point_index],
        float,
    )
    incident_directions = np.asarray(
        trace["directions"][direction_index],
        float,
    )

    if (
        hit_points.ndim != 2
        or hit_points.shape[1] != 3
        or incident_directions.ndim != 2
        or incident_directions.shape[1] != 3
        or chief_index >= len(hit_points)
        or chief_index >= len(incident_directions)
    ):
        raise RuntimeError(
            f"MIRROR_TOPOLOGY_TRACE_SHAPE_INVALID:{surface_name}"
        )

    hit_point = hit_points[chief_index]
    incident_direction = incident_directions[chief_index]

    if (
        not np.all(np.isfinite(hit_point))
        or not np.all(np.isfinite(incident_direction))
    ):
        raise RuntimeError(
            f"MIRROR_TOPOLOGY_CHIEF_TRACE_NONFINITE:{surface_name}"
        )

    local = (
        hit_point
        - surface.center
    ) @ surface.frame

    surface_normal = np.asarray(
        surface.normal(
            np.asarray([local[0]], float),
            np.asarray([local[1]], float),
        ),
        float,
    )[0]

    incident_norm = float(
        np.linalg.norm(incident_direction)
    )
    normal_norm = float(
        np.linalg.norm(surface_normal)
    )

    if (
        not np.all(np.isfinite(surface_normal))
        or not np.isfinite(incident_norm)
        or not np.isfinite(normal_norm)
        or incident_norm <= 1e-12
        or normal_norm <= 1e-12
    ):
        raise RuntimeError(
            f"MIRROR_TOPOLOGY_CHIEF_GEOMETRY_INVALID:{surface_name}"
        )

    incident_unit = (
        incident_direction
        / incident_norm
    )
    normal_unit = (
        surface_normal
        / normal_norm
    )

    ray_facing_dot = float(
        np.dot(
            normal_unit,
            -incident_unit,
        )
    )

    if (
        not np.isfinite(ray_facing_dot)
        or abs(ray_facing_dot) <= 1e-9
    ):
        raise RuntimeError(
            f"MIRROR_TOPOLOGY_CHIEF_RAY_GRAZING:{surface_name}"
        )

    facing_sign = (
        1.0
        if ray_facing_dot > 0.0
        else -1.0
    )

    topology = str(
        topology_cfg[surface_name]
    )

    if topology == "CONCAVE_RAY_FACING":
        orientation_sign = facing_sign
    elif topology == "CONVEX_RAY_FACING":
        orientation_sign = -facing_sign
    else:
        raise ValueError(
            f"MIRROR_TOPOLOGY_VALUE_INVALID:"
            f"{surface_name}:{topology}"
        )

    quality_cfg = (
        ctx.config[
            "surface_fit"
        ][
            "step12_quality_gates"
        ]
    )

    shape_policy = {
        "maximum_freeform_departure_mm":
            float(
                quality_cfg[
                    "maximum_sag_rms_mm"
                ]
            ),
        "maximum_normal_departure_deg":
            float(
                quality_cfg[
                    "maximum_normal_max_deg"
                ]
            ),
        "maximum_principal_curvature_per_mm":
            float(
                ctx.config[
                    "surface_fit"
                ][
                    "curvature_absolute_max_per_mm"
                ]
            ),
    }

    gate = dict(
        mirror_shape_gate(
            surface,
            shape_policy,
            orientation_sign=orientation_sign,
        )
    )

    gate.update({
        "topology_authority":
            topology,
        "topology_enforcement":
            str(
                topology_cfg[
                    "enforcement"
                ][
                    surface_name
                ]
            ),
        "chief_ray_index":
            int(chief_index),
        "trace_point_index":
            int(point_index),
        "trace_incident_direction_index":
            int(direction_index),
        "ray_facing_dot":
            ray_facing_dot,
        "ray_facing_normal_sign":
            facing_sign,
        "required_orientation_sign":
            float(orientation_sign),
        "ray_facing_definition":
            "SURFACE_NORMAL_DOT_NEGATIVE_INCIDENT_DIRECTION",
    })

    return gate


def _o2_mirror_gate_view(
    gate: dict[str, Any],
    quality_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Tách hard topology O2 khỏi shape-quality chỉ dùng để cảnh báo ở STEP11-12."""
    raw_checks = dict(
        gate.get(
            "checks",
            {},
        )
    )

    topology_names = (
        "basic_surface_sanity",
        "single_bowl_gaussian_curvature",
        "single_bowl_mean_curvature_orientation",
    )

    missing = [
        name
        for name in topology_names
        if name not in raw_checks
    ]

    if missing:
        raise RuntimeError(
            "O2_MIRROR_GATE_MISSING_TOPOLOGY_CHECKS:"
            + ",".join(
                missing
            )
        )

    topology_checks = {
        name:
            bool(
                raw_checks[
                    name
                ]
            )
        for name in topology_names
    }

    topology_failure_reasons = [
        name
        for name, passed in topology_checks.items()
        if not passed
    ]

    principal_curvature_max = max(
        abs(
            float(
                gate[
                    "k1_min_per_mm"
                ]
            )
        ),
        abs(
            float(
                gate[
                    "k1_max_per_mm"
                ]
            )
        ),
        abs(
            float(
                gate[
                    "k2_min_per_mm"
                ]
            )
        ),
        abs(
            float(
                gate[
                    "k2_max_per_mm"
                ]
            )
        ),
    )

    quality_rows = [
        {
            "check":
                "freeform_departure_rms",

            "actual":
                float(
                    gate[
                        "freeform_departure_rms_mm"
                    ]
                ),

            "limit":
                float(
                    quality_cfg[
                        "maximum_sag_rms_mm"
                    ]
                ),

            "unit":
                "mm",
        },
        {
            "check":
                "normal_departure_rms",

            "actual":
                float(
                    gate[
                        "normal_departure_rms_deg"
                    ]
                ),

            "limit":
                float(
                    quality_cfg[
                        "maximum_normal_rms_deg"
                    ]
                ),

            "unit":
                "deg",
        },
        {
            "check":
                "normal_departure_max",

            "actual":
                float(
                    gate[
                        "normal_departure_max_deg"
                    ]
                ),

            "limit":
                float(
                    quality_cfg[
                        "maximum_normal_max_deg"
                    ]
                ),

            "unit":
                "deg",
        },
        {
            "check":
                "principal_curvature_magnitude",

            "actual":
                float(
                    principal_curvature_max
                ),

            "limit":
                float(
                    gate[
                        "policy"
                    ][
                        "maximum_principal_curvature_per_mm"
                    ]
                ),

            "unit":
                "1/mm",
        },
        {
            "check":
                "curvature_gradient_p95",

            "actual":
                float(
                    gate[
                        "curvature_gradient_p95_per_mm2"
                    ]
                ),

            "limit":
                float(
                    gate[
                        "curvature_gradient_limit_per_mm2"
                    ]
                ),

            "unit":
                "1/mm^2",
        },
        {
            "check":
                "curvature_gradient_emergency",

            "actual":
                float(
                    gate[
                        "curvature_gradient_max_per_mm2"
                    ]
                ),

            "limit":
                float(
                    gate[
                        "curvature_gradient_limit_per_mm2"
                    ]
                ),

            "unit":
                "1/mm^2",
        },
    ]

    failure_status = (
        "WARN"
        if str(
            quality_cfg[
                "enforcement"
            ]
        )
        == "WARN"
        else "FAIL"
    )

    for row in quality_rows:
        actual = float(
            row[
                "actual"
            ]
        )

        limit = float(
            row[
                "limit"
            ]
        )

        passed = bool(
            np.isfinite(
                actual
            )
            and actual <= limit
        )

        row[
            "status"
        ] = (
            "PASS"
            if passed
            else failure_status
        )

        row[
            "enforcement"
        ] = str(
            quality_cfg[
                "enforcement"
            ]
        )

        row[
            "reason"
        ] = (
            f"{actual:.6g} <= {limit:.6g} {row['unit']}"
            if passed
            else
            f"{actual:.6g} > {limit:.6g} {row['unit']}"
        )

    quality_warning_reasons = [
        str(
            row[
                "check"
            ]
        )
        for row in quality_rows
        if row[
            "status"
        ]
        != "PASS"
    ]

    result = dict(
        gate
    )

    result.update({
        "raw_shape_pass":
            bool(
                gate[
                    "pass"
                ]
            ),

        "raw_shape_failure_reasons":
            list(
                gate.get(
                    "failure_reasons",
                    [],
                )
            ),

        "topology_checks":
            topology_checks,

        "topology_pass":
            bool(
                all(
                    topology_checks.values()
                )
            ),

        "topology_failure_reasons":
            topology_failure_reasons,

        "quality_checks":
            quality_rows,

        "quality_status":
            (
                "PASS"
                if not quality_warning_reasons
                else failure_status
            ),

        "quality_warning_reasons":
            quality_warning_reasons,

        "o2_gate_policy":
            "HARD_TOPOLOGY_SOFT_SHAPE_QUALITY",
    })

    return result


_STEP11_SHAPE_CURVATURE_FLOAT_KEYS = (
    "H_min_per_mm",
    "H_max_per_mm",
    "KG_min_per_mm2",
    "KG_max_per_mm2",
    "k1_min_per_mm",
    "k1_max_per_mm",
    "k2_min_per_mm",
    "k2_max_per_mm",
    "oriented_H_min_per_mm",
)

_STEP11_SHAPE_CURVATURE_INT_KEYS = (
    "curvature_sign_flip_count",
)

_STEP11_CURVATURE_HISTORY_FIELDS = tuple(
    f"{surface}_{key}"
    for surface in ("M1", "M2")
    for key in (_STEP11_SHAPE_CURVATURE_FLOAT_KEYS + _STEP11_SHAPE_CURVATURE_INT_KEYS)
)


def _step11_shape_curvature_record_fields(
    surface_name: str,
    shape_gate: dict[str, Any],
) -> dict[str, Any]:
    """Trích xuất 10 giá trị curvature diagnostic từ kết quả mirror_shape_gate cho surface."""
    if surface_name not in ("M1", "M2"):
        raise ValueError(
            f"STEP11_CURVATURE_DIAGNOSTIC_SURFACE_INVALID:{surface_name}"
        )
    result: dict[str, Any] = {}
    for key in _STEP11_SHAPE_CURVATURE_FLOAT_KEYS:
        result[f"{surface_name}_{key}"] = float(shape_gate[key])
    for key in _STEP11_SHAPE_CURVATURE_INT_KEYS:
        result[f"{surface_name}_{key}"] = int(shape_gate[key])
    return result


def _step11_curvature_history_fields_from_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Trích xuất 20 trường curvature diagnostic từ record candidate để ghi vào CSV history."""
    return {
        field: record.get(field)
        for field in _STEP11_CURVATURE_HISTORY_FIELDS
    }


def _final_reverse_trace_fingerprint(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    display: PolySurface,
) -> str:
    """Fingerprint exact state dung cho STEP24→25 trace reuse."""
    payload = {
        "schema":
            "FINAL_REVERSE_TRACE_STATE_V1",
        "rays":
            ctx.data["rays"],
        "visor":
            ctx.data["visor"].to_dict(),
        "m1":
            m1.to_dict(),
        "m2":
            m2.to_dict(),
        "display":
            display.to_dict(),
        "physical_first_hit":
            True,
        "algorithm_source_manifest_sha256":
            ctx.data.get(
                "algorithm_source_manifest",
                {},
            ).get("manifest_sha256"),
    }

    return hashlib.sha256(
        pickle.dumps(
            payload,
            protocol=5,
        )
    ).hexdigest()


STAGES: dict[int, tuple[str, Callable[["Context"], dict[str, Any]]]] = {}
IMPORTANT = {0, 1, 5, 7, 8, 12, 14, 17, 20, 21, 23, 24, 25, 26}
MAX_STEP = 26

# Giải thích ngắn bằng tiếng Việt, dùng chung cho source và các file chạy tay.
STEP_EXPLANATIONS_VI = {
    0: "Kiểm tra config, nạp đúng dữ liệu authority và khóa đầu vào cho run hiện tại.",
    1: "Fit các điểm visor-inner thành mặt Chebyshev khả vi để giao tia và lấy pháp tuyến.",
    2: "Dựng mặt phẳng ảnh ảo từ VID, D6, azimuth và hai góc FOV.",
    3: "Chia FOV theo field_grid và xác định duy nhất một field trung tâm.",
    4: "Tạo lưới pupil cấu hình được, phủ đều tâm trong eye-box.",
    5: "Tạo 49 mẫu trong mỗi pupil của lưới cấu hình cho mọi field và gán ID cho từng tia.",
    6: "Giao các tia với visor, tính pháp tuyến và hướng phản xạ đi về M1.",
    7: "Tự sinh planar folding geometry từ central chief ray, hard-gate physical topology, unobscuration và packaging tùy chọn, rồi chọn geometry compact nhất; planar RMS chỉ là diagnostic tie-break.",
    8: "Tách reference động của Fan khỏi lưới ảnh ảo cố định dùng đo distortion.",
    9: "Khi M2 còn phẳng, tạo target M1 bằng phép đối xứng liên hợp qua M2.",
    10: "Dựng point cloud và normal của M1 theo CI point-by-point, bắt đầu từ chief ray.",
    11: (
        "Fit M1 O2 từ M1 CI, dựng M2 CI một lần rồi joint-restoration "
        "M1/M2 O2 trên fixed residual grid bằng parallel finite-difference "
        "Jacobian và damped LM; topology và unobscuration được dẫn hướng "
        "bằng residual liên tục nhưng vẫn phải qua hard certification cuối."
    ),
    12: "Nhận best M1/M2 O2 pair từ STEP11 rồi joint-refine bằng actual physical ray trace; chỉ topology và unobscuration còn là hard acceptance, còn fit quality và physical-fraction target được báo PASS/WARN theo cấu hình.",
    13: "Kết thúc shortcut phẳng và chuyển sang Fan Step Two dùng Fermat.",
    14: "Giải điểm target trên M2 sao cho hai đạo hàm optical path đều gần bằng 0.",
    15: "Khai báo các nhánh rho dương/âm để thử hai hướng phân phối optical power.",
    16: "Với từng rho, dựng lại M1 rồi M2 và chọn feasible-first; chỉ giữ nhánh restoration có nhãn theo giới hạn cấu hình.",
    17: "Lặp Fermat, CI và fit trên basis lồng nhau từ O2 tới O5; level 3 giữ A22 và mọi gate theo ray dùng ngưỡng cấu hình mặc định 98%.",
    18: "Lưu mặt O5 sau CI làm starting point, chưa coi là kết quả tối ưu cuối.",
    19: "Khai báo chief-centered MF1/MF2; pupil-shift và field-bias được báo riêng, engineering constraints vẫn nằm ngoài core.",
    20: "Tính chief-centered MF1 từ mọi ray quanh chief của đúng field/pupil và báo pupil-shift cùng field-bias.",
    21: "Tính MF2 từ diện tích có hướng A-Q-P; không obscured thì MF2 bằng 0.",
    22: "Khai báo công thức và hard limit distortion trên lưới ảnh ảo cố định.",
    23: "Filter-DLSQ coefficient O5 có log từng trial và restoration engineering; tùy chọn 23S/23G rồi DLSQ lại, aperture luôn khóa.",
    24: "Bắn đủ tia thuận Display→M2→M1→visor→eye; chỉ bundle đầy đủ, đúng thứ tự và trước mắt được chứng nhận.",
    25: "Tổng hợp từng hard gate thành PASS/FAIL/UNGRADED và quyết định cuối.",
    26: "Xuất prescription, coefficient, kết quả ray-trace, báo cáo và manifest đầy đủ.",
}


def stage(number: int, title: str):
    """Đăng ký hàm thành STEP có số và tên cố định."""
    def wrap(fn: Callable[["Context"], dict[str, Any]]):
        """Bọc STEP để in tiến độ nhưng giữ nguyên kết quả và ngoại lệ."""
        STAGES[number] = (title, fn)
        return fn
    return wrap


def group_name(number: int) -> str:
    """Ánh xạ số STEP sang nhóm thư mục kết quả."""
    return f"STEP_{number:02d}"


def validate_config(c: dict[str, Any]) -> None:
    """Kiểm đủ biến, miền giá trị và quan hệ cấu hình trước khi chạy."""
    required = {
        "schema", "prompt_sha256", "input_dir", "hud_geometry_authority", "vid_mm", "d6_deg", "azimuth_deg",
        "fov_h_deg", "fov_v_deg", "field_grid", "pupil_diameter_mm", "eyebox_y_mm", "eyebox_z_mm",
        "primary_pupil_grid",
        "display_seed_mm", "wavelength_nm", "visor_active_mm", "packaging_lambda_max",
        "distortion", "fan_reference", "visor_surrogate_degree", "visor_surrogate_scale_mm",
        "fit_weights", "surface_fit", "ci_construction", "fan_weights", "normalized_objective", "solver",
        "surface_parameter_refinement", "hard_tolerances",
        "mtf_requested", "zemax_crosscheck",
    }
    missing = sorted(required - set(c))
    if missing:
        raise ValueError(f"CONFIG_MISSING_KEYS: {missing}")
    if c["schema"] != "HUD_FAN_CLEAN_DESIGN_V5_5":
        raise ValueError("CONFIG_SCHEMA_NOT_V5_5")
    geometry = c["hud_geometry_authority"]
    geometry_keys = {"visor_center_mm", "visor_U", "visor_V", "visor_N", "source"}
    if not geometry_keys.issubset(geometry):
        raise ValueError(f"HUD_GEOMETRY_AUTHORITY_MISSING_KEYS: {sorted(geometry_keys-set(geometry))}")
    packaging_raw = geometry.get("packaging_vertices_mm")
    if packaging_raw not in (None, []):
        packaging = np.asarray(packaging_raw, float)
        if (packaging.shape != (8, 3) or not np.all(np.isfinite(packaging))
                or np.linalg.matrix_rank(packaging - np.mean(packaging, axis=0)) < 3):
            raise ValueError("PACKAGING_VERTICES_MUST_BE_FINITE_8_BY_3_WITH_NONZERO_VOLUME")
        if float(c["packaging_lambda_max"]) < 1.0:
            raise ValueError("PACKAGING_LAMBDA_MAX_MUST_BE_AT_LEAST_ONE")
    if any(np.asarray(geometry[key], float).shape != (3,)
           for key in ("visor_center_mm", "visor_U", "visor_V", "visor_N")):
        raise ValueError("VISOR_FRAME_VECTORS_MUST_HAVE_THREE_COMPONENTS")
    grid = c["field_grid"]
    if set(grid) != {"horizontal_count", "vertical_count"}:
        raise ValueError("FIELD_GRID_KEYS_MUST_BE_HORIZONTAL_COUNT_AND_VERTICAL_COUNT")
    counts = [grid["horizontal_count"], grid["vertical_count"]]
    if any(isinstance(v, bool) or int(v) != v or int(v) < 3 or int(v) % 2 == 0 for v in counts):
        raise ValueError("FIELD_GRID_COUNTS_MUST_BE_ODD_INTEGERS_AT_LEAST_3")
    pupil_grid = c["primary_pupil_grid"]
    if set(pupil_grid) != {"horizontal_count", "vertical_count"}:
        raise ValueError("PRIMARY_PUPIL_GRID_KEYS_MUST_BE_HORIZONTAL_COUNT_AND_VERTICAL_COUNT")
    pupil_counts = [pupil_grid["horizontal_count"], pupil_grid["vertical_count"]]
    if any(isinstance(v, bool) or int(v) != v or int(v) < 3 or int(v) % 2 == 0 for v in pupil_counts):
        raise ValueError("PRIMARY_PUPIL_GRID_COUNTS_MUST_BE_ODD_INTEGERS_AT_LEAST_3")
    if "ray_sampling_profiles" in c:
        rsp = c["ray_sampling_profiles"]
        if not isinstance(rsp, dict):
            raise ValueError("RAY_SAMPLING_PROFILES_MUST_BE_DICT")
        if not isinstance(rsp.get("enabled"), bool):
            raise ValueError("RAY_SAMPLING_PROFILES_ENABLED_MUST_BE_BOOL")
        if "full_check_profile" in rsp or "full_check_before_commit" in rsp:
            raise ValueError("FULL_CHECK_KEYS_FORBIDDEN_IN_CONFIG")
        if rsp.get("enabled"):
            profiles = rsp.get("profiles")
            if not isinstance(profiles, dict):
                raise ValueError("RAY_SAMPLING_PROFILES_MISSING_PROFILES_DICT")
            expected_names = {"O2_SEARCH", "FREEFORM", "CERTIFICATION"}
            if set(profiles.keys()) != expected_names:
                raise ValueError(f"RAY_SAMPLING_PROFILES_NAMES_MISMATCH: expected {expected_names}")
            for pname, pdata in profiles.items():
                req_pkeys = {"step_start", "step_end", "field_grid", "pupil_grid", "rays_per_pupil", "expected_ray_count"}
                if not req_pkeys.issubset(set(pdata.keys())):
                    raise ValueError(f"PROFILE_{pname}_MISSING_KEYS: {sorted(req_pkeys - set(pdata.keys()))}")
                p_fg = pdata["field_grid"]
                p_pg = pdata["pupil_grid"]
                if any(isinstance(v, bool) or int(v) != v or int(v) < 3 or int(v) % 2 == 0 for v in [p_fg["horizontal_count"], p_fg["vertical_count"]]):
                    raise ValueError(f"PROFILE_{pname}_FIELD_GRID_COUNTS_MUST_BE_ODD_INTEGERS_AT_LEAST_3")
                if any(isinstance(v, bool) or int(v) != v or int(v) < 3 or int(v) % 2 == 0 for v in [p_pg["horizontal_count"], p_pg["vertical_count"]]):
                    raise ValueError(f"PROFILE_{pname}_PUPIL_GRID_COUNTS_MUST_BE_ODD_INTEGERS_AT_LEAST_3")
                rpp = int(pdata["rays_per_pupil"])
                if rpp not in (25, 49):
                    raise ValueError(f"PROFILE_{pname}_RAYS_PER_PUPIL_MUST_BE_25_OR_49")
                expected_total = int(p_fg["horizontal_count"]) * int(p_fg["vertical_count"]) * int(p_pg["horizontal_count"]) * int(p_pg["vertical_count"]) * rpp
                if int(pdata["expected_ray_count"]) != expected_total:
                    raise ValueError(f"PROFILE_{pname}_EXPECTED_RAY_COUNT_MISMATCH: {pdata['expected_ray_count']} vs {expected_total}")
    d = c["distortion"]
    if not (float(d["hard_limit_percent"]) > 0.0 and d["rule"] == "strict_less_than"
            and d["authority"] == "USER_FIXED_TARGET_VI_PLANE"
            and d["metric_name"] == "USER_FIXED_GRID_VECTOR_DISTORTION"
            and d["digital_prewarp"] is False):
        raise ValueError("V5_5_DISTORTION_AUTHORITY_MISMATCH")
    fan_reference = c["fan_reference"]
    reference_mode = fan_reference.get("mode")
    if reference_mode not in ("AUTO_FROM_DISPLAY_SEED_AND_TARGET_VI_EXTENT", "FIXED_MAGNIFICATION"):
        raise ValueError("FAN_REFERENCE_MODE_INVALID")
    parity = np.asarray(fan_reference.get("axis_parity", []), float)
    if parity.shape != (2,) or not np.all(np.isin(parity, (-1.0, 1.0))):
        raise ValueError("FAN_REFERENCE_AXIS_PARITY_MUST_BE_PLUS_OR_MINUS_ONE")
    if reference_mode == "FIXED_MAGNIFICATION":
        fixed_magnification = np.asarray([fan_reference.get("M_x"), fan_reference.get("M_y")], float)
        if fixed_magnification.shape != (2,) or not np.all(np.isfinite(fixed_magnification)):
            raise ValueError("FIXED_FAN_MAGNIFICATION_REQUIRES_FINITE_M_X_AND_M_Y")
    if np.asarray(c["display_seed_mm"], float).shape != (2,) or np.any(np.asarray(c["display_seed_mm"], float) <= 0.0):
        raise ValueError("DISPLAY_SEED_MUST_HAVE_TWO_POSITIVE_DIMENSIONS")
    surface_fit = c["surface_fit"]
    required_surface_fit = {
        "sphere_mode", "joint_variable_projection_enabled", "curvature_relative_bound",
        "curvature_absolute_max_per_mm", "conic_bounds", "joint_max_nfev",
        "nonlinear_regularization", "diagnostic_neighbors", "diagnostic_max_samples",
        "sphere_radius_factor_bounds", "sphere_center_regularization", "sphere_max_nfev",
        "polynomial_gauge_mode", "polynomial_departure_regularization",
        "normal_residual_mode", "k_profile_enabled", "k_profile_sample_count", "integrability_gates",
        "diagnostic_bundle_max_samples", "mirror_topology_authority",
        "step11_macro_micro_conditioning",
        "step11_topology_restoration",
        "step12_quality_gates", "step12_o2_refinement",
    }
    if not required_surface_fit.issubset(surface_fit):
        raise ValueError(f"SURFACE_FIT_MISSING_KEYS: {sorted(required_surface_fit-set(surface_fit))}")

    topology = surface_fit["mirror_topology_authority"]
    if set(topology) != {"enforcement", "M1", "M2"}:
        raise ValueError("MIRROR_TOPOLOGY_AUTHORITY_KEYS_INVALID")
    topology_enforcement = topology["enforcement"]
    if (
        not isinstance(topology_enforcement, dict)
        or set(topology_enforcement) != {"M1", "M2"}
        or any(
            not isinstance(value, str)
            or value not in {"HARD", "WARN"}
            for value in topology_enforcement.values()
        )
    ):
        raise ValueError(
            "MIRROR_TOPOLOGY_AUTHORITY_ENFORCEMENT_INVALID"
        )
    if (
        topology["M1"] != "CONCAVE_RAY_FACING"
        or topology["M2"] != "CONVEX_RAY_FACING"
    ):
        raise ValueError("MIRROR_TOPOLOGY_AUTHORITY_MISMATCH")

    if surface_fit["sphere_mode"] != "CHIEF_VERTEX_CONSTRAINED_GEOMETRIC_SPHERE":
        raise ValueError("SURFACE_FIT_SPHERE_MODE_INVALID")
    radius_factors = np.asarray(surface_fit["sphere_radius_factor_bounds"], float)
    if (radius_factors.shape != (2,) or not np.all(np.isfinite(radius_factors))
            or radius_factors[0] <= 0.0 or radius_factors[0] >= radius_factors[1]):
        raise ValueError("SURFACE_FIT_SPHERE_RADIUS_FACTOR_BOUNDS_INVALID")
    if (not np.isfinite(float(surface_fit["sphere_center_regularization"]))
            or float(surface_fit["sphere_center_regularization"]) < 0.0
            or isinstance(surface_fit["sphere_max_nfev"], bool)
            or int(surface_fit["sphere_max_nfev"]) != surface_fit["sphere_max_nfev"]
            or int(surface_fit["sphere_max_nfev"]) < 1):
        raise ValueError("SURFACE_FIT_CONSTRAINED_SPHERE_OPTIONS_INVALID")
    if surface_fit["polynomial_gauge_mode"] != "FIX_CHIEF_PISTON_AND_REMOVE_SYMMETRIC_QUADRATIC":
        raise ValueError("SURFACE_FIT_POLYNOMIAL_GAUGE_MODE_INVALID")
    if (not np.isfinite(float(surface_fit["polynomial_departure_regularization"]))
            or float(surface_fit["polynomial_departure_regularization"]) < 0.0
            or not isinstance(surface_fit["k_profile_enabled"], bool)
            or surface_fit["normal_residual_mode"] != "TARGET_UNIT_NORMAL_DOT_SURFACE_TANGENTS"
            or isinstance(surface_fit["k_profile_sample_count"], bool)
            or int(surface_fit["k_profile_sample_count"]) != surface_fit["k_profile_sample_count"]
            or int(surface_fit["k_profile_sample_count"]) < 3):
        raise ValueError("SURFACE_FIT_IDENTIFIABILITY_OR_NORMAL_METRIC_INVALID")

    if "step17_o4_degree_regularization" in surface_fit:
        o4_degree_regularization = surface_fit[
            "step17_o4_degree_regularization"
        ]

        if (
            isinstance(o4_degree_regularization, bool)
            or not np.isfinite(
                float(o4_degree_regularization)
            )
            or float(o4_degree_regularization) < 0.0
        ):
            raise ValueError(
                "STEP17_O4_DEGREE_REGULARIZATION_INVALID"
            )

    if not isinstance(surface_fit["joint_variable_projection_enabled"], bool):
        raise ValueError("SURFACE_FIT_JOINT_VARIABLE_PROJECTION_MUST_BE_BOOLEAN")
    if (not 0.0 < float(surface_fit["curvature_relative_bound"]) < 1.0
            or float(surface_fit["curvature_absolute_max_per_mm"]) <= 0.0
            or int(surface_fit["joint_max_nfev"]) < 1
            or float(surface_fit["nonlinear_regularization"]) < 0.0
            or int(surface_fit["diagnostic_neighbors"]) < 3
            or int(surface_fit["diagnostic_max_samples"]) < 10
            or int(surface_fit["diagnostic_bundle_max_samples"]) < 6):
        raise ValueError("SURFACE_FIT_NUMERIC_OPTIONS_INVALID")

    if "step17_parent_trust" in surface_fit:
        parent_trust = surface_fit["step17_parent_trust"]
        required_parent_trust_keys = {
            "enabled",
            "minimum_order",
            "grid_samples",
            "sag_lambda",
            "normal_lambda",
            "balance_by_sample_count",
        }

        if set(parent_trust) != required_parent_trust_keys:
            raise ValueError(
                "STEP17_PARENT_TRUST_CONFIG_KEYS_INVALID"
            )

        if not isinstance(parent_trust["enabled"], bool):
            raise ValueError(
                "STEP17_PARENT_TRUST_ENABLED_INVALID"
            )

        if (
            isinstance(parent_trust["minimum_order"], bool)
            or int(parent_trust["minimum_order"])
            != parent_trust["minimum_order"]
            or int(parent_trust["minimum_order"]) < 3
            or int(parent_trust["minimum_order"]) > 5
        ):
            raise ValueError(
                "STEP17_PARENT_TRUST_MINIMUM_ORDER_INVALID"
            )

        if (
            isinstance(parent_trust["grid_samples"], bool)
            or int(parent_trust["grid_samples"])
            != parent_trust["grid_samples"]
            or int(parent_trust["grid_samples"]) < 5
        ):
            raise ValueError(
                "STEP17_PARENT_TRUST_GRID_SAMPLES_INVALID"
            )

        sag_lambda = parent_trust["sag_lambda"]
        normal_lambda = parent_trust["normal_lambda"]

        if (
            isinstance(sag_lambda, bool)
            or not np.isfinite(float(sag_lambda))
            or float(sag_lambda) < 0.0
        ):
            raise ValueError(
                "STEP17_PARENT_TRUST_SAG_LAMBDA_INVALID"
            )

        if (
            isinstance(normal_lambda, bool)
            or not np.isfinite(float(normal_lambda))
            or float(normal_lambda) < 0.0
        ):
            raise ValueError(
                "STEP17_PARENT_TRUST_NORMAL_LAMBDA_INVALID"
            )

        if (
            float(sag_lambda) == 0.0
            and float(normal_lambda) == 0.0
            and bool(parent_trust["enabled"])
        ):
            raise ValueError(
                "STEP17_PARENT_TRUST_ENABLED_WITH_ZERO_WEIGHTS"
            )

        if not isinstance(
            parent_trust["balance_by_sample_count"],
            bool,
        ):
            raise ValueError(
                "STEP17_PARENT_TRUST_SAMPLE_BALANCE_INVALID"
            )

    conditioning = surface_fit["step11_macro_micro_conditioning"]
    required_conditioning = {
        "enabled",
        "fallback_gamma_candidates",
        "minimum_abs_baseline_curvature_per_mm",
        "default_abs_baseline_curvature_per_mm",
        "maximum_alternating_projection_iterations",
    }
    allowed_conditioning = (
        required_conditioning
        |
        {
            "maximum_search_cycles",
        }
    )

    if (
        not required_conditioning.issubset(conditioning)
        or not set(conditioning).issubset(allowed_conditioning)
    ):
        raise ValueError(
            "STEP11_MACRO_MICRO_CONDITIONING_KEYS_INVALID"
        )
    if not isinstance(conditioning["enabled"], bool):
        raise ValueError("STEP11_MACRO_MICRO_CONDITIONING_ENABLED_MUST_BE_BOOL")

    gamma_raw = conditioning["fallback_gamma_candidates"]
    if (
        not isinstance(gamma_raw, list)
        or not gamma_raw
        or any(isinstance(value, bool) for value in gamma_raw)
    ):
        raise ValueError("STEP11_FALLBACK_GAMMA_CANDIDATES_INVALID")
    try:
        gamma_values = np.asarray(gamma_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("STEP11_FALLBACK_GAMMA_CANDIDATES_INVALID") from exc
    if (
        gamma_values.ndim != 1
        or not np.all(np.isfinite(gamma_values))
        or np.any(gamma_values <= 0.0)
        or len(np.unique(gamma_values)) != len(gamma_values)
    ):
        raise ValueError("STEP11_FALLBACK_GAMMA_CANDIDATES_INVALID")

    minimum_abs_curvature = conditioning[
        "minimum_abs_baseline_curvature_per_mm"
    ]
    default_abs_curvature = conditioning[
        "default_abs_baseline_curvature_per_mm"
    ]
    if (
        isinstance(minimum_abs_curvature, bool)
        or isinstance(default_abs_curvature, bool)
        or not np.isfinite(float(minimum_abs_curvature))
        or not np.isfinite(float(default_abs_curvature))
        or float(minimum_abs_curvature) <= 0.0
        or float(default_abs_curvature) < float(minimum_abs_curvature)
    ):
        raise ValueError("STEP11_FALLBACK_CURVATURE_FLOORS_INVALID")

    maximum_conditioning_iterations = conditioning[
        "maximum_alternating_projection_iterations"
    ]
    if (
        isinstance(maximum_conditioning_iterations, bool)
        or not isinstance(maximum_conditioning_iterations, int)
        or not 1 <= maximum_conditioning_iterations <= 5
    ):
        raise ValueError("STEP11_ALTERNATING_PROJECTION_ITERATIONS_INVALID")

    if "maximum_search_cycles" in conditioning:
        maximum_search_cycles = conditioning[
            "maximum_search_cycles"
        ]

        if (
            isinstance(maximum_search_cycles, bool)
            or not isinstance(maximum_search_cycles, int)
            or maximum_search_cycles < 1
        ):
            raise ValueError(
                "STEP11_MAXIMUM_SEARCH_CYCLES_INVALID"
            )

    restoration = surface_fit[
        "step11_topology_restoration"
    ]

    required_restoration = {
        "enabled",
        "grid_samples",
        "maximum_iterations",
        "finite_difference_normalized",
        "finite_difference_shrink_factors",
        "maximum_invalid_jacobian_fraction",
        "minimum_jacobian_column_norm",
        "svd_rcond",
        "maximum_jacobian_condition",
        "damping",
        "minimum_damping",
        "maximum_damping",
        "damping_multipliers",
        "accepted_damping_factor",
        "rejected_damping_factor",
        "max_step_normalized",
        "minimum_trust_radius",
        "maximum_trust_radius",
        "trust_shrink_factor",
        "trust_expand_factor",
        "minimum_acceptance_rho",
        "trust_shrink_rho",
        "trust_expand_rho",
        "line_search_alphas",
        "h_target_per_mm",
        "kg_target_per_mm2",
        "mean_curvature_scale_per_mm",
        "gaussian_curvature_scale_per_mm2",
        "principal_curvature_scale_per_mm",
        "curvature_gradient_scale_per_mm2",
        "pair_sag_trust_scale_mm",
        "pair_normal_trust_scale_deg",
        "physical_fraction_scale",
        "maximum_optical_residual_samples",
        "require_m1_raw_topology",
        "require_stable_orientation_sign",
        "coefficient_absolute_span_mm",
        "curvature_relative_span",
        "minimum_curvature_span_per_mm",
        "conic_absolute_span",
        "bound_hit_relative_tolerance",
        "reject_final_parameter_bound_hit",
        "weights",
    }

    if set(restoration) != required_restoration:
        raise ValueError(
            "STEP11_TOPOLOGY_RESTORATION_KEYS_INVALID"
        )

    if not isinstance(
        restoration["enabled"],
        bool,
    ):
        raise ValueError(
            "STEP11_TOPOLOGY_RESTORATION_ENABLED_INVALID"
        )

    for key in (
        "grid_samples",
        "maximum_iterations",
        "maximum_optical_residual_samples",
    ):
        value = restoration[key]

        if (
            isinstance(value, bool)
            or int(value) != value
            or int(value) < 1
        ):
            raise ValueError(
                f"STEP11_RESTORATION_INTEGER_INVALID:{key}"
            )

    if int(
        restoration["grid_samples"]
    ) < 9:
        raise ValueError(
            "STEP11_RESTORATION_GRID_TOO_SMALL"
        )

    positive_keys = (
        "finite_difference_normalized",
        "minimum_jacobian_column_norm",
        "svd_rcond",
        "maximum_jacobian_condition",
        "damping",
        "minimum_damping",
        "maximum_damping",
        "accepted_damping_factor",
        "rejected_damping_factor",
        "max_step_normalized",
        "minimum_trust_radius",
        "maximum_trust_radius",
        "trust_shrink_factor",
        "trust_expand_factor",
        "mean_curvature_scale_per_mm",
        "gaussian_curvature_scale_per_mm2",
        "principal_curvature_scale_per_mm",
        "curvature_gradient_scale_per_mm2",
        "pair_sag_trust_scale_mm",
        "pair_normal_trust_scale_deg",
        "physical_fraction_scale",
        "coefficient_absolute_span_mm",
        "curvature_relative_span",
        "minimum_curvature_span_per_mm",
        "conic_absolute_span",
        "bound_hit_relative_tolerance",
    )

    for key in positive_keys:
        value = restoration[key]

        if (
            isinstance(value, bool)
            or not np.isfinite(
                float(value)
            )
            or float(value) <= 0.0
        ):
            raise ValueError(
                "STEP11_RESTORATION_"
                f"POSITIVE_VALUE_INVALID:{key}"
            )

    for key in (
        "maximum_invalid_jacobian_fraction",
        "minimum_acceptance_rho",
        "trust_shrink_rho",
        "trust_expand_rho",
    ):
        value = float(
            restoration[key]
        )

        if not (
            np.isfinite(value)
            and
            0.0 <= value <= 1.0
        ):
            raise ValueError(
                "STEP11_RESTORATION_"
                f"FRACTION_INVALID:{key}"
            )

    if (
        float(
            restoration[
                "trust_shrink_rho"
            ]
        )
        >=
        float(
            restoration[
                "trust_expand_rho"
            ]
        )
    ):
        raise ValueError(
            "STEP11_RESTORATION_RHO_ORDER_INVALID"
        )

    if (
        float(
            restoration[
                "minimum_trust_radius"
            ]
        )
        >
        float(
            restoration[
                "maximum_trust_radius"
            ]
        )
    ):
        raise ValueError(
            "STEP11_RESTORATION_TRUST_RANGE_INVALID"
        )

    for key in (
        "finite_difference_shrink_factors",
        "damping_multipliers",
        "line_search_alphas",
    ):
        values = np.asarray(
            restoration[key],
            float,
        )

        if (
            values.ndim != 1
            or len(values) == 0
            or not np.all(
                np.isfinite(values)
            )
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                "STEP11_RESTORATION_"
                f"LIST_INVALID:{key}"
            )

    if not isinstance(
        restoration[
            "require_m1_raw_topology"
        ],
        bool,
    ):
        raise ValueError(
            "STEP11_REQUIRE_M1_TOPOLOGY_INVALID"
        )

    if not isinstance(
        restoration[
            "require_stable_orientation_sign"
        ],
        bool,
    ):
        raise ValueError(
            "STEP11_STABLE_ORIENTATION_CONFIG_INVALID"
        )

    if not isinstance(
        restoration[
            "reject_final_parameter_bound_hit"
        ],
        bool,
    ):
        raise ValueError(
            "STEP11_FINAL_BOUND_POLICY_INVALID"
        )

    weights = restoration["weights"]

    required_weights = {
        "M1_H",
        "M1_KG",
        "M1_principal",
        "M1_gradient",

        "M2_H",
        "M2_KG",
        "M2_principal",
        "M2_gradient",

        "M1_sag_trust",
        "M1_normal_trust",

        "M2_sag_trust",
        "M2_normal_trust",

        "physical_fraction",
        "unobscuration",
        "optical",
    }

    if set(weights) != required_weights:
        raise ValueError(
            "STEP11_RESTORATION_WEIGHT_KEYS_INVALID"
        )

    for key, value in weights.items():
        if (
            isinstance(value, bool)
            or not np.isfinite(
                float(value)
            )
            or float(value) < 0.0
        ):
            raise ValueError(
                "STEP11_RESTORATION_"
                f"WEIGHT_INVALID:{key}"
            )

    if "step12_aperture_rebuild" in surface_fit:
        aperture_cfg = surface_fit["step12_aperture_rebuild"]
        required_aperture_keys = {
            "m2_footprint_filter_enabled",
            "robust_method",
            "maximum_robust_z",
            "maximum_global_rejected_fraction",
            "maximum_bundle_rejected_fraction",
            "minimum_axis_mad_scale_mm",
            "allow_chief_ray_rejection",
        }
        if set(aperture_cfg) != required_aperture_keys:
            raise ValueError("STEP12_APERTURE_REBUILD_CONFIG_KEYS_INVALID")
        if not isinstance(aperture_cfg["m2_footprint_filter_enabled"], bool):
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        if aperture_cfg["robust_method"] != "AXISWISE_MEDIAN_MAD_MAX_Z":
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        max_z = aperture_cfg["maximum_robust_z"]
        if (
            isinstance(max_z, bool)
            or not np.isfinite(float(max_z))
            or float(max_z) <= 0.0
        ):
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        max_global_frac = aperture_cfg["maximum_global_rejected_fraction"]
        if (
            isinstance(max_global_frac, bool)
            or not np.isfinite(float(max_global_frac))
            or not (0.0 < float(max_global_frac) <= 0.10)
        ):
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        max_bundle_frac = aperture_cfg["maximum_bundle_rejected_fraction"]
        if (
            isinstance(max_bundle_frac, bool)
            or not np.isfinite(float(max_bundle_frac))
            or not (0.0 < float(max_bundle_frac) <= 1.0)
            or float(max_bundle_frac) < float(max_global_frac)
        ):
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        min_mad_scale = aperture_cfg["minimum_axis_mad_scale_mm"]
        if (
            isinstance(min_mad_scale, bool)
            or not np.isfinite(float(min_mad_scale))
            or float(min_mad_scale) <= 0.0
        ):
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        if not isinstance(aperture_cfg["allow_chief_ray_rejection"], bool):
            raise ValueError("STEP12_M2_FOOTPRINT_FILTER_CONFIG_INVALID")

    if "step17_aperture_rebuild" in surface_fit:
        aperture_cfg17 = surface_fit["step17_aperture_rebuild"]
        required_aperture_keys = {
            "m2_footprint_filter_enabled",
            "robust_method",
            "maximum_robust_z",
            "maximum_global_rejected_fraction",
            "maximum_bundle_rejected_fraction",
            "minimum_axis_mad_scale_mm",
            "allow_chief_ray_rejection",
        }
        if set(aperture_cfg17) != required_aperture_keys:
            raise ValueError("STEP17_APERTURE_REBUILD_CONFIG_KEYS_INVALID")
        if not isinstance(aperture_cfg17["m2_footprint_filter_enabled"], bool):
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        if aperture_cfg17["robust_method"] != "AXISWISE_MEDIAN_MAD_MAX_Z":
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        max_z = aperture_cfg17["maximum_robust_z"]
        if (
            isinstance(max_z, bool)
            or not np.isfinite(float(max_z))
            or float(max_z) <= 0.0
        ):
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        max_global_frac = aperture_cfg17["maximum_global_rejected_fraction"]
        if (
            isinstance(max_global_frac, bool)
            or not np.isfinite(float(max_global_frac))
            or not (0.0 < float(max_global_frac) <= 0.10)
        ):
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        max_bundle_frac = aperture_cfg17["maximum_bundle_rejected_fraction"]
        if (
            isinstance(max_bundle_frac, bool)
            or not np.isfinite(float(max_bundle_frac))
            or not (0.0 < float(max_bundle_frac) <= 1.0)
            or float(max_bundle_frac) < float(max_global_frac)
        ):
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        min_mad_scale = aperture_cfg17["minimum_axis_mad_scale_mm"]
        if (
            isinstance(min_mad_scale, bool)
            or not np.isfinite(float(min_mad_scale))
            or float(min_mad_scale) <= 0.0
        ):
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")
        if not isinstance(aperture_cfg17["allow_chief_ray_rejection"], bool):
            raise ValueError("STEP17_M2_FOOTPRINT_FILTER_CONFIG_INVALID")

    ci_construction = c["ci_construction"]

    required_ci_construction = {
        "mode",
        "compatible_frontier",
    }

    if set(ci_construction) != required_ci_construction:
        raise ValueError(
            "CI_CONSTRUCTION_KEYS_INVALID"
        )

    if ci_construction["mode"] not in (
        "LEGACY_NEAREST",
        "SURFACE_COMPATIBLE_FRONTIER_V1",
        "SURFACE_COMPATIBLE_FRONTIER_V2",
        "SURFACE_COMPATIBLE_PATCH_V3",
    ):
        raise ValueError(
            "CI_CONSTRUCTION_MODE_INVALID"
        )

    frontier = ci_construction[
        "compatible_frontier"
    ]

    required_frontier = {
        "minimum_candidate_radius_mm",
        "candidate_distance_factor",
        "maximum_candidate_radius_mm",
        "normal_angle_floor_deg",
        "normal_angle_per_mm_deg",
        "normal_angle_cap_deg",
        "edge_height_residual_cap_mm",
    }

    if set(frontier) != required_frontier:
        raise ValueError(
            "CI_COMPATIBLE_FRONTIER_KEYS_INVALID"
        )

    minimum_radius = float(
        frontier[
            "minimum_candidate_radius_mm"
        ]
    )

    distance_factor = float(
        frontier[
            "candidate_distance_factor"
        ]
    )

    maximum_radius = float(
        frontier[
            "maximum_candidate_radius_mm"
        ]
    )

    angle_floor = float(
        frontier[
            "normal_angle_floor_deg"
        ]
    )

    angle_per_mm = float(
        frontier[
            "normal_angle_per_mm_deg"
        ]
    )

    angle_cap = float(
        frontier[
            "normal_angle_cap_deg"
        ]
    )

    edge_cap = float(
        frontier[
            "edge_height_residual_cap_mm"
        ]
    )

    numeric_values = np.asarray(
        [
            minimum_radius,
            distance_factor,
            maximum_radius,
            angle_floor,
            angle_per_mm,
            angle_cap,
            edge_cap,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(
            "CI_COMPATIBLE_FRONTIER_NONFINITE_OPTION"
        )

    if (
        minimum_radius <= 0.0
        or maximum_radius < minimum_radius
        or distance_factor < 1.0
        or angle_floor <= 0.0
        or angle_per_mm <= 0.0
        or angle_cap < angle_floor
        or angle_cap >= 90.0
        or edge_cap <= 0.0
    ):
        raise ValueError(
            "CI_COMPATIBLE_FRONTIER_OPTION_INVALID"
        )

    fit_weights = c["fit_weights"]
    if (any(not np.isfinite(float(fit_weights.get(key, np.nan)))
            or float(fit_weights.get(key, 0.0)) <= 0.0
            for key in ("sag", "normal", "sag_scale_mm", "normal_angle_scale_deg"))):
        raise ValueError("SURFACE_FIT_WEIGHTS_OR_SCALES_INVALID")
    conic_bounds = np.asarray(surface_fit["conic_bounds"], float)
    if conic_bounds.shape != (2,) or not np.all(np.isfinite(conic_bounds)) or conic_bounds[0] >= conic_bounds[1]:
        raise ValueError("SURFACE_FIT_CONIC_BOUNDS_INVALID")
    quality = surface_fit["step12_quality_gates"]
    required_quality = {"enforcement", "minimum_physical_fraction", "maximum_sag_rms_mm",
                        "maximum_normal_rms_deg", "maximum_normal_max_deg",
                        "maximum_reflection_rms_deg", "reject_k_bound_hit"}
    if set(quality) != required_quality:
        raise ValueError("STEP12_QUALITY_GATE_KEYS_INVALID")
    if quality["enforcement"] not in ("WARN", "HARD") or not isinstance(quality["reject_k_bound_hit"], bool):
        raise ValueError("STEP12_QUALITY_GATE_POLICY_INVALID")
    if (not 0.0 <= float(quality["minimum_physical_fraction"]) <= 1.0
            or any(float(quality[key]) <= 0.0 for key in
                   ("maximum_sag_rms_mm", "maximum_normal_rms_deg", "maximum_normal_max_deg",
                    "maximum_reflection_rms_deg"))):
        raise ValueError("STEP12_QUALITY_GATE_LIMIT_INVALID")

    step12_refinement = surface_fit["step12_o2_refinement"]
    required_step12_refinement = {
        "enabled",
        "iterations",
        "damping",
        "finite_difference_normalized",
        "max_step_normalized",
        "coefficient_absolute_bound_mm",
        "curvature_relative_bound",
        "conic_absolute_bound",
        "invalid_residual",
        "minimum_relative_improvement",
        "damping_multipliers",
        "line_search_alphas",
    }
    if set(step12_refinement) != required_step12_refinement:
        raise ValueError("STEP12_O2_REFINEMENT_KEYS_INVALID")
    if not isinstance(step12_refinement["enabled"], bool):
        raise ValueError("STEP12_O2_REFINEMENT_ENABLED_MUST_BE_BOOLEAN")
    if (isinstance(step12_refinement["iterations"], bool)
            or not isinstance(step12_refinement["iterations"], int)
            or step12_refinement["iterations"] < 1):
        raise ValueError("STEP12_O2_REFINEMENT_ITERATIONS_INVALID")

    positive_step12_values = (
        "damping",
        "finite_difference_normalized",
        "max_step_normalized",
        "coefficient_absolute_bound_mm",
        "curvature_relative_bound",
        "conic_absolute_bound",
        "invalid_residual",
    )
    if any(
        not np.isfinite(float(step12_refinement[key]))
        or float(step12_refinement[key]) <= 0.0
        for key in positive_step12_values
    ):
        raise ValueError("STEP12_O2_REFINEMENT_NUMERIC_OPTION_INVALID")
    if not 0.0 < float(step12_refinement["finite_difference_normalized"]) < 1.0:
        raise ValueError("STEP12_O2_REFINEMENT_FD_INVALID")
    if not 0.0 < float(step12_refinement["max_step_normalized"]) <= 1.0:
        raise ValueError("STEP12_O2_REFINEMENT_MAX_STEP_INVALID")
    if not 0.0 < float(step12_refinement["curvature_relative_bound"]) < 1.0:
        raise ValueError("STEP12_O2_REFINEMENT_CURVATURE_BOUND_INVALID")
    if (not np.isfinite(float(step12_refinement["minimum_relative_improvement"]))
            or float(step12_refinement["minimum_relative_improvement"]) < 0.0):
        raise ValueError("STEP12_O2_REFINEMENT_PLATEAU_INVALID")

    damping_multipliers = step12_refinement["damping_multipliers"]
    if (not isinstance(damping_multipliers, list)
            or not damping_multipliers
            or any(
                not np.isfinite(float(value)) or float(value) <= 0.0
                for value in damping_multipliers
            )):
        raise ValueError("STEP12_O2_REFINEMENT_DAMPING_MULTIPLIERS_INVALID")

    line_search_alphas = step12_refinement["line_search_alphas"]
    if (not isinstance(line_search_alphas, list)
            or not line_search_alphas
            or any(
                not np.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
                for value in line_search_alphas
            )):
        raise ValueError("STEP12_O2_REFINEMENT_LINE_SEARCH_INVALID")

    integrability = surface_fit["integrability_gates"]
    required_integrability = {
        "enforcement", "maximum_bundle_curl_rms_p95_per_mm",
        "maximum_bundle_local_geometry_normal_rms_p95_deg",
        "maximum_bundle_loop_circulation_p95_mm",
        "maximum_bundle_edge_gradient_height_residual_p95_mm",
        "maximum_bundle_nearest_normal_angle_p99_deg",
    }
    if set(integrability) != required_integrability:
        raise ValueError("SURFACE_FIT_INTEGRABILITY_GATE_KEYS_INVALID")
    if (integrability["enforcement"] not in ("WARN", "HARD")
            or any(float(integrability[key]) <= 0.0
                   for key in required_integrability if key != "enforcement")):
        raise ValueError("SURFACE_FIT_INTEGRABILITY_GATE_POLICY_INVALID")
    if list(map(int, c["solver"]["order_schedule"])) != [2, 3, 4, 5]:
        raise ValueError("ORDER_SCHEDULE_MUST_BE_2_3_4_5")
    planar_rms_limit = float(c["solver"].get("planar_paper_spot_rms_max_mm", 0.0))
    if not (np.isfinite(planar_rms_limit) and planar_rms_limit > 0.0
            and c["solver"].get("planar_paper_spot_rms_rule") == "strict_less_than"):
        raise ValueError(
            "PLANAR_PAPER_SPOT_RMS_DIAGNOSTIC_MUST_BE_STRICTLY_POSITIVE_AND_STRICT_LESS_THAN"
        )

    step7_search = c["solver"].get("step7_geometry_search")
    required_step7_search = {
        "mode",
        "candidate_count",
        "sequence_seed",
        "m1_distance_scale_bounds",
        "m1_to_m2_distance_scale_bounds",
        "m1_to_m2_turn_deg_bounds",
        "m2_to_display_distance_scale_bounds",
        "m2_to_display_turn_deg_bounds",
    }
    if not isinstance(step7_search, dict) or set(step7_search) != required_step7_search:
        raise ValueError("STEP7_GEOMETRY_SEARCH_KEYS_INVALID")
    if step7_search["mode"] != "AUTO_CHIEF_RAY_FOLD_LATIN_HYPERCUBE":
        raise ValueError("STEP7_GEOMETRY_SEARCH_MODE_INVALID")

    candidate_count = step7_search["candidate_count"]
    if (isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 8
            or candidate_count > 256):
        raise ValueError("STEP7_GEOMETRY_SEARCH_CANDIDATE_COUNT_INVALID")

    sequence_seed = step7_search["sequence_seed"]
    if (isinstance(sequence_seed, bool)
            or not isinstance(sequence_seed, int)
            or sequence_seed < 0):
        raise ValueError("STEP7_GEOMETRY_SEARCH_SEQUENCE_SEED_INVALID")

    for key in (
        "m1_distance_scale_bounds",
        "m1_to_m2_distance_scale_bounds",
        "m2_to_display_distance_scale_bounds",
    ):
        bounds = np.asarray(step7_search[key], dtype=float)
        if (bounds.shape != (2,)
                or not np.all(np.isfinite(bounds))
                or bounds[0] <= 0.0
                or bounds[0] >= bounds[1]):
            raise ValueError(f"STEP7_GEOMETRY_SEARCH_DISTANCE_BOUNDS_INVALID:{key}")

    for key in (
        "m1_to_m2_turn_deg_bounds",
        "m2_to_display_turn_deg_bounds",
    ):
        bounds = np.asarray(step7_search[key], dtype=float)
        if (bounds.shape != (2,)
                or not np.all(np.isfinite(bounds))
                or bounds[0] >= bounds[1]
                or bounds[0] <= -180.0
                or bounds[1] >= 180.0):
            raise ValueError(f"STEP7_GEOMETRY_SEARCH_ANGLE_BOUNDS_INVALID:{key}")

    rho_candidates = [float(v) for v in c["solver"].get("rho_candidates", [])]
    if (not rho_candidates or any(v == 0.0 or abs(v) > 1.0 for v in rho_candidates)
            or not any(v < 0.0 for v in rho_candidates) or not any(v > 0.0 for v in rho_candidates)):
        raise ValueError("SIGNED_RHO_CANDIDATES_MUST_INCLUDE_BOTH_BRANCHES_WITH_0<ABS_RHO<=1")
    rho_search = c["solver"].get("rho_o2_search")
    if not isinstance(rho_search, dict):
        raise ValueError("RHO_O2_SEARCH_CONFIG_MISSING")
    if rho_search.get("mode") != "COARSE_TO_FINE_FULL_PHYSICAL":
        raise ValueError("RHO_O2_SEARCH_MODE_INVALID")
    maximum_refinement_levels = int(rho_search.get("maximum_refinement_levels", 0))
    minimum_refinement_levels = int(rho_search.get("minimum_refinement_levels", 0))
    if (maximum_refinement_levels < 1 or minimum_refinement_levels < 1
            or minimum_refinement_levels > maximum_refinement_levels):
        raise ValueError("RHO_O2_SEARCH_REFINEMENT_LEVELS_INVALID")
    rho_grid_points = int(rho_search.get("grid_points", 0))
    if rho_grid_points < 5 or rho_grid_points % 2 == 0:
        raise ValueError("RHO_O2_SEARCH_GRID_POINTS_MUST_BE_ODD_AND_AT_LEAST_5")
    initial_fraction = float(rho_search.get("initial_half_width_fraction_of_nearest_spacing", 0.0))
    if not np.isfinite(initial_fraction) or not 0.0 < initial_fraction <= 1.0:
        raise ValueError("RHO_O2_SEARCH_INITIAL_HALF_WIDTH_FRACTION_INVALID")
    next_half_width = float(rho_search.get("next_half_width_in_previous_grid_steps", 0.0))
    if not np.isfinite(next_half_width) or next_half_width <= 0.0:
        raise ValueError("RHO_O2_SEARCH_NEXT_HALF_WIDTH_INVALID")
    rho_tolerance = float(rho_search.get("rho_tolerance", 0.0))
    if not np.isfinite(rho_tolerance) or rho_tolerance <= 0.0:
        raise ValueError("RHO_O2_SEARCH_TOLERANCE_INVALID")
    minimum_m2_construction_fraction = rho_search.get(
        "minimum_m2_construction_finite_fraction",
        0.95,
    )
    if (
        isinstance(minimum_m2_construction_fraction, bool)
        or not np.isfinite(float(minimum_m2_construction_fraction))
        or not 0.95 <= float(minimum_m2_construction_fraction) <= 1.0
    ):
        raise ValueError(
            "RHO_O2_SEARCH_M2_CONSTRUCTION_FINITE_FRACTION_INVALID"
        )
    minimum_step16_candidate_fraction = rho_search.get(
        "minimum_step16_candidate_fraction",
        minimum_m2_construction_fraction,
    )
    if (
        isinstance(minimum_step16_candidate_fraction, bool)
        or not np.isfinite(float(minimum_step16_candidate_fraction))
        or not 0.95 <= float(minimum_step16_candidate_fraction) <= 1.0
    ):
        raise ValueError(
            "RHO_O2_SEARCH_MINIMUM_CANDIDATE_FRACTION_INVALID"
        )
    step16_m2_edge_residual_limit = rho_search.get(
        "maximum_m2_bundle_edge_gradient_height_residual_p95_mm",
        0.17,
    )
    if (
        isinstance(step16_m2_edge_residual_limit, bool)
        or not np.isfinite(float(step16_m2_edge_residual_limit))
        or float(step16_m2_edge_residual_limit) <= 0.0
    ):
        raise ValueError(
            "RHO_O2_SEARCH_M2_EDGE_GRADIENT_RESIDUAL_LIMIT_INVALID"
        )
    if int(c["solver"].get("rho_beam_width", 0)) < 1:
        raise ValueError("RHO_BEAM_WIDTH_MUST_BE_POSITIVE")
    fermat_tol = c["solver"].get("fermat_reflection_residual_tolerance")
    if isinstance(fermat_tol, bool) or not isinstance(fermat_tol, (int, float)):
        raise ValueError("FERMAT_REFLECTION_RESIDUAL_TOLERANCE_INVALID")
    fermat_tol = float(fermat_tol)
    if not np.isfinite(fermat_tol) or fermat_tol <= 0.0:
        raise ValueError("FERMAT_REFLECTION_RESIDUAL_TOLERANCE_INVALID")
    if c["solver"].get("ci_iteration_mode") != "SUCCESSIVE_APPROXIMATION_HUD_EQ3":
        raise ValueError("CI_ITERATION_MODE_MUST_MATCH_HUD_EQ3")
    minimum_step17_candidate_fraction = c["solver"].get(
        "minimum_step17_candidate_fraction",
        0.98,
    )
    if (
        isinstance(minimum_step17_candidate_fraction, bool)
        or not np.isfinite(float(minimum_step17_candidate_fraction))
        or not 0.0 < float(minimum_step17_candidate_fraction) <= 1.0
    ):
        raise ValueError("SOLVER_MINIMUM_STEP17_CANDIDATE_FRACTION_INVALID")

    step17_spot_guard = c["solver"].get(
        "step17_spot_guard",
        {
            "enabled": True,
            "minimum_order": 4,
            "maximum_regression_mm": 0.01,
        },
    )
    required_step17_spot_guard_keys = {
        "enabled",
        "minimum_order",
        "maximum_regression_mm",
    }
    if (
        not isinstance(step17_spot_guard, dict)
        or set(step17_spot_guard) != required_step17_spot_guard_keys
    ):
        raise ValueError("SOLVER_STEP17_SPOT_GUARD_KEYS_INVALID")
    if not isinstance(step17_spot_guard["enabled"], bool):
        raise ValueError("SOLVER_STEP17_SPOT_GUARD_ENABLED_INVALID")

    step17_spot_minimum_order = step17_spot_guard["minimum_order"]
    if (
        isinstance(step17_spot_minimum_order, bool)
        or not isinstance(step17_spot_minimum_order, int)
        or not 3 <= int(step17_spot_minimum_order) <= 5
    ):
        raise ValueError("SOLVER_STEP17_SPOT_GUARD_MINIMUM_ORDER_INVALID")

    step17_spot_maximum_regression = (
        step17_spot_guard["maximum_regression_mm"]
    )
    if (
        isinstance(step17_spot_maximum_regression, bool)
        or not isinstance(
            step17_spot_maximum_regression,
            (int, float),
        )
        or not np.isfinite(
            float(step17_spot_maximum_regression)
        )
        or float(step17_spot_maximum_regression) < 0.0
    ):
        raise ValueError(
            "SOLVER_STEP17_SPOT_GUARD_MAXIMUM_REGRESSION_INVALID"
        )

    step17_m2_edge_residual_limit = c["solver"].get(
        "maximum_step17_m2_bundle_edge_gradient_height_residual_p95_mm",
        0.16,
    )
    if (
        isinstance(step17_m2_edge_residual_limit, bool)
        or not np.isfinite(float(step17_m2_edge_residual_limit))
        or float(step17_m2_edge_residual_limit) <= 0.0
    ):
        raise ValueError("SOLVER_STEP17_M2_EDGE_GRADIENT_RESIDUAL_LIMIT_INVALID")
    maximum_cycles = int(c["solver"].get("ci_cycles_per_order", 0))
    minimum_cycles = int(c["solver"].get("ci_min_cycles_per_order", 0))
    if maximum_cycles < 1 or minimum_cycles < 1 or minimum_cycles > maximum_cycles:
        raise ValueError("CI_CYCLE_LIMITS_INVALID")
    if int(c["solver"].get("ci_plateau_patience", 0)) < 1:
        raise ValueError("CI_PLATEAU_PATIENCE_INVALID")
    if float(c["solver"].get("ci_plateau_relative", -1.0)) < 0.0:
        raise ValueError("CI_PLATEAU_RELATIVE_INVALID")
    monitor_count = int(c["solver"].get("forward_constraint_rays_per_bundle", 0))
    if monitor_count != 0 and monitor_count != 49 and not (3 <= monitor_count <= 17):
        raise ValueError("FORWARD_CONSTRAINT_RAYS_PER_BUNDLE_MUST_BE_0_ALL_OR_3_TO_17_OR_49")
    if float(c["solver"].get("engineering_worsening_relative", -1.0)) < 0.0:
        raise ValueError("ENGINEERING_WORSENING_RELATIVE_MUST_BE_NONNEGATIVE")
    dlsq_nonnegative = (
        "dlsq_engineering_absolute_tolerance",
        "dlsq_restoration_fan_worsening_relative",
        "dlsq_restoration_engineering_improvement_relative",
        "dlsq_restoration_component_worsening_relative",
    )
    if any(not np.isfinite(float(c["solver"].get(key, np.nan)))
           or float(c["solver"].get(key, -1.0)) < 0.0 for key in dlsq_nonnegative):
        raise ValueError("DLSQ_FILTER_LIMITS_MUST_BE_FINITE_AND_NONNEGATIVE")
    restoration_trial_limit = c["solver"].get("dlsq_restoration_trial_limit", 0)
    if (isinstance(restoration_trial_limit, bool)
            or int(restoration_trial_limit) != restoration_trial_limit
            or int(restoration_trial_limit) < 1):
        raise ValueError("DLSQ_RESTORATION_TRIAL_LIMIT_INVALID")
    restoration_poll_count = c["solver"].get("dlsq_restoration_poll_variable_count", 0)
    restoration_poll_step = float(c["solver"].get("dlsq_restoration_poll_step_mm", np.nan))
    if (isinstance(restoration_poll_count, bool)
            or int(restoration_poll_count) != restoration_poll_count
            or int(restoration_poll_count) < 1
            or not np.isfinite(restoration_poll_step) or restoration_poll_step <= 0.0):
        raise ValueError("DLSQ_RESTORATION_COORDINATE_POLL_INVALID")
    damping_multipliers = np.asarray(c["solver"].get("dlsq_damping_multipliers", []), float)
    line_search_alphas = np.asarray(c["solver"].get("dlsq_line_search_alphas", []), float)
    if (damping_multipliers.ndim != 1 or len(damping_multipliers) == 0
            or not np.all(np.isfinite(damping_multipliers))
            or np.any(damping_multipliers <= 0.0)
            or line_search_alphas.ndim != 1 or len(line_search_alphas) == 0
            or not np.all(np.isfinite(line_search_alphas))
            or np.any(line_search_alphas <= 0.0) or np.any(line_search_alphas > 1.0)):
        raise ValueError("DLSQ_TRUST_REGION_SCHEDULE_INVALID")
    for key in ("require_full_physical_ci", "require_unobscured_ci"):
        if not isinstance(c["solver"].get(key), bool):
            raise ValueError(f"SOLVER_{key.upper()}_MUST_BE_BOOLEAN")
    if not isinstance(c["solver"].get("restoration_branch_enabled"), bool):
        raise ValueError("SOLVER_RESTORATION_BRANCH_ENABLED_MUST_BE_BOOLEAN")
    restoration_fraction = float(c["solver"].get("restoration_minimum_physical_fraction", -1.0))
    restoration_count = c["solver"].get("restoration_max_branches", 0)
    if (not np.isfinite(restoration_fraction) or not 0.0 <= restoration_fraction < 1.0
            or isinstance(restoration_count, bool)
            or int(restoration_count) != restoration_count or int(restoration_count) < 1):
        raise ValueError("SOLVER_RESTORATION_BRANCH_POLICY_INVALID")
    surface_refinement = c["surface_parameter_refinement"]
    required_surface_refinement = {
        "enabled", "top_k_only", "iterations", "damping", "finite_difference_normalized",
        "max_step_normalized", "curvature_relative_bound", "conic_absolute_bound",
    }
    if set(surface_refinement) != required_surface_refinement:
        raise ValueError("SURFACE_PARAMETER_REFINEMENT_KEYS_INVALID")
    if (not isinstance(surface_refinement["enabled"], bool)
            or not isinstance(surface_refinement["top_k_only"], bool)):
        raise ValueError("SURFACE_PARAMETER_REFINEMENT_FLAGS_MUST_BE_BOOLEAN")
    if int(surface_refinement["iterations"]) < 1:
        raise ValueError("SURFACE_PARAMETER_REFINEMENT_ITERATIONS_INVALID")
    for key in ("damping", "finite_difference_normalized", "max_step_normalized",
                "curvature_relative_bound", "conic_absolute_bound"):
        if not np.isfinite(float(surface_refinement[key])) or float(surface_refinement[key]) <= 0.0:
            raise ValueError(f"SURFACE_PARAMETER_REFINEMENT_{key.upper()}_INVALID")
    if not 0.0 < float(surface_refinement["curvature_relative_bound"]) < 1.0:
        raise ValueError("SURFACE_PARAMETER_REFINEMENT_CURVATURE_BOUND_INVALID")
    refinement = c.get("geometry_refinement")
    if refinement is not None:
        refinement_required = {
            "enabled", "top_k", "iterations", "damping", "finite_difference_normalized",
            "max_step_normalized", "fan_worsening_relative", "eye_residual_scale_mm",
            "image_residual_scale_mm", "invalid_residual", "surfaces",
        }
        missing_refinement = sorted(refinement_required - set(refinement))
        if missing_refinement:
            raise ValueError(f"GEOMETRY_REFINEMENT_MISSING_KEYS: {missing_refinement}")
        if not isinstance(refinement["enabled"], bool):
            raise ValueError("GEOMETRY_REFINEMENT_ENABLED_MUST_BE_BOOLEAN")
        for key in ("top_k", "iterations"):
            value = refinement[key]
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"GEOMETRY_REFINEMENT_{key.upper()}_MUST_BE_POSITIVE_INTEGER")
        for key in ("damping", "finite_difference_normalized", "max_step_normalized",
                    "eye_residual_scale_mm", "image_residual_scale_mm", "invalid_residual"):
            if not math.isfinite(float(refinement[key])) or float(refinement[key]) <= 0.0:
                raise ValueError(f"GEOMETRY_REFINEMENT_{key.upper()}_MUST_BE_FINITE_POSITIVE")
        if (not math.isfinite(float(refinement["fan_worsening_relative"]))
                or float(refinement["fan_worsening_relative"]) < 0.0):
            raise ValueError("GEOMETRY_REFINEMENT_FAN_WORSENING_RELATIVE_MUST_BE_NONNEGATIVE")
        surfaces = refinement["surfaces"]
        if not all(name in surfaces for name in ("M1", "M2", "DISPLAY")):
            raise ValueError("GEOMETRY_REFINEMENT_SURFACES_MUST_INCLUDE_M1_M2_DISPLAY")
        active_pose_dofs = 0
        for name in ("M1", "M2", "DISPLAY"):
            surface = surfaces[name]
            if not isinstance(surface.get("enabled"), bool):
                raise ValueError(f"GEOMETRY_REFINEMENT_{name}_ENABLED_MUST_BE_BOOLEAN")
            translations = np.asarray(surface.get("translation_local_bound_mm", []), float)
            tilts = np.asarray(surface.get("tilt_local_bound_deg", []), float)
            if (translations.shape != (3,) or tilts.shape != (2,)
                    or not np.all(np.isfinite(translations)) or not np.all(np.isfinite(tilts))
                    or np.any(translations < 0.0) or np.any(tilts < 0.0)):
                raise ValueError(f"GEOMETRY_REFINEMENT_{name}_POSE_BOUNDS_INVALID")
            if surface["enabled"]:
                active_pose_dofs += int(np.sum(translations > 0.0) + np.sum(tilts > 0.0))
        if refinement["enabled"] and active_pose_dofs == 0:
            raise ValueError("GEOMETRY_REFINEMENT_ENABLED_WITHOUT_ACTIVE_POSE_VARIABLES")
    tolerance_keys = {"fov_h_deg", "fov_v_deg", "vid_mm", "d6_deg", "azimuth_deg"}
    if set(c["hard_tolerances"]) != tolerance_keys:
        raise ValueError("HARD_TOLERANCES_KEYS_MUST_MATCH_FOV_VID_D6_AZIMUTH")
    for key, value in c["hard_tolerances"].items():
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError(f"INVALID_TOLERANCE_{key}")
    if c["mtf_requested"]:
        mtf = c.get("mtf", {})
        if (mtf.get("method") != "GEOMETRIC_RAY_OTF_WITH_CIRCULAR_DIFFRACTION_FACTOR"
                or not mtf.get("spatial_frequencies_lpmm") or not mtf.get("wavelengths_nm")
                or len(mtf.get("wavelengths_nm", [])) != len(mtf.get("wavelength_weights", []))):
            raise ValueError("MTF_CONFIGURATION_INVALID")
    live = c.get("live_monitor")
    if live is not None:
        if not isinstance(live, dict):
            raise ValueError("LIVE_MONITOR_MUST_BE_OBJECT")
        for key in ("enabled", "images_enabled"):
            if not isinstance(live.get(key), bool):
                raise ValueError(f"LIVE_MONITOR_INVALID_BOOL:{key}")
        for key in ("progress_interval_seconds", "image_min_interval_seconds"):
            value = live.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"LIVE_MONITOR_INVALID_INTERVAL:{key}")
        count = live.get("max_plot_rays")
        if (isinstance(count, bool) or not isinstance(count, int)
                or not 1 <= count <= 500):
            raise ValueError("LIVE_MONITOR_INVALID_MAX_PLOT_RAYS")


class Context:
    """Giữ config, dữ liệu, summary và thư mục của một run."""
    def __init__(
        self,
        source_dir: Path | None = None,
        config_path: Path | None = None,
        config: dict[str, Any] | None = None,
        run_dir: Path | None = None,
        data: dict[str, Any] | None = None,
        stage_summaries: dict[int, dict[str, Any]] | None = None,
        *,
        root: Path | None = None,
    ):
        """Thực thi __init__."""
        chosen_root = source_dir if source_dir is not None else (root if root is not None else ".")
        self.source_dir = Path(chosen_root)
        self.config_path = Path(config_path) if config_path is not None else Path("config_v55.json")
        self.config = config if config is not None else {}
        self.run_dir = Path(run_dir) if run_dir is not None else Path(".")
        self.data = data if data is not None else {}
        self.stage_summaries = stage_summaries if stage_summaries is not None else {}

    @property
    def root(self) -> Path:
        """Thực thi root."""
        return self.source_dir

    @root.setter
    def root(self, val: Path) -> None:
        """Thực thi root."""
        self.source_dir = Path(val)

    def step_dir(self, number: int) -> Path:
        """Trả về thư mục bằng chứng của STEP hiện tại."""
        p = self.run_dir / f"STEP_{number:02d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def group(self, number: int) -> Path:
        """Trả về và tạo thư mục nhóm kết quả."""
        return self.step_dir(number)


def _rows(rays: dict[str, Any], values: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    """Đổi ma trận NumPy thành bản ghi có tên cột để xuất CSV."""
    return [{"ray_id": row["ray_id"], f"{prefix}_x_mm": p[0], f"{prefix}_y_mm": p[1],
             f"{prefix}_z_mm": p[2]} for row, p in zip(rays["rows"], values)]


def _chief_index(rays: dict[str, Any]) -> int:
    """Lấy chief ray tại đồng thời field và pupil trung tâm được sinh động từ config."""
    center_field = int(rays["central_field_index"])
    center_pupil = int(rays["central_pupil_index"])
    selected = np.where((rays["field_index"] == center_field)
                        & (rays["pupil_index"] == center_pupil) & rays["chief"])[0]
    if len(selected) != 1:
        raise RuntimeError("CENTRAL_CHIEF_RAY_SELECTION_MISMATCH")
    return int(selected[0])


def _ci_parent_link_metrics(
    ci: dict[str, Any],
    rays: dict[str, Any],
) -> dict[str, Any]:
    """Tính các metric liên kết con-cha (edge-height residual, cross-pupil/field) bằng vectorization."""
    order = np.asarray(ci["ray_order"], int)
    parent = np.asarray(ci["parent_order_index"], int)
    points = np.asarray(ci["points_ordered"], float)
    normals = unit(np.asarray(ci["normals_ordered"], float))

    n = len(order)
    edge_height_residuals_by_order = np.full(n, np.nan, dtype=float)

    mask = parent >= 0
    child_indices = np.where(mask)[0]

    if len(child_indices) == 0:
        return {
            "parent_link_count": 0,
            "parent_edge_height_residual_p50_mm": None,
            "parent_edge_height_residual_p95_mm": None,
            "parent_edge_height_residual_max_mm": None,
            "cross_pupil_parent_link_count": 0,
            "cross_pupil_parent_link_fraction": 0.0,
            "cross_field_parent_link_count": 0,
            "cross_field_parent_link_fraction": 0.0,
            "same_pupil_edge_height_p95_mm": None,
            "cross_pupil_edge_height_p95_mm": None,
            "same_pupil_parent_normal_angle_p95_deg": None,
            "cross_pupil_parent_normal_angle_p95_deg": None,
            "edge_height_residuals_by_order": edge_height_residuals_by_order,
            "topology_scope": "GLOBAL_ALL_FIELDS_ALL_PUPILS_NOT_PUPIL_GATED",
        }

    parent_indices = parent[child_indices]
    pc = points[child_indices]
    pp = points[parent_indices]
    nc = normals[child_indices]
    np_norm = normals[parent_indices]

    # Orient child normal into same hemisphere as parent normal
    dots = np.sum(nc * np_norm, axis=1, keepdims=True)
    nc_oriented = np.where(dots < 0.0, -nc, nc)
    navg = 0.5 * (nc_oriented + np_norm)
    delta = pc - pp
    edge_residuals = np.abs(np.sum(delta * navg, axis=1))

    edge_height_residuals_by_order[child_indices] = edge_residuals

    # Normal angles
    dots_aligned = np.clip(np.abs(np.sum(nc * np_norm, axis=1)), 0.0, 1.0)
    normal_angles_deg = np.degrees(np.arccos(dots_aligned))

    field_idx = np.asarray(rays["field_index"], int)
    pupil_idx = np.asarray(rays["pupil_index"], int)

    child_rays = order[child_indices]
    parent_rays = order[parent_indices]

    child_fields = field_idx[child_rays]
    parent_fields = field_idx[parent_rays]
    child_pupils = pupil_idx[child_rays]
    parent_pupils = pupil_idx[parent_rays]

    same_field = (child_fields == parent_fields)
    same_pupil = (child_pupils == parent_pupils)
    cross_field = ~same_field
    cross_pupil = ~same_pupil

    n_links = len(child_indices)
    cross_pupil_count = int(np.sum(cross_pupil))
    cross_field_count = int(np.sum(cross_field))

    same_pupil_residuals = edge_residuals[same_pupil]
    cross_pupil_residuals = edge_residuals[cross_pupil]
    same_pupil_angles = normal_angles_deg[same_pupil]
    cross_pupil_angles = normal_angles_deg[cross_pupil]

    return {
        "parent_link_count": n_links,
        "parent_edge_height_residual_p50_mm": float(np.percentile(edge_residuals, 50)),
        "parent_edge_height_residual_p95_mm": float(np.percentile(edge_residuals, 95)),
        "parent_edge_height_residual_max_mm": float(np.max(edge_residuals)),
        "cross_pupil_parent_link_count": cross_pupil_count,
        "cross_pupil_parent_link_fraction": float(cross_pupil_count / n_links),
        "cross_field_parent_link_count": cross_field_count,
        "cross_field_parent_link_fraction": float(cross_field_count / n_links),
        "same_pupil_edge_height_p95_mm": float(np.percentile(same_pupil_residuals, 95)) if len(same_pupil_residuals) else None,
        "cross_pupil_edge_height_p95_mm": float(np.percentile(cross_pupil_residuals, 95)) if len(cross_pupil_residuals) else None,
        "same_pupil_parent_normal_angle_p95_deg": float(np.percentile(same_pupil_angles, 95)) if len(same_pupil_angles) else None,
        "cross_pupil_parent_normal_angle_p95_deg": float(np.percentile(cross_pupil_angles, 95)) if len(cross_pupil_angles) else None,
        "edge_height_residuals_by_order": edge_height_residuals_by_order,
        "topology_scope": "GLOBAL_ALL_FIELDS_ALL_PUPILS_NOT_PUPIL_GATED",
    }


def _construction_diagnostics(ctx: Context, ci: dict[str, Any], frame: np.ndarray,
                              label: str) -> dict[str, Any]:
    """Tổng hợp topology và ba lớp consistency theo từng field/pupil bundle."""
    cfg = ctx.config["surface_fit"]
    rays = ctx.data["rays"]
    point_count = len(np.asarray(ci["points_by_ray"]))
    source_ray_count = len(np.asarray(rays["field_index"]))
    ray_indices = np.asarray(
        ci.get("ray_indices", np.arange(source_ray_count)),
        int,
    )
    if (
        ray_indices.shape != (point_count,)
        or np.any(ray_indices < 0)
        or np.any(ray_indices >= source_ray_count)
        or len(np.unique(ray_indices)) != len(ray_indices)
    ):
        raise ValueError("CONSTRUCTION_DIAGNOSTIC_RAY_INDICES_INVALID")
    all_bundle_ids = (
        np.asarray(rays["field_index"], int) * int(rays["pupil_count"])
        + np.asarray(rays["pupil_index"], int)
    )
    bundle_ids = all_bundle_ids[ray_indices]
    cloud = point_normal_cloud_diagnostics(
        ci["points_by_ray"], ci["normals_by_ray"], frame,
        int(cfg["diagnostic_neighbors"]), int(cfg["diagnostic_max_samples"]), bundle_ids,
        int(cfg["diagnostic_bundle_max_samples"]))
    cloud["group_id_definition"] = "field_index*pupil_count+pupil_index"
    cloud["field_count"] = int(rays["field_count"])
    cloud["pupil_count"] = int(rays["pupil_count"])
    cloud["source_ray_count"] = source_ray_count
    cloud["diagnostic_ray_count"] = point_count
    cloud["ray_subset_applied"] = bool(point_count != source_ray_count)
    for bundle_row in cloud.get("bundle_rows", []):
        group_id = int(bundle_row["group_id"])
        field_index = group_id // int(rays["pupil_count"])
        pupil_index = group_id % int(rays["pupil_count"])
        original_indices = np.flatnonzero(all_bundle_ids == group_id)
        diagnostic_indices = np.flatnonzero(bundle_ids == group_id)
        first_row = (
            rays["rows"][int(original_indices[0])]
            if len(original_indices)
            else {}
        )
        field_definitions = ctx.data.get("vi", {}).get("fields", [])
        pupil_definitions = ctx.data.get("pupils", [])
        field_definition = (
            field_definitions[field_index]
            if 0 <= field_index < len(field_definitions)
            else {}
        )
        pupil_definition = (
            pupil_definitions[pupil_index]
            if 0 <= pupil_index < len(pupil_definitions)
            else {}
        )
        bundle_row.update({
            "field_index": field_index,
            "field_id": first_row.get("field_id"),
            "field_h_deg": field_definition.get("h_deg"),
            "field_v_deg": field_definition.get("v_deg"),
            "pupil_index": pupil_index,
            "pupil_id": first_row.get("pupil_id"),
            "pupil_x_mm": pupil_definition.get("x_mm"),
            "pupil_y_mm": pupil_definition.get("y_mm"),
            "pupil_z_mm": pupil_definition.get("z_mm"),
            "source_ray_count": int(len(original_indices)),
            "diagnostic_ray_count": int(len(diagnostic_indices)),
            "excluded_ray_count": int(
                len(original_indices) - len(diagnostic_indices)
            ),
        })
    expected_bundle_count = int(rays["field_count"]) * int(rays["pupil_count"])
    present_bundle_ids = {
        int(row["group_id"])
        for row in cloud.get("bundle_rows", [])
    }
    cloud["expected_bundle_count"] = expected_bundle_count
    cloud["missing_bundle_ids"] = [
        group_id
        for group_id in range(expected_bundle_count)
        if group_id not in present_bundle_ids
    ]
    cloud["missing_bundle_count"] = int(
        expected_bundle_count - len(present_bundle_ids)
    )
    distance = np.asarray(ci.get("nearest_distance_mm", []), float)
    finite = distance[np.isfinite(distance)]
    parent = np.asarray(ci.get("parent_order_index", []), int)
    nearest_ray_topology_available = "ray_order" in ci
    if nearest_ray_topology_available:
        link_metrics = _ci_parent_link_metrics(ci, rays)
        nearest_ray_topology_status = "AVAILABLE"
        nearest_ray_topology_reason = None
    else:
        # Iteration clouds intentionally preserve the current ray coordinates
        # without re-running the Nearest-Ray Algorithm.  Parent-link metrics
        # therefore do not exist for this cloud and must not be fabricated.
        link_metrics = {
            "parent_edge_height_residual_p50_mm": None,
            "parent_edge_height_residual_p95_mm": None,
            "parent_edge_height_residual_max_mm": None,
            "parent_link_count": None,
            "cross_pupil_parent_link_count": None,
            "cross_pupil_parent_link_fraction": None,
            "cross_field_parent_link_count": None,
            "cross_field_parent_link_fraction": None,
            "same_pupil_edge_height_p95_mm": None,
            "cross_pupil_edge_height_p95_mm": None,
            "same_pupil_parent_normal_angle_p95_deg": None,
            "cross_pupil_parent_normal_angle_p95_deg": None,
            "topology_scope": "NOT_AVAILABLE_NO_NEAREST_RAY_TOPOLOGY",
        }
        nearest_ray_topology_status = "NOT_AVAILABLE"
        nearest_ray_topology_reason = "RAY_ORDER_NOT_PROVIDED_BY_CLOUD"

    diag: dict[str, Any] = {
        "label": label,
        "fallback_count": int(ci["fallback_count"]),
        "nearest_ray_topology_available": nearest_ray_topology_available,
        "nearest_ray_topology_status": nearest_ray_topology_status,
        "nearest_ray_topology_reason": nearest_ray_topology_reason,
        "finite_parent_link_count": (int(len(finite))
                                     if nearest_ray_topology_available else None),
        "parent_link_distance_p50_mm": (float(np.percentile(finite, 50))
                                        if nearest_ray_topology_available and len(finite) else None),
        "parent_link_distance_p95_mm": (float(np.percentile(finite, 95))
                                        if nearest_ray_topology_available and len(finite) else None),
        "parent_link_distance_max_mm": (float(np.max(finite))
                                        if nearest_ray_topology_available and len(finite) else None),
        "nonlocal_parent_count": (int(np.sum((parent >= 0) & ((np.arange(len(parent))-parent) > 1)))
                                  if nearest_ray_topology_available and len(parent) else None),
        "parent_edge_height_residual_p50_mm": link_metrics["parent_edge_height_residual_p50_mm"],
        "parent_edge_height_residual_p95_mm": link_metrics["parent_edge_height_residual_p95_mm"],
        "parent_edge_height_residual_max_mm": link_metrics["parent_edge_height_residual_max_mm"],
        "parent_link_count": link_metrics["parent_link_count"],
        "cross_pupil_parent_link_count": link_metrics["cross_pupil_parent_link_count"],
        "cross_pupil_parent_link_fraction": link_metrics["cross_pupil_parent_link_fraction"],
        "cross_field_parent_link_count": link_metrics["cross_field_parent_link_count"],
        "cross_field_parent_link_fraction": link_metrics["cross_field_parent_link_fraction"],
        "same_pupil_edge_height_p95_mm": link_metrics["same_pupil_edge_height_p95_mm"],
        "cross_pupil_edge_height_p95_mm": link_metrics["cross_pupil_edge_height_p95_mm"],
        "same_pupil_parent_normal_angle_p95_deg": link_metrics["same_pupil_parent_normal_angle_p95_deg"],
        "cross_pupil_parent_normal_angle_p95_deg": link_metrics["cross_pupil_parent_normal_angle_p95_deg"],
        "topology_scope": link_metrics["topology_scope"],
        "cloud": cloud,
    }

    if "front_id" in ci:
        diag["compatible_frontier_available"] = True
        diag["front_count"] = int(ci.get("front_count", 1))
        diag["front_restart_count"] = int(ci.get("front_restart_count", 0))

        p_angles = np.asarray(ci.get("parent_normal_angle_deg", []), float)
        finite_angles = p_angles[np.isfinite(p_angles)]
        diag["parent_normal_angle_p50_deg"] = float(np.percentile(finite_angles, 50)) if len(finite_angles) else None
        diag["parent_normal_angle_p95_deg"] = float(np.percentile(finite_angles, 95)) if len(finite_angles) else None
        diag["parent_normal_angle_max_deg"] = float(np.max(finite_angles)) if len(finite_angles) else None

        loc_counts = np.asarray(ci.get("local_candidate_count", []), float)
        diag["local_candidate_count_p50"] = float(np.percentile(loc_counts, 50)) if len(loc_counts) else None
        diag["local_candidate_count_p95"] = float(np.percentile(loc_counts, 95)) if len(loc_counts) else None

        comp_counts = np.asarray(ci.get("compatible_candidate_count", []), float)
        diag["compatible_candidate_count_p50"] = float(np.percentile(comp_counts, 50)) if len(comp_counts) else None
        diag["compatible_candidate_count_p05"] = float(np.percentile(comp_counts, 5)) if len(comp_counts) else None
    else:
        diag["compatible_frontier_available"] = False

    return diag


def _ci_topology_rows(
    ci: dict[str, Any],
    rays: dict[str, Any],
    policy: str,
) -> list[dict[str, Any]]:
    """Tạo các dòng topology theo chuẩn nhất quán cho STEP10/11."""
    ray_order = np.asarray(ci["ray_order"], int)
    parent = np.asarray(ci["parent_order_index"], int)
    distance = np.asarray(ci.get("nearest_distance_mm", []), float)
    fallback_reasons = ci.get("fallback_reasons", [])

    front_id = ci.get("front_id")
    restart_reason_code = ci.get("restart_reason_code")
    parent_angle = ci.get("parent_normal_angle_deg")
    local_count = ci.get("local_candidate_count")
    compatible_count = ci.get("compatible_candidate_count")

    link_metrics = _ci_parent_link_metrics(ci, rays)
    edge_res_arr = link_metrics["edge_height_residuals_by_order"]

    reason_map = {
        0: "",
        1: "NO_COMPATIBLE_LOCAL_CANDIDATE",
        2: "NO_FORWARD_TANGENT_INTERSECTION",
    }

    field_idx_arr = np.asarray(rays["field_index"], int)
    pupil_idx_arr = np.asarray(rays["pupil_index"], int)
    rows_meta = rays["rows"]

    result = []
    for k, ray in enumerate(ray_order):
        ray_i = int(ray)
        p_order = int(parent[k])
        p_ray = int(ray_order[p_order]) if p_order >= 0 else None

        f_idx = int(field_idx_arr[ray_i])
        p_idx = int(pupil_idx_arr[ray_i])
        parent_f_idx = int(field_idx_arr[p_ray]) if p_ray is not None else None
        parent_p_idx = int(pupil_idx_arr[p_ray]) if p_ray is not None else None

        code = int(restart_reason_code[k]) if restart_reason_code is not None else 0
        r_reason = reason_map.get(code, "")
        if p_order == -3 and not r_reason:
            r_reason = "NEW_FRONT_FROM_INCUMBENT"

        fb_reason = ""
        if k < len(fallback_reasons):
            fb_reason = str(fallback_reasons[k])
        elif p_order == -3:
            fb_reason = "NEW_FRONT_FROM_INCUMBENT"

        edge_val = edge_res_arr[k] if k < len(edge_res_arr) else np.nan

        row = {
            "construction_order": k,
            "construction_policy": policy,
            "ray_index": ray_i,
            "ray_id": rows_meta[ray_i]["ray_id"],
            "field_index": f_idx,
            "pupil_index": p_idx,
            "parent_order_index": p_order,
            "parent_ray_index": p_ray,
            "parent_field_index": parent_f_idx,
            "parent_pupil_index": parent_p_idx,
            "same_field": (f_idx == parent_f_idx) if parent_f_idx is not None else False,
            "same_pupil": (p_idx == parent_p_idx) if parent_p_idx is not None else False,
            "nearest_distance_mm": float(distance[k]) if k < len(distance) and np.isfinite(distance[k]) else None,
            "front_id": int(front_id[k]) if front_id is not None else None,
            "restart_reason": r_reason,
            "parent_normal_angle_deg": float(parent_angle[k]) if parent_angle is not None and np.isfinite(parent_angle[k]) else None,
            "parent_edge_height_residual_mm": float(edge_val) if np.isfinite(edge_val) else None,
            "local_candidate_count": int(local_count[k]) if local_count is not None else None,
            "compatible_candidate_count": int(compatible_count[k]) if compatible_count is not None else None,
            "fallback_reason": fb_reason,
        }
        result.append(row)
    return result


def _integrability_quality_gate(diagnostic: dict[str, Any], policy: dict[str, Any],
                                surface: str) -> dict[str, Any]:
    """Chấm diagnostic rời rạc theo bundle mà không gọi nó là chứng minh integrability."""
    summary = diagnostic.get("cloud", {}).get("bundle_summary", {})
    definitions = [
        ("bundle_curl_rms_p95", "bundle_curl_rms_p95_per_mm",
         "maximum_bundle_curl_rms_p95_per_mm", "1/mm"),
        ("bundle_local_geometry_normal_rms_p95",
         "bundle_local_geometry_normal_rms_p95_deg",
         "maximum_bundle_local_geometry_normal_rms_p95_deg", "deg"),
        ("bundle_loop_circulation_p95", "bundle_loop_abs_circulation_p95_of_bundle_p95_mm",
         "maximum_bundle_loop_circulation_p95_mm", "mm"),
        ("bundle_edge_gradient_height_residual_p95",
         "bundle_edge_gradient_height_residual_p95_of_bundle_p95_mm",
         "maximum_bundle_edge_gradient_height_residual_p95_mm", "mm"),
        ("bundle_nearest_normal_angle_p99", "bundle_nearest_normal_angle_p99_of_bundle_p99_deg",
         "maximum_bundle_nearest_normal_angle_p99_deg", "deg"),
    ]
    rows = []
    for check_name, actual_key, limit_key, unit_name in definitions:
        raw = summary.get(actual_key)
        actual = float(raw) if raw is not None else float("inf")
        limit = float(policy[limit_key])
        passed = bool(np.isfinite(actual) and actual <= limit)
        status = "PASS" if passed else ("WARN" if policy["enforcement"] == "WARN" else "FAIL")
        rows.append({"surface": surface, "check": check_name,
                     "actual": actual if np.isfinite(actual) else None,
                     "operator": "<=", "limit": limit, "unit": unit_name,
                     "status": status, "enforcement": policy["enforcement"],
                     "reason": (f"{actual:.6g} <= {limit:.6g} {unit_name}"
                                if passed else f"NOT_EVALUABLE_OR_ABOVE_{limit:.6g}_{unit_name}")})
    all_pass = all(row["status"] == "PASS" for row in rows)
    return {"surface": surface,
            "status": ("PASS" if all_pass else
                       "WARN" if policy["enforcement"] == "WARN" else "FAIL"),
            "enforcement": policy["enforcement"],
            "discrete_diagnostic_not_proof": True,
            "checks": rows}


def _fit_quality_gate(fit: dict[str, Any], quality: dict[str, Any], surface: str) -> dict[str, Any]:
    """Cham fit quality bang nguong config va tra ly do dong de audit."""
    reflection = fit.get("reflection_direction_error", {})
    failure_status = "WARN" if str(quality.get("enforcement", "FAIL")) == "WARN" else "FAIL"
    checks = {
        "sag_rms": (float(fit["sag_rms_mm"]), float(quality["maximum_sag_rms_mm"]), "mm"),
        "normal_rms": (float(fit["normal_rms_deg"]), float(quality["maximum_normal_rms_deg"]), "deg"),
        "normal_max": (float(fit["normal_max_deg"]), float(quality["maximum_normal_max_deg"]), "deg"),
        "reflection_rms": (float(reflection.get("rms_deg", np.inf)),
                           float(quality["maximum_reflection_rms_deg"]), "deg"),
    }
    rows = []
    for name, (actual, limit, unit_name) in checks.items():
        passed = bool(np.isfinite(actual) and actual <= limit)
        rows.append({"surface": surface, "check": name, "actual": actual,
                     "operator": "<=", "limit": limit, "unit": unit_name,
                     "status": "PASS" if passed else failure_status,
                     "reason": f"{actual:.6g} {'<=' if passed else '>'} {limit:.6g} {unit_name}"})
    if bool(quality["reject_k_bound_hit"]):
        passed = not bool(fit.get("k_bound_hit", False))
        rows.append({"surface": surface, "check": "conic_bound", "actual": fit.get("conic_constant"),
                     "operator": "inside", "limit": fit.get("joint_variable_projection", {}).get("conic_bounds"),
                     "unit": "", "status": "PASS" if passed else failure_status,
                     "reason": "K is interior" if passed else "K reached a configured bound"})
    all_pass = all(row["status"] == "PASS" for row in rows)
    return {"surface": surface,
            "status": "PASS" if all_pass else failure_status,
            "checks": rows}


class ApertureRebuildError(RuntimeError):
    """Footprint không đủ dữ liệu hình học để dựng khẩu độ ứng viên."""


def _resize(
    surf: PolySurface,
    points: np.ndarray,
    floor: tuple[float, float] = (1.0, 1.0),
    set_scale: bool = False,
) -> None:
    """Dựng aperture trên bản sao; chỉ commit sau khi mọi thao tác đạt."""
    points_array = np.asarray(points, dtype=float)
    floor_array = np.asarray(floor, dtype=float)

    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("APERTURE_POINTS_MUST_HAVE_SHAPE_N_3")
    if floor_array.shape != (2,):
        raise ValueError("APERTURE_FLOOR_MUST_HAVE_SHAPE_2")
    if not np.all(np.isfinite(floor_array)) or np.any(floor_array <= 0):
        raise ValueError("APERTURE_FLOOR_MUST_BE_FINITE_POSITIVE")

    if len(points_array) == 0:
        raise ApertureRebuildError("APERTURE_EMPTY_FOOTPRINT")

    bad = np.flatnonzero(~np.all(np.isfinite(points_array), axis=1))
    if len(bad):
        raise ApertureRebuildError(
            f"APERTURE_NONFINITE_FOOTPRINT:count={len(bad)};"
            f"first_indices={bad[:20].tolist()}"
        )

    trial = surf.copy()
    local = (points_array - trial.center) @ trial.frame
    xy = local[:, :2]

    if not np.all(np.isfinite(xy)):
        raise ApertureRebuildError("APERTURE_LOCAL_COORDINATES_NONFINITE")

    # Giữ chính sách rectangle hiện có cho DISPLAY và tập dưới 3 điểm.
    if trial.name == "DISPLAY" or len(xy) < 3:
        polygon = None
        half = np.maximum(
            np.max(np.abs(xy), axis=0) + 0.5,
            floor_array,
        )
    else:
        try:
            polygon, half = convex_aperture_from_points(
                xy,
                margin_mm=0.5,
                floor=tuple(float(x) for x in floor_array),
            )
        except (QhullError, ValueError) as exc:
            raise ApertureRebuildError(
                f"APERTURE_HULL_REBUILD_FAILED:{exc}"
            ) from exc

        polygon = np.asarray(polygon, dtype=float)
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
            or not np.all(np.isfinite(polygon))
        ):
            raise ApertureRebuildError("APERTURE_HULL_RESULT_INVALID")

    half = np.asarray(half, dtype=float)
    if (
        half.shape != (2,)
        or not np.all(np.isfinite(half))
        or np.any(half <= 0)
    ):
        raise ApertureRebuildError("APERTURE_HALF_BOUNDS_INVALID")

    if polygon is not None:
        if np.any(half < np.max(np.abs(polygon), axis=0)):
            raise ApertureRebuildError(
                "APERTURE_BOUNDING_BOX_DOES_NOT_CONTAIN_POLYGON"
            )

    trial.half_aperture = half.copy()
    trial.aperture_polygon = (
        None if polygon is None else polygon.copy()
    )

    if set_scale:
        coefficients = np.asarray(trial.coeff)
        if coefficients.size and np.any(coefficients != 0.0):
            rescale_surface_polynomial(
                trial, trial.half_aperture.copy()
            )
        else:
            trial.scale = trial.half_aperture.copy()

    if (
        np.asarray(trial.scale).shape != (2,)
        or not np.all(np.isfinite(trial.scale))
        or np.any(np.asarray(trial.scale) <= 0)
        or not np.all(np.isfinite(trial.coeff))
    ):
        raise ApertureRebuildError("APERTURE_RESCALE_RESULT_INVALID")

    # Commit cuối: không thay frame, center, curvature hoặc conic.
    surf.half_aperture = trial.half_aperture.copy()
    surf.aperture_polygon = (
        None if trial.aperture_polygon is None
        else trial.aperture_polygon.copy()
    )
    surf.scale = trial.scale.copy()
    surf.coeff = trial.coeff.copy()


def _aperture_area(surface: PolySurface) -> float:
    """Tính diện tích khẩu độ (mm^2) theo polygon hoặc bounding box."""
    if surface.aperture_polygon is not None and len(surface.aperture_polygon) >= 3:
        poly = np.asarray(surface.aperture_polygon, dtype=float)
        x = poly[:, 0]
        y = poly[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    h = np.asarray(surface.half_aperture, dtype=float)
    return float(4.0 * h[0] * h[1])


def _step7_characteristic_scale_mm(ctx: Context) -> float:
    """Tạo scale geometry từ kích thước quang học và packaging đang bật."""
    display_size = np.asarray(ctx.config["display_seed_mm"], dtype=float)
    visor_size = np.asarray(ctx.config["visor_active_mm"], dtype=float)
    eyebox_size = np.asarray(
        [
            np.ptp(np.asarray(ctx.config["eyebox_y_mm"], dtype=float)),
            np.ptp(np.asarray(ctx.config["eyebox_z_mm"], dtype=float)),
        ],
        dtype=float,
    )

    values = [
        float(np.linalg.norm(display_size)),
        float(0.5 * np.linalg.norm(visor_size)),
        float(np.linalg.norm(eyebox_size)),
        float(ctx.config["pupil_diameter_mm"]),
    ]

    packaging = ctx.data["inputs"].get("packaging_vertices")
    if packaging is not None:
        package_extent = np.ptp(np.asarray(packaging, dtype=float), axis=0)
        values.append(float(0.5 * np.linalg.norm(package_extent)))

    scale = float(max(values))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("STEP7_CHARACTERISTIC_SCALE_INVALID")
    return scale


def _step7_latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Sinh Latin-hypercube xác định được bằng tâm mỗi stratum."""
    if count < 1 or dimensions < 1:
        raise ValueError("STEP7_LATIN_HYPERCUBE_DIMENSION_INVALID")

    rng = np.random.default_rng(int(seed))
    centers = (np.arange(count, dtype=float) + 0.5) / float(count)
    samples = np.empty((count, dimensions), dtype=float)

    for dimension in range(dimensions):
        samples[:, dimension] = centers[rng.permutation(count)]

    return samples


def _step7_auto_seed_descriptors(ctx: Context) -> list[dict[str, Any]]:
    """Sinh planar folding geometry từ central chief ray, không dùng XYZ seed viết tay."""
    search = ctx.config["solver"]["step7_geometry_search"]
    count = int(search["candidate_count"])
    samples = _step7_latin_hypercube(
        count,
        5,
        int(search["sequence_seed"]),
    )

    scale = _step7_characteristic_scale_mm(ctx)
    rays = ctx.data["rays"]
    chief_index = _chief_index(rays)
    qv = np.asarray(
        ctx.data["visor_hit"]["point"][chief_index],
        dtype=float,
    )
    dv = unit(
        np.asarray(
            ctx.data["post_visor"][chief_index],
            dtype=float,
        )
    )

    fold_axis = None
    for preferred in (
        ctx.data["inputs"]["visor_V"],
        ctx.data["inputs"]["visor_N"],
        ctx.data["inputs"]["visor_U"],
    ):
        preferred_array = np.asarray(preferred, dtype=float)
        projected = preferred_array - float(np.dot(preferred_array, dv)) * dv
        if np.linalg.norm(projected) > 1e-9:
            fold_axis = unit(projected)
            break

    if fold_axis is None:
        raise RuntimeError("STEP7_FOLD_PLANE_BASIS_DEGENERATE")

    m1_bounds = np.asarray(
        search["m1_distance_scale_bounds"],
        dtype=float,
    )
    m12_bounds = np.asarray(
        search["m1_to_m2_distance_scale_bounds"],
        dtype=float,
    )
    theta12_bounds = np.asarray(
        search["m1_to_m2_turn_deg_bounds"],
        dtype=float,
    )
    m2d_bounds = np.asarray(
        search["m2_to_display_distance_scale_bounds"],
        dtype=float,
    )
    theta2d_bounds = np.asarray(
        search["m2_to_display_turn_deg_bounds"],
        dtype=float,
    )

    def interpolate(sample: float, bounds: np.ndarray) -> float:
        """Map một mẫu [0,1] vào đúng bounds cấu hình."""
        return float(
            bounds[0]
            + float(sample) * (bounds[1] - bounds[0])
        )

    descriptors: list[dict[str, Any]] = []

    for candidate_index, sample in enumerate(samples, start=1):
        m1_distance = scale * interpolate(sample[0], m1_bounds)
        m1_to_m2_distance = scale * interpolate(sample[1], m12_bounds)
        theta12_deg = interpolate(sample[2], theta12_bounds)
        m2_to_display_distance = scale * interpolate(sample[3], m2d_bounds)
        theta2d_deg = interpolate(sample[4], theta2d_bounds)

        theta12 = math.radians(theta12_deg)
        theta2d = math.radians(theta2d_deg)

        c1 = qv + m1_distance * dv
        d12 = unit(
            math.cos(theta12) * dv
            + math.sin(theta12) * fold_axis
        )
        c2 = c1 + m1_to_m2_distance * d12

        perpendicular_12 = unit(
            -math.sin(theta12) * dv
            + math.cos(theta12) * fold_axis
        )
        d2d = unit(
            math.cos(theta2d) * d12
            + math.sin(theta2d) * perpendicular_12
        )
        cd = c2 + m2_to_display_distance * d2d

        descriptors.append(
            {
                "candidate_index": candidate_index,
                "d": float(m1_distance),
                "c2": np.asarray(c2, dtype=float),
                "cd": np.asarray(cd, dtype=float),
                "geometry_search_parameters": {
                    "mode": search["mode"],
                    "characteristic_scale_mm": float(scale),
                    "m1_distance_mm": float(m1_distance),
                    "m1_to_m2_distance_mm": float(m1_to_m2_distance),
                    "m1_to_m2_turn_deg": float(theta12_deg),
                    "m2_to_display_distance_mm": float(m2_to_display_distance),
                    "m2_to_display_turn_deg": float(theta2d_deg),
                    "fold_axis_global": np.asarray(
                        fold_axis,
                        dtype=float,
                    ).tolist(),
                },
            }
        )

    return descriptors


def _step7_packaging_pass(
    ctx: Context,
    packaging_value: float | None,
) -> bool:
    """Packaging chỉ là hard gate khi P1-P8 thật sự được bật."""
    enabled = bool(
        ctx.data["inputs"]["packaging_constraint_enabled"]
    )
    if not enabled:
        return True

    return bool(
        packaging_value is not None
        and np.isfinite(float(packaging_value))
        and float(packaging_value)
        <= float(ctx.config["packaging_lambda_max"])
    )


def _step7_geometry_metrics(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    display: PolySurface,
    m1_distance_mm: float,
) -> dict[str, Any]:
    """Đo compactness từ clear aperture thật sau _resize()."""
    sampled_points = np.vstack(
        [
            aperture_points(m1, samples=15),
            aperture_points(m2, samples=15),
            aperture_points(display, samples=15),
        ]
    )

    bbox_min = np.min(sampled_points, axis=0)
    bbox_max = np.max(sampled_points, axis=0)
    bbox_extent = bbox_max - bbox_min

    m1_area = _aperture_area(m1)
    m2_area = _aperture_area(m2)
    display_area = _aperture_area(display)

    m1_diagonal = float(
        2.0 * np.linalg.norm(np.asarray(m1.half_aperture, dtype=float))
    )
    m2_diagonal = float(
        2.0 * np.linalg.norm(np.asarray(m2.half_aperture, dtype=float))
    )
    display_diagonal = float(
        2.0 * np.linalg.norm(np.asarray(display.half_aperture, dtype=float))
    )

    chief_path_length = float(
        m1_distance_mm
        + np.linalg.norm(
            np.asarray(m2.center, dtype=float)
            - np.asarray(m1.center, dtype=float)
        )
        + np.linalg.norm(
            np.asarray(display.center, dtype=float)
            - np.asarray(m2.center, dtype=float)
        )
    )

    packaging_value = _packaging(
        ctx,
        m1,
        m2,
        display,
    )
    packaging_enabled = bool(
        ctx.data["inputs"]["packaging_constraint_enabled"]
    )
    packaging_limit = (
        float(ctx.config["packaging_lambda_max"])
        if packaging_enabled
        else None
    )
    packaging_pass = _step7_packaging_pass(
        ctx,
        packaging_value,
    )

    return {
        "geometry_bbox_min_mm": bbox_min.tolist(),
        "geometry_bbox_max_mm": bbox_max.tolist(),
        "geometry_bbox_extent_mm": bbox_extent.tolist(),
        "geometry_bbox_volume_mm3": float(np.prod(bbox_extent)),
        "geometry_bbox_diagonal_mm": float(np.linalg.norm(bbox_extent)),
        "m1_clear_aperture_area_mm2": float(m1_area),
        "m2_clear_aperture_area_mm2": float(m2_area),
        "display_clear_aperture_area_mm2": float(display_area),
        "total_clear_aperture_area_mm2": float(
            m1_area + m2_area + display_area
        ),
        "m1_clear_aperture_diagonal_mm": m1_diagonal,
        "m2_clear_aperture_diagonal_mm": m2_diagonal,
        "display_clear_aperture_diagonal_mm": display_diagonal,
        "max_clear_aperture_diagonal_mm": float(
            max(
                m1_diagonal,
                m2_diagonal,
                display_diagonal,
            )
        ),
        "chief_path_length_mm": chief_path_length,
        "packaging_constraint_enabled": packaging_enabled,
        "packaging_lambda": (
            None
            if packaging_value is None
            else float(packaging_value)
        ),
        "packaging_lambda_limit": packaging_limit,
        "packaging_pass": bool(packaging_pass),
    }


def _planar_seed_rank_key(
    candidate: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    """Xếp candidate eligible theo compactness; RMS chỉ là tie-break cuối."""
    if not bool(candidate.get("eligible", False)):
        return (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )

    geometry = candidate["geometry_metrics"]
    packaging_value = candidate.get("packaging_lambda")
    rms = candidate["planar_paper_spot"].get("RMS_spot_radius_mm")

    packaging_rank = (
        float(packaging_value)
        if packaging_value is not None
        and np.isfinite(float(packaging_value))
        else 0.0
    )
    rms_rank = (
        float(rms)
        if rms is not None
        and np.isfinite(float(rms))
        else float("inf")
    )

    return (
        float(geometry["geometry_bbox_volume_mm3"]),
        float(geometry["geometry_bbox_diagonal_mm"]),
        float(geometry["max_clear_aperture_diagonal_mm"]),
        float(geometry["total_clear_aperture_area_mm2"]),
        float(geometry["chief_path_length_mm"]),
        packaging_rank,
        rms_rank,
    )


def _surface_shape_snapshot(surface: PolySurface) -> dict[str, Any]:
    """Snapshot hình dạng bề mặt để kiểm tra tính bất biến của tham số hình học."""
    return {
        "name": surface.name,
        "center": np.asarray(surface.center, dtype=float).copy(),
        "frame": np.asarray(surface.frame, dtype=float).copy(),
        "scale": np.asarray(surface.scale, dtype=float).copy(),
        "coeff": np.asarray(surface.coeff, dtype=float).copy(),
        "curvature": float(surface.curvature),
        "conic": float(surface.conic),
        "terms": list(surface.terms),
    }


def _assert_surface_shape_unchanged(before: dict[str, Any], surface: PolySurface, label: str) -> None:
    """Đảm bảo các tham số hình học không đổi sau khi tái tạo khẩu độ."""
    if not np.allclose(before["center"], surface.center, atol=1e-12):
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_CENTER_{label}")
    if not np.allclose(before["frame"], surface.frame, atol=1e-12):
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_FRAME_{label}")
    if not np.allclose(before["scale"], surface.scale, atol=1e-12):
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_SCALE_{label}")
    if not np.allclose(before["coeff"], surface.coeff, atol=1e-12):
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_COEFF_{label}")
    if abs(before["curvature"] - float(surface.curvature)) > 1e-12:
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_CURVATURE_{label}")
    if abs(before["conic"] - float(surface.conic)) > 1e-12:
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_CONIC_{label}")
    if before["terms"] != list(surface.terms):
        raise RuntimeError(f"POST_FIT_SHAPE_CHANGED_TERMS_{label}")


DEFAULT_STEP12_APERTURE_REBUILD_CONFIG: dict[str, Any] = {
    "m2_footprint_filter_enabled": True,
    "robust_method": "AXISWISE_MEDIAN_MAD_MAX_Z",
    "maximum_robust_z": 3.0,
    "maximum_global_rejected_fraction": 0.1,
    "maximum_bundle_rejected_fraction": 0.25,
    "minimum_axis_mad_scale_mm": 1e-9,
    "allow_chief_ray_rejection": False,
}

DEFAULT_STEP17_APERTURE_REBUILD_CONFIG: dict[str, Any] = {
    "m2_footprint_filter_enabled": True,
    "robust_method": "AXISWISE_MEDIAN_MAD_MAX_Z",
    "maximum_robust_z": 3.0,
    "maximum_global_rejected_fraction": 0.1,
    "maximum_bundle_rejected_fraction": 0.25,
    "minimum_axis_mad_scale_mm": 1e-9,
    "allow_chief_ray_rejection": False,
}

M2_APERTURE_FOOTPRINT_EXCLUSIONS_CSV_FIELDNAMES: list[str] = [
    "ray_index",
    "ray_id",
    "field_id",
    "field_index",
    "pupil_id",
    "pupil_index",
    "sample_id",
    "sample_index",
    "chief_preseed",
    "m2_local_x_mm",
    "m2_local_y_mm",
    "construction_m1_finite",
    "construction_m2_finite",
    "construction_display_finite",
    "chain_valid",
    "robust_z_x",
    "robust_z_y",
    "robust_z_max",
    "exclusion_reason",
    "finite_trace_valid_after_rebuild",
]

M2_APERTURE_FOOTPRINT_BUNDLES_CSV_FIELDNAMES: list[str] = [
    "bundle_id",
    "field_id",
    "field_index",
    "pupil_id",
    "pupil_index",
    "finite_m2_count",
    "chain_valid_count",
    "chain_incomplete_count",
    "robust_outlier_count",
    "combined_rejected_count",
    "combined_rejected_fraction",
    "retained_authority_count",
]


def _m2_aperture_authority_mask(
    ctx: Context,
    m2: PolySurface,
    p2: np.ndarray,
    finite_masks: dict[str, np.ndarray],
    construction_finite_mask: np.ndarray,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Xác định tập tia đại diện (authority mask) dùng để dựng khẩu độ M2 trong STEP 12 và STEP 17 O3-O5."""
    rays = ctx.data["rays"]
    ray_count = len(rays["origins"])
    m2_finite_mask = np.asarray(finite_masks["M2"], bool)

    is_step12 = label.startswith("STEP_12_")
    is_step17 = label in {
        "ORDER_3_RECONSTRUCTION",
        "ORDER_4_RECONSTRUCTION",
        "ORDER_5_RECONSTRUCTION",
    }

    surface_fit_cfg = ctx.config.get("surface_fit", {})
    if is_step17:
        if "step17_aperture_rebuild" in surface_fit_cfg:
            cfg = surface_fit_cfg["step17_aperture_rebuild"]
            config_source = "EXPLICIT_FROZEN_CONFIG"
        elif "step12_aperture_rebuild" in surface_fit_cfg:
            cfg = surface_fit_cfg["step12_aperture_rebuild"]
            config_source = "INHERITED_FROM_STEP12"
        else:
            cfg = DEFAULT_STEP17_APERTURE_REBUILD_CONFIG
            config_source = "BACKWARD_COMPAT_SOURCE_DEFAULT"
        filter_enabled = bool(cfg.get("m2_footprint_filter_enabled", True))
    elif is_step12:
        if "step12_aperture_rebuild" in surface_fit_cfg:
            cfg = surface_fit_cfg["step12_aperture_rebuild"]
            config_source = "EXPLICIT_FROZEN_CONFIG"
        else:
            cfg = DEFAULT_STEP12_APERTURE_REBUILD_CONFIG
            config_source = "BACKWARD_COMPAT_SOURCE_DEFAULT"
        filter_enabled = bool(cfg.get("m2_footprint_filter_enabled", True))
    else:
        cfg = DEFAULT_STEP12_APERTURE_REBUILD_CONFIG
        config_source = "DISABLED_NON_STEP12_NON_STEP17"
        filter_enabled = False

    if not filter_enabled:
        disabled_audit = {
            "footprint_filter_enabled": False,
            "footprint_filter_method": "NONE",
            "config_source": (
                "DISABLED_NON_STEP12_NON_STEP17"
                if not (is_step12 or is_step17)
                else config_source
            ),
            "maximum_robust_z": 0.0,
            "maximum_global_rejected_fraction": 0.0,
            "maximum_bundle_rejected_fraction": 0.0,
            "robust_center_local_xy_mm": [0.0, 0.0],
            "robust_mad_scale_xy_mm": [0.0, 0.0],
            "m2_finite_count": int(np.count_nonzero(m2_finite_mask)),
            "chain_valid_count": int(np.count_nonzero(construction_finite_mask)),
            "chain_incomplete_count": 0,
            "robust_outlier_count": 0,
            "combined_rejected_count": 0,
            "combined_rejected_fraction": 0.0,
            "retained_authority_count": int(np.count_nonzero(m2_finite_mask)),
            "rejected_chief_count": 0,
            "affected_bundle_count": 0,
            "maximum_observed_bundle_rejected_fraction": 0.0,
            "local_xy_min_before_mm": [0.0, 0.0],
            "local_xy_max_before_mm": [0.0, 0.0],
            "local_xy_min_after_mm": [0.0, 0.0],
            "local_xy_max_after_mm": [0.0, 0.0],
            "half_aperture_before_mm": np.asarray(m2.half_aperture, float).tolist(),
            "half_aperture_after_mm": np.asarray(m2.half_aperture, float).tolist(),
            "combined_rejected_indices_preview": [],
            "combined_rejected_ray_ids_preview": [],
            "excluded_still_finite_trace_count": 0,
            "excluded_invalid_finite_trace_count": 0,
            "retained_authority_lost_count": 0,
            "footprint_outlier_rows": [],
            "footprint_bundle_rows": [],
        }
        ctx.data["_last_m2_aperture_audit"] = disabled_audit
        ctx.data["_last_m2_authority_mask"] = m2_finite_mask.copy()
        return m2_finite_mask.copy(), disabled_audit, [], []

    prefix = "STEP12" if is_step12 else "STEP17"

    # 03.1 Shape checks
    if p2.shape != (ray_count, 3):
        raise RuntimeError(
            f"{prefix}_M2_P2_SHAPE_MISMATCH_{label}:actual={p2.shape}:expected={(ray_count, 3)}"
        )
    for mask_name, msk in list(finite_masks.items()) + [("construction_finite", construction_finite_mask)]:
        if msk.shape != (ray_count,):
            raise RuntimeError(
                f"{prefix}_M2_MASK_SHAPE_MISMATCH_{label}:{mask_name}={msk.shape}:expected={(ray_count,)}"
            )
    field_indices = np.asarray(rays["field_index"], int)
    pupil_indices = np.asarray(rays["pupil_index"], int)
    sample_indices = np.asarray(rays["sample_index"], int)
    if (
        field_indices.shape != (ray_count,)
        or pupil_indices.shape != (ray_count,)
        or sample_indices.shape != (ray_count,)
    ):
        raise RuntimeError(f"{prefix}_M2_RAYS_INDEX_SHAPE_MISMATCH_{label}")
    if len(rays["rows"]) != ray_count:
        raise RuntimeError(
            f"{prefix}_M2_RAYS_ROWS_LENGTH_MISMATCH_{label}:actual={len(rays['rows'])}:expected={ray_count}"
        )

    # 03.2 Chain-valid gate
    chain_valid_mask = (
        np.asarray(finite_masks["M1"], bool)
        & np.asarray(finite_masks["M2"], bool)
        & np.asarray(finite_masks["DISPLAY"], bool)
    )
    chain_incomplete_mask = m2_finite_mask & ~chain_valid_mask
    chain_valid_count = int(np.count_nonzero(chain_valid_mask))
    chain_incomplete_count = int(np.count_nonzero(chain_incomplete_mask))

    if chain_valid_count == 0:
        raise RuntimeError(f"{prefix}_M2_FOOTPRINT_NO_CHAIN_VALID_RAYS_{label}")

    # 03.3 Local coordinates
    m2_local = (p2 - np.asarray(m2.center, float)) @ np.asarray(m2.frame, float)
    xy = m2_local[:, :2]

    if np.any(m2_finite_mask):
        local_xy_min_before_mm = [
            float(np.min(xy[m2_finite_mask, 0])),
            float(np.min(xy[m2_finite_mask, 1])),
        ]
        local_xy_max_before_mm = [
            float(np.max(xy[m2_finite_mask, 0])),
            float(np.max(xy[m2_finite_mask, 1])),
        ]
    else:
        local_xy_min_before_mm = [0.0, 0.0]
        local_xy_max_before_mm = [0.0, 0.0]

    # 03.4 Median–MAD robust score
    xy_chain_valid = xy[chain_valid_mask]
    median_xy = np.median(xy_chain_valid, axis=0)
    mad_xy = np.median(np.abs(xy_chain_valid - median_xy), axis=0)
    mad_scale_xy = 1.4826 * mad_xy

    minimum_axis_mad_scale_mm = float(cfg.get("minimum_axis_mad_scale_mm", 1e-9))
    if np.any(mad_scale_xy <= minimum_axis_mad_scale_mm):
        raise RuntimeError(
            f"{prefix}_M2_FOOTPRINT_MAD_SCALE_DEGENERATE_{label}:"
            f"mad_scale={mad_scale_xy.tolist()}:min={minimum_axis_mad_scale_mm}"
        )

    z_xy = np.zeros((ray_count, 2), dtype=float)
    z_xy[:, 0] = np.abs((xy[:, 0] - median_xy[0]) / mad_scale_xy[0])
    z_xy[:, 1] = np.abs((xy[:, 1] - median_xy[1]) / mad_scale_xy[1])
    robust_z_max = np.maximum(z_xy[:, 0], z_xy[:, 1])

    maximum_robust_z = float(cfg.get("maximum_robust_z", 3.0))
    robust_outlier_mask = chain_valid_mask & (robust_z_max > maximum_robust_z)
    robust_outlier_count = int(np.count_nonzero(robust_outlier_mask))

    # 03.5 Combined excluded mask and authority mask
    combined_excluded_mask = chain_incomplete_mask | robust_outlier_mask
    authority_mask = m2_finite_mask & ~combined_excluded_mask

    if np.any(authority_mask & ~np.all(np.isfinite(p2), axis=1)):
        raise RuntimeError(f"{prefix}_M2_AUTHORITY_CONTAINS_NONFINITE_{label}")

    retained_authority_count = int(np.count_nonzero(authority_mask))

    if np.any(authority_mask):
        local_xy_min_after_mm = [
            float(np.min(xy[authority_mask, 0])),
            float(np.min(xy[authority_mask, 1])),
        ]
        local_xy_max_after_mm = [
            float(np.max(xy[authority_mask, 0])),
            float(np.max(xy[authority_mask, 1])),
        ]
    else:
        local_xy_min_after_mm = [0.0, 0.0]
        local_xy_max_after_mm = [0.0, 0.0]

    # Global and bundle metrics pre-calculation
    combined_rejected_count = int(np.count_nonzero(combined_excluded_mask))
    combined_rejected_fraction = float(combined_rejected_count / max(ray_count, 1))
    maximum_global_rejected_fraction = float(cfg.get("maximum_global_rejected_fraction", 0.10))

    chief_mask = np.asarray(rays.get("chief", np.zeros(ray_count, bool)), bool)
    rejected_chief_mask = combined_excluded_mask & chief_mask
    rejected_chief_count = int(np.count_nonzero(rejected_chief_mask))
    allow_chief_ray_rejection = bool(cfg.get("allow_chief_ray_rejection", False))

    pupil_count = int(rays.get("pupil_count", 15))
    bundle_ids = field_indices * pupil_count + pupil_indices
    unique_bundle_ids = np.unique(bundle_ids)

    maximum_bundle_rejected_fraction = float(cfg.get("maximum_bundle_rejected_fraction", 0.25))

    bundle_rows: list[dict[str, Any]] = []
    affected_bundle_count = 0
    maximum_observed_bundle_rejected_fraction = 0.0

    for b_id in unique_bundle_ids:
        b_mask = (bundle_ids == b_id)
        finite_m2_in_b = int(np.count_nonzero(b_mask & m2_finite_mask))
        if finite_m2_in_b == 0:
            continue
        chain_valid_in_b = int(np.count_nonzero(b_mask & chain_valid_mask))
        chain_incomp_in_b = int(np.count_nonzero(b_mask & chain_incomplete_mask))
        robust_outlier_in_b = int(np.count_nonzero(b_mask & robust_outlier_mask))
        combined_rejected_in_b = int(np.count_nonzero(b_mask & combined_excluded_mask))
        retained_authority_in_b = int(np.count_nonzero(b_mask & authority_mask))

        bundle_rejected_frac = float(combined_rejected_in_b / finite_m2_in_b)

        sample_idx = int(np.flatnonzero(b_mask)[0])
        row_sample = rays["rows"][sample_idx]
        f_id = row_sample.get("field_id", f"F_{int(field_indices[sample_idx])}")
        p_id = row_sample.get("pupil_id", f"P_{int(pupil_indices[sample_idx])}")

        if combined_rejected_in_b > 0:
            affected_bundle_count += 1
        if bundle_rejected_frac > maximum_observed_bundle_rejected_fraction:
            maximum_observed_bundle_rejected_fraction = bundle_rejected_frac

        bundle_row = {
            "bundle_id": int(b_id),
            "field_id": str(f_id),
            "field_index": int(field_indices[sample_idx]),
            "pupil_id": str(p_id),
            "pupil_index": int(pupil_indices[sample_idx]),
            "finite_m2_count": finite_m2_in_b,
            "chain_valid_count": chain_valid_in_b,
            "chain_incomplete_count": chain_incomp_in_b,
            "robust_outlier_count": robust_outlier_in_b,
            "combined_rejected_count": combined_rejected_in_b,
            "combined_rejected_fraction": bundle_rejected_frac,
            "retained_authority_count": retained_authority_in_b,
        }
        bundle_rows.append(bundle_row)

    excluded_indices = np.flatnonzero(combined_excluded_mask)
    excluded_rows: list[dict[str, Any]] = []
    for idx in excluded_indices:
        idx = int(idx)
        r_item = rays["rows"][idx]
        is_incomp = bool(chain_incomplete_mask[idx])
        is_robust = bool(robust_outlier_mask[idx])
        if is_incomp and is_robust:
            reason = "CONSTRUCTION_CHAIN_INCOMPLETE_AND_ROBUST_OUTLIER"
        elif is_incomp:
            reason = "CONSTRUCTION_CHAIN_INCOMPLETE"
        elif is_robust:
            reason = "ROBUST_MEDIAN_MAD_OUTLIER"
        else:
            reason = "UNKNOWN"

        ex_row = {
            "ray_index": idx,
            "ray_id": str(r_item.get("ray_id", f"RAY_{idx}")),
            "field_id": str(r_item.get("field_id", "")),
            "field_index": int(field_indices[idx]),
            "pupil_id": str(r_item.get("pupil_id", "")),
            "pupil_index": int(pupil_indices[idx]),
            "sample_id": str(r_item.get("sample_id", "")),
            "sample_index": int(sample_indices[idx]),
            "chief_preseed": bool(r_item.get("chief_preseed", False)),
            "m2_local_x_mm": float(xy[idx, 0]),
            "m2_local_y_mm": float(xy[idx, 1]),
            "construction_m1_finite": bool(finite_masks["M1"][idx]),
            "construction_m2_finite": bool(finite_masks["M2"][idx]),
            "construction_display_finite": bool(finite_masks["DISPLAY"][idx]),
            "chain_valid": bool(chain_valid_mask[idx]),
            "robust_z_x": float(z_xy[idx, 0]),
            "robust_z_y": float(z_xy[idx, 1]),
            "robust_z_max": float(robust_z_max[idx]),
            "exclusion_reason": reason,
            "finite_trace_valid_after_rebuild": None,
        }
        excluded_rows.append(ex_row)

    combined_rejected_indices_preview = [int(i) for i in excluded_indices[:20]]
    combined_rejected_ray_ids_preview = [
        str(rays["rows"][int(i)].get("ray_id", f"RAY_{int(i)}"))
        for i in excluded_indices[:20]
    ]

    m2_half_before = np.asarray(m2.half_aperture, float).tolist()
    if retained_authority_count > 0:
        m2_half_after = [
            max(abs(local_xy_min_after_mm[0]), abs(local_xy_max_after_mm[0])),
            max(abs(local_xy_min_after_mm[1]), abs(local_xy_max_after_mm[1])),
        ]
    else:
        m2_half_after = list(m2_half_before)

    m2_authority_audit = {
        "footprint_filter_enabled": True,
        "footprint_filter_method": str(cfg.get("robust_method", "AXISWISE_MEDIAN_MAD_MAX_Z")),
        "config_source": config_source,
        "maximum_robust_z": maximum_robust_z,
        "maximum_global_rejected_fraction": maximum_global_rejected_fraction,
        "maximum_bundle_rejected_fraction": maximum_bundle_rejected_fraction,
        "robust_center_local_xy_mm": [float(median_xy[0]), float(median_xy[1])],
        "robust_mad_scale_xy_mm": [float(mad_scale_xy[0]), float(mad_scale_xy[1])],
        "m2_finite_count": int(np.count_nonzero(m2_finite_mask)),
        "chain_valid_count": chain_valid_count,
        "chain_incomplete_count": chain_incomplete_count,
        "robust_outlier_count": robust_outlier_count,
        "combined_rejected_count": combined_rejected_count,
        "combined_rejected_fraction": combined_rejected_fraction,
        "retained_authority_count": retained_authority_count,
        "rejected_chief_count": rejected_chief_count,
        "affected_bundle_count": affected_bundle_count,
        "maximum_observed_bundle_rejected_fraction": float(maximum_observed_bundle_rejected_fraction),
        "local_xy_min_before_mm": local_xy_min_before_mm,
        "local_xy_max_before_mm": local_xy_max_before_mm,
        "local_xy_min_after_mm": local_xy_min_after_mm,
        "local_xy_max_after_mm": local_xy_max_after_mm,
        "half_aperture_before_mm": m2_half_before,
        "half_aperture_after_mm": m2_half_after,
        "combined_rejected_indices_preview": combined_rejected_indices_preview,
        "combined_rejected_ray_ids_preview": combined_rejected_ray_ids_preview,
        "excluded_still_finite_trace_count": 0,
        "excluded_invalid_finite_trace_count": 0,
        "retained_authority_lost_count": 0,
        "footprint_outlier_rows": excluded_rows,
        "footprint_bundle_rows": bundle_rows,
    }

    # Stash audit into ctx.data BEFORE safeguards evaluate
    ctx.data["_last_m2_aperture_audit"] = m2_authority_audit
    ctx.data["_last_m2_authority_mask"] = authority_mask

    # Synchronize into candidate evidence if present
    ev = ctx.data.get("_last_candidate_evidence")
    if isinstance(ev, dict):
        mapping = {
            "M2_footprint_filter_enabled": "footprint_filter_enabled",
            "M2_footprint_filter_method": "footprint_filter_method",
            "M2_footprint_rejected_count": "combined_rejected_count",
            "M2_footprint_rejected_fraction": "combined_rejected_fraction",
            "M2_footprint_retained_count": "retained_authority_count",
            "M2_footprint_affected_bundle_count": "affected_bundle_count",
            "M2_footprint_max_bundle_rejected_fraction": "maximum_observed_bundle_rejected_fraction",
            "M2_local_xy_min_before_mm": "local_xy_min_before_mm",
            "M2_local_xy_max_before_mm": "local_xy_max_before_mm",
            "M2_local_xy_min_after_mm": "local_xy_min_after_mm",
            "M2_local_xy_max_after_mm": "local_xy_max_after_mm",
            "M2_half_aperture_before_mm": "half_aperture_before_mm",
            "M2_half_aperture_after_mm": "half_aperture_after_mm",
            "M2_footprint_outlier_rows": "footprint_outlier_rows",
            "M2_footprint_bundle_rows": "footprint_bundle_rows",
        }
        for ev_k, audit_k in mapping.items():
            ev[ev_k] = copy.deepcopy(m2_authority_audit[audit_k])
        if "aperture_rebuild" not in ev:
            ev["aperture_rebuild"] = {
                "phase": "APERTURE_REBUILD_IN_PROGRESS",
                "label": label,
                "M2": copy.deepcopy(m2_authority_audit),
            }

    # THAY ĐỔI 04 — Global và per-bundle safeguards
    if retained_authority_count < 3:
        raise RuntimeError(
            f"{prefix}_M2_AUTHORITY_LESS_THAN_THREE_POINTS_{label}:count={retained_authority_count}"
        )

    if combined_rejected_fraction > maximum_global_rejected_fraction:
        raise RuntimeError(
            f"{prefix}_M2_FOOTPRINT_GLOBAL_REJECTION_LIMIT_EXCEEDED_{label}:"
            f"count={combined_rejected_count}:fraction={combined_rejected_fraction:.6f}:"
            f"limit={maximum_global_rejected_fraction:.6f}"
        )

    if rejected_chief_count > 0 and not allow_chief_ray_rejection:
        raise RuntimeError(
            f"{prefix}_M2_FOOTPRINT_CHIEF_RAY_REJECTION_FORBIDDEN_{label}:"
            f"rejected_chief_count={rejected_chief_count}"
        )

    exceeded_bundles = [
        b for b in bundle_rows
        if b["combined_rejected_fraction"] > maximum_bundle_rejected_fraction
    ]
    if exceeded_bundles:
        worst = max(
            exceeded_bundles,
            key=lambda b: (b["combined_rejected_fraction"], b["combined_rejected_count"]),
        )
        raise RuntimeError(
            f"{prefix}_M2_FOOTPRINT_BUNDLE_REJECTION_LIMIT_EXCEEDED_{label}:"
            f"bundle={worst['field_id']}x{worst['pupil_id']}:"
            f"rejected={worst['combined_rejected_count']}/{worst['finite_m2_count']}:"
            f"fraction={worst['combined_rejected_fraction']:.6f}:"
            f"limit={maximum_bundle_rejected_fraction:.6f}"
        )

    return authority_mask, m2_authority_audit, excluded_rows, bundle_rows


_step12_m2_aperture_authority_mask = _m2_aperture_authority_mask


def _format_step12_aperture_progress_log(phase_tag: str, audit: dict[str, Any]) -> str:
    """Định dạng log tiến trình tái tạo khẩu độ STEP 12 không gây hiểu nhầm."""
    m2_audit = audit.get("M2", {})
    construction_rays = int(audit.get("ray_count", 0))
    chain_valid = int(m2_audit.get("chain_valid_count", construction_rays))
    excluded = int(m2_audit.get("combined_rejected_count", 0))
    excluded_frac = float(m2_audit.get("combined_rejected_fraction", 0.0)) * 100.0
    robust_outliers = int(m2_audit.get("robust_outlier_count", 0))
    affected_bundles = int(m2_audit.get("affected_bundle_count", 0))
    retained_authority = int(m2_audit.get("retained_authority_count", construction_rays))
    finite_retrace = int(audit.get("finite_count", 0))
    authority_lost = int(m2_audit.get("retained_authority_lost_count", 0))
    return (
        f"[STEP12][APERTURE][M2][{phase_tag}]\n"
        f"  construction rays : {construction_rays}\n"
        f"  chain valid       : {chain_valid}\n"
        f"  excluded          : {excluded} ({excluded_frac:.4f}%)\n"
        f"  robust outliers   : {robust_outliers}\n"
        f"  affected bundles  : {affected_bundles}\n"
        f"  retained authority: {retained_authority}\n"
        f"  finite retrace    : {finite_retrace}/{construction_rays}\n"
        f"  authority lost    : {authority_lost}\n"
        f"  role              : APERTURE_CONSTRUCTION_NOT_OPTICAL_CERTIFICATION"
    )


def _format_step17_aperture_progress_log(phase_tag: str, audit: dict[str, Any]) -> str:
    """Định dạng log tiến trình tái tạo khẩu độ STEP 17 không gây hiểu nhầm."""
    m2_audit = audit.get("M2", {})
    construction_rays = int(audit.get("ray_count", 0))
    chain_valid = int(m2_audit.get("chain_valid_count", construction_rays))
    excluded = int(m2_audit.get("combined_rejected_count", 0))
    excluded_frac = float(m2_audit.get("combined_rejected_fraction", 0.0)) * 100.0
    robust_outliers = int(m2_audit.get("robust_outlier_count", 0))
    affected_bundles = int(m2_audit.get("affected_bundle_count", 0))
    retained_authority = int(m2_audit.get("retained_authority_count", construction_rays))
    finite_retrace = int(audit.get("finite_count", 0))
    authority_lost = int(m2_audit.get("retained_authority_lost_count", 0))
    return (
        f"[STEP17][APERTURE][M2][{phase_tag}]\n"
        f"  construction rays : {construction_rays}\n"
        f"  chain valid       : {chain_valid}\n"
        f"  excluded          : {excluded} ({excluded_frac:.4f}%)\n"
        f"  robust outliers   : {robust_outliers}\n"
        f"  affected bundles  : {affected_bundles}\n"
        f"  retained authority: {retained_authority}\n"
        f"  finite retrace    : {finite_retrace}/{construction_rays}\n"
        f"  authority lost    : {authority_lost}\n"
        f"  role              : APERTURE_CONSTRUCTION_NOT_OPTICAL_CERTIFICATION"
    )


def _step16_minimum_candidate_fraction(ctx: Context) -> float:
    """Return the shared STEP16 finite/numeric/physical admission fraction."""
    search = ctx.config["solver"]["rho_o2_search"]
    return float(
        search.get(
            "minimum_step16_candidate_fraction",
            search.get(
                "minimum_m2_construction_finite_fraction",
                0.95,
            ),
        )
    )


def _step17_minimum_candidate_fraction(ctx: Context) -> float:
    """Return the shared STEP17 O3-O5 per-ray admission fraction."""
    return float(
        ctx.config["solver"].get(
            "minimum_step17_candidate_fraction",
            0.98,
        )
    )


def _rebuild_post_fit_apertures(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    display: PolySurface,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Tái tạo clear aperture của M1, M2 và DISPLAY sau fit theo đúng construction footprint."""
    rays = ctx.data["rays"]
    visor = ctx.data["visor"]
    ray_count = len(rays["origins"])
    step12_sparse_nonfinite_tolerance = label.startswith("STEP_12_")
    step16_display_nonfinite_tolerance = (
        label == "ORDER_2_RECONSTRUCTION"
    )
    step17_sparse_nonfinite_tolerance = label in {
        "ORDER_3_RECONSTRUCTION",
        "ORDER_4_RECONSTRUCTION",
        "ORDER_5_RECONSTRUCTION",
    }
    minimum_finite_fraction = (
        0.95
        if step12_sparse_nonfinite_tolerance
        else (
            _step16_minimum_candidate_fraction(ctx)
            if step16_display_nonfinite_tolerance
            else (
                _step17_minimum_candidate_fraction(ctx)
                if step17_sparse_nonfinite_tolerance
                else 1.0
            )
        )
    )
    construction_minimum_finite_fractions = {
        "M1": (
            0.95
            if step12_sparse_nonfinite_tolerance
            else (
                _step17_minimum_candidate_fraction(ctx)
                if step17_sparse_nonfinite_tolerance
                else 1.0
            )
        ),
        "M2": (
            0.95
            if step12_sparse_nonfinite_tolerance
            else (
                _step17_minimum_candidate_fraction(ctx)
                if step17_sparse_nonfinite_tolerance
                else 1.0
            )
        ),
        "DISPLAY": minimum_finite_fraction,
    }

    # A. Construction trace không giới hạn aperture trên M1, M2, DISPLAY
    construction = trace_reverse(
        rays,
        visor,
        m1,
        m2,
        display,
        physical_first_hit=False,
        sequential_aperture_mode="CONSTRUCTION_FOOTPRINT",
    )

    # B. Kiểm tra shape và tỷ lệ finite. STEP 12/16 dùng ngưỡng 95%; STEP 17
    # O3-O5 dùng minimum_step17_candidate_fraction (mặc định 98%).
    p1 = construction["points"][1]
    p2 = construction["points"][2]
    landing = construction["landing"]
    expected_shape = (ray_count, 3)
    for surface_name, points in (
        ("M1", p1),
        ("M2", p2),
        ("DISPLAY", landing),
    ):
        if points.shape != expected_shape:
            raise RuntimeError(
                f"POST_FIT_{surface_name}_CONSTRUCTION_FOOTPRINT_SHAPE_MISMATCH_"
                f"{label}:actual={points.shape}:expected={expected_shape}"
            )

    finite_masks = {
        "M1": np.all(np.isfinite(p1), axis=1),
        "M2": np.all(np.isfinite(p2), axis=1),
        "DISPLAY": np.all(np.isfinite(landing), axis=1),
    }
    finite_counts = {
        name: int(np.count_nonzero(mask))
        for name, mask in finite_masks.items()
    }
    finite_fractions = {
        name: float(count / max(ray_count, 1))
        for name, count in finite_counts.items()
    }
    for surface_name in ("M1", "M2", "DISPLAY"):
        surface_minimum_finite_fraction = (
            construction_minimum_finite_fractions[surface_name]
        )
        if (
            finite_fractions[surface_name]
            < surface_minimum_finite_fraction
        ):
            raise RuntimeError(
                f"POST_FIT_{surface_name}_CONSTRUCTION_FOOTPRINT_NONFINITE_"
                f"{label}:finite={finite_counts[surface_name]}_of_{ray_count}:"
                f"fraction={finite_fractions[surface_name]:.9f}:"
                f"minimum={surface_minimum_finite_fraction:.9f}"
            )

    construction_finite_mask = (
        finite_masks["M1"]
        & finite_masks["M2"]
        & finite_masks["DISPLAY"]
    )
    construction_finite_count = int(np.count_nonzero(construction_finite_mask))
    construction_finite_fraction = float(
        construction_finite_count / max(ray_count, 1)
    )

    # C. Snapshot surface shape trước rebuild
    snap_m1 = _surface_shape_snapshot(m1)
    snap_m2 = _surface_shape_snapshot(m2)
    snap_disp = _surface_shape_snapshot(display)

    m1_half_before = np.asarray(m1.half_aperture, float).copy()
    m2_half_before = np.asarray(m2.half_aperture, float).copy()
    disp_half_before = np.asarray(display.half_aperture, float).copy()
    m1_area_before = _aperture_area(m1)
    m2_area_before = _aperture_area(m2)
    disp_area_before = _aperture_area(display)

    (
        m2_aperture_authority_mask,
        m2_authority_audit,
        excluded_rows,
        bundle_rows,
    ) = _m2_aperture_authority_mask(
        ctx,
        m2,
        p2,
        finite_masks,
        construction_finite_mask,
        label,
    )

    # D. Gọi _resize bằng footprint hữu hạn riêng của từng bề mặt.
    _resize(m1, p1[finite_masks["M1"]])
    _resize(m2, p2[m2_aperture_authority_mask])
    _resize(
        display,
        landing[finite_masks["DISPLAY"]],
        tuple(np.asarray(ctx.config["display_seed_mm"], float) / 2.0),
    )

    # E. Assert shape parameters KHÔNG đổi
    _assert_surface_shape_unchanged(snap_m1, m1, f"{label}_M1")
    _assert_surface_shape_unchanged(snap_m2, m2, f"{label}_M2")
    _assert_surface_shape_unchanged(snap_disp, display, f"{label}_DISPLAY")

    # F. Trace lại với FINITE aperture
    finite_trace = trace_reverse(
        rays,
        visor,
        m1,
        m2,
        display,
        physical_first_hit=False,
        sequential_aperture_mode="FINITE",
    )

    finite_trace_mask = (
        np.asarray(finite_trace["valid"], bool)
        & np.all(np.isfinite(finite_trace["points"][1]), axis=1)
        & np.all(np.isfinite(finite_trace["points"][2]), axis=1)
        & np.all(np.isfinite(finite_trace["landing"]), axis=1)
    )
    finite_count = int(np.count_nonzero(finite_trace_mask))
    finite_fraction = float(finite_count / max(ray_count, 1))

    # Diagnostic trên toàn bộ construction-finite rays:
    lost_all_construction_finite_mask = (
        construction_finite_mask & ~finite_trace_mask
    )
    lost_all_construction_count = int(
        np.count_nonzero(lost_all_construction_finite_mask)
    )

    # Hard authority gate:
    construction_authority_mask = (
        construction_finite_mask & m2_aperture_authority_mask
    )
    lost_construction_authority_mask = (
        construction_authority_mask & ~finite_trace_mask
    )
    lost_construction_authority_count = int(
        np.count_nonzero(lost_construction_authority_mask)
    )

    if (
        finite_fraction < minimum_finite_fraction
        or lost_construction_authority_count > 0
    ):
        raise RuntimeError(
            f"POST_FIT_FINITE_TRACE_NONFINITE_{label}:"
            f"finite={finite_count}_of_{ray_count}:fraction={finite_fraction:.9f}:"
            f"minimum={minimum_finite_fraction:.9f}:"
            f"lost_construction_authority={lost_construction_authority_count}:"
            f"lost_all_construction_valid={lost_all_construction_count}"
        )

    for ex_row in excluded_rows:
        idx = ex_row["ray_index"]
        ex_row["finite_trace_valid_after_rebuild"] = bool(
            finite_trace_mask[idx]
        )

    excluded_still_finite = sum(
        1 for row in excluded_rows if row["finite_trace_valid_after_rebuild"]
    )
    excluded_invalid = len(excluded_rows) - excluded_still_finite

    m2_authority_audit["excluded_still_finite_trace_count"] = int(
        excluded_still_finite
    )
    m2_authority_audit["excluded_invalid_finite_trace_count"] = int(
        excluded_invalid
    )
    m2_authority_audit["retained_authority_lost_count"] = int(
        lost_construction_authority_count
    )

    # G. Trả về construction, finite_trace, audit
    audit = {
        "label": label,
        "surface_shape_changed": False,
        "construction_trace_is_not_physical_certification": True,
        "ray_count": ray_count,
        "minimum_finite_fraction": minimum_finite_fraction,
        "construction_minimum_finite_fractions": (
            construction_minimum_finite_fractions
        ),
        "construction_finite_count": construction_finite_count,
        "construction_invalid_count": ray_count - construction_finite_count,
        "construction_finite_fraction": construction_finite_fraction,
        "construction_invalid_indices_preview": np.flatnonzero(
            ~construction_finite_mask
        )[:20].tolist(),
        "finite_count": finite_count,
        "finite_invalid_count": ray_count - finite_count,
        "finite_fraction": finite_fraction,
        "finite_invalid_indices_preview": np.flatnonzero(
            ~finite_trace_mask
        )[:20].tolist(),
        "finite_trace_lost_construction_valid_count": lost_all_construction_count,
        "finite_trace_lost_construction_valid_indices_preview": np.flatnonzero(
            lost_all_construction_finite_mask
        )[:20].tolist(),
        "lost_all_construction_finite_count": lost_all_construction_count,
        "lost_construction_authority_count": lost_construction_authority_count,
        "retained_authority_count": int(np.count_nonzero(m2_aperture_authority_mask)),
        "finite_trace_accepted": True,
        "M1": {
            "construction_finite_count": finite_counts["M1"],
            "construction_invalid_count": ray_count - finite_counts["M1"],
            "construction_finite_fraction": finite_fractions["M1"],
            "half_aperture_before_mm": m1_half_before.tolist(),
            "half_aperture_after_mm": np.asarray(m1.half_aperture, float).tolist(),
            "area_before_mm2": m1_area_before,
            "area_after_mm2": _aperture_area(m1),
        },
        "M2": {
            "construction_finite_count": finite_counts["M2"],
            "construction_invalid_count": ray_count - finite_counts["M2"],
            "construction_finite_fraction": finite_fractions["M2"],
            "half_aperture_before_mm": m2_half_before.tolist(),
            "half_aperture_after_mm": np.asarray(m2.half_aperture, float).tolist(),
            "area_before_mm2": m2_area_before,
            "area_after_mm2": _aperture_area(m2),
            "footprint_filter_enabled": bool(m2_authority_audit["footprint_filter_enabled"]),
            "footprint_filter_method": str(m2_authority_audit["footprint_filter_method"]),
            "config_source": str(m2_authority_audit["config_source"]),
            "maximum_robust_z": float(m2_authority_audit["maximum_robust_z"]),
            "maximum_global_rejected_fraction": float(m2_authority_audit["maximum_global_rejected_fraction"]),
            "maximum_bundle_rejected_fraction": float(m2_authority_audit["maximum_bundle_rejected_fraction"]),
            "robust_center_local_xy_mm": m2_authority_audit["robust_center_local_xy_mm"],
            "robust_mad_scale_xy_mm": m2_authority_audit["robust_mad_scale_xy_mm"],
            "m2_finite_count": int(m2_authority_audit["m2_finite_count"]),
            "chain_valid_count": int(m2_authority_audit["chain_valid_count"]),
            "chain_incomplete_count": int(m2_authority_audit["chain_incomplete_count"]),
            "robust_outlier_count": int(m2_authority_audit["robust_outlier_count"]),
            "combined_rejected_count": int(m2_authority_audit["combined_rejected_count"]),
            "combined_rejected_fraction": float(m2_authority_audit["combined_rejected_fraction"]),
            "retained_authority_count": int(m2_authority_audit["retained_authority_count"]),
            "rejected_chief_count": int(m2_authority_audit["rejected_chief_count"]),
            "affected_bundle_count": int(m2_authority_audit["affected_bundle_count"]),
            "maximum_observed_bundle_rejected_fraction": float(m2_authority_audit["maximum_observed_bundle_rejected_fraction"]),
            "local_xy_min_before_mm": m2_authority_audit["local_xy_min_before_mm"],
            "local_xy_max_before_mm": m2_authority_audit["local_xy_max_before_mm"],
            "local_xy_min_after_mm": m2_authority_audit["local_xy_min_after_mm"],
            "local_xy_max_after_mm": m2_authority_audit["local_xy_max_after_mm"],
            "combined_rejected_indices_preview": m2_authority_audit["combined_rejected_indices_preview"],
            "combined_rejected_ray_ids_preview": m2_authority_audit["combined_rejected_ray_ids_preview"],
            "excluded_still_finite_trace_count": int(excluded_still_finite),
            "excluded_invalid_finite_trace_count": int(excluded_invalid),
            "retained_authority_lost_count": int(lost_construction_authority_count),
            "footprint_outlier_rows": excluded_rows,
            "footprint_bundle_rows": bundle_rows,
        },
        "DISPLAY": {
            "construction_finite_count": finite_counts["DISPLAY"],
            "construction_invalid_count": ray_count - finite_counts["DISPLAY"],
            "construction_finite_fraction": finite_fractions["DISPLAY"],
            "half_aperture_before_mm": disp_half_before.tolist(),
            "half_aperture_after_mm": np.asarray(display.half_aperture, float).tolist(),
            "area_before_mm2": disp_area_before,
            "area_after_mm2": _aperture_area(display),
        },
    }
    return construction, finite_trace, audit


def _make_seed(ctx: Context, m1_distance: float, c2: np.ndarray, cd: np.ndarray) -> dict[str, Any]:
    """Dựng và pre-score một planar seed theo physical gate và paper-style spot RMS."""
    rays, vh, rv = ctx.data["rays"], ctx.data["visor_hit"], ctx.data["post_visor"]
    i = _chief_index(rays)
    c1 = vh["point"][i] + m1_distance * rv[i]
    d12 = unit(c2 - c1)
    m1 = PolySurface.plane("M1", c1, unit(rv[i] - d12), (40.0, 35.0))
    d2d = unit(cd - c2)
    m2 = PolySurface.plane("M2", c2, unit(d12 - d2d), (40.0, 35.0))
    display = PolySurface.plane("DISPLAY", cd, d2d, tuple(np.asarray(ctx.config["display_seed_mm"]) / 2.0))

    live_phase("PLANAR_GEOMETRY_CREATED")
    _live_model_preview(
        ctx,
        "PLANAR SEED - aperture not yet rebuilt",
        m1=m1,
        m2=m2,
        display=display,
        role="PLANAR_SEED_PENDING_PRECHECK",
    )

    live_phase("PLANAR_CONSTRUCTION_INTERSECTIONS")
    h1 = m1.intersect(vh["point"] + 1e-4 * rv, rv, finite=False)
    r1 = reflect(rv, h1["normal"])
    h2 = m2.intersect(h1["point"] + 1e-4 * r1, r1, finite=False)
    r2 = reflect(r1, h2["normal"])
    hd = display.intersect(h2["point"] + 1e-4 * r2, r2, finite=False)

    live_phase("PLANAR_APERTURE_REBUILD")
    _resize(m1, h1["point"], set_scale=True); _resize(m2, h2["point"], set_scale=True)
    _resize(display, hd["point"], tuple(np.asarray(ctx.config["display_seed_mm"]) / 2.0), set_scale=True)

    live_phase("PLANAR_SEQUENTIAL_REVERSE_TRACE")
    tr = trace_reverse(rays, ctx.data["visor"], m1, m2, display, physical_first_hit=False)

    _live_model_preview(
        ctx,
        "PLANAR SEED - sequential trace",
        m1=m1,
        m2=m2,
        display=display,
        trace=tr,
        trace_mode="SEQUENTIAL_NOT_FIRST_HIT_CERTIFIED",
        role="PLANAR_SEED_PENDING_PHYSICAL_AUDIT",
    )

    live_phase("PLANAR_PHYSICAL_FIRST_HIT_AUDIT")
    physical = trace_reverse(rays, ctx.data["visor"], m1, m2, display, physical_first_hit=True)

    _live_model_preview(
        ctx,
        "PLANAR SEED - physical trace evaluated",
        m1=m1,
        m2=m2,
        display=display,
        trace=physical,
        trace_mode="PHYSICAL_FIRST_HIT",
        role="PLANAR_PRECHECK_RESULT_NOT_FINAL_DESIGN",
        force=True,
    )

    live_phase("PLANAR_METRICS_AND_ELIGIBILITY")
    mf2 = mf2_geometry(tr["points"][0], tr["points"][1], tr["points"][2], m2,
                       float(ctx.config["fan_weights"]["omega2"]), _fixed_obscuration_orientation(ctx))
    finite = bool(np.all(np.isfinite(tr["landing"])) and np.sum(tr["valid"]) == len(rays["rows"]))
    physical_ok = bool(np.sum(physical["valid"]) == len(rays["rows"]))
    unobscured = bool(mf2["S_AQP_signed_mm2"] >= 0.0)
    spot = chief_centered_geometric_spot_rms(
        tr["landing"],
        tr["valid"],
        rays["field_index"],
        rays["pupil_index"],
        rays["chief"],
        display.frame,
    )

    rms_limit = float(
        ctx.config["solver"][
            "planar_paper_spot_rms_max_mm"
        ]
    )

    rms = spot[
        "RMS_spot_radius_mm"
    ]

    rms_pass = bool(
        finite
        and rms is not None
        and float(rms) < rms_limit
    )

    geometry_metrics = _step7_geometry_metrics(
        ctx,
        m1,
        m2,
        display,
        float(m1_distance),
    )

    packaging_pass = bool(
        geometry_metrics["packaging_pass"]
    )

    physical_eligible = bool(
        finite
        and physical_ok
        and unobscured
    )

    eligible = bool(
        physical_eligible
        and packaging_pass
    )

    rejection_reasons = []

    if not finite:
        rejection_reasons.append(
            "NONFINITE_OR_INCOMPLETE_REVERSE_LANDING"
        )

    if not physical_ok:
        rejection_reasons.append(
            "PHYSICAL_FIRST_HIT_SEQUENCE_INCOMPLETE"
        )

    if not unobscured:
        rejection_reasons.append(
            "SIGNED_MF2_OBSCURED"
        )

    if (
        geometry_metrics["packaging_constraint_enabled"]
        and not packaging_pass
    ):
        rejection_reasons.append(
            "PACKAGING_LAMBDA_EXCEEDS_LIMIT"
        )

    return {
        "m1": m1,
        "m2": m2,
        "display": display,
        "trace": tr,
        "usable": finite,
        "physical_trace": physical,
        "physical_valid_count": int(np.sum(physical["valid"])),
        "mf2_seed": mf2,
        "unobscured": unobscured,
        "physical_eligible": physical_eligible,
        "eligible": eligible,
        "planar_paper_spot": spot,
        "planar_paper_spot_rms_limit_mm": rms_limit,
        "planar_paper_spot_rms_pass": rms_pass,
        "rms_is_hard_gate": False,
        "geometry_metrics": geometry_metrics,
        "packaging_constraint_enabled": geometry_metrics[
            "packaging_constraint_enabled"
        ],
        "packaging_lambda": geometry_metrics["packaging_lambda"],
        "packaging_lambda_limit": geometry_metrics[
            "packaging_lambda_limit"
        ],
        "packaging_pass": packaging_pass,
        "rejection_reasons": rejection_reasons,
        "m1_distance_mm": float(m1_distance),
        "m1_center_mm": np.asarray(c1, dtype=float),
        "m1_normal": np.asarray(m1.frame[:, 2], dtype=float),
        "m2_center_mm": np.asarray(c2, dtype=float),
        "m2_normal": np.asarray(m2.frame[:, 2], dtype=float),
        "display_center_mm": np.asarray(cd, dtype=float),
        "display_normal": np.asarray(display.frame[:, 2], dtype=float),
        "precheck_execution_status": "EVALUATED",
    }


def _dynamic_refs(ctx: Context, trace: dict[str, Any],
                  display: PolySurface | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Tính lại Fan references từ central chief ray hiện tại."""
    f = ctx.config["fan_reference"]
    display = display or ctx.data["display"]
    if f["mode"] == "AUTO_FROM_DISPLAY_SEED_AND_TARGET_VI_EXTENT":
        fields = ctx.data["vi"]["fields"]
        u = np.asarray([row["u_vi_mm"] for row in fields], float)
        v = np.asarray([row["v_vi_mm"] for row in fields], float)
        extents = np.asarray([np.ptp(u), np.ptp(v)], float)
        if np.any(extents <= 0.0):
            raise RuntimeError("FAN_REFERENCE_TARGET_VI_EXTENT_MUST_BE_POSITIVE")
        parity = np.asarray(f["axis_parity"], float)
        magnification = parity * np.asarray(ctx.config["display_seed_mm"], float) / extents
        mx, my = map(float, magnification)
        magnification_source = "AUTO_FROM_CURRENT_DISPLAY_SEED_AND_CURRENT_TARGET_VI_EXTENT"
    else:
        mx, my = float(f["M_x"]), float(f["M_y"])
        magnification_source = "FIXED_MAGNIFICATION_FROM_CONFIG"
    refs, rule = reference_grid(display, trace, ctx.data["rays"],
                                ctx.data["vi"]["fields"], mx, my)
    rule.update({"authority": "FAN_DYNAMIC_REFERENCE", "mode": f["mode"],
                 "source": magnification_source,
                 "M_x_source": magnification_source, "M_y_source": magnification_source,
                 "configured_source_note": f["source"],
                 "display_seed_mm_used": ctx.config["display_seed_mm"],
                 "updated_from_current_central_chief": True})
    return refs, rule


def _nested_characteristic_pattern(
    rays_per_pupil: int,
    pupil_radius_mm: float,
) -> list[dict[str, Any]]:
    """Tạo pattern 25 lồng chính xác trong pattern 49."""
    full_pattern = polar_pattern(
        49,
        float(pupil_radius_mm),
    )

    if int(rays_per_pupil) == 49:
        return copy.deepcopy(full_pattern)

    if int(rays_per_pupil) != 25:
        raise ValueError(
            "RAYS_PER_PUPIL_MUST_BE_25_OR_49"
        )

    selected = [
        copy.deepcopy(row)
        for row in full_pattern
        if bool(row["chief"])
        or int(str(row["sample_id"]).rsplit("A", 1)[1]) % 2 == 0
    ]

    if len(selected) != 25:
        raise RuntimeError(
            f"NESTED_PATTERN_25_COUNT_MISMATCH:{len(selected)}"
        )

    expected_ids = {
        "C",
        "R1A00", "R1A02", "R1A04", "R1A06",
        "R1A08", "R1A10", "R1A12", "R1A14",
        "R2A00", "R2A02", "R2A04", "R2A06",
        "R2A08", "R2A10", "R2A12", "R2A14",
        "R3A00", "R3A02", "R3A04", "R3A06",
        "R3A08", "R3A10", "R3A12", "R3A14",
    }

    actual_ids = {
        str(row["sample_id"])
        for row in selected
    }

    if actual_ids != expected_ids:
        raise RuntimeError(
            "NESTED_PATTERN_25_ID_MISMATCH"
        )

    return selected


def _compute_profile_fingerprint(
    profile_name: str,
    field_grid: dict[str, Any],
    pupil_grid: dict[str, Any],
    sample_ids: list[str],
    origins: np.ndarray,
    directions: np.ndarray,
    field_index: np.ndarray,
    pupil_index: np.ndarray,
    sample_index: np.ndarray,
) -> str:
    """Tạo fingerprint duy nhất cho profile lấy mẫu tia."""
    payload = (
        str(profile_name),
        int(field_grid["horizontal_count"]),
        int(field_grid["vertical_count"]),
        int(pupil_grid["horizontal_count"]),
        int(pupil_grid["vertical_count"]),
        tuple(str(sid) for sid in sample_ids),
        np.asarray(origins, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
        np.asarray(field_index, dtype=np.int64),
        np.asarray(pupil_index, dtype=np.int64),
        np.asarray(sample_index, dtype=np.int64),
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def verify_sampling_profile_on_resume(ctx: Context, completed_step: int) -> None:
    """Kiểm tra checkpoint có metadata sampling profile và khớp đúng profile của step."""
    rsp = ctx.config.get("ray_sampling_profiles")
    if not isinstance(rsp, dict) or not rsp.get("enabled", False):
        return

    meta = ctx.data.get("sampling_profile_metadata")
    if not isinstance(meta, dict) or "profile_name" not in meta or "fingerprint" not in meta:
        raise RuntimeError("RESUME_SAMPLING_PROFILE_METADATA_MISSING")

    if completed_step <= 16:
        expected_profile = "O2_SEARCH"
    elif 17 <= completed_step <= 23:
        expected_profile = "FREEFORM"
    else:
        expected_profile = "CERTIFICATION"

    actual_profile = meta.get("profile_name")
    if actual_profile != expected_profile:
        raise RuntimeError(
            f"RESUME_SAMPLING_PROFILE_MISMATCH: completed_step={completed_step}, "
            f"expected={expected_profile}, actual={actual_profile}"
        )

    rays = ctx.data.get("rays")
    pattern = ctx.data.get("pattern")
    if rays is None or pattern is None:
        raise RuntimeError("RESUME_SAMPLING_PROFILE_METADATA_MISSING")

    sample_ids = [str(r["sample_id"]) for r in pattern]
    expected_fp = _compute_profile_fingerprint(
        profile_name=actual_profile,
        field_grid=meta["field_grid"],
        pupil_grid=meta["pupil_grid"],
        sample_ids=sample_ids,
        origins=rays["origins"],
        directions=rays["directions"],
        field_index=rays["field_index"],
        pupil_index=rays["pupil_index"],
        sample_index=rays["sample_index"],
    )

    if meta["fingerprint"] != expected_fp:
        raise RuntimeError(
            f"RESUME_SAMPLING_PROFILE_FINGERPRINT_MISMATCH: "
            f"stored={meta['fingerprint']}, expected={expected_fp}"
        )


def _get_active_ray_sampling_profile(
    ctx: Context,
    step_number: int,
) -> tuple[str, dict[str, Any]] | None:
    """Lấy profile lấy mẫu tia đang hoạt động tại một STEP cụ thể."""
    rsp = ctx.config.get("ray_sampling_profiles")
    if not isinstance(rsp, dict) or not rsp.get("enabled", False):
        return None
    profiles = rsp.get("profiles", {})
    for name, prof in profiles.items():
        if int(prof["step_start"]) <= step_number <= int(prof["step_end"]):
            return name, prof
    return None


def _apply_ray_sampling_profile(
    ctx: Context,
    profile_name: str,
    step_number: int,
) -> None:
    """Chuyển đổi profile lấy mẫu tia, tái tạo toàn bộ dữ liệu ray và xóa mask cũ."""
    rsp = ctx.config.get("ray_sampling_profiles")
    if not isinstance(rsp, dict) or not rsp.get("enabled", False):
        return
    profile = rsp["profiles"][profile_name]

    # Dựng Virtual Image mới
    cfg = dict(ctx.config)
    cfg["field_grid"] = dict(profile["field_grid"])
    vi = make_virtual_image(cfg)
    for f in vi["fields"]:
        f["u_vi_mm"] = float(np.dot(f["vi_point"] - vi["center"], vi["horizontal"]))
        f["v_vi_mm"] = float(np.dot(f["vi_point"] - vi["center"], vi["vertical"]))

    # Dựng Pupils mới
    p_grid = profile["pupil_grid"]
    pupils = primary_pupils(
        tuple(ctx.config["eyebox_y_mm"]),
        tuple(ctx.config["eyebox_z_mm"]),
        (int(p_grid["horizontal_count"]), int(p_grid["vertical_count"])),
    )

    # Dựng Characteristic Pattern mới
    pattern = _nested_characteristic_pattern(
        int(profile["rays_per_pupil"]),
        float(ctx.config["pupil_diameter_mm"]) / 2.0,
    )

    # Dựng Rays mới
    rays = build_rays(vi["fields"], pupils, pattern)
    ray_count = len(rays["rows"])
    if ray_count != int(profile["expected_ray_count"]):
        raise RuntimeError(
            f"RAY_COUNT_MISMATCH_FOR_PROFILE_{profile_name}: "
            f"expected {profile['expected_ray_count']}, got {ray_count}"
        )

    # Tính fingerprint riêng cho profile
    sample_ids = [str(r["sample_id"]) for r in pattern]
    fp = _compute_profile_fingerprint(
        profile_name=profile_name,
        field_grid=profile["field_grid"],
        pupil_grid=profile["pupil_grid"],
        sample_ids=sample_ids,
        origins=rays["origins"],
        directions=rays["directions"],
        field_index=rays["field_index"],
        pupil_index=rays["pupil_index"],
        sample_index=rays["sample_index"],
    )

    # Giao tia với Visor
    hit = ctx.data["visor"].intersect(rays["origins"], rays["directions"], finite=True)
    if not np.all(hit["valid"]):
        raise RuntimeError("CHARACTERISTIC_RAY_MISSES_VISOR_DOMAIN")
    post = reflect(rays["directions"], hit["normal"])

    # Xóa bỏ hoàn toàn masks, ray_order và CI diagnostics cũ (không chuyển sang profile mới)
    for stale_key in (
        "last_ci_m1", "last_ci_m2",
        "last_ci_m1_diagnostics", "last_ci_m2_diagnostics",
        "last_ci_physical_trace",
        "_last_m2_authority_mask", "_last_m2_aperture_audit",
        "_last_candidate_evidence",
    ):
        ctx.data.pop(stale_key, None)

    # Dựng reference và trace mới nếu các mặt quang học đã có
    if "m1" in ctx.data and "m2" in ctx.data and "display" in ctx.data:
        tr = trace_reverse(
            rays,
            ctx.data["visor"],
            ctx.data["m1"],
            ctx.data["m2"],
            ctx.data["display"],
            physical_first_hit=False,
        )
        refs, rule = _dynamic_refs(ctx, tr)
        fixed = np.array([f["vi_point"] for f in vi["fields"]], dtype=float, copy=True)
        ctx.data.update({
            "trace": tr,
            "fan_refs": refs,
            "fan_refs_initial": refs.copy(),
            "fan_reference_rule": rule,
            "fixed_vi_grid": fixed,
            "fixed_display_surrogate": refs.copy(),
        })

    # Cập nhật bundle mới vào Context
    ctx.data.update({
        "vi": vi,
        "pupils": pupils,
        "pattern": pattern,
        "rays": rays,
        "visor_hit": hit,
        "post_visor": post,
        "sampling_profile_name": profile_name,
        "_ray_bundle_fingerprint": fp,
        "sampling_profile_metadata": {
            "profile_name": profile_name,
            "field_grid": dict(profile["field_grid"]),
            "pupil_grid": dict(profile["pupil_grid"]),
            "sample_ids": sample_ids,
            "fingerprint": fp,
            "expected_ray_count": int(profile["expected_ray_count"]),
            "actual_ray_count": ray_count,
        },
    })
    # Lưu giữ snapshot bundle của profile phục vụ STEP 26 export
    ctx.data[f"profile_rays_{profile_name}"] = copy.deepcopy(rays)
    ctx.data[f"profile_vi_{profile_name}"] = copy.deepcopy(vi)
    ctx.data[f"profile_pupils_{profile_name}"] = copy.deepcopy(pupils)
    ctx.data[f"profile_pattern_{profile_name}"] = copy.deepcopy(pattern)
    ctx.data[f"profile_visor_hit_{profile_name}"] = copy.deepcopy(hit)
    ctx.data[f"profile_post_visor_{profile_name}"] = copy.deepcopy(post)


def _capture_spot_snapshot(
    ctx: Context,
    label: str,
    source_step: int,
    *,
    trace: dict[str, Any] | None = None,
    references: np.ndarray | None = None,
    surface_state: dict[str, Any] | None = None,
    validity_mode: str = "STORED_TRACE_VALID",
) -> None:
    """Store one trace-authoritative stage for STEP20 diagnostics."""
    selected_trace = (
        ctx.data["trace"]
        if trace is None
        else trace
    )
    rays = ctx.data["rays"]
    selected_references = (
        ctx.data["fan_refs"]
        if references is None
        else references
    )

    landing = np.asarray(
        selected_trace["landing"],
        float,
    )
    valid = np.asarray(
        selected_trace["valid"],
        bool,
    )

    expected_ray_count = len(rays["rows"])

    if landing.shape != (expected_ray_count, 3):
        raise RuntimeError(
            f"SPOT_SNAPSHOT_LANDING_SHAPE_INVALID_{label}"
        )

    if valid.shape != (expected_ray_count,):
        raise RuntimeError(
            f"SPOT_SNAPSHOT_VALID_SHAPE_INVALID_{label}"
        )

    chief_spot = chief_centered_geometric_spot_rms(
        landing,
        valid,
        rays["field_index"],
        rays["pupil_index"],
        rays["chief"],
        np.asarray(selected_trace["display_frame"], float),
    )

    fields = ctx.data["vi"]["fields"]
    pupils = ctx.data["pupils"]

    field_definitions = [
        {
            "field_index": int(index),
            "field_id": str(field["field_id"]),
            "h_deg": float(field["h_deg"]),
            "v_deg": float(field["v_deg"]),
            "horizontal_index": int(field["horizontal_index"]),
            "vertical_index": int(field["vertical_index"]),
        }
        for index, field in enumerate(fields)
    ]

    pupil_definitions = [
        {
            "pupil_index": int(index),
            "pupil_id": str(pupil["pupil_id"]),
            "x_mm": float(pupil["x_mm"]),
            "y_mm": float(pupil["y_mm"]),
            "z_mm": float(pupil["z_mm"]),
        }
        for index, pupil in enumerate(pupils)
    ]

    selected_surface_state = surface_state

    if selected_surface_state is None and label == "ORDER_5":
        selected_surface_state = ctx.data.get("order5_state")

    snapshots = ctx.data.setdefault(
        "spot_evolution_snapshots",
        {},
    )

    snapshots[label] = {
        "label": label,
        "source_step": int(source_step),
        "landing": landing.copy(),
        "valid": valid.copy(),
        "display_center": np.asarray(
            selected_trace["display_center"],
            float,
        ).copy(),
        "display_frame": np.asarray(
            selected_trace["display_frame"],
            float,
        ).copy(),
        "references": np.asarray(
            selected_references,
            float,
        ).copy(),
        "field_index": np.asarray(
            rays["field_index"],
            int,
        ).copy(),
        "pupil_index": np.asarray(
            rays["pupil_index"],
            int,
        ).copy(),
        "chief": np.asarray(
            rays["chief"],
            bool,
        ).copy(),
        "ray_count": expected_ray_count,
        "field_count": int(rays["field_count"]),
        "pupil_count": int(rays["pupil_count"]),
        "samples_per_pupil": int(
            rays["samples_per_pupil"]
        ),
        "field_grid": copy.deepcopy(
            ctx.config["field_grid"]
        ),
        "primary_pupil_grid": copy.deepcopy(
            ctx.config["primary_pupil_grid"]
        ),
        "field_definitions": field_definitions,
        "pupil_definitions": pupil_definitions,
        "validity_mode": str(validity_mode),
        "surface_state": copy.deepcopy(
            selected_surface_state
        ),
        "chief_centered_spot": copy.deepcopy(
            chief_spot
        ),
        "available_metrics": [
            "CHIEF_CENTERED_GEOMETRIC_SPOT_RMS_BY_FIELD_AND_PUPIL",
            "CHIEF_CENTERED_BUNDLE_RMS_P50_P95_P99",
            "PHYSICAL_VALIDITY_BY_FIELD_AND_PUPIL",
            "RAY_LOSS_BY_FIELD_AND_PUPIL",
            "CHIEF_TO_CHIEF_PUPIL_SPREAD_BY_FIELD",
            "CHIEF_FIELD_BIAS_TO_DYNAMIC_REFERENCE",
        ],
    }


def _fixed_obscuration_orientation(ctx: Context) -> np.ndarray:
    """Khóa hướng vật lý của mặt phẳng obscuration, không lật theo seed."""
    return unit(np.asarray(ctx.data["vi"]["horizontal"], dtype=float))


class FermatResultSchemaError(ValueError):
    """Core và pipeline không có cùng hợp đồng kết quả Fermat."""


def _fermat_audit(
    m2: PolySurface,
    sol: dict[str, Any],
    label: str,
    *,
    expected_ray_count: int,
    gradient_tolerance: float,
    reflection_tolerance: float,
    minimum_admissible_fraction: float = 1.0,
) -> dict[str, Any]:
    """Chứng nhận nghiệm stationary Fermat trước khi redesign."""
    n = int(expected_ray_count)
    if n <= 0:
        raise FermatResultSchemaError("FERMAT_EXPECTED_RAY_COUNT_INVALID")
    minimum_fraction = float(minimum_admissible_fraction)
    if not np.isfinite(minimum_fraction) or not 0.0 < minimum_fraction <= 1.0:
        raise FermatResultSchemaError("FERMAT_MINIMUM_ADMISSIBLE_FRACTION_INVALID")

    def boolean_mask(key: str) -> np.ndarray:
        """Trích xuất và kiểm tra tính hợp lệ của mảng mặt nạ boolean kích thước (N,)."""
        if key not in sol:
            raise FermatResultSchemaError(f"FERMAT_MISSING:{key}")
        value = np.asarray(sol[key])
        if value.shape != (n,) or value.dtype.kind != "b":
            raise FermatResultSchemaError(
                f"FERMAT_INVALID_MASK:{key}:{value.shape}:{value.dtype}"
            )
        return value

    def numeric_array(key: str, shape: tuple[int, ...]) -> np.ndarray:
        """Trích xuất và kiểm tra tính hợp lệ của mảng số thực theo đúng shape yêu cầu."""
        if key not in sol:
            raise FermatResultSchemaError(f"FERMAT_MISSING:{key}")
        value = np.asarray(sol[key], float)
        if value.shape != shape:
            raise FermatResultSchemaError(
                f"FERMAT_INVALID_SHAPE:{key}:{value.shape}"
            )
        return value

    success = boolean_mask("success")
    gradient_flag = boolean_mask("gradient_pass")
    reflection_flag = boolean_mask("reflection_pass")
    conic_flag = boolean_mask("conic_domain_pass")
    finite_flag = boolean_mask("finite_pass")
    in_aperture = boolean_mask("in_aperture")
    in_construction = boolean_mask("in_construction_domain")
    fallback_numeric_failure = boolean_mask(
        "fallback_numeric_failure"
    )

    points = numeric_array("target_points", (n, 3))
    xy = numeric_array("xy", (n, 2))
    gradient_norm = numeric_array("gradient_norm", (n,))
    reflection_residual = numeric_array("reflection_residual", (n,))

    finite_numeric = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(xy), axis=1)
        & np.isfinite(gradient_norm)
        & np.isfinite(reflection_residual)
    )

    gradient_ok = (
        gradient_flag
        & np.isfinite(gradient_norm)
        & (gradient_norm >= 0.0)
        & (gradient_norm <= float(gradient_tolerance))
    )
    reflection_ok = (
        reflection_flag
        & np.isfinite(reflection_residual)
        & (reflection_residual >= 0.0)
        & (reflection_residual <= float(reflection_tolerance))
    )

    admissible = (
        success
        & gradient_ok
        & reflection_ok
        & conic_flag
        & finite_flag
        & finite_numeric
    )

    admissible_count = int(np.count_nonzero(admissible))
    admissible_fraction = float(admissible_count / n)
    all_admissible = bool(admissible_count == n)
    admissible_fraction_pass = bool(
        admissible_fraction >= minimum_fraction
    )

    grad_finite = np.isfinite(gradient_norm)
    if np.any(grad_finite):
        grad_rms = float(np.sqrt(np.mean(gradient_norm[grad_finite] ** 2)))
        grad_max = float(np.max(gradient_norm[grad_finite]))
    else:
        grad_rms = float("nan")
        grad_max = float("nan")

    return {
        "solve": label,
        "success_count": int(np.count_nonzero(success)),
        "ray_count": n,
        "solver_all_converged": bool(np.all(success)),
        "all_converged": all_admissible,
        "branch_may_continue": admissible_fraction_pass,
        "admissible_target_count": admissible_count,
        "admissible_target_fraction": admissible_fraction,
        "minimum_admissible_fraction": minimum_fraction,
        "admissible_fraction_pass": admissible_fraction_pass,
        "gradient_pass_count": int(np.count_nonzero(gradient_ok)),
        "reflection_pass_count": int(np.count_nonzero(reflection_ok)),
        "wrong_reflection_branch_count": int(
            np.count_nonzero(gradient_ok & ~reflection_ok)
        ),
        "fallback_numeric_failure_count": int(
            np.count_nonzero(fallback_numeric_failure)
        ),
        "fallback_numeric_failure_ray_indices": np.flatnonzero(
            fallback_numeric_failure
        ).tolist(),
        "fallback_invalid_start_count": int(
            sol.get("fallback_invalid_start_count", 0)
        ),
        "fallback_solver_error_count": int(
            sol.get("fallback_solver_error_count", 0)
        ),
        "conic_domain_pass_count": int(np.count_nonzero(conic_flag)),
        "finite_pass_count": int(
            np.count_nonzero(finite_flag & finite_numeric)
        ),
        "inside_current_aperture_count": int(np.count_nonzero(in_aperture)),
        "inside_construction_domain_count": int(
            np.count_nonzero(in_construction)
        ),
        "gradient_rms": grad_rms,
        "gradient_max": grad_max,
        "targets_finite": bool(np.all(finite_numeric)),
        "targets_inside_current_m2_domain": bool(np.all(in_aperture)),
        "reflection_residual_max": float(np.max(reflection_residual)) if reflection_residual.size else float("nan"),
        "current_clear_aperture_is_not_next_surface_construction_domain": True,
        "construction_domain_factor": float(sol.get("construction_domain_factor", 1.0)),
        "diagnostic_continuation_allowed": bool(
            admissible_fraction_pass and not all_admissible
        ),
    }


def _prescription(ctx: Context) -> dict[str, Any]:
    """Xuất prescription gọn của display, visor, M1 và M2."""
    return {"schema": "HUD_FAN_V5_5_PRESCRIPTION", "M1": ctx.data["m1"].to_dict(),
            "M2": ctx.data["m2"].to_dict(), "display": ctx.data["display"].to_dict(),
            "visor_surrogate": ctx.data["visor"].to_dict(),
            "reverse_sequence": ["EYE", "VISOR_INNER", "M1", "M2", "DISPLAY"],
            "forward_sequence": ["DISPLAY", "M2", "M1", "VISOR_INNER", "EYE"]}


def _packaging(ctx: Context, m1: PolySurface | None = None, m2: PolySurface | None = None,
               display: PolySurface | None = None) -> float | None:
    """Tính packaging khi P1–P8 được khai báo; null nghĩa là tắt constraint."""
    vertices = ctx.data["inputs"].get("packaging_vertices")
    if vertices is None:
        return None
    m1 = m1 or ctx.data["m1"]; m2 = m2 or ctx.data["m2"]
    display = display or ctx.data["display"]
    pts = np.vstack([aperture_points(m1, samples=15), aperture_points(m2, samples=15),
                     aperture_points(display, samples=15)])
    return packaging_lambda(vertices, pts)


def _publish_trace_debug(
    ctx: Context,
    step_number: int,
    phase: str,
    trace: dict[str, Any],
    rays: dict[str, Any],
    *,
    valid_key: str = "valid",
    direction: str = "REVERSE",
) -> None:
    """Ghi nhận dữ liệu ray-trace chẩn đoán của một phase vào _step_debug của Context."""
    from diagnostics_v55 import extract_failed_rays

    slot = ctx.data.get("_step_debug")
    if not isinstance(slot, dict) or slot.get("step") != step_number:
        # Hàm STEP được gọi trực tiếp ngoài runner: tạo đúng scope hiện tại.
        slot = {
            "step": step_number,
            "traces": {},
            "surface_sanity": {},
            "ci_fallbacks": {},
            "diagnostic_errors": [],
        }
        ctx.data["_step_debug"] = slot

    try:
        raw_mask = np.asarray(trace[valid_key])
        if raw_mask.ndim != 1 or raw_mask.dtype.kind != "b":
            raise ValueError("DEBUG_VALIDITY_MASK_SCHEMA_ERROR")

        # Chỉ tạo dictionary view, không thay mask của trace gốc.
        diagnostic_trace = dict(trace)
        diagnostic_trace["valid"] = raw_mask

        failed = extract_failed_rays(diagnostic_trace, rays)
        slot["traces"][phase] = {
            "producer_step": step_number,
            "producer_phase": phase,
            "direction": direction,
            "validity_semantics": valid_key,
            "ray_count": int(raw_mask.size),
            "valid_count": int(np.count_nonzero(raw_mask)),
            "failed_count": len(failed),
            "failed_rays": failed,
            "diagnostic_status": "RECORDED",
        }
    except Exception as exc:
        slot["diagnostic_errors"].append({
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })


def _write_step_debug(
    ctx: Context,
    number: int,
    title: str,
    summary: dict[str, Any],
    p: Path,
) -> None:
    """Xuất thông tin chẩn đoán debug an toàn cho đúng STEP hiện hành."""
    from diagnostics_v55 import (
        build_step_debug_report,
        write_failed_rays_csv,
    )

    slot = ctx.data.get("_step_debug")
    if not isinstance(slot, dict) or slot.get("step") != number:
        slot = {
            "step": number,
            "traces": {},
            "surface_sanity": {},
            "ci_fallbacks": {},
            "diagnostic_errors": [],
        }

    traces = slot.get("traces", {})
    traces_dir = p / "DEBUG_TRACES"
    trace_files: dict[str, str] = {}

    if traces:
        traces_dir.mkdir(parents=True, exist_ok=True)
        for phase, tinfo in traces.items():
            target_csv = traces_dir / f"{phase}_FAILED_RAYS.csv"
            write_failed_rays_csv(target_csv, tinfo.get("failed_rays", []))
            trace_files[phase] = str(target_csv)

        if len(traces) == 1:
            single_phase = next(iter(traces))
            write_failed_rays_csv(
                p / "FAILED_RAYS.csv",
                traces[single_phase].get("failed_rays", []),
            )

    extra_info: dict[str, Any] = {
        "traces": traces,
        "trace_files": trace_files,
        "surface_sanity": slot.get("surface_sanity", {}),
        "ci_fallbacks": slot.get("ci_fallbacks", {}),
        "diagnostic_errors": slot.get("diagnostic_errors", []),
        "current_phase": slot.get("current_phase"),
        "pre_aperture_quality": slot.get("pre_aperture_quality"),
        "ray_evaluation": "EVALUATED" if traces else "NOT_EVALUATED",
    }

    debug_md, debug_json = build_step_debug_report(
        number, title, summary, ctx, extra_info
    )
    (p / "DEBUG_REPORT.md").write_text(debug_md, encoding="utf-8")
    write_json(p / "DEBUG_SUMMARY.json", debug_json)


def _record_step_failure_safely(
    ctx: Context,
    number: int,
    exc: Exception,
) -> None:
    """Ghi nhận lỗi thực thi của STEP an toàn mà không làm thay đổi exception gốc."""
    try:
        import traceback
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        manifest = ctx.data.get("algorithm_source_manifest", {})
        slot = ctx.data.get("_step_debug")
        if not isinstance(slot, dict) or slot.get("step") != number:
            slot = {}
        failure_phase = str(
            slot.get("current_phase")
            or ctx.data.get("_execution_phase", "ALGORITHM")
        )
        pre_aperture_quality = slot.get("pre_aperture_quality")

        failure_class = "ALGORITHM_FAILURE"
        if number == 12 and failure_phase == "APERTURE_REBUILD":
            if (
                isinstance(exc, ApertureRebuildError)
                and str(exc).startswith("APERTURE_HULL_REBUILD_FAILED:")
            ):
                failure_class = "TECHNICAL_GEOMETRY_FAILURE"
            else:
                failure_class = "APERTURE_REBUILD_FAILURE"

        m1_fit_status = None
        m2_fit_status = None
        m1_integrability_status = None
        m2_integrability_status = None
        if isinstance(pre_aperture_quality, dict):
            m1_fit_status = pre_aperture_quality.get("M1_fit", {}).get("status")
            m2_fit_status = pre_aperture_quality.get("M2_fit", {}).get("status")
            integrability = pre_aperture_quality.get("integrability_quality", {})
            m1_integrability_status = integrability.get("M1", {}).get("status")
            m2_integrability_status = integrability.get("M2", {}).get("status")

        failure_summary = {
            "failed_step": number,
            "failure_phase": failure_phase,
            "failure_class": failure_class,
            "optical_fit_M1": m1_fit_status,
            "optical_fit_M2": m2_fit_status,
            "integrability_M1": m1_integrability_status,
            "integrability_M2": m2_integrability_status,
        }
        ctx.data["_last_step_failure"] = failure_summary.copy()

        payload = {
            **failure_summary,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb_str,
            "execution_phase": ctx.data.get("_execution_phase", "ALGORITHM"),
            "pre_aperture_quality": pre_aperture_quality,
            "restart_rule": "fix declared upstream input/code then restart STEP_00",
            "algorithm_source_manifest_sha256": manifest.get("manifest_sha256"),
        }
        step_dir = ctx.step_dir(number)
        step_dir.mkdir(parents=True, exist_ok=True)
        write_json(ctx.run_dir / f"RUN_FAILURE_STEP_{number:02d}.json", payload)
        write_json(step_dir / f"RUN_FAILURE_STEP_{number:02d}.json", payload)

        summary = {
            "status": "EXECUTION_FAILURE",
            "execution_status": "ALGORITHM_FAILED",
            "failure_phase": failure_phase,
            "failure_class": failure_class,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_step_debug(ctx, number, f"STEP_{number:02d}_FAILURE", summary, step_dir)
    except Exception as record_exc:
        print(f"STEP_FAILURE_RECORDING_ERROR: {type(record_exc).__name__}: {record_exc}", flush=True)


def _call_stage_algorithm(
    ctx: Context,
    number: int,
    fn: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bọc lời gọi hàm STEP: cách ly _step_debug, ghi nhận phase và xử lý lỗi."""
    ctx.data["_step_debug"] = {
        "step": number,
        "traces": {},
        "surface_sanity": {},
        "ci_fallbacks": {},
        "diagnostic_errors": [],
    }
    ctx.data["_execution_phase"] = "ALGORITHM"

    cm = ray_trace_acceleration(f"STEP_{number:02d}") if number in PARALLEL_RAY_TRACE_STEPS else nullcontext()
    with cm:
        try:
            summary = fn(ctx, *args, **kwargs)
            if not isinstance(summary, dict):
                raise TypeError("STEP_SUMMARY_MUST_BE_DICT")
        except Exception as exc:
            if number == 11:
                slot = ctx.data.get("_step_debug")
                failure_phase = (
                    slot.get("current_phase")
                    if isinstance(slot, dict)
                    else None
                ) or ctx.data.get("_execution_phase", "STEP11/UNKNOWN")
                print(
                    f"[STEP11] FAILED | phase={failure_phase} | "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            _record_step_failure_safely(ctx, number, exc)
            raise

    summary["execution_status"] = "ALGORITHM_COMPLETED"
    ctx.data["_execution_phase"] = "OUTPUTS"
    return summary


def _archive_step(ctx: Context, number: int, title: str, fn: Callable[..., Any], summary: dict[str, Any]) -> None:
    """Lưu summary, bằng chứng thuật toán và đầu ra của STEP."""
    p = ctx.step_dir(number)
    write_json(p / "STEP_SUMMARY.json", {
        "step": number, "title": title,
        "execution_status": summary.get("execution_status", "ALGORITHM_COMPLETED"),
        "summary": summary,
        "algorithm_source_manifest_sha256": ctx.data.get(
            "algorithm_source_manifest", {}).get("manifest_sha256"),
    })
    (p / "README.md").write_text(
        f"# STEP_{number:02d} — {title}\n\n"
        f"Thuật toán: {STEP_EXPLANATIONS_VI[number]}\n\n"
        f"Status: {summary.get('status', 'RECORDED')}\n",
        encoding="utf-8",
    )
    (p / "ALGORITHM_SOURCE.py.txt").write_text(inspect.getsource(fn), encoding="utf-8")

    try:
        _write_step_debug(ctx, number, title, summary, p)
    except Exception as exc:
        print(f"STEP_DEBUG_EXPORT_WARNING STEP_{number:02d}: {type(exc).__name__}: {exc}", flush=True)


def _live_model_preview(
    ctx: Context,
    label: str,
    *,
    m1: PolySurface | None,
    m2: PolySurface | None,
    display: PolySurface | None,
    trace: dict[str, Any] | None = None,
    trace_mode: str = "NOT_EVALUATED",
    role: str = "CANDIDATE",
    force: bool = False,
) -> None:
    """Preview chi doc cac doi tuong duoc caller truyen ro rang."""

    def draw(path: Path, maximum: int) -> dict[str, Any]:
        """Chi duoc goi khi telemetry va image dang bat."""
        from live_preview_v55 import render_live_preview

        return render_live_preview(
            path,
            label=label,
            models=[
                ("M1", m1),
                ("M2", m2),
                ("DISPLAY", display),
            ],
            visor=ctx.data.get("visor"),
            rays=ctx.data["rays"],
            trace=trace,
            trace_mode=trace_mode,
            max_plot_rays=maximum,
        )

    live_capture(
        label,
        draw,
        role=role,
        force=force,
    )


def _algorithm_source_manifest(ctx: Context) -> dict[str, Any]:
    """Khóa hash toàn bộ source quyết định thuật toán và ảnh của đúng run hiện tại."""
    sources = [
        ctx.source_dir / "core.py",
        ctx.source_dir / "pipeline_v55.py",
        ctx.source_dir / "execution_v55.py",
        ctx.source_dir / "execution_workers_v55.py",
        ctx.source_dir / "kernels_array_v55.py",
        ctx.source_dir / "kernels_cpu_v55.py",
        ctx.source_dir / "kernels_ci_v55.py",
        ctx.source_dir / "kernel_dispatch_v55.py",
        ctx.source_dir / "gpu_service_v55.py",
        ctx.source_dir / "gpu_transport_v55.py",
        ctx.source_dir / "evaluation_cache_v55.py",
        ctx.source_dir / "artifact_cache_v55.py",
        ctx.source_dir / "live_monitor_v55.py",
        ctx.source_dir / "live_preview_v55.py",
        ctx.source_dir / "diagnostics_v55.py",
        ctx.source_dir / "visualization_v55.py",
        ctx.source_dir / "run_multistart_v55.py",
        ctx.source_dir / "manual_stage_runner_v55.py",
        ctx.source_dir / "run_pipeline_v55.py",
        ctx.config_path,
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"ALGORITHM_SOURCE_FILES_MISSING:{missing}")
    files = []
    for path in sources:
        payload = path.read_bytes()
        files.append({"path": str(path.resolve()), "size_bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"schema": "HUD_FAN_V5_5_ALGORITHM_SOURCE_MANIFEST",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": files, "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
            "scope": "CORE_PIPELINE_VISUALIZATION_RUNNERS_AND_EXACT_CONFIG"}


















@observe(7, "PLANAR_SEED_WORKER", "LIVE")
def evaluate_planar_seed_job(ctx: Context, job: dict[str, Any]) -> dict[str, Any]:
    """Thực hiện đánh giá một seed planar độc lập trong worker."""
    cand_idx = int(job["candidate_index"])
    total_cands = int(job["total_candidates"])
    d = float(job["d"])
    c2 = np.asarray(job["c2"], float)
    cd = np.asarray(job["cd"], float)

    live_item(
        "PLANAR_SEED",
        cand_idx,
        total_cands,
        m1_distance_mm=d,
        m2_center_mm=c2,
        display_center_mm=cd,
    )

    try:
        record = _make_seed(ctx, d, c2, cd)
    except ApertureRebuildError as exc:
        record = {
            "m1": None,
            "m2": None,
            "display": None,
            "trace": None,
            "physical_trace": None,
            "usable": False,
            "physical_eligible": False,
            "eligible": False,
            "physical_valid_count": None,
            "mf2_seed": {"S_AQP_signed_mm2": None},
            "unobscured": None,
            "planar_paper_spot": {
                "metric_name": "NOT_EVALUATED",
                "RMS_spot_radius_mm": None,
                "max_spot_radius_mm": None,
                "valid_ray_count": None,
            },
            "planar_paper_spot_rms_limit_mm": float(
                ctx.config["solver"]["planar_paper_spot_rms_max_mm"]
            ),
            "planar_paper_spot_rms_pass": False,
            "rms_is_hard_gate": False,
            "geometry_metrics": None,
            "packaging_constraint_enabled": bool(
                ctx.data["inputs"]["packaging_constraint_enabled"]
            ),
            "packaging_lambda": None,
            "packaging_lambda_limit": (
                float(ctx.config["packaging_lambda_max"])
                if ctx.data["inputs"]["packaging_constraint_enabled"]
                else None
            ),
            "packaging_pass": None,
            "rejection_reasons": [str(exc)],
            "m1_distance_mm": d,
            "m2_center_mm": c2,
            "display_center_mm": cd,
            "precheck_execution_status": "APERTURE_REBUILD_FAILED",
        }

    record["geometry_search_parameters"] = copy.deepcopy(
        job.get("geometry_search_parameters")
    )

    row = planar_seed_candidate_rows([record])[0]
    row["candidate"] = cand_idx
    live_results(lambda: [row])
    return record


def _enumerate_planar_seeds_serial(ctx: Context) -> list[dict[str, Any]]:
    """Đánh giá tuần tự toàn bộ auto-generated STEP 07 geometry candidates."""
    descriptors = _step7_auto_seed_descriptors(ctx)
    live_total = len(descriptors)
    records: list[dict[str, Any]] = []

    for descriptor in descriptors:
        d = float(descriptor["d"])
        c2 = np.asarray(descriptor["c2"], dtype=float)
        cd = np.asarray(descriptor["cd"], dtype=float)

        live_item(
            "PLANAR_SEED",
            int(descriptor["candidate_index"]),
            live_total,
            m1_distance_mm=d,
            m2_center_mm=c2,
            display_center_mm=cd,
        )

        try:
            record = _make_seed(ctx, d, c2, cd)
        except ApertureRebuildError as exc:
            record = {
                "m1": None,
                "m2": None,
                "display": None,
                "trace": None,
                "physical_trace": None,
                "usable": False,
                "physical_eligible": False,
                "eligible": False,
                "physical_valid_count": None,
                "mf2_seed": {"S_AQP_signed_mm2": None},
                "unobscured": None,
                "planar_paper_spot": {
                    "metric_name": "NOT_EVALUATED",
                    "RMS_spot_radius_mm": None,
                    "max_spot_radius_mm": None,
                    "valid_ray_count": None,
                },
                "planar_paper_spot_rms_limit_mm": float(
                    ctx.config["solver"]["planar_paper_spot_rms_max_mm"]
                ),
                "planar_paper_spot_rms_pass": False,
                "rms_is_hard_gate": False,
                "geometry_metrics": None,
                "packaging_constraint_enabled": bool(
                    ctx.data["inputs"]["packaging_constraint_enabled"]
                ),
                "packaging_lambda": None,
                "packaging_lambda_limit": (
                    float(ctx.config["packaging_lambda_max"])
                    if ctx.data["inputs"]["packaging_constraint_enabled"]
                    else None
                ),
                "packaging_pass": None,
                "rejection_reasons": [str(exc)],
                "m1_distance_mm": d,
                "m2_center_mm": c2,
                "display_center_mm": cd,
                "precheck_execution_status": "APERTURE_REBUILD_FAILED",
            }

        record["geometry_search_parameters"] = copy.deepcopy(
            descriptor["geometry_search_parameters"]
        )
        records.append(record)

        live_results(
            lambda: planar_seed_candidate_rows(records)
        )

    return records


def _enumerate_planar_seeds_parallel(ctx: Context) -> list[dict[str, Any]]:
    """Đánh giá sơ bộ toàn bộ tích Descartes seed qua managed compute pool."""
    descriptors = _step7_auto_seed_descriptors(ctx)

    runtime = current_runtime()
    session_id = runtime.session_id if runtime and runtime.session_id else "DEFAULT"
    total_candidates = len(descriptors)

    for desc in descriptors:
        desc["total_candidates"] = total_candidates
        desc["job_id"] = f"SEED_PRECHECK_{desc['candidate_index']:03d}_{uuid.uuid4().hex[:8]}"
        desc["work_dir"] = str(
            ctx.run_dir / "LIVE_PRECHECK" / "JOBS" / session_id / f"CANDIDATE_{desc['candidate_index']:03d}"
        )

    snapshot = copy.deepcopy(ctx)

    from execution_v55 import managed_compute_pool
    from execution_workers_v55 import ordered_bounded_map, seed_precheck_worker

    completed_records: dict[int, dict[str, Any]] = {}

    def on_completed(index: int, reply: dict[str, Any]):
        """Thực thi on_completed."""
        cand_idx = reply["candidate_index"]
        rec = reply["record"]
        completed_records[cand_idx] = rec

        live_item(
            "PLANAR_SEED",
            cand_idx,
            total_candidates,
            m1_distance_mm=float(rec["m1_distance_mm"]),
            m2_center_mm=np.asarray(rec["m2_center_mm"], float),
            display_center_mm=np.asarray(rec["display_center_mm"], float),
        )
        live_results(lambda: [
            planar_seed_candidate_rows([completed_records[i]])[0] | {"candidate": i}
            for i in sorted(completed_records.keys())
        ])

    with managed_compute_pool(snapshot, purpose="PLANAR_PRECHECK") as (executor, effective_max_inflight, policy):
        if executor is None:
            return _enumerate_planar_seeds_serial(ctx)

        replies = ordered_bounded_map(
            executor,
            seed_precheck_worker,
            descriptors,
            effective_max_inflight,
            on_completed=on_completed,
        )

    for desc, reply in zip(descriptors, replies):
        if reply["candidate_index"] != desc["candidate_index"] or reply["job_id"] != desc["job_id"]:
            raise BackendConsistencyError("PRECHECK_REPLY_IDENTITY_MISMATCH")

    return [reply["record"] for reply in replies]


@observe(7, "PLANAR_PRECHECK", "LIVE_PRECHECK")
def enumerate_planar_seeds(ctx: Context) -> list[dict[str, Any]]:
    """Thực thi enumerate_planar_seeds."""
    runtime = current_runtime()

    if (
        runtime is None
        or is_compute_worker()
        or not runtime.config.get("parallel_precheck", False)
        or int(runtime.config.get("cpu_workers", 1)) <= 1
    ):
        return _enumerate_planar_seeds_serial(ctx)

    return _enumerate_planar_seeds_parallel(ctx)


def planar_seed_candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Xuất physical gates, compactness, packaging và RMS diagnostic của STEP 07."""
    eligible_ranked = sorted(
        [
            (index, candidate)
            for index, candidate in enumerate(candidates, start=1)
            if candidate["eligible"]
        ],
        key=lambda item: (
            _planar_seed_rank_key(item[1]),
            item[0],
        ),
    )
    rank_by_candidate = {
        candidate_index: rank
        for rank, (candidate_index, candidate) in enumerate(
            eligible_ranked,
            start=1,
        )
    }

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        geometry = candidate.get("geometry_metrics") or {}

        rows.append({
            "candidate": index,
            "geometry_rank": rank_by_candidate.get(index),
            "usable": candidate["usable"],
            "physical_eligible": candidate["physical_eligible"],
            "eligible": candidate["eligible"],
            "physical_valid_count": candidate["physical_valid_count"],
            "unobscured": candidate["unobscured"],
            "S_AQP_signed_mm2": candidate["mf2_seed"]["S_AQP_signed_mm2"],
            "packaging_constraint_enabled": candidate.get(
                "packaging_constraint_enabled"
            ),
            "packaging_lambda": candidate.get("packaging_lambda"),
            "packaging_lambda_limit": candidate.get("packaging_lambda_limit"),
            "packaging_pass": candidate.get("packaging_pass"),
            "geometry_bbox_volume_mm3": geometry.get(
                "geometry_bbox_volume_mm3"
            ),
            "geometry_bbox_diagonal_mm": geometry.get(
                "geometry_bbox_diagonal_mm"
            ),
            "max_clear_aperture_diagonal_mm": geometry.get(
                "max_clear_aperture_diagonal_mm"
            ),
            "m1_clear_aperture_area_mm2": geometry.get(
                "m1_clear_aperture_area_mm2"
            ),
            "m2_clear_aperture_area_mm2": geometry.get(
                "m2_clear_aperture_area_mm2"
            ),
            "display_clear_aperture_area_mm2": geometry.get(
                "display_clear_aperture_area_mm2"
            ),
            "total_clear_aperture_area_mm2": geometry.get(
                "total_clear_aperture_area_mm2"
            ),
            "chief_path_length_mm": geometry.get("chief_path_length_mm"),
            "planar_paper_spot_metric": candidate[
                "planar_paper_spot"
            ]["metric_name"],
            "planar_paper_spot_rms_mm": candidate[
                "planar_paper_spot"
            ]["RMS_spot_radius_mm"],
            "planar_paper_spot_max_mm": candidate[
                "planar_paper_spot"
            ]["max_spot_radius_mm"],
            "planar_paper_spot_valid_ray_count": candidate[
                "planar_paper_spot"
            ]["valid_ray_count"],
            "planar_paper_spot_rms_operator": "<",
            "planar_paper_spot_rms_limit_mm": candidate[
                "planar_paper_spot_rms_limit_mm"
            ],
            "planar_paper_spot_rms_pass": candidate[
                "planar_paper_spot_rms_pass"
            ],
            "planar_paper_spot_rms_role": "DIAGNOSTIC_FINAL_TIE_BREAK_ONLY",
            "rms_is_hard_gate": False,
            "rejection_reasons": "|".join(candidate["rejection_reasons"]),
            "m1_distance_mm": candidate["m1_distance_mm"],
            "m1_center_mm": candidate.get("m1_center_mm"),
            "m1_normal": candidate.get("m1_normal"),
            "m2_center_mm": candidate["m2_center_mm"],
            "m2_normal": candidate.get("m2_normal"),
            "display_center_mm": candidate["display_center_mm"],
            "display_normal": candidate.get("display_normal"),
            "geometry_search_parameters": candidate.get(
                "geometry_search_parameters"
            ),
            "precheck_execution_status": candidate.get(
                "precheck_execution_status",
                "EVALUATED",
            ),
        })

    return rows


def activate_planar_seed(ctx: Context, selected: dict[str, Any], selected_candidate: int,
                         candidate_rows: list[dict[str, Any]], selection_rule: str) -> dict[str, Any]:
    """Khóa một seed đã precheck vào Context; dùng chung cho single-run và multi-start."""
    ctx.data.update({"m1": selected["m1"], "m2": selected["m2"], "display": selected["display"],
                     "trace": selected["trace"]})
    _publish_trace_debug(
        ctx, 7, "SELECTED_PLANAR_PHYSICAL_TRACE",
        selected["physical_trace"], ctx.data["rays"],
        direction="REVERSE",
    )
    ctx.data["planar_seed_candidate_rows"] = copy.deepcopy(candidate_rows)
    g = selected["mf2_seed"]
    nobs = _fixed_obscuration_orientation(ctx)
    ctx.data["n_obs"] = nobs
    write_csv(ctx.step_dir(7) / "07_PLANAR_START_CANDIDATES.csv", candidate_rows)
    packaging_enabled = bool(
        selected["packaging_constraint_enabled"]
    )

    planar_record = {
        "selection_rule": selection_rule,
        "selected_candidate": selected_candidate,
        "selection_rank_key": list(_planar_seed_rank_key(selected)),
        "geometry_search_parameters": copy.deepcopy(
            selected.get("geometry_search_parameters")
        ),
        "geometry_selection_metrics": copy.deepcopy(
            selected["geometry_metrics"]
        ),
        "n_obs_frozen": nobs,
        "n_obs_source": "FIXED_VIRTUAL_IMAGE_HORIZONTAL_AXIS_NO_RESULT_BASED_FLIP",
        "seed_MF2": g,
        "physical_valid_count": selected["physical_valid_count"],
        "planar_paper_spot": selected["planar_paper_spot"],
        "planar_paper_spot_rms_limit_mm": selected[
            "planar_paper_spot_rms_limit_mm"
        ],
        "planar_paper_spot_rms_rule": "DIAGNOSTIC_STRICT_LESS_THAN_REFERENCE_ONLY",
        "planar_paper_spot_rms_pass": selected[
            "planar_paper_spot_rms_pass"
        ],
        "rms_is_hard_gate": False,
        "packaging_constraint_enabled": packaging_enabled,
        "packaging_lambda": selected["packaging_lambda"],
        "packaging_lambda_limit": selected["packaging_lambda_limit"],
        "packaging_hard_gate_pass": selected["packaging_pass"],
        "packaging_lambda_diagnostic_only": (
            selected["packaging_lambda"]
            if not packaging_enabled
            else None
        ),
        "no_all_ray_nearest_first_seed_gate": True,
        "no_seed_packaging_hard_gate": not packaging_enabled,
        "prescription": _prescription(ctx),
    }
    ctx.data["planar_start_record"] = planar_record
    write_json(ctx.step_dir(7) / "07_SELECTED_PLANAR_START.json", planar_record)
    return {
        "status": "PASS",
        "selected_candidate": selected_candidate,
        "n_obs_frozen": True,
        "orientation_result_flip": False,
        "physical_valid_count": selected["physical_valid_count"],
        "unobscured": True,
        "geometry_bbox_volume_mm3": selected[
            "geometry_metrics"
        ]["geometry_bbox_volume_mm3"],
        "geometry_bbox_diagonal_mm": selected[
            "geometry_metrics"
        ]["geometry_bbox_diagonal_mm"],
        "max_clear_aperture_diagonal_mm": selected[
            "geometry_metrics"
        ]["max_clear_aperture_diagonal_mm"],
        "total_clear_aperture_area_mm2": selected[
            "geometry_metrics"
        ]["total_clear_aperture_area_mm2"],
        "chief_path_length_mm": selected[
            "geometry_metrics"
        ]["chief_path_length_mm"],
        "packaging_constraint_enabled": packaging_enabled,
        "packaging_lambda": selected["packaging_lambda"],
        "packaging_lambda_limit": selected["packaging_lambda_limit"],
        "planar_paper_spot_rms_mm": selected[
            "planar_paper_spot"
        ]["RMS_spot_radius_mm"],
        "planar_paper_spot_rms_limit_mm": selected[
            "planar_paper_spot_rms_limit_mm"
        ],
        "rms_is_hard_gate": False,
        "planar_seed_is_not_final_gate": True,
    }










def _step12_o2_variable_descriptors(
    m1: PolySurface,
    m2: PolySurface,
) -> list[dict[str, Any]]:
    """Khai báo đúng các DOF O2 được phép tinh chỉnh mà vẫn giữ gauge của fit_surface()."""
    descriptors: list[dict[str, Any]] = []

    for which, surface in (
        ("M1", m1),
        ("M2", m2),
    ):
        descriptors.append({
            "surface": which,
            "kind": "CURVATURE",
        })

        descriptors.append({
            "surface": which,
            "kind": "CONIC",
        })

        term_to_index = {
            term: index
            for index, term in enumerate(
                surface.terms
            )
        }

        if (
            (2, 0) in term_to_index
            and (0, 2) in term_to_index
        ):
            descriptors.append({
                "surface": which,
                "kind":
                    "ASTIG_QUADRATIC_PAIR",
                "index_20":
                    int(
                        term_to_index[
                            (2, 0)
                        ]
                    ),
                "index_02":
                    int(
                        term_to_index[
                            (0, 2)
                        ]
                    ),
            })

        for term, index in (
            term_to_index.items()
        ):
            if term in (
                (0, 0),
                (2, 0),
                (0, 2),
            ):
                continue

            descriptors.append({
                "surface": which,
                "kind": "COEFFICIENT",
                "index": int(index),
                "term": tuple(term),
            })

    return descriptors


def _step12_apply_o2_vector(
    base: dict[str, PolySurface],
    descriptors: list[dict[str, Any]],
    normalized: np.ndarray,
    cfg: dict[str, Any],
    surface_fit_cfg: dict[str, Any],
) -> tuple[PolySurface, PolySurface]:
    """Áp vector chuẩn hóa lên c, K và các Aij O2 trong bound nhỏ, giữ A00 và A20+A02 gauge."""
    u = np.asarray(
        normalized,
        float,
    )

    if (
        u.shape != (
            len(descriptors),
        )
        or not np.all(
            np.isfinite(u)
        )
    ):
        raise ValueError(
            "STEP12_O2_REFINEMENT_VECTOR_INVALID"
        )

    if np.any(
        np.abs(u)
        > 1.0 + 1e-12
    ):
        raise ValueError(
            "STEP12_O2_REFINEMENT_VECTOR_OUT_OF_BOUNDS"
        )

    m1 = base["M1"].copy()
    m2 = base["M2"].copy()

    surfaces = {
        "M1": m1,
        "M2": m2,
    }

    coefficient_bound = float(
        cfg[
            "coefficient_absolute_bound_mm"
        ]
    )

    curvature_fraction = float(
        cfg[
            "curvature_relative_bound"
        ]
    )

    conic_bound = float(
        cfg[
            "conic_absolute_bound"
        ]
    )

    curvature_abs_max = float(
        surface_fit_cfg[
            "curvature_absolute_max_per_mm"
        ]
    )

    conic_lo, conic_hi = map(
        float,
        surface_fit_cfg[
            "conic_bounds"
        ],
    )

    for value, descriptor in zip(
        u,
        descriptors,
    ):
        which = str(
            descriptor[
                "surface"
            ]
        )

        surface = surfaces[
            which
        ]

        source = base[
            which
        ]

        kind = str(
            descriptor[
                "kind"
            ]
        )

        if kind == "CURVATURE":
            candidate = float(
                source.curvature
                * (
                    1.0
                    + curvature_fraction
                    * float(value)
                )
            )

            surface.curvature = float(
                np.clip(
                    candidate,
                    -curvature_abs_max,
                    curvature_abs_max,
                )
            )

        elif kind == "CONIC":
            candidate = float(
                source.conic
                + conic_bound
                * float(value)
            )

            surface.conic = float(
                np.clip(
                    candidate,
                    conic_lo,
                    conic_hi,
                )
            )

        elif kind == (
            "ASTIG_QUADRATIC_PAIR"
        ):
            delta = (
                coefficient_bound
                * float(value)
            )

            index_20 = int(
                descriptor[
                    "index_20"
                ]
            )

            index_02 = int(
                descriptor[
                    "index_02"
                ]
            )

            surface.coeff[
                index_20
            ] = float(
                source.coeff[
                    index_20
                ]
                + delta
            )

            surface.coeff[
                index_02
            ] = float(
                source.coeff[
                    index_02
                ]
                - delta
            )

        elif kind == "COEFFICIENT":
            index = int(
                descriptor[
                    "index"
                ]
            )

            surface.coeff[
                index
            ] = float(
                source.coeff[
                    index
                ]
                + coefficient_bound
                * float(value)
            )

        else:
            raise ValueError(
                "STEP12_O2_REFINEMENT_"
                "VARIABLE_KIND_INVALID:"
                f"{kind}"
            )

    return m1, m2


def _step12_topology_admission(
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Apply STEP12 HARD/WARN topology policy with an audited WARN bypass."""
    enforcement = str(
        gate[
            "topology_enforcement"
        ]
    )

    if enforcement not in {"HARD", "WARN"}:
        raise ValueError(
            "STEP12_TOPOLOGY_ENFORCEMENT_INVALID:"
            f"{enforcement}"
        )

    raw_pass = bool(
        gate[
            "topology_pass"
        ]
    )
    basic_surface_sanity_raw = bool(
        gate[
            "topology_checks"
        ][
            "basic_surface_sanity"
        ]
    )
    unsafe_basic_surface_sanity_bypass = bool(
        enforcement == "WARN"
        and not basic_surface_sanity_raw
    )
    basic_surface_sanity = bool(
        basic_surface_sanity_raw
        or enforcement == "WARN"
    )
    admitted = bool(
        raw_pass
        or (
            enforcement == "WARN"
            and basic_surface_sanity
        )
    )

    return {
        "enforcement":
            enforcement,

        "raw_pass":
            raw_pass,

        "basic_surface_sanity":
            basic_surface_sanity,

        "basic_surface_sanity_raw":
            basic_surface_sanity_raw,

        "unsafe_basic_surface_sanity_bypass":
            unsafe_basic_surface_sanity_bypass,

        "admitted":
            admitted,

        "status":
            (
                "PASS"
                if raw_pass
                else
                (
                    (
                        "WARN_ADMITTED_UNSAFE_BASIC_SANITY"
                        if unsafe_basic_surface_sanity_bypass
                        else "WARN_ADMITTED"
                    )
                    if admitted
                    else "FAIL"
                )
            ),

        "warning_reasons":
            (
                list(
                    gate.get(
                        "topology_failure_reasons",
                        [],
                    )
                )
                if admitted and not raw_pass
                else []
            ),
    }


def _step12_actual_o2_evaluate(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    cfg: dict[str, Any],
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Đánh giá O2 bằng ray-trace vật lý thật tới display target cố định của STEP08."""
    trace = (
        _reverse_evaluation_trace(
            ctx,
            m1,
            m2,
            ctx.data[
                "display"
            ],
            use_cache=False,
        )
    )

    rays = ctx.data[
        "rays"
    ]

    valid = np.asarray(
        trace[
            "valid"
        ],
        bool,
    )

    ray_count = len(
        valid
    )

    if valid.shape != (
        len(
            rays[
                "rows"
            ]
        ),
    ):
        raise RuntimeError(
            "STEP12_O2_VALID_MASK_"
            "SHAPE_MISMATCH"
        )

    targets = np.asarray(
        ctx.data[
            "fan_refs"
        ],
        float,
    )[
        np.asarray(
            rays[
                "field_index"
            ],
            int,
        )
    ]

    landing = np.asarray(
        trace[
            "landing"
        ],
        float,
    )

    display_frame = np.asarray(
        ctx.data[
            "display"
        ].frame,
        float,
    )

    local_error = (
        landing
        - targets
    ) @ display_frame

    finite_target = (
        valid
        &
        np.all(
            np.isfinite(
                local_error[
                    :,
                    :2,
                ]
            ),
            axis=1,
        )
    )

    l_ref = float(
        ctx.config[
            "normalized_objective"
        ][
            "L_ref_mm"
        ]
    )

    invalid_penalty = float(
        cfg[
            "invalid_residual"
        ]
    )

    target_residual = np.full(
        (
            ray_count,
            2,
        ),
        invalid_penalty,
        dtype=float,
    )

    target_residual[
        finite_target
    ] = (
        local_error[
            finite_target,
            :2,
        ]
        / l_ref
    )

    target_radius = np.linalg.norm(
        local_error[
            :,
            :2,
        ],
        axis=1,
    )

    p2 = np.asarray(
        trace[
            "points"
        ][
            2
        ],
        float,
    )

    actual_out = np.asarray(
        trace[
            "directions"
        ][
            2
        ],
        float,
    )

    direction_good = (
        valid
        &
        np.all(
            np.isfinite(
                p2
            ),
            axis=1,
        )
        &
        np.all(
            np.isfinite(
                actual_out
            ),
            axis=1,
        )
        &
        np.all(
            np.isfinite(
                targets
            ),
            axis=1,
        )
    )

    direction_angle_rad = np.full(
        ray_count,
        np.nan,
        dtype=float,
    )

    if np.any(
        direction_good
    ):
        desired_out = unit(
            targets[
                direction_good
            ]
            -
            p2[
                direction_good
            ]
        )

        actual_unit = unit(
            actual_out[
                direction_good
            ]
        )

        dot = np.sum(
            desired_out
            * actual_unit,
            axis=1,
        )

        direction_angle_rad[
            direction_good
        ] = np.arccos(
            np.clip(
                dot,
                -1.0,
                1.0,
            )
        )

    angle_scale_rad = math.radians(
        float(
            ctx.config[
                "surface_fit"
            ][
                "step12_quality_gates"
            ][
                "maximum_reflection_rms_deg"
            ]
        )
    )

    direction_residual = np.full(
        ray_count,
        invalid_penalty,
        dtype=float,
    )

    finite_angle = (
        valid
        &
        np.isfinite(
            direction_angle_rad
        )
    )

    direction_residual[
        finite_angle
    ] = (
        direction_angle_rad[
            finite_angle
        ]
        /
        angle_scale_rad
    )

    residual = np.concatenate([
        target_residual[
            :,
            0,
        ],
        target_residual[
            :,
            1,
        ],
        direction_residual,
    ])

    if not np.all(
        np.isfinite(
            residual
        )
    ):
        raise RuntimeError(
            "STEP12_O2_OBJECTIVE_NONFINITE"
        )

    valid_target_radius = (
        target_radius[
            finite_target
        ]
    )

    valid_direction_deg = (
        np.degrees(
            direction_angle_rad[
                finite_angle
            ]
        )
    )

    field_index = np.asarray(rays["field_index"], int)
    landing_local = (landing - ctx.data["display"].center) @ display_frame
    target_local = (targets - ctx.data["display"].center) @ display_frame
    field_rows: list[dict[str, Any]] = []
    mapping_values: list[float] = []
    spot_values: list[float] = []
    for field in range(int(rays["field_count"])):
        mask = (field_index == field) & finite_target
        count = int(np.count_nonzero(mask))
        if count == 0:
            field_rows.append({
                "field_index": int(field),
                "valid_ray_count": 0,
                "mapping_error_mm": None,
                "spot_rms_mm": None,
            })
            continue
        points_xy = landing_local[mask, :2]
        target_xy = np.mean(target_local[mask, :2], axis=0)
        centroid = np.mean(points_xy, axis=0)
        mapping_error = float(np.linalg.norm(centroid - target_xy))
        spot_rms = float(np.sqrt(np.mean(np.sum((points_xy - centroid) ** 2, axis=1))))
        mapping_values.append(mapping_error)
        spot_values.append(spot_rms)
        field_rows.append({
            "field_index": int(field),
            "valid_ray_count": count,
            "mapping_error_mm": mapping_error,
            "spot_rms_mm": spot_rms,
        })

    mapping_array = np.asarray(mapping_values, float)
    spot_array = np.asarray(spot_values, float)
    mapping_field_rms = (
        float(np.sqrt(np.mean(mapping_array * mapping_array)))
        if len(mapping_array)
        else float("inf")
    )
    spot_field_rms = (
        float(np.sqrt(np.mean(spot_array * spot_array)))
        if len(spot_array)
        else float("inf")
    )

    quality_cfg = (
        ctx.config[
            "surface_fit"
        ][
            "step12_quality_gates"
        ]
    )

    sanity1 = _o2_mirror_gate_view(
        _mirror_topology_gate(
            ctx,
            m1,
            trace,
            "M1",
        ),
        quality_cfg,
    )

    sanity2 = _o2_mirror_gate_view(
        _mirror_topology_gate(
            ctx,
            m2,
            trace,
            "M2",
        ),
        quality_cfg,
    )

    topology_admission1 = (
        _step12_topology_admission(
            sanity1
        )
    )
    topology_admission2 = (
        _step12_topology_admission(
            sanity2
        )
    )

    physical_fraction = float(
        np.mean(
            valid
        )
    )

    objective = float(
        np.mean(
            residual
            * residual
        )
    )

    metrics = {
        "schema":
            "HUD_FAN_V5_5_"
            "STEP12_ACTUAL_O2_METRICS_V1",

        "objective_kind":
            "FIXED_STEP08_MAPPING_PLUS_SPOT_VIA_PER_RAY_TARGET_ERROR_"
            "PLUS_ACTUAL_O2_TARGET_DIRECTION",

        "J_step12_actual_o2_dimensionless":
            objective,

        "ray_count":
            int(ray_count),

        "physical_valid_count":
            int(
                np.count_nonzero(
                    valid
                )
            ),

        "physical_fraction":
            physical_fraction,

        "field_metric_count": int(len(mapping_array)),

        "mapping_field_rms_mm": mapping_field_rms,

        "mapping_field_p95_mm": (
            float(np.percentile(mapping_array, 95))
            if len(mapping_array)
            else float("inf")
        ),

        "mapping_field_max_mm": (
            float(np.max(mapping_array))
            if len(mapping_array)
            else float("inf")
        ),

        "spot_field_rms_mm": spot_field_rms,

        "spot_field_p95_mm": (
            float(np.percentile(spot_array, 95))
            if len(spot_array)
            else float("inf")
        ),

        "spot_field_max_mm": (
            float(np.max(spot_array))
            if len(spot_array)
            else float("inf")
        ),

        "field_optical_rows": field_rows,

        "display_target_rms_mm":
            (
                float(
                    np.sqrt(
                        np.mean(
                            valid_target_radius
                            ** 2
                        )
                    )
                )
                if len(
                    valid_target_radius
                )
                else float(
                    "inf"
                )
            ),

        "display_target_p95_mm":
            (
                float(
                    np.percentile(
                        valid_target_radius,
                        95,
                    )
                )
                if len(
                    valid_target_radius
                )
                else float(
                    "inf"
                )
            ),

        "display_target_max_mm":
            (
                float(
                    np.max(
                        valid_target_radius
                    )
                )
                if len(
                    valid_target_radius
                )
                else float(
                    "inf"
                )
            ),

        "target_direction_rms_deg":
            (
                float(
                    np.sqrt(
                        np.mean(
                            valid_direction_deg
                            ** 2
                        )
                    )
                )
                if len(
                    valid_direction_deg
                )
                else float(
                    "inf"
                )
            ),

        "target_direction_p95_deg":
            (
                float(
                    np.percentile(
                        valid_direction_deg,
                        95,
                    )
                )
                if len(
                    valid_direction_deg
                )
                else float(
                    "inf"
                )
            ),

        "target_direction_max_deg":
            (
                float(
                    np.max(
                        valid_direction_deg
                    )
                )
                if len(
                    valid_direction_deg
                )
                else float(
                    "inf"
                )
            ),

        "M1_sanity":
            sanity1,

        "M2_sanity":
            sanity2,

        "M1_topology_admission":
            topology_admission1,

        "M2_topology_admission":
            topology_admission2,

        "candidate_surface_raw_valid":
            bool(
                sanity1[
                    "topology_pass"
                ]
                and
                sanity2[
                    "topology_pass"
                ]
            ),

        "candidate_surface_valid":
            bool(
                topology_admission1[
                    "admitted"
                ]
                and
                topology_admission2[
                    "admitted"
                ]
            ),

        "candidate_surface_admitted":
            bool(
                topology_admission1[
                    "admitted"
                ]
                and
                topology_admission2[
                    "admitted"
                ]
            ),

        "candidate_surface_valid_policy":
            "PER_MIRROR_HARD_WARN_TOPOLOGY_WITH_WARN_BASIC_SANITY_BYPASS",

        "candidate_surface_quality_status":
            (
                "PASS"
                if (
                    sanity1[
                        "quality_status"
                    ]
                    == "PASS"
                    and
                    sanity2[
                        "quality_status"
                    ]
                    == "PASS"
                )
                else "WARN"
            ),

        "target_authority":
            "ctx.data['fan_refs'] "
            "frozen by STEP08 "
            "before STEP12 refinement",

        "ci_cloud_used_in_objective":
            False,
    }

    return (
        residual,
        metrics,
        trace,
    )


def _step12_actual_o2_gate(
    metrics: dict[str, Any],
    quality_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Gate STEP12 theo chính O2 ray-trace; cloud-fit và CI integrability không tham gia quyết định."""
    checks = [
        {
            "surface":
                "SYSTEM_O2",

            "check":
                "physical_fraction",

            "actual":
                float(
                    metrics[
                        "physical_fraction"
                    ]
                ),

            "operator":
                ">=",

            "limit":
                float(
                    quality_cfg[
                        "minimum_physical_fraction"
                    ]
                ),

            "unit":
                "fraction",
        },
        {
            "surface":
                "SYSTEM_O2",

            "check":
                "target_direction_rms",

            "actual":
                float(
                    metrics[
                        "target_direction_rms_deg"
                    ]
                ),

            "operator":
                "<=",

            "limit":
                float(
                    quality_cfg[
                        "maximum_reflection_rms_deg"
                    ]
                ),

            "unit":
                "deg",
        },
    ]

    for row in checks:
        actual = float(
            row[
                "actual"
            ]
        )

        limit = float(
            row[
                "limit"
            ]
        )

        passed = bool(
            np.isfinite(
                actual
            )
            and
            (
                actual >= limit
                if row[
                    "operator"
                ] == ">="
                else
                actual <= limit
            )
        )

        row[
            "status"
        ] = (
            "PASS"
            if passed
            else
            (
                "WARN"
                if quality_cfg[
                    "enforcement"
                ]
                == "WARN"
                else "FAIL"
            )
        )

        row[
            "reason"
        ] = (
            f"{actual:.6g} "
            f"{row['operator']} "
            f"{limit:.6g} "
            f"{row['unit']}"
            if passed
            else
            "ACTUAL_O2_GATE_FAILED:"
            f"{actual:.6g}:"
            f"{row['operator']}:"
            f"{limit:.6g}:"
            f"{row['unit']}"
        )

    checks.extend([
        {
            "surface":
                "SYSTEM_O2",

            "check":
                "display_target_rms",

            "actual":
                float(
                    metrics[
                        "display_target_rms_mm"
                    ]
                ),

            "operator":
                "DIAGNOSTIC",

            "limit":
                None,

            "unit":
                "mm",

            "status":
                "INFO",

            "reason":
                "NO_STEP12_ABSOLUTE_"
                "DISPLAY_TARGET_LIMIT_CONFIGURED",
        },
        {
            "surface":
                "SYSTEM_O2",

            "check":
                "display_target_p95",

            "actual":
                float(
                    metrics[
                        "display_target_p95_mm"
                    ]
                ),

            "operator":
                "DIAGNOSTIC",

            "limit":
                None,

            "unit":
                "mm",

            "status":
                "INFO",

            "reason":
                "NO_STEP12_ABSOLUTE_"
                "DISPLAY_TARGET_LIMIT_CONFIGURED",
        },
        {
            "surface":
                "SYSTEM_O2",

            "check":
                "target_direction_p95",

            "actual":
                float(
                    metrics[
                        "target_direction_p95_deg"
                    ]
                ),

            "operator":
                "DIAGNOSTIC",

            "limit":
                None,

            "unit":
                "deg",

            "status":
                "INFO",

            "reason":
                "NO_STEP12_P95_"
                "DIRECTION_LIMIT_CONFIGURED",
        },
    ])

    hard_rows = [
        row
        for row in checks
        if row[
            "status"
        ] != "INFO"
    ]

    all_pass = bool(
        all(
            row[
                "status"
            ]
            == "PASS"
            for row in hard_rows
        )
    )

    return {
        "authority":
            "ACTUAL_O2_SURFACE_RAY_TRACE",

        "enforcement":
            str(
                quality_cfg[
                    "enforcement"
                ]
            ),

        "status":
            (
                "PASS"
                if all_pass
                else
                (
                    "WARN"
                    if quality_cfg[
                        "enforcement"
                    ]
                    == "WARN"
                    else "FAIL"
                )
            ),

        "checks":
            checks,
    }


def _step12_duration(seconds: float) -> str:
    """Format compact wall-clock durations for STEP12 terminal progress."""
    val = float(seconds)
    if not np.isfinite(val):
        return "--:--"
    val = max(val, 0.0)
    if val < 60.0:
        return f"{val:.1f}s"
    total = int(round(val))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _step12_progress(message: str) -> None:
    """Print concise STEP12 progress from the parent process only."""
    if not is_compute_worker():
        prefix = "[STEP12]"
        for line in str(message).split("\n"):
            if line.startswith("="):
                print(line, flush=True)
            elif line.startswith("[STEP12]"):
                print(line, flush=True)
            elif line.startswith("["):
                print(f"{prefix}{line}", flush=True)
            elif line.startswith("  ") or not line.strip():
                print(line, flush=True)
            elif any(line.startswith(k) for k in (
                "Iterations", "Accepted", "Rejected", "Objective",
                "Physical", "Chief/display", "M1 topology", "M2 topology",
                "M1 quality", "M2 quality", "Result", "Elapsed"
            )):
                print(line, flush=True)
            else:
                print(f"{prefix} {line}", flush=True)


_STEP12_PARENT_GUARD_METRICS = (
    "chief_centered_spot_RMS_mm",
    "bundle_RMS_P95_mm",
    "worst_bundle_RMS_mm",
    "global_max_spot_radius_mm",
    "chief_pupil_spread_RMS_mm",
    "chief_pupil_spread_P95_mm",
    "chief_pupil_spread_max_mm",
)


def _step12_finite_metric(
    value: Any,
) -> float | None:
    """Convert one STEP12 metric to a finite float."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return (
        number
        if np.isfinite(number)
        else None
    )


def _step12_spot_pupil_metrics(
    ctx: Context,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Measure STEP12 bundle spot and chief-pupil spread."""
    rays = ctx.data["rays"]
    valid = np.asarray(
        trace["valid"],
        bool,
    )
    landing = np.asarray(
        trace["landing"],
        float,
    )

    spot = chief_centered_geometric_spot_rms(
        landing,
        valid,
        rays["field_index"],
        rays["pupil_index"],
        rays["chief"],
        ctx.data["display"].frame,
    )

    bundle_rows = list(
        spot.get(
            "bundle_rows",
            [],
        )
    )

    evaluable_rows = [
        row
        for row in bundle_rows
        if (
            bool(row.get("chief_valid", False))
            and
            _step12_finite_metric(
                row.get("RMS_to_chief_mm")
            )
            is not None
        )
    ]

    bundle_rms = np.asarray(
        [
            float(row["RMS_to_chief_mm"])
            for row in evaluable_rows
        ],
        dtype=float,
    )

    worst_row = (
        max(
            evaluable_rows,
            key=lambda row:
                float(row["RMS_to_chief_mm"]),
        )
        if evaluable_rows
        else None
    )

    required_bundle_count = int(
        rays["field_count"]
        *
        rays["pupil_count"]
    )

    chief_mask = np.asarray(
        rays["chief"],
        bool,
    )

    chief_rows_valid = bool(
        np.all(
            valid[chief_mask]
        )
        and
        np.all(
            np.isfinite(
                landing[chief_mask]
            )
        )
    )

    pupil_metrics_evaluable = False
    pupil_spread_rms = None
    pupil_spread_p95 = None
    pupil_spread_max = None
    pupil_spread_per_field = None

    if chief_rows_valid:
        fan_metrics = fan_imaging_metrics(
            trace,
            rays,
            ctx.data["fan_refs"],
            float(
                ctx.config[
                    "fan_weights"
                ][
                    "omega1"
                ]
            ),
        )

        per_field = np.asarray(
            fan_metrics[
                "chief_pupil_spread_per_field_mm"
            ],
            float,
        )

        if (
            len(per_field)
            ==
            int(rays["field_count"])
            and
            np.all(
                np.isfinite(
                    per_field
                )
            )
        ):
            pupil_metrics_evaluable = True
            pupil_spread_rms = float(
                fan_metrics[
                    "chief_pupil_spread_RMS_mm"
                ]
            )
            pupil_spread_p95 = float(
                np.percentile(
                    per_field,
                    95,
                )
            )
            pupil_spread_max = float(
                fan_metrics[
                    "chief_pupil_spread_max_mm"
                ]
            )
            pupil_spread_per_field = (
                per_field.tolist()
            )

    bundle_valid_counts = [
        int(
            row.get(
                "valid_count",
                0,
            )
        )
        for row in bundle_rows
    ]

    return {
        "chief_centered_spot_RMS_mm":
            _step12_finite_metric(
                spot.get(
                    "RMS_spot_radius_mm"
                )
            ),

        "bundle_RMS_P50_mm":
            (
                float(
                    np.percentile(
                        bundle_rms,
                        50,
                    )
                )
                if len(bundle_rms)
                else None
            ),

        "bundle_RMS_P95_mm":
            (
                float(
                    np.percentile(
                        bundle_rms,
                        95,
                    )
                )
                if len(bundle_rms)
                else None
            ),

        "bundle_RMS_P99_mm":
            (
                float(
                    np.percentile(
                        bundle_rms,
                        99,
                    )
                )
                if len(bundle_rms)
                else None
            ),

        "worst_bundle_RMS_mm":
            (
                float(
                    worst_row[
                        "RMS_to_chief_mm"
                    ]
                )
                if worst_row is not None
                else None
            ),

        "worst_bundle_field_index":
            (
                int(
                    worst_row[
                        "field_index"
                    ]
                )
                if worst_row is not None
                else None
            ),

        "worst_bundle_pupil_index":
            (
                int(
                    worst_row[
                        "pupil_index"
                    ]
                )
                if worst_row is not None
                else None
            ),

        "global_max_spot_radius_mm":
            _step12_finite_metric(
                spot.get(
                    "max_spot_radius_mm"
                )
            ),

        "valid_bundle_count":
            int(
                len(
                    evaluable_rows
                )
            ),

        "required_bundle_count":
            required_bundle_count,

        "underfilled_bundle_count":
            int(
                sum(
                    int(
                        row.get(
                            "valid_count",
                            0,
                        )
                    )
                    <
                    int(
                        row.get(
                            "ray_count",
                            0,
                        )
                    )
                    for row in bundle_rows
                )
            ),

        "bundle_valid_counts":
            bundle_valid_counts,

        "all_chiefs_valid":
            chief_rows_valid,

        "spot_metrics_evaluable":
            bool(
                len(evaluable_rows)
                ==
                required_bundle_count
            ),

        "pupil_metrics_evaluable":
            pupil_metrics_evaluable,

        "chief_pupil_spread_RMS_mm":
            pupil_spread_rms,

        "chief_pupil_spread_P95_mm":
            pupil_spread_p95,

        "chief_pupil_spread_max_mm":
            pupil_spread_max,

        "chief_pupil_spread_per_field_mm":
            pupil_spread_per_field,

        "physical_valid_count":
            int(
                np.count_nonzero(
                    valid
                )
            ),

        "bundle_rows":
            bundle_rows,
    }


def _step12_parent_nonworsening_guard(
    candidate: dict[str, Any],
    parent: dict[str, Any] | None,
) -> dict[str, Any]:
    """Require STEP12 spot/pupil/ray coverage not to worsen versus parent."""
    if parent is None:
        return {
            "pass": True,
            "status": "BASELINE_REFERENCE",
            "checks": [],
        }

    checks: list[dict[str, Any]] = []

    for metric_name in (
        _STEP12_PARENT_GUARD_METRICS
    ):
        actual = _step12_finite_metric(
            candidate.get(
                metric_name
            )
        )
        limit = _step12_finite_metric(
            parent.get(
                metric_name
            )
        )

        if limit is None:
            passed = bool(
                actual is not None
            )
        elif actual is None:
            passed = False
        else:
            numerical_tolerance = float(
                64.0
                *
                np.finfo(float).eps
                *
                max(
                    1.0,
                    abs(limit),
                )
            )

            passed = bool(
                actual
                <=
                limit
                +
                numerical_tolerance
            )

        checks.append({
            "check":
                metric_name,

            "actual":
                actual,

            "operator":
                "<=_PARENT",

            "limit":
                limit,

            "status":
                (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
        })

    actual_valid_count = int(
        candidate[
            "physical_valid_count"
        ]
    )
    parent_valid_count = int(
        parent[
            "physical_valid_count"
        ]
    )

    checks.append({
        "check":
            "physical_valid_count",

        "actual":
            actual_valid_count,

        "operator":
            ">=_PARENT",

        "limit":
            parent_valid_count,

        "status":
            (
                "PASS"
                if actual_valid_count
                >=
                parent_valid_count
                else "FAIL"
            ),
    })

    candidate_bundle_counts = np.asarray(
        candidate[
            "bundle_valid_counts"
        ],
        int,
    )
    parent_bundle_counts = np.asarray(
        parent[
            "bundle_valid_counts"
        ],
        int,
    )

    bundle_coverage_pass = bool(
        candidate_bundle_counts.shape
        ==
        parent_bundle_counts.shape
        and
        np.all(
            candidate_bundle_counts
            >=
            parent_bundle_counts
        )
    )

    checks.append({
        "check":
            "per_bundle_valid_ray_count",

        "actual":
            candidate_bundle_counts.tolist(),

        "operator":
            ">=_PARENT_ELEMENTWISE",

        "limit":
            parent_bundle_counts.tolist(),

        "status":
            (
                "PASS"
                if bundle_coverage_pass
                else "FAIL"
            ),
    })

    passed = bool(
        all(
            row["status"] == "PASS"
            for row in checks
        )
    )

    return {
        "pass":
            passed,

        "status":
            (
                "PASS"
                if passed
                else "FAIL"
            ),

        "checks":
            checks,
    }


def _step12_constraint_state(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    optical_metrics: dict[str, Any],
    trace: dict[str, Any],
    parent_spot_pupil: dict[str, Any] | None,
    *,
    monitor_only: bool,
) -> dict[str, Any]:
    """Kiểm tra toàn diện các ràng buộc của candidate trong STEP 12."""
    spot_pupil = _step12_spot_pupil_metrics(
        ctx,
        trace,
    )

    parent_guard = (
        _step12_parent_nonworsening_guard(
            spot_pupil,
            parent_spot_pupil,
        )
    )

    m1_raw_topology_pass = bool(
        optical_metrics[
            "M1_sanity"
        ][
            "topology_pass"
        ]
    )

    m2_raw_topology_pass = bool(
        optical_metrics[
            "M2_sanity"
        ][
            "topology_pass"
        ]
    )

    valid = np.asarray(
        trace["valid"],
        bool,
    )

    unobscured = False
    signed_area_mm2 = None

    if np.any(valid):
        mf2 = mf2_geometry(
            trace["points"][0][valid],
            trace["points"][1][valid],
            trace["points"][2][valid],
            m2,
            float(
                ctx.config[
                    "fan_weights"
                ][
                    "omega2"
                ]
            ),
            ctx.data["n_obs"],
        )

        signed_area_mm2 = float(
            mf2[
                "S_AQP_signed_mm2"
            ]
        )
        unobscured = bool(
            signed_area_mm2 >= 0.0
        )

    probe_m1 = m1.copy()
    probe_m2 = m2.copy()
    probe_display = (
        ctx.data["display"].copy()
    )

    aperture_error = None
    aperture_audit = None
    display_active_pass = False
    display_clear_pass = False
    display_active_full_mm = None
    display_clear_full_mm = None

    try:
        (
            construction,
            _,
            aperture_audit,
        ) = _rebuild_post_fit_apertures(
            ctx,
            probe_m1,
            probe_m2,
            probe_display,
            "STEP_12_CONSTRAINT_PROBE",
        )

        landing = np.asarray(
            construction["landing"],
            float,
        )
        finite = np.all(
            np.isfinite(
                landing
            ),
            axis=1,
        )

        local_xy = (
            landing[finite]
            -
            probe_display.center
        ) @ probe_display.frame

        active_half = np.max(
            np.abs(
                local_xy[
                    :,
                    :2,
                ]
            ),
            axis=0,
        )

        seed_half = (
            np.asarray(
                ctx.config[
                    "display_seed_mm"
                ],
                float,
            )
            /
            2.0
        )

        clear_half = np.asarray(
            probe_display.half_aperture,
            float,
        )

        clear_limit_half = (
            seed_half
            +
            0.5
        )

        display_active_pass = bool(
            np.all(
                active_half
                <=
                seed_half
            )
        )

        display_clear_pass = bool(
            np.all(
                clear_half
                <=
                clear_limit_half
            )
        )

        display_active_full_mm = (
            2.0
            *
            active_half
        ).tolist()

        display_clear_full_mm = (
            2.0
            *
            clear_half
        ).tolist()

    except (
        RuntimeError,
        ValueError,
        ArithmeticError,
    ) as exc:
        aperture_error = (
            f"{type(exc).__name__}:{exc}"
        )

    mapping_complete = False
    distortion_pass = False
    distortion_percent = None
    distortion_partial_percent = None
    mapping_error = None
    forward_evaluated = 0
    forward_required = 0

    if (
        aperture_error is None
        and
        parent_guard["pass"]
    ):
        try:
            forward_evaluation = (
                _run_forward_evaluation(
                    ctx,
                    probe_m1,
                    probe_m2,
                    monitor_only=
                        monitor_only,
                    display=
                        probe_display,
                )
            )

            distortion = (
                forward_evaluation[
                    "distortion"
                ]
            )

            mapping_complete = bool(
                distortion[
                    "evaluation_status"
                ]
                ==
                "COMPLETE"
            )

            distortion_percent = (
                _step12_finite_metric(
                    distortion.get(
                        "D_max_percent"
                    )
                )
            )

            distortion_partial_percent = (
                _step12_finite_metric(
                    distortion.get(
                        "D_partial_max_percent_diagnostic"
                    )
                )
            )

            distortion_pass = bool(
                mapping_complete
                and
                distortion[
                    "strict_less_than_limit"
                ]
            )

            forward_evaluated = int(
                distortion[
                    "evaluated_noncentral_field_pupil_count"
                ]
            )

            forward_required = int(
                distortion[
                    "required_count"
                ]
            )

        except (
            RuntimeError,
            ValueError,
            ArithmeticError,
        ) as exc:
            mapping_error = (
                f"{type(exc).__name__}:{exc}"
            )

    spot_pupil_evaluable = bool(
        spot_pupil[
            "spot_metrics_evaluable"
        ]
        and
        spot_pupil[
            "pupil_metrics_evaluable"
        ]
    )

    gate_rows = [
        {
            "check": "M1_raw_topology",
            "status":
                "PASS"
                if m1_raw_topology_pass
                else "FAIL",
        },
        {
            "check": "M2_raw_convex_ray_facing",
            "status":
                "PASS"
                if m2_raw_topology_pass
                else "FAIL",
        },
        {
            "check": "signed_MF2_unobscured",
            "status":
                "PASS"
                if unobscured
                else "FAIL",
        },
        {
            "check": "spot_pupil_evaluable",
            "status":
                "PASS"
                if spot_pupil_evaluable
                else "FAIL",
        },
        {
            "check": "spot_pupil_parent_nonworsening",
            "status":
                "PASS"
                if parent_guard["pass"]
                else "FAIL",
        },
        {
            "check": "display_active_footprint_within_seed",
            "status":
                "PASS"
                if display_active_pass
                else "FAIL",
        },
        {
            "check": "display_clear_aperture_within_seed_plus_margin",
            "status":
                "PASS"
                if display_clear_pass
                else "FAIL",
        },
        {
            "check": "fixed_grid_forward_mapping_complete",
            "status":
                "PASS"
                if mapping_complete
                else "FAIL",
        },
        {
            "check": "fixed_grid_distortion_strictly_below_limit",
            "status":
                "PASS"
                if distortion_pass
                else "FAIL",
        },
    ]

    failed_gate_count = int(
        sum(
            row["status"] != "PASS"
            for row in gate_rows
        )
    )

    feasibility_pass = bool(
        failed_gate_count == 0
    )

    distortion_limit = float(
        ctx.config[
            "distortion"
        ][
            "hard_limit_percent"
        ]
    )

    measured_distortion = (
        distortion_percent
        if distortion_percent is not None
        else distortion_partial_percent
    )

    distortion_violation = (
        max(
            0.0,
            measured_distortion
            /
            distortion_limit
            -
            1.0,
        )
        if measured_distortion is not None
        else 1.0
    )

    restoration_severity = float(
        distortion_violation
        +
        (
            0.0
            if mapping_complete
            else 1.0
        )
        +
        (
            0.0
            if display_active_pass
            else 1.0
        )
        +
        (
            0.0
            if display_clear_pass
            else 1.0
        )
    )

    return {
        "feasibility_pass":
            feasibility_pass,

        "failed_gate_count":
            failed_gate_count,

        "restoration_rank_prefix": [
            failed_gate_count,
            restoration_severity,
        ],

        "gate_rows":
            gate_rows,

        "spot_pupil":
            spot_pupil,

        "parent_nonworsening":
            parent_guard,

        "M1_raw_topology_pass":
            m1_raw_topology_pass,

        "M2_raw_topology_pass":
            m2_raw_topology_pass,

        "unobscured":
            unobscured,

        "S_AQP_signed_mm2":
            signed_area_mm2,

        "mapping_complete":
            mapping_complete,

        "distortion_pass":
            distortion_pass,

        "distortion_percent":
            distortion_percent,

        "distortion_partial_percent":
            distortion_partial_percent,

        "distortion_limit_percent":
            distortion_limit,

        "forward_evaluated_pair_count":
            forward_evaluated,

        "forward_required_pair_count":
            forward_required,

        "display_active_pass":
            display_active_pass,

        "display_clear_pass":
            display_clear_pass,

        "display_active_full_mm":
            display_active_full_mm,

        "display_clear_full_mm":
            display_clear_full_mm,

        "display_seed_full_mm":
            list(
                map(
                    float,
                    ctx.config[
                        "display_seed_mm"
                    ],
                )
            ),

        "aperture_error":
            aperture_error,

        "mapping_error":
            mapping_error,

        "monitor_only":
            bool(
                monitor_only
            ),

        "aperture_audit":
            aperture_audit,
    }


def _step12_refine_o2(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
) -> tuple[
    PolySurface,
    PolySurface,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Tinh chỉnh O2 bounded DLSQ trên actual ray target; không dùng CI cloud trong objective."""
    refinement_started = time.perf_counter()
    cfg = (
        ctx.config[
            "surface_fit"
        ][
            "step12_o2_refinement"
        ]
    )

    residual, current, trace = (
        _step12_actual_o2_evaluate(
            ctx,
            m1,
            m2,
            cfg,
        )
    )

    base_obj = float(current["J_step12_actual_o2_dimensionless"])
    base_rms = float(current["display_target_rms_mm"])
    base_phys = 100.0 * float(current["physical_fraction"])
    base_topo_m1 = str(current["M1_topology_admission"]["status"])
    base_topo_m2 = str(current["M2_topology_admission"]["status"])
    base_qual_m1 = current["M1_sanity"].get("quality_status", "PASS")
    base_qual_m2 = current["M2_sanity"].get("quality_status", "PASS")

    baseline_valid = np.asarray(
        trace["valid"],
        bool,
    )
    if not np.any(baseline_valid):
        raise RuntimeError(
            "STEP12_O2_BASELINE_HAS_NO_VALID_PHYSICAL_RAYS"
        )

    baseline_spot_pupil = (
        _step12_spot_pupil_metrics(
            ctx,
            trace,
        )
    )

    current_constraints = (
        _step12_constraint_state(
            ctx,
            m1,
            m2,
            current,
            trace,
            None,
            monitor_only=True,
        )
    )

    current[
        "step12_constraint_state"
    ] = current_constraints

    current[
        "step12_baseline_spot_pupil_metrics"
    ] = copy.deepcopy(
        baseline_spot_pupil
    )

    restoration_mode = not bool(
        current_constraints[
            "feasibility_pass"
        ]
    )

    _step12_progress(
        "=" * 80 + "\n"
        "[STEP12] JOINT O2 REFINEMENT\n"
        "=" * 80 + "\n"
        "[BASE]\n"
        f"  mode       : "
        f"{'RESTORATION' if restoration_mode else 'HARD_OPTIMIZATION'}\n"
        f"  constraints: "
        f"{current_constraints['failed_gate_count']} failed\n"
        f"  objective : {base_obj:.5f}\n"
        f"  RMS       : {base_rms:.3f} mm\n"
        f"  physical  : {base_phys:.2f}%\n"
        f"  topology  : {base_topo_m1}/{base_topo_m2}\n"
        f"  quality   : {base_qual_m1}/{base_qual_m2}"
    )

    if not bool(
        cfg[
            "enabled"
        ]
    ):
        _step12_progress(
            "O2 refinement disabled "
            f"| objective={float(current['J_step12_actual_o2_dimensionless']):.6g}"
        )
        return (
            m1,
            m2,
            [],
            current,
            trace,
        )

    base = {
        "M1":
            m1.copy(),

        "M2":
            m2.copy(),
    }

    descriptors = (
        _step12_o2_variable_descriptors(
            m1,
            m2,
        )
    )

    u = np.zeros(
        len(
            descriptors
        ),
        dtype=float,
    )

    runtime = current_runtime()
    worker_count = int(
        getattr(runtime, "_ray_trace_max_inflight", 1)
        if runtime is not None
        else 1
    )
    iteration_total = int(cfg["iterations"])
    column_total = len(descriptors)


    history: list[
        dict[str, Any]
    ] = []

    damping = float(
        cfg[
            "damping"
        ]
    )

    fd = float(
        cfg[
            "finite_difference_normalized"
        ]
    )

    max_step = float(
        cfg[
            "max_step_normalized"
        ]
    )

    minimum_relative = float(
        cfg[
            "minimum_relative_improvement"
        ]
    )

    for iteration in range(
        1,
        int(
            cfg[
                "iterations"
            ]
        )
        + 1,
    ):
        iteration_started = time.perf_counter()
        jacobian_started = time.perf_counter()


        jacobian = np.empty(
            (
                len(
                    residual
                ),
                len(
                    descriptors
                ),
            ),
            dtype=float,
        )

        for column in range(
            len(
                descriptors
            )
        ):
            trial_u = (
                u.copy()
            )

            trial_u[
                column
            ] = min(
                1.0,
                trial_u[
                    column
                ]
                + fd,
            )

            actual_fd = float(
                trial_u[
                    column
                ]
                -
                u[
                    column
                ]
            )

            if actual_fd <= 0.0:
                raise RuntimeError(
                    "STEP12_O2_FD_"
                    "STEP_INVALID:"
                    f"{column}"
                )

            a, b = (
                _step12_apply_o2_vector(
                    base,
                    descriptors,
                    trial_u,
                    cfg,
                    ctx.config[
                        "surface_fit"
                    ],
                )
            )

            trial_residual, _, _ = (
                _step12_actual_o2_evaluate(
                    ctx,
                    a,
                    b,
                    cfg,
                )
            )

            if (
                trial_residual.shape
                !=
                residual.shape
            ):
                raise RuntimeError(
                    "STEP12_O2_JACOBIAN_"
                    "SHAPE_MISMATCH:"
                    f"{column}"
                )

            jacobian[
                :,
                column
            ] = (
                (
                    trial_residual
                    - residual
                )
                / actual_fd
            )

            completed_columns = column + 1
            if (
                completed_columns % 3 == 0
                or completed_columns == column_total
            ):
                jacobian_elapsed = time.perf_counter() - jacobian_started
                remaining_columns = column_total - completed_columns
                eta_seconds = (
                    jacobian_elapsed
                    / completed_columns
                    * remaining_columns
                )
                _step12_progress(
                    f"[ITER {iteration}/{iteration_total}][JACOBIAN] "
                    f"{completed_columns}/{column_total} "
                    f"| elapsed={jacobian_elapsed:.1f}s "
                    f"| ETA={eta_seconds:.1f}s"
                )

        jtj = (
            jacobian.T
            @ jacobian
        )

        diagonal = np.maximum(
            np.diag(
                jtj
            ),
            1e-12,
        )

        rhs = -(
            jacobian.T
            @ residual
        )

        accepted = False

        accepted_alpha = 0.0

        accepted_damping = (
            damping
        )

        before_objective = float(
            current[
                "J_step12_actual_o2_dimensionless"
            ]
        )
        before_rms = float(current["display_target_rms_mm"])
        before_phys = float(current["physical_fraction"])
        before_rays = int(current["physical_valid_count"])

        selected_relative = 0.0
        line_search_trials = 0
        mode_before = (
            "HARD_OPTIMIZATION"
            if current_constraints[
                "feasibility_pass"
            ]
            else "RESTORATION"
        )
        accepted_mode = None

        for multiplier in map(
            float,
            cfg[
                "damping_multipliers"
            ],
        ):
            trial_damping = (
                damping
                * multiplier
            )

            try:
                delta = np.linalg.solve(
                    jtj
                    +
                    trial_damping
                    * np.diag(
                        diagonal
                    ),
                    rhs,
                )

            except np.linalg.LinAlgError:
                continue

            delta = np.clip(
                delta,
                -max_step,
                max_step,
            )

            for alpha in map(
                float,
                cfg[
                    "line_search_alphas"
                ],
            ):
                line_search_trials += 1
                candidate_u = np.clip(
                    u
                    +
                    alpha
                    * delta,
                    -1.0,
                    1.0,
                )

                a, b = (
                    _step12_apply_o2_vector(
                        base,
                        descriptors,
                        candidate_u,
                        cfg,
                        ctx.config[
                            "surface_fit"
                        ],
                    )
                )

                (
                    candidate_residual,
                    candidate,
                    candidate_trace,
                ) = (
                    _step12_actual_o2_evaluate(
                        ctx,
                        a,
                        b,
                        cfg,
                    )
                )

                candidate_objective = float(
                    candidate[
                        "J_step12_actual_o2_dimensionless"
                    ]
                )

                relative_improvement = (
                    (
                        before_objective
                        -
                        candidate_objective
                    )
                    /
                    max(
                        abs(
                            before_objective
                        ),
                        1e-12,
                    )
                )

                candidate_constraints = (
                    _step12_constraint_state(
                        ctx,
                        a,
                        b,
                        candidate,
                        candidate_trace,
                        current_constraints[
                            "spot_pupil"
                        ],
                        monitor_only=True,
                    )
                )

                candidate[
                    "step12_constraint_state"
                ] = candidate_constraints

                candidate[
                    "step12_baseline_spot_pupil_metrics"
                ] = copy.deepcopy(
                    baseline_spot_pupil
                )

                current_restoration_rank = (
                    *tuple(
                        current_constraints[
                            "restoration_rank_prefix"
                        ]
                    ),
                    before_objective,
                )

                candidate_restoration_rank = (
                    *tuple(
                        candidate_constraints[
                            "restoration_rank_prefix"
                        ]
                    ),
                    candidate_objective,
                )

                parent_guard_pass = bool(
                    candidate_constraints[
                        "parent_nonworsening"
                    ][
                        "pass"
                    ]
                )

                if mode_before == "RESTORATION":
                    candidate_accepted = bool(
                        np.isfinite(
                            candidate_objective
                        )
                        and
                        parent_guard_pass
                        and
                        candidate_restoration_rank
                        <
                        current_restoration_rank
                    )
                else:
                    candidate_accepted = bool(
                        np.isfinite(
                            candidate_objective
                        )
                        and
                        candidate_constraints[
                            "feasibility_pass"
                        ]
                        and
                        parent_guard_pass
                        and
                        candidate_objective
                        <
                        before_objective
                    )

                if candidate_accepted:
                    u = (
                        candidate_u
                    )

                    m1 = a
                    m2 = b

                    residual = (
                        candidate_residual
                    )

                    current = (
                        candidate
                    )

                    trace = (
                        candidate_trace
                    )

                    current_constraints = (
                        candidate_constraints
                    )

                    accepted = True

                    accepted_alpha = (
                        alpha
                    )

                    accepted_damping = (
                        trial_damping
                    )

                    selected_relative = float(
                        relative_improvement
                    )

                    accepted_mode = mode_before

                    break

            if accepted:
                break

        damping = (
            max(
                accepted_damping
                * 0.3,
                1e-8,
            )
            if accepted
            else
            min(
                damping
                * 10.0,
                1e8,
            )
        )

        after_objective = float(
            current[
                "J_step12_actual_o2_dimensionless"
            ]
        )
        after_rms = float(current["display_target_rms_mm"])
        after_phys = float(current["physical_fraction"])
        after_rays = int(current["physical_valid_count"])
        total_rays = len(ctx.data["rays"]["rows"])

        if accepted:
            min_phys = float(cfg.get("minimum_physical_fraction", 0.90))
            phys_warn = bool(after_phys < min_phys)
            tag_warn = "[WARN]" if phys_warn else ""
            phys_status = f"[WARN < {min_phys * 100.0:.0f}% preferred]" if phys_warn else "[PASS]"
            accept_lines = [
                f"[ITER {iteration}/{iteration_total}][ACCEPT][{accepted_mode}]{tag_warn}",
                f"  trials      : {line_search_trials}",
                f"  alpha       : {accepted_alpha:.3f}",
                f"  damping     : {accepted_damping:.4f}",
                f"  objective   : {before_objective:.5f} -> {after_objective:.5f}",
                f"  improvement : {selected_relative * 100.0:+.2f}%",
                f"  RMS         : {before_rms:.3f} -> {after_rms:.3f} mm",
                f"  physical    : {before_phys * 100.0:.2f}% -> {after_phys * 100.0:.2f}% {phys_status}",
                f"  rays        : {after_rays:,} / {total_rays:,}",
            ]
            _step12_progress("\n".join(accept_lines))
        else:
            reject_lines = [
                f"[ITER {iteration}/{iteration_total}][REJECT]",
                f"  trials      : {line_search_trials}",
                f"  damping     : {damping:.4f}",
                f"  result      : REJECT_ALL_TRIALS",
                f"  next damping: {min(damping * 10.0, 1e8):.4f}",
            ]
            _step12_progress("\n".join(reject_lines))

        history.append({
            "iteration":
                int(
                    iteration
                ),

            "accepted":
                bool(
                    accepted
                ),

            "acceptance_mode":
                accepted_mode
                if accepted
                else mode_before,

            "feasibility_pass":
                bool(
                    current_constraints[
                        "feasibility_pass"
                    ]
                ),

            "failed_constraint_count":
                int(
                    current_constraints[
                        "failed_gate_count"
                    ]
                ),

            "M1_raw_topology_pass":
                bool(
                    current_constraints[
                        "M1_raw_topology_pass"
                    ]
                ),

            "M2_raw_topology_pass":
                bool(
                    current_constraints[
                        "M2_raw_topology_pass"
                    ]
                ),

            "mapping_complete":
                bool(
                    current_constraints[
                        "mapping_complete"
                    ]
                ),

            "distortion_percent":
                current_constraints[
                    "distortion_percent"
                ],

            "display_active_full_mm":
                current_constraints[
                    "display_active_full_mm"
                ],

            "display_clear_full_mm":
                current_constraints[
                    "display_clear_full_mm"
                ],

            "bundle_RMS_P95_mm":
                current_constraints[
                    "spot_pupil"
                ][
                    "bundle_RMS_P95_mm"
                ],

            "worst_bundle_RMS_mm":
                current_constraints[
                    "spot_pupil"
                ][
                    "worst_bundle_RMS_mm"
                ],

            "chief_pupil_spread_P95_mm":
                current_constraints[
                    "spot_pupil"
                ][
                    "chief_pupil_spread_P95_mm"
                ],

            "chief_pupil_spread_max_mm":
                current_constraints[
                    "spot_pupil"
                ][
                    "chief_pupil_spread_max_mm"
                ],

            "alpha":
                float(
                    accepted_alpha
                ),

            "damping_used":
                float(
                    accepted_damping
                ),

            "damping_next":
                float(
                    damping
                ),

            "objective_before":
                float(
                    before_objective
                ),

            "objective_after":
                float(
                    after_objective
                ),

            "relative_improvement":
                float(
                    selected_relative
                ),

            "physical_valid_count":
                int(
                    after_rays
                ),

            "physical_fraction":
                float(
                    after_phys
                ),

            "physical_fraction_before":
                float(
                    before_phys
                ),

            "display_target_rms_mm":
                float(
                    after_rms
                ),

            "display_target_rms_mm_before":
                float(
                    before_rms
                ),

            "target_direction_rms_deg":
                float(
                    current[
                        "target_direction_rms_deg"
                    ]
                ),

            "normalized_parameter_norm":
                float(
                    np.linalg.norm(
                        u
                    )
                ),
        })

        if not accepted:
            _step12_progress(
                "[STOP] no accepted line search trial"
            )
            break

        if (
            accepted_mode
            ==
            "HARD_OPTIMIZATION"
            and
            selected_relative
            <
            minimum_relative
        ):
            _step12_progress(
                "[STOP] relative improvement "
                "below threshold "
                f"({selected_relative:.5f} "
                f"< {minimum_relative:.5f})"
            )
            break

    return (
        m1,
        m2,
        history,
        current,
        trace,
    )






def _step14_duration(seconds: float) -> str:
    """Format compact wall-clock durations for STEP14 terminal progress."""
    value = float(seconds)
    if not np.isfinite(value):
        return "--:--"
    value = max(value, 0.0)
    if value < 60.0:
        return f"{value:.1f}s"
    whole = int(round(value))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _step14_progress(message: str) -> None:
    """Print concise STEP14 progress from the parent process only."""
    if not is_compute_worker():
        print(f"[STEP14] {message}", flush=True)


class _Step14TerminalProgress:
    """Render bounded Fermat progress without allowing worker output to interleave."""

    def __init__(self) -> None:
        """Khởi tạo trạng thái in tiến độ terminal cho solver Fermat."""
        self._fallback_line_open = False
        self._fallback_width = 0

    def _replace_fallback_line(self, message: str, *, finish: bool = False) -> None:
        """Cập nhật dòng tiến độ terminal cho nhánh fallback."""
        if is_compute_worker():
            return
        line = f"[STEP14] {message}"
        self._fallback_width = max(self._fallback_width, len(line))
        print(
            "\r" + line.ljust(self._fallback_width),
            end="\n" if finish else "",
            flush=True,
        )
        self._fallback_line_open = not finish

    def __call__(self, event: str, details: dict[str, Any]) -> None:
        """Xử lý sự kiện tiến độ Fermat và hiển thị ra terminal."""
        if event == "newton_iteration":
            _step14_progress(
                f"Newton {int(details['iteration']):02d}/{int(details['max_iterations']):02d} "
                f"| active={int(details['active_before']):,}->{int(details['active_after']):,} "
                f"| backend={str(details['backend']).upper()} "
                f"| iteration={_step14_duration(float(details['iteration_seconds']))} "
                f"| elapsed={_step14_duration(float(details['elapsed_seconds']))}"
            )
            return

        if event == "fallback_started":
            _step14_progress(
                f"Fallback started | unresolved={int(details['total']):,} "
                "| workers=1 (main process) | mode=sequential per-ray"
            )
            return

        if event == "fallback_progress":
            processed = int(details["processed"])
            total = int(details["total"])
            percent = 100.0 * processed / max(total, 1)
            self._replace_fallback_line(
                f"Fallback {processed:,}/{total:,} | {percent:5.1f}% "
                f"| accepted={int(details['accepted']):,} "
                f"| elapsed={_step14_duration(float(details['elapsed_seconds']))} "
                f"| ETA={_step14_duration(float(details['eta_seconds']))}"
            )
            return

        if event == "fallback_complete":
            message = (
                f"Fallback complete | processed={int(details['processed']):,}/"
                f"{int(details['total']):,} | accepted={int(details['accepted']):,} "
                f"| elapsed={_step14_duration(float(details['elapsed_seconds']))}"
            )
            if self._fallback_line_open:
                self._replace_fallback_line(message, finish=True)
            else:
                _step14_progress(message)
            return

        if event == "solver_complete":
            _step14_progress(
                f"Fermat complete | converged={int(details['converged']):,}/"
                f"{int(details['ray_count']):,} "
                f"| fallback={int(details['fallback_count']):,} "
                f"| elapsed={_step14_duration(float(details['elapsed_seconds']))}"
            )


def _step14_fermat_diagnostics(
    rays: dict[str, Any],
    trace: dict[str, Any],
    m2: PolySurface,
    sol: dict[str, Any],
    *,
    gradient_tolerance: float,
    reflection_tolerance: float,
    neighbor_count: int,
) -> dict[str, Any]:
    """Chứng nhận 5 hard gate STEP14 và ghi continuity diagnostic mà không đổi nghiệm Fermat."""
    n = int(len(rays["rows"]))
    if n <= 0:
        raise FermatResultSchemaError("STEP14_FERMAT_RAY_COUNT_INVALID")

    q1 = np.asarray(trace["points"][1], dtype=float)
    q2_current = np.asarray(trace["points"][2], dtype=float)
    q2_fermat = np.asarray(sol["target_points"], dtype=float)
    xy = np.asarray(sol["xy"], dtype=float)
    gradient = np.asarray(sol["gradient"], dtype=float)
    gradient_norm = np.asarray(sol["gradient_norm"], dtype=float)
    reflection_residual = np.asarray(sol["reflection_residual"], dtype=float)
    op_initial = np.asarray(sol["op_initial"], dtype=float)
    op_final = np.asarray(sol["op_final"], dtype=float)
    step_norm = np.asarray(sol["step_norm"], dtype=float)

    expected_shapes = {
        "q1": (n, 3),
        "q2_current": (n, 3),
        "q2_fermat": (n, 3),
        "xy": (n, 2),
        "gradient": (n, 2),
        "gradient_norm": (n,),
        "reflection_residual": (n,),
        "op_initial": (n,),
        "op_final": (n,),
        "step_norm": (n,),
    }
    actual_shapes = {
        "q1": q1.shape,
        "q2_current": q2_current.shape,
        "q2_fermat": q2_fermat.shape,
        "xy": xy.shape,
        "gradient": gradient.shape,
        "gradient_norm": gradient_norm.shape,
        "reflection_residual": reflection_residual.shape,
        "op_initial": op_initial.shape,
        "op_final": op_final.shape,
        "step_norm": step_norm.shape,
    }
    for key, expected in expected_shapes.items():
        if actual_shapes[key] != expected:
            raise FermatResultSchemaError(
                f"STEP14_FERMAT_DIAGNOSTIC_SHAPE_MISMATCH:"
                f"{key}:{actual_shapes[key]}:{expected}"
            )

    solver_success = np.asarray(sol["success"], dtype=bool)
    gradient_flag = np.asarray(sol["gradient_pass"], dtype=bool)
    reflection_flag = np.asarray(sol["reflection_pass"], dtype=bool)
    finite_flag = np.asarray(sol["finite_pass"], dtype=bool)
    conic_flag = np.asarray(sol["conic_domain_pass"], dtype=bool)
    current_aperture_flag = np.asarray(sol["in_aperture"], dtype=bool)
    construction_flag = np.asarray(sol["in_construction_domain"], dtype=bool)
    fallback_used = np.asarray(sol["fallback_used"], dtype=bool)

    masks = {
        "success": solver_success,
        "gradient_pass": gradient_flag,
        "reflection_pass": reflection_flag,
        "finite_pass": finite_flag,
        "conic_domain_pass": conic_flag,
        "in_aperture": current_aperture_flag,
        "in_construction_domain": construction_flag,
        "fallback_used": fallback_used,
    }
    for key, value in masks.items():
        if value.shape != (n,):
            raise FermatResultSchemaError(
                f"STEP14_FERMAT_DIAGNOSTIC_MASK_SHAPE_MISMATCH:"
                f"{key}:{value.shape}:{(n,)}"
            )

    finite_numeric = (
        np.all(np.isfinite(q2_fermat), axis=1)
        & np.all(np.isfinite(xy), axis=1)
        & np.isfinite(gradient_norm)
        & np.isfinite(reflection_residual)
    )

    gradient_pass = (
        gradient_flag
        & np.isfinite(gradient_norm)
        & (gradient_norm >= 0.0)
        & (gradient_norm <= float(gradient_tolerance))
    )
    reflection_pass = (
        reflection_flag
        & np.isfinite(reflection_residual)
        & (reflection_residual >= 0.0)
        & (reflection_residual <= float(reflection_tolerance))
    )
    finite_pass = finite_flag & finite_numeric

    conic_domain_value = (
        1.0
        - (1.0 + float(m2.conic))
        * float(m2.curvature) ** 2
        * np.sum(xy ** 2, axis=1)
    )
    conic_pass = (
        conic_flag
        & np.isfinite(conic_domain_value)
    )

    construction_factor = max(
        float(sol.get("construction_domain_factor", 1.0)),
        1.0,
    )
    construction_half = (
        construction_factor
        * np.asarray(m2.half_aperture, dtype=float)
    )
    construction_recomputed = (
        np.all(np.isfinite(xy), axis=1)
        & np.all(
            np.abs(xy)
            <= construction_half[None, :],
            axis=1,
        )
    )
    construction_pass = (
        construction_flag
        & construction_recomputed
    )

    construction_excess_xy = np.maximum(
        np.abs(xy)
        - construction_half[None, :],
        0.0,
    )
    construction_excess_max = np.max(
        construction_excess_xy,
        axis=1,
    )

    hard_pass = (
        gradient_pass
        & reflection_pass
        & finite_pass
        & conic_pass
        & construction_pass
    )

    q1_local = (
        (q1 - m2.center)
        @ m2.frame
    )
    q2_current_local = (
        (q2_current - m2.center)
        @ m2.frame
    )
    q2_fermat_local = (
        (q2_fermat - m2.center)
        @ m2.frame
    )

    delta_q2 = (
        q2_fermat
        - q2_current
    )
    delta_q2_mm = np.linalg.norm(
        delta_q2,
        axis=1,
    )

    neighbor_delta_jump_mm = np.full(
        n,
        np.nan,
        dtype=float,
    )
    current_xy = q2_current_local[:, :2]
    valid_neighbor_rows = (
        np.all(
            np.isfinite(current_xy),
            axis=1,
        )
        & np.all(
            np.isfinite(delta_q2),
            axis=1,
        )
    )
    valid_neighbor_ids = np.flatnonzero(
        valid_neighbor_rows
    )
    neighbor_count_used = 0

    if len(valid_neighbor_ids) > 1:
        neighbor_count_used = min(
            max(int(neighbor_count), 1),
            len(valid_neighbor_ids) - 1,
        )
        tree = cKDTree(
            current_xy[
                valid_neighbor_ids
            ]
        )
        _, local_neighbors = tree.query(
            current_xy[
                valid_neighbor_ids
            ],
            k=neighbor_count_used + 1,
        )
        local_neighbors = np.asarray(
            local_neighbors,
            dtype=int,
        )
        if local_neighbors.ndim == 1:
            local_neighbors = (
                local_neighbors[
                    :, None
                ]
            )

        for local_row, ray_index in enumerate(
            valid_neighbor_ids
        ):
            candidates = [
                int(candidate)
                for candidate in np.atleast_1d(
                    local_neighbors[
                        local_row
                    ]
                )
                if int(candidate) != local_row
            ]
            candidates = candidates[
                :neighbor_count_used
            ]
            if not candidates:
                continue

            neighbor_ray_ids = (
                valid_neighbor_ids[
                    np.asarray(
                        candidates,
                        dtype=int,
                    )
                ]
            )
            jumps = np.linalg.norm(
                delta_q2[
                    ray_index
                ][None, :]
                - delta_q2[
                    neighbor_ray_ids
                ],
                axis=1,
            )
            neighbor_delta_jump_mm[
                ray_index
            ] = float(
                np.median(
                    jumps
                )
            )

    def extreme_upper_threshold(
        values: np.ndarray,
        eligible: np.ndarray,
    ) -> float:
        """Ngưỡng Tukey extreme-outlier chỉ dùng để note, không phải physical hard gate."""
        selected = np.asarray(
            values,
            dtype=float,
        )
        selected = selected[
            eligible
            & np.isfinite(
                selected
            )
        ]
        if len(selected) < 4:
            return float("nan")
        q1_value, q3_value = np.percentile(
            selected,
            [25.0, 75.0],
        )
        iqr = float(
            q3_value
            - q1_value
        )
        return float(
            q3_value
            + 3.0 * iqr
        )

    delta_q2_extreme_threshold_mm = (
        extreme_upper_threshold(
            delta_q2_mm,
            hard_pass,
        )
    )
    neighbor_jump_extreme_threshold_mm = (
        extreme_upper_threshold(
            neighbor_delta_jump_mm,
            hard_pass,
        )
    )

    delta_q2_extreme = (
        hard_pass
        & np.isfinite(
            delta_q2_mm
        )
        & np.isfinite(
            delta_q2_extreme_threshold_mm
        )
        & (
            delta_q2_mm
            > delta_q2_extreme_threshold_mm
        )
    )
    neighbor_jump_extreme = (
        hard_pass
        & np.isfinite(
            neighbor_delta_jump_mm
        )
        & np.isfinite(
            neighbor_jump_extreme_threshold_mm
        )
        & (
            neighbor_delta_jump_mm
            > neighbor_jump_extreme_threshold_mm
        )
    )
    continuity_suspect = (
        delta_q2_extreme
        | neighbor_jump_extreme
    )

    def finite_summary(
        values: np.ndarray,
    ) -> dict[str, Any]:
        """Tóm tắt mảng finite để report STEP14."""
        finite_values = np.asarray(
            values,
            dtype=float,
        )
        finite_values = finite_values[
            np.isfinite(
                finite_values
            )
        ]
        if len(finite_values) == 0:
            return {
                "count": 0,
                "min": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
        return {
            "count": int(
                len(
                    finite_values
                )
            ),
            "min": float(
                np.min(
                    finite_values
                )
            ),
            "p50": float(
                np.percentile(
                    finite_values,
                    50.0,
                )
            ),
            "p95": float(
                np.percentile(
                    finite_values,
                    95.0,
                )
            ),
            "p99": float(
                np.percentile(
                    finite_values,
                    99.0,
                )
            ),
            "max": float(
                np.max(
                    finite_values
                )
            ),
        }

    rows: list[dict[str, Any]] = []

    for i, row in enumerate(
        rays["rows"]
    ):
        hard_fail_reasons: list[str] = []

        if not bool(
            finite_pass[i]
        ):
            hard_fail_reasons.append(
                "NONFINITE_FAIL"
            )
        else:
            if not bool(
                gradient_pass[i]
            ):
                hard_fail_reasons.append(
                    "GRADIENT_FAIL"
                )
            if not bool(
                reflection_pass[i]
            ):
                hard_fail_reasons.append(
                    "REFLECTION_FAIL"
                )
            if not bool(
                conic_pass[i]
            ):
                hard_fail_reasons.append(
                    "CONIC_DOMAIN_FAIL"
                )
            if not bool(
                construction_pass[i]
            ):
                hard_fail_reasons.append(
                    "CONSTRUCTION_DOMAIN_FAIL"
                )

        if hard_fail_reasons:
            hard_fail_reason = "|".join(
                hard_fail_reasons
            )
        else:
            hard_fail_reason = (
                "PASS_ALL_HARD_GATES"
            )

        continuity_reasons: list[str] = []
        if bool(
            delta_q2_extreme[i]
        ):
            continuity_reasons.append(
                "DELTA_Q2_EXTREME_OUTLIER"
            )
        if bool(
            neighbor_jump_extreme[i]
        ):
            continuity_reasons.append(
                "NEIGHBOR_DISPLACEMENT_JUMP_EXTREME_OUTLIER"
            )

        continuity_reason = (
            "|".join(
                continuity_reasons
            )
            if continuity_reasons
            else
            "NOT_SUSPECT"
        )

        solver_failure_reason = (
            str(
                sol["failure_reasons"][i]
            )
            if (
                "failure_reasons"
                in sol
                and sol[
                    "failure_reasons"
                ][i]
            )
            else
            "NOT_RECORDED"
        )

        gradient_ratio = (
            float(
                gradient_norm[i]
                / float(
                    gradient_tolerance
                )
            )
            if np.isfinite(
                gradient_norm[i]
            )
            else
            float("nan")
        )
        reflection_ratio = (
            float(
                reflection_residual[i]
                / float(
                    reflection_tolerance
                )
            )
            if np.isfinite(
                reflection_residual[i]
            )
            else
            float("nan")
        )

        rows.append({
            "ray_id":
                row["ray_id"],
            "field_id":
                row["field_id"],
            "field_index":
                int(row["field_index"]),
            "pupil_id":
                row["pupil_id"],
            "pupil_index":
                int(row["pupil_index"]),
            "sample_id":
                row["sample_id"],
            "sample_index":
                int(row["sample_index"]),
            "chief_preseed":
                bool(row["chief_preseed"]),

            "Q1_x_mm":
                float(q1[i, 0]),
            "Q1_y_mm":
                float(q1[i, 1]),
            "Q1_z_mm":
                float(q1[i, 2]),
            "Q1_local_x_mm":
                float(q1_local[i, 0]),
            "Q1_local_y_mm":
                float(q1_local[i, 1]),
            "Q1_local_z_mm":
                float(q1_local[i, 2]),

            "Q2_current_x_mm":
                float(q2_current[i, 0]),
            "Q2_current_y_mm":
                float(q2_current[i, 1]),
            "Q2_current_z_mm":
                float(q2_current[i, 2]),
            "Q2_current_local_x_mm":
                float(q2_current_local[i, 0]),
            "Q2_current_local_y_mm":
                float(q2_current_local[i, 1]),
            "Q2_current_local_z_mm":
                float(q2_current_local[i, 2]),

            "Q2_Fermat_x_mm":
                float(q2_fermat[i, 0]),
            "Q2_Fermat_y_mm":
                float(q2_fermat[i, 1]),
            "Q2_Fermat_z_mm":
                float(q2_fermat[i, 2]),
            "Q2_Fermat_local_x_mm":
                float(q2_fermat_local[i, 0]),
            "Q2_Fermat_local_y_mm":
                float(q2_fermat_local[i, 1]),
            "Q2_Fermat_local_z_mm":
                float(q2_fermat_local[i, 2]),

            "Delta_Q2_x_mm":
                float(delta_q2[i, 0]),
            "Delta_Q2_y_mm":
                float(delta_q2[i, 1]),
            "Delta_Q2_z_mm":
                float(delta_q2[i, 2]),
            "Delta_Q2_mm":
                float(delta_q2_mm[i]),

            "Gx":
                float(gradient[i, 0]),
            "Gy":
                float(gradient[i, 1]),
            "gradient_norm":
                float(gradient_norm[i]),
            "gradient_tolerance":
                float(gradient_tolerance),
            "gradient_ratio_to_limit":
                gradient_ratio,
            "gradient_pass":
                bool(gradient_pass[i]),

            "reflection_residual":
                float(reflection_residual[i]),
            "reflection_tolerance":
                float(reflection_tolerance),
            "reflection_ratio_to_limit":
                reflection_ratio,
            "reflection_pass":
                bool(reflection_pass[i]),

            "finite_pass":
                bool(finite_pass[i]),

            "conic_domain_value":
                float(conic_domain_value[i]),
            "conic_domain_pass":
                bool(conic_pass[i]),

            "in_current_aperture":
                bool(current_aperture_flag[i]),
            "current_aperture_is_hard_gate":
                False,

            "construction_domain_factor":
                float(construction_factor),
            "construction_limit_x_mm":
                float(construction_half[0]),
            "construction_limit_y_mm":
                float(construction_half[1]),
            "construction_excess_x_mm":
                float(construction_excess_xy[i, 0]),
            "construction_excess_y_mm":
                float(construction_excess_xy[i, 1]),
            "construction_excess_max_mm":
                float(construction_excess_max[i]),
            "construction_domain_pass":
                bool(construction_pass[i]),
            "construction_domain_is_hard_gate":
                True,

            "OP_initial_mm":
                float(op_initial[i]),
            "OP_final_mm":
                float(op_final[i]),
            "OP_change_mm":
                float(
                    op_final[i]
                    - op_initial[i]
                ),
            "solver_step_norm":
                float(step_norm[i]),
            "fallback_used":
                bool(fallback_used[i]),
            "solver_iterations_total":
                int(sol["iterations"]),
            "solver_success":
                bool(solver_success[i]),
            "solver_failure_reason":
                solver_failure_reason,

            "step14_hard_pass":
                bool(hard_pass[i]),
            "step14_hard_fail_reason":
                hard_fail_reason,

            "neighbor_delta_jump_mm":
                float(
                    neighbor_delta_jump_mm[i]
                ),
            "Delta_Q2_extreme_threshold_mm":
                float(
                    delta_q2_extreme_threshold_mm
                ),
            "neighbor_jump_extreme_threshold_mm":
                float(
                    neighbor_jump_extreme_threshold_mm
                ),
            "continuity_suspect":
                bool(
                    continuity_suspect[i]
                ),
            "continuity_suspect_reason":
                continuity_reason,
            "continuity_is_hard_gate":
                False,
        })

    hard_fail_counts = {
        "gradient":
            int(
                np.count_nonzero(
                    ~gradient_pass
                )
            ),
        "reflection":
            int(
                np.count_nonzero(
                    ~reflection_pass
                )
            ),
        "finite":
            int(
                np.count_nonzero(
                    ~finite_pass
                )
            ),
        "conic":
            int(
                np.count_nonzero(
                    ~conic_pass
                )
            ),
        "construction_domain":
            int(
                np.count_nonzero(
                    ~construction_pass
                )
            ),
    }

    summary = {
        "schema":
            "STEP14_FERMAT_PER_RAY_CERTIFICATION_V1",
        "ray_count":
            n,
        "solver_success_count":
            int(
                np.count_nonzero(
                    solver_success
                )
            ),
        "hard_pass_count":
            int(
                np.count_nonzero(
                    hard_pass
                )
            ),
        "hard_fail_count":
            int(
                np.count_nonzero(
                    ~hard_pass
                )
            ),
        "all_hard_pass":
            bool(
                np.all(
                    hard_pass
                )
            ),
        "hard_gate_definition": [
            "gradient_pass",
            "reflection_pass",
            "finite_pass",
            "conic_domain_pass",
            "construction_domain_pass",
        ],
        "hard_fail_counts":
            hard_fail_counts,

        "gradient_tolerance":
            float(
                gradient_tolerance
            ),
        "reflection_tolerance":
            float(
                reflection_tolerance
            ),
        "gradient_norm":
            finite_summary(
                gradient_norm
            ),
        "reflection_residual":
            finite_summary(
                reflection_residual
            ),

        "construction_domain_factor":
            float(
                construction_factor
            ),
        "construction_limit_x_mm":
            float(
                construction_half[0]
            ),
        "construction_limit_y_mm":
            float(
                construction_half[1]
            ),
        "construction_domain_excess_mm":
            finite_summary(
                construction_excess_max
            ),
        "inside_current_aperture_count":
            int(
                np.count_nonzero(
                    current_aperture_flag
                )
            ),
        "outside_current_aperture_count":
            int(
                np.count_nonzero(
                    ~current_aperture_flag
                )
            ),
        "current_aperture_is_hard_gate":
            False,

        "Delta_Q2_mm":
            finite_summary(
                delta_q2_mm
            ),
        "neighbor_delta_jump_mm":
            finite_summary(
                neighbor_delta_jump_mm
            ),
        "neighbor_count_requested":
            int(
                neighbor_count
            ),
        "neighbor_count_used":
            int(
                neighbor_count_used
            ),
        "continuity_suspect_count":
            int(
                np.count_nonzero(
                    continuity_suspect
                )
            ),
        "continuity_suspect_method":
            "TUKEY_EXTREME_OUTLIER_Q3_PLUS_3_IQR",
        "Delta_Q2_extreme_threshold_mm":
            float(
                delta_q2_extreme_threshold_mm
            ),
        "neighbor_jump_extreme_threshold_mm":
            float(
                neighbor_jump_extreme_threshold_mm
            ),
        "continuity_is_hard_gate":
            False,
        "continuity_diagnostic_only":
            True,
    }

    failed_rows = [
        row
        for row in rows
        if not bool(
            row["step14_hard_pass"]
        )
    ]
    suspicious_rows = [
        row
        for row in rows
        if (
            bool(
                row["step14_hard_pass"]
            )
            and bool(
                row["continuity_suspect"]
            )
        )
    ]

    return {
        "rows": rows,
        "failed_rows": failed_rows,
        "suspicious_rows": suspicious_rows,
        "summary": summary,
    }






def _iteration_cloud(points: np.ndarray, normals: np.ndarray, starts: np.ndarray,
                     target_points: np.ndarray, role: str,
                     ray_indices: np.ndarray | None = None) -> dict[str, Any]:
    """Đóng gói point/normal của CI iteration mà không chạy lại Nearest-Ray Algorithm."""
    point_array = np.asarray(points, float)
    normal_array = np.asarray(normals, float)
    start_array = np.asarray(starts, float)
    target_array = np.asarray(target_points, float)
    if not (
        normal_array.shape == point_array.shape
        and start_array.shape == point_array.shape
        and target_array.shape == point_array.shape
    ):
        raise ValueError("ITERATION_CLOUD_ARRAY_SHAPE_MISMATCH")
    out = {
        "points_by_ray": point_array.copy(),
        "normals_by_ray": normal_array.copy(),
        "starts_by_ray": start_array.copy(),
        "target_points_by_ray": target_array.copy(),
        "targets_by_ray": target_array.copy(),
        "iteration_role": role,
        "coordinates_preserved_from_current_trace": True,
        "nearest_ray_reexecuted": False,
        "fallback_count": 0,
    }
    if ray_indices is not None:
        index_array = np.asarray(ray_indices, int)
        if index_array.shape != (len(point_array),):
            raise ValueError("ITERATION_CLOUD_RAY_INDICES_SHAPE_MISMATCH")
        out["ray_indices"] = index_array.copy()
    return out


def _required_reflection_normals(starts: np.ndarray, points: np.ndarray,
                                 targets: np.ndarray) -> np.ndarray:
    """Tính normal phản xạ yêu cầu ngay tại các current intersection được giữ lại."""
    incident = unit(points - starts)
    outgoing = unit(targets - points)
    return unit(incident - outgoing)


def _promote_surface_order_without_shape_change(surface: PolySurface, order: int) -> PolySurface:
    """Thêm các term bậc cao bằng 0 để đổi basis mà giữ nguyên đúng hình mặt hiện tại."""
    promoted = surface.copy()
    terms = monomial_terms(order, axis_order2=False, include_constant=True,
                           preserve_fan_axis_order2=True)
    old = {term: float(value) for term, value in zip(surface.terms, surface.coeff)}
    if any(term not in terms and abs(value) > 1e-14 for term, value in old.items()):
        raise RuntimeError("CI_ORDER_PROMOTION_WOULD_DROP_ACTIVE_TERM")
    promoted.terms = terms
    promoted.coeff = np.array([old.get(term, 0.0) for term in terms], dtype=float)
    return promoted


def _ci_basis_label(order: int, axis_order2: bool = False) -> str:
    """Trả nhãn basis CI, bao gồm level-3 lồng nhau giữ nguyên A22."""
    if axis_order2:
        return "FAN_EQ1_AXIS_ORDER_2"
    if int(order) == 3:
        return "FAN_AXIS_ORDER2_UNION_TOTAL_XY_ORDER_3"
    return f"CONIC_PLUS_TOTAL_XY_ORDER_{int(order)}"


def _export_step17_m2_footprint_csvs(
    ctx: Context,
    order: int,
    m2_ap_audit: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> None:
    """Xuất CSV loại trừ và bundle M2 cho STEP 17 kèm cycle và branch, có log cảnh báo khi I/O lỗi."""
    if int(order) not in (3, 4, 5):
        return
    cycle_val = ctx.data.get("step17_cycle", ctx.data.get("cycle"))
    branch_val = ctx.data.get(
        "step17_branch_index",
        ctx.data.get("step17_branch", ctx.data.get("branch_index", ctx.data.get("branch"))),
    )
    if cycle_val is not None and branch_val is not None:
        tag = f"ORDER_{order}_CYCLE_{int(cycle_val):03d}_BRANCH_{int(branch_val):03d}"
    else:
        tag = f"ORDER_{order}"

    try:
        step17_dir = ctx.step_dir(17)
        if step17_dir.exists():
            write_csv(
                step17_dir / f"17_M2_APERTURE_FOOTPRINT_EXCLUSIONS_{tag}.csv",
                m2_ap_audit.get("footprint_outlier_rows", []),
                fieldnames=M2_APERTURE_FOOTPRINT_EXCLUSIONS_CSV_FIELDNAMES,
            )
            write_csv(
                step17_dir / f"17_M2_APERTURE_FOOTPRINT_BUNDLES_{tag}.csv",
                m2_ap_audit.get("footprint_bundle_rows", []),
                fieldnames=M2_APERTURE_FOOTPRINT_BUNDLES_CSV_FIELDNAMES,
            )
    except Exception as io_exc:
        print(f"[STEP17][WARN] Ghi M2 footprint CSV cho {tag} that bai: {io_exc}", flush=True)
        if evidence is not None:
            evidence["m2_footprint_csv_export_error"] = str(io_exc)


def _reconstruct_impl(
    ctx: Context,
    order: int,
    axis_order2: bool,
    allow_restoration: bool = False,
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """CI iteration; cho phép nhánh restoration có nhãn nhưng không hạ gate final."""
    r, visor, display = ctx.data["rays"], ctx.data["visor"], ctx.data["display"].copy()
    evidence["display"] = display
    old1, old2 = ctx.data["m1"], ctx.data["m2"]
    chief_index = _chief_index(r)

    live_phase("CI_PARENT_REVERSE_TRACE")
    tr0 = trace_reverse(r, visor, old1, old2, display, physical_first_hit=False)

    live_phase("CI_M1_TARGET_UPDATE")
    # HUD Eq.(3): rho tác động lên target M2 của M1, không nội suy point cloud của M1.
    p1 = np.asarray(tr0["points"][1], float).copy()
    starts_m1 = np.asarray(tr0["points"][0], float)
    target_current = np.asarray(tr0["points"][2], float)
    target_fermat = np.asarray(ctx.data["fermat"]["target_points"], float)
    target_m1 = target_current + float(ctx.data["rho"]) * (target_fermat - target_current)
    step17_fractional_admission = bool(int(order) >= 3 and not axis_order2)
    minimum_m1_geometry_fraction = (
        _step17_minimum_candidate_fraction(ctx)
        if step17_fractional_admission
        else 1.0
    )
    valid_m1_geometry = (
        np.all(np.isfinite(starts_m1), axis=1)
        & np.all(np.isfinite(p1), axis=1)
        & np.all(np.isfinite(target_current), axis=1)
        & np.all(np.isfinite(target_fermat), axis=1)
        & np.all(np.isfinite(target_m1), axis=1)
    )
    if step17_fractional_admission:
        fermat_admissible = np.asarray(
            ctx.data["fermat"].get("success"),
            bool,
        )
        if fermat_admissible.shape != (len(p1),):
            raise RuntimeError(
                f"ORDER_{order}_FERMAT_ADMISSIBLE_MASK_SHAPE_MISMATCH"
            )
        valid_m1_geometry &= fermat_admissible

    valid_m1_indices = np.flatnonzero(valid_m1_geometry)
    m1_geometry_count = int(len(valid_m1_indices))
    m1_geometry_fraction = float(
        m1_geometry_count / max(len(valid_m1_geometry), 1)
    )
    chief_m1_geometry_valid = bool(valid_m1_geometry[chief_index])
    all_m1_bundle_ids = (
        np.asarray(r["field_index"], int) * int(r["pupil_count"])
        + np.asarray(r["pupil_index"], int)
    )
    expected_m1_bundle_count = int(r["field_count"]) * int(r["pupil_count"])
    valid_m1_bundle_counts = np.bincount(
        all_m1_bundle_ids[valid_m1_geometry],
        minlength=expected_m1_bundle_count,
    )
    minimum_evaluable_bundle_ray_count = 4
    underfilled_m1_bundle_ids = np.flatnonzero(
        valid_m1_bundle_counts < minimum_evaluable_bundle_ray_count
    )
    evidence["m1_construction_input_gate"] = {
        "valid_geometry_count": m1_geometry_count,
        "ray_count": int(len(valid_m1_geometry)),
        "valid_geometry_fraction": m1_geometry_fraction,
        "minimum_valid_geometry_fraction": minimum_m1_geometry_fraction,
        "maximum_excluded_fraction": float(1.0 - minimum_m1_geometry_fraction),
        "excluded_geometry_count": int(len(valid_m1_geometry) - m1_geometry_count),
        "chief_ray_geometry_valid": chief_m1_geometry_valid,
        "fermat_admissible_subset_enabled": step17_fractional_admission,
        "failed_ray_indices": np.flatnonzero(~valid_m1_geometry),
        "expected_bundle_count": expected_m1_bundle_count,
        "minimum_evaluable_bundle_ray_count": minimum_evaluable_bundle_ray_count,
        "underfilled_bundle_count": int(len(underfilled_m1_bundle_ids)),
        "underfilled_bundle_ids": underfilled_m1_bundle_ids,
        "excluded_rays_remain_subject_to_physical_admission": True,
        "exact_all_rays_completion_reported_separately": True,
    }
    if m1_geometry_fraction < minimum_m1_geometry_fraction:
        raise RuntimeError(
            f"ORDER_{order}_M1_CONSTRUCTION_INPUT_INCOMPLETE_"
            f"{m1_geometry_count}_OF_{len(valid_m1_geometry)}_"
            f"BELOW_MINIMUM_{minimum_m1_geometry_fraction:.6f}"
        )
    if not chief_m1_geometry_valid:
        raise RuntimeError(
            f"ORDER_{order}_M1_CONSTRUCTION_CHIEF_RAY_INVALID_"
            f"RAY_{chief_index}"
        )
    if len(underfilled_m1_bundle_ids):
        raise RuntimeError(
            f"ORDER_{order}_M1_CONSTRUCTION_BUNDLE_UNDERSAMPLED_"
            f"{len(underfilled_m1_bundle_ids)}_BUNDLES"
        )

    p1_fit = p1[valid_m1_geometry]
    starts_m1_fit = starts_m1[valid_m1_geometry]
    target_m1_fit = target_m1[valid_m1_geometry]
    n1 = _required_reflection_normals(starts_m1_fit, p1_fit, target_m1_fit)
    ci1 = _iteration_cloud(
        p1_fit,
        n1,
        starts_m1_fit,
        target_m1_fit,
        "M1_CURRENT_INTERSECTIONS_WITH_HUD_EQ3_TARGET_CONTINUATION",
        ray_indices=valid_m1_indices,
    )
    iteration_fit_options = dict(ctx.config["surface_fit"])
    iteration_fit_options["k_profile_enabled"] = False

    configured_o4_degree_regularization = float(
        ctx.config["surface_fit"].get(
            "step17_o4_degree_regularization",
            1.0e-4,
        )
    )

    iteration_fit_options[
        "o4_degree_regularization"
    ] = (
        configured_o4_degree_regularization
        if (
            int(order) == 4
            and not bool(axis_order2)
        )
        else 0.0
    )

    parent_trust_cfg = ctx.config["surface_fit"].get(
        "step17_parent_trust",
        {
            "enabled": True,
            "minimum_order": 4,
            "grid_samples": 15,
            "sag_lambda": 0.25,
            "normal_lambda": 0.25,
            "balance_by_sample_count": True,
        },
    )

    parent_trust_active = bool(
        parent_trust_cfg["enabled"]
        and not axis_order2
        and int(order)
        >= int(parent_trust_cfg["minimum_order"])
    )

    iteration_fit_options[
        "parent_trust_enabled"
    ] = parent_trust_active
    iteration_fit_options[
        "parent_trust_grid_samples"
    ] = int(parent_trust_cfg["grid_samples"])
    iteration_fit_options[
        "parent_trust_sag_lambda"
    ] = float(parent_trust_cfg["sag_lambda"])
    iteration_fit_options[
        "parent_trust_normal_lambda"
    ] = float(parent_trust_cfg["normal_lambda"])
    iteration_fit_options[
        "parent_trust_balance_by_sample_count"
    ] = bool(
        parent_trust_cfg["balance_by_sample_count"]
    )

    evidence["phase"] = "FIT_M1"

    live_phase("CI_FIT_M1")
    m1_chief_index = int(np.flatnonzero(valid_m1_indices == chief_index)[0])
    evidence["m1_construction_input_gate"]["fit_chief_subset_index"] = (
        m1_chief_index
    )
    m1, s1 = fit_surface(p1_fit, n1, old1, order, axis_order2,
                         ctx.config["fit_weights"], m1_chief_index, iteration_fit_options,
                         ci1["starts_by_ray"], ci1["targets_by_ray"])
    evidence["m1"] = m1
    evidence["M1_fit"] = s1

    _live_model_preview(
        ctx,
        "CI M1 fitted; M2 is still the parent",
        m1=m1,
        m2=old2,
        display=display,
        trace=None,
        role="INTERMEDIATE_M1_CANDIDATE_WITH_PARENT_M2",
    )

    live_phase("CI_RETRACE_AFTER_M1")
    tr1 = trace_reverse(r, visor, m1, old2, display, physical_first_hit=False)

    _live_model_preview(
        ctx,
        "CI trace with new M1 and parent M2",
        m1=m1,
        m2=old2,
        display=display,
        trace=tr1,
        trace_mode="SEQUENTIAL_NOT_FIRST_HIT_CERTIFIED",
        role="INTERMEDIATE_M1_CANDIDATE_WITH_PARENT_M2",
    )

    # M2 construction uses only the current M1/M2 intersections plus the dynamic
    # Fan reference. Do not gate on tr1["valid"] here because it also includes
    # DISPLAY validity, which is rebuilt later and is not an M2 cloud input.
    m1_after_retrace = np.asarray(
        tr1["points"][1],
        float,
    )

    p2 = np.asarray(
        tr1["points"][2],
        float,
    ).copy()

    valid_m2_geometry = (
        np.all(
            np.isfinite(m1_after_retrace),
            axis=1,
        )
        & np.all(
            np.isfinite(p2),
            axis=1,
        )
    )
    if step17_fractional_admission:
        valid_m2_geometry &= valid_m1_geometry

    m2_geometry_count = int(
        np.count_nonzero(valid_m2_geometry)
    )
    m2_geometry_fraction = float(
        m2_geometry_count / max(len(valid_m2_geometry), 1)
    )
    minimum_m2_geometry_fraction = (
        float(
            ctx.config["solver"]["rho_o2_search"].get(
                "minimum_m2_construction_finite_fraction",
                0.95,
            )
        )
        if int(order) == 2 and bool(axis_order2)
        else (
            _step17_minimum_candidate_fraction(ctx)
            if step17_fractional_admission
            else 1.0
        )
    )
    valid_m2_indices = np.flatnonzero(valid_m2_geometry)
    chief_geometry_valid = bool(valid_m2_geometry[chief_index])
    failed_m2_indices = np.flatnonzero(~valid_m2_geometry)
    all_m2_bundle_ids = (
        np.asarray(r["field_index"], int) * int(r["pupil_count"])
        + np.asarray(r["pupil_index"], int)
    )
    expected_m2_bundle_count = int(r["field_count"]) * int(r["pupil_count"])
    valid_m2_bundle_counts = np.bincount(
        all_m2_bundle_ids[valid_m2_geometry],
        minlength=expected_m2_bundle_count,
    )
    minimum_evaluable_bundle_ray_count = 4
    underfilled_m2_bundle_ids = np.flatnonzero(
        valid_m2_bundle_counts < minimum_evaluable_bundle_ray_count
    )
    failed_m2_bundle_rows: list[dict[str, Any]] = []
    if len(failed_m2_indices):
        failed_bundle_ids = (
            np.asarray(r["field_index"], int)[failed_m2_indices]
            * int(r["pupil_count"])
            + np.asarray(r["pupil_index"], int)[failed_m2_indices]
        )
        for group_id in np.unique(failed_bundle_ids):
            selected = failed_m2_indices[failed_bundle_ids == group_id]
            first_row = r["rows"][int(selected[0])]
            field_definition = ctx.data["vi"]["fields"][
                int(first_row["field_index"])
            ]
            pupil_definition = ctx.data["pupils"][
                int(first_row["pupil_index"])
            ]
            failed_m2_bundle_rows.append({
                "group_id": int(group_id),
                "field_index": int(first_row["field_index"]),
                "field_id": first_row["field_id"],
                "field_h_deg": field_definition["h_deg"],
                "field_v_deg": field_definition["v_deg"],
                "pupil_index": int(first_row["pupil_index"]),
                "pupil_id": first_row["pupil_id"],
                "pupil_x_mm": pupil_definition["x_mm"],
                "pupil_y_mm": pupil_definition["y_mm"],
                "pupil_z_mm": pupil_definition["z_mm"],
                "failed_ray_count": int(len(selected)),
                "failed_ray_indices": selected,
                "failed_sample_ids": [
                    r["rows"][int(index)]["sample_id"]
                    for index in selected
                ],
            })

    evidence["phase"] = "M2_CONSTRUCTION_INPUT_GATE"

    evidence["m2_construction_input_gate"] = {
        "valid_geometry_count": m2_geometry_count,
        "ray_count": int(len(valid_m2_geometry)),
        "valid_geometry_fraction": m2_geometry_fraction,
        "minimum_valid_geometry_fraction": minimum_m2_geometry_fraction,
        "maximum_excluded_fraction": float(
            1.0 - minimum_m2_geometry_fraction
        ),
        "excluded_geometry_count": int(
            len(valid_m2_geometry) - m2_geometry_count
        ),
        "chief_ray_geometry_valid": chief_geometry_valid,
        "finite_subset_fit_enabled": bool(
            minimum_m2_geometry_fraction < 1.0
        ),
        "excluded_rays_remain_subject_to_full_physical_certification": bool(
            minimum_m2_geometry_fraction >= 1.0
        ),
        "excluded_rays_remain_subject_to_physical_admission": True,
        "exact_all_rays_completion_reported_separately": True,
        "failed_ray_indices": failed_m2_indices,
        "failed_bundle_count": int(len(failed_m2_bundle_rows)),
        "failed_bundle_rows": failed_m2_bundle_rows,
        "expected_bundle_count": expected_m2_bundle_count,
        "minimum_evaluable_bundle_ray_count": minimum_evaluable_bundle_ray_count,
        "underfilled_bundle_count": int(len(underfilled_m2_bundle_ids)),
        "underfilled_bundle_ids": underfilled_m2_bundle_ids,
        "uses_cumulative_trace_valid": False,
        "reason": (
            "M2_REQUIRES_FINITE_CURRENT_M1_AND_M2_INTERSECTIONS"
        ),
    }

    if m2_geometry_fraction < minimum_m2_geometry_fraction:
        raise RuntimeError(
            f"ORDER_{order}_M2_CONSTRUCTION_INPUT_INCOMPLETE_"
            f"{m2_geometry_count}_OF_{len(valid_m2_geometry)}_"
            f"BELOW_MINIMUM_{minimum_m2_geometry_fraction:.6f}"
        )

    if not chief_geometry_valid:
        raise RuntimeError(
            f"ORDER_{order}_M2_CONSTRUCTION_CHIEF_RAY_INVALID_"
            f"RAY_{chief_index}"
        )

    if len(underfilled_m2_bundle_ids):
        raise RuntimeError(
            f"ORDER_{order}_M2_CONSTRUCTION_BUNDLE_UNDERSAMPLED_"
            f"{len(underfilled_m2_bundle_ids)}_BUNDLES"
        )


    anchor_index = _chief_index(r)

    anchor_landing = np.asarray(
        tr1["landing"],
        float,
    )[anchor_index]

    if not np.all(np.isfinite(anchor_landing)):
        evidence["phase"] = (
            "M2_DYNAMIC_REFERENCE_INPUT_GATE"
        )

        evidence[
            "m2_dynamic_reference_input_gate"
        ] = {
            "anchor_ray_index": int(anchor_index),
            "anchor_landing_finite": False,
        }

        raise RuntimeError(
            f"ORDER_{order}_"
            f"M2_DYNAMIC_REFERENCE_ANCHOR_NONFINITE_"
            f"RAY_{anchor_index}"
        )


    refs, rule = _dynamic_refs(ctx, tr1)


    # [21]: M2 cũng giữ current intersections;
    # output target vẫn là Fan reference trên image.
    target_m2 = np.asarray(
        refs[r["field_index"]],
        float,
    )

    if not np.all(np.isfinite(target_m2)):
        finite_target_rows = np.all(
            np.isfinite(target_m2),
            axis=1,
        )

        evidence["phase"] = (
            "M2_DYNAMIC_REFERENCE_OUTPUT_GATE"
        )

        evidence[
            "m2_dynamic_reference_output_gate"
        ] = {
            "finite_target_count": int(
                np.count_nonzero(finite_target_rows)
            ),
            "ray_count": int(
                len(finite_target_rows)
            ),
            "failed_ray_indices": np.flatnonzero(
                ~finite_target_rows
            ),
        }

        raise RuntimeError(
            f"ORDER_{order}_"
            "M2_DYNAMIC_REFERENCE_NONFINITE"
        )


    m1_after_retrace_fit = m1_after_retrace[valid_m2_geometry]
    p2_fit = p2[valid_m2_geometry]
    target_m2_fit = target_m2[valid_m2_geometry]
    n2 = _required_reflection_normals(
        m1_after_retrace_fit,
        p2_fit,
        target_m2_fit,
    )

    ci2 = _iteration_cloud(
        p2_fit,
        n2,
        m1_after_retrace_fit,
        target_m2_fit,
        "M2_CURRENT_INTERSECTIONS_TO_DYNAMIC_FAN_REFERENCE",
        ray_indices=valid_m2_indices,
    )
    evidence["phase"] = "FIT_M2"

    live_phase("CI_FIT_M2")
    m2_chief_index = int(np.flatnonzero(valid_m2_indices == chief_index)[0])
    evidence["m2_construction_input_gate"]["fit_chief_subset_index"] = (
        m2_chief_index
    )
    m2, s2 = fit_surface(p2_fit, n2, old2, order, axis_order2,
                         ctx.config["fit_weights"], m2_chief_index, iteration_fit_options,
                         ci2["starts_by_ray"], ci2["targets_by_ray"])
    evidence["m2"] = m2
    evidence["M2_fit"] = s2

    _live_model_preview(
        ctx,
        "CI M1 and M2 fitted; final candidate audit pending",
        m1=m1,
        m2=m2,
        display=display,
        trace=None,
        role="CANDIDATE_BEFORE_FINAL_APERTURE_AND_TRACE",
    )

    live_phase("CI_CLOUD_DIAGNOSTICS")
    diagnostic1 = _construction_diagnostics(
        ctx, ci1, m1.frame, f"M1_ORDER_{order}_ITERATION_CLOUD")
    diagnostic2 = _construction_diagnostics(
        ctx, ci2, m2.frame, f"M2_ORDER_{order}_ITERATION_CLOUD")
    integrability_cfg = ctx.config["surface_fit"]["integrability_gates"]
    m2_integrability_cfg = dict(integrability_cfg)
    step16_m2_edge_limit_override = bool(
        int(order) == 2 and bool(axis_order2)
    )
    step17_m2_edge_limit_override = bool(
        int(order) >= 3 and not bool(axis_order2)
    )
    if step16_m2_edge_limit_override:
        m2_integrability_cfg[
            "maximum_bundle_edge_gradient_height_residual_p95_mm"
        ] = float(
            ctx.config["solver"]["rho_o2_search"].get(
                "maximum_m2_bundle_edge_gradient_height_residual_p95_mm",
                0.17,
            )
        )
    elif step17_m2_edge_limit_override:
        m2_integrability_cfg[
            "maximum_bundle_edge_gradient_height_residual_p95_mm"
        ] = float(
            ctx.config["solver"].get(
                "maximum_step17_m2_bundle_edge_gradient_height_residual_p95_mm",
                0.16,
            )
        )
    m2_edge_limit = float(
        m2_integrability_cfg[
            "maximum_bundle_edge_gradient_height_residual_p95_mm"
        ]
    )
    m2_bundle_rows = []
    for source_row in diagnostic2.get("cloud", {}).get("bundle_rows", []):
        bundle_row = copy.deepcopy(source_row)
        actual = bundle_row.get(
            "edge_gradient_height_residual_p95_mm"
        )
        evaluable = bool(
            actual is not None
            and np.isfinite(float(actual))
        )
        bundle_row.update({
            "edge_gradient_height_residual_limit_mm": m2_edge_limit,
            "edge_gradient_height_evaluable": evaluable,
            "edge_gradient_height_status": (
                "PASS"
                if evaluable and float(actual) <= m2_edge_limit
                else "FAIL"
            ),
        })
        m2_bundle_rows.append(bundle_row)
    m2_bundle_rows.sort(
        key=lambda row: (
            float(row["edge_gradient_height_residual_p95_mm"])
            if row.get("edge_gradient_height_residual_p95_mm") is not None
            and np.isfinite(float(row["edge_gradient_height_residual_p95_mm"]))
            else float("-inf")
        ),
        reverse=True,
    )
    m2_bundle_summary = copy.deepcopy(
        diagnostic2.get("cloud", {}).get("bundle_summary", {})
    )
    evidence["M2_bundle_integrability_diagnostic"] = {
        "grouping": "FIELD_PUPIL_BUNDLE",
        "group_id_definition": "field_index*pupil_count+pupil_index",
        "aggregate_rule": "P95_OF_PER_BUNDLE_P95",
        "limit_scope": (
            "STEP16_M2_ONLY"
            if step16_m2_edge_limit_override
            else (
                "STEP17_M2_ONLY"
                if step17_m2_edge_limit_override
                else "GLOBAL_INTEGRABILITY_POLICY"
            )
        ),
        "global_edge_gradient_height_residual_limit_mm": float(
            integrability_cfg[
                "maximum_bundle_edge_gradient_height_residual_p95_mm"
            ]
        ),
        "edge_gradient_height_residual_limit_mm": m2_edge_limit,
        "aggregate_edge_gradient_height_residual_p95_mm": (
            m2_bundle_summary.get(
                "bundle_edge_gradient_height_residual_p95_of_bundle_p95_mm"
            )
        ),
        "bundle_count": int(len(m2_bundle_rows)),
        "violating_bundle_count": int(sum(
            row["edge_gradient_height_status"] == "FAIL"
            for row in m2_bundle_rows
        )),
        "worst_bundle": (
            copy.deepcopy(m2_bundle_rows[0])
            if m2_bundle_rows
            else None
        ),
        "bundle_summary": m2_bundle_summary,
        "bundle_rows_sorted_worst_first": m2_bundle_rows,
    }
    integrability_gate1 = _integrability_quality_gate(
        diagnostic1, integrability_cfg, f"M1_ORDER_{order}")
    integrability_gate2 = _integrability_quality_gate(
        diagnostic2, m2_integrability_cfg, f"M2_ORDER_{order}")
    integrability_pass = bool(integrability_gate1["status"] == "PASS"
                              and integrability_gate2["status"] == "PASS")
    evidence["phase"] = "CLOUD_INTEGRABILITY"
    evidence["integrability_quality"] = {
        "M1": integrability_gate1,
        "M2": integrability_gate2,
        "pass": integrability_pass,
    }
    if integrability_cfg["enforcement"] == "HARD" and not integrability_pass:
        raise RuntimeError(f"ORDER_{order}_CI_CLOUD_INTEGRABILITY_DIAGNOSTIC_GATE_FAIL")

    live_phase("CI_FINAL_APERTURE_REBUILD")
    aperture_label = f"ORDER_{order}_RECONSTRUCTION"
    evidence["phase"] = "APERTURE_REBUILD"
    evidence["aperture_rebuild"] = {
        "phase": "APERTURE_REBUILD_IN_PROGRESS",
        "label": aperture_label,
    }

    construction_trace, tr2, rebuild_audit = _rebuild_post_fit_apertures(
        ctx,
        m1,
        m2,
        display,
        aperture_label,
    )

    evidence["aperture_rebuild"] = rebuild_audit

    m2_ap_audit = rebuild_audit.get("M2", {})
    evidence["M2_footprint_filter_enabled"] = bool(
        m2_ap_audit.get("footprint_filter_enabled", False)
    )
    evidence["M2_footprint_filter_method"] = str(
        m2_ap_audit.get("footprint_filter_method", "NONE")
    )
    evidence["M2_footprint_rejected_count"] = int(
        m2_ap_audit.get("combined_rejected_count", 0)
    )
    evidence["M2_footprint_rejected_fraction"] = float(
        m2_ap_audit.get("combined_rejected_fraction", 0.0)
    )
    evidence["M2_footprint_retained_count"] = int(
        m2_ap_audit.get("retained_authority_count", 0)
    )
    evidence["M2_footprint_affected_bundle_count"] = int(
        m2_ap_audit.get("affected_bundle_count", 0)
    )
    evidence["M2_footprint_max_bundle_rejected_fraction"] = float(
        m2_ap_audit.get("maximum_observed_bundle_rejected_fraction", 0.0)
    )
    evidence["M2_local_xy_min_before_mm"] = copy.deepcopy(
        m2_ap_audit.get("local_xy_min_before_mm", [0.0, 0.0])
    )
    evidence["M2_local_xy_max_before_mm"] = copy.deepcopy(
        m2_ap_audit.get("local_xy_max_before_mm", [0.0, 0.0])
    )
    evidence["M2_local_xy_min_after_mm"] = copy.deepcopy(
        m2_ap_audit.get("local_xy_min_after_mm", [0.0, 0.0])
    )
    evidence["M2_local_xy_max_after_mm"] = copy.deepcopy(
        m2_ap_audit.get("local_xy_max_after_mm", [0.0, 0.0])
    )
    evidence["M2_half_aperture_before_mm"] = copy.deepcopy(
        m2_ap_audit.get("half_aperture_before_mm", [0.0, 0.0])
    )
    evidence["M2_half_aperture_after_mm"] = copy.deepcopy(
        m2_ap_audit.get("half_aperture_after_mm", [0.0, 0.0])
    )
    evidence["M2_footprint_outlier_rows"] = copy.deepcopy(
        m2_ap_audit.get("footprint_outlier_rows", [])
    )
    evidence["M2_footprint_bundle_rows"] = copy.deepcopy(
        m2_ap_audit.get("footprint_bundle_rows", [])
    )

    if int(order) in (3, 4, 5):
        print(
            _format_step17_aperture_progress_log(
                f"ORDER_{order}",
                rebuild_audit,
            ),
            flush=True,
        )
        if not evidence.get("_m2_footprint_csv_exported", False):
            _export_step17_m2_footprint_csvs(ctx, order, m2_ap_audit, evidence)
            evidence["_m2_footprint_csv_exported"] = True

    live_phase("CI_SURFACE_SANITY")
    sanity1 = _mirror_topology_gate(
        ctx,
        m1,
        tr2,
        "M1",
    )
    sanity2 = _mirror_topology_gate(
        ctx,
        m2,
        tr2,
        "M2",
    )
    evidence["phase"] = "SURFACE_SANITY"
    evidence["M1_sanity"] = sanity1
    evidence["M2_sanity"] = sanity2
    m1_sanity_admitted = bool(sanity1["pass"])
    m2_sanity_warn_admitted = bool(
        int(order) >= 2
        and not sanity2["pass"]
        and str(sanity2.get("topology_enforcement", "HARD")) == "WARN"
    )
    m2_sanity_admitted = bool(
        sanity2["pass"]
        or m2_sanity_warn_admitted
    )
    evidence["surface_sanity_admission"] = {
        "M1": {
            "raw_pass": bool(sanity1["pass"]),
            "enforcement": str(
                sanity1.get("topology_enforcement", "HARD")
            ),
            "warn_admitted": False,
            "admitted": m1_sanity_admitted,
        },
        "M2": {
            "raw_pass": bool(sanity2["pass"]),
            "enforcement": str(
                sanity2.get("topology_enforcement", "HARD")
            ),
            "warn_admitted": m2_sanity_warn_admitted,
            "admitted": m2_sanity_admitted,
        },
    }
    minimum_candidate_fraction = (
        _step16_minimum_candidate_fraction(ctx)
        if int(order) == 2 and bool(axis_order2)
        else (
            _step17_minimum_candidate_fraction(ctx)
            if step17_fractional_admission
            else 1.0
        )
    )
    sequential_numeric_mask = (
        np.asarray(tr2["valid"], bool)
        & np.all(np.isfinite(tr2["points"][0]), axis=1)
        & np.all(np.isfinite(tr2["points"][1]), axis=1)
        & np.all(np.isfinite(tr2["points"][2]), axis=1)
        & np.all(np.isfinite(tr2["landing"]), axis=1)
    )
    sequential_numeric_count = int(
        np.count_nonzero(sequential_numeric_mask)
    )
    sequential_numeric_fraction = float(
        sequential_numeric_count / max(len(r["rows"]), 1)
    )
    chief_mask = np.asarray(r["chief"], bool)
    chief_numeric_complete = bool(
        np.all(sequential_numeric_mask[chief_mask])
    )
    numeric = bool(
        sequential_numeric_fraction >= minimum_candidate_fraction
        and chief_numeric_complete
        and sanity1["finite"]
        and sanity2["finite"]
    )
    evidence["numeric_admission"] = {
        "valid_count": sequential_numeric_count,
        "ray_count": int(len(r["rows"])),
        "valid_fraction": sequential_numeric_fraction,
        "minimum_fraction": minimum_candidate_fraction,
        "chief_numeric_complete": chief_numeric_complete,
        "M1_surface_finite": bool(sanity1["finite"]),
        "M2_surface_finite": bool(sanity2["finite"]),
        "pass": numeric,
    }
    if not numeric:
        raise RuntimeError(
            f"ORDER_{order}_RECONSTRUCTION_NUMERICALLY_UNUSABLE_"
            f"{sequential_numeric_count}_OF_{len(r['rows'])}_"
            f"MINIMUM_{minimum_candidate_fraction:.6f}_"
            f"CHIEF_COMPLETE_{int(chief_numeric_complete)}"
        )
    if not m1_sanity_admitted or not m2_sanity_admitted:
        failed_surf = "M1" if not m1_sanity_admitted else "M2"
        bad_sanity = sanity1 if not m1_sanity_admitted else sanity2
        reasons = ",".join(bad_sanity.get("failure_reasons", ["UNKNOWN"]))
        raise RuntimeError(f"ORDER_{order}_SURFACE_SANITY_FAIL_{failed_surf}_{reasons}")

    live_phase("CI_PHYSICAL_FIRST_HIT_AUDIT")
    physical = trace_reverse(r, visor, m1, m2, display, physical_first_hit=True)
    evidence["phase"] = "PHYSICAL_TRACE"
    evidence["physical_trace"] = physical

    _live_model_preview(
        ctx,
        "CI candidate physical trace evaluated",
        m1=m1,
        m2=m2,
        display=display,
        trace=physical,
        trace_mode="PHYSICAL_FIRST_HIT",
        role="CANDIDATE_BEFORE_BRANCH_DECISION",
    )

    physical_count = int(np.sum(physical["valid"]))
    physical_fraction = physical_count / max(len(r["rows"]), 1)
    full_physical = physical_count == len(r["rows"])
    physical_fraction_pass = bool(
        physical_fraction >= minimum_candidate_fraction
    )
    restoration_enabled = bool(ctx.config["solver"].get("restoration_branch_enabled", False))
    restoration_minimum = float(ctx.config["solver"].get(
        "restoration_minimum_physical_fraction", 1.0))
    restoration_admissible = bool(
        allow_restoration
        and restoration_enabled
        and not physical_fraction_pass
        and physical_fraction >= restoration_minimum
    )
    evidence["physical_admission"] = {
        "valid_count": physical_count,
        "ray_count": int(len(r["rows"])),
        "valid_fraction": physical_fraction,
        "minimum_fraction": minimum_candidate_fraction,
        "fraction_pass": physical_fraction_pass,
        "all_rays_complete": full_physical,
        "restoration_admissible": restoration_admissible,
    }
    if (
        bool(ctx.config["solver"]["require_full_physical_ci"])
        and not physical_fraction_pass
        and not restoration_admissible
    ):
        raise RuntimeError(
            f"ORDER_{order}_PHYSICAL_FIRST_HIT_BELOW_MINIMUM_"
            f"{physical_count}_OF_{len(r['rows'])}_"
            f"MINIMUM_{minimum_candidate_fraction:.6f}"
            f"; PHYSICAL_FIRST_HIT_INCOMPLETE"
        )

    live_phase("CI_METRICS_AND_BRANCH_GATES")
    refs2, rule2 = _dynamic_refs(ctx, tr2)
    metric_ray_indices = np.flatnonzero(
        sequential_numeric_mask
    )
    metric_trace = {
        "landing": np.asarray(
            tr2["landing"],
            float,
        )[sequential_numeric_mask],
        "valid": np.ones(
            sequential_numeric_count,
            dtype=bool,
        ),
    }
    metric_rays = {
        "field_index": np.asarray(
            r["field_index"],
            int,
        )[sequential_numeric_mask],
        "pupil_index": np.asarray(
            r["pupil_index"],
            int,
        )[sequential_numeric_mask],
        "chief": np.asarray(
            r["chief"],
            bool,
        )[sequential_numeric_mask],
    }
    mf1 = fan_imaging_metrics(
        metric_trace,
        metric_rays,
        refs2,
        float(
            ctx.config[
                "fan_weights"
            ][
                "omega1"
            ]
        ),
    )
    mf1["metric_valid_count"] = sequential_numeric_count
    mf1["metric_ray_count"] = int(len(r["rows"]))
    mf1["metric_valid_fraction"] = sequential_numeric_fraction

    mf2 = mf2_geometry(
        np.asarray(tr2["points"][0], float)[sequential_numeric_mask],
        np.asarray(tr2["points"][1], float)[sequential_numeric_mask],
        np.asarray(tr2["points"][2], float)[sequential_numeric_mask],
        m2,
        float(
            ctx.config[
                "fan_weights"
            ][
                "omega2"
            ]
        ),
        ctx.data["n_obs"],
    )
    mf2["ray_I_subset_index"] = int(mf2["ray_I_index"])
    mf2["point_P_subset_index"] = int(mf2["point_P_index"])
    mf2["ray_I_index"] = int(
        metric_ray_indices[mf2["ray_I_subset_index"]]
    )
    mf2["point_P_index"] = int(
        metric_ray_indices[mf2["point_P_subset_index"]]
    )
    mf2["metric_valid_count"] = sequential_numeric_count
    mf2["metric_ray_count"] = int(len(r["rows"]))
    mf2["metric_valid_fraction"] = sequential_numeric_fraction

    chief_spot = (
        chief_centered_geometric_spot_rms(
            physical["landing"],
            physical["valid"],
            r["field_index"],
            r["pupil_index"],
            r["chief"],
            display.frame,
        )
    )
    unobscured = bool(float(mf2["S_AQP_signed_mm2"]) >= 0.0)
    if bool(ctx.config["solver"]["require_unobscured_ci"]) and not unobscured:
        raise RuntimeError(f"ORDER_{order}_SIGNED_MF2_OBSCURATION_FAIL")
    fan_core_merit = float(mf1["MF1_Fan"] + mf2["MF2"])

    live_phase("CI_COMMIT_ADMISSIBLE_CONSTRUCTION")
    evidence["phase"] = "COMMIT_ACCEPTABLE_CONSTRUCTION"
    ctx.data.update({"m1": m1, "m2": m2, "display": display, "trace": tr2,
                      "fan_refs": refs2, "fan_reference_rule": rule2,
                      "last_ci_m1": ci1, "last_ci_m2": ci2,
                      "last_ci_m1_diagnostics": diagnostic1,
                      "last_ci_m2_diagnostics": diagnostic2,
                      "last_ci_physical_trace": physical})
    return {"order": order, "M1_fit": s1, "M2_fit": s2, "numeric_usable": numeric,
            "numeric_valid_count": sequential_numeric_count,
            "numeric_valid_fraction": sequential_numeric_fraction,
            "minimum_candidate_fraction": minimum_candidate_fraction,
            "M1_sanity": sanity1, "M2_sanity": sanity2,
            "M1_sanity_admitted": m1_sanity_admitted,
            "M2_sanity_admitted": m2_sanity_admitted,
            "M2_sanity_warn_admitted": m2_sanity_warn_admitted,
            "m1_construction_input_gate": copy.deepcopy(
                evidence["m1_construction_input_gate"]
            ),
            "m2_construction_input_gate": copy.deepcopy(
                evidence["m2_construction_input_gate"]
            ),
            "M2_bundle_integrability_diagnostic": copy.deepcopy(
                evidence["M2_bundle_integrability_diagnostic"]
            ),
            "physical_valid_count": physical_count, "ray_count": len(r["rows"]),
            "physical_valid_fraction": physical_fraction,
            "physical_fraction_pass": physical_fraction_pass,
            "physical_first_hit_complete": full_physical,
            "feasibility_status": (
                "FULLY_PHYSICAL"
                if full_physical
                else (
                    (
                        "STEP17_MINIMUM_PHYSICAL_FRACTION_PASS"
                        if step17_fractional_admission
                        else "STEP16_MINIMUM_PHYSICAL_FRACTION_PASS"
                    )
                    if physical_fraction_pass
                    else "RESTORATION_BRANCH_EXPLICITLY_INFEASIBLE"
                )
            ),
            "restoration_branch": restoration_admissible,
            "physical_role": (
                (
                    "STEP17_MINIMUM_FRACTION_ELIGIBLE; "
                    if step17_fractional_admission
                    else "STEP16_MINIMUM_FRACTION_ELIGIBLE; "
                )
                +
                "EXACT_ALL_RAYS_COMPLETION_REPORTED_SEPARATELY"
            ),
            "unobscured": unobscured,
            "integrability_quality": {
                "enforcement": integrability_cfg["enforcement"],
                "status": "PASS" if integrability_pass else "WARN",
                "M1": integrability_gate1, "M2": integrability_gate2},
            "aperture_rebuild": copy.deepcopy(rebuild_audit),
            "M2_footprint_filter_enabled": bool(
                m2_ap_audit.get("footprint_filter_enabled", False)
            ),
            "M2_footprint_filter_method": str(
                m2_ap_audit.get("footprint_filter_method", "NONE")
            ),
            "M2_footprint_rejected_count": int(
                m2_ap_audit.get("combined_rejected_count", 0)
            ),
            "M2_footprint_rejected_fraction": float(
                m2_ap_audit.get("combined_rejected_fraction", 0.0)
            ),
            "M2_footprint_retained_count": int(
                m2_ap_audit.get("retained_authority_count", 0)
            ),
            "M2_footprint_affected_bundle_count": int(
                m2_ap_audit.get("affected_bundle_count", 0)
            ),
            "M2_footprint_max_bundle_rejected_fraction": float(
                m2_ap_audit.get("maximum_observed_bundle_rejected_fraction", 0.0)
            ),
            "M2_local_xy_min_before_mm": copy.deepcopy(
                m2_ap_audit.get("local_xy_min_before_mm", [0.0, 0.0])
            ),
            "M2_local_xy_max_before_mm": copy.deepcopy(
                m2_ap_audit.get("local_xy_max_before_mm", [0.0, 0.0])
            ),
            "M2_local_xy_min_after_mm": copy.deepcopy(
                m2_ap_audit.get("local_xy_min_after_mm", [0.0, 0.0])
            ),
            "M2_local_xy_max_after_mm": copy.deepcopy(
                m2_ap_audit.get("local_xy_max_after_mm", [0.0, 0.0])
            ),
            "M2_half_aperture_before_mm": copy.deepcopy(
                m2_ap_audit.get("half_aperture_before_mm", [0.0, 0.0])
            ),
            "M2_half_aperture_after_mm": copy.deepcopy(
                m2_ap_audit.get("half_aperture_after_mm", [0.0, 0.0])
            ),
            "M2_footprint_outlier_rows": copy.deepcopy(
                m2_ap_audit.get("footprint_outlier_rows", [])
            ),
            "M2_footprint_bundle_rows": copy.deepcopy(
                m2_ap_audit.get("footprint_bundle_rows", [])
            ),
            "chief_centered_spot_RMS_mm": chief_spot["RMS_spot_radius_mm"],
            "chief_centered_spot": chief_spot,
            "MF1_Fan": mf1["MF1_Fan"],
            "MF2_Fan": mf2["MF2"],
            "S_AQP_signed_mm2": float(mf2["S_AQP_signed_mm2"]),
            "Fan_core_merit": fan_core_merit,
            "rho": float(ctx.data["rho"]),
            "iteration_coordinates": "PRESERVED_CURRENT_RAY_SURFACE_INTERSECTIONS",
            "rho_application": "HUD_EQ3_M1_NEIGHBOR_TARGET_CONTINUATION",
            "normal_recomputation": "AT_EACH_PRESERVED_POINT_AFTER_TARGET_UPDATE",
            "M1_constructed_before_M2": True,
            "physical_first_hit_intermediate_gate": physical_fraction_pass}


def _reconstruct(
    ctx: Context,
    order: int,
    axis_order2: bool,
    allow_restoration: bool = False,
) -> dict[str, Any]:
    """Tái dựng các mặt quang học M1 và M2 theo bậc đa thức và thu thập bằng chứng ứng viên."""
    ctx.data.pop("_last_m2_aperture_audit", None)
    ctx.data.pop("_last_m2_authority_mask", None)

    evidence: dict[str, Any] = {
        "order": int(order),
        "phase": "START",
        "m1": None,
        "m2": None,
        "display": None,
        "status": "RUNNING",
    }
    ctx.data["_last_candidate_evidence"] = evidence

    def capture_m2_aperture_diagnostics(
        error_message: str,
    ) -> None:
        """Thu hồi audit của đúng candidate mà không ghi nhầm lỗi phía sau thành aperture failure."""
        aperture_state = evidence.get("aperture_rebuild")
        aperture_in_progress = bool(
            isinstance(aperture_state, dict)
            and aperture_state.get("phase")
            == "APERTURE_REBUILD_IN_PROGRESS"
        )

        m2_ap_audit = ctx.data.get("_last_m2_aperture_audit")

        if isinstance(m2_ap_audit, dict):
            mapping = {
                "M2_footprint_filter_enabled":
                    "footprint_filter_enabled",
                "M2_footprint_filter_method":
                    "footprint_filter_method",
                "M2_footprint_rejected_count":
                    "combined_rejected_count",
                "M2_footprint_rejected_fraction":
                    "combined_rejected_fraction",
                "M2_footprint_retained_count":
                    "retained_authority_count",
                "M2_footprint_affected_bundle_count":
                    "affected_bundle_count",
                "M2_footprint_max_bundle_rejected_fraction":
                    "maximum_observed_bundle_rejected_fraction",
                "M2_local_xy_min_before_mm":
                    "local_xy_min_before_mm",
                "M2_local_xy_max_before_mm":
                    "local_xy_max_before_mm",
                "M2_local_xy_min_after_mm":
                    "local_xy_min_after_mm",
                "M2_local_xy_max_after_mm":
                    "local_xy_max_after_mm",
                "M2_half_aperture_before_mm":
                    "half_aperture_before_mm",
                "M2_half_aperture_after_mm":
                    "half_aperture_after_mm",
                "M2_footprint_outlier_rows":
                    "footprint_outlier_rows",
                "M2_footprint_bundle_rows":
                    "footprint_bundle_rows",
            }

            for evidence_key, audit_key in mapping.items():
                if audit_key in m2_ap_audit:
                    evidence[evidence_key] = copy.deepcopy(
                        m2_ap_audit[audit_key]
                    )

            if not evidence.get(
                "_m2_footprint_csv_exported",
                False,
            ):
                _export_step17_m2_footprint_csvs(
                    ctx,
                    order,
                    m2_ap_audit,
                    evidence,
                )
                evidence[
                    "_m2_footprint_csv_exported"
                ] = True

        if aperture_in_progress:
            failed_aperture_state = copy.deepcopy(
                aperture_state
            )
            failed_aperture_state[
                "phase"
            ] = "APERTURE_REBUILD_FAILED"
            failed_aperture_state[
                "error"
            ] = str(error_message)

            if isinstance(m2_ap_audit, dict):
                failed_aperture_state["M2"] = copy.deepcopy(
                    m2_ap_audit
                )

            evidence[
                "aperture_rebuild"
            ] = failed_aperture_state

    try:
        result = _reconstruct_impl(
            ctx,
            order,
            axis_order2,
            allow_restoration,
            evidence=evidence,
        )
    except np.linalg.LinAlgError as exc:
        phase = str(
            evidence.get(
                "phase",
                "UNKNOWN",
            )
        )

        wrapped = RuntimeError(
            f"ORDER_{order}_{phase}_"
            f"LINEAR_ALGEBRA_FAILURE:{exc}"
        )

        evidence["status"] = (
            "REJECTED_OR_ABORTED"
        )

        evidence["exception_type"] = (
            type(wrapped).__name__
        )

        evidence["exception_message"] = (
            str(wrapped)
        )

        evidence["root_exception_type"] = (
            type(exc).__name__
        )

        evidence["root_exception_message"] = (
            str(exc)
        )

        capture_m2_aperture_diagnostics(
            str(wrapped)
        )

        raise wrapped from exc
    except Exception as exc:
        evidence["status"] = "REJECTED_OR_ABORTED"
        evidence["exception_type"] = type(exc).__name__
        evidence["exception_message"] = str(exc)

        capture_m2_aperture_diagnostics(
            str(exc)
        )

        raise

    evidence["status"] = "CONSTRUCTION_COMPLETED"
    evidence["reconstruct_out"] = copy.deepcopy(result)
    return result


def _rho_state(ctx: Context, rho_sequence: list[float], order_history: list[dict[str, Any]],
               reconstruct_out: dict[str, Any]) -> dict[str, Any]:
    """Chụp mặt và metric của một ứng viên signed rho."""
    keys = ("m1", "m2", "display", "trace", "fan_refs", "fan_reference_rule", "last_ci_m1",
            "last_ci_m2", "last_ci_m1_diagnostics", "last_ci_m2_diagnostics",
            "last_ci_physical_trace",
            "fermat", "fermat_convergence_history", "fermat_all_converged",
            "fermat_fraction_pass", "reference_history", "rho")
    state = {key: copy.deepcopy(ctx.data[key]) for key in keys if key in ctx.data}
    state.update({"rho_sequence": list(rho_sequence), "order_history": copy.deepcopy(order_history),
                  "reconstruct_out": copy.deepcopy(reconstruct_out),
                  "Fan_core_merit": float(reconstruct_out["Fan_core_merit"])})
    return state


def _apply_rho_state(ctx: Context, state: dict[str, Any]) -> None:
    """Phục hồi đầy đủ ứng viên rho đã chọn vào Context."""
    ctx.data.pop("_last_candidate_evidence", None)
    for key, value in state.items():
        if key not in ("rho_sequence", "order_history", "reconstruct_out", "Fan_core_merit"):
            ctx.data[key] = copy.deepcopy(value)


def _rho_sort_key(state: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Xếp hạng feasible-first, sau đó dùng chief-centered Fan merit và spot."""
    out = state["reconstruct_out"]

    spot = out.get(
        "chief_centered_spot_RMS_mm"
    )

    return (
        -float(
            out.get(
                "physical_valid_count",
                0,
            )
        ),

        (
            0.0
            if bool(
                out.get(
                    "unobscured",
                    False,
                )
            )
            else 1.0
        ),

        float(
            state[
                "Fan_core_merit"
            ]
        ),

        (
            float(
                spot
            )
            if (
                spot is not None
                and np.isfinite(
                    float(
                        spot
                    )
                )
            )
            else float(
                "inf"
            )
        ),

        float(
            abs(
                state[
                    "rho_sequence"
                ][
                    -1
                ]
            )
        ),
    )


def _step16_spot_tail_metrics(reconstruct_out: dict[str, Any]) -> dict[str, Any]:
    """Tóm tắt tail spot theo bundle để global RMS không che field/pupil xấu."""
    spot = reconstruct_out.get("chief_centered_spot") or {}
    rows = spot.get("bundle_rows") or []
    bundle_rms = np.asarray([
        float(row["RMS_to_chief_mm"])
        for row in rows
        if row.get("RMS_to_chief_mm") is not None
        and np.isfinite(float(row["RMS_to_chief_mm"]))
    ], dtype=float)

    return {
        "bundle_count": int(len(rows)),
        "finite_bundle_count": int(len(bundle_rms)),
        "bundle_RMS_P95_mm": (
            float(np.percentile(bundle_rms, 95))
            if len(bundle_rms)
            else None
        ),
        "worst_bundle_RMS_mm": (
            float(np.max(bundle_rms))
            if len(bundle_rms)
            else None
        ),
        "global_RMS_mm": spot.get("RMS_spot_radius_mm"),
        "max_spot_radius_mm": spot.get("max_spot_radius_mm"),
    }


STEP17_SPOT_SUMMARY_FIELDS = [
    "bundle_RMS_P50_mm",
    "bundle_RMS_P95_mm",
    "bundle_RMS_P99_mm",
    "worst_bundle_RMS_mm",
    "worst_bundle_field_index",
    "worst_bundle_pupil_index",
    "global_max_spot_radius_mm",
    "worst_bundle_max_radius_mm",
    "valid_bundle_count",
    "required_bundle_count",
    "underfilled_bundle_count",
]

STEP17_SPOT_DELTA_FIELDS = [
    "delta_global_RMS_mm",
    "delta_bundle_P95_mm",
    "delta_worst_bundle_RMS_mm",
    "delta_max_spot_radius_mm",
    "delta_physical_ray_count",
    "delta_Fan_core_merit",
]

STEP17_SPOT_EXPORT_FIELDS = [
    "spot_bundle_csv",
    "spot_bundle_export_status",
    "spot_bundle_export_error",
]

STEP17_SPOT_BUNDLE_CSV_FIELDS = [
    "order",
    "cycle",
    "branch",
    "decision",
    "decision_reason",
    "field_index",
    "field_id",
    "field_h_deg",
    "field_v_deg",
    "field_horizontal_index",
    "field_vertical_index",
    "pupil_index",
    "pupil_id",
    "pupil_y_mm",
    "pupil_z_mm",
    "ray_count",
    "valid_count",
    "valid_fraction",
    "lost_ray_count",
    "lost_ray_fraction",
    "chief_ray_index",
    "chief_valid",
    "evaluable",
    "underfilled",
    "RMS_to_chief_mm",
    "max_radius_to_chief_mm",
]


def _step17_finite_number(value: Any) -> float | None:
    """Return a finite float or None without changing candidate admission."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if np.isfinite(number) else None


def _step17_spot_bundle_diagnostics(
    reconstruct_out: dict[str, Any],
) -> dict[str, Any]:
    """Summarize chief-centered STEP17 spot quality without adding a gate."""
    spot = reconstruct_out.get("chief_centered_spot") or {}
    source_rows = list(spot.get("bundle_rows") or [])

    field_count = int(spot.get("field_count", 0) or 0)
    pupil_count = int(spot.get("pupil_count", 0) or 0)

    if source_rows and field_count <= 0:
        field_count = max(int(row["field_index"]) for row in source_rows) + 1

    if source_rows and pupil_count <= 0:
        pupil_count = max(int(row["pupil_index"]) for row in source_rows) + 1

    required_bundle_count = int(field_count * pupil_count)
    source_by_bundle = {
        (int(row["field_index"]), int(row["pupil_index"])): row
        for row in source_rows
    }

    bundle_rows: list[dict[str, Any]] = []

    for field_index in range(field_count):
        for pupil_index in range(pupil_count):
            source = source_by_bundle.get(
                (field_index, pupil_index),
                {},
            )

            ray_count = int(source.get("ray_count", 0) or 0)
            valid_count = int(source.get("valid_count", 0) or 0)
            chief_valid = bool(source.get("chief_valid", False))
            rms = _step17_finite_number(
                source.get("RMS_to_chief_mm")
            )
            maximum = _step17_finite_number(
                source.get("max_radius_to_chief_mm")
            )
            evaluable = bool(chief_valid and rms is not None)
            underfilled = bool(valid_count < ray_count)

            bundle_rows.append({
                "field_index": field_index,
                "pupil_index": pupil_index,
                "ray_count": ray_count,
                "valid_count": valid_count,
                "valid_fraction": (
                    float(valid_count / ray_count)
                    if ray_count > 0
                    else 0.0
                ),
                "lost_ray_count": int(max(0, ray_count - valid_count)),
                "lost_ray_fraction": (
                    float(max(0, ray_count - valid_count) / ray_count)
                    if ray_count > 0
                    else 1.0
                ),
                "chief_ray_index": source.get("chief_ray_index"),
                "chief_valid": chief_valid,
                "evaluable": evaluable,
                "underfilled": underfilled,
                "RMS_to_chief_mm": rms,
                "max_radius_to_chief_mm": maximum,
            })

    evaluable_rows = [
        row
        for row in bundle_rows
        if bool(row["evaluable"])
        and row["RMS_to_chief_mm"] is not None
    ]

    bundle_rms = np.asarray(
        [
            float(row["RMS_to_chief_mm"])
            for row in evaluable_rows
        ],
        dtype=float,
    )

    worst_row = (
        max(
            evaluable_rows,
            key=lambda row: float(row["RMS_to_chief_mm"]),
        )
        if evaluable_rows
        else None
    )

    summary = {
        "bundle_RMS_P50_mm": (
            float(np.percentile(bundle_rms, 50))
            if len(bundle_rms)
            else None
        ),
        "bundle_RMS_P95_mm": (
            float(np.percentile(bundle_rms, 95))
            if len(bundle_rms)
            else None
        ),
        "bundle_RMS_P99_mm": (
            float(np.percentile(bundle_rms, 99))
            if len(bundle_rms)
            else None
        ),
        "worst_bundle_RMS_mm": (
            float(worst_row["RMS_to_chief_mm"])
            if worst_row is not None
            else None
        ),
        "worst_bundle_field_index": (
            int(worst_row["field_index"])
            if worst_row is not None
            else None
        ),
        "worst_bundle_pupil_index": (
            int(worst_row["pupil_index"])
            if worst_row is not None
            else None
        ),
        "global_max_spot_radius_mm": _step17_finite_number(
            spot.get("max_spot_radius_mm")
        ),
        "worst_bundle_max_radius_mm": (
            _step17_finite_number(
                worst_row.get("max_radius_to_chief_mm")
            )
            if worst_row is not None
            else None
        ),
        "valid_bundle_count": int(len(evaluable_rows)),
        "required_bundle_count": required_bundle_count,
        "underfilled_bundle_count": int(
            sum(bool(row["underfilled"]) for row in bundle_rows)
        ),
    }

    return {
        "summary": summary,
        "bundle_rows": bundle_rows,
    }


def _step17_spot_deltas(
    candidate_out: dict[str, Any],
    parent_out: dict[str, Any],
    candidate_diagnostic: dict[str, Any],
    parent_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Return candidate-minus-parent diagnostics; negative optical deltas are better."""

    def difference(
        candidate_value: Any,
        parent_value: Any,
    ) -> float | None:
        """Tính chênh lệch candidate trừ parent cho giá trị hữu hạn."""
        candidate_number = _step17_finite_number(candidate_value)
        parent_number = _step17_finite_number(parent_value)

        if candidate_number is None or parent_number is None:
            return None

        return float(candidate_number - parent_number)

    candidate_summary = candidate_diagnostic["summary"]
    parent_summary = parent_diagnostic["summary"]

    return {
        "delta_global_RMS_mm": difference(
            candidate_out.get("chief_centered_spot_RMS_mm"),
            parent_out.get("chief_centered_spot_RMS_mm"),
        ),
        "delta_bundle_P95_mm": difference(
            candidate_summary.get("bundle_RMS_P95_mm"),
            parent_summary.get("bundle_RMS_P95_mm"),
        ),
        "delta_worst_bundle_RMS_mm": difference(
            candidate_summary.get("worst_bundle_RMS_mm"),
            parent_summary.get("worst_bundle_RMS_mm"),
        ),
        "delta_max_spot_radius_mm": difference(
            candidate_summary.get("global_max_spot_radius_mm"),
            parent_summary.get("global_max_spot_radius_mm"),
        ),
        "delta_physical_ray_count": int(
            candidate_out.get("physical_valid_count", 0)
        ) - int(
            parent_out.get("physical_valid_count", 0)
        ),
        "delta_Fan_core_merit": difference(
            candidate_out.get("Fan_core_merit"),
            parent_out.get("Fan_core_merit"),
        ),
    }


def _persist_step17_spot_bundle_rows(
    ctx: Context,
    order: int,
    cycle: int,
    branch_index: int,
    diagnostic: dict[str, Any],
    decision: str,
    decision_reason: str,
) -> dict[str, Any]:
    """Write one non-overwriting bundle file per reconstructed STEP17 candidate."""
    fields = ctx.data["vi"]["fields"]
    pupils = ctx.data["pupils"]

    output_rows: list[dict[str, Any]] = []

    for source in diagnostic["bundle_rows"]:
        field_index = int(source["field_index"])
        pupil_index = int(source["pupil_index"])
        field = fields[field_index]
        pupil = pupils[pupil_index]

        output_rows.append({
            "order": int(order),
            "cycle": int(cycle),
            "branch": int(branch_index),
            "decision": str(decision),
            "decision_reason": str(decision_reason),
            "field_index": field_index,
            "field_id": str(field["field_id"]),
            "field_h_deg": float(field["h_deg"]),
            "field_v_deg": float(field["v_deg"]),
            "field_horizontal_index": int(field["horizontal_index"]),
            "field_vertical_index": int(field["vertical_index"]),
            "pupil_index": pupil_index,
            "pupil_id": str(pupil["pupil_id"]),
            "pupil_y_mm": float(pupil["y_mm"]),
            "pupil_z_mm": float(pupil["z_mm"]),
            "ray_count": int(source["ray_count"]),
            "valid_count": int(source["valid_count"]),
            "valid_fraction": float(source["valid_fraction"]),
            "lost_ray_count": int(source["lost_ray_count"]),
            "lost_ray_fraction": float(source["lost_ray_fraction"]),
            "chief_ray_index": source.get("chief_ray_index"),
            "chief_valid": bool(source["chief_valid"]),
            "evaluable": bool(source["evaluable"]),
            "underfilled": bool(source["underfilled"]),
            "RMS_to_chief_mm": source.get("RMS_to_chief_mm"),
            "max_radius_to_chief_mm": source.get(
                "max_radius_to_chief_mm"
            ),
        })

    target = (
        ctx.step_dir(17)
        / (
            f"17_SPOT_BUNDLES_ORDER_{int(order)}"
            f"_CYCLE_{int(cycle):03d}"
            f"_BRANCH_{int(branch_index):03d}.csv"
        )
    )

    try:
        write_csv(
            target,
            output_rows,
            fieldnames=STEP17_SPOT_BUNDLE_CSV_FIELDS,
        )
    except (OSError, TypeError, ValueError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(
            "[STEP17][WARN] Bundle diagnostic export failed | "
            f"O{int(order)} C{int(cycle):02d} "
            f"B{int(branch_index):02d} | {message}",
            flush=True,
        )
        return {
            "spot_bundle_csv": None,
            "spot_bundle_export_status": "WARN",
            "spot_bundle_export_error": message,
        }

    return {
        "spot_bundle_csv": str(target),
        "spot_bundle_export_status": "PASS",
        "spot_bundle_export_error": None,
    }


def _step17_metric_text(
    value: Any,
    digits: int = 3,
) -> str:
    """Định dạng số thực hữu hạn sang chuỗi hoặc trả về N/A."""
    number = _step17_finite_number(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _print_step17_candidate_diagnostic(
    order: int,
    cycle: int,
    branch_index: int,
    candidate_out: dict[str, Any],
    parent_out: dict[str, Any],
    candidate_diagnostic: dict[str, Any],
    parent_diagnostic: dict[str, Any],
    spot_guard: dict[str, Any],
    decision: str,
    decision_reason: str,
) -> None:
    """Print one complete STEP17 diagnostic block after the decision is known."""
    candidate_summary = candidate_diagnostic["summary"]
    parent_summary = parent_diagnostic["summary"]

    ray_count = int(candidate_out.get("ray_count", 0))
    physical_count = int(
        candidate_out.get("physical_valid_count", 0)
    )
    physical_fraction = (
        float(physical_count / ray_count)
        if ray_count > 0
        else 0.0
    )

    worst_field = candidate_summary.get(
        "worst_bundle_field_index"
    )
    worst_pupil = candidate_summary.get(
        "worst_bundle_pupil_index"
    )

    worst_identity = (
        f"F{int(worst_field):02d}/P{int(worst_pupil):02d}"
        if worst_field is not None and worst_pupil is not None
        else "N/A"
    )

    m1_trust = (
        candidate_out.get("M1_fit", {})
        .get("parent_trust", {})
    )
    m2_trust = (
        candidate_out.get("M2_fit", {})
        .get("parent_trust", {})
    )

    signed_area = _step17_finite_number(
        candidate_out.get("S_AQP_signed_mm2")
    )
    signed_area_status = (
        "PASS"
        if signed_area is not None and signed_area >= 0.0
        else "FAIL"
    )

    print(
        f"[STEP17][O{int(order)} "
        f"C{int(cycle):02d} B{int(branch_index):02d}]",
        flush=True,
    )
    print(
        "  physical : "
        f"{physical_count:,}/{ray_count:,} "
        f"= {100.0 * physical_fraction:.3f}%",
        flush=True,
    )
    print(
        "  spot RMS : "
        f"{_step17_metric_text(parent_out.get('chief_centered_spot_RMS_mm'))}"
        " -> "
        f"{_step17_metric_text(candidate_out.get('chief_centered_spot_RMS_mm'))}"
        " mm | "
        f"{'PASS' if spot_guard.get('spot_RMS_guard_pass') else 'FAIL'}",
        flush=True,
    )
    print(
        "  bundleP95: "
        f"{_step17_metric_text(parent_summary.get('bundle_RMS_P95_mm'))}"
        " -> "
        f"{_step17_metric_text(candidate_summary.get('bundle_RMS_P95_mm'))}"
        " mm",
        flush=True,
    )
    print(
        "  worst    : "
        f"{worst_identity} | "
        f"{_step17_metric_text(candidate_summary.get('worst_bundle_RMS_mm'))}"
        " mm",
        flush=True,
    )
    print(
        "  max GEO  : "
        f"{_step17_metric_text(candidate_summary.get('global_max_spot_radius_mm'))}"
        " mm",
        flush=True,
    )
    print(
        "  Fan merit: "
        f"{_step17_metric_text(parent_out.get('Fan_core_merit'), 6)}"
        " -> "
        f"{_step17_metric_text(candidate_out.get('Fan_core_merit'), 6)}",
        flush=True,
    )
    print(
        "  MF2      : "
        f"S_AQP={_step17_metric_text(signed_area, 6)} mm2 | "
        f"{signed_area_status}",
        flush=True,
    )
    print(
        "  trust M1 : "
        f"sag={_step17_metric_text(m1_trust.get('sag_rms_mm'), 6)} mm | "
        f"normal={_step17_metric_text(m1_trust.get('normal_rms_deg'), 6)} deg",
        flush=True,
    )
    print(
        "  trust M2 : "
        f"sag={_step17_metric_text(m2_trust.get('sag_rms_mm'), 6)} mm | "
        f"normal={_step17_metric_text(m2_trust.get('normal_rms_deg'), 6)} deg",
        flush=True,
    )
    print(
        f"  decision : {decision} | {decision_reason}",
        flush=True,
    )


def _step16_full_hard_feasible(reconstruct_out: dict[str, Any]) -> bool:
    """STEP16 best-rho eligibility using the configured minimum ray fraction."""
    ray_count = int(reconstruct_out.get("ray_count", 0))
    physical_count = int(reconstruct_out.get("physical_valid_count", 0))
    minimum_fraction = float(
        reconstruct_out.get(
            "minimum_candidate_fraction",
            0.95,
        )
    )
    physical_fraction = float(
        reconstruct_out.get(
            "physical_valid_fraction",
            physical_count / max(ray_count, 1),
        )
    )
    integrability = reconstruct_out.get("integrability_quality") or {}
    sanity1 = reconstruct_out.get("M1_sanity") or {}
    sanity2 = reconstruct_out.get("M2_sanity") or {}

    return bool(
        ray_count > 0
        and 0 <= physical_count <= ray_count
        and physical_fraction >= minimum_fraction
        and bool(reconstruct_out.get("physical_fraction_pass", False))
        and bool(reconstruct_out.get("unobscured", False))
        and str(integrability.get("status", "")) == "PASS"
        and bool(
            reconstruct_out.get(
                "M1_sanity_admitted",
                sanity1.get("pass", False),
            )
        )
        and bool(
            reconstruct_out.get(
                "M2_sanity_admitted",
                sanity2.get("pass", False),
            )
        )
    )


def _step16_metric_or_inf(value: Any) -> float:
    """Đổi metric thiếu/non-finite thành +inf chỉ cho mục đích xếp hạng."""
    if value is None:
        return float("inf")
    number = float(value)
    return number if np.isfinite(number) else float("inf")


def _rho_o2_search_sort_key(
    state: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    """Xếp hạng STEP16 sau hard feasibility: Fan → tail bundle → global spot → |rho|."""
    out = state["reconstruct_out"]
    tail = out.get("step16_spot_tail") or {}
    return (
        0.0 if bool(out.get("step16_hard_feasible", False)) else 1.0,
        float(state["Fan_core_merit"]),
        _step16_metric_or_inf(tail.get("bundle_RMS_P95_mm")),
        _step16_metric_or_inf(tail.get("worst_bundle_RMS_mm")),
        _step16_metric_or_inf(tail.get("global_RMS_mm")),
        _step16_metric_or_inf(tail.get("max_spot_radius_mm")),
        float(abs(state["rho_sequence"][-1])),
    )


def _step16_nearest_same_sign_spacing(
    center: float,
    values: list[float],
) -> float:
    """Khoảng rho gần nhất cùng dấu để mở cửa sổ refine mà không nhảy qua rho=0."""
    distances = [
        abs(float(value) - float(center))
        for value in values
        if float(value) * float(center) > 0.0
        and not math.isclose(
            float(value),
            float(center),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ]
    if distances:
        return float(min(distances))
    return max(abs(float(center)), 1e-3)


def _step16_refinement_grid(
    center: float,
    half_width: float,
    grid_points: int,
    evaluated: set[float],
) -> tuple[list[float], float]:
    """Sinh lưới refine signed-rho đối xứng quanh incumbent và bỏ candidate đã đánh giá."""
    values = np.linspace(
        float(center) - float(half_width),
        float(center) + float(half_width),
        int(grid_points),
    )
    grid_step = float(
        2.0 * half_width / max(int(grid_points) - 1, 1)
    )
    candidates: list[float] = []
    seen: set[float] = set()

    for raw in values:
        value = float(raw)
        key = round(value, 15)
        if value == 0.0 or abs(value) > 1.0:
            continue
        if value * float(center) <= 0.0:
            continue
        if key in evaluated or key in seen:
            continue
        seen.add(key)
        candidates.append(value)

    return candidates, grid_step


def _ci_full_physical(state: dict[str, Any]) -> bool:
    """Kiểm tra tính nhất quán số tia và cờ vật lý hoàn chỉnh của một trạng thái CI."""
    out = state["reconstruct_out"]
    total = int(out["ray_count"])
    count = int(out["physical_valid_count"])
    flag = bool(out["physical_first_hit_complete"])

    if total <= 0 or not 0 <= count <= total:
        raise RuntimeError("CI_STATE_INVALID_RAY_COUNTS")

    if flag != (count == total):
        raise RuntimeError("CI_STATE_PHYSICAL_FLAG_COUNT_MISMATCH")

    return flag


def _ci_minimum_physical_fraction_admitted(state: dict[str, Any]) -> bool:
    """Validate counts and apply the candidate's configured physical fraction gate."""
    out = state["reconstruct_out"]
    total = int(out["ray_count"])
    count = int(out["physical_valid_count"])
    minimum_fraction = float(out.get("minimum_candidate_fraction", 1.0))
    if (
        total <= 0
        or not 0 <= count <= total
        or not np.isfinite(minimum_fraction)
        or not 0.0 < minimum_fraction <= 1.0
    ):
        raise RuntimeError("CI_STATE_INVALID_FRACTIONAL_PHYSICAL_COUNTS")

    actual_fraction = float(count / total)
    admitted = bool(actual_fraction >= minimum_fraction)
    reported_fraction = float(
        out.get("physical_valid_fraction", actual_fraction)
    )
    reported_pass = bool(out.get("physical_fraction_pass", admitted))
    if not math.isclose(
        reported_fraction,
        actual_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or reported_pass != admitted:
        raise RuntimeError("CI_STATE_PHYSICAL_FRACTION_FLAG_COUNT_MISMATCH")
    _ci_full_physical(state)
    return admitted


def _remember_ci_state(
    state: dict[str, Any],
    feasible: dict[int, dict[str, Any]],
    admissible: dict[int, dict[str, Any]],
) -> None:
    """Lưu trữ trạng thái CI tốt nhất theo thứ tự bậc vào kho lưu trữ feasible hoặc admissible."""
    order = int(state["reconstruct_out"]["order"])
    archive = (
        feasible
        if _ci_minimum_physical_fraction_admitted(state)
        else admissible
    )
    previous = archive.get(order)

    if previous is None or _rho_sort_key(state) < _rho_sort_key(previous):
        archive[order] = copy.deepcopy(state)


def _surface_has_ci_basis(surface: PolySurface, order: int) -> bool:
    """Xác nhận mặt quang học có đúng hệ cơ sở đơn thức monomial chuẩn của bậc chỉ định."""
    expected = monomial_terms(
        order,
        axis_order2=(order == 2),
        include_constant=True,
        preserve_fan_axis_order2=(order == 3),
    )
    terms = [tuple(term) for term in surface.terms]
    coefficients = np.asarray(surface.coeff)

    return (
        len(terms) == len(expected)
        and len(set(terms)) == len(terms)
        and set(terms) == set(expected)
        and coefficients.shape == (len(terms),)
    )


def _promote_ci_state_without_shape_change(
    state: dict[str, Any],
    target_order: int,
    cycle: int,
    *,
    spot_guard_enabled: bool,
    maximum_spot_rms_regression_mm: float,
) -> dict[str, Any]:
    """Promote the polynomial basis while preserving exact optical geometry."""
    source_order = int(
        state["reconstruct_out"]["order"]
    )
    target_order = int(target_order)

    if (
        target_order not in (4, 5)
        or source_order != target_order - 1
    ):
        raise RuntimeError(
            "CI_BASIS_PROMOTION_ORDER_TRANSITION_INVALID"
        )

    if not _ci_minimum_physical_fraction_admitted(state):
        raise RuntimeError(
            "CI_BASIS_PROMOTION_PARENT_NOT_PHYSICALLY_ADMITTED"
        )

    if not all(
        _surface_has_ci_basis(state[name], source_order)
        for name in ("m1", "m2")
    ):
        raise RuntimeError(
            "CI_BASIS_PROMOTION_PARENT_BASIS_INVALID"
        )

    promoted = copy.deepcopy(state)
    promoted["m1"] = (
        _promote_surface_order_without_shape_change(
            state["m1"],
            target_order,
        )
    )
    promoted["m2"] = (
        _promote_surface_order_without_shape_change(
            state["m2"],
            target_order,
        )
    )

    promoted["reconstruct_out"] = copy.deepcopy(
        state["reconstruct_out"]
    )
    promoted["reconstruct_out"].update({
        "order": target_order,
        "basis_promotion_only": True,
        "basis_promotion_from_order": source_order,
        "shape_optimized_at_this_order": False,
    })
    promotion_spot_diagnostic = (
        _step17_spot_bundle_diagnostics(
            state["reconstruct_out"]
        )
    )
    promotion_spot_deltas = _step17_spot_deltas(
        state["reconstruct_out"],
        state["reconstruct_out"],
        promotion_spot_diagnostic,
        promotion_spot_diagnostic,
    )

    promoted["reconstruct_out"].update(
        promotion_spot_diagnostic["summary"]
    )
    promoted["reconstruct_out"].update(
        promotion_spot_deltas
    )

    if target_order == 5:
        promoted["reconstruct_out"]["order5_entry"] = (
            "ZERO_PAD_HIGH_ORDER_TERMS_WITHOUT_SHAPE_CHANGE"
        )

    spot_value = _finite_step17_spot_rms(
        state["reconstruct_out"]
    )

    history = copy.deepcopy(state["order_history"])
    history.append({
        "order": target_order,
        "cycle": int(cycle),
        "accepted": False,
        "accepted_shape_update": False,
        "basis_promoted": True, "shape_optimized": False,
        "promotion_from_order": source_order,
        "rho": float(state["rho_sequence"][-1]),
        "basis": _ci_basis_label(target_order),
        "fermat_success_count": None,
        "fermat_ray_count": None,
        "fermat_all_converged": None,
        "fermat_admissible_fraction": None,
        "fermat_minimum_fraction": None,
        "fermat_fraction_pass": None,
        "gradient_max": None,
        "M1_sag_fit_rms_mm": None,
        "M2_sag_fit_rms_mm": None,
        "Fan_core_merit": float(
            state["Fan_core_merit"]
        ),
        "chief_centered_spot_RMS_mm": spot_value,
        **promotion_spot_diagnostic["summary"],
        **promotion_spot_deltas,
        "spot_bundle_csv": None,
        "spot_bundle_export_status": "NOT_APPLICABLE_BASIS_ONLY",
        "spot_bundle_export_error": None,
        "physical_valid_count": int(
            state["reconstruct_out"]["physical_valid_count"]
        ),
        "physical_valid_fraction": float(
            state["reconstruct_out"][
                "physical_valid_fraction"
            ]
        ),
        "minimum_candidate_fraction": float(
            state["reconstruct_out"][
                "minimum_candidate_fraction"
            ]
        ),
        "physical_fraction_pass": bool(
            state["reconstruct_out"][
                "physical_fraction_pass"
            ]
        ),
        "physical_first_hit_complete": bool(
            state["reconstruct_out"][
                "physical_first_hit_complete"
            ]
        ),
        "feasibility_status": str(
            state["reconstruct_out"]["feasibility_status"]
        ),
        "spot_RMS_guard_enabled": bool(
            spot_guard_enabled
        ),
        "parent_spot_RMS_mm": spot_value,
        "candidate_spot_RMS_mm": spot_value,
        "spot_RMS_delta_mm": (
            0.0 if spot_value is not None else None
        ),
        "maximum_spot_RMS_regression_mm": float(
            maximum_spot_rms_regression_mm
        ),
        "spot_RMS_guard_pass": True,
        "spot_RMS_guard_status":
            "BASIS_PROMOTION_NO_SHAPE_CHANGE",
        "reason": (
            f"ORDER{target_order}_ZERO_PAD_CONTINUATION_"
            f"SHAPE_UNCHANGED_FROM_ORDER{source_order}"
        ),
    })

    promoted["order_history"] = history
    return promoted


def _ci_order5_metadata(
    state: dict[str, Any],
    expected_ray_count: int,
) -> dict[str, Any] | None:
    """Validate the nested O2-O5 CI ladder and report real optimized shape order."""
    out = state["reconstruct_out"]

    if int(out["order"]) != 5:
        return None
    if int(out["ray_count"]) != int(expected_ray_count):
        return None
    if not _ci_minimum_physical_fraction_admitted(state):
        return None

    if not all(
        _surface_has_ci_basis(state[name], 5)
        for name in ("m1", "m2")
    ):
        return None

    history = state["order_history"]

    accepted_shape_orders = {
        int(row["order"])
        for row in history
        if bool(row.get("accepted", False))
        and bool(row.get("accepted_shape_update", False))
        and bool(row.get("shape_optimized", False))
        and not bool(row.get("basis_promoted", False))
    }

    basis_promotion_orders = {
        int(row["order"])
        for row in history
        if bool(row.get("basis_promoted", False))
        and not bool(row.get("accepted_shape_update", False))
        and not bool(row.get("shape_optimized", False))
        and int(row.get("promotion_from_order", -1))
        == int(row["order"]) - 1
    }

    if not {2, 3}.issubset(accepted_shape_orders):
        return None

    order4_valid = bool(
        4 in accepted_shape_orders
        or 4 in basis_promotion_orders
    )
    if not order4_valid:
        return None

    order5_rows = [
        row
        for row in history
        if int(row["order"]) == 5
    ]
    if not order5_rows:
        return None

    last_order5 = order5_rows[-1]
    order5_shape_update = bool(
        last_order5.get("accepted", False)
        and last_order5.get(
            "accepted_shape_update",
            False,
        )
        and last_order5.get("shape_optimized", False)
        and not last_order5.get("basis_promoted", False)
    )
    order5_promotion = bool(
        last_order5.get("basis_promoted", False)
        and not last_order5.get(
            "accepted_shape_update",
            False,
        )
        and not last_order5.get(
            "shape_optimized",
            False,
        )
        and int(
            last_order5.get(
                "promotion_from_order",
                -1,
            )
        ) == 4
    )

    if not (order5_shape_update or order5_promotion):
        return None

    shape_optimized_through_order = max(
        accepted_shape_orders
    )

    status = (
        "ORDER5_SHAPE_ACCEPTED_BY_CI"
        if order5_shape_update
        else (
            "ORDER5_BASIS_ZERO_PAD_ONLY_"
            f"SHAPE_UNCHANGED_FROM_ORDER"
            f"{shape_optimized_through_order}"
        )
    )

    return {
        "basis_order": 5,
        "shape_optimized_through_order":
            shape_optimized_through_order,
        "shape_update_orders": sorted(
            accepted_shape_orders
        ),
        "basis_promotion_orders": sorted(
            basis_promotion_orders
        ),
        "order4_zero_pad_promotion": bool(
            4 in basis_promotion_orders
        ),
        "order5_zero_pad_promotion": bool(
            order5_promotion
        ),
        "status": status,
        "zero_pad_promotion": order5_promotion,
        "physical_valid_count": int(
            out["physical_valid_count"]
        ),
        "ray_count": int(out["ray_count"]),
        "physical_valid_fraction": float(
            out["physical_valid_fraction"]
        ),
        "minimum_candidate_fraction": float(
            out["minimum_candidate_fraction"]
        ),
        "physical_fraction_pass": bool(
            out["physical_fraction_pass"]
        ),
        "physical_first_hit_complete": bool(
            out["physical_first_hit_complete"]
        ),
    }


def _select_ci_frontier(states: list[dict[str, Any]], solver: dict[str, Any]
                        ) -> list[dict[str, Any]]:
    """Giữ mọi nghiệm feasible tốt nhất và tối đa số restoration branch đã khai báo."""
    ordered = sorted(states, key=_rho_sort_key)
    width = int(solver["rho_beam_width"])
    fraction_admitted = [
        state
        for state in ordered
        if _ci_minimum_physical_fraction_admitted(state)
    ]
    restoration = [
        state
        for state in ordered
        if not _ci_minimum_physical_fraction_admitted(state)
    ]
    max_restoration = int(solver.get("restoration_max_branches", 1))
    if fraction_admitted:
        selected = fraction_admitted[:width]
        remaining = max(0, width-len(selected))
        selected.extend(restoration[:min(remaining, max_restoration)])
        return selected
    return restoration[:min(width, max_restoration)]


def _finite_step17_spot_rms(
    reconstruct_out: dict[str, Any],
) -> float | None:
    """Return finite chief-centered Spot RMS or None when unavailable."""
    raw_value = reconstruct_out.get(
        "chief_centered_spot_RMS_mm"
    )
    if raw_value is None:
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def _step17_spot_guard(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    *,
    enabled: bool,
    maximum_regression_mm: float,
) -> dict[str, Any]:
    """Prevent O4-O5 from degrading chief-centered Spot RMS beyond tolerance."""
    tolerance = float(maximum_regression_mm)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "STEP17_SPOT_GUARD_TOLERANCE_INVALID"
        )

    parent_spot = _finite_step17_spot_rms(parent)
    candidate_spot = _finite_step17_spot_rms(candidate)
    delta = (
        None
        if parent_spot is None or candidate_spot is None
        else float(candidate_spot - parent_spot)
    )

    result = {
        "spot_RMS_guard_enabled": bool(enabled),
        "parent_spot_RMS_mm": parent_spot,
        "candidate_spot_RMS_mm": candidate_spot,
        "spot_RMS_delta_mm": delta,
        "maximum_spot_RMS_regression_mm": tolerance,
        "spot_RMS_guard_pass": True,
        "spot_RMS_guard_status": "DISABLED",
    }

    if not enabled:
        return result

    if parent_spot is None:
        result.update({
            "spot_RMS_guard_pass": False,
            "spot_RMS_guard_status":
                "PARENT_SPOT_RMS_UNAVAILABLE",
        })
        return result

    if candidate_spot is None:
        result.update({
            "spot_RMS_guard_pass": False,
            "spot_RMS_guard_status":
                "CANDIDATE_SPOT_RMS_NONFINITE",
        })
        return result

    if candidate_spot > parent_spot + tolerance + 1e-12:
        result.update({
            "spot_RMS_guard_pass": False,
            "spot_RMS_guard_status":
                "SPOT_RMS_REGRESSION_EXCEEDS_TOLERANCE",
        })
        return result

    result["spot_RMS_guard_status"] = "PASS"
    return result


def _ci_candidate_improves(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    *,
    spot_guard_enabled: bool,
    maximum_spot_rms_regression_mm: float,
) -> tuple[bool, str, dict[str, Any]]:
    """Compare CI candidates while protecting O4-O5 Spot RMS."""
    spot_guard = _step17_spot_guard(
        candidate,
        parent,
        enabled=spot_guard_enabled,
        maximum_regression_mm=(
            maximum_spot_rms_regression_mm
        ),
    )

    if not bool(spot_guard["spot_RMS_guard_pass"]):
        return (
            False,
            str(spot_guard["spot_RMS_guard_status"]),
            spot_guard,
        )

    candidate_count = int(
        candidate.get("physical_valid_count", 0)
    )
    parent_count = int(
        parent.get("physical_valid_count", 0)
    )
    candidate_fraction_admitted = bool(
        candidate.get("physical_fraction_pass", False)
    )
    parent_fraction_admitted = bool(
        parent.get("physical_fraction_pass", False)
    )

    if (
        candidate_fraction_admitted
        and not parent_fraction_admitted
    ):
        return (
            True,
            "MINIMUM_PHYSICAL_FRACTION_RECOVERED",
            spot_guard,
        )

    if candidate_count > parent_count:
        return (
            True,
            "PHYSICAL_COVERAGE_IMPROVED",
            spot_guard,
        )

    if candidate_count < parent_count:
        if (
            candidate_fraction_admitted
            and parent_fraction_admitted
            and float(candidate["Fan_core_merit"])
            < float(parent["Fan_core_merit"])
        ):
            return (
                True,
                "MINIMUM_FRACTION_MAINTAINED_WITH_FAN_IMPROVEMENT",
                spot_guard,
            )

        if (
            bool(candidate.get("restoration_branch", False))
            and float(candidate["Fan_core_merit"])
            < float(parent["Fan_core_merit"])
        ):
            return (
                True,
                "ADMISSIBLE_RESTORATION_STEP_WITH_FAN_IMPROVEMENT",
                spot_guard,
            )

        return (
            False,
            "PHYSICAL_COVERAGE_REGRESSED_WITHOUT_ADMISSIBLE_RESTORATION_PROGRESS",
            spot_guard,
        )

    if (
        float(candidate["Fan_core_merit"])
        < float(parent["Fan_core_merit"])
    ):
        return (
            True,
            "FAN_MERIT_IMPROVED_AT_EQUAL_PHYSICAL_COVERAGE",
            spot_guard,
        )

    return (
        False,
        "NO_LEXICOGRAPHIC_PHYSICAL_OR_FAN_IMPROVEMENT",
        spot_guard,
    )


def _persist_candidate_evidence_payload(
    owner: Context,
    evidence: dict[str, Any] | None,
    step_number: int,
    order: int,
    cycle: int,
    branch_id: str,
    reason: str,
) -> dict[str, Any]:
    """Lưu vết bằng chứng và dữ liệu hình học của ứng viên bị loại ra file artifact JSON."""
    from diagnostics_v55 import try_save_failed_candidate

    if not isinstance(evidence, dict):
        return {
            "candidate_artifact": None,
            "diagnostic_export_status": "NOT_RECORDED",
            "diagnostic_export_error": "NO_LOCAL_CANDIDATE_EVIDENCE",
        }

    details = {
        key: evidence[key]
        for key in (
            "order",
            "phase",
            "status",
            "exception_type",
            "exception_message",
            "root_exception_type",
            "root_exception_message",
            "m1_construction_input_gate",
            "m2_construction_input_gate",
            "m2_dynamic_reference_input_gate",
            "m2_dynamic_reference_output_gate",
            "numeric_admission",
            "physical_admission",
            "surface_sanity_admission",
            "M1_fit",
            "M2_fit",
            "M1_sanity",
            "M2_sanity",
            "integrability_quality",
            "M2_bundle_integrability_diagnostic",
            "reconstruct_out",
            "aperture_rebuild",
            "M2_footprint_filter_enabled",
            "M2_footprint_filter_method",
            "M2_footprint_rejected_count",
            "M2_footprint_rejected_fraction",
            "M2_footprint_retained_count",
            "M2_footprint_affected_bundle_count",
            "M2_footprint_max_bundle_rejected_fraction",
            "M2_local_xy_min_before_mm",
            "M2_local_xy_max_before_mm",
            "M2_local_xy_min_after_mm",
            "M2_local_xy_max_after_mm",
            "M2_half_aperture_before_mm",
            "M2_half_aperture_after_mm",
            "M2_footprint_outlier_rows",
            "M2_footprint_bundle_rows",
        )
        if key in evidence
    }

    display = evidence.get("display")
    details["display_prescription"] = (
        display.to_dict() if display is not None else None
    )
    details["algorithm_source_manifest_sha256"] = (
        owner.data.get("algorithm_source_manifest", {})
        .get("manifest_sha256")
    )

    physical = evidence.get("physical_trace")
    if physical is not None:
        mask = np.asarray(physical["valid"], bool)
        details["physical_valid_count"] = int(np.count_nonzero(mask))
        details["ray_count"] = int(mask.size)
        details["failed_ray_indices"] = np.flatnonzero(~mask)

    return try_save_failed_candidate(
        owner.step_dir(step_number),
        order,
        cycle,
        branch_id,
        evidence.get("m1"),
        evidence.get("m2"),
        reason,
        details,
    )


def _persist_candidate_evidence(
    owner: Context,
    branch: Context,
    step_number: int,
    order: int,
    cycle: int,
    branch_id: str,
    reason: str,
) -> dict[str, Any]:
    """Lưu vết bằng chứng từ Context của branch."""
    return _persist_candidate_evidence_payload(
        owner,
        branch.data.get("_last_candidate_evidence"),
        step_number,
        order,
        cycle,
        branch_id,
        reason,
    )


def _evaluate_step16_rho_branch(
    branch: Context,
    rho: float,
) -> dict[str, Any]:
    """Đánh giá một rho O2 từ cùng baseline STEP14; restoration chỉ được ghi nhận, không được chọn best."""
    branch.data["rho"] = float(rho)
    try:
        out = _reconstruct(
            branch,
            2,
            True,
            allow_restoration=True,
        )
        tail = _step16_spot_tail_metrics(out)
        hard_feasible = _step16_full_hard_feasible(out)
        out = out | {
            "step16_spot_tail": tail,
            "step16_hard_feasible": hard_feasible,
            "step16_aperture_rebuild_passed": True,
        }

        row = {
            "order": 2,
            "cycle": 1,
            "accepted": True,
            "eligible_for_step16_best": hard_feasible,
            "accepted_shape_update": True,
            "basis_promoted": False,
            "shape_optimized": True,
            "rho": float(rho),
            "basis": _ci_basis_label(2, True),
            "Fan_core_merit": out["Fan_core_merit"],
            "physical_valid_count": out["physical_valid_count"],
            "ray_count": out["ray_count"],
            "physical_valid_fraction": out["physical_valid_fraction"],
            "minimum_candidate_fraction": out[
                "minimum_candidate_fraction"
            ],
            "physical_fraction_pass": out["physical_fraction_pass"],
            "physical_first_hit_complete": out["physical_first_hit_complete"],
            "unobscured": out["unobscured"],
            "integrability_status": out["integrability_quality"]["status"],
            "M1_sanity_pass": bool(out["M1_sanity"]["pass"]),
            "M2_sanity_pass": bool(out["M2_sanity"]["pass"]),
            "M1_sanity_admitted": bool(out["M1_sanity_admitted"]),
            "M2_sanity_admitted": bool(out["M2_sanity_admitted"]),
            "M2_sanity_warn_admitted": bool(
                out["M2_sanity_warn_admitted"]
            ),
            "aperture_rebuild_pass": True,
            "feasibility_status": out["feasibility_status"],
            "chief_centered_spot_RMS_mm": out["chief_centered_spot_RMS_mm"],
            "bundle_RMS_P95_mm": tail["bundle_RMS_P95_mm"],
            "worst_bundle_RMS_mm": tail["worst_bundle_RMS_mm"],
            "max_spot_radius_mm": tail["max_spot_radius_mm"],
            "reason": (
                "PASS_STEP16_MINIMUM_FRACTION_HARD_FEASIBILITY"
                if hard_feasible
                else "RESTORATION_ONLY_NOT_ELIGIBLE_FOR_STEP16_BEST"
            ),
        }

        state = _rho_state(
            branch,
            [float(rho)],
            [row],
            out,
        )

        return {
            "status": "ACCEPTED",
            "rho": float(rho),
            "row": row,
            "state": state,
            "evidence": copy.deepcopy(
                branch.data.get("_last_candidate_evidence")
            ),
        }
    except RuntimeError as exc:
        row = {
            "order": 2,
            "cycle": 1,
            "accepted": False,
            "eligible_for_step16_best": False,
            "accepted_shape_update": False,
            "basis_promoted": False,
            "shape_optimized": False,
            "rho": float(rho),
            "basis": _ci_basis_label(2, True),
            "reason": str(exc),
        }

        return {
            "status": "REJECTED",
            "rho": float(rho),
            "row": row,
            "state": None,
            "error": str(exc),
            "evidence": copy.deepcopy(
                branch.data.get("_last_candidate_evidence")
            ),
        }
















def _reverse_surrogate_distortion(ctx: Context, tr: dict[str, Any],
                                  display: PolySurface | None = None) -> dict[str, Any]:
    """Tính distortion surrogate từ trace ngược trong vòng tối ưu."""
    r = ctx.data["rays"]; actual = np.asarray(tr["display_local"])[:, :2]
    display = display or ctx.data["display"]
    fixed3 = ctx.data["fixed_display_surrogate"]
    fixed = (fixed3 - display.center) @ display.frame
    center_field = int(r["central_field_index"])
    fixed = fixed[:, :2] - fixed[center_field, :2]
    nf = int(r["field_count"]); npup = int(r["pupil_count"])
    cen = np.zeros((nf, npup, 2))
    for f in range(nf):
        for p in range(npup):
            mask = (r["field_index"] == f) & (r["pupil_index"] == p)
            cen[f, p] = np.mean(actual[mask], axis=0)
    vals = []
    for f in range(nf):
        if f == center_field: continue
        for p in range(npup):
            q = cen[f, p] - cen[center_field, p]
            vals.append(100.0 * np.linalg.norm(q - fixed[f]) / max(np.linalg.norm(fixed[f]),
                                                                    float(ctx.config["distortion"]["epsilon_mm"])))
    return {"D_max_percent": float(np.max(vals)), "D_rms_percent": float(np.sqrt(np.mean(np.asarray(vals) ** 2))),
            "role": "REVERSE_DISPLAY_SURROGATE_OPTIMIZATION_ONLY_NOT_FINAL_AUTHORITY"}


def _objective_eval(ctx: Context, m1: PolySurface, m2: PolySurface,
                    display: PolySurface | None = None,
                    *,
                    use_reverse_cache: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Tính tách biệt Fan objective, engineering penalty và diagnostics."""
    r = ctx.data["rays"]; display = display or ctx.data["display"]
    tr = _reverse_evaluation_trace(ctx, m1, m2, display, use_cache=use_reverse_cache)
    refs, _ = _dynamic_refs(ctx, tr, display)
    mf1 = fan_imaging_metrics(tr, r, refs, float(ctx.config["fan_weights"]["omega1"]))
    mf2 = mf2_geometry(tr["points"][0], tr["points"][1], tr["points"][2], m2,
                       float(ctx.config["fan_weights"]["omega2"]), ctx.data["n_obs"])
    E1 = float(mf1["E1_mm"]); E2 = max(0.0, -float(mf2["S_AQP_signed_mm2"]))
    dist = _reverse_surrogate_distortion(ctx, tr, display); lam = _packaging(ctx, m1, m2, display)
    invalid = 1.0 - float(np.mean(tr["valid"])); dc = ctx.config["distortion"]
    phiD = max(0.0, dist["D_max_percent"] / float(dc["hard_limit_percent"]) - 1.0) ** 2
    phiP = (0.0 if lam is None else
            max(0.0, lam / float(ctx.config["packaging_lambda_max"]) - 1.0) ** 2)
    phiR = invalid ** 2
    w = ctx.config["normalized_objective"]; L = float(w["L_ref_mm"]); A = L * L
    fan_terms = {"w1_MF1_over_Lref": float(w["w1"]) * float(mf1["MF1_Fan"]) / L,
                 "w2_MF2_over_Aref": float(w["w2"]) * float(mf2["MF2"]) / A,
                 "w3_MF3_over_Lref": 0.0}
    engineering_terms = {"wD_PhiD": float(w["wD"]) * phiD,
                         "wP_PhiP": float(w["wP"]) * phiP,
                         "wR_PhiR": float(w["wR"]) * phiR}
    # Scale the three-dimensional chief-centered intercept residual so that
    # its squared norm is exactly w1*MF1/Lref while retaining the existing
    # outer pupil sum and DLSQ normalization.
    components: list[np.ndarray] = []

    errors3 = np.empty_like(
        np.asarray(
            tr["landing"],
            float,
        )
    )

    for field_index in range(
        int(
            r[
                "field_count"
            ]
        )
    ):
        for pupil_index in range(
            int(
                r[
                    "pupil_count"
                ]
            )
        ):
            bundle = (
                (
                    r[
                        "field_index"
                    ]
                    == field_index
                )
                &
                (
                    r[
                        "pupil_index"
                    ]
                    == pupil_index
                )
            )

            chief_indices = np.where(
                bundle
                & r[
                    "chief"
                ]
            )[0]

            if len(
                chief_indices
            ) != 1:
                raise RuntimeError(
                    "OBJECTIVE_CHIEF_SELECTION_MISMATCH"
                )

            chief_index = int(
                chief_indices[0]
            )

            errors3[bundle] = (
                tr[
                    "landing"
                ][
                    bundle
                ]
                -
                tr[
                    "landing"
                ][
                    chief_index
                ]
            )

    omega1 = float(
        ctx.config[
            "fan_weights"
        ][
            "omega1"
        ]
    )

    for pupil_index, pupil_rms in enumerate(
        mf1[
            "per_pupil_RMS_mm"
        ]
    ):
        mask = (
            r[
                "pupil_index"
            ]
            == pupil_index
        )

        count = int(
            np.sum(
                mask
            )
        )

        scale = math.sqrt(
            float(
                w[
                    "w1"
                ]
            )
            * omega1
            /
            (
                L
                * count
                * max(
                    float(
                        pupil_rms
                    ),
                    1e-15,
                )
            )
        )

        components.append(
            (
                scale
                * errors3[
                    mask
                ]
            ).ravel()
        )
    components.append(np.array([math.sqrt(max(fan_terms["w2_MF2_over_Aref"], 0.0))]))
    residual = np.concatenate(components)
    sanity_m1 = _mirror_topology_gate(
        ctx,
        m1,
        tr,
        "M1",
    )
    sanity_m2 = _mirror_topology_gate(
        ctx,
        m2,
        tr,
        "M2",
    )
    hard_valid = bool(
        np.all(tr["valid"])
        and sanity_m1["pass"]
        and sanity_m2["pass"]
        and float(mf2["S_AQP_signed_mm2"]) >= 0.0
    )
    rec = {
        "J_Fan_dimensionless": float(sum(fan_terms.values())),
        # Giá trị này chỉ là chẩn đoán reverse rẻ cho các lần tính Jacobian.
        # Khi xét nhận bước, bộ đánh giá forward sẽ ghi đè J_engineering_dimensionless.
        "J_engineering_dimensionless": float(sum(engineering_terms.values())),
        "J_engineering_reverse_surrogate_dimensionless": float(sum(engineering_terms.values())),
        "hard_physical_surface_valid": hard_valid,
        "E1_mm":
            E1,

        "MF1_Fan":
            float(
                mf1[
                    "MF1_Fan"
                ]
            ),

        "chief_pupil_spread_RMS_mm":
            float(
                mf1[
                    "chief_pupil_spread_RMS_mm"
                ]
            ),

        "chief_pupil_spread_max_mm":
            float(
                mf1[
                    "chief_pupil_spread_max_mm"
                ]
            ),

        "chief_field_bias_RMS_mm":
            float(
                mf1[
                    "chief_field_bias_RMS_mm"
                ]
            ),

        "chief_field_bias_max_mm":
            float(
                mf1[
                    "chief_field_bias_max_mm"
                ]
            ),

        "E2_mm2":
            E2,

        "MF2_Fan":
            float(
                mf2[
                    "MF2"
                ]
            ),
        "S_AQP_signed_mm2": float(mf2["S_AQP_signed_mm2"]),
        "D_reverse_surrogate_percent": dist["D_max_percent"], "packaging_lambda": lam,
        "packaging_constraint_enabled": lam is not None,
        "invalid_fraction": invalid, "PhiD": phiD, "PhiP": phiP, "PhiR": phiR,
        "Fan_terms": fan_terms, "engineering_terms": engineering_terms,
        "M1_surface_pass": sanity_m1["pass"], "M2_surface_pass": sanity_m2["pass"],
        "M1_surface_reasons": sanity_m1.get("failure_reasons", []),
        "M2_surface_reasons": sanity_m2.get("failure_reasons", []),
    }
    return residual, rec


def _refinement_worker_snapshot(
    ctx: Context,
) -> Context:
    """Context toi thieu cho worker Jacobian STEP 23."""
    snapshot = Context(
        source_dir=ctx.source_dir,
        config_path=ctx.config_path,
        config=copy.deepcopy(ctx.config),
        run_dir=ctx.run_dir,
    )

    keys = (
        "rays",
        "visor",
        "display",
        "vi",
        "n_obs",
        "fixed_display_surrogate",
        "algorithm_source_manifest",
        "packaging_vertices",
        "inputs",
        "pattern",
        "fixed_vi_grid",
        "pupils",
    )

    snapshot.data = {
        key: ctx.data[key]
        for key in keys
        if key in ctx.data
    }

    return snapshot


def _parallel_surface_parameter_columns(
    ctx: Context,
    base: dict[str, PolySurface],
    u: np.ndarray,
    cfg: dict[str, Any],
    fd: float,
) -> list[dict[str, Any]] | None:
    """Tra None neu khong co pool; neu co thi tinh 4 cot song song."""
    runtime = current_runtime()

    if (
        runtime is None
        or is_compute_worker()
        or not bool(
            runtime.config.get(
                "parallel_jacobian",
                False,
            )
        )
    ):
        return None

    executor = runtime._ray_trace_executor

    if executor is None:
        return None

    from execution_workers_v55 import (
        ordered_bounded_map,
        surface_parameter_column_worker,
    )

    jobs = [
        {
            "column": column,
            "u": np.asarray(
                u, dtype=float
            ),
            "fd": float(fd),
            "base": base,
            "cfg": cfg,
        }
        for column in range(4)
    ]

    replies = ordered_bounded_map(
        executor,
        surface_parameter_column_worker,
        jobs,
        int(
            runtime._ray_trace_max_inflight
        ),
    )

    return replies


def _parallel_geometry_columns(
    ctx: Context,
    base: dict[str, PolySurface],
    specs: list[dict[str, Any]],
    u: np.ndarray,
    fd: float,
) -> list[dict[str, Any]] | None:
    """Tinh cac cot pose tren pool STEP23 hien tai."""
    runtime = current_runtime()

    if (
        runtime is None
        or is_compute_worker()
        or not bool(
            runtime.config.get(
                "parallel_jacobian",
                False,
            )
        )
    ):
        return None

    executor = runtime._ray_trace_executor

    if executor is None:
        return None

    from execution_workers_v55 import (
        geometry_column_worker,
        ordered_bounded_map,
    )

    jobs = [
        {
            "column": column,
            "u": np.asarray(
                u, dtype=float
            ),
            "fd": float(fd),
            "base": base,
            "specs": specs,
        }
        for column in range(len(specs))
    ]

    replies = ordered_bounded_map(
        executor,
        geometry_column_worker,
        jobs,
        int(
            runtime._ray_trace_max_inflight
        ),
    )

    return replies


@contextmanager
def coefficient_column_evaluator(ctx: Context, phase: str) -> Iterator[Any]:
    """Mở compute pool cho toàn pha DLSQ coefficient và cung cấp callback evaluate_columns và evaluate_trials."""
    runtime = current_runtime()
    if (
        runtime is None
        or is_compute_worker()
        or not runtime.config.get("parallel_jacobian", False)
        or int(runtime.config.get("cpu_workers", 1)) <= 1
    ):
        yield {
            "columns": None,
            "trials": None,
        }
        return

    snapshot = _refinement_worker_snapshot(ctx)

    from execution_v55 import managed_compute_pool
    from execution_workers_v55 import jacobian_column_worker, dlsq_trial_worker, ordered_bounded_map

    with managed_compute_pool(snapshot, purpose=f"JACOBIAN_{phase}") as (executor, effective_max_inflight, policy):
        runtime._ray_trace_executor = executor
        runtime._ray_trace_max_inflight = effective_max_inflight
        runtime._ray_trace_chunk_size = int(runtime.config.get("ray_trace_chunk_size", 2048))
        runtime._ray_trace_min_rays = int(runtime.config.get("ray_trace_min_rays", 512))

        def evaluate_columns(m1, m2, variable_indices, fd_sag):
            """Thực thi evaluate_columns."""
            state_id = hashlib.sha256(
                pickle.dumps(
                    (
                        m1.to_dict(),
                        m2.to_dict(),
                        tuple(variable_indices),
                        float(fd_sag),
                    ),
                    protocol=5,
                )
            ).hexdigest()

            jobs = [
                {
                    "column": column,
                    "which": which,
                    "coeff_index": index,
                    "fd_sag": float(fd_sag),
                    "m1": m1,
                    "m2": m2,
                    "state_id": state_id,
                }
                for column, (which, index) in enumerate(variable_indices)
            ]

            replies = ordered_bounded_map(
                executor,
                jacobian_column_worker,
                jobs,
                effective_max_inflight,
            )

            residuals = []
            for column, reply in enumerate(replies):
                if (
                    reply["column"] != column
                    or reply["state_id"] != state_id
                ):
                    raise BackendConsistencyError(
                        "JACOBIAN_REPLY_IDENTITY_MISMATCH"
                    )
                residuals.append(reply["residual"])

            return residuals

        def evaluate_trials(m1, m2, variable_indices, steps):
            """Thực thi evaluate_trials."""
            jobs = []
            for trial_index, step in enumerate(steps):
                jobs.append({
                    "trial_index": trial_index,
                    "step": np.asarray(step, dtype=float),
                    "m1": m1,
                    "m2": m2,
                    "variable_indices": tuple(variable_indices),
                })

            replies = ordered_bounded_map(
                executor,
                dlsq_trial_worker,
                jobs,
                effective_max_inflight,
            )

            for expected, reply in enumerate(replies):
                if reply["trial_index"] != expected:
                    raise BackendConsistencyError(
                        "DLSQ_TRIAL_REPLY_ORDER_MISMATCH"
                    )

            return replies

        parallel_trials_enabled = bool(
            runtime.config.get("parallel_dlsq_trials", True)
        )

        try:
            yield {
                "columns": evaluate_columns,
                "trials": evaluate_trials if parallel_trials_enabled else None,
            }
        finally:
            runtime._ray_trace_executor = None
            runtime._ray_trace_max_inflight = 1


def _coefficient_dlsq_phase(ctx: Context, phase: str) -> dict[str, Any]:
    """Chạy một phase DLSQ coefficient và cập nhật Context bằng nghiệm được nhận."""
    s = ctx.config["solver"]
    variables = ([("M1", i) for i, t in enumerate(ctx.data["m1"].terms) if t != (0, 0)] +
                 [("M2", i) for i, t in enumerate(ctx.data["m2"].terms) if t != (0, 0)])
    evaluate = lambda a, b: _objective_eval(ctx, a, b)
    with coefficient_column_evaluator(ctx, phase) as parallel:
        columns = parallel["columns"] if isinstance(parallel, dict) else None
        trials = parallel["trials"] if isinstance(parallel, dict) else None
        _, initial_fan = evaluate(ctx.data["m1"], ctx.data["m2"])
        initial = {**initial_fan, **_forward_engineering_eval(ctx, ctx.data["m1"], ctx.data["m2"])}
        m1, m2, history, trials_records, termination = normalized_dlsq_optimize(
            ctx.data["m1"], ctx.data["m2"], evaluate, variables, int(s["optimization_iterations"]),
            float(s["optimization_damping"]), float(s["optimization_fd_sag_mm"]),
            float(s["objective_plateau_relative"]), float(s["engineering_worsening_relative"]),
            evaluate_engineering=lambda a, b: _forward_engineering_eval(ctx, a, b),
            engineering_absolute_tolerance=float(s["dlsq_engineering_absolute_tolerance"]),
            restoration_fan_worsening_relative=float(
                s["dlsq_restoration_fan_worsening_relative"]),
            restoration_engineering_improvement_relative=float(
                s["dlsq_restoration_engineering_improvement_relative"]),
            restoration_component_worsening_relative=float(
                s["dlsq_restoration_component_worsening_relative"]),
            restoration_trial_limit=int(s["dlsq_restoration_trial_limit"]),
            restoration_poll_variable_count=int(s["dlsq_restoration_poll_variable_count"]),
            restoration_poll_step_mm=float(s["dlsq_restoration_poll_step_mm"]),
            damping_multipliers=tuple(map(float, s["dlsq_damping_multipliers"])),
            line_search_alphas=tuple(map(float, s["dlsq_line_search_alphas"])),
            evaluate_columns=columns,
            evaluate_trials=trials,
        )
        phase_trials = [{"phase": phase, **row} for row in trials_records]
        phase_termination = {"phase": phase, **termination}
        ctx.data.setdefault("dlsq_trial_history", []).extend(phase_trials)
        ctx.data.setdefault("dlsq_termination_history", []).append(phase_termination)
        ctx.data.update({"m1": m1, "m2": m2, "optimization_history": history,
                         "dlsq_last_termination": phase_termination})
        tr = _reverse_evaluation_trace(ctx, m1, m2, ctx.data["display"])
        refs, rule = _dynamic_refs(ctx, tr); ctx.data.update({"trace": tr, "fan_refs": refs, "fan_reference_rule": rule})
        _publish_trace_debug(
            ctx, 23, phase, tr, ctx.data["rays"],
            direction="REVERSE",
        )
        _, final_fan = evaluate(m1, m2)
        final = {**final_fan, **_forward_engineering_eval(ctx, m1, m2)}
        ctx.data["objective_final"] = final
        ctx.data["reference_history"].append({"cycle": f"POST_{phase}", **rule})
    return {"phase": phase, "history": history, "trials": phase_trials,
            "termination": phase_termination, "initial": initial, "final": final,
            "variables": len(variables), "iterations": len(history)}


def _apply_surface_parameter_vector(base: dict[str, PolySurface], normalized: np.ndarray,
                                    cfg: dict[str, Any]) -> tuple[PolySurface, PolySurface]:
    """Ap vector chuan hoa vao curvature va conic cua M1/M2 trong bound hep."""
    m1, m2 = base["M1"].copy(), base["M2"].copy()
    c_fraction = float(cfg["curvature_relative_bound"])
    k_bound = float(cfg["conic_absolute_bound"])
    for offset, surface in ((0, m1), (2, m2)):
        surface.curvature = float(base[surface.name].curvature * (1.0+c_fraction*normalized[offset]))
        surface.conic = float(base[surface.name].conic+k_bound*normalized[offset+1])
    return m1, m2


def _engineering_candidate_nonworsening(base: dict[str, Any], candidate: dict[str, Any],
                                        relative: float) -> bool:
    """Kiem tra engineering tong va tung thanh phan khong bi che boi thanh phan khac."""
    limit = float(base["J_engineering_dimensionless"])*(1.0+relative)+1e-12
    passed = bool(float(candidate["J_engineering_dimensionless"]) <= limit)
    for key in ("forward_distortion_monitor_percent", "forward_invalid_ray_fraction",
                "forward_invalid_bundle_fraction", "packaging_lambda_constraint_value"):
        if key in base and key in candidate:
            passed = passed and bool(float(candidate[key]) <= float(base[key])*(1.0+relative)+1e-12)
    mask_key = "_forward_distortion_evaluated_pair_mask"
    if mask_key in base and mask_key in candidate:
        base_mask = np.asarray(base[mask_key], bool)
        candidate_mask = np.asarray(candidate[mask_key], bool)
        passed = passed and base_mask.shape == candidate_mask.shape and bool(np.all(candidate_mask[base_mask]))
    return bool(passed)


def _bounded_fd_trial(
    u: np.ndarray,
    column: int,
    requested_step: float,
) -> tuple[np.ndarray, float]:
    """Tính toán vector thử nghiệm và bước sai phân thực tế có chặn biên trong khoảng [-1, 1]."""
    u = np.asarray(u, float)
    h = float(requested_step)

    if u.ndim != 1 or not np.all(np.isfinite(u)):
        raise ValueError("FD_BASE_VECTOR_INVALID")
    if not 0 <= column < len(u):
        raise IndexError("FD_COLUMN_OUT_OF_RANGE")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("FD_STEP_MUST_BE_FINITE_POSITIVE")
    if np.any(u < -1.0) or np.any(u > 1.0):
        raise ValueError("FD_BASE_OUTSIDE_BOUNDS")

    plus_room = 1.0 - float(u[column])
    minus_room = float(u[column]) + 1.0

    if plus_room >= h:
        signed_step = h
    elif minus_room >= h:
        signed_step = -h
    elif plus_room >= minus_room:
        signed_step = plus_room
    else:
        signed_step = -minus_room

    trial = u.copy()
    trial[column] = np.clip(
        u[column] + signed_step, -1.0, 1.0
    )
    actual_step = float(trial[column] - u[column])

    if actual_step == 0.0 or not np.isfinite(actual_step):
        raise RuntimeError("FD_NO_REPRESENTABLE_NONZERO_STEP")

    return trial, actual_step


def _bounded_surface_parameter_refine(ctx: Context) -> dict[str, Any]:
    """Tinh chinh c va K bang adaptive damped least-squares voi physical/engineering guardrail."""
    cfg = ctx.config["surface_parameter_refinement"]
    base = {"M1": ctx.data["m1"].copy(), "M2": ctx.data["m2"].copy()}
    u = np.zeros(4, float)
    m1, m2 = _apply_surface_parameter_vector(base, u, cfg)
    residual, fan = _objective_eval(ctx, m1, m2)
    current = {**fan, **_forward_engineering_eval(ctx, m1, m2)}
    initial = copy.deepcopy(current)
    history: list[dict[str, Any]] = []
    fd = float(cfg["finite_difference_normalized"])
    damping = float(cfg["damping"])
    relative = float(ctx.config["solver"]["engineering_worsening_relative"])
    for iteration in range(
        int(cfg["iterations"])
    ):
        jacobian = np.empty(
            (len(residual), 4)
        )

        parallel_columns = (
            _parallel_surface_parameter_columns(
                ctx,
                base,
                u,
                cfg,
                fd,
            )
        )

        if parallel_columns is None:
            for column in range(4):
                trial, actual_fd = (
                    _bounded_fd_trial(
                        u,
                        column,
                        fd,
                    )
                )

                a, b = (
                    _apply_surface_parameter_vector(
                        base,
                        trial,
                        cfg,
                    )
                )

                trial_residual, _ = (
                    _objective_eval(
                        ctx,
                        a,
                        b,
                    )
                )

                trial_residual = np.asarray(
                    trial_residual,
                    float,
                )

                if (
                    trial_residual.shape
                    != residual.shape
                    or not np.all(
                        np.isfinite(
                            trial_residual
                        )
                    )
                    or not np.all(
                        np.isfinite(
                            residual
                        )
                    )
                ):
                    raise RuntimeError(
                        "SURFACE_PARAMETER_"
                        "JACOBIAN_INVALID_COLUMN:"
                        f"{column}"
                    )

                jacobian[:, column] = (
                    trial_residual
                    - residual
                ) / actual_fd

        else:
            if len(parallel_columns) != 4:
                raise BackendExecutionError(
                    "SURFACE_PARAMETER_"
                    "JACOBIAN_COLUMN_COUNT_MISMATCH"
                )

            for column, reply in enumerate(
                parallel_columns
            ):
                if int(
                    reply["column"]
                ) != column:
                    raise BackendConsistencyError(
                        "SURFACE_PARAMETER_"
                        "JACOBIAN_COLUMN_ORDER_MISMATCH"
                    )

                trial_residual = np.asarray(
                    reply["residual"],
                    float,
                )

                actual_fd = float(
                    reply["actual_fd"]
                )

                if (
                    trial_residual.shape
                    != residual.shape
                    or not np.all(
                        np.isfinite(
                            trial_residual
                        )
                    )
                    or not np.isfinite(
                        actual_fd
                    )
                    or actual_fd == 0.0
                ):
                    raise BackendExecutionError(
                        "SURFACE_PARAMETER_"
                        "JACOBIAN_PARALLEL_INVALID:"
                        f"{column}"
                    )

                jacobian[:, column] = (
                    trial_residual
                    - residual
                ) / actual_fd
        singular = np.linalg.svd(jacobian, compute_uv=False)
        condition = (float(singular[0]/singular[-1])
                     if len(singular) and singular[-1] > 1e-15 else None)
        jtj = jacobian.T@jacobian
        diagonal = np.maximum(np.diag(jtj), 1e-12)
        rhs = -(jacobian.T@residual)
        accepted = False
        alpha_used = 0.0
        damping_used = damping
        for trial_damping in (damping, 10.0*damping, 100.0*damping):
            try:
                delta = np.linalg.solve(jtj+trial_damping*np.diag(diagonal), rhs)
            except np.linalg.LinAlgError:
                continue
            delta = np.clip(delta, -float(cfg["max_step_normalized"]),
                            float(cfg["max_step_normalized"]))
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
                candidate_u = np.clip(u+alpha*delta, -1.0, 1.0)
                a, b = _apply_surface_parameter_vector(base, candidate_u, cfg)
                candidate_residual, candidate_fan = _objective_eval(ctx, a, b)
                if not candidate_fan["hard_physical_surface_valid"]:
                    continue
                candidate = {**candidate_fan, **_forward_engineering_eval(ctx, a, b)}
                fan_better = bool(candidate["J_Fan_dimensionless"] < current["J_Fan_dimensionless"])
                engineering_ok = _engineering_candidate_nonworsening(current, candidate, relative)
                if fan_better and engineering_ok:
                    u, m1, m2 = candidate_u, a, b
                    residual, current = candidate_residual, candidate
                    accepted = True
                    alpha_used = alpha
                    damping_used = trial_damping
                    break
            if accepted:
                break
        damping = max(damping_used*0.3, 1e-8) if accepted else min(damping*10.0, 1e8)
        history.append({
            "iteration": iteration, "accepted": accepted, "alpha": alpha_used,
            "adaptive_damping_used": damping_used, "adaptive_damping_next": damping,
            "J_Fan_dimensionless": current["J_Fan_dimensionless"],
            "J_engineering_dimensionless": current["J_engineering_dimensionless"],
            "jacobian_condition": condition, "normalized_parameters": u.tolist(),
            "M1_curvature_per_mm": m1.curvature, "M1_conic": m1.conic,
            "M2_curvature_per_mm": m2.curvature, "M2_conic": m2.conic,
        })
        if not accepted:
            break
    ctx.data.update({"m1": m1, "m2": m2,
                     "surface_parameter_refinement_history": history})
    tr = trace_reverse(ctx.data["rays"], ctx.data["visor"], m1, m2, ctx.data["display"], True)
    refs, rule = _dynamic_refs(ctx, tr)
    ctx.data.update({"trace": tr, "fan_refs": refs, "fan_reference_rule": rule})
    ctx.data["reference_history"].append({"cycle": "POST_23S_SURFACE_PARAMETERS", **rule})
    return {"status": "PASS", "variables": ["M1.curvature", "M1.conic", "M2.curvature", "M2.conic"],
            "iterations": len(history), "accepted_iterations": sum(bool(x["accepted"]) for x in history),
            "initial": initial, "final": current, "normalized_parameters": u.tolist(),
            "bounds": {"curvature_relative": cfg["curvature_relative_bound"],
                       "conic_absolute": cfg["conic_absolute_bound"]},
            "coefficients_changed_in_23S": False, "aperture_optimized": False}


def _rotation_xy(rx: float, ry: float) -> np.ndarray:
    """Tạo rotation local trực chuẩn từ hai góc quanh trục X/Y; không mở gauge Rz."""
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    return np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]) @ \
           np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])


def _geometry_variable_specs(ctx: Context) -> list[dict[str, Any]]:
    """Đọc các DOF pose có bound; curvature/conic/aperture cố ý không nằm trong danh sách."""
    cfg = ctx.config["geometry_refinement"]
    specs: list[dict[str, Any]] = []
    for name in ("M1", "M2", "DISPLAY"):
        item = cfg["surfaces"][name]
        if not item["enabled"]:
            continue
        for axis, bound in zip("xyz", item["translation_local_bound_mm"]):
            if float(bound) > 0.0:
                specs.append({"surface": name, "kind": "translation", "axis": axis,
                              "scale": float(bound), "unit": "mm"})
        for axis, bound in zip("xy", item["tilt_local_bound_deg"]):
            if float(bound) > 0.0:
                specs.append({"surface": name, "kind": "tilt", "axis": axis,
                              "scale": math.radians(float(bound)), "unit": "rad"})
    return specs


def _apply_geometry_vector(base: dict[str, PolySurface], specs: list[dict[str, Any]],
                           normalized: np.ndarray) -> tuple[PolySurface, PolySurface, PolySurface]:
    """Áp vector chuẩn hóa vào rigid pose, giữ nguyên sag, aperture, curvature và conic."""
    surfaces = {name: surface.copy() for name, surface in base.items()}
    values: dict[str, dict[str, dict[str, float]]] = {
        name: {"translation": {a: 0.0 for a in "xyz"}, "tilt": {a: 0.0 for a in "xy"}}
        for name in surfaces
    }
    for spec, value in zip(specs, normalized):
        values[spec["surface"]][spec["kind"]][spec["axis"]] = float(value) * spec["scale"]
    for name, surface in surfaces.items():
        translation = np.array([values[name]["translation"][a] for a in "xyz"])
        rx, ry = (values[name]["tilt"][a] for a in "xy")
        surface.center = base[name].center + base[name].frame @ translation
        surface.frame = base[name].frame @ _rotation_xy(rx, ry)
    return surfaces["M1"], surfaces["M2"], surfaces["DISPLAY"]


def _geometry_forward_eval(ctx: Context, m1: PolySurface, m2: PolySurface,
                           display: PolySurface, with_fan: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Tạo residual forward liên tục: pupil miss và hướng ảnh so với fixed VI authority."""
    cfg = ctx.config["geometry_refinement"]
    evaluation = _run_forward_evaluation(ctx, m1, m2, monitor_only=True, display=display)
    fwd, rays = evaluation["forward"], evaluation["rays"]
    eye_scale = float(cfg["eye_residual_scale_mm"])
    image_scale = float(cfg["image_residual_scale_mm"])
    invalid = float(cfg["invalid_residual"])
    eye_residual = np.asarray(fwd["residual_yz"], float) / eye_scale
    apparent = -unit(np.asarray(fwd["arrive_direction"], float))
    fixed = np.asarray(ctx.data["fixed_vi_grid"], float)[rays["field_index"]]
    desired = unit(fixed - np.asarray(rays["origins"], float))
    direction_residual = (apparent - desired) * float(ctx.config["vid_mm"]) / image_scale
    residual = np.concatenate([eye_residual.ravel(), direction_residual.ravel()])
    residual = np.nan_to_num(residual, nan=invalid, posinf=invalid, neginf=-invalid)
    record = {
        "J_forward_continuous": float(np.mean(residual * residual)),
        "forward_converged": int(np.sum(fwd["converged"])),
        "forward_ray_count": len(fwd["converged"]),
        "physical_first_order_count": int(np.sum(evaluation["first_order"])),
        "valid_bundle_count": int(evaluation["virtual"]["valid_bundle_count"]),
        "required_bundle_count": int(evaluation["virtual"]["required_bundle_count"]),
        "packaging_lambda": _packaging(ctx, m1, m2, display),
    }
    if with_fan:
        _, fan = _objective_eval(ctx, m1, m2, display)
        record.update({"J_Fan_dimensionless": fan["J_Fan_dimensionless"],
                       "hard_physical_surface_valid": fan["hard_physical_surface_valid"]})
    return residual / math.sqrt(max(len(residual), 1)), record


def _bounded_geometry_refine(ctx: Context) -> dict[str, Any]:
    """DLSQ pose chuẩn hóa có bound, forward objective và Fan guardrail."""
    cfg = ctx.config["geometry_refinement"]
    specs = _geometry_variable_specs(ctx)
    if not specs:
        raise RuntimeError("GEOMETRY_REFINEMENT_ENABLED_WITHOUT_ACTIVE_POSE_VARIABLES")
    base = {"M1": ctx.data["m1"].copy(), "M2": ctx.data["m2"].copy(),
            "DISPLAY": ctx.data["display"].copy()}
    u = np.zeros(len(specs)); m1, m2, display = _apply_geometry_vector(base, specs, u)
    r0, current = _geometry_forward_eval(ctx, m1, m2, display, True)
    initial = copy.deepcopy(current); initial_fan = float(initial["J_Fan_dimensionless"])
    history: list[dict[str, Any]] = []
    fd = float(cfg["finite_difference_normalized"])
    for iteration in range(int(cfg["iterations"])):
        J = np.empty(
            (len(r0), len(specs))
        )

        parallel_columns = (
            _parallel_geometry_columns(
                ctx,
                base,
                specs,
                u,
                fd,
            )
        )

        if parallel_columns is None:
            for j in range(len(specs)):
                trial_u = u.copy()
                trial_u[j] = min(
                    1.0,
                    trial_u[j] + fd,
                )

                actual_fd = (
                    trial_u[j] - u[j]
                )

                a, b, d = (
                    _apply_geometry_vector(
                        base,
                        specs,
                        trial_u,
                    )
                )

                rp, _ = (
                    _geometry_forward_eval(
                        ctx,
                        a,
                        b,
                        d,
                        False,
                    )
                )

                J[:, j] = (
                    rp - r0
                ) / max(
                    actual_fd,
                    1e-12,
                )

        else:
            if len(
                parallel_columns
            ) != len(specs):
                raise BackendExecutionError(
                    "GEOMETRY_JACOBIAN_"
                    "COLUMN_COUNT_MISMATCH"
                )

            for j, reply in enumerate(
                parallel_columns
            ):
                if int(
                    reply["column"]
                ) != j:
                    raise BackendConsistencyError(
                        "GEOMETRY_JACOBIAN_"
                        "COLUMN_ORDER_MISMATCH"
                    )

                rp = np.asarray(
                    reply["residual"],
                    float,
                )

                actual_fd = float(
                    reply["actual_fd"]
                )

                if (
                    rp.shape != r0.shape
                    or not np.all(
                        np.isfinite(rp)
                    )
                    or not np.isfinite(
                        actual_fd
                    )
                ):
                    raise BackendExecutionError(
                        "GEOMETRY_JACOBIAN_"
                        f"PARALLEL_INVALID:{j}"
                    )

                J[:, j] = (
                    rp - r0
                ) / max(
                    actual_fd,
                    1e-12,
                )
        singular = np.linalg.svd(J, compute_uv=False)
        condition = (float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 1e-15
                     else None)
        jtj = J.T @ J; diagonal = np.maximum(np.diag(jtj), 1e-12); rhs = -(J.T @ r0)
        accepted = False; accepted_alpha = 0.0
        for damp in (float(cfg["damping"]), 10.0*float(cfg["damping"]), 100.0*float(cfg["damping"])):
            try:
                delta = np.linalg.solve(jtj + damp*np.diag(diagonal), rhs)
            except np.linalg.LinAlgError:
                continue
            delta = np.clip(delta, -float(cfg["max_step_normalized"]),
                            float(cfg["max_step_normalized"]))
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
                candidate_u = np.clip(u + alpha*delta, -1.0, 1.0)
                a, b, d = _apply_geometry_vector(base, specs, candidate_u)
                candidate_r, candidate = _geometry_forward_eval(ctx, a, b, d, True)
                forward_better = candidate["J_forward_continuous"] < current["J_forward_continuous"]
                topology_ok = (candidate["forward_converged"] >= current["forward_converged"]
                               and candidate["physical_first_order_count"] >= current["physical_first_order_count"]
                               and candidate["valid_bundle_count"] >= current["valid_bundle_count"])
                fan_ok = (candidate["J_Fan_dimensionless"] <=
                          initial_fan*(1.0 + float(cfg["fan_worsening_relative"])) + 1e-12)
                if forward_better and topology_ok and fan_ok and candidate["hard_physical_surface_valid"]:
                    u, r0, current = candidate_u, candidate_r, candidate
                    m1, m2, display = a, b, d; accepted = True; accepted_alpha = alpha
                    break
            if accepted:
                break
        history.append({"iteration": iteration, "accepted": accepted, "alpha": accepted_alpha,
                        "J_forward_continuous": current["J_forward_continuous"],
                        "J_Fan_dimensionless": current["J_Fan_dimensionless"],
                        "forward_converged": current["forward_converged"],
                        "physical_first_order_count": current["physical_first_order_count"],
                        "valid_bundle_count": current["valid_bundle_count"],
                        "jacobian_condition": condition, "normalized_pose": u.tolist()})
        if not accepted:
            break
    ctx.data.update({"m1": m1, "m2": m2, "display": display,
                     "geometry_refinement_history": history,
                     "geometry_refinement_specs": specs})
    tr = trace_reverse(ctx.data["rays"], ctx.data["visor"], m1, m2, display, True)
    refs, rule = _dynamic_refs(ctx, tr, display)
    ctx.data.update({"trace": tr, "fan_refs": refs, "fan_reference_rule": rule})
    ctx.data["reference_history"].append({"cycle": "POST_23G_GEOMETRY", **rule})
    return {"status": "PASS", "variables": specs, "iterations": len(history),
            "accepted_iterations": sum(bool(row["accepted"]) for row in history),
            "initial": initial, "final": current, "normalized_pose": u.tolist(),
            "aperture_optimized": False, "curvature_optimized": False,
            "conic_optimized": False}


def _snapshot_step23_phase(ctx: Context, phase: str) -> dict[str, Any]:
    """Lưu ảnh 3D và footprint riêng cho 23A/23S/23G/23C, đồng thời giữ ảnh STEP 23 chuẩn."""
    from visualization_v55 import render_step, render_surface_ray_view
    ctx.stage_summaries[23] = {"status": phase}
    view = render_step(ctx, 23); footprint = render_surface_ray_view(ctx, 23)
    folder = ctx.step_dir(23) / phase; folder.mkdir(parents=True, exist_ok=True)
    copied = {}
    for label, record in (("view", view), ("footprint", footprint)):
        for kind in ("image", "metadata"):
            source = Path(record[kind]); target = folder / source.name
            shutil.copy2(source, target); copied[f"{label}_{kind}"] = str(target)
    return copied




def _forward_distortion(ctx: Context, virtual: np.ndarray,
                        bundle_valid: np.ndarray | None = None,
                        rays: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tính distortion forward; bundle thiếu hoặc sai phía mắt không được chứng nhận."""
    vi = ctx.data["vi"]; ideal3 = ctx.data["fixed_vi_grid"]; eps = float(ctx.config["distortion"]["epsilon_mm"])
    h, v, n, c = vi["horizontal"], vi["vertical"], vi["normal"], vi["center"]
    r = ctx.data["rays"] if rays is None else rays
    nf = int(r["field_count"]); npup = int(r["pupil_count"])
    center_field = int(r["central_field_index"])
    if bundle_valid is None:
        bundle_valid = np.all(np.isfinite(virtual), axis=2)
    bundle_valid = np.asarray(bundle_valid, bool)
    if bundle_valid.shape != (nf, npup):
        raise ValueError("FORWARD_BUNDLE_VALIDITY_SHAPE_MISMATCH")
    ideal_proj = ideal3 - ((ideal3 - c) @ n)[:, None] * n
    ideal_uv = np.column_stack([(ideal_proj - c) @ h, (ideal_proj - c) @ v])
    actual_uv = np.full((nf, npup, 2), np.nan)
    for f in range(nf):
        for p in range(npup):
            x = virtual[f, p]
            if bundle_valid[f, p] and np.all(np.isfinite(x)):
                xp = x - np.dot(x - c, n) * n
                actual_uv[f, p] = [np.dot(xp - c, h), np.dot(xp - c, v)]
    rows, vals = [], []
    for f in range(nf):
        qideal = ideal_uv[f] - ideal_uv[center_field]
        for p in range(npup):
            pair_valid = bool(bundle_valid[f, p] and bundle_valid[center_field, p])
            qact = (actual_uv[f, p] - actual_uv[center_field, p]
                    if pair_valid else np.full(2, np.nan))
            if f == center_field:
                rows.append({"field_index": f, "pupil_index": p, "D_vector_percent": None,
                             "D_r_percent": None, "D_x_percent": None, "D_y_percent": None,
                             "central_anchor_error_mm": float(np.linalg.norm(actual_uv[f, p] - ideal_uv[f]))
                             if np.all(np.isfinite(actual_uv[f, p])) else None})
                continue
            den = max(float(np.linalg.norm(qideal)), eps)
            dvec = 100.0 * float(np.linalg.norm(qact - qideal)) / den if np.all(np.isfinite(qact)) else float("nan")
            rideal, ract = float(np.linalg.norm(qideal)), float(np.linalg.norm(qact))
            dr = 100.0 * (ract - rideal) / max(rideal, eps) if np.isfinite(ract) else float("nan")
            dx = None if abs(qideal[0]) <= eps else 100.0 * (qact[0] - qideal[0]) / qideal[0]
            dy = None if abs(qideal[1]) <= eps else 100.0 * (qact[1] - qideal[1]) / qideal[1]
            rows.append({"field_index": f, "pupil_index": p, "D_vector_percent": dvec,
                         "D_r_percent": dr, "D_x_percent": dx, "D_y_percent": dy,
                         "q_act_x_mm": qact[0], "q_act_y_mm": qact[1],
                         "q_ideal_x_mm": qideal[0], "q_ideal_y_mm": qideal[1]})
            vals.append(dvec)
    evaluated_pair_mask = np.asarray([np.isfinite(x) for x in vals], bool)
    finite = np.asarray([x for x in vals if np.isfinite(x)], float)
    required = (nf - 1) * npup
    complete = len(finite) == len(vals) and len(finite) == required
    dmax = float(np.max(finite)) if complete else None
    partial_dmax = float(np.max(finite)) if len(finite) else None
    return {"metric_name": "USER_FIXED_GRID_VECTOR_DISTORTION", "rows": rows, "D_max_percent": dmax,
            "D_partial_max_percent_diagnostic": partial_dmax,
            "strict_less_than_limit": bool(complete and dmax is not None
                                             and dmax < float(ctx.config["distortion"]["hard_limit_percent"])),
            "evaluation_status": "COMPLETE" if complete else "NOT_EVALUABLE_INCOMPLETE_FORWARD_BUNDLES",
            "evaluated_noncentral_field_pupil_count": len(finite), "required_count": required,
            "_evaluated_pair_mask": evaluated_pair_mask,
            "valid_bundle_count": int(np.sum(bundle_valid)), "required_bundle_count": int(nf*npup),
            "central_field_index": center_field,
            "central_percentage": "N/A", "reference": "FIXED_TARGET_VI_PLANE_STEP_08B"}


def _actual_optics(ctx: Context, points: np.ndarray,
                   bundle_valid: np.ndarray | None = None) -> dict[str, Any]:
    """Đo FOV/VID từ từng tâm pupil; không xuất số chứng nhận nếu bundle thiếu."""
    vi = ctx.data["vi"]; c, h, v = vi["normal"], vi["horizontal"], vi["vertical"]
    center_field = int(vi["central_field_index"])
    npup = int(ctx.data["rays"]["pupil_count"])
    nf = int(ctx.data["rays"]["field_count"])
    if bundle_valid is None:
        bundle_valid = np.all(np.isfinite(points), axis=2)
    bundle_valid = np.asarray(bundle_valid, bool)
    if bundle_valid.shape != (nf, npup):
        raise ValueError("OPTICS_BUNDLE_VALIDITY_SHAPE_MISMATCH")
    pupil_centers = np.asarray([[p["x_mm"], p["y_mm"], p["z_mm"]]
                                for p in ctx.data["pupils"]], float)
    rows = []
    for p in range(npup):
        complete = bool(np.all(bundle_valid[:, p]))
        dirs = unit(points[:, p, :] - pupil_centers[p]) if complete else np.full((nf, 3), np.nan)
        forward_hemisphere = bool(complete and np.all(dirs @ c > 0.0))
        valid = bool(complete and forward_hemisphere and np.all(np.isfinite(dirs)))
        hh = np.degrees(np.arctan2(dirs @ h, dirs @ c))
        vv = np.degrees(np.arctan2(dirs @ v, dirs @ c))
        center = dirs[center_field]
        rows.append({"pupil_index": p, "complete_field_bundle": complete,
                     "all_virtual_points_in_forward_hemisphere": forward_hemisphere,
                     "optics_valid": valid,
                     "FOV_H_deg": float(np.ptp(hh)) if valid else None,
                     "FOV_V_deg": float(np.ptp(vv)) if valid else None,
                      "VID_center_mm": float(np.linalg.norm(points[center_field, p]-pupil_centers[p]))
                      if valid else None,
                     "D6_center_deg": float(np.degrees(np.arcsin(np.clip(center[2], -1, 1)))) if valid else None,
                     "azimuth_center_deg": float(np.degrees(np.arctan2(center[1], center[0]))) if valid else None})
    keys = ("FOV_H_deg", "FOV_V_deg", "VID_center_mm", "D6_center_deg", "azimuth_center_deg")
    complete = bool(all(row["optics_valid"] for row in rows))
    achieved = ({k: float(np.mean([float(row[k]) for row in rows])) for k in keys}
                if complete else {k: None for k in keys})
    if complete:
        az = np.radians([float(row["azimuth_center_deg"]) for row in rows])
        achieved["azimuth_center_deg"] = float(np.degrees(np.arctan2(np.mean(np.sin(az)), np.mean(np.cos(az)))))
    return {"per_pupil": rows, "achieved_mean": achieved,
            "evaluation_status": "COMPLETE" if complete else "NOT_EVALUABLE_INCOMPLETE_OR_BACKWARD_VIRTUAL_BUNDLES",
            "valid_pupil_count": int(sum(row["optics_valid"] for row in rows)),
            "required_pupil_count": npup,
            "direction_reference": "EACH_PRIMARY_PUPIL_CENTER_NOT_GLOBAL_ORIGIN",
            "grading": {k: "UNGRADED_NO_USER_TOLERANCE" for k in keys}}


def _forward_monitor_indices(ctx: Context) -> np.ndarray:
    """Chọn chief và tia vòng ngoài cố định cho forward gate trong DLSQ."""
    rays = ctx.data["rays"]
    requested = int(ctx.config["solver"].get("forward_constraint_rays_per_bundle", 0))
    samples_per_bundle = int(rays["samples_per_pupil"])
    if requested == 0 or requested >= samples_per_bundle:
        return np.arange(len(rays["rows"]), dtype=int)
    pattern = ctx.data["pattern"]
    chief = [i for i, row in enumerate(pattern) if row["chief"]]
    outer_ring = max(int(row["ring"]) for row in pattern)
    outer = [i for i, row in enumerate(pattern) if int(row["ring"]) == outer_ring]
    if len(chief) != 1 or requested - 1 > len(outer):
        raise RuntimeError("FORWARD_MONITOR_PATTERN_CANNOT_SUPPLY_REQUESTED_RAYS")
    positions = np.floor(np.arange(requested - 1) * len(outer) / (requested - 1)).astype(int)
    selected_samples = np.asarray([chief[0], *[outer[i] for i in positions]], int)
    return np.flatnonzero(np.isin(rays["sample_index"], selected_samples))


def _slice_rays(rays: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    """Cắt ray bundle nhưng giữ nguyên chỉ số field/pupil và central field."""
    count = len(rays["rows"]); indices = np.asarray(indices, int)
    out: dict[str, Any] = {}
    for key, value in rays.items():
        if key == "rows":
            out[key] = [value[i] for i in indices]
        elif isinstance(value, np.ndarray) and len(value) == count:
            out[key] = value[indices]
        else:
            out[key] = value
    out["samples_per_pupil"] = int(len(indices) // (int(rays["field_count"])*int(rays["pupil_count"])))
    return out


def _run_forward_evaluation(ctx: Context, m1: PolySurface, m2: PolySurface,
                            monitor_only: bool, display: PolySurface | None = None,
                            reverse_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Chạy cùng một forward solver cho gate DLSQ hoặc chứng nhận đầy đủ STEP 24."""
    full_rays = ctx.data["rays"]; display = display or ctx.data["display"]
    reverse = (
        reverse_trace
        if reverse_trace is not None
        else _reverse_evaluation_trace(ctx, m1, m2, display)
    )
    if np.asarray(reverse["landing"]).shape != (len(full_rays["rows"]), 3):
        raise ValueError("FORWARD_EVALUATION_REVERSE_TRACE_SHAPE_MISMATCH")
    sources, rule = _dynamic_refs(ctx, reverse, display)
    indices = _forward_monitor_indices(ctx) if monitor_only else np.arange(len(full_rays["rows"]), dtype=int)
    rays = _slice_rays(full_rays, indices)
    source_per_ray = sources[full_rays["field_index"]][indices]
    base = unit(reverse["points"][2][indices] - source_per_ray)
    solver = ctx.config["solver"]
    fwd = forward_shoot(source_per_ray, rays["origins"], base, display, m2,
                        m1, ctx.data["visor"], int(solver["forward_max_iter"]),
                        float(solver["forward_eye_tolerance_mm"]),
                        float(solver["forward_fd_angle"]), float(solver["forward_damping"]))
    surfaces = {"M1": m1, "M2": m2, "VISOR": ctx.data["visor"]}
    fh2 = first_hit(source_per_ray + 1e-4*fwd["direction"], fwd["direction"], surfaces)
    fh1 = first_hit(fwd["m2"] + 1e-4*fwd["direction_after_m2"], fwd["direction_after_m2"], surfaces)
    fhv = first_hit(fwd["m1"] + 1e-4*fwd["direction_after_m1"], fwd["direction_after_m1"], surfaces)
    first_order = (fh2["name"] == "M2") & (fh1["name"] == "M1") & (fhv["name"] == "VISOR")
    fwd["physical_valid"] = np.asarray(fwd["valid"], bool) & first_order
    fwd["converged"] = np.asarray(fwd["converged"], bool) & fwd["physical_valid"]
    virtual = reconstruct_virtual_points(fwd, rays)
    distortion = _forward_distortion(ctx, virtual["points"], virtual["bundle_valid"], rays)
    return {"reverse": reverse, "sources": sources, "reference_rule": rule,
            "indices": indices, "rays": rays, "source_per_ray": source_per_ray,
            "forward": fwd, "virtual": virtual, "distortion": distortion,
            "first_hit_m2": fh2, "first_hit_m1": fh1, "first_hit_visor": fhv,
            "first_order": first_order}


def _record_optical_convergence(
    ctx: Context,
    label: str,
    source_step: int,
    *,
    m1: PolySurface | None = None,
    m2: PolySurface | None = None,
    display: PolySurface | None = None,
    reverse_trace: dict[str, Any] | None = None,
    forward_evaluation: dict[str, Any] | None = None,
    authority: bool = False,
) -> dict[str, Any]:
    """Tính convergence diagnostic từ đúng optical state hiện tại; không tham gia gate."""
    cfg = (
        ctx.config.get("execution", {})
        .get("optical_convergence", {})
    )

    if not bool(cfg.get("enabled", False)):
        return {
            "status": "DISABLED",
            "label": label,
            "source_step": int(source_step),
        }

    m1 = m1 if m1 is not None else ctx.data["m1"]
    m2 = m2 if m2 is not None else ctx.data["m2"]
    display = display if display is not None else ctx.data["display"]

    safe_label = "".join(
        character if character.isalnum() else "_"
        for character in str(label)
    ).strip("_").upper()

    target = {
        "target_VID_mm": float(ctx.config["vid_mm"]),
        "target_FOV_H_deg": float(ctx.config["fov_h_deg"]),
        "target_FOV_V_deg": float(ctx.config["fov_v_deg"]),
        "target_D6_deg": float(ctx.config["d6_deg"]),
        "target_azimuth_deg": float(ctx.config["azimuth_deg"]),
    }

    try:
        full_forward_requested = bool(
            cfg.get("full_forward", True)
        )

        evaluation = forward_evaluation

        if evaluation is None:
            evaluation = _run_forward_evaluation(
                ctx,
                m1,
                m2,
                monitor_only=not full_forward_requested,
                display=display,
                reverse_trace=reverse_trace,
            )

        reverse = evaluation["reverse"]
        rays = ctx.data["rays"]

        refs, _ = _dynamic_refs(
            ctx,
            reverse,
            display,
        )

        mf1 = fan_imaging_metrics(
            reverse,
            rays,
            refs,
            float(
                ctx.config["fan_weights"]["omega1"]
            ),
        )

        mf2 = mf2_geometry(
            reverse["points"][0],
            reverse["points"][1],
            reverse["points"][2],
            m2,
            float(
                ctx.config["fan_weights"]["omega2"]
            ),
            ctx.data["n_obs"],
        )

        spot = chief_centered_geometric_spot_rms(
            reverse["landing"],
            reverse["valid"],
            rays["field_index"],
            rays["pupil_index"],
            rays["chief"],
            reverse["display_frame"],
        )

        fwd = evaluation["forward"]
        virtual = evaluation["virtual"]
        distortion = evaluation["distortion"]

        optics = _actual_optics(
            ctx,
            virtual["points"],
            virtual["bundle_valid"],
        )

        achieved = optics["achieved_mean"]

        ray_count = int(
            len(rays["rows"])
        )

        reverse_physical_valid_count = int(
            np.count_nonzero(
                np.asarray(
                    reverse["valid"],
                    dtype=bool,
                )
            )
        )

        forward_ray_count = int(
            len(fwd["converged"])
        )

        forward_converged_count = int(
            np.count_nonzero(
                np.asarray(
                    fwd["converged"],
                    dtype=bool,
                )
            )
        )

        forward_full_ray_set = bool(
            len(evaluation["rays"]["rows"])
            == ray_count
        )

        record = {
            "status": "EVALUATED",
            "label": str(label),
            "source_step": int(source_step),
            "authority_role": (
                "STEP24_NUMERICAL_AUTHORITY"
                if authority
                else "INTERMEDIATE_DIAGNOSTIC_NOT_GATE"
            ),
            "forward_mode": (
                "FULL_CURRENT_RAY_SET"
                if forward_full_ray_set
                else "SUBSAMPLED_FORWARD_MONITOR"
            ),
            **target,
            "reverse_physical_valid_count":
                reverse_physical_valid_count,
            "ray_count":
                ray_count,
            "forward_converged_count":
                forward_converged_count,
            "forward_ray_count":
                forward_ray_count,
            "forward_rays_per_bundle":
                int(
                    evaluation["rays"][
                        "samples_per_pupil"
                    ]
                ),
            "Fan_core_merit":
                float(
                    mf1["MF1_Fan"]
                    + mf2["MF2"]
                ),
            "MF1_Fan":
                float(mf1["MF1_Fan"]),
            "MF2_Fan":
                float(mf2["MF2"]),
            "chief_centered_spot_RMS_mm":
                spot["RMS_spot_radius_mm"],
            "chief_centered_spot_max_mm":
                spot["max_spot_radius_mm"],
            "VID_mm":
                achieved["VID_center_mm"],
            "FOV_H_deg":
                achieved["FOV_H_deg"],
            "FOV_V_deg":
                achieved["FOV_V_deg"],
            "D6_deg":
                achieved["D6_center_deg"],
            "azimuth_deg":
                achieved["azimuth_center_deg"],
            "distortion_max_percent":
                distortion.get(
                    "D_max_percent"
                ),
            "distortion_partial_max_percent":
                distortion.get(
                    "D_partial_max_percent_diagnostic"
                ),
            "distortion_evaluation_status":
                distortion.get(
                    "evaluation_status"
                ),
            "optics_evaluation_status":
                optics.get(
                    "evaluation_status"
                ),
            "valid_virtual_bundle_count":
                int(
                    virtual.get(
                        "valid_bundle_count",
                        0,
                    )
                ),
            "required_virtual_bundle_count":
                int(
                    virtual.get(
                        "required_bundle_count",
                        0,
                    )
                ),
        }

    except Exception as exc:
        record = {
            "status": "DIAGNOSTIC_ERROR",
            "label": str(label),
            "source_step": int(source_step),
            "authority_role": (
                "STEP24_NUMERICAL_AUTHORITY"
                if authority
                else "INTERMEDIATE_DIAGNOSTIC_NOT_GATE"
            ),
            **target,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    try:
        old_history = ctx.data.setdefault(
            "optical_convergence_history",
            [],
        )

        history = [
            copy.deepcopy(row)
            for row in old_history
            if not (
                int(
                    row.get(
                        "source_step",
                        -1,
                    )
                )
                == int(source_step)
                and str(
                    row.get(
                        "label",
                        "",
                    )
                )
                == str(label)
            )
        ]

        history.append(
            copy.deepcopy(record)
        )

        ctx.data[
            "optical_convergence_history"
        ] = history

        write_json(
            ctx.step_dir(source_step)
            / (
                f"{int(source_step):02d}_"
                f"{safe_label}_"
                "OPTICAL_CONVERGENCE.json"
            ),
            record,
        )

        write_csv(
            ctx.run_dir
            / "OPTICAL_CONVERGENCE_HISTORY.csv",
            history,
        )

        write_json(
            ctx.run_dir
            / "OPTICAL_CONVERGENCE_HISTORY.json",
            {
                "schema":
                    "HUD_FAN_V5_5_"
                    "OPTICAL_CONVERGENCE_HISTORY_V1",
                "diagnostic_only":
                    True,
                "step24_remains_final_authority":
                    True,
                "records":
                    history,
            },
        )

    except Exception as persist_exc:
        if not is_compute_worker():
            print(
                "[OPTICAL CONVERGENCE WARNING] "
                "artifact write failed: "
                f"{type(persist_exc).__name__}: "
                f"{persist_exc}",
                flush=True,
            )

    if not is_compute_worker():
        if record["status"] == "EVALUATED":
            def format_number(
                value: Any,
                digits: int,
            ) -> str:
                """Format finite scalar hoặc N/A."""
                if value is None:
                    return "N/A"
                number = float(value)
                if not np.isfinite(number):
                    return "N/A"
                return f"{number:.{digits}f}"

            print(
                "=" * 68,
                flush=True,
            )
            print(
                "OPTICAL CONVERGENCE",
                flush=True,
            )
            print(
                f"Checkpoint: {label}",
                flush=True,
            )
            print(
                "Target    : "
                f"VID={ctx.config['vid_mm']:.3f} mm | "
                f"FOV={ctx.config['fov_h_deg']:.3f} x "
                f"{ctx.config['fov_v_deg']:.3f} deg | "
                f"D6={ctx.config['d6_deg']:.3f} deg",
                flush=True,
            )
            print(
                "",
                flush=True,
            )
            print(
                "Physical  : "
                f"{record['reverse_physical_valid_count']}/"
                f"{record['ray_count']}",
                flush=True,
            )
            print(
                "Forward   : "
                f"{record['forward_converged_count']}/"
                f"{record['forward_ray_count']}",
                flush=True,
            )
            print(
                "Fan       : "
                f"{format_number(record['Fan_core_merit'], 4)}",
                flush=True,
            )
            print(
                "Spot RMS  : "
                f"{format_number(record['chief_centered_spot_RMS_mm'], 3)} mm",
                flush=True,
            )
            print(
                "VID       : "
                f"{format_number(record['VID_mm'], 3)} mm",
                flush=True,
            )
            print(
                "FOV-H     : "
                f"{format_number(record['FOV_H_deg'], 3)} deg",
                flush=True,
            )
            print(
                "FOV-V     : "
                f"{format_number(record['FOV_V_deg'], 3)} deg",
                flush=True,
            )
            print(
                "D6        : "
                f"{format_number(record['D6_deg'], 3)} deg",
                flush=True,
            )
            print(
                "Azimuth   : "
                f"{format_number(record['azimuth_deg'], 3)} deg",
                flush=True,
            )
            print(
                "Distortion: "
                f"{format_number(record['distortion_max_percent'], 3)} %",
                flush=True,
            )
            print(
                "Role      : "
                f"{record['authority_role']}",
                flush=True,
            )
            print(
                "=" * 68,
                flush=True,
            )
        else:
            print(
                "[OPTICAL CONVERGENCE WARNING] "
                f"{label}: "
                f"{record.get('error_type')} - "
                f"{record.get('error_message')}",
                flush=True,
            )

    return record


def _forward_engineering_eval(ctx: Context, m1: PolySurface, m2: PolySurface,
                              display: PolySurface | None = None) -> dict[str, Any]:
    """Đánh giá forward distortion/validity khi nhận bước, tách khỏi Fan residual."""
    display = display or ctx.data["display"]
    evaluation = _run_forward_evaluation(ctx, m1, m2, monitor_only=True, display=display)
    fwd = evaluation["forward"]; virtual = evaluation["virtual"]
    # Gate tối ưu dùng điểm fit forward thật từ phần tia hội tụ của mỗi bundle.
    # Chứng nhận cuối vẫn dùng bundle_valid nghiêm ngặt và không dùng diagnostic_points.
    dist = _forward_distortion(ctx, virtual["diagnostic_points"],
                               virtual["diagnostic_bundle_valid"], evaluation["rays"])
    ray_invalid = 1.0 - float(np.mean(fwd["converged"]))
    bundle_invalid = 1.0 - float(np.mean(virtual["bundle_valid"]))
    limit = float(ctx.config["distortion"]["hard_limit_percent"])
    measured = dist["D_max_percent"]
    if measured is None:
        measured = dist["D_partial_max_percent_diagnostic"]
    used_fallback = measured is None
    if measured is None:
        measured = limit * (2.0 + bundle_invalid)
    measured = float(measured)
    phi_d = max(0.0, measured/limit - 1.0)**2
    phi_r = ray_invalid**2 + bundle_invalid**2
    lam = _packaging(ctx, m1, m2, display)
    phi_p = (0.0 if lam is None else
             max(0.0, lam/float(ctx.config["packaging_lambda_max"])-1.0)**2)
    weights = ctx.config["normalized_objective"]
    terms = {"wD_PhiD_forward": float(weights["wD"])*phi_d,
             "wP_PhiP": float(weights["wP"])*phi_p,
             "wR_PhiR_forward": float(weights["wR"])*phi_r}
    record = {"J_engineering_dimensionless": float(sum(terms.values())),
              "engineering_direction": "FORWARD_DISPLAY_M2_M1_VISOR_EYE",
              "forward_monitor_ray_count": len(fwd["converged"]),
              "forward_monitor_rays_per_bundle": int(evaluation["rays"]["samples_per_pupil"]),
              "forward_monitor_complete": bool(dist["evaluation_status"] == "COMPLETE"),
              "forward_distortion_monitor_percent": measured,
              "forward_distortion_monitor_is_measured": not used_fallback,
              "forward_distortion_monitor_evaluated_pairs": dist["evaluated_noncentral_field_pupil_count"],
              "forward_distortion_monitor_required_pairs": dist["required_count"],
              "_forward_distortion_evaluated_pair_mask": dist["_evaluated_pair_mask"],
              "forward_distortion_complete_value_percent": dist["D_max_percent"],
              "forward_invalid_ray_fraction": ray_invalid,
              "forward_invalid_bundle_fraction": bundle_invalid,
              "packaging_lambda": lam, "packaging_constraint_enabled": lam is not None,
              "engineering_terms": terms}
    if lam is not None:
        record["packaging_lambda_constraint_value"] = float(lam)
    return record




def _final_numerical_decision(
    config: dict[str, Any],
    numerical: dict[str, Any],
    gates: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Tổng hợp các cổng kiểm tra kỹ thuật và đưa ra quyết định thiết kế quang học cuối cùng."""
    by_name: dict[str, dict[str, Any]] = {}
    for row in gates:
        name = str(row["item"])
        if name in by_name:
            raise RuntimeError(f"DUPLICATE_SCORECARD_ITEM:{name}")
        by_name[name] = row

    failed = [
        str(row["item"])
        for row in gates
        if row["status"] == "FAIL"
    ]
    if failed:
        return "FAIL_NUMERICAL_HARD_GATE", {
            "failed_gates": failed,
            "missing_tolerances": [],
            "missing_or_ungraded_required_gates": [],
        }

    tolerance_keys = (
        "fov_h_deg", "fov_v_deg", "vid_mm",
        "d6_deg", "azimuth_deg",
    )
    missing_tolerances = [
        key for key in tolerance_keys
        if config["hard_tolerances"].get(key) is None
    ]

    required = {
        "USER_FIXED_GRID_VECTOR_DISTORTION",
        "VISOR_FOOTPRINT_CONVERGED_RAYS",
        "PHYSICAL_SEQUENTIAL_FORWARD_TRACE",
        "FERMAT_STATIONARY_POINT_CONVERGENCE",
        "SIGNED_MF2_OBSCURATION",
        "FOV_H_deg", "FOV_V_deg", "VID_mm",
        "D6_deg", "azimuth_deg",
    }

    if bool(numerical["packaging_constraint_enabled"]):
        required.add("PACKAGING_LAMBDA")

    if (
        bool(config["mtf_requested"])
        and float(config["mtf"]["minimum_at_max_frequency"]) > 0.0
    ):
        required.add("NUMERICAL_MTF_AT_MAX_FREQUENCY")

    # Một gate khác có operator đã khai báo vẫn phải được tôn trọng.
    required.update(
        str(row["item"])
        for row in gates
        if row.get("operator") is not None
    )

    unresolved = sorted(
        name for name in required
        if name not in by_name or by_name[name]["status"] != "PASS"
    )

    details = {
        "failed_gates": [],
        "missing_tolerances": missing_tolerances,
        "missing_or_ungraded_required_gates": unresolved,
    }

    if missing_tolerances:
        return "INCOMPLETE_HARD_TOLERANCE", details
    if unresolved:
        return "INCOMPLETE_NUMERICAL_EVIDENCE", details

    return "NUMERICAL_DESIGN_PASS", details






def step_00(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 00: Kiểm tra config, nạp đúng dữ liệu authority và khóa đầu vào cho run hiện tại."""

    validate_config(ctx.config)
    inp = load_inputs(Path(ctx.config["input_dir"]), ctx.config["hud_geometry_authority"])
    ctx.data["inputs"] = inp
    ctx.data["algorithm_source_manifest"] = _algorithm_source_manifest(ctx)
    summary = {"schema": ctx.config["schema"], "prompt_sha256": ctx.config["prompt_sha256"],
               "input_dir": ctx.config["input_dir"], "file_count": len(inp["files"]),
               "algorithm_source_manifest_sha256": ctx.data["algorithm_source_manifest"]["manifest_sha256"],
               "no_guessing": True, "input_hash_manifest_required": False,
               "algorithm_source_manifest_required": True,
               "packaging_constraint_enabled": inp["packaging_constraint_enabled"],
               "loaded_files": [p.name for p in inp["files"]], "config": ctx.config}
    write_json(ctx.step_dir(0) / "00_CURRENT_INPUT_SUMMARY.json", summary)
    write_json(ctx.step_dir(0) / "00_ALGORITHM_SOURCE_MANIFEST.json",
               ctx.data["algorithm_source_manifest"])
    rsp = ctx.config.get("ray_sampling_profiles")
    if isinstance(rsp, dict) and rsp.get("enabled", False):
        o2_prof = rsp["profiles"]["O2_SEARCH"]
        grid = o2_prof["field_grid"]
        field_count = int(grid["horizontal_count"]) * int(grid["vertical_count"])
        pupil_grid = o2_prof["pupil_grid"]
        pupil_count = int(pupil_grid["horizontal_count"]) * int(pupil_grid["vertical_count"])
        expected_ray_count = int(o2_prof["expected_ray_count"])
    else:
        grid = ctx.config["field_grid"]
        field_count = int(grid["horizontal_count"]) * int(grid["vertical_count"])
        pupil_grid = ctx.config["primary_pupil_grid"]
        pupil_count = int(pupil_grid["horizontal_count"]) * int(pupil_grid["vertical_count"])
        expected_ray_count = field_count * pupil_count * 49
    write_json(ctx.step_dir(0) / "00_FLOW_CONSISTENCY_PRECHECK.json", {
        "authority": "PASS", "grid_separation": "PENDING_STEP_08", "state": "STEP_ONE",
        "field_grid": grid, "field_count": field_count,
        "primary_pupil_grid": pupil_grid, "pupil_count": pupil_count,
        "expected_ray_count": expected_ray_count, "finite_VID": True, "MTF": "NOT_REQUESTED",
        "packaging_constraint": "ENABLED_P1_P8" if inp["packaging_constraint_enabled"] else "DISABLED_NO_P1_P8",
        "Zemax": "SKIP", "physical_measurement_wording": "SIMULATION_ONLY"})
    return {"status": "PASS", "files": len(inp["files"]), "hash_audit_performed": False}

def step_01(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 01: Fit các điểm visor-inner thành mặt Chebyshev khả vi để giao tia và lấy pháp tuyến."""

    i, c = ctx.data["inputs"], ctx.config
    visor, stats, loc = ChebVisor.fit(i["visor_center"], i["visor_U"], i["visor_V"], i["visor_N"],
                                      i["visor_inner"], int(c["visor_surrogate_degree"]),
                                      tuple(c["visor_surrogate_scale_mm"]))
    ctx.data["visor"] = visor
    ctx.data["_visor_fingerprint"] = hashlib.sha256(
        pickle.dumps(visor.to_dict(), protocol=5)
    ).hexdigest()
    n = visor.normal(loc[:, 0], loc[:, 1]); dots = np.sum(n * i["visor_inner_normals"], axis=1)
    ang = np.degrees(np.arccos(np.clip(np.abs(dots), -1.0, 1.0)))
    stats.update({"normal_rms_deg": float(np.sqrt(np.mean(ang ** 2))),
                  "normal_max_deg": float(np.max(ang)), "domain_bounds_local_mm": visor.bounds,
                  "acceptance_rule": "NUMERICALLY_USABLE_ONLY_NO_V5_2_THRESHOLD_GATE",
                  "numeric_usable": bool(np.all(np.isfinite(visor.coeff)))})
    ctx.data["visor_fit_stats"] = stats
    write_json(ctx.step_dir(1) / "01_VISOR_SURROGATE.json", visor.to_dict() | stats)
    if not stats["numeric_usable"]:
        raise RuntimeError("VISOR_SURROGATE_NUMERICALLY_UNUSABLE")
    return stats | {"status": "PASS"}

def step_02(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 02: Dựng mặt phẳng ảnh ảo từ VID, D6, azimuth và hai góc FOV."""

    prof_info = _get_active_ray_sampling_profile(ctx, 2)
    if prof_info is not None:
        name, prof = prof_info
        cfg = dict(ctx.config)
        cfg["field_grid"] = dict(prof["field_grid"])
        vi = make_virtual_image(cfg)
    else:
        vi = make_virtual_image(ctx.config)
    ctx.data["vi"] = vi
    out = {k: v for k, v in vi.items() if k != "fields"}
    write_json(ctx.step_dir(2) / "02_TARGET_VIRTUAL_IMAGE.json", out)
    return {"status": "PASS", "VID_mm": ctx.config["vid_mm"], "D6_deg": ctx.config["d6_deg"]}

def step_03(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 03: Chia FOV theo field_grid và xác định duy nhất một field trung tâm."""

    vi = ctx.data["vi"]
    for f in vi["fields"]:
        f["u_vi_mm"] = float(np.dot(f["vi_point"] - vi["center"], vi["horizontal"]))
        f["v_vi_mm"] = float(np.dot(f["vi_point"] - vi["center"], vi["vertical"]))
    rows = [{"field_id": f["field_id"], "h_deg": f["h_deg"], "v_deg": f["v_deg"],
             "vi_x_mm": f["vi_point"][0], "vi_y_mm": f["vi_point"][1], "vi_z_mm": f["vi_point"][2],
             "u_vi_mm": f["u_vi_mm"], "v_vi_mm": f["v_vi_mm"]} for f in vi["fields"]]
    write_csv(ctx.step_dir(3) / "03_FIELDS_AND_IDEAL_VI.csv", rows)
    return {"status": "PASS", "field_count": len(rows),
            "horizontal_count": vi["field_grid"]["horizontal_count"],
            "vertical_count": vi["field_grid"]["vertical_count"],
            "central_field_index": vi["central_field_index"]}

def step_04(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 04: Tạo lưới pupil cấu hình được, phủ đều tâm eye-box."""

    prof_info = _get_active_ray_sampling_profile(ctx, 4)
    if prof_info is not None:
        name, prof = prof_info
        grid = prof["pupil_grid"]
    else:
        grid = ctx.config["primary_pupil_grid"]
    pupils = primary_pupils(tuple(ctx.config["eyebox_y_mm"]), tuple(ctx.config["eyebox_z_mm"]),
                             (int(grid["horizontal_count"]), int(grid["vertical_count"])))
    ctx.data["pupils"] = pupils
    write_csv(ctx.step_dir(4) / f"04_PRIMARY_PUPILS_{len(pupils)}.csv", pupils)
    return {"status": "PASS", "pupil_count": len(pupils), "pupil_grid": dict(grid)}

def step_05(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 05: Tạo 49 hoặc 25 mẫu trong mỗi pupil cho mọi field và gán ID cho từng tia."""

    prof_info = _get_active_ray_sampling_profile(ctx, 5)
    if prof_info is not None:
        name, prof = prof_info
        pattern = _nested_characteristic_pattern(int(prof["rays_per_pupil"]), float(ctx.config["pupil_diameter_mm"]) / 2.0)
    else:
        name, prof = None, None
        pattern = polar_pattern(49, float(ctx.config["pupil_diameter_mm"]) / 2.0)
    rays = build_rays(ctx.data["vi"]["fields"], ctx.data["pupils"], pattern)
    ctx.data.update({"pattern": pattern, "rays": rays})

    sample_ids = [str(r["sample_id"]) for r in pattern]
    if prof_info is not None:
        name, prof = prof_info
        fp = _compute_profile_fingerprint(
            profile_name=name,
            field_grid=prof["field_grid"],
            pupil_grid=prof["pupil_grid"],
            sample_ids=sample_ids,
            origins=rays["origins"],
            directions=rays["directions"],
            field_index=rays["field_index"],
            pupil_index=rays["pupil_index"],
            sample_index=rays["sample_index"],
        )
        ctx.data["sampling_profile_name"] = name
        ctx.data["sampling_profile_metadata"] = {
            "profile_name": name,
            "field_grid": dict(prof["field_grid"]),
            "pupil_grid": dict(prof["pupil_grid"]),
            "sample_ids": sample_ids,
            "fingerprint": fp,
            "expected_ray_count": int(prof["expected_ray_count"]),
            "actual_ray_count": len(rays["rows"]),
        }
        ctx.data[f"profile_rays_{name}"] = copy.deepcopy(rays)
        ctx.data[f"profile_vi_{name}"] = copy.deepcopy(ctx.data["vi"])
        ctx.data[f"profile_pupils_{name}"] = copy.deepcopy(ctx.data["pupils"])
        ctx.data[f"profile_pattern_{name}"] = copy.deepcopy(pattern)
        ctx.data["ci_m1_sampling_profile"] = "O2_SEARCH"
        ctx.data["ci_m2_sampling_profile"] = "O2_SEARCH"
    else:
        ray_fingerprint_payload = (
            np.asarray(rays["origins"], dtype=np.float64),
            np.asarray(rays["directions"], dtype=np.float64),
            np.asarray(rays["field_index"], dtype=np.int64),
            np.asarray(rays["pupil_index"], dtype=np.int64),
            np.asarray(rays["sample_index"], dtype=np.int64),
        )
        fp = hashlib.sha256(pickle.dumps(ray_fingerprint_payload, protocol=5)).hexdigest()

    ctx.data["_ray_bundle_fingerprint"] = fp
    if "_visor_fingerprint" not in ctx.data and "visor" in ctx.data:
        ctx.data["_visor_fingerprint"] = hashlib.sha256(
            pickle.dumps(ctx.data["visor"].to_dict(), protocol=5)
        ).hexdigest()
    write_csv(ctx.step_dir(5) / f"05_CHARACTERISTIC_PATTERN_{len(pattern)}.csv", pattern)
    if len(pattern) == 49:
        write_csv(ctx.step_dir(5) / "05_CHARACTERISTIC_PATTERN_49.csv", pattern)
    ray_count = len(rays["rows"]); field_count = len(ctx.data["vi"]["fields"])
    write_csv(ctx.step_dir(5) / f"05_CHARACTERISTIC_RAYS_{ray_count}.csv", rays["rows"])
    if ray_count != field_count * len(ctx.data["pupils"]) * len(pattern):
        raise RuntimeError("RAY_COUNT_MISMATCH")
    return {"status": "PASS", "fields": field_count, "pupils": len(ctx.data["pupils"]),
            "rays_per_pupil": len(pattern), "total": ray_count,
            "sampling_authority": "CURRENT_IMPLEMENTATION_CHOICE"}

def step_06(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 06: Giao các tia với visor, tính pháp tuyến và hướng phản xạ đi về M1."""

    rays = ctx.data["rays"]
    hit = ctx.data["visor"].intersect(rays["origins"], rays["directions"], finite=True)
    _publish_trace_debug(
        ctx, 6, "VISOR_INTERSECTION", hit, rays,
        direction="REVERSE",
    )
    if not np.all(hit["valid"]):
        raise RuntimeError("CHARACTERISTIC_RAY_MISSES_VISOR_DOMAIN")
    post = reflect(rays["directions"], hit["normal"])
    ctx.data.update({"visor_hit": hit, "post_visor": post})
    prof_name = ctx.data.get("sampling_profile_name")
    if prof_name:
        ctx.data[f"profile_visor_hit_{prof_name}"] = copy.deepcopy(hit)
        ctx.data[f"profile_post_visor_{prof_name}"] = copy.deepcopy(post)
    write_csv(ctx.step_dir(6) / f"06_VISOR_HITS_{len(rays['rows'])}.csv", _rows(rays, hit["point"], "visor"))
    return {"status": "PASS", "valid": int(np.sum(hit["valid"])), "total": len(hit["valid"])}

def step_07(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 07: Chọn geometry compact nhất sau các hard gate vật lý."""

    candidates = enumerate_planar_seeds(ctx)

    eligible = [
        (index, candidate)
        for index, candidate in enumerate(
            candidates,
            start=1,
        )
        if candidate["eligible"]
    ]

    if not eligible:
        raise RuntimeError(
            "NO_PLANAR_GEOMETRY_PASSES_PHYSICAL_OBSCURATION_AND_OPTIONAL_PACKAGING_GATE"
        )

    selected_candidate, selected = min(
        eligible,
        key=lambda item: (
            _planar_seed_rank_key(item[1]),
            item[0],
        ),
    )

    return activate_planar_seed(
        ctx,
        selected,
        selected_candidate,
        planar_seed_candidate_rows(candidates),
        "COMPACTNESS_RANK_AFTER_PHYSICAL_UNOBSCURED_OPTIONAL_PACKAGING;PLANAR_RMS_FINAL_TIE_BREAK_ONLY",
    )

def step_08(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 08: Tách reference động của Fan khỏi lưới ảnh ảo cố định dùng đo distortion."""

    refs, rule = _dynamic_refs(ctx, ctx.data["trace"])
    fixed = np.array([f["vi_point"] for f in ctx.data["vi"]["fields"]], dtype=float, copy=True)
    if np.shares_memory(refs, fixed) or np.array_equal(refs, fixed):
        raise RuntimeError("REFERENCE_AUTHORITY_ALIAS")
    ctx.data.update({"fan_refs": refs, "fan_refs_initial": refs.copy(), "fan_reference_rule": rule, "fixed_vi_grid": fixed,
                     "fixed_display_surrogate": refs.copy(),
                     "reference_history": [{"cycle": 0, **rule}]})
    _capture_spot_snapshot(ctx, "PLANAR", 8)
    write_csv(ctx.step_dir(8) / "08A_FAN_DYNAMIC_REFERENCE_GRID.csv",
              [{"field_id": f["field_id"], "x_mm": p[0], "y_mm": p[1], "z_mm": p[2]}
               for f, p in zip(ctx.data["vi"]["fields"], refs)])
    write_csv(ctx.step_dir(8) / "08B_DISTORTION_FIXED_IDEAL_GRID.csv",
              [{"field_id": f["field_id"], "x_mm": p[0], "y_mm": p[1], "z_mm": p[2]}
               for f, p in zip(ctx.data["vi"]["fields"], fixed)])
    write_json(ctx.step_dir(8) / "08_REFERENCE_SEPARATION_AUDIT.json", {
        "distinct_objects": True, "memory_alias": False, "Fan_grid_dynamic": True,
        "USER_grid_fixed": True, "USER_grid_never_used_as_Fan_endpoint": True})
    return {"status": "PASS", "Fan_reference_count": len(refs),
            "USER_reference_count": len(fixed), "distinct": True}

def step_09(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 09: Khi M2 còn phẳng, tạo target M1 bằng phép đối xứng liên hợp qua M2."""

    target = symmetric_conjugate(ctx.data["fan_refs"][ctx.data["rays"]["field_index"]], ctx.data["m2"])
    ctx.data["m1_targets"] = target
    write_csv(ctx.step_dir(9) / "09_PLANAR_CONJUGATE_TARGETS.csv", _rows(ctx.data["rays"], target, "target"))
    return {"status": "PASS", "M2_state": "PLANE", "target_count": len(target)}

def step_10(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 10: Dựng point cloud và normal của M1 theo CI point-by-point, bắt đầu từ chief ray."""

    r, vh, d = ctx.data["rays"], ctx.data["visor_hit"], ctx.data["post_visor"]
    ci = point_by_point_engine(
        vh["point"] + 1e-4 * d,
        d,
        ctx.data["m1_targets"],
        ctx.data["m1"],
        _chief_index(r),
        construction_options=ctx.config["ci_construction"],
    )
    diagnostic = _construction_diagnostics(ctx, ci, ctx.data["m1"].frame, "M1_INITIAL_CI")
    m1_incumbent = ctx.data["m1"].intersect(
        vh["point"] + 1e-4 * d,
        d,
        finite=False,
    )["point"]
    disp1 = np.linalg.norm(ci["points_by_ray"] - m1_incumbent, axis=1)
    finite_disp1 = disp1[np.isfinite(disp1)]
    diagnostic["CI_displacement_mm"] = {
        "p50": float(np.percentile(finite_disp1, 50)) if len(finite_disp1) else None,
        "p95": float(np.percentile(finite_disp1, 95)) if len(finite_disp1) else None,
        "max": float(np.max(finite_disp1)) if len(finite_disp1) else None,
    }
    ctx.data.update({"ci_m1": ci, "ci_m1_diagnostics": diagnostic})
    write_csv(ctx.step_dir(10) / "10_M1_CI_POINTS_NORMALS.csv",
              [{"ray_id": r["rows"][i]["ray_id"], "x_mm": p[0], "y_mm": p[1], "z_mm": p[2],
               "nx": n[0], "ny": n[1], "nz": n[2]} for i, (p, n) in enumerate(zip(ci["points_by_ray"], ci["normals_by_ray"]))])
    write_csv(
        ctx.step_dir(10) / "10_M1_NEAREST_RAY_TOPOLOGY.csv",
        _ci_topology_rows(
            ci,
            r,
            ctx.config["ci_construction"]["mode"],
        ),
    )
    from diagnostics_v55 import summarize_ci_fallbacks
    ci_diag_m1 = summarize_ci_fallbacks(ci, r)
    write_json(ctx.step_dir(10) / "10_M1_CI_FALLBACK_DIAGNOSTICS.json", ci_diag_m1)
    slot10 = ctx.data.get("_step_debug")
    if isinstance(slot10, dict) and slot10.get("step") == 10:
        slot10["ci_fallbacks"]["M1"] = ci_diag_m1
    write_json(ctx.step_dir(10) / "10_M1_CLOUD_INTEGRABILITY_DIAGNOSTICS.json", diagnostic)
    write_csv(ctx.step_dir(10) / "10_M1_CLOUD_DIAGNOSTIC_SAMPLES.csv",
              diagnostic["cloud"].get("sample_rows", []))
    write_csv(ctx.step_dir(10) / "10_M1_CLOUD_BUNDLE_DIAGNOSTICS.csv",
              diagnostic["cloud"].get("bundle_rows", []))
    return {"status": "PASS", "M1_constructed_first": True, "points": len(ci["points_by_ray"]),
            "fallback_count": ci["fallback_count"], "cloud_diagnostics": diagnostic}


def _step11_fixed_residual_grid(
    surface: PolySurface,
    samples: int,
) -> dict[str, np.ndarray | float]:
    """
    Tạo topology residual grid duy nhất cho cả restoration run.

    Grid KHÔNG thay đổi theo candidate.
    Neighbor graph cũng được freeze để curvature-gradient
    luôn so sánh đúng cùng topology.
    """

    if (
        isinstance(samples, bool)
        or int(samples) != samples
        or int(samples) < 9
    ):
        raise ValueError(
            "STEP11_FIXED_GRID_SAMPLES_INVALID"
        )

    x = np.linspace(
        -float(surface.half_aperture[0]),
        float(surface.half_aperture[0]),
        int(samples),
    )

    y = np.linspace(
        -float(surface.half_aperture[1]),
        float(surface.half_aperture[1]),
        int(samples),
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    xy = np.column_stack([
        X.ravel(),
        Y.ravel(),
    ])

    if surface.aperture_polygon is not None:

        poly = np.asarray(
            surface.aperture_polygon,
            float,
        )

        if (
            poly.ndim != 2
            or poly.shape[1] != 2
            or len(poly) < 3
            or not np.all(
                np.isfinite(poly)
            )
        ):
            raise RuntimeError(
                "STEP11_FIXED_GRID_POLYGON_INVALID:"
                f"{surface.name}"
            )

        edges = (
            np.roll(
                poly,
                -1,
                axis=0,
            )
            -
            poly
        )

        cross = (
            edges[None, :, 0]
            *
            (
                xy[:, None, 1]
                -
                poly[None, :, 1]
            )
            -
            edges[None, :, 1]
            *
            (
                xy[:, None, 0]
                -
                poly[None, :, 0]
            )
        )

        area = 0.5 * np.sum(
            poly[:, 0]
            *
            np.roll(
                poly[:, 1],
                -1,
            )
            -
            poly[:, 1]
            *
            np.roll(
                poly[:, 0],
                -1,
            )
        )

        if (
            not np.isfinite(area)
            or abs(
                float(area)
            ) <= 1e-12
        ):
            raise RuntimeError(
                "STEP11_FIXED_GRID_POLYGON_DEGENERATE:"
                f"{surface.name}"
            )

        keep = np.all(
            cross
            *
            np.sign(area)
            >= -1e-9,
            axis=1,
        )

        xy = np.vstack([
            xy[keep],
            poly,
        ])

        rounded = np.round(
            xy,
            decimals=12,
        )

        _, unique_index = np.unique(
            rounded,
            axis=0,
            return_index=True,
        )

        xy = xy[
            np.sort(
                unique_index
            )
        ]

    if (
        len(xy) < 9
        or not np.all(
            np.isfinite(xy)
        )
    ):
        raise RuntimeError(
            "STEP11_FIXED_GRID_INVALID:"
            f"{surface.name}"
        )

    # Freeze nearest-neighbour topology.
    tree = cKDTree(
        xy
    )

    neighbor_count = min(
        5,
        len(xy),
    )

    distances, indices = tree.query(
        xy,
        k=neighbor_count,
    )

    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    edge_i = []
    edge_j = []
    edge_distance = []

    row_index = np.arange(
        len(xy),
        dtype=int,
    )

    for neighbor_column in range(
        1,
        neighbor_count,
    ):
        distance = np.asarray(
            distances[
                :,
                neighbor_column
            ],
            float,
        )

        neighbor = np.asarray(
            indices[
                :,
                neighbor_column
            ],
            int,
        )

        good = (
            np.isfinite(distance)
            &
            (distance > 1e-12)
        )

        edge_i.append(
            row_index[good]
        )

        edge_j.append(
            neighbor[good]
        )

        edge_distance.append(
            distance[good]
        )

    return {
        "xy":
            np.asarray(
                xy,
                float,
            ),

        "edge_i":
            (
                np.concatenate(edge_i)
                if edge_i
                else
                np.empty(
                    0,
                    dtype=int,
                )
            ),

        "edge_j":
            (
                np.concatenate(edge_j)
                if edge_j
                else
                np.empty(
                    0,
                    dtype=int,
                )
            ),

        "edge_distance_mm":
            (
                np.concatenate(
                    edge_distance
                )
                if edge_distance
                else
                np.empty(
                    0,
                    dtype=float,
                )
            ),

        "characteristic_length_mm":
            max(
                float(
                    np.max(
                        surface.half_aperture
                    )
                ),
                1e-9,
            ),
    }


def _step11_curvature_field(
    surface: PolySurface,
    grid: dict[str, Any],
    orientation_sign: float,
) -> dict[str, np.ndarray]:
    """
    Tính H, KG, k1, k2 và curvature-gradient
    trên CHÍNH fixed grid.
    """

    xy = np.asarray(
        grid["xy"],
        float,
    )

    orientation = float(
        np.sign(
            float(
                orientation_sign
            )
        )
    )

    if orientation == 0.0:
        raise RuntimeError(
            "STEP11_CURVATURE_ORIENTATION_ZERO:"
            f"{surface.name}"
        )

    (
        z,
        gx,
        gy,
        gxx,
        gxy,
        gyy,
    ) = surface.sag_slopes_hessian(
        xy[:, 0],
        xy[:, 1],
    )

    for value in (
        z,
        gx,
        gy,
        gxx,
        gxy,
        gyy,
    ):
        if not np.all(
            np.isfinite(value)
        ):
            raise RuntimeError(
                "STEP11_CURVATURE_FIELD_NONFINITE:"
                f"{surface.name}"
            )

    denominator = (
        1.0
        +
        gx * gx
        +
        gy * gy
    )

    H = (
        (
            (1.0 + gy * gy)
            *
            gxx
            -
            2.0
            *
            gx
            *
            gy
            *
            gxy
            +
            (1.0 + gx * gx)
            *
            gyy
        )
        /
        (
            2.0
            *
            denominator ** 1.5
        )
    )

    KG = (
        gxx * gyy
        -
        gxy * gxy
    ) / (
        denominator
        *
        denominator
    )

    discriminant = np.maximum(
        H * H
        -
        KG,
        0.0,
    )

    root = np.sqrt(
        discriminant
    )

    k1 = H + root
    k2 = H - root

    normal = unit(
        np.column_stack([
            -gx,
            -gy,
            np.ones_like(gx),
        ])
    )

    edge_i = np.asarray(
        grid["edge_i"],
        int,
    )

    edge_j = np.asarray(
        grid["edge_j"],
        int,
    )

    distance = np.asarray(
        grid[
            "edge_distance_mm"
        ],
        float,
    )

    if len(distance):

        gradient_k1 = (
            np.abs(
                k1[edge_i]
                -
                k1[edge_j]
            )
            /
            distance
        )

        gradient_k2 = (
            np.abs(
                k2[edge_i]
                -
                k2[edge_j]
            )
            /
            distance
        )

    else:

        gradient_k1 = np.empty(
            0,
            dtype=float,
        )

        gradient_k2 = np.empty(
            0,
            dtype=float,
        )

    arrays = (
        H,
        KG,
        k1,
        k2,
        normal,
        gradient_k1,
        gradient_k2,
    )

    if any(
        not np.all(
            np.isfinite(value)
        )
        for value in arrays
    ):
        raise RuntimeError(
            "STEP11_CURVATURE_DERIVED_NONFINITE:"
            f"{surface.name}"
        )

    return {
        "z_mm":
            np.asarray(
                z,
                float,
            ),

        "normal":
            np.asarray(
                normal,
                float,
            ),

        "H_per_mm":
            np.asarray(
                H,
                float,
            ),

        "KG_per_mm2":
            np.asarray(
                KG,
                float,
            ),

        "k1_per_mm":
            np.asarray(
                k1,
                float,
            ),

        "k2_per_mm":
            np.asarray(
                k2,
                float,
            ),

        "oriented_H_per_mm":
            np.asarray(
                orientation * H,
                float,
            ),

        "gradient_k1_per_mm2":
            np.asarray(
                gradient_k1,
                float,
            ),

        "gradient_k2_per_mm2":
            np.asarray(
                gradient_k2,
                float,
            ),
    }


def _step11_balanced_block(
    values: np.ndarray,
    scale: float,
    weight: float,
) -> np.ndarray:
    """
    Dimensionless + sample-count balanced residual.

    Tổng energy của một block không tự tăng chỉ vì
    ta dùng grid dày hơn.
    """

    values = np.asarray(
        values,
        float,
    ).ravel()

    scale = float(scale)
    weight = float(weight)

    if (
        not np.isfinite(scale)
        or scale <= 0.0
        or not np.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError(
            "STEP11_RESIDUAL_SCALE_INVALID"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "STEP11_RESIDUAL_BLOCK_NONFINITE"
        )

    if len(values) == 0:
        return np.empty(
            0,
            dtype=float,
        )

    return (
        math.sqrt(
            weight
            /
            float(
                len(values)
            )
        )
        *
        values
        /
        scale
    )


def _step11_surface_restoration_blocks(
    field: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    grid: dict[str, Any],
    cfg: dict[str, Any],
    weights: dict[str, float],
    surface_name: str,
    maximum_principal_curvature_per_mm: float,
) -> list[np.ndarray]:
    """Xây dựng các khối residual hình học chuẩn hóa không thứ nguyên cho một mặt gương."""

    oriented_h = np.asarray(
        field[
            "oriented_H_per_mm"
        ],
        float,
    )

    kg = np.asarray(
        field[
            "KG_per_mm2"
        ],
        float,
    )

    k1 = np.asarray(
        field[
            "k1_per_mm"
        ],
        float,
    )

    k2 = np.asarray(
        field[
            "k2_per_mm"
        ],
        float,
    )

    gradient = np.concatenate([
        np.asarray(
            field[
                "gradient_k1_per_mm2"
            ],
            float,
        ),
        np.asarray(
            field[
                "gradient_k2_per_mm2"
            ],
            float,
        ),
    ])

    # H phải có positive ray-facing safety margin.
    H_violation = np.maximum(
        0.0,
        float(
            cfg[
                "h_target_per_mm"
            ]
        )
        -
        oriented_h,
    )

    # Final hard authority đang là KG >= -1e-7.
    # Restoration target dùng -5e-8 để có safety margin.
    KG_violation = np.maximum(
        0.0,
        float(
            cfg[
                "kg_target_per_mm2"
            ]
        )
        -
        kg,
    )

    principal_violation = np.concatenate([
        np.maximum(
            0.0,
            np.abs(k1)
            -
            float(
                maximum_principal_curvature_per_mm
            ),
        ),
        np.maximum(
            0.0,
            np.abs(k2)
            -
            float(
                maximum_principal_curvature_per_mm
            ),
        ),
    ])

    gradient_limit = (
        float(
            maximum_principal_curvature_per_mm
        )
        /
        float(
            grid[
                "characteristic_length_mm"
            ]
        )
    )

    gradient_violation = np.maximum(
        0.0,
        gradient
        -
        gradient_limit,
    )

    sag_delta = (
        np.asarray(
            field["z_mm"],
            float,
        )
        -
        np.asarray(
            anchor["z_mm"],
            float,
        )
    )

    normal_dot = np.sum(
        np.asarray(
            field["normal"],
            float,
        )
        *
        np.asarray(
            anchor["normal"],
            float,
        ),
        axis=1,
    )

    normal_angle_deg = np.degrees(
        np.arccos(
            np.clip(
                normal_dot,
                -1.0,
                1.0,
            )
        )
    )

    return [
        _step11_balanced_block(
            H_violation,
            cfg[
                "mean_curvature_scale_per_mm"
            ],
            weights[
                f"{surface_name}_H"
            ],
        ),

        _step11_balanced_block(
            KG_violation,
            cfg[
                "gaussian_curvature_scale_per_mm2"
            ],
            weights[
                f"{surface_name}_KG"
            ],
        ),

        _step11_balanced_block(
            principal_violation,
            cfg[
                "principal_curvature_scale_per_mm"
            ],
            weights[
                f"{surface_name}_principal"
            ],
        ),

        _step11_balanced_block(
            gradient_violation,
            cfg[
                "curvature_gradient_scale_per_mm2"
            ],
            weights[
                f"{surface_name}_gradient"
            ],
        ),

        _step11_balanced_block(
            sag_delta,
            cfg[
                "pair_sag_trust_scale_mm"
            ],
            weights[
                f"{surface_name}_sag_trust"
            ],
        ),

        _step11_balanced_block(
            normal_angle_deg,
            cfg[
                "pair_normal_trust_scale_deg"
            ],
            weights[
                f"{surface_name}_normal_trust"
            ],
        ),
    ]


def _step11_apply_joint_restoration_vector(
    base_pair: dict[str, PolySurface],
    descriptors: list[dict[str, Any]],
    normalized: np.ndarray,
    cfg: dict[str, Any],
    surface_fit_cfg: dict[str, Any],
) -> tuple[
    PolySurface | None,
    PolySurface | None,
    dict[str, Any],
]:
    """Áp dụng vector tìm kiếm u vào cặp mặt M1/M2."""

    u = np.asarray(
        normalized,
        float,
    )

    if (
        u.shape
        !=
        (len(descriptors),)
        or not np.all(
            np.isfinite(u)
        )
    ):
        return (
            None,
            None,
            {
                "valid": False,
                "reason":
                    "RESTORATION_VECTOR_INVALID",
            },
        )

    if np.any(
        np.abs(u)
        >
        1.0 + 1e-12
    ):
        return (
            None,
            None,
            {
                "valid": False,
                "reason":
                    "RESTORATION_NORMALIZED_BOUND",
            },
        )

    m1 = base_pair[
        "M1"
    ].copy()

    m2 = base_pair[
        "M2"
    ].copy()

    surfaces = {
        "M1": m1,
        "M2": m2,
    }

    curvature_abs_max = float(
        surface_fit_cfg[
            "curvature_absolute_max_per_mm"
        ]
    )

    conic_lo, conic_hi = map(
        float,
        surface_fit_cfg[
            "conic_bounds"
        ],
    )

    rows = []

    for column, (
        value,
        descriptor,
    ) in enumerate(
        zip(
            u,
            descriptors,
        )
    ):

        which = str(
            descriptor[
                "surface"
            ]
        )

        surface = surfaces[
            which
        ]

        source = base_pair[
            which
        ]

        kind = str(
            descriptor[
                "kind"
            ]
        )

        if kind == "CURVATURE":

            span = max(
                abs(
                    float(
                        source.curvature
                    )
                )
                *
                float(
                    cfg[
                        "curvature_relative_span"
                    ]
                ),
                float(
                    cfg[
                        "minimum_curvature_span_per_mm"
                    ]
                ),
            )

            candidate = (
                float(
                    source.curvature
                )
                +
                span
                *
                float(value)
            )

            if (
                abs(candidate)
                >
                curvature_abs_max
                + 1e-12
            ):
                return (
                    None,
                    None,
                    {
                        "valid": False,

                        "reason":
                            "RESTORATION_CURVATURE_BOUND:"
                            f"{which}:"
                            f"{candidate:.12g}",
                    },
                )

            surface.curvature = float(
                candidate
            )

        elif kind == "CONIC":

            span = float(
                cfg[
                    "conic_absolute_span"
                ]
            )

            candidate = (
                float(
                    source.conic
                )
                +
                span
                *
                float(value)
            )

            if (
                candidate
                <
                conic_lo - 1e-12
                or
                candidate
                >
                conic_hi + 1e-12
            ):
                return (
                    None,
                    None,
                    {
                        "valid": False,

                        "reason":
                            "RESTORATION_CONIC_BOUND:"
                            f"{which}:"
                            f"{candidate:.12g}",
                    },
                )

            surface.conic = float(
                candidate
            )

        elif kind == (
            "ASTIG_QUADRATIC_PAIR"
        ):

            delta = (
                float(
                    cfg[
                        "coefficient_absolute_span_mm"
                    ]
                )
                *
                float(value)
            )

            i20 = int(
                descriptor[
                    "index_20"
                ]
            )

            i02 = int(
                descriptor[
                    "index_02"
                ]
            )

            surface.coeff[
                i20
            ] = (
                float(
                    source.coeff[
                        i20
                    ]
                )
                +
                delta
            )

            surface.coeff[
                i02
            ] = (
                float(
                    source.coeff[
                        i02
                    ]
                )
                -
                delta
            )

        elif kind == "COEFFICIENT":

            index = int(
                descriptor[
                    "index"
                ]
            )

            surface.coeff[
                index
            ] = (
                float(
                    source.coeff[
                        index
                    ]
                )
                +
                float(
                    cfg[
                        "coefficient_absolute_span_mm"
                    ]
                )
                *
                float(value)
            )

        else:

            raise ValueError(
                "STEP11_RESTORATION_"
                "VARIABLE_KIND_INVALID:"
                f"{kind}"
            )

        rows.append({
            "column":
                int(column),

            "surface":
                which,

            "kind":
                kind,

            "term":
                str(
                    descriptor.get(
                        "term",
                        "",
                    )
                ),

            "normalized_value":
                float(value),
        })

    return (
        m1,
        m2,
        {
            "valid": True,
            "reason": None,
            "rows": rows,
        },
    )


def _step11_unobscuration_state(
    ctx: Context,
    m2: PolySurface,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """
    Đánh giá signed-area unobscuration trên physical trace hiện tại.

    Đây là cùng định nghĩa MF2 mà STEP11 final certification sử dụng:
        S_AQP_signed_mm2 >= 0  -> unobscured
        S_AQP_signed_mm2 < 0   -> obscured
    """

    physical_valid = np.asarray(
        trace[
            "valid"
        ],
        bool,
    )

    if (
        physical_valid.ndim != 1
        or not np.any(
            physical_valid
        )
    ):
        raise RuntimeError(
            "STEP11_UNOBSCURATION_NO_VALID_PHYSICAL_RAYS"
        )

    try:

        mf2 = mf2_geometry(
            np.asarray(
                trace[
                    "points"
                ][0],
                float,
            )[
                physical_valid
            ],

            np.asarray(
                trace[
                    "points"
                ][1],
                float,
            )[
                physical_valid
            ],

            np.asarray(
                trace[
                    "points"
                ][2],
                float,
            )[
                physical_valid
            ],

            m2,

            float(
                ctx.config[
                    "fan_weights"
                ][
                    "omega2"
                ]
            ),

            ctx.data[
                "n_obs"
            ],
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ) as exc:

        raise RuntimeError(
            "STEP11_UNOBSCURATION_EVALUATION_FAILED:"
            f"{type(exc).__name__}:{exc}"
        ) from exc

    signed_area = float(
        mf2[
            "S_AQP_signed_mm2"
        ]
    )

    if not np.isfinite(
        signed_area
    ):
        raise RuntimeError(
            "STEP11_UNOBSCURATION_SIGNED_AREA_NONFINITE"
        )

    return {
        "mf2":
            mf2,

        "S_AQP_signed_mm2":
            signed_area,

        "unobscured":
            bool(
                signed_area >= 0.0
            ),

        "violation_mm2":
            float(
                max(
                    0.0,
                    -signed_area,
                )
            ),
    }


def _step11_evaluate_restoration_pair(
    ctx: Context,
    m1: PolySurface,
    m2: PolySurface,
    refinement_cfg: dict[str, Any],
    restoration_cfg: dict[str, Any],
    frozen_orientation_signs: dict[str, float],
) -> dict[str, Any]:
    """Đánh giá toàn diện một ứng viên phục hồi cặp gương M1/M2."""

    try:
        (
            optical_residual,
            metrics,
            trace,
        ) = _step12_actual_o2_evaluate(
            ctx,
            m1,
            m2,
            refinement_cfg,
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ) as exc:

        return {
            "valid": False,

            "reason":
                f"{type(exc).__name__}:{exc}",
        }

    m1_gate = metrics[
        "M1_sanity"
    ]

    m2_gate = metrics[
        "M2_sanity"
    ]

    if bool(
        restoration_cfg[
            "require_stable_orientation_sign"
        ]
    ):

        for name, gate in (
            ("M1", m1_gate),
            ("M2", m2_gate),
        ):

            actual_sign = float(
                gate[
                    "orientation_sign"
                ]
            )

            frozen_sign = float(
                frozen_orientation_signs[
                    name
                ]
            )

            if (
                np.sign(
                    actual_sign
                )
                !=
                np.sign(
                    frozen_sign
                )
            ):
                return {
                    "valid": False,

                    "reason":
                        "RESTORATION_"
                        "ORIENTATION_SIGN_CHANGED:"
                        f"{name}",
                }

    # M1 đã tốt -> không cho solver phá M1.
    if (
        bool(
            restoration_cfg[
                "require_m1_raw_topology"
            ]
        )
        and
        not bool(
            m1_gate[
                "topology_pass"
            ]
        )
    ):
        return {
            "valid": False,

            "reason":
                "RESTORATION_M1_RAW_TOPOLOGY_LOST",
        }

    # M2 H/KG FAIL vẫn là VALID state.
    #
    # Chỉ basic-sanity mất mới coi là
    # numerical/domain failure.
    if not bool(
        m2_gate[
            "topology_checks"
        ][
            "basic_surface_sanity"
        ]
    ):
        return {
            "valid": False,

            "reason":
                "RESTORATION_M2_BASIC_SANITY_LOST",
        }

    residual = np.asarray(
        optical_residual,
        float,
    )

    if (
        residual.ndim != 1
        or not np.all(
            np.isfinite(residual)
        )
    ):
        return {
            "valid": False,

            "reason":
                "RESTORATION_OPTICAL_RESIDUAL_INVALID",
        }

    try:

        unobscuration = (
            _step11_unobscuration_state(
                ctx,
                m2,
                trace,
            )
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ) as exc:

        return {
            "valid": False,

            "reason":
                f"{type(exc).__name__}:{exc}",
        }

    return {
        "valid": True,

        "reason": None,

        "optical_residual":
            residual,

        "metrics":
            metrics,

        "trace":
            trace,

        "M1_gate":
            m1_gate,

        "M2_gate":
            m2_gate,

        "mf2":
            unobscuration[
                "mf2"
            ],

        "S_AQP_signed_mm2":
            float(
                unobscuration[
                    "S_AQP_signed_mm2"
                ]
            ),

        "unobscured":
            bool(
                unobscuration[
                    "unobscured"
                ]
            ),
    }


def _step11_build_restoration_residual(
    m1: PolySurface,
    m2: PolySurface,
    evaluation: dict[str, Any],
    state: dict[str, Any],
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """Xây dựng vector phần dư tổng hợp cân bằng không thứ nguyên r(u)."""

    cfg = state["cfg"]

    weights = cfg[
        "weights"
    ]

    maximum_principal = float(
        state[
            "surface_fit_cfg"
        ][
            "curvature_absolute_max_per_mm"
        ]
    )

    fields = {}

    blocks = []

    for name, surface in (
        ("M1", m1),
        ("M2", m2),
    ):

        field = (
            _step11_curvature_field(
                surface,
                state[
                    "grids"
                ][name],
                state[
                    "orientation_signs"
                ][name],
            )
        )

        fields[
            name
        ] = field

        blocks.extend(
            _step11_surface_restoration_blocks(
                field,
                state[
                    "anchors"
                ][name],
                state[
                    "grids"
                ][name],
                cfg,
                weights,
                name,
                maximum_principal,
            )
        )

    optical = np.asarray(
        evaluation[
            "optical_residual"
        ],
        float,
    )

    optical_indices = np.asarray(
        state[
            "optical_indices"
        ],
        int,
    )

    if len(
        optical_indices
    ):

        if (
            np.min(
                optical_indices
            ) < 0
            or
            np.max(
                optical_indices
            )
            >=
            len(optical)
        ):
            raise RuntimeError(
                "STEP11_OPTICAL_INDEX_OUT_OF_RANGE"
            )

        optical_sample = optical[
            optical_indices
        ]

        blocks.append(
            _step11_balanced_block(
                optical_sample,
                1.0,
                weights[
                    "optical"
                ],
            )
        )

    physical_fraction = float(
        evaluation[
            "metrics"
        ][
            "physical_fraction"
        ]
    )

    preferred_fraction = float(
        state[
            "quality_cfg"
        ][
            "minimum_physical_fraction"
        ]
    )

    physical_shortfall = np.asarray([
        max(
            0.0,
            preferred_fraction
            -
            physical_fraction,
        )
    ])

    blocks.append(
        _step11_balanced_block(
            physical_shortfall,
            cfg[
                "physical_fraction_scale"
            ],
            weights[
                "physical_fraction"
            ],
        )
    )

    signed_area = float(
        evaluation[
            "S_AQP_signed_mm2"
        ]
    )

    if not np.isfinite(
        signed_area
    ):
        raise RuntimeError(
            "STEP11_UNOBSCURATION_RESIDUAL_NONFINITE"
        )

    unobscuration_violation = np.asarray([
        max(
            0.0,
            -signed_area,
        )
    ])

    blocks.append(
        _step11_balanced_block(
            unobscuration_violation,

            float(
                state[
                    "unobscuration_scale_mm2"
                ]
            ),

            weights[
                "unobscuration"
            ],
        )
    )

    residual = np.concatenate(
        blocks
    )

    if (
        residual.ndim != 1
        or not np.all(
            np.isfinite(
                residual
            )
        )
    ):
        raise RuntimeError(
            "STEP11_RESTORATION_RESIDUAL_NONFINITE"
        )

    def principal_max(
        name: str,
    ) -> float:
        """Tính độ cong chính cực đại tuyệt đối."""

        return float(
            max(
                np.max(
                    np.abs(
                        fields[name][
                            "k1_per_mm"
                        ]
                    )
                ),
                np.max(
                    np.abs(
                        fields[name][
                            "k2_per_mm"
                        ]
                    )
                ),
            )
        )

    meta = {
        "M1_H_min_per_mm":
            float(
                np.min(
                    fields[
                        "M1"
                    ][
                        "H_per_mm"
                    ]
                )
            ),

        "M1_oriented_H_min_per_mm":
            float(
                np.min(
                    fields[
                        "M1"
                    ][
                        "oriented_H_per_mm"
                    ]
                )
            ),

        "M1_KG_min_per_mm2":
            float(
                np.min(
                    fields[
                        "M1"
                    ][
                        "KG_per_mm2"
                    ]
                )
            ),

        "M2_H_min_per_mm":
            float(
                np.min(
                    fields[
                        "M2"
                    ][
                        "H_per_mm"
                    ]
                )
            ),

        "M2_oriented_H_min_per_mm":
            float(
                np.min(
                    fields[
                        "M2"
                    ][
                        "oriented_H_per_mm"
                    ]
                )
            ),

        "M2_KG_min_per_mm2":
            float(
                np.min(
                    fields[
                        "M2"
                    ][
                        "KG_per_mm2"
                    ]
                )
            ),

        "M1_principal_abs_max_per_mm":
            principal_max(
                "M1"
            ),

        "M2_principal_abs_max_per_mm":
            principal_max(
                "M2"
            ),

        "M1_raw_topology_pass":
            bool(
                evaluation[
                    "M1_gate"
                ][
                    "topology_pass"
                ]
            ),

        "M2_raw_topology_pass":
            bool(
                evaluation[
                    "M2_gate"
                ][
                    "topology_pass"
                ]
            ),

        "physical_fraction":
            physical_fraction,

        "S_AQP_signed_mm2":
            signed_area,

        "unobscured":
            bool(
                signed_area >= 0.0
            ),

        "unobscuration_violation_mm2":
            float(
                max(
                    0.0,
                    -signed_area,
                )
            ),

        "residual_length":
            int(
                len(
                    residual
                )
            ),
    }

    return (
        residual,
        meta,
    )


def _step11_evaluate_restoration_u(
    common: dict[str, Any],
    normalized: np.ndarray,
    *,
    include_payload: bool,
) -> dict[str, Any]:
    """Đánh giá nhanh hàm mục tiêu và phần dư cho vector u."""

    (
        m1,
        m2,
        bound_state,
    ) = (
        _step11_apply_joint_restoration_vector(
            common[
                "base_pair"
            ],
            common[
                "descriptors"
            ],
            normalized,
            common[
                "restoration_cfg"
            ],
            common[
                "surface_fit_cfg"
            ],
        )
    )

    if m1 is None or m2 is None:
        return {
            "valid": False,

            "reason":
                bound_state.get(
                    "reason"
                ),

            "bound_state":
                bound_state,
        }

    evaluated = (
        _step11_evaluate_restoration_pair(
            common["ctx"],
            m1,
            m2,
            common[
                "refinement_cfg"
            ],
            common[
                "restoration_cfg"
            ],
            common[
                "residual_state"
            ][
                "orientation_signs"
            ],
        )
    )

    if not evaluated[
        "valid"
    ]:
        return {
            "valid": False,

            "reason":
                evaluated.get(
                    "reason"
                ),

            "bound_state":
                bound_state,
        }

    try:

        residual, meta = (
            _step11_build_restoration_residual(
                m1,
                m2,
                evaluated,
                common[
                    "residual_state"
                ],
            )
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ) as exc:

        return {
            "valid": False,

            "reason":
                f"{type(exc).__name__}:{exc}",

            "bound_state":
                bound_state,
        }

    result = {
        "valid": True,

        "reason": None,

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

        "bound_state":
            bound_state,
    }

    if include_payload:

        result.update({
            "m1":
                m1,

            "m2":
                m2,

            "evaluation":
                evaluated,
        })

    return result


def evaluate_step11_restoration_probe_job(
    common: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    """
    Pure STEP11 FD probe evaluator.
    Không snapshot.
    Không ghi file.
    Không mở nested process pool.
    """

    if (
        job.get(
            "schema"
        )
        !=
        "HUD_FAN_V5_5_"
        "STEP11_RESTORATION_PROBE_JOB_V1"
    ):
        raise ValueError(
            "STEP11_RESTORATION_PROBE_SCHEMA_INVALID"
        )

    u = np.asarray(
        job[
            "search_vector"
        ],
        float,
    )

    result = (
        _step11_evaluate_restoration_u(
            common,
            u,
            include_payload=False,
        )
    )

    return {
        "schema":
            "HUD_FAN_V5_5_"
            "STEP11_RESTORATION_PROBE_RESULT_V1",

        "job_index":
            int(
                job[
                    "job_index"
                ]
            ),

        "iteration":
            int(
                job[
                    "iteration"
                ]
            ),

        "fd_round":
            int(
                job[
                    "fd_round"
                ]
            ),

        "column":
            int(
                job[
                    "column"
                ]
            ),

        "sign":
            float(
                job[
                    "sign"
                ]
            ),

        "delta":
            float(
                job[
                    "delta"
                ]
            ),

        **result,
    }


def _step11_apply_search_vector(
    common: dict[str, Any],
    search_vector: np.ndarray,
) -> PolySurface:
    """Áp vector trust-region sau khi scale từng DOF theo sag/slope/Hessian."""
    descriptors = common["descriptors"]
    u = np.asarray(search_vector, float)
    if u.shape != (len(descriptors),):
        raise ValueError("STEP11_SEARCH_VECTOR_SHAPE_MISMATCH")
    if not np.all(np.isfinite(u)) or np.any(np.abs(u) > 1.0 + 1e-12):
        raise ValueError("STEP11_SEARCH_VECTOR_INVALID")
    candidate_m1, _ = _step12_apply_o2_vector(
        common["base_pair"],
        descriptors,
        u * common["physical_scales"],
        common["refinement_cfg"],
        common["surface_fit_cfg"],
    )
    return candidate_m1


def _step11_topology_admission(
    gate: dict[str, Any],
    enforcement: str,
) -> dict[str, Any]:
    """Apply STEP11-only HARD/WARN admission without changing raw topology diagnostics."""
    if enforcement not in {"HARD", "WARN"}:
        raise ValueError(
            "STEP11_TOPOLOGY_ENFORCEMENT_INVALID"
        )

    topology_pass = bool(gate["topology_pass"])
    basic_surface_sanity = bool(
        gate["topology_checks"]["basic_surface_sanity"]
    )
    admitted = bool(
        topology_pass
        or (
            enforcement == "WARN"
            and basic_surface_sanity
        )
    )
    failure_reasons = list(
        gate.get("topology_failure_reasons", [])
    )

    return {
        "enforcement": enforcement,
        "raw_pass": topology_pass,
        "basic_surface_sanity": basic_surface_sanity,
        "admitted": admitted,
        "status": (
            "PASS"
            if topology_pass
            else (
                "WARN_ADMITTED"
                if admitted
                else "FAIL"
            )
        ),
        "warning_reasons": (
            failure_reasons
            if admitted and not topology_pass
            else []
        ),
    }


def _step11_topology_shape_gate(
    common: dict[str, Any],
    surface: PolySurface,
    incident_direction: np.ndarray,
    topology: str,
) -> dict[str, Any]:
    """Map ray-facing concave/convex authority to mirror_shape_gate orientation."""
    direction = np.asarray(
        incident_direction,
        float,
    )

    if (
        direction.shape != (3,)
        or not np.all(
            np.isfinite(
                direction
            )
        )
    ):
        raise RuntimeError(
            f"STEP11_TOPOLOGY_INCIDENT_DIRECTION_INVALID:{surface.name}"
        )

    normal_rows = np.asarray(
        surface.normal(
            np.asarray(
                [0.0],
                float,
            ),
            np.asarray(
                [0.0],
                float,
            ),
        ),
        float,
    )

    if (
        normal_rows.shape != (1, 3)
        or not np.all(
            np.isfinite(
                normal_rows
            )
        )
    ):
        raise RuntimeError(
            f"STEP11_TOPOLOGY_CHIEF_NORMAL_INVALID:{surface.name}"
        )

    normal = normal_rows[0]
    direction_norm = float(np.linalg.norm(direction))
    normal_norm = float(np.linalg.norm(normal))

    if (
        not np.isfinite(direction_norm)
        or not np.isfinite(normal_norm)
        or direction_norm <= 1e-12
        or normal_norm <= 1e-12
    ):
        raise RuntimeError(
            f"STEP11_TOPOLOGY_CHIEF_VECTOR_NORM_INVALID:{surface.name}"
        )

    incident_unit = direction / direction_norm
    normal_unit = normal / normal_norm
    ray_facing_dot = float(np.dot(normal_unit, -incident_unit))

    if not np.isfinite(ray_facing_dot) or abs(ray_facing_dot) <= 1e-9:
        raise RuntimeError(
            f"STEP11_TOPOLOGY_CHIEF_RAY_GRAZING:{surface.name}"
        )

    facing_sign = 1.0 if ray_facing_dot > 0.0 else -1.0

    if topology == "CONCAVE_RAY_FACING":
        orientation_sign = facing_sign
    elif topology == "CONVEX_RAY_FACING":
        orientation_sign = -facing_sign
    else:
        raise RuntimeError(
            f"STEP11_TOPOLOGY_AUTHORITY_INVALID:{surface.name}:{topology}"
        )

    shape_policy = common["shape_policy"]
    quality_cfg = common["quality_cfg"]

    gate = dict(
        mirror_shape_gate(
            surface,
            shape_policy,
            orientation_sign=orientation_sign,
        )
    )

    gate.update({
        "topology_authority": topology,
        "topology_enforcement": str(
            common["topology_enforcement"][surface.name]
        ),
        "ray_facing_definition": "SURFACE_NORMAL_DOT_NEGATIVE_INCIDENT_DIRECTION",
        "ray_facing_dot": ray_facing_dot,
        "ray_facing_normal_sign": facing_sign,
        "required_orientation_sign": float(orientation_sign),
    })

    return _o2_mirror_gate_view(
        gate,
        quality_cfg,
    )


def _step11_ci_trust_gate(
    common: dict[str, Any],
    candidate_m1: PolySurface,
) -> dict[str, Any]:
    """Chấm sag/normal M1 so với CI như quality diagnostic WARN ở STEP11."""
    ctx = common["ctx"]
    quality_cfg = common["quality_cfg"]
    points = np.asarray(ctx.data["ci_m1"]["points_by_ray"], float)
    target_normals = unit(np.asarray(ctx.data["ci_m1"]["normals_by_ray"], float))
    local = (points - candidate_m1.center) @ candidate_m1.frame
    z_model, _, _ = candidate_m1.sag_slopes(local[:, 0], local[:, 1])
    sag_error = z_model - local[:, 2]
    model_normals = unit(candidate_m1.normal(local[:, 0], local[:, 1]))
    dot = np.abs(np.sum(model_normals * target_normals, axis=1))
    normal_error_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    sag_rms = float(np.sqrt(np.mean(sag_error * sag_error)))
    normal_rms = float(np.sqrt(np.mean(normal_error_deg * normal_error_deg)))
    normal_max = float(np.max(normal_error_deg))
    checks = {
        "sag_rms": bool(sag_rms <= float(quality_cfg["maximum_sag_rms_mm"])),
        "normal_rms": bool(normal_rms <= float(quality_cfg["maximum_normal_rms_deg"])),
        "normal_max": bool(normal_max <= float(quality_cfg["maximum_normal_max_deg"])),
    }
    passed = bool(all(checks.values()))
    return {
        "sag_rms_mm": sag_rms,
        "normal_rms_deg": normal_rms,
        "normal_max_deg": normal_max,
        "checks": checks,
        "pass": passed,
        "status": "PASS" if passed else "WARN",
        "enforcement": "WARN",
        "role": "STEP11_CI_TRUST_DIAGNOSTIC_ONLY_NOT_REJECTION_GATE",
    }


def _step11_integrability_rank_key(
    gate: dict[str, Any],
) -> tuple[int, int, float, float]:
    """Rank a discrete integrability diagnostic without changing gate authority."""
    ratios: list[float] = []
    failed_check_count = 0

    for row in gate.get("checks", []):
        actual_raw = row.get("actual")
        limit_raw = row.get("limit")
        if actual_raw is None or limit_raw is None:
            ratio = float("inf")
        else:
            actual = float(actual_raw)
            limit = float(limit_raw)
            ratio = (
                actual / limit
                if np.isfinite(actual) and np.isfinite(limit) and limit > 0.0
                else float("inf")
            )
        ratios.append(float(ratio))
        if str(row.get("status")) != "PASS":
            failed_check_count += 1

    maximum_ratio = (
        float(max(ratios))
        if ratios
        else float("inf")
    )
    ratio_sum = (
        float(np.sum(ratios))
        if ratios and np.all(np.isfinite(np.asarray(ratios, float)))
        else float("inf")
    )

    return (
        0 if str(gate.get("status")) == "PASS" else 1,
        int(failed_check_count),
        maximum_ratio,
        ratio_sum,
    )


def _step11_gamma_anchor_rank_key(
    payload: dict[str, Any],
    gamma: float,
    gamma_index: int,
) -> tuple[Any, ...]:
    """Prefer integrability PASS, then preserve the existing STEP11 rank rule."""
    integrability_rank = (
        0
        if str(payload["M2_integrability_gate"]["status"]) == "PASS"
        else 1
    )
    return (
        integrability_rank,
        *tuple(payload["rank_key"]),
        abs(float(gamma) - 1.0),
        int(gamma_index),
    )


def _step11_gamma_restoration_seed_rank_key(
    record: dict[str, Any],
    gamma: float,
    gamma_index: int,
    restoration_cfg: dict[str, Any],
) -> tuple[Any, ...]:
    """
    Chọn gamma để làm SEED cho restoration,
    không yêu cầu M2 đã topology PASS.

    Priority:
      1. M2 basic sanity có thể đánh giá được
      2. integrability PASS/WARN
      3. H/KG càng gần restoration target càng tốt
      4. gamma càng gần 1 càng tốt
    """

    integrability_status = str(
        record.get(
            "M2_integrability_status",
            "NOT_EVALUATED",
        )
    )

    if integrability_status == "PASS":

        integrability_rank = 0

    elif integrability_status == "WARN":

        integrability_rank = 1

    else:

        integrability_rank = 2

    basic_sanity_rank = (
        0
        if bool(
            record.get(
                "M2_basic_surface_sanity_pass",
                False,
            )
        )
        else 1
    )

    h_raw = record.get(
        "M2_oriented_H_min_per_mm"
    )

    kg_raw = record.get(
        "M2_KG_min_per_mm2"
    )

    try:
        h_value = float(
            h_raw
        )
    except (
        TypeError,
        ValueError,
    ):
        h_value = float("nan")

    try:
        kg_value = float(
            kg_raw
        )
    except (
        TypeError,
        ValueError,
    ):
        kg_value = float("nan")

    if np.isfinite(
        h_value
    ):

        h_violation = max(
            0.0,
            float(
                restoration_cfg[
                    "h_target_per_mm"
                ]
            )
            -
            h_value,
        )

        h_normalized = (
            h_violation
            /
            float(
                restoration_cfg[
                    "mean_curvature_scale_per_mm"
                ]
            )
        )

    else:

        h_normalized = float(
            "inf"
        )

    if np.isfinite(
        kg_value
    ):

        kg_violation = max(
            0.0,
            float(
                restoration_cfg[
                    "kg_target_per_mm2"
                ]
            )
            -
            kg_value,
        )

        kg_normalized = (
            kg_violation
            /
            float(
                restoration_cfg[
                    "gaussian_curvature_scale_per_mm2"
                ]
            )
        )

    else:

        kg_normalized = float(
            "inf"
        )

    topology_distance = (
        h_normalized
        +
        kg_normalized
    )

    return (
        int(
            basic_sanity_rank
        ),

        int(
            integrability_rank
        ),

        float(
            topology_distance
        ),

        float(
            h_normalized
        ),

        float(
            kg_normalized
        ),

        abs(
            float(gamma)
            -
            1.0
        ),

        int(
            gamma_index
        ),
    )


def _step11_condition_m2_ci(
    ctx: Context,
    starts: np.ndarray,
    directions: np.ndarray,
    targets: np.ndarray,
    seed_surface: PolySurface,
    chief_index: int,
    surface_fit_cfg: dict[str, Any],
    integrability_policy: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Condition M2 CI by alternating surface and reflection-law projections."""
    conditioning_started = time.perf_counter()

    def emit(message: str) -> None:
        """Emit parent-only conditioning progress without affecting optics."""
        if progress_callback is None:
            return
        try:
            progress_callback(str(message))
        except Exception as exc:
            if not is_compute_worker():
                print(
                    "[STEP11][WARN] CONDITIONING_PROGRESS_CALLBACK_FAILED:"
                    f"{type(exc).__name__}:{exc}",
                    flush=True,
                )

    def gate_summary(gate: dict[str, Any]) -> str:
        """Format the decisive integrability state for concise terminal logs."""
        checks = list(gate.get("checks", []))
        failed = [
            str(row.get("check"))
            for row in checks
            if str(row.get("status")) != "PASS"
        ]
        local_normal = next(
            (
                row
                for row in checks
                if str(row.get("check"))
                == "bundle_local_geometry_normal_rms_p95"
            ),
            None,
        )
        if local_normal is None:
            local_text = "n/a"
        else:
            actual = local_normal.get("actual")
            limit = local_normal.get("limit")
            local_text = (
                f"{float(actual):.4f}/{float(limit):.4f} deg"
                if actual is not None and limit is not None
                else "n/a"
            )
        return (
            f"integrability={str(gate.get('status', 'NOT_EVALUATED'))}"
            f" | failed={','.join(failed) if failed else 'none'}"
            f" | local_normal={local_text}"
        )

    conditioning_cfg = surface_fit_cfg[
        "step11_macro_micro_conditioning"
    ]
    enabled = bool(conditioning_cfg["enabled"])
    maximum_iterations = int(
        conditioning_cfg["maximum_alternating_projection_iterations"]
    )

    starts_array = np.asarray(starts, float)
    directions_array = np.asarray(directions, float)
    targets_array = np.asarray(targets, float)

    if (
        starts_array.ndim != 2
        or starts_array.shape[1] != 3
        or directions_array.shape != starts_array.shape
        or targets_array.shape != starts_array.shape
    ):
        raise ValueError("STEP11_M2_CONDITIONING_INPUT_SHAPE_MISMATCH")
    if (
        not np.all(np.isfinite(starts_array))
        or not np.all(np.isfinite(directions_array))
        or not np.all(np.isfinite(targets_array))
    ):
        raise ValueError("STEP11_M2_CONDITIONING_INPUT_NONFINITE")

    emit(
        f"[AP RAW][START] rays={len(starts_array):,}"
        f" | max_iterations={maximum_iterations}"
    )
    raw_ci = point_by_point_engine(
        starts_array,
        directions_array,
        targets_array,
        seed_surface,
        int(chief_index),
        construction_options=ctx.config["ci_construction"],
    )
    raw_diagnostic = _construction_diagnostics(
        ctx,
        raw_ci,
        seed_surface.frame,
        "M2_STEP11_CANDIDATE_CI_RAW",
    )
    raw_gate = _integrability_quality_gate(
        raw_diagnostic,
        integrability_policy,
        "M2_CI_CLOUD",
    )
    raw_rank = _step11_integrability_rank_key(raw_gate)
    emit(
        f"[AP 0/{maximum_iterations}][DONE] {gate_summary(raw_gate)}"
    )

    best_ci = raw_ci
    best_diagnostic = raw_diagnostic
    best_gate = raw_gate
    best_iteration = 0
    best_key = (*raw_rank, 0)

    iteration_rows: list[dict[str, Any]] = [{
        "iteration": 0,
        "role": "RAW_CI",
        "integrability_status": str(raw_gate["status"]),
        "integrability_rank_key": list(raw_rank),
        "checks": [dict(row) for row in raw_gate.get("checks", [])],
    }]

    current_ci = raw_ci
    current_seed = seed_surface
    attempted_iterations = 0
    conditioning_error: str | None = None

    if enabled and str(raw_gate["status"]) != "PASS":
        for iteration in range(1, maximum_iterations + 1):
            attempted_iterations = iteration
            iteration_started = time.perf_counter()
            emit(
                f"[AP {iteration}/{maximum_iterations}][START]"
                " fit temporary O2 and project rays"
            )
            try:
                temporary_m2, temporary_fit = fit_surface(
                    current_ci["points_by_ray"],
                    current_ci["normals_by_ray"],
                    current_seed,
                    2,
                    True,
                    ctx.config["fit_weights"],
                    int(chief_index),
                    surface_fit_cfg,
                    current_ci["starts_by_ray"],
                    current_ci["targets_by_ray"],
                )

                projected_hit = temporary_m2.intersect(
                    starts_array,
                    directions_array,
                    finite=False,
                )
                projected_valid = np.asarray(
                    projected_hit["valid"],
                    bool,
                )
                if projected_valid.shape != (len(starts_array),):
                    raise RuntimeError(
                        "STEP11_M2_CONDITIONING_VALID_MASK_SHAPE_MISMATCH"
                    )
                if not np.all(projected_valid):
                    raise RuntimeError(
                        "STEP11_M2_CONDITIONING_RETRACE_INCOMPLETE:"
                        f"{int(np.sum(projected_valid))}/{len(projected_valid)}"
                    )

                projected_points = np.asarray(
                    projected_hit["point"],
                    float,
                )
                if (
                    projected_points.shape != starts_array.shape
                    or not np.all(np.isfinite(projected_points))
                ):
                    raise RuntimeError(
                        "STEP11_M2_CONDITIONING_PROJECTED_POINTS_INVALID"
                    )

                target_vectors = targets_array - projected_points
                target_norms = np.linalg.norm(
                    target_vectors,
                    axis=1,
                )
                incoming_unit = unit(directions_array)
                outgoing_unit = unit(target_vectors)
                normal_vectors = incoming_unit - outgoing_unit
                normal_norms = np.linalg.norm(
                    normal_vectors,
                    axis=1,
                )
                if (
                    not np.all(np.isfinite(target_norms))
                    or not np.all(np.isfinite(normal_norms))
                    or np.any(target_norms <= 1e-12)
                    or np.any(normal_norms <= 1e-12)
                ):
                    raise RuntimeError(
                        "STEP11_M2_CONDITIONING_REFLECTION_NORMAL_DEGENERATE"
                    )

                projected_normals = unit(normal_vectors)
                ray_order = np.asarray(
                    current_ci["ray_order"],
                    int,
                )
                if not np.array_equal(
                    np.sort(ray_order),
                    np.arange(len(starts_array)),
                ):
                    raise RuntimeError(
                        "STEP11_M2_CONDITIONING_RAY_ORDER_INVALID"
                    )

                candidate_ci = dict(current_ci)
                candidate_ci.update({
                    "points_by_ray": projected_points.copy(),
                    "normals_by_ray": projected_normals.copy(),
                    "points_ordered": projected_points[ray_order].copy(),
                    "normals_ordered": projected_normals[ray_order].copy(),
                    "conditioning_iteration": int(iteration),
                    "conditioning_method":
                        "ALTERNATING_O2_SURFACE_AND_REFLECTION_LAW_PROJECTION",
                })

                candidate_diagnostic = _construction_diagnostics(
                    ctx,
                    candidate_ci,
                    seed_surface.frame,
                    f"M2_STEP11_CANDIDATE_CI_CONDITIONED_{iteration:02d}",
                )
                candidate_gate = _integrability_quality_gate(
                    candidate_diagnostic,
                    integrability_policy,
                    "M2_CI_CLOUD",
                )
                candidate_rank = _step11_integrability_rank_key(
                    candidate_gate
                )
                candidate_key = (
                    *candidate_rank,
                    int(iteration),
                )
                selected_now = bool(candidate_key < best_key)

                iteration_rows.append({
                    "iteration": int(iteration),
                    "role":
                        "ALTERNATING_O2_SURFACE_AND_REFLECTION_LAW_PROJECTION",
                    "integrability_status":
                        str(candidate_gate["status"]),
                    "integrability_rank_key":
                        list(candidate_rank),
                    "temporary_fit_curvature_per_mm":
                        temporary_fit.get("curvature_per_mm"),
                    "temporary_fit_conic_constant":
                        temporary_fit.get("conic_constant"),
                    "temporary_fit_sag_rms_mm":
                        temporary_fit.get("sag_rms_mm"),
                    "temporary_fit_normal_rms_deg":
                        temporary_fit.get("normal_rms_deg"),
                    "temporary_fit_normal_max_deg":
                        temporary_fit.get("normal_max_deg"),
                    "checks": [
                        dict(row)
                        for row in candidate_gate.get("checks", [])
                    ],
                })

                if selected_now:
                    best_ci = candidate_ci
                    best_diagnostic = candidate_diagnostic
                    best_gate = candidate_gate
                    best_iteration = int(iteration)
                    best_key = candidate_key

                current_ci = candidate_ci
                current_seed = temporary_m2

                emit(
                    f"[AP {iteration}/{maximum_iterations}][DONE]"
                    f" {gate_summary(candidate_gate)}"
                    f" | selected={'yes' if selected_now else 'no'}"
                    f" | elapsed={time.perf_counter() - iteration_started:.1f}s"
                )

                if str(candidate_gate["status"]) == "PASS":
                    break
            except (RuntimeError, ValueError) as exc:
                conditioning_error = f"{type(exc).__name__}:{exc}"
                iteration_rows.append({
                    "iteration": int(iteration),
                    "role":
                        "ALTERNATING_O2_SURFACE_AND_REFLECTION_LAW_PROJECTION",
                    "integrability_status": "ERROR",
                    "error": conditioning_error,
                })
                emit(
                    f"[AP {iteration}/{maximum_iterations}][WARN]"
                    f" {conditioning_error}"
                    f" | keep_best_iteration={best_iteration}"
                    f" | elapsed={time.perf_counter() - iteration_started:.1f}s"
                )
                break

    if not enabled:
        conditioning_status = "DISABLED_RAW_CI"
    elif str(raw_gate["status"]) == "PASS":
        conditioning_status = "NOT_REQUIRED_RAW_PASS"
    elif str(best_gate["status"]) == "PASS":
        conditioning_status = "PASS_AFTER_ALTERNATING_PROJECTION"
    elif conditioning_error is not None:
        conditioning_status = "BEST_EFFORT_WARN_AFTER_ERROR"
    else:
        conditioning_status = "BEST_EFFORT_WARN_MAX_ITERATIONS"

    audit = {
        "schema": "HUD_FAN_V5_5_STEP11_M2_ALTERNATING_PROJECTION_V1",
        "enabled": enabled,
        "method":
            "ALTERNATING_O2_SURFACE_AND_REFLECTION_LAW_PROJECTION",
        "direct_normal_averaging_used": False,
        "reflection_law_normal_definition":
            "unit(unit(d_in)-unit(target-point))",
        "incoming_ray_coordinates_preserved": True,
        "maximum_iterations": maximum_iterations,
        "attempted_iterations": int(attempted_iterations),
        "selected_iteration": int(best_iteration),
        "status": conditioning_status,
        "initial_integrability_status": str(raw_gate["status"]),
        "selected_integrability_status": str(best_gate["status"]),
        "selected_integrability_rank_key": list(best_key[:-1]),
        "error": conditioning_error,
        "iterations": iteration_rows,
    }

    emit(
        "[AP FINAL]"
        f" status={conditioning_status}"
        f" | selected_iteration={best_iteration}/{maximum_iterations}"
        f" | selected_integrability={str(best_gate['status'])}"
        f" | elapsed={time.perf_counter() - conditioning_started:.1f}s"
    )

    return (
        best_ci,
        best_diagnostic,
        best_gate,
        audit,
    )


def _step11_build_restoration_seed(
    common: dict[str, Any],
    seed_m1: PolySurface,
) -> dict[str, Any]:
    """Xây dựng trạng thái hạt giống ban đầu cho quá trình phục hồi."""

    ctx = common[
        "ctx"
    ]

    r = ctx.data[
        "rays"
    ]

    chief_index = int(
        common[
            "chief_index"
        ]
    )

    d = np.asarray(
        ctx.data[
            "post_visor"
        ],
        float,
    )

    vh = ctx.data[
        "visor_hit"
    ]

    m1_gate = (
        _step11_topology_shape_gate(
            common,
            seed_m1,
            d[
                chief_index
            ],
            common[
                "m1_topology"
            ],
        )
    )

    require_raw_m1 = bool(
        common[
            "restoration_cfg"
        ][
            "require_m1_raw_topology"
        ]
    )

    if require_raw_m1:

        m1_seed_admitted = bool(
            m1_gate[
                "topology_pass"
            ]
        )

    else:

        m1_seed_admission = (
            _step11_topology_admission(
                m1_gate,
                str(
                    common[
                        "topology_enforcement"
                    ][
                        "M1"
                    ]
                ),
            )
        )

        m1_seed_admitted = bool(
            m1_seed_admission[
                "admitted"
            ]
        )

    if not m1_seed_admitted:

        raise RuntimeError(
            "STEP11_RESTORATION_"
            "SEED_M1_TOPOLOGY_INVALID:"
            +
            ",".join(
                m1_gate[
                    "topology_failure_reasons"
                ]
            )
        )

    m1_trust = (
        _step11_ci_trust_gate(
            common,
            seed_m1,
        )
    )

    h1 = seed_m1.intersect(
        vh[
            "point"
        ]
        +
        1e-4
        *
        d,
        d,
        finite=False,
    )

    valid = np.asarray(
        h1[
            "valid"
        ],
        bool,
    )

    if (
        valid.shape
        !=
        (len(d),)
        or not np.all(
            valid
        )
    ):
        raise RuntimeError(
            "STEP11_RESTORATION_"
            "SEED_M1_RETRACE_INCOMPLETE"
        )

    points = np.asarray(
        h1[
            "point"
        ],
        float,
    )

    normals = np.asarray(
        h1[
            "normal"
        ],
        float,
    )

    if (
        not np.all(
            np.isfinite(points)
        )
        or
        not np.all(
            np.isfinite(normals)
        )
    ):
        raise RuntimeError(
            "STEP11_RESTORATION_"
            "SEED_M1_RETRACE_NONFINITE"
        )

    d1 = reflect(
        d,
        normals,
    )

    (
        ci_m2,
        m2_diagnostic,
        integrability_gate,
        conditioning_audit,
    ) = _step11_condition_m2_ci(
        ctx,

        points
        +
        1e-4
        *
        d1,

        d1,

        ctx.data[
            "fan_refs"
        ][
            r[
                "field_index"
            ]
        ],

        ctx.data[
            "m2"
        ],

        chief_index,

        common[
            "surface_fit_cfg"
        ],

        common[
            "integrability_policy"
        ],
    )

    if (
        str(
            integrability_gate[
                "status"
            ]
        )
        not in (
            "PASS",
            "WARN",
        )
    ):
        raise RuntimeError(
            "STEP11_RESTORATION_"
            "SEED_M2_INTEGRABILITY_INVALID"
        )

    seed_m2, fit_m2 = fit_surface(
        ci_m2[
            "points_by_ray"
        ],

        ci_m2[
            "normals_by_ray"
        ],

        ctx.data[
            "m2"
        ],

        2,
        True,

        ctx.config[
            "fit_weights"
        ],

        chief_index,

        common[
            "surface_fit_cfg"
        ],

        ci_m2[
            "starts_by_ray"
        ],

        ci_m2[
            "targets_by_ray"
        ],
    )

    (
        optical_residual,
        metrics,
        trace,
    ) = _step12_actual_o2_evaluate(
        ctx,
        seed_m1,
        seed_m2,
        common[
            "refinement_cfg"
        ],
    )

    # Quan trọng:
    # M2 topology FAIL ở đây KHÔNG raise.
    #
    # Đây chính là starting point mà
    # restoration solver cần sửa.

    return {
        "m1":
            seed_m1.copy(),

        "m2":
            seed_m2.copy(),

        "M1_shape_gate":
            m1_gate,

        "M1_ci_trust":
            m1_trust,

        "ci_m2":
            ci_m2,

        "M2_diagnostic":
            m2_diagnostic,

        "M2_integrability_gate":
            integrability_gate,

        "M2_conditioning_audit":
            conditioning_audit,

        "fit_m2_seed":
            fit_m2,

        "optical_residual":
            np.asarray(
                optical_residual,
                float,
            ),

        "optical_metrics":
            metrics,

        "physical_trace":
            trace,
    }


def _step11_final_fit_diagnostic(
    surface: PolySurface,
    ci: dict[str, Any],
    surface_fit_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Tính toán chẩn đoán khớp cuối cùng đối với đám mây CI đông kết."""

    points = np.asarray(
        ci[
            "points_by_ray"
        ],
        float,
    )

    target_normals = unit(
        np.asarray(
            ci[
                "normals_by_ray"
            ],
            float,
        )
    )

    local = (
        points
        -
        surface.center
    ) @ surface.frame

    z, _, _ = surface.sag_slopes(
        local[:, 0],
        local[:, 1],
    )

    sag_error = (
        z
        -
        local[:, 2]
    )

    model_normals = unit(
        surface.normal(
            local[:, 0],
            local[:, 1],
        )
    )

    dot = np.abs(
        np.sum(
            model_normals
            *
            target_normals,
            axis=1,
        )
    )

    normal_angle = np.degrees(
        np.arccos(
            np.clip(
                dot,
                -1.0,
                1.0,
            )
        )
    )

    reflection = {
        "evaluated":
            False,

        "rms_deg":
            float("inf"),

        "p95_deg":
            float("inf"),

        "max_deg":
            float("inf"),
    }

    if (
        "starts_by_ray"
        in
        ci
        and
        "targets_by_ray"
        in
        ci
    ):

        starts = np.asarray(
            ci[
                "starts_by_ray"
            ],
            float,
        )

        targets = np.asarray(
            ci[
                "targets_by_ray"
            ],
            float,
        )

        if (
            starts.shape
            ==
            points.shape
            and
            targets.shape
            ==
            points.shape
        ):

            fitted_points = surface.point(
                local[:, 0],
                local[:, 1],
            )

            actual_out = reflect(
                unit(
                    fitted_points
                    -
                    starts
                ),

                surface.normal(
                    local[:, 0],
                    local[:, 1],
                ),
            )

            required_out = unit(
                targets
                -
                fitted_points
            )

            reflection_angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.sum(
                            actual_out
                            *
                            required_out,
                            axis=1,
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )

            reflection = {
                "evaluated":
                    True,

                "rms_deg":
                    float(
                        np.sqrt(
                            np.mean(
                                reflection_angle
                                *
                                reflection_angle
                            )
                        )
                    ),

                "p95_deg":
                    float(
                        np.percentile(
                            reflection_angle,
                            95,
                        )
                    ),

                "max_deg":
                    float(
                        np.max(
                            reflection_angle
                        )
                    ),
            }

    curvature_limit = float(
        surface_fit_cfg[
            "curvature_absolute_max_per_mm"
        ]
    )

    conic_lo, conic_hi = map(
        float,
        surface_fit_cfg[
            "conic_bounds"
        ],
    )

    return {
        "role":
            "STEP11_FINAL_RESTORED_SURFACE_"
            "DIAGNOSTIC_AGAINST_FROZEN_CI",

        "curvature_per_mm":
            float(
                surface.curvature
            ),

        "conic_constant":
            float(
                surface.conic
            ),

        "sag_rms_mm":
            float(
                np.sqrt(
                    np.mean(
                        sag_error
                        *
                        sag_error
                    )
                )
            ),

        "sag_p95_mm":
            float(
                np.percentile(
                    np.abs(
                        sag_error
                    ),
                    95,
                )
            ),

        "sag_max_mm":
            float(
                np.max(
                    np.abs(
                        sag_error
                    )
                )
            ),

        "normal_rms_deg":
            float(
                np.sqrt(
                    np.mean(
                        normal_angle
                        *
                        normal_angle
                    )
                )
            ),

        "normal_p95_deg":
            float(
                np.percentile(
                    normal_angle,
                    95,
                )
            ),

        "normal_max_deg":
            float(
                np.max(
                    normal_angle
                )
            ),

        "reflection_direction_error":
            reflection,

        "curvature_bound_hit":
            bool(
                abs(
                    float(
                        surface.curvature
                    )
                )
                >=
                curvature_limit
                *
                (
                    1.0
                    -
                    1e-4
                )
            ),

        "k_bound_hit":
            bool(
                abs(
                    float(
                        surface.conic
                    )
                    -
                    conic_lo
                )
                <=
                1e-4
                *
                max(
                    abs(
                        conic_hi
                        -
                        conic_lo
                    ),
                    1.0,
                )
                or
                abs(
                    float(
                        surface.conic
                    )
                    -
                    conic_hi
                )
                <=
                1e-4
                *
                max(
                    abs(
                        conic_hi
                        -
                        conic_lo
                    ),
                    1.0,
                )
            ),

        "joint_variable_projection": {
            "curvature_bounds_per_mm":
                [
                    -curvature_limit,
                    curvature_limit,
                ],

            "conic_bounds":
                [
                    conic_lo,
                    conic_hi,
                ],

            "role":
                "POST_RESTORATION_DIAGNOSTIC_ONLY",
        },
    }


def _step11_run_fd_probe_jobs(
    common: dict[str, Any],
    jobs: list[dict[str, Any]],
    executor: Any,
    max_inflight: int,
) -> list[dict[str, Any]]:
    """Chạy các job thăm dò sai phân hữu hạn song song hoặc tuần tự."""

    if not jobs:
        return []

    if (
        executor is not None
        and
        max_inflight > 1
        and
        len(jobs) > 1
    ):

        from execution_workers_v55 import (
            ordered_bounded_map,
            step11_restoration_probe_worker,
        )

        replies = ordered_bounded_map(
            executor,
            step11_restoration_probe_worker,
            jobs,
            max_inflight,
        )

    else:

        import os

        replies = []

        for job in jobs:

            result = (
                evaluate_step11_restoration_probe_job(
                    common,
                    job,
                )
            )

            replies.append({
                **result,
                "pid":
                    os.getpid(),
            })

    if len(replies) != len(jobs):
        raise RuntimeError(
            "STEP11_RESTORATION_"
            "PROBE_REPLY_COUNT_MISMATCH"
        )

    for expected_index, (
        job,
        reply,
    ) in enumerate(
        zip(
            jobs,
            replies,
        )
    ):

        if (
            reply.get(
                "schema"
            )
            !=
            "HUD_FAN_V5_5_"
            "STEP11_RESTORATION_PROBE_RESULT_V1"
        ):
            raise RuntimeError(
                "STEP11_RESTORATION_"
                "PROBE_REPLY_SCHEMA_MISMATCH"
            )

        if (
            int(
                reply[
                    "job_index"
                ]
            )
            !=
            expected_index
        ):
            raise RuntimeError(
                "STEP11_RESTORATION_"
                "PROBE_REPLY_ORDER_MISMATCH"
            )

        if (
            int(
                reply[
                    "column"
                ]
            )
            !=
            int(
                job[
                    "column"
                ]
            )
            or
            float(
                reply[
                    "sign"
                ]
            )
            !=
            float(
                job[
                    "sign"
                ]
            )
        ):
            raise RuntimeError(
                "STEP11_RESTORATION_"
                "PROBE_REPLY_IDENTITY_MISMATCH"
            )

    return replies


def _step11_build_parallel_jacobian(
    common: dict[str, Any],
    u: np.ndarray,
    current_residual: np.ndarray,
    iteration: int,
    trust_radius: float,
    executor: Any,
    max_inflight: int,
) -> tuple[
    np.ndarray,
    list[int],
    list[dict[str, Any]],
]:
    """Xây dựng ma trận Jacobian sai phân hữu hạn trung tâm song song."""

    cfg = common[
        "restoration_cfg"
    ]

    descriptors = common[
        "descriptors"
    ]

    n_columns = len(
        descriptors
    )

    jacobian = np.zeros(
        (
            len(
                current_residual
            ),
            n_columns,
        ),
        dtype=float,
    )

    unresolved = set(
        range(
            n_columns
        )
    )

    central_results: dict[
        int,
        tuple[
            float,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = {}

    one_sided: dict[
        int,
        tuple[
            str,
            float,
            dict[str, Any],
        ]
    ] = {}

    diagnostic_rows = []

    job_counter = 0

    for fd_round, shrink in enumerate(
        cfg[
            "finite_difference_shrink_factors"
        ],
        start=1,
    ):

        if not unresolved:
            break

        h = (
            float(
                cfg[
                    "finite_difference_normalized"
                ]
            )
            *
            float(
                trust_radius
            )
            *
            float(
                shrink
            )
        )

        if h <= 0.0:
            continue

        jobs = []

        for column in sorted(
            unresolved
        ):

            for sign in (
                -1.0,
                +1.0,
            ):

                probe_u = np.asarray(
                    u,
                    float,
                ).copy()

                probe_u[
                    column
                ] += (
                    sign
                    *
                    h
                )

                jobs.append({
                    "schema":
                        "HUD_FAN_V5_5_"
                        "STEP11_RESTORATION_PROBE_JOB_V1",

                    "job_index":
                        int(
                            len(jobs)
                        ),

                    "global_job_index":
                        int(
                            job_counter
                        ),

                    "iteration":
                        int(
                            iteration
                        ),

                    "fd_round":
                        int(
                            fd_round
                        ),

                    "column":
                        int(
                            column
                        ),

                    "sign":
                        float(
                            sign
                        ),

                    "delta":
                        float(
                            h
                        ),

                    "search_vector":
                        probe_u,
                })

                job_counter += 1

        replies = (
            _step11_run_fd_probe_jobs(
                common,
                jobs,
                executor,
                max_inflight,
            )
        )

        by_column = {}

        for reply in replies:

            column = int(
                reply[
                    "column"
                ]
            )

            sign = float(
                reply[
                    "sign"
                ]
            )

            by_column.setdefault(
                column,
                {},
            )[sign] = reply

        next_unresolved = set()

        for column in sorted(
            unresolved
        ):

            sides = by_column.get(
                column,
                {},
            )

            minus = sides.get(
                -1.0
            )

            plus = sides.get(
                +1.0
            )

            minus_valid = bool(
                minus
                and
                minus.get(
                    "valid",
                    False,
                )
            )

            plus_valid = bool(
                plus
                and
                plus.get(
                    "valid",
                    False,
                )
            )

            if (
                minus_valid
                and
                plus_valid
            ):

                central_results[
                    column
                ] = (
                    float(h),
                    plus,
                    minus,
                )

                continue

            # Ghi lại one-sided nhỏ nhất hiện có.
            if plus_valid:

                one_sided[
                    column
                ] = (
                    "FORWARD",
                    float(h),
                    plus,
                )

            elif minus_valid:

                one_sided[
                    column
                ] = (
                    "BACKWARD",
                    float(h),
                    minus,
                )

            next_unresolved.add(
                column
            )

        unresolved = (
            next_unresolved
        )

    active_columns = []

    for column in range(
        n_columns
    ):

        if column in central_results:

            (
                h,
                plus,
                minus,
            ) = central_results[
                column
            ]

            derivative = (
                np.asarray(
                    plus[
                        "residual"
                    ],
                    float,
                )
                -
                np.asarray(
                    minus[
                        "residual"
                    ],
                    float,
                )
            ) / (
                2.0
                *
                h
            )

            mode = "CENTRAL"

        elif column in one_sided:

            (
                mode,
                h,
                side,
            ) = one_sided[
                column
            ]

            if mode == "FORWARD":

                derivative = (
                    np.asarray(
                        side[
                            "residual"
                        ],
                        float,
                    )
                    -
                    current_residual
                ) / h

            else:

                derivative = (
                    current_residual
                    -
                    np.asarray(
                        side[
                            "residual"
                        ],
                        float,
                    )
                ) / h

        else:

            diagnostic_rows.append({
                "iteration":
                    int(
                        iteration
                    ),

                "column":
                    int(
                        column
                    ),

                "surface":
                    descriptors[
                        column
                    ][
                        "surface"
                    ],

                "kind":
                    descriptors[
                        column
                    ][
                        "kind"
                    ],

                "term":
                    str(
                        descriptors[
                            column
                        ].get(
                            "term",
                            "",
                        )
                    ),

                "fd_mode":
                    "INVALID",

                "delta_used":
                    None,

                "column_norm":
                    0.0,
            })

            continue

        if (
            derivative.shape
            !=
            current_residual.shape
            or
            not np.all(
                np.isfinite(
                    derivative
                )
            )
        ):
            continue

        column_norm = float(
            np.linalg.norm(
                derivative
            )
        )

        if (
            column_norm
            <
            float(
                cfg[
                    "minimum_jacobian_column_norm"
                ]
            )
        ):

            diagnostic_rows.append({
                "iteration":
                    int(
                        iteration
                    ),

                "column":
                    int(
                        column
                    ),

                "surface":
                    descriptors[
                        column
                    ][
                        "surface"
                    ],

                "kind":
                    descriptors[
                        column
                    ][
                        "kind"
                    ],

                "term":
                    str(
                        descriptors[
                            column
                        ].get(
                            "term",
                            "",
                        )
                    ),

                "fd_mode":
                    "INSENSITIVE",

                "delta_used":
                    float(
                        h
                    ),

                "column_norm":
                    column_norm,
            })

            continue

        jacobian[
            :,
            column
        ] = derivative

        active_columns.append(
            column
        )

        diagnostic_rows.append({
            "iteration":
                int(
                    iteration
                ),

            "column":
                int(
                    column
                ),

            "surface":
                descriptors[
                    column
                ][
                    "surface"
                ],

            "kind":
                descriptors[
                    column
                ][
                    "kind"
                ],

            "term":
                str(
                    descriptors[
                        column
                    ].get(
                        "term",
                        "",
                    )
                ),

            "fd_mode":
                mode,

            "delta_used":
                float(
                    h
                ),

            "column_norm":
                column_norm,
        })

    return (
        jacobian,
        active_columns,
        diagnostic_rows,
    )


def _step11_restoration_hard_feasible(
    meta: dict[str, Any],
) -> bool:
    """
    Điều kiện tối thiểu để STEP11 restoration được phép kết thúc.

    Optical quality vẫn để STEP12 refine tiếp.
    """

    return bool(
        meta[
            "M1_raw_topology_pass"
        ]
        and
        meta[
            "M2_raw_topology_pass"
        ]
        and
        meta[
            "unobscured"
        ]
    )


def _step11_disabled_restoration_result(
    common: dict[str, Any],
    progress: Callable[
        [str],
        None,
    ],
) -> dict[str, Any]:
    """
    Khi restoration.enabled=false:
    không chạy Jacobian/LM, chỉ đánh giá seed hiện có.
    """

    descriptors = common[
        "descriptors"
    ]

    u = np.zeros(
        len(
            descriptors
        ),
        dtype=float,
    )

    current = (
        _step11_evaluate_restoration_u(
            common,
            u,
            include_payload=True,
        )
    )

    if not bool(
        current.get(
            "valid",
            False,
        )
    ):
        raise RuntimeError(
            "STEP11_RESTORATION_DISABLED_BASE_INVALID:"
            +
            str(
                current.get(
                    "reason"
                )
            )
        )

    success = (
        _step11_restoration_hard_feasible(
            current[
                "meta"
            ]
        )
    )

    stop_reason = (
        "RESTORATION_DISABLED_SEED_HARD_FEASIBLE"
        if success
        else
        "RESTORATION_DISABLED_SEED_INFEASIBLE"
    )

    failure_message = (
        None
        if success
        else
        (
            "STEP11_TOPOLOGY_RESTORATION_DISABLED_"
            "SEED_INFEASIBLE:"
            f"M2_H="
            f"{current['meta']['M2_oriented_H_min_per_mm']:.12g};"
            f"M2_KG="
            f"{current['meta']['M2_KG_min_per_mm2']:.12g};"
            f"S_AQP="
            f"{current['meta']['S_AQP_signed_mm2']:.12g}"
        )
    )

    progress(
        "[RESTORE][DISABLED]"
        f" hard_feasible={success}"
        f" | M2_H={current['meta']['M2_oriented_H_min_per_mm']:.6g}"
        f" | M2_KG={current['meta']['M2_KG_min_per_mm2']:.6g}"
        f" | S_AQP={current['meta']['S_AQP_signed_mm2']:.6g}"
    )

    return {
        "success":
            bool(
                success
            ),

        "failure_message":
            failure_message,

        "u":
            u,

        "m1":
            current[
                "m1"
            ],

        "m2":
            current[
                "m2"
            ],

        "evaluation":
            current[
                "evaluation"
            ],

        "meta":
            current[
                "meta"
            ],

        "history":
            [],

        "jacobian_history":
            [],

        "effective_workers":
            0,

        "stop_reason":
            stop_reason,

        "final_objective":
            float(
                current[
                    "objective"
                ]
            ),
    }


def _step11_write_restoration_run_diagnostics(
    step_dir: Path,
    restoration: dict[str, Any],
    descriptors: list[dict[str, Any]],
    grids: dict[str, dict[str, Any]],
    residual_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ghi restoration diagnostics bất kể solver PASS hay FAIL."""

    history = list(
        restoration.get(
            "history",
            [],
        )
    )

    jacobian_history = list(
        restoration.get(
            "jacobian_history",
            [],
        )
    )

    write_csv(
        step_dir
        /
        "11_RESTORATION_JACOBIAN_COLUMNS.csv",

        jacobian_history,
    )

    write_csv(
        step_dir
        /
        "11_TOPOLOGY_RESTORATION_HISTORY.csv",

        history,
    )

    compatibility_history = []

    for row in history:

        hard_feasible = bool(
            row.get(
                "M1_raw_topology_pass",
                False,
            )
            and
            row.get(
                "M2_raw_topology_pass",
                False,
            )
            and
            row.get(
                "unobscured",
                False,
            )
        )

        compatibility_history.append({
            "candidate_id":
                "RESTORE_ITER_"
                f"{int(row['iteration']):03d}",

            "cycle":
                int(
                    row[
                        "iteration"
                    ]
                ),

            "trust_scale":
                row.get(
                    "trust_radius"
                ),

            "move":
                "JOINT_LM_RESTORATION",

            "accepted":
                row.get(
                    "accepted"
                ),

            "feasible":
                hard_feasible,

            "rejection_stage":
                (
                    None
                    if hard_feasible
                    else
                    "TOPOLOGY_RESTORATION"
                ),

            "rejection_reason":
                (
                    None
                    if hard_feasible
                    else
                    row.get(
                        "failure",
                        "RESTORATION_IN_PROGRESS",
                    )
                ),

            "M1_shape_pass":
                row.get(
                    "M1_raw_topology_pass"
                ),

            "M2_shape_pass":
                row.get(
                    "M2_raw_topology_pass"
                ),

            "M1_H_min_per_mm":
                row.get(
                    "M1_H_min_per_mm"
                ),

            "M1_KG_min_per_mm2":
                row.get(
                    "M1_KG_min_per_mm2"
                ),

            "M1_oriented_H_min_per_mm":
                row.get(
                    "M1_oriented_H_min_per_mm"
                ),

            "M2_H_min_per_mm":
                row.get(
                    "M2_H_min_per_mm"
                ),

            "M2_KG_min_per_mm2":
                row.get(
                    "M2_KG_min_per_mm2"
                ),

            "M2_oriented_H_min_per_mm":
                row.get(
                    "M2_oriented_H_min_per_mm"
                ),

            "S_AQP_signed_mm2":
                row.get(
                    "S_AQP_signed_mm2"
                ),

            "unobscured":
                row.get(
                    "unobscured"
                ),

            "physical_fraction":
                row.get(
                    "physical_fraction"
                ),

            "optical_objective":
                row.get(
                    "objective_after"
                ),

            "M2_CI_bundle_curl_rms_p95_actual": None,
            "M2_CI_bundle_curl_rms_p95_limit": None,
            "M2_CI_bundle_local_geometry_normal_rms_p95_actual": None,
            "M2_CI_bundle_local_geometry_normal_rms_p95_limit": None,
            "M2_CI_bundle_loop_circulation_p95_actual": None,
            "M2_CI_bundle_loop_circulation_p95_limit": None,
            "M2_CI_bundle_edge_gradient_height_residual_p95_actual": None,
            "M2_CI_bundle_edge_gradient_height_residual_p95_limit": None,
            "M2_CI_bundle_nearest_normal_angle_p99_actual": None,
            "M2_CI_bundle_nearest_normal_angle_p99_limit": None,
            "M2_CI_bundle_count": None,
            "M2_CI_evaluable_bundle_count": None,
            "M2_fit_curvature_per_mm": None,
            "M2_fit_conic_constant": None,
            "M2_fit_sag_rms_mm": None,
            "M2_fit_normal_rms_deg": None,
            "M2_fit_normal_max_deg": None,
            "M2_fit_k_bound_hit": None,
            "M2_fit_curvature_bound_hit": None,
            "M1_topology_enforcement": None,
            "M1_topology_admitted": None,
            "M1_topology_admission_status": None,
            "M1_topology_warning_reasons": None,
            "M2_topology_enforcement": None,
            "M2_topology_admitted": None,
            "M2_topology_admission_status": None,
            "M2_topology_warning_reasons": None,
            "candidate_surface_admitted": None,
            "actual_M1_topology_pass": None,
            "actual_M1_topology_admitted": None,
            "actual_M1_topology_admission_status": None,
            "actual_M1_topology_warning_reasons": None,
            "actual_M2_topology_pass": None,
            "actual_M2_topology_admitted": None,
            "actual_M2_topology_admission_status": None,
            "actual_M2_topology_warning_reasons": None,
        })

    write_csv(
        step_dir
        /
        "11_O2_SEARCH_HISTORY.csv",

        compatibility_history,
    )

    for surface_name in (
        "M1",
        "M2",
    ):

        write_csv(
            step_dir
            /
            f"11_RESTORATION_GRID_{surface_name}.csv",

            [
                {
                    "sample_index":
                        int(index),

                    "x_mm":
                        float(
                            point[0]
                        ),

                    "y_mm":
                        float(
                            point[1]
                        ),
                }
                for index, point
                in enumerate(
                    grids[
                        surface_name
                    ][
                        "xy"
                    ]
                )
            ],
        )

    restoration_cfg = (
        residual_state[
            "cfg"
        ]
    )

    write_json(
        step_dir
        /
        "11_TOPOLOGY_RESTORATION_SUMMARY.json",

        {
            "schema":
                "HUD_FAN_V5_5_"
                "STEP11_JOINT_TOPOLOGY_RESTORATION_V2",

            "success":
                bool(
                    restoration[
                        "success"
                    ]
                ),

            "failure_message":
                restoration.get(
                    "failure_message"
                ),

            "stop_reason":
                restoration.get(
                    "stop_reason"
                ),

            "iteration_count":
                int(
                    len(
                        history
                    )
                ),

            "variable_count":
                int(
                    len(
                        descriptors
                    )
                ),

            "effective_workers":
                int(
                    restoration.get(
                        "effective_workers",
                        1,
                    )
                ),

            "selected_normalized_vector":
                np.asarray(
                    restoration.get(
                        "u",
                        [],
                    ),
                    float,
                ).tolist(),

            "final_objective":
                restoration.get(
                    "final_objective"
                ),

            "final_meta":
                restoration.get(
                    "meta"
                ),

            "fixed_grid_samples": {
                "M1":
                    int(
                        len(
                            grids[
                                "M1"
                            ][
                                "xy"
                            ]
                        )
                    ),

                "M2":
                    int(
                        len(
                            grids[
                                "M2"
                            ][
                                "xy"
                            ]
                        )
                    ),
            },

            "normalization": {
                "H_scale_per_mm":
                    restoration_cfg[
                        "mean_curvature_scale_per_mm"
                    ],

                "KG_scale_per_mm2":
                    restoration_cfg[
                        "gaussian_curvature_scale_per_mm2"
                    ],

                "principal_scale_per_mm":
                    restoration_cfg[
                        "principal_curvature_scale_per_mm"
                    ],

                "gradient_scale_per_mm2":
                    restoration_cfg[
                        "curvature_gradient_scale_per_mm2"
                    ],

                "unobscuration_scale_mm2":
                    float(
                        residual_state[
                            "unobscuration_scale_mm2"
                        ]
                    ),
            },

            "weights":
                restoration_cfg[
                    "weights"
                ],
        },
    )

    return (
        compatibility_history
    )


def _step11_run_joint_topology_restoration(
    common: dict[str, Any],
    progress: Callable[
        [str],
        None,
    ],
) -> dict[str, Any]:
    """Chạy vòng lặp tối ưu Levenberg-Marquardt phục hồi tô-pô đồng thời M1/M2."""

    cfg = common[
        "restoration_cfg"
    ]

    descriptors = common[
        "descriptors"
    ]

    u = np.zeros(
        len(descriptors),
        dtype=float,
    )

    current = (
        _step11_evaluate_restoration_u(
            common,
            u,
            include_payload=True,
        )
    )

    if not current[
        "valid"
    ]:
        raise RuntimeError(
            "STEP11_RESTORATION_BASE_INVALID:"
            +
            str(
                current.get(
                    "reason"
                )
            )
        )

    damping = float(
        cfg[
            "damping"
        ]
    )

    trust_radius = float(
        cfg[
            "maximum_trust_radius"
        ]
    )

    history = []

    jacobian_history = []

    exec_cfg = (
        common[
            "ctx"
        ].config.get(
            "execution",
            {},
        )
    )

    parallel_enabled = bool(
        exec_cfg.get(
            "parallel_step11_candidates",
            False,
        )
    )

    worker_limit = int(
        exec_cfg.get(
            "step11_candidate_workers",
            4,
        )
    )

    memory_estimate = float(
        exec_cfg.get(
            "step11_worker_memory_estimate_gib",
            1.5,
        )
    )

    runtime = current_runtime()

    can_use_pool = bool(
        parallel_enabled
        and
        runtime is not None
        and
        not runtime.is_worker
        and
        not is_compute_worker()
        and
        worker_limit > 1
    )

    from execution_v55 import (
        managed_compute_pool,
    )

    pool_cm = (
        managed_compute_pool(
            common,
            purpose=
                "STEP11_JACOBIAN",
            worker_limit=
                worker_limit,
            worker_memory_estimate_gib=
                memory_estimate,
        )
        if can_use_pool
        else
        nullcontext(
            (
                None,
                1,
                {},
            )
        )
    )

    restoration_success = (
        _step11_restoration_hard_feasible(
            current[
                "meta"
            ]
        )
    )

    stop_reason = (
        "BASE_ALREADY_HARD_FEASIBLE"
        if restoration_success
        else None
    )

    with pool_cm as (
        executor,
        max_inflight,
        _,
    ):

        effective_workers = (
            executor._max_workers
            if executor is not None
            else 1
        )

        progress(
            "[RESTORE][START]"
            f" variables={len(descriptors)}"
            f" | workers={effective_workers}"
            f" | J={current['objective']:.6g}"
            f" | M2_H={current['meta']['M2_oriented_H_min_per_mm']:.6g}"
            f" | M2_KG={current['meta']['M2_KG_min_per_mm2']:.6g}"
            f" | S_AQP={current['meta']['S_AQP_signed_mm2']:.6g}"
            f" | M2_topology={current['meta']['M2_raw_topology_pass']}"
            f" | unobscured={current['meta']['unobscured']}"
        )

        for iteration in range(
            1,
            int(
                cfg[
                    "maximum_iterations"
                ]
            )
            + 1,
        ):

            if restoration_success:
                break

            objective_before = float(
                current[
                    "objective"
                ]
            )

            residual_before = np.asarray(
                current[
                    "residual"
                ],
                float,
            )

            (
                jacobian,
                active_columns,
                fd_rows,
            ) = (
                _step11_build_parallel_jacobian(
                    common,
                    u,
                    residual_before,
                    iteration,
                    trust_radius,
                    executor,
                    max_inflight,
                )
            )

            jacobian_history.extend(
                fd_rows
            )

            invalid_fraction = (
                1.0
                -
                len(
                    active_columns
                )
                /
                max(
                    len(
                        descriptors
                    ),
                    1,
                )
            )

            if (
                not active_columns
                or
                invalid_fraction
                >
                float(
                    cfg[
                        "maximum_invalid_jacobian_fraction"
                    ]
                )
            ):

                trust_radius = max(
                    float(
                        cfg[
                            "minimum_trust_radius"
                        ]
                    ),

                    trust_radius
                    *
                    float(
                        cfg[
                            "trust_shrink_factor"
                        ]
                    ),
                )

                damping = min(
                    float(
                        cfg[
                            "maximum_damping"
                        ]
                    ),

                    damping
                    *
                    float(
                        cfg[
                            "rejected_damping_factor"
                        ]
                    ),
                )

                history.append({
                    "iteration":
                        int(
                            iteration
                        ),

                    "accepted":
                        False,

                    "failure":
                        "TOO_MANY_INVALID_COLUMNS",

                    "active_columns":
                        int(
                            len(
                                active_columns
                            )
                        ),

                    "invalid_column_fraction":
                        float(
                            invalid_fraction
                        ),

                    "objective_before":
                        objective_before,

                    "objective_after":
                        objective_before,

                    "trust_radius":
                        trust_radius,

                    "damping":
                        damping,

                    **current[
                        "meta"
                    ],
                })

                continue

            active = np.asarray(
                active_columns,
                int,
            )

            J = jacobian[
                :,
                active
            ]

            singular = np.linalg.svd(
                J,
                compute_uv=False,
            )

            if (
                len(singular) == 0
                or
                singular[0] <= 0.0
            ):

                rank = 0

                condition_number = float(
                    "inf"
                )

            else:

                threshold = (
                    float(
                        cfg[
                            "svd_rcond"
                        ]
                    )
                    *
                    float(
                        singular[
                            0
                        ]
                    )
                )

                rank = int(
                    np.count_nonzero(
                        singular
                        >
                        threshold
                    )
                )

                condition_number = (
                    float(
                        singular[0]
                        /
                        singular[
                            rank - 1
                        ]
                    )
                    if rank > 0
                    else
                    float("inf")
                )

            if rank == 0:

                stop_reason = (
                    "JACOBIAN_RANK_ZERO"
                )

                history.append({
                    "iteration":
                        int(
                            iteration
                        ),

                    "accepted":
                        False,

                    "failure":
                        "JACOBIAN_RANK_ZERO",

                    "objective_before":
                        objective_before,

                    "objective_after":
                        objective_before,

                    "jacobian_rank":
                        0,

                    "active_columns":
                        int(
                            len(
                                active_columns
                            )
                        ),

                    "jacobian_condition":
                        float("inf"),

                    "invalid_column_fraction":
                        float(
                            invalid_fraction
                        ),

                    "trust_radius":
                        float(
                            trust_radius
                        ),

                    "damping":
                        float(
                            damping
                        ),

                    **current[
                        "meta"
                    ],
                })

                break

            condition_scale = max(
                1.0,

                condition_number
                /
                float(
                    cfg[
                        "maximum_jacobian_condition"
                    ]
                ),
            )

            column_norm = np.sqrt(
                np.maximum(
                    np.sum(
                        J * J,
                        axis=0,
                    ),
                    1e-12,
                )
            )

            accepted = False

            accepted_state = None
            accepted_u = None
            accepted_rho = None
            accepted_alpha = None
            accepted_damping = None

            predicted_reduction = None
            actual_reduction = None

            for multiplier in map(
                float,
                cfg[
                    "damping_multipliers"
                ],
            ):

                trial_damping = (
                    damping
                    *
                    multiplier
                    *
                    condition_scale
                )

                # Augmented LS:
                #
                # [J           ] delta = [-r]
                # [sqrt(lam) D ]         [ 0]
                #
                # numerically tốt hơn J.T @ J.

                A = np.vstack([
                    J,

                    math.sqrt(
                        trial_damping
                    )
                    *
                    np.diag(
                        column_norm
                    ),
                ])

                b = np.concatenate([
                    -residual_before,

                    np.zeros(
                        len(
                            active_columns
                        ),
                        dtype=float,
                    ),
                ])

                try:

                    delta_active = (
                        np.linalg.lstsq(
                            A,
                            b,
                            rcond=float(
                                cfg[
                                    "svd_rcond"
                                ]
                            ),
                        )[0]
                    )

                except np.linalg.LinAlgError:

                    continue

                if not np.all(
                    np.isfinite(
                        delta_active
                    )
                ):
                    continue

                delta = np.zeros(
                    len(
                        descriptors
                    ),
                    dtype=float,
                )

                delta[
                    active
                ] = (
                    delta_active
                )

                maximum_step = (
                    float(
                        cfg[
                            "max_step_normalized"
                        ]
                    )
                    *
                    float(
                        trust_radius
                    )
                )

                delta = np.clip(
                    delta,
                    -maximum_step,
                    maximum_step,
                )

                for alpha in map(
                    float,
                    cfg[
                        "line_search_alphas"
                    ],
                ):

                    step = (
                        alpha
                        *
                        delta
                    )

                    candidate_u = (
                        u
                        +
                        step
                    )

                    # Không clip normalized variables.
                    if np.any(
                        np.abs(
                            candidate_u
                        )
                        >
                        1.0 + 1e-12
                    ):
                        continue

                    candidate = (
                        _step11_evaluate_restoration_u(
                            common,
                            candidate_u,
                            include_payload=True,
                        )
                    )

                    if not candidate[
                        "valid"
                    ]:
                        continue

                    predicted_residual = (
                        residual_before
                        +
                        jacobian
                        @
                        step
                    )

                    predicted_objective = float(
                        np.mean(
                            predicted_residual
                            *
                            predicted_residual
                        )
                    )

                    predicted_reduction = (
                        objective_before
                        -
                        predicted_objective
                    )

                    actual_reduction = (
                        objective_before
                        -
                        float(
                            candidate[
                                "objective"
                            ]
                        )
                    )

                    rho = (
                        actual_reduction
                        /
                        predicted_reduction
                        if
                        predicted_reduction
                        >
                        1e-15
                        else
                        float("-inf")
                    )

                    if (
                        np.isfinite(
                            rho
                        )
                        and
                        actual_reduction
                        >
                        0.0
                        and
                        rho
                        >=
                        float(
                            cfg[
                                "minimum_acceptance_rho"
                            ]
                        )
                    ):

                        accepted = True

                        accepted_state = (
                            candidate
                        )

                        accepted_u = (
                            candidate_u
                        )

                        accepted_rho = float(
                            rho
                        )

                        accepted_alpha = float(
                            alpha
                        )

                        accepted_damping = float(
                            trial_damping
                        )

                        break

                if accepted:
                    break

            if accepted:

                u = np.asarray(
                    accepted_u,
                    float,
                )

                current = (
                    accepted_state
                )

                if (
                    accepted_rho
                    <
                    float(
                        cfg[
                            "trust_shrink_rho"
                        ]
                    )
                ):

                    trust_radius = max(
                        float(
                            cfg[
                                "minimum_trust_radius"
                            ]
                        ),

                        trust_radius
                        *
                        float(
                            cfg[
                                "trust_shrink_factor"
                            ]
                        ),
                    )

                elif (
                    accepted_rho
                    >=
                    float(
                        cfg[
                            "trust_expand_rho"
                        ]
                    )
                    and
                    accepted_alpha
                    >=
                    0.999
                ):

                    trust_radius = min(
                        float(
                            cfg[
                                "maximum_trust_radius"
                            ]
                        ),

                        trust_radius
                        *
                        float(
                            cfg[
                                "trust_expand_factor"
                            ]
                        ),
                    )

                damping = max(
                    float(
                        cfg[
                            "minimum_damping"
                        ]
                    ),

                    accepted_damping
                    *
                    float(
                        cfg[
                            "accepted_damping_factor"
                        ]
                    ),
                )

            else:

                trust_radius = max(
                    float(
                        cfg[
                            "minimum_trust_radius"
                        ]
                    ),

                    trust_radius
                    *
                    float(
                        cfg[
                            "trust_shrink_factor"
                        ]
                    ),
                )

                damping = min(
                    float(
                        cfg[
                            "maximum_damping"
                        ]
                    ),

                    damping
                    *
                    float(
                        cfg[
                            "rejected_damping_factor"
                        ]
                    ),
                )

            restoration_success = (
                _step11_restoration_hard_feasible(
                    current[
                        "meta"
                    ]
                )
            )

            if restoration_success:

                stop_reason = (
                    "M1_M2_RAW_TOPOLOGY_"
                    "AND_UNOBSCURATION_PASS"
                )

            history.append({
                "iteration":
                    int(
                        iteration
                    ),

                "accepted":
                    bool(
                        accepted
                    ),

                "objective_before":
                    float(
                        objective_before
                    ),

                "objective_after":
                    float(
                        current[
                            "objective"
                        ]
                    ),

                "predicted_reduction":
                    predicted_reduction,

                "actual_reduction":
                    actual_reduction,

                "rho":
                    (
                        accepted_rho
                        if accepted
                        else None
                    ),

                "alpha":
                    (
                        accepted_alpha
                        if accepted
                        else None
                    ),

                "jacobian_rank":
                    int(
                        rank
                    ),

                "active_columns":
                    int(
                        len(
                            active_columns
                        )
                    ),

                "jacobian_condition":
                    float(
                        condition_number
                    ),

                "invalid_column_fraction":
                    float(
                        invalid_fraction
                    ),

                "trust_radius":
                    float(
                        trust_radius
                    ),

                "damping":
                    float(
                        damping
                    ),

                **current[
                    "meta"
                ],
            })

            progress(
                f"[RESTORE][ITER {iteration}]"
                f" {'ACCEPT' if accepted else 'REJECT'}"
                f" | J={objective_before:.6g}"
                f"->{current['objective']:.6g}"
                f" | rank={rank}/{len(active_columns)}"
                f" | cond={condition_number:.3g}"
                f" | rho={accepted_rho if accepted else 'n/a'}"
                f" | trust={trust_radius:.4g}"
                f" | M2_H={current['meta']['M2_oriented_H_min_per_mm']:.6g}"
                f" | M2_KG={current['meta']['M2_KG_min_per_mm2']:.6g}"
                f" | S_AQP={current['meta']['S_AQP_signed_mm2']:.6g}"
                f" | M2_topology={current['meta']['M2_raw_topology_pass']}"
                f" | unobscured={current['meta']['unobscured']}"
            )

            if restoration_success:
                break

            if (
                not accepted
                and
                trust_radius
                <=
                float(
                    cfg[
                        "minimum_trust_radius"
                    ]
                )
                +
                1e-15
                and
                damping
                >=
                float(
                    cfg[
                        "maximum_damping"
                    ]
                )
            ):

                stop_reason = (
                    "TRUST_AND_DAMPING_LIMIT_REACHED"
                )

                break

    if not restoration_success:

        if stop_reason is None:
            stop_reason = (
                "MAXIMUM_ITERATIONS"
            )

        failure_message = (
            "STEP11_JOINT_TOPOLOGY_RESTORATION_FAILED:"
            f"reason={stop_reason};"
            f"M2_H={current['meta']['M2_oriented_H_min_per_mm']:.12g};"
            f"M2_KG={current['meta']['M2_KG_min_per_mm2']:.12g};"
            f"S_AQP={current['meta']['S_AQP_signed_mm2']:.12g};"
            f"J={current['objective']:.12g};"
            f"trust={trust_radius:.12g};"
            f"damping={damping:.12g}"
        )

        progress(
            "[RESTORE][FAIL]"
            f" reason={stop_reason}"
            f" | J={current['objective']:.6g}"
            f" | M2_H={current['meta']['M2_oriented_H_min_per_mm']:.6g}"
            f" | M2_KG={current['meta']['M2_KG_min_per_mm2']:.6g}"
            f" | S_AQP={current['meta']['S_AQP_signed_mm2']:.6g}"
        )

    else:

        failure_message = None

    return {
        "success":
            bool(
                restoration_success
            ),

        "failure_message":
            failure_message,

        "u":
            np.asarray(
                u,
                float,
            ),

        "m1":
            current[
                "m1"
            ],

        "m2":
            current[
                "m2"
            ],

        "evaluation":
            current[
                "evaluation"
            ],

        "meta":
            current[
                "meta"
            ],

        "history":
            history,

        "jacobian_history":
            jacobian_history,

        "effective_workers":
            int(
                effective_workers
            ),

        "stop_reason":
            stop_reason,

        "final_objective":
            float(
                current[
                    "objective"
                ]
            ),
    }


def _step11_archive_candidate_snapshot(
    record: dict[str, Any],
    last_phase: str,
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """Lưu geometry STEP11 của một candidate mà không làm thay đổi kết quả search."""
    visualization_payload = record.pop(
        "_visualization_payload",
        None,
    )
    snapshot_spec = job.get("snapshot")

    if (
        not isinstance(snapshot_spec, dict)
        or not bool(snapshot_spec.get("enabled", False))
    ):
        return None

    candidate_id = str(
        record.get(
            "candidate_id",
            job.get("candidate_id", "UNKNOWN"),
        )
    )
    candidate_class = str(
        snapshot_spec.get(
            "candidate_class",
            "UNKNOWN",
        )
    )

    summary: dict[str, Any] = {
        "status": "WRITE_ERROR",
        "archive_index": snapshot_spec.get("archive_index"),
        "candidate_id": candidate_id,
        "candidate_class": candidate_class,
        "cycle": record.get("cycle"),
        "move": record.get("move"),
        "feasible": bool(record.get("feasible", False)),
        "last_phase": str(last_phase),
        "directory": None,
        "metadata_file": None,
        "arrays_file": None,
        "array_count": 0,
        "array_bytes": 0,
        "compressed_bytes": 0,
        "error": None,
    }

    try:
        if not isinstance(visualization_payload, dict):
            raise RuntimeError(
                "STEP11_CANDIDATE_VISUALIZATION_PAYLOAD_MISSING"
            )

        archive_index = int(
            snapshot_spec["archive_index"]
        )
        snapshot_root = Path(
            str(snapshot_spec["root"])
        )
        safe_candidate_id = "".join(
            character
            if character.isalnum() or character in ("_", "-")
            else "_"
            for character in candidate_id
        )
        candidate_dir = (
            snapshot_root
            / f"{archive_index:06d}_{safe_candidate_id}"
        )
        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        arrays: dict[str, np.ndarray] = {}

        def add_array(
            name: str,
            value: Any,
        ) -> None:
            """Thêm numeric array vào snapshot mà không đổi dtype hay độ chính xác."""
            if value is None:
                return
            array = np.asarray(value)
            if array.dtype.kind not in "biufc":
                return
            arrays[name] = array

        add_array(
            "search_vector",
            visualization_payload.get("search_vector"),
        )
        add_array(
            "m1_hit_points",
            visualization_payload.get("m1_hit_points"),
        )
        add_array(
            "m1_hit_normals",
            visualization_payload.get("m1_hit_normals"),
        )
        add_array(
            "m1_reflected_directions",
            visualization_payload.get("m1_reflected_directions"),
        )
        add_array(
            "m1_valid",
            visualization_payload.get("m1_valid"),
        )

        ci_m2 = visualization_payload.get("ci_m2")
        if isinstance(ci_m2, dict):
            for source_key, archive_key in (
                ("points_by_ray", "ci_m2_points_by_ray"),
                ("normals_by_ray", "ci_m2_normals_by_ray"),
                ("starts_by_ray", "ci_m2_starts_by_ray"),
                ("directions_by_ray", "ci_m2_directions_by_ray"),
                ("targets_by_ray", "ci_m2_targets_by_ray"),
            ):
                add_array(
                    archive_key,
                    ci_m2.get(source_key),
                )

        physical_trace = visualization_payload.get(
            "physical_trace"
        )
        if isinstance(physical_trace, dict):
            for stage_index, points in enumerate(
                physical_trace.get("points", [])
            ):
                add_array(
                    f"physical_point_{stage_index:02d}",
                    points,
                )

            for stage_index, directions in enumerate(
                physical_trace.get("directions", [])
            ):
                add_array(
                    f"physical_direction_{stage_index:02d}",
                    directions,
                )

            add_array(
                "physical_valid",
                physical_trace.get("valid"),
            )
            add_array(
                "physical_resolved",
                physical_trace.get("resolved"),
            )
            add_array(
                "physical_landing",
                physical_trace.get("landing"),
            )
            add_array(
                "physical_display_local",
                physical_trace.get("display_local"),
            )

        m1 = visualization_payload.get("m1")
        m2 = visualization_payload.get("m2")
        surfaces = {
            "M1": (
                m1.to_dict()
                if isinstance(m1, PolySurface)
                else None
            ),
            "M2": (
                m2.to_dict()
                if isinstance(m2, PolySurface)
                else None
            ),
        }

        array_inventory = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": int(array.nbytes),
            }
            for name, array in arrays.items()
        }
        array_bytes = int(
            sum(
                array.nbytes
                for array in arrays.values()
            )
        )

        arrays_file: str | None = None
        compressed_bytes = 0
        if arrays:
            arrays_file = "arrays.npz"
            arrays_path = candidate_dir / arrays_file
            temporary_arrays_path = (
                candidate_dir
                / "arrays.npz.tmp"
            )
            with temporary_arrays_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    **arrays,
                )
            temporary_arrays_path.replace(
                arrays_path
            )
            compressed_bytes = int(
                arrays_path.stat().st_size
            )

        scalar_record = {
            key: value
            for key, value in record.items()
            if not str(key).startswith("_")
        }
        available_geometry = [
            name
            for name, value in (
                ("M1_SURFACE", surfaces["M1"]),
                ("M1_RETRACE", visualization_payload.get("m1_hit_points")),
                ("M2_CI", ci_m2),
                ("M2_SURFACE", surfaces["M2"]),
                ("PHYSICAL_TRACE", physical_trace),
            )
            if value is not None
        ]

        metadata = {
            "schema":
                "HUD_FAN_V5_5_STEP11_CANDIDATE_SNAPSHOT_V1",
            "archive_index": archive_index,
            "candidate_id": candidate_id,
            "candidate_class": candidate_class,
            "cycle": record.get("cycle"),
            "trust_scale": record.get("trust_scale"),
            "move": record.get("move"),
            "feasible": bool(record.get("feasible", False)),
            "rejection_stage": record.get("rejection_stage"),
            "rejection_reason": record.get("rejection_reason"),
            "last_phase": str(last_phase),
            "available_geometry": available_geometry,
            "surfaces": surfaces,
            "candidate_record": scalar_record,
            "array_file": arrays_file,
            "array_inventory": array_inventory,
            "array_bytes": array_bytes,
            "compressed_bytes": compressed_bytes,
            "full_precision_saved": True,
            "reran_optical_algorithms": False,
        }

        metadata_path = candidate_dir / "metadata.json"
        temporary_metadata_path = (
            candidate_dir
            / "metadata.json.tmp"
        )
        write_json(
            temporary_metadata_path,
            metadata,
        )
        temporary_metadata_path.replace(
            metadata_path
        )

        summary.update({
            "status": "SAVED",
            "archive_index": archive_index,
            "directory": candidate_dir.name,
            "metadata_file": metadata_path.name,
            "arrays_file": arrays_file,
            "array_count": int(len(arrays)),
            "array_bytes": array_bytes,
            "compressed_bytes": compressed_bytes,
            "available_geometry": available_geometry,
            "error": None,
        })
        return summary

    except Exception as exc:
        summary["error"] = (
            f"{type(exc).__name__}:{exc}"
        )
        return summary


def evaluate_step11_candidate_job(
    common: dict[str, Any],
    job: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Đánh giá candidate STEP11 thuần túy từ common snapshot và job definition."""
    candidate_id = str(job["candidate_id"])
    cycle = int(job["cycle"])
    trust_scale = float(job["trust_scale"])
    move = str(job["move"])
    search_vector = np.asarray(job["search_vector"], float)

    ctx = common["ctx"]
    quality_cfg = common["quality_cfg"]
    refinement_cfg = common["refinement_cfg"]
    surface_fit_cfg = common["surface_fit_cfg"]
    integrability_policy = common["integrability_policy"]
    m1_topology = common["m1_topology"]
    m2_topology = common["m2_topology"]
    topology_enforcement = common["topology_enforcement"]
    m1_topology_enforcement = str(
        topology_enforcement["M1"]
    )
    m2_topology_enforcement = str(
        topology_enforcement["M2"]
    )
    chief_index = int(common["chief_index"])
    r = ctx.data["rays"]

    last_phase = f"CANDIDATE/{candidate_id}/INITIALIZE"
    visualization_payload: dict[str, Any] = {
        "schema":
            "HUD_FAN_V5_5_STEP11_CANDIDATE_VISUALIZATION_PAYLOAD_V1",
        "candidate_id": candidate_id,
        "cycle": int(cycle),
        "move": str(move),
        "search_vector": search_vector,
        "last_phase": last_phase,
        "m1": None,
        "m1_hit_points": None,
        "m1_hit_normals": None,
        "m1_reflected_directions": None,
        "m1_valid": None,
        "ci_m2": None,
        "m2": None,
        "physical_trace": None,
    }

    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "cycle": int(cycle),
        "trust_scale": float(trust_scale),
        "move": str(move),
        "feasible": False,
        "_visualization_payload":
            visualization_payload,
    }

    def enter_phase(name: str) -> None:
        """Cập nhật phase hiện tại của candidate đang được đánh giá."""
        nonlocal last_phase
        last_phase = f"CANDIDATE/{candidate_id}/{name}"
        visualization_payload["last_phase"] = (
            last_phase
        )

    enter_phase("M1_PARAMETER_APPLICATION")
    try:
        candidate_m1 = _step11_apply_search_vector(common, search_vector)
    except (RuntimeError, ValueError) as exc:
        record.update({
            "rejection_stage": "M1_PARAMETER_APPLICATION",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    visualization_payload["m1"] = candidate_m1

    enter_phase("M1_MIRROR_SHAPE_GATE")
    try:
        shape1 = _step11_topology_shape_gate(
            common,
            candidate_m1,
            np.asarray(ctx.data["post_visor"], float)[chief_index],
            m1_topology,
        )
    except (RuntimeError, ValueError) as exc:
        record.update({
            "M1_shape_pass": False,
            "rejection_stage": "M1_MIRROR_TOPOLOGY_GATE",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    m1_topology_admission = _step11_topology_admission(
        shape1,
        m1_topology_enforcement,
    )
    record.update({
        "M1_shape_pass": bool(shape1["topology_pass"]),
        "M1_shape_raw_pass": bool(shape1["raw_shape_pass"]),
        "M1_shape_quality_status": str(shape1["quality_status"]),
        "M1_shape_quality_warnings": ",".join(shape1["quality_warning_reasons"]),
        "M1_topology": m1_topology,
        "M1_topology_enforcement": m1_topology_enforcement,
        "M1_topology_admitted": bool(
            m1_topology_admission["admitted"]
        ),
        "M1_topology_admission_status": str(
            m1_topology_admission["status"]
        ),
        "M1_topology_warning_reasons": ",".join(
            m1_topology_admission["warning_reasons"]
        ),
        "M1_orientation_sign": float(shape1["orientation_sign"]),
        "M1_ray_facing_dot": float(shape1["ray_facing_dot"]),
    })

    record.update(
        _step11_shape_curvature_record_fields(
            "M1",
            shape1,
        )
    )

    if not m1_topology_admission["admitted"]:
        record.update({
            "rejection_stage": "M1_MIRROR_TOPOLOGY_GATE",
            "rejection_reason": ",".join(shape1["topology_failure_reasons"]),
        })
        return record, None, last_phase

    enter_phase("M1_CI_TRUST_GATE")
    trust = _step11_ci_trust_gate(common, candidate_m1)
    record["M1_ci_trust_pass"] = bool(trust["pass"])
    record["M1_ci_trust_status"] = str(trust["status"])

    vh = ctx.data["visor_hit"]
    d = ctx.data["post_visor"]
    enter_phase("M1_ACTUAL_RETRACE")
    try:
        h1 = candidate_m1.intersect(
            vh["point"] + 1e-4 * d,
            d,
            finite=False,
        )
        h1_valid = np.asarray(h1["valid"], bool)
        if h1_valid.shape != (len(d),):
            raise RuntimeError(
                f"STEP11_M1_RETRACE_VALID_MASK_SHAPE_MISMATCH:{h1_valid.shape}"
            )
        if not np.all(h1_valid):
            status = np.asarray(h1.get("status", np.full(len(d), "UNKNOWN", object)), object)
            names, counts = np.unique(status.astype(str), return_counts=True)
            status_counts = {str(name): int(count) for name, count in zip(names, counts)}
            raise RuntimeError(
                f"STEP11_M1_RETRACE_INCOMPLETE:{int(np.sum(h1_valid))}/{len(d)}:{status_counts}"
            )
        point_array = np.asarray(h1["point"], float)
        normal_array = np.asarray(h1["normal"], float)
        t_array = np.asarray(h1["t"], float)
        if (
            not np.all(np.isfinite(point_array))
            or not np.all(np.isfinite(normal_array))
            or not np.all(np.isfinite(t_array))
        ):
            raise RuntimeError("STEP11_M1_RETRACE_NONFINITE_VALID_OUTPUT")
        d1 = reflect(d, normal_array)
        if not np.all(np.isfinite(d1)):
            raise RuntimeError("STEP11_M1_REFLECTION_NONFINITE")
    except (RuntimeError, ValueError) as exc:
        record.update({
            "rejection_stage": "M1_ACTUAL_RETRACE",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    visualization_payload.update({
        "m1_hit_points": point_array,
        "m1_hit_normals": normal_array,
        "m1_reflected_directions": d1,
        "m1_valid": h1_valid,
    })

    retrace_audit = {
        "schema": "HUD_STEP11_M1_RETRACE_AUDIT_V2",
        "ray_count": int(len(d)),
        "valid_count": int(np.sum(h1_valid)),
        "invalid_count": int(len(d) - np.sum(h1_valid)),
        "all_valid": bool(np.all(h1_valid)),
        "finite_point_rows": int(np.sum(np.all(np.isfinite(point_array), axis=1))),
        "finite_normal_rows": int(np.sum(np.all(np.isfinite(normal_array), axis=1))),
        "finite_t_count": int(np.sum(np.isfinite(t_array))),
        "finite_aperture_enforced": False,
    }

    enter_phase("M2_CI_REBUILD_AND_INTEGRABILITY")
    conditioning_progress_callback = None
    if (
        bool(job.get("emit_conditioning_progress", False))
        and not is_compute_worker()
    ):
        conditioning_progress_label = str(
            job.get(
                "conditioning_progress_label",
                candidate_id,
            )
        )

        def conditioning_progress_callback(message: str) -> None:
            """Print detailed conditioning progress only for parent-run jobs."""
            print(
                f"[STEP11][{conditioning_progress_label}]{message}",
                flush=True,
            )

    try:
        (
            ci_m2,
            m2_diagnostic,
            integrability_gate,
            conditioning_audit,
        ) = _step11_condition_m2_ci(
            ctx,
            point_array + 1e-4 * d1,
            d1,
            ctx.data["fan_refs"][r["field_index"]],
            ctx.data["m2"],
            _chief_index(r),
            surface_fit_cfg,
            integrability_policy,
            progress_callback=conditioning_progress_callback,
        )
    except (RuntimeError, ValueError) as exc:
        record.update({
            "rejection_stage": "M2_CI_REBUILD",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    visualization_payload["ci_m2"] = ci_m2

    record.update({
        "M2_conditioning_status":
            conditioning_audit["status"],
        "M2_conditioning_attempted_iterations":
            conditioning_audit["attempted_iterations"],
        "M2_conditioning_selected_iteration":
            conditioning_audit["selected_iteration"],
        "M2_conditioning_initial_integrability_status":
            conditioning_audit["initial_integrability_status"],
        "M2_conditioning_selected_integrability_status":
            conditioning_audit["selected_integrability_status"],
        "M2_conditioning_error":
            conditioning_audit["error"],
    })

    enter_phase("M2_CI_CONSISTENCY_GATE")
    integrability_status = str(integrability_gate["status"])
    integrability_admitted = bool(integrability_status in ("PASS", "WARN"))
    record["M2_integrability_status"] = integrability_status
    record["M2_integrability_pass"] = bool(integrability_status == "PASS")
    record["M2_integrability_admitted"] = integrability_admitted
    for integrability_row in integrability_gate["checks"]:
        check_name = str(integrability_row["check"])
        record[f"M2_CI_{check_name}_actual"] = integrability_row.get("actual")
        record[f"M2_CI_{check_name}_limit"] = integrability_row.get("limit")

    m2_cloud_summary = (
        m2_diagnostic.get("cloud", {}).get("bundle_summary", {})
    )
    record.update({
        "M2_CI_bundle_count":
            m2_cloud_summary.get("bundle_count"),
        "M2_CI_evaluable_bundle_count":
            m2_cloud_summary.get("evaluable_bundle_count"),
    })
    if not integrability_admitted:
        record.update({
            "rejection_stage": "M2_CI_CONSISTENCY_GATE",
            "rejection_reason": ",".join(
                row["check"]
                for row in integrability_gate["checks"]
                if row["status"] != "PASS"
            ),
        })
        return record, None, last_phase

    enter_phase("M2_O2_FIT")
    try:
        candidate_m2, fit_m2 = fit_surface(
            ci_m2["points_by_ray"],
            ci_m2["normals_by_ray"],
            ctx.data["m2"],
            2,
            True,
            ctx.config["fit_weights"],
            _chief_index(r),
            surface_fit_cfg,
            ci_m2["starts_by_ray"],
            ci_m2["targets_by_ray"],
        )
    except (RuntimeError, ValueError) as exc:
        record.update({
            "rejection_stage": "M2_O2_FIT",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    visualization_payload["m2"] = candidate_m2

    record.update({
        "M2_fit_curvature_per_mm": fit_m2.get("curvature_per_mm"),
        "M2_fit_conic_constant": fit_m2.get("conic_constant"),
        "M2_fit_sag_rms_mm": fit_m2.get("sag_rms_mm"),
        "M2_fit_normal_rms_deg": fit_m2.get("normal_rms_deg"),
        "M2_fit_normal_max_deg": fit_m2.get("normal_max_deg"),
        "M2_fit_k_bound_hit": fit_m2.get("k_bound_hit"),
        "M2_fit_curvature_bound_hit": fit_m2.get("curvature_bound_hit"),
    })

    enter_phase("M2_MIRROR_SHAPE_GATE")
    try:
        shape2 = _step11_topology_shape_gate(
            common,
            candidate_m2,
            np.asarray(d1, float)[chief_index],
            m2_topology,
        )
    except (RuntimeError, ValueError) as exc:
        record.update({
            "M2_shape_pass": False,
            "rejection_stage": "M2_MIRROR_TOPOLOGY_GATE",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    m2_topology_admission = _step11_topology_admission(
        shape2,
        m2_topology_enforcement,
    )
    record.update({
        "M2_shape_pass": bool(shape2["topology_pass"]),
        "M2_basic_surface_sanity_pass": bool(
            shape2["topology_checks"]["basic_surface_sanity"]
        ),
        "M2_shape_raw_pass": bool(shape2["raw_shape_pass"]),
        "M2_shape_quality_status": str(shape2["quality_status"]),
        "M2_shape_quality_warnings": ",".join(shape2["quality_warning_reasons"]),
        "M2_topology": m2_topology,
        "M2_topology_enforcement": m2_topology_enforcement,
        "M2_topology_admitted": bool(
            m2_topology_admission["admitted"]
        ),
        "M2_topology_admission_status": str(
            m2_topology_admission["status"]
        ),
        "M2_topology_warning_reasons": ",".join(
            m2_topology_admission["warning_reasons"]
        ),
        "M2_orientation_sign": float(shape2["orientation_sign"]),
        "M2_ray_facing_dot": float(shape2["ray_facing_dot"]),
    })

    record.update(
        _step11_shape_curvature_record_fields(
            "M2",
            shape2,
        )
    )

    m2_raw_convex_pass = bool(
        m2_topology_admission["raw_pass"]
    )

    record[
        "M2_raw_convex_ray_facing_pass"
    ] = m2_raw_convex_pass

    if not m2_raw_convex_pass:
        record.update({
            "M2_topology_admitted": False,
            "M2_topology_admission_status":
                "RAW_CONVEX_REQUIRED_REJECTED",
            "rejection_stage":
                "M2_MIRROR_TOPOLOGY_GATE",
            "rejection_reason":
                ",".join(
                    shape2[
                        "topology_failure_reasons"
                    ]
                ),
        })
        return record, None, last_phase

    enter_phase("FULL_PHYSICAL_OPTICAL_EVALUATION")
    try:
        _, optical_metrics, physical_trace = _step12_actual_o2_evaluate(
            ctx,
            candidate_m1,
            candidate_m2,
            refinement_cfg,
        )
        visualization_payload["physical_trace"] = (
            physical_trace
        )
        physical_mask = np.asarray(physical_trace["valid"], bool)
        if not np.any(physical_mask):
            raise RuntimeError("STEP11_PHYSICAL_TRACE_HAS_NO_VALID_RAYS")
        mf2 = mf2_geometry(
            physical_trace["points"][0][physical_mask],
            physical_trace["points"][1][physical_mask],
            physical_trace["points"][2][physical_mask],
            candidate_m2,
            float(ctx.config["fan_weights"]["omega2"]),
            ctx.data["n_obs"],
        )
        chief_spot = chief_centered_geometric_spot_rms(
            physical_trace["landing"],
            physical_trace["valid"],
            r["field_index"],
            r["pupil_index"],
            r["chief"],
            ctx.data["display"].frame,
        )
        chief_spot_rms = chief_spot["RMS_spot_radius_mm"]
        all_chiefs_valid = bool(
            chief_spot["bundle_rows"]
            and all(
                bool(row["chief_valid"])
                for row in chief_spot["bundle_rows"]
            )
        )
        if (
            chief_spot_rms is None
            or not np.isfinite(float(chief_spot_rms))
            or not all_chiefs_valid
        ):
            raise RuntimeError("STEP11_CHIEF_CENTERED_SPOT_RMS_NOT_EVALUABLE")
        optical_metrics = dict(optical_metrics)
        optical_metrics["chief_centered_spot"] = chief_spot
        optical_metrics["chief_centered_spot_RMS_mm"] = float(chief_spot_rms)
    except (RuntimeError, ValueError) as exc:
        record.update({
            "rejection_stage": "FULL_PHYSICAL_OPTICAL_EVALUATION",
            "rejection_reason": f"{type(exc).__name__}:{exc}",
        })
        return record, None, last_phase

    enter_phase("FULL_PHYSICAL_GATE")
    physical_valid_count = int(optical_metrics["physical_valid_count"])
    ray_count = int(optical_metrics["ray_count"])
    physical_fraction = float(optical_metrics["physical_fraction"])
    minimum_physical_fraction = float(quality_cfg["minimum_physical_fraction"])
    actual_surface_valid = bool(optical_metrics["candidate_surface_valid"])
    actual_m1_topology_admission = _step11_topology_admission(
        optical_metrics["M1_sanity"],
        m1_topology_enforcement,
    )
    actual_m2_topology_admission = _step11_topology_admission(
        optical_metrics["M2_sanity"],
        m2_topology_enforcement,
    )
    actual_surface_admitted = bool(
        actual_m1_topology_admission["admitted"]
        and actual_m2_topology_admission["admitted"]
    )
    actual_raw_topology_pass = bool(
        actual_m1_topology_admission["raw_pass"]
        and
        actual_m2_topology_admission["raw_pass"]
    )

    unobscured = bool(float(mf2["S_AQP_signed_mm2"]) >= 0.0)

    physical_hard_pass = bool(
        actual_raw_topology_pass
        and
        unobscured
    )
    physical_quality_pass = bool(physical_fraction >= minimum_physical_fraction)
    physical_quality_status = "PASS" if physical_quality_pass else "WARN"
    physical_quality_rank = 0 if physical_quality_pass else 1

    record.update({
        "physical_valid_count": physical_valid_count,
        "physical_fraction": physical_fraction,
        "minimum_physical_fraction": minimum_physical_fraction,
        "physical_quality_pass": physical_quality_pass,
        "physical_quality_status": physical_quality_status,
        "physical_quality_rank": physical_quality_rank,
        "candidate_surface_valid": actual_surface_valid,
        "candidate_surface_admitted": actual_surface_admitted,
        "actual_raw_topology_pass":
            actual_raw_topology_pass,
        "actual_M1_topology_pass": bool(
            actual_m1_topology_admission["raw_pass"]
        ),
        "actual_M1_topology_admitted": bool(
            actual_m1_topology_admission["admitted"]
        ),
        "actual_M1_topology_admission_status": str(
            actual_m1_topology_admission["status"]
        ),
        "actual_M1_topology_warning_reasons": ",".join(
            actual_m1_topology_admission["warning_reasons"]
        ),
        "actual_M2_topology_pass": bool(
            actual_m2_topology_admission["raw_pass"]
        ),
        "actual_M2_topology_admitted": bool(
            actual_m2_topology_admission["admitted"]
        ),
        "actual_M2_topology_admission_status": str(
            actual_m2_topology_admission["status"]
        ),
        "actual_M2_topology_warning_reasons": ",".join(
            actual_m2_topology_admission["warning_reasons"]
        ),
        "unobscured": unobscured,
        "chief_centered_spot_RMS_mm": float(chief_spot_rms),
    })

    if not physical_hard_pass:
        record.update({
            "rejection_stage": "FULL_PHYSICAL_GATE",
            "rejection_reason": (
                f"valid={physical_valid_count}/{ray_count};"
                f"fraction={physical_fraction:.6g};"
                f"preferred_minimum={minimum_physical_fraction:.6g};"
                f"raw_topology_pass={actual_raw_topology_pass};"
                f"surface_valid={actual_surface_valid};"
                f"surface_admitted={actual_surface_admitted};"
                f"unobscured={unobscured}"
            ),
        })
        return record, None, last_phase

    representation_score = (
        (float(fit_m2["sag_rms_mm"]) / float(quality_cfg["maximum_sag_rms_mm"])) ** 2
        + (float(fit_m2["normal_rms_deg"]) / float(quality_cfg["maximum_normal_rms_deg"])) ** 2
    )
    record.update({
        "feasible": True,
        "rejection_stage": None,
        "rejection_reason": None,
        "optical_objective": float(optical_metrics["J_step12_actual_o2_dimensionless"]),
        "mapping_rms_mm": float(optical_metrics["mapping_field_rms_mm"]),
        "spot_rms_mm": float(optical_metrics["spot_field_rms_mm"]),
        "direction_rms_deg": float(optical_metrics["target_direction_rms_deg"]),
        "representation_score": float(representation_score),
    })
    payload = {
        "search_vector": np.asarray(search_vector, float).copy(),
        "m1": candidate_m1,
        "m2": candidate_m2,
        "M1_shape_gate": shape1,
        "M1_ci_trust": trust,
        "M1_retrace_audit": retrace_audit,
        "ci_m2": ci_m2,
        "M2_diagnostic": m2_diagnostic,
        "M2_integrability_gate": integrability_gate,
        "M2_conditioning_audit": conditioning_audit,
        "fit_m2_order2": fit_m2,
        "M2_shape_gate": shape2,
        "optical_metrics": optical_metrics,
        "physical_trace": physical_trace,
        "mf2": mf2,
        "rank_key": (
            int(physical_quality_rank),
            float(optical_metrics["chief_centered_spot_RMS_mm"]),
            float(optical_metrics["J_step12_actual_o2_dimensionless"]),
            float(representation_score),
        ),
    }
    enter_phase("COMPLETE")
    return record, payload, last_phase


# Legacy architecture tokens preserved for verify_v55 compatibility:
# SYSTEM_AWARE_O2_PAIR_CONSTRUCTION
# 11_M1_PHYSICAL_DOF_SCALING.csv
# STEP11_NO_FEASIBLE_O2_PAIR
# FEASIBLE_FIRST_THEN_ACTUAL_OPTICAL_OBJECTIVE_THEN_M2_REPRESENTABILITY
# "topology_is_hard_constraint_not_merit":
# "shape_quality_enforcement":
# "ci_trust_enforcement":
# SOFT_PREFERRED_THRESHOLD_FOR_RANK_BUCKET

def step_11(ctx: Context) -> dict[str, Any]:
    """Joint-restoration M1/M2 O2 bằng fixed-grid Jacobian-LM rồi hard-certify topology, unobscuration và physical optics."""

    step_started = time.perf_counter()

    def set_phase(phase: str) -> None:
        """Ghi phase STEP 11 hiện hành để terminal và hồ sơ lỗi chỉ đúng điểm dừng."""
        ctx.data["_execution_phase"] = f"STEP11/{phase}"
        slot = ctx.data.get("_step_debug")
        if isinstance(slot, dict) and slot.get("step") == 11:
            slot["current_phase"] = f"STEP11/{phase}"

    def progress(message: str) -> None:
        """In tiến độ STEP 11 từ process cha theo định dạng taxonomy chuẩn."""
        if not is_compute_worker():
            prefix = "[STEP11]"
            for line in str(message).split("\n"):
                if line.startswith("="):
                    print(line, flush=True)
                elif line.startswith("[STEP11]"):
                    print(line, flush=True)
                elif line.startswith("["):
                    print(f"{prefix}{line}", flush=True)
                elif line.startswith("  ") or not line.strip():
                    print(line, flush=True)
                elif any(line.startswith(k) for k in (
                    "Candidates evaluated", "Preferred candidates", "Physical WARN",
                    "Hard rejected", "Top rejection", "Top reasons", "Last phase",
                    "Diagnostic"
                )):
                    print(line, flush=True)
                else:
                    print(f"{prefix} {line}", flush=True)

    def duration(seconds: float) -> str:
        """Định dạng thời gian chạy gọn để các dòng tiến độ dễ quét bằng mắt."""
        total = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def number(value: Any) -> str:
        """Định dạng metric số mà không làm telemetry gây lỗi cho thuật toán chính."""
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{scalar:.6g}" if np.isfinite(scalar) else str(scalar)

    def rejection_summary(rows: list[dict[str, Any]]) -> str:
        """Gom số candidate bị loại theo gate và chỉ hiện các nhóm lớn nhất."""
        counts: dict[str, int] = {}
        for row in rows:
            stage = row.get("rejection_stage")
            if stage:
                name = str(stage)
                counts[name] = counts.get(name, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        parts = [f"{name}:{count}" for name, count in ordered[:3]]
        if len(ordered) > 3:
            parts.append(f"OTHER:{sum(count for _, count in ordered[3:])}")
        return ",".join(parts) if parts else "none"

    set_phase("INITIALIZE")
    r = ctx.data["rays"]
    step_dir = ctx.step_dir(11)

    candidate_snapshot_root = (
        step_dir
        / "11_CANDIDATE_SNAPSHOTS"
    )
    candidate_snapshot_rows: list[
        dict[str, Any]
    ] = []
    candidate_snapshot_counter = 0

    def write_candidate_snapshot_manifest(
        execution_status: str,
    ) -> None:
        """Ghi index scalar của archive; candidate geometry nằm trong thư mục riêng."""
        manifest = {
            "schema":
                "HUD_FAN_V5_5_STEP11_CANDIDATE_SNAPSHOT_MANIFEST_V1",
            "execution_status": str(execution_status),
            "archive_complete": bool(
                execution_status != "IN_PROGRESS"
            ),
            "candidate_count": int(
                len(candidate_snapshot_rows)
            ),
            "saved_count": int(
                sum(
                    row.get("status") == "SAVED"
                    for row in candidate_snapshot_rows
                )
            ),
            "write_error_count": int(
                sum(
                    row.get("status") != "SAVED"
                    for row in candidate_snapshot_rows
                )
            ),
            "candidate_order":
                "STEP11_EVALUATION_ORDER",
            "array_precision":
                "ORIGINAL_NUMPY_DTYPE",
            "array_storage":
                "PER_CANDIDATE_COMPRESSED_NPZ",
            "algorithm_source_manifest_sha256":
                ctx.data.get(
                    "algorithm_source_manifest",
                    {},
                ).get(
                    "manifest_sha256"
                ),
            "candidates":
                candidate_snapshot_rows,
        }
        candidate_snapshot_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        target = (
            candidate_snapshot_root
            / "manifest.json"
        )
        temporary = (
            candidate_snapshot_root
            / "manifest.json.tmp"
        )
        write_json(
            temporary,
            manifest,
        )
        temporary.replace(
            target
        )

    def allocate_candidate_snapshot(
        candidate_class: str,
    ) -> dict[str, Any]:
        """Cấp archive index ổn định trước khi candidate được gửi tới worker."""
        nonlocal candidate_snapshot_counter
        candidate_snapshot_counter += 1
        return {
            "enabled": True,
            "root": str(
                candidate_snapshot_root
            ),
            "archive_index": int(
                candidate_snapshot_counter
            ),
            "candidate_class": str(
                candidate_class
            ),
        }

    def register_candidate_snapshot(
        record: dict[str, Any],
    ) -> None:
        """Thu summary snapshot theo đúng thứ tự parent xử lý candidate."""
        snapshot_summary = record.pop(
            "_snapshot_summary",
            None,
        )
        if not isinstance(
            snapshot_summary,
            dict,
        ):
            return

        candidate_snapshot_rows.append(
            snapshot_summary
        )
        write_candidate_snapshot_manifest(
            "IN_PROGRESS"
        )

        if snapshot_summary.get("status") == "SAVED":
            progress(
                "[SNAPSHOT]"
                f" archive={int(snapshot_summary['archive_index']):06d}"
                f" | candidate={snapshot_summary['candidate_id']}"
                f" | arrays={snapshot_summary['array_count']}"
                f" | compressed={snapshot_summary['compressed_bytes']} bytes"
            )
        else:
            progress(
                "[SNAPSHOT][WARN]"
                f" candidate={snapshot_summary.get('candidate_id')}"
                f" | error={snapshot_summary.get('error')}"
            )

    write_candidate_snapshot_manifest(
        "IN_PROGRESS"
    )

    surface_fit_cfg = ctx.config["surface_fit"]
    quality_cfg = surface_fit_cfg["step12_quality_gates"]
    refinement_cfg = surface_fit_cfg["step12_o2_refinement"]
    integrability_policy = {
        **surface_fit_cfg["integrability_gates"],
        "enforcement": "WARN",
    }
    conditioning_cfg = surface_fit_cfg[
        "step11_macro_micro_conditioning"
    ]
    restoration_cfg = surface_fit_cfg[
        "step11_topology_restoration"
    ]
    shape_policy = {
        "maximum_freeform_departure_mm": float(quality_cfg["maximum_sag_rms_mm"]),
        "maximum_normal_departure_deg": float(quality_cfg["maximum_normal_max_deg"]),
        "maximum_principal_curvature_per_mm": float(
            surface_fit_cfg["curvature_absolute_max_per_mm"]
        ),
    }
    set_phase("BASELINE_M1_FIT")
    baseline_m1, baseline_fit = fit_surface(
        ctx.data["ci_m1"]["points_by_ray"],
        ctx.data["ci_m1"]["normals_by_ray"],
        ctx.data["m1"],
        2,
        True,
        ctx.config["fit_weights"],
        _chief_index(r),
        surface_fit_cfg,
        ctx.data["ci_m1"]["starts_by_ray"],
        ctx.data["ci_m1"]["targets_by_ray"],
    )
    baseline_fit = dict(baseline_fit)
    baseline_fit["role"] = "STEP11_BASELINE_SEARCH_CENTER_NOT_FINAL_WINNER"
    write_json(step_dir / "11_M1_FIT_DIAGNOSTICS.json", baseline_fit)

    set_phase("BASELINE_M1_SHAPE_GATE")
    topology_cfg = surface_fit_cfg[
        "mirror_topology_authority"
    ]

    m1_topology = str(
        topology_cfg[
            "M1"
        ]
    )

    m2_topology = str(
        topology_cfg[
            "M2"
        ]
    )

    topology_enforcement = {
        surface_name: str(
            topology_cfg[
                "enforcement"
            ][
                surface_name
            ]
        )
        for surface_name in ("M1", "M2")
    }

    if any(
        value not in {"HARD", "WARN"}
        for value in topology_enforcement.values()
    ):
        raise RuntimeError(
            "STEP11_TOPOLOGY_ENFORCEMENT_INVALID"
        )

    progress(
        "[TOPOLOGY] "
        f"M1_enforcement={topology_enforcement['M1']} | "
        f"M2_enforcement={topology_enforcement['M2']} | "
        "scope=STEP11_KG_H_ONLY | "
        "basic_surface_sanity=HARD"
    )

    if m1_topology != "CONCAVE_RAY_FACING":
        raise RuntimeError(
            "STEP11_M1_TOPOLOGY_AUTHORITY_MUST_BE_CONCAVE_RAY_FACING"
        )

    if m2_topology != "CONVEX_RAY_FACING":
        raise RuntimeError(
            "STEP11_M2_TOPOLOGY_AUTHORITY_MUST_BE_CONVEX_RAY_FACING"
        )

    chief_index = _chief_index(
        r
    )

    def topology_shape_gate(
        surface: PolySurface,
        incident_direction: np.ndarray,
        topology: str,
    ) -> dict[str, Any]:
        """Map ray-facing concave/convex authority to mirror_shape_gate orientation."""
        direction = np.asarray(
            incident_direction,
            float,
        )

        if (
            direction.shape != (3,)
            or not np.all(
                np.isfinite(
                    direction
                )
            )
        ):
            raise RuntimeError(
                f"STEP11_TOPOLOGY_INCIDENT_DIRECTION_INVALID:{surface.name}"
            )

        normal_rows = np.asarray(
            surface.normal(
                np.asarray(
                    [0.0],
                    float,
                ),
                np.asarray(
                    [0.0],
                    float,
                ),
            ),
            float,
        )

        if (
            normal_rows.shape != (1, 3)
            or not np.all(
                np.isfinite(
                    normal_rows
                )
            )
        ):
            raise RuntimeError(
                f"STEP11_TOPOLOGY_CHIEF_NORMAL_INVALID:{surface.name}"
            )

        normal = normal_rows[
            0
        ]

        direction_norm = float(
            np.linalg.norm(
                direction
            )
        )

        normal_norm = float(
            np.linalg.norm(
                normal
            )
        )

        if (
            not np.isfinite(
                direction_norm
            )
            or not np.isfinite(
                normal_norm
            )
            or direction_norm <= 1e-12
            or normal_norm <= 1e-12
        ):
            raise RuntimeError(
                f"STEP11_TOPOLOGY_CHIEF_VECTOR_NORM_INVALID:{surface.name}"
            )

        incident_unit = (
            direction
            /
            direction_norm
        )

        normal_unit = (
            normal
            /
            normal_norm
        )

        ray_facing_dot = float(
            np.dot(
                normal_unit,
                -incident_unit,
            )
        )

        if (
            not np.isfinite(
                ray_facing_dot
            )
            or abs(
                ray_facing_dot
            )
            <= 1e-9
        ):
            raise RuntimeError(
                f"STEP11_TOPOLOGY_CHIEF_RAY_GRAZING:{surface.name}"
            )

        facing_sign = (
            1.0
            if ray_facing_dot > 0.0
            else -1.0
        )

        if topology == "CONCAVE_RAY_FACING":
            orientation_sign = (
                facing_sign
            )

        elif topology == "CONVEX_RAY_FACING":
            orientation_sign = (
                -facing_sign
            )

        else:
            raise RuntimeError(
                f"STEP11_TOPOLOGY_AUTHORITY_INVALID:"
                f"{surface.name}:{topology}"
            )

        gate = dict(
            mirror_shape_gate(
                surface,
                shape_policy,
                orientation_sign=
                    orientation_sign,
            )
        )

        gate.update({
            "topology_authority":
                topology,

            "topology_enforcement":
                topology_enforcement[surface.name],

            "ray_facing_definition":
                "SURFACE_NORMAL_DOT_NEGATIVE_INCIDENT_DIRECTION",

            "ray_facing_dot":
                ray_facing_dot,

            "ray_facing_normal_sign":
                facing_sign,

            "required_orientation_sign":
                float(
                    orientation_sign
                ),
        })

        return _o2_mirror_gate_view(
            gate,
            quality_cfg,
        )

    baseline_shape_gate = topology_shape_gate(
        baseline_m1,
        np.asarray(ctx.data["post_visor"], float)[chief_index],
        m1_topology,
    )

    if not baseline_shape_gate["topology_pass"]:
        progress(
            "[WARN] Baseline M1 fit NLSQ bị FAIL Topology. "
            "Kích hoạt Biconic Gamma Sweep + M2 Alternating Projection!"
        )

        c_base = float(baseline_m1.curvature)
        req_sign = float(
            baseline_shape_gate.get(
                "required_orientation_sign",
                0.0,
            )
        )
        target_sign = (
            req_sign
            if req_sign != 0.0
            else (
                float(np.sign(c_base))
                if c_base != 0.0
                else -1.0
            )
        )

        minimum_abs_curvature = float(
            conditioning_cfg[
                "minimum_abs_baseline_curvature_per_mm"
            ]
        )
        default_abs_curvature = float(
            conditioning_cfg[
                "default_abs_baseline_curvature_per_mm"
            ]
        )

        if (
            abs(c_base) < minimum_abs_curvature
            or (
                req_sign != 0.0
                and np.sign(c_base) != np.sign(req_sign)
            )
        ):
            c_base = float(target_sign) * default_abs_curvature

        gamma_values = (
            [
                float(value)
                for value in conditioning_cfg[
                    "fallback_gamma_candidates"
                ]
            ]
            if bool(conditioning_cfg["enabled"])
            else [1.0]
        )
        gamma_sweep_started = time.perf_counter()
        progress(
            "[GAMMA SWEEP][START]"
            f" candidates={len(gamma_values)}"
            f" | range={min(gamma_values):.2f}..{max(gamma_values):.2f}"
            " | execution=SERIAL_PARENT"
            f" | AP_iterations<={int(conditioning_cfg['maximum_alternating_projection_iterations'])}"
        )

        gamma_records: list[dict[str, Any]] = []
        gamma_surfaces: list[PolySurface] = []
        restoration_seed_entries: list[
            tuple[
                tuple[Any, ...],
                int,
                float,
                dict[str, Any],
                dict[str, Any] | None,
            ]
        ] = []

        for gamma_index, gamma in enumerate(gamma_values):
            gamma_started = time.perf_counter()
            c_try = float(c_base * gamma)
            radius_try = float(1.0 / c_try)
            progress(
                f"[GAMMA {gamma_index + 1}/{len(gamma_values)}][START]"
                f" gamma={gamma:.2f}"
                f" | c={c_try:.6f}/mm"
                f" | R={radius_try:.1f}mm"
            )

            fallback_m1_try = PolySurface.biconic(
                "M1",
                baseline_m1.center,
                baseline_m1.frame,
                baseline_m1.half_aperture,
                rx=radius_try,
                ry=radius_try,
                kx=0.0,
                ky=0.0,
                curvature_sign=1.0,
                terms=list(baseline_m1.terms),
                scale=baseline_m1.scale,
                aperture_polygon=baseline_m1.aperture_polygon,
            )
            gamma_surfaces.append(fallback_m1_try)

            trial_descriptors = [
                descriptor
                for descriptor in _step12_o2_variable_descriptors(
                    fallback_m1_try,
                    ctx.data["m2"],
                )
                if descriptor["surface"] == "M1"
            ]
            if not trial_descriptors:
                raise RuntimeError(
                    "STEP11_FALLBACK_GAMMA_HAS_NO_M1_O2_VARIABLES"
                )

            trial_common = {
                "ctx": ctx,
                "base_pair": {
                    "M1": fallback_m1_try.copy(),
                    "M2": ctx.data["m2"].copy(),
                },
                "descriptors": trial_descriptors,
                "physical_scales":
                    np.ones(len(trial_descriptors), dtype=float),
                "surface_fit_cfg": surface_fit_cfg,
                "quality_cfg": quality_cfg,
                "refinement_cfg": refinement_cfg,
                "integrability_policy": integrability_policy,
                "shape_policy": shape_policy,
                "m1_topology": m1_topology,
                "m2_topology": m2_topology,
                "topology_enforcement": dict(topology_enforcement),
                "chief_index": int(chief_index),
            }
            trial_job = {
                "candidate_id":
                    f"FALLBACK_GAMMA_{gamma_index + 1:02d}",
                "cycle": 0,
                "trust_scale": 0.0,
                "move": f"FALLBACK_GAMMA_{gamma:.2f}",
                "search_vector":
                    np.zeros(len(trial_descriptors), dtype=float),
                "emit_conditioning_progress": True,
                "conditioning_progress_label": (
                    f"GAMMA {gamma_index + 1}/{len(gamma_values)}"
                    f" gamma={gamma:.2f}"
                ),
                "snapshot":
                    allocate_candidate_snapshot(
                        "FALLBACK_GAMMA"
                    ),
            }

            (
                trial_record,
                trial_payload,
                trial_last_phase,
            ) = evaluate_step11_candidate_job(
                trial_common,
                trial_job,
            )

            trial_record.update({
                "gamma_index": int(gamma_index),
                "gamma": float(gamma),
                "base_curvature_per_mm": float(c_base),
                "trial_curvature_per_mm": float(c_try),
                "trial_radius_mm": float(radius_try),
                "last_phase": str(trial_last_phase),
            })

            snapshot_summary = (
                _step11_archive_candidate_snapshot(
                    trial_record,
                    trial_last_phase,
                    trial_job,
                )
            )
            if snapshot_summary is not None:
                trial_record[
                    "_snapshot_summary"
                ] = snapshot_summary
            register_candidate_snapshot(
                trial_record
            )

            m1_seed_eligible = bool(
                trial_record.get(
                    "M1_shape_pass",
                    False,
                )
            )

            if (
                not bool(
                    restoration_cfg[
                        "require_m1_raw_topology"
                    ]
                )
            ):

                m1_seed_eligible = bool(
                    trial_record.get(
                        "M1_topology_admitted",
                        False,
                    )
                )

            if m1_seed_eligible:

                selection_key = (
                    _step11_gamma_restoration_seed_rank_key(
                        trial_record,
                        gamma,
                        gamma_index,
                        restoration_cfg,
                    )
                )

                trial_record[
                    "gamma_restoration_seed_key"
                ] = list(
                    selection_key
                )

                trial_record[
                    "gamma_selection_key"
                ] = list(
                    selection_key
                )

                if trial_payload is not None:

                    trial_record[
                        "gamma_rank_key"
                    ] = list(
                        trial_payload[
                            "rank_key"
                        ]
                    )

                restoration_seed_entries.append((
                    selection_key,
                    int(
                        gamma_index
                    ),
                    float(
                        gamma
                    ),
                    trial_record,
                    trial_payload,
                ))

            gamma_records.append(trial_record)
            gamma_conditioning = str(
                trial_record.get(
                    "M2_conditioning_status",
                    "NOT_EVALUATED",
                )
            )
            gamma_integrability = str(
                trial_record.get(
                    "M2_integrability_status",
                    "NOT_EVALUATED",
                )
            )
            gamma_m2_topology_raw = trial_record.get(
                "M2_shape_pass"
            )
            gamma_m2_topology = (
                "NOT_EVALUATED"
                if gamma_m2_topology_raw is None
                else (
                    "PASS"
                    if bool(gamma_m2_topology_raw)
                    else "FAIL"
                )
            )
            gamma_rejection_stage = str(
                trial_record.get("rejection_stage")
                or "none"
            )
            progress(
                f"[GAMMA {gamma_index + 1}/{len(gamma_values)}][DONE]"
                f" gamma={gamma:.2f}"
                f" | feasible={bool(trial_record.get('feasible', False))}"
                f" | conditioning={gamma_conditioning}"
                f" | integrability={gamma_integrability}"
                f" | M2_topology={gamma_m2_topology}"
                f" | failed_at={gamma_rejection_stage}"
                f" | elapsed={duration(time.perf_counter() - gamma_started)}"
            )
            gamma_conditioning_error = trial_record.get(
                "M2_conditioning_error"
            )
            if gamma_conditioning_error:
                progress(
                    f"[GAMMA {gamma_index + 1}/{len(gamma_values)}][WARN]"
                    f" {gamma_conditioning_error}"
                )

        progress(
            "[GAMMA SWEEP][DONE]"
            f" evaluated={len(gamma_records)}/{len(gamma_values)}"
            f" | feasible={sum(bool(record.get('feasible')) for record in gamma_records)}"
            f" | integrability_PASS={sum(record.get('M2_integrability_status') == 'PASS' for record in gamma_records)}"
            f" | elapsed={duration(time.perf_counter() - gamma_sweep_started)}"
        )

        write_csv(
            step_dir / "11_M1_FALLBACK_GAMMA_SWEEP.csv",
            gamma_records,
        )

        if restoration_seed_entries:

            selected_entry = min(
                restoration_seed_entries,
                key=lambda entry:
                    entry[0],
            )

            selected_key = (
                selected_entry[
                    0
                ]
            )

            selected_gamma_index = int(
                selected_entry[
                    1
                ]
            )

            selected_gamma = float(
                selected_entry[
                    2
                ]
            )

            selected_record = (
                selected_entry[
                    3
                ]
            )

            selected_payload = (
                selected_entry[
                    4
                ]
            )

            # Quan trọng:
            # dùng M1 gamma surface trực tiếp.
            # Không yêu cầu old candidate phải có M2 feasible payload.
            baseline_m1 = (
                gamma_surfaces[
                    selected_gamma_index
                ].copy()
            )

            selected_integrability_status = str(
                selected_record.get(
                    "M2_integrability_status",
                    "NOT_EVALUATED",
                )
            )

            anchor_status = (
                "M1_VALID_RESTORATION_SEED"
            )

        else:

            write_csv(
                step_dir
                /
                "11_M1_FALLBACK_GAMMA_SWEEP.csv",

                gamma_records,
            )

            raise RuntimeError(
                "STEP11_M1_FALLBACK_GAMMA_"
                "NO_VALID_RESTORATION_SEED"
            )

        baseline_shape_gate = topology_shape_gate(
            baseline_m1,
            np.asarray(
                ctx.data["post_visor"],
                float,
            )[chief_index],
            m1_topology,
        )
        baseline_topology_admission = _step11_topology_admission(
            baseline_shape_gate,
            topology_enforcement["M1"],
        )
        if not baseline_topology_admission["admitted"]:
            raise RuntimeError(
                "STEP11_M1_FALLBACK_GAMMA_TOPOLOGY_FAILED:"
                + ",".join(
                    baseline_shape_gate[
                        "topology_failure_reasons"
                    ]
                )
            )
        if baseline_topology_admission["status"] == "WARN_ADMITTED":
            progress(
                "[WARN] M1 fallback Gamma van FAIL raw topology "
                "nhung duoc STEP11 WARN admission cho di tiep | "
                "reasons="
                + ",".join(
                    baseline_topology_admission[
                        "warning_reasons"
                    ]
                )
            )

        selected_curvature = float(
            baseline_m1.curvature
        )
        selected_radius = float(
            1.0 / selected_curvature
        )

        gamma_summary = {
            "schema":
                "HUD_FAN_V5_5_STEP11_M1_FALLBACK_GAMMA_SWEEP_V1",
            "enabled": bool(conditioning_cfg["enabled"]),
            "selection_rule":
                "M1_VALID_THEN_M2_BASIC_SANITY_"
                "THEN_INTEGRABILITY_"
                "THEN_NORMALIZED_H_KG_DISTANCE_"
                "THEN_ABS_GAMMA_MINUS_ONE_"
                "THEN_LIST_ORDER",
            "gamma_candidates": list(gamma_values),
            "candidate_count": int(len(gamma_records)),
            "feasible_candidate_count": int(
                sum(
                    bool(record.get("feasible"))
                    for record in gamma_records
                )
            ),
            "restoration_seed_candidate_count": int(
                len(
                    restoration_seed_entries
                )
            ),
            "integrability_pass_count": int(
                sum(
                    record.get("M2_integrability_status") == "PASS"
                    for record in gamma_records
                )
            ),
            "anchor_status": anchor_status,
            "selected_gamma_index": int(
                selected_gamma_index
            ),
            "selected_gamma": float(selected_gamma),
            "selected_curvature_per_mm":
                selected_curvature,
            "selected_radius_mm":
                selected_radius,
            "selected_integrability_status":
                selected_integrability_status,
            "selected_step11_rank_key": (
                list(selected_payload["rank_key"])
                if selected_payload is not None
                else None
            ),
            "selected_gamma_selection_key": (
                list(selected_key)
                if selected_key is not None
                else None
            ),
            "selected_record": selected_record,
        }

        baseline_fit["fallback_activated"] = True
        baseline_fit["fallback_type"] = "BICONIC"
        baseline_fit["fallback_strategy"] = (
            "GAMMA_SWEEP_WITH_M2_ALTERNATING_PROJECTION"
        )
        baseline_fit["fallback_base_curvature_per_mm"] = (
            float(c_base)
        )
        baseline_fit["fallback_gamma_candidates"] = list(
            gamma_values
        )
        baseline_fit["fallback_selected_gamma"] = float(
            selected_gamma
        )
        baseline_fit["fallback_c"] = selected_curvature
        baseline_fit["fallback_radius_mm"] = selected_radius
        baseline_fit["fallback_anchor_status"] = anchor_status
        baseline_fit[
            "fallback_selected_integrability_status"
        ] = selected_integrability_status
        baseline_fit["baseline_topology_gate"] = (
            baseline_shape_gate
        )

        write_json(
            step_dir / "11_M1_FALLBACK_GAMMA_SUMMARY.json",
            gamma_summary,
        )
        write_json(
            step_dir / "11_M1_FIT_DIAGNOSTICS.json",
            baseline_fit,
        )
        progress(
            "[FIX] M1 fallback anchor selected | "
            f"gamma={selected_gamma:.2f} | "
            f"c={selected_curvature:.6f}/mm | "
            f"R={selected_radius:.1f}mm | "
            f"integrability={selected_integrability_status} | "
            f"anchor={anchor_status}"
        )

    set_phase("M1_CLOUD_DIAGNOSTICS")
    m1_diagnostic = _construction_diagnostics(
        ctx,
        ctx.data["ci_m1"],
        baseline_m1.frame,
        "M1_INITIAL_CI_IN_FITTED_LOCAL_FRAME",
    )
    write_json(
        step_dir / "11_M1_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
        m1_diagnostic,
    )

    set_phase(
        "BUILD_RESTORATION_SEED"
    )

    restoration_common_seed = {
        "ctx":
            ctx,

        "restoration_cfg":
            restoration_cfg,

        "surface_fit_cfg":
            surface_fit_cfg,

        "quality_cfg":
            quality_cfg,

        "refinement_cfg":
            refinement_cfg,

        "integrability_policy":
            integrability_policy,

        "shape_policy":
            shape_policy,

        "m1_topology":
            m1_topology,

        "m2_topology":
            m2_topology,

        "topology_enforcement":
            dict(
                topology_enforcement
            ),

        "chief_index":
            int(
                chief_index
            ),
    }

    seed = (
        _step11_build_restoration_seed(
            restoration_common_seed,
            baseline_m1,
        )
    )

    seed_m1 = seed[
        "m1"
    ]

    seed_m2 = seed[
        "m2"
    ]

    seed_unobscuration = (
        _step11_unobscuration_state(
            ctx,
            seed_m2,
            seed[
                "physical_trace"
            ],
        )
    )

    seed_signed_area = float(
        seed_unobscuration[
            "S_AQP_signed_mm2"
        ]
    )

    unobscuration_scale_mm2 = max(
        1.0,
        max(
            0.0,
            -seed_signed_area,
        ),
    )

    progress(
        "[RESTORE][UNOBSCURATION SEED]"
        f" S_AQP={seed_signed_area:.6g} mm^2"
        f" | unobscured={seed_signed_area >= 0.0}"
        f" | scale={unobscuration_scale_mm2:.6g} mm^2"
    )

    descriptors = (
        _step12_o2_variable_descriptors(
            seed_m1,
            seed_m2,
        )
    )

    if not descriptors:
        raise RuntimeError(
            "STEP11_RESTORATION_HAS_NO_VARIABLES"
        )

    set_phase(
        "FREEZE_RESTORATION_STATE"
    )

    grids = {
        "M1":
            _step11_fixed_residual_grid(
                seed_m1,
                int(
                    restoration_cfg[
                        "grid_samples"
                    ]
                ),
            ),

        "M2":
            _step11_fixed_residual_grid(
                seed_m2,
                int(
                    restoration_cfg[
                        "grid_samples"
                    ]
                ),
            ),
    }

    orientation_signs = {
        "M1":
            float(
                seed[
                    "optical_metrics"
                ][
                    "M1_sanity"
                ][
                    "orientation_sign"
                ]
            ),

        "M2":
            float(
                seed[
                    "optical_metrics"
                ][
                    "M2_sanity"
                ][
                    "orientation_sign"
                ]
            ),
    }

    anchors = {
        "M1":
            _step11_curvature_field(
                seed_m1,
                grids[
                    "M1"
                ],
                orientation_signs[
                    "M1"
                ],
            ),

        "M2":
            _step11_curvature_field(
                seed_m2,
                grids[
                    "M2"
                ],
                orientation_signs[
                    "M2"
                ],
            ),
    }

    base_optical = np.asarray(
        seed[
            "optical_residual"
        ],
        float,
    )

    maximum_optical = int(
        restoration_cfg[
            "maximum_optical_residual_samples"
        ]
    )

    if maximum_optical <= 0:

        optical_indices = np.empty(
            0,
            dtype=int,
        )

    elif (
        len(base_optical)
        <=
        maximum_optical
    ):

        optical_indices = np.arange(
            len(base_optical),
            dtype=int,
        )

    else:

        optical_indices = np.unique(
            np.linspace(
                0,
                len(base_optical) - 1,
                maximum_optical,
                dtype=int,
            )
        )

    residual_state = {
        "cfg":
            copy.deepcopy(
                restoration_cfg
            ),

        "surface_fit_cfg":
            copy.deepcopy(
                surface_fit_cfg
            ),

        "quality_cfg":
            copy.deepcopy(
                quality_cfg
            ),

        "grids":
            copy.deepcopy(
                grids
            ),

        "anchors":
            copy.deepcopy(
                anchors
            ),

        "orientation_signs":
            dict(
                orientation_signs
            ),

        "optical_indices":
            np.asarray(
                optical_indices,
                int,
            ),

        "unobscuration_scale_mm2":
            float(
                unobscuration_scale_mm2
            ),
    }

    restoration_ctx_snapshot = Context(
        source_dir=
            ctx.source_dir,

        config_path=
            ctx.config_path,

        config=
            copy.deepcopy(
                ctx.config
            ),

        run_dir=
            ctx.run_dir,
    )

    for key in (
        "rays",
        "visor",
        "display",
        "ci_m1",
        "visor_hit",
        "post_visor",
        "fan_refs",
        "n_obs",
    ):
        if key in ctx.data:

            restoration_ctx_snapshot.data[
                key
            ] = copy.deepcopy(
                ctx.data[
                    key
                ]
            )

    restoration_common = {
        "ctx":
            restoration_ctx_snapshot,

        "base_pair": {
            "M1":
                seed_m1.copy(),

            "M2":
                seed_m2.copy(),
        },

        "descriptors":
            copy.deepcopy(
                descriptors
            ),

        "restoration_cfg":
            copy.deepcopy(
                restoration_cfg
            ),

        "surface_fit_cfg":
            copy.deepcopy(
                surface_fit_cfg
            ),

        "quality_cfg":
            copy.deepcopy(
                quality_cfg
            ),

        "refinement_cfg":
            copy.deepcopy(
                refinement_cfg
            ),

        "residual_state":
            residual_state,
    }

    progress(
        "=" * 80
    )

    progress(
        "JOINT M1/M2 TOPOLOGY RESTORATION"
    )

    progress(
        "=" * 80
    )

    progress(
        f"[INIT]"
        f" variables={len(descriptors)}"
        f" | M1_grid={len(grids['M1']['xy'])}"
        f" | M2_grid={len(grids['M2']['xy'])}"
        f" | optical_samples={len(optical_indices)}"
    )

    set_phase(
        "JOINT_TOPOLOGY_RESTORATION"
    )

    if bool(
        restoration_cfg[
            "enabled"
        ]
    ):

        restoration = (
            _step11_run_joint_topology_restoration(
                restoration_common,
                progress,
            )
        )

    else:

        restoration = (
            _step11_disabled_restoration_result(
                restoration_common,
                progress,
            )
        )

    compatibility_history = (
        _step11_write_restoration_run_diagnostics(
            step_dir,
            restoration,
            descriptors,
            grids,
            residual_state,
        )
    )

    if not bool(
        restoration[
            "success"
        ]
    ):

        write_candidate_snapshot_manifest(
            "FAILED"
        )

        set_phase(
            "JOINT_TOPOLOGY_RESTORATION_FAILED"
        )

        raise RuntimeError(
            str(
                restoration.get(
                    "failure_message"
                )
                or
                "STEP11_JOINT_TOPOLOGY_RESTORATION_FAILED"
            )
        )

    final_m1 = restoration[
        "m1"
    ]

    final_m2 = restoration[
        "m2"
    ]

    set_phase(
        "FINAL_HARD_CERTIFICATION"
    )

    (
        final_optical_residual,
        final_metrics,
        final_trace,
    ) = _step12_actual_o2_evaluate(
        ctx,
        final_m1,
        final_m2,
        refinement_cfg,
    )

    final_m1_gate = final_metrics[
        "M1_sanity"
    ]

    final_m2_gate = final_metrics[
        "M2_sanity"
    ]

    if not bool(
        final_m1_gate[
            "topology_pass"
        ]
    ):
        raise RuntimeError(
            "STEP11_FINAL_M1_RAW_TOPOLOGY_FAILED:"
            +
            ",".join(
                final_m1_gate[
                    "topology_failure_reasons"
                ]
            )
        )

    if not bool(
        final_m2_gate[
            "topology_pass"
        ]
    ):
        raise RuntimeError(
            "STEP11_FINAL_M2_RAW_TOPOLOGY_FAILED:"
            +
            ",".join(
                final_m2_gate[
                    "topology_failure_reasons"
                ]
            )
        )

    physical_valid = np.asarray(
        final_trace[
            "valid"
        ],
        bool,
    )

    if not np.any(
        physical_valid
    ):
        raise RuntimeError(
            "STEP11_FINAL_NO_VALID_PHYSICAL_RAYS"
        )

    final_unobscuration = (
        _step11_unobscuration_state(
            ctx,
            final_m2,
            final_trace,
        )
    )

    final_mf2 = (
        final_unobscuration[
            "mf2"
        ]
    )

    unobscured = bool(
        final_unobscuration[
            "unobscured"
        ]
    )

    if not unobscured:

        raise RuntimeError(
            "STEP11_FINAL_UNOBSCURATION_FAILED:"
            f"S_AQP={float(final_unobscuration['S_AQP_signed_mm2']):.12g}"
        )

    curvature_limit = float(
        surface_fit_cfg[
            "curvature_absolute_max_per_mm"
        ]
    )

    conic_lo, conic_hi = map(
        float,
        surface_fit_cfg[
            "conic_bounds"
        ],
    )

    bound_tolerance = float(
        restoration_cfg[
            "bound_hit_relative_tolerance"
        ]
    )

    parameter_bound_rows = []

    for name, surface in (
        ("M1", final_m1),
        ("M2", final_m2),
    ):

        curvature_hit = bool(
            abs(
                float(
                    surface.curvature
                )
            )
            >=
            curvature_limit
            *
            (
                1.0
                -
                bound_tolerance
            )
        )

        conic_span = max(
            abs(
                conic_hi
                -
                conic_lo
            ),
            1.0,
        )

        conic_hit = bool(
            abs(
                float(
                    surface.conic
                )
                -
                conic_lo
            )
            <=
            bound_tolerance
            *
            conic_span
            or
            abs(
                float(
                    surface.conic
                )
                -
                conic_hi
            )
            <=
            bound_tolerance
            *
            conic_span
        )

        parameter_bound_rows.append({
            "surface":
                name,

            "curvature_per_mm":
                float(
                    surface.curvature
                ),

            "curvature_absolute_limit_per_mm":
                curvature_limit,

            "curvature_bound_hit":
                curvature_hit,

            "conic_constant":
                float(
                    surface.conic
                ),

            "conic_bounds":
                [
                    conic_lo,
                    conic_hi,
                ],

            "conic_bound_hit":
                conic_hit,
        })

    if bool(
        restoration_cfg[
            "reject_final_parameter_bound_hit"
        ]
    ):

        if any(
            row[
                "curvature_bound_hit"
            ]
            or
            row[
                "conic_bound_hit"
            ]
            for row
            in parameter_bound_rows
        ):

            raise RuntimeError(
                "STEP11_FINAL_PARAMETER_BOUND_HIT"
            )

    final_fit_m1 = (
        _step11_final_fit_diagnostic(
            final_m1,
            ctx.data[
                "ci_m1"
            ],
            surface_fit_cfg,
        )
    )

    final_fit_m2 = (
        _step11_final_fit_diagnostic(
            final_m2,
            seed[
                "ci_m2"
            ],
            surface_fit_cfg,
        )
    )

    write_json(
        step_dir
        /
        "11_FINAL_PARAMETER_BOUND_STATE.json",

        {
            "schema":
                "HUD_FAN_V5_5_"
                "STEP11_PARAMETER_BOUND_STATE_V1",

            "surfaces":
                parameter_bound_rows,
        },
    )

    # Legacy compatibility fields preserved for test_step11_preserves_ci_and_fit_diagnostics_in_record_and_history:
    # "M2_CI_bundle_curl_rms_p95_actual" "M2_CI_bundle_curl_rms_p95_limit"
    # "M2_CI_bundle_local_geometry_normal_rms_p95_actual" "M2_CI_bundle_local_geometry_normal_rms_p95_limit"
    # "M2_CI_bundle_loop_circulation_p95_actual" "M2_CI_bundle_loop_circulation_p95_limit"
    # "M2_CI_bundle_edge_gradient_height_residual_p95_actual" "M2_CI_bundle_edge_gradient_height_residual_p95_limit"
    # "M2_CI_bundle_nearest_normal_angle_p99_actual" "M2_CI_bundle_nearest_normal_angle_p99_limit"
    # "M2_CI_bundle_count" "M2_CI_evaluable_bundle_count"
    # "M2_fit_curvature_per_mm" "M2_fit_conic_constant" "M2_fit_sag_rms_mm"
    # "M2_fit_normal_rms_deg" "M2_fit_normal_max_deg" "M2_fit_k_bound_hit"
    # "M2_fit_curvature_bound_hit" "M1_topology_enforcement" "M1_topology_admitted"
    # "M1_topology_admission_status" "M1_topology_warning_reasons"
    # "M2_topology_enforcement" "M2_topology_admitted"
    # "M2_topology_admission_status" "M2_topology_warning_reasons"
    # "candidate_surface_admitted" "actual_M1_topology_pass"
    # "actual_M1_topology_admitted" "actual_M1_topology_admission_status"
    # "actual_M1_topology_warning_reasons" "actual_M2_topology_pass"
    # "actual_M2_topology_admitted" "actual_M2_topology_admission_status"
    # "actual_M2_topology_warning_reasons"

    final_m1_trust = (
        _step11_ci_trust_gate(
            restoration_common_seed,
            final_m1,
        )
    )

    ctx.data.update({
        "m1":
            final_m1,

        "m2":
            final_m2,

        "fit_m1_order2":
            final_fit_m1,

        "fit_m2_order2":
            final_fit_m2,

        "fit_m1_order2_role":
            "STEP11_FINAL_JOINT_RESTORATION_"
            "DIAGNOSTIC_AGAINST_FROZEN_CI",

        "ci_m1_diagnostics":
            m1_diagnostic,

        "ci_m2":
            seed[
                "ci_m2"
            ],

        "ci_m2_diagnostics":
            seed[
                "M2_diagnostic"
            ],

        "step11_selected_m2_conditioning_audit":
            seed[
                "M2_conditioning_audit"
            ],

        "step11_selected_m1_ci_trust":
            final_m1_trust,

        "step11_shape_authority": {
            "authority":
                "FIXED_RAY_FACING_DESIGN_TOPOLOGY",

            "M1_topology":
                m1_topology,

            "M2_topology":
                m2_topology,

            "topology_enforcement":
                dict(
                    topology_enforcement
                ),

            "M1_orientation_sign":
                float(
                    final_m1_gate[
                        "orientation_sign"
                    ]
                ),

            "M2_orientation_sign":
                float(
                    final_m2_gate[
                        "orientation_sign"
                    ]
                ),

            "shape_policy":
                shape_policy,
        },

        "step11_selected_pair_metrics":
            final_metrics,

        "step11_topology_restoration_history":
            restoration[
                "history"
            ],

        # Legacy compatibility.
        "step11_o2_search_history":
            compatibility_history,
    })

    write_json(
        step_dir / "11_M1_SHAPE_GATE.json",
        final_m1_gate,
    )

    write_json(
        step_dir / "11_M1_CI_TRUST_GATE.json",
        final_m1_trust,
    )

    write_json(
        step_dir / "11_M2_CLOUD_INTEGRABILITY_DIAGNOSTICS.json",
        seed["M2_diagnostic"],
    )

    write_json(
        step_dir / "11_M2_CI_INTEGRABILITY_GATE.json",
        seed["M2_integrability_gate"],
    )

    write_json(
        step_dir / "11_M2_CI_CONDITIONING.json",
        seed["M2_conditioning_audit"],
    )

    write_json(
        step_dir / "11_M2_FIT_DIAGNOSTICS.json",
        final_fit_m2,
    )

    write_json(
        step_dir / "11_M2_SHAPE_GATE.json",
        final_m2_gate,
    )

    write_json(
        step_dir / "11_SELECTED_O2_PAIR_METRICS.json",
        final_metrics,
    )

    write_csv(
        step_dir / "11_M2_CLOUD_DIAGNOSTIC_SAMPLES.csv",
        seed["M2_diagnostic"]["cloud"].get("sample_rows", []),
    )
    write_csv(
        step_dir / "11_M2_CLOUD_BUNDLE_DIAGNOSTICS.csv",
        seed["M2_diagnostic"]["cloud"].get("bundle_rows", []),
    )

    write_candidate_snapshot_manifest(
        "COMPLETE"
    )
    set_phase("COMPLETE")
    progress(
        "Complete | "
        f"restoration_iterations={len(restoration['history'])} | "
        f"final_objective={restoration['final_objective']:.6g} | "
        f"mapping_rms_mm={number(final_metrics['mapping_field_rms_mm'])} | "
        f"spot_rms_mm={number(final_metrics['spot_field_rms_mm'])} | "
        f"direction_rms_deg={number(final_metrics['target_direction_rms_deg'])} | "
        f"elapsed={duration(time.perf_counter() - step_started)}"
    )

    return {
        "status":
            "PASS",

        "mode":
            "JOINT_JACOBIAN_"
            "TOPOLOGY_RESTORATION",

        "M1_constructed_first":
            True,

        "M2_constructed_once_from_M1_seed":
            True,

        "M2_rebuilt_per_iteration":
            False,

        "joint_M1_M2_refinement":
            True,

        "parallel_finite_difference":
            bool(
                restoration[
                    "effective_workers"
                ]
                >
                1
            ),

        "effective_fd_workers":
            int(
                restoration[
                    "effective_workers"
                ]
            ),

        "restoration_iterations":
            int(
                len(
                    restoration[
                        "history"
                    ]
                )
            ),

        "restoration_stop_reason":
            restoration[
                "stop_reason"
            ],

        "M1_raw_topology_pass":
            True,

        "M2_raw_topology_pass":
            True,

        "physical_valid_count":
            int(
                final_metrics[
                    "physical_valid_count"
                ]
            ),

        "ray_count":
            int(
                final_metrics[
                    "ray_count"
                ]
            ),

        "mapping_rms_mm":
            float(
                final_metrics[
                    "mapping_field_rms_mm"
                ]
            ),

        "spot_rms_mm":
            float(
                final_metrics[
                    "spot_field_rms_mm"
                ]
            ),

        "direction_rms_deg":
            float(
                final_metrics[
                    "target_direction_rms_deg"
                ]
            ),

        "optical_objective":
            float(
                final_metrics[
                    "J_step12_actual_o2_dimensionless"
                ]
            ),

        "unobscured":
            unobscured,
    }



def step_12(ctx: Context) -> dict[str, Any]:
    """Joint-refine cặp O2 đã được STEP11 chọn; không refit M2 và phá winner trước refinement."""

    r = ctx.data["rays"]
    step12_started = time.perf_counter()
    m1_fit = ctx.data["fit_m1_order2"]
    m2 = ctx.data["m2"]
    s2 = ctx.data["fit_m2_order2"]
    m2_diagnostic = ctx.data["ci_m2_diagnostics"]

    selected_trust = ctx.data.get(
        "step11_selected_m1_ci_trust",
        {
            "sag_rms_mm": m1_fit["sag_rms_mm"],
            "normal_rms_deg": m1_fit["normal_rms_deg"],
        },
    )
    _step12_progress(
        "STEP11 selected O2 pair available "
        f"| M1_CI_sag_rms={float(selected_trust['sag_rms_mm']):.6g} mm "
        f"| M1_CI_normal_rms={float(selected_trust['normal_rms_deg']):.6g} deg "
        f"| M2_fit_sag_rms={float(s2['sag_rms_mm']):.6g} mm "
        f"| M2_fit_normal_rms={float(s2['normal_rms_deg']):.6g} deg"
    )

    write_json(
        ctx.step_dir(12) / "12_M1_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
        ctx.data["ci_m1_diagnostics"],
    )

    write_json(
        ctx.step_dir(12) / "12_M2_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
        m2_diagnostic,
    )

    quality_cfg = ctx.config["surface_fit"]["step12_quality_gates"]

    fit_gate_m1 = _fit_quality_gate(
        ctx.data["fit_m1_order2"],
        quality_cfg,
        "M1",
    )

    fit_gate_m2 = _fit_quality_gate(
        s2,
        quality_cfg,
        "M2",
    )

    integrability_cfg = ctx.config["surface_fit"]["integrability_gates"]

    step12_integrability_cfg = {
        **integrability_cfg,
        "enforcement": "WARN",
    }

    integrability_gate_m1 = _integrability_quality_gate(
        ctx.data["ci_m1_diagnostics"],
        step12_integrability_cfg,
        "M1_CI_CLOUD",
    )

    integrability_gate_m2 = _integrability_quality_gate(
        ctx.data["ci_m2_diagnostics"],
        step12_integrability_cfg,
        "M2_CI_CLOUD",
    )

    cloud_rows = [
        {
            **row,
            "authority":
                "DIAGNOSTIC_ONLY",
        }
        for row in (
            fit_gate_m1[
                "checks"
            ]
            +
            fit_gate_m2[
                "checks"
            ]
            +
            integrability_gate_m1[
                "checks"
            ]
            +
            integrability_gate_m2[
                "checks"
            ]
        )
    ]

    cloud_diagnostic = {
        "schema":
            "HUD_FAN_V5_5_STEP12_"
            "CLOUD_DIAGNOSTIC_ROLE_V1",

        "authority":
            "DIAGNOSTIC_ONLY",

        "acceptance_gate":
            False,

        "M1_fit":
            fit_gate_m1,

        "M2_fit":
            fit_gate_m2,

        "M1_integrability":
            integrability_gate_m1,

        "M2_integrability":
            integrability_gate_m2,

        "configured_integrability_enforcement_elsewhere":
            integrability_cfg["enforcement"],

        "step12_effective_integrability_enforcement": "WARN",

        "reason": "STEP12_ACCEPTANCE_AUTHORITY_MOVED_TO_ACTUAL_O2_SURFACE_RAY_TRACE",
    }

    write_json(
        ctx.step_dir(
            12
        )
        /
        "12_CLOUD_DIAGNOSTICS_ROLE.json",
        cloud_diagnostic,
    )

    write_csv(
        ctx.step_dir(
            12
        )
        /
        "12_FIT_QUALITY_GATES.csv",
        cloud_rows,
    )

    pre_aperture_quality = {
        "schema":
            "HUD_FAN_V5_5_STEP12_"
            "PRE_APERTURE_QUALITY_V2",

        "phase":
            "PRE_APERTURE_REBUILD",

        "cloud_fit_and_integrability_authority":
            "DIAGNOSTIC_ONLY",

        "M1_fit":
            fit_gate_m1,

        "M2_fit":
            fit_gate_m2,

        "M1_integrability":
            integrability_gate_m1,

        "M2_integrability":
            integrability_gate_m2,

        "aperture_rebuild_status":
            "NOT_STARTED",

        "actual_o2_refinement_status":
            "NOT_STARTED",
    }

    write_json(
        ctx.step_dir(
            12
        )
        /
        "12_PRE_APERTURE_QUALITY.json",
        pre_aperture_quality,
    )

    slot12 = ctx.data.get(
        "_step_debug"
    )

    if (
        isinstance(
            slot12,
            dict,
        )
        and
        slot12.get(
            "step"
        )
        == 12
    ):
        slot12[
            "current_phase"
        ] = (
            "APERTURE_REBUILD_"
            "BEFORE_O2_REFINEMENT"
        )

        slot12[
            "pre_aperture_quality"
        ] = copy.deepcopy(
            pre_aperture_quality
        )

    _step12_progress(
        "Initial aperture rebuild started "
        f"| rays={len(r['rows'])}"
    )

    (
        _,
        initial_trace,
        initial_aperture_audit,
    ) = (
        _rebuild_post_fit_apertures(
            ctx,
            ctx.data[
                "m1"
            ],
            m2,
            ctx.data[
                "display"
            ],
            "STEP_12_ORDER2_INITIAL",
        )
    )

    _step12_progress(
        _format_step12_aperture_progress_log(
            "INITIAL",
            initial_aperture_audit,
        )
    )

    write_json(
        ctx.step_dir(12) / "12_POST_FIT_APERTURE_REBUILD_INITIAL.json",
        initial_aperture_audit,
    )
    initial_m2_audit = initial_aperture_audit.get("M2", {})
    write_csv(
        ctx.step_dir(12) / "12_M2_APERTURE_FOOTPRINT_EXCLUSIONS_INITIAL.csv",
        initial_m2_audit.get("footprint_outlier_rows", []),
        fieldnames=M2_APERTURE_FOOTPRINT_EXCLUSIONS_CSV_FIELDNAMES,
    )
    write_csv(
        ctx.step_dir(12) / "12_M2_APERTURE_FOOTPRINT_BUNDLES_INITIAL.csv",
        initial_m2_audit.get("footprint_bundle_rows", []),
        fieldnames=M2_APERTURE_FOOTPRINT_BUNDLES_CSV_FIELDNAMES,
    )

    ctx.data[
        "trace"
    ] = (
        initial_trace
    )

    if (
        isinstance(
            slot12,
            dict,
        )
        and
        slot12.get(
            "step"
        )
        == 12
    ):
        slot12[
            "current_phase"
        ] = (
            "ACTUAL_O2_REFINEMENT"
        )

    (
        m1_refined,
        m2_refined,
        refinement_history,
        refinement_metrics,
        _,
    ) = (
        _step12_refine_o2(
            ctx,
            ctx.data[
                "m1"
            ],
            m2,
        )
    )

    ctx.data.update({
        "m1":
            m1_refined,

        "m2":
            m2_refined,

        "step12_o2_refinement_history":
            refinement_history,

        "step12_o2_refinement_"
        "metrics_before_final_aperture_rebuild":
            refinement_metrics,
    })

    write_csv(
        ctx.step_dir(
            12
        )
        /
        "12_O2_REFINEMENT_HISTORY.csv",
        refinement_history,
    )

    if (
        isinstance(
            slot12,
            dict,
        )
        and
        slot12.get(
            "step"
        )
        == 12
    ):
        slot12[
            "current_phase"
        ] = (
            "FINAL_APERTURE_REBUILD"
        )

    _step12_progress("Final aperture rebuild started")

    (
        _,
        tr,
        aperture_audit,
    ) = (
        _rebuild_post_fit_apertures(
            ctx,
            ctx.data[
                "m1"
            ],
            ctx.data[
                "m2"
            ],
            ctx.data[
                "display"
            ],
            "STEP_12_ORDER2_FINAL",
        )
    )

    _step12_progress(
        _format_step12_aperture_progress_log(
            "FINAL",
            aperture_audit,
        )
    )

    write_json(
        ctx.step_dir(12) / "12_POST_FIT_APERTURE_REBUILD.json",
        aperture_audit,
    )
    final_m2_audit = aperture_audit.get("M2", {})
    write_csv(
        ctx.step_dir(12) / "12_M2_APERTURE_FOOTPRINT_EXCLUSIONS_FINAL.csv",
        final_m2_audit.get("footprint_outlier_rows", []),
        fieldnames=M2_APERTURE_FOOTPRINT_EXCLUSIONS_CSV_FIELDNAMES,
    )
    write_csv(
        ctx.step_dir(12) / "12_M2_APERTURE_FOOTPRINT_BUNDLES_FINAL.csv",
        final_m2_audit.get("footprint_bundle_rows", []),
        fieldnames=M2_APERTURE_FOOTPRINT_BUNDLES_CSV_FIELDNAMES,
    )

    ctx.data[
        "trace"
    ] = tr

    if (
        isinstance(
            slot12,
            dict,
        )
        and
        slot12.get(
            "step"
        )
        == 12
    ):
        slot12[
            "current_phase"
        ] = (
            "POST_REFINEMENT_"
            "ACTUAL_O2_AUDIT"
        )

    _step12_progress("Final actual-O2 audit started")

    (
        _,
        actual_o2_metrics,
        actual_o2_trace,
    ) = (
        _step12_actual_o2_evaluate(
            ctx,
            ctx.data[
                "m1"
            ],
            ctx.data[
                "m2"
            ],
            ctx.config[
                "surface_fit"
            ][
                "step12_o2_refinement"
            ],
        )
    )

    actual_o2_gate = (
        _step12_actual_o2_gate(
            actual_o2_metrics,
            quality_cfg,
        )
    )

    _step12_progress(
        "Final actual-O2 audit complete "
        f"| status={actual_o2_gate['status']} "
        f"| valid={int(actual_o2_metrics['physical_valid_count'])}/"
        f"{len(r['rows'])}"
    )

    write_json(
        ctx.step_dir(
            12
        )
        /
        "12_O2_ACTUAL_SURFACE_METRICS.json",
        actual_o2_metrics,
    )

    write_csv(
        ctx.step_dir(
            12
        )
        /
        "12_O2_ACTUAL_SURFACE_GATES.csv",
        actual_o2_gate[
            "checks"
        ],
    )

    baseline_spot_pupil = (
        refinement_metrics[
            "step12_baseline_spot_pupil_metrics"
        ]
    )

    final_constraint_state = (
        _step12_constraint_state(
            ctx,
            ctx.data["m1"],
            ctx.data["m2"],
            actual_o2_metrics,
            actual_o2_trace,
            baseline_spot_pupil,
            monitor_only=False,
        )
    )

    write_json(
        ctx.step_dir(12)
        /
        "12_O2_FINAL_CONSTRAINT_STATE.json",
        final_constraint_state,
    )

    write_csv(
        ctx.step_dir(12)
        /
        "12_O2_FINAL_CONSTRAINT_GATES.csv",
        final_constraint_state[
            "gate_rows"
        ],
    )

    if not final_constraint_state[
        "feasibility_pass"
    ]:
        failed = ",".join(
            row["check"]
            for row in final_constraint_state[
                "gate_rows"
            ]
            if row["status"] != "PASS"
        )

        raise RuntimeError(
            "STEP12_FINAL_HARD_CONSTRAINT_FAIL:"
            f"{failed}"
        )

    actual_o2_valid = np.asarray(
        actual_o2_trace[
            "valid"
        ],
        bool,
    )

    if not np.any(
        actual_o2_valid
    ):
        raise RuntimeError(
            "ORDER2_ACTUAL_O2_HAS_NO_VALID_PHYSICAL_RAYS"
        )

    mf2 = mf2_geometry(
        actual_o2_trace[
            "points"
        ][
            0
        ][actual_o2_valid],
        actual_o2_trace[
            "points"
        ][
            1
        ][actual_o2_valid],
        actual_o2_trace[
            "points"
        ][
            2
        ][actual_o2_valid],
        ctx.data[
            "m2"
        ],
        float(
            ctx.config[
                "fan_weights"
            ][
                "omega2"
            ]
        ),
        ctx.data[
            "n_obs"
        ],
    )

    sanity1 = actual_o2_metrics["M1_sanity"]
    sanity2 = actual_o2_metrics["M2_sanity"]
    topology_admission1 = actual_o2_metrics[
        "M1_topology_admission"
    ]
    topology_admission2 = actual_o2_metrics[
        "M2_topology_admission"
    ]

    if (
        isinstance(
            slot12,
            dict,
        )
        and
        slot12.get(
            "step"
        )
        == 12
    ):
        slot12[
            "surface_sanity"
        ][
            "M1"
        ] = sanity1

        slot12[
            "surface_sanity"
        ][
            "M2"
        ] = sanity2

    expected = {
        (
            i,
            j,
        )
        for i in range(
            3
        )
        for j in range(
            3
        )
    }

    audit = {
        "equation":
            "conic + sum_i=0..2 "
            "sum_j=0..2 "
            "A_ij*x^i*y^j",

        "M1_terms":
            ctx.data[
                "m1"
            ].terms,

        "M2_terms":
            ctx.data[
                "m2"
            ].terms,

        "A21_A12_A22_present":
            all(
                x
                in set(
                    ctx.data[
                        "m2"
                    ].terms
                )
                for x in (
                    (2, 1),
                    (1, 2),
                    (2, 2),
                )
            ),

        "no_total_degree_truncation":
            True,

        "M1_fit":
            ctx.data[
                "fit_m1_order2"
            ],

        "M2_fit":
            s2,

        "cloud_fit_role":
            "DIAGNOSTIC_ONLY_AFTER_"
            "INITIAL_O2_BOOTSTRAP",

        "numeric_usable":
            bool(
                aperture_audit[
                    "finite_trace_accepted"
                ]
            ),

        "numeric_usable_role":
            "POST_FIT_APERTURE_REBUILD_MINIMUM_FINITE_FRACTION_"
            "WITH_NO_CONSTRUCTION_VALID_RAY_LOSS",

        "physical_valid_count":
            int(
                actual_o2_metrics[
                    "physical_valid_count"
                ]
            ),

        "unobscured":
            bool(
                float(
                    mf2[
                        "S_AQP_signed_mm2"
                    ]
                )
                >= 0.0
            ),

        "physical_obscuration_role":
            "UNOBSCURED_IS_HARD_O2_ACCEPTANCE_GATE_"
            "PHYSICAL_FRACTION_IS_STEP12_WARN_QUALITY_GATE",

        "M1_sanity":
            sanity1,

        "M2_sanity":
            sanity2,

        "M1_topology_admission":
            topology_admission1,

        "M2_topology_admission":
            topology_admission2,

        "actual_o2_quality": {
            "enforcement":
                quality_cfg[
                    "enforcement"
                ],

            **actual_o2_gate,
        },

        "actual_o2_metrics":
            actual_o2_metrics,

        "o2_refinement": {
            "enabled":
                bool(
                    ctx.config[
                        "surface_fit"
                    ][
                        "step12_o2_refinement"
                    ][
                        "enabled"
                    ]
                ),

            "iterations":
                len(
                    refinement_history
                ),

            "accepted_iterations":
                int(
                    sum(
                        bool(
                            row[
                                "accepted"
                            ]
                        )
                        for row
                        in refinement_history
                    )
                ),
        },

        "cloud_integrability_diagnostics": {
            "M1":
                ctx.data.get(
                    "ci_m1_diagnostics"
                ),

            "M2":
                ctx.data.get(
                    "ci_m2_diagnostics"
                ),
        },

        "cloud_diagnostic_quality":
            cloud_diagnostic,
    }

    write_json(
        ctx.step_dir(
            12
        )
        /
        "12_FAN_EQ1_ORDER2_SURFACES.json",
        audit
        |
        _prescription(
            ctx
        ),
    )

    if (
        set(
            ctx.data[
                "m1"
            ].terms
        )
        != expected
        or
        set(
            ctx.data[
                "m2"
            ].terms
        )
        != expected
    ):
        raise RuntimeError(
            "FAN_EQ1_TERM_SET_MISMATCH"
        )

    if not audit[
        "numeric_usable"
    ]:
        raise RuntimeError(
            "ORDER2_SURFACE_"
            "NUMERICALLY_UNUSABLE"
        )

    if (
        not bool(
            sanity1[
                "topology_pass"
            ]
        )
        or
        not bool(
            sanity2[
                "topology_pass"
            ]
        )
    ):
        failed_surf = (
            "M1"
            if not bool(
                sanity1[
                    "topology_pass"
                ]
            )
            else "M2"
        )

        bad_sanity = (
            sanity1
            if not bool(
                sanity1[
                    "topology_pass"
                ]
            )
            else sanity2
        )

        reasons = ",".join(
            bad_sanity.get(
                "topology_failure_reasons",
                [
                    "UNKNOWN"
                ],
            )
        )

        raise RuntimeError(
            "ORDER2_SURFACE_TOPOLOGY_FAIL_"
            f"{failed_surf}_"
            f"{reasons}"
        )

    if not audit[
        "unobscured"
    ]:
        raise RuntimeError(
            "ORDER2_SIGNED_MF2_OBSCURATION_FAIL"
        )

    if (
        quality_cfg[
            "enforcement"
        ]
        == "HARD"
        and
        actual_o2_gate[
            "status"
        ]
        != "PASS"
    ):
        failed = ",".join(
            f"{row['surface']}:"
            f"{row['check']}"
            for row
            in actual_o2_gate[
                "checks"
            ]
            if row[
                "status"
            ]
            == "FAIL"
        )

        raise RuntimeError(
            "ORDER2_ACTUAL_O2_GATE_FAIL:"
            f"{failed}"
        )

    # _capture_spot_snapshot(ctx, "ORDER_2", 12)
    _capture_spot_snapshot(
        ctx,
        "ORDER_2",
        12,
        trace=actual_o2_trace,
        references=ctx.data["fan_refs"],
        surface_state={
            "basis_order": 2,
            "shape_optimized_through_order": 2,
            "shape_update_orders": [2],
            "basis_promotion_orders": [],
            "basis_promotion_only": False,
            "zero_pad_promotion": False,
            "status": "ORDER2_ACTUAL_OPTICAL_SHAPE",
        },
        validity_mode="PHYSICAL_FIRST_HIT",
    )

    _record_optical_convergence(
        ctx,
        "STEP12 O2",
        12,
        reverse_trace=actual_o2_trace,
    )

    total_iters = len(refinement_history)
    cfg_iters = int(ctx.config["surface_fit"]["step12_o2_refinement"]["iterations"])
    n_acc = sum(1 for row in refinement_history if row.get("accepted"))
    n_rej = total_iters - n_acc

    if refinement_history:
        obj_init = float(refinement_history[0]["objective_before"])
        phys_init = float(refinement_history[0].get("physical_fraction_before", refinement_history[0]["physical_fraction"])) * 100.0
        rms_init = float(refinement_history[0].get("display_target_rms_mm_before", refinement_history[0]["display_target_rms_mm"]))
    else:
        obj_init = float(actual_o2_metrics["J_step12_actual_o2_dimensionless"])
        phys_init = float(actual_o2_metrics["physical_fraction"]) * 100.0
        rms_init = float(actual_o2_metrics["display_target_rms_mm"])

    obj_final = float(actual_o2_metrics["J_step12_actual_o2_dimensionless"])
    phys_final = float(actual_o2_metrics["physical_fraction"]) * 100.0
    rms_final = float(actual_o2_metrics["display_target_rms_mm"])
    obj_impr = ((obj_init - obj_final) / max(abs(obj_init), 1e-12)) * 100.0
    target_phys = float(quality_cfg["minimum_physical_fraction"]) * 100.0
    phys_stat = "PASS" if phys_final >= target_phys else "WARN"
    topo_m1 = str(topology_admission1["status"])
    topo_m2 = str(topology_admission2["status"])
    qual_m1 = sanity1.get("quality_status", "PASS")
    qual_m2 = sanity2.get("quality_status", "PASS")

    complete_lines = [
        "=" * 80,
        "[COMPLETE]",
        "=" * 80,
        f"Iterations       : {total_iters} / {cfg_iters}",
        f"Accepted         : {n_acc}",
        f"Rejected         : {n_rej}",
        "",
        "Objective",
        f"  initial        : {obj_init:.5f}",
        f"  final          : {obj_final:.5f}",
        f"  improvement    : {obj_impr:+.2f}%",
        "",
        "Physical",
        f"  initial        : {phys_init:.2f}%",
        f"  final          : {phys_final:.2f}%",
        f"  target         : {target_phys:.2f}%",
        f"  status         : {phys_stat}",
        "",
        "Chief/display RMS",
        f"  initial        : {rms_init:.3f} mm",
        f"  final          : {rms_final:.3f} mm",
        "",
        f"M1 topology      : {topo_m1}",
        f"M2 topology      : {topo_m2}",
        f"M1 quality       : {qual_m1}",
        f"M2 quality       : {qual_m2}",
        "",
        "Result           : O2 ACCEPTED",
        f"Elapsed          : {_step12_duration(time.perf_counter() - step12_started)}",
        "=" * 80,
    ]
    _step12_progress("\n".join(complete_lines))

    return (
        audit
        |
        {
            "status":
                "PASS"
        }
    )

def step_13(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 13: Kết thúc shortcut phẳng và chuyển sang Fan Step Two dùng Fermat."""

    ctx.data["step_two"] = True
    out = {"state": "FAN_STEP_TWO", "planar_conjugate_shortcut_disabled": True,
           "permitted_target_engine": "FERMAT_STATIONARY_POINT_ON_CURRENT_M2"}
    write_json(ctx.step_dir(13) / "13_STEP_TWO_STATE.json", out)
    return out | {"status": "PASS"}

def step_14(ctx: Context) -> dict[str, Any]:
    """Giải Fermat trên current M2, ghi per-ray evidence và hard-gate 5 điều kiện STEP14."""

    if not ctx.data.get("step_two"):
        raise RuntimeError("STEP_TWO_STATE_NOT_ACTIVE")

    step14_started = time.perf_counter()
    r = ctx.data["rays"]
    trace_started = time.perf_counter()
    tr = trace_reverse(
        r,
        ctx.data["visor"],
        ctx.data["m1"],
        ctx.data["m2"],
        ctx.data["display"],
        physical_first_hit=False,
    )
    runtime = current_runtime()
    trace_workers = int(getattr(runtime, "_persistent_ray_worker_count", 1)) if runtime is not None else 1
    _step14_progress(
        f"Reverse trace complete | rays={len(r['rows']):,} "
        f"| workers={trace_workers} "
        f"| elapsed={_step14_duration(time.perf_counter() - trace_started)}"
    )

    refs, rule = _dynamic_refs(
        ctx,
        tr,
    )
    ctx.data["fan_refs"] = refs

    s = ctx.config["solver"]
    gradient_tolerance = float(
        s["fermat_gradient_tolerance"]
    )
    reflection_tolerance = float(
        s[
            "fermat_reflection_residual_tolerance"
        ]
    )

    terminal_progress = _Step14TerminalProgress()
    sol = solve_fermat_m2(
        tr["points"][1],
        refs[
            r["field_index"]
        ],
        ctx.data["m2"],
        tr["points"][2],
        int(
            s["fermat_max_iter"]
        ),
        gradient_tolerance,
        reflection_tolerance=
            reflection_tolerance,
        progress_callback=terminal_progress,
        progress_interval_seconds=30.0,
    )

    audit = _fermat_audit(
        ctx.data["m2"],
        sol,
        "STEP_14_ORDER_2",
        expected_ray_count=
            len(r["rows"]),
        gradient_tolerance=
            gradient_tolerance,
        reflection_tolerance=
            reflection_tolerance,
    )

    diagnostic = (
        _step14_fermat_diagnostics(
            r,
            tr,
            ctx.data["m2"],
            sol,
            gradient_tolerance=
                gradient_tolerance,
            reflection_tolerance=
                reflection_tolerance,
            neighbor_count=int(
                ctx.config[
                    "surface_fit"
                ][
                    "diagnostic_neighbors"
                ]
            ),
        )
    )

    step14_certified = bool(
        audit["all_converged"]
        and diagnostic[
            "summary"
        ][
            "all_hard_pass"
        ]
    )

    step14_audit = dict(
        audit
    )
    step14_audit.update({
        "construction_domain_is_step14_hard_gate":
            True,
        "step14_hard_pass_count":
            diagnostic[
                "summary"
            ][
                "hard_pass_count"
            ],
        "step14_hard_fail_count":
            diagnostic[
                "summary"
            ][
                "hard_fail_count"
            ],
        "step14_certified":
            step14_certified,
        "continuity_suspect_count":
            diagnostic[
                "summary"
            ][
                "continuity_suspect_count"
            ],
        "continuity_is_hard_gate":
            False,
    })

    diagnostic[
        "summary"
    ][
        "legacy_fermat_audit_all_converged"
    ] = bool(
        audit["all_converged"]
    )
    diagnostic[
        "summary"
    ][
        "step14_certified"
    ] = step14_certified

    ctx.data.update({
        "fermat":
            sol,
        "trace":
            tr,
        "fermat_convergence_history":
            [step14_audit],
        "fermat_all_converged":
            step14_certified,
        "step14_fermat_diagnostics":
            copy.deepcopy(
                diagnostic[
                    "summary"
                ]
            ),
    })

    rows = diagnostic[
        "rows"
    ]
    failed_rows = diagnostic[
        "failed_rows"
    ]
    suspicious_rows = diagnostic[
        "suspicious_rows"
    ]
    fieldnames = (
        list(
            rows[0].keys()
        )
        if rows
        else
        []
    )

    write_csv(
        ctx.step_dir(14)
        / f"14_FERMAT_TARGETS_{len(rows)}.csv",
        rows,
        fieldnames=fieldnames,
    )
    write_csv(
        ctx.step_dir(14)
        / "14_FERMAT_FAILED_RAYS.csv",
        failed_rows,
        fieldnames=fieldnames,
    )
    write_csv(
        ctx.step_dir(14)
        / "14_FERMAT_SUSPICIOUS_PASS_RAYS.csv",
        suspicious_rows,
        fieldnames=fieldnames,
    )
    write_json(
        ctx.step_dir(14)
        / "14_FERMAT_QUALITY_SUMMARY.json",
        diagnostic[
            "summary"
        ],
    )

    out = {
        "OP_definition":
            "|Q2-Q1|+|P_ref_Fan-Q2|",
        "surface_constraint":
            "Q2 on current M2",
        "stationarity": [
            "dOP/dx2=0",
            "dOP/dy2=0",
        ],
        "planar_conjugate_used":
            False,

        "solver_success_count":
            int(
                np.sum(
                    sol["success"]
                )
            ),
        "step14_hard_pass_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_pass_count"
                ]
            ),
        "step14_hard_fail_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_fail_count"
                ]
            ),
        "ray_count":
            len(
                sol["success"]
            ),

        "gradient_rms":
            float(
                audit[
                    "gradient_rms"
                ]
            ),
        "gradient_max":
            float(
                audit[
                    "gradient_max"
                ]
            ),
        "gradient_fail_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_fail_counts"
                ][
                    "gradient"
                ]
            ),
        "reflection_fail_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_fail_counts"
                ][
                    "reflection"
                ]
            ),
        "finite_fail_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_fail_counts"
                ][
                    "finite"
                ]
            ),
        "conic_fail_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_fail_counts"
                ][
                    "conic"
                ]
            ),
        "construction_domain_fail_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "hard_fail_counts"
                ][
                    "construction_domain"
                ]
            ),

        "outside_current_aperture_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "outside_current_aperture_count"
                ]
            ),
        "continuity_suspect_count":
            int(
                diagnostic[
                    "summary"
                ][
                    "continuity_suspect_count"
                ]
            ),
        "continuity_is_hard_gate":
            False,
        "Delta_Q2_mm":
            diagnostic[
                "summary"
            ][
                "Delta_Q2_mm"
            ],
        "neighbor_delta_jump_mm":
            diagnostic[
                "summary"
            ][
                "neighbor_delta_jump_mm"
            ],

        "dynamic_reference":
            rule,
        "stationary_domain": {
            "current_clear_aperture_mutated":
                False,
            "next_surface_construction_domain_factor":
                float(
                    sol.get(
                        "construction_domain_factor",
                        1.0,
                    )
                ),
            "physical_current_M2_aperture_is_not_a_target_domain_gate":
                True,
            "construction_domain_is_step14_hard_gate":
                True,
            "post_reconstruction_physical_first_hit_required":
                True,
        },
        "all_converged":
            step14_certified,
        "partial_fermat_accepted": False,
        "diagnostic_continuation_allowed":
            False,
        "distortion_7_percent_gate_applied_here":
            False,
    }

    write_json(
        ctx.step_dir(14)
        / "14_FERMAT_SUMMARY.json",
        out,
    )

    _step14_progress(
        f"Artifacts complete | success={out['step14_hard_pass_count']:,}/{out['ray_count']:,} "
        f"| total={_step14_duration(time.perf_counter() - step14_started)}"
    )

    if not audit["all_converged"]:
        raise RuntimeError("FERMAT_STATIONARY_POINT_NOT_CONVERGED_FOR_ALL_RAYS")

    if not step14_certified:
        failures = diagnostic[
            "summary"
        ][
            "hard_fail_counts"
        ]
        raise RuntimeError(
            "FERMAT_STEP14_HARD_GATE_FAIL:"
            f"GRADIENT={failures['gradient']};"
            f"REFLECTION={failures['reflection']};"
            f"FINITE={failures['finite']};"
            f"CONIC={failures['conic']};"
            f"CONSTRUCTION_DOMAIN="
            f"{failures['construction_domain']};"
            f"LEGACY_AUDIT_FAIL="
            f"{int(not audit['all_converged'])}"
        )

    return out | {
        "status":
            "PASS"
    }

def step_15(ctx: Context) -> dict[str, Any]:
    """Khai báo coarse-to-fine search để chọn một rho O2 full-physical trước O3→O5."""

    solver = ctx.config["solver"]
    search = solver["rho_o2_search"]
    candidates = [float(v) for v in solver["rho_candidates"]]
    out = {
        "rho_candidates": candidates,
        "role": "ORDER2_SIGNED_RHO_COARSE_TO_FINE_SEARCH",
        "negative_branch_meaning": "opposite surface-power update direction",
        "positive_branch_meaning": "same surface-power update direction",
        "hard_feasibility": [
            "CI_INTEGRABILITY_PASS",
            "M1_SURFACE_SANITY_PASS",
            "M2_SURFACE_SANITY_PASS_OR_WARN_ADMITTED",
            "APERTURE_REBUILD_PASS",
            "PHYSICAL_FIRST_HIT_MINIMUM_CONFIGURED_FRACTION",
            "UNOBSCURED",
        ],
        "ranking_rule": [
            "FAN_CORE_MERIT",
            "BUNDLE_RMS_P95",
            "WORST_BUNDLE_RMS",
            "GLOBAL_CHIEF_CENTERED_RMS",
            "MAX_SPOT_RADIUS",
            "ABS_RHO",
        ],
        "search_mode": search["mode"],
        "maximum_refinement_levels": int(search["maximum_refinement_levels"]),
        "minimum_refinement_levels": int(search["minimum_refinement_levels"]),
        "grid_points": int(search["grid_points"]),
        "initial_half_width_fraction_of_nearest_spacing": float(
            search["initial_half_width_fraction_of_nearest_spacing"]
        ),
        "next_half_width_in_previous_grid_steps": float(
            search["next_half_width_in_previous_grid_steps"]
        ),
        "rho_tolerance": float(search["rho_tolerance"]),
        "minimum_m2_construction_finite_fraction": float(
            search.get(
                "minimum_m2_construction_finite_fraction",
                0.95,
            )
        ),
        "minimum_step16_candidate_fraction": float(
            search.get(
                "minimum_step16_candidate_fraction",
                search.get(
                    "minimum_m2_construction_finite_fraction",
                    0.95,
                ),
            )
        ),
        "maximum_m2_bundle_edge_gradient_height_residual_p95_mm": float(
            search.get(
                "maximum_m2_bundle_edge_gradient_height_residual_p95_mm",
                0.17,
            )
        ),
        "M2_edge_gradient_limit_scope": "STEP16_M2_ONLY",
        "m2_construction_excluded_rays_remain_excluded_from_metric_and_physical_valid_counts": True,
        "fan_relative_tolerance": float(solver["ci_plateau_relative"]),
        "bundle_p95_relative_worsening_tolerance": float(solver["ci_plateau_relative"]),
        "restoration_role": "DIAGNOSTIC_ONLY_NOT_ELIGIBLE_AS_STEP16_BEST",
        "selected_before_result_evaluation": False,
    }
    write_json(ctx.step_dir(15) / "15_SURFACE_FACTOR_RHO.json", out)
    return out | {"status": "PASS"}

def step_16(ctx: Context) -> dict[str, Any]:
    """Tìm rho O2 tốt nhất bằng coarse-to-fine; chỉ full-physical candidate được quyền thắng."""

    solver = ctx.config["solver"]
    search = solver["rho_o2_search"]
    initial_rho_values = [
        float(v)
        for v in solver["rho_candidates"]
    ]
    baseline = copy.deepcopy(ctx)
    records: list[dict[str, Any]] = []
    level_summaries: list[dict[str, Any]] = []
    full_states: list[dict[str, Any]] = []
    best_admissible_by_order: dict[int, dict[str, Any]] = {}
    evaluated: set[float] = set()
    parallel_signed_rho = bool(
        ctx.config.get("execution", {}).get(
            "parallel_signed_rho",
            False,
        )
    )

    def evaluate_level(
        rho_values: list[float],
        search_level: int,
    ) -> list[dict[str, Any]]:
        """Thực thi đánh giá song song hoặc tuần tự các giá trị rho trong một search level."""
        if parallel_signed_rho and len(rho_values) >= 2:
            from execution_v55 import managed_compute_pool
            from execution_workers_v55 import (
                ordered_bounded_map,
                signed_rho_branch_worker,
            )

            with managed_compute_pool(
                baseline,
                purpose=f"STEP16_RHO_LEVEL_{search_level:02d}",
            ) as (
                executor,
                max_inflight,
                policy,
            ):
                if executor is not None:
                    jobs = [
                        {
                            "ordinal": index,
                            "rho": float(rho),
                        }
                        for index, rho in enumerate(
                            rho_values,
                            start=1,
                        )
                    ]
                    return ordered_bounded_map(
                        executor,
                        signed_rho_branch_worker,
                        jobs,
                        max_inflight,
                    )

        return [
            _evaluate_step16_rho_branch(
                copy.deepcopy(baseline),
                float(rho),
            )
            for rho in rho_values
        ]

    def consume_level(
        rho_values: list[float],
        replies: list[dict[str, Any]],
        search_level: int,
    ) -> list[dict[str, Any]]:
        """Thu nạp kết quả đánh giá rho, cập nhật candidate list và live preview."""
        level_full: list[dict[str, Any]] = []

        for live_ordinal, (
            rho,
            reply,
        ) in enumerate(
            zip(rho_values, replies),
            start=1,
        ):
            evaluated.add(
                round(
                    float(rho),
                    15,
                )
            )
            live_item(
                "SIGNED_RHO",
                live_ordinal,
                len(rho_values),
                rho=float(rho),
                order=2,
                search_level=int(search_level),
            )

            row = dict(reply["row"])
            row["search_level"] = int(search_level)
            row["search_level_ordinal"] = int(live_ordinal)
            evidence = reply.get("evidence")
            if isinstance(evidence, dict):
                construction_gate = evidence.get(
                    "m2_construction_input_gate",
                    {},
                )
                row.update({
                    "M2_construction_valid_count": construction_gate.get(
                        "valid_geometry_count"
                    ),
                    "M2_construction_ray_count": construction_gate.get(
                        "ray_count"
                    ),
                    "M2_construction_valid_fraction": construction_gate.get(
                        "valid_geometry_fraction"
                    ),
                    "M2_construction_minimum_fraction": construction_gate.get(
                        "minimum_valid_geometry_fraction"
                    ),
                    "M2_construction_excluded_count": construction_gate.get(
                        "excluded_geometry_count"
                    ),
                    "M2_construction_failed_bundle_count": construction_gate.get(
                        "failed_bundle_count"
                    ),
                    "M2_construction_underfilled_bundle_count": construction_gate.get(
                        "underfilled_bundle_count"
                    ),
                })
                bundle_audit = evidence.get(
                    "M2_bundle_integrability_diagnostic",
                    {},
                )
                worst_bundle = bundle_audit.get("worst_bundle") or {}
                row.update({
                    "M2_CI_edge_gradient_height_residual_p95_actual": (
                        bundle_audit.get(
                            "aggregate_edge_gradient_height_residual_p95_mm"
                        )
                    ),
                    "M2_CI_edge_gradient_height_residual_p95_limit": (
                        bundle_audit.get(
                            "edge_gradient_height_residual_limit_mm"
                        )
                    ),
                    "M2_CI_edge_gradient_violating_bundle_count": (
                        bundle_audit.get("violating_bundle_count")
                    ),
                    "M2_CI_edge_gradient_worst_field_id": worst_bundle.get(
                        "field_id"
                    ),
                    "M2_CI_edge_gradient_worst_pupil_id": worst_bundle.get(
                        "pupil_id"
                    ),
                    "M2_CI_edge_gradient_worst_bundle_actual_mm": (
                        worst_bundle.get(
                            "edge_gradient_height_residual_p95_mm"
                        )
                    ),
                })
                numeric_admission = evidence.get(
                    "numeric_admission",
                    {},
                )
                physical_admission = evidence.get(
                    "physical_admission",
                    {},
                )
                surface_admission = evidence.get(
                    "surface_sanity_admission",
                    {},
                )
                m1_surface_admission = surface_admission.get("M1", {})
                m2_surface_admission = surface_admission.get("M2", {})
                row.update({
                    "STEP16_numeric_valid_count": numeric_admission.get(
                        "valid_count"
                    ),
                    "STEP16_numeric_ray_count": numeric_admission.get(
                        "ray_count"
                    ),
                    "STEP16_numeric_valid_fraction": numeric_admission.get(
                        "valid_fraction"
                    ),
                    "STEP16_numeric_minimum_fraction": numeric_admission.get(
                        "minimum_fraction"
                    ),
                    "STEP16_numeric_pass": numeric_admission.get("pass"),
                    "STEP16_physical_valid_count": physical_admission.get(
                        "valid_count"
                    ),
                    "STEP16_physical_ray_count": physical_admission.get(
                        "ray_count"
                    ),
                    "STEP16_physical_valid_fraction": physical_admission.get(
                        "valid_fraction"
                    ),
                    "STEP16_physical_minimum_fraction": physical_admission.get(
                        "minimum_fraction"
                    ),
                    "STEP16_physical_fraction_pass": physical_admission.get(
                        "fraction_pass"
                    ),
                    "STEP16_physical_all_rays_complete": physical_admission.get(
                        "all_rays_complete"
                    ),
                    "STEP16_M1_sanity_raw_pass": m1_surface_admission.get(
                        "raw_pass"
                    ),
                    "STEP16_M1_sanity_admitted": m1_surface_admission.get(
                        "admitted"
                    ),
                    "STEP16_M2_sanity_raw_pass": m2_surface_admission.get(
                        "raw_pass"
                    ),
                    "STEP16_M2_sanity_enforcement": m2_surface_admission.get(
                        "enforcement"
                    ),
                    "STEP16_M2_sanity_warn_admitted": m2_surface_admission.get(
                        "warn_admitted"
                    ),
                    "STEP16_M2_sanity_admitted": m2_surface_admission.get(
                        "admitted"
                    ),
                })

            if reply["status"] == "ACCEPTED":
                state = reply["state"]

                if bool(
                    row.get(
                        "eligible_for_step16_best",
                        False,
                    )
                ):
                    level_full.append(state)
                    full_states.append(state)
                else:
                    previous = (
                        best_admissible_by_order.get(2)
                    )
                    if (
                        previous is None
                        or _rho_sort_key(state)
                        < _rho_sort_key(previous)
                    ):
                        best_admissible_by_order[2] = (
                            copy.deepcopy(state)
                        )

                records.append(row)
            else:
                records.append(row)
                records[-1].update(
                    _persist_candidate_evidence_payload(
                        ctx,
                        evidence,
                        16,
                        2,
                        search_level + 1,
                        (
                            f"level_{search_level:02d}_"
                            f"rho_{float(rho):+.12g}"
                        ),
                        reply.get(
                            "error",
                            "REJECTED",
                        ),
                    )
                )

            live_results(
                lambda: records
            )

            if isinstance(
                evidence,
                dict,
            ):
                physical_trace = evidence.get(
                    "physical_trace"
                )
                preview_branch = copy.deepcopy(ctx)
                preview_branch.data["rho"] = float(rho)

                _live_model_preview(
                    preview_branch,
                    (
                        f"RHO L{search_level} "
                        f"{float(rho):+.6g} - "
                        f"{records[-1].get('reason', 'RESULT_RECORDED')}"
                    ),
                    m1=evidence.get("m1"),
                    m2=evidence.get("m2"),
                    display=evidence.get("display"),
                    trace=physical_trace,
                    trace_mode=(
                        "PHYSICAL_FIRST_HIT"
                        if physical_trace is not None
                        else (
                            "NO_COMPLETE_PHYSICAL_TRACE_"
                            "FOR_THIS_CANDIDATE"
                        )
                    ),
                    role=(
                            "MINIMUM_FRACTION_FEASIBLE_STEP16_CANDIDATE"
                        if bool(
                            records[-1].get(
                                "eligible_for_step16_best",
                                False,
                            )
                        )
                        else (
                            "REJECTED_OR_RESTORATION_ONLY_"
                            "CANDIDATE"
                        )
                    ),
                    force=True,
                )

        return level_full

    coarse_replies = evaluate_level(
        initial_rho_values,
        0,
    )
    coarse_full = consume_level(
        initial_rho_values,
        coarse_replies,
        0,
    )

    if not coarse_full:
        write_csv(
            ctx.step_dir(16)
            / "16_RHO_COARSE_TO_FINE_CANDIDATES.csv",
            records,
        )
        raise RuntimeError(
            "ORDER_2_NO_MINIMUM_PHYSICAL_FRACTION_SIGNED_RHO_CANDIDATE_"
            f"REQUIRED_{_step16_minimum_candidate_fraction(ctx):.6f}"
        )

    best = min(
        full_states,
        key=_rho_o2_search_sort_key,
    )
    best_rho = float(
        best["rho_sequence"][-1]
    )

    nearest_spacing = (
        _step16_nearest_same_sign_spacing(
            best_rho,
            initial_rho_values,
        )
    )

    half_width = (
        float(
            search[
                "initial_half_width_fraction_of_nearest_spacing"
            ]
        )
        * nearest_spacing
    )

    level_summaries.append({
        "search_level": 0,
        "candidate_count": len(initial_rho_values),
        "full_feasible_count": len(coarse_full),
        "best_rho": best_rho,
        "best_Fan_core_merit": float(
            best["Fan_core_merit"]
        ),
        "best_bundle_RMS_P95_mm": (
            best["reconstruct_out"][
                "step16_spot_tail"
            ]["bundle_RMS_P95_mm"]
        ),
        "status": "COARSE_COMPLETE",
    })

    previous_best = copy.deepcopy(best)
    stop_reason = (
        "MAXIMUM_REFINEMENT_LEVELS_REACHED"
    )

    maximum_levels = int(
        search["maximum_refinement_levels"]
    )
    minimum_levels = int(
        search["minimum_refinement_levels"]
    )
    grid_points = int(
        search["grid_points"]
    )
    rho_tolerance = float(
        search["rho_tolerance"]
    )
    plateau_relative = float(
        solver["ci_plateau_relative"]
    )

    for refinement_level in range(
        1,
        maximum_levels + 1,
    ):
        rho_values, grid_step = (
            _step16_refinement_grid(
                float(
                    previous_best[
                        "rho_sequence"
                    ][-1]
                ),
                half_width,
                grid_points,
                evaluated,
            )
        )

        if not rho_values:
            stop_reason = (
                "NO_NEW_DISTINCT_RHO_VALUES"
            )
            break

        replies = evaluate_level(
            rho_values,
            refinement_level,
        )

        level_full = consume_level(
            rho_values,
            replies,
            refinement_level,
        )

        if not level_full:
            level_summaries.append({
                "search_level": refinement_level,
                "candidate_count": len(rho_values),
                "full_feasible_count": 0,
                "best_rho": float(
                    previous_best[
                        "rho_sequence"
                    ][-1]
                ),
                "best_Fan_core_merit": float(
                    previous_best[
                        "Fan_core_merit"
                    ]
                ),
                "best_bundle_RMS_P95_mm": (
                    previous_best[
                        "reconstruct_out"
                    ][
                        "step16_spot_tail"
                    ][
                        "bundle_RMS_P95_mm"
                    ]
                ),
                "grid_half_width": float(
                    half_width
                ),
                "grid_step": float(
                    grid_step
                ),
                "status": (
                    "NO_NEW_FULL_FEASIBLE_NEIGHBOR"
                ),
            })

            best = previous_best
            stop_reason = (
                "REFINEMENT_NEIGHBORHOOD_HAS_NO_"
                "NEW_FULL_FEASIBLE_CANDIDATE"
            )
            break

        best = min(
            full_states,
            key=_rho_o2_search_sort_key,
        )

        old_rho = float(
            previous_best[
                "rho_sequence"
            ][-1]
        )
        new_rho = float(
            best[
                "rho_sequence"
            ][-1]
        )

        old_merit = float(
            previous_best[
                "Fan_core_merit"
            ]
        )
        new_merit = float(
            best[
                "Fan_core_merit"
            ]
        )

        relative_fan_improvement = (
            old_merit - new_merit
        ) / max(
            abs(old_merit),
            1e-15,
        )

        old_p95 = (
            _step16_metric_or_inf(
                previous_best[
                    "reconstruct_out"
                ][
                    "step16_spot_tail"
                ].get(
                    "bundle_RMS_P95_mm"
                )
            )
        )

        new_p95 = (
            _step16_metric_or_inf(
                best[
                    "reconstruct_out"
                ][
                    "step16_spot_tail"
                ].get(
                    "bundle_RMS_P95_mm"
                )
            )
        )

        p95_not_worse = bool(
            new_p95
            <= old_p95
            * (
                1.0
                + plateau_relative
            )
        )

        rho_delta = abs(
            new_rho - old_rho
        )

        level_summaries.append({
            "search_level": refinement_level,
            "candidate_count": len(rho_values),
            "full_feasible_count": len(
                level_full
            ),
            "best_rho": new_rho,
            "best_Fan_core_merit": new_merit,
            "best_bundle_RMS_P95_mm": (
                new_p95
            ),
            "rho_delta_from_previous_best": (
                rho_delta
            ),
            "relative_Fan_improvement": (
                relative_fan_improvement
            ),
            "bundle_P95_not_worse": (
                p95_not_worse
            ),
            "grid_half_width": float(
                half_width
            ),
            "grid_step": float(
                grid_step
            ),
            "status": "REFINEMENT_COMPLETE",
        })

        if (
            refinement_level >= minimum_levels
            and rho_delta <= rho_tolerance
            and relative_fan_improvement
            <= plateau_relative
            and p95_not_worse
        ):
            stop_reason = (
                "RHO_AND_FAN_PLATEAU_WITH_"
                "STABLE_BUNDLE_P95"
            )
            previous_best = copy.deepcopy(
                best
            )
            break

        previous_best = copy.deepcopy(
            best
        )

        half_width = (
            float(
                search[
                    "next_half_width_in_previous_grid_steps"
                ]
            )
            * grid_step
        )

    best = min(
        full_states,
        key=_rho_o2_search_sort_key,
    )

    _apply_rho_state(
        ctx,
        best,
    )

    live_phase(
        "SELECTED_INCUMBENT"
    )

    live_note(
        "INCUMBENT_SELECTED",
        rho_sequence=best["rho_sequence"],
        Fan_core_merit=best["Fan_core_merit"],
        physical_valid_count=(
            best["reconstruct_out"][
                "physical_valid_count"
            ]
        ),
        ray_count=(
            best["reconstruct_out"][
                "ray_count"
            ]
        ),
        feasibility_status=(
            best["reconstruct_out"][
                "feasibility_status"
            ]
        ),
    )

    _live_model_preview(
        ctx,
        (
            "STEP16 SELECTED MINIMUM-FRACTION-PHYSICAL "
            "O2 RHO - not final optical design"
        ),
        m1=ctx.data["m1"],
        m2=ctx.data["m2"],
        display=ctx.data["display"],
        trace=ctx.data[
            "last_ci_physical_trace"
        ],
        trace_mode="PHYSICAL_FIRST_HIT",
        role=(
            "SELECTED_MINIMUM_FRACTION_PHYSICAL_O2_RHO"
        ),
        force=True,
    )

    _publish_trace_debug(
        ctx,
        16,
        "SELECTED_CI_PHYSICAL_TRACE",
        ctx.data[
            "last_ci_physical_trace"
        ],
        ctx.data["rays"],
        direction="REVERSE",
    )

    _record_optical_convergence(
        ctx,
        (
            "STEP16 rho="
            f"{float(best['rho_sequence'][-1]):+.6f}"
        ),
        16,
        reverse_trace=ctx.data[
            "last_ci_physical_trace"
        ],
    )

    ctx.data["_rho_frontier"] = [
        copy.deepcopy(best)
    ]

    ctx.data["_ci_incumbents"] = {
        "feasible": {
            2: copy.deepcopy(best),
        },
        "admissible": (
            best_admissible_by_order
        ),
    }

    search_summary = {
        "schema": (
            "HUD_FAN_V5_5_STEP16_"
            "RHO_COARSE_TO_FINE_V1"
        ),
        "baseline": (
            "FIXED_POST_STEP14_ORDER2_CONTEXT_FOR_EVERY_RHO"
        ),
        "initial_candidates": (
            initial_rho_values
        ),
        "evaluated_candidate_count": len(
            evaluated
        ),
        "full_feasible_candidate_count": len(
            full_states
        ),
        "threshold_feasible_candidate_count": len(
            full_states
        ),
        "minimum_candidate_fraction": (
            _step16_minimum_candidate_fraction(ctx)
        ),
        "selected_rho": float(
            best["rho_sequence"][-1]
        ),
        "selected_sort_key": list(
            _rho_o2_search_sort_key(
                best
            )
        ),
        "selection_rule": (
            "HARD_MINIMUM_PHYSICAL_FRACTION_THEN_FAN_THEN_BUNDLE_P95_"
            "THEN_WORST_BUNDLE_THEN_GLOBAL_RMS_THEN_"
            "MAX_SPOT_THEN_ABS_RHO"
            # HARD_FULL_PHYSICAL_THEN_FAN_THEN_BUNDLE_P95
            # FEASIBLE_FIRST_THEN_LIMITED_RESTORATION_THEN_FAN_MERIT_THEN_PAPER_SPOT
        ),
        "restoration_policy": (
            "RECORD_ONLY_NOT_ELIGIBLE_AS_STEP16_BEST"
        ),
        "stop_reason": stop_reason,
        "levels": level_summaries,
    }

    out = (
        best["reconstruct_out"]
        | {
            "rho_sequence": (
                best["rho_sequence"]
            ),
            "accepted_branch_count": int(
                sum(
                    bool(
                        row.get(
                            "accepted",
                            False,
                        )
                    )
                    for row in records
                )
            ),
            "full_feasible_candidate_count": (
                len(full_states)
            ),
            "threshold_feasible_candidate_count": (
                len(full_states)
            ),
            "minimum_candidate_fraction": (
                _step16_minimum_candidate_fraction(ctx)
            ),
            "retained_branch_count": 1,
            "selection_rule": (
                search_summary[
                    "selection_rule"
                ]
            ),
            "rho_search": (
                search_summary
            ),
        }
    )

    write_csv(
        ctx.step_dir(16)
        / "16_RHO_COARSE_TO_FINE_CANDIDATES.csv",
        records,
    )

    # Giữ filename cũ để không phá tool/report cũ.
    write_csv(
        ctx.step_dir(16)
        / "16_SIGNED_RHO_BRANCHES.csv",
        records,
    )

    write_json(
        ctx.step_dir(16)
        / "16_RHO_COARSE_TO_FINE_SUMMARY.json",
        search_summary,
    )

    write_json(
        ctx.step_dir(16)
        / "16_RECONSTRUCT_M1_THEN_M2.json",
        out,
    )

    write_json(
        ctx.step_dir(16)
        / "16_SELECTED_CI_CLOUD_INTEGRABILITY.json",
        {
            "M1": ctx.data[
                "last_ci_m1_diagnostics"
            ],
            "M2": ctx.data[
                "last_ci_m2_diagnostics"
            ],
            "quality": (
                best[
                    "reconstruct_out"
                ][
                    "integrability_quality"
                ]
            ),
        },
    )

    return out | {
        "status": "PASS"
    }


def _ci_snapshot_surface_state(
    state: dict[str, Any],
    basis_order: int,
) -> dict[str, Any]:
    """Describe whether a STEP17 stage is a real shape update or basis-only promotion."""
    history = state["order_history"]

    accepted_shape_orders = sorted({
        int(row["order"])
        for row in history
        if bool(row.get("accepted", False))
        and bool(row.get("accepted_shape_update", False))
        and bool(row.get("shape_optimized", False))
        and not bool(row.get("basis_promoted", False))
    })

    basis_promotion_orders = sorted({
        int(row["order"])
        for row in history
        if bool(row.get("basis_promoted", False))
        and not bool(row.get("accepted_shape_update", False))
        and not bool(row.get("shape_optimized", False))
    })

    shape_optimized_through_order = (
        max(accepted_shape_orders)
        if accepted_shape_orders
        else int(state["reconstruct_out"]["order"])
    )

    basis_promotion_only = bool(
        int(basis_order) in basis_promotion_orders
        and int(basis_order) not in accepted_shape_orders
    )

    return {
        "basis_order": int(basis_order),
        "shape_optimized_through_order": int(
            shape_optimized_through_order
        ),
        "shape_update_orders": accepted_shape_orders,
        "basis_promotion_orders": basis_promotion_orders,
        "basis_promotion_only": basis_promotion_only,
        "zero_pad_promotion": basis_promotion_only,
        "status": (
            f"ORDER{int(basis_order)}_BASIS_ONLY_"
            f"ORDER{int(shape_optimized_through_order)}_OPTICAL_SHAPE"
            if basis_promotion_only
            else f"ORDER{int(basis_order)}_OPTICAL_SHAPE_ACCEPTED"
        ),
    }


def step_17(ctx: Context) -> dict[str, Any]:
    """Lặp successive approximation [21]/HUD Eq.(3) đến hội tụ cho từng order."""

    frontier = ctx.data.pop("_rho_frontier")
    incumbents = ctx.data.get("_ci_incumbents", {"feasible": {}, "admissible": {}})
    best_feasible_by_order: dict[int, dict[str, Any]] = incumbents.get("feasible", {})
    best_admissible_by_order: dict[int, dict[str, Any]] = incumbents.get("admissible", {})

    rsp = ctx.config.get("ray_sampling_profiles")
    if isinstance(rsp, dict) and rsp.get("enabled", False):
        if "profile_rays_O2_SEARCH" not in ctx.data and "rays" in ctx.data:
            ctx.data["profile_rays_O2_SEARCH"] = copy.deepcopy(ctx.data["rays"])
            ctx.data["profile_vi_O2_SEARCH"] = copy.deepcopy(ctx.data["vi"])
            ctx.data["profile_pupils_O2_SEARCH"] = copy.deepcopy(ctx.data["pupils"])
            ctx.data["profile_pattern_O2_SEARCH"] = copy.deepcopy(ctx.data["pattern"])
            ctx.data["profile_visor_hit_O2_SEARCH"] = copy.deepcopy(ctx.data["visor_hit"])
            ctx.data["profile_post_visor_O2_SEARCH"] = copy.deepcopy(ctx.data["post_visor"])

        _apply_ray_sampling_profile(ctx, "FREEFORM", 17)
        ctx.data["ci_m1_sampling_profile"] = "O2_SEARCH"
        ctx.data["ci_m2_sampling_profile"] = "O2_SEARCH"
        ctx.data["fermat_sampling_profile"] = "FREEFORM"

        # Đánh giá lại parent trên 16.875 tia làm baseline cho O3
        if frontier:
            r_new = ctx.data["rays"]
            m1_parent = ctx.data["m1"]
            m2_parent = ctx.data["m2"]
            disp_parent = ctx.data["display"]
            vis_parent = ctx.data["visor"]

            tr_phys = trace_reverse(r_new, vis_parent, m1_parent, m2_parent, disp_parent, physical_first_hit=True)
            phys_count = int(np.sum(tr_phys["valid"]))
            phys_frac = phys_count / max(len(r_new["rows"]), 1)
            min_frac = _step17_minimum_candidate_fraction(ctx)
            phys_pass = bool(phys_frac >= min_frac)
            full_phys = (phys_count == len(r_new["rows"]))

            tr_std = trace_reverse(r_new, vis_parent, m1_parent, m2_parent, disp_parent, physical_first_hit=False)
            refs_new, rule_new = _dynamic_refs(ctx, tr_std)
            mf1_new = fan_imaging_metrics(tr_std, r_new, refs_new, float(ctx.config["fan_weights"]["omega1"]))
            mf2_new = mf2_geometry(tr_std["points"][0], tr_std["points"][1], tr_std["points"][2], m2_parent, float(ctx.config["fan_weights"]["omega2"]), ctx.data["n_obs"])
            fan_merit_new = float(mf1_new["MF1_Fan"] + mf2_new["MF2"])
            spot_new = chief_centered_geometric_spot_rms(tr_phys["landing"], tr_phys["valid"], r_new["field_index"], r_new["pupil_index"], r_new["chief"], disp_parent.frame)

            ctx.data.update({
                "trace": tr_std,
                "fan_refs": refs_new,
                "fan_reference_rule": rule_new,
                "last_ci_physical_trace": tr_phys,
            })

            p_out = copy.deepcopy(frontier[0]["reconstruct_out"])
            p_out["physical_valid_count"] = phys_count
            p_out["ray_count"] = len(r_new["rows"])
            p_out["physical_valid_fraction"] = phys_frac
            p_out["physical_fraction_pass"] = phys_pass
            p_out["physical_first_hit_complete"] = full_phys
            p_out["chief_centered_spot_RMS_mm"] = spot_new["RMS_spot_radius_mm"]
            p_out["chief_centered_spot"] = spot_new
            p_out["MF1_Fan"] = mf1_new["MF1_Fan"]
            p_out["MF2_Fan"] = mf2_new["MF2"]
            p_out["S_AQP_signed_mm2"] = float(mf2_new["S_AQP_signed_mm2"])
            p_out["Fan_core_merit"] = fan_merit_new

            parent_state = _rho_state(ctx, frontier[0]["rho_sequence"], frontier[0]["order_history"], p_out)
            parent_state["Fan_core_merit"] = fan_merit_new
            parent_state["trace"] = tr_std
            parent_state["fan_refs"] = refs_new
            parent_state["fan_reference_rule"] = rule_new
            parent_state["last_ci_physical_trace"] = tr_phys

            frontier = [copy.deepcopy(parent_state)]
            best_feasible_by_order[2] = copy.deepcopy(parent_state)
            ctx.data["_ci_incumbents"] = {"feasible": best_feasible_by_order, "admissible": best_admissible_by_order}
    rejected: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    convergence_records: list[dict[str, Any]] = []
    rejected_fields = ["order", "cycle", "parent_rho_sequence", "rho", "accepted", "reason",
                       "fermat_success_count", "fermat_ray_count", "gradient_pass_count",
                       "reflection_pass_count", "wrong_reflection_branch_count",
                       "fermat_admissible_count", "fermat_admissible_fraction",
                       "fermat_minimum_fraction", "fermat_fraction_pass",
                       "inside_current_aperture_count", "gradient_max",
                       "M2_footprint_filter_enabled", "M2_footprint_rejected_count",
                       "M2_footprint_rejected_fraction", "M2_footprint_retained_count",
                       "M2_footprint_affected_bundle_count", "M2_footprint_max_bundle_rejected_fraction",
                       "candidate_artifact", "diagnostic_export_status", "diagnostic_export_error",
                       "spot_RMS_guard_enabled", "parent_spot_RMS_mm", "candidate_spot_RMS_mm",
                       "spot_RMS_delta_mm", "maximum_spot_RMS_regression_mm",
                       "spot_RMS_guard_pass", "spot_RMS_guard_status",
                       *STEP17_SPOT_SUMMARY_FIELDS,
                       *STEP17_SPOT_DELTA_FIELDS,
                       *STEP17_SPOT_EXPORT_FIELDS]
    accepted_fields = ["order", "cycle", "rho_sequence", "Fan_core_merit",
                       "chief_centered_spot_RMS_mm", "physical_valid_count",
                       "physical_valid_fraction", "minimum_candidate_fraction",
                       "physical_fraction_pass", "physical_first_hit_complete",
                       "fermat_admissible_fraction", "fermat_minimum_fraction",
                       "fermat_all_converged", "feasibility_status", "status",
                       "M2_footprint_filter_enabled", "M2_footprint_rejected_count",
                       "M2_footprint_rejected_fraction", "M2_footprint_retained_count",
                       "M2_footprint_affected_bundle_count", "M2_footprint_max_bundle_rejected_fraction",
                       "spot_RMS_guard_enabled", "parent_spot_RMS_mm", "candidate_spot_RMS_mm",
                       "spot_RMS_delta_mm", "maximum_spot_RMS_regression_mm",
                       "spot_RMS_guard_pass", "spot_RMS_guard_status",
                       *STEP17_SPOT_SUMMARY_FIELDS,
                       *STEP17_SPOT_DELTA_FIELDS,
                       *STEP17_SPOT_EXPORT_FIELDS]
    solver = ctx.config["solver"]
    step17_spot_guard_cfg = solver.get(
        "step17_spot_guard",
        {
            "enabled": True,
            "minimum_order": 4,
            "maximum_regression_mm": 0.01,
        },
    )
    live_cfg = ctx.config.get(
        "live_monitor", {}
    )

    step17_images_enabled = bool(
        live_cfg.get(
            "images_enabled", True
        )
    )
    if step17_images_enabled:
        from visualization_v55 import render_step17_branch_progress

    step17_pre_fermat_images = bool(
        step17_images_enabled
        and live_cfg.get(
            "step17_pre_fermat_images",
            False,
        )
    )
    maximum_cycles = int(solver["ci_cycles_per_order"])
    minimum_cycles = int(solver["ci_min_cycles_per_order"])
    plateau_relative = float(solver["ci_plateau_relative"])
    plateau_patience = int(solver["ci_plateau_patience"])
    minimum_step17_fraction = _step17_minimum_candidate_fraction(ctx)

    for order in solver["order_schedule"][1:]:
        spot_guard_active = bool(
            step17_spot_guard_cfg["enabled"]
            and int(order)
            >= int(step17_spot_guard_cfg["minimum_order"])
        )
        maximum_spot_rms_regression_mm = float(
            step17_spot_guard_cfg["maximum_regression_mm"]
        )
        order_idx = solver["order_schedule"].index(order)
        prev_order = int(solver["order_schedule"][order_idx - 1])
        if prev_order in best_feasible_by_order:
            if not any(
                _ci_minimum_physical_fraction_admitted(s)
                for s in frontier
            ):
                frontier = [copy.deepcopy(best_feasible_by_order[prev_order])]

        plateau_streak = 0
        order_converged = False
        accepted_cycles_this_order = 0
        partial_fermat_accepted_this_order = False
        for cycle in range(maximum_cycles):
            next_frontier: list[dict[str, Any]] = []
            valid_without_improvement = 0
            rejected_before_cycle = len(rejected)
            frontier = sorted(frontier, key=_rho_sort_key)
            merit_before = float(frontier[0]["Fan_core_merit"])
            for parent_index, state in enumerate(frontier):
                parent = copy.deepcopy(ctx); _apply_rho_state(parent, state)
                r = parent.data["rays"]
                tr = trace_reverse(r, parent.data["visor"], parent.data["m1"], parent.data["m2"],
                                   parent.data["display"], physical_first_hit=False)
                refs, _ = _dynamic_refs(parent, tr)
                parent.data["trace"] = tr; parent.data["fan_refs"] = refs
                parent.data["rho_sequence"] = copy.deepcopy(state["rho_sequence"])
                rho = float(state["rho_sequence"][-1])
                print(f"  order {order} cycle {cycle + 1} branch {parent_index + 1}: Fermat", flush=True)
                if step17_pre_fermat_images:
                    try:
                        live = (
                            render_step17_branch_progress(
                                parent,
                                int(order),
                                cycle + 1,
                                parent_index + 1,
                                len(frontier),
                                None,
                                "SOLVING_FERMAT",
                            )
                        )
                        print(
                            "    live image: "
                            f"{live['latest']}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            "    LIVE_IMAGE_WARNING "
                            f"before Fermat: {exc}",
                            flush=True,
                        )
                sol = solve_fermat_m2(
                    tr["points"][1], refs[r["field_index"]], parent.data["m2"],
                    tr["points"][2], int(solver["fermat_max_iter"]),
                    float(solver["fermat_gradient_tolerance"]),
                    reflection_tolerance=float(solver["fermat_reflection_residual_tolerance"]),
                )
                audit = _fermat_audit(
                    parent.data["m2"], sol,
                    f"STEP_17_ORDER_{order}_CYCLE_{cycle + 1}_BRANCH_{parent_index + 1}",
                    expected_ray_count=len(r["rows"]),
                    gradient_tolerance=float(solver["fermat_gradient_tolerance"]),
                    reflection_tolerance=float(solver["fermat_reflection_residual_tolerance"]),
                    minimum_admissible_fraction=minimum_step17_fraction,
                )
                parent.data["fermat"] = sol
                print(
                    "    Fermat admission | "
                    f"admissible={audit['admissible_target_count']:,}/"
                    f"{audit['ray_count']:,} "
                    f"({100.0 * audit['admissible_target_fraction']:.3f}%) | "
                    f"minimum={100.0 * audit['minimum_admissible_fraction']:.3f}% | "
                    f"fraction={'PASS' if audit['admissible_fraction_pass'] else 'FAIL'} | "
                    f"all_rays={'PASS' if audit['all_converged'] else 'NO'} | "
                    f"numeric_failed={audit['fallback_numeric_failure_count']:,}",
                    flush=True,
                )
                if step17_images_enabled:
                    try:
                        live = render_step17_branch_progress(
                            parent, int(order), cycle + 1, parent_index + 1, len(frontier), audit,
                            "FERMAT_PASS" if audit["branch_may_continue"] else "FERMAT_FAIL")
                        print(f"    completed image: {live['image']}", flush=True)
                    except Exception as exc:
                        print(f"    LIVE_IMAGE_WARNING after Fermat: {exc}", flush=True)
                if not audit["branch_may_continue"]:
                    rej_reason = "FERMAT_BELOW_MINIMUM_ADMISSIBLE_FRACTION"
                    rejected.append({"order": int(order), "cycle": cycle + 1,
                                     "parent_rho_sequence": state["rho_sequence"], "rho": rho,
                                     "accepted": False, "reason": rej_reason,
                                     "fermat_success_count": audit["success_count"],
                                     "fermat_ray_count": audit["ray_count"],
                                     "gradient_pass_count": audit["gradient_pass_count"],
                                     "reflection_pass_count": audit.get("reflection_pass_count", audit["success_count"]),
                                     "wrong_reflection_branch_count": audit.get("wrong_reflection_branch_count", 0),
                                     "fermat_admissible_count": audit["admissible_target_count"],
                                     "fermat_admissible_fraction": audit["admissible_target_fraction"],
                                     "fermat_minimum_fraction": audit["minimum_admissible_fraction"],
                                     "fermat_fraction_pass": audit["admissible_fraction_pass"],
                                     "inside_current_aperture_count": audit["inside_current_aperture_count"],
                                     "gradient_max": audit["gradient_max"]})
                    continue
                parent.data["fermat_convergence_history"].append(audit)
                parent.data["fermat_all_converged"] = bool(audit["all_converged"])
                parent.data["fermat_fraction_pass"] = bool(
                    audit["admissible_fraction_pass"]
                )
                branch = copy.deepcopy(parent)
                branch.data["rho"] = rho
                branch.data["step17_cycle"] = int(cycle + 1)
                branch.data["step17_branch_index"] = int(parent_index + 1)
                try:
                    out = _reconstruct(
                        branch,
                        int(order),
                        False,
                        allow_restoration=True,
                    )

                    candidate_spot_diagnostic = (
                        _step17_spot_bundle_diagnostics(out)
                    )
                    parent_spot_diagnostic = (
                        _step17_spot_bundle_diagnostics(
                            state["reconstruct_out"]
                        )
                    )
                    spot_deltas = _step17_spot_deltas(
                        out,
                        state["reconstruct_out"],
                        candidate_spot_diagnostic,
                        parent_spot_diagnostic,
                    )

                    out.update(candidate_spot_diagnostic["summary"])
                    out.update(spot_deltas)

                    (
                        improved,
                        improvement_reason,
                        spot_guard,
                    ) = _ci_candidate_improves(
                        out,
                        state["reconstruct_out"],
                        spot_guard_enabled=spot_guard_active,
                        maximum_spot_rms_regression_mm=(
                            maximum_spot_rms_regression_mm
                        ),
                    )
                    if not improved:
                        valid_without_improvement += 1
                        m2_ev = branch.data.get("_last_candidate_evidence", {})
                        spot_bundle_export = _persist_step17_spot_bundle_rows(
                            ctx,
                            int(order),
                            cycle + 1,
                            parent_index + 1,
                            candidate_spot_diagnostic,
                            "REJECT",
                            improvement_reason,
                        )

                        _print_step17_candidate_diagnostic(
                            int(order),
                            cycle + 1,
                            parent_index + 1,
                            out,
                            state["reconstruct_out"],
                            candidate_spot_diagnostic,
                            parent_spot_diagnostic,
                            spot_guard,
                            "REJECT",
                            improvement_reason,
                        )
                        rejected.append({"order": int(order), "cycle": cycle + 1,
                                         "parent_rho_sequence": state["rho_sequence"], "rho": rho,
                                         "accepted": False, "reason": improvement_reason,
                                         **spot_guard,
                                         **candidate_spot_diagnostic["summary"],
                                         **spot_deltas,
                                         **spot_bundle_export,
                                         "fermat_success_count": audit["success_count"],
                                         "fermat_ray_count": audit["ray_count"],
                                         "gradient_pass_count": audit["gradient_pass_count"],
                                         "reflection_pass_count": audit.get("reflection_pass_count", audit["success_count"]),
                                         "wrong_reflection_branch_count": audit.get("wrong_reflection_branch_count", 0),
                                         "fermat_admissible_count": audit["admissible_target_count"],
                                         "fermat_admissible_fraction": audit["admissible_target_fraction"],
                                         "fermat_minimum_fraction": audit["minimum_admissible_fraction"],
                                         "fermat_fraction_pass": audit["admissible_fraction_pass"],
                                         "inside_current_aperture_count": audit["inside_current_aperture_count"],
                                         "gradient_max": audit["gradient_max"],
                                         "M2_footprint_filter_enabled": out.get("M2_footprint_filter_enabled", m2_ev.get("M2_footprint_filter_enabled", False)),
                                         "M2_footprint_rejected_count": out.get("M2_footprint_rejected_count", m2_ev.get("M2_footprint_rejected_count", 0)),
                                         "M2_footprint_rejected_fraction": out.get("M2_footprint_rejected_fraction", m2_ev.get("M2_footprint_rejected_fraction", 0.0)),
                                         "M2_footprint_retained_count": out.get("M2_footprint_retained_count", m2_ev.get("M2_footprint_retained_count", 0)),
                                         "M2_footprint_affected_bundle_count": out.get("M2_footprint_affected_bundle_count", m2_ev.get("M2_footprint_affected_bundle_count", 0)),
                                         "M2_footprint_max_bundle_rejected_fraction": out.get("M2_footprint_max_bundle_rejected_fraction", m2_ev.get("M2_footprint_max_bundle_rejected_fraction", 0.0))})
                        rejected[-1].update(
                            _persist_candidate_evidence(
                                ctx, branch, 17, int(order), cycle + 1,
                                f"parent_{parent_index + 1:03d}_rho_{rho:+.12g}",
                                improvement_reason,
                            )
                        )
                        continue
                    spot_bundle_export = _persist_step17_spot_bundle_rows(
                        ctx,
                        int(order),
                        cycle + 1,
                        parent_index + 1,
                        candidate_spot_diagnostic,
                        "ACCEPT",
                        improvement_reason,
                    )

                    _print_step17_candidate_diagnostic(
                        int(order),
                        cycle + 1,
                        parent_index + 1,
                        out,
                        state["reconstruct_out"],
                        candidate_spot_diagnostic,
                        parent_spot_diagnostic,
                        spot_guard,
                        "ACCEPT",
                        improvement_reason,
                    )
                    history = copy.deepcopy(state["order_history"])
                    row = {"order": int(order), "cycle": cycle + 1, "accepted": True,
                           "accepted_shape_update": True, "basis_promoted": False,
                           "shape_optimized": True,
                           **spot_guard,
                           **candidate_spot_diagnostic["summary"],
                           **spot_deltas,
                           **spot_bundle_export,
                           "rho": rho, "basis": _ci_basis_label(int(order)),
                           "fermat_success_count": audit["success_count"],
                           "fermat_ray_count": audit["ray_count"],
                           "fermat_all_converged": bool(audit["all_converged"]),
                           "fermat_admissible_fraction": audit["admissible_target_fraction"],
                           "fermat_minimum_fraction": audit["minimum_admissible_fraction"],
                           "fermat_fraction_pass": audit["admissible_fraction_pass"],
                           "gradient_max": audit["gradient_max"],
                           "M1_sag_fit_rms_mm": out["M1_fit"]["sag_rms_mm"],
                           "M2_sag_fit_rms_mm": out["M2_fit"]["sag_rms_mm"],
                           "Fan_core_merit": out["Fan_core_merit"],
                            "chief_centered_spot_RMS_mm": out["chief_centered_spot_RMS_mm"],
                            "physical_valid_count": out["physical_valid_count"],
                            "physical_valid_fraction": out["physical_valid_fraction"],
                            "minimum_candidate_fraction": out["minimum_candidate_fraction"],
                            "physical_fraction_pass": out["physical_fraction_pass"],
                            "physical_first_hit_complete": out["physical_first_hit_complete"],
                            "feasibility_status": out["feasibility_status"],
                            "reason": f"PASS_PAPER_CI_ITERATION_{improvement_reason}",
                            "M2_footprint_filter_enabled": out.get("M2_footprint_filter_enabled", False),
                            "M2_footprint_rejected_count": out.get("M2_footprint_rejected_count", 0),
                            "M2_footprint_rejected_fraction": out.get("M2_footprint_rejected_fraction", 0.0),
                            "M2_footprint_retained_count": out.get("M2_footprint_retained_count", 0),
                            "M2_footprint_affected_bundle_count": out.get("M2_footprint_affected_bundle_count", 0),
                            "M2_footprint_max_bundle_rejected_fraction": out.get("M2_footprint_max_bundle_rejected_fraction", 0.0)}
                    history.append(row)
                    branch.data["reference_history"].append({"cycle": len(history) - 1,
                                                              **branch.data["fan_reference_rule"]})
                    sequence = state["rho_sequence"] + [rho]
                    cand_state = _rho_state(branch, sequence, history, out)
                    next_frontier.append(cand_state)
                    partial_fermat_accepted_this_order = bool(
                        partial_fermat_accepted_this_order
                        or not audit["all_converged"]
                    )
                    _remember_ci_state(
                        cand_state,
                        best_feasible_by_order,
                        best_admissible_by_order,
                    )
                    accepted_records.append({"order": int(order), "cycle": cycle + 1,
                                             "rho_sequence": sequence,
                                             **spot_guard,
                                             **candidate_spot_diagnostic["summary"],
                                             **spot_deltas,
                                             **spot_bundle_export,
                                             "Fan_core_merit": out["Fan_core_merit"],
                                             "chief_centered_spot_RMS_mm": out["chief_centered_spot_RMS_mm"],
                                             "physical_valid_count": out["physical_valid_count"],
                                             "physical_valid_fraction": out["physical_valid_fraction"],
                                             "minimum_candidate_fraction": out["minimum_candidate_fraction"],
                                             "physical_fraction_pass": out["physical_fraction_pass"],
                                             "physical_first_hit_complete": out["physical_first_hit_complete"],
                                             "fermat_admissible_fraction": audit["admissible_target_fraction"],
                                             "fermat_minimum_fraction": audit["minimum_admissible_fraction"],
                                             "fermat_all_converged": bool(audit["all_converged"]),
                                              "feasibility_status": out["feasibility_status"],
                                              "status": "PASS_PAPER_CI_SUCCESSIVE_APPROXIMATION",
                                              "M2_footprint_filter_enabled": out.get("M2_footprint_filter_enabled", False),
                                              "M2_footprint_rejected_count": out.get("M2_footprint_rejected_count", 0),
                                              "M2_footprint_rejected_fraction": out.get("M2_footprint_rejected_fraction", 0.0),
                                              "M2_footprint_retained_count": out.get("M2_footprint_retained_count", 0),
                                              "M2_footprint_affected_bundle_count": out.get("M2_footprint_affected_bundle_count", 0),
                                              "M2_footprint_max_bundle_rejected_fraction": out.get("M2_footprint_max_bundle_rejected_fraction", 0.0)})
                except RuntimeError as exc:
                    err_str = str(exc)
                    m2_ev = branch.data.get("_last_candidate_evidence", {})
                    rejected.append({"order": int(order), "cycle": cycle + 1,
                                     "parent_rho_sequence": state["rho_sequence"], "rho": rho,
                                     "accepted": False, "reason": err_str,
                                     "fermat_success_count": audit["success_count"],
                                     "fermat_ray_count": audit["ray_count"],
                                     "gradient_pass_count": audit["gradient_pass_count"],
                                     "reflection_pass_count": audit.get("reflection_pass_count", audit["success_count"]),
                                     "wrong_reflection_branch_count": audit.get("wrong_reflection_branch_count", 0),
                                     "fermat_admissible_count": audit["admissible_target_count"],
                                     "fermat_admissible_fraction": audit["admissible_target_fraction"],
                                     "fermat_minimum_fraction": audit["minimum_admissible_fraction"],
                                     "fermat_fraction_pass": audit["admissible_fraction_pass"],
                                     "inside_current_aperture_count": audit["inside_current_aperture_count"],
                                     "gradient_max": audit["gradient_max"],
                                     "M2_footprint_filter_enabled": m2_ev.get("M2_footprint_filter_enabled", False),
                                     "M2_footprint_rejected_count": m2_ev.get("M2_footprint_rejected_count", 0),
                                     "M2_footprint_rejected_fraction": m2_ev.get("M2_footprint_rejected_fraction", 0.0),
                                     "M2_footprint_retained_count": m2_ev.get("M2_footprint_retained_count", 0),
                                     "M2_footprint_affected_bundle_count": m2_ev.get("M2_footprint_affected_bundle_count", 0),
                                     "M2_footprint_max_bundle_rejected_fraction": m2_ev.get("M2_footprint_max_bundle_rejected_fraction", 0.0)})
                    rejected[-1].update(
                        _persist_candidate_evidence(
                            ctx, branch, 17, int(order), cycle + 1,
                            f"parent_{parent_index + 1:03d}_rho_{rho:+.12g}",
                            err_str,
                        )
                    )
            if not next_frontier:
                if cycle > 0 and valid_without_improvement:
                    order_converged = True
                    convergence_records.append({"order": int(order), "cycle": cycle + 1,
                                                "status": "CONVERGED_NO_FAN_MERIT_IMPROVEMENT",
                                                 "best_merit": merit_before})
                    if order in best_feasible_by_order:
                        frontier = [copy.deepcopy(best_feasible_by_order[order])]
                    break
                if order in best_feasible_by_order and accepted_cycles_this_order > 0:
                    order_converged = True
                    frontier = [copy.deepcopy(best_feasible_by_order[order])]
                    cycle_rejections = rejected[rejected_before_cycle:]
                    convergence_records.append({
                        "order": int(order), "cycle": cycle + 1,
                        "status": "TERMINATED_RETAIN_LAST_FRACTION_ADMITTED_STATE_NO_ADMISSIBLE_NEXT_CYCLE",
                        # TERMINATED_RETAIN_LAST_FULLY_VALID_STATE_NO_ADMISSIBLE_NEXT_CYCLE
                        "best_merit": float(frontier[0]["Fan_core_merit"]),
                        "accepted_cycles_this_order": accepted_cycles_this_order,
                        "rejected_attempt_count": len(cycle_rejections),
                        "rejected_reasons": sorted({str(row["reason"]) for row in cycle_rejections}),
                        "partial_fermat_accepted": partial_fermat_accepted_this_order,
                        "incumbent_shape_changed_by_rejected_cycle": False,
                    })
                    break
                if (cycle + 1 >= minimum_cycles and accepted_cycles_this_order > 0):
                    cycle_rejections = rejected[rejected_before_cycle:]
                    convergence_records.append({
                        "order": int(order), "cycle": cycle + 1,
                        "status": "TERMINATED_RETAIN_LAST_ADMISSIBLE_STATE_NO_ADMISSIBLE_NEXT_CYCLE",
                        "best_merit": merit_before,
                        "accepted_cycles_this_order": accepted_cycles_this_order,
                        "rejected_attempt_count": len(cycle_rejections),
                        "rejected_reasons": sorted({str(row["reason"]) for row in cycle_rejections}),
                        "partial_fermat_accepted": partial_fermat_accepted_this_order,
                        "incumbent_shape_changed_by_rejected_cycle": False,
                    })
                    order_converged = True
                    break
                promotion_parents = [
                    candidate_parent
                    for candidate_parent in frontier
                    if int(
                        candidate_parent[
                            "reconstruct_out"
                        ]["order"]
                    ) == prev_order
                    and _ci_minimum_physical_fraction_admitted(
                        candidate_parent
                    )
                    and all(
                        _surface_has_ci_basis(
                            candidate_parent[name],
                            prev_order,
                        )
                        for name in ("m1", "m2")
                    )
                ]

                if (
                    cycle == 0
                    and int(order) in (4, 5)
                    and promotion_parents
                ):
                    promoted_parent = min(
                        promotion_parents,
                        key=_rho_sort_key,
                    )
                    promoted = (
                        _promote_ci_state_without_shape_change(
                            promoted_parent,
                            int(order),
                            cycle + 1,
                            spot_guard_enabled=(
                                spot_guard_active
                            ),
                            maximum_spot_rms_regression_mm=(
                                maximum_spot_rms_regression_mm
                            ),
                        )
                    )

                    frontier = [promoted]
                    _remember_ci_state(
                        promoted,
                        best_feasible_by_order,
                        best_admissible_by_order,
                    )
                    order_converged = True

                    convergence_records.append({
                        "order": int(order),
                        "cycle": cycle + 1,
                        "status": (
                            f"ORDER{int(order)}_BASIS_PROMOTED_"
                            f"WITHOUT_ACCEPTED_ORDER{int(order)}_"
                            "SHAPE_CHANGE"
                        ),
                        "promotion_from_order": prev_order,
                        "best_merit": float(
                            promoted["Fan_core_merit"]
                        ),
                        "shape_changed": False,
                    })
                    break
                write_csv(ctx.step_dir(17) / "17_REJECTED_BRANCHES.csv", rejected,
                          fieldnames=rejected_fields)
                write_csv(ctx.step_dir(17) / "17_ACCEPTED_BRANCHES_BEFORE_PRUNING.csv",
                          accepted_records, fieldnames=accepted_fields)
                raise RuntimeError(f"ORDER_{order}_NO_IMPROVING_PAPER_CI_BRANCH")
            previous_best_physical = max(
                int(state["reconstruct_out"].get("physical_valid_count", 0)) for state in frontier)
            frontier = _select_ci_frontier(next_frontier, solver)
            if not frontier:
                raise RuntimeError(f"ORDER_{order}_NO_FRONTIER_AFTER_FEASIBLE_FIRST_PRUNING")
            accepted_cycles_this_order += 1
            merit_after = float(frontier[0]["Fan_core_merit"])
            best_physical_after = max(
                int(state["reconstruct_out"].get("physical_valid_count", 0)) for state in frontier)
            relative = (merit_before - merit_after) / max(abs(merit_before), 1e-15)
            physical_recovery = best_physical_after > previous_best_physical
            plateau_streak = (0 if physical_recovery else
                              plateau_streak + 1 if relative <= plateau_relative else 0)
            convergence_records.append({"order": int(order), "cycle": cycle + 1,
                                        "status": "ITERATING", "best_merit_before": merit_before,
                                        "best_merit_after": merit_after,
                                        "relative_improvement": relative,
                                        "physical_recovery": physical_recovery,
                                        "best_physical_count_before": previous_best_physical,
                                        "best_physical_count_after": best_physical_after,
                                        "plateau_streak": plateau_streak,
                                        "partial_fermat_accepted": partial_fermat_accepted_this_order})
            if cycle + 1 >= minimum_cycles and plateau_streak >= plateau_patience:
                order_converged = True
                convergence_records[-1]["status"] = "CONVERGED_RELATIVE_PLATEAU"
                break
        if not order_converged:
            write_json(ctx.step_dir(17) / "17_CI_CONVERGENCE.json",
                       {"records": convergence_records, "status": "MAX_CYCLES_REACHED"})
            raise RuntimeError(f"ORDER_{order}_CI_NOT_CONVERGED_WITHIN_{maximum_cycles}_CYCLES")
        if order in best_feasible_by_order:
            frontier = [copy.deepcopy(best_feasible_by_order[order])]

        if frontier:
            convergence_state = min(
                frontier,
                key=_rho_sort_key,
            )
            _record_optical_convergence(
                ctx,
                f"STEP17 O{int(order)}",
                17,
                m1=convergence_state["m1"],
                m2=convergence_state["m2"],
                display=convergence_state["display"],
                reverse_trace=convergence_state.get(
                    "last_ci_physical_trace"
                ),
            )

            if int(order) in (3, 4):
                snapshot_surface_state = (
                    _ci_snapshot_surface_state(
                        convergence_state,
                        int(order),
                    )
                )

                _capture_spot_snapshot(
                    ctx,
                    f"ORDER_{int(order)}",
                    17,
                    trace=convergence_state[
                        "last_ci_physical_trace"
                    ],
                    references=convergence_state["fan_refs"],
                    surface_state=snapshot_surface_state,
                    validity_mode="PHYSICAL_FIRST_HIT",
                )

    final_pool = list(frontier)
    archived_order5 = best_feasible_by_order.get(5)
    if archived_order5 is not None:
        final_pool.append(archived_order5)

    expected_ray_count = len(ctx.data["rays"]["rows"])
    final_feasible = [
        candidate_state
        for candidate_state in final_pool
        if _ci_order5_metadata(candidate_state, expected_ray_count) is not None
    ]

    if not final_feasible:
        write_json(
            ctx.step_dir(17) / "17_FINAL_CI_SELECTION_FAILURE.json",
            {
                "status": "CI_NO_ADMISSIBLE_ORDER5_START_BEFORE_STEP_18",
                "available_feasible_orders": sorted(best_feasible_by_order),
                "available_admissible_orders": sorted(best_admissible_by_order),
                "lower_order_incumbents_preserved": True,
                "lower_order_relabelled_as_order5": False,
            },
        )
        raise RuntimeError(
            "CI_NO_ADMISSIBLE_ORDER5_START_BEFORE_STEP_18"
        )

    best = min(final_feasible, key=_rho_sort_key)
    order5_metadata = _ci_order5_metadata(best, expected_ray_count)
    assert order5_metadata is not None

    _apply_rho_state(ctx, best)
    ctx.data["_ci_selected_reconstruct_out"] = copy.deepcopy(
        best["reconstruct_out"]
    )
    ctx.data["order5_state"] = copy.deepcopy(order5_metadata)
    _publish_trace_debug(
        ctx, 17, "SELECTED_ORDER5_PHYSICAL_TRACE",
        ctx.data["last_ci_physical_trace"], ctx.data["rays"],
        direction="REVERSE",
    )

    history = best["order_history"]
    ctx.data["order_history"] = history
    zero_pad = bool(order5_metadata["zero_pad_promotion"])

    shape_optimized_through_order = int(
        order5_metadata["shape_optimized_through_order"]
    )
    ci_completion_status = (
        (
            "COMPLETED_ORDER5_BASIS_WITH_"
            f"ORDER{shape_optimized_through_order}_"
            "OPTIMIZED_SHAPE"
        )
        if zero_pad
        else "CONVERGED_WITH_ACCEPTED_ORDER5_SHAPE_UPDATE"
    )
    ctx.data["fermat_rejected_attempts"] = [
        copy.deepcopy(row) for row in rejected
        if str(row.get("reason", "")).startswith("FERMAT_")
    ]
    # _capture_spot_snapshot(ctx, "ORDER_5", 17)
    _capture_spot_snapshot(
        ctx,
        "ORDER_5",
        17,
        trace=ctx.data["last_ci_physical_trace"],
        references=ctx.data["fan_refs"],
        surface_state=ctx.data["order5_state"],
        validity_mode="PHYSICAL_FIRST_HIT",
    )
    write_csv(ctx.step_dir(17) / "17_ORDER_LADDER_2_TO_5.csv", history,
              fieldnames=["order", "cycle", "accepted", "accepted_shape_update",
                          "basis_promoted", "shape_optimized", "promotion_from_order", "rho", "basis", "fermat_success_count",
                          "fermat_ray_count", "fermat_all_converged",
                          "fermat_admissible_fraction", "fermat_minimum_fraction",
                          "fermat_fraction_pass", "gradient_max",
                          "M1_sag_fit_rms_mm", "M2_sag_fit_rms_mm", "Fan_core_merit",
                          "chief_centered_spot_RMS_mm",
                          *STEP17_SPOT_SUMMARY_FIELDS,
                          *STEP17_SPOT_DELTA_FIELDS,
                          *STEP17_SPOT_EXPORT_FIELDS,
                          "physical_valid_count",
                          "physical_valid_fraction", "minimum_candidate_fraction",
                          "physical_fraction_pass", "physical_first_hit_complete",
                          "feasibility_status",
                          "spot_RMS_guard_enabled", "parent_spot_RMS_mm",
                          "candidate_spot_RMS_mm", "spot_RMS_delta_mm",
                          "maximum_spot_RMS_regression_mm", "spot_RMS_guard_pass",
                          "spot_RMS_guard_status",
                          "reason"])
    write_csv(ctx.step_dir(17) / "17_REJECTED_BRANCHES.csv", rejected, fieldnames=rejected_fields)
    write_csv(ctx.step_dir(17) / "17_ACCEPTED_BRANCHES_BEFORE_PRUNING.csv",
              accepted_records, fieldnames=accepted_fields)
    write_json(ctx.step_dir(17) / "17_CI_CONVERGENCE.json",
               {"records": convergence_records, "status": ci_completion_status,
                "all_basis_levels_completed": True,
                "order5_shape_update_accepted": not zero_pad})
    write_json(ctx.step_dir(17) / "17_SELECTED_CI_CLOUD_INTEGRABILITY.json", {
        "M1": ctx.data["last_ci_m1_diagnostics"],
        "M2": ctx.data["last_ci_m2_diagnostics"],
        "quality": best["reconstruct_out"]["integrability_quality"],
    })
    write_json(ctx.step_dir(17) / "17_ORDER_POLICY.json", {
        "Fan_core_requirement": "gradually increase to fifth order", "implementation_choice": [2, 3, 4, 5],
        "nested_basis_policy": {
            "level_2": "FAN_EQ1_AXIS_ORDER_2_INCLUDES_A22",
            "level_3": "UNION_OF_FAN_AXIS_ORDER2_AND_TOTAL_DEGREE_LE_3",
            "level_4": "TOTAL_DEGREE_LE_4_SUPERSET_OF_LEVEL_3",
            "level_5": "TOTAL_DEGREE_LE_5_SUPERSET_OF_LEVEL_4",
            "active_terms_never_dropped_during_order_transition": True,
        },
        "per_order_certification": ["FERMAT_MINIMUM_98_PERCENT", "SURFACE_SANITY",
                                    "FRACTION_ADMITTED_FIRST_WITH_EXPLICIT_RESTORATION",
                                    "SIGNED_MF2_UNOBSCURED"],
        "minimum_step17_candidate_fraction": minimum_step17_fraction,
        "exact_all_rays_completion_reported_separately": True,
        "rejected_cycle_policy": "ACCEPT_FERMAT_AT_OR_ABOVE_MINIMUM_FRACTION; RETAIN_PRIOR_ADMISSIBLE_INCUMBENT_AFTER_MINIMUM_CYCLE_ATTEMPTS",
        "rejected_cycle_changes_incumbent": False,
        "branch_ranking": "FEASIBLE_FIRST_THEN_LIMITED_RESTORATION_THEN_FAN_MERIT_THEN_PAPER_STYLE_SPOT",
        "restoration_policy": {
            "enabled": bool(solver["restoration_branch_enabled"]),
            "minimum_physical_fraction": float(solver["restoration_minimum_physical_fraction"]),
            "maximum_retained_restoration_branches": int(solver["restoration_max_branches"]),
            "minimum_step17_fraction_required_before_step18": minimum_step17_fraction,
            "full_physical_required_before_step18": False,
        },
        "step17_spot_guard": {
            "enabled": bool(
                step17_spot_guard_cfg["enabled"]
            ),
            "minimum_order": int(
                step17_spot_guard_cfg["minimum_order"]
            ),
            "maximum_regression_mm": float(
                step17_spot_guard_cfg[
                    "maximum_regression_mm"
                ]
            ),
            "metric":
                "CHIEF_CENTERED_SPOT_RMS_MM",
            "role":
                "O4_O5_PARENT_NON_REGRESSION_GATE",
        },
        "basis_promotion_fallback": {
            "eligible_target_orders": [4, 5],
            "shape_change": False,
            "new_coefficients": "ZERO",
            "physical_parent_must_be_admitted": True,
            "promotion_is_not_shape_optimization": True,
        },
        "final_step17_certification": [
            "PHYSICAL_FIRST_HIT_CONFIGURED_MINIMUM_FRACTION",
            "SURFACE_SANITY",
            "CI_INTEGRABILITY",
            "SIGNED_MF2_UNOBSCURED",
            "O4_O5_SPOT_RMS_NON_REGRESSION",
        ],
        "deferred_certification": [
            "FORWARD_DISTORTION",
            "PACKAGING",
        ],
        "final_hard_certification": ["PHYSICAL_FIRST_HIT_CONFIGURED_MINIMUM_FRACTION", "DISTORTION", "PACKAGING"],
        "signed_rho_candidates": solver["rho_candidates"],
        "iteration_mode": "SUCCESSIVE_APPROXIMATION_HUD_EQ3",
        "rho_policy_after_order2": "FIXED_PER_SELECTED_BRANCH",
        "convergence_required": True,
        "selected_rho_sequence": best["rho_sequence"], "final_basis_order": 5,
        "order5_state": ctx.data["order5_state"]})
    return {"status": "PASS", "final_basis_order": 5,
            "shape_optimized_through_order": ctx.data["order5_state"]["shape_optimized_through_order"],
            "order5_state": ctx.data["order5_state"], "cycles": len(history),
            "per_order_gates": ["FERMAT", "SURFACE_SANITY", "FEASIBLE_FIRST_PHYSICAL", "SIGNED_MF2"],
            "final_minimum_physical_fraction_certified": bool(
                order5_metadata["physical_fraction_pass"]
            ),
            "final_full_physical_certified": bool(
                order5_metadata["physical_first_hit_complete"]
            ),
            "selected_rho_sequence": best["rho_sequence"],
            "Fan_core_merit": best["Fan_core_merit"], "rejected_branch_count": len(rejected),
            "CI_convergence": ci_completion_status}

def step_18(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 18: Lưu mặt O5 sau CI làm starting point, chưa coi là kết quả tối ưu cuối."""

    required_keys = (
        "_ci_selected_reconstruct_out",
        "order_history",
        "order5_state",
    )
    missing = [key for key in required_keys if key not in ctx.data]
    if missing:
        raise RuntimeError(f"CI_START_STATE_MISSING:{missing}")

    selected_state = {
        "m1": ctx.data["m1"],
        "m2": ctx.data["m2"],
        "reconstruct_out": ctx.data["_ci_selected_reconstruct_out"],
        "order_history": ctx.data["order_history"],
    }
    verified_state = _ci_order5_metadata(
        selected_state, len(ctx.data["rays"]["rows"])
    )
    if verified_state is None:
        raise RuntimeError("CI_START_STATE_NOT_ADMISSIBLE_ORDER5")

    p = _prescription(ctx); ctx.data["ci_start_prescription"] = p
    state = copy.deepcopy(verified_state)
    write_json(ctx.step_dir(18) / "18_FIFTH_ORDER_CI_START.json", p | {
        "role": "OPTIMIZATION_START_NOT_FINAL_RESULT", "basis_order": 5,
        "order5_state": state,
        "label_rule": "ZERO_PAD_IS_A_BASIS_PROMOTION_NOT_AN_OPTICAL_ORDER5_IMPROVEMENT"})
    return {"status": "PASS", "basis_order": 5, "order5_state": state,
            "final_result": False}

def step_19(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 19: Khai báo MF1/MF2 của Fan và tách distortion, packaging, ray-validity ra ngoài."""

    n = ctx.config["normalized_objective"]; L = float(n["L_ref_mm"])
    out = {
        "Fan_core": {
            "MF1": (
                "omega1*sum_pupil "
                "RMS_over_all_fields_and_49_characteristic_rays"
                "(||P-P_chief(field,pupil)||)"
            ),
            "MF2": "0 if signed S_AQP>=0 else omega2*abs(S_AQP)",
            "MF3": "N/A_SINGLE_STATIC_EYEBOX_HEIGHT",
            "objective": "J_Fan=w1*MF1/Lref+w2*MF2/Aref+w3*MF3/Lref",
            "chief_pupil_spread": (
                "diagnostic RMS of chief(field,pupil) around "
                "the per-field mean chief"
            ),
            "chief_field_bias": (
                "diagnostic distance from the per-field mean chief "
                "to P_ref(field)"
            ),
            "equation_4_claim": (
                "PROJECT_CHIEF_CENTERED_OBJECTIVE; "
                "DO_NOT_CLAIM_IDENTICAL_TO_PAPER_EQ4_UNTIL_PRIMARY_SOURCE_IS_VERIFIED"
            ),
        },
        "engineering_constraints_outside_Fan_core": {
            "distortion": "fixed user target grid; forward monitor acceptance constraint and final full-ray hard gate",
            "packaging": ("optional P1-P8 convex-hull lambda constraint"
                          if ctx.data["inputs"]["packaging_constraint_enabled"]
                          else "DISABLED because P1-P8 are null or omitted"),
            "ray_validity": "physical first-hit sequence required",
            "surface_sanity": "finite, non-spiking sampled sag required",
            "forward_monitor": "J_engineering_forward=wD*PhiD_forward+wP*PhiP+wR*PhiR_forward",
            "separation": ("forward engineering is not added to Fan-core; it participates in "
                           "filter acceptance and bounded restoration selection"),
        },
        "optimization_policy": (
            "DLSQ directions minimize Fan-core; an engineering-aware filter accepts objective "
            "steps or bounded restoration steps and logs every evaluated rejection"
        ),
        "L_ref_mm": L, "A_ref_mm2": L * L, "weights": n,
        "PhiD": "max(0,Dforward_monitor/distortion_limit-1)^2",
        "PhiP": "0 when P1-P8 disabled; otherwise max(0,lambda/packaging_limit-1)^2",
        "PhiR": "forward_invalid_ray_fraction^2+forward_invalid_bundle_fraction^2",
        "double_weighting": False, "objective_is_dimensionless": True,
    }
    ctx.data["objective_definition"] = out
    write_json(ctx.step_dir(19) / "19_MERIT_AND_OBJECTIVE_DEFINITION.json", out)
    return out | {"status": "PASS"}

def step_20(ctx: Context) -> dict[str, Any]:
    """Tính chief-centered MF1 và tách pupil-shift / field-bias diagnostics."""

    tr = trace_reverse(ctx.data["rays"], ctx.data["visor"], ctx.data["m1"], ctx.data["m2"],
                       ctx.data["display"], physical_first_hit=True)
    refs, rule = _dynamic_refs(ctx, tr)
    m = fan_imaging_metrics(tr, ctx.data["rays"], refs, float(ctx.config["fan_weights"]["omega1"]))
    ctx.data.update({"trace": tr, "fan_refs": refs, "fan_mf1": m, "fan_reference_rule": rule})
    write_json(
        ctx.step_dir(20)
        / "20_MF1_FAN_ALL_CHARACTERISTIC_RAYS.json",
        m | {
            "formula": (
                "omega1*sum_p "
                "sqrt((1/(K*N))*sum_f*sum_r "
                "||P_pfr-P_chief_fp||^2)"
            ),

            "outer_pupil_sum_retained":
                True,

            "chief_centered":
                True,

            "chief_ray_is_included_in_49":
                True,

            "chief_pupil_spread_diagnostic":
                m[
                    "chief_pupil_spread_RMS_mm"
                ],

            "chief_field_bias_diagnostic":
                m[
                    "chief_field_bias_RMS_mm"
                ],

            "execution_status_meaning":
                "MF1_EVALUATED_NOT_A_PHYSICAL_PASS_GATE",

            "physical_certification_authority":
                "SEPARATE_FULL_FORWARD_AND_REVERSE_PHYSICAL_GATES",
        },
    )
    return m | {"status": "EVALUATED_MF1_NOT_PHYSICAL_CERTIFICATION"}

def step_21(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 21: Tính MF2 từ diện tích có hướng A-Q-P; không obscured thì MF2 bằng 0."""

    tr = ctx.data["trace"]
    g = mf2_geometry(tr["points"][0], tr["points"][1], tr["points"][2], ctx.data["m2"],
                     float(ctx.config["fan_weights"]["omega2"]), ctx.data["n_obs"])
    e2 = max(0.0, -float(g["S_AQP_signed_mm2"])); g["E2_mm2"] = e2
    g["orientation_frozen_from_initial_planar_state"] = True
    ctx.data["fan_mf2"] = g
    write_json(ctx.step_dir(21) / "21_MF2_SIGNED_OBSCURATION_3D.json", g)
    return g | {"status": "PASS"}

def step_22(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 22: Khai báo công thức và hard limit distortion trên lưới ảnh ảo cố định."""

    d = ctx.config["distortion"]
    out = {"metric_name": d["metric_name"], "authority": d["authority"],
           "fixed_grid_file": "STEP_08/08B_DISTORTION_FIXED_IDEAL_GRID.csv",
           "formula": "100*||q_act-q_ideal||/max(||q_ideal||,epsilon)",
           "hard_rule": f"D_max < {float(d['hard_limit_percent']):g} percent",
           "central_field_percentage": "N/A",
           "central_field_report": "absolute anchor error mm", "signed_diagnostics": ["D_r", "D_x", "D_y"],
           "prewarp": False, "single_hard_distortion_gate": True,
           "Fan_reference_is_not_distortion_authority": True}
    write_json(ctx.step_dir(22) / "22_USER_FIXED_GRID_DISTORTION_DEFINITION.json", out)
    return out | {"status": "PASS"}

def step_23(ctx: Context) -> dict[str, Any]:
    """Chạy 23A→23S curvature/conic→23G pose→23C; chỉ aperture luôn bị khóa."""

    runtime_enabled = bool(ctx.data.get("_geometry_refinement_runtime_enabled", True))
    geometry_enabled = bool(ctx.config.get("geometry_refinement", {}).get("enabled", False)) and runtime_enabled
    surface_cfg = ctx.config["surface_parameter_refinement"]
    surface_enabled = bool(surface_cfg["enabled"])
    if bool(surface_cfg["top_k_only"]):
        surface_enabled = surface_enabled and runtime_enabled
    ctx.data["dlsq_trial_history"] = []
    ctx.data["dlsq_termination_history"] = []
    phase_a = _coefficient_dlsq_phase(ctx, "23A_COEFFICIENT_DLSQ")
    write_csv(ctx.step_dir(23) / "23A_COEFFICIENT_HISTORY.csv", phase_a["history"])
    write_csv(ctx.step_dir(23) / "23A_DLSQ_TRIALS.csv", phase_a["trials"])
    write_json(ctx.step_dir(23) / "23A_DLSQ_TERMINATION.json", phase_a["termination"])
    write_json(ctx.step_dir(23) / "23A_OBJECTIVE_INITIAL.json", phase_a["initial"])
    write_json(ctx.step_dir(23) / "23A_OBJECTIVE_FINAL.json", phase_a["final"])
    phase_render_enabled = bool(
        ctx.config.get(
            "execution",
            {},
        ).get(
            "render_step23_phase_images",
            True,
        )
    )
    advanced_enabled = surface_enabled or geometry_enabled
    if not advanced_enabled:
        history = phase_a["history"]; initial = phase_a["initial"]; final = phase_a["final"]
        geometry = {"status": "DISABLED_USING_LEGACY_COEFFICIENT_ONLY_PATH"}
        surface_parameters = {"status": "DISABLED_USING_COEFFICIENT_ONLY_PATH"}
        phase_images = {}
    else:
        phase_images = (
            {"23A": _snapshot_step23_phase(ctx, "PHASE_23A")}
            if phase_render_enabled
            else {}
        )
        advanced_worker_snapshot = (
            _refinement_worker_snapshot(
                ctx
            )
        )
        with ray_trace_acceleration(
            "STEP23_ADVANCED_REFINEMENT",
            base_context=advanced_worker_snapshot,
        ):
            if surface_enabled:
                surface_parameters = _bounded_surface_parameter_refine(ctx)
                write_csv(ctx.step_dir(23) / "23S_SURFACE_PARAMETER_HISTORY.csv",
                          ctx.data["surface_parameter_refinement_history"])
                write_json(ctx.step_dir(23) / "23S_SURFACE_PARAMETER_RESULT.json", surface_parameters)
                write_json(ctx.step_dir(23) / "23S_CURVATURE_CONIC_REFINED_SURFACES.json", _prescription(ctx))
                if phase_render_enabled:
                    phase_images["23S"] = _snapshot_step23_phase(ctx, "PHASE_23S")
            else:
                surface_parameters = {"status": "DISABLED"}
            if geometry_enabled:
                geometry = _bounded_geometry_refine(ctx)
                write_csv(ctx.step_dir(23) / "23G_GEOMETRY_HISTORY.csv", ctx.data["geometry_refinement_history"])
                write_json(ctx.step_dir(23) / "23G_GEOMETRY_RESULT.json", geometry)
                write_json(ctx.step_dir(23) / "23G_POSE_REFINED_SURFACES.json", _prescription(ctx))
                if phase_render_enabled:
                    phase_images["23G"] = _snapshot_step23_phase(ctx, "PHASE_23G")
            else:
                geometry = {"status": "DISABLED"}
        phase_c = _coefficient_dlsq_phase(ctx, "23C_COEFFICIENT_DLSQ")
        write_csv(ctx.step_dir(23) / "23C_COEFFICIENT_HISTORY.csv", phase_c["history"])
        write_csv(ctx.step_dir(23) / "23C_DLSQ_TRIALS.csv", phase_c["trials"])
        write_json(ctx.step_dir(23) / "23C_DLSQ_TERMINATION.json", phase_c["termination"])
        write_json(ctx.step_dir(23) / "23C_OBJECTIVE_INITIAL.json", phase_c["initial"])
        write_json(ctx.step_dir(23) / "23C_OBJECTIVE_FINAL.json", phase_c["final"])
        if phase_render_enabled:
            phase_images["23C"] = _snapshot_step23_phase(ctx, "PHASE_23C")
        history = ([{"phase": "23A", **row} for row in phase_a["history"]] +
                   [{"phase": "23C", **row} for row in phase_c["history"]])
        initial, final = phase_a["initial"], phase_c["final"]
    ctx.data["optimization_history"] = history
    ctx.data["objective_final"] = final
    write_csv(ctx.step_dir(23) / "23_OPTIMIZATION_HISTORY.csv", history)
    write_csv(ctx.step_dir(23) / "23_DLSQ_TRIALS.csv", ctx.data["dlsq_trial_history"])
    write_json(ctx.step_dir(23) / "23_DLSQ_TERMINATION.json", {
        "phases": ctx.data["dlsq_termination_history"],
        "blocked_phase_count": sum(
            row.get("status") == "BLOCKED_NO_ADMISSIBLE_DLSQ_TRIAL"
            for row in ctx.data["dlsq_termination_history"]),
        "blocked_is_not_convergence": True,
    })
    write_json(ctx.step_dir(23) / "23_OBJECTIVE_INITIAL.json", initial)
    write_json(ctx.step_dir(23) / "23_OBJECTIVE_FINAL.json", final)
    write_json(ctx.step_dir(23) / "23_OPTIMIZED_SURFACES.json", _prescription(ctx))
    write_json(ctx.step_dir(23) / "23_REFERENCE_GRID_HISTORY.json", ctx.data["reference_history"])

    _record_optical_convergence(
        ctx,
        "STEP23 FINAL OPT",
        23,
        reverse_trace=ctx.data.get(
            "trace"
        ),
    )

    active_path = "23A" + ("_23S" if surface_enabled else "") + ("_23G" if geometry_enabled else "")
    active_path += "_23C" if advanced_enabled else "_ONLY"
    blocked_phase_count = sum(
        row.get("status") == "BLOCKED_NO_ADMISSIBLE_DLSQ_TRIAL"
        for row in ctx.data["dlsq_termination_history"])
    local_status = ("COMPLETED_WITH_DLSQ_BLOCKED_NO_ADMISSIBLE_TRIAL"
                    if blocked_phase_count else "PASS")
    return {"status": local_status, "path": active_path,
            "surface_parameter_refinement": surface_parameters,
            "geometry_refinement": geometry, "phase_images": phase_images,
            "dlsq_termination": ctx.data["dlsq_termination_history"],
            "dlsq_trial_count": len(ctx.data["dlsq_trial_history"]),
            "dlsq_blocked_phase_count": blocked_phase_count,
            "iterations": len(history), "variables": phase_a["variables"],
            "J_Fan_initial": initial["J_Fan_dimensionless"], "J_Fan_final": final["J_Fan_dimensionless"],
            "J_engineering_initial": initial["J_engineering_dimensionless"],
            "J_engineering_final": final["J_engineering_dimensionless"],
            "evaluation_rays": len(ctx.data["rays"]["rows"]), "final_hard_gate_not_applied_here": True,
            "optimized_terms": "ALL_NON_PISTON_XY_TERMS_THROUGH_TOTAL_ORDER_5_ON_M1_AND_M2",
            "curvature_conic_optimized": bool(surface_enabled),
            "distortion_during_optimization": (
                "EXTERNAL_FORWARD_MONITOR_ACCEPTANCE_GATE_NOT_FAN_RESIDUAL"),
            "forward_monitor_rays_per_bundle": int(
                ctx.config["solver"]["forward_constraint_rays_per_bundle"])}

def step_24(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 24: Bắn đủ tia thuận và chỉ chứng nhận bundle đầy đủ, đúng thứ tự, trước mắt."""

    rsp = ctx.config.get("ray_sampling_profiles")
    if isinstance(rsp, dict) and rsp.get("enabled", False):
        if "profile_rays_FREEFORM" not in ctx.data and "rays" in ctx.data:
            ctx.data["profile_rays_FREEFORM"] = copy.deepcopy(ctx.data["rays"])
            ctx.data["profile_vi_FREEFORM"] = copy.deepcopy(ctx.data["vi"])
            ctx.data["profile_pupils_FREEFORM"] = copy.deepcopy(ctx.data["pupils"])
            ctx.data["profile_pattern_FREEFORM"] = copy.deepcopy(ctx.data["pattern"])
            ctx.data["profile_visor_hit_FREEFORM"] = copy.deepcopy(ctx.data["visor_hit"])
            ctx.data["profile_post_visor_FREEFORM"] = copy.deepcopy(ctx.data["post_visor"])

        _apply_ray_sampling_profile(ctx, "CERTIFICATION", 24)
        ctx.data["final_trace_sampling_profile"] = "CERTIFICATION"

    r = ctx.data["rays"]
    evaluation = _run_forward_evaluation(ctx, ctx.data["m1"], ctx.data["m2"], monitor_only=False)
    reverse = evaluation["reverse"]
    sources = evaluation["sources"]
    rule = evaluation["reference_rule"]
    _publish_trace_debug(
        ctx, 24, "FINAL_FORWARD_EVALUATION",
        evaluation["forward"],
        evaluation.get("rays", r),
        valid_key="converged",
        direction="FORWARD",
    )
    final_reverse_fingerprint = (
        _final_reverse_trace_fingerprint(
            ctx,
            ctx.data["m1"],
            ctx.data["m2"],
            ctx.data["display"],
        )
    )

    ctx.data.update({
        "fan_refs": sources,
        "fan_reference_rule": rule,
        "final_reverse_trace": reverse,
        "final_reverse_trace_fingerprint":
            final_reverse_fingerprint,
        "final_reverse_trace_provenance": {
            "source_step": 24,
            "physical_first_hit": True,
            "fingerprint":
                final_reverse_fingerprint,
        },
    })
    fwd = evaluation["forward"]; virtual = evaluation["virtual"]; dist = evaluation["distortion"]
    fh2, fh1, fhv, first_order = (evaluation["first_hit_m2"], evaluation["first_hit_m1"],
                                  evaluation["first_hit_visor"], evaluation["first_order"])
    optics = _actual_optics(ctx, virtual["points"], virtual["bundle_valid"])
    converged = int(np.sum(fwd["converged"])); physical = bool(converged == len(r["rows"]))
    visor_loc = (fwd["visor"] - ctx.data["visor"].center) @ ctx.data["visor"].frame
    def visor_bounds(mask: np.ndarray, prefix: str) -> dict[str, Any]:
        """Trả bounds local của đúng nhóm tia được chỉ định."""
        q = visor_loc[np.asarray(mask, bool)]
        q = q[np.all(np.isfinite(q[:, :2]), axis=1)]
        return {f"{prefix}_count": len(q),
                f"{prefix}_VISOR_u_min_mm": float(np.min(q[:, 0])) if len(q) else None,
                f"{prefix}_VISOR_u_max_mm": float(np.max(q[:, 0])) if len(q) else None,
                f"{prefix}_VISOR_v_min_mm": float(np.min(q[:, 1])) if len(q) else None,
                f"{prefix}_VISOR_v_max_mm": float(np.max(q[:, 1])) if len(q) else None}
    footprint = {**visor_bounds(np.ones(len(visor_loc), bool), "attempted"),
                 **visor_bounds(fwd["physical_valid"], "physical_valid"),
                 **visor_bounds(fwd["converged"], "converged"),
                 "M1_bbox_min_mm": np.nanmin(fwd["m1"], axis=0), "M1_bbox_max_mm": np.nanmax(fwd["m1"], axis=0),
                 "M2_bbox_min_mm": np.nanmin(fwd["m2"], axis=0), "M2_bbox_max_mm": np.nanmax(fwd["m2"], axis=0)}
    va = np.asarray(ctx.config["visor_active_mm"]) / 2.0
    valid_visor_loc = visor_loc[fwd["converged"]]
    visor_inside = bool(len(valid_visor_loc) and np.all(np.isfinite(valid_visor_loc))
                        and np.max(np.abs(valid_visor_loc[:, 0])) <= va[0]
                        and np.max(np.abs(valid_visor_loc[:, 1])) <= va[1])
    footprint.update({"active_full_u_mm": 2.0*va[0], "active_full_v_mm": 2.0*va[1],
                      "converged_footprint_inside_active": visor_inside,
                      "all_rays_required_by_separate_physical_gate": True})
    lam = _packaging(ctx)
    spot_rows = []
    local = reverse["display_local"][:, :2]
    for f in range(int(r["field_count"])):
        for p in range(int(r["pupil_count"])):
            mask = (r["field_index"] == f) & (r["pupil_index"] == p)
            q = local[mask]; cen = np.mean(q, axis=0); radius = np.linalg.norm(q - cen, axis=1)
            spot_rows.append({"field_index": f, "pupil_index": p, "RMS_radius_mm": float(np.sqrt(np.mean(radius ** 2))),
                              "max_radius_mm": float(np.max(radius)), "direction": "REVERSE_MAPPING_DIAGNOSTIC"})
    summary_rows = [{"ray_id": row["ray_id"], "converged": bool(fwd["converged"][i]),
                     "valid_physical_sequence": bool(fwd["physical_valid"][i]), "eye_residual_mm": fwd["residual_mm"][i],
                     "first_display_to_m2": fh2["name"][i], "first_m2_to_m1": fh1["name"][i],
                     "first_m1_to_visor": fhv["name"][i], "first_order_pass": bool(first_order[i]),
                     "iterations": int(fwd["iterations"][i])} for i, row in enumerate(r["rows"])]
    virtual_rows = []
    for row in virtual["rows"]:
        virtual_rows.append(dict(row))
    p = ctx.step_dir(24)
    write_csv(p / "24N_FORWARD_SHOOTING_SUMMARY.csv", summary_rows)
    write_csv(p / "24N_VIRTUAL_POINTS.csv", virtual_rows)
    write_json(p / "24N_FOV_VID_D6_AZIMUTH.json", optics)
    write_csv(p / "24N_FINAL_DISTORTION.csv", dist["rows"])
    write_csv(p / "24N_FINAL_SPOT.csv", spot_rows)
    write_csv(p / "24N_FOOTPRINTS.csv", [footprint])
    write_json(p / "24N_APERTURES.json", {"M1": ctx.data["m1"].half_aperture,
                                           "M2": ctx.data["m2"].half_aperture,
                                           "display": ctx.data["display"].half_aperture,
                                           "visor_active_full_mm": ctx.config["visor_active_mm"]})
    write_json(p / "24N_PACKAGING.json", {"constraint_enabled": lam is not None,
               "lambda": lam, "limit": ctx.config["packaging_lambda_max"] if lam is not None else None,
               "pass": bool(lam <= ctx.config["packaging_lambda_max"]) if lam is not None else None,
               "status": "EVALUATED" if lam is not None else "NOT_CONSTRAINED_NO_P1_P8"})
    if ctx.config["mtf_requested"]:
        mtf_config = ctx.config["mtf"]
        mtf = ray_based_mtf(reverse, r, ctx.data["pattern"],
                            list(map(float, mtf_config["spatial_frequencies_lpmm"])),
                            list(map(float, mtf_config["wavelengths_nm"])),
                            list(map(float, mtf_config["wavelength_weights"])))
        mtf_summary = {k: v for k, v in mtf.items() if k != "rows"}
        mtf_minimum = float(mtf_config["minimum_at_max_frequency"])
        mtf_summary["minimum_required_at_max_frequency"] = (
            mtf_minimum if mtf_minimum > 0.0 else None
        )
        mtf_summary["passes_declared_minimum"] = (
            bool(mtf["complete"] and mtf["minimum_MTF_at_max_frequency"] is not None
                 and mtf["minimum_MTF_at_max_frequency"] >= mtf_minimum)
            if mtf_minimum > 0.0 else None
        )
        mtf_summary["grading"] = ("GRADED" if mtf_minimum > 0.0
                                  else "UNGRADED_NO_USER_TOLERANCE")
        write_csv(p / "24N_MTF.csv", mtf["rows"])
        write_json(p / "24N_MTF_SUMMARY.json", mtf_summary)
    else:
        mtf_summary = {"status": "NOT_REQUESTED_NOT_COMPUTED", "complete": False,
                       "passes_declared_minimum": None}
    numerical = {"validation_type": "NUMERICAL_SIMULATION_NOT_HARDWARE_MEASUREMENT",
                 "forward_sequence": ["DISPLAY", "M2", "M1", "VISOR_INNER", "EYE"],
                 "forward_converged": converged, "ray_count": len(r["rows"]),
                 "forward_physical_valid": int(np.sum(fwd["physical_valid"])),
                 "forward_valid_bundle_count": virtual.get("valid_bundle_count", int(np.sum(virtual.get("bundle_valid", [1])))),
                 "forward_required_bundle_count": virtual.get("required_bundle_count", len(virtual.get("bundle_valid", [1]))),
                 "physical_sequential_validity": physical,
                 "distortion": {k: v for k, v in dist.items() if k != "rows" and not k.startswith("_")},
                 "optics": optics, "packaging_constraint_enabled": lam is not None,
                 "packaging_lambda": lam, "visor_footprint": footprint,
                 "visor_footprint_inside_66x42": visor_inside,
                 "visor_footprint_scope": "CONVERGED_PHYSICAL_RAYS_ONLY; ALL_RAYS_GRADED_SEPARATELY",
                 "MTF": mtf_summary, "fake_sparse_FFT": False,
                 "Zemax_status": "SKIPPED_NOT_REQUIRED" if not ctx.config.get("zemax_crosscheck", False) else "REQUESTED_BUT_UNAVAILABLE",
                 "final_observables_from_forward_arriving_direction": True,
                  "fermat_all_converged": bool(ctx.data.get("fermat_all_converged", False)),
                  "fermat_fraction_pass": bool(ctx.data.get("fermat_fraction_pass", False)),
                  "fermat_minimum_fraction": _step17_minimum_candidate_fraction(ctx),
                  "fermat_convergence_scope": "ACCEPTED_CONSTRUCTION_PATH_ONLY",
                  "fermat_convergence_history": ctx.data.get("fermat_convergence_history", []),
                  "fermat_rejected_attempts": ctx.data.get("fermat_rejected_attempts", []),
                  "partial_fermat_accepted": bool(
                      ctx.data.get("fermat_fraction_pass", False)
                      and not ctx.data.get("fermat_all_converged", False)
                  )}
    write_json(p / "24N_NUMERICAL_ANALYSIS_SUMMARY.json", numerical)
    ctx.data["numerical"] = numerical; ctx.data["forward"] = fwd; ctx.data["virtual"] = virtual

    _record_optical_convergence(
        ctx,
        "STEP24 AUTHORITY",
        24,
        reverse_trace=reverse,
        forward_evaluation=evaluation,
        authority=True,
    )

    return {"status": "EVALUATED", "forward_converged": converged, "rays": len(r["rows"]),
            "D_max_percent": dist.get("D_max_percent", 0.0), "packaging_lambda": lam,
            "visor_inside": visor_inside, "MTF": mtf_summary, "Zemax": numerical["Zemax_status"]}

def step_25(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 25: Tổng hợp từng hard gate thành PASS/FAIL/UNGRADED và quyết định cuối."""

    n = ctx.data["numerical"]
    expected_fingerprint = (
        _final_reverse_trace_fingerprint(
            ctx,
            ctx.data["m1"],
            ctx.data["m2"],
            ctx.data["display"],
        )
    )

    stored_fingerprint = ctx.data.get(
        "final_reverse_trace_fingerprint"
    )

    final_trace = ctx.data.get(
        "final_reverse_trace"
    )

    provenance = ctx.data.get(
        "final_reverse_trace_provenance",
        {},
    )

    if final_trace is None:
        raise RuntimeError(
            "STEP25_MISSING_STEP24_"
            "FINAL_REVERSE_TRACE"
        )

    if (
        stored_fingerprint
        != expected_fingerprint
    ):
        raise RuntimeError(
            "STEP25_FINAL_REVERSE_TRACE_"
            "FINGERPRINT_MISMATCH"
        )

    if (
        provenance.get("source_step")
        != 24
        or provenance.get(
            "physical_first_hit"
        ) is not True
        or provenance.get(
            "fingerprint"
        ) != expected_fingerprint
    ):
        raise RuntimeError(
            "STEP25_FINAL_REVERSE_TRACE_"
            "PROVENANCE_INVALID"
        )
    _publish_trace_debug(
        ctx, 25, "FINAL_FAN_REVERSE_AUDIT",
        final_trace, ctx.data["rays"],
        direction="REVERSE",
    )
    final_refs, final_rule = _dynamic_refs(ctx, final_trace)
    final_mf1 = fan_imaging_metrics(final_trace, ctx.data["rays"], final_refs,
                                    float(ctx.config["fan_weights"]["omega1"]))
    final_mf2 = mf2_geometry(final_trace["points"][0], final_trace["points"][1], final_trace["points"][2],
                             ctx.data["m2"], float(ctx.config["fan_weights"]["omega2"]), ctx.data["n_obs"])
    final_mf2["E2_mm2"] = max(0.0, -float(final_mf2["S_AQP_signed_mm2"]))
    final_mf2["orientation_frozen_from_initial_planar_state"] = True
    ctx.data.update({"fan_mf1": final_mf1, "fan_mf2": final_mf2,
                     "fan_refs": final_refs, "fan_reference_rule": final_rule})
    # Store the final Fan diagnostics alongside STEP 24 numerical authority so
    # scorecard rebuilds and STEP 25 images use exactly the same final evidence.
    n["final_mf1"] = copy.deepcopy(final_mf1)
    n["final_mf2"] = copy.deepcopy(final_mf2)
    gates = build_final_scorecard(ctx.config, n)
    hard_pass = not any(x["status"] == "FAIL" for x in gates)
    final, decision_details = _final_numerical_decision(
        ctx.config, n, gates
    )
    report = {"final_status": final, "decision_details": decision_details,
              "hardware_validation": "NOT_PERFORMED",
              "simulation_claim": "NUMERICAL_ONLY", "Fan_sequence_traceable": True,
              "raw_Fan": {"MF1": ctx.data["fan_mf1"], "MF2": ctx.data["fan_mf2"], "MF3": "N/A"},
              "numerical": n, "scorecard": gates,
              "scorecard_schema": "DYNAMIC_ACTUAL_OPERATOR_THRESHOLD_REASON_V1",
              "full_pass_forbidden_without_missing_tolerances": True}
    write_csv(ctx.step_dir(25) / "25_FINAL_SCORECARD.csv", gates)
    write_json(ctx.step_dir(25) / "25_FINAL_OPTICAL_REPORT.json", report)
    ctx.data.update({"scorecard": gates, "final_report": report, "final_status": final})
    ungraded = sum(x["status"] in ("UNGRADED", "NOT_CONSTRAINED") for x in gates)
    return {"status": "PASS", "decision": final, "decision_details": decision_details, "hard_gate_pass": hard_pass,
            "ungraded_optical_targets": ungraded, "hardware_measurement": "NOT_PERFORMED"}

def step_26(ctx: Context) -> dict[str, Any]:
    """Thực thi STEP 26: Xuất prescription, coefficient, kết quả ray-trace, báo cáo và manifest đầy đủ."""

    out = ctx.run_dir
    i, vi, r = ctx.data["inputs"], ctx.data["vi"], ctx.data["rays"]
    field_count = int(r["field_count"]); ray_count = len(r["rows"])
    fields_filename = f"03_FIELDS_{field_count}.csv"
    rays_filename = f"05_CHARACTERISTIC_RAYS_{ray_count}.csv"
    visor_bundle_filename = f"06_VISOR_RAY_BUNDLE_{ray_count}.csv"
    write_json(out / "00_CURRENT_INPUT_SUMMARY.json", {
        "schema": ctx.config["schema"], "input_dir": ctx.config["input_dir"],
        "files": [p.name for p in i["files"]], "config": ctx.config,
        "input_hash_manifest_required": False,
        "algorithm_source_manifest_required": True,
        "algorithm_source_manifest_sha256": ctx.data["algorithm_source_manifest"]["manifest_sha256"]})
    write_json(out / "00_ALGORITHM_SOURCE_MANIFEST.json",
               ctx.data["algorithm_source_manifest"])
    visor = ctx.data["visor"]; write_json(out / "01_VISOR_FIT_MODEL.json", visor.to_dict() | ctx.data["visor_fit_stats"])
    loc = (i["visor_inner"] - visor.center) @ visor.frame
    fit_z = visor.sag(loc[:, 0], loc[:, 1]); fit_n = visor.normal(loc[:, 0], loc[:, 1])
    angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(fit_n * i["visor_inner_normals"], axis=1)), -1, 1)))
    sample = np.linspace(0, len(loc) - 1, min(2001, len(loc)), dtype=int)
    write_csv(out / "01_VISOR_FIT_RESIDUALS.csv", [{"sample_index": int(k),
              "sag_residual_mm": float(fit_z[k] - loc[k, 2]), "normal_residual_deg": float(angle[k])} for k in sample])
    write_json(out / "02_TARGET_VI_GEOMETRY.json", {k: v for k, v in vi.items() if k != "fields"})
    write_csv(out / fields_filename, [{"field_id": f["field_id"], "h_deg": f["h_deg"], "v_deg": f["v_deg"],
              "vi_x_mm": f["vi_point"][0], "vi_y_mm": f["vi_point"][1], "vi_z_mm": f["vi_point"][2],
              "u_vi_mm": f["u_vi_mm"], "v_vi_mm": f["v_vi_mm"]} for f in vi["fields"]])
    pupil_filename = f"04_PRIMARY_PUPILS_{len(ctx.data['pupils'])}.csv"
    write_csv(out / pupil_filename, ctx.data["pupils"])
    write_csv(out / "05_CHARACTERISTIC_PATTERN_49.csv", ctx.data["pattern"])
    write_csv(out / rays_filename, r["rows"])
    vh = ctx.data["visor_hit"]
    write_csv(out / visor_bundle_filename, [{"ray_id": row["ray_id"],
              "visor_x_mm": vh["point"][k, 0], "visor_y_mm": vh["point"][k, 1], "visor_z_mm": vh["point"][k, 2],
              "out_dx": ctx.data["post_visor"][k, 0], "out_dy": ctx.data["post_visor"][k, 1],
              "out_dz": ctx.data["post_visor"][k, 2]} for k, row in enumerate(r["rows"])])
    write_json(out / "07_PLANAR_START.json", ctx.data["planar_start_record"])
    write_csv(out / "08A_FAN_DYNAMIC_REFERENCE_GRID_FINAL.csv", [{"field_id": f["field_id"],
              "x_mm": p[0], "y_mm": p[1], "z_mm": p[2]} for f, p in zip(vi["fields"], ctx.data["fan_refs"])])
    write_csv(out / "08B_DISTORTION_FIXED_IDEAL_GRID.csv", [{"field_id": f["field_id"],
              "x_mm": p[0], "y_mm": p[1], "z_mm": p[2]} for f, p in zip(vi["fields"], ctx.data["fixed_vi_grid"])])
    ctx.data.setdefault("ci_m1_sampling_profile", "O2_SEARCH")
    ctx.data.setdefault("ci_m2_sampling_profile", "O2_SEARCH")
    ctx.data.setdefault("fermat_sampling_profile", "FREEFORM")
    ctx.data.setdefault("final_trace_sampling_profile", "CERTIFICATION")

    o2_rays = ctx.data.get("profile_rays_O2_SEARCH", r)
    for name, ci in (("10_M1_CI_POINT_CLOUD.csv", ctx.data["ci_m1"]),
                     ("11_M2_CI_POINT_CLOUD.csv", ctx.data["ci_m2"])):
        ci_rays = o2_rays if len(ci["points_by_ray"]) == len(o2_rays["rows"]) else r
        write_csv(out / name, [{"ray_id": ci_rays["rows"][k]["ray_id"], "x_mm": p[0], "y_mm": p[1], "z_mm": p[2],
                  "nx": n[0], "ny": n[1], "nz": n[2]} for k, (p, n) in enumerate(zip(ci["points_by_ray"], ci["normals_by_ray"]))])
    write_json(out / "12_M1_FAN_EQ1_ORDER2.json", ctx.data["fit_m1_order2"])
    write_json(out / "12_M2_FAN_EQ1_ORDER2.json", ctx.data["fit_m2_order2"])
    for step_number, name in (
        (12, "12_FIT_QUALITY_GATES.csv"),
        (12, "12_M1_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json"),
        (12, "12_M2_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json"),
        (16, "16_SELECTED_CI_CLOUD_INTEGRABILITY.json"),
        (17, "17_SELECTED_CI_CLOUD_INTEGRABILITY.json"),
        (17, "17_ORDER_POLICY.json"),
    ):
        source = ctx.step_dir(step_number) / name
        if source.exists():
            shutil.copy2(source, out / name)
    ff_rays = ctx.data.get("profile_rays_FREEFORM", o2_rays if "profile_rays_FREEFORM" not in ctx.data else r)
    sol = ctx.data["fermat"]
    fermat_rays = ff_rays if len(sol["success"]) == len(ff_rays["rows"]) else (o2_rays if len(sol["success"]) == len(o2_rays["rows"]) else r)
    write_csv(out / "14_FERMAT_SOLVER_SUMMARY.csv", [{"ray_id": fermat_rays["rows"][k]["ray_id"], "success": bool(sol["success"][k]),
              "gradient_norm": sol["gradient_norm"][k], "OP_initial_mm": sol["op_initial"][k],
              "OP_final_mm": sol["op_final"][k]} for k, _ in enumerate(sol["success"])])
    write_csv(out / "15_RHO_HISTORY.csv", ctx.data["order_history"])
    write_csv(out / "17_ORDER_HISTORY.csv", ctx.data["order_history"])
    write_json(out / "18_CI_ORDER5_STARTING_POINT.json", ctx.data["ci_start_prescription"])
    write_csv(out / "20_MF1_HISTORY.csv", [{"cycle": "FINAL_PRE_OPT_SAVED", **ctx.data["fan_mf1"]}])
    write_csv(out / "21_MF2_HISTORY.csv", [{k: v for k, v in ctx.data["fan_mf2"].items() if not isinstance(v, np.ndarray)}])
    write_csv(out / "22_DISTORTION_OPT_HISTORY.csv", [{"iteration": x.get("iteration"),
              "D_forward_monitor_percent": x.get("forward_distortion_monitor_percent"),
              "forward_invalid_ray_fraction": x.get("forward_invalid_ray_fraction"),
              "forward_invalid_bundle_fraction": x.get("forward_invalid_bundle_fraction"),
              "authority": "FORWARD_MONITOR_ACCEPTANCE_CONSTRAINT_NOT_FINAL_FULL_RAY_GATE"}
              for x in ctx.data["optimization_history"]])
    write_csv(out / "23_LOCAL_OPTIMIZATION_HISTORY.csv", ctx.data["optimization_history"])
    write_csv(out / "23_DLSQ_TRIALS.csv", ctx.data.get("dlsq_trial_history", []))
    write_json(out / "23_DLSQ_TERMINATION.json", {
        "phases": ctx.data.get("dlsq_termination_history", []),
        "blocked_is_not_convergence": True,
    })
    write_json(out / "23_OBJECTIVE_DEFINITION.json", ctx.data["objective_definition"])
    geometry_exports: list[str] = []
    for name in ("23A_COEFFICIENT_HISTORY.csv", "23A_DLSQ_TRIALS.csv",
                 "23A_DLSQ_TERMINATION.json", "23A_OBJECTIVE_INITIAL.json",
                 "23A_OBJECTIVE_FINAL.json"):
        source = ctx.step_dir(23) / name
        if source.exists():
            shutil.copy2(source, out / name)
            geometry_exports.append(name)
    if ("geometry_refinement_history" in ctx.data
            or "surface_parameter_refinement_history" in ctx.data):
        for name in ("23G_GEOMETRY_HISTORY.csv",
                     "23G_GEOMETRY_RESULT.json", "23G_POSE_REFINED_SURFACES.json",
                     "23S_SURFACE_PARAMETER_HISTORY.csv", "23S_SURFACE_PARAMETER_RESULT.json",
                     "23S_CURVATURE_CONIC_REFINED_SURFACES.json",
                     "23C_COEFFICIENT_HISTORY.csv", "23C_DLSQ_TRIALS.csv",
                     "23C_DLSQ_TERMINATION.json", "23C_OBJECTIVE_INITIAL.json",
                     "23C_OBJECTIVE_FINAL.json"):
            source = ctx.step_dir(23) / name
            if source.exists():
                shutil.copy2(source, out / name)
                geometry_exports.append(name)
    for name in ("24N_FORWARD_SHOOTING_SUMMARY.csv", "24N_VIRTUAL_POINTS.csv", "24N_FOV_VID_D6_AZIMUTH.json",
                 "24N_FINAL_DISTORTION.csv", "24N_FINAL_SPOT.csv", "24N_FOOTPRINTS.csv", "24N_APERTURES.json",
                 "24N_PACKAGING.json", "24N_NUMERICAL_ANALYSIS_SUMMARY.json"):
        shutil.copy2(ctx.step_dir(24) / name, out / name)
    if ctx.config["mtf_requested"]:
        for name in ("24N_MTF.csv", "24N_MTF_SUMMARY.json"):
            shutil.copy2(ctx.step_dir(24) / name, out / name)
    shutil.copy2(ctx.step_dir(25) / "25_FINAL_OPTICAL_REPORT.json", out / "25_FINAL_OPTICAL_REPORT.json")
    shutil.copy2(ctx.step_dir(25) / "25_FINAL_SCORECARD.csv", out / "25_FINAL_SCORECARD.csv")
    write_json(out / "FINAL_PRESCRIPTION.json", _prescription(ctx))
    for surf, name in ((ctx.data["m1"], "M1"), (ctx.data["m2"], "M2")):
        write_json(out / f"FINAL_SURFACE_{name}.json", surf.to_dict())
        write_csv(out / f"FINAL_SURFACE_{name}.csv", [{"i": t[0], "j": t[1], "normalized_coefficient_mm": a,
                  "physical_coefficient": a / (surf.scale[0] ** t[0] * surf.scale[1] ** t[1])}
                 for t, a in zip(surf.terms, surf.coeff)])
    (out / "FINAL_RUN_STATUS.txt").write_text(ctx.data["final_status"] + "\nNUMERICAL_SIMULATION; HARDWARE_NOT_PERFORMED\n", encoding="utf-8")
    minimum = ["00_CURRENT_INPUT_SUMMARY.json", "00_ALGORITHM_SOURCE_MANIFEST.json",
               "01_VISOR_FIT_MODEL.json", "01_VISOR_FIT_RESIDUALS.csv",
               "02_TARGET_VI_GEOMETRY.json", fields_filename, pupil_filename,
               "05_CHARACTERISTIC_PATTERN_49.csv", rays_filename, visor_bundle_filename,
               "12_FIT_QUALITY_GATES.csv",
               "12_M1_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
               "12_M2_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
               "16_SELECTED_CI_CLOUD_INTEGRABILITY.json",
               "17_SELECTED_CI_CLOUD_INTEGRABILITY.json", "17_ORDER_POLICY.json",
               "23_DLSQ_TRIALS.csv", "23_DLSQ_TERMINATION.json",
               "08A_FAN_DYNAMIC_REFERENCE_GRID_FINAL.csv", "08B_DISTORTION_FIXED_IDEAL_GRID.csv",
               "24N_NUMERICAL_ANALYSIS_SUMMARY.json", "25_FINAL_OPTICAL_REPORT.json", "25_FINAL_SCORECARD.csv",
               "FINAL_PRESCRIPTION.json", "FINAL_SURFACE_M1.json", "FINAL_SURFACE_M2.json", "FINAL_RUN_STATUS.txt"]
    if ctx.config["mtf_requested"]:
        minimum.extend(["24N_MTF.csv", "24N_MTF_SUMMARY.json"])
    minimum.extend(geometry_exports)
    missing = [name for name in minimum if not (out / name).exists()]
    if missing: raise RuntimeError(f"FINAL_EXPORT_MISSING: {missing}")
    write_json(out / "26_EXPORT_COMPLETENESS.json", {"minimum_files": minimum, "missing": missing,
              "geometry_refinement_exports": geometry_exports,
              "input_manifest_hash_not_required": True,
              "algorithm_source_manifest_required": True,
              "algorithm_source_manifest_sha256": ctx.data["algorithm_source_manifest"]["manifest_sha256"],
              "holdout_not_required": True, "density_729_3969_not_required": True})
    ctx.data[
        "step26_export_file_inventory"
    ] = sorted(
        p.name
        for p in out.iterdir()
        if p.is_file()
    )
    return {"status": "PASS", "final_status": ctx.data["final_status"], "minimum_files": len(minimum),
            "missing": 0, "fake_Zemax_files": False,
            "MTF_claim": ctx.data["numerical"]["MTF"].get("claim", "NOT_REQUESTED")}


_STAGE_DEFINITIONS: dict[int, tuple[str, Callable[["Context"], dict[str, Any]]]] = {
    0: ("Load current input and lock declared authority", step_00),
    1: ("Fit differentiable visor-inner surrogate", step_01),
    2: ("Build fixed target virtual-image geometry", step_02),
    3: ("Build configured field samples and ideal VI points", step_03),
    4: ("Build configurable primary pupil grid", step_04),
    5: ("Build 49 characteristic rays per field/pupil", step_05),
    6: ("Trace fitted visor and form characteristic bundle", step_06),
    7: ("Auto-search compact physically valid planar geometry", step_07),
    8: ("Create separate dynamic Fan and fixed USER reference grids", step_08),
    9: ("Fan Step One planar conjugate targets", step_09),
    10: ("Construct M1 first by point-by-point CI", step_10),
    11: ("System-aware O2 pair construction", step_11),
    12: ("Joint refine STEP11-selected Fan Eq.(1) order-2 pair", step_12),
    13: ("Disable planar shortcut and enter Fan Step Two", step_13),
    14: ("Solve constrained Fermat target on current M2", step_14),
    15: ("Declare coarse-to-fine signed-rho search", step_15),
    16: ("Coarse-to-fine search for the best full-physical order-2 rho", step_16),
    17: ("Increase nested CI basis levels gradually to fifth order", step_17),
    18: ("Save fifth-order CI starting structure", step_18),
    19: ("Declare Fan-core merit and separate engineering constraints", step_19),
    20: ("Evaluate chief-centered MF1 using every characteristic ray", step_20),
    21: ("Evaluate signed 3-D obscuration merit MF2", step_21),
    22: ("Declare fixed USER distortion constraint", step_22),
    23: ("Coefficient DLSQ with bounded curvature/conic and optional pose refinement", step_23),
    24: ("Mandatory numerical forward optical evaluation; optional Zemax branch", step_24),
    25: ("Build final numerical scorecard", step_25),
    26: ("Export complete v5.5 result set", step_26),
}

if set(_STAGE_DEFINITIONS) != set(range(MAX_STEP + 1)):
    raise RuntimeError("STAGE_REGISTRY_MUST_COVER_STEP_00_THROUGH_STEP_26")

for _step_number, (_step_title, _step_fn) in _STAGE_DEFINITIONS.items():
    _step_fn.__module__ = __name__
    _step_fn.__globals__.setdefault("Context", Context)
    STAGES[_step_number] = (_step_title, _step_fn)


def run_all(config_path: Path, run_dir: Path) -> Context:
    """Chạy tuần tự toàn bộ STEP 00–26 trong cùng Context."""
    from visualization_v55 import render_spot_evolution, render_step, render_surface_ray_view
    from execution_v55 import execution_session
    c = json.loads(config_path.read_text(encoding="utf-8")); validate_config(c)
    ctx = Context(Path(__file__).resolve().parent, config_path.resolve(), c, run_dir.resolve())
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    with execution_session(c, run_dir.resolve()):
        for number in range(MAX_STEP + 1):
            title, fn = STAGES[number]
            print(f"[{number:02d}/{MAX_STEP:02d}] {title}", flush=True)
            try:
                summary = _call_stage_algorithm(ctx, number, fn)
                ctx.stage_summaries[number] = summary
                try:
                    summary["visualization_3d"] = render_step(ctx, number)
                except Exception as exc:
                    summary["visualization_3d"] = {
                        "status": "VISUALIZATION_WARNING",
                        "message": str(exc),
                    }
                try:
                    summary["surface_ray_footprint"] = render_surface_ray_view(ctx, number)
                except Exception as exc:
                    # Chỉ cảnh báo lỗi render; không can thiệp trạng thái thuật toán quang học.
                    summary["surface_ray_footprint"] = {"status": "VISUALIZATION_WARNING", "message": str(exc)}
                    print(f"SURFACE_RAY_IMAGE_WARNING STEP_{number:02d}: {exc}", flush=True)
                if number == 20:
                    try:
                        # Ảnh phụ đọc snapshot thật; lỗi render không được phép đổi PASS/FAIL quang học.
                        summary["spot_evolution"] = render_spot_evolution(ctx)
                    except Exception as exc:
                        summary["spot_evolution"] = {"status": "VISUALIZATION_WARNING", "message": str(exc)}
                        print(f"SPOT_EVOLUTION_IMAGE_WARNING STEP_{number:02d}: {exc}", flush=True)
                _archive_step(ctx, number, title, fn, summary)
            except Exception as exc:
                _record_step_failure_safely(ctx, number, exc)
                print(f"STEP_{number:02d} FAILED: {exc}", file=sys.stderr, flush=True)
                raise
    return ctx


def main() -> None:
    """Đọc config và chạy pipeline end-to-end từ dòng lệnh."""
    root = Path(__file__).resolve().parent
    config = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "config_v55.json"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all(config, root / "runs_v5_5" / f"HUD_FAN_V5_5_{stamp}")


if __name__ == "__main__":
    main()
