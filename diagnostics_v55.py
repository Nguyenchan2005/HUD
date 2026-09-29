"""Hệ thống chẩn đoán và điều tra lỗi dùng chung cho HUD Fan pipeline v5.5.

Cung cấp các công cụ:
1. Xuất dữ liệu chẩn đoán an toàn (JSON và CSV nguyên tử, phân tách non-finite).
2. Trích xuất danh sách và tọa độ các tia không đạt (FAILED_RAYS.csv).
3. Định dạng và lưu vết các vi phạm hình học mặt cong (surface sanity violations).
4. Lưu giữ cấu hình và hình học của các ứng viên bị loại (FAILED_CANDIDATES).
5. Phân tích nguyên nhân fallback trong quá trình dựng mặt theo CI.
6. Tổng hợp báo cáo điều tra lỗi chi tiết (DEBUG_REPORT.md và DEBUG_SUMMARY.json).
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

import numpy as np


class DiagnosticSchemaError(ValueError):
    """Dữ liệu debug không đúng hợp đồng; không phải một optical gate."""


def _diagnostic_jsonable(
    value: Any,
    nonfinite: list[dict[str, str]],
    location: str = "$",
) -> Any:
    """Chuyển bản dữ liệu xuất; không mutate dữ liệu tính toán."""
    if isinstance(value, np.ndarray):
        return _diagnostic_jsonable(value.tolist(), nonfinite, location)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        nonfinite.append({"path": location, "value": repr(value)})
        return None

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DiagnosticSchemaError(
                    f"NON_STRING_JSON_KEY at {location}: {type(key).__name__}"
                )
            result[key] = _diagnostic_jsonable(
                item, nonfinite, f"{location}.{key}"
            )
        return result

    if isinstance(value, (list, tuple)):
        return [
            _diagnostic_jsonable(item, nonfinite, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]

    raise DiagnosticSchemaError(
        f"UNSUPPORTED_DIAGNOSTIC_TYPE at {location}: {type(value).__name__}"
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Ghi file tạm cùng thư mục rồi thay thế file đích một cách nguyên tử."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def write_diagnostic_json(path: Path, payload: dict[str, Any]) -> None:
    """JSON nghiêm ngặt; non-finite có vị trí giải thích riêng."""
    nonfinite: list[dict[str, str]] = []
    data = _diagnostic_jsonable(payload, nonfinite)

    if "_diagnostic_serialization" in data:
        raise DiagnosticSchemaError(
            "RESERVED_KEY:_diagnostic_serialization"
        )

    data["_diagnostic_serialization"] = {
        "schema": "HUD_DIAGNOSTIC_JSON_V1",
        "nonfinite_values": nonfinite,
        "nonfinite_export_policy": "NULL_WITH_EXPLICIT_PATH",
    }

    atomic_write_text(
        path,
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
    )


def save_failed_candidate(
    step_dir: Path,
    order: int,
    cycle: int,
    branch_id: str,
    m1: Any,
    m2: Any,
    reason: str,
    details: dict[str, Any] | None = None,
) -> Path:
    """Lưu đúng mặt ứng viên được caller truyền vào."""
    out_dir = Path(step_dir) / "FAILED_CANDIDATES"
    safe_branch = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(branch_id)
    ).strip("._") or "branch"

    # UUID chỉ phục vụ tên artifact, không tham gia RNG của thuật toán.
    suffix = uuid.uuid4().hex
    target = out_dir / (
        f"ORDER_{int(order)}_CYCLE_{int(cycle)}_"
        f"{safe_branch}_{suffix}.json"
    )

    def prescription(surface: Any) -> Any:
        """Trích xuất cấu trúc dictionary prescription của mặt quang học ứng viên."""
        if surface is None:
            return None
        if not callable(getattr(surface, "to_dict", None)):
            raise DiagnosticSchemaError(
                "CANDIDATE_SURFACE_HAS_NO_TO_DICT"
            )
        return surface.to_dict()

    payload = {
        "schema": "HUD_FAILED_CANDIDATE_V1",
        "order": int(order),
        "cycle": int(cycle),
        "branch_id": str(branch_id),
        "rejection_reason": str(reason),
        "details": {} if details is None else details,
        "m1_prescription": prescription(m1),
        "m2_prescription": prescription(m2),
        "missing_surface_means": "NOT_CONSTRUCTED_OR_NOT_RECORDED",
    }
    write_diagnostic_json(target, payload)
    return target


def try_save_failed_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lỗi ghi debug không được thay lỗi quang học của caller."""
    try:
        path = save_failed_candidate(*args, **kwargs)
        return {
            "candidate_artifact": str(path),
            "diagnostic_export_status": "WRITTEN",
            "diagnostic_export_error": None,
        }
    except Exception as exc:
        # Chỉ bao quanh việc xuất artifact; không bao quanh solver.
        message = f"{type(exc).__name__}: {exc}"
        print(f"DIAGNOSTIC_EXPORT_WARNING: {message}", flush=True)
        return {
            "candidate_artifact": None,
            "diagnostic_export_status": "FAILED",
            "diagnostic_export_error": message,
        }


FAILED_RAYS_FIELDNAMES = [
    "ray_index",
    "ray_id",
    "ray_id_source",
    "field_index",
    "pupil_index",
    "sample_index",
    "first_failure_surface",
    "first_failure_reason",
    "first_failure_stage",
    "point_x_mm",
    "point_y_mm",
    "point_z_mm",
    "dir_x",
    "dir_y",
    "dir_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "residual_mm",
]


def extract_failed_rays(
    trace_result: dict[str, Any],
    rays_dict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Trích xuất danh sách các tia thất bại từ kết quả trace_reverse kèm tọa độ và lý do lỗi."""
    if not isinstance(trace_result, dict) or "valid" not in trace_result:
        raise DiagnosticSchemaError("TRACE_VALID_MASK_NOT_RECORDED")

    valid = np.asarray(trace_result["valid"])
    if valid.ndim != 1 or valid.dtype.kind != "b":
        raise DiagnosticSchemaError("TRACE_VALID_MASK_MUST_BE_BOOL_N")

    n = len(valid)

    def vector_field(key: str, default: Any, dtype: Any) -> np.ndarray:
        """Trích xuất mảng vector 1 chiều có kiểm tra kích thước N."""
        result = np.asarray(trace_result.get(key, default), dtype=dtype)
        if result.shape != (n,):
            raise DiagnosticSchemaError(
                f"{key}: expected {(n,)}, got {result.shape}"
            )
        return result

    def xyz_field(key: str) -> np.ndarray:
        """Trích xuất ma trận tọa độ không gian 3 chiều có kích thước (N, 3)."""
        result = np.asarray(
            trace_result.get(key, np.full((n, 3), np.nan)),
            dtype=float,
        )
        if result.shape != (n, 3):
            raise DiagnosticSchemaError(
                f"{key}: expected {(n, 3)}, got {result.shape}"
            )
        return result

    first_surf = vector_field(
        "first_failure_surface", np.full(n, "UNKNOWN", object), object
    )
    first_reason = vector_field(
        "first_failure_reason", np.full(n, "NOT_RECORDED", object), object
    )
    first_stage = vector_field(
        "first_failure_stage_index", np.full(n, -1, int), int
    )
    first_res = vector_field(
        "first_failure_residual", np.full(n, np.nan), float
    )
    first_pt = xyz_field("first_failure_point")
    first_dir = xyz_field("first_failure_dir")
    first_norm = xyz_field("first_failure_normal")

    if rays_dict is None:
        rows = [{} for _ in range(n)]
    else:
        rows = rays_dict.get("rows")
        if not isinstance(rows, list) or len(rows) != n:
            raise DiagnosticSchemaError("RAY_ROWS_DO_NOT_MATCH_TRACE")

    failed_indices = np.flatnonzero(~valid)
    records: list[dict[str, Any]] = []

    for idx in failed_indices:
        i = int(idx)
        ray_info = rows[i]
        raw_id = ray_info.get("ray_id") if isinstance(ray_info, dict) else None
        has_id = bool(raw_id and str(raw_id).strip())

        surf_str = str(first_surf[i]).strip() if first_surf[i] is not None else ""
        reason_str = str(first_reason[i]).strip() if first_reason[i] is not None else ""

        records.append({
            "ray_index": i,
            "ray_id": str(raw_id) if has_id else f"DIAGNOSTIC_INDEX_{i}",
            "ray_id_source": "ORIGINAL" if has_id else "LOCAL_DIAGNOSTIC_INDEX",
            "field_index": ray_info.get("field_index", -1) if isinstance(ray_info, dict) else -1,
            "pupil_index": ray_info.get("pupil_index", -1) if isinstance(ray_info, dict) else -1,
            "sample_index": ray_info.get("sample_index", -1) if isinstance(ray_info, dict) else -1,
            "first_failure_surface": surf_str if surf_str else "UNKNOWN",
            "first_failure_reason": reason_str if reason_str else "NOT_RECORDED",
            "first_failure_stage": int(first_stage[i]),
            "point_x_mm": float(first_pt[i, 0]) if np.isfinite(first_pt[i, 0]) else None,
            "point_y_mm": float(first_pt[i, 1]) if np.isfinite(first_pt[i, 1]) else None,
            "point_z_mm": float(first_pt[i, 2]) if np.isfinite(first_pt[i, 2]) else None,
            "dir_x": float(first_dir[i, 0]) if np.isfinite(first_dir[i, 0]) else None,
            "dir_y": float(first_dir[i, 1]) if np.isfinite(first_dir[i, 1]) else None,
            "dir_z": float(first_dir[i, 2]) if np.isfinite(first_dir[i, 2]) else None,
            "normal_x": float(first_norm[i, 0]) if np.isfinite(first_norm[i, 0]) else None,
            "normal_y": float(first_norm[i, 1]) if np.isfinite(first_norm[i, 1]) else None,
            "normal_z": float(first_norm[i, 2]) if np.isfinite(first_norm[i, 2]) else None,
            "residual_mm": float(first_res[i]) if np.isfinite(first_res[i]) else None,
        })
    return records


def write_failed_rays_csv(path: Path, failed_rays: list[dict[str, Any]]) -> None:
    """Ghi danh sách tia thất bại ra định dạng CSV chuẩn, luôn có header kể cả khi 0 lỗi."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FAILED_RAYS_FIELDNAMES, extrasaction="raise")
    writer.writeheader()
    if failed_rays:
        writer.writerows(failed_rays)
    atomic_write_text(path, stream.getvalue())


def format_sanity_violation(surface_name: str, sanity: dict[str, Any]) -> str:
    """Định dạng kết quả kiểm tra surface_sanity thành bảng báo cáo markdown."""
    lines = [
        f"### Kiểm tra bề mặt: {surface_name}",
        f"- **Trạng thái**: {'PASS' if sanity.get('pass') else 'FAIL'}",
        f"- **Các điều kiện không đạt**: {', '.join(sanity.get('failure_reasons', [])) or 'Không có'}",
        f"- **Độ võng sag (min/max/PV)**: {sanity.get('sag_min_mm', 'nan')} / {sanity.get('sag_max_mm', 'nan')} / {sanity.get('sag_pv_mm', 'nan')} mm",
        f"- **Độ dốc cực đại**: {sanity.get('max_slope', 'nan')}",
        f"- **Miền conic min**: {sanity.get('minimum_conic_domain_argument', 'nan')}",
        f"- **Phát hiện gai sag (spike)**: {sanity.get('spike_detected', False)}",
    ]
    details = sanity.get("violation_details", {})
    if details:
        lines.append("- **Chi tiết vi phạm**:")
        for check_name, info in details.items():
            val = info.get("value")
            thresh = info.get("threshold")
            coord = info.get("coord_local_mm", [])
            lines.append(f"  - `{check_name}`: giá trị thực tế = {val}, ngưỡng = {thresh}, tọa độ kiểm tra = {coord}")
    return "\n".join(lines)


def summarize_ci_fallbacks(ci_result: dict[str, Any], rays_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tổng hợp chẩn đoán fallback từ kết quả dựng mặt theo Nearest-Ray."""
    parent = np.asarray(ci_result.get("parent_order_index", []))
    ray_order = np.asarray(ci_result.get("ray_order", []))
    fallback_reasons = ci_result.get("fallback_reasons", [])

    fallback_mask = parent == -2
    front_restart_mask = parent == -3
    fallback_indices = np.where(fallback_mask)[0]

    rows = rays_dict.get("rows", []) if rays_dict else []
    items = []
    for k in fallback_indices:
        ray_idx = int(ray_order[k]) if k < len(ray_order) else -1
        ray_info = rows[ray_idx] if 0 <= ray_idx < len(rows) else {}
        items.append({
            "construction_order": int(k),
            "ray_index": ray_idx,
            "ray_id": ray_info.get("ray_id", f"RAY_{ray_idx:05d}"),
            "field_index": ray_info.get("field_index", -1),
            "pupil_index": ray_info.get("pupil_index", -1),
            "reason": str(fallback_reasons[k]) if k < len(fallback_reasons) else "FALLBACK_SEED_SURFACE",
        })

    return {
        "fallback_count": int(np.sum(fallback_mask)),
        "total_rays": len(parent),
        "fallback_ratio": float(np.sum(fallback_mask)) / max(len(parent), 1),
        "front_restart_count": int(np.sum(front_restart_mask)),
        "front_restart_ratio": float(np.sum(front_restart_mask)) / max(len(parent), 1),
        "candidate_rejections": ci_result.get("fallback_candidate_rejections", {}),
        "fallback_events": items,
    }


def build_step_debug_report(
    step_num: int,
    title: str,
    summary: dict[str, Any],
    ctx: Any = None,
    extra_info: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Tạo báo cáo chẩn đoán lỗi chi tiết (markdown và JSON) cho một lần chạy STEP."""
    execution_status = summary.get("execution_status", summary.get("status", "RECORDED"))
    optical_status = summary.get("status", "NOT_EVALUATED")
    extra = extra_info or {}

    lines = [
        f"# BÁO CÁO CHẨN ĐOÁN LỖI: STEP_{step_num:02d} — {title}",
        "",
        f"- **Trạng thái thực thi thuật toán**: `{execution_status}`",
        f"- **Trạng thái quang học ghi nhận**: `{optical_status}`",
        f"- **Loại bước**: Nhóm {step_num // 5}",
        "",
        "## 1. Tóm tắt kết quả vận hành",
        "",
    ]

    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)):
            lines.append(f"- **{k}**: `{v}`")

    lines.append("")
    lines.append("## 2. Kiểm tra điều kiện quang học và bằng chứng hình học")
    lines.append("")

    traces_info = extra.get("traces")
    if isinstance(traces_info, dict) and traces_info:
        for phase_name, pdata in traces_info.items():
            lines.append(f"### Pha: {phase_name}")
            lines.append(f"- **Bước phát sinh**: STEP_{pdata.get('producer_step', step_num):02d}")
            lines.append(f"- **Chiều truy vết**: `{pdata.get('direction', 'REVERSE')}`")
            lines.append(f"- **Ngữ nghĩa hợp lệ**: `{pdata.get('validity_semantics', 'valid')}`")
            lines.append(f"- **Số tia khảo sát**: {pdata.get('ray_count', 0)}")
            lines.append(f"- **Số tia hợp lệ**: {pdata.get('valid_count', 0)}")
            lines.append(f"- **Số tia không đạt**: {pdata.get('failed_count', 0)}")
    elif extra.get("failed_rays_summary"):
        frs = extra["failed_rays_summary"]
        lines.append(f"- **Số tia không đạt**: {frs.get('count', 0)}")
        lines.append(f"- **Mặt đầu tiên xuất hiện lỗi**: `{frs.get('first_surface', 'N/A')}`")
        lines.append(f"- **Mã lỗi phát hiện**: `{frs.get('primary_reason', 'N/A')}`")
        surface_counts = frs.get("by_surface", {})
        if surface_counts:
            lines.append("  - Phân bổ theo mặt phát hiện đầu tiên:")
            for sname, scnt in surface_counts.items():
                lines.append(f"    - `{sname}`: {scnt} tia")
    else:
        lines.append("- **Đánh giá tia**: `NOT_EVALUATED` hoặc không có dữ liệu truy vết tia tại bước này.")

    if "surface_sanity" in extra and isinstance(extra["surface_sanity"], dict):
        for sname, sdata in extra["surface_sanity"].items():
            lines.append("")
            lines.append(format_sanity_violation(sname, sdata))

    if "ci_fallbacks" in extra and isinstance(extra["ci_fallbacks"], dict):
        fb = extra["ci_fallbacks"]
        lines.append("")
        lines.append(f"- **CI Fallback**: {fb.get('fallback_count', 0)} / {fb.get('total_rays', 0)} tia phải dùng mặt seed.")
        if fb.get("candidate_rejections"):
            lines.append(f"  - Nguyên nhân loại trừ tiếp tuyến cha: {fb.get('candidate_rejections')}")

    if extra.get("diagnostic_errors"):
        lines.append("")
        lines.append("### Lỗi phát sinh trong quá trình xuất chẩn đoán:")
        for err in extra["diagnostic_errors"]:
            lines.append(f"- `{err.get('phase', 'UNKNOWN')}`: {err.get('error_type', 'Error')} - {err.get('message', '')}")

    lines.append("")
    lines.append("## 3. Phân biệt vị trí phát hiện và nguyên nhân gốc")
    lines.append("")
    lines.append("- **Vị trí phát hiện**: Đối chiếu chi tiết tại bảng các tia không đạt và các file chẩn đoán tương ứng.")
    lines.append("- **Khuyến nghị điều tra**:")
    lines.append("  1. Đối chiếu góc tới và pháp tuyến tại mặt trước đó xem có làm lệch hướng chùm tia hay không.")
    lines.append("  2. Kiểm tra biên khẩu độ clear aperture đã đủ bao trọn footprint thực tế hay chưa.")
    lines.append("  3. Tránh việc nới lỏng ngưỡng trước khi xác định rõ sai lệch hình học.")

    debug_report_md = "\n".join(lines) + "\n"
    debug_summary_json = {
        "step": step_num,
        "title": title,
        "execution_status": execution_status,
        "optical_status": optical_status,
        "summary": summary,
        "diagnostic_details": extra,
    }
    return debug_report_md, debug_summary_json
