"""Dựng lại ảnh hậu kỳ từ state đã lưu của multistart mà không chạy lại thuật toán quang học.

Quét toàn bộ STEP 00-26, tự động nhận diện state thành công hoặc partial failure state.
Không lọc theo danh sách bước cấu hình.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core import PolySurface, aperture_boundary_xy, aperture_contains_xy, write_json
from pipeline_v55 import _algorithm_source_manifest
from visualization_v55 import (
    _equal,
    _poly_surface,
    _scatter,
    _set_labels,
    _trace_lines,
    _visor_surface,
    render_planar_seed_precheck,
    render_spot_evolution,
    render_step,
    render_surface_ray_view,
)

import math


ROOT = Path(__file__).resolve().parent
MULTISTART_ROOT = ROOT / "multistart_runs_v5_5"
SURFACE_CAD_STEPS = {10, 11, 12, 16, 17, 23}

STEP11_SEARCH_DASHBOARD_PNG = "11_SEARCH_DASHBOARD.png"
STEP11_SEARCH_DASHBOARD_JSON = "11_SEARCH_DASHBOARD.json"
STEP11_TOPOLOGY_MAPS_PNG = "11_TOPOLOGY_MAPS.png"
STEP11_TOPOLOGY_MAPS_JSON = "11_TOPOLOGY_MAPS.json"
STEP11_PHYSICAL_VS_RMS_PNG = "11_PHYSICAL_VS_RMS.png"
STEP11_PHYSICAL_VS_RMS_JSON = "11_PHYSICAL_VS_RMS.json"
STEP11_HISTORY_CSV = "11_O2_SEARCH_HISTORY.csv"
STEP11_RAY_FOOTPRINT_PNG = "11_SURFACE_RAY_FOOTPRINT.png"
STEP11_SURFACE_CAD_PNG = "11_SURFACE_CAD_VIEW.png"
STEP11_CANDIDATE_SNAPSHOT_DIR = "11_CANDIDATE_SNAPSHOTS"
STEP11_CANDIDATE_SNAPSHOT_MANIFEST_JSON = "manifest.json"
STEP11_CANDIDATE_IMAGE_DIR = "11_CANDIDATE_IMAGES"
STEP11_CANDIDATE_RENDER_MANIFEST_JSON = (
    "11_CANDIDATE_RENDER_MANIFEST.json"
)
STEP11_CANDIDATE_3D_MAX_RAYS = 240


def _latest_multistart_root() -> Path:
    """Tìm thư mục multistart run mới nhất dựa trên timestamp tên thư mục."""
    candidates = sorted(
        path
        for path in MULTISTART_ROOT.glob("HUD_FAN_V5_5_MULTISTART_*")
        if path.is_dir()
    )

    if not candidates:
        raise RuntimeError(f"NO_MULTISTART_RUN_FOUND:{MULTISTART_ROOT}")

    return candidates[-1].resolve()


def _find_state(run_dir: Path, step: int) -> tuple[Path | None, str]:
    """Tìm kiếm file state lưu trữ của STEP, ưu tiên RENDER_STATES sau đó là CHECKPOINTS."""
    render_state = (
        run_dir
        / "RENDER_STATES"
        / f"STEP_{step:02d}"
        / "state.pkl.gz"
    )
    if render_state.is_file():
        return render_state, "RENDER_STATE"

    checkpoint = (
        run_dir
        / "CHECKPOINTS"
        / f"STEP_{step:02d}"
        / "state.pkl.gz"
    )
    if checkpoint.is_file():
        return checkpoint, "CHECKPOINT"

    return None, "NONE"


def _load_state(state_path: Path, run_dir: Path) -> Any:
    """Nạp Context từ file state nén và cập nhật lại run_dir hiện tại."""
    with gzip.open(state_path, "rb") as stream:
        ctx = pickle.load(stream)
    ctx.run_dir = run_dir.resolve()
    return ctx


def _assert_manifest_matches(ctx: Any, allow_source_mismatch: bool) -> dict[str, Any]:
    """Kiểm tra mã băm SHA256 mã nguồn thuật toán giữa state và môi trường hiện tại."""
    stored = ctx.data.get("algorithm_source_manifest", {}).get("manifest_sha256")
    current = _algorithm_source_manifest(ctx).get("manifest_sha256")
    matched = bool(stored and current and stored == current)

    if not matched and not allow_source_mismatch:
        raise RuntimeError(
            f"DEFERRED_RENDER_SOURCE_MANIFEST_MISMATCH:stored={stored}:current={current}:"
            "use --allow-source-mismatch only for inspection"
        )

    return {
        "stored": stored,
        "current": current,
        "matched": matched,
    }


def _update_step_summary(run_dir: Path, step: int, records: dict[str, Any]) -> None:
    """Cập nhật bản ghi render hậu kỳ vào file STEP_SUMMARY.json nếu file này tồn tại."""
    path = run_dir / f"STEP_{step:02d}" / "STEP_SUMMARY.json"
    if not path.exists():
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.setdefault("summary", {})
    summary.update(records)
    payload["visualization_render_mode"] = "POSTHOC_FROM_SAVED_STATE"
    payload["visualization_rendered_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, payload)


def _is_step_rendered(run_dir: Path, step: int) -> bool:
    """Kiểm tra ảnh hậu kỳ bắt buộc, gồm CAD surface metadata tại các STEP thay đổi mặt."""
    step_dir = run_dir / f"STEP_{step:02d}"
    required = [
        step_dir / f"{step:02d}_3D_SPATIAL_VIEW.png",
        step_dir / f"{step:02d}_3D_VIEW_METADATA.json",
    ]
    if step in SURFACE_CAD_STEPS:
        required.append(step_dir / f"{step:02d}_SURFACE_CAD_VIEW.json")

    if step == 11:
        snapshot_manifest = (
            step_dir
            / STEP11_CANDIDATE_SNAPSHOT_DIR
            / STEP11_CANDIDATE_SNAPSHOT_MANIFEST_JSON
        )
        if snapshot_manifest.is_file():
            required.append(
                step_dir
                / STEP11_CANDIDATE_RENDER_MANIFEST_JSON
            )

    return all(p.exists() for p in required)


def _step11_read_history_rows(step_dir: Path) -> list[dict[str, Any]]:
    """Đọc file 11_O2_SEARCH_HISTORY.csv của STEP11."""
    path = step_dir / STEP11_HISTORY_CSV
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return [dict(row) for row in reader]


def _step11_bool(value: Any) -> bool:
    """Chuyển đổi giá trị chuỗi hoặc bool sang kiểu boolean chuẩn."""
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _step11_float(value: Any, default: float = float("nan")) -> float:
    """Chuyển đổi giá trị sang kiểu float an toàn với giá trị mặc định."""
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _step11_topology_label(row: dict[str, Any]) -> str:
    """Gán nhãn phân loại candidate theo feasibility, cảnh báo và gate vi phạm."""
    reason = str(row.get("rejection_reason", "")).strip()
    stage = str(row.get("rejection_stage", "")).strip()
    feasible = _step11_bool(row.get("feasible", False))
    physical_status = str(row.get("physical_quality_status", "")).strip().upper()

    if feasible:
        if physical_status == "WARN":
            return "PHYSICAL_WARN"
        return "FEASIBLE"

    if stage == "M1_MIRROR_TOPOLOGY_GATE" or stage == "M2_MIRROR_TOPOLOGY_GATE":
        return "TOPOLOGY_REJECT"

    if stage == "FULL_PHYSICAL_GATE":
        return "PHYSICAL_REJECT"

    if reason:
        return "OTHER_REJECT"

    return "UNKNOWN"


def _step11_find_selected_candidate_id(
    ctx: Any,
    rows: list[dict[str, Any]],
) -> str | None:
    """Tìm ID candidate tốt nhất được STEP11 chọn từ metrics hoặc history rows."""
    metrics = ctx.data.get("step11_selected_pair_metrics", {})
    if isinstance(metrics, dict):
        candidate_id = metrics.get("candidate_id")
        if candidate_id:
            return str(candidate_id)

    feasible_rows = [row for row in rows if _step11_bool(row.get("feasible", False))]
    if not feasible_rows:
        return None

    def rank_key(row: dict[str, Any]) -> tuple[float, float]:
        """Khóa xếp hạng kết hợp physical bucket và chief spot RMS."""
        physical_bucket = 0.0 if str(row.get("physical_quality_status", "")).upper() != "WARN" else 1.0
        rms = _step11_float(row.get("chief_centered_spot_RMS_mm"))
        return (physical_bucket, rms)

    best = min(feasible_rows, key=rank_key)
    return str(best.get("candidate_id", "")) or None


def _step11_orientation_sign_from_topology(topology: str) -> float:
    """Xác định dấu định hướng mặt phản xạ theo ray-facing topology."""
    label = str(topology).strip().upper()
    if label == "CONCAVE_RAY_FACING":
        return -1.0
    if label == "CONVEX_RAY_FACING":
        return 1.0
    return 1.0


def _step11_surface_curvature_maps(
    surface: PolySurface,
    topology: str,
) -> dict[str, Any]:
    """Tính Gaussian curvature và oriented mean curvature trên lưới local của surface."""
    half = np.asarray(surface.half_aperture, float)
    x = np.linspace(-half[0], half[0], 121)
    y = np.linspace(-half[1], half[1], 101)
    X, Y = np.meshgrid(x, y)
    xf = X.ravel()
    yf = Y.ravel()

    inside = aperture_contains_xy(
        surface.half_aperture,
        surface.aperture_polygon,
        xf,
        yf,
    ).reshape(X.shape)

    Z, ZX, ZY = surface.sag_slopes(xf, yf)
    Z = np.asarray(Z, float).reshape(X.shape)
    ZX = np.asarray(ZX, float).reshape(X.shape)
    ZY = np.asarray(ZY, float).reshape(X.shape)

    Z[~inside] = np.nan
    ZX[~inside] = np.nan
    ZY[~inside] = np.nan

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    ZXX = np.gradient(ZX, dx, axis=1)
    ZXY_1 = np.gradient(ZX, dy, axis=0)
    ZYX_1 = np.gradient(ZY, dx, axis=1)
    ZYY = np.gradient(ZY, dy, axis=0)
    ZXY = 0.5 * (ZXY_1 + ZYX_1)

    denom_g = np.power(1.0 + ZX * ZX + ZY * ZY, 2.0)
    KG = (ZXX * ZYY - ZXY * ZXY) / denom_g

    denom_h = 2.0 * np.power(1.0 + ZX * ZX + ZY * ZY, 1.5)
    H = ((1.0 + ZY * ZY) * ZXX - 2.0 * ZX * ZY * ZXY + (1.0 + ZX * ZX) * ZYY) / denom_h

    orientation_sign = _step11_orientation_sign_from_topology(topology)
    oriented_h = orientation_sign * H

    mask_bad = ~inside | ~np.isfinite(KG) | ~np.isfinite(oriented_h)
    KG = np.asarray(KG, float)
    oriented_h = np.asarray(oriented_h, float)
    KG[mask_bad] = np.nan
    oriented_h[mask_bad] = np.nan

    return {
        "X": X,
        "Y": Y,
        "KG": KG,
        "oriented_H": oriented_h,
        "orientation_sign": float(orientation_sign),
        "kg_min": float(np.nanmin(KG)) if np.isfinite(np.nanmin(KG)) else float("nan"),
        "kg_max": float(np.nanmax(KG)) if np.isfinite(np.nanmax(KG)) else float("nan"),
        "oh_min": float(np.nanmin(oriented_h)) if np.isfinite(np.nanmin(oriented_h)) else float("nan"),
        "oh_max": float(np.nanmax(oriented_h)) if np.isfinite(np.nanmax(oriented_h)) else float("nan"),
    }


def _render_step11_search_dashboard(
    step_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dựng ảnh dashboard 4 panel tổng hợp candidate search và footprint."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    png_path = step_dir / STEP11_SEARCH_DASHBOARD_PNG
    json_path = step_dir / STEP11_SEARCH_DASHBOARD_JSON

    if not rows:
        payload = {
            "status": "SKIPPED_NO_HISTORY_ROWS",
            "image": str(png_path),
        }
        write_json(json_path, payload)
        return payload

    stage_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}

    for row in rows:
        stage = str(row.get("rejection_stage", "")).strip() or "FEASIBLE_OR_UNSET"
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        label = _step11_topology_label(row)
        label_counts[label] = label_counts.get(label, 0) + 1

        reason_text = str(row.get("rejection_reason", "")).strip()
        if reason_text:
            for reason in [part.strip() for part in reason_text.split(",") if part.strip()]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    stage_items = sorted(stage_counts.items(), key=lambda item: item[1], reverse=True)
    label_items = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    fig = plt.figure(figsize=(16, 11), dpi=150)
    grid = fig.add_gridspec(2, 2, hspace=.28, wspace=.18)

    ax_stage = fig.add_subplot(grid[0, 0])
    ax_reason = fig.add_subplot(grid[0, 1])
    ax_labels = fig.add_subplot(grid[1, 0])
    ax_thumb = fig.add_subplot(grid[1, 1])

    ax_stage.barh(
        [name for name, _ in stage_items],
        [count for _, count in stage_items],
    )
    ax_stage.set_title("STEP11 rejection stage count")
    ax_stage.set_xlabel("candidate count")
    ax_stage.grid(True, alpha=.20)

    ax_reason.barh(
        [name for name, _ in top_reasons],
        [count for _, count in top_reasons],
    )
    ax_reason.set_title("Top rejection reasons")
    ax_reason.set_xlabel("candidate count")
    ax_reason.grid(True, alpha=.20)

    ax_labels.barh(
        [name for name, _ in label_items],
        [count for _, count in label_items],
    )
    ax_labels.set_title("Candidate bucket summary")
    ax_labels.set_xlabel("candidate count")
    ax_labels.grid(True, alpha=.20)

    thumb_path = step_dir / STEP11_RAY_FOOTPRINT_PNG
    if thumb_path.is_file():
        img = mpimg.imread(thumb_path)
        ax_thumb.imshow(img)
        ax_thumb.set_title("Existing STEP11 ray-validity footprint")
        ax_thumb.axis("off")
    else:
        ax_thumb.text(
            0.5,
            0.5,
            "Ray footprint image not found",
            ha="center",
            va="center",
            transform=ax_thumb.transAxes,
        )
        ax_thumb.set_title("Existing STEP11 ray-validity footprint")
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])

    fig.suptitle(
        "STEP 11 - Candidate Search Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "RENDERED",
        "image": str(png_path),
        "row_count": len(rows),
        "stage_counts": stage_counts,
        "reason_counts_top8": dict(top_reasons),
        "label_counts": label_counts,
    }
    write_json(json_path, payload)
    return payload


def _render_step11_physical_vs_rms(
    ctx: Any,
    step_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dựng scatter plot giữa physical fraction và chief-centered spot RMS theo bucket."""
    import matplotlib.pyplot as plt

    png_path = step_dir / STEP11_PHYSICAL_VS_RMS_PNG
    json_path = step_dir / STEP11_PHYSICAL_VS_RMS_JSON

    selected_candidate_id = _step11_find_selected_candidate_id(ctx, rows)

    groups: dict[str, list[tuple[float, float, str]]] = {
        "FEASIBLE": [],
        "PHYSICAL_WARN": [],
        "TOPOLOGY_REJECT": [],
        "PHYSICAL_REJECT": [],
        "OTHER_REJECT": [],
        "UNKNOWN": [],
    }

    for row in rows:
        x = _step11_float(row.get("physical_fraction"))
        y = _step11_float(row.get("chief_centered_spot_RMS_mm"))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        label = _step11_topology_label(row)
        candidate_id = str(row.get("candidate_id", ""))
        groups.setdefault(label, []).append((x, y, candidate_id))

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)

    style = {
        "FEASIBLE": dict(marker="o", s=34, alpha=.75, label="feasible / preferred"),
        "PHYSICAL_WARN": dict(marker="^", s=42, alpha=.80, label="physical warn < 0.90"),
        "TOPOLOGY_REJECT": dict(marker="x", s=42, alpha=.80, label="topology reject"),
        "PHYSICAL_REJECT": dict(marker="s", s=32, alpha=.70, label="physical reject"),
        "OTHER_REJECT": dict(marker="d", s=28, alpha=.70, label="other reject"),
        "UNKNOWN": dict(marker=".", s=24, alpha=.60, label="unknown"),
    }

    selected_xy = None

    for name, points in groups.items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, **style[name])

        if selected_candidate_id:
            for x, y, candidate_id in points:
                if candidate_id == selected_candidate_id:
                    selected_xy = (x, y)

    if selected_xy is not None:
        ax.scatter(
            [selected_xy[0]],
            [selected_xy[1]],
            marker="*",
            s=220,
            linewidths=1.2,
            label=f"selected: {selected_candidate_id}",
        )

    ax.axvline(0.90, linestyle="--", linewidth=1.2)
    ax.set_title("STEP11 physical fraction vs chief-centered RMS")
    ax.set_xlabel("physical fraction")
    ax.set_ylabel("chief-centered spot RMS [mm]")
    ax.grid(True, alpha=.20)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "RENDERED",
        "image": str(png_path),
        "selected_candidate_id": selected_candidate_id,
        "group_counts": {name: len(points) for name, points in groups.items()},
    }
    write_json(json_path, payload)
    return payload


def _render_step11_topology_maps(
    ctx: Any,
    step_dir: Path,
) -> dict[str, Any]:
    """Dựng 4 heatmap 2x2 gồm Gaussian curvature và oriented mean curvature cho M1/M2."""
    import matplotlib.pyplot as plt

    png_path = step_dir / STEP11_TOPOLOGY_MAPS_PNG
    json_path = step_dir / STEP11_TOPOLOGY_MAPS_JSON

    m1 = ctx.data.get("m1")
    m2 = ctx.data.get("m2")
    if not isinstance(m1, PolySurface) or not isinstance(m2, PolySurface):
        payload = {
            "status": "SKIPPED_NO_SELECTED_STEP11_SURFACES",
            "image": str(png_path),
        }
        write_json(json_path, payload)
        return payload

    authority = ctx.config.get("surface_fit", {}).get("mirror_topology_authority", {})
    m1_topology = str(authority.get("M1", "CONCAVE_RAY_FACING"))
    m2_topology = str(authority.get("M2", "CONVEX_RAY_FACING"))

    m1_maps = _step11_surface_curvature_maps(m1, m1_topology)
    m2_maps = _step11_surface_curvature_maps(m2, m2_topology)

    fig = plt.figure(figsize=(13.5, 10.5), dpi=150)
    grid = fig.add_gridspec(2, 2, hspace=.25, wspace=.18)

    panels = [
        ("M1 Gaussian curvature", m1_maps["X"], m1_maps["Y"], m1_maps["KG"], "KG [1/mm²]"),
        ("M1 oriented-H", m1_maps["X"], m1_maps["Y"], m1_maps["oriented_H"], "orientation_sign × H [1/mm]"),
        ("M2 Gaussian curvature", m2_maps["X"], m2_maps["Y"], m2_maps["KG"], "KG [1/mm²]"),
        ("M2 oriented-H", m2_maps["X"], m2_maps["Y"], m2_maps["oriented_H"], "orientation_sign × H [1/mm]"),
    ]

    summaries = {}

    for idx, (title, X, Y, Z, cbar_label) in enumerate(panels):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        image = ax.pcolormesh(X, Y, Z, shading="auto")
        fig.colorbar(image, ax=ax, fraction=.046, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel("local x [mm]")
        ax.set_ylabel("local y [mm]")
        ax.set_aspect("equal", adjustable="box")

        summaries[title] = {
            "min": float(np.nanmin(Z)) if np.isfinite(np.nanmin(Z)) else float("nan"),
            "max": float(np.nanmax(Z)) if np.isfinite(np.nanmax(Z)) else float("nan"),
        }

    fig.suptitle(
        "STEP 11 - Topology Maps of Selected O2 Pair",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "RENDERED",
        "image": str(png_path),
        "m1_topology": m1_topology,
        "m2_topology": m2_topology,
        "summaries": summaries,
    }
    write_json(json_path, payload)
    return payload


def _step11_plot_candidate_footprint(
    ax: Any,
    surface: PolySurface | None,
    hits: np.ndarray | None,
    field_index: np.ndarray,
    title: str,
) -> int:
    """Vẽ toàn bộ footprint hữu hạn trong local frame của candidate surface."""
    ax.set_title(
        title,
        fontsize=9.5,
        fontweight="bold",
    )

    if not isinstance(surface, PolySurface):
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Surface chưa được tạo\nở phase candidate này",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return 0

    boundary = aperture_boundary_xy(
        surface.half_aperture,
        surface.aperture_polygon,
    )
    closed_boundary = np.vstack([
        boundary,
        boundary[0],
    ])
    ax.plot(
        closed_boundary[:, 0],
        closed_boundary[:, 1],
        color="#111111",
        lw=1.2,
        label="clear aperture",
    )

    if hits is None:
        ax.text(
            0.5,
            0.5,
            "Footprint chưa được tính\nở phase candidate này",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xlabel("local x [mm]")
        ax.set_ylabel("local y [mm]")
        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.grid(alpha=.2)
        return 0

    hit_array = np.asarray(
        hits,
        float,
    )
    finite = np.all(
        np.isfinite(hit_array),
        axis=1,
    )
    finite_hits = hit_array[finite]

    if not len(finite_hits):
        ax.text(
            0.5,
            0.5,
            "Không có footprint hữu hạn",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xlabel("local x [mm]")
        ax.set_ylabel("local y [mm]")
        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.grid(alpha=.2)
        return 0

    local = (
        finite_hits
        - np.asarray(surface.center, float)
    ) @ np.asarray(surface.frame, float)

    if (
        field_index.shape == (len(hit_array),)
    ):
        colors = field_index[finite]
    else:
        colors = np.zeros(
            len(finite_hits),
            dtype=int,
        )

    ax.scatter(
        local[:, 0],
        local[:, 1],
        c=colors,
        cmap="turbo",
        s=2.0,
        alpha=.38,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_xlabel("local x [mm]")
    ax.set_ylabel("local y [mm]")
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.grid(alpha=.2)
    ax.legend(
        fontsize=7,
        loc="upper right",
    )
    ax.text(
        0.02,
        0.02,
        f"all finite points: {len(finite_hits):,}",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
    )
    return int(len(finite_hits))


def render_step11_candidate_sequence(
    ctx: Any,
    step_dir: Path,
) -> dict[str, Any]:
    """Dựng ảnh tuần tự cho toàn bộ STEP11 candidate từ snapshot đã lưu."""
    import matplotlib.pyplot as plt

    snapshot_root = (
        step_dir
        / STEP11_CANDIDATE_SNAPSHOT_DIR
    )
    image_root = (
        step_dir
        / STEP11_CANDIDATE_IMAGE_DIR
    )
    image_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_paths = sorted(
        snapshot_root.glob(
            "*/metadata.json"
        )
    )
    render_rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []

    if not metadata_paths:
        result = {
            "schema":
                "HUD_FAN_V5_5_STEP11_CANDIDATE_RENDER_MANIFEST_V1",
            "status":
                "SKIPPED_NO_CANDIDATE_SNAPSHOTS",
            "snapshot_count": 0,
            "rendered_count": 0,
            "warning_count": 0,
            "image_directory": str(image_root),
            "rows": [],
            "load_errors": [],
            "reran_optical_algorithms": False,
        }
        write_json(
            step_dir
            / STEP11_CANDIDATE_RENDER_MANIFEST_JSON,
            result,
        )
        return result

    data = ctx.data
    rays = data.get("rays", {})
    ray_origins = np.asarray(
        rays.get(
            "origins",
            np.empty((0, 3)),
        ),
        float,
    )
    field_index = np.asarray(
        rays.get(
            "field_index",
            np.empty(0, dtype=int),
        ),
        int,
    )
    visor_hits = np.asarray(
        data.get(
            "visor_hit",
            {},
        ).get(
            "point",
            np.empty((0, 3)),
        ),
        float,
    )

    for metadata_path in metadata_paths:
        figure = None
        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
            candidate_record = metadata.get(
                "candidate_record",
                {},
            )
            archive_index = int(
                metadata["archive_index"]
            )
            candidate_id = str(
                metadata["candidate_id"]
            )
            candidate_class = str(
                metadata.get(
                    "candidate_class",
                    "UNKNOWN",
                )
            )

            arrays: dict[str, np.ndarray] = {}
            arrays_file = metadata.get(
                "array_file"
            )
            if arrays_file:
                arrays_path = (
                    metadata_path.parent
                    / str(arrays_file)
                )
                with np.load(
                    arrays_path,
                    allow_pickle=False,
                ) as archive:
                    arrays = {
                        key: np.asarray(
                            archive[key]
                        ).copy()
                        for key in archive.files
                    }

            surface_payload = metadata.get(
                "surfaces",
                {},
            )
            m1_payload = surface_payload.get(
                "M1"
            )
            m2_payload = surface_payload.get(
                "M2"
            )
            m1 = (
                PolySurface.from_dict(m1_payload)
                if isinstance(m1_payload, dict)
                else None
            )
            m2 = (
                PolySurface.from_dict(m2_payload)
                if isinstance(m2_payload, dict)
                else None
            )

            figure = plt.figure(
                figsize=(18, 9),
                dpi=145,
            )
            grid = figure.add_gridspec(
                2,
                2,
                width_ratios=(1.25, 1.0),
                height_ratios=(1.0, 1.0),
                hspace=.28,
                wspace=.20,
            )
            ax3d = figure.add_subplot(
                grid[:, 0],
                projection="3d",
            )
            ax_m1 = figure.add_subplot(
                grid[0, 1]
            )
            ax_m2 = figure.add_subplot(
                grid[1, 1]
            )

            geometry: list[np.ndarray] = []

            visor = data.get("visor")
            if visor is not None:
                geometry.append(
                    _visor_surface(
                        ax3d,
                        visor,
                        "#5bc0eb",
                    )
                )

            if isinstance(m1, PolySurface):
                geometry.append(
                    _poly_surface(
                        ax3d,
                        m1,
                        "#ff9f1c",
                        "candidate M1",
                    )
                )

            if isinstance(m2, PolySurface):
                geometry.append(
                    _poly_surface(
                        ax3d,
                        m2,
                        "#e71d36",
                        "candidate M2",
                    )
                )

            display = data.get("display")
            if isinstance(display, PolySurface):
                geometry.append(
                    _poly_surface(
                        ax3d,
                        display,
                        "#2ec4b6",
                        "display",
                    )
                )

            physical_point_keys = sorted(
                key
                for key in arrays
                if key.startswith(
                    "physical_point_"
                )
            )
            trace_sequences: list[np.ndarray] = []

            if physical_point_keys:
                if (
                    ray_origins.ndim == 2
                    and ray_origins.shape[1:] == (3,)
                ):
                    trace_sequences.append(
                        ray_origins
                    )
                trace_sequences.extend(
                    arrays[key]
                    for key in physical_point_keys
                )
            else:
                for candidate_array in (
                    ray_origins,
                    visor_hits,
                    arrays.get("m1_hit_points"),
                    arrays.get("ci_m2_points_by_ray"),
                ):
                    if candidate_array is None:
                        continue
                    candidate_array = np.asarray(
                        candidate_array,
                        float,
                    )
                    if (
                        candidate_array.ndim == 2
                        and candidate_array.shape[1:] == (3,)
                    ):
                        trace_sequences.append(
                            candidate_array
                        )

            if len(trace_sequences) >= 2:
                traced = _trace_lines(
                    ax3d,
                    trace_sequences,
                    "#8338ec",
                    "saved candidate ray path",
                    STEP11_CANDIDATE_3D_MAX_RAYS,
                    .22,
                )
                geometry.append(
                    traced
                )

            ci_points = arrays.get(
                "ci_m2_points_by_ray"
            )
            if ci_points is not None:
                geometry.append(
                    _scatter(
                        ax3d,
                        ci_points,
                        "#d90429",
                        "M2 CI points",
                        3.0,
                        1800,
                        .35,
                    )
                )

            _set_labels(
                ax3d,
                (
                    f"STEP11 candidate {archive_index:06d}"
                    f" — {candidate_id}"
                ),
            )
            _equal(
                ax3d,
                geometry,
            )
            handles, labels = (
                ax3d.get_legend_handles_labels()
            )
            if handles:
                unique = dict(
                    zip(
                        labels,
                        handles,
                    )
                )
                ax3d.legend(
                    unique.values(),
                    unique.keys(),
                    fontsize=6.5,
                    loc="upper left",
                    framealpha=.72,
                )

            m1_hits = arrays.get(
                "physical_point_01",
                arrays.get(
                    "m1_hit_points"
                ),
            )
            m2_hits = arrays.get(
                "physical_point_02",
                arrays.get(
                    "ci_m2_points_by_ray"
                ),
            )

            m1_count = (
                _step11_plot_candidate_footprint(
                    ax_m1,
                    m1,
                    m1_hits,
                    field_index,
                    "M1 footprint — all finite rays",
                )
            )
            m2_count = (
                _step11_plot_candidate_footprint(
                    ax_m2,
                    m2,
                    m2_hits,
                    field_index,
                    "M2 footprint / CI cloud — all finite rays",
                )
            )

            feasible = bool(
                metadata.get(
                    "feasible",
                    False,
                )
            )
            rejection_stage = str(
                metadata.get(
                    "rejection_stage"
                )
                or "none"
            )
            rejection_reason = str(
                metadata.get(
                    "rejection_reason"
                )
                or "none"
            )
            title = (
                f"STEP 11 CANDIDATE SEQUENCE"
                f" — #{archive_index:06d}"
                f" — {candidate_id}\n"
                f"class={candidate_class}"
                f" | cycle={metadata.get('cycle')}"
                f" | move={metadata.get('move')}"
                f" | feasible={feasible}"
                f" | last_phase={metadata.get('last_phase')}"
            )
            figure.suptitle(
                title,
                fontsize=12,
                fontweight="bold",
            )
            figure.text(
                .01,
                .012,
                (
                    f"rejection_stage={rejection_stage}"
                    f" | rejection_reason={rejection_reason[:240]}"
                    " | source=saved candidate arrays"
                    " | optical algorithms were not rerun"
                ),
                fontsize=7.3,
                color="#343a40",
            )
            figure.subplots_adjust(
                left=.045,
                right=.975,
                bottom=.075,
                top=.87,
                wspace=.22,
                hspace=.30,
            )

            safe_candidate_id = "".join(
                character
                if character.isalnum()
                or character in ("_", "-")
                else "_"
                for character in candidate_id
            )
            stem = (
                f"{archive_index:06d}_"
                f"{safe_candidate_id}"
            )
            png_path = (
                image_root
                / f"{stem}.png"
            )
            temporary_png_path = (
                image_root
                / f".{stem}.tmp.png"
            )
            image_metadata_path = (
                image_root
                / f"{stem}.json"
            )

            figure.savefig(
                temporary_png_path,
                format="png",
                bbox_inches="tight",
            )
            plt.close(
                figure
            )
            figure = None
            temporary_png_path.replace(
                png_path
            )

            image_record = {
                "schema":
                    "HUD_FAN_V5_5_STEP11_CANDIDATE_IMAGE_V1",
                "status": "RENDERED",
                "archive_index": archive_index,
                "candidate_id": candidate_id,
                "candidate_class": candidate_class,
                "cycle": metadata.get("cycle"),
                "move": metadata.get("move"),
                "feasible": feasible,
                "rejection_stage":
                    metadata.get("rejection_stage"),
                "rejection_reason":
                    metadata.get("rejection_reason"),
                "last_phase":
                    metadata.get("last_phase"),
                "snapshot_metadata":
                    str(metadata_path),
                "snapshot_arrays": (
                    str(
                        metadata_path.parent
                        / str(arrays_file)
                    )
                    if arrays_file
                    else None
                ),
                "image": str(png_path),
                "m1_footprint_count": m1_count,
                "m2_footprint_count": m2_count,
                "three_d_maximum_ray_paths":
                    STEP11_CANDIDATE_3D_MAX_RAYS,
                "footprints_use_all_finite_points":
                    True,
                "reran_optical_algorithms":
                    False,
                "candidate_record":
                    candidate_record,
            }
            write_json(
                image_metadata_path,
                image_record,
            )
            render_rows.append(
                image_record
            )

        except Exception as exc:
            if figure is not None:
                plt.close(
                    figure
                )
            load_errors.append({
                "metadata": str(
                    metadata_path
                ),
                "error":
                    f"{type(exc).__name__}:{exc}",
            })

    result = {
        "schema":
            "HUD_FAN_V5_5_STEP11_CANDIDATE_RENDER_MANIFEST_V1",
        "status": (
            "RENDERED"
            if render_rows
            else "VISUALIZATION_WARNING"
        ),
        "snapshot_count": int(
            len(metadata_paths)
        ),
        "rendered_count": int(
            len(render_rows)
        ),
        "warning_count": int(
            len(load_errors)
        ),
        "image_directory": str(
            image_root
        ),
        "rows": render_rows,
        "load_errors": load_errors,
        "three_d_maximum_ray_paths":
            STEP11_CANDIDATE_3D_MAX_RAYS,
        "footprints_use_all_finite_points":
            True,
        "reran_optical_algorithms":
            False,
    }
    write_json(
        step_dir
        / STEP11_CANDIDATE_RENDER_MANIFEST_JSON,
        result,
    )
    return result


def render_step11_enhanced_visuals(
    ctx: Any,
    step_dir: Path,
) -> dict[str, Any]:
    """Render đầy đủ bộ ảnh STEP11 để debug candidate search."""
    rows = _step11_read_history_rows(step_dir)

    records = {
        "status": "RENDERED",
        "history_row_count": len(rows),
        "search_dashboard": _render_step11_search_dashboard(step_dir, rows),
        "physical_vs_rms": _render_step11_physical_vs_rms(ctx, step_dir, rows),
        "topology_maps": _render_step11_topology_maps(ctx, step_dir),
        "candidate_sequence": render_step11_candidate_sequence(
            ctx,
            step_dir,
        ),
        "ray_footprint_expected": str(step_dir / STEP11_RAY_FOOTPRINT_PNG),
        "surface_cad_expected": str(step_dir / STEP11_SURFACE_CAD_PNG),
    }
    return records


def render_surface_cad_view(ctx: Any, step: int) -> dict[str, Any]:
    """Vẽ riêng các mặt M1/M2 đã thực sự tồn tại trong saved state, không refit và không ray-trace."""
    if step not in SURFACE_CAD_STEPS:
        return {
            "status": "SKIPPED_STEP_HAS_NO_REQUESTED_SURFACE_CHANGE",
            "step": int(step),
        }

    data = ctx.data
    step_dir = ctx.step_dir(step)
    metadata_path = step_dir / f"{step:02d}_SURFACE_CAD_VIEW.json"

    rows: list[dict[str, Any]] = []

    if step == 10:
        record = {
            "schema": "HUD_FAN_V5_5_SURFACE_CAD_VIEW_V1",
            "status": "SKIPPED_NO_FITTED_SURFACE",
            "step": 10,
            "reason": (
                "STEP_10_STORES_M1_CI_POINT_NORMAL_CLOUD_ONLY; "
                "M1_REMAINS_THE_PLANAR_SEED_UNTIL_STEP_11_FITS_ORDER_2"
            ),
            "reran_optical_algorithms": False,
            "data_source": "SAVED_CONTEXT_ONLY",
            "algorithm_source_manifest_sha256": data.get(
                "algorithm_source_manifest", {}
            ).get("manifest_sha256"),
        }
        write_json(metadata_path, record)
        return {
            "status": record["status"],
            "metadata": str(metadata_path),
            "reason": record["reason"],
        }

    if step == 11:
        m1 = data.get("m1")
        m2 = data.get("m2")
        if (isinstance(m1, PolySurface) and isinstance(m2, PolySurface)
                and "step11_selected_pair_metrics" in data):
            rows.append({
                "label": "STEP 11 - selected system-aware O2 pair",
                "order": 2,
                "source": "ctx.data['m1']/['m2'] selected by STEP11 feasible-first O2 search",
                "archive_class": "SELECTED_INCUMBENT",
                "m1": m1,
                "m2": m2,
            })
    elif step == 12:
        m1 = data.get("m1") if "fit_m1_order2" in data else None
        m2 = data.get("m2") if "fit_m2_order2" in data else None
        if isinstance(m1, PolySurface) or isinstance(m2, PolySurface):
            rows.append({
                "label": "STEP 12 - joint-refined Fan O2 surfaces",
                "order": 2,
                "source": "ctx.data after STEP12 refinement of the STEP11-selected O2 pair",
                "archive_class": "CURRENT_OR_PARTIAL_STEP_STATE",
                "m1": m1 if isinstance(m1, PolySurface) else None,
                "m2": m2 if isinstance(m2, PolySurface) else None,
            })
    elif step == 16:
        incumbents = data.get("_ci_incumbents")
        if isinstance(incumbents, dict):
            m1 = data.get("m1")
            m2 = data.get("m2")
            if isinstance(m1, PolySurface) and isinstance(m2, PolySurface):
                rows.append({
                    "label": "STEP 16 - selected signed-rho O2 incumbent",
                    "order": 2,
                    "source": "ctx.data selected by STEP 16 before _rho_frontier publication",
                    "archive_class": "SELECTED_INCUMBENT",
                    "m1": m1,
                    "m2": m2,
                })
    elif step == 17:
        incumbents = data.get("_ci_incumbents", {})
        feasible = incumbents.get("feasible", {}) if isinstance(incumbents, dict) else {}
        admissible = incumbents.get("admissible", {}) if isinstance(incumbents, dict) else {}
        for order in (2, 3, 4, 5):
            state = feasible.get(order)
            archive_key = "feasible"
            archive_class = "FEASIBLE"
            if state is None:
                state = admissible.get(order)
                archive_key = "admissible"
                archive_class = "ADMISSIBLE_RESTORATION"
            if not isinstance(state, dict):
                continue
            m1 = state.get("m1")
            m2 = state.get("m2")
            if not isinstance(m1, PolySurface) or not isinstance(m2, PolySurface):
                continue
            rows.append({
                "label": f"STEP 17 - archived O{order}",
                "order": int(order),
                "source": f"ctx.data['_ci_incumbents']['{archive_key}'][{order}]",
                "archive_class": archive_class,
                "m1": m1,
                "m2": m2,
            })
    elif step == 23:
        m1 = data.get("m1")
        m2 = data.get("m2")
        if isinstance(m1, PolySurface) and isinstance(m2, PolySurface):
            rows.append({
                "label": "STEP 23 - current O5 surface state",
                "order": 5,
                "source": "ctx.data current STEP 23 state; may be partial if execution failed",
                "archive_class": "CURRENT_OR_PARTIAL_STEP_STATE",
                "m1": m1,
                "m2": m2,
            })

    if not rows:
        record = {
            "schema": "HUD_FAN_V5_5_SURFACE_CAD_VIEW_V1",
            "status": "SKIPPED_NO_FITTED_SURFACE_IN_SAVED_STATE",
            "step": int(step),
            "reran_optical_algorithms": False,
            "data_source": "SAVED_CONTEXT_ONLY",
            "algorithm_source_manifest_sha256": data.get(
                "algorithm_source_manifest", {}
            ).get("manifest_sha256"),
        }
        write_json(metadata_path, record)
        return {
            "status": record["status"],
            "metadata": str(metadata_path),
        }

    import matplotlib.pyplot as plt

    def surface_mesh(surface: PolySurface) -> dict[str, Any]:
        """Tạo lưới bề mặt hiển thị từ biên khẩu độ và đa thức."""
        half = np.asarray(surface.half_aperture, float)
        x = np.linspace(-half[0], half[0], 61)
        y = np.linspace(-half[1], half[1], 49)
        X, Y = np.meshgrid(x, y)
        xf = X.ravel()
        yf = Y.ravel()

        inside = aperture_contains_xy(
            surface.half_aperture,
            surface.aperture_polygon,
            xf,
            yf,
        )

        raw_conic_argument = (
            1.0
            - (1.0 + float(surface.conic))
            * float(surface.curvature) ** 2
            * (xf * xf + yf * yf)
        )

        domain = np.isfinite(raw_conic_argument) & (
            raw_conic_argument >= 0.0
        )

        z, _, _ = surface.sag_slopes(
            xf,
            yf,
        )

        world = surface.point(
            xf,
            yf,
        )

        good = (
            np.asarray(inside, bool)
            & domain
            & np.isfinite(z)
            & np.all(np.isfinite(world), axis=1)
        )

        local_z = np.asarray(
            z,
            float,
        ).copy()

        world = np.asarray(
            world,
            float,
        ).copy()

        local_z[~good] = np.nan
        world[~good] = np.nan

        boundary_xy = aperture_boundary_xy(
            surface.half_aperture,
            surface.aperture_polygon,
        )

        boundary_z, _, _ = surface.sag_slopes(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
        )

        boundary_world = surface.point(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
        )

        return {
            "X": X,
            "Y": Y,
            "Z": local_z.reshape(X.shape),
            "world": world.reshape(*X.shape, 3),
            "good_world": world[good],
            "boundary_xy": boundary_xy,
            "boundary_z": np.asarray(boundary_z, float),
            "boundary_world": np.asarray(boundary_world, float),
            "valid_grid_count": int(np.count_nonzero(good)),
        }

    def equal_axes(
        ax: Any,
        clouds: list[np.ndarray],
    ) -> None:
        """Thiết lập tỷ lệ khung trục bằng nhau cho đồ thị không gian 3D."""
        finite_clouds = [
            np.asarray(cloud, float).reshape(-1, 3)
            for cloud in clouds
            if cloud is not None and np.asarray(cloud).size
        ]

        finite_clouds = [
            cloud[np.all(np.isfinite(cloud), axis=1)]
            for cloud in finite_clouds
        ]

        finite_clouds = [
            cloud
            for cloud in finite_clouds
            if len(cloud)
        ]

        if not finite_clouds:
            return

        cloud = np.vstack(
            finite_clouds
        )

        lower = np.min(
            cloud,
            axis=0,
        )

        upper = np.max(
            cloud,
            axis=0,
        )

        center = (
            lower + upper
        ) / 2.0

        radius = max(
            float(
                np.max(
                    upper - lower
                )
            ) / 2.0,
            1e-6,
        )

        ax.set_xlim(
            center[0] - radius,
            center[0] + radius,
        )

        ax.set_ylim(
            center[1] - radius,
            center[1] + radius,
        )

        ax.set_zlim(
            center[2] - radius,
            center[2] + radius,
        )

        ax.set_box_aspect(
            (1, 1, 1)
        )

    fig = plt.figure(
        figsize=(
            19,
            max(
                6.0,
                5.2 * len(rows),
            ),
        ),
        dpi=140,
    )

    grid = fig.add_gridspec(
        len(rows),
        3,
        width_ratios=(
            1.0,
            1.0,
            1.15,
        ),
        hspace=.30,
        wspace=.12,
    )

    model_records: list[
        dict[str, Any]
    ] = []

    for row_index, row in enumerate(
        rows
    ):
        meshes: dict[
            str,
            dict[str, Any] | None,
        ] = {
            "M1": None,
            "M2": None,
        }

        surfaces = {
            "M1": row["m1"],
            "M2": row["m2"],
        }

        for column_index, name in enumerate(
            (
                "M1",
                "M2",
            )
        ):
            ax = fig.add_subplot(
                grid[
                    row_index,
                    column_index,
                ],
                projection="3d",
            )

            surface = surfaces[
                name
            ]

            if not isinstance(
                surface,
                PolySurface,
            ):
                ax.set_axis_off()

                ax.text2D(
                    0.5,
                    0.5,
                    (
                        f"{name} has no fitted surface "
                        "in this saved state"
                    ),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

                continue

            mesh = surface_mesh(
                surface
            )

            meshes[
                name
            ] = mesh

            ax.plot_surface(
                mesh["X"],
                mesh["Y"],
                mesh["Z"],
                cmap="viridis",
                alpha=.92,
                linewidth=.16,
                edgecolor="#4f4f4f",
                shade=False,
            )

            boundary_xy = mesh[
                "boundary_xy"
            ]

            boundary_z = mesh[
                "boundary_z"
            ]

            closed_xy = np.vstack(
                [
                    boundary_xy,
                    boundary_xy[0],
                ]
            )

            closed_z = np.r_[
                boundary_z,
                boundary_z[0],
            ]

            ax.plot(
                closed_xy[:, 0],
                closed_xy[:, 1],
                closed_z,
                color="#111111",
                lw=1.4,
            )

            finite_z = mesh["Z"][
                np.isfinite(
                    mesh["Z"]
                )
            ]

            sag_min = (
                float(
                    np.min(
                        finite_z
                    )
                )
                if len(finite_z)
                else float("nan")
            )

            sag_max = (
                float(
                    np.max(
                        finite_z
                    )
                )
                if len(finite_z)
                else float("nan")
            )

            ax.set_title(
                f"{row['label']} | {name} local sag view\n"
                f"sag {sag_min:.6g}..{sag_max:.6g} mm | "
                f"c={float(surface.curvature):.6g}/mm | "
                f"K={float(surface.conic):.6g}",
                fontsize=9.5,
            )

            ax.set_xlabel(
                "local x [mm]"
            )

            ax.set_ylabel(
                "local y [mm]"
            )

            ax.set_zlabel(
                "sag z [mm]"
            )

            ax.view_init(
                elev=25,
                azim=-55,
            )

            ax.set_proj_type(
                "ortho"
            )

            model_records.append({
                "state_label": row["label"],
                "order": row["order"],
                "archive_class": row["archive_class"],
                "surface": name,
                "source": row["source"],
                "valid_grid_count": mesh["valid_grid_count"],
                "prescription": surface.to_dict(),
            })

        ax_global = fig.add_subplot(
            grid[
                row_index,
                2,
            ],
            projection="3d",
        )

        global_clouds: list[
            np.ndarray
        ] = []

        for name, color in (
            (
                "M1",
                "#ff9f1c",
            ),
            (
                "M2",
                "#e71d36",
            ),
        ):
            surface = surfaces[
                name
            ]

            mesh = meshes[
                name
            ]

            if (
                not isinstance(
                    surface,
                    PolySurface,
                )
                or mesh is None
            ):
                continue

            world = mesh[
                "world"
            ]

            ax_global.plot_surface(
                world[:, :, 0],
                world[:, :, 1],
                world[:, :, 2],
                color=color,
                alpha=.72,
                linewidth=.18,
                edgecolor=color,
                shade=False,
            )

            boundary_world = mesh[
                "boundary_world"
            ]

            closed_world = np.vstack(
                [
                    boundary_world,
                    boundary_world[0],
                ]
            )

            ax_global.plot(
                closed_world[:, 0],
                closed_world[:, 1],
                closed_world[:, 2],
                color="#111111",
                lw=1.2,
            )

            center = np.asarray(
                surface.center,
                float,
            )

            ax_global.text(
                center[0],
                center[1],
                center[2],
                name,
            )

            global_clouds.append(
                mesh[
                    "good_world"
                ]
            )

        ax_global.set_title(
            (
                f"{row['label']} | "
                "global M1/M2 only"
            ),
            fontsize=9.5,
        )

        ax_global.set_xlabel(
            "X [mm]"
        )

        ax_global.set_ylabel(
            "Y [mm]"
        )

        ax_global.set_zlabel(
            "Z [mm]"
        )

        ax_global.view_init(
            elev=22,
            azim=-55,
        )

        ax_global.set_proj_type(
            "ortho"
        )

        equal_axes(
            ax_global,
            global_clouds,
        )

    fig.suptitle(
        f"STEP {step:02d} - SAVED-SURFACE CAD VIEW\n"
        "M1/M2 only; no rays, no visor, no display, "
        "no refit, no optical re-execution",
        fontsize=12,
        fontweight="bold",
        y=.985,
    )

    fig.subplots_adjust(
        left=.035,
        right=.985,
        bottom=.04,
        top=.84,
        hspace=.34,
        wspace=.14,
    )

    png = (
        step_dir
        / f"{step:02d}_SURFACE_CAD_VIEW.png"
    )

    temporary_png = (
        step_dir
        / f".{step:02d}_SURFACE_CAD_VIEW.tmp.png"
    )

    fig.savefig(
        temporary_png,
        format="png",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    temporary_png.replace(
        png
    )

    record = {
        "schema":
            "HUD_FAN_V5_5_SURFACE_CAD_VIEW_V1",

        "status":
            "RENDERED",

        "step":
            int(step),

        "image":
            str(png),

        "row_count":
            len(rows),

        "rows": [
            {
                "label":
                    row["label"],

                "order":
                    row["order"],

                "source":
                    row["source"],

                "archive_class":
                    row["archive_class"],

                "has_m1":
                    isinstance(
                        row["m1"],
                        PolySurface,
                    ),

                "has_m2":
                    isinstance(
                        row["m2"],
                        PolySurface,
                    ),
            }
            for row in rows
        ],

        "models":
            model_records,

        "coordinate_views": [
            (
                "LOCAL_X_Y_SAG_AUTOSCALED_AXES_"
                "FOR_SHAPE_INSPECTION_WITH_TRUE_MM_COORDINATES"
            ),
            (
                "GLOBAL_X_Y_Z_EQUAL_AXIS_SCALE_"
                "M1_M2_ONLY"
            ),
        ],

        "reran_optical_algorithms":
            False,

        "refit_performed":
            False,

        "ray_trace_performed":
            False,

        "data_source":
            "SAVED_CONTEXT_ONLY",

        "algorithm_source_manifest_sha256":
            data.get(
                "algorithm_source_manifest",
                {},
            ).get(
                "manifest_sha256"
            ),

        "rendered_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    write_json(
        metadata_path,
        record,
    )

    return {
        "status":
            "RENDERED",

        "image":
            str(png),

        "metadata":
            str(metadata_path),

        "row_count":
            len(rows),
    }


def _render_artifact_fallback(run_dir: Path, step: int) -> dict[str, Any] | None:
    """Tạo ảnh diagnostic từ các file CSV hoặc JSON của STEP cho run cũ không có state pkl."""
    step_dir = run_dir / f"STEP_{step:02d}"
    if not step_dir.is_dir():
        return None

    if step == 11:
        history_csv = step_dir / STEP11_HISTORY_CSV
        if history_csv.is_file():
            try:
                rows = _step11_read_history_rows(step_dir)
                diag_res = _render_step11_search_dashboard(step_dir, rows)
                if diag_res.get("status") == "RENDERED":
                    return diag_res
            except Exception:
                pass

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate_csv = step_dir / f"{step:02d}_FIT_QUALITY_GATES.csv"
    fail_json = run_dir / f"RUN_FAILURE_STEP_{step:02d}.json"
    if not fail_json.exists():
        fail_json = step_dir / f"RUN_FAILURE_STEP_{step:02d}.json"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252526")

    rendered = False
    if gate_csv.is_file():
        try:
            with open(gate_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                labels = [f"{r.get('surface', '')} {r.get('check', '')}" for r in rows]
                statuses = [r.get("status", "") for r in rows]
                colors = ["#4ec9b0" if s == "PASS" else "#f44747" for s in statuses]
                y_pos = np.arange(len(labels))
                ax.barh(y_pos, [1] * len(labels), color=colors, height=0.6)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, color="white", fontsize=8)
                ax.set_xticks([])
                for i, r in enumerate(rows):
                    act = str(r.get("actual", ""))[:8]
                    lim = str(r.get("limit", ""))
                    stat = str(r.get("status", ""))
                    ax.text(0.05, i, f"Status: {stat} | Actual: {act} | Limit: {lim}",
                            va="center", color="white", fontweight="bold", fontsize=8)
                ax.set_title(f"STEP {step:02d} - Quality Gates (Artifact Diagnostic)", color="white", fontsize=12)
                rendered = True
        except Exception:
            pass

    if not rendered and fail_json.is_file():
        try:
            fail_data = json.loads(fail_json.read_text(encoding="utf-8"))
            msg = fail_data.get("message", "FAILED")
            exc_type = fail_data.get("exception_type", "Exception")
            ax.text(0.1, 0.7, f"STEP {step:02d} FAILED", color="#f44747", fontsize=15, fontweight="bold")
            ax.text(0.1, 0.5, f"Exception: {exc_type}", color="white", fontsize=11)
            ax.text(0.1, 0.25, f"Message:\n{msg}", color="#dcdcaa", fontsize=9, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            rendered = True
        except Exception:
            pass

    if not rendered:
        plt.close(fig)
        return None

    target_png = step_dir / f"{step:02d}_SAVED_ARTIFACT_DIAGNOSTIC.png"
    fig.tight_layout()
    fig.savefig(target_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    meta = {
        "render_mode": "ARTIFACT_ONLY_FALLBACK",
        "full_3d_state_available": False,
        "reason": "LEGACY_RUN_HAS_NO_FAILURE_RENDER_STATE",
        "step": step,
        "image": str(target_png),
    }
    write_json(step_dir / f"{step:02d}_SAVED_ARTIFACT_DIAGNOSTIC.json", meta)
    return meta


def _render_precheck(root: Path, common_ctx: Any, force: bool = False) -> dict[str, Any]:
    """Dựng lại ảnh so sánh RMS và cập nhật kết quả precheck planar từ dữ liệu đã lưu."""
    candidates_json = root / "07_ALL_PLANAR_START_CANDIDATES.json"
    target_img = root / "07_ALL_PLANAR_START_RMS_COMPARISON.png"
    target_meta = root / "07_ALL_PLANAR_START_RMS_COMPARISON.json"

    if not candidates_json.exists():
        return {
            "status": "SKIPPED_NO_PRECHECK_JSON",
            "message": f"Precheck data not found: {candidates_json}",
        }

    if not force and target_img.exists() and target_meta.exists():
        return {
            "status": "REUSED_EXISTING",
            "image": str(target_img),
            "metadata": str(target_meta),
        }

    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])

    solver_cfg = common_ctx.config.get("solver", {})
    limit_mm = float(solver_cfg.get("planar_paper_spot_rms_max_mm", 2.6))
    manifest_sha = common_ctx.data.get("algorithm_source_manifest", {}).get("manifest_sha256")

    precheck_view = render_planar_seed_precheck(
        root,
        rows,
        limit_mm,
        manifest_sha,
    )

    multistart_precheck = root / "07_MULTISTART_PRECHECK.json"
    if multistart_precheck.exists():
        precheck_data = json.loads(multistart_precheck.read_text(encoding="utf-8"))
        precheck_data["planar_paper_spot_rms_comparison_view"] = precheck_view
        precheck_data["visualization_render_mode"] = "POSTHOC_FROM_SAVED_PRECHECK_DATA"
        write_json(multistart_precheck, precheck_data)

    return precheck_view


def _discover_design_runs(root: Path) -> list[Path]:
    """Tìm kiếm tất cả thư mục kết quả baseline và geometry refinement trong run."""
    runs: list[Path] = []
    baseline_dirs = sorted(
        [p for p in root.glob("SEED_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    runs.extend(baseline_dirs)

    geom_root = root / "GEOMETRY_REFINEMENT"
    if geom_root.is_dir():
        geom_selection = root / "GEOMETRY_REFINEMENT_SELECTION.json"
        if geom_selection.exists():
            try:
                sel_data = json.loads(geom_selection.read_text(encoding="utf-8"))
                for entry in sel_data.get("selected", []):
                    cand = entry.get("candidate")
                    cand_dir = geom_root / f"SEED_{cand:03d}"
                    if cand_dir.is_dir() and cand_dir not in runs:
                        runs.append(cand_dir)
            except Exception:
                pass
        for p in sorted(geom_root.glob("SEED_*")):
            if p.is_dir() and p not in runs:
                runs.append(p)

    return runs


def _render_container_steps(
    run_dir: Path,
    allow_source_mismatch: bool,
    force: bool = False,
    start_step: int = 0,
    end_step: int = 26,
) -> dict[str, Any]:
    """Dựng ảnh cho một thư mục container (COMMON hoặc SEED_xxx) bằng cách quét tất cả STEP thực sự có dữ liệu."""
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "status": "COMPLETE",
        "steps": [],
    }

    first_ctx: Any = None

    for step in range(start_step, end_step + 1):
        step_rec: dict[str, Any] = {"step": step}
        step_dir = run_dir / f"STEP_{step:02d}"
        fail_file = run_dir / f"RUN_FAILURE_STEP_{step:02d}.json"

        state_path, state_source = _find_state(run_dir, step)

        if state_path is None:
            # Không có file state pkl.gz. Kiểm tra xem STEP có folder hoặc fail file không.
            if step_dir.is_dir() or fail_file.is_file():
                # Có artifact nhưng thiếu state pkl -> Thử render artifact fallback
                artifact_res = _render_artifact_fallback(run_dir, step)
                if artifact_res is not None:
                    step_rec["status"] = "ARTIFACT_ONLY_RENDERED"
                    step_rec["details"] = artifact_res
                else:
                    step_rec["status"] = "ARTIFACTS_EXIST_BUT_FULL_STATE_MISSING"
            else:
                step_rec["status"] = "NOT_EXECUTED"

            report["steps"].append(step_rec)
            continue

        # Đã tìm thấy file state
        is_partial_fail = False
        if state_source == "RENDER_STATE":
            meta_path = state_path.parent / "metadata.json"
            if meta_path.is_file():
                try:
                    s_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if s_meta.get("state_kind") == "PARTIAL_FAILURE_STATE" or s_meta.get("execution_status") == "ALGORITHM_FAILED":
                        is_partial_fail = True
                except Exception:
                    pass

        if not force and _is_step_rendered(run_dir, step):
            step_rec["status"] = "REUSED_EXISTING"
            report["steps"].append(step_rec)
            continue

        try:
            ctx = _load_state(state_path, run_dir)
            if first_ctx is None:
                first_ctx = ctx
                report["manifest"] = _assert_manifest_matches(ctx, allow_source_mismatch)

            records: dict[str, Any] = {}

            try:
                records["visualization_3d"] = render_step(ctx, step)
            except Exception as exc:
                records["visualization_3d"] = {
                    "status": "VISUALIZATION_WARNING",
                    "message": str(exc),
                }

            try:
                records["surface_ray_footprint"] = render_surface_ray_view(ctx, step)
            except Exception as exc:
                records["surface_ray_footprint"] = {
                    "status": "VISUALIZATION_WARNING",
                    "message": str(exc),
                }

            if step in SURFACE_CAD_STEPS:
                try:
                    records["surface_cad"] = render_surface_cad_view(ctx, step)
                except Exception as exc:
                    records["surface_cad"] = {
                        "status": "VISUALIZATION_WARNING",
                        "message": str(exc),
                    }

            if step == 11:
                try:
                    records["step11_search_diagnostic"] = render_step11_enhanced_visuals(
                        ctx,
                        step_dir,
                    )
                except Exception as exc:
                    records["step11_search_diagnostic"] = {
                        "status": "VISUALIZATION_WARNING",
                        "message": str(exc),
                    }

            if step == 20:
                try:
                    records["spot_evolution"] = render_spot_evolution(ctx)
                except Exception as exc:
                    records["spot_evolution"] = {
                        "status": "VISUALIZATION_WARNING",
                        "message": str(exc),
                    }

            _update_step_summary(run_dir, step, records)

            if is_partial_fail:
                step_rec["status"] = "RENDERED_FROM_PARTIAL_FAILURE_STATE"
            elif state_source == "RENDER_STATE":
                step_rec["status"] = "RENDERED_FROM_COMPLETED_STATE"
            else:
                step_rec["status"] = "RENDERED_FROM_CHECKPOINT"

            step_rec["records"] = records
        except Exception as exc:
            step_rec["status"] = "ERROR"
            step_rec["message"] = str(exc)

        report["steps"].append(step_rec)

    return report


def render_all_deferred(
    root: Path,
    force: bool = False,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Điều phối toàn bộ quá trình render hậu kỳ cho precheck, common và mọi design."""
    print(f"=== POST-HOC RENDERING: {root} ===", flush=True)

    common_dir = root / "COMMON"
    common_ctx: Any = None
    common_report: dict[str, Any] | None = None

    if common_dir.is_dir():
        print("Rendering COMMON container (STEP 00-06) ...", flush=True)
        common_report = _render_container_steps(
            common_dir,
            allow_source_mismatch,
            force=force,
            start_step=0,
            end_step=6,
        )
        # Nạp common_ctx để render precheck nếu có
        for step in (6, 0):
            st_path, _ = _find_state(common_dir, step)
            if st_path is not None:
                try:
                    common_ctx = _load_state(st_path, common_dir)
                    break
                except Exception:
                    pass

    precheck_report: dict[str, Any] = {"status": "NOT_AVAILABLE"}
    if common_ctx is not None:
        precheck_report = _render_precheck(root, common_ctx, force=force)

    design_runs = _discover_design_runs(root)
    print(f"Found {len(design_runs)} design run directories to render.", flush=True)

    designs_reports: list[dict[str, Any]] = []
    for idx, d_dir in enumerate(design_runs, 1):
        print(f"[{idx:02d}/{len(design_runs):02d}] Rendering design {d_dir.name} ...", flush=True)
        rep = _render_container_steps(
            d_dir,
            allow_source_mismatch,
            force=force,
            start_step=7,
            end_step=26,
        )
        designs_reports.append(rep)

    full_report: dict[str, Any] = {
        "schema": "HUD_FAN_V5_5_DEFERRED_VISUALIZATION_REPORT_V1",
        "root": str(root),
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "reran_optical_algorithms": False,
        "precheck": precheck_report,
        "common": common_report,
        "designs": designs_reports,
    }

    report_path = root / "DEFERRED_VISUALIZATION_REPORT.json"
    write_json(report_path, full_report)
    print(f"Saved deferred visualization report to {report_path}", flush=True)
    return full_report


def main() -> None:
    """Phân tích tham số dòng lệnh và thực thi render hậu kỳ cho multistart run."""
    parser = argparse.ArgumentParser(
        description="Render saved multistart checkpoints and states post-hoc without rerunning optics"
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Multistart root; default = latest timestamped run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already existing images",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Allow rendering with a different source revision for inspection only",
    )
    args = parser.parse_args()

    multistart_root = args.run.resolve() if args.run is not None else _latest_multistart_root()
    render_all_deferred(
        multistart_root,
        force=args.force,
        allow_source_mismatch=args.allow_source_mismatch,
    )


if __name__ == "__main__":
    main()
