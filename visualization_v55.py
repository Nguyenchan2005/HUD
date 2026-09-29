"""Vẽ hình 3D và bằng chứng số riêng cho thuật toán của từng STEP đã lưu."""

from __future__ import annotations

import gzip
import json
import shutil
import csv
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from core import ChebVisor, PolySurface, build_final_scorecard, write_csv, write_json


ROOT = Path(__file__).resolve().parent
MANUAL_ROOT = ROOT / "manual_flow_v5_5"


def _algorithm_manifest_hash(ctx: Any) -> str | None:
    """Gắn metadata ảnh với đúng revision source đã sinh dữ liệu của run."""
    return ctx.data.get("algorithm_source_manifest", {}).get("manifest_sha256")


STEP_FOCUS = {
    0: ("Input authority", "Raw visor samples and the P1–P8 packaging envelope"),
    1: ("Visor surrogate fit", "Measured visor points versus the differentiable fitted surface"),
    2: ("Virtual-image specification", "VID, D6 and FOV define the target VI plane"),
    3: ("Configured sampled fields", "The configured angular field grid mapped onto the VI plane"),
    4: ("Configurable primary pupil grid", "A configurable uniform grid covers the current eye-box"),
    5: ("Characteristic-ray sampling", "The configured fields use 49 samples at each configured pupil"),
    6: ("Visor reflection bundle", "Eye/VI rays intersect the fitted visor and reflect toward M1"),
    7: ("Physical planar seed", "Selected M1/M2/display planes, ordered first hits and signed MF2"),
    8: ("Reference separation", "Dynamic Fan references are distinct from the fixed distortion grid"),
    9: ("Planar conjugate targets", "Step-One targets are symmetric conjugates through planar M2"),
    10: ("M1 point-by-point CI", "Nearest-ray selection and nearest previously built tangent plane"),
    11: ("System-aware O2 pair construction", "Search M1 O2 under hard shape/CI gates, rebuild and fit M2 per candidate, then rank only full-physical feasible pairs"),
    12: ("Joint order-2 actual-surface refinement", "The STEP11-selected M1/M2 O2 pair is locally refined while the same hard mirror-shape gate remains active"),
    13: ("Enter Fan Step Two", "The planar shortcut is disabled before stationary-path redesign"),
    14: ("Fermat stationary targets", "Every Q2 target must satisfy both in-surface OP gradients"),
    15: ("Order-2 rho search policy", "Signed coarse candidates and coarse-to-fine full-physical ranking are declared before evaluation"),
    16: ("Order-2 coarse-to-fine rho search", "All rho candidates use the same post-STEP14 baseline; only the best full-physical O2 state continues"),
    17: ("Nested CI basis ladder 2→5", "Level 3 retains A22; Fermat/CI must recover 100% physical trace before STEP 18"),
    18: ("Order-5 starting point", "The fifth-order CI prescription is frozen before local optimization"),
    19: ("Merit separation", "MF1/MF2/MF3 form Fan core; engineering constraints remain external"),
    20: ("Fan MF1", "All mapping residuals and the outer sum of configured pupil RMS terms"),
    21: ("Fan MF2", "Oriented A–Q–P triangle gives the signed 3-D obscuration merit"),
    22: ("Fixed distortion rule", "The user grid remains fixed and is not a Fan endpoint"),
    23: ("Full order-5 filter DLSQ", "Fan objective steps plus bounded engineering-restoration trials with explicit rejection logs"),
    24: ("Forward optical evaluation", "Only complete physical bundles in front of each pupil certify VI/FOV/distortion"),
    25: ("Final scorecard", "Every declared hard gate is shown without converting failures to passes"),
    26: ("Traceable export", "Final prescription, surfaces, diagnostics, checkpoints and views are exported"),
}


def _sample(a: np.ndarray, maximum: int) -> np.ndarray:
    """Lấy mẫu đều để hình 3D rõ nhưng không quá nặng."""
    a = np.asarray(a, float)
    if len(a) <= maximum:
        return a
    return a[np.linspace(0, len(a) - 1, maximum, dtype=int)]


def _scatter(ax: Any, p: np.ndarray, color: str, label: str, size: float = 5.0,
             maximum: int = 1200, alpha: float = 0.8) -> np.ndarray:
    """Vẽ tập điểm 3D hợp lệ với màu và nhãn yêu cầu."""
    q = _sample(np.asarray(p, float).reshape(-1, 3), maximum)
    good = np.all(np.isfinite(q), axis=1); q = q[good]
    if len(q):
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=size, c=color, label=label, alpha=alpha, depthshade=False)
    return q


def _poly_surface(ax: Any, surf: PolySurface, color: str, label: str) -> np.ndarray:
    """Lấy lưới điểm 3D của gương polynomial."""
    hx, hy = np.asarray(surf.half_aperture, float)
    x = np.linspace(-hx, hx, 21); y = np.linspace(-hy, hy, 17)
    X, Y = np.meshgrid(x, y); p = surf.point(X.ravel(), Y.ravel()).reshape(len(y), len(x), 3)
    if surf.aperture_polygon is not None:
        poly = np.asarray(surf.aperture_polygon, float)
        edges = np.roll(poly, -1, axis=0) - poly
        xv, yv = X.ravel(), Y.ravel()
        cross = (edges[None, :, 0] * (yv[:, None] - poly[None, :, 1])
                 - edges[None, :, 1] * (xv[:, None] - poly[None, :, 0]))
        area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                            - poly[:, 1] * np.roll(poly[:, 0], -1))
        inside = np.all(cross * np.sign(area) >= -1e-9, axis=1).reshape(X.shape)
        p[~inside] = np.nan
    ax.plot_surface(p[:, :, 0], p[:, :, 1], p[:, :, 2], color=color, alpha=0.35,
                    linewidth=0.2, edgecolor=color, shade=False)
    c = np.asarray(surf.center); ax.scatter([c[0]], [c[1]], [c[2]], c=color, s=22, label=label)
    return p.reshape(-1, 3)


def _visor_surface(ax: Any, visor: Any, color: str = "#5bc0eb") -> np.ndarray:
    """Lấy lưới điểm 3D của visor Chebyshev."""
    xmin, xmax, ymin, ymax = visor.bounds
    x = np.linspace(xmin, xmax, 25); y = np.linspace(ymin, ymax, 19)
    X, Y = np.meshgrid(x, y); p = visor.point(X.ravel(), Y.ravel()).reshape(len(y), len(x), 3)
    ax.plot_surface(p[:, :, 0], p[:, :, 1], p[:, :, 2], color=color, alpha=0.22,
                    linewidth=0.15, edgecolor=color, shade=False)
    return p.reshape(-1, 3)


def _packaging(ax: Any, vertices: np.ndarray) -> np.ndarray:
    """Tính hoặc vẽ packaging P1–P8 theo ngữ cảnh gọi."""
    p = np.asarray(vertices, float)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#444444", s=24, marker="s", label="P1–P8 envelope")
    try:
        hull = ConvexHull(p)
        edges = set()
        for tri in hull.simplices:
            for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edges.add(tuple(sorted((int(i), int(j)))))
        for i, j in edges:
            ax.plot(p[[i, j], 0], p[[i, j], 1], p[[i, j], 2], color="#777777", lw=0.8, alpha=0.65)
    except Exception:
        pass
    return p


def _field_shape(data: dict[str, Any]) -> tuple[int, int]:
    """Đọc kích thước field grid động từ run hoặc config."""
    grid = data["vi"].get("field_grid", {})
    return int(grid.get("vertical_count", 3)), int(grid.get("horizontal_count", 3))


def _center_field(data: dict[str, Any]) -> int:
    """Xác định field trung tâm cho mọi lưới kích thước lẻ."""
    return int(data["vi"].get("central_field_index", len(data["vi"]["fields"]) // 2))


def _grid_lines(ax: Any, points: np.ndarray, color: str, label: str,
                linewidth: float = 1.4, shape: tuple[int, int] = (3, 3)) -> np.ndarray:
    """Tạo đường nối hàng/cột của field grid."""
    nv, nh = shape
    p = np.asarray(points, float).reshape(nv, nh, 3)
    for i in range(nv):
        ax.plot(p[i, :, 0], p[i, :, 1], p[i, :, 2], color=color, lw=linewidth)
    for i in range(nh):
        ax.plot(p[:, i, 0], p[:, i, 1], p[:, i, 2], color=color, lw=linewidth)
    ax.scatter(p[:, :, 0], p[:, :, 1], p[:, :, 2], color=color, s=25, label=label, depthshade=False)
    return p.reshape(-1, 3)


def _trace_lines(ax: Any, sequences: list[np.ndarray], color: str, label: str,
                 maximum: int = 90, alpha: float = 0.18) -> np.ndarray:
    """Vẽ các đoạn tia hợp lệ qua từng phần tử quang."""
    if not sequences:
        return np.empty((0, 3))
    arrays = [np.asarray(x, float) for x in sequences]
    count = min(len(x) for x in arrays)
    ids = np.linspace(0, count - 1, min(count, maximum), dtype=int)
    first = True
    all_points = []
    for i in ids:
        p = np.vstack([x[i] for x in arrays]); good = np.all(np.isfinite(p), axis=1); p = p[good]
        if len(p) >= 2:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, lw=0.55, alpha=alpha,
                    label=label if first else None)
            first = False; all_points.append(p)
    return np.vstack(all_points) if all_points else np.empty((0, 3))


def _set_labels(ax: Any, title: str) -> None:
    """Gắn tên trục và đơn vị mm cho đồ thị 3D."""
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X forward [mm]", fontsize=8)
    ax.set_ylabel("Y right [mm]", fontsize=8)
    ax.set_zlabel("Z up [mm]", fontsize=8)
    ax.tick_params(labelsize=7); ax.view_init(elev=22, azim=-58)


def _equal(ax: Any, points: list[np.ndarray], padding: float = 0.08) -> None:
    """Cân tỷ lệ ba trục để hình học không bị kéo méo."""
    valid = []
    for p in points:
        q = np.asarray(p, float).reshape(-1, 3); q = q[np.all(np.isfinite(q), axis=1)]
        if len(q): valid.append(q)
    if not valid:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); return
    q = np.vstack(valid); lo = np.min(q, axis=0); hi = np.max(q, axis=0)
    center = 0.5 * (lo + hi); span = max(float(np.max(hi - lo)), 1.0) * (0.5 + padding)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def _draw(ctx: Any, ax: Any, step: int, mode: str, layers: list[dict[str, Any]]) -> list[np.ndarray]:
    """Vẽ mô hình hệ quang nền cho panel của STEP."""
    d = ctx.data; points = []
    inputs = d.get("inputs")
    if inputs is not None:
        if inputs.get("packaging_vertices") is not None:
            package = _packaging(ax, inputs["packaging_vertices"]); points.append(package)
            layers.append({"layer": "optional_P1_P8_packaging_constraint", "count": 8})
        raw = _scatter(ax, inputs["visor_inner"], "#8ecae6", "Visor INNER samples", 1.0, 1200, 0.2)
        points.append(raw); layers.append({"layer": "visor_inner_authority", "count": len(inputs["visor_inner"])})
    if "visor" in d:
        q = _visor_surface(ax, d["visor"]); points.append(q); layers.append({"layer": "visor_surrogate", "count": len(q)})
    if "m1" in d:
        q = _poly_surface(ax, d["m1"], "#ff9f1c", "M1"); points.append(q); layers.append({"layer": "M1_surface", "count": len(q)})
    if "m2" in d:
        q = _poly_surface(ax, d["m2"], "#e71d36", "M2"); points.append(q); layers.append({"layer": "M2_surface", "count": len(q)})
    if "display" in d:
        q = _poly_surface(ax, d["display"], "#2ec4b6", "Display"); points.append(q); layers.append({"layer": "display", "count": len(q)})
    if "pupils" in d:
        p = np.array([[x["x_mm"], x["y_mm"], x["z_mm"]] for x in d["pupils"]])
        q = _scatter(ax, p, "#000000", "Primary pupils", 35, 20, 1.0); points.append(q)
        layers.append({"layer": "primary_pupils", "count": len(p)})
    if "vi" in d:
        vi = np.array([x["vi_point"] for x in d["vi"]["fields"]])
        if mode == "context":
            q = _grid_lines(ax, vi, "#6a4c93", "Fixed ideal VI",
                            shape=_field_shape(d)); points.append(q)
            eye = np.zeros((1, 3)); q = _scatter(ax, eye, "#000000", "Eye origin", 45, 1, 1); points.append(q)
            q = _trace_lines(ax, [np.zeros_like(vi), vi], "#6a4c93",
                             "Target field directions", len(vi), 0.45); points.append(q)
        layers.append({"layer": "ideal_virtual_image", "count": len(vi)})
    if "rays" in d and step == 5 and mode != "context":
        r = d["rays"]; target = np.array([d["vi"]["fields"][i]["vi_point"] for i in r["field_index"]])
        q = _trace_lines(ax, [r["origins"], target], "#8d99ae", "VI→pupil rays", 70, 0.10); points.append(q)
    if "visor_hit" in d:
        q = _scatter(ax, d["visor_hit"]["point"], "#0077b6", "Visor ray hits", 3, 900, 0.55); points.append(q)
        layers.append({"layer": "visor_hits", "count": len(d["visor_hit"]["point"])})
    tr = d.get("trace")
    if tr is not None and "rays" in d and mode != "context":
        q = _trace_lines(ax, [d["rays"]["origins"], *tr["points"]], "#8338ec", "Reverse trace", 95, 0.16); points.append(q)
        layers.append({"layer": "reverse_trace", "count": len(d["rays"]["rows"])})
    if "fan_refs" in d and mode != "context":
        q = _grid_lines(ax, d["fan_refs"], "#00a896", "Dynamic Fan references",
                        shape=_field_shape(d)); points.append(q)
        layers.append({"layer": "fan_reference", "count": len(d["fan_refs"])})
    if "fixed_vi_grid" in d and mode == "context":
        q = _grid_lines(ax, d["fixed_vi_grid"], "#6a4c93", "Fixed USER distortion grid",
                        2.0, _field_shape(d)); points.append(q)
    if "ci_m1" in d and step in (10, 11, 12):
        q = _scatter(ax, d["ci_m1"]["points_by_ray"], "#fb8500", "M1 CI cloud", 3, 1000, 0.5); points.append(q)
        layers.append({"layer": "M1_CI_cloud", "count": len(d["ci_m1"]["points_by_ray"])})
    if "ci_m2" in d and step in (11, 12):
        q = _scatter(ax, d["ci_m2"]["points_by_ray"], "#d00000", "M2 CI cloud", 3, 1000, 0.5); points.append(q)
        layers.append({"layer": "M2_CI_cloud", "count": len(d["ci_m2"]["points_by_ray"])})
    if "fermat" in d and step in (14, 15, 16, 17):
        q = _scatter(ax, d["fermat"]["target_points"], "#ff006e", "Fermat Q2 targets", 3, 1000, 0.55); points.append(q)
        layers.append({"layer": "Fermat_Q2", "count": len(d["fermat"]["target_points"])})
    if "fan_mf2" in d and mode != "context":
        g = d["fan_mf2"]; q = np.vstack([g["A"], g["Q"], g["P"], g["A"]])
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color="#d90429", lw=2.2, label="MF2 signed A-Q-P")
        points.append(q); layers.append({"layer": "MF2_AQP", "count": 3})
    if "forward" in d:
        f = d["forward"]
        if mode != "context":
            src = d["fan_refs"][d["rays"]["field_index"]]
            q = _trace_lines(ax, [src, f["m2"], f["m1"], f["visor"], f["eye_hit"]],
                             "#00b4d8", "Forward shooting", 110, 0.20); points.append(q)
        if mode == "context" and "virtual" in d:
            vp = d["virtual"]["points"].reshape(-1, 3)
            q = _scatter(ax, vp, "#f72585", "Reconstructed virtual points", 18, 200, 0.8); points.append(q)
        layers.append({"layer": "forward_shooting", "count": len(f["eye_hit"]),
                       "converged": int(np.sum(f["converged"]))})
    return points


def _read_rows(ctx: Any, step: int, filename: str) -> list[dict[str, str]]:
    """Đọc CSV bằng chứng hoặc trả rỗng nếu chưa tồn tại."""
    path = ctx.step_dir(step) / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _text_panel(ax: Any, title: str, lines: list[str]) -> None:
    """Vẽ bảng chữ tóm tắt metric và điều kiện STEP."""
    ax.set_axis_off(); ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(str(line), width=70) or [""])
    ax.text(0.01, 0.96, "\n".join(wrapped), transform=ax.transAxes, va="top", ha="left",
            fontsize=8.2, family="DejaVu Sans",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8f9fa", "edgecolor": "#adb5bd"})


def _gate_audit_panel(ax: Any, gates: list[dict[str, Any]]) -> None:
    """Vẽ actual/rule/reason của các gate được chấm bằng dữ liệu scorecard."""
    graded = [row for row in gates if row["status"] != "UNGRADED"]
    colors = {"PASS": "#2a9d8f", "FAIL": "#d90429",
              "NOT_CONSTRAINED": "#6c757d"}
    ax.set_axis_off()
    ax.set_title("Automatic hard-gate audit — actual value / current rule / reason",
                 fontsize=10, loc="left", fontweight="bold")
    if not graded:
        ax.text(.02, .9, "No graded or explicitly disabled gate is available.",
                transform=ax.transAxes, va="top", fontsize=8.5)
        return
    row_height = .90 / len(graded)
    for index, gate in enumerate(graded):
        top = .95 - index * row_height
        bottom = top - row_height + .008
        status = str(gate["status"])
        background = "#f8f9fa" if index % 2 == 0 else "#eef1f3"
        ax.add_patch(plt.Rectangle((0, bottom), 1, row_height - .008,
                                  transform=ax.transAxes, facecolor=background,
                                  edgecolor="#ced4da", lw=.55))
        ax.add_patch(plt.Rectangle((.012, bottom + .018), .135, row_height - .044,
                                  transform=ax.transAxes,
                                  facecolor=colors.get(status, "#adb5bd"),
                                  edgecolor="none"))
        ax.text(.0795, bottom + (row_height - .008) / 2, status.replace("_", "\n"),
                transform=ax.transAxes, ha="center", va="center", color="white",
                fontsize=7.1, fontweight="bold")
        label = "\n".join(textwrap.wrap(str(gate["item"]).replace("_", " "),
                                        width=27, break_long_words=False))
        ax.text(.16, bottom + (row_height - .008) / 2, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7.15, fontweight="bold", color="#212529")
        detail_lines = [f"Actual: {gate['actual_display']}",
                        f"Rule: {gate['rule_display']}",
                        f"Why: {gate['reason']}"]
        detail = "\n".join(part for line in detail_lines
                            for part in textwrap.wrap(line, width=88,
                                                     subsequent_indent="      ",
                                                     break_long_words=False))
        ax.text(.48, bottom + (row_height - .008) / 2, detail,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=6.75, color="#212529", linespacing=1.12)


def render_planar_seed_precheck(root: Path, rows: list[dict[str, Any]],
                                 rms_limit_mm: float,
                                 algorithm_source_manifest_sha256: str | None = None
                                 ) -> dict[str, str]:
    """Xuất bảng hình RMS paper-style cho mọi planar seed trước khi chạy tối ưu sâu."""
    root = Path(root)
    values = np.asarray([np.nan if row.get("planar_paper_spot_rms_mm") is None
                         else float(row["planar_paper_spot_rms_mm"]) for row in rows], float)
    colors = [
        "#2a9d8f"
        if bool(row.get("eligible"))
        else (
            "#d90429"
            if bool(row.get("physical_eligible"))
            and bool(row.get("packaging_constraint_enabled"))
            and row.get("packaging_pass") is False
            else "#adb5bd"
        )
        for row in rows
    ]
    fig = plt.figure(figsize=(15, 8), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1.0), hspace=.28)
    ax = fig.add_subplot(grid[0]); table_ax = fig.add_subplot(grid[1])
    x = np.arange(1, len(rows) + 1)
    ax.bar(x, values, color=colors)
    ax.axhline(float(rms_limit_mm), color="#d90429", ls="--", lw=1.3,
               label=f"diagnostic reference only: RMS < {rms_limit_mm:g} mm")
    ax.set(title="STEP 07 — planar RMS diagnostic after geometry/physics evaluation",
           xlabel="auto-generated planar geometry", ylabel="RMS spot radius [mm]")
    ax.set_xticks(x); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    table_ax.set_axis_off()
    headers = ["seed", "rank", "RMS [mm]", "physical", "packaging", "decision / reason"]
    matrix = []
    for row in rows:
        status = row.get("precheck_execution_status", "EVALUATED")
        if status != "EVALUATED":
            phys_col = "NOT_EVALUATED"
        else:
            phys_col = "PASS" if bool(row.get("physical_eligible")) else "FAIL"

        if not bool(row.get("packaging_constraint_enabled")):
            packaging_col = "OFF"
        elif row.get("packaging_lambda") is None:
            packaging_col = "N/A"
        else:
            packaging_col = (
                f"{float(row['packaging_lambda']):.4g}"
                f" <= {float(row['packaging_lambda_limit']):.4g}"
            )

        matrix.append([
            str(row["candidate"]),
            "-" if row.get("geometry_rank") is None
            else str(row["geometry_rank"]),
            "N/A" if row.get("planar_paper_spot_rms_mm") is None
            else f"{float(row['planar_paper_spot_rms_mm']):.6g}",
            phys_col,
            packaging_col,
            "ELIGIBLE" if bool(row.get("eligible"))
            else str(row.get("rejection_reasons") or "REJECTED"),
        ])
    table = table_ax.table(cellText=matrix, colLabels=headers, cellLoc="center", loc="center",
                           colWidths=[.06, .08, .12, .11, .14, .49])
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.35)
    fig.suptitle(
        f"All {len(rows)} STEP 07 geometries; green = eligible, red = packaging fail after physical pass, gray = physical precheck fail; RMS is diagnostic only",
        fontsize=11,
        fontweight="bold",
    )
    png = root / "07_ALL_PLANAR_START_RMS_COMPARISON.png"
    metadata = root / "07_ALL_PLANAR_START_RMS_COMPARISON.json"
    temporary = root / ".07_ALL_PLANAR_START_RMS_COMPARISON.tmp.png"
    fig.savefig(temporary, format="png", bbox_inches="tight"); plt.close(fig)
    temporary.replace(png)
    record = {
        "schema": "HUD_FAN_V5_5_PLANAR_PAPER_SPOT_RMS_PRECHECK",
        "metric_name": "CENTROIDED_GEOMETRIC_SPOT_RMS_RADIUS_ALL_FIELD_PUPIL_RAYS",
        "aggregation": "per-field/pupil centroid, then RMS pooled over all valid rays",
        "rms_reference": f"RMS < {float(rms_limit_mm):g} mm",
        "rms_is_hard_gate": False,
        "candidate_count": len(rows),
        "physical_eligible_count": sum(bool(row.get("physical_eligible")) for row in rows),
        "eligible_count": sum(bool(row.get("eligible")) for row in rows),
        "algorithm_source_manifest_sha256": algorithm_source_manifest_sha256,
        "rows": rows, "image": str(png),
    }
    write_json(metadata, record)
    return {"image": str(png), "metadata": str(metadata)}


def _ci_focus(ax: Any, ci: dict[str, Any], color: str, name: str,
              layers: list[dict[str, Any]]) -> np.ndarray:
    """Vẽ riêng dựng hình CI, ray và target của STEP."""
    ordered = np.asarray(ci["points_ordered"], float)
    shown = ordered[:min(180, len(ordered))]
    ax.plot(shown[:, 0], shown[:, 1], shown[:, 2], color=color, lw=1.2,
            label=f"{name} construction order (first {len(shown)})")
    ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], color=color, s=5, alpha=.55)
    normals = np.asarray(ci["normals_ordered"], float)
    qi = np.linspace(0, len(shown) - 1, min(12, len(shown)), dtype=int)
    ax.quiver(shown[qi, 0], shown[qi, 1], shown[qi, 2], normals[qi, 0], normals[qi, 1], normals[qi, 2],
              length=2.0, normalize=True, color="#023047", linewidth=.7, label="constructed normals")
    k = min(60, len(ordered) - 1); parent = int(ci["parent_order_index"][k])
    if parent >= 0:
        p0, n = ordered[parent], normals[parent]
        seed = np.array([1., 0., 0.]) if abs(n[0]) < .85 else np.array([0., 1., 0.])
        e1 = np.cross(n, seed); e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(n, e1); uv = np.linspace(-2.0, 2.0, 5)
        U, V = np.meshgrid(uv, uv); plane = p0 + U[..., None] * e1 + V[..., None] * e2
        ax.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2], color="#90e0ef", alpha=.5,
                        linewidth=.2, edgecolor="#0077b6")
        ax.plot(ordered[[parent, k], 0], ordered[[parent, k], 1], ordered[[parent, k], 2],
                "o-", color="#d90429", lw=2, label=f"nearest tangent parent {parent}→{k}")
    layers.append({"layer": f"{name}_ordered_CI_with_nearest_tangent_plane",
                   "count": len(ordered), "fallback_count": int(ci["fallback_count"])})
    return ordered


def _focused_geometry(ctx: Any, ax: Any, step: int, layers: list[dict[str, Any]]) -> list[np.ndarray]:
    """Chọn lớp hình học cần nhấn mạnh theo thuật toán STEP."""
    d = ctx.data; points: list[np.ndarray] = []
    if step <= 1:
        if "inputs" in d:
            if d["inputs"].get("packaging_vertices") is not None:
                points.append(_packaging(ax, d["inputs"]["packaging_vertices"]))
            points.append(_scatter(ax, d["inputs"]["visor_inner"], "#219ebc", "raw visor-inner points", 2, 1800, .3))
        if step == 1 and "visor" in d: points.append(_visor_surface(ax, d["visor"], "#00b4d8"))
    elif step in (2, 3):
        vi = np.asarray([f["vi_point"] for f in d["vi"]["fields"]])
        nv, nh = _field_shape(d)
        points.append(_grid_lines(ax, vi, "#6a4c93", f"{nh}×{nv} target VI grid",
                                  2.0, (nv, nh)))
        eye = np.zeros((len(vi), 3)); points.append(_trace_lines(
            ax, [eye, vi], "#8338ec", "field directions", len(vi), .7))
        points.append(_scatter(ax, np.zeros((1, 3)), "#000", "eye origin", 50, 1, 1))
    elif step in (4, 5):
        p = np.asarray([[x["x_mm"], x["y_mm"], x["z_mm"]] for x in d["pupils"]])
        points.append(_scatter(ax, p, "#111", f"{len(p)} primary pupils", 55, 10, 1))
        if step == 5:
            center_field = _center_field(d)
            ids = np.where((d["rays"]["field_index"] == center_field))[0]
            target = np.asarray([d["vi"]["fields"][center_field]["vi_point"]] * len(ids))
            points.append(_trace_lines(ax, [d["rays"]["origins"][ids], target], "#8338ec",
                                       "central-field characteristic rays", 100, .12))
    elif step == 6:
        points.append(_visor_surface(ax, d["visor"])); points.append(_scatter(ax, d["visor_hit"]["point"],
                      "#0077b6", f"{len(d['rays']['rows'])} visor intersections", 3, 1400, .5))
        ids = np.linspace(0, len(d["rays"]["rows"]) - 1, 80, dtype=int)
        start = d["rays"]["origins"][ids]; hit = d["visor_hit"]["point"][ids]
        end = hit + 18.0 * d["post_visor"][ids]
        points.append(_trace_lines(ax, [start, hit, end], "#fb8500", "incident → reflected", 80, .35))
    elif step in (10, 11):
        key = "ci_m1" if step == 10 else "ci_m2"
        points.append(_ci_focus(ax, d[key], "#fb8500" if step == 10 else "#d00000",
                                "M1" if step == 10 else "M2", layers))
    elif step == 14:
        points.append(_poly_surface(ax, d["m2"], "#e71d36", "current M2"))
        q1, q2 = d["trace"]["points"][1], d["fermat"]["target_points"]
        targets = d["fan_refs"][d["rays"]["field_index"]]
        points.append(_trace_lines(ax, [q1, q2, targets], "#ff006e", "Q1→Q2→P_ref optical path", 100, .25))
        points.append(_scatter(ax, q2, "#ff006e", "stationary Q2 targets", 4, 1200, .55))
    elif step == 20:
        points.append(_poly_surface(ax, d["display"], "#2ec4b6", "display"))
        land = d["trace"]["landing"]; ref = d["fan_refs"][d["rays"]["field_index"]]
        points.append(_trace_lines(ax, [land, ref], "#d90429", "all-ray MF1 residual vectors", 180, .45))
        points.append(_scatter(ax, d["fan_refs"], "#00a896", "dynamic Fan references", 35, 20, 1))
    elif step == 21:
        for key, color in (("m1", "#ff9f1c"), ("m2", "#e71d36")):
            points.append(_poly_surface(ax, d[key], color, key.upper()))
        g = d["fan_mf2"]; tri = np.vstack([g["A"], g["Q"], g["P"], g["A"]])
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], "o-", color="#d90429", lw=3, label="signed A–Q–P triangle")
        points.append(tri)
    elif step == 24 and "forward" in d:
        for key, color in (("display", "#2ec4b6"), ("m2", "#e71d36"), ("m1", "#ff9f1c")):
            points.append(_poly_surface(ax, d[key], color, key.upper()))
        points.append(_visor_surface(ax, d["visor"])); f = d["forward"]
        src = d["fan_refs"][d["rays"]["field_index"]]; good = np.asarray(f["converged"], bool)
        ids = np.where(good)[0]
        if len(ids):
            points.append(_trace_lines(ax, [src[ids], f["m2"][ids], f["m1"][ids],
                          f["visor"][ids], f["eye_hit"][ids]], "#00b4d8",
                          "converged: Display→M2→M1→visor→eye", 90, .28))
        # A failed nonlinear shot can contain an arbitrarily remote trial eye/visor
        # intersection.  Plotting that raw trial point hides the physical HUD by
        # rescaling the 3-D axes.  Keep the numerical failure in the diagnostics,
        # while the spatial panel shows only its finite in-package path to M1.
        bad = np.where(~good)[0]
        local = bad[(np.abs(f["m1"][bad, 0]) < 80.0)
                    & (np.abs(f["m1"][bad, 1]) < 50.0)
                    & (np.abs(f["m1"][bad, 2]) < 100.0)]
        if len(local):
            points.append(_trace_lines(ax, [src[local], f["m2"][local], f["m1"][local]],
                          "#d90429", "failed shots: finite path through M1", 55, .38))
            points.append(_scatter(ax, f["m1"][local], "#d90429",
                          "failed solver endpoint shown", 12, 120, .8))
    else:
        if "visor" in d: points.append(_visor_surface(ax, d["visor"]))
        for key, color in (("m1", "#ff9f1c"), ("m2", "#e71d36"), ("display", "#2ec4b6")):
            if key in d: points.append(_poly_surface(ax, d[key], color, key.upper()))
        if "trace" in d and "rays" in d:
            points.append(_trace_lines(ax, [d["rays"]["origins"], *d["trace"]["points"]],
                                       "#8338ec", "current reverse mapping", 90, .18))
        if step in (8, 22) and "fan_refs" in d:
            points.append(_grid_lines(ax, d["fan_refs"], "#00a896", "dynamic Fan reference",
                                      2, _field_shape(d)))
        if step in (9,) and "m1_targets" in d:
            points.append(_scatter(ax, d["m1_targets"], "#f72585", "planar conjugate targets", 4, 1200, .5))
        if step in (12,) and "ci_m1" in d:
            points.append(_scatter(ax, d["ci_m1"]["points_by_ray"], "#fb8500", "M1 CI cloud", 2, 900, .35))
            points.append(_scatter(ax, d["ci_m2"]["points_by_ray"], "#d00000", "M2 CI cloud", 2, 900, .35))
        if step in (15, 16, 17) and "fermat" in d:
            points.append(_scatter(ax, d["fermat"]["target_points"], "#ff006e", "Fermat construction targets", 3, 900, .4))
        if (step in (25, 26) and "inputs" in d
                and d["inputs"].get("packaging_vertices") is not None):
            points.append(_packaging(ax, d["inputs"]["packaging_vertices"]))
    layers.append({"layer": "step_specific_geometry", "step": step, "focus": STEP_FOCUS[step][0]})
    return points


def _diagnostics(ctx: Any, ax1: Any, ax2: Any, step: int, evidence: list[str]) -> None:
    """Vẽ residual, distortion, MTF hoặc khẩu độ theo STEP."""
    d = ctx.data; cfg = ctx.config
    ax1.grid(True, alpha=.2); ax2.grid(True, alpha=.2)
    if step == 0:
        packaging = d["inputs"].get("packaging_vertices")
        if packaging is not None:
            p = np.asarray(packaging); ax1.scatter(p[:, 0], p[:, 2], c=np.arange(len(p)), cmap="viridis", s=60)
            for i, q in enumerate(p): ax1.annotate(f"P{i+1}", (q[0], q[2]), fontsize=7)
            ax1.set(title="Declared optional P1–P8 envelope", xlabel="X [mm]", ylabel="Z [mm]")
            evidence.append("P1–P8 packaging constraint enabled")
        else:
            _text_panel(ax1, "Packaging constraint", ["P1–P8: NOT DECLARED", "Optical search is not clipped by packaging."])
            evidence.append("P1–P8 packaging constraint disabled")
        files = d["inputs"]["files"]; sizes = [x.stat().st_size / 1e6 for x in files]
        ax2.barh(range(len(files)), sizes, color="#219ebc"); ax2.set_yticks(range(len(files)), [x.name[:28] for x in files], fontsize=6)
        ax2.set(title="Loaded authority files", xlabel="size [MB]"); evidence += [f"{len(files)} input files loaded"]
    elif step == 1:
        visor = d["visor"]; loc = (d["inputs"]["visor_inner"] - visor.center) @ visor.frame
        residual = visor.sag(loc[:, 0], loc[:, 1]) - loc[:, 2]
        ax1.hist(residual, bins=80, color="#219ebc"); ax1.set(title="Visor sag-fit residual", xlabel="fit − data [mm]", ylabel="count")
        nums = [(k, v) for k, v in d["visor_fit_stats"].items() if isinstance(v, (int, float, np.number))]
        _text_panel(ax2, "Differentiable fit evidence", [f"{k}: {float(v):.6g}" for k, v in nums[:10]])
        evidence += [f"sag RMS={np.sqrt(np.mean(residual**2)):.6g} mm", "raw points overlaid with fitted visor"]
    elif step in (2, 3):
        fields = d["vi"]["fields"]; h = [f["h_deg"] for f in fields]; v = [f["v_deg"] for f in fields]
        nv, nh = _field_shape(d); field_count = len(fields)
        ax1.scatter(h, v, s=70, c=range(field_count), cmap="plasma")
        for f in fields: ax1.annotate(f["field_id"], (f["h_deg"], f["v_deg"]), fontsize=7)
        ax1.set(title=f"{nh}×{nv} angular field samples ({field_count} fields)",
                xlabel="horizontal field [deg]", ylabel="vertical field [deg]")
        vals = [cfg["fov_h_deg"], cfg["fov_v_deg"], cfg["d6_deg"], cfg["vid_mm"] / 1000]
        ax2.bar(["FOV-H°", "FOV-V°", "D6°", "VID m"], vals, color=["#8338ec", "#8338ec", "#fb8500", "#219ebc"])
        ax2.set_title("Declared VI specification"); evidence += [f"{nh}×{nv} field grid", "VI points computed from FOV/D6/VID"]
    elif step == 4:
        p = d["pupils"]; y = [x["y_mm"] for x in p]; z = [x["z_mm"] for x in p]
        ax1.scatter(y, z, s=110, c="#111")
        for x in p:
            ax1.annotate(x["pupil_id"], (x["y_mm"], x["z_mm"])); ax1.add_patch(plt.Circle((x["y_mm"], x["z_mm"]), cfg["pupil_diameter_mm"]/2, fill=False, alpha=.35))
        diameter = float(cfg["pupil_diameter_mm"])
        ax1.axis("equal"); ax1.set(title=f"Primary pupil locations and {diameter:g} mm footprints", xlabel="Y [mm]", ylabel="Z [mm]")
        grid = cfg["primary_pupil_grid"]
        _text_panel(ax2, "Eye-box sampling", [f"Grid: {grid['horizontal_count']} × {grid['vertical_count']} = {len(p)} centers", f"Y range: {cfg['eyebox_y_mm']} mm", f"Z range: {cfg['eyebox_z_mm']} mm", f"Pupil diameter: {cfg['pupil_diameter_mm']} mm"])
        evidence += [f"{len(p)} pupil centers", "finite pupil footprint drawn at each center"]
    elif step == 5:
        pat = d["pattern"]; yy = [x["dy_mm"] for x in pat]; zz = [x["dz_mm"] for x in pat]; ring = [x["ring"] for x in pat]
        ax1.scatter(yy, zz, c=ring, cmap="viridis", s=35); ax1.axis("equal"); ax1.set(title="49-point polar characteristic pattern", xlabel="ΔY [mm]", ylabel="ΔZ [mm]")
        field_count = int(d["rays"].get("field_count", len(d["vi"]["fields"])))
        counts = [int(np.sum(d["rays"]["field_index"] == f)) for f in range(field_count)]
        ax2.bar(range(field_count), counts, color="#8338ec"); ax2.set(title="Ray count per field", xlabel="field index", ylabel="rays")
        evidence += [f"pattern={len(pat)}", f"total rays={len(d['rays']['rows'])}"]
    elif step == 6:
        loc = (d["visor_hit"]["point"] - d["visor"].center) @ d["visor"].frame
        ax1.scatter(loc[:, 0], loc[:, 1], s=2, alpha=.35, c=d["rays"]["field_index"], cmap="plasma")
        ray_count = len(d["rays"]["rows"])
        ax1.set(title=f"{ray_count} hit footprint on fitted visor", xlabel="visor local u [mm]", ylabel="visor local v [mm]")
        ax2.hist(d["post_visor"][:, 2], bins=60, color="#fb8500"); ax2.set(title="Reflected direction Z component", xlabel="dZ", ylabel="count")
        evidence += [f"valid visor hits={int(np.sum(d['visor_hit']['valid']))}/{ray_count}", "incident and reflected segments shown in 3-D"]
    elif step == 7:
        rec = d["planar_start_record"]; g = rec["seed_MF2"]
        required = len(d["rays"]["rows"])
        rows = d.get("planar_seed_candidate_rows", [])
        limit = float(rec["planar_paper_spot_rms_limit_mm"])
        rms = np.asarray([np.nan if row.get("planar_paper_spot_rms_mm") is None
                          else float(row["planar_paper_spot_rms_mm"]) for row in rows], float)
        colors = ["#2a9d8f" if bool(row.get("eligible")) else
                  ("#d90429" if bool(row.get("physical_eligible")) else "#adb5bd")
                  for row in rows]
        ax1.bar(np.arange(1, len(rows) + 1), rms, color=colors)
        ax1.axhline(limit, color="#d90429", ls="--", lw=1.2,
                    label=f"diagnostic reference: RMS < {limit:g} mm")
        ax1.set(title="Planar chief-centered RMS — geometry gate already decided separately",
                xlabel="geometry candidate", ylabel="RMS spot radius [mm]")
        ax1.set_xticks(np.arange(1, len(rows) + 1)); ax1.legend(fontsize=7)
        spot = rec["planar_paper_spot"]
        _text_panel(ax2, "Seed acceptance evidence", [
            f"candidate: {rec['selected_candidate']}",
            f"physical first-hit: {rec['physical_valid_count']}/{required}",
            f"signed S_AQP: {g['S_AQP_signed_mm2']:.6g} mm²",
            f"planar chief-centered RMS diagnostic: {spot['RMS_spot_radius_mm']:.6g} mm",
            f"packaging enabled: {rec['packaging_constraint_enabled']}; lambda: {rec['packaging_lambda']}",
            "STEP 07 selection uses physical topology + compactness; RMS is only the final tie-break.",
        ])
        evidence += [f"physical first-hit {rec['physical_valid_count']}/{required}",
                     f"planar chief-centered RMS={spot['RMS_spot_radius_mm']:.6g} mm",
                     f"diagnostic reference <{limit:g} mm"]
    elif step == 8:
        fan = (d["fan_refs"] - d["display"].center) @ d["display"].frame
        fixed = np.asarray([[f["u_vi_mm"], f["v_vi_mm"]] for f in d["vi"]["fields"]])
        ax1.plot(fan[:, 0], fan[:, 1], "o", color="#00a896"); ax1.set(title="Dynamic Fan grid on display", xlabel="display u [mm]", ylabel="display v [mm]")
        ax2.plot(fixed[:, 0], fixed[:, 1], "s", color="#6a4c93"); ax2.set(title="Fixed distortion grid on VI plane", xlabel="VI u [mm]", ylabel="VI v [mm]")
        evidence += ["separate arrays and authorities", "fixed grid never used as Fan endpoint"]
    elif step == 9:
        target = d["m1_targets"]; ref = d["fan_refs"][d["rays"]["field_index"]]
        dist = np.linalg.norm(target - ref, axis=1); ax1.hist(dist, bins=60, color="#f72585")
        ax1.set(title="Planar conjugate displacement", xlabel="|target − Fan reference| [mm]", ylabel="count")
        _text_panel(ax2, "Step-One construction", ["P_target = mirror(P_ref) through planar M2", f"targets: {len(target)}", "Valid only while M2 is planar"])
        evidence += [f"{len(target)} symmetric-conjugate targets", "planar-only formula declared"]
    elif step in (10, 11):
        ci = d["ci_m1" if step == 10 else "ci_m2"]; distance = np.asarray(ci["nearest_distance_mm"], float)
        good = np.isfinite(distance); ax1.semilogy(np.where(good)[0], distance[good], color="#fb8500" if step == 10 else "#d00000", lw=.8)
        ax1.set(title="Nearest tangent-plane propagation distance", xlabel="construction order k", ylabel="distance [mm]")
        parent = np.asarray(ci["parent_order_index"]); ax2.plot(parent, lw=.7, color="#023047"); ax2.plot(np.arange(len(parent)), "--", lw=.5, color="#adb5bd")
        ax2.set(title="Selected previous tangent-plane index", xlabel="construction order k", ylabel="parent index")
        evidence += [f"constructed points={len(parent)}", f"fallbacks={ci['fallback_count']}", "representative tangent plane drawn in 3-D"]
    elif step == 12:
        m1, m2 = d["fit_m1_order2"], d["fit_m2_order2"]
        selected_m1 = d.get("step11_selected_m1_ci_trust", m1)
        ax1.bar(["M1 CI trust sag RMS", "M2 fit sag RMS"], [selected_m1["sag_rms_mm"], m2["sag_rms_mm"]], color=["#ff9f1c", "#e71d36"]); ax1.set(title="Selected order-2 pair residual diagnostics", ylabel="RMS [mm]")
        terms = np.zeros((3, 3));
        for i, j in d["m2"].terms: terms[j, i] = 1
        ax2.imshow(terms, origin="lower", cmap="Greens", vmin=0, vmax=1); ax2.set_xticks(range(3)); ax2.set_yticks(range(3)); ax2.set(title="Complete axis-order-2 term set", xlabel="i", ylabel="j")
        evidence += ["A21, A12 and A22 present", "CI clouds and fitted surfaces overlaid",
                     f"M1 selected CI-trust normal RMS={selected_m1['normal_rms_deg']:.6g} deg",
                     f"M2 normal RMS={m2['normal_rms_deg']:.6g} deg; K-bound={m2.get('k_bound_hit')}",
                     f"M1 constrained sphere R={m1['base_sphere_radius_mm']:.6g} mm; chief residual={m1['chief_vertex_radial_residual_to_geometric_sphere_mm']:.3g} mm",
                     f"M2 constrained sphere R={m2['base_sphere_radius_mm']:.6g} mm; chief residual={m2['chief_vertex_radial_residual_to_geometric_sphere_mm']:.3g} mm",
                     "normal objective=unit-normal/tangent-plane; integrability grouped by field/pupil"]
    elif step == 13:
        _text_panel(ax1, "State transition", ["Fan Step One", "planar conjugate", "↓ disable shortcut", "Fan Step Two", "Fermat stationary target on current M2"])
        _text_panel(ax2, "Allowed next operation", ["Only Eq. (2) stationary-path target generation", "No return to symmetric planar conjugate after this step"])
        evidence += ["step_two=True", "planar shortcut disabled"]
    elif step == 14:
        g = np.sort(np.asarray(d["fermat"]["gradient_norm"], float)); ax1.semilogy(np.arange(len(g)), np.maximum(g, 1e-16), color="#ff006e")
        ax1.axhline(cfg["solver"]["fermat_gradient_tolerance"], color="#d90429", ls="--", label="tolerance"); ax1.legend(fontsize=7); ax1.set(title="Fermat gradient certification", xlabel="sorted ray", ylabel="||∇OP||")
        delta = np.asarray(d["fermat"]["op_final"]) - np.asarray(d["fermat"]["op_initial"]); ax2.hist(delta, bins=60, color="#8338ec"); ax2.set(title="Stationary solve optical-path change", xlabel="OP_final − OP_initial [mm]", ylabel="count")
        evidence += [f"success={int(np.sum(d['fermat']['success']))}/{len(g)}", f"gradient max={np.max(g):.6g}"]
    elif step == 15:
        rho = np.asarray(cfg["solver"]["rho_candidates"], float); ax1.axhline(0, color="#555", lw=.8); ax1.scatter(rho, np.zeros_like(rho), c=np.where(rho > 0, "#2a9d8f", "#e76f51"), s=90)
        for x in rho: ax1.annotate(f"{x:+g}", (x, 0), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7)
        ax1.set(title="Declared signed rho branches", xlabel="rho", yticks=[])
        _text_panel(ax2, "Branch rule", ["rho < 0: opposite update direction", "rho > 0: same update direction", "Fermat + surface + MF2 gates", "rank: full physical, limited restoration, Fan merit, chief-centered spot", "100% physical remains mandatory before STEP 18"])
        evidence += [f"signed candidates={len(rho)}", "both convex/concave update directions retained"]
    elif step == 16:
        frontier = d.get("_rho_frontier", []); rho = [x["rho_sequence"][-1] for x in frontier]; merit = [x["Fan_core_merit"] for x in frontier]
        ax1.scatter(rho, merit, s=80, c=np.where(np.asarray(rho) > 0, "#2a9d8f", "#e76f51")); ax1.set(title="Accepted order-2 signed-rho branches", xlabel="rho", ylabel="Fan-core merit")
        required = len(d["rays"]["rows"])
        physical = [x["reconstruct_out"]["physical_valid_count"] for x in frontier]
        colors = ["#2a9d8f" if x["reconstruct_out"].get("physical_first_hit_complete") else "#f4a261" for x in frontier]
        ax2.bar(range(len(frontier)), physical, color=colors); ax2.axhline(required, color="#d90429", ls="--"); ax2.set(title="Feasible-first physical coverage", xlabel="retained branch", ylabel="valid rays")
        restoration_count = sum(not x["reconstruct_out"].get("physical_first_hit_complete", False) for x in frontier)
        evidence += [f"retained branches={len(frontier)}; restoration={restoration_count}", "M1 reconstructed before M2", "final continuation still requires 100% physical and nonnegative MF2"]
    elif step == 17:
        hist = d["order_history"]; orders = [int(x["order"]) for x in hist]; merit = [float(x["Fan_core_merit"]) for x in hist]
        optimized = np.asarray([bool(x.get("shape_optimized", x.get("accepted", False))) for x in hist])
        ax1.plot(orders, merit, "-", color="#8338ec", alpha=.65)
        if np.any(optimized):
            ax1.scatter(np.asarray(orders)[optimized], np.asarray(merit)[optimized],
                        color="#8338ec", marker="o", label="accepted shape update")
        if np.any(~optimized):
            ax1.scatter(np.asarray(orders)[~optimized], np.asarray(merit)[~optimized],
                        facecolors="none", edgecolors="#fb8500", marker="s", s=70,
                        label="basis promotion only")
        ax1.legend(fontsize=7); ax1.set_xticks([2,3,4,5]); ax1.set(title="Nested CI level ladder", xlabel="CI basis level", ylabel="Fan-core merit")
        required = len(d["rays"]["rows"])
        phys = [int(x["physical_valid_count"]) for x in hist]; ax2.plot(orders, phys, "o-", color="#219ebc"); ax2.axhline(required, color="#d90429", ls="--"); ax2.set_xticks([2,3,4,5]); ax2.set(title="Physical audit at each order", xlabel="order", ylabel="valid first hits")
        state = d.get("order5_state", {})
        evidence += [f"rho sequence={[float(x['rho']) for x in hist]}", "nested basis: Fan O2 subset union L3 subset total O4 subset total O5",
                     f"Order-5 state: {state.get('status', 'N/A')}",
                     f"final full physical: {bool(phys and phys[-1] == required)}"]
    elif step == 18:
        for ax, surf, color in ((ax1, d["m1"], "#ff9f1c"), (ax2, d["m2"], "#e71d36")):
            order = [sum(t) for t in surf.terms]; values = np.abs(surf.coeff); ax.scatter(order, values, c=color, alpha=.75); ax.set_yscale("log"); ax.set(title=f"{surf.name} order-5 coefficient spectrum", xlabel="total order i+j", ylabel="|normalized coefficient| [mm]")
        evidence += [f"M1 terms={len(d['m1'].terms)}", f"M2 terms={len(d['m2'].terms)}", "saved as optimization start, not final"]
    elif step == 19:
        _text_panel(ax1, "Chief-centered optical core — DLSQ residual", ["MF1 = ω1 Σ_pupil RMS(P − P_chief(field,pupil))", "P diagnostic = chief-to-chief pupil spread", "B diagnostic = mean-chief to P_ref field bias", "MF2 = 0 if S_AQP≥0 else ω2|S_AQP|", "J_Fan = w1 MF1/L + w2 MF2/A"])
        _text_panel(ax2, "External engineering checks", ["fixed-grid distortion", "packaging lambda", "physical first-hit", "surface sanity", "These are not added to the Fan residual."])
        evidence += ["Fan and engineering objectives shown separately", "no double weighting"]
    elif step == 20:
        m = d["fan_mf1"]
        pupil_rms = m["per_pupil_RMS_mm"]
        ax1.bar(range(len(pupil_rms)), pupil_rms, color="#8338ec")
        ax1.set(title=f"{len(pupil_rms)} all-ray pupil RMS terms (design merit)", xlabel="pupil index", ylabel="RMS [mm]")
        landing = np.asarray(d["trace"]["landing"], float)
        err = np.empty(len(landing), float)
        for field_index in range(int(d["rays"]["field_count"])):
            for pupil_index in range(int(d["rays"]["pupil_count"])):
                bundle = ((d["rays"]["field_index"] == field_index) &
                          (d["rays"]["pupil_index"] == pupil_index))
                chief_indices = np.where(bundle & d["rays"]["chief"])[0]
                if len(chief_indices) != 1:
                    raise RuntimeError("STEP20_VIEW_CHIEF_SELECTION_MISMATCH")
                chief_index = int(chief_indices[0])
                err[bundle] = np.linalg.norm(landing[bundle] - landing[chief_index], axis=1)
        valid = np.asarray(d["trace"]["valid"], bool)
        if np.any(valid):
            ax2.hist(err[valid], bins=70, color="#2a9d8f", alpha=.72,
                     label=f"physical-valid {int(np.sum(valid))}")
        if np.any(~valid):
            ax2.hist(err[~valid], bins=70, color="#d90429", alpha=.72,
                     label=f"physical-invalid {int(np.sum(~valid))}")
        ax2.legend(fontsize=7)
        ax2.set(title=f"All {len(err)} MF1 errors — validity shown separately",
                xlabel="3-D error [mm]", ylabel="count")
        rays_per_pupil = len(err) // len(m["per_pupil_RMS_mm"])
        evidence += [f"E1={m['E1_mm']:.6g} mm", f"MF1 all-ray design merit={m['MF1_Fan']:.6g}",
                     f"reverse physical-valid={m['physical_valid_ray_count']}/{m['active_ray_count']}",
                     f"certification={m['physical_certification_status']}",
                     f"{rays_per_pupil} rays in each pupil term"]
    elif step == 21:
        g = d["fan_mf2"]; A,Q,P = map(np.asarray, (g["A"],g["Q"],g["P"])); e1 = Q-A; e1 /= max(np.linalg.norm(e1),1e-12); e2=np.cross(g["projection_orientation"],e1); e2/=max(np.linalg.norm(e2),1e-12)
        uv=np.asarray([[0,0],[np.dot(Q-A,e1),np.dot(Q-A,e2)],[np.dot(P-A,e1),np.dot(P-A,e2)],[0,0]])
        ax1.plot(uv[:,0],uv[:,1],"o-",color="#d90429",lw=2); ax1.fill(uv[:3,0],uv[:3,1],color="#d90429",alpha=.18); ax1.set(title="Oriented A–Q–P triangle",xlabel="oriented axis 1 [mm]",ylabel="oriented axis 2 [mm]")
        _text_panel(ax2,"Signed obscuration result",[f"L_min={g['L_min_mm']:.6g} mm",f"D_min={g['D_min_mm']:.6g} mm",f"S_AQP={g['S_AQP_signed_mm2']:.6g} mm²",f"MF2={g['MF2']:.6g}","orientation frozen from initial planar layout"])
        evidence += ["A, Q and P shown in 3-D and oriented 2-D", "signed-area rule displayed"]
    elif step == 22:
        fixed = np.asarray([[f["u_vi_mm"],f["v_vi_mm"]] for f in d["vi"]["fields"]]); fan=(d["fan_refs"]-d["display"].center)@d["display"].frame
        ax1.scatter(fixed[:,0],fixed[:,1],marker="s",s=70,color="#6a4c93"); ax1.set(title="Fixed USER ideal grid",xlabel="VI u [mm]",ylabel="VI v [mm]")
        ax2.scatter(fan[:,0],fan[:,1],marker="o",s=70,color="#00a896"); ax2.set(title="Dynamic Fan reference (not distortion authority)",xlabel="display u [mm]",ylabel="display v [mm]")
        evidence += [f"hard rule Dmax < {cfg['distortion']['hard_limit_percent']}%", "central field is an absolute anchor, not a percentage"]
    elif step == 23:
        h=d["optimization_history"]; it=np.arange(len(h)); before=[x["J_Fan_before"] for x in h]; after=[x["J_Fan_after"] for x in h]
        ax1.plot(it,before,"o--",label="before",color="#adb5bd");ax1.plot(it,after,"o-",label="after",color="#8338ec");ax1.legend(fontsize=7);ax1.set(title="Fan-core filter-DLSQ history",xlabel="recorded iteration",ylabel="J_Fan")
        trials = [row for row in d.get("dlsq_trial_history", [])
                  if bool(row.get("forward_engineering_evaluated", False))
                  and row.get("J_engineering_candidate") is not None]
        if trials:
            rejected_trials = [row for row in trials if not bool(row.get("accepted", False))]
            accepted_trials = [row for row in trials if bool(row.get("accepted", False))]
            if rejected_trials:
                ax2.scatter([float(row["J_Fan_candidate"]) for row in rejected_trials],
                            [float(row["J_engineering_candidate"]) for row in rejected_trials],
                            s=18, alpha=.45, color="#d90429", label="rejected trial")
            if accepted_trials:
                ax2.scatter([float(row["J_Fan_candidate"]) for row in accepted_trials],
                            [float(row["J_engineering_candidate"]) for row in accepted_trials],
                            s=65, marker="*", color="#2a9d8f", label="accepted filter step")
            ax2.legend(fontsize=7); ax2.set(title="DLSQ filter decisions",xlabel="trial J_Fan",ylabel="trial J_engineering")
        else:
            eng=[x["J_engineering_after"] for x in h];ax2.plot(it,eng,"o-",color="#fb8500",label="engineering monitor");ax2.set(title="External filter monitor",xlabel="recorded iteration",ylabel="J_engineering")
        blocked = sum(row.get("status") == "BLOCKED_NO_ADMISSIBLE_DLSQ_TRIAL"
                      for row in d.get("dlsq_termination_history", []))
        evidence += ["40 non-piston order-5 variables",f"J_Fan {before[0]:.6g}→{after[-1]:.6g}",f"accepted steps={sum(bool(x['accepted']) for x in h)}",f"evaluated DLSQ trials={len(d.get('dlsq_trial_history', []))}; blocked phases={blocked}"]
    elif step == 24:
        f=d["forward"]; rays=d["rays"]
        nf=int(rays.get("field_count", len(d["vi"]["fields"]))); npup=int(rays.get("pupil_count", len(d["pupils"])))
        mat=np.zeros((nf,npup));
        for fi in range(nf):
            for pi in range(npup):
                mask=(d["rays"]["field_index"]==fi)&(d["rays"]["pupil_index"]==pi);mat[fi,pi]=np.sum(f["converged"][mask])
        im=ax1.imshow(mat,aspect="auto",vmin=0,vmax=49,cmap="RdYlGn");plt.colorbar(im,ax=ax1,fraction=.046);ax1.set(title="Forward convergence per field/pupil",xlabel="pupil",ylabel="field")
        mtf_rows=_read_rows(ctx,24,"24N_MTF.csv")
        for key,color in (("MTF_u","#219ebc"),("MTF_v","#e76f51")):
            by={}
            for row in mtf_rows:
                if row.get(key) not in (None,""):by.setdefault(float(row["frequency_lpmm"]),[]).append(float(row[key]))
            if by: ax2.plot(sorted(by),[min(by[x]) for x in sorted(by)],"o-",color=color,label=f"minimum {key}")
        ax2.legend(fontsize=7);ax2.set(title="Numerical ray-based MTF envelope",xlabel="spatial frequency [lp/mm]",ylabel="MTF",ylim=(0,1.02))
        ray_count=len(rays["rows"]); dist=d["numerical"]["distortion"]
        evidence += [f"forward converged={int(np.sum(f['converged']))}/{ray_count}",f"distortion bundles={dist['evaluated_noncentral_field_pupil_count']}/{dist['required_count']}","MTF explicitly numerical approximation"]
    elif step == 25:
        # Rebuild from the checkpoint's own config + STEP 24 numerical data.
        # This also upgrades old checkpoints whose scorecard had only four columns.
        gates = build_final_scorecard(cfg, d["numerical"])
        d["scorecard"] = gates
        status=[x["status"] for x in gates]
        _gate_audit_panel(ax1, gates)
        packaging_text = (f"{d['numerical']['packaging_lambda']:.6g}"
                          if d['numerical']['packaging_lambda'] is not None else "NOT CONSTRAINED")
        ungraded = [x for x in gates if x["status"] == "UNGRADED"]
        ungraded_text = "; ".join(f"{x['item']}={x['actual_display'].split('=', 1)[-1]}"
                                  for x in ungraded)
        _text_panel(ax2,"Final numerical decision",[
            f"status: {d['final_status']}",
            f"forward: {d['numerical']['forward_converged']}/{d['numerical']['ray_count']}",
            f"packaging λ: {packaging_text}",
            f"distortion: {d['numerical']['distortion']['evaluation_status']}",
            f"visor footprint inside: {d['numerical']['visor_footprint_inside_66x42']}",
            f"reported but ungraded ({len(ungraded)}): {ungraded_text}",
            "The values, operators, limits and reasons above come from this run's config and STEP 24 data.",
        ])
        evidence += [f"PASS={status.count('PASS')}",f"FAIL={status.count('FAIL')}",f"UNGRADED={status.count('UNGRADED')}"]
    else:
        saved_inventory = d.get(
            "step26_export_file_inventory"
        )

        if saved_inventory is None:
            file_names = [
                p.name
                for p in ctx.run_dir.iterdir()
                if p.is_file()
            ]
        else:
            file_names = [
                str(name)
                for name in saved_inventory
            ]

        ext = {}

        for name in file_names:
            suffix = (
                Path(name).suffix.lower()
                or "no extension"
            )

            ext[suffix] = (
                ext.get(
                    suffix,
                    0,
                )
                + 1
            )
        ax1.bar(ext.keys(),ext.values(),color="#219ebc");ax1.tick_params(axis="x",rotation=35);ax1.set(title="Exported root files by type",ylabel="count")
        vals=np.r_[np.abs(d["m1"].coeff),np.abs(d["m2"].coeff)];ax2.semilogy(np.arange(len(vals)),np.maximum(vals,1e-16),"o",ms=3,color="#8338ec");ax2.set(title="Final M1+M2 normalized coefficients",xlabel="coefficient index",ylabel="|coefficient| [mm]")
        evidence += [f"root export files={len(file_names)}",f"final status={d['final_status']}","prescription and both surfaces exported"]


def render_step(ctx: Any, step: int) -> dict[str, Any]:
    """Kết xuất ảnh ba panel và metadata bằng chứng của STEP."""
    status = ctx.stage_summaries.get(step, {}).get("status", "STATE")
    status = f"EXECUTION {status}"
    if step == 25 and "final_status" in ctx.data:
        status = f"OPTICAL {ctx.data['final_status']}"
    elif step == 26 and "final_status" in ctx.data:
        status = f"EXPORTED • OPTICAL {ctx.data['final_status']}"
    focus, purpose = STEP_FOCUS[step]
    if "vi" in ctx.data:
        nv, nh = _field_shape(ctx.data); field_count = len(ctx.data["vi"]["fields"])
        if step == 3:
            focus = f"{nh}×{nv} sampled fields"
            purpose = f"The configured {field_count}-field angular grid mapped onto the VI plane"
        elif step == 5 and "rays" in ctx.data:
            purpose = (f"49 pupil samples × {field_count} fields × "
                       f"{ctx.data['rays'].get('pupil_count', len(ctx.data['pupils']))} pupils = "
                       f"{len(ctx.data['rays']['rows'])} rays")
        elif step == 20 and "rays" in ctx.data:
            purpose = (f"All {len(ctx.data['rays']['rows'])} mapping residuals and the "
                       "outer sum of pupil RMS terms")
    title = f"STEP_{step:02d} — {focus} — {status}"
    fig = plt.figure(figsize=((20, 10) if step == 25 else (18, 9)), dpi=150)
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=((1.0, 1.55) if step == 25 else (1.2, 1.0)),
        height_ratios=((1.65, .55) if step == 25 else (1.0, 1.0)),
        hspace=.28, wspace=.2,
    )
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax1 = fig.add_subplot(grid[0, 1]); ax2 = fig.add_subplot(grid[1, 1])
    layers: list[dict[str, Any]] = []; evidence: list[str] = []
    geometry = _focused_geometry(ctx, ax3d, step, layers)
    _set_labels(ax3d, f"3-D algorithm evidence — {focus}")
    _equal(ax3d, geometry)
    handles, labels = ax3d.get_legend_handles_labels(); unique = dict(zip(labels, handles))
    if unique: ax3d.legend(unique.values(), unique.keys(), fontsize=6.5, loc="upper left", framealpha=.72)
    _diagnostics(ctx, ax1, ax2, step, evidence)
    fig.suptitle(title + "\n" + purpose, fontsize=13, fontweight="bold")
    fig.text(.01, .012, "Evidence in this image: " + " • ".join(evidence), fontsize=7.5, color="#343a40")
    fig.subplots_adjust(left=.045, right=.975, bottom=.075, top=.885,
                        wspace=.24, hspace=.30)
    folder = ctx.step_dir(step); png = folder / f"{step:02d}_3D_SPATIAL_VIEW.png"
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)
    unique_layers = []
    seen = set()
    for item in layers:
        key = json.dumps(item, sort_keys=True)
        if key not in seen: seen.add(key); unique_layers.append(item)
    metadata = {"schema": "HUD_FAN_V5_5_STEP_ALGORITHM_VIEW", "step": step, "title": title,
                "source": "checkpoint values available after this exact step", "invented_geometry": False,
                "algorithm_source_manifest_sha256": _algorithm_manifest_hash(ctx),
                "coordinate_system": {"X": "forward", "Y": "right", "Z": "up", "unit": "mm"},
                "algorithm_focus": focus, "algorithm_purpose": purpose,
                "panels": ["step-specific 3-D algorithm evidence", "primary numerical diagnostic",
                           "secondary numerical/logic diagnostic"],
                "evidence": evidence,
                "layers": unique_layers, "image": png.name}
    if step == 25:
        metadata["scorecard_schema"] = "DYNAMIC_ACTUAL_OPERATOR_THRESHOLD_REASON_V1"
        metadata["gate_audit"] = ctx.data["scorecard"]
    metadata_path = folder / f"{step:02d}_3D_VIEW_METADATA.json"
    write_json(metadata_path, metadata)
    # Mirror the latest view into the source STEP folder so it is visible beside algorithm.py.
    manual_folder = MANUAL_ROOT / f"STEP_{step:02d}"
    manual_folder.mkdir(parents=True, exist_ok=True)
    latest_png = manual_folder / "LATEST_3D_SPATIAL_VIEW.png"
    latest_metadata = manual_folder / "LATEST_3D_VIEW_METADATA.json"
    shutil.copy2(png, latest_png)
    latest_record = dict(metadata)
    latest_record.update({"image": latest_png.name, "origin_run_step_image": str(png),
                          "latest_mirror_for_manual_inspection": True})
    write_json(latest_metadata, latest_record)
    return {"image": str(png), "metadata": str(folder / f"{step:02d}_3D_VIEW_METADATA.json"),
            "manual_latest_image": str(latest_png), "layer_count": len(unique_layers)}


def render_step17_branch_progress(ctx: Any, order: int, cycle: int, branch_index: int,
                                  branch_total: int, audit: dict[str, Any] | None,
                                  state: str) -> dict[str, str]:
    """Xuất ảnh tiến độ STEP 17 khi một nhánh bắt đầu hoặc vừa giải Fermat xong."""
    data = ctx.data
    trace = data["trace"]
    rays = data["rays"]
    folder = ctx.step_dir(17) / "LIVE_BRANCHES"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"ORDER_{order:02d}_CYCLE_{cycle:02d}_BRANCH_{branch_index:02d}"
    png = folder / f"{stem}.png"
    metadata_path = folder / f"{stem}.json"

    fig = plt.figure(figsize=(18, 9), dpi=130)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1.0), hspace=.28, wspace=.22)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_gradient = fig.add_subplot(grid[0, 1])
    ax_counts = fig.add_subplot(grid[1, 1])

    geometry: list[np.ndarray] = []
    geometry.append(_poly_surface(ax3d, data["m1"], "#ff9f1c", f"M1 parent trước order {order}"))
    geometry.append(_poly_surface(ax3d, data["m2"], "#e71d36", f"M2 parent trước order {order}"))
    if "visor" in data:
        geometry.append(_visor_surface(ax3d, data["visor"], "#00b4d8"))

    # Trước khi solver xong, hiển thị đường tia của parent đang được xử lý.
    ids = np.linspace(0, len(rays["rows"]) - 1, min(90, len(rays["rows"])), dtype=int)
    geometry.append(_trace_lines(ax3d, [rays["origins"][ids],
                                        trace["points"][0][ids], trace["points"][1][ids],
                                        trace["points"][2][ids]],
                                 "#8338ec", "parent reverse rays", len(ids), .20))

    if "fermat" in data:
        solution = data["fermat"]
        targets = np.asarray(solution["target_points"], float)
        geometry.append(_scatter(ax3d, targets, "#ff006e", "Fermat stationary targets",
                                 4, 1000, .45))
        refs = data["fan_refs"][rays["field_index"]]
        geometry.append(_trace_lines(ax3d, [trace["points"][1], targets, refs],
                                     "#ff006e", "Q1→Q2→P_ref", 90, .25))

    _set_labels(ax3d, f"Hình học đang xử lý — order {order}, branch {branch_index}/{branch_total}")
    _equal(ax3d, geometry)
    handles, labels = ax3d.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax3d.legend(unique.values(), unique.keys(), fontsize=6.5, loc="upper left")

    tolerance = float(ctx.config["solver"]["fermat_gradient_tolerance"])
    if audit is None:
        _text_panel(ax_gradient, "Trạng thái Fermat", [
            "ĐANG GIẢI — chưa có nghiệm mới để đánh giá",
            f"order: {order}", f"cycle: {cycle}",
            f"branch: {branch_index}/{branch_total}",
            f"parent rho sequence: {data.get('rho_sequence', [])}",
            f"số tia phải giải: {len(rays['rows'])}",
        ])
        _text_panel(ax_counts, "Điều kiện sẽ kiểm", [
            f"||∇OP|| ≤ {tolerance:.3g}",
            f"tối thiểu {100.0 * float(ctx.config['solver'].get('minimum_step17_candidate_fraction', 0.98)):.2f}% tia phải hội tụ",
            "target phải nằm trong aperture M2 hiện tại",
            "ảnh này sẽ được ghi đè khi branch giải xong",
        ])
    else:
        gradient = np.sort(np.asarray(data["fermat"]["gradient_norm"], float))
        ax_gradient.semilogy(np.arange(len(gradient)), np.maximum(gradient, 1e-16),
                             color="#ff006e", lw=.9)
        ax_gradient.axhline(tolerance, color="#d90429", ls="--", label="tolerance")
        ax_gradient.set(title="Fermat gradient của branch", xlabel="tia đã sắp xếp",
                        ylabel="||∇OP||")
        ax_gradient.grid(True, alpha=.2); ax_gradient.legend(fontsize=7)
        labels_count = ["solver success", "admissible", "inside aperture", "required minimum"]
        counts = [int(audit["success_count"]), int(audit["admissible_target_count"]),
                  int(audit["inside_current_aperture_count"]),
                  int(np.ceil(float(audit["minimum_admissible_fraction"])
                              * int(audit["ray_count"])))]
        colors = ["#2a9d8f", "#219ebc", "#8338ec", "#264653"]
        ax_counts.bar(labels_count, counts, color=colors)
        ax_counts.tick_params(axis="x", rotation=15)
        ax_counts.set(title=f"Chứng nhận branch: {'PASS' if audit['branch_may_continue'] else 'FAIL'}",
                      ylabel="số tia")
        ax_counts.grid(True, axis="y", alpha=.2)

    title = (f"STEP 17 LIVE — order {order}, cycle {cycle}, branch "
             f"{branch_index}/{branch_total} — {state}")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(.01, .012, "Ảnh tiến độ; không phải kết quả tối ưu cuối của STEP 17.",
             fontsize=8, color="#343a40")
    fig.subplots_adjust(left=.045, right=.975, bottom=.08, top=.90)
    temporary_png = folder / f".{stem}.tmp.png"
    fig.savefig(temporary_png, format="png", bbox_inches="tight")
    plt.close(fig)
    temporary_png.replace(png)

    record = {
        "schema": "HUD_FAN_V5_5_STEP17_LIVE_BRANCH_VIEW",
        "order": int(order), "cycle": int(cycle),
        "branch_index": int(branch_index), "branch_total": int(branch_total),
        "state": state, "parent_rho_sequence": data.get("rho_sequence", []),
        "ray_count": len(rays["rows"]), "audit": audit,
        "algorithm_source_manifest_sha256": _algorithm_manifest_hash(ctx),
        "image": str(png), "final_step_result": False,
    }
    write_json(metadata_path, record)
    write_json(ctx.step_dir(17) / "LIVE_PROGRESS.json", record)

    manual_folder = MANUAL_ROOT / "STEP_17"
    manual_folder.mkdir(parents=True, exist_ok=True)
    latest_png = manual_folder / "LATEST_BRANCH_PROGRESS.png"
    latest_json = manual_folder / "LATEST_BRANCH_PROGRESS.json"
    shutil.copy2(png, latest_png)
    write_json(latest_json, record | {"image": str(latest_png), "source_image": str(png)})
    return {"image": str(png), "metadata": str(metadata_path), "latest": str(latest_png)}


def _receiver_payload(ctx: Any, step: int) -> dict[str, Any]:
    """Tự chọn bề mặt nhận tia và hit points phù hợp nhất với dữ liệu của STEP."""
    data = ctx.data
    rays = data.get("rays", {})

    def payload(name: str, surface: Any, hits: np.ndarray, starts: np.ndarray | None,
                center: np.ndarray, frame: np.ndarray, role: str) -> dict[str, Any]:
        """Chuẩn hóa dữ liệu nhiều loại mặt về một schema vẽ chung."""
        hit_array = np.asarray(hits, float)
        start_array = None if starts is None else np.asarray(starts, float)
        field = np.asarray(rays.get("field_index", np.zeros(len(hit_array), int)), int)
        if len(field) != len(hit_array):
            field = np.zeros(len(hit_array), int)
        return {"name": name, "surface": surface, "hits": hit_array, "starts": start_array,
                "center": np.asarray(center, float), "frame": np.asarray(frame, float),
                "field_index": field, "role": role}

    # STEP 00–01 mới có dữ liệu visor; vẫn vẽ mặt/point cloud dù chưa có tia quang.
    if step <= 1:
        inputs = data["inputs"]
        surface = data.get("visor")
        frame = surface.frame if surface is not None else np.column_stack(
            [inputs["visor_U"], inputs["visor_V"], inputs["visor_N"]])
        center = surface.center if surface is not None else inputs["visor_center"]
        return payload("VISOR", surface, inputs["visor_inner"], None, center, frame,
                       "authority surface; characteristic rays are not available yet")

    # STEP 02–03 quan sát mặt phẳng ảnh ảo và các hướng field xuất phát từ mắt.
    if step in (2, 3):
        vi = data["vi"]
        hits = np.asarray([field["vi_point"] for field in vi["fields"]], float)
        starts = np.zeros_like(hits)
        frame = np.column_stack([vi["horizontal"], vi["vertical"], vi["normal"]])
        return payload("VIRTUAL_IMAGE_PLANE", None, hits, starts, vi["center"], frame,
                       "declared field rays received by the virtual-image plane")

    # STEP 04–05 quan sát pupil/eyebox như mặt nhận các tia ngược từ ảnh ảo.
    if step in (4, 5):
        if "rays" in data:
            hits = np.asarray(rays["origins"], float)
            starts = np.asarray([data["vi"]["fields"][i]["vi_point"]
                                 for i in rays["field_index"]], float)
        else:
            hits = np.asarray([[p["x_mm"], p["y_mm"], p["z_mm"]]
                               for p in data["pupils"]], float)
            starts = None
        frame = np.column_stack([np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                                 np.array([1., 0., 0.])])
        return payload("EYEBOX_PUPIL_PLANE", None, hits, starts, np.zeros(3), frame,
                       "pupil samples receiving reverse field rays")

    if step == 6:
        visor = data["visor"]
        return payload("VISOR", visor, data["visor_hit"]["point"], rays["origins"],
                       visor.center, visor.frame, "first optical surface receiving rays from eye")

    # Các STEP seed trước CI dùng footprint thật trên M1 của trace ngược hiện tại.
    if step in (7, 8, 9) and "trace" in data:
        trace = data["trace"]
        return payload("M1", data["m1"], trace["points"][1], trace["points"][0],
                       data["m1"].center, data["m1"].frame,
                       "planar-seed M1 receiving rays reflected by visor")

    if step == 10:
        hits = data["ci_m1"]["points_by_ray"]
        starts = data["visor_hit"]["point"]
        return payload("M1_CI_POINT_CLOUD", None, hits, starts,
                       data["m1"].center, data["m1"].frame,
                       "point-by-point CI targets receiving post-visor rays")

    if step == 11:
        direction = data["post_visor"]
        visor_hit = data["visor_hit"]["point"]
        m1_hit = data["m1"].intersect(visor_hit + 1e-4 * direction, direction, finite=False)
        hits = data["ci_m2"]["points_by_ray"]
        return payload("M2_CI_POINT_CLOUD", None, hits, m1_hit["point"],
                       data["m2"].center, data["m2"].frame,
                       "point-by-point CI targets receiving rays reflected by fitted M1")

    # Fermat trực tiếp tìm target trên M2; ưu tiên hiển thị đúng Q1→Q2 của bước đó.
    if step in (14, 15) and "fermat" in data:
        trace = data["trace"]
        return payload("M2_FERMAT_TARGET", data["m2"], data["fermat"]["target_points"],
                       trace["points"][1], data["m2"].center, data["m2"].frame,
                       "stationary Fermat targets receiving rays from M1")

    if step == 21 and "trace" in data:
        trace = data["trace"]
        return payload("M2", data["m2"], trace["points"][2], trace["points"][1],
                       data["m2"].center, data["m2"].frame,
                       "M2 footprint used together with signed obscuration geometry")

    # Forward validation kết thúc tại eye plane; STEP 25–26 ưu tiên footprint visor hard gate.
    if step == 24 and "forward" in data:
        forward = data["forward"]
        eye_frame = np.column_stack([np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                                     np.array([1., 0., 0.])])
        return payload("EYE_PLANE", None, forward["eye_hit"], forward["visor"],
                       np.zeros(3), eye_frame, "final forward rays received by eye plane")
    if step in (25, 26) and "forward" in data:
        forward = data["forward"]
        visor = data["visor"]
        return payload("VISOR_FINAL_FOOTPRINT", visor, forward["visor"], forward["m1"],
                       visor.center, visor.frame, "final visor footprint used by hard gate")

    # Khi đã có trace đầy đủ, display là bề mặt nhận cuối của đường tia ngược.
    if "trace" in data and "display" in data:
        trace = data["trace"]
        return payload("DISPLAY", data["display"], trace["landing"], trace["points"][-2],
                       data["display"].center, data["display"].frame,
                       "display receiving the current reverse-traced rays")

    raise RuntimeError(f"NO_RECEIVING_SURFACE_DATA_FOR_STEP_{step:02d}")


def render_surface_ray_view(ctx: Any, step: int) -> dict[str, Any]:
    """Vẽ mặt nhận tia, tia tới và footprint local bằng dữ liệu thật của một STEP."""
    item = _receiver_payload(ctx, step)
    hits = np.asarray(item["hits"], float)
    starts = item["starts"]
    finite_hits = np.all(np.isfinite(hits), axis=1)
    if starts is not None:
        finite_rays = finite_hits & np.all(np.isfinite(starts), axis=1)
    else:
        finite_rays = np.zeros(len(hits), bool)

    fig = plt.figure(figsize=(17, 8), dpi=125)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.0, .85), wspace=.27)
    ax3d = fig.add_subplot(grid[0], projection="3d")
    ax_footprint = fig.add_subplot(grid[1])
    ax_stats = fig.add_subplot(grid[2])
    geometry: list[np.ndarray] = []

    surface = item["surface"]
    if isinstance(surface, ChebVisor):
        geometry.append(_visor_surface(ax3d, surface, "#00b4d8"))
    elif isinstance(surface, PolySurface):
        geometry.append(_poly_surface(ax3d, surface, "#90e0ef", item["name"]))
    else:
        geometry.append(_scatter(ax3d, hits[finite_hits], "#90e0ef", item["name"],
                                 5, 1600, .35))

    if np.any(finite_rays):
        indices = np.where(finite_rays)[0]
        indices = indices[np.linspace(0, len(indices) - 1, min(120, len(indices)), dtype=int)]
        geometry.append(_trace_lines(ax3d, [starts[indices], hits[indices]], "#8338ec",
                                     "incident rays", len(indices), .28))
    geometry.append(_scatter(ax3d, hits[finite_hits], "#d90429", "ray hits",
                             6, 1800, .55))
    _set_labels(ax3d, f"STEP {step:02d}: tia tới và {item['name']}")
    _equal(ax3d, geometry)
    handles, labels = ax3d.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax3d.legend(unique.values(), unique.keys(), fontsize=7, loc="upper left")

    local = (hits - item["center"]) @ item["frame"]
    field = item["field_index"]
    plotted = finite_hits & np.all(np.isfinite(local[:, :2]), axis=1)
    scatter = ax_footprint.scatter(local[plotted, 0], local[plotted, 1],
                                   c=field[plotted], cmap="plasma", s=5, alpha=.5)
    if np.any(plotted):
        fig.colorbar(scatter, ax=ax_footprint, fraction=.045, label="field index")
    ax_footprint.set(title=f"Footprint trên {item['name']}", xlabel="local u [mm]",
                     ylabel="local v [mm]")
    ax_footprint.set_aspect("equal", adjustable="datalim"); ax_footprint.grid(True, alpha=.2)

    if np.any(finite_rays):
        lengths = np.linalg.norm(hits[finite_rays] - starts[finite_rays], axis=1)
        ax_stats.hist(lengths, bins=55, color="#8338ec", alpha=.8)
        ax_stats.set(title="Phân bố chiều dài đoạn tia tới", xlabel="độ dài [mm]", ylabel="số tia")
        statistics = {"segment_min_mm": float(np.min(lengths)),
                      "segment_median_mm": float(np.median(lengths)),
                      "segment_max_mm": float(np.max(lengths))}
    else:
        _text_panel(ax_stats, "Trạng thái dữ liệu tia", [
            "STEP này mới khai báo/fit bề mặt.",
            "Chưa có characteristic-ray hit để vẽ đoạn tia tới.",
            item["role"],
        ])
        statistics = {}
    ax_stats.grid(True, alpha=.2)

    fig.suptitle(f"STEP {step:02d} — RECEIVING SURFACE & RAY FOOTPRINT\n{item['role']}",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=.04, right=.97, bottom=.10, top=.86, wspace=.28)
    folder = ctx.step_dir(step)
    png = folder / f"{step:02d}_SURFACE_RAY_FOOTPRINT.png"
    temporary_png = folder / f".{step:02d}_SURFACE_RAY_FOOTPRINT.tmp.png"
    fig.savefig(temporary_png, format="png", bbox_inches="tight")
    plt.close(fig); temporary_png.replace(png)

    record = {"schema": "HUD_FAN_V5_5_RECEIVING_SURFACE_RAY_VIEW", "step": int(step),
              "receiver": item["name"], "role": item["role"],
              "hit_count": int(np.sum(finite_hits)), "input_row_count": len(hits),
              "ray_segment_count": int(np.sum(finite_rays)), "statistics": statistics,
              "image": str(png), "data_source": "CURRENT_STEP_CONTEXT_NO_INVENTED_RAYS",
              "algorithm_source_manifest_sha256": _algorithm_manifest_hash(ctx)}
    metadata = folder / f"{step:02d}_SURFACE_RAY_FOOTPRINT.json"
    write_json(metadata, record)
    manual_folder = MANUAL_ROOT / f"STEP_{step:02d}"
    manual_folder.mkdir(parents=True, exist_ok=True)
    latest_png = manual_folder / "LATEST_SURFACE_RAY_FOOTPRINT.png"
    latest_json = manual_folder / "LATEST_SURFACE_RAY_FOOTPRINT.json"
    shutil.copy2(png, latest_png)
    write_json(latest_json, record | {"image": str(latest_png), "source_image": str(png)})
    return {"image": str(png), "metadata": str(metadata), "latest": str(latest_png),
            "receiver": item["name"], "hit_count": int(np.sum(finite_hits))}


def _spot_snapshot_from_context(ctx: Any, label: str, source_step: int) -> dict[str, Any]:
    """Tạo snapshot spot từ checkpoint cũ bằng đúng landing và Fan reference của STEP."""
    trace = ctx.data["trace"]
    rays = ctx.data["rays"]
    return {
        "label": label, "source_step": int(source_step),
        "landing": np.asarray(trace["landing"], float).copy(),
        "valid": np.asarray(trace["valid"], bool).copy(),
        "display_center": np.asarray(trace["display_center"], float).copy(),
        "display_frame": np.asarray(trace["display_frame"], float).copy(),
        "references": np.asarray(ctx.data["fan_refs"], float).copy(),
        "field_index": np.asarray(rays["field_index"], int).copy(),
        "pupil_index": np.asarray(rays["pupil_index"], int).copy(),
        "field_count": int(rays["field_count"]),
        "pupil_count": int(rays["pupil_count"]),
        "field_grid": dict(ctx.config["field_grid"]),
        "available_metrics": [
            "RMS_2D_DEVIATION_FROM_STAGE_DYNAMIC_FAN_REFERENCE",
            "CENTROIDED_GEOMETRIC_SPOT_RMS_RADIUS_BY_FIELD_AND_PUPIL",
        ],
    }


PAPER_FAN_FIGURE8_SPOT_RMS_UM = {
    "PLANAR": 1371.9,
    "ORDER_2": 327.1,
    "ORDER_5": 15.7,
}


def _prepare_dual_spot_metrics(ctx: Any) -> tuple[list[dict[str, Any]], int, int]:
    """Chuẩn bị mapping RMS và spot RMS quanh centroid từ cùng snapshot thật."""
    snapshots = ctx.data.get("spot_evolution_snapshots", {})
    required = [
        ("PLANAR", "(a) Planar"),
        ("ORDER_2", "(b) Order-2 freeform"),
        ("ORDER_5", "(e) Final Order-5"),
    ]

    missing_required = [
        key
        for key, _ in required
        if key not in snapshots
    ]

    if missing_required:
        raise RuntimeError(
            "SPOT_EVOLUTION_REQUIRED_SNAPSHOTS_MISSING: "
            f"{missing_required}"
        )

    ordered_candidates = [
        ("PLANAR", "(a) Planar"),
        ("ORDER_2", "(b) Order-2 freeform"),
        ("ORDER_3", "(c) Best Order-3"),
        ("ORDER_4", "(d) Best Order-4"),
        ("ORDER_5", "(e) Final Order-5"),
    ]

    ordered = [
        item
        for item in ordered_candidates
        if item[0] in snapshots
    ]

    prepared: list[dict[str, Any]] = []
    for key, title in ordered:
        snap = snapshots[key]
        surface_state = snap.get("surface_state")
        if surface_state:
            basis_order = int(
                surface_state.get(
                    "basis_order",
                    int(key.split("_")[-1])
                    if key.startswith("ORDER_")
                    else 0,
                )
            )
            optimized_order = int(
                surface_state.get(
                    "shape_optimized_through_order",
                    basis_order,
                )
            )
            basis_only = bool(
                surface_state.get("basis_promotion_only", False)
                or surface_state.get("zero_pad_promotion", False)
            )

            if basis_only:
                panel_letter = {
                    "ORDER_3": "c",
                    "ORDER_4": "d",
                    "ORDER_5": "e",
                }.get(key, "?")

                title = (
                    f"({panel_letter}) O{basis_order} basis / "
                    f"O{optimized_order} optical shape"
                )
        landing = np.asarray(snap["landing"], float)
        frame = np.asarray(snap["display_frame"], float)
        field_index = np.asarray(snap["field_index"], int)
        pupil_index = np.asarray(
            snap.get("pupil_index", ctx.data["rays"]["pupil_index"]), int)
        references = np.asarray(snap["references"], float)
        mapping_xy = ((landing - references[field_index]) @ frame)[:, :2]
        finite = np.all(np.isfinite(mapping_xy), axis=1)
        valid = np.asarray(snap["valid"], bool) & finite
        field_count = int(snap["field_count"])
        pupil_count = int(snap.get("pupil_count", ctx.data["rays"]["pupil_count"]))

        chief = np.asarray(
            snap.get(
                "chief",
                ctx.data["rays"]["chief"],
            ),
            bool,
        )

        chief_spot = snap.get("chief_centered_spot")

        if chief_spot is None:
            raise RuntimeError(
                f"CHIEF_CENTERED_SPOT_SNAPSHOT_MISSING_{key}"
            )

        chief_bundle_rows: list[dict[str, Any]] = []

        for source_row in chief_spot["bundle_rows"]:
            ray_count = int(source_row["ray_count"])
            valid_count = int(source_row["valid_count"])

            chief_bundle_rows.append({
                "stage": key,
                "source_step": int(snap["source_step"]),
                "field_index": int(source_row["field_index"]),
                "pupil_index": int(source_row["pupil_index"]),
                "ray_count": ray_count,
                "valid_count": valid_count,
                "valid_fraction": (
                    float(valid_count / ray_count)
                    if ray_count > 0
                    else 0.0
                ),
                "lost_ray_count": int(
                    max(0, ray_count - valid_count)
                ),
                "lost_ray_fraction": (
                    float(max(0, ray_count - valid_count) / ray_count)
                    if ray_count > 0
                    else 1.0
                ),
                "chief_valid": bool(
                    source_row["chief_valid"]
                ),
                "underfilled": bool(
                    valid_count < ray_count
                ),
                "RMS_to_chief_mm": source_row.get(
                    "RMS_to_chief_mm"
                ),
                "max_radius_to_chief_mm": source_row.get(
                    "max_radius_to_chief_mm"
                ),
            })

        centroided_xy = np.full_like(mapping_xy, np.nan)
        bundle_rows: list[dict[str, Any]] = []
        for field in range(field_count):
            for pupil in range(pupil_count):
                bundle = (field_index == field) & (pupil_index == pupil)
                good = bundle & valid
                if not np.any(good):
                    bundle_rows.append({
                        "stage": key, "source_step": int(snap["source_step"]),
                        "field_index": field, "pupil_index": pupil,
                        "point_count": int(np.sum(bundle & finite)), "valid_count": 0,
                        "RMS_spot_radius_um": None, "max_spot_radius_um": None,
                        "centroid_reference_du_mm": None, "centroid_reference_dv_mm": None,
                        "centroid_reference_offset_um": None,
                    })
                    continue
                centroid = np.mean(mapping_xy[good], axis=0)
                drawable = bundle & finite
                centroided_xy[drawable] = mapping_xy[drawable] - centroid
                radius = np.linalg.norm(centroided_xy[good], axis=1)
                bundle_rows.append({
                    "stage": key, "source_step": int(snap["source_step"]),
                    "field_index": field, "pupil_index": pupil,
                    "point_count": int(np.sum(drawable)), "valid_count": int(np.sum(good)),
                    "RMS_spot_radius_um": float(np.sqrt(np.mean(radius ** 2)) * 1000.0),
                    "max_spot_radius_um": float(np.max(radius) * 1000.0),
                    "centroid_reference_du_mm": float(centroid[0]),
                    "centroid_reference_dv_mm": float(centroid[1]),
                    "centroid_reference_offset_um": float(np.linalg.norm(centroid) * 1000.0),
                })

        mapping_radius = np.linalg.norm(mapping_xy[finite], axis=1)
        mapping_valid_radius = np.linalg.norm(mapping_xy[valid], axis=1)
        spot_drawable = np.all(np.isfinite(centroided_xy), axis=1)
        spot_radius = np.linalg.norm(centroided_xy[valid], axis=1)
        centroid_offsets = np.asarray([
            float(row["centroid_reference_offset_um"]) / 1000.0
            for row in bundle_rows if row["centroid_reference_offset_um"] is not None
            for _ in range(int(row["valid_count"]))
        ], float)
        mapping_rms_mm = float(np.sqrt(np.mean(mapping_radius ** 2)))
        mapping_valid_rms_mm = float(np.sqrt(np.mean(mapping_valid_radius ** 2)))
        spot_rms_mm = float(np.sqrt(np.mean(spot_radius ** 2)))
        centroid_reference_rms_mm = float(np.sqrt(np.mean(centroid_offsets ** 2)))
        decomposition_error = abs(mapping_valid_rms_mm ** 2 - spot_rms_mm ** 2 -
                                  centroid_reference_rms_mm ** 2)
        prepared.append({
            "key": key, "title": title, "snapshot": snap,
            "surface_state": surface_state,
            "field_index": field_index, "pupil_index": pupil_index,
            "mapping_xy": mapping_xy, "mapping_finite": finite,
            "centroided_xy": centroided_xy, "spot_drawable": spot_drawable,
            "valid": valid, "bundle_rows": bundle_rows,
            "mapping_rms_um": mapping_rms_mm * 1000.0,
            "mapping_valid_ray_rms_um": mapping_valid_rms_mm * 1000.0,
            "mapping_max_um": float(np.max(mapping_radius) * 1000.0),
            "spot_rms_um": spot_rms_mm * 1000.0,
            "spot_max_um": float(np.max(spot_radius) * 1000.0),
            "centroid_reference_rms_um": centroid_reference_rms_mm * 1000.0,
            "rms_variance_decomposition_error_mm2": float(decomposition_error),
            "paper_benchmark_um": PAPER_FAN_FIGURE8_SPOT_RMS_UM.get(key),
            "chief": chief,
            "chief_bundle_rows": chief_bundle_rows,
            "chief_spot_rms_mm": chief_spot["RMS_spot_radius_mm"],
            "chief_spot_max_mm": chief_spot["max_spot_radius_mm"],
            "field_count": field_count,
            "pupil_count": pupil_count,
        })

    first_grid = prepared[0]["snapshot"]["field_grid"]
    nh = int(first_grid["horizontal_count"]); nv = int(first_grid["vertical_count"])
    if any(int(item["snapshot"]["field_count"]) != nh * nv for item in prepared):
        raise RuntimeError("SPOT_EVOLUTION_FIELD_GRID_MISMATCH")
    return prepared, nh, nv


def _render_spot_grid(prepared: list[dict[str, Any]], nh: int, nv: int,
                      mode: str, target: Path) -> float:
    """Vẽ ảnh ba trạng thái cho mapping RMS hoặc centroided spot RMS."""
    if mode not in ("mapping", "centroided"):
        raise ValueError("SPOT_GRID_MODE_MUST_BE_MAPPING_OR_CENTROIDED")
    xy_key = "mapping_xy" if mode == "mapping" else "centroided_xy"
    draw_key = "mapping_finite" if mode == "mapping" else "spot_drawable"
    rms_key = "mapping_rms_um" if mode == "mapping" else "spot_rms_um"
    components = [np.abs(item[xy_key][item[draw_key]]).ravel() for item in prepared]
    common_limit = max(1.05 * float(np.max(np.concatenate(components))), 1e-6)

    stage_count = len(prepared)
    figure_width = max(20.0, 6.2 * stage_count)

    fig = plt.figure(
        figsize=(figure_width, 9.4),
        dpi=135,
    )
    outer = fig.add_gridspec(
        1,
        stage_count,
        wspace=.13,
        left=.025,
        right=.985,
        bottom=.105,
        top=.875,
    )
    for stage_index, item in enumerate(prepared):
        sub = outer[stage_index].subgridspec(nv + 1, nh, height_ratios=[.32] + [1.] * nv,
                                              hspace=.22, wspace=.18)
        title_ax = fig.add_subplot(sub[0, :]); title_ax.set_axis_off()
        if mode == "mapping":
            label = (f"{item['title']}\nMapping RMS to Fan reference: "
                     f"{item[rms_key]:.3f} µm\n"
                     f"valid-ray split: spot {item['spot_rms_um']:.3f} µm ⊕ centroid-reference "
                     f"{item['centroid_reference_rms_um']:.3f} µm")
        else:
            benchmark = item["paper_benchmark_um"]

            if benchmark is None:
                label = (
                    f"{item['title']}\n"
                    f"Centroided spot RMS: {item[rms_key]:.3f} µm\n"
                    "No paper benchmark assigned to this intermediate order"
                )
            else:
                ratio = item[rms_key] / benchmark
                label = (
                    f"{item['title']}\n"
                    f"Centroided spot RMS: {item[rms_key]:.3f} µm\n"
                    f"Paper Fig. 8: {benchmark:.1f} µm "
                    f"(current/paper {ratio:.2f}×)"
                )
        title_ax.text(.5, .55, label, ha="center", va="center", fontsize=10.5,
                      fontweight="bold")
        field_index = item["field_index"]
        for field in range(nh * nv):
            row, column = divmod(field, nh)
            ax = fig.add_subplot(sub[row + 1, column])
            mask = (field_index == field) & item[draw_key]
            good = mask & item["valid"]
            bad = mask & ~item["valid"]
            ax.scatter(item[xy_key][good, 0], item[xy_key][good, 1], s=2.2,
                       color="#ef233c", alpha=.62, linewidths=0)
            if np.any(bad):
                ax.scatter(item[xy_key][bad, 0], item[xy_key][bad, 1], s=10,
                           marker="x", color="#111111", linewidths=.6)
            ax.axhline(0, color="#adb5bd", lw=.35); ax.axvline(0, color="#adb5bd", lw=.35)
            ax.set_xlim(-common_limit, common_limit); ax.set_ylim(-common_limit, common_limit)
            ax.set_aspect("equal"); ax.grid(True, alpha=.18, lw=.35)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(length=0)
            ax.set_title(f"FOV{field + 1}", fontsize=7.2, pad=2)

    if mode == "mapping":
        fig.suptitle("STEP 20A — REFERENCE-LOCKED MAPPING RMS", fontsize=15,
                     fontweight="bold")
        note = (f"Common axes ±{common_limit:.6g} mm • Reference is NOT removed • "
                "RMS includes spot spread and bundle-centroid mapping error • "
                "Not the paper RMS spot radius • black × = physical-invalid ray")
    else:
        fig.suptitle("STEP 20 — PAPER-STYLE CENTROIDED GEOMETRIC SPOT RMS", fontsize=15,
                     fontweight="bold")
        note = (f"Common axes ±{common_limit:.6g} mm • One centroid removed independently per "
                "(field, pupil) bundle • RMS uses valid rays only • Paper values are visual "
                "benchmarks; sampling and geometry differ • Not diffraction spot")
    fig.text(.5, .025, note, ha="center", fontsize=8, color="#343a40")
    temporary = target.with_name(f".{target.stem}.tmp.png")
    fig.savefig(temporary, format="png", bbox_inches="tight")
    plt.close(fig); temporary.replace(target)
    return common_limit


def _step20_bundle_matrix(
    item: dict[str, Any],
    metric: str,
) -> np.ndarray:
    """Tạo ma trận field x pupil từ dữ liệu bundle của stage."""
    field_count = int(item["field_count"])
    pupil_count = int(item["pupil_count"])

    matrix = np.full(
        (field_count, pupil_count),
        np.nan,
        dtype=float,
    )

    for row in item["chief_bundle_rows"]:
        field_index = int(row["field_index"])
        pupil_index = int(row["pupil_index"])

        if metric == "CHIEF_RMS_MM":
            value = row.get("RMS_to_chief_mm")
        elif metric == "RAY_LOSS_PERCENT":
            value = 100.0 * float(
                row["lost_ray_fraction"]
            )
        else:
            raise ValueError(
                "STEP20_BUNDLE_HEATMAP_METRIC_INVALID"
            )

        if value is not None and np.isfinite(float(value)):
            matrix[field_index, pupil_index] = float(value)

    return matrix


def _render_step20_bundle_heatmap(
    prepared: list[dict[str, Any]],
    metric: str,
    target: Path,
) -> float:
    """Vẽ heatmap chẩn đoán theo từng bundle field x pupil cho các stage."""
    matrices = [
        _step20_bundle_matrix(item, metric)
        for item in prepared
    ]

    finite_values = np.concatenate([
        matrix[np.isfinite(matrix)]
        for matrix in matrices
        if np.any(np.isfinite(matrix))
    ])

    maximum = (
        float(np.max(finite_values))
        if len(finite_values)
        else 1.0
    )
    maximum = max(maximum, 1e-12)

    stage_count = len(prepared)
    fig, axes = plt.subplots(
        1,
        stage_count,
        figsize=(5.6 * stage_count, 11.0),
        dpi=135,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    image = None

    for stage_index, (ax, item, matrix) in enumerate(
        zip(axes, prepared, matrices)
    ):
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=(
                "viridis"
                if metric == "CHIEF_RMS_MM"
                else "magma"
            ),
            vmin=0.0,
            vmax=maximum,
        )

        ax.set_title(
            item["title"],
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Pupil index")

        pupil_count = int(item["pupil_count"])
        ax.set_xticks(np.arange(pupil_count))
        ax.set_xticklabels(
            [
                f"P{index:02d}"
                for index in range(pupil_count)
            ],
            rotation=90,
            fontsize=6,
        )

        if stage_index == 0:
            field_count = int(item["field_count"])
            ax.set_ylabel("Field index")
            ax.set_yticks(np.arange(field_count))
            ax.set_yticklabels(
                [
                    f"F{index:02d}"
                    for index in range(field_count)
                ],
                fontsize=5.5,
            )

        if np.any(np.isfinite(matrix)):
            worst_flat_index = int(
                np.nanargmax(matrix)
            )
            worst_field, worst_pupil = np.unravel_index(
                worst_flat_index,
                matrix.shape,
            )
            ax.scatter(
                [worst_pupil],
                [worst_field],
                marker="s",
                facecolors="none",
                edgecolors="#00ffff",
                linewidths=1.5,
                s=70,
            )

    if image is None:
        raise RuntimeError(
            "STEP20_BUNDLE_HEATMAP_HAS_NO_IMAGE"
        )

    if metric == "CHIEF_RMS_MM":
        title = (
            "STEP 20 — CHIEF-CENTERED BUNDLE RMS "
            "BY FIELD × PUPIL"
        )
        colorbar_label = "RMS to chief ray (mm)"
    else:
        title = (
            "STEP 20 — PHYSICAL RAY LOSS "
            "BY FIELD × PUPIL"
        )
        colorbar_label = "Lost physical rays (%)"

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )

    colorbar = fig.colorbar(
        image,
        ax=axes.tolist(),
        shrink=.92,
        pad=.015,
    )
    colorbar.set_label(colorbar_label)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.tmp.png"
    )
    fig.savefig(
        temporary,
        format="png",
        bbox_inches="tight",
    )
    plt.close(fig)
    temporary.replace(target)

    return maximum


def render_spot_evolution(ctx: Any) -> dict[str, Any]:
    """Xuất duy nhất ảnh centroided geometric spot RMS tương ứng cách trình bày Fig. 8."""
    prepared, nh, nv = _prepare_dual_spot_metrics(ctx)
    folder = ctx.step_dir(20)

    png = (
        folder
        / "20_SPOT_EVOLUTION_PLANAR_ORDER2_ORDER5.png"
    )
    axis_limit = _render_spot_grid(
        prepared,
        nh,
        nv,
        "centroided",
        png,
    )

    bundle_rms_png = (
        folder
        / "20_BUNDLE_RMS_HEATMAP_PLANAR_O2_O3_O4_O5.png"
    )
    bundle_rms_maximum = _render_step20_bundle_heatmap(
        prepared,
        "CHIEF_RMS_MM",
        bundle_rms_png,
    )

    ray_loss_png = (
        folder
        / "20_PHYSICAL_RAY_LOSS_HEATMAP_PLANAR_O2_O3_O4_O5.png"
    )
    ray_loss_maximum = _render_step20_bundle_heatmap(
        prepared,
        "RAY_LOSS_PERCENT",
        ray_loss_png,
    )

    metric_rows = []
    for item in prepared:
        snap = item["snapshot"]
        metric_rows.extend(item["bundle_rows"])
        metric_rows.append({
            "stage": item["key"], "source_step": snap["source_step"],
            "field_index": "ALL", "pupil_index": "ALL",
            "point_count": int(np.sum(item["spot_drawable"])),
            "valid_count": int(np.sum(item["valid"])),
            "RMS_spot_radius_um": item["spot_rms_um"],
            "max_spot_radius_um": item["spot_max_um"],
            "centroid_reference_du_mm": None, "centroid_reference_dv_mm": None,
            "centroid_reference_offset_um": item["centroid_reference_rms_um"],
        })

    csv_path = folder / "20_SPOT_EVOLUTION_METRICS.csv"
    write_csv(csv_path, metric_rows)

    chief_bundle_rows = [
        row
        for item in prepared
        for row in item["chief_bundle_rows"]
    ]

    chief_bundle_csv = (
        folder
        / "20_CHIEF_CENTERED_BUNDLE_METRICS.csv"
    )
    write_csv(
        chief_bundle_csv,
        chief_bundle_rows,
    )

    summary = {
        "schema": "HUD_FAN_V5_5_PAPER_STYLE_CENTROIDED_SPOT_EVOLUTION",
        "image": str(png),
        "bundle_RMS_heatmap": str(bundle_rms_png),
        "physical_ray_loss_heatmap": str(ray_loss_png),
        "metric_csv": str(csv_path),
        "chief_bundle_metric_csv": str(chief_bundle_csv),
        "bundle_RMS_heatmap_maximum_mm": bundle_rms_maximum,
        "physical_ray_loss_heatmap_maximum_percent": ray_loss_maximum,
        "stage_sequence": [
            item["key"]
            for item in prepared
        ],
        "metric": "CENTROIDED_GEOMETRIC_SPOT_RMS_RADIUS_BY_FIELD_AND_PUPIL",
        "grouping": "ONE_BUNDLE_PER_FIELD_AND_PUPIL", "centroid_removed": True,
        "formula": "sqrt(mean(||xy_ray-centroid_xy(field,pupil)||^2))",
        "invalid_ray_policy": "PHYSICAL_INVALID_RAYS_ARE_DRAWN_AS_BLACK_X_AND_EXCLUDED_FROM_RMS",
        "not_diffraction_spot": True, "common_axis_limit_mm": axis_limit,
        "field_grid": {"horizontal_count": nh, "vertical_count": nv},
        "paper_benchmark": {
            "source": "Fan_et_al_Figure_8",
            "values_um": PAPER_FAN_FIGURE8_SPOT_RMS_UM,
            "role": "VISUAL_REFERENCE_ONLY_NOT_ACCEPTANCE_THRESHOLD",
            "comparability_warning":
                "Same spot-radius concept, but paper sampling, windshield, geometry and optimization differ.",
            "paper_values_reused_as_current_run_data": False,
        },
        "stages": [{
            "label": item["key"], "source_step": item["snapshot"]["source_step"],
            "RMS_spot_radius_um": item["spot_rms_um"],
            "max_spot_radius_um": item["spot_max_um"],
            "paper_Figure8_RMS_spot_um": item["paper_benchmark_um"],
            "current_over_paper_ratio": (
                item["spot_rms_um"] / item["paper_benchmark_um"]
                if item["paper_benchmark_um"] is not None
                else None
            ),
            "valid_count": int(np.sum(item["valid"])),
            "point_count": int(np.sum(item["spot_drawable"])),
            "chief_centered_RMS_mm": item["chief_spot_rms_mm"],
            "chief_centered_max_radius_mm": item["chief_spot_max_mm"],
            "valid_bundle_count": int(
                sum(
                    row["RMS_to_chief_mm"] is not None
                    for row in item["chief_bundle_rows"]
                )
            ),
            "required_bundle_count": int(
                len(item["chief_bundle_rows"])
            ),
            "underfilled_bundle_count": int(
                sum(
                    bool(row["underfilled"])
                    for row in item["chief_bundle_rows"]
                )
            ),
            "surface_state": item["surface_state"],
            "validity_mode": item["snapshot"].get(
                "validity_mode",
                "LEGACY_UNKNOWN",
            ),
        } for item in prepared],
        "data_source": "CURRENT_RUN_RAY_LANDINGS", "paper_values_reused": False,
        "algorithm_source_manifest_sha256": _algorithm_manifest_hash(ctx),
        "single_step20_spot_output": True,
    }
    metadata = folder / "20_SPOT_EVOLUTION_METADATA.json"
    write_json(metadata, summary)

    # Remove obsolete dual-output artifacts if an old STEP 20 folder is rendered
    # again.  The canonical files above now contain the paper-style calculation.
    for obsolete in (
        "20_CENTROIDED_SPOT_EVOLUTION_PLANAR_ORDER2_ORDER5.png",
        "20_CENTROIDED_SPOT_EVOLUTION_METADATA.json",
        "20_CENTROIDED_SPOT_EVOLUTION_METRICS.csv",
        "20_DUAL_RMS_DATA_FLOW.json",
    ):
        (folder / obsolete).unlink(missing_ok=True)

    manual_folder = MANUAL_ROOT / "STEP_20"; manual_folder.mkdir(parents=True, exist_ok=True)
    latest = manual_folder / "LATEST_SPOT_EVOLUTION.png"
    shutil.copy2(png, latest)
    latest_bundle_rms = (
        manual_folder
        / "LATEST_BUNDLE_RMS_HEATMAP.png"
    )
    latest_ray_loss = (
        manual_folder
        / "LATEST_PHYSICAL_RAY_LOSS_HEATMAP.png"
    )

    shutil.copy2(
        bundle_rms_png,
        latest_bundle_rms,
    )
    shutil.copy2(
        ray_loss_png,
        latest_ray_loss,
    )
    write_json(manual_folder / "LATEST_SPOT_EVOLUTION.json",
               summary | {"image": str(latest), "source_image": str(png)})
    for obsolete in ("LATEST_CENTROIDED_SPOT_EVOLUTION.png",
                     "LATEST_CENTROIDED_SPOT_EVOLUTION.json",
                     "LATEST_DUAL_RMS_DATA_FLOW.json"):
        (manual_folder / obsolete).unlink(missing_ok=True)
    return {
        "image": str(png),
        "bundle_RMS_heatmap": str(bundle_rms_png),
        "physical_ray_loss_heatmap": str(ray_loss_png),
        "metadata": str(metadata),
        "metric_csv": str(csv_path),
        "chief_bundle_metric_csv": str(chief_bundle_csv),
        "latest": str(latest),
        "latest_bundle_RMS_heatmap": str(latest_bundle_rms),
        "latest_physical_ray_loss_heatmap": str(latest_ray_loss),
        "stages": summary["stages"],
        "metric": summary["metric"],
        "paper_style": True,
    }


def render_saved_step(step: int, run_dir: Path | None = None) -> dict[str, Any]:
    """Nạp checkpoint rồi vẽ lại STEP, không chạy lại quang học."""
    if run_dir is None:
        active = json.loads((MANUAL_ROOT / "ACTIVE_RUN.json").read_text(encoding="utf-8"))
        run_dir = Path(active["run_dir"])
    checkpoint = Path(run_dir) / "CHECKPOINTS" / f"STEP_{step:02d}" / "state.pkl.gz"
    if not checkpoint.exists():
        raise RuntimeError(f"CHECKPOINT_NOT_FOUND_FOR_STEP_{step:02d}: {checkpoint}")
    with gzip.open(checkpoint, "rb") as stream:
        ctx = __import__("pickle").load(stream)
    return render_step(ctx, step)


def render_saved_surface_ray_view(step: int, run_dir: Path | None = None) -> dict[str, Any]:
    """Nạp checkpoint rồi vẽ lại mặt nhận tia và footprint của một STEP đã chạy."""
    if run_dir is None:
        active = json.loads((MANUAL_ROOT / "ACTIVE_RUN.json").read_text(encoding="utf-8"))
        run_dir = Path(active["run_dir"])
    checkpoint = Path(run_dir) / "CHECKPOINTS" / f"STEP_{step:02d}" / "state.pkl.gz"
    if not checkpoint.exists():
        raise RuntimeError(f"CHECKPOINT_NOT_FOUND_FOR_STEP_{step:02d}: {checkpoint}")
    with gzip.open(checkpoint, "rb") as stream:
        ctx = __import__("pickle").load(stream)
    return render_surface_ray_view(ctx, step)


def render_saved_spot_evolution(run_dir: Path | None = None) -> dict[str, Any]:
    """Tạo lại ảnh so sánh spot từ checkpoint STEP 08, 12 và 17 của một run."""
    if run_dir is None:
        active = json.loads((MANUAL_ROOT / "ACTIVE_RUN.json").read_text(encoding="utf-8"))
        run_dir = Path(active["run_dir"])
    run_dir = Path(run_dir)
    contexts = {}
    for label, step in (("PLANAR", 8), ("ORDER_2", 12), ("ORDER_5", 17)):
        checkpoint = run_dir / "CHECKPOINTS" / f"STEP_{step:02d}" / "state.pkl.gz"
        if not checkpoint.exists():
            raise RuntimeError(f"SPOT_EVOLUTION_CHECKPOINT_MISSING: {checkpoint}")
        with gzip.open(checkpoint, "rb") as stream:
            contexts[label] = __import__("pickle").load(stream)
    base_checkpoint = run_dir / "CHECKPOINTS" / "STEP_20" / "state.pkl.gz"
    if not base_checkpoint.exists():
        raise RuntimeError(f"SPOT_EVOLUTION_CHECKPOINT_MISSING: {base_checkpoint}")
    with gzip.open(base_checkpoint, "rb") as stream:
        base = __import__("pickle").load(stream)
    base.data["spot_evolution_snapshots"] = {
        label: _spot_snapshot_from_context(contexts[label], label, step)
        for label, step in (("PLANAR", 8), ("ORDER_2", 12), ("ORDER_5", 17))
    }
    return render_spot_evolution(base)


def render_all_saved(run_dir: Path) -> list[dict[str, Any]]:
    """Vẽ lại mọi STEP có checkpoint trong run."""
    output = []
    for step in range(27):
        folder = Path(run_dir) / f"STEP_{step:02d}"
        image = folder / f"{step:02d}_3D_SPATIAL_VIEW.png"
        metadata = folder / f"{step:02d}_3D_VIEW_METADATA.json"
        if image.exists() and metadata.exists():
            print(f"reuse STEP_{step:02d}", flush=True)
            info = json.loads(metadata.read_text(encoding="utf-8"))
            output.append({"image": str(image), "metadata": str(metadata),
                           "layer_count": len(info.get("layers", []))})
        else:
            print(f"render STEP_{step:02d}", flush=True)
            output.append(render_saved_step(step, run_dir))
    write_json(Path(run_dir) / "3D_VIEW_INDEX.json", {"schema": "HUD_FAN_V5_5_3D_VIEW_INDEX", "views": output})
    return output
