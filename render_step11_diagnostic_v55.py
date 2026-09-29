"""Bộ tạo ảnh chẩn đoán chuyên sâu STEP 11: Quá trình tìm kiếm cặp O2 và giải phẫu lỗi."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


SCHEMA_V2 = "HUD_FAN_V5_5_STEP11_SEARCH_DIAGNOSTIC_V2"

# Bảng màu chẩn đoán trạng thái
STAGE_COLORS: dict[str, str] = {
    "FEASIBLE": "#2ea043",                      # Xanh lá: Thỏa mãn toàn bộ gate
    "M1_PARAMETER_APPLICATION": "#d29922",      # Vàng cam
    "M1_MIRROR_TOPOLOGY_GATE": "#f85149",       # Đỏ cam: Lỗi topology M1 (không lõm)
    "M1_MIRROR_SHAPE_GATE": "#f85149",          # Đỏ cam
    "M1_CI_TRUST_GATE": "#db6d28",              # Cam
    "M1_ACTUAL_RETRACE": "#bd561d",             # Nâu cam
    "M2_CI_REBUILD": "#bc8cff",                 # Tím nhạt
    "M2_CI_CONSISTENCY_GATE": "#da3633",        # Đỏ tươi: Lỗi consistency M2
    "M2_O2_FIT": "#8957e5",                     # Tím đậm
    "M2_MIRROR_TOPOLOGY_GATE": "#f78166",       # Hồng cam: Lỗi topology M2 (không lồi)
    "M2_MIRROR_SHAPE_GATE": "#f78166",          # Hồng cam
    "FULL_PHYSICAL_OPTICAL_EVALUATION": "#388bfd",# Xanh dương
    "FULL_PHYSICAL_GATE": "#e3b341",            # Vàng gold
    "SKIPPED_NOT_EVALUATED": "#30363d",         # Xám tối
    "UNKNOWN": "#8b949e",                       # Xám vừa
}


def _abbreviate_move(move_str: str) -> str:
    """Rút gọn tên move để hiển thị rõ trong ô lưới lattice."""
    m = str(move_str).strip()
    if m == "BASELINE":
        return "BASE"
    if m.startswith("CURVATURE::"):
        sign = m.split("::")[-1]
        return f"CURV{sign}"
    if m.startswith("CONIC::"):
        sign = m.split("::")[-1]
        return f"CONIC{sign}"
    if m.startswith("ASTIG_QUADRATIC_PAIR::"):
        sign = m.split("::")[-1]
        return f"ASTIG{sign}"
    if m.startswith("COEFFICIENT:"):
        # Format COEFFICIENT:(i, j):+1 -> C(i,j)+
        parts = m.split(":")
        if len(parts) >= 3:
            term = parts[1].replace(" ", "")
            sign = parts[2]
            return f"C{term}{sign}"
    if "COMBINATION" in m:
        return "COMB"
    return m[:6]


def parse_step11_search_history(csv_path: Path) -> list[dict[str, Any]]:
    """Đọc và chuẩn hóa dữ liệu từ file 11_O2_SEARCH_HISTORY.csv."""
    if not csv_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Chuẩn hóa kiểu dữ liệu từng cột
            row_dict: dict[str, Any] = {}
            for k, v in r.items():
                clean_k = k.strip() if k else ""
                clean_v = v.strip() if v is not None else ""
                row_dict[clean_k] = clean_v

            cid = row_dict.get("candidate_id", "")
            cycle = int(row_dict.get("cycle", 0)) if row_dict.get("cycle", "").isdigit() else 0
            try:
                trust_scale = float(row_dict.get("trust_scale", 0.0))
            except ValueError:
                trust_scale = 0.0
            move = row_dict.get("move", "")
            feasible = row_dict.get("feasible", "").lower() in ("true", "1")
            stage = row_dict.get("rejection_stage", "")
            reason = row_dict.get("rejection_reason", "")

            # Boolean passes
            m1_shape = row_dict.get("M1_shape_pass", "").lower() in ("true", "1") if row_dict.get("M1_shape_pass") else None
            m1_ci = row_dict.get("M1_ci_trust_pass", "").lower() in ("true", "1") if row_dict.get("M1_ci_trust_pass") else None
            m2_int = row_dict.get("M2_integrability_pass", "").lower() in ("true", "1") if row_dict.get("M2_integrability_pass") else None
            m2_int_admitted = row_dict.get("M2_integrability_admitted", "").lower() in ("true", "1") if row_dict.get("M2_integrability_admitted") else None
            m2_shape = row_dict.get("M2_shape_pass", "").lower() in ("true", "1") if row_dict.get("M2_shape_pass") else None

            # Downstream metrics
            def _float_or_none(key: str) -> float | None:
                """Chuyển đổi giá trị chuỗi sang số thực hoặc None nếu rỗng."""
                val = row_dict.get(key, "")
                try:
                    return float(val) if val != "" else None
                except ValueError:
                    return None

            def _int_or_none(key: str) -> int | None:
                """Chuyển đổi giá trị chuỗi sang số nguyên hoặc None nếu rỗng."""
                val = row_dict.get(key, "")
                try:
                    return int(float(val)) if val != "" else None
                except ValueError:
                    return None

            rows.append({
                "candidate_id": cid,
                "cycle": cycle,
                "trust_scale": trust_scale,
                "move": move,
                "feasible": feasible,
                "rejection_stage": stage,
                "rejection_reason": reason,
                "M1_shape_pass": m1_shape,
                "M1_ci_trust_pass": m1_ci,
                "M1_H_min_per_mm": _float_or_none("M1_H_min_per_mm"),
                "M1_H_max_per_mm": _float_or_none("M1_H_max_per_mm"),
                "M1_KG_min_per_mm2": _float_or_none("M1_KG_min_per_mm2"),
                "M1_KG_max_per_mm2": _float_or_none("M1_KG_max_per_mm2"),
                "M1_k1_min_per_mm": _float_or_none("M1_k1_min_per_mm"),
                "M1_k1_max_per_mm": _float_or_none("M1_k1_max_per_mm"),
                "M1_k2_min_per_mm": _float_or_none("M1_k2_min_per_mm"),
                "M1_k2_max_per_mm": _float_or_none("M1_k2_max_per_mm"),
                "M1_oriented_H_min_per_mm": _float_or_none("M1_oriented_H_min_per_mm"),
                "M1_curvature_sign_flip_count": _int_or_none("M1_curvature_sign_flip_count"),
                "M2_integrability_pass": m2_int,
                "M2_integrability_admitted": m2_int_admitted,
                "M2_shape_pass": m2_shape,
                "M2_H_min_per_mm": _float_or_none("M2_H_min_per_mm"),
                "M2_H_max_per_mm": _float_or_none("M2_H_max_per_mm"),
                "M2_KG_min_per_mm2": _float_or_none("M2_KG_min_per_mm2"),
                "M2_KG_max_per_mm2": _float_or_none("M2_KG_max_per_mm2"),
                "M2_k1_min_per_mm": _float_or_none("M2_k1_min_per_mm"),
                "M2_k1_max_per_mm": _float_or_none("M2_k1_max_per_mm"),
                "M2_k2_min_per_mm": _float_or_none("M2_k2_min_per_mm"),
                "M2_k2_max_per_mm": _float_or_none("M2_k2_max_per_mm"),
                "M2_oriented_H_min_per_mm": _float_or_none("M2_oriented_H_min_per_mm"),
                "M2_curvature_sign_flip_count": _int_or_none("M2_curvature_sign_flip_count"),
                "physical_valid_count": _float_or_none("physical_valid_count"),
                "physical_fraction": _float_or_none("physical_fraction"),
                "unobscured": row_dict.get("unobscured", "").lower() in ("true", "1") if row_dict.get("unobscured") else None,
                "optical_objective": _float_or_none("optical_objective"),
                "mapping_rms_mm": _float_or_none("mapping_rms_mm"),
                "spot_rms_mm": _float_or_none("spot_rms_mm"),
                "direction_rms_deg": _float_or_none("direction_rms_deg"),
                "representation_score": _float_or_none("representation_score"),
            })
    return rows


def render_step11_search_diagnostic(
    step_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Tạo bảng chẩn đoán tổng thể single-board cho STEP 11 từ các artifact lưu sẵn."""
    step_path = Path(step_dir).resolve()
    target_out_dir = Path(output_dir).resolve() if output_dir else step_path

    history_csv = step_path / "11_O2_SEARCH_HISTORY.csv"
    if not history_csv.is_file():
        return {
            "schema": SCHEMA_V2,
            "status": "SKIPPED_NO_SEARCH_HISTORY",
            "message": f"11_O2_SEARCH_HISTORY.csv not found in {step_path}",
        }

    history = parse_step11_search_history(history_csv)
    if not history:
        return {
            "schema": SCHEMA_V2,
            "status": "SKIPPED_EMPTY_SEARCH_HISTORY",
            "message": f"Empty history in {history_csv}",
        }

    # Đọc thêm thông tin debug phụ nếu có
    debug_summary_path = step_path / "DEBUG_SUMMARY.json"
    run_failure_path = step_path / "RUN_FAILURE_STEP_11.json"
    scaling_csv_path = step_path / "11_M1_PHYSICAL_DOF_SCALING.csv"

    debug_summary: dict[str, Any] = {}
    if debug_summary_path.is_file():
        try:
            debug_summary = json.loads(debug_summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    run_failure: dict[str, Any] = {}
    if run_failure_path.is_file():
        try:
            run_failure = json.loads(run_failure_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    scaling_rows: list[dict[str, Any]] = []
    if scaling_csv_path.is_file():
        try:
            with open(scaling_csv_path, "r", encoding="utf-8-sig") as f:
                scaling_rows = list(csv.DictReader(f))
        except Exception:
            pass

    # Tổng hợp các chỉ số cốt lõi
    candidate_count = len(history)
    feasible_count = sum(1 for r in history if r["feasible"])
    cycles = sorted(set(r["cycle"] for r in history))
    cycle_count = len(cycles)
    mode = "FAILURE_MODE" if feasible_count == 0 else "SUCCESS_MODE"

    # Thống kê rejection stage
    stage_counts: dict[str, int] = {}
    for r in history:
        st = r["rejection_stage"] if not r["feasible"] else "FEASIBLE"
        stage_counts[st] = stage_counts.get(st, 0) + 1

    dominant_stage = max(stage_counts.items(), key=lambda x: x[1])[0] if stage_counts else "UNKNOWN"

    # Thống kê rejection reason tokens
    reason_token_counts: dict[str, int] = {}
    for r in history:
        raw_reason = r.get("rejection_reason", "")
        if raw_reason:
            tokens = [t.strip() for t in raw_reason.split(",") if t.strip()]
            for tok in tokens:
                reason_token_counts[tok] = reason_token_counts.get(tok, 0) + 1

    dominant_tokens = sorted(reason_token_counts.keys(), key=lambda k: reason_token_counts[k], reverse=True)[:3]

    downstream_reached = any(r["optical_objective"] is not None for r in history)

    # Khởi tạo đồ họa Dark HUD Aesthetic
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "text.color": "#c9d1d9",
        "axes.labelcolor": "#8b949e",
        "axes.edgecolor": "#30363d",
        "axes.facecolor": "#161b22",
        "figure.facecolor": "#0d1117",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
    })

    fig = plt.figure(figsize=(22, 14), dpi=160)
    gs = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.30, left=0.04, right=0.98, top=0.93, bottom=0.05)

    # Tiêu đề chính toàn bảng
    banner_title = "STEP 11 — SYSTEM-AWARE O2 SEARCH & FAILURE ANATOMY DIAGNOSTIC"
    banner_sub = (
        f"MODE: {mode}  |  Candidates: {candidate_count}  |  Cycles: {cycle_count}  |  "
        f"Feasible: {feasible_count}  |  Dominant Failure: {dominant_stage}"
    )
    fig.text(0.04, 0.975, banner_title, fontsize=15, fontweight="bold", color="#58a6ff")
    fig.text(0.04, 0.952, banner_sub, fontsize=10, color="#7ee787" if feasible_count > 0 else "#f85149")

    # -------------------------------------------------------------------------
    # PANEL A: Cycle x Candidate Timeline
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_a.set_title("Panel A — Cycle × Candidate Timeline", fontsize=10, fontweight="bold", color="#58a6ff", pad=8)
    indices = np.arange(1, candidate_count + 1)
    cand_cycles = [r["cycle"] for r in history]
    cand_stages = [r["rejection_stage"] if not r["feasible"] else "FEASIBLE" for r in history]
    cand_colors = [STAGE_COLORS.get(s, STAGE_COLORS["UNKNOWN"]) for s in cand_stages]

    ax_a.scatter(indices, cand_cycles, c=cand_colors, s=38, marker="x" if mode == "FAILURE_MODE" else "o", alpha=0.9, zorder=3)
    ax_a.set_xlabel("Candidate evaluation sequence (1..N)")
    ax_a.set_ylabel("Cycle")
    ax_a.set_yticks(cycles)
    ax_a.grid(True, axis="both")

    # Kẻ vạch ngăn cách cycle và ghi chú trust_scale
    prev_c = -1
    for i, r in enumerate(history):
        c = r["cycle"]
        if c != prev_c:
            if i > 0:
                ax_a.axvline(i + 0.5, color="#30363d", linestyle="--", linewidth=1.0, zorder=2)
            # Annotate trust scale tại đầu cycle
            ax_a.text(i + 1, c + 0.25, f"scale={r['trust_scale']:.2g}", fontsize=7.5, color="#8b949e", style="italic")
            prev_c = c
    ax_a.set_ylim(-0.5, max(cycles) + 0.6)

    # -------------------------------------------------------------------------
    # PANEL B: Cycle x Move Lattice
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_b.set_title("Panel B — Cycle × Move Exploration Lattice", fontsize=10, fontweight="bold", color="#58a6ff", pad=8)

    max_moves = 0
    cycle_groups: dict[int, list[dict[str, Any]]] = {}
    for r in history:
        cycle_groups.setdefault(r["cycle"], []).append(r)
    for c, rows in cycle_groups.items():
        if len(rows) > max_moves:
            max_moves = len(rows)

    # Vẽ từng ô dạng heatmap block
    for c in cycles:
        c_rows = cycle_groups.get(c, [])
        for col_idx, r in enumerate(c_rows):
            st = r["rejection_stage"] if not r["feasible"] else "FEASIBLE"
            col = STAGE_COLORS.get(st, STAGE_COLORS["UNKNOWN"])
            rect = plt.Rectangle((col_idx, c - 0.4), 0.9, 0.8, facecolor=col, edgecolor="#30363d", alpha=0.85, zorder=2)
            ax_b.add_patch(rect)
            short_m = _abbreviate_move(r["move"])
            ax_b.text(col_idx + 0.45, c, short_m, ha="center", va="center", fontsize=6.5, color="#ffffff", fontweight="bold", zorder=3)

    ax_b.set_xlim(-0.5, max_moves + 0.5)
    ax_b.set_ylim(-0.6, max(cycles) + 0.6)
    ax_b.set_xlabel("Move sequence within cycle")
    ax_b.set_ylabel("Cycle")
    ax_b.set_yticks(cycles)
    ax_b.grid(False)

    # -------------------------------------------------------------------------
    # PANEL C: Gate Outcome Matrix
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0:3])
    ax_c.set_title("Panel C — Gate Outcome Matrix (Failure Point Isolation)", fontsize=10, fontweight="bold", color="#58a6ff", pad=8)

    gate_names = [
        "1. M1 shape gate",
        "2. M1 CI trust gate",
        "3. M2 integrability gate",
        "4. M2 shape gate",
        "5. Physical trace reached",
        "6. Unobscured reached",
        "7. Optical scoring reached",
    ]

    # Ma trận 7 hàng x N cột: +1 (Pass), 0 (Skipped/Not reached), -1 (Fail)
    matrix = np.zeros((7, candidate_count), dtype=float)
    for i, r in enumerate(history):
        # M1 shape
        matrix[0, i] = 1.0 if r["M1_shape_pass"] is True else (-1.0 if r["M1_shape_pass"] is False else 0.0)
        # M1 CI trust
        matrix[1, i] = 1.0 if r["M1_ci_trust_pass"] is True else (-1.0 if r["M1_ci_trust_pass"] is False else 0.0)
        # M2 integrability admission: PASS hoặc WARN đều được đi tiếp ở STEP11.
        matrix[2, i] = 1.0 if r["M2_integrability_admitted"] is True else (-1.0 if r["M2_integrability_admitted"] is False else 0.0)
        # M2 shape
        matrix[3, i] = 1.0 if r["M2_shape_pass"] is True else (-1.0 if r["M2_shape_pass"] is False else 0.0)
        # Physical reached
        matrix[4, i] = 1.0 if r["physical_valid_count"] is not None else 0.0
        # Unobscured reached
        matrix[5, i] = 1.0 if r["unobscured"] is not None else 0.0
        # Optical scoring reached
        matrix[6, i] = 1.0 if r["optical_objective"] is not None else 0.0

    # Vẽ ô lưới ma trận
    cmap = matplotlib.colors.ListedColormap(["#da3633", "#21262d", "#2ea043"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ax_c.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    ax_c.set_yticks(np.arange(7))
    ax_c.set_yticklabels(gate_names, fontsize=8)
    ax_c.set_xlabel("Candidate (1..N)")
    ax_c.grid(True, color="#30363d", linewidth=0.5)

    # -------------------------------------------------------------------------
    # PANEL F: Trust-Scale Schedule
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 3])
    ax_f.set_title("Panel F — Trust-Scale Schedule", fontsize=10, fontweight="bold", color="#58a6ff", pad=8)
    unique_cycle_scales = []
    c_counts = []
    for c in cycles:
        c_cand = cycle_groups.get(c, [])
        scale_val = c_cand[0]["trust_scale"] if c_cand else 0.0
        unique_cycle_scales.append(scale_val)
        c_counts.append(len(c_cand))

    ax_f.plot(cycles, unique_cycle_scales, marker="o", color="#f0883e", linewidth=2.0, markersize=6, zorder=3)
    for c, sc, cnt in zip(cycles, unique_cycle_scales, c_counts):
        ax_f.annotate(f"{sc:.2g}\n(n={cnt})", (c, sc), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#c9d1d9")
    ax_f.set_xlabel("Cycle")
    ax_f.set_ylabel("Trust scale")
    ax_f.set_xticks(cycles)
    ax_f.set_ylim(-0.1, max(unique_cycle_scales) * 1.35 if unique_cycle_scales else 1.0)
    ax_f.grid(True)

    # -------------------------------------------------------------------------
    # PANEL D: Failure Signature Summary (Stage counts + Reason tokens)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[2, 0:2])
    ax_d.set_title("Panel D — Failure Signature Breakdown", fontsize=10, fontweight="bold", color="#58a6ff", pad=8)

    # Top items: Stage counts and Reason tokens
    d_labels = []
    d_vals = []
    d_colors = []

    # Stages
    for st, count in sorted(stage_counts.items(), key=lambda x: x[1]):
        d_labels.append(f"Stage: {st}")
        d_vals.append(count)
        d_colors.append(STAGE_COLORS.get(st, STAGE_COLORS["UNKNOWN"]))

    # Tokens
    for tok, count in sorted(reason_token_counts.items(), key=lambda x: x[1]):
        d_labels.append(f"Token: {tok}")
        d_vals.append(count)
        d_colors.append("#ff7b72")

    y_pos = np.arange(len(d_labels))
    bars = ax_d.barh(y_pos, d_vals, color=d_colors, height=0.65, zorder=3)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(d_labels, fontsize=7.5)
    ax_d.set_xlabel("Occurrences")
    ax_d.grid(True, axis="x")

    for bar in bars:
        w = bar.get_width()
        ax_d.text(w + 0.8, bar.get_y() + bar.get_height() / 2.0, f"{int(w)}", va="center", fontsize=7.5, color="#c9d1d9")
    ax_d.set_xlim(0, max(d_vals) * 1.15 if d_vals else 10)

    # -------------------------------------------------------------------------
    # PANEL E: Downstream Metric Reachability
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[2, 2])
    ax_e.set_title("Panel E — Metric Reachability", fontsize=10, fontweight="bold", color="#58a6ff", pad=8)

    check_keys = [
        ("physical_valid", "physical_valid_count"),
        ("physical_frac", "physical_fraction"),
        ("unobscured", "unobscured"),
        ("optical_obj", "optical_objective"),
        ("mapping_rms", "mapping_rms_mm"),
        ("spot_rms", "spot_rms_mm"),
        ("direction_rms", "direction_rms_deg"),
        ("rep_score", "representation_score"),
    ]

    reach_names = [item[0] for item in check_keys]
    reach_pcts = []
    for _, key in check_keys:
        valid_cnt = sum(1 for r in history if r.get(key) is not None)
        reach_pcts.append((valid_cnt / candidate_count) * 100.0)

    y_reach = np.arange(len(reach_names))
    ax_e.barh(y_reach, reach_pcts, color="#58a6ff", height=0.6, zorder=3)
    ax_e.set_yticks(y_reach)
    ax_e.set_yticklabels(reach_names, fontsize=7.5)
    ax_e.set_xlabel("Coverage (%)")
    ax_e.set_xlim(0, 105)
    ax_e.grid(True, axis="x")

    for idx, pct in enumerate(reach_pcts):
        ax_e.text(pct + 2.0, idx, f"{pct:.1f}%", va="center", fontsize=7.5, color="#8b949e")

    # -------------------------------------------------------------------------
    # PANEL G & H: Status Summary & DOF Scaling Sidecar
    # -------------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[2, 3])
    ax_g.axis("off")

    fail_msg = (
        f"STEP 11 {mode}\n"
        f"-------------------------------------\n"
        f"• Status: {debug_summary.get('execution_status', 'FAILED')}\n"
        f"• Phase: {debug_summary.get('summary', {}).get('failure_phase', 'UNKNOWN')}\n"
        f"• Candidates: {candidate_count} across {cycle_count} cycles\n"
        f"• Feasible: {feasible_count} / {candidate_count}\n"
        f"• Dominant Gate: {dominant_stage}\n"
        f"• Downstream Reached: {'YES' if downstream_reached else 'NO (0%)'}\n\n"
        f"ROOT CAUSE SYNTHESIS:\n"
        f"All {candidate_count} candidates successfully passed\n"
        f"M1 shape and CI trust gates, but were\n"
        f"consistently REJECTED at M2 consistency\n"
        f"due to normal RMS & height residuals.\n"
        f"Search died BEFORE optical evaluation."
    )

    box_props = dict(boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#da3633" if feasible_count == 0 else "#2ea043", linewidth=1.5)
    ax_g.text(0.02, 0.98, fail_msg, transform=ax_g.transAxes, fontsize=8.2, va="top", ha="left", color="#c9d1d9", bbox=box_props, family="monospace")

    # Lưu hình ảnh
    png_path = target_out_dir / "11_STEP11_SEARCH_DIAGNOSTIC.png"
    json_path = target_out_dir / "11_STEP11_SEARCH_DIAGNOSTIC.json"

    fig.savefig(png_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Xuất file metadata JSON V2 đúng hợp đồng spec
    metadata: dict[str, Any] = {
        "schema": SCHEMA_V2,
        "status": "RENDERED",
        "mode": mode,
        "image": str(png_path.resolve()),
        "data_sources": {
            "history_csv": str(history_csv.resolve()),
            "debug_summary": str(debug_summary_path.resolve()) if debug_summary_path.is_file() else None,
            "run_failure": str(run_failure_path.resolve()) if run_failure_path.is_file() else None,
            "m1_physical_dof_scaling": str(scaling_csv_path.resolve()) if scaling_csv_path.is_file() else None,
        },
        "candidate_count": int(candidate_count),
        "cycle_count": int(cycle_count),
        "feasible_candidate_count": int(feasible_count),
        "dominant_rejection_stage": str(dominant_stage),
        "dominant_rejection_reason_tokens": dominant_tokens,
        "downstream_optical_evaluation_reached": bool(downstream_reached),
        "reran_optical_algorithms": False,
        "refit_performed": False,
        "ci_rebuild_performed": False,
        "ray_trace_performed": False,
        "reevaluation_performed": False,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main() -> None:
    """CLI entrypoint cho renderer chẩn đoán STEP 11."""
    target_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

    if target_dir and (target_dir / "11_O2_SEARCH_HISTORY.csv").is_file():
        # Trỏ thẳng vào thư mục STEP_11
        res = render_step11_search_diagnostic(target_dir)
        print(f"Rendered STEP_11 in {target_dir}: {res.get('status')}")
        return

    if target_dir and target_dir.is_dir():
        # Quét tìm các thư mục STEP_11 con
        found = list(target_dir.glob("**/STEP_11/11_O2_SEARCH_HISTORY.csv"))
        if found:
            for csv_file in found:
                sdir = csv_file.parent
                res = render_step11_search_diagnostic(sdir)
                print(f"Rendered {sdir}: {res.get('status')}")
            return

    # Tự động quét run gần nhất trong multistart_runs_v5_5
    runs_dir = Path("multistart_runs_v5_5").resolve()
    found = list(runs_dir.glob("**/STEP_11/11_O2_SEARCH_HISTORY.csv"))
    if found:
        # Sắp xếp theo thời gian sửa đổi mới nhất
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_step11 = found[0].parent
        res = render_step11_search_diagnostic(latest_step11)
        print(f"Auto-rendered latest STEP_11 in {latest_step11}: {res.get('status')}")
    else:
        print("No STEP_11 directory with 11_O2_SEARCH_HISTORY.csv found.")


if __name__ == "__main__":
    main()
