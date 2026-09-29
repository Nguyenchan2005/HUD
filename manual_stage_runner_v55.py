"""Chạy thủ công một STEP, nối checkpoint và lưu ảnh kiểm tra tương ứng."""

from __future__ import annotations

import gzip
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

from core import write_json
from pipeline_v55 import (Context, IMPORTANT, MAX_STEP, STAGES, _algorithm_source_manifest,
                           _archive_step, _call_stage_algorithm, _record_step_failure_safely,
                           validate_config)
from visualization_v55 import render_spot_evolution, render_step, render_surface_ray_view


ROOT = Path(__file__).resolve().parent
MANUAL_ROOT = ROOT / "manual_flow_v5_5"
MANUAL_RUNS = ROOT / "manual_runs_v5_5"
ACTIVE = MANUAL_ROOT / "ACTIVE_RUN.json"


def _save(ctx: Context, step: int) -> Path:
    """Ghi checkpoint nén theo cách thay thế nguyên tử."""
    folder = ctx.run_dir / "CHECKPOINTS" / f"STEP_{step:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "state.pkl.gz"; temporary = folder / "state.pkl.gz.tmp"
    compresslevel = int(
        ctx.config.get(
            "execution", {}
        ).get(
            "checkpoint_compresslevel",
            1,
        )
    )
    with gzip.open(
        temporary,
        "wb",
        compresslevel=compresslevel,
    ) as stream:
        pickle.dump(ctx, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    write_json(folder / "checkpoint.json", {
        "schema": "HUD_FAN_V5_5_MANUAL_CHECKPOINT",
        "completed_step": step,
        "checkpoint": str(target),
        "gzip_compresslevel": compresslevel,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_source_manifest_sha256": ctx.data.get(
            "algorithm_source_manifest", {}).get("manifest_sha256")})
    return target


def _new() -> Context:
    """Tạo run thủ công mới và khóa config cùng input tại STEP 00."""
    config_path = ROOT / "config_v55.json"
    c = json.loads(config_path.read_text(encoding="utf-8")); validate_config(c)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = MANUAL_RUNS / f"HUD_FAN_V5_5_MANUAL_{stamp}"; run.mkdir(parents=True)
    ctx = Context(ROOT, config_path.resolve(), c, run)
    ctx.data["_frozen_config_value"] = c
    return ctx


def _resume(step: int) -> Context:
    """Nạp checkpoint trước và xác nhận config không đổi giữa run."""
    if not ACTIVE.exists(): raise RuntimeError("NO_ACTIVE_V5_5_RUN: run STEP_00 first")
    active = json.loads(ACTIVE.read_text(encoding="utf-8"))
    if int(active["next_step"]) != step:
        raise RuntimeError(f"MANUAL_STEP_ORDER_ERROR: expected STEP_{int(active['next_step']):02d}")
    checkpoint = Path(active["checkpoint"])
    with gzip.open(checkpoint, "rb") as stream: ctx = pickle.load(stream)
    current = json.loads(ctx.config_path.read_text(encoding="utf-8"))
    if current != ctx.data["_frozen_config_value"]:
        raise RuntimeError("CONFIG_CHANGED_AFTER_STEP_00: restart the manual run at STEP_00")
    frozen_manifest = ctx.data.get("algorithm_source_manifest", {}).get("manifest_sha256")
    current_manifest = _algorithm_source_manifest(ctx).get("manifest_sha256")
    if frozen_manifest is None or frozen_manifest != current_manifest:
        raise RuntimeError("ALGORITHM_SOURCE_CHANGED_AFTER_STEP_00: restart the manual run at STEP_00")
    return ctx


def run_step(step: int) -> Path:
    """Chạy đúng một STEP rồi lưu checkpoint, archive và ảnh 3D."""
    if step not in STAGES: raise ValueError(f"step must be 0..{MAX_STEP}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    ctx = _new() if step == 0 else _resume(step)
    from execution_v55 import execution_session
    with execution_session(ctx.config, ctx.run_dir):
        title, fn = STAGES[step]; print(f"[{step:02d}/{MAX_STEP:02d}] {title}", flush=True)
        try:
            summary = _call_stage_algorithm(ctx, step, fn)
            ctx.stage_summaries[step] = summary
            try:
                summary["visualization_3d"] = render_step(ctx, step)
            except Exception as exc:
                summary["visualization_3d"] = {"status": "VISUALIZATION_WARNING", "message": str(exc)}
                print(f"VISUALIZATION_3D_WARNING STEP_{step:02d}: {exc}", flush=True)
            try:
                summary["surface_ray_footprint"] = render_surface_ray_view(ctx, step)
            except Exception as exc:
                # Hình phụ không được phép làm thay đổi PASS/FAIL của thuật toán quang học.
                summary["surface_ray_footprint"] = {"status": "VISUALIZATION_WARNING", "message": str(exc)}
                print(f"SURFACE_RAY_IMAGE_WARNING STEP_{step:02d}: {exc}", flush=True)
            if step == 20:
                try:
                    # Ảnh so sánh chỉ đọc các snapshot đã khóa; lỗi vẽ không làm đổi kết quả quang học.
                    summary["spot_evolution"] = render_spot_evolution(ctx)
                except Exception as exc:
                    summary["spot_evolution"] = {"status": "VISUALIZATION_WARNING", "message": str(exc)}
                    print(f"SPOT_EVOLUTION_IMAGE_WARNING STEP_{step:02d}: {exc}", flush=True)
            _archive_step(ctx, step, title, fn, summary)
            ctx.data["_execution_phase"] = "CHECKPOINT"
            checkpoint = _save(ctx, step)
            write_json(ACTIVE, {"schema": "HUD_FAN_V5_5_MANUAL_ACTIVE_RUN",
                       "status": "COMPLETE" if step == MAX_STEP else "READY", "run_dir": str(ctx.run_dir),
                       "completed_step": step, "next_step": None if step == MAX_STEP else step + 1,
                       "checkpoint": str(checkpoint), "updated_utc": datetime.now(timezone.utc).isoformat(),
                       "v5_2_checkpoint_reuse": False})
            print(f"STEP_COMPLETE {checkpoint}", flush=True); return checkpoint
        except Exception as exc:
            _record_step_failure_safely(ctx, step, exc)
            print(f"STEP_FAILED {exc}", file=sys.stderr, flush=True); raise


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("Usage: py -3.10 manual_stage_runner_v55.py STEP_NUMBER")
    run_step(int(sys.argv[1]))
