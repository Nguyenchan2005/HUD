"""Precheck mọi planar geometry ở STEP 07, chỉ chạy sâu geometry rank #1 tới STEP 26."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import pickle
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import write_json
from pipeline_v55 import (
    Context, STAGES, _algorithm_source_manifest, _archive_step,
    _call_stage_algorithm, _record_step_failure_safely,
    _planar_seed_rank_key,
    activate_planar_seed, enumerate_planar_seeds,
    planar_seed_candidate_rows, validate_config,
)
from execution_v55 import (
    execution_session,
    BackendExecutionError,
    BackendConsistencyError,
)


ROOT = Path(__file__).resolve().parent
MULTISTART_ROOT = ROOT / "multistart_runs_v5_5"
SELECTION_RULE = (
    "STEP07_EVALUATE_ALL_ELIGIBLE_GEOMETRIES;"
    "DEEP_RUN_ONLY_STEP07_GEOMETRY_RANK_1;"
    "OPTIONAL_POST_STEP22_REFINEMENT_APPLIES_ONLY_TO_THE_SELECTED_DESIGN"
)


def _save_checkpoint(ctx: Context, step: int) -> Path:
    """Lưu Context sau mỗi STEP để bảo toàn bằng chứng và hỗ trợ điều tra lỗi."""
    folder = ctx.run_dir / "CHECKPOINTS" / f"STEP_{step:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "state.pkl.gz"
    temporary = folder / "state.pkl.gz.tmp"
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
    meta = ctx.data.get("sampling_profile_metadata")
    chk_record = {
        "schema": "HUD_FAN_V5_5_MULTISTART_CHECKPOINT",
        "completed_step": step,
        "checkpoint": str(target),
        "gzip_compresslevel": compresslevel,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_source_manifest_sha256": ctx.data.get(
            "algorithm_source_manifest", {}).get("manifest_sha256"),
    }
    if meta:
        chk_record["sampling_profile"] = meta.get("profile_name")
        chk_record["sampling_profile_fingerprint"] = meta.get("fingerprint")
    write_json(folder / "checkpoint.json", chk_record)
    return target


def _load_checkpoint(path: Path, expected_step: int | None = None) -> Context:
    """Nạp Context đã nén của phần common khi tiếp tục một multi-start run."""
    with gzip.open(path, "rb") as stream:
        ctx = pickle.load(stream)
    if expected_step is not None:
        from pipeline_v55 import verify_sampling_profile_on_resume
        verify_sampling_profile_on_resume(ctx, expected_step)
    return ctx


def _deferred_render_steps(ctx: Context) -> set[int]:
    """Tập STEP cần giữ state để render hậu kỳ; multistart không render inline."""
    raw = ctx.config.get(
        "execution", {}
    ).get(
        "render_steps", []
    )

    if not raw:
        return set(range(27))

    steps = {
        int(value)
        for value in raw
    }

    invalid = sorted(
        step
        for step in steps
        if step < 0 or step > 26
    )

    if invalid:
        raise ValueError(
            f"DEFERRED_RENDER_STEP_OUT_OF_RANGE:{invalid}"
        )

    return steps


def _execution_checkpoint_steps(
    ctx: Context,
) -> set[int]:
    """
    Checkpoint bắt buộc bao phủ:
    - checkpoint production đã cấu hình,
    - STEP 06 cho common multistart,
    - STEP 22 cho geometry refinement,
    - mọi STEP cần render hậu kỳ.
    """
    raw = ctx.config.get(
        "execution", {}
    ).get(
        "checkpoint_steps", []
    )

    if not raw:
        return set(range(27))

    steps = {
        int(value)
        for value in raw
    }

    steps.add(6)
    steps.add(22)

    # Không được để một deferred render STEP thiếu state.
    steps.update(_deferred_render_steps(ctx))

    return steps


def _save_render_state(
    ctx: Context,
    step: int,
    *,
    execution_status: str,
    exception: Exception | None = None,
) -> Path:
    """Lưu state chỉ phục vụ hậu kỳ visualization mà không thay đổi optical result."""
    folder = (
        ctx.run_dir
        / "RENDER_STATES"
        / f"STEP_{step:02d}"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        folder
        / "state.pkl.gz"
    )

    temporary = (
        folder
        / "state.pkl.gz.tmp"
    )

    compresslevel = int(
        ctx.config.get(
            "execution",
            {},
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
        pickle.dump(
            ctx,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary.replace(
        target
    )

    metadata = {
        "schema":
            "HUD_FAN_V5_5_RENDER_STATE_V1",

        "step":
            int(step),

        "execution_status":
            str(execution_status),

        "state_kind":
            (
                "COMPLETED_STEP_STATE"
                if execution_status
                == "ALGORITHM_COMPLETED"
                else "PARTIAL_FAILURE_STATE"
            ),

        "algorithm_source_manifest_sha256":
            ctx.data.get(
                "algorithm_source_manifest",
                {},
            ).get(
                "manifest_sha256"
            ),

        "exception_type":
            (
                None
                if exception is None
                else type(exception).__name__
            ),

        "exception_message":
            (
                None
                if exception is None
                else str(exception)
            ),
    }

    write_json(
        folder / "metadata.json",
        metadata,
    )

    return target


def _execute_stage(
    ctx: Context,
    number: int,
) -> dict[str, Any]:
    """Chạy một STEP bằng đúng thuật toán, lưu render-state, archive dữ liệu và lưu checkpoint."""
    title, fn = STAGES[number]

    print(
        f"[{number:02d}/26] {title}",
        flush=True,
    )

    checkpoint_steps = (
        _execution_checkpoint_steps(ctx)
    )

    try:
        summary = _call_stage_algorithm(
            ctx,
            number,
            fn,
        )

        ctx.stage_summaries[number] = summary

        # Lưu render-state cho mọi STEP thành công
        _save_render_state(
            ctx,
            number,
            execution_status="ALGORITHM_COMPLETED",
        )

        deferred = {
            "status": "DEFERRED",
            "renderer":
                "render_multistart_saved_v55.py",
            "source":
                "RENDER_STATE",
        }

        summary["visualization_3d"] = dict(
            deferred
        )

        summary[
            "surface_ray_footprint"
        ] = dict(
            deferred
        )

        if number == 20:
            summary[
                "spot_evolution"
            ] = dict(
                deferred
            )

        _archive_step(
            ctx,
            number,
            title,
            fn,
            summary,
        )

        if number in checkpoint_steps:
            ctx.data[
                "_execution_phase"
            ] = "CHECKPOINT"

            _save_checkpoint(
                ctx,
                number,
            )

        return summary

    except Exception as exc:
        try:
            _save_render_state(
                ctx,
                number,
                execution_status="ALGORITHM_FAILED",
                exception=exc,
            )
        except Exception as state_exc:
            print(
                "RENDER_STATE_SAVE_WARNING "
                f"STEP_{number:02d}: "
                f"{type(state_exc).__name__}: "
                f"{state_exc}",
                flush=True,
            )

        _record_step_failure_safely(
            ctx,
            number,
            exc,
        )
        raise


def _execute_selected_step_07(ctx: Context, seed: dict[str, Any], candidate_index: int,
                              candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Khóa đúng seed của nhánh hiện tại thay vì tự động lấy seed eligible đầu tiên."""
    print(f"[07/26] Activate eligible planar seed {candidate_index:02d}", flush=True)
    checkpoint_steps = (
        _execution_checkpoint_steps(ctx)
    )
    try:
        summary = _call_stage_algorithm(
            ctx,
            7,
            activate_planar_seed,
            copy.deepcopy(seed),
            candidate_index,
            candidate_rows,
            SELECTION_RULE,
        )
        ctx.stage_summaries[7] = summary

        # Lưu render-state cho STEP 07 thành công
        _save_render_state(
            ctx,
            7,
            execution_status="ALGORITHM_COMPLETED",
        )

        deferred = {
            "status": "DEFERRED",
            "renderer":
                "render_multistart_saved_v55.py",
            "source":
                "RENDER_STATE",
        }

        summary[
            "visualization_3d"
        ] = dict(
            deferred
        )

        summary[
            "surface_ray_footprint"
        ] = dict(
            deferred
        )

        _archive_step(ctx, 7, "Activate one independently evaluated planar start",
                      activate_planar_seed, summary)
        if 7 in checkpoint_steps:
            ctx.data["_execution_phase"] = "CHECKPOINT"
            _save_checkpoint(ctx, 7)
        return summary
    except Exception as exc:
        try:
            _save_render_state(
                ctx,
                7,
                execution_status="ALGORITHM_FAILED",
                exception=exc,
            )
        except Exception as state_exc:
            print(
                "RENDER_STATE_SAVE_WARNING "
                f"STEP_07: "
                f"{type(state_exc).__name__}: "
                f"{state_exc}",
                flush=True,
            )

        _record_step_failure_safely(ctx, 7, exc)
        raise


def _finite(value: Any) -> float | None:
    """Chuẩn hóa một scalar hữu hạn để ghi bảng xếp hạng."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _completed_result(ctx: Context, candidate_index: int,
                      variant: str = "BASELINE") -> dict[str, Any]:
    """Rút các hard gate và metric so sánh từ Context đã qua STEP 25/26."""
    numerical = ctx.data["numerical"]
    scorecard = ctx.data["scorecard"]
    failures = [row["item"] for row in scorecard if row["status"] == "FAIL"]
    passes = [row["item"] for row in scorecard if row["status"] == "PASS"]
    rays = int(numerical["ray_count"])
    required_bundles = int(numerical["forward_required_bundle_count"])
    mtf = numerical.get("MTF", {}).get("minimum_MTF_at_max_frequency")
    return {
        "candidate": candidate_index,
        "variant": variant,
        "design_id": f"SEED_{candidate_index:03d}_{variant}",
        "execution_status": "COMPLETED",
        "completed_step": 26,
        "run_dir": str(ctx.run_dir),
        "algorithm_source_manifest_sha256": ctx.data.get(
            "algorithm_source_manifest", {}).get("manifest_sha256"),
        "final_status": ctx.data["final_status"],
        "hard_gate_pass": not failures,
        "hard_gate_fail_count": len(failures),
        "hard_gate_pass_count": len(passes),
        "failed_hard_gates": failures,
        "forward_converged": int(numerical["forward_converged"]),
        "ray_count": rays,
        "forward_converged_fraction": float(numerical["forward_converged"]) / max(rays, 1),
        "forward_valid_bundle_count": int(numerical["forward_valid_bundle_count"]),
        "forward_required_bundle_count": required_bundles,
        "forward_valid_bundle_fraction": float(numerical["forward_valid_bundle_count"]) / max(required_bundles, 1),
        "distortion_evaluation_status": numerical["distortion"]["evaluation_status"],
        "distortion_D_max_percent": _finite(numerical["distortion"]["D_max_percent"]),
        "packaging_constraint_enabled": bool(numerical["packaging_constraint_enabled"]),
        "packaging_lambda": _finite(numerical["packaging_lambda"]),
        "visor_footprint_inside": bool(numerical["visor_footprint_inside_66x42"]),
        "fermat_all_converged": bool(numerical["fermat_all_converged"]),
        "MF1_Fan": _finite(ctx.data["fan_mf1"]["MF1_Fan"]),
        "E1_mm": _finite(ctx.data["fan_mf1"]["E1_mm"]),
        "MTF_at_max_frequency": _finite(mtf),
    }


def _ranking_key(row: dict[str, Any], config: dict[str, Any]) -> tuple[Any, ...]:
    """Khóa lexicographic: hard gates luôn đứng trước các metric chất lượng mềm."""
    distortion = row.get("distortion_D_max_percent")
    packaging = row.get("packaging_lambda")
    mtf = row.get("MTF_at_max_frequency")
    packaging_limit = float(config["packaging_lambda_max"])
    packaging_ratio = (float(packaging) / packaging_limit
                       if packaging is not None and row.get("packaging_constraint_enabled") else 0.0)
    return (
        0 if row.get("execution_status") == "COMPLETED" else 1,
        0 if row.get("hard_gate_pass") else 1,
        int(row.get("hard_gate_fail_count", 10**6)),
        -float(row.get("forward_converged_fraction", 0.0)),
        -float(row.get("forward_valid_bundle_fraction", 0.0)),
        0 if distortion is not None else 1,
        float(distortion) if distortion is not None else 10**12,
        packaging_ratio,
        float(row.get("MF1_Fan")) if row.get("MF1_Fan") is not None else 10**12,
        -float(mtf) if mtf is not None else 0.0,
        int(row["candidate"]),
        str(row.get("variant", "BASELINE")),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Ghi bảng phẳng UTF-8, chuyển list hard gate thành chuỗi dễ đọc."""
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ";".join(map(str, value)) if isinstance(value, list) else value
                             for key, value in row.items()})


def _publish_best(root: Path, best: dict[str, Any], ranking: list[dict[str, Any]]) -> None:
    """Công bố seed thắng và sao chép bộ prescription/báo cáo gọn để sử dụng ngay."""
    source = Path(best["run_dir"])
    output = root / "BEST_RESULT"
    output.mkdir(parents=True, exist_ok=True)
    names = [
        "00_ALGORITHM_SOURCE_MANIFEST.json", "FINAL_RUN_STATUS.txt", "FINAL_PRESCRIPTION.json", "FINAL_SURFACE_M1.json",
        "FINAL_SURFACE_M1.csv", "FINAL_SURFACE_M2.json", "FINAL_SURFACE_M2.csv",
        "12_FIT_QUALITY_GATES.csv", "16_SELECTED_CI_CLOUD_INTEGRABILITY.json",
        "12_M1_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
        "12_M2_CLOUD_INTEGRABILITY_IN_FITTED_FRAME.json",
        "17_SELECTED_CI_CLOUD_INTEGRABILITY.json", "17_ORDER_POLICY.json",
        "17_ORDER_HISTORY.csv", "23_LOCAL_OPTIMIZATION_HISTORY.csv",
        "23_DLSQ_TRIALS.csv", "23_DLSQ_TERMINATION.json",
        "24N_NUMERICAL_ANALYSIS_SUMMARY.json", "24N_MTF_SUMMARY.json",
        "25_FINAL_OPTICAL_REPORT.json", "25_FINAL_SCORECARD.csv", "26_EXPORT_COMPLETENESS.json",
    ]
    copied = []
    for name in names:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
            copied.append(name)
    write_json(output / "BEST_SEED.json", {
        "schema": "HUD_FAN_V5_5_MULTISTART_BEST_SEED",
        "selection_rule": SELECTION_RULE,
        "best": best,
        "source_run_dir": str(source),
        "algorithm_source_manifest_sha256": best.get("algorithm_source_manifest_sha256"),
        "copied_files": copied,
        "ranking_design_order": [row.get("design_id", f"SEED_{row['candidate']:03d}")
                                  for row in ranking],
    })


def _status(root: Path, **values: Any) -> None:
    """Cập nhật trạng thái tổng để theo dõi một run dài mà không đọc log."""
    write_json(root / "MULTISTART_STATUS.json", {
        "schema": "HUD_FAN_V5_5_MULTISTART_STATUS",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    })


def _normalize_result_identity(row: dict[str, Any], variant: str = "BASELINE") -> dict[str, Any]:
    """Bổ sung định danh cho result cũ để resume tương thích với run tạo trước nhánh 23G."""
    row.setdefault("variant", variant)
    row.setdefault("design_id", f"SEED_{int(row['candidate']):03d}_{row['variant']}")
    return row


def _select_step7_deep_run_schedule(
    candidates: list[dict[str, Any]],
) -> tuple[
    list[tuple[int, dict[str, Any]]],
    list[tuple[int, dict[str, Any]]],
]:
    """Schedule STEP07 geometries in rank order until one deep run completes."""
    eligible = sorted(
        [
            (index, seed)
            for index, seed in enumerate(
                candidates,
                start=1,
            )
            if seed["eligible"]
        ],
        key=lambda item: (
            _planar_seed_rank_key(item[1]),
            item[0],
        ),
    )

    scheduled = eligible

    return eligible, scheduled


def _run_geometry_top_k(root: Path, config: dict[str, Any],
                        baseline_ranking: list[dict[str, Any]], resume: bool) -> list[dict[str, Any]]:
    """Chạy 23A→23G→23C→26 từ checkpoint STEP 22, chỉ cho top-K seed baseline."""
    refinement = config.get("geometry_refinement", {})
    surface_refinement = config.get("surface_parameter_refinement", {})
    if not refinement.get("enabled", False) and not surface_refinement.get("enabled", False):
        return []
    selected = baseline_ranking[:min(int(refinement["top_k"]), len(baseline_ranking))]
    selection_record = {
        "schema": "HUD_FAN_V5_5_TOP_K_GEOMETRY_SELECTION",
        "source": "BASELINE_FULL_FORWARD_STEP_24_AND_HARD_GATE_STEP_25_RANKING",
        "top_k_requested": int(refinement["top_k"]),
        "selected": [{"candidate": row["candidate"], "baseline_rank": row["rank"],
                      "baseline_design_id": row["design_id"]} for row in selected],
        "flow": "STEP_22_CHECKPOINT_TO_23A_TO_OPTIONAL_23S_TO_OPTIONAL_23G_TO_23C_TO_24_TO_25_TO_26",
    }
    write_json(root / "GEOMETRY_REFINEMENT_SELECTION.json", selection_record)
    results: list[dict[str, Any]] = []
    for ordinal, baseline in enumerate(selected, 1):
        candidate_index = int(baseline["candidate"])
        refined_dir = root / "GEOMETRY_REFINEMENT" / f"SEED_{candidate_index:03d}"
        existing_result = refined_dir / "SEED_RESULT.json"
        if resume and existing_result.exists():
            old = _normalize_result_identity(
                json.loads(existing_result.read_text(encoding="utf-8")), "GEOMETRY_REFINED")
            if old.get("execution_status") in ("COMPLETED", "FAILED_DURING_PIPELINE"):
                print(f"=== SKIP TERMINAL GEOMETRY SEED {candidate_index:02d}: "
                      f"{old.get('execution_status')} ===", flush=True)
                results.append(old)
                continue
        checkpoint = root / f"SEED_{candidate_index:03d}" / "CHECKPOINTS" / "STEP_22" / "state.pkl.gz"
        print(f"=== GEOMETRY TOP-K SEED {candidate_index:02d} "
              f"({ordinal}/{len(selected)}) FROM STEP_22 ===", flush=True)
        last_step = 22
        try:
            if not checkpoint.exists():
                raise RuntimeError(f"GEOMETRY_STEP_22_CHECKPOINT_MISSING: {checkpoint}")
            ctx = _load_checkpoint(checkpoint, expected_step=22)
            ctx.run_dir = refined_dir
            ctx.config["geometry_refinement"] = copy.deepcopy(refinement)
            ctx.config["surface_parameter_refinement"] = copy.deepcopy(surface_refinement)
            ctx.data["_geometry_refinement_runtime_enabled"] = True
            ctx.data["multistart_variant"] = "GEOMETRY_REFINED"
            for step in range(23, 27):
                _execute_stage(ctx, step)
                last_step = step
            result = _completed_result(ctx, candidate_index, "GEOMETRY_REFINED")
        except KeyboardInterrupt:
            _status(root, status="INTERRUPTED_GEOMETRY_REFINEMENT",
                    current_candidate=candidate_index)
            raise
        except (BackendExecutionError, BackendConsistencyError):
            err_report = {
                "schema": "HUD_RUN_EXECUTION_FAILURE_V1",
                "candidate": candidate_index,
                "variant": "GEOMETRY_REFINED",
                "completed_step": last_step,
                "failed_step": last_step + 1,
                "traceback": traceback.format_exc(),
            }
            write_json(root / "RUN_EXECUTION_FAILURE.json", err_report)
            raise
        except Exception as exc:
            result = {
                "candidate": candidate_index,
                "variant": "GEOMETRY_REFINED",
                "design_id": f"SEED_{candidate_index:03d}_GEOMETRY_REFINED",
                "execution_status": "FAILED_DURING_PIPELINE",
                "completed_step": last_step,
                "failed_step": last_step + 1,
                "run_dir": str(refined_dir),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            refined_dir.mkdir(parents=True, exist_ok=True)
            write_json(refined_dir / "RUN_FAILURE.json", result | {"traceback": traceback.format_exc()})
            print(f"GEOMETRY_SEED_{candidate_index:03d}_FAILED STEP_{last_step + 1:02d}: {exc}",
                  file=sys.stderr, flush=True)
        write_json(refined_dir / "SEED_RESULT.json", result)
        results.append(result)
        _status(root, status="RUNNING_GEOMETRY_REFINEMENT", current_candidate=candidate_index,
                selected_candidates=[int(row["candidate"]) for row in selected],
                processed_geometry_designs=[row["design_id"] for row in results])
    return results


def _assert_multistart_inline_rendering_disabled(
    config: dict[str, Any],
) -> None:
    """
    Multistart chỉ compute.
    Mọi PNG phải được dựng hậu kỳ.
    """
    execution = config.get(
        "execution",
        {},
    )

    live = config.get(
        "live_monitor",
        {},
    )

    if bool(
        execution.get(
            "multistart_inline_rendering",
            False,
        )
    ):
        raise ValueError(
            "MULTISTART_INLINE_RENDERING_MUST_REMAIN_DISABLED"
        )

    if bool(
        execution.get(
            "render_step23_phase_images",
            False,
        )
    ):
        raise ValueError(
            "MULTISTART_STEP23_PHASE_IMAGES_MUST_REMAIN_DISABLED"
        )

    if bool(
        live.get(
            "images_enabled",
            False,
        )
    ):
        raise ValueError(
            "MULTISTART_LIVE_IMAGES_MUST_REMAIN_DISABLED"
        )


def run_multistart(config_path: Path, root: Path, max_seeds: int | None = None,
                   precheck_only: bool = False, resume: bool = False) -> dict[str, Any]:
    """Chạy common STEP 00–06, precheck mọi STEP 07 geometry, rồi chỉ chạy rank #1 tới STEP 26."""
    requested_config = json.loads(config_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=resume)
    common_checkpoint = root / "COMMON" / "CHECKPOINTS" / "STEP_06" / "state.pkl.gz"
    if resume:
        if not common_checkpoint.exists():
            raise RuntimeError(f"MULTISTART_COMMON_CHECKPOINT_MISSING: {common_checkpoint}")
        base = _load_checkpoint(common_checkpoint, expected_step=6)
        frozen_manifest = base.data.get("algorithm_source_manifest", {}).get("manifest_sha256")
        current_manifest = _algorithm_source_manifest(base)
        if frozen_manifest is None or frozen_manifest != current_manifest["manifest_sha256"]:
            raise RuntimeError(
                "MULTISTART_RESUME_ALGORITHM_SOURCE_CHANGED_START_A_NEW_RUN_DIRECTORY")
        config = copy.deepcopy(base.config)
        refinement_keys = {"geometry_refinement", "_comment_geometry_refinement",
                           "surface_parameter_refinement", "_comment_surface_parameter_refinement"}
        frozen_without_refinement = {k: v for k, v in config.items() if k not in refinement_keys}
        requested_without_refinement = {k: v for k, v in requested_config.items()
                                        if k not in refinement_keys}
        if frozen_without_refinement != requested_without_refinement:
            raise RuntimeError("MULTISTART_CONFIG_CHANGED_OUTSIDE_GEOMETRY_REFINEMENT")
        if "geometry_refinement" in requested_config:
            config["geometry_refinement"] = copy.deepcopy(requested_config["geometry_refinement"])
        if "surface_parameter_refinement" in requested_config:
            config["surface_parameter_refinement"] = copy.deepcopy(
                requested_config["surface_parameter_refinement"])
        base.config = copy.deepcopy(config)
        validate_config(config)
    else:
        config = requested_config
        validate_config(config)
        base = Context(ROOT, config_path.resolve(), config, root / "COMMON")
        base.data["_frozen_config_value"] = copy.deepcopy(config)

    _assert_multistart_inline_rendering_disabled(
        config
    )
    if "algorithm_source_manifest" not in base.data:
        base.data["algorithm_source_manifest"] = _algorithm_source_manifest(base)

    write_json(
        root
        / "DEFERRED_VISUALIZATION_PLAN.json",
        {
            "schema":
                "HUD_FAN_V5_5_DEFERRED_VISUALIZATION_PLAN_V1",

            "inline_rendering":
                False,

            "renderer":
                "render_multistart_saved_v55.py",

            "render_steps":
                sorted(
                    _deferred_render_steps(base)
                ),

            "checkpoint_steps":
                sorted(
                    _execution_checkpoint_steps(base)
                ),

            "precheck_image":
                "07_ALL_PLANAR_START_RMS_COMPARISON.png",

            "algorithm_source_manifest_sha256":
                base.data.get(
                    "algorithm_source_manifest",
                    {},
                ).get(
                    "manifest_sha256"
                ),
        },
    )

    with execution_session(config, root):
        if not resume:
            for step in range(7):
                _execute_stage(base, step)

        print(
            "[07/PRECHECK] Evaluating configured planar seeds; "
            "STEP 06 has returned. Per-seed results will appear live.",
            flush=True,
        )
        candidates = enumerate_planar_seeds(base)
        candidate_rows = planar_seed_candidate_rows(candidates)

        eligible, scheduled = _select_step7_deep_run_schedule(
            candidates
        )

        selected_candidate_index = (
            int(scheduled[0][0])
            if scheduled
            else None
        )

        if max_seeds is not None and max_seeds != 1:
            print(
                "[07/SELECT] NOTICE: --max-seeds is retained for CLI compatibility; "
                "BEST_ONLY STEP 07 policy always deep-runs exactly one geometry.",
                flush=True,
            )
        _write_csv(root / "07_ALL_PLANAR_START_CANDIDATES.csv", candidate_rows)
        write_json(
            root
            / "07_ALL_PLANAR_START_CANDIDATES.json",
            {
                "schema":
                    "HUD_FAN_V5_5_PLANAR_START_CANDIDATES_V1",

                "rows":
                    candidate_rows,

                "algorithm_source_manifest_sha256":
                    base.data.get(
                        "algorithm_source_manifest",
                        {},
                    ).get(
                        "manifest_sha256"
                    ),
            },
        )
        precheck_view = {
            "status":
                "DEFERRED",

            "renderer":
                "render_multistart_saved_v55.py",

            "image":
                str(
                    root
                    / "07_ALL_PLANAR_START_RMS_COMPARISON.png"
                ),

            "metadata":
                str(
                    root
                    / "07_ALL_PLANAR_START_RMS_COMPARISON.json"
                ),
        }
        write_json(root / "07_MULTISTART_PRECHECK.json", {
            "total_candidates": len(candidates),
            "physical_eligible_candidates": [
                index
                for index, seed in enumerate(candidates, 1)
                if seed["physical_eligible"]
            ],
            "eligible_candidates": [
                index
                for index, seed in enumerate(candidates, 1)
                if seed["eligible"]
            ],
            "eligible_candidates_in_geometry_rank_order": [
                index
                for index, seed in eligible
            ],
            "scheduled_candidates": [
                index
                for index, seed in scheduled
            ],
            "selected_candidate": selected_candidate_index,
            "selected_geometry_rank": (
                1
                if scheduled
                else None
            ),
            "deep_run_policy": (
                "STEP07_GEOMETRY_RANK_ORDER_UNTIL_FIRST_COMPLETED"
            ),
            "deep_run_count": len(scheduled),
            "max_seeds_argument": max_seeds,
            "max_seeds_controls_deep_run_count": False,
            "step7_geometry_search_mode":
                config["solver"]["step7_geometry_search"]["mode"],
            "packaging_constraint_enabled":
                bool(base.data["inputs"]["packaging_constraint_enabled"]),
            "packaging_lambda_limit":
                (
                    float(config["packaging_lambda_max"])
                    if base.data["inputs"]["packaging_constraint_enabled"]
                    else None
                ),
            "eligibility_rule":
                "FINITE_AND_FULL_PHYSICAL_FIRST_HIT_AND_UNOBSCURED_AND_OPTIONAL_PACKAGING_LAMBDA",
            "geometry_ranking_rule":
                "BBOX_VOLUME_THEN_BBOX_DIAGONAL_THEN_MAX_OPTIC_DIAGONAL_THEN_TOTAL_APERTURE_AREA_THEN_CHIEF_PATH_THEN_PACKAGING_THEN_PLANAR_RMS",
            "planar_paper_spot_rms_metric":
                "CHIEF_CENTERED_GEOMETRIC_SPOT_RMS_ALL_FIELD_PUPIL_RAYS",
            "planar_paper_spot_rms_rule":
                "DIAGNOSTIC_ONLY_AFTER_STEP07_GEOMETRY_GATES",
            "rms_is_hard_gate": False,
            "planar_paper_spot_rms_comparison_view": precheck_view,
            "selection_rule": SELECTION_RULE,
            "precheck_only": precheck_only,
        })
        if not eligible:
            raise RuntimeError(
                "NO_STEP7_GEOMETRY_PASSES_PHYSICAL_OBSCURATION_AND_OPTIONAL_PACKAGING_GATE"
            )

        print(
            f"[07/SELECT] eligible={len(eligible)}; "
            f"selected candidate={selected_candidate_index:02d}; "
            "geometry_rank=1; deep_run_count=1",
            flush=True,
        )

        if precheck_only:
            _status(
                root,
                status="PRECHECK_COMPLETE",
                total_candidates=len(candidates),
                eligible_count=len(eligible),
                selected_candidate=selected_candidate_index,
                scheduled_candidates=[
                    index
                    for index, seed in scheduled
                ],
                completed_candidates=[],
            )
            return {
                "status": "PRECHECK_COMPLETE",
                "root": str(root),
                "eligible_count": len(eligible),
                "selected_candidate": selected_candidate_index,
                "deep_run_count": len(scheduled),
            }

        results: list[dict[str, Any]] = []
        completed_indices: list[int] = []

        for ordinal, (candidate_index, seed) in enumerate(
            scheduled,
            start=1,
        ):
            continue_to_next_geometry = False
            seed_dir = root / f"SEED_{candidate_index:03d}"
            existing_result = seed_dir / "SEED_RESULT.json"
            if resume and existing_result.exists():
                old = _normalize_result_identity(
                    json.loads(existing_result.read_text(encoding="utf-8")), "BASELINE")
                if old.get("execution_status") in ("COMPLETED", "FAILED_DURING_PIPELINE"):
                    print(f"=== SKIP TERMINAL SEED {candidate_index:02d}: "
                          f"{old.get('execution_status')} ===", flush=True)
                    results.append(old)
                    if old.get("execution_status") == "COMPLETED":
                        completed_indices.append(candidate_index)
                        break
                    elif str(old.get("message")) == "STEP11_NO_RAW_CONVEX_RAY_FACING_M2_CANDIDATE":
                        continue
                    else:
                        break
            print(
                f"=== SELECTED STEP7 GEOMETRY {candidate_index:02d} "
                f"({ordinal}/{len(scheduled)}) ===",
                flush=True,
            )
            ctx = copy.deepcopy(base)
            ctx.run_dir = seed_dir
            ctx.data["multistart_candidate"] = candidate_index
            ctx.data["step7_selected_geometry_rank"] = int(ordinal)
            if int(ordinal) == 1:
                ctx.data["step7_selected_geometry_rank"] = 1
            ctx.data["step7_deep_run_policy"] = (
                "BEST_FIRST_WITH_RAW_CONVEX_M2_FALLBACK"
            )
            ctx.data["_geometry_refinement_runtime_enabled"] = False
            ctx.data["multistart_variant"] = "BASELINE"
            last_step = 6
            try:
                _execute_selected_step_07(ctx, seed, candidate_index, candidate_rows)
                last_step = 7
                for step in range(8, 27):
                    _execute_stage(ctx, step)
                    last_step = step
                result = _completed_result(ctx, candidate_index, "BASELINE")
                completed_indices.append(candidate_index)
            except KeyboardInterrupt:
                _status(root, status="INTERRUPTED", current_candidate=candidate_index,
                        completed_candidates=completed_indices)
                raise
            except (BackendExecutionError, BackendConsistencyError):
                err_report = {
                    "schema": "HUD_RUN_EXECUTION_FAILURE_V1",
                    "candidate": candidate_index,
                    "variant": "BASELINE",
                    "completed_step": last_step,
                    "failed_step": last_step + 1,
                    "traceback": traceback.format_exc(),
                }
                write_json(root / "RUN_EXECUTION_FAILURE.json", err_report)
                raise
            except Exception as exc:
                failure_detail = ctx.data.get("_last_step_failure")
                if not isinstance(failure_detail, dict):
                    failure_detail = {}
                result = {
                    "candidate": candidate_index,
                    "variant": "BASELINE",
                    "design_id": f"SEED_{candidate_index:03d}_BASELINE",
                    "execution_status": "FAILED_DURING_PIPELINE",
                    "completed_step": last_step,
                    "failed_step": last_step + 1,
                    "run_dir": str(seed_dir),
                    "failure_phase": failure_detail.get("failure_phase"),
                    "failure_class": failure_detail.get("failure_class"),
                    "optical_fit_M1": failure_detail.get("optical_fit_M1"),
                    "optical_fit_M2": failure_detail.get("optical_fit_M2"),
                    "integrability_M1": failure_detail.get("integrability_M1"),
                    "integrability_M2": failure_detail.get("integrability_M2"),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                continue_to_next_geometry = bool(
                    str(exc)
                    ==
                    "STEP11_NO_RAW_CONVEX_RAY_FACING_M2_CANDIDATE"
                )

                result[
                    "step7_geometry_fallback_allowed"
                ] = continue_to_next_geometry

                write_json(seed_dir / "RUN_FAILURE.json", result | {"traceback": traceback.format_exc()})
                print(f"SEED_{candidate_index:03d}_FAILED STEP_{last_step + 1:02d}: {exc}",
                      file=sys.stderr, flush=True)
            write_json(seed_dir / "SEED_RESULT.json", result)
            results.append(result)
            _status(
                root,
                status="RUNNING_SELECTED_GEOMETRY",
                current_candidate=candidate_index,
                selected_candidate=selected_candidate_index,
                eligible_count=len(eligible),
                scheduled_count=len(scheduled),
                completed_candidates=completed_indices,
                processed_candidates=[
                    row["candidate"]
                    for row in results
                ],
            )

            if (
                result[
                    "execution_status"
                ]
                ==
                "COMPLETED"
            ):
                break

            if not continue_to_next_geometry:
                break

            print(
                "[STEP07 FALLBACK] "
                f"SEED_{candidate_index:03d} produced no raw-convex M2; "
                "trying the next eligible STEP07 geometry.",
                flush=True,
            )

        baseline_completed = [row for row in results if row["execution_status"] == "COMPLETED"]
        baseline_ranking = sorted(baseline_completed, key=lambda row: _ranking_key(row, config))
        for rank, row in enumerate(baseline_ranking, 1):
            row["rank"] = rank
        geometry_results = _run_geometry_top_k(root, config, baseline_ranking, resume)
        completed = baseline_completed + [row for row in geometry_results
                                          if row["execution_status"] == "COMPLETED"]
        ranking = sorted(completed, key=lambda row: _ranking_key(row, config))
        for rank, row in enumerate(ranking, 1):
            row["rank"] = rank
        failed = ([row for row in results if row["execution_status"] != "COMPLETED"] +
                  [row for row in geometry_results if row["execution_status"] != "COMPLETED"])
        comparison = ranking + failed
        write_json(root / "MULTISTART_COMPARISON.json", {
            "schema": "HUD_FAN_V5_5_MULTISTART_COMPARISON",
            "selection_rule": SELECTION_RULE,
            "ranking_priority": [
                "completed execution", "all hard gates pass", "fewer failed hard gates",
                "higher forward ray fraction", "higher valid bundle fraction",
                "evaluable then lower distortion", "lower packaging ratio", "lower MF1", "higher MTF tie-break",
            ],
            "results": comparison,
        })
        _write_csv(root / "MULTISTART_COMPARISON.csv", comparison)
        if not ranking:
            write_json(root / "MULTISTART_FAILURE_SUMMARY.json", {
                "schema": "HUD_FAN_V5_5_MULTISTART_FAILURE_SUMMARY_V1",
                "status": "FAILED_NO_COMPLETED_SEED",
                "eligible_count": len(eligible),
                "failure_count": len(failed),
                "failures": [{
                    "candidate": row.get("candidate"),
                    "variant": row.get("variant"),
                    "failed_step": row.get("failed_step"),
                    "failure_phase": row.get("failure_phase"),
                    "failure_class": row.get("failure_class"),
                    "optical_fit_M1": row.get("optical_fit_M1"),
                    "optical_fit_M2": row.get("optical_fit_M2"),
                    "integrability_M1": row.get("integrability_M1"),
                    "integrability_M2": row.get("integrability_M2"),
                    "exception_type": row.get("exception_type"),
                    "message": row.get("message"),
                } for row in failed],
            })
            _status(root, status="FAILED_NO_COMPLETED_SEED", eligible_count=len(eligible),
                    completed_candidates=[])
            raise RuntimeError("MULTISTART_NO_SEED_COMPLETED_STEP_26")
        best = ranking[0]
        manifest_hash = base.data.get("algorithm_source_manifest", {}).get("manifest_sha256")
        _status(root, status="FINALIZING_BEST_RESULT", eligible_count=len(eligible),
                completed_count=len(completed), failed_count=len(failed),
                algorithm_source_manifest_sha256=manifest_hash)
        try:
            _publish_best(root, best, ranking)
        except Exception as exc:
            _status(root, status="FAILED_DURING_BEST_RESULT_PUBLICATION",
                    exception_type=type(exc).__name__, message=str(exc),
                    algorithm_source_manifest_sha256=manifest_hash)
            raise
        final = {
            "status": "COMPLETE",
            "root": str(root),
            "step7_eligible_count": len(eligible),
            "step7_selected_candidate": selected_candidate_index,
            "step7_selected_geometry_rank": 1,
            "deep_run_policy": "BEST_STEP7_GEOMETRY_ONLY",
            "baseline_deep_run_count": len(scheduled),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "best_candidate": best["candidate"],
            "best_variant": best["variant"],
            "best_design_id": best["design_id"],
            "best_run_dir": best["run_dir"],
            "best_final_status": best["final_status"],
            "algorithm_source_manifest_sha256": manifest_hash,
        }
        _status(root, **final)
        return final


def main() -> None:
    """CLI cho run mới, precheck nhanh hoặc tiếp tục một multi-start run bị ngắt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config_v55.json")
    parser.add_argument("--output", type=Path, help="Thư mục run mới; mặc định tự sinh timestamp")
    parser.add_argument("--resume", type=Path, help="Tiếp tục run và bỏ qua seed đã COMPLETE")
    parser.add_argument(
        "--max-seeds",
        type=int,
        help=(
            "LEGACY compatibility only; BEST_ONLY STEP 07 policy always "
            "deep-runs exactly one geometry."
        ),
    )
    parser.add_argument("--precheck-only", action="store_true", help="Chỉ chạy STEP 00–07 precheck")
    args = parser.parse_args()
    if args.max_seeds is not None and args.max_seeds < 1:
        parser.error("--max-seeds must be at least 1")
    if args.resume and args.output:
        parser.error("use either --resume or --output, not both")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.resume.resolve() if args.resume else (
        args.output.resolve() if args.output else MULTISTART_ROOT / f"HUD_FAN_V5_5_MULTISTART_{stamp}"
    )
    result = run_multistart(args.config.resolve(), output, args.max_seeds,
                            args.precheck_only, resume=args.resume is not None)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
