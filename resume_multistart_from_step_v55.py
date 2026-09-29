"""Continue one saved multistart design from a completed STEP_xx state.

The runner never reconstructs optical state from CSV/JSON.  It loads the
completed pickle state saved by run_multistart_v55.py, preserves the frozen
configuration in that state, and calls the original stage executor for every
subsequent step in the same SEED_xxx directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import write_json
from execution_v55 import execution_session
from pipeline_v55 import STAGES, _algorithm_source_manifest, validate_config
from run_multistart_v55 import (
    SELECTION_RULE,
    _completed_result,
    _execute_stage,
    _load_checkpoint,
    _normalize_result_identity,
    _ranking_key,
    _status,
    _write_csv,
)


ROOT = Path(__file__).resolve().parent
MULTISTART_ROOT = (ROOT / "multistart_runs_v5_5").resolve()
MAX_STEP = max(STAGES)
STEP_FOLDER_PATTERN = re.compile(r"STEP_(\d{2})", re.IGNORECASE)
SEED_FOLDER_PATTERN = re.compile(r"SEED_(\d{3})", re.IGNORECASE)


def _utc_now() -> str:
    """Return an auditable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject any other top-level type."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def _sha256(path: Path) -> str:
    """Hash the exact state file used for the continuation."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_from_user(raw: str) -> Path:
    """Accept a pasted quoted or unquoted Windows path."""
    text = raw.strip().strip('"').strip("'").strip()
    if not text:
        raise ValueError("STEP_FOLDER_PATH_EMPTY")
    return Path(text).expanduser().resolve()


def _resolve_step_folder(path: Path) -> tuple[Path, Path, Path, int]:
    """Validate STEP_xx -> SEED_xxx -> timestamped multistart ownership."""
    if not path.is_dir():
        raise FileNotFoundError(f"STEP_FOLDER_NOT_FOUND:{path}")
    try:
        path.relative_to(MULTISTART_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"STEP_FOLDER_MUST_BE_INSIDE_MULTISTART_ROOT:{MULTISTART_ROOT}"
        ) from exc

    step_match = STEP_FOLDER_PATTERN.fullmatch(path.name)
    if step_match is None:
        raise ValueError(f"EXPECTED_STEP_XX_FOLDER:{path}")
    completed_step = int(step_match.group(1))
    if completed_step < 0 or completed_step > MAX_STEP:
        raise ValueError(f"STEP_OUT_OF_RANGE:{completed_step}")

    seed_dir = path.parent.resolve()
    if SEED_FOLDER_PATTERN.fullmatch(seed_dir.name) is None:
        raise ValueError(f"EXPECTED_SEED_XXX_PARENT:{seed_dir}")

    if seed_dir.parent.name == "GEOMETRY_REFINEMENT":
        multistart_root = seed_dir.parent.parent.resolve()
    else:
        multistart_root = seed_dir.parent.resolve()
    try:
        multistart_root.relative_to(MULTISTART_ROOT)
    except ValueError as exc:
        raise RuntimeError("MULTISTART_ROOT_RESOLUTION_FAILED") from exc
    return path, seed_dir, multistart_root, completed_step


def _completed_state_path(seed_dir: Path, step: int) -> tuple[Path, str, dict[str, Any]]:
    """Prefer a production checkpoint, otherwise use a completed render state."""
    checkpoint_dir = seed_dir / "CHECKPOINTS" / f"STEP_{step:02d}"
    checkpoint = checkpoint_dir / "state.pkl.gz"
    checkpoint_meta = checkpoint_dir / "checkpoint.json"
    if checkpoint.is_file() and checkpoint_meta.is_file():
        metadata = _read_json(checkpoint_meta)
        if int(metadata.get("completed_step", -1)) == step:
            return checkpoint, "CHECKPOINT", metadata

    render_dir = seed_dir / "RENDER_STATES" / f"STEP_{step:02d}"
    render_state = render_dir / "state.pkl.gz"
    render_meta = render_dir / "metadata.json"
    if render_state.is_file() and render_meta.is_file():
        metadata = _read_json(render_meta)
        if (
            int(metadata.get("step", -1)) == step
            and metadata.get("execution_status") == "ALGORITHM_COMPLETED"
            and metadata.get("state_kind") == "COMPLETED_STEP_STATE"
        ):
            return render_state, "RENDER_STATE", metadata
        raise RuntimeError(
            f"STEP_STATE_IS_NOT_COMPLETED:{render_dir}:"
            f"status={metadata.get('execution_status')}:"
            f"kind={metadata.get('state_kind')}"
        )

    raise FileNotFoundError(
        f"COMPLETED_STATE_NOT_FOUND:STEP_{step:02d}:"
        f"checked={checkpoint};{render_state}"
    )


def _load_and_validate_context(
    state_path: Path,
    seed_dir: Path,
    completed_step: int,
) -> Any:
    """Load only a trusted local state and verify its identity and completion."""
    ctx = _load_checkpoint(state_path)
    saved_run_dir = Path(ctx.run_dir).resolve()
    if saved_run_dir != seed_dir:
        raise RuntimeError(
            f"STATE_RUN_DIR_MISMATCH:saved={saved_run_dir}:requested={seed_dir}"
        )
    summary = ctx.stage_summaries.get(completed_step)
    if not isinstance(summary, dict):
        raise RuntimeError(f"STATE_MISSING_COMPLETED_SUMMARY:STEP_{completed_step:02d}")
    if (
        summary.get("execution_status") != "ALGORITHM_COMPLETED"
        or summary.get("status") not in ("PASS", "COMPLETE")
    ):
        raise RuntimeError(
            f"STATE_SUMMARY_NOT_COMPLETED:STEP_{completed_step:02d}:"
            f"status={summary.get('status')}:"
            f"execution_status={summary.get('execution_status')}"
        )
    unexpected_later = sorted(
        int(step)
        for step in ctx.stage_summaries
        if int(step) > completed_step
    )
    if unexpected_later:
        raise RuntimeError(
            f"STATE_CONTAINS_LATER_STEPS:{unexpected_later}:"
            "select the latest completed STEP folder instead"
        )
    validate_config(ctx.config)
    from pipeline_v55 import verify_sampling_profile_on_resume
    verify_sampling_profile_on_resume(ctx, completed_step)
    ctx.source_dir = ROOT
    ctx.run_dir = seed_dir
    return ctx


def _candidate_identity(ctx: Any, seed_dir: Path) -> tuple[int, str]:
    """Recover the same multistart identity stored by the original runner."""
    match = SEED_FOLDER_PATTERN.fullmatch(seed_dir.name)
    folder_candidate = int(match.group(1)) if match is not None else -1
    candidate = int(ctx.data.get("multistart_candidate", folder_candidate))
    if candidate != folder_candidate:
        raise RuntimeError(
            f"STATE_CANDIDATE_MISMATCH:state={candidate}:folder={folder_candidate}"
        )
    variant = str(ctx.data.get("multistart_variant", "BASELINE"))
    return candidate, variant


def _downstream_targets(
    seed_dir: Path,
    multistart_root: Path,
    start_step: int,
    end_step: int,
) -> list[Path]:
    """List stale downstream artifacts that would otherwise mix two attempts."""
    targets: list[Path] = []
    for step in range(start_step, end_step + 1):
        targets.extend([
            seed_dir / f"STEP_{step:02d}",
            seed_dir / "CHECKPOINTS" / f"STEP_{step:02d}",
            seed_dir / "RENDER_STATES" / f"STEP_{step:02d}",
            seed_dir / f"RUN_FAILURE_STEP_{step:02d}.json",
        ])
    targets.extend([
        seed_dir / "RUN_FAILURE.json",
        seed_dir / "SEED_RESULT.json",
    ])

    root_export_pattern = re.compile(r"(?:\d{2}[A-Z]?_|FINAL_).+")
    for child in seed_dir.iterdir():
        if child.is_file() and root_export_pattern.fullmatch(child.name):
            targets.append(child)

    targets.extend([
        multistart_root / "MULTISTART_FAILURE_SUMMARY.json",
        multistart_root / "GEOMETRY_REFINEMENT_SELECTION.json",
        multistart_root / "DEFERRED_VISUALIZATION_REPORT.json",
    ])
    best_result = multistart_root / "BEST_RESULT"
    if best_result.exists():
        targets.append(best_result)

    unique: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.resolve()
        if target.exists() and resolved not in seen:
            resolved.relative_to(multistart_root)
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _archive_targets(
    targets: list[Path],
    multistart_root: Path,
    attempt_dir: Path,
) -> list[dict[str, Any]]:
    """Move stale artifacts into the recoverable resume-attempt directory."""
    records: list[dict[str, Any]] = []
    archive_root = attempt_dir / "BEFORE_RESUME"
    for source in targets:
        relative = source.relative_to(multistart_root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"RESUME_ARCHIVE_DESTINATION_EXISTS:{destination}")
        shutil.move(str(source), str(destination))
        records.append({
            "source": str(source),
            "archive": str(destination),
            "kind": "directory" if destination.is_dir() else "file",
        })
    return records


def _comparison_results(multistart_root: Path, replacement: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace the stale result for this design while preserving other designs."""
    comparison_path = multistart_root / "MULTISTART_COMPARISON.json"
    rows: list[dict[str, Any]] = []
    if comparison_path.is_file():
        payload = _read_json(comparison_path)
        raw_rows = payload.get("results", [])
        if isinstance(raw_rows, list):
            rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    if not rows:
        for result_path in sorted(multistart_root.glob("SEED_*/SEED_RESULT.json")):
            payload = _read_json(result_path)
            rows.append(payload)

    replacement_id = replacement["design_id"]
    rows = [
        _normalize_result_identity(row, str(row.get("variant", "BASELINE")))
        for row in rows
        if row.get("design_id") != replacement_id
    ]
    rows.append(replacement)
    return rows


def _update_multistart_aggregate(
    multistart_root: Path,
    result: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Publish an honest baseline result without launching optional refinement branches."""
    rows = _comparison_results(multistart_root, result)
    completed = [row for row in rows if row.get("execution_status") == "COMPLETED"]
    failed = [row for row in rows if row.get("execution_status") != "COMPLETED"]
    ranking = sorted(completed, key=lambda row: _ranking_key(row, config))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    for row in failed:
        row.pop("rank", None)
    comparison = ranking + failed
    write_json(multistart_root / "MULTISTART_COMPARISON.json", {
        "schema": "HUD_FAN_V5_5_MULTISTART_COMPARISON",
        "selection_rule": SELECTION_RULE,
        "continuation_note": (
            "UPDATED_BY_RESUME_FROM_COMPLETED_STEP;"
            "OPTIONAL_GEOMETRY_REFINEMENT_NOT_LAUNCHED_BY_THIS_RUNNER"
        ),
        "results": comparison,
    })
    _write_csv(multistart_root / "MULTISTART_COMPARISON.csv", comparison)

    refinement_pending = bool(
        config.get("geometry_refinement", {}).get("enabled", False)
        or config.get("surface_parameter_refinement", {}).get("enabled", False)
    )
    _status(
        multistart_root,
        status=(
            "BASELINE_RESUME_COMPLETE_REFINEMENT_PENDING"
            if refinement_pending
            else "COMPLETE"
        ),
        resumed_design_id=result["design_id"],
        resumed_completed_step=26,
        completed_count=len(completed),
        failed_count=len(failed),
        optional_geometry_refinement_pending=refinement_pending,
        algorithm_source_manifest_sha256=result.get(
            "algorithm_source_manifest_sha256"
        ),
    )


def _plan_payload(
    step_dir: Path,
    seed_dir: Path,
    multistart_root: Path,
    completed_step: int,
    start_step: int,
    end_step: int,
    state_path: Path,
    state_source: str,
    state_metadata: dict[str, Any],
    state_sha256: str,
    ctx: Any,
    archive_targets: list[Path],
) -> dict[str, Any]:
    """Build the immutable portion of the resume audit."""
    previous_manifest = ctx.data.get("algorithm_source_manifest", {})
    current_manifest = _algorithm_source_manifest(ctx)
    return {
        "schema": "HUD_FAN_V5_5_RESUME_FROM_COMPLETED_STEP_V1",
        "requested_step_folder": str(step_dir),
        "seed_dir": str(seed_dir),
        "multistart_root": str(multistart_root),
        "completed_step": completed_step,
        "start_step": start_step,
        "end_step": end_step,
        "steps_to_execute": list(range(start_step, end_step + 1)),
        "state_source": state_source,
        "state_path": str(state_path),
        "state_sha256": state_sha256,
        "state_metadata": state_metadata,
        "config_policy": "FROZEN_CONFIG_FROM_COMPLETED_STATE",
        "saved_source_manifest_sha256": previous_manifest.get("manifest_sha256"),
        "current_source_manifest_sha256": current_manifest.get("manifest_sha256"),
        "source_changed": (
            previous_manifest.get("manifest_sha256")
            != current_manifest.get("manifest_sha256")
        ),
        "archive_targets": [str(path) for path in archive_targets],
        "algorithm_guarantee": (
            "SUBSEQUENT_STEPS_USE_ORIGINAL_STAGES_AND_RUN_MULTISTART_EXECUTOR"
        ),
    }


def resume_from_completed_step(
    step_folder: Path,
    *,
    end_step: int = MAX_STEP,
    dry_run: bool = False,
    assume_yes: bool = False,
) -> dict[str, Any]:
    """Validate, archive stale downstream output, and continue in place."""
    step_dir, seed_dir, multistart_root, completed_step = _resolve_step_folder(
        step_folder
    )
    start_step = completed_step + 1
    if start_step > MAX_STEP:
        raise ValueError("STEP_26_HAS_NO_SUBSEQUENT_STEP")
    if end_step < start_step or end_step > MAX_STEP:
        raise ValueError(
            f"END_STEP_MUST_BE_BETWEEN_{start_step:02d}_AND_{MAX_STEP:02d}"
        )

    state_path, state_source, state_metadata = _completed_state_path(
        seed_dir,
        completed_step,
    )
    state_sha256 = _sha256(state_path)
    ctx = _load_and_validate_context(state_path, seed_dir, completed_step)
    candidate, variant = _candidate_identity(ctx, seed_dir)
    archive_targets = _downstream_targets(
        seed_dir,
        multistart_root,
        start_step,
        end_step,
    )
    plan = _plan_payload(
        step_dir,
        seed_dir,
        multistart_root,
        completed_step,
        start_step,
        end_step,
        state_path,
        state_source,
        state_metadata,
        state_sha256,
        ctx,
        archive_targets,
    )

    print("=" * 88, flush=True)
    print("RESUME MULTISTART FROM COMPLETED STEP", flush=True)
    print("=" * 88, flush=True)
    print(f"Seed folder : {seed_dir}", flush=True)
    print(f"State       : {state_source} STEP_{completed_step:02d}", flush=True)
    print(f"Execute     : STEP_{start_step:02d} -> STEP_{end_step:02d}", flush=True)
    print("Output      : same SEED folder", flush=True)
    print("Config      : frozen config stored in the completed state", flush=True)
    print(f"Source code : {'CHANGED' if plan['source_changed'] else 'UNCHANGED'}", flush=True)
    print(f"Archive     : {len(archive_targets)} stale downstream target(s)", flush=True)

    if dry_run:
        print("DRY RUN: no file was changed and no STEP was executed.", flush=True)
        return {"status": "DRY_RUN_OK", **plan}

    if not assume_yes:
        answer = input("Type YES to continue in place: ").strip()
        if answer != "YES":
            raise RuntimeError("RESUME_CANCELLED_BY_USER")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    attempt_dir = seed_dir / "RESUME_ATTEMPTS" / (
        f"RESUME_FROM_STEP_{completed_step:02d}_{stamp}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    plan["attempt_dir"] = str(attempt_dir)
    plan["started_utc"] = _utc_now()
    write_json(attempt_dir / "RESUME_PLAN.json", plan)
    archived = _archive_targets(archive_targets, multistart_root, attempt_dir)
    write_json(attempt_dir / "ARCHIVED_ARTIFACTS.json", {
        "schema": "HUD_FAN_V5_5_RESUME_ARCHIVED_ARTIFACTS_V1",
        "records": archived,
    })

    previous_manifest = ctx.data.get("algorithm_source_manifest", {})
    current_manifest = _algorithm_source_manifest(ctx)
    history = ctx.data.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise RuntimeError("RESUME_HISTORY_MUST_BE_LIST")
    history.append({
        "completed_step_source": completed_step,
        "start_step": start_step,
        "end_step": end_step,
        "state_path": str(state_path),
        "state_sha256": state_sha256,
        "attempt_dir": str(attempt_dir),
        "saved_source_manifest_sha256": previous_manifest.get("manifest_sha256"),
        "current_source_manifest_sha256": current_manifest.get("manifest_sha256"),
        "resumed_utc": _utc_now(),
    })
    ctx.data["algorithm_source_manifest"] = current_manifest
    ctx.data.pop("_last_step_failure", None)
    ctx.data.pop("_step_debug", None)
    ctx.data["_execution_phase"] = "RESUME_FROM_COMPLETED_STEP"

    last_completed = completed_step
    try:
        _status(
            multistart_root,
            status="RESUMING_FROM_COMPLETED_STEP",
            current_candidate=candidate,
            variant=variant,
            completed_step=completed_step,
            next_step=start_step,
            end_step=end_step,
            attempt_dir=str(attempt_dir),
            source_changed=plan["source_changed"],
        )
        with execution_session(ctx.config, multistart_root):
            for step in range(start_step, end_step + 1):
                _execute_stage(ctx, step)
                last_completed = step

        if end_step == MAX_STEP:
            result = _completed_result(ctx, candidate, variant)
            write_json(seed_dir / "SEED_RESULT.json", result)
            _update_multistart_aggregate(multistart_root, result, ctx.config)
            status = "COMPLETED"
        else:
            result = {
                "candidate": candidate,
                "variant": variant,
                "design_id": f"SEED_{candidate:03d}_{variant}",
                "execution_status": "PARTIAL_CONTINUATION_COMPLETE",
                "completed_step": last_completed,
                "next_step": last_completed + 1,
                "run_dir": str(seed_dir),
                "algorithm_source_manifest_sha256": current_manifest.get(
                    "manifest_sha256"
                ),
            }
            _status(
                multistart_root,
                status="PARTIAL_CONTINUATION_COMPLETE",
                **result,
            )
            status = "PARTIAL_CONTINUATION_COMPLETE"

        outcome = {
            "schema": "HUD_FAN_V5_5_RESUME_RESULT_V1",
            "status": status,
            "completed_step_source": completed_step,
            "last_completed_step": last_completed,
            "attempt_dir": str(attempt_dir),
            "result": result,
            "finished_utc": _utc_now(),
        }
        write_json(attempt_dir / "RESUME_RESULT.json", outcome)
        return outcome
    except KeyboardInterrupt:
        outcome = {
            "schema": "HUD_FAN_V5_5_RESUME_RESULT_V1",
            "status": "INTERRUPTED",
            "completed_step_source": completed_step,
            "last_completed_step": last_completed,
            "next_step": last_completed + 1,
            "attempt_dir": str(attempt_dir),
            "finished_utc": _utc_now(),
        }
        write_json(attempt_dir / "RESUME_RESULT.json", outcome)
        _status(multistart_root, **outcome)
        raise
    except Exception as exc:
        failure_detail = ctx.data.get("_last_step_failure")
        if not isinstance(failure_detail, dict):
            failure_detail = {}
        failed_step = last_completed + 1
        result = {
            "candidate": candidate,
            "variant": variant,
            "design_id": f"SEED_{candidate:03d}_{variant}",
            "execution_status": "FAILED_DURING_RESUME",
            "completed_step": last_completed,
            "failed_step": failed_step,
            "run_dir": str(seed_dir),
            "failure_phase": failure_detail.get("failure_phase"),
            "failure_class": failure_detail.get("failure_class"),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        write_json(seed_dir / "RUN_FAILURE.json", {
            **result,
            "traceback": traceback.format_exc(),
        })
        write_json(seed_dir / "SEED_RESULT.json", result)
        outcome = {
            "schema": "HUD_FAN_V5_5_RESUME_RESULT_V1",
            "status": "FAILED",
            "completed_step_source": completed_step,
            "last_completed_step": last_completed,
            "failed_step": failed_step,
            "attempt_dir": str(attempt_dir),
            "result": result,
            "traceback": traceback.format_exc(),
            "finished_utc": _utc_now(),
        }
        write_json(attempt_dir / "RESUME_RESULT.json", outcome)
        _status(
            multistart_root,
            status="FAILED_DURING_RESUME",
            current_candidate=candidate,
            completed_step=last_completed,
            failed_step=failed_step,
            attempt_dir=str(attempt_dir),
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def main() -> None:
    """CLI accepting a pasted STEP folder or a positional path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step_folder",
        nargs="?",
        help="Completed folder such as ...\\SEED_043\\STEP_11",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=MAX_STEP,
        help=f"Last STEP to run; default {MAX_STEP}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate state and show the plan without changing files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final in-place continuation confirmation",
    )
    args = parser.parse_args()

    raw_path = args.step_folder
    if raw_path is None:
        raw_path = input("Paste completed STEP folder: ")
    step_folder = _path_from_user(raw_path)
    result = resume_from_completed_step(
        step_folder,
        end_step=args.end_step,
        dry_run=args.dry_run,
        assume_yes=args.yes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
