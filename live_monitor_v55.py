"""Theo doi truc tiep HUD; khong tham gia quyet dinh quang hoc."""

from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


_ACTIVE: ContextVar[Any] = ContextVar(
    "hud_v55_live_monitor", default=None
)


def _print_safely(message: str) -> None:
    """Loi terminal khong duoc thay doi ket qua thuat toan."""
    try:
        print(message, flush=True)
    except Exception:
        pass


def _plain(value: Any) -> Any:
    """Chuyen metadata sang JSON, khong sua doi du lieu nguon."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("LIVE_METADATA_KEYS_MUST_BE_STRINGS")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    raise TypeError(f"LIVE_UNSUPPORTED_TYPE:{type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    """Ghi file tam, sau do thay file dich."""
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()

        for attempt in range(3):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.02)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class LiveMonitor:
    """Mot writer cho mot lan goi precheck hoac STEP16."""

    def __init__(
        self,
        root: Path,
        metadata: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        """Tao session rieng, khong dung lai anh cua session cu."""
        self.root = Path(root).resolve()
        self.options = copy.deepcopy(options)
        self.session_id = (
            f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        )
        self.session_dir = (
            self.root / "sessions" / self.session_id
        )
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self.sequence = 0
        self.operation_sequence = 0
        self.image_sequence = 0
        self.disabled = False
        self.last_image_clock = -math.inf
        self.phase_clock = time.perf_counter()
        self.warning_keys: set[tuple[str, str]] = set()

        self.state: dict[str, Any] = {
            "schema": "HUD_LIVE_MONITOR_V1",
            **_plain(metadata),
            "session_id": self.session_id,
            "pid": os.getpid(),
            "session_status": "RUNNING",
            "optical_certification": "NOT_PERFORMED_BY_MONITOR",
            "item": None,
            "completed_items": 0,
            "phase": "INITIALIZING",
            "phase_started_unix": time.time(),
            "task": None,
            "results": [],
            "last_result": None,
            "latest_image": None,
            "last_algorithm_event_unix": time.time(),
        }

    def emit(
        self,
        event: str,
        *,
        algorithm_event: bool = True,
        **updates: Any,
    ) -> None:
        """Ghi event; loi telemetry khong duoc nem vao solver."""
        if self.disabled:
            return

        try:
            values = _plain(updates)
            self.state.update(values)
            self.sequence += 1
            now = time.time()

            self.state.update({
                "event": event,
                "event_sequence": self.sequence,
                "updated_unix": now,
            })
            if algorithm_event:
                self.state["last_algorithm_event_unix"] = now

            entry = {
                "event": event,
                "event_sequence": self.sequence,
                "at_unix": now,
                "session_id": self.session_id,
                "item": self.state["item"],
                "phase": self.state["phase"],
                "updates": values,
            }

            event_text = json.dumps(
                entry, ensure_ascii=False, allow_nan=False
            )
            with (self.session_dir / "events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(event_text + "\n")
                stream.flush()

            _atomic_text(
                self.root / "latest.json",
                json.dumps(
                    self.state,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
            )

            item = self.state.get("item") or {}
            task = self.state.get("task") or {}
            suffix = ""
            if task:
                suffix = (
                    f" usable_processed={task.get('completed', '?')}"
                    f"/{task.get('total', '?')}"
                    f" batch={task.get('batch_total', '?')}"
                )

            _print_safely(
                f"[{self.state['scope']}] {event}"
                f" item={item.get('ordinal', '-')}"
                f"/{item.get('total', '-')}"
                f" phase={self.state['phase']}{suffix}"
            )

            if event == "ITEM_RESULT":
                _print_safely(
                    "  RESULT "
                    + json.dumps(
                        self.state["last_result"],
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
        except Exception as exc:
            self.disabled = True
            _print_safely(
                "LIVE_MONITOR_DISABLED: "
                f"{type(exc).__name__}: {exc}"
            )

    def warning(self, origin: str, exc: Exception) -> None:
        """Ghi loi preview mot lan cho moi nhom loi."""
        key = (origin, type(exc).__name__)
        if key in self.warning_keys:
            return
        self.warning_keys.add(key)
        self.emit(
            "MONITOR_WARNING",
            algorithm_event=False,
            last_monitor_warning={
                "origin": origin,
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )


def observe(
    step: int,
    scope: str,
    relative_directory: str,
) -> Callable:
    """Gan session telemetry vao ham, khong thay ket qua ham."""

    def decorate(function: Callable) -> Callable:
        """Tao wrapper giu nguyen metadata ham goc."""

        @wraps(function)
        def wrapped(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            """Goi function dung mot lan va reset ContextVar."""
            monitor = None
            options = ctx.config.get("live_monitor")

            if options and options.get("enabled", False):
                try:
                    monitor = LiveMonitor(
                        Path(ctx.run_dir) / relative_directory,
                        {
                            "step": int(step),
                            "scope": scope,
                            "run_dir": str(Path(ctx.run_dir).resolve()),
                            "source_manifest_sha256": (
                                ctx.data.get(
                                    "algorithm_source_manifest", {}
                                ).get("manifest_sha256")
                            ),
                        },
                        options,
                    )
                except Exception as exc:
                    _print_safely(
                        "LIVE_MONITOR_INIT_WARNING: "
                        f"{type(exc).__name__}: {exc}"
                    )

            token = _ACTIVE.set(monitor)
            try:
                if monitor is not None:
                    monitor.emit("SESSION_STARTED")
                    _print_safely(
                        f"LIVE_DIRECTORY: {monitor.root}"
                    )

                result = function(ctx, *args, **kwargs)

                if monitor is not None:
                    monitor.emit(
                        "SESSION_RETURNED",
                        session_status="FUNCTION_RETURNED",
                        function_result_status=(
                            str(result.get("status", "RETURNED"))
                            if isinstance(result, dict)
                            else "RETURNED"
                        ),
                    )
                return result
            except BaseException as exc:
                if monitor is not None:
                    monitor.emit(
                        "SESSION_ABORTED",
                        session_status="ABORTED",
                        exception_type=type(exc).__name__,
                        exception_message=str(exc),
                    )
                raise
            finally:
                _ACTIVE.reset(token)

        return wrapped

    return decorate


def live_item(
    kind: str,
    ordinal: int,
    total: int,
    **details: Any,
) -> None:
    """Bao bat dau seed/rho; chua tang completed_items."""
    monitor = _ACTIVE.get()
    if monitor is None:
        return
    monitor.phase_clock = time.perf_counter()
    monitor.emit(
        "ITEM_STARTED",
        item={
            "kind": kind,
            "ordinal": int(ordinal),
            "total": int(total),
            "details": details,
            "started_unix": time.time(),
        },
        phase="ITEM_START",
        phase_started_unix=time.time(),
        phase_details={},
        task=None,
    )


def live_phase(name: str, **details: Any) -> None:
    """Bao pha truoc khi goi phep tinh nang."""
    monitor = _ACTIVE.get()
    if monitor is None:
        return
    now = time.perf_counter()
    previous = {
        "name": monitor.state["phase"],
        "elapsed_seconds": now - monitor.phase_clock,
    }
    monitor.phase_clock = now
    monitor.emit(
        "PHASE_STARTED",
        previous_phase=previous,
        phase=str(name),
        phase_started_unix=time.time(),
        phase_details=details,
        task=None,
    )


def live_results(
    producer: Callable[[], list[dict[str, Any]]],
) -> None:
    """Lay bang ket qua da co trong barrier rieng cua telemetry."""
    monitor = _ACTIVE.get()
    if monitor is None or monitor.disabled:
        return
    try:
        rows = producer()
        monitor.emit(
            "ITEM_RESULT",
            completed_items=len(rows),
            results=rows,
            last_result=rows[-1] if rows else None,
        )
    except Exception as exc:
        monitor.warning("RESULT_TABLE", exc)


def live_note(event: str, **details: Any) -> None:
    """Ghi thong tin lua chon da duoc thuat toan quyet dinh."""
    monitor = _ACTIVE.get()
    if monitor is not None:
        monitor.emit(event, note=details)


def progress_indices(
    values: Any,
    operation: str,
    *,
    batch_total: int,
) -> Iterator[Any]:
    """Giu nguyen thu tu loop; dem usable rays da xu ly."""
    monitor = _ACTIVE.get()
    if monitor is None or monitor.disabled:
        yield from values
        return

    total = len(values)
    monitor.operation_sequence += 1
    task = {
        "id": monitor.operation_sequence,
        "operation": operation,
        "completed": 0,
        "total": total,
        "batch_total": int(batch_total),
        "not_in_usable_loop": int(batch_total) - total,
        "started_unix": time.time(),
        "count_meaning": "PROCESSED_INPUTS_NOT_VALID_RAYS",
    }
    monitor.emit("OPERATION_STARTED", task=task)

    last_emit = time.perf_counter()
    interval = float(
        monitor.options["progress_interval_seconds"]
    )

    for completed, value in enumerate(values, start=1):
        yield value

        # Dem sau than loop. "continue" van quay lai day.
        if completed % 64 == 0 or completed == total:
            now = time.perf_counter()
            if now - last_emit >= interval or completed == total:
                task = {**task, "completed": completed}
                monitor.emit(
                    "OPERATION_FINISHED"
                    if completed == total
                    else "OPERATION_PROGRESS",
                    task=task,
                )
                last_emit = now

    if total == 0:
        monitor.emit("OPERATION_FINISHED", task=task)


def live_capture(
    label: str,
    renderer: Callable[[Path, int], dict[str, Any]],
    *,
    role: str,
    force: bool = False,
) -> None:
    """Ve snapshot tren main thread, khong goi them optical solver."""
    monitor = _ACTIVE.get()
    if (
        monitor is None
        or monitor.disabled
        or not monitor.options["images_enabled"]
    ):
        return

    now = time.perf_counter()
    interval = float(
        monitor.options["image_min_interval_seconds"]
    )
    if not force and now - monitor.last_image_clock < interval:
        return

    monitor.image_sequence += 1
    filename = f"snap_{monitor.image_sequence:06d}.png"
    destination = monitor.session_dir / filename
    temporary = destination.with_name(
        f".{destination.stem}.tmp.png"
    )

    source_item = copy.deepcopy(monitor.state["item"])
    source_phase = str(monitor.state["phase"])
    source_event = monitor.sequence
    started = time.perf_counter()

    try:
        drawing_info = renderer(
            temporary,
            int(monitor.options["max_plot_rays"]),
        )
        os.replace(temporary, destination)
        monitor.last_image_clock = time.perf_counter()

        monitor.emit(
            "IMAGE_READY",
            algorithm_event=False,
            latest_image={
                "path": destination.relative_to(
                    monitor.root
                ).as_posix(),
                "label": label,
                "role": role,
                "item": source_item,
                "phase": source_phase,
                "source_event_sequence": source_event,
                "created_unix": time.time(),
                "render_seconds": (
                    time.perf_counter() - started
                ),
                "source_manifest_sha256": monitor.state[
                    "source_manifest_sha256"
                ],
                "drawing": drawing_info,
            },
        )
    except Exception as exc:
        monitor.warning("PREVIEW_RENDER", exc)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
