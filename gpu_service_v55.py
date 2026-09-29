"""Dedicated GPU Owner Service running CuPy in an isolated process with IPC."""

from __future__ import annotations

import multiprocessing as mp
import os
import secrets
import time
from typing import Any

from kernels_array_v55 import cheb_ray_equations, fermat_eval

CUDA_FUNCTIONS = {
    "cheb_ray_equations": cheb_ray_equations,
    "fermat_eval": fermat_eval,
}

ARRAY_FIELDS = {
    "cheb_ray_equations": ("coeff", "scale", "ol", "dl"),
    "fermat_eval": (
        "xy", "q1", "targets", "center", "frame", "scale", "coeff",
    ),
}


def _execute_cuda_request(
    cp,
    kernel_name: str,
    payload: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Thực thi _execute_cuda_request."""
    wall_start = time.perf_counter()

    upload_start = cp.cuda.Event()
    upload_end = cp.cuda.Event()
    kernel_end = cp.cuda.Event()
    upload_start.record()

    device_payload = dict(payload)
    for key in ARRAY_FIELDS[kernel_name]:
        if key in payload:
            device_payload[key] = cp.asarray(payload[key], dtype=cp.float64)

    upload_end.record()
    result = CUDA_FUNCTIONS[kernel_name](cp, **device_payload)
    kernel_end.record()

    if isinstance(result, tuple):
        host_result = tuple(cp.asnumpy(item) for item in result)
    else:
        host_result = cp.asnumpy(result)

    cp.cuda.get_current_stream().synchronize()

    row_count = 0
    if "ol" in payload and hasattr(payload["ol"], "__len__"):
        row_count = len(payload["ol"])
    elif "xy" in payload and hasattr(payload["xy"], "__len__"):
        row_count = len(payload["xy"])

    timing = {
        "upload_ms": float(cp.cuda.get_elapsed_time(upload_start, upload_end)),
        "kernel_ms": float(cp.cuda.get_elapsed_time(upload_end, kernel_end)),
        "wall_seconds_in_owner": time.perf_counter() - wall_start,
        "dtype": "float64",
        "backend": "cuda",
        "actual_backend": "cuda",
        "kernel": kernel_name,
        "row_count": row_count,
    }

    return host_result, timing


def _gpu_owner_loop(
    authkey,
    device_index,
    memory_pool_limit_gib,
    request_timeout_seconds,
    max_message_bytes,
    ready_send,
    stop_event,
):
    """Thực thi _gpu_owner_loop."""
    import hashlib
    import pickle
    import socket
    import time

    from gpu_transport_v55 import receive_message, send_message

    server = None
    ready_sent = False

    try:
        import cupy as cp

        with cp.cuda.Device(device_index):
            cp.get_default_memory_pool().set_limit(
                size=int(memory_pool_limit_gib * 1024**3)
            )
            cp.zeros(1, dtype=cp.float64)
            cp.cuda.get_current_stream().synchronize()

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("127.0.0.1", 0))
            server.listen(16)
            server.settimeout(0.25)

            props = cp.cuda.runtime.getDeviceProperties(device_index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            ready_send.send({
                "status": "READY",
                "address": server.getsockname(),
                "device": {
                    "name": name,
                    "major": int(props["major"]),
                    "minor": int(props["minor"]),
                    "cupy": cp.__version__,
                    "driver_version": int(
                        cp.cuda.runtime.driverGetVersion()
                    ),
                    "runtime_version": int(
                        cp.cuda.runtime.runtimeGetVersion()
                    ),
                },
            })
            ready_sent = True
            ready_send.close()

            while not stop_event.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue

                with conn:
                    deadline = (
                        time.monotonic()
                        + float(request_timeout_seconds)
                    )
                    message = None

                    try:
                        message = receive_message(
                            conn, authkey, deadline,
                            max_message_bytes,
                        )

                        if not isinstance(message, dict):
                            raise TypeError("GPU_REQUEST_NOT_DICT")
                        if message.get("schema") != "HUD_GPU_RPC_V1":
                            raise ValueError("GPU_REQUEST_SCHEMA")

                        kernel_name = message["kernel_name"]
                        if kernel_name not in CUDA_FUNCTIONS:
                            raise ValueError("GPU_KERNEL_NOT_WHITELISTED")

                        blob = message["payload_blob"]
                        fingerprint = hashlib.sha256(blob).hexdigest()
                        if fingerprint != message["input_fingerprint"]:
                            raise ValueError("GPU_INPUT_FINGERPRINT_MISMATCH")

                        encoded_kernel, payload = pickle.loads(blob)
                        if encoded_kernel != kernel_name:
                            raise ValueError("GPU_KERNEL_PAYLOAD_MISMATCH")

                        output, timing = _execute_cuda_request(
                            cp, kernel_name, payload
                        )
                        response = {
                            "schema": "HUD_GPU_RPC_V1",
                            "status": "SUCCESS",
                            "request_id": message["request_id"],
                            "kernel_name": kernel_name,
                            "input_fingerprint": fingerprint,
                            "output": output,
                            "timing": timing,
                        }
                    except Exception as exc:
                        response = {
                            "schema": "HUD_GPU_RPC_V1",
                            "status": "ERROR",
                            "request_id": (
                                message.get("request_id")
                                if isinstance(message, dict) else None
                            ),
                            "kernel_name": (
                                message.get("kernel_name")
                                if isinstance(message, dict) else None
                            ),
                            "input_fingerprint": (
                                message.get("input_fingerprint")
                                if isinstance(message, dict) else None
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }

                    try:
                        send_message(
                            conn, response, authkey, deadline,
                            max_message_bytes,
                        )
                    except (
                        OSError, EOFError, TimeoutError, ValueError
                    ):
                        # Caller có thể đã timeout/đóng kết nối.
                        # Không giữ connection và không nhận request tiếp
                        # trên connection này.
                        pass

    except Exception as exc:
        if not ready_sent:
            try:
                ready_send.send({
                    "status": "START_FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            except Exception:
                pass
        else:
            print(
                f"GPU_OWNER_FATAL {type(exc).__name__}: {exc}",
                flush=True,
            )
    finally:
        if server is not None:
            server.close()
        try:
            ready_send.close()
        except Exception:
            pass


class GpuOwnerService:
    """Quản lý tiến trình GPU chủ và cung cấp thông tin kết nối IPC."""

    def __init__(
        self,
        device_index: int = 0,
        memory_pool_limit_gib: float = 8.0,
        *,
        startup_timeout_seconds: float = 60.0,
        shutdown_timeout_seconds: float = 5.0,
        request_timeout_seconds: float = 180.0,
        max_message_bytes: int = 64 * 1024 * 1024,
    ):
        """Thực thi __init__."""
        self.device_index = device_index
        self.memory_pool_limit_gib = memory_pool_limit_gib
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.max_message_bytes = max_message_bytes

        self.authkey = secrets.token_bytes(32)
        self.process: mp.Process | None = None
        self._address: tuple[str, int] | None = None
        self._stop_event: Any = None
        self.device_info: dict[str, Any] = {}

    def start(self) -> None:
        """Thực thi start."""
        if self.process is not None:
            raise RuntimeError("GPU_SERVICE_ALREADY_STARTED")

        ctx = mp.get_context("spawn")
        receive_ready, send_ready = ctx.Pipe(duplex=False)
        self._stop_event = ctx.Event()

        self.process = ctx.Process(
            target=_gpu_owner_loop,
            args=(
                self.authkey,
                self.device_index,
                self.memory_pool_limit_gib,
                self.request_timeout_seconds,
                self.max_message_bytes,
                send_ready,
                self._stop_event,
            ),
            daemon=False,
        )

        self.process.start()
        send_ready.close()

        try:
            if not receive_ready.poll(self.startup_timeout_seconds):
                raise TimeoutError("GPU_STARTUP_TIMEOUT")

            message = receive_ready.recv()
            if message.get("status") != "READY":
                raise RuntimeError(
                    "GPU_STARTUP_FAILED:"
                    + str(message.get("error", message))
                )
            if not self.process.is_alive():
                raise RuntimeError("GPU_OWNER_EXITED_AFTER_READY")

            self._address = tuple(message["address"])
            self.device_info = dict(message["device"])
        except BaseException:
            self.stop()
            raise
        finally:
            receive_ready.close()

    def get_endpoint(self) -> tuple[str, int, bytes]:
        """Thực thi get_endpoint."""
        if self.process is None or not self.process.is_alive() or self._address is None:
            raise RuntimeError("GPU_SERVICE_NOT_STARTED_OR_DEAD")
        return (self._address[0], int(self._address[1]), self.authkey)

    def stop(self) -> dict[str, Any]:
        """Thực thi stop."""
        process = self.process
        if process is None:
            return {"status": "NOT_RUNNING"}

        forced = False
        if self._stop_event is not None:
            self._stop_event.set()
        process.join(timeout=self.shutdown_timeout_seconds)

        if process.is_alive():
            forced = True
            process.terminate()
            process.join(timeout=2.0)

        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)

        if process.is_alive():
            raise RuntimeError("GPU_OWNER_COULD_NOT_BE_STOPPED")

        exitcode = process.exitcode
        process.close()
        self.process = None
        self._address = None

        return {
            "status": "FORCED_STOP" if forced else "STOPPED",
            "exitcode": exitcode,
        }
