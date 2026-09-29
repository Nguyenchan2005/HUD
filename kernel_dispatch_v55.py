"""Kernel dispatch layer routing execution to CPU numpy or CUDA GPU owner."""

from __future__ import annotations

import hashlib
import pickle
import time
import uuid
from typing import Any
import numpy as np

import kernels_array_v55
from execution_v55 import (
    BackendConsistencyError,
    BackendExecutionError,
    current_runtime,
    selected_backend,
)

ROW_FIELDS = {
    "cheb_ray_equations": ("ol", "dl"),
    "fermat_eval": ("xy", "q1", "targets"),
}


def _merge_outputs(parts: list[Any]) -> Any:
    """Thực thi _merge_outputs."""
    if not parts:
        raise BackendExecutionError("GPU_EMPTY_PARTS")
    if isinstance(parts[0], tuple):
        width = len(parts[0])
        if any(not isinstance(p, tuple) or len(p) != width for p in parts):
            raise BackendConsistencyError("GPU_TUPLE_SCHEMA_MISMATCH")
        return tuple(
            np.concatenate([part[i] for part in parts], axis=0)
            for i in range(width)
        )
    return np.concatenate(parts, axis=0)


def _check_part_shape_and_dtype(kernel_name: str, part: Any, batch_count: int, payload: dict[str, Any]) -> None:
    """Thực thi _check_part_shape_and_dtype."""
    if kernel_name == "cheb_ray_equations":
        if not isinstance(part, np.ndarray):
            raise BackendConsistencyError("CHEB_OUTPUT_NOT_NDARRAY")
        if part.dtype != np.float64:
            raise BackendConsistencyError(f"CHEB_OUTPUT_DTYPE_NOT_FLOAT64:{part.dtype}")
        coeff = np.asarray(payload["coeff"])
        expected_width = max(2, coeff.shape[0] + coeff.shape[1] - 1)
        if part.shape != (batch_count, expected_width):
            raise BackendConsistencyError(
                f"CHEB_OUTPUT_SHAPE_MISMATCH:{part.shape} vs {(batch_count, expected_width)}"
            )
    elif kernel_name == "fermat_eval":
        if not isinstance(part, tuple) or len(part) != 4:
            raise BackendConsistencyError("FERMAT_OUTPUT_NOT_4_TUPLE")
        grad, pts, op, refl = part
        for name, arr, expected_shape in (
            ("gradient", grad, (batch_count, 2)),
            ("points", pts, (batch_count, 3)),
            ("op", op, (batch_count,)),
            ("reflection", refl, (batch_count,)),
        ):
            if not isinstance(arr, np.ndarray):
                raise BackendConsistencyError(f"FERMAT_{name}_NOT_NDARRAY")
            if arr.dtype != np.float64:
                raise BackendConsistencyError(f"FERMAT_{name}_DTYPE_NOT_FLOAT64:{arr.dtype}")
            if arr.shape != expected_shape:
                raise BackendConsistencyError(
                    f"FERMAT_{name}_SHAPE_MISMATCH:{arr.shape} vs {expected_shape}"
                )


def _one_gpu_call(runtime, kernel_name: str, payload: dict[str, Any]) -> Any:
    """Thực thi _one_gpu_call."""
    from gpu_transport_v55 import request

    gpu = runtime.config.get("gpu", {})
    blob = pickle.dumps((kernel_name, payload), protocol=5)
    fingerprint = hashlib.sha256(blob).hexdigest()
    request_id = uuid.uuid4().hex

    message = {
        "schema": "HUD_GPU_RPC_V1",
        "request_id": request_id,
        "kernel_name": kernel_name,
        "input_fingerprint": fingerprint,
        "payload_blob": blob,
    }

    timeout = float(gpu.get("request_timeout_seconds", 180.0))
    max_bytes = int(gpu.get("max_message_mib", 64)) * 1024**2

    started = time.perf_counter()
    try:
        response = request(
            runtime.gpu_endpoint,
            message,
            timeout,
            max_bytes,
        )

        for key, expected in (
            ("schema", "HUD_GPU_RPC_V1"),
            ("request_id", request_id),
            ("kernel_name", kernel_name),
            ("input_fingerprint", fingerprint),
        ):
            if response.get(key) != expected:
                raise BackendConsistencyError(
                    f"GPU_RESPONSE_MISMATCH:{key}"
                )

        if response.get("status") != "SUCCESS":
            raise BackendExecutionError(
                f"GPU_KERNEL_FAILED:{response.get('error_type')}:"
                f"{response.get('error')}"
            )

        client_wall = time.perf_counter() - started
        timing = response.get("timing", {})
        runtime.record(
            "KERNEL_COMPLETED",
            requested_backend="cuda",
            actual_backend="cuda",
            kernel=kernel_name,
            row_count=timing.get("row_count", 0),
            request_id=request_id,
            input_fingerprint=fingerprint,
            client_wall_seconds=client_wall,
            gpu_timing=timing,
        )
        return response["output"]

    except (BackendExecutionError, BackendConsistencyError):
        raise
    except Exception as exc:
        runtime.record(
            "KERNEL_FAILED",
            kernel=kernel_name,
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise BackendExecutionError(
            f"GPU_TRANSPORT_FAILED:{type(exc).__name__}:{exc}"
        ) from exc


def dispatch_kernel(kernel_name: str, payload: dict[str, Any]) -> Any:
    """Điều phối kernel tới CPU hoặc GPU theo chính sách runtime đã chọn."""
    backend = selected_backend(kernel_name)
    if backend == "reference":
        raise BackendExecutionError("REFERENCE_BACKEND_SHOULD_USE_INLINE_PATH")

    if backend == "cpu":
        fn = getattr(kernels_array_v55, kernel_name, None)
        if fn is None:
            raise BackendExecutionError(f"KERNEL_NOT_FOUND_IN_CPU_ARRAY:{kernel_name}")
        return fn(np, **payload)

    if backend == "cuda":
        runtime = current_runtime()
        if runtime is None or runtime.gpu_endpoint is None:
            raise BackendExecutionError(f"CUDA_REQUEST_WITHOUT_ENDPOINT:{kernel_name}")

        if kernel_name not in ROW_FIELDS:
            # Whole kernel call without batching
            return _one_gpu_call(runtime, kernel_name, payload)

        row_fields = ROW_FIELDS[kernel_name]
        first_field = row_fields[0]
        if first_field not in payload:
            raise BackendExecutionError(f"MISSING_ROW_FIELD:{first_field}")

        first_arr = payload[first_field]
        n_rows = len(first_arr)
        for rf in row_fields[1:]:
            if len(payload.get(rf, [])) != n_rows:
                raise ValueError(f"ROW_FIELD_LENGTH_MISMATCH:{rf}")

        if n_rows == 0:
            runtime.record(
                "KERNEL_COMPLETED",
                requested_backend="cuda",
                actual_backend="cpu_empty",
                kernel=kernel_name,
                row_count=0,
                note="EMPTY_INPUT_NO_CUDA_LAUNCH",
            )
            fn = getattr(kernels_array_v55, kernel_name, None)
            if fn is None:
                raise BackendExecutionError(f"KERNEL_NOT_FOUND_IN_CPU_ARRAY:{kernel_name}")
            return fn(np, **payload)

        gpu_cfg = runtime.config.get("gpu", {})
        ray_batch_size = max(1, int(gpu_cfg.get("ray_batch_size", 4096)))

        parts: list[Any] = []
        for start in range(0, n_rows, ray_batch_size):
            end = min(start + ray_batch_size, n_rows)
            batch_count = end - start

            batch_payload = dict(payload)
            for rf in row_fields:
                batch_payload[rf] = payload[rf][start:end]

            part = _one_gpu_call(runtime, kernel_name, batch_payload)
            _check_part_shape_and_dtype(kernel_name, part, batch_count, payload)
            parts.append(part)

        return _merge_outputs(parts)

    raise BackendExecutionError(f"UNRECOGNIZED_BACKEND:{backend}")
