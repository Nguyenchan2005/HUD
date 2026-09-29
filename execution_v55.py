"""Execution management layer: Runtime sessions, process state, and backend policy."""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class BackendExecutionError(Exception):
    """Lỗi khi thực thi kernel hoặc dịch vụ tăng tốc."""
    pass


class BackendConsistencyError(Exception):
    """Lỗi khi kết quả số học của backend không nhất quán với reference."""
    pass


KNOWN_KERNELS = {
    "plane_batch",
    "fixed_design_lstsq",
    "forward_active_fd",
    "point_by_point_ci",
    "cheb_ray_equations",
    "fermat_eval",
}

_CURRENT_RUNTIME: contextvars.ContextVar[ExecutionRuntime | None] = contextvars.ContextVar(
    "_CURRENT_RUNTIME", default=None
)
_IS_WORKER_PROCESS = False


def is_compute_worker() -> bool:
    """Kiểm tra process hiện tại có phải là worker con."""
    return _IS_WORKER_PROCESS


def mark_as_compute_worker(policy: dict[str, Any]) -> None:
    """Đánh dấu process hiện tại là worker và thiết lập runtime nếu có."""
    global _IS_WORKER_PROCESS
    _IS_WORKER_PROCESS = True

    if not isinstance(policy, dict):
        raise ValueError("WORKER_POLICY_MUST_BE_DICT")
    if "backend_plan" not in policy:
        raise ValueError("WORKER_POLICY_MISSING_BACKEND_PLAN")

    exec_cfg = copy.deepcopy(policy.get("execution_config", {}))
    runtime = ExecutionRuntime(
        mode=str(exec_cfg.get("mode", "reference")),
        config=exec_cfg,
        run_dir=Path(policy.get("run_dir", ".")),
        is_worker=True,
        gpu_endpoint=policy.get("gpu_endpoint"),
        implementation_fingerprint=str(policy.get("implementation_fingerprint", "")),
        backend_plan=dict(policy.get("backend_plan", {})),
        session_id=str(policy.get("session_id", "")),
    )
    _CURRENT_RUNTIME.set(runtime)


@dataclass
class ExecutionRuntime:
    """Đối tượng runtime cục bộ quản lý backend, cache và GPU client."""
    mode: str = "reference"  # "reference", "qualification", "accelerated"
    config: dict[str, Any] = field(default_factory=dict)
    run_dir: Path = field(default_factory=lambda: Path("."))
    is_worker: bool = False
    gpu_endpoint: tuple[str, int, bytes] | None = None
    reverse_cache_enabled: bool = True
    implementation_fingerprint: str = ""
    backend_plan: dict[str, str] = field(default_factory=dict)
    session_id: str = ""
    source_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    gpu_device_info: dict[str, Any] = field(default_factory=dict)
    reverse_cache: Any = None
    _gpu_owner: Any = None
    _managed_pools: list[Any] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ray_trace_executor: Any = None
    _ray_trace_max_inflight: int = 1
    _ray_trace_chunk_size: int = 2048
    _ray_trace_min_rays: int = 2048
    _persistent_ray_executor: Any = None
    _persistent_ray_worker_count: int = 1
    _persistent_ray_policy: dict[str, Any] | None = None

    def __post_init__(self):
        """Thực thi __post_init__."""
        from evaluation_cache_v55 import ReverseTraceEvaluationCache
        entries = int(self.config.get("reverse_cache_entries", 2))
        mib = float(self.config.get("reverse_cache_budget_mib", 256))
        self.reverse_cache = ReverseTraceEvaluationCache(max_entries=entries, max_bytes_mib=mib)
        if self.mode == "reference":
            self.reverse_cache_enabled = False

    def selected_backend(self, kernel_name: str) -> str:
        """Thực thi selected_backend."""
        if self.mode == "reference":
            return "reference"

        backend = self.backend_plan.get(kernel_name, "reference")
        if backend not in {"reference", "cpu", "cuda"}:
            raise BackendExecutionError(
                f"INVALID_EFFECTIVE_BACKEND:{kernel_name}:{backend}"
            )

        if backend == "cuda" and self.gpu_endpoint is None:
            raise BackendExecutionError(
                f"CUDA_SELECTED_WITHOUT_OWNER:{kernel_name}"
            )

        return backend

    def record(self, event: str, **details: Any) -> None:
        """Chỉ nhận metadata nhỏ; không ghi payload, authkey hoặc arrays lớn."""
        folder = self.run_dir / "EXECUTION" / self.session_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        row = {
            "event": event,
            "unix_time": time.time(),
            "pid": os.getpid(),
            "worker": self.is_worker,
            "session_id": self.session_id,
            **details,
        }

        path = folder / f"PID_{os.getpid()}.jsonl"
        with self._lock:
            try:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
            except Exception as exc:
                print(
                    f"EXECUTION_TELEMETRY_ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def worker_policy(self) -> dict[str, Any]:
        """Tạo chính sách truyền cho worker con qua IPC."""
        return {
            "execution_config": copy.deepcopy(self.config),
            "backend_plan": copy.deepcopy(self.backend_plan),
            "session_id": self.session_id,
            "run_dir": str(self.run_dir),
            "implementation_fingerprint": self.implementation_fingerprint,
            "gpu_endpoint": self.gpu_endpoint,
        }


def current_runtime() -> ExecutionRuntime | None:
    """Lấy runtime phiên thực thi hiện tại trong ngữ cảnh."""
    return _CURRENT_RUNTIME.get()


def selected_backend(kernel_name: str) -> str:
    """Trả backend được chọn ('reference', 'cpu', 'cuda') cho kernel cụ thể."""
    runtime = current_runtime()
    if runtime is None:
        return "reference"
    return runtime.selected_backend(kernel_name)


def compute_software_fingerprint(source_dir: Path) -> str:
    """Tính SHA256 nhận diện bộ mã nguồn và các thư viện số học."""
    manifest_files = [
        "core.py",
        "pipeline_v55.py",
        "kernels_array_v55.py",
        "kernels_cpu_v55.py",
        "kernels_ci_v55.py",
        "execution_v55.py",
        "execution_workers_v55.py",
        "kernel_dispatch_v55.py",
        "gpu_service_v55.py",
        "gpu_transport_v55.py",
        "ci_reference_v55.py",
        "ci_validation_v55.py",
        "run_qualification_v55.py",
    ]
    file_hashes: list[str] = []
    for fname in sorted(manifest_files):
        fpath = source_dir / fname
        if fpath.is_file():
            payload = fpath.read_bytes()
            h = hashlib.sha256(payload).hexdigest()
            file_hashes.append(f"{fname}:{h}")
        else:
            file_hashes.append(f"{fname}:MISSING")

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    pkg_versions = []
    for mod_name in ("numpy", "scipy", "numba", "threadpoolctl"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "__version__"):
            pkg_versions.append(f"{mod_name}:{mod.__version__}")
        else:
            try:
                imported = __import__(mod_name)
                pkg_versions.append(f"{mod_name}:{getattr(imported, '__version__', 'unknown')}")
            except ImportError:
                pkg_versions.append(f"{mod_name}:NOT_INSTALLED")

    raw = ";".join(file_hashes) + "|" + py_version + "|" + ";".join(pkg_versions)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_execution_config(exec_cfg: dict[str, Any]) -> None:
    """Xác thực tính hợp lệ của cấu hình execution."""
    mode = exec_cfg.get("mode", "reference")
    if mode not in {"reference", "qualification", "accelerated"}:
        raise ValueError(f"INVALID_EXECUTION_MODE:{mode}")

    cpu_workers = exec_cfg.get("cpu_workers", 1)
    if isinstance(cpu_workers, bool) or not isinstance(cpu_workers, int) or cpu_workers < 1:
        raise ValueError("CPU_WORKERS_MUST_BE_INT_GE_1")

    max_inflight = exec_cfg.get("max_inflight_cpu_jobs", 1)
    if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight < 1 or max_inflight > cpu_workers:
        raise ValueError("MAX_INFLIGHT_MUST_BE_INT_BETWEEN_1_AND_CPU_WORKERS")

    for key in ("host_memory_budget_gib", "host_available_reserve_gib", "worker_memory_estimate_gib", "admission_wait_seconds"):
        val = exec_cfg.get(key)
        if val is not None:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"INVALID_CONFIG_VALUE:{key}:{val}")

    if "parallel_ray_trace" in exec_cfg and not isinstance(exec_cfg["parallel_ray_trace"], bool):
        raise ValueError("PARALLEL_RAY_TRACE_MUST_BE_BOOL")
    if "parallel_jacobian" in exec_cfg and not isinstance(
        exec_cfg["parallel_jacobian"], bool
    ):
        raise ValueError("PARALLEL_JACOBIAN_MUST_BE_BOOL")
    if "parallel_step11_candidates" in exec_cfg and not isinstance(
        exec_cfg["parallel_step11_candidates"], bool
    ):
        raise ValueError("PARALLEL_STEP11_CANDIDATES_MUST_BE_BOOL")
    if "step11_candidate_workers" in exec_cfg:
        val = exec_cfg["step11_candidate_workers"]
        if (
            isinstance(val, bool)
            or not isinstance(val, int)
            or val < 1
            or val > cpu_workers
            or val > max_inflight
        ):
            raise ValueError(
                "STEP11_CANDIDATE_WORKERS_MUST_BE_INT_BETWEEN_1_AND_GLOBAL_LIMITS"
            )
    if "step11_worker_memory_estimate_gib" in exec_cfg:
        val = exec_cfg["step11_worker_memory_estimate_gib"]
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val <= 0
        ):
            raise ValueError(
                "STEP11_WORKER_MEMORY_ESTIMATE_GIB_MUST_BE_POSITIVE_FINITE"
            )

    required_accelerated = exec_cfg.get(
        "required_accelerated_kernels", []
    )
    if not isinstance(required_accelerated, list):
        raise ValueError(
            "REQUIRED_ACCELERATED_KERNELS_MUST_BE_LIST"
        )
    for kernel_name in required_accelerated:
        if (
            not isinstance(kernel_name, str)
            or kernel_name not in KNOWN_KERNELS
        ):
            raise ValueError(
                f"UNKNOWN_REQUIRED_ACCELERATED_KERNEL:{kernel_name}"
            )

    checkpoint_level = exec_cfg.get(
        "checkpoint_compresslevel", 1
    )
    if (
        isinstance(checkpoint_level, bool)
        or not isinstance(checkpoint_level, int)
        or not 0 <= checkpoint_level <= 9
    ):
        raise ValueError(
            "CHECKPOINT_COMPRESSLEVEL_MUST_BE_INT_0_TO_9"
        )
    for key in ("ray_trace_min_rays", "ray_trace_chunk_size"):
        if key in exec_cfg:
            val = exec_cfg[key]
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{key.upper()}_MUST_BE_INT_GE_1")

    if "dynamic_ray_chunks" in exec_cfg and not isinstance(exec_cfg["dynamic_ray_chunks"], bool):
        raise ValueError("DYNAMIC_RAY_CHUNKS_MUST_BE_BOOL")
    if "ray_trace_chunks_per_worker" in exec_cfg:
        chunks_per_worker = exec_cfg["ray_trace_chunks_per_worker"]
        if isinstance(chunks_per_worker, bool) or not isinstance(chunks_per_worker, int) or chunks_per_worker < 1:
            raise ValueError("RAY_TRACE_CHUNKS_PER_WORKER_MUST_BE_GE_1")
    if "ray_trace_chunk_min" in exec_cfg:
        chunk_min = exec_cfg["ray_trace_chunk_min"]
        if isinstance(chunk_min, bool) or not isinstance(chunk_min, int) or chunk_min < 1:
            raise ValueError("RAY_TRACE_CHUNK_MIN_MUST_BE_GE_1")
    if "ray_trace_chunk_max" in exec_cfg:
        chunk_max = exec_cfg["ray_trace_chunk_max"]
        lower = int(exec_cfg.get("ray_trace_chunk_min", 1))
        if isinstance(chunk_max, bool) or not isinstance(chunk_max, int) or chunk_max < lower:
            raise ValueError("RAY_TRACE_CHUNK_MAX_MUST_BE_GE_CHUNK_MIN")

    gpu_cfg = exec_cfg.get("gpu", {})
    if not isinstance(gpu_cfg, dict):
        raise ValueError("GPU_CONFIG_MUST_BE_DICT")

    require_cuda = gpu_cfg.get(
        "require_cuda_kernels",
        [],
    )
    if not isinstance(require_cuda, list):
        raise ValueError(
            "GPU_REQUIRE_CUDA_KERNELS_MUST_BE_LIST"
        )
    for kernel_name in require_cuda:
        if (
            not isinstance(kernel_name, str)
            or kernel_name not in KNOWN_KERNELS
        ):
            raise ValueError(
                f"UNKNOWN_REQUIRED_CUDA_KERNEL:{kernel_name}"
            )

    owner_processes = gpu_cfg.get("owner_processes", 1)
    if owner_processes != 1:
        raise ValueError("GPU_OWNER_PROCESSES_MUST_BE_1")

    failure_policy = gpu_cfg.get("failure_policy", "raise")
    if failure_policy != "raise":
        raise ValueError("GPU_FAILURE_POLICY_MUST_BE_RAISE")

    min_fermat_rows = gpu_cfg.get("min_fermat_rows", 256)
    if (
        isinstance(min_fermat_rows, bool)
        or not isinstance(min_fermat_rows, int)
        or min_fermat_rows < 1
    ):
        raise ValueError(
            "GPU_MIN_FERMAT_ROWS_MUST_BE_INT_GE_1"
        )


def load_qualification_profile(
    profile_path: Path,
    expected_fingerprint: str,
) -> dict[str, str]:
    """Tải và xác thực hồ sơ qualification; trả về effective backend plan."""
    if not profile_path.is_file():
        raise FileNotFoundError(f"QUALIFICATION_FILE_NOT_FOUND:{profile_path}")

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if data.get("schema") != "HUD_BACKEND_QUALIFICATION_V1":
        raise ValueError("INVALID_QUALIFICATION_SCHEMA")

    # Đã bỏ vĩnh viễn kiểm tra software_fingerprint và test_report_sha256 theo yêu cầu người dùng.
    kernel_results = data.get("kernel_results", {})
    backend_plan: dict[str, str] = {}

    for k in KNOWN_KERNELS:
        if k in kernel_results:
            info = kernel_results[k]
            backend = info.get("backend")
            passed = bool(info.get("passed", False))
            failed = int(info.get("cases_failed", 0))
            cases_passed = int(info.get("cases_passed", 0))
            if backend not in ("cpu", "cuda"):
                raise ValueError(f"INVALID_QUALIFIED_BACKEND:{k}:{backend}")
            if passed and failed == 0 and cases_passed > 0:
                backend_plan[k] = backend
            else:
                backend_plan[k] = "reference"
        else:
            backend_plan[k] = "reference"

    return backend_plan


def get_hud_tree_rss_gib() -> float:
    """Đo tổng RSS (GiB) của cây tiến trình HUD thuộc process hiện tại."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        proc = psutil.Process()
        total_rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                if child.is_running():
                    total_rss += child.memory_info().rss
            except Exception:
                pass
        return float(total_rss / (1024 ** 3))
    except Exception:
        return 0.0


def get_memory_stats_gib() -> dict[str, float]:
    """Đo RAM khả dụng và process RSS."""
    stats = {"process_rss_gib": 0.0, "system_available_gib": 0.0}
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            stats["system_available_gib"] = float(mem.available / (1024 ** 3))
            stats["process_rss_gib"] = get_hud_tree_rss_gib()
        except Exception:
            pass
    return stats


@contextmanager
def execution_session(
    config: dict[str, Any],
    run_dir: Path | str,
    *,
    purpose: str = "production",
    qualification_plan: dict[str, str] | None = None,
) -> Iterator[ExecutionRuntime]:
    """Mở một phiên thực thi có quản lý tài nguyên, GPU service và cache."""
    if is_compute_worker():
        raise RuntimeError("COMPUTE_WORKER_CANNOT_OPEN_EXECUTION_SESSION")

    existing = current_runtime()
    if existing is not None:
        raise RuntimeError("NESTED_EXECUTION_SESSION_NOT_ALLOWED")

    exec_cfg = copy.deepcopy(config.get("execution", {}))
    validate_execution_config(exec_cfg)

    mode = str(exec_cfg.get("mode", "reference"))
    if purpose == "production" and mode == "qualification":
        raise ValueError("PRODUCTION_CANNOT_USE_QUALIFICATION_MODE")

    run_path = Path(run_dir)
    session_id = uuid.uuid4().hex
    source_dir = Path(__file__).resolve().parent
    fingerprint = compute_software_fingerprint(source_dir)

    backend_plan: dict[str, str] = {}
    if mode == "reference":
        backend_plan = {k: "reference" for k in KNOWN_KERNELS}
    elif purpose == "qualification":
        if qualification_plan is not None:
            backend_plan = dict(qualification_plan)
        else:
            backend_plan = {k: "reference" for k in KNOWN_KERNELS}
    elif mode == "accelerated":
        qfile_str = exec_cfg.get("qualification_file")
        if not qfile_str:
            raise ValueError("ACCELERATED_MODE_REQUIRES_QUALIFICATION_FILE")
        qfile = Path(qfile_str)
        if not qfile.is_absolute():
            qfile = source_dir / qfile
        backend_plan = load_qualification_profile(qfile, fingerprint)

    if (purpose == "production" and mode == "accelerated"
            and backend_plan.get("point_by_point_ci") != "cpu"):
        raise BackendExecutionError("CI_CPU_BACKEND_NOT_QUALIFIED")

    if purpose == "production" and mode == "accelerated":
        required_accelerated = exec_cfg.get(
            "required_accelerated_kernels",
            ["point_by_point_ci"],
        )

        for kernel_name in required_accelerated:
            backend = backend_plan.get(
                kernel_name, "reference"
            )
            if backend == "reference":
                raise BackendExecutionError(
                    "REQUIRED_ACCELERATED_KERNEL_NOT_QUALIFIED:"
                    f"{kernel_name}"
                )

    required_cuda = exec_cfg.get(
        "gpu",
        {},
    ).get(
        "require_cuda_kernels",
        [],
    )

    if (
        purpose == "production"
        and mode == "accelerated"
    ):
        for kernel_name in required_cuda:
            backend = backend_plan.get(
                kernel_name,
                "reference",
            )

            if backend != "cuda":
                raise BackendExecutionError(
                    "REQUIRED_CUDA_KERNEL_NOT_QUALIFIED:"
                    f"{kernel_name}:"
                    f"{backend}"
                )

    ci_preflight = None
    if backend_plan.get("point_by_point_ci") == "cpu":
        ci_mode = config.get("ci_construction", {}).get("mode", "LEGACY_NEAREST")
        if ci_mode == "LEGACY_NEAREST":
            from kernels_ci_v55 import warmup_ci_numba
            ci_preflight = warmup_ci_numba()
        elif ci_mode == "SURFACE_COMPATIBLE_FRONTIER_V1":
            from kernels_ci_v55 import warmup_ci_compatible_numba
            ci_preflight = warmup_ci_compatible_numba(
                edge_height_residual_cap_mm=math.inf,
                construction_policy="SURFACE_COMPATIBLE_FRONTIER_V1",
            )
        elif ci_mode == "SURFACE_COMPATIBLE_FRONTIER_V2":
            from kernels_ci_v55 import warmup_ci_compatible_numba
            edge_cap = float(
                config.get("ci_construction", {})
                      .get("compatible_frontier", {})
                      .get("edge_height_residual_cap_mm", 0.05)
            )
            ci_preflight = warmup_ci_compatible_numba(
                edge_height_residual_cap_mm=edge_cap,
                construction_policy="SURFACE_COMPATIBLE_FRONTIER_V2",
            )
        else:
            from kernels_ci_v55 import warmup_ci_numba
            ci_preflight = warmup_ci_numba()

    runtime = ExecutionRuntime(
        mode=mode,
        config=exec_cfg,
        run_dir=run_path,
        is_worker=False,
        implementation_fingerprint=fingerprint,
        backend_plan=backend_plan,
        session_id=session_id,
        source_dir=source_dir,
    )

    token = _CURRENT_RUNTIME.set(runtime)

    owner = None
    gpu_cfg = exec_cfg.get("gpu", {})
    gpu_needed = ("cuda" in backend_plan.values()) and bool(gpu_cfg.get("enabled", False))

    if gpu_needed:
        try:
            from gpu_service_v55 import GpuOwnerService
            owner = GpuOwnerService(
                device_index=int(gpu_cfg.get("device", 0)),
                memory_pool_limit_gib=float(gpu_cfg.get("memory_pool_limit_gib", 8.0)),
                startup_timeout_seconds=float(gpu_cfg.get("startup_timeout_seconds", 60.0)),
                shutdown_timeout_seconds=float(gpu_cfg.get("shutdown_timeout_seconds", 5.0)),
                request_timeout_seconds=float(gpu_cfg.get("request_timeout_seconds", 180.0)),
                max_message_bytes=int(gpu_cfg.get("max_message_mib", 64)) * 1024**2,
            )
            owner.start()
            runtime._gpu_owner = owner
            runtime.gpu_endpoint = owner.get_endpoint()
            runtime.gpu_device_info = dict(owner.device_info)
            runtime.record(
                "GPU_READY",
                device=runtime.gpu_device_info,
            )
        except Exception as exc:
            runtime.record(
                "GPU_START_FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            _CURRENT_RUNTIME.reset(token)
            raise BackendExecutionError(
                f"GPU_STARTUP_FAILED:{type(exc).__name__}:{exc}"
            ) from exc

    manifest = {
        "schema": "HUD_EXECUTION_MANIFEST_V1",
        "session_id": session_id,
        "mode": mode,
        "purpose": purpose,
        "effective_backend_plan": backend_plan,
        "ci_preflight": ci_preflight,
        "software_fingerprint": fingerprint,
        "gpu_device_info": runtime.gpu_device_info,
        "gpu_enabled": gpu_needed,
        "config_snapshot": {
            "cpu_workers": exec_cfg.get("cpu_workers"),
            "max_inflight_cpu_jobs": exec_cfg.get("max_inflight_cpu_jobs"),
            "required_accelerated_kernels": exec_cfg.get(
                "required_accelerated_kernels", []
            ),
            "required_cuda_kernels": gpu_cfg.get(
                "require_cuda_kernels",
                [],
            ),
            "checkpoint_compresslevel": exec_cfg.get(
                "checkpoint_compresslevel", 1
            ),
            "host_memory_budget_gib": exec_cfg.get("host_memory_budget_gib"),
            "host_available_reserve_gib": exec_cfg.get("host_available_reserve_gib"),
            "worker_memory_estimate_gib": exec_cfg.get("worker_memory_estimate_gib"),
            "parallel_step11_candidates": exec_cfg.get("parallel_step11_candidates"),
            "step11_candidate_workers": exec_cfg.get("step11_candidate_workers"),
            "step11_worker_memory_estimate_gib": exec_cfg.get("step11_worker_memory_estimate_gib"),
            "render_backend": "INLINE",
        },
    }

    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    try:
        (run_path / "EXECUTION_MANIFEST.json").write_text(manifest_json, encoding="utf-8")
        session_folder = run_path / "EXECUTION" / session_id
        session_folder.mkdir(parents=True, exist_ok=True)
        (session_folder / "EXECUTION_MANIFEST.json").write_text(manifest_json, encoding="utf-8")
    except Exception:
        pass

    runtime.record("SESSION_STARTED", mode=mode, purpose=purpose)
    if ci_preflight is not None:
        runtime.record("CI_JIT_READY", **ci_preflight)

    try:
        yield runtime
    finally:
        # Close any open pools
        for pool in list(runtime._managed_pools):
            try:
                pool.shutdown(wait=True)
            except Exception:
                pass
        runtime._managed_pools.clear()

        # Stop GPU owner
        if owner is not None:
            try:
                stop_res = owner.stop()
                runtime.record("GPU_STOPPED", **stop_res)
            except Exception as exc:
                runtime.record("GPU_STOP_FAILED", error=str(exc))

        runtime.record("SESSION_CLOSED")
        _CURRENT_RUNTIME.reset(token)


def compute_dynamic_ray_chunk_size(
    ray_count: int,
    worker_count: int,
    exec_cfg: dict[str, Any],
) -> int:
    """Chia ray đủ nhỏ để worker luôn có hàng đợi, nhưng không tạo quá nhiều IPC."""
    if ray_count < 1:
        return 1

    if not bool(
        exec_cfg.get(
            "dynamic_ray_chunks",
            False,
        )
    ):
        return int(
            exec_cfg.get(
                "ray_trace_chunk_size",
                2048,
            )
        )

    workers = max(
        1,
        int(worker_count),
    )

    chunks_per_worker = max(
        1,
        int(
            exec_cfg.get(
                "ray_trace_chunks_per_worker",
                3,
            )
        ),
    )

    lower = max(
        1,
        int(
            exec_cfg.get(
                "ray_trace_chunk_min",
                256,
            )
        ),
    )

    upper = max(
        lower,
        int(
            exec_cfg.get(
                "ray_trace_chunk_max",
                2048,
            )
        ),
    )

    target_jobs = workers * chunks_per_worker
    raw = int(
        math.ceil(
            ray_count
            / max(
                target_jobs,
                1,
            )
        )
    )

    return max(
        lower,
        min(
            upper,
            raw,
        ),
    )


def _admitted_worker_count(
    runtime: ExecutionRuntime,
    purpose: str,
    *,
    worker_limit: int | None = None,
    worker_memory_estimate_gib: float | None = None,
) -> int:
    """Tính worker chỉ từ ngân sách RAM khai báo trong config."""
    cfg = runtime.config
    cpu_workers = int(cfg.get("cpu_workers", 4))
    max_inflight = int(cfg.get("max_inflight_cpu_jobs", 4))
    if worker_limit is not None:
        cpu_workers = min(cpu_workers, int(worker_limit))
        max_inflight = min(max_inflight, int(worker_limit))

    host_budget = float(cfg.get("host_memory_budget_gib", 10.0))
    estimate = float(
        worker_memory_estimate_gib
        if worker_memory_estimate_gib is not None
        else cfg.get("worker_memory_estimate_gib", 2.0)
    )

    # Admission tĩnh: không đọc RAM trống hay RSS của tiến trình lúc chạy.
    # host_memory_budget_gib là toàn bộ RAM người dùng cấp cho worker;
    # không trừ thêm reserve hay bất kỳ lượng RAM runtime nào.
    budget_for_workers = max(0.0, host_budget)

    limit_by_ram = int(
        math.floor(
            budget_for_workers
            / max(
                estimate,
                0.1,
            )
        )
    )

    if limit_by_ram <= 0:
        runtime.record(
            "MEMORY_ADMISSION_BLOCKED",
            purpose=purpose,
            host_budget=host_budget,
            worker_memory_estimate=estimate,
            reason="CONFIGURED_RAM_BUDGET_INSUFFICIENT",
        )
        print(
            f"[{purpose}] CONFIGURED_RESOURCE_BUDGET_INSUFFICIENT: Configured worker RAM budget ({budget_for_workers:.2f} GiB) "
            f"< worker estimate ({estimate:.2f} GiB). Falling back to verified serial execution.",
            flush=True,
        )
        return 0

    effective_workers = max(
        1,
        min(
            cpu_workers,
            max_inflight,
            limit_by_ram,
        ),
    )
    return effective_workers


def get_or_create_persistent_ray_pool(
    purpose: str,
) -> tuple[Any, int, dict[str, Any]]:
    """Tạo hoặc lấy pool persistent cho generic ray jobs trong suốt session."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from execution_workers_v55 import worker_init

    runtime = current_runtime()
    if runtime is None:
        raise RuntimeError("PERSISTENT_RAY_POOL_REQUIRES_ACTIVE_RUNTIME")
    if runtime.is_worker or is_compute_worker():
        raise RuntimeError("WORKER_CANNOT_OPEN_PERSISTENT_RAY_POOL")

    if runtime._persistent_ray_executor is not None:
        return (
            runtime._persistent_ray_executor,
            runtime._persistent_ray_worker_count,
            runtime._persistent_ray_policy,
        )

    workers = _admitted_worker_count(
        runtime,
        purpose,
    )
    if workers <= 0:
        return None, 1, runtime.worker_policy()

    policy = runtime.worker_policy()
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(
            policy,
            None,
        ),
    )

    runtime._persistent_ray_executor = executor
    runtime._persistent_ray_worker_count = workers
    runtime._persistent_ray_policy = policy
    runtime._managed_pools.append(
        executor
    )

    return executor, workers, policy


@contextmanager
def managed_compute_pool(
    base_context: Any,
    purpose: str = "general",
    *,
    worker_limit: int | None = None,
    worker_memory_estimate_gib: float | None = None,
) -> Iterator[tuple[Any, int, dict[str, Any]]]:
    """Mở và quản lý ProcessPoolExecutor kiểm soát ngân sách RAM và không tạo pool lồng."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    runtime = current_runtime()
    if runtime is None:
        raise RuntimeError("MANAGED_COMPUTE_POOL_REQUIRES_ACTIVE_RUNTIME")
    if runtime.is_worker or is_compute_worker():
        raise RuntimeError("WORKER_CANNOT_OPEN_COMPUTE_POOL")

    effective_workers = _admitted_worker_count(
        runtime,
        purpose,
        worker_limit=worker_limit,
        worker_memory_estimate_gib=worker_memory_estimate_gib,
    )
    if effective_workers <= 0:
        yield None, 1, runtime.worker_policy()
        return

    effective_max_inflight = effective_workers

    from execution_workers_v55 import worker_init

    policy = runtime.worker_policy()
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(policy, base_context),
    )
    runtime._managed_pools.append(executor)

    try:
        yield executor, effective_max_inflight, policy
    finally:
        if executor in runtime._managed_pools:
            runtime._managed_pools.remove(executor)
        executor.shutdown(wait=True)


@contextmanager
def ray_trace_acceleration(
    purpose: str = "RAY_TRACE",
    base_context: Any = None,
) -> Iterator[None]:
    """Context manager quản lý worker pool cho parallel ray trace."""
    runtime = current_runtime()
    if runtime is None or runtime.is_worker or is_compute_worker():
        yield
        return

    exec_cfg = runtime.config
    if not bool(exec_cfg.get("parallel_ray_trace", False)):
        yield
        return

    if runtime._ray_trace_executor is not None:
        # Đã có executor hoạt động (ví dụ trong Jacobian context)
        yield
        return

    chunk_size = int(exec_cfg.get("ray_trace_chunk_size", 2048))
    min_rays = int(exec_cfg.get("ray_trace_min_rays", 512))
    use_persistent = bool(exec_cfg.get("persistent_ray_pool", False))

    if use_persistent:
        executor, workers, policy = get_or_create_persistent_ray_pool(purpose)
        runtime._ray_trace_executor = executor
        runtime._ray_trace_max_inflight = workers
        runtime._ray_trace_chunk_size = chunk_size
        runtime._ray_trace_min_rays = min_rays
        runtime.record(
            (
                "RAY_TRACE_ACCELERATION_ACTIVE"
                if executor is not None
                else "RAY_TRACE_ACCELERATION_SERIAL_FALLBACK"
            ),
            purpose=purpose,
            parallel_requested=True,
            persistent_pool=True,
            executor_active=(
                executor is not None
            ),
            effective_max_inflight=int(
                workers
            ),
            chunk_size=int(
                chunk_size
            ),
            min_rays=int(
                min_rays
            ),
        )
        try:
            yield
        finally:
            runtime._ray_trace_executor = None
            runtime._ray_trace_max_inflight = 1
            runtime._ray_trace_chunk_size = 2048
            runtime._ray_trace_min_rays = 2048
    else:
        with managed_compute_pool(
            base_context,
            purpose=purpose,
        ) as (
            executor,
            effective_max_inflight,
            policy,
        ):
            runtime._ray_trace_executor = executor
            runtime._ray_trace_max_inflight = effective_max_inflight
            runtime._ray_trace_chunk_size = chunk_size
            runtime._ray_trace_min_rays = min_rays
            runtime.record(
                (
                    "RAY_TRACE_ACCELERATION_ACTIVE"
                    if executor is not None
                    else "RAY_TRACE_ACCELERATION_SERIAL_FALLBACK"
                ),
                purpose=purpose,
                parallel_requested=True,
                persistent_pool=False,
                executor_active=(
                    executor is not None
                ),
                effective_max_inflight=int(
                    effective_max_inflight
                ),
                chunk_size=int(
                    chunk_size
                ),
                min_rays=int(
                    min_rays
                ),
            )
            try:
                yield
            finally:
                runtime._ray_trace_executor = None
                runtime._ray_trace_max_inflight = 1
                runtime._ray_trace_chunk_size = 2048
                runtime._ray_trace_min_rays = 2048


@contextmanager
def timed_operation(
    runtime: ExecutionRuntime | None,
    event: str,
    **metadata: Any,
) -> Iterator[dict[str, Any]]:
    """Đo thời gian wall-clock và biến thiên bộ nhớ của một tác vụ số học/quang học."""
    t0 = time.perf_counter()
    mem_before = get_memory_stats_gib()
    if runtime is not None:
        runtime.record(
            f"{event}_STARTED",
            rss_before_gib=mem_before["process_rss_gib"],
            available_ram_gib=mem_before["system_available_gib"],
            **metadata,
        )
    res_ctx: dict[str, Any] = {}
    try:
        yield res_ctx
    finally:
        t1 = time.perf_counter()
        mem_after = get_memory_stats_gib()
        wall = t1 - t0
        res_ctx["wall_seconds"] = wall
        if runtime is not None:
            runtime.record(
                f"{event}_FINISHED",
                wall_seconds=wall,
                rss_after_gib=mem_after["process_rss_gib"],
                available_ram_gib=mem_after["system_available_gib"],
                **metadata,
            )

