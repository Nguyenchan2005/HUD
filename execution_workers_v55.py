"""Worker tasks and bounded parallel execution for Jacobian columns and Multistart seeds."""

from __future__ import annotations

import copy
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import Any, Callable

_BASE_CONTEXT = None
_BLAS_LIMITER = None


def worker_init(policy: dict[str, Any], base_context: Any) -> None:
    """Khởi tạo process worker, nạp chính sách và giới hạn BLAS threads."""
    global _BASE_CONTEXT, _BLAS_LIMITER

    import copy
    from threadpoolctl import threadpool_limits
    from execution_v55 import (
        current_runtime,
        mark_as_compute_worker,
    )

    mark_as_compute_worker(copy.deepcopy(policy))
    _BASE_CONTEXT = base_context

    blas_threads = int(policy["execution_config"].get("blas_threads_per_worker", 1))
    _BLAS_LIMITER = threadpool_limits(limits=blas_threads)

    runtime = current_runtime()
    if runtime is not None:
        runtime.record(
            "WORKER_INITIALIZED",
            backend_plan=runtime.backend_plan,
        )


def ordered_bounded_map(
    executor,
    function: Callable[[Any], Any],
    tasks: list[Any],
    max_inflight: int,
    on_completed: Callable[[int, Any], None] | None = None,
    on_tick: Callable[[], None] | None = None,
    before_submit: Callable[[int, Any, int], None] | None = None,
    on_submitted: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """Thực thi song song giữ nguyên thứ tự kết quả và giới hạn số task cùng lúc."""
    if max_inflight < 1:
        raise ValueError("MAX_INFLIGHT_MUST_BE_POSITIVE")

    results = [None] * len(tasks)
    pending = {}
    next_index = 0

    def submit_available():
        """Thực thi submit_available."""
        nonlocal next_index
        while next_index < len(tasks) and len(pending) < max_inflight:
            idx = next_index
            task = tasks[idx]
            if before_submit is not None:
                try:
                    before_submit(idx, task, len(pending))
                except Exception as cb_exc:
                    print(f"BEFORE_SUBMIT_CALLBACK_ERROR: {cb_exc}", flush=True)

            future = executor.submit(function, task)
            pending[future] = idx
            next_index += 1

            if on_submitted is not None:
                try:
                    on_submitted(idx, task)
                except Exception as cb_exc:
                    print(f"ON_SUBMITTED_CALLBACK_ERROR: {cb_exc}", flush=True)

    try:
        submit_available()
        while pending:
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if on_tick is not None:
                try:
                    on_tick()
                except Exception as cb_exc:
                    print(f"ON_TICK_CALLBACK_ERROR: {cb_exc}", flush=True)

            for future in sorted(done, key=lambda item: pending[item]):
                index = pending.pop(future)
                result = future.result()
                results[index] = result

                if on_completed is not None:
                    try:
                        on_completed(index, result)
                    except Exception as cb_exc:
                        print(f"ON_COMPLETED_CALLBACK_ERROR: {cb_exc}", flush=True)

            submit_available()
    except BaseException:
        for future in pending:
            future.cancel()
        raise

    return results


def seed_precheck_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Thực hiện precheck một seed planar độc lập."""
    import copy
    import os
    from pathlib import Path
    import numpy as np

    from execution_v55 import current_runtime
    from pipeline_v55 import evaluate_planar_seed_job

    if _BASE_CONTEXT is None:
        raise RuntimeError("PRECHECK_WORKER_NOT_INITIALIZED")

    c2 = np.asarray(job["c2"], dtype=float)
    cd = np.asarray(job["cd"], dtype=float)
    if c2.shape != (3,) or cd.shape != (3,):
        raise ValueError("PRECHECK_COORDINATES_MUST_BE_VECTOR3")

    ctx = copy.deepcopy(_BASE_CONTEXT)
    ctx.run_dir = Path(job["work_dir"])

    runtime = current_runtime()
    if runtime is not None:
        runtime.record(
            "JOB_STARTED",
            kind="PRECHECK",
            job_id=job["job_id"],
            candidate=job["candidate_index"],
        )

    record = evaluate_planar_seed_job(ctx, job)

    if runtime is not None:
        runtime.record(
            "JOB_FINISHED",
            kind="PRECHECK",
            job_id=job["job_id"],
            candidate=job["candidate_index"],
        )

    return {
        "job_id": job["job_id"],
        "candidate_index": int(job["candidate_index"]),
        "pid": os.getpid(),
        "record": record,
        "work_dir": str(ctx.run_dir),
    }


def jacobian_column_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Tính toán sai phân của một cột Jacobian trong tiến trình worker."""
    import numpy as np
    from pipeline_v55 import _objective_eval

    if _BASE_CONTEXT is None:
        raise RuntimeError("JACOBIAN_WORKER_NOT_INITIALIZED")

    which = job["which"]
    if which not in ("M1", "M2"):
        raise ValueError("JACOBIAN_SURFACE_INVALID")

    m1 = job["m1"].copy()
    m2 = job["m2"].copy()
    surface = m1 if which == "M1" else m2

    index = int(job["coeff_index"])
    if not 0 <= index < len(surface.coeff):
        raise IndexError("JACOBIAN_COEFFICIENT_INDEX")

    surface.coeff[index] += float(job["fd_sag"])
    residual, _ = _objective_eval(_BASE_CONTEXT, m1, m2)

    return {
        "column": int(job["column"]),
        "state_id": job["state_id"],
        "residual": np.asarray(residual, dtype=float),
    }


def surface_parameter_column_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Tinh mot cot Jacobian curvature/conic tren worker."""
    import numpy as np

    from pipeline_v55 import (
        _apply_surface_parameter_vector,
        _bounded_fd_trial,
        _objective_eval,
    )

    if _BASE_CONTEXT is None:
        raise RuntimeError(
            "SURFACE_PARAMETER_WORKER_NOT_INITIALIZED"
        )

    column = int(job["column"])
    u = np.asarray(
        job["u"], dtype=float
    )

    trial, actual_fd = _bounded_fd_trial(
        u,
        column,
        float(job["fd"]),
    )

    m1, m2 = _apply_surface_parameter_vector(
        job["base"],
        trial,
        job["cfg"],
    )

    residual, _ = _objective_eval(
        _BASE_CONTEXT,
        m1,
        m2,
    )

    return {
        "column": column,
        "actual_fd": float(actual_fd),
        "residual": np.asarray(
            residual,
            dtype=float,
        ),
    }


def geometry_column_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Tinh mot cot Jacobian pose forward tren worker."""
    import numpy as np

    from pipeline_v55 import (
        _apply_geometry_vector,
        _geometry_forward_eval,
    )

    if _BASE_CONTEXT is None:
        raise RuntimeError(
            "GEOMETRY_WORKER_NOT_INITIALIZED"
        )

    column = int(job["column"])
    u = np.asarray(
        job["u"], dtype=float
    )

    trial_u = u.copy()
    trial_u[column] = min(
        1.0,
        trial_u[column]
        + float(job["fd"]),
    )

    actual_fd = float(
        trial_u[column] - u[column]
    )

    m1, m2, display = (
        _apply_geometry_vector(
            job["base"],
            job["specs"],
            trial_u,
        )
    )

    residual, _ = _geometry_forward_eval(
        _BASE_CONTEXT,
        m1,
        m2,
        display,
        False,
    )

    return {
        "column": column,
        "actual_fd": actual_fd,
        "residual": np.asarray(
            residual,
            dtype=float,
        ),
    }


def poly_surface_intersect_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Thực thi giao tia PolySurface trên worker con độc lập."""
    from core import PolySurface

    surface = PolySurface.from_dict(job["surface"])
    t_min = float(job.get("t_min", 0.0))
    result = surface.intersect(
        job["origins"],
        job["directions"],
        finite=bool(job["finite"]),
        t_min=t_min,
    )
    return {
        "chunk_index": int(job["chunk_index"]),
        "start": int(job["start"]),
        "end": int(job["end"]),
        "result": result,
    }


def cheb_visor_intersect_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Thực thi giao tia ChebVisor trên worker con độc lập."""
    from core import ChebVisor

    visor = ChebVisor.from_dict(
        job["visor"]
    )

    result = visor.intersect(
        job["origins"],
        job["directions"],
        finite=bool(
            job["finite"]
        ),
        t_min=float(
            job["t_min"]
        ),
    )

    return {
        "chunk_index":
            int(job["chunk_index"]),
        "start":
            int(job["start"]),
        "end":
            int(job["end"]),
        "result":
            result,
    }


def signed_rho_branch_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Đánh giá một nhánh rho độc lập của STEP16 trên worker."""
    import copy

    from pipeline_v55 import (
        _evaluate_step16_rho_branch,
    )

    if _BASE_CONTEXT is None:
        raise RuntimeError(
            "SIGNED_RHO_BASE_CONTEXT_NOT_INITIALIZED"
        )

    branch = copy.deepcopy(
        _BASE_CONTEXT
    )

    return _evaluate_step16_rho_branch(
        branch,
        float(job["rho"]),
    )


def dlsq_trial_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Đánh giá một bước thử DLSQ độc lập trên worker."""
    import numpy as np

    from pipeline_v55 import (
        _objective_eval,
    )

    if _BASE_CONTEXT is None:
        raise RuntimeError(
            "DLSQ_TRIAL_BASE_CONTEXT_NOT_INITIALIZED"
        )

    m1 = job["m1"].copy()
    m2 = job["m2"].copy()

    variable_indices = (
        job["variable_indices"]
    )

    step = np.asarray(
        job["step"],
        dtype=float,
    )

    for value, (
        which,
        index,
    ) in zip(
        step,
        variable_indices,
    ):
        surface = (
            m1
            if which == "M1"
            else m2
        )

        surface.coeff[
            int(index)
        ] += float(value)

    residual, record = _objective_eval(
        _BASE_CONTEXT,
        m1,
        m2,
        use_reverse_cache=False,
    )

    return {
        "trial_index":
            int(job["trial_index"]),
        "step":
            step,
        "residual":
            np.asarray(
                residual,
                dtype=float,
            ),
        "record":
            record,
    }


def step11_candidate_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Đánh giá một coordinate candidate độc lập của STEP11 trong worker process."""
    import os
    import numpy as np
    from execution_v55 import current_runtime
    from pipeline_v55 import (
        _step11_archive_candidate_snapshot,
        evaluate_step11_candidate_job,
    )

    if _BASE_CONTEXT is None:
        raise RuntimeError("STEP11_WORKER_BASE_CONTEXT_NOT_INITIALIZED")

    if job.get("schema") != "HUD_FAN_V5_5_STEP11_CANDIDATE_JOB_V1":
        raise ValueError(f"STEP11_INVALID_JOB_SCHEMA:{job.get('schema')}")

    candidate_id = job.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("STEP11_JOB_CANDIDATE_ID_INVALID")

    job_index = job.get("job_index")
    if isinstance(job_index, bool) or not isinstance(job_index, int) or job_index < 0:
        raise ValueError("STEP11_JOB_INDEX_INVALID")

    candidate_number = job.get("candidate_number")
    if isinstance(candidate_number, bool) or not isinstance(candidate_number, int) or candidate_number < 1:
        raise ValueError("STEP11_JOB_CANDIDATE_NUMBER_INVALID")

    cycle = job.get("cycle")
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 1:
        raise ValueError("STEP11_JOB_CYCLE_INVALID")

    search_vector = np.asarray(job.get("search_vector"), dtype=float)
    if search_vector.ndim != 1:
        raise ValueError("STEP11_JOB_SEARCH_VECTOR_NOT_1D")
    if not np.all(np.isfinite(search_vector)):
        raise ValueError("STEP11_JOB_SEARCH_VECTOR_NONFINITE")
    if np.any(np.abs(search_vector) > 1.0 + 1e-12):
        raise ValueError("STEP11_JOB_SEARCH_VECTOR_OUT_OF_BOUNDS")

    runtime = current_runtime()
    if runtime is not None:
        runtime.record(
            "JOB_STARTED",
            kind="STEP11_CANDIDATE",
            candidate_id=candidate_id,
            cycle=cycle,
        )

    try:
        record, payload, last_phase = evaluate_step11_candidate_job(
            _BASE_CONTEXT,
            job,
        )
    except Exception as exc:
        raise RuntimeError(
            f"STEP11_WORKER_CANDIDATE_FAILED:"
            f"candidate_id={candidate_id}:"
            f"cycle={cycle}:"
            f"move={job.get('move')}:"
            f"pid={os.getpid()}:"
            f"exc={type(exc).__name__}:{exc}"
        ) from exc

    snapshot_summary = (
        _step11_archive_candidate_snapshot(
            record,
            last_phase,
            job,
        )
    )
    if snapshot_summary is not None:
        record["_snapshot_summary"] = (
            snapshot_summary
        )

    if runtime is not None:
        runtime.record(
            "JOB_COMPLETED",
            kind="STEP11_CANDIDATE",
            candidate_id=candidate_id,
            cycle=cycle,
        )

    return {
        "schema": "HUD_FAN_V5_5_STEP11_CANDIDATE_REPLY_V1",
        "job_index": int(job_index),
        "candidate_number": int(candidate_number),
        "candidate_id": str(candidate_id),
        "cycle": int(cycle),
        "pid": os.getpid(),
        "record": record,
        "payload": payload,
        "last_phase": last_phase,
    }


def step11_restoration_probe_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate đúng một +delta hoặc -delta
    của STEP11 restoration Jacobian.
    """

    import os

    from execution_v55 import (
        current_runtime,
    )

    from pipeline_v55 import (
        evaluate_step11_restoration_probe_job,
    )

    if _BASE_CONTEXT is None:
        raise RuntimeError(
            "STEP11_RESTORATION_"
            "WORKER_BASE_CONTEXT_NOT_INITIALIZED"
        )

    if (
        job.get(
            "schema"
        )
        !=
        "HUD_FAN_V5_5_"
        "STEP11_RESTORATION_PROBE_JOB_V1"
    ):
        raise ValueError(
            "STEP11_RESTORATION_"
            "INVALID_JOB_SCHEMA"
        )

    runtime = current_runtime()

    if runtime is not None:
        runtime.record(
            "JOB_STARTED",
            kind=
                "STEP11_RESTORATION_PROBE",
            iteration=
                int(
                    job[
                        "iteration"
                    ]
                ),
            column=
                int(
                    job[
                        "column"
                    ]
                ),
            sign=
                float(
                    job[
                        "sign"
                    ]
                ),
        )

    try:

        result = (
            evaluate_step11_restoration_probe_job(
                _BASE_CONTEXT,
                job,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "STEP11_RESTORATION_PROBE_FAILED:"
            f"iteration={job.get('iteration')}:"
            f"column={job.get('column')}:"
            f"sign={job.get('sign')}:"
            f"pid={os.getpid()}:"
            f"{type(exc).__name__}:{exc}"
        ) from exc

    if runtime is not None:
        runtime.record(
            "JOB_COMPLETED",
            kind=
                "STEP11_RESTORATION_PROBE",
            iteration=
                int(
                    job[
                        "iteration"
                    ]
                ),
            column=
                int(
                    job[
                        "column"
                    ]
                ),
            sign=
                float(
                    job[
                        "sign"
                    ]
                ),
        )

    return {
        **result,

        "pid":
            os.getpid(),
    }

