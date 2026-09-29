"""Pure numerical kernels for Chebyshev equations and Fermat evaluations (CPU/GPU array compatible)."""

from __future__ import annotations

from typing import Any


def _row_convolve(xp: Any, a: Any, b: Any) -> Any:
    """Tích đa thức hệ số tăng dần, độc lập cho từng hàng."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("ROW_CONVOLUTION_SHAPE_MISMATCH")

    out = xp.zeros(
        (a.shape[0], a.shape[1] + b.shape[1] - 1),
        dtype=xp.float64,
    )
    for index in range(a.shape[1]):
        out[:, index:index + b.shape[1]] += (
            a[:, index:index + 1] * b
        )
    return out


def _affine_chebyshev(
    xp: Any,
    a: Any,
    b: Any,
    max_degree: int,
) -> list[Any]:
    """Hệ số của T_n(a+b*t), n=0..max_degree."""
    if max_degree < 0:
        raise ValueError("CHEBYSHEV_DEGREE_INVALID")

    count = a.shape[0]
    table = [xp.ones((count, 1), dtype=xp.float64)]
    if max_degree == 0:
        return table

    table.append(xp.stack((a, b), axis=1))

    for degree in range(2, max_degree + 1):
        previous = table[-1]
        older = table[-2]
        current = xp.zeros((count, degree + 1), dtype=xp.float64)

        current[:, :degree] += (
            (2.0 * a)[:, None] * previous
        )
        current[:, 1:] += (
            (2.0 * b)[:, None] * previous
        )
        current[:, :degree - 1] -= older
        table.append(current)

    return table


def cheb_ray_equations(
    xp: Any,
    coeff: Any,
    scale: Any,
    ol: Any,
    dl: Any,
) -> Any:
    """Hệ số F(t)=z_ray(t)-sag(x_ray(t),y_ray(t))."""
    coeff = xp.asarray(coeff, dtype=xp.float64)
    scale = xp.asarray(scale, dtype=xp.float64)
    ol = xp.asarray(ol, dtype=xp.float64)
    dl = xp.asarray(dl, dtype=xp.float64)

    if coeff.ndim != 2 or min(coeff.shape) < 1:
        raise ValueError("CHEB_COEFFICIENT_SHAPE_INVALID")
    if scale.shape != (2,):
        raise ValueError("CHEB_SCALE_SHAPE_INVALID")
    if ol.ndim != 2 or ol.shape[1] != 3 or dl.shape != ol.shape:
        raise ValueError("CHEB_RAY_SHAPE_MISMATCH")

    nx, ny = coeff.shape
    width = max(2, nx + ny - 1)

    tx = _affine_chebyshev(
        xp, ol[:, 0] / scale[0], dl[:, 0] / scale[0], nx - 1
    )
    ty = _affine_chebyshev(
        xp, ol[:, 1] / scale[1], dl[:, 1] / scale[1], ny - 1
    )

    sag = xp.zeros((ol.shape[0], width), dtype=xp.float64)

    for ix in range(nx):
        for iy in range(ny):
            term = _row_convolve(
                xp,
                coeff[ix, iy] * tx[ix],
                ty[iy],
            )
            sag[:, :term.shape[1]] += term

    equation = -sag
    equation[:, 0] += ol[:, 2]
    equation[:, 1] += dl[:, 2]
    return equation


def _unit_rows(xp: Any, value: Any, eps: float) -> Any:
    """Thực thi _unit_rows."""
    norm = xp.linalg.norm(value, axis=-1, keepdims=True)
    return value / xp.maximum(norm, eps)


def poly_sag_slopes(
    xp: Any,
    x: Any,
    y: Any,
    scale: Any,
    terms: list[tuple[int, int]],
    coeff: Any,
    curvature: float,
    conic: float,
) -> tuple[Any, Any, Any]:
    """Thực thi poly_sag_slopes."""
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    scale = xp.asarray(scale, dtype=xp.float64)
    coeff = xp.asarray(coeff, dtype=xp.float64)

    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("POLY_XY_SHAPE_MISMATCH")
    if scale.shape != (2,) or coeff.shape != (len(terms),):
        raise ValueError("POLY_PARAMETER_SHAPE_MISMATCH")

    c = float(curvature)
    k = float(conic)
    r2 = x * x + y * y

    # Giữ công thức và clamp tính toán của _conic() hiện tại.
    # Clamp này KHÔNG thay kiểm raw conic domain ở caller.
    argument = xp.maximum(
        1.0 - (1.0 + k) * c * c * r2, 1e-12
    )
    root = xp.sqrt(argument)
    den = 1.0 + root
    zc = c * r2 / den
    dden = -(1.0 + k) * c * c / (2.0 * root)
    dz = c / den - c * r2 * dden / (den * den)

    X = x / scale[0]
    Y = y / scale[1]

    if not terms:
        zero = xp.zeros_like(x)
        return zc + zero, 2.0 * x * dz + zero, 2.0 * y * dz + zero

    b0, bx, by = [], [], []
    for i, j in terms:
        b0.append(X ** i * Y ** j)
        bx.append(
            xp.zeros_like(X) if i == 0
            else i * X ** (i - 1) * Y ** j / scale[0]
        )
        by.append(
            xp.zeros_like(X) if j == 0
            else j * X ** i * Y ** (j - 1) / scale[1]
        )

    return (
        zc + xp.stack(b0, axis=1) @ coeff,
        2.0 * x * dz + xp.stack(bx, axis=1) @ coeff,
        2.0 * y * dz + xp.stack(by, axis=1) @ coeff,
    )


def fermat_eval(
    xp: Any,
    xy: Any,
    q1: Any,
    targets: Any,
    center: Any,
    frame: Any,
    scale: Any,
    terms: list[tuple[int, int]],
    coeff: Any,
    curvature: float,
    conic: float,
    eps: float,
) -> tuple[Any, Any, Any, Any]:
    """Thực thi fermat_eval."""
    xy = xp.asarray(xy, dtype=xp.float64)
    q1 = xp.asarray(q1, dtype=xp.float64)
    targets = xp.asarray(targets, dtype=xp.float64)
    center = xp.asarray(center, dtype=xp.float64)
    frame = xp.asarray(frame, dtype=xp.float64)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("FERMAT_KERNEL_XY_SHAPE")
    if q1.shape != (len(xy), 3) or targets.shape != q1.shape:
        raise ValueError("FERMAT_KERNEL_ENDPOINT_SHAPE")
    if center.shape != (3,) or frame.shape != (3, 3):
        raise ValueError("FERMAT_KERNEL_FRAME_SHAPE")

    z, gx, gy = poly_sag_slopes(
        xp, xy[:, 0], xy[:, 1], scale, terms, coeff,
        curvature, conic,
    )
    p = center + xp.stack((xy[:, 0], xy[:, 1], z), axis=1) @ frame.T

    tx = frame[:, 0] + gx[:, None] * frame[:, 2]
    ty = frame[:, 1] + gy[:, None] * frame[:, 2]
    normal_local = _unit_rows(
        xp,
        xp.stack((-gx, -gy, xp.ones_like(gx)), axis=1),
        eps,
    )
    normal = normal_local @ frame.T

    incoming = _unit_rows(xp, p - q1, eps)
    outgoing = _unit_rows(xp, targets - p, eps)
    difference = incoming - outgoing

    gradient = xp.stack((
        xp.sum(difference * tx, axis=1),
        xp.sum(difference * ty, axis=1),
    ), axis=1)

    optical_path = (
        xp.linalg.norm(p - q1, axis=1)
        + xp.linalg.norm(targets - p, axis=1)
    )
    cos_in = xp.sum(incoming * normal, axis=1)[:, None]
    reflected = _unit_rows(
        xp, incoming - 2.0 * cos_in * normal, eps
    )
    reflection_residual = xp.linalg.norm(
        outgoing - reflected, axis=1
    )
    return gradient, p, optical_path, reflection_residual
