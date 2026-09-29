"""Ve anh trung gian tu hinh hoc va trace da duoc tinh."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core import aperture_contains_xy


def render_live_preview(
    path: Path,
    *,
    label: str,
    models: list[tuple[str, Any]],
    visor: Any,
    rays: dict[str, Any],
    trace: dict[str, Any] | None,
    trace_mode: str,
    max_plot_rays: int,
) -> dict[str, Any]:
    """Ve snapshot, khong mutate surface, rays hoac trace."""
    figure = Figure(figsize=(11, 7), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, projection="3d")

    missing_models: list[str] = []
    plotted_models: list[str] = []
    extents: list[np.ndarray] = []
    shown_ray_ids: list[str] = []
    ray_count = len(rays["rows"])
    valid_count: int | None = None

    try:
        for name, surface in models:
            if surface is None:
                missing_models.append(name)
                continue

            hx, hy = np.asarray(
                surface.half_aperture, float
            )
            x = np.linspace(-hx, hx, 21)
            y = np.linspace(-hy, hy, 17)
            X, Y = np.meshgrid(x, y)
            xf, yf = X.ravel(), Y.ravel()

            inside = aperture_contains_xy(
                surface.half_aperture,
                surface.aperture_polygon,
                xf,
                yf,
            )

            # Chi la mask ve, khong thay raw model/gate.
            delta = (
                1.0
                - (1.0 + float(surface.conic))
                * float(surface.curvature) ** 2
                * (xf * xf + yf * yf)
            )
            domain = np.isfinite(delta) & (delta >= 0.0)

            world = np.asarray(
                surface.point(xf, yf), float
            ).copy()
            if world.shape != (len(xf), 3):
                raise ValueError("PREVIEW_SURFACE_POINT_SCHEMA")

            good = (
                np.asarray(inside, bool)
                & domain
                & np.all(np.isfinite(world), axis=1)
            )
            world[~good] = np.nan
            grid = world.reshape(len(y), len(x), 3)

            if np.any(good):
                axes.plot_surface(
                    grid[:, :, 0],
                    grid[:, :, 1],
                    grid[:, :, 2],
                    alpha=0.30,
                    linewidth=0.15,
                )
                extents.append(world[good])
                plotted_models.append(name)

            center = np.asarray(surface.center, float)
            if center.shape == (3,) and np.all(np.isfinite(center)):
                axes.text(*center, name)

        if visor is not None:
            xmin, xmax, ymin, ymax = visor.bounds
            X, Y = np.meshgrid(
                np.linspace(xmin, xmax, 25),
                np.linspace(ymin, ymax, 19),
            )
            points = np.asarray(
                visor.point(X.ravel(), Y.ravel()), float
            ).copy()
            good = np.all(np.isfinite(points), axis=1)
            points[~good] = np.nan
            grid = points.reshape(*X.shape, 3)

            if np.any(good):
                axes.plot_surface(
                    grid[:, :, 0],
                    grid[:, :, 1],
                    grid[:, :, 2],
                    alpha=0.18,
                    linewidth=0.1,
                )
                extents.append(points[good])

        if trace is not None:
            mask = np.asarray(trace["valid"])
            if (
                mask.shape != (ray_count,)
                or mask.dtype.kind != "b"
            ):
                raise ValueError("PREVIEW_TRACE_VALID_SCHEMA")

            paths = [
                np.asarray(rays["origins"], float),
                *[
                    np.asarray(point_array, float)
                    for point_array in trace["points"]
                ],
            ]
            if len(paths) != 5 or any(
                value.shape != (ray_count, 3)
                for value in paths
            ):
                raise ValueError("PREVIEW_REVERSE_PATH_SCHEMA")

            valid_count = int(np.count_nonzero(mask))

            # Cung indices cho cung ray set, khong chon lai theo ket qua.
            indices = np.linspace(
                0,
                ray_count - 1,
                min(max_plot_rays, ray_count),
                dtype=int,
            ) if ray_count else np.empty(0, dtype=int)

            for index in indices:
                if not mask[index]:
                    continue
                polyline = np.stack(
                    [value[index] for value in paths]
                )
                if not np.all(np.isfinite(polyline)):
                    continue

                axes.plot(
                    polyline[:, 0],
                    polyline[:, 1],
                    polyline[:, 2],
                    linewidth=0.55,
                    alpha=0.6,
                )
                extents.append(polyline)
                shown_ray_ids.append(
                    str(rays["rows"][index].get(
                        "ray_id", f"LOCAL_INDEX_{index}"
                    ))
                )

        if extents:
            cloud = np.vstack(extents)
            lower = np.min(cloud, axis=0)
            upper = np.max(cloud, axis=0)
            center = (lower + upper) / 2.0
            radius = max(
                float(np.max(upper - lower)) / 2.0,
                1.0,
            )
            axes.set_xlim(center[0] - radius, center[0] + radius)
            axes.set_ylim(center[1] - radius, center[1] + radius)
            axes.set_zlim(center[2] - radius, center[2] + radius)
            axes.set_box_aspect((1, 1, 1))

        axes.set_xlabel("X [mm]")
        axes.set_ylabel("Y [mm]")
        axes.set_zlabel("Z [mm]")
        axes.view_init(elev=22, azim=-55)

        counts = (
            "trace not yet evaluated for this snapshot"
            if valid_count is None
            else f"recorded valid={valid_count}/{ray_count}"
        )
        axes.set_title(
            f"{label}\n{trace_mode}; {counts}\n"
            "PREVIEW ONLY - NOT FINAL OPTICAL CERTIFICATION"
        )

        figure.savefig(path, format="png", dpi=110)

        return {
            "trace_mode": trace_mode,
            "ray_count": ray_count,
            "recorded_valid_count": valid_count,
            "drawn_ray_count": len(shown_ray_ids),
            "drawn_ray_ids": shown_ray_ids,
            "plotted_models": plotted_models,
            "missing_models": missing_models,
            "invalid_ray_polylines_not_drawn": True,
            "sampling_is_for_drawing_only": True,
        }
    finally:
        figure.clear()
