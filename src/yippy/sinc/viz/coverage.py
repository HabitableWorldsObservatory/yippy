"""Stored nodes, virtual reflections, and stencil-served geometry."""

import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Circle, Rectangle

from yippy.sinc.kernel import exact_quotient_floor_v1
from yippy.sinc.viz import _require
from yippy.sinc.viz._common import lattice_parts, query_xy, require_x64, single_ax


def plot_coverage(
    lattice, *, domain=None, quadrant=False, query=None, ax=None, node_kw=None
):
    """Map stored nodes and the half-open rectangle served by complete stencils.

    Args:
        lattice: PixelGrid or ``(origin_xy, spacing_xy, shape_yx)`` tuple.
        domain: Optional Domain box or origin-centered annulus to outline.
        quadrant: Reflect stored nodes across both axes without duplicates.
            Requires a zero-origin positive quadrant lattice.
        query: Optional finite ``(x, y)`` query to mark with its tap rectangle.
        ax: Single axes; None creates a figure. Limits stay fixed on updates.
        node_kw: Scatter overrides for the stored nodes.

    Returns:
        PlotResult with ``scatter`` holding stored nodes, virtual nodes when
        requested, and the query when given. ``fill`` is the served rectangle
        (absent for empty coverage). ``ellipse`` lists the outline patches
        (``Rectangle`` or ``Circle``; eyepiece's vocabulary has no generic
        patch key): domain outlines, then the query's 32-by-32 tap
        rectangle, when present. ``update(query)``
        moves preallocated artists, including when query was initially omitted.

    Raises:
        ValueError: Invalid lattice, query, axes shape, or disabled x64 when
            drawing a query's exact stencil.

    Example::

        from yippy.sinc.viz import plot_coverage
        result = plot_coverage(((0, 0), (0.25, 0.25), (40, 40)),
                               quadrant=True, query=(0.2, 0.3))
        result.update((-0.2, 0.4))
    """
    origin, spacing, shape = lattice_parts(lattice)
    if quadrant and np.any(origin != 0):
        raise ValueError("quadrant coverage requires zero lattice origin")
    point = None if query is None else query_xy(query)
    ep = _require.eyepiece()
    ax = single_ax(ax)
    nx, ny = shape[::-1].astype(int)
    xx, yy = np.meshgrid(
        origin[0] + np.arange(nx) * spacing[0], origin[1] + np.arange(ny) * spacing[1]
    )
    stored = np.column_stack((xx.ravel(), yy.ravel()))
    # Shrink markers on dense grids so neighboring nodes stay distinct.
    size = float(np.clip(8.0 * (24.0 / max(nx, ny)) ** 2, 1.0, 8.0))
    options = {"s": size, "zorder": 2, **(node_kw or {})}
    if "c" not in options:
        options.setdefault("color", ep.style.role("data"))
    points = [ax.scatter(*stored.T, **options)]
    if quadrant:
        reflected = []
        for sx, sy in ((-1, 1), (1, -1), (-1, -1)):
            keep = (stored[:, 0] != 0) if sx < 0 else np.ones(len(stored), bool)
            keep &= (stored[:, 1] != 0) if sy < 0 else np.ones(len(stored), bool)
            reflected.append(stored[keep] * (sx, sy))
        virtual = np.concatenate(reflected)
        points.append(
            ax.scatter(
                *virtual.T,
                s=size,
                facecolors="none",
                edgecolors=ep.style.neutral(0.4),
                zorder=1,
            )
        )
    upper = (shape[::-1] - 16) * spacing + (0 if quadrant else origin)
    lower = -upper if quadrant else origin + 15 * spacing
    artists = {"scatter": points}
    if np.all(upper > lower):
        served = Rectangle(
            lower,
            *(upper - lower),
            facecolor=ep.style.neutral(0.2),
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(served)
        artists["fill"] = served
    outlines = []
    if domain is not None:
        if domain.kind == "box":
            limits = np.asarray(domain.limits_xy_lod)
            outlines.append(
                Rectangle(
                    limits[0],
                    *(limits[1] - limits[0]),
                    fill=False,
                    edgecolor=ep.style.neutral(0.45),
                )
            )
        else:
            outlines.extend(
                Circle(
                    (0, 0), float(radius), fill=False, edgecolor=ep.style.neutral(0.45)
                )
                for radius in np.asarray(domain.radii_lod)
                if radius > 0
            )
        for patch in outlines:
            ax.add_patch(patch)

    def tap_corner(point):
        require_x64()
        floor = np.asarray(
            exact_quotient_floor_v1(
                jnp.asarray(point, dtype=jnp.float64),
                jnp.asarray(origin),
                jnp.asarray(spacing),
            )
        )
        return origin + (floor - 15.5) * spacing

    initial = np.empty((0, 2)) if point is None else point.reshape(1, 2)
    marker = ax.scatter(
        *initial.T, marker="x", s=40, color=ep.style.role("text"), zorder=4
    )
    corner = (np.nan, np.nan) if point is None else tap_corner(point)
    rectangle = Rectangle(
        corner,
        *(32 * spacing),
        fill=False,
        edgecolor=ep.style.neutral(0.75),
        linewidth=1.5,
        zorder=3,
        visible=point is not None,
    )
    ax.add_patch(rectangle)

    def update(query):
        point = query_xy(query)
        corner = tap_corner(point)
        marker.set_offsets(point.reshape(1, 2))
        rectangle.set_xy(corner)
        rectangle.set_visible(True)
        if not any(p is marker for p in points):
            points.append(marker)
            outlines.append(rectangle)
            artists["ellipse"] = outlines

    if point is not None:
        update(point)

    if outlines:
        artists["ellipse"] = outlines
    ax.set_aspect("equal")
    ep.plot.label_lod(ax)
    ax.autoscale_view()
    return ep.plot.PlotResult(ax, artists, update)
