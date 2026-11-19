"""The two-axis tap stencil of one read, drawn over the node lattice."""

import numpy as np
from matplotlib.colors import to_rgba

from yippy.sinc.viz import _require
from yippy.sinc.viz._common import (
    lattice_parts,
    query_xy,
    require_x64,
    single_ax,
    stencil_profile,
    stencil_reader,
)


def plot_stencil_2d(
    query,
    source,
    *,
    profile=None,
    quadrant=None,
    order="spatial",
    n_taps=None,
    fold=0.0,
    max_area=120.0,
    ax=None,
):
    """Draw the taps of a read, marker area proportional to the weight.

    Each tap sits at its node, with area ``max_area * |w_x w_y|`` (a weight of
    one, a read exactly on a node, fills ``max_area``) and the color of its
    sign (the ``positive`` and ``negative`` roles). The stored lattice is
    scenery behind it. Taps not yet counted are faint ghost dots, so the whole
    stencil is visible before it accumulates.

    Args:
        query: Finite ``(x, y)`` source position.
        source: A yippy.sinc reader, whose own stencil, lattice and reflection are
            drawn; or a lattice, a PixelGrid or ``(origin_xy, spacing_xy,
            shape_yx)`` tuple, drawn with ``profile``.
        profile: Raw KernelProfile of a lattice source. A prefiltered profile
            needs its reader, which holds the designed band.
        quadrant: Lattice source only (a reader supplies it). The lattice stores
            only the positive quadrant. Taps at negative nodes are hollow (they
            exist only by symmetry) and mirrored nodes are drawn as dim dots.
            Requires a zero lattice origin. Default False.
        order: ``"spatial"`` counts taps row by row, ``"weight"`` by decreasing
            weight magnitude, so the first taps counted are the ones that matter.
        n_taps: Number of taps counted so far (0 to the stencil's tap count);
            None counts all.
        fold: Quadrant only. 0 draws taps at their signed nodes, 1 draws taps
            at negative nodes on the stored node they read, 0 to 1 slides them.
        max_area: Marker area in points squared of a weight of one.
        ax: Single axes; None creates a figure. Limits stay fixed on updates.

    Returns:
        PlotResult whose ``scatter`` holds, in order, the stored nodes, the
        mirrored nodes (quadrant only), the ghost dots, the counted taps and
        the query marker. ``update(query=None, n_taps=None, fold=None)`` moves
        the existing artists; an argument left None keeps its current value.

    Raises:
        TypeError: ``profile`` is not a kernel profile.
        ValueError: Invalid lattice, query, order, counts, axes shape, or
            disabled x64; a profile with a reader or none with a lattice.

    Example::

        from yippy.sinc.kernel import KernelProfile
        from yippy.sinc.viz import plot_stencil_2d
        result = plot_stencil_2d((1.0, 2.0), ((0, 0), (0.38, 0.38), (31, 31)),
                                 profile=KernelProfile(), quadrant=True,
                                 order="weight", n_taps=100)
        result.update((0.2, 2.0), n_taps=1024, fold=1.0)
    """
    if hasattr(source, "stencil"):
        if profile is not None or quadrant is not None:
            raise ValueError("a reader supplies its own profile and reflection")
        origin, spacing, shape, quadrant, at = stencil_reader(source)
    else:
        if profile is None:
            raise ValueError("a lattice source needs a profile")
        quadrant = bool(quadrant)
        origin, spacing, shape = lattice_parts(source)
        at = stencil_profile(profile, origin, spacing, shape[::-1], quadrant)
    if quadrant and np.any(origin != 0):
        raise ValueError("quadrant stencils require zero lattice origin")
    if order not in ("spatial", "weight"):
        raise ValueError("order must be 'spatial' or 'weight'")
    ep = _require.eyepiece()
    require_x64()
    nx, ny = shape[::-1].astype(int)
    positive, negative = ep.style.role("positive"), ep.style.role("negative")
    scenery = ep.style.neutral(0.35)
    state = {"query": query_xy(query), "n": n_taps, "fold": float(fold)}
    if not 0.0 <= state["fold"] <= 1.0 or (state["fold"] > 0 and not quadrant):
        raise ValueError("fold must lie in [0, 1] and needs a quadrant lattice")

    count = at(state["query"])[0].shape[1] ** 2

    def tap_table():
        """Return positions, weights, validity and count rank of every tap."""
        nodes, weight, ok = at(state["query"])
        tx, ty = np.meshgrid(nodes[0], nodes[1])
        weights = (weight[1][:, None] * weight[0][None, :]).ravel()
        tx, ty = tx.ravel(), ty.ravel()
        rank = np.arange(count)
        if order == "weight":
            rank[np.argsort(-np.abs(weights), kind="stable")] = np.arange(count)
        valid = (ok[1][:, None] & ok[0][None, :]).ravel()
        f = state["fold"]
        fx = tx + f * (np.abs(tx) - tx) if quadrant else tx
        fy = ty + f * (np.abs(ty) - ty) if quadrant else ty
        xy = np.column_stack((origin[0] + fx * spacing[0], origin[1] + fy * spacing[1]))
        return xy, weights, valid, rank

    ax = single_ax(ax)
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    stored = np.column_stack(
        (origin[0] + xx.ravel() * spacing[0], origin[1] + yy.ravel() * spacing[1])
    )
    points = [ax.scatter(*stored.T, s=3, color=scenery, zorder=1)]
    if quadrant:
        off_x, off_y = stored[:, 0] != 0, stored[:, 1] != 0
        mirrored = np.concatenate(
            (
                stored[off_x] * (-1, 1),
                stored[off_y] * (1, -1),
                stored[off_x & off_y] * (-1, -1),
            )
        )
        points.append(
            ax.scatter(*mirrored.T, s=2, color=ep.style.neutral(0.2), zorder=1)
        )
    xy = tap_table()[0]
    ghost = ax.scatter(*xy.T, s=2.5, color=ep.style.neutral(0.5), zorder=2)
    taps = ax.scatter([], [], zorder=3, linewidths=0.8)
    marker = ax.scatter([], [], marker="x", s=40, color=ep.style.role("text"), zorder=4)
    points += [ghost, taps, marker]

    def redraw():
        xy, weights, valid, rank = tap_table()
        limit = count if state["n"] is None else int(state["n"])
        if not 0 <= limit <= count:
            raise ValueError(f"n_taps must lie between 0 and {count}")
        counted = rank < limit
        sign = np.where(weights >= 0, 0, 1)
        edge = np.array([to_rgba(positive), to_rgba(negative)])[sign]
        face = edge.copy()
        face[~valid, 3] = 0.0
        ghost.set_offsets(xy)
        taps.set_offsets(xy[counted])
        taps.set_sizes(max_area * np.abs(weights[counted]))
        taps.set_facecolors(face[counted])
        taps.set_edgecolors(edge[counted])
        marker.set_offsets(state["query"].reshape(1, 2))

    redraw()
    low = np.minimum(stored.min(0), -stored.max(0) if quadrant else stored.min(0))
    high = stored.max(0)
    pad = 4 * spacing
    ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax.set_aspect("equal")
    ep.plot.label_lod(ax)

    def update(query=None, n_taps=None, fold=None):
        if query is not None:
            state["query"] = query_xy(query)
        if n_taps is not None:
            state["n"] = n_taps
        if fold is not None:
            if not 0.0 <= float(fold) <= 1.0 or (float(fold) > 0 and not quadrant):
                raise ValueError("fold must lie in [0, 1] and needs a quadrant lattice")
            state["fold"] = float(fold)
        redraw()

    return ep.plot.PlotResult(ax, {"scatter": points}, update)
