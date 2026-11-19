"""Kernel shape and signed one-dimensional stencil views."""

import jax.numpy as jnp
import numpy as np
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from yippy.sinc.kernel import KernelProfile, weights
from yippy.sinc.viz import _require
from yippy.sinc.viz._common import (
    require_x64,
    single_ax,
    stencil_profile,
    stencil_reader,
)


def plot_kernel(profile=None, *, ax=None, show_ideal=True, line_kw=None):
    """Compare the fixed Kaiser-sinc kernel with its ideal sinc reference.

    Args:
        profile: Optional default KernelProfile, validated but otherwise ignored.
        ax: Single axes; None creates a figure with constrained layout.
        show_ideal: Whether to draw the ideal sinc as scenery.
        line_kw: Overrides for the kernel's Line2D.

    Returns:
        PlotResult with ``line`` for the kernel and ``lines`` for the ideal
        sinc (first, if present), then the two support-edge reference lines.
        The kernel is fixed, so the result has no ``update``.

    Raises:
        TypeError: The profile is not a KernelProfile.
        ValueError: The axes shape or profile is invalid, or x64 is disabled.

    Example::

        from yippy.sinc.viz import plot_kernel
        result = plot_kernel()
    """
    ep = _require.eyepiece()
    require_x64()
    if profile is not None:
        if not isinstance(profile, KernelProfile):
            raise TypeError("profile must be a KernelProfile or None")
        profile.__check_init__()
    offsets = np.linspace(-17.0, 17.0, 2001)
    values = np.asarray(weights(jnp.asarray(offsets, dtype=jnp.float64)))
    ax = single_ax(ax)
    options = {"color": ep.style.role("data"), **normalize_kwargs(line_kw, Line2D)}
    line = ax.plot(offsets, values, **options)[0]
    refs = []
    if show_ideal:
        refs.append(
            ax.plot(
                offsets,
                np.sinc(offsets),
                color=ep.style.neutral(0.45),
                linewidth=line.get_linewidth() * 0.6,
                zorder=1,
            )[0]
        )
    for edge in (-16, 16):
        refs.append(
            ep.plot.ref_line(ax, edge, orient="v", color=ep.style.neutral(0.2)).artists[
                "line"
            ]
        )
    ax.set_xlabel("offset from query (samples)")
    ax.set_ylabel("weight")
    return ep.plot.PlotResult(ax, {"line": line, "lines": refs})


def plot_stencil(
    query,
    *,
    reader=None,
    profile=None,
    axis=0,
    origin=None,
    spacing=None,
    n_nodes=None,
    samples=None,
    ax=None,
    stem_kw=None,
):
    """Draw the signed weights of one axis of a read, and optional node samples.

    The taps are those the profile reads with: pass a ``reader`` to draw its own
    stencil on its own lattice, or a raw ``KernelProfile`` with the lattice
    keywords. A prefiltered profile needs its reader, which holds the designed
    band.

    Args:
        query: Finite query position along ``axis``, in the units of the lattice.
        reader: A yippy.sinc reader. Supplies profile, origin, spacing, node count
            and reflection; the lattice keywords must then be left None.
        profile: KernelProfile to draw when there is no reader.
        axis: 0 for the x axis of a reader's lattice, 1 for y.
        origin: Position of stored node zero (profile only; default 0).
        spacing: Positive node spacing (profile only; default 1).
        n_nodes: Optional positive count of stored nodes indexed from zero
            (profile only).
        samples: One-dimensional values indexed from zero, or a function of
            node position. Only existing nodes in the stencil view are shown.
        ax: Single axes; None creates a figure. Limits stay fixed on updates.
        stem_kw: LineCollection overrides for the stems.

    Returns:
        PlotResult whose ``update(query)`` moves existing artists. ``collection``
        holds one stem per tap; its segments' endpoint y values are the current
        weights. Recover taps as ``round((endpoint_x - origin) / spacing)``.
        ``line`` is the query line. ``scatter`` holds node ticks, tap endpoints
        (hollow outside storage), and sample dots when requested, in that order.

    Raises:
        TypeError: ``profile`` is not a kernel profile.
        ValueError: Invalid geometry, samples, axes shape, or disabled x64;
            neither or both of reader and profile; lattice keywords with a reader;
            a prefiltered profile without a reader.

    Example::

        from yippy.sinc.kernel import KernelProfile
        from yippy.sinc.viz import plot_stencil
        result = plot_stencil(0.3, profile=KernelProfile(), n_nodes=40)
        result.update(0.5)
    """
    if (reader is None) == (profile is None):
        raise ValueError("pass exactly one of reader and profile")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    query = float(np.asarray(query))
    sample_values = (
        None if samples is None or callable(samples) else np.asarray(samples)
    )
    ep = _require.eyepiece()
    require_x64()
    if reader is not None:
        if origin is not None or spacing is not None or n_nodes is not None:
            raise ValueError("a reader supplies origin, spacing and n_nodes")
        origins, spacings, shape, _, at = stencil_reader(reader)
        origin, spacing = float(origins[axis]), float(spacings[axis])
        n_nodes = int(shape[::-1][axis])
    else:
        origin = 0.0 if origin is None else float(origin)
        spacing = 1.0 if spacing is None else float(spacing)
        if not np.isfinite(origin) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("origin must be finite and spacing must be positive")
        if n_nodes is not None and (int(n_nodes) != n_nodes or n_nodes < 1):
            raise ValueError("n_nodes must be a positive integer")
        size = None if n_nodes is None else np.full(2, int(n_nodes))
        at = stencil_profile(
            profile, np.full(2, origin), np.full(2, spacing), size, False
        )
    if sample_values is not None and sample_values.ndim != 1:
        raise ValueError("samples must be a one-dimensional array")
    data_color = ep.style.role("data")
    scenery = ep.style.neutral(0.4)

    def resolve(point):
        point = float(np.asarray(point))
        if not np.isfinite(point):
            raise ValueError("query must be finite")
        require_x64()
        found = at(np.full(2, point))
        taps, weight, valid = (a[axis] for a in found)
        positions = origin + taps * spacing
        colors = np.array([to_rgba(data_color if v else scenery) for v in valid])
        faces = colors.copy()
        faces[~valid, 3] = 0.0
        nodes = np.arange(taps[0] - 1, taps[-1] + 2)
        if n_nodes is not None:
            nodes = nodes[(nodes >= 0) & (nodes < n_nodes)]
        node_positions = origin + nodes * spacing
        sample_points = np.empty((0, 2))
        if samples is not None:
            if sample_values is not None:
                nodes = nodes[(nodes >= 0) & (nodes < len(sample_values))]
            x = origin + nodes * spacing
            y = np.asarray(
                samples(x) if callable(samples) else sample_values[nodes], dtype=float
            )
            sample_points = np.column_stack((x, np.broadcast_to(y, x.shape)))
        segments = np.stack(
            (
                np.column_stack((positions, np.zeros(len(taps)))),
                np.column_stack((positions, weight)),
            ),
            axis=1,
        )
        return (
            point,
            positions,
            weight,
            colors,
            faces,
            node_positions,
            sample_points,
            segments,
        )

    state = resolve(query)
    ax = single_ax(ax)
    options = {"colors": state[3], "linewidths": 1.5, **(stem_kw or {})}
    stems = LineCollection(state[-1], **options)
    ax.add_collection(stems)
    nodes = ax.scatter(
        state[5], np.zeros(len(state[5])), marker="|", s=12, color=scenery
    )
    endpoints = ax.scatter(
        state[1], state[2], s=18, facecolors=state[4], edgecolors=state[3]
    )
    points = [nodes, endpoints]
    if samples is not None:
        points.append(ax.scatter(*state[6].T, s=18, color=ep.style.role("model")))
    query_line = ep.plot.ref_line(
        ax, query, orient="v", color=ep.style.role("text"), line_kw={"linestyle": "-"}
    ).artists["line"]
    ax.set_xlim(state[1][0] - spacing, state[1][-1] + spacing)
    ax.autoscale_view(scalex=False)
    ax.set_xlabel("node position (samples)")
    ax.set_ylabel("weight / sample value" if samples is not None else "weight")

    def update(query):
        state = resolve(query)
        stems.set_segments(state[-1])
        if not any(
            k in (stem_kw or {}) for k in ("color", "colors", "edgecolor", "edgecolors")
        ):
            stems.set_color(state[3])
        nodes.set_offsets(np.column_stack((state[5], np.zeros(len(state[5])))))
        endpoints.set_offsets(np.column_stack((state[1], state[2])))
        endpoints.set_edgecolors(state[3])
        endpoints.set_facecolors(state[4])
        query_line.set_xdata([state[0], state[0]])
        if samples is not None:
            points[-1].set_offsets(state[6])

    return ep.plot.PlotResult(
        ax, {"collection": stems, "line": query_line, "scatter": points}, update
    )
