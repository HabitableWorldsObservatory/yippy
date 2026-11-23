"""Shared axes and precision validation at the plotting boundary."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import AbstractKernelProfile, read_stencil


def single_ax(ax):
    """Resolve one axes, rejecting arrays with an explicit shape error.

    Args:
        ax: A matplotlib axes, or None to create a new figure with one axes.

    Returns:
        The axes to draw on.
    """
    if ax is None:
        return plt.subplots(layout="constrained")[1]
    if not isinstance(ax, plt.Axes):
        shape = np.asarray(ax, dtype=object).shape
        raise ValueError(f"expected a single ax, got axes shape {shape}")
    return ax


def require_x64():
    """Refuse precision-sensitive kernel arithmetic without changing JAX."""
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before plotting yippy.sinc stencils")


def query_xy(query):
    """Convert one finite coordinate pair to a host array.

    Args:
        query: Source position (x, y), two finite numbers.

    Returns:
        Float array of shape (2,).
    """
    point = np.asarray(query, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("query must be a finite (x, y) pair")
    return point


def lattice_parts(lattice):
    """Return validated ``(origin, spacing, shape_yx)`` host arrays of a lattice.

    Accepts a PixelGrid or an ``(origin_xy, spacing_xy, shape_yx)`` tuple.

    Args:
        lattice: A PixelGrid, or a tuple of the (x, y) origin, the positive
            (x, y) node spacing and the integer (rows, columns) shape.

    Returns:
        A tuple ``(origin, spacing, shape)``: float arrays of shape (2,) for
        the origin and spacing and an array of shape (2,) for the shape.
    """
    if hasattr(lattice, "basis"):
        origin, spacing, shape = (
            origin_xy(lattice),
            spacing_xy(lattice),
            lattice.shape,
        )
    else:
        origin, spacing, shape = lattice
    origin, spacing = np.asarray(origin, dtype=float), np.asarray(spacing, dtype=float)
    shape = np.asarray(shape)
    if (
        origin.shape != (2,)
        or spacing.shape != (2,)
        or shape.shape != (2,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
        or np.any(shape < 1)
        or np.any(shape != np.floor(shape))
    ):
        raise ValueError(
            "lattice needs finite origin, positive spacing, and integer shape_yx"
        )
    return origin, spacing, shape


def stencil_reader(reader):
    """Return ``(origin, spacing, shape_yx, reflected, at)`` of a reader's lattice.

    ``at(query_xy)`` returns the reader's own stencil as host arrays
    ``(nodes, weights, valid)``, each [2, taps].

    Args:
        reader: A reader that exposes ``stencil`` and a sampled ``grid``.

    Returns:
        A tuple ``(origin, spacing, shape_yx, reflected, at)``: the lattice
        arrays, whether the table is read by reflection, and the stencil
        function.
    """
    origin, spacing, shape = lattice_parts(reader.grid.source)

    def at(query):
        found = reader.stencil(jnp.asarray(query, dtype=jnp.float64))
        return tuple(np.asarray(a) for a in (found.nodes, found.weights, found.valid))

    return origin, spacing, shape, reader.grid.reflection.kind != "none", at


def stencil_profile(profile, origin, spacing, size_xy, reflected):
    """Return ``at(query_xy)`` for the stencil of an explicit kernel profile.

    A prefiltered profile needs the weights of its designed band, which only a
    reader carries, so it is refused here.

    Args:
        profile: The kernel profile whose stencil is drawn.
        origin: Coordinate of node zero, per axis (x, y).
        spacing: Node spacing, per axis (x, y), in the unit of ``origin``.
        size_xy: Stored node count per axis (x first).
        reflected: Whether the table is read by reflection.

    Returns:
        A function ``at(query_xy)`` returning the host arrays
        ``(nodes, weights, valid)``, each [2, taps].
    """
    if not isinstance(profile, AbstractKernelProfile):
        raise TypeError("profile must be a kernel profile; or pass a reader")
    origin, spacing = jnp.asarray(origin), jnp.asarray(spacing)

    def at(query):
        found = read_stencil(
            profile,
            jnp.asarray(query, dtype=jnp.float64),
            origin,
            spacing,
            size_xy,
            reflected,
        )
        return tuple(np.asarray(a) for a in (found.nodes, found.weights, found.valid))

    return at
