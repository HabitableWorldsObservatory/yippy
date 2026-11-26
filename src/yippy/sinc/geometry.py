"""Source-lattice operations on shared, static optixstuff geometry."""

import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.grids import PixelGrid


def array_namespace(*values):
    """Keep concrete host coordinates wide until their origin is subtracted.

    Args:
        *values: Coordinates or pytrees of them.

    Returns:
        ``jax.numpy`` if any leaf is a JAX array or tracer, else ``numpy``.
    """
    if any(
        isinstance(v, (jax.Array, jax.core.Tracer)) for v in jax.tree.leaves(values)
    ):
        return jnp
    return np


def origin_xy(grid):
    """Coordinate of the first stored node, in the grid's units.

    Args:
        grid: Source pixel grid.

    Returns:
        Array of two values, the (x, y) coordinate of pixel (0, 0) in the
        grid's units.
    """
    return np.asarray(grid.basis) @ np.array([-grid.center_x, -grid.center_y])


def raster_axes(grid):
    """Host binary64 pixel-center axes of a grid: x per column, y per row.

    Args:
        grid: Source pixel grid.

    Returns:
        Tuple of the x coordinates, shape (columns,), taken along the first
        row, and the y coordinates, shape (rows,), taken down the first
        column, in the grid's units.
    """
    rows, columns = grid.shape
    (a, b), (c, d) = np.asarray(grid.basis, dtype=np.float64)
    dx = np.arange(columns, dtype=np.float64) - grid.center_x
    dy = np.arange(rows, dtype=np.float64) - grid.center_y
    return a * dx + b * -grid.center_y, c * -grid.center_x + d * dy


def spacing_xy(grid):
    """Diagonal source pitches; admission rejects rotated or sheared sources.

    Args:
        grid: Source pixel grid.

    Returns:
        Array of two values, the (x, y) node pitches taken from the diagonal of
        the grid basis, in the grid's units.
    """
    return np.array((grid.basis[0][0], grid.basis[1][1]))


def source_grid(origin_lod, spacing_lod, shape_yx, frame):
    """Build a shared axis-aligned source grid from declared node coordinates.

    PixelGrid stores the zero-coordinate center rather than an origin. The
    recovered origin is checked against delivered offsets during admission.

    Args:
        origin_lod: Coordinate (x, y) of the first stored node, in lambda/D.
        spacing_lod: Node pitches (x, y), in lambda/D; both must be nonzero.
        shape_yx: Array shape of the grid as (rows, columns).
        frame: Coordinate frame object attached to the grid.

    Returns:
        Axis-aligned grid in lambda/D with the given shape and pitches.
    """
    x, y = map(float, origin_lod)
    hx, hy = map(float, spacing_lod)
    return PixelGrid(shape_yx, -x / hx, -y / hy, ((hx, 0.0), (0.0, hy)), "lod", frame)
