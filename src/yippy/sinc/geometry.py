"""Source-lattice operations on shared, static optixstuff geometry."""

import jax.numpy as jnp
from optixstuff.grids import PixelGrid


def origin_xy(grid):
    """Coordinate of the first stored node, in the grid's units.

    Args:
        grid: Source pixel grid.

    Returns:
        Array of two values, the (x, y) coordinate of pixel (0, 0) in the
        grid's units.
    """
    return jnp.asarray(grid.pixel_to_unit(0.0, 0.0))


def spacing_xy(grid):
    """Diagonal source pitches; admission rejects rotated or sheared sources.

    Args:
        grid: Source pixel grid.

    Returns:
        Array of two values, the (x, y) node pitches taken from the diagonal of
        the grid basis, in the grid's units.
    """
    return jnp.asarray((grid.basis[0][0], grid.basis[1][1]))


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
