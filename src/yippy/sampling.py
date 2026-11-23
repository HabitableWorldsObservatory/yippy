"""Detector sampling conventions for native FITS images and delivery planes."""

from dataclasses import replace

import jax.numpy as jnp
from optixstuff.coronagraph import on_grid
from optixstuff.grids import Image, Measure

# A stored stellar map holds the intensity sampled at each cell center times the
# cell area, not the integral over the cell; a detector pixel integrates the
# intensity over its footprint. The map is therefore read as the integral of a
# reconstruction of the samples over each pixel, at every pitch including the
# native one. The halo spans about eight decades and the table ends inside the
# bright field, so a band-limited (sinc) reconstruction carries the cut into the
# dark hole and rings below its floor. The not-a-knot cubic spline is accurate on
# a field band-limited at the table pitch (about 0.05 percent at the median and
# 0.3 percent at the 95th percentile per pixel in the hole) and uses the table
# alone at its edge; linear smooths the hole by several percent per pixel.
STELLAR_RECONSTRUCTION = "cubic_spline"

# The spline has negative lobes, so the pixel read is floored at this fraction of
# the linear footprint read, which has no negative weight. The maximum of two
# continuous reads is continuous in pointing, wavelength and stellar diameter,
# with a kink where the floor starts to bind. On eac1_aavc_2d the floor binds
# nowhere between 1 and 24 lambda/D: only at the central null of the largest
# stars and within four cells of the table edge, at pitches finer than the
# table.
STELLAR_FLOOR_FRACTION = 0.1


def stellar_on_grid(image, grid):
    """Read a native stellar map onto a detector grid, floored to stay positive.

    The pixel value is the larger of the declared reconstruction's footprint
    integral and ``STELLAR_FLOOR_FRACTION`` times the linear footprint integral
    of the same samples. The linear margin (0.5 cell) lies inside the cubic
    spline's (2 cells), so the floor does not change the supported pixels.

    Args:
        image: Native stellar map, a point density with the stellar
            reconstruction.
        grid: Detector grid to read onto.

    Returns:
        Cell-integrated stellar flux fractions on ``grid``, declaring the
        reconstruction of ``image``.
    """
    read = on_grid(image, grid)
    floor = on_grid(replace(image, reconstruction="linear"), grid)
    return replace(
        read, data=jnp.maximum(read.data, STELLAR_FLOOR_FRACTION * floor.data)
    )


def sample_units(sample_kind, grid):
    """Return the decoded measure and the factor from stored to decoded values.

    Args:
        sample_kind: How stored pixel values sample the image: ``"point"``,
            ``"integral"`` or ``"mean"``.
        grid: Pixel grid whose cell area converts stored values to the measure.

    Returns:
        Pair of (decoded measure, multiplicative factor taking stored values to
        decoded values).
    """
    if sample_kind == "point":
        return Measure.density("grid"), 1.0 / grid.cell_area("grid")
    if sample_kind == "integral":
        return Measure.integral(), 1.0
    if sample_kind == "mean":
        return Measure.integral(), grid.cell_area("grid")
    raise ValueError("sample_kind must be point, integral or mean")


def fraction_image(
    data,
    grid,
    axes,
    measure,
    *,
    quantity="stellar_flux_fraction",
    reconstruction=None,
):
    """Recover the native image measure from per-pixel photon diagnostics.

    reconstruction overrides the default (nearest for integrals, band_limited for
    densities); a stellar halo spanning many decades declares the cubic spline,
    because a band-limited interpolant of a table cut inside the bright field
    rings below the dark-hole floor.

    Args:
        data: Per-pixel photon fractions, with any leading axes named by ``axes``
            preceding (y, x).
        grid: Pixel grid of the native image.
        axes: Names of the leading axes of ``data``.
        measure: Measure of the stored native values, used to decide whether the
            result is returned as stored or converted to a density.
        quantity: Name of the physical quantity carried by the image.
        reconstruction: Reconstruction override between pixels, or None for the
            default.

    Returns:
        Image of the recovered native measure on ``grid``.
    """
    image = Image(
        data,
        grid,
        axes,
        Measure.integral(),
        quantity,
        reconstruction=reconstruction
        or ("nearest" if measure == Measure.integral() else "band_limited"),
    )
    return image if measure == image.measure else image.density("grid")
