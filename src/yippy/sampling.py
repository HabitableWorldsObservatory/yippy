"""Detector sampling conventions for native FITS images and delivery planes."""

from optixstuff.grids import Image, Measure

# The stellar halo spans about eight decades from the bright field to the dark hole,
# so a band-limited (sinc) interpolant rings at the level of the hole floor along
# the rows and columns of the bright pixels and goes negative. Linear cannot. A log
# interpolant is positive too but biased low: between two pixels it returns their
# geometric mean, and a speckle field varies by large factors pixel to pixel. Rebuilding
# dropped pixels of a shipped map from every second pixel, the dark-hole flux (2.5 to
# 16 lambda/D) is low by 8 to 15 % with log and within 1 % with linear.
STELLAR_RECONSTRUCTION = "linear"


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
    outside="zero",
    quantity="stellar_flux_fraction",
    reconstruction=None,
):
    """Recover the native image measure from per-pixel photon diagnostics.

    reconstruction overrides the default (nearest for integrals, band_limited for
    densities); a stellar halo spanning many decades declares linear, because a
    band-limited interpolant rings below the dark-hole floor.

    Args:
        data: Per-pixel photon fractions, with any leading axes named by ``axes``
            preceding (y, x).
        grid: Pixel grid of the native image.
        axes: Names of the leading axes of ``data``.
        measure: Measure of the stored native values, used to decide whether the
            result is returned as stored or converted to a density.
        outside: Fill rule for positions outside the grid.
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
        outside=outside,
    )
    return image if measure == image.measure else image.density("grid")
