"""Detector sampling conventions for native FITS images and delivery planes."""

from hwoutils.grids import Image, Measure


def sample_units(sample_kind, grid):
    """Return the decoded measure and the factor from stored to decoded values."""
    if sample_kind == "point":
        return Measure.density("grid"), 1.0 / grid.cell_area("grid")
    if sample_kind == "integral":
        return Measure.integral(), 1.0
    if sample_kind == "mean":
        return Measure.integral(), grid.cell_area("grid")
    raise ValueError("sample_kind must be point, integral or mean")


def fraction_image(
    data, grid, axes, measure, *, outside="zero", quantity="stellar_flux_fraction"
):
    """Recover the native image measure from per-pixel photon diagnostics."""
    image = Image(
        data,
        grid,
        axes,
        Measure.integral(),
        quantity,
        reconstruction="nearest" if measure == Measure.integral() else "band_limited",
        outside=outside,
    )
    return image if measure == image.measure else image.density("grid")
