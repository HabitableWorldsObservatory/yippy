"""Independent pixel-footprint reads of a tabulated map, in SciPy and NumPy.

A stored map holds point samples at the cell centers times the cell area. A
detector pixel integrates the reconstructed intensity over its footprint. These
matrices compute that integral along one axis from codes that share nothing with
optixstuff: SciPy interpolating B-splines (exact antiderivatives) and a direct
Fourier sum for the band-limited read. Positions and widths are in source cells,
with sample n at coordinate n; a read of samples ``t`` is ``wy @ t @ wx.T``.
"""

import numpy as np
from scipy.interpolate import make_interp_spline

# Units of machine epsilon allowed per pixel, times the sum of absolute terms.
ROUNDING_UNITS = 16


def axis_positions(source_grid, target_grid):
    """Target pixel centers in source cells and the pixel width, for x then y.

    Both grids must share their unit and have no rotation. The geometry is
    typed here from the grid centers and steps, not taken from the reader.
    """
    if source_grid.unit != target_grid.unit:
        raise ValueError("convert both grids to one unit first")
    ratio = target_grid.step / source_grid.step
    ny, nx = target_grid.shape
    x = (np.arange(nx) - target_grid.center_x) * ratio + source_grid.center_x
    y = (np.arange(ny) - target_grid.center_y) * ratio + source_grid.center_y
    return (x, ratio), (y, ratio)


def spline_footprint(n_source, positions, width, order):
    """Pixel integrals of the order-``order`` interpolating spline of each sample.

    SciPy's interpolating spline with its default (not-a-knot) end condition,
    extended by its end pieces and integrated over each pixel clipped to the
    table footprint [-1/2, N - 1/2]. Order 1 is the hat function with the
    samples beyond the table taken as zero, so its reach ends one cell past
    the table and it is not clipped.
    """
    positions = np.asarray(positions, dtype=float)
    if order == 1:
        knots = np.arange(-1.0, n_source + 1.0)
        values = np.pad(np.eye(n_source), ((1, 1), (0, 0)))
        low, high = -1.0, float(n_source)
    else:
        knots = np.arange(float(n_source))
        values = np.eye(n_source)
        low, high = -0.5, n_source - 0.5
    antiderivative = make_interp_spline(knots, values, k=order).antiderivative()
    upper = np.clip(positions + width / 2, low, high)
    lower = np.clip(positions - width / 2, low, high)
    return antiderivative(upper) - antiderivative(lower)


def band_limited_footprint(n_source, positions, width, pad):
    """Pixel integrals of the periodic band-limited interpolant of each sample.

    The table is zero-padded to ``pad`` samples with the table at offset
    ``(pad - n_source) // 2``; the Fourier sum of the padded table, multiplied
    by the transform of the pixel box, is evaluated at each pixel center.
    """
    offset = (pad - n_source) // 2
    frequency = np.fft.fftfreq(pad)
    centers = np.asarray(positions, dtype=float) + offset
    box = width * np.sinc(frequency * width)
    columns = np.arange(n_source) + offset
    to_target = np.exp(2j * np.pi * centers[:, None] * frequency[None, :]) * box
    from_source = np.exp(-2j * np.pi * frequency[:, None] * columns[None, :])
    return (to_target @ from_source).real / pad


def read(table, axes, matrix):
    """Apply one footprint matrix per axis to a (ny, nx) table.

    Args:
        table: Stored values, the sample times the cell area.
        axes: ``axis_positions`` output.
        matrix: Callable ``(n_source, positions, width) -> matrix``.

    Returns:
        The footprint read on the target grid, in the units of ``table``.
    """
    (x, wx), (y, wy) = axes
    ny, nx = table.shape
    return matrix(ny, y, wy) @ table @ matrix(nx, x, wx).T


def floored_spline_read(table, axes, fraction=0.1):
    """The declared stellar read: cubic spline, floored at a linear fraction."""
    spline = read(table, axes, lambda n, p, w: spline_footprint(n, p, w, 3))
    linear = read(table, axes, lambda n, p, w: spline_footprint(n, p, w, 1))
    return np.maximum(spline, fraction * linear)


def assert_stellar_read(actual, table, source_grid, target_grid, *, where=None):
    """Assert a stellar map read equals the floored spline read to rounding.

    Tolerance basis: floating point. Each pixel is a sum of signed products of
    footprint weights and stored values, so its rounding error is bounded by a
    few units of the result's machine epsilon times the same sum taken over
    absolute values; ``ROUNDING_UNITS`` of them are allowed.

    Args:
        actual: Pixel values returned by the library.
        table: Stored native values (the sample times the cell area).
        source_grid: Native grid of ``table``.
        target_grid: Grid of ``actual``, in the unit of ``source_grid``.
        where: Boolean mask of the pixels to compare; None compares all.
    """
    axes = axis_positions(source_grid, target_grid)
    table = np.asarray(table, dtype=float)
    truth = floored_spline_read(table, axes)
    magnitude = read(
        np.abs(table), axes, lambda n, p, w: np.abs(spline_footprint(n, p, w, 3))
    ) + read(np.abs(table), axes, lambda n, p, w: spline_footprint(n, p, w, 1))
    actual = np.asarray(actual)
    excess = np.abs(actual.astype(float) - truth) - (
        ROUNDING_UNITS * np.finfo(actual.dtype).eps * magnitude
    )
    where = np.ones(truth.shape, bool) if where is None else np.asarray(where)
    assert where.any(), "no pixel to compare"
    assert np.all(excess[where] <= 0.0), (
        f"{np.sum(excess[where] > 0)} pixels differ from the floored spline read "
        "by more than rounding"
    )
