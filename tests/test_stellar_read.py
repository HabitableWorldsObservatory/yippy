"""The pixel read of the stellar map: accuracy, positivity and continuity.

A stored stellar map holds the intensity at each cell center times the cell
area. ``stellar_map`` reads it onto a detector as the footprint integral of its
cubic spline, floored at a tenth of the linear footprint integral. These tests
run on the public ``eac1_aavc_2d`` map. Their references are independent codes
(SciPy splines and a direct Fourier sum, ``footprint_truth``).
"""

from dataclasses import replace

import numpy as np
import pytest
from footprint_truth import (
    assert_stellar_read,
    axis_positions,
    band_limited_footprint,
    read,
    spline_footprint,
)
from optixstuff.coronagraph import on_grid
from optixstuff.grids import Measure, PixelGrid, coverage

from yippy.sampling import STELLAR_FLOOR_FRACTION, STELLAR_RECONSTRUCTION

# A star of 0.0325 lambda/D, between two tabulated diameters.
DIAMETER_LOD = 0.0325
# Detector pitches near the table pitch, in lambda/D, with the offset of the
# optical axis from the central pixel; the 0.25 rungs are aligned and offset.
NEAR_TABLE_PITCHES = ((0.2275, 0.03), (0.25, 0.0), (0.25, 0.0925), (0.275, 0.05))
# A coarser detector, the table pitch, and the two finer pitches where the
# spline rings at the central null of a large star.
SUPPORT_PITCHES = (0.4532, 0.25, 0.0625, 0.03125)


def _detector(stellar_grid, pitch, extent_lod, offset_lod=0.0):
    """Square grid of side ``extent_lod`` about the stellar axis, shifted."""
    n = int(np.ceil(extent_lod / pitch / 2)) * 2
    center = (n - 1) / 2 - offset_lod / pitch
    return PixelGrid.regular(
        (n, n), pitch, "lod", frame=stellar_grid.frame, center=(center, center)
    )


def _radius(grid):
    """Distance of each pixel center from the optical axis, in lambda/D."""
    rows, columns = np.indices(grid.shape)
    return np.hypot(columns - grid.center_x, rows - grid.center_y) * grid.step


def _supported(coro, grid):
    """Pixels whose corners lie two cells inside the stellar table."""
    return coverage(
        coro.stellar_grid,
        grid,
        reconstruction=STELLAR_RECONSTRUCTION,
        measure=Measure.density("grid"),
    )


def _tapered(table, stellar_grid):
    """The table times a raised-cosine window that is one from 2 to 16 lambda/D.

    The window falls to zero inside 1 lambda/D and beyond 20 lambda/D, so the
    band-limited reference does not carry the bright field into the hole.
    """
    rows, columns = np.indices(table.shape)
    r = (
        np.hypot(columns - stellar_grid.center_x, rows - stellar_grid.center_y)
        * stellar_grid.step
    )
    outer = np.clip((20.0 - r) / 4.0, 0.0, 1.0)
    inner = np.clip(r - 1.0, 0.0, 1.0)
    return (
        table
        * (0.5 - 0.5 * np.cos(np.pi * outer))
        * (0.5 - 0.5 * np.cos(np.pi * inner))
    )


def _aperture_masks(grid):
    """Pixel-center masks of 0.7 lambda/D apertures, 24 on each of six rings."""
    rows, columns = np.indices(grid.shape)
    x = (columns - grid.center_x) * grid.step
    y = (rows - grid.center_y) * grid.step
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    return [
        np.hypot(x - ring * np.cos(angle), y - ring * np.sin(angle)) < 0.7
        for ring in (4.0, 6.0, 8.27, 10.0, 12.0, 15.0)
        for angle in angles
    ]


def test_aperture_sums_match_independent_footprint_reads(shipped_coro):
    """Fails if the stellar map is read with a less accurate reconstruction.

    Aperture sums of radius 0.7 lambda/D on rings from 4 to 15 lambda/D, at
    pitches from 0.2275 to 0.275 lambda/D, against two references read from the
    same samples: (a) the band-limited footprint of the table tapered outside
    the rings, the independent row, and (b) the footprint of the order-5
    interpolating spline, an order check in the same family as the read and
    labelled so. Tolerance 0.3 percent for both. Basis (a): the tapered
    band-limited read was measured within 0.02 percent of the splines in these
    apertures. Basis (b): spike, twice the largest measured (0.14 percent).
    Measured here: at most 0.096 percent against (a) and 0.089 percent against
    (b). Mutants: a linear declaration (5.8 to 7.4 percent) and the Keys cubic
    (0.48 to 0.59 percent at three of the four pitches) fail.

    Every pixel is also compared to rounding with the SciPy footprint read of
    the floored spline (``footprint_truth``), which verifies the declaration,
    the floor and the scale of the read on a structured map. The pixel bound
    propagates SciPy's two antiderivative operand magnitudes and includes
    coordinate and contraction roundoff on both sides (see assert_stellar_read).
    """
    coro = shipped_coro
    native = coro.stellar_map(None, DIAMETER_LOD)
    table = np.asarray(native.integrated().data, dtype=float)
    tapered = _tapered(table, coro.stellar_grid)
    for pitch, offset in NEAR_TABLE_PITCHES:
        grid = _detector(coro.stellar_grid, pitch, 34.0, offset)
        actual = np.asarray(coro.stellar_map(None, DIAMETER_LOD, grid=grid).data)
        axes = axis_positions(coro.stellar_grid, grid)
        references = {
            "band-limited, tapered": read(
                tapered, axes, lambda n, p, w: band_limited_footprint(n, p, w, 2 * n)
            ),
            "order-5 spline": read(
                table, axes, lambda n, p, w: spline_footprint(n, p, w, 5)
            ),
        }
        for name, reference in references.items():
            for mask in _aperture_masks(grid):
                ratio = actual[mask].sum() / reference[mask].sum()
                assert abs(ratio - 1.0) < 3e-3, (pitch, offset, name, ratio)
        assert_stellar_read(actual, table, coro.stellar_grid, grid)


@pytest.mark.medium(reason="every tabulated diameter on four grids up to 2048^2, 6 s")
def test_stellar_read_is_positive_and_its_floor_idle_in_the_hole(shipped_coro):
    """Fails if a supported stellar pixel is not positive, or the floor binds.

    (2a) No supported pixel is negative or zero, at every tabulated diameter,
    on a coarser grid, on the table pitch and at the two finer pitches where
    the spline rings at the central null of the largest stars. Truth: the
    linear footprint read has no negative weight, so the floored read is
    positive wherever the table is. Mutant: without the floor, four pixels
    inside 0.3 lambda/D are negative at 0.0625 lambda/D for the 1 lambda/D star.

    (2b) Between 1 and 24 lambda/D no supported pixel is changed by the floor:
    the read equals the unfloored spline read exactly. Basis: measured on this
    map; the floor binds only inside 0.3 lambda/D and beyond 31 lambda/D,
    within four cells of the table edge. Mutant: a floor at the full linear
    read binds through the hole.

    2b is scoped to the AAVC map. On eac1_optimal_order_6_1d at 0.03125
    lambda/D the floor binds on 14 pixels at 4.11 to 4.20 lambda/D for the
    point star, whose map sums to 1.4e-27 of the unocculted star: the spline
    rings on numerical zero there and the floor does what it is for.
    """
    coro = shipped_coro
    for pitch in SUPPORT_PITCHES:
        grid = _detector(coro.stellar_grid, pitch, 64.0)
        supported = _supported(coro, grid)
        assert supported.sum() > 0.8 * supported.size
        hole = supported & (_radius(grid) > 1.0) & (_radius(grid) < 24.0)
        for diameter in coro._host.stellar_intens.diams.value:
            native = coro.stellar_map(None, float(diameter))
            floored = np.asarray(
                coro.stellar_map(None, float(diameter), grid=grid).data
            )
            assert floored[supported].min() > 0.0, (pitch, diameter)
            spline = np.asarray(on_grid(native, grid).data)
            assert np.array_equal(floored[hole], spline[hole]), (pitch, diameter)


@pytest.mark.medium(reason="21 pointings of a 1024^2 grid, three reads each, 4 s")
def test_stellar_read_is_continuous_where_the_floor_binds(shipped_coro):
    """Fails if the floor switches reads instead of taking their maximum.

    This proves that the floor is a maximum and not a switch. It does not test
    continuity by itself: the continuity of each read in pointing is
    optixstuff's ``test_no_switch_at_equal_pitch``, and a maximum of two
    continuous reads is continuous.

    (2c) A pointing offset swept along the diagonal in 20 steps of 0.05 cell,
    at 0.0625 lambda/D for the 1 lambda/D star, where the floor binds at the
    central null. The maximum of two reads moves by no more than the larger
    of their moves, so each supported pixel of the floored read changes by at
    most the larger of the spline read's change and the floor's change.
    Tolerance basis: exact algebra, plus rounding (16 units of float32
    epsilon of the pixel). Mutant: a switch to the linear read where the
    spline is negative fails at the first step.

    The bound is stated against the spline and floor reads, not against the
    linear read alone: the linear footprint read is smoother than the spline
    (weights 1/8, 3/4, 1/8 per axis at the table pitch), so near a speckle
    null the spline legitimately changes per step by up to 26 percent of its
    value more than the linear read does here, and 13 percent at the table
    pitch.
    """
    coro = shipped_coro
    diameter = float(coro._host.stellar_intens.diams.value[-1])
    native = coro.stellar_map(None, diameter)
    linear = replace(native, reconstruction="linear")
    pitch = 0.0625
    step = 0.05 * coro.stellar_grid.step
    previous = None
    for k in range(21):
        grid = _detector(coro.stellar_grid, pitch, 64.0, -k * step)
        reads = (
            np.asarray(coro.stellar_map(None, diameter, grid=grid).data, dtype=float),
            np.asarray(on_grid(native, grid).data, dtype=float),
            STELLAR_FLOOR_FRACTION
            * np.asarray(on_grid(linear, grid).data, dtype=float),
            _supported(coro, grid),
        )
        if previous is not None:
            floored, spline, floor, supported = reads
            before, spline0, floor0, supported0 = previous
            both = supported & supported0
            allowed = np.maximum(np.abs(spline - spline0), np.abs(floor - floor0))
            rounding = 16 * np.finfo(np.float32).eps * np.abs(floored)
            change = np.abs(floored - before)
            assert np.all((change - allowed - rounding)[both] <= 0.0), k
        previous = reads
    floor_bound = previous[0] > previous[1]
    assert floor_bound[previous[3]].any(), "the floor never binds on this sweep"
