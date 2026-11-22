"""Stellar and sky maps hold only what the table holds: zero past its grid.

A product resampled onto a grid twice the table must not repeat the table
(a mirror copy of the dark hole would score as light the data never measured).
The comparison value is the same resample with the reflected boundary, so the
test asserts what changed and what did not.
"""

import dataclasses

import numpy as np
import pytest
from optixstuff.grids import PixelGrid, coverage, reconstruction_margin, resample

from yippy import open_yip


@pytest.fixture(scope="module")
def coro(synthetic_2d):
    """Coronagraph over the quarterly symmetric analytic package."""
    return open_yip(synthetic_2d.path).prepare()


def _wide_grid(grid, factor=2):
    n = grid.shape[0] * factor
    return PixelGrid.regular(
        (n, n), grid.step, grid.unit, frame=grid.frame, orientation=grid.orientation
    )


def _products(coro):
    wavelength = coro.frame.reference_wavelength_nm
    return {
        "stellar": coro._native_stellar_map(0.0),
        "sky": coro.sky_trans,
    }, wavelength


@pytest.mark.parametrize("name", ["stellar", "sky"])
def test_map_is_zero_past_the_table_and_unchanged_two_pixels_inside(coro, name):
    """Fails if the map is reflected past the table or its interior moves.

    Outside the table (pixels whose footprint is wholly past the grid) the
    resample is exactly 0. Two or more source pixels inside it equals the
    reflected resample to 1e-12. The outermost cell moves with the boundary
    and is not asserted.
    """
    products, _ = _products(coro)
    image = products[name]
    assert image.outside == "zero"
    wide = _wide_grid(image.grid)
    zero = np.asarray(resample(image, wide).data)
    reflect = np.asarray(
        resample(dataclasses.replace(image, outside="reflect"), wide).data
    )
    xs, ys = (np.asarray(a) for a in wide.coordinates())
    half = image.grid.shape[0] * image.grid.step / 2
    wholly_outside = (np.abs(xs) > half + image.grid.step / 2 + 1e-9) | (
        np.abs(ys) > half + image.grid.step / 2 + 1e-9
    )
    assert wholly_outside.sum() > 0
    assert (zero[wholly_outside] == 0.0).all()
    assert (reflect[wholly_outside] != 0.0).any()
    margin = max(2.0, reconstruction_margin(image.reconstruction))
    inside = coverage(image.grid, wide, margin)
    assert inside.sum() > 0
    np.testing.assert_allclose(zero[inside], reflect[inside], rtol=1e-12, atol=0.0)
