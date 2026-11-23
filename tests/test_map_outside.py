"""Stellar and sky maps hold only what the table holds: zero past its grid.

A product resampled onto a grid twice the table must not repeat the table
(a mirror copy of the dark hole would score as light the data never measured).
"""

import numpy as np
import pytest
from optixstuff.grids import PixelGrid, resample

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
def test_map_is_zero_past_the_table(coro, name):
    """Fails if the map holds light past the table.

    Outside the table (pixels whose footprint is wholly past the grid) the
    resample is exactly 0.
    """
    products, _ = _products(coro)
    image = products[name]
    wide = _wide_grid(image.grid)
    zero = np.asarray(resample(image, wide).data)
    xs, ys = (np.asarray(a) for a in wide.coordinates())
    half = image.grid.shape[0] * image.grid.step / 2
    wholly_outside = (np.abs(xs) > half + image.grid.step / 2 + 1e-9) | (
        np.abs(ys) > half + image.grid.step / 2 + 1e-9
    )
    assert wholly_outside.sum() > 0
    assert (zero[wholly_outside] == 0.0).all()
