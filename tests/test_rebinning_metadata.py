"""Rebinned PSFs require matching coordinate metadata and fresh cache entries."""

import copy

import astropy.io.fits as fits
import numpy as np
from optixstuff.grids import Frame, PixelGrid

from yippy._offax import OffAx


def test_rebinning_transforms_optical_center(tmp_path):
    """The input optical center at index 3.5 maps to index 1.5 in 2x bins."""
    fits.writeto(tmp_path / "psfs.fits", np.ones((4, 8, 8)))
    fits.writeto(
        tmp_path / "offsets.fits",
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    offax = OffAx(
        fits.getdata(tmp_path / "psfs.fits"),
        fits.getdata(tmp_path / "offsets.fits"),
        x_symmetric=True,
        y_symmetric=True,
        downsample_shape=(4, 4),
        grid=PixelGrid.focal(8, 0.1, Frame(500, 6, "circumscribed")),
    )
    assert offax.grid.center_x == 1.5
    assert offax.grid.center_y == 1.5


def test_optical_center_axes_are_not_transposed(tmp_path):
    """center_x measures columns and center_y measures rows, not the reverse.

    A square PSF stack cannot see a row/column transposition, so this uses 8
    rows by 12 columns. ``center_x`` feeds ``PixelGrid.pixel_to_unit`` for an
    x position, so it must come from the column count. Tolerance basis: exact halves.
    """
    fits.writeto(tmp_path / "psfs.fits", np.ones((4, 8, 12)))
    fits.writeto(
        tmp_path / "offsets.fits",
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    offax = OffAx(
        fits.getdata(tmp_path / "psfs.fits"),
        fits.getdata(tmp_path / "offsets.fits"),
        x_symmetric=True,
        y_symmetric=True,
        grid=PixelGrid.regular(
            (8, 12), 0.1, "lod", frame=Frame(500, 6, "circumscribed")
        ),
    )
    assert offax.grid.center_x == 5.5
    assert offax.grid.center_y == 3.5


def test_performance_cache_distinguishes_rebinned_psfs(coro):
    """Changing detector bins changes the aperture calculation's cache entry."""
    from yippy._yip import _YIPData

    host = _YIPData(coro.package)
    reduced = copy.copy(host)
    reduced.npixels = host.npixels // 2
    assert reduced._perf_filename() != host._perf_filename()


def test_datacube_cache_does_not_reuse_pre_rebinning_schema(coro):
    """Cubes made with center-sampled PSFs cannot be reused for integrated bins."""
    from yippy._yip import _YIPData

    host = _YIPData(coro.package)
    current = host._datacube_cache_path
    parts = current.name.split("_")
    dtype_tag = next(part for part in parts if part in ("f32", "f64"))
    old = (
        f"psf_datacube_quarter_{dtype_tag}_{host.npixels}px_"
        f"{host.package.source.digest}.npy"
    )
    assert current.name != old
