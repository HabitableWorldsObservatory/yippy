"""Rebinned PSFs require matching coordinate metadata and fresh cache entries."""

import copy

import astropy.io.fits as fits
import astropy.units as u
import numpy as np

from yippy.offax import OffAx


def test_rebinning_transforms_optical_center(tmp_path):
    """The input optical center at index 4 maps to index 1.75 in 2x bins."""
    fits.writeto(tmp_path / "psfs.fits", np.ones((4, 8, 8)))
    fits.writeto(
        tmp_path / "offsets.fits",
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    offax = OffAx(
        tmp_path,
        "psfs.fits",
        "offsets.fits",
        0.1 * u.arcsec / u.pix,
        True,
        True,
        downsample_shape=(4, 4),
    )
    assert offax.center_x.to_value(u.pix) == 1.75
    assert offax.center_y.to_value(u.pix) == 1.75


def test_optical_center_axes_are_not_transposed(tmp_path):
    """center_x measures columns and center_y measures rows, not the reverse.

    A square PSF stack cannot see a row/column transposition, so this uses 8
    rows by 12 columns. ``center_x`` feeds ``convert_to_lod`` for an x position,
    so it must come from the column count. Tolerance basis: alg (exact halves).
    """
    fits.writeto(tmp_path / "psfs.fits", np.ones((4, 8, 12)))
    fits.writeto(
        tmp_path / "offsets.fits",
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    offax = OffAx(
        tmp_path,
        "psfs.fits",
        "offsets.fits",
        0.1 * u.arcsec / u.pix,
        True,
        True,
    )
    assert offax.center_x.to_value(u.pix) == 6.0
    assert offax.center_y.to_value(u.pix) == 4.0


def test_performance_cache_distinguishes_rebinned_psfs(coro):
    """Changing detector bins changes the aperture calculation's cache entry."""
    reduced = copy.copy(coro)
    reduced.npixels = coro.npixels // 2
    assert reduced._perf_filename() != coro._perf_filename()


def test_datacube_cache_does_not_reuse_pre_rebinning_schema(coro):
    """Cubes made with center-sampled PSFs cannot be reused for integrated bins."""
    current = coro._datacube_cache_path
    parts = current.name.split("_")
    dtype_tag = next(part for part in parts if part in ("f32", "f64"))
    legacy = (
        f"psf_datacube_quarter_{dtype_tag}_{coro.npixels}px_"
        f"{coro._source_signature()}.npy"
    )
    assert current.name != legacy
