"""FITS byte order must be converted before JAX PSF downsampling."""

import astropy.io.fits as fits
import astropy.units as u
import jax
import numpy as np
import pytest

from yippy.offax import OffAx


@pytest.fixture(autouse=True)
def enable_x64():
    """Keep the FITS precision test independent of other modules' array dtypes."""
    with jax.enable_x64():
        yield


def test_downsample_fits_psfs(tmp_path):
    """Exercise the same big-endian FITS boundary as the EAC1 YIP."""
    psfs = np.ones((4, 8, 8), dtype=np.float64) / 64
    offsets = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fits.writeto(tmp_path / "psfs.fits", psfs)
    fits.writeto(tmp_path / "offsets.fits", offsets)
    assert not fits.getdata(tmp_path / "psfs.fits").dtype.isnative
    result = OffAx(
        tmp_path,
        "psfs.fits",
        "offsets.fits",
        0.1 * u.arcsec / u.pix,
        True,
        True,
        downsample_shape=(4, 4),
    )
    assert result.flat_psfs.shape == (4, 4, 4)
    assert np.isfinite(result.flat_psfs).all()
