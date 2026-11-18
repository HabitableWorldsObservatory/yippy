"""FITS byte order must be converted before JAX PSF downsampling."""

import astropy.io.fits as fits
import jax
import numpy as np
import pytest
from optixstuff.grids import Frame, PixelGrid

from yippy.header import HeaderData
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
    # The parent reader bypassed metadata; it supplied no default PIXSCALE.
    # A physical grid now needs it, unless an explicit grid is supplied.
    with pytest.raises(ValueError, match="PIXSCALE"):
        HeaderData.from_fits_header(fits.getheader(tmp_path / "psfs.fits")).pixel_grid()
    result = OffAx(
        fits.getdata(tmp_path / "psfs.fits"),
        fits.getdata(tmp_path / "offsets.fits"),
        x_symmetric=True,
        y_symmetric=True,
        downsample_shape=(4, 4),
        grid=PixelGrid.focal(8, 0.1, Frame(500, 6, "circumscribed")),
    )
    assert result.flat_psfs.shape == (4, 4, 4)
    assert np.isfinite(result.flat_psfs).all()
