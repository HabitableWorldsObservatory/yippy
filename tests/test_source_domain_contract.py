"""A source outside the source domain is zero and flagged, for every yippy reader.

The contract is optixstuff's (see AbstractImagingCoronagraph): SourcePSFs.valid
is False for a source outside the implementer's source domain, with a finite
all-zero image, and off_axis_psf returns zeros rather than NaN.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from optixstuff.grids import PixelGrid
from sinc_fixture import save_manifest, write_delivery
from synthetic_yip import write_synthetic_yip

from yippy import open_yip


@pytest.fixture(autouse=True, scope="module")
def _clear_caches():
    yield
    jax.clear_caches()


def _yippy(kind):
    def build(tmp_path_factory):
        root = tmp_path_factory.mktemp(f"yip_{kind}")
        written = write_synthetic_yip(root / "package", kind=kind)
        # The package tabulates offsets out to 6 lambda/D.
        return open_yip(written.path).prepare(), (1.3 * 6.0, 0.0)

    return build


def _yippy_sinc(tmp_path_factory):
    root = tmp_path_factory.mktemp("yip_sinc")
    metadata = write_delivery(root)
    metadata["checks"] = []
    metadata["served"]["bounds_lod"] = [[-0.2, 0.2], [-0.2, 0.5]]
    save_manifest(root, metadata)
    with fits.open(root / "legacy" / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data *= 1 + 0.2 * np.cos(2 * np.pi * (np.arange(4) - 1.5) / 4)
    # The served box reaches 0.5 lambda/D along y; 1.3 times that is outside.
    return open_yip(root).prepare(interpolation="sinc"), (0.0, 1.3 * 0.5)


IMPLEMENTERS = {
    "yippy_1d": _yippy("1d"),
    "yippy_2d": _yippy("2dq"),
    "yippy_sinc": _yippy_sinc,
}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            name,
            marks=(
                [pytest.mark.float64_only(reason="YIP sinc construction requires x64")]
                if name == "yippy_sinc"
                else []
            ),
        )
        for name in IMPLEMENTERS
    ],
)
def implementer(request, tmp_path_factory):
    """One yippy reader and a source position outside its domain."""
    return IMPLEMENTERS[request.param](tmp_path_factory)


def test_source_outside_the_domain_is_zero_and_flagged(implementer):
    """Fails if a source beyond the data gives valid True, NaN or nonzero light.

    A source at 1.3 times the source domain has valid False for that source
    alone, a finite image of exact zeros, and an off-axis PSF of zeros, not
    NaN. A second, inside source in the same call keeps its own flag.
    """
    coro, (x_out, y_out) = implementer
    wavelength = coro.frame.reference_wavelength_nm
    grid = PixelGrid.focal(16, 0.5, coro.frame)
    result = coro.source_psfs(
        wavelength,
        jnp.array([0.0, x_out]),
        jnp.array([0.0, y_out]),
        grid=grid,
    )
    np.testing.assert_array_equal(np.asarray(result.valid), [True, False])
    image = np.asarray(result.image.data)
    assert np.isfinite(image).all()
    assert (image[1] == 0.0).all()
    separation = float(np.hypot(x_out, y_out))
    off_axis = np.asarray(coro.off_axis_psf(wavelength, separation, grid).data)
    assert np.isfinite(off_axis).all() and (off_axis == 0.0).all()
