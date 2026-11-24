"""Check zero-valued stellar samples and the positive-only interpolation path."""

import hashlib

import jax
import numpy as np
import pytest
from astropy.io import fits
from lod_unit import lod
from synthetic_yip import STELLAR_LEVEL, write_synthetic_yip

from yippy import open_yip

# SHA-256 of each full native float32 map in row-major, little-endian order.
# Recorded on CPU at these diameter knots from yippy 98cb161.
EAC1_MAP_DIGESTS = {
    0.0: "d14552b3f287087fe795f9a790a6b7a2d7785dd689c9785c753aff49d101d4ec",
    0.001: "ef28b185432fdc327cb0b652ddf26e5a1ead46ddb7a21c55e9b63ea2f42c2db5",
    0.12589254117941676: (
        "0909db50336b0d06f31ae50105c98895dfcef124cdbbaf919a59663318374ec3"
    ),
}


@pytest.fixture(scope="module")
def zero_coro(tmp_path_factory):
    """Two diameter knots with a rising, dark, and falling pixel."""
    directory = tmp_path_factory.mktemp("stellar_zeros")
    # A float32-representable log keeps rounding small on the background pixels.
    write_synthetic_yip(
        directory, npix=24, max_offset_lod=2.0, stellar_level=np.exp(-16.0)
    )
    with fits.open(directory / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data = hdus[0].data[:2].copy()
        hdus[0].data[0, 0, 0] = 0
        hdus[0].data[1, 0, 0] = STELLAR_LEVEL
        hdus[0].data[:, 0, 1] = 0
        hdus[0].data[:, 0, 2] = [5.6379300493792775e-11, 0]
        hdus[0].data[:, 0, 3] = np.exp([-16.0, -15.0])
    fits.writeto(
        directory / "stellar_intens_diam_list.fits",
        np.array([0.0, 0.1]),
        overwrite=True,
    )
    with jax.enable_x64(False):
        return open_yip(directory).prepare()


def test_zero_stellar_samples_are_preserved(zero_coro):
    """Use the delivered samples as truth, with 1e-6 for float32 rounding."""
    host = zero_coro._host.stellar_intens
    diameters = np.array([0.0, 0.05, 0.1])
    expected = host(diameters * lod).integrated().data
    actual = zero_coro.stellar_map(None, diameters).integrated().data
    for maps in (expected, actual):
        np.testing.assert_array_equal(maps[:, 0, 1], 0)
        assert maps[0, 0, 0] == 0
        np.testing.assert_allclose(maps[2, 0, 0], STELLAR_LEVEL, rtol=1e-6, atol=0)
        assert 0 < maps[1, 0, 0] < STELLAR_LEVEL
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(
        host.diams.value, zero_coro.package.stellar_diameters_lod
    )
    np.testing.assert_array_equal(host.ln_interp.x, host.diams.value)
    np.testing.assert_array_equal(host.psfs, zero_coro.package.stellar_intensity)
    np.testing.assert_array_equal(
        np.argwhere(host.zero_pixels), [[0, 0], [0, 1], [0, 2]]
    )


def test_final_knot_keeps_a_zero_sample_exact(zero_coro):
    """Endpoint rounding must preserve a delivered zero after a positive sample."""
    host = zero_coro._host.stellar_intens
    endpoint = host.diams[-1]
    assert host(endpoint).data[0, 2] == 0
    assert zero_coro.stellar_map(None, endpoint.value).data[0, 2] == 0


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_masked_stellar_pixels_are_constant_outside_knots(zero_coro, backend):
    """Both ends stay nonnegative and constant for pixels with a zero knot."""
    diameters = np.array([-0.02, -0.01, 0.0, 0.1, 0.45, 2.0])
    if backend == "scipy":
        host = zero_coro._host.stellar_intens
        pixels = host._evaluate_array(diameters * lod)[:, 0, :4]
    else:

        def read(diameter):
            return zero_coro.stellar_map(None, diameter).integrated().data[..., 0, :4]

        with jax.enable_x64(False):
            pixels = np.asarray(jax.jit(read)(diameters))
            gradients = np.asarray(jax.vmap(jax.jacfwd(read))(diameters))
        assert np.all(np.isfinite(gradients))
        np.testing.assert_array_equal(gradients[[0, 1, 4, 5], :3], 0)
    assert np.all(pixels[:, :3] >= 0)
    np.testing.assert_array_equal(pixels[[0, 1, 4, 5], :3], pixels[[2, 2, 3, 3], :3])
    # The positive control pixel still extrapolates.
    assert pixels[0, 3] < pixels[2, 3]
    assert pixels[5, 3] > pixels[3, 3]


def test_final_knot_keeps_a_small_positive_sample_exact(tmp_path):
    """Keep the last sample exact after a bright neighboring knot."""
    write_synthetic_yip(tmp_path, npix=24, max_offset_lod=2.0)
    with fits.open(tmp_path / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[:, 0, 0] = [0.0, 1e-2, 1e-12]

    with jax.enable_x64(False):
        coro = open_yip(tmp_path).prepare()
        read = jax.jit(lambda d: coro.stellar_map(None, d).integrated().data[0, 0])
        final = read(np.float32(0.2))
    assert final.dtype == np.float32
    assert final == np.float32(1e-12)
    assert coro._host.stellar_intens._evaluate_array(0.2 * lod)[0, 0] == 1e-12


def test_subnormal_float32_stellar_sample_uses_pchip(tmp_path):
    """A positive float32 subnormal needs the linear-intensity spline."""
    write_synthetic_yip(tmp_path, npix=24, max_offset_lod=2.0)
    with fits.open(tmp_path / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[:, 0, 0] = [1e-2, 1e-39, 1e-2]

    with jax.enable_x64(False):
        coro = open_yip(tmp_path).prepare()

        def read(diameter):
            return coro.stellar_map(None, diameter).integrated().data[0, 0]

        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(read)))(
            np.array([0.0, 0.05, 0.1, 0.15, 0.2], dtype=np.float32)
        )
    assert not coro._host.stellar_intens.zero_pixels[0, 0]
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    np.testing.assert_allclose(np.asarray(values)[[1, 3]], 2.5e-3, rtol=1e-6)


def test_stellar_diameter_gradient_is_finite(zero_coro):
    """NaNs in the unselected log spline must not contaminate the gradient."""

    def total(diameter_lod):
        return zero_coro.stellar_map(None, diameter_lod).data.sum()

    assert np.isfinite(jax.grad(total)(0.05))


@pytest.mark.parametrize("sample", [-1e-10, np.nan])
def test_invalid_stellar_sample_names_the_file(tmp_path, sample):
    """A negative or nonfinite sample is rejected with its source file in the error."""
    write_synthetic_yip(tmp_path, npix=24, max_offset_lod=2.0)
    with fits.open(tmp_path / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[0, 0, 0] = sample
    package = open_yip(tmp_path)
    with pytest.raises(ValueError, match=r"stellar_intens\.fits"):
        package.prepare()


def test_positive_eac1_stellar_map_matches_base_bitwise(shipped_coro):
    """Check three full maps against the stored float32 reference from 98cb161."""
    assert not np.any(shipped_coro._host.stellar_intens.zero_pixels)
    with jax.enable_x64(False):
        for diameter, digest in EAC1_MAP_DIGESTS.items():
            image = shipped_coro.stellar_map(None, diameter)
            data = np.asarray(image.data)
            assert data.dtype == np.float32
            assert (
                hashlib.sha256(data.astype("<f4", copy=False).tobytes()).hexdigest()
                == digest
            )
    assert shipped_coro._stellar_linear_interp is None
