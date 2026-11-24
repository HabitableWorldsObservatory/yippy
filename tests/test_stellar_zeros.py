"""Check zero-valued stellar samples and the positive-only interpolation path."""

import jax
import numpy as np
import pytest
from astropy.io import fits
from lod_unit import lod
from scipy.interpolate import CubicSpline, PchipInterpolator
from synthetic_yip import STELLAR_LEVEL, write_synthetic_yip

from yippy import open_yip


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
    with jax.enable_x64():
        read = jax.jit(lambda d: zero_coro.stellar_map(None, d).integrated().data[0, 2])
        value = np.asarray(read(np.float64(endpoint.value)))
    assert value == 0
    assert value.dtype == np.float32


@pytest.fixture(scope="module")
def three_knot_coro(tmp_path_factory):
    """Two zero minima next to a positive-only pixel with steep extrapolation."""
    directory = tmp_path_factory.mktemp("stellar_three_knots")
    write_synthetic_yip(directory, npix=24, max_offset_lod=2.0)
    with fits.open(directory / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[:, 0, 0] = [0.011384385, 0, 0.011384385]
        hdus[0].data[:, 0, 1] = [1e-35, 1e-20, 0.01]
        hdus[0].data[:, 0, 2] = [0.8047643574968019, 0, 0.8047643574968019]
        hdus[0].data[:, 0, 3] = [np.finfo(np.float32).tiny, 1e-20, 1e-10]
        hdus[0].data[:, 0, 4] = [1e-2, -0.0, 1e-2]
        hdus[0].data[:, 0, 5] = [0.0, 0.01, 0.02]
    with jax.enable_x64(False):
        return open_yip(directory).prepare()


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_pchip_stays_between_bracketing_samples(three_knot_coro, backend):
    """A query just below a zero knot can undershoot in either polynomial."""
    coro = three_knot_coro
    if backend == "scipy":
        knot = np.float64(0.1)
        near = np.nextafter(knot, np.float64(0))
        image = coro._host.stellar_intens(np.array([near, knot]) * lod)
        values = image.integrated().data
        column = 2
    else:
        knot = np.float32(0.1)
        near = np.nextafter(knot, np.float32(0))
        with jax.enable_x64(False):
            values = jax.jit(lambda d: coro.stellar_map(None, d).integrated().data)(
                np.array([near, knot])
            )
        column = 0
    sample = coro.package.stellar_intensity[0, 0, column]
    assert 0 <= values[0, 0, column] <= sample
    assert values[1, 0, column] == 0
    assert np.signbit(np.asarray(values)[1, 0, 4])


def test_bound_keeps_unit_slope_at_a_monotone_knot(three_knot_coro):
    """Selecting the bound must not halve the derivative where it is reached."""
    coro = three_knot_coro

    def read(d):
        return coro.stellar_map(None, d).integrated().data[0, 5]

    with jax.enable_x64(False):
        slope = jax.jit(jax.grad(read))
        at = float(slope(np.float32(0.1)))
        beside = float(slope(np.float32(0.0999)))
    assert at == pytest.approx(beside, rel=1e-2)


def test_stellar_zero_mask_is_static(three_knot_coro):
    """The host zero mask is shared metadata and absent from the pytree leaves."""
    coro = three_knot_coro
    expected = coro._host.stellar_intens.zero_pixels
    assert expected[0, 0] and not expected[0, 1]
    assert not any(
        getattr(leaf, "dtype", None) == np.dtype(bool)
        and np.shape(leaf) == expected.shape
        and np.array_equal(leaf, expected)
        for leaf in jax.tree.leaves(coro)
    )


def test_smallest_normal_stellar_sample_uses_log_spline(three_knot_coro):
    """The smallest normal float32 sample is positive in both evaluators."""
    coro = three_knot_coro
    samples = coro.package.stellar_intensity[:, 0, 3]
    diameters = coro.package.stellar_diameters_lod
    expected = np.exp(CubicSpline(diameters, np.log(samples))(0.05))
    host = coro._host.stellar_intens(0.05 * lod).integrated().data[0, 3]
    with jax.enable_x64(False):
        actual = coro.stellar_map(None, 0.05).integrated().data[0, 3]
    np.testing.assert_allclose([host, actual], expected, rtol=1e-5, atol=0)


def test_reverse_mode_ignores_overflow_in_adjacent_stellar_pixel(three_knot_coro):
    """The clamped masked pixel has zero derivative even beside an infinite one."""
    coro = three_knot_coro
    with jax.enable_x64(False):
        assert np.isposinf(coro.stellar_map(None, 1.0).data[0, 1])
        grad = jax.jit(jax.grad(lambda d: coro.stellar_map(None, d).data[0, 0]))
        assert grad(np.float32(1.0)) == 0


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


def test_subnormal_stellar_sample_requires_wider_working_dtype(tmp_path):
    """A positive 1e-39 sample is subnormal in float32 and normal in float64."""
    write_synthetic_yip(tmp_path, npix=24, max_offset_lod=2.0)
    with fits.open(tmp_path / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[:, 0, 0] = [1e-2, 1e-39, 1e-2]

    with jax.enable_x64(False):
        with pytest.raises(ValueError, match=r"1 positive stellar sample.*float32"):
            open_yip(tmp_path).prepare()
    with jax.enable_x64():
        coro = open_yip(tmp_path).prepare()
        assert coro._stellar_linear_interp is None
        image = coro.stellar_map(None, 0.1).integrated()
        value = np.asarray(image.data)[0, 0]
    assert value.dtype == np.float64
    assert value == np.float64(1e-39)


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


def test_positive_aavc_stellar_map_uses_log_cubic_spline(shipped_coro):
    """Check exact delivered knots and a pixel with a 74% log/PCHIP gap."""
    diameters = shipped_coro.package.stellar_diameters_lod
    delivered = shipped_coro.package.stellar_intensity
    diameter, row, column = 0.1, 133, 133
    samples = delivered[:, row, column].astype(np.float64)
    expected = np.exp(CubicSpline(diameters, np.log(samples))(diameter))
    linear = PchipInterpolator(diameters, samples)(diameter)
    # The measured relative gap is 0.7394, well above float32 rtol=2e-5.
    np.testing.assert_allclose(abs(linear - expected) / expected, 0.7394, atol=1e-4)

    with jax.enable_x64(False):
        knots = np.asarray(shipped_coro.stellar_map(None, diameters).integrated().data)
        expected_bits = delivered.astype(np.float32).view(np.uint32)
        assert knots.dtype == np.float32
        np.testing.assert_array_equal(knots.view(np.uint32), expected_bits)
        pixel = shipped_coro.stellar_map(None, diameter).integrated().data[row, column]
    np.testing.assert_allclose(pixel, expected, rtol=2e-5, atol=0)
