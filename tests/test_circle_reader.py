"""Analytic circle means that detect coordinate, scale and normalization errors."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import i0, j1

from yippy.performance import azimuthal_mean_profile, stellar_axis_index

# Share compilation across cases to keep this analytic unit file inexpensive.
_circle_mean = jax.jit(azimuthal_mean_profile)
_selectable_circle_mean = eqx.filter_jit(azimuthal_mean_profile)


def test_keys_bit_identical():
    """Pin float64 CPU bits captured before adding the quintic reader."""
    y, x = np.indices((16, 19))
    image = ((7 * x + 11 * y) % 23) / 32.0
    radii = np.array([0, 0.3, 2.25, 6.5, 9, 11.25, 14, 20])
    expected = np.array(
        [
            4602838248358674432,
            4602361469332650918,
            4599785533096927238,
            4599657244376053817,
            4600173806480846896,
            4598346295681330144,
            0,
            0,
        ],
        dtype=np.uint64,
    )
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        default = np.asarray(_circle_mean(image, (6.25, 9.5), radii))
        explicit = np.asarray(
            _selectable_circle_mean(
                image, jnp.array([6.25, 9.5]), radii, interpolation="keys"
            )
        )
    np.testing.assert_array_equal(default.view(np.uint64), expected)
    np.testing.assert_array_equal(explicit.view(np.uint64), expected)


@pytest.mark.parametrize("center", [(63.5, 63.5), (59.25, 67.75)])
def test_quadratic_circle_mean(center):
    """Preserve the axis, row/column order, radius scale and pixel-area units."""
    pixel_scale = 0.25
    y, x = (np.indices((128, 128)) - 63.5) * pixel_scale
    # Linear and unequal quadratic terms expose swapped axes/offset centers.
    density = 2 + 0.1 * x + 0.2 * y + 0.3 * x**2 + 0.7 * y**2
    radii_lod = np.array([0, 3, 3.03125, 5, 10])
    cy, cx = (np.asarray(center) - 63.5) * pixel_scale
    expected = 2 + 0.1 * cx + 0.2 * cy + 0.3 * cx**2 + 0.7 * cy**2
    expected = expected + radii_lod**2 / 2
    coro = SimpleNamespace(
        stellar_intens=SimpleNamespace(
            center_y=center[0] + 0.5, center_x=center[1] + 0.5
        )
    )
    assert stellar_axis_index(coro) == center
    with jax.enable_x64():
        actual = np.asarray(
            _circle_mean(
                density * pixel_scale**2,
                stellar_axis_index(coro),
                radii_lod / pixel_scale,
            )
        )
    np.testing.assert_allclose(actual / pixel_scale**2, expected, rtol=1e-12)


def test_displaced_gaussian_circle_mean():
    """A smooth asymmetric map agrees with its closed-form full-circle mean."""
    pixel_scale = 0.25
    y, x = (np.indices((128, 128)) - 63.5) * pixel_scale
    image = np.exp(-((x - 1) ** 2 + (y + 2) ** 2) / 18)
    radii = np.array([0, 3, 3.03125, 5, 10])
    expected = np.exp(-(radii**2 + 5) / 18) * i0(radii * np.sqrt(5) / 9)
    with jax.enable_x64():
        actual = np.asarray(_circle_mean(image, (63.5, 63.5), radii / pixel_scale))
    # An accuracy requirement against the analytic mean, not stored reader values.
    np.testing.assert_allclose(actual, expected, rtol=5e-5)


def test_partial_circle_normalizes_only_in_image_arcs():
    """A constant stays constant on short corner arcs and vanishes outside."""
    corner = np.hypot(63.5, 63.5)
    radii = np.array([0, 64, 80, corner - 0.001, corner, corner + 0.001])
    with jax.enable_x64():
        actual = np.asarray(_circle_mean(np.ones((128, 128)), (63.5, 63.5), radii))
    np.testing.assert_allclose(actual, [1, 1, 1, 1, 1, 0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pixel_scale", [0.25, 0.125])
@pytest.mark.parametrize("kind", ["airy", "gaussian"])
def test_quintic_analytic_circle_mean(pixel_scale, kind):
    """Quintic reaches twice the measured spline error at both pixel scales."""
    size = round(32 / pixel_scale)
    center = (size - 1) / 2
    y, x = (np.indices((size, size)) - center) * pixel_scale
    radii = np.array([3.0, 5.0, 10.0])
    if kind == "airy":
        z = np.pi * np.hypot(y, x)
        image = (2 * j1(z) / z) ** 2
        expected = (2 * j1(np.pi * radii) / (np.pi * radii)) ** 2
        bounds = (
            np.array([0.0844264, 0.0672181, 0.0387699])
            if pixel_scale == 0.25
            else np.array([0.000600227, 0.000424109, 0.000228349])
        )
    else:
        image = np.exp(-((x - 1) ** 2 + (y + 2) ** 2) / 18)
        expected = np.exp(-(radii**2 + 5) / 18) * i0(radii * np.sqrt(5) / 9)
        bounds = (
            np.array([2.91480e-9, 1.07434e-8, 6.25964e-8])
            if pixel_scale == 0.25
            else np.array([4.75842e-11, 1.60183e-10, 9.59399e-10])
        )
    with jax.enable_x64():
        actual = np.asarray(
            _selectable_circle_mean(
                image,
                (center, center),
                radii / pixel_scale,
                n_theta=2048,
                interpolation="quintic",
            )
        )
    assert np.all(100 * np.abs(actual / expected - 1) < 2 * bounds)


def test_quintic_partial_circle_and_tangency():
    """Quintic preserves in-image normalization, tangencies and exterior zero."""
    corner = np.hypot(7.5, 9.5)
    with jax.enable_x64():
        actual = np.asarray(
            _selectable_circle_mean(
                np.ones((16, 20)),
                (7.5, 9.5),
                np.array([0, 8, 10, corner - 0.001, corner, corner + 0.001]),
                interpolation="quintic",
            )
        )
    np.testing.assert_allclose(actual, [1, 1, 1, 1, 1, 0], rtol=1e-12, atol=1e-12)


def test_quintic_image_gradient():
    """The prefilter and spatial evaluation participate in reverse-mode AD."""

    def total(image):
        return azimuthal_mean_profile(
            image,
            (7.5, 9.5),
            jnp.array([0.0, 2.3, 9.0, 20.0]),
            interpolation="quintic",
        ).sum()

    with jax.enable_x64():
        image = jnp.arange(320, dtype=float).reshape(16, 20) / 320
        gradient = np.asarray(eqx.filter_jit(jax.grad(total))(image))
    assert np.all(np.isfinite(gradient))
    # Three nonempty circles: adding a constant adds it to each mean.
    np.testing.assert_allclose(gradient.sum(), 3, rtol=1e-12)


def test_unknown_interpolation_rejected():
    """A misspelled option must not silently read a different interpolant."""
    with pytest.raises(ValueError, match="interpolation"):
        azimuthal_mean_profile(
            np.ones((8, 8)), (3.5, 3.5), np.array([1]), interpolation="qunitic"
        )


def test_quintic_coordinate_gradients():
    """Radius and center derivatives agree with finite differences off tangencies."""
    from jax.test_util import check_grads

    y, x = np.indices((16, 20))
    image = np.exp(-((x - 10) ** 2 + (y - 6) ** 2) / 20)

    def read(parameters):
        return azimuthal_mean_profile(
            image,
            (parameters[0], parameters[1]),
            parameters[2:],
            interpolation="quintic",
        ).sum()

    with jax.enable_x64():
        parameters = jnp.array([7.3, 9.2, 2.3, 4.1])
        gradient = np.asarray(eqx.filter_jit(jax.grad(read))(parameters))
        assert np.all(np.isfinite(gradient))
        check_grads(
            eqx.filter_jit(read),
            (parameters,),
            order=1,
            modes=("rev",),
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )


def test_quintic_nearest_sample_extension():
    """Nonconstant edge samples match an independently prefiltered nearest halo."""
    y, x = np.indices((8, 11))
    image = ((7 * x + 11 * y) % 23) / 32.0
    centers = np.array([[0.2, 0.4], [0.1, 9.8], [6.9, 0.25], [6.8, 9.7]])
    # SciPy degree-5 spline_filter on samples edge-padded by 48 pixels,
    # evaluated with map_coordinates(order=5, prefilter=False), float64.
    expected = [
        0.14425356796312336,
        0.1354142496969667,
        0.29370149975107673,
        0.28342768901968746,
    ]

    @eqx.filter_jit
    def read(image, centers):
        return jax.vmap(
            lambda center: azimuthal_mean_profile(
                image,
                center,
                jnp.array([0.0]),
                interpolation="quintic",
            )[0]
        )(centers)

    with jax.enable_x64():
        actual = np.asarray(read(image, centers))
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 4)])
def test_quintic_interpolates_small_images(shape):
    """Degenerate and small rectangular images reproduce their pixel-center data."""
    image = np.arange(np.prod(shape), dtype=float).reshape(shape) / 7
    centers = np.indices(shape).reshape(2, -1).T

    @eqx.filter_jit
    def read(image, centers):
        return jax.vmap(
            lambda center: azimuthal_mean_profile(
                image,
                center,
                jnp.array([0.0]),
                interpolation="quintic",
            )[0]
        )(centers)

    with jax.enable_x64():
        actual = np.asarray(read(image, centers))
    np.testing.assert_allclose(actual, image.ravel(), rtol=1e-13, atol=1e-14)
