"""Analytic circle means that detect coordinate, scale and normalization errors."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from scipy.special import i0

from yippy.performance import azimuthal_mean_profile, stellar_axis_index

# Share compilation across cases to keep this analytic unit file inexpensive.
_circle_mean = jax.jit(azimuthal_mean_profile)


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
