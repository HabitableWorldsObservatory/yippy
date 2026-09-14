"""Yippy's public rotation utility must share the corrected transform geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yippy.jax_funcs import fft_rotate_jax


@pytest.fixture(autouse=True)
def enable_x64():
    """Check geometry with sufficient precision to expose sub-pixel offsets."""
    with jax.enable_x64():
        yield


@pytest.mark.parametrize("n", [63, 64])
def test_rotated_psf_centroid_and_angle_derivative(n):
    """Both odd and even PSFs retain astrometry through the public yippy API."""
    y, x = jnp.mgrid[:n, :n] - (n - 1) / 2
    psf = jnp.exp(-((x - 6) ** 2 + (y + 4) ** 2) / 18)

    def centroid(angle):
        rotated = fft_rotate_jax(psf, angle)
        return (rotated * y).sum() / rotated.sum()

    np.testing.assert_allclose(centroid(30.0), 3 - 2 * np.sqrt(3), atol=1e-8)
    np.testing.assert_allclose(jax.grad(centroid)(0.0), np.pi / 30, atol=1e-9)


def test_optical_center_and_padding_options():
    """An explicit optical center is carried through the yippy entry point."""
    image = jnp.zeros((32, 32)).at[16, 20].set(1.0)
    rotated = fft_rotate_jax(image, 90.0, center=(16.0, 16.0), pad_factor=1.5)
    expected = np.zeros((32, 32))
    expected[20, 16] = 1.0
    np.testing.assert_allclose(rotated, expected, atol=1e-12)
