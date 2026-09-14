"""Verify detector rebinning and rotation on the cached EAC1 coronagraph PSFs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.transforms import downsample_psfs

from yippy.jax_funcs import fft_rotate_jax


@pytest.mark.parametrize("size", [64, 100])
def test_real_psf_downsampling_conserves_flux(coro, size):
    """Every stored PSF retains its integral at integer and noninteger ratios."""
    with jax.enable_x64():
        source = jnp.asarray(coro.offax.flat_psfs, dtype=jnp.float64)
        output, scale = downsample_psfs(source, 0.25, (size, size))
        np.testing.assert_allclose(
            output.sum(axis=(1, 2)), source.sum(axis=(1, 2)), rtol=1e-12, atol=1e-30
        )
        assert scale == 0.25 * source.shape[1] / size
        assert np.isfinite(output).all()
        assert float(output.min()) >= 0


def test_real_psf_rotation_padding_convergence(coro):
    """Increasing padding converges while core photometry stays stable."""
    with jax.enable_x64():
        source = jnp.asarray(coro.offax.flat_psfs[37], dtype=jnp.float64)
        outputs = [
            np.asarray(fft_rotate_jax(source, 30.0, pad_factor=p))
            for p in (0.5, 1.5, 3.5)
        ]
    reference = outputs[-1]
    n = reference.shape[0]
    y, x = np.mgrid[:n, :n]
    peak_y, peak_x = np.unravel_index(np.argmax(source), source.shape)
    cy = cx = (n - 1) / 2
    expected_y = cy + (peak_y - cy) * np.sqrt(3) / 2 + (peak_x - cx) / 2
    expected_x = cx + (peak_x - cx) * np.sqrt(3) / 2 - (peak_y - cy) / 2
    core = np.hypot(y - expected_y, x - expected_x) <= 8
    # The provisional photometric screen is 0.01%; full faint-wing accuracy
    # is measured separately and is not inferred from this aperture result.
    np.testing.assert_allclose(outputs[0][core].sum(), reference[core].sum(), rtol=1e-4)
    assert np.linalg.norm(outputs[1] - reference) < np.linalg.norm(
        outputs[0] - reference
    )
