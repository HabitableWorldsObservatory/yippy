"""Unit tests for yippy's FFT-based PSF interpolation pipeline.

Tests the interpolation-specific behavior: flux conservation through
the full pipeline, extrapolation, continuous sweeps, dark hole contrast,
shift masking, and nearest neighbor lookup.

Core FFT shift tests (fidelity, Parseval, clamp) live in
hwoutils/tests/test_fft.py.
"""

import numpy as np
import pytest
from conftest import requires_available_yip

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def offax(coro):
    """The OffAx/OffJAX object from the session coronagraph."""
    return coro._offaxis_backend


@pytest.fixture(scope="session")
def sample_psf(offax):
    """A single reference PSF from the middle of the offset grid."""
    mid = len(offax.x_offsets) // 2
    idx = offax.offset_to_flat_idx[mid, 0]
    return np.array(offax.flat_psfs[int(idx)])


@pytest.fixture(scope="session")
def grid_offsets(offax):
    """All x-offsets as a numpy array."""
    return np.array(offax.x_offsets)


# ---------------------------------------------------------------------------
# Test 2: Flux conservation through the full pipeline
# ---------------------------------------------------------------------------


@requires_available_yip
class TestFluxConservation:
    """Verify total flux is preserved through interpolation."""

    def test_grid_point_flux(self, coro, offax, grid_offsets):
        """Interpolation at a grid point must return exact flux match."""
        for i in range(min(5, len(grid_offsets))):
            x_val = float(grid_offsets[i])
            if x_val < 0.5:
                continue
            ref_idx = int(offax.offset_to_flat_idx[i, 0])
            ref_psf = np.array(offax.flat_psfs[ref_idx])
            ref_flux = np.sum(ref_psf)

            interp_psf = np.array(offax.create_psf(x_val, 0.0).data)
            interp_flux = np.sum(interp_psf)
            np.testing.assert_allclose(
                interp_flux,
                ref_flux,
                rtol=0.01,
                err_msg=f"Flux mismatch at grid point x={x_val}",
            )

    def test_midpoint_flux_bounded(self, offax, grid_offsets):
        """Flux at midpoints should be within 20% of neighbor range."""
        for i in range(min(5, len(grid_offsets) - 1)):
            x_lo = float(grid_offsets[i])
            x_hi = float(grid_offsets[i + 1])
            if x_lo < 0.5:
                continue
            x_mid = (x_lo + x_hi) / 2.0

            idx_lo = int(offax.offset_to_flat_idx[i, 0])
            idx_hi = int(offax.offset_to_flat_idx[i + 1, 0])
            flux_lo = float(np.sum(np.array(offax.flat_psfs[idx_lo])))
            flux_hi = float(np.sum(np.array(offax.flat_psfs[idx_hi])))

            interp_psf = np.array(offax.create_psf(x_mid, 0.0).data)
            interp_flux = float(np.sum(interp_psf))

            flux_min = min(flux_lo, flux_hi) * 0.8
            flux_max = max(flux_lo, flux_hi) * 1.2
            assert flux_min <= interp_flux <= flux_max, (
                f"Midpoint flux {interp_flux:.6f} outside range "
                f"[{flux_min:.6f}, {flux_max:.6f}] at x={x_mid}"
            )


# ---------------------------------------------------------------------------
# Test 10: Dark hole contrast preservation
# ---------------------------------------------------------------------------


@requires_available_yip
class TestDarkHoleContrast:
    """Verify interpolation does not inject artifacts into the dark hole."""

    def test_dark_hole_intensity_stable(self, offax, grid_offsets):
        """Mean intensity in the 3-10 lam/D annulus of an interpolated PSF.

        Must not exceed that of the neighboring grid-point PSFs.
        This test ensures that the interpolation process does not introduce
        significant artifacts that would artificially brighten the dark hole
        region compared to the PSFs at the original grid points.
        """
        pixel_scale_arcsec = float(offax.grid.step)

        # Pick a midpoint between two grid points
        mid_idx = len(grid_offsets) // 2
        x_lo = float(grid_offsets[mid_idx])
        x_hi = float(grid_offsets[mid_idx + 1])
        x_mid = (x_lo + x_hi) / 2.0

        # Get the grid-point reference PSFs
        ref_lo = np.array(offax.create_psf(x_lo, 0.0).data)
        ref_hi = np.array(offax.create_psf(x_hi, 0.0).data)

        # Get the interpolated midpoint PSF
        interp_psf = np.array(offax.create_psf(x_mid, 0.0).data)

        ny, nx = interp_psf.shape
        cy, cx = ny // 2, nx // 2
        yy, xx = np.mgrid[:ny, :nx]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * pixel_scale_arcsec

        annulus = (rr >= 3.0) & (rr <= 10.0)
        if not np.any(annulus):
            pytest.skip("PSF too small for 3-10 lam/D annulus")

        baseline = max(
            np.mean(ref_lo[annulus]),
            np.mean(ref_hi[annulus]),
        )
        interp_mean = np.mean(interp_psf[annulus])

        # Interpolated dark hole should not be significantly brighter
        # than the brightest neighbor
        assert interp_mean < baseline * 1.5, (
            f"Interpolated dark hole mean ({interp_mean:.2e}) exceeds "
            f"1.5x neighbor baseline ({baseline:.2e})"
        )

    def test_synthesized_psf_is_finite(self, offax, grid_offsets):
        """Synthesized PSF at a midpoint must be finite and positive."""
        x_lo = float(grid_offsets[len(grid_offsets) // 2])
        x_hi = float(grid_offsets[len(grid_offsets) // 2 + 1])
        x_mid = (x_lo + x_hi) / 2.0

        psf = np.array(offax.create_psf(x_mid, 0.0).data)
        assert np.all(np.isfinite(psf))
        assert np.sum(psf) > 0


# ---------------------------------------------------------------------------
# Test 12: create_shift_mask boundary logic
# ---------------------------------------------------------------------------


class TestCreateShiftMask:
    """Verify branchless masking for shifted PSFs."""

    def test_positive_x_shift(self):
        """Positive x-shift should mask the left columns."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 2.3, 0.0, x_grid, y_grid, fill_val=1.0)

        # ceil(2.3) = 3, so left 3 columns should be zero
        assert float(jnp.sum(mask[:, :3])) == 0.0
        # rest should be 1.0
        assert float(jnp.min(mask[:, 3:])) == 1.0

    def test_negative_y_shift(self):
        """Negative y-shift should mask the bottom rows."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 0.0, -1.5, x_grid, y_grid, fill_val=1.0)

        # ceil(1.5) = 2, so bottom 2 rows should be zero
        assert float(jnp.sum(mask[8:, :])) == 0.0
        # rest should be 1.0
        assert float(jnp.min(mask[:8, :])) == 1.0

    def test_fill_value_applied(self):
        """Fill value should be used for valid pixels."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 0.0, 0.0, x_grid, y_grid, fill_val=3.5)
        assert float(jnp.min(mask)) == 3.5

    def test_combined_shift(self):
        """Positive x and negative y shifts mask correct edges."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 2.3, -1.5, x_grid, y_grid, fill_val=2.0)

        # Left 3 cols masked
        assert float(jnp.sum(mask[:, :3])) == 0.0
        # Bottom 2 rows masked
        assert float(jnp.sum(mask[8:, :])) == 0.0
        # Interior should be fill_val
        assert float(mask[5, 5]) == 2.0


# ---------------------------------------------------------------------------
# Test 13: Nearest neighbor index lookup
# ---------------------------------------------------------------------------


class TestNearestNeighborLookup:
    """Verify index clamping in get_near_inds_offsets_1D and _2D."""

    def test_1d_interior_point(self):
        """Query between two grid points returns correct bounding indices."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_near_inds_offsets_1D

        x_offsets = jnp.array([0.0, 1.0, 2.0, 3.0])
        y_offsets = jnp.array([0.0])

        inds, offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, 1.2, 0.0)

        # Should bound between index 1 (x=1.0) and index 2 (x=2.0)
        assert int(inds[0, 0]) == 1
        assert int(inds[1, 0]) == 2
        np.testing.assert_allclose(float(offsets[0, 0]), 1.0)
        np.testing.assert_allclose(float(offsets[1, 0]), 2.0)

    def test_1d_boundary_clamp(self):
        """Query at the left edge should clamp to index 0."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_near_inds_offsets_1D

        x_offsets = jnp.array([1.0, 2.0, 3.0])
        y_offsets = jnp.array([0.0])

        inds, _offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, 0.5, 0.0)

        # Both should clamp to index 0
        assert int(inds[0, 0]) == 0
        assert int(inds[1, 0]) == 0

    def test_2d_interior_point(self):
        """Query in the interior of a 2D grid returns 4 bounding corners."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_near_inds_offsets_2D

        x_offs = jnp.array([0.0, 1.0, 2.0])
        y_offs = jnp.array([0.0, 1.0, 2.0])

        inds, offsets = get_near_inds_offsets_2D(x_offs, y_offs, 1.2, 0.8)

        # Should return 4 corners: (1,0), (1,1), (2,0), (2,1)
        assert inds.shape == (4, 2)
        assert offsets.shape == (4, 2)
        # x indices should be 1 and 2
        assert int(inds[0, 0]) == 1  # x_low
        assert int(inds[2, 0]) == 2  # x_high
        # y indices should be 0 and 1
        assert int(inds[0, 1]) == 0  # y_low
        assert int(inds[1, 1]) == 1  # y_high
