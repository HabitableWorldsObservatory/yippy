"""Tests for record evaluation and optixstuff radial map projections.

Uses the session-scoped ``coro`` fixture from conftest.py.
"""

import numpy as np
import pytest
from conftest import requires_available_yip

from yippy.performance import stellar_intensity_curves

pytestmark = requires_available_yip


class TestSeparationMap:
    """Tests for the radial map of the native grid."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        r = coro.grid.radius()
        assert r.shape == (coro.grid.shape[0], coro.grid.shape[0])

    def test_center_is_zero(self, coro):
        """Separation at the grid center is zero."""
        r = coro.grid.radius()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        assert r[cy, cx] == pytest.approx(0, abs=coro.grid.step)

    def test_radially_symmetric(self, coro):
        """Values at equal pixel distance from center should be equal."""
        r = coro.grid.radius()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 10
        assert r[cy, cx + d] == pytest.approx(np.asarray(r[cy + d, cx]), rel=1e-10)
        # The axis sits between pixels cx - 1 and cx, so cx + d mirrors to
        # cx - 1 - d.
        assert r[cy, cx + d] == pytest.approx(np.asarray(r[cy, cx - 1 - d]), rel=1e-10)

    def test_all_nonnegative(self, coro):
        """All separations are non-negative."""
        r = coro.grid.radius()
        assert np.all(r >= 0)


class TestCoreMeanIntensityMap:
    """Tests for radial core_mean_intensity_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = stellar_intensity_curves(coro, 0.0, interpolation="keys").evaluate(
            np.hypot(*coro.stellar_grid.coordinates())
        )
        assert m.shape == (coro.grid.shape[0], coro.grid.shape[0])

    def test_radially_symmetric(self, coro):
        """Output should be perfectly radially symmetric."""
        m = stellar_intensity_curves(coro, 0.0, interpolation="keys").evaluate(
            np.hypot(*coro.stellar_grid.coordinates())
        )
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 10
        assert m[cy, cx + d] == pytest.approx(np.asarray(m[cy + d, cx]), rel=1e-10)
        assert m[cy, cx + d] == pytest.approx(np.asarray(m[cy - 1 - d, cx]), rel=1e-10)

    def test_positive_values(self, coro):
        """All values are non-negative."""
        m = stellar_intensity_curves(coro, 0.0, interpolation="keys").evaluate(
            np.hypot(*coro.stellar_grid.coordinates())
        )
        assert np.all(m >= 0)

    def test_matches_1d(self, coro):
        """Map values should match point-query at the same separation."""
        r = coro.grid.radius()
        m = stellar_intensity_curves(coro, 0.0, interpolation="keys").evaluate(
            np.hypot(*coro.stellar_grid.coordinates())
        )
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 15
        sep = r[cy, cx + d]
        expected = stellar_intensity_curves(coro, 0.0, interpolation="keys").evaluate(
            sep
        )
        assert m[cy, cx + d] == pytest.approx(np.asarray(expected), rel=1e-10)


class TestNoiseFloorAyoMap:
    """Post-processing residual density maps use leakage_density_lod2 / ppf."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = (
            coro.curves().evaluate(
                "leakage_density_lod2", np.hypot(*coro.grid.coordinates())
            )
            / 30.0
        )
        assert m.shape == (coro.grid.shape[0], coro.grid.shape[0])

    def test_ppf_scaling(self, coro):
        """Result should scale inversely with ppf."""
        m30 = (
            coro.curves().evaluate(
                "leakage_density_lod2", np.hypot(*coro.grid.coordinates())
            )
            / 30.0
        )
        m60 = (
            coro.curves().evaluate(
                "leakage_density_lod2", np.hypot(*coro.grid.coordinates())
            )
            / 60.0
        )
        np.testing.assert_allclose(m30, 2.0 * m60, rtol=1e-12)

    def test_equals_istar_over_ppf(self, coro):
        """Residual density is exactly contract leakage density divided by ppf."""
        ppf = 25.0
        istar = coro.curves().evaluate(
            "leakage_density_lod2", np.hypot(*coro.grid.coordinates())
        )
        nf = (
            coro.curves().evaluate(
                "leakage_density_lod2", np.hypot(*coro.grid.coordinates())
            )
            / ppf
        )
        np.testing.assert_allclose(nf, istar / ppf, rtol=1e-12)
