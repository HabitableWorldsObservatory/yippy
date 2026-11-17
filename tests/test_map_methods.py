"""Tests for record evaluation and optixstuff radial map projections.

Uses the session-scoped ``coro`` fixture from conftest.py.
"""

import numpy as np
import pytest
from conftest import requires_available_yip
from optixstuff import YippyCoronagraph

from yippy import EqxCoronagraph
from yippy.performance import stellar_intensity_curves

pytestmark = requires_available_yip


class TestSeparationMap:
    """Tests for radial separation_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        assert r.shape == (coro.npixels, coro.npixels)

    def test_center_is_zero(self, coro):
        """Separation at the grid center is zero."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        assert r[cy, cx] == pytest.approx(0, abs=coro.pixel_scale_arcsec.value)

    def test_radially_symmetric(self, coro):
        """Values at equal pixel distance from center should be equal."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 10
        assert r[cy, cx + d] == pytest.approx(r[cy + d, cx], rel=1e-10)
        # The axis sits between pixels cx - 1 and cx, so cx + d mirrors to
        # cx - 1 - d.
        assert r[cy, cx + d] == pytest.approx(r[cy, cx - 1 - d], rel=1e-10)

    def test_all_nonnegative(self, coro):
        """All separations are non-negative."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        assert np.all(r >= 0)


class TestCoreMeanIntensityMap:
    """Tests for radial core_mean_intensity_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
        ).evaluate(
            "core_mean_intensity", np.hypot(*coro.stellar_intens.grid.coordinates())
        )
        assert m.shape == (coro.npixels, coro.npixels)

    def test_radially_symmetric(self, coro):
        """Output should be perfectly radially symmetric."""
        m = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
        ).evaluate(
            "core_mean_intensity", np.hypot(*coro.stellar_intens.grid.coordinates())
        )
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 10
        assert m[cy, cx + d] == pytest.approx(m[cy + d, cx], rel=1e-10)
        assert m[cy, cx + d] == pytest.approx(m[cy - 1 - d, cx], rel=1e-10)

    def test_positive_values(self, coro):
        """All values are non-negative."""
        m = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
        ).evaluate(
            "core_mean_intensity", np.hypot(*coro.stellar_intens.grid.coordinates())
        )
        assert np.all(m >= 0)

    def test_matches_1d(self, coro):
        """Map values should match point-query at the same separation."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        m = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
        ).evaluate(
            "core_mean_intensity", np.hypot(*coro.stellar_intens.grid.coordinates())
        )
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 15
        sep = r[cy, cx + d]
        expected = stellar_intensity_curves(
            EqxCoronagraph(yippy_coro=coro), 0.0, interpolation="keys"
        ).evaluate("core_mean_intensity", sep)
        assert m[cy, cx + d] == pytest.approx(expected, rel=1e-10)


class TestNoiseFloorAyoMap:
    """Post-processing residual density maps use leakage_density_lod2 / ppf."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = (
            YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro))
            .curves()
            .evaluate("leakage_density_lod2", np.hypot(*coro.grid.coordinates()))
            / 30.0
        )
        assert m.shape == (coro.npixels, coro.npixels)

    def test_ppf_scaling(self, coro):
        """Result should scale inversely with ppf."""
        m30 = (
            YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro))
            .curves()
            .evaluate("leakage_density_lod2", np.hypot(*coro.grid.coordinates()))
            / 30.0
        )
        m60 = (
            YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro))
            .curves()
            .evaluate("leakage_density_lod2", np.hypot(*coro.grid.coordinates()))
            / 60.0
        )
        np.testing.assert_allclose(m30, 2.0 * m60, rtol=1e-12)

    def test_equals_istar_over_ppf(self, coro):
        """Residual density is exactly contract leakage density divided by ppf."""
        ppf = 25.0
        istar = (
            YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro))
            .curves()
            .evaluate("leakage_density_lod2", np.hypot(*coro.grid.coordinates()))
        )
        nf = (
            YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro))
            .curves()
            .evaluate("leakage_density_lod2", np.hypot(*coro.grid.coordinates()))
            / ppf
        )
        np.testing.assert_allclose(nf, istar / ppf, rtol=1e-12)


class TestThroughputMap:
    """Tests for radial throughput_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).throughput_map()
        assert m.shape == (coro.npixels, coro.npixels)

    def test_bounded(self, coro):
        """All values are in [0, 1]."""
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).throughput_map()
        assert np.all(m >= 0)
        assert np.all(m <= 1.0)

    def test_radially_symmetric(self, coro):
        """Equidistant points from center have equal throughput."""
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).throughput_map()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 10
        assert m[cy, cx + d] == pytest.approx(m[cy + d, cx], rel=1e-10)

    def test_matches_1d(self, coro):
        """Map values match the 1D interpolator at the same separation."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).throughput_map()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 15
        sep = r[cy, cx + d]
        expected = coro.performance.evaluate("throughput", sep)
        assert m[cy, cx + d] == pytest.approx(expected, rel=1e-10)


class TestCoreAreaMap:
    """Tests for radial core_area_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).core_area_map()
        assert m.shape == (coro.npixels, coro.npixels)

    def test_positive(self, coro):
        """All core areas are non-negative."""
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).core_area_map()
        assert np.all(m >= 0)

    def test_matches_1d(self, coro):
        """Map values match the 1D interpolator at the same separation."""
        r = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).separation_map()
        m = YippyCoronagraph(backend=EqxCoronagraph(yippy_coro=coro)).core_area_map()
        cy, cx = round(coro.grid.center_y), round(coro.grid.center_x)
        d = 15
        sep = r[cy, cx + d]
        expected = coro.performance.evaluate("core_area_lod2", sep)
        assert m[cy, cx + d] == pytest.approx(expected, rel=1e-10)
