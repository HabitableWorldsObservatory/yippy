"""Tests for prepared YIP curves and retained stellar-product measurements.

Uses a real coronagraph YIP via the session-scoped ``coro`` fixture from
conftest.py (downloaded via pooch).
"""

import jax
import numpy as np
import pytest
from conftest import requires_available_yip
from optixstuff.apertures import Circular, HalfMaximum
from optixstuff.performance import PerformanceCurves
from scipy.interpolate import PchipInterpolator, make_interp_spline

from yippy import Coronagraph
from yippy._yip import _YIPData
from yippy.performance import stellar_intensity_curves

# =============================================================================
# Performance Curves (require real coronagraph data)
# =============================================================================


@requires_available_yip
class TestThroughputCurve:
    """Tests for throughput curve computation."""

    def test_values_bounded(self, coro):
        """Throughput values should be in [0, 1]."""
        throughput = coro.curves().throughput
        assert np.all(throughput >= 0)
        assert np.all(throughput <= 1.0)

    def test_separations_sorted(self, coro):
        """Separations should be sorted ascending."""
        sep = coro.curves().separation_lod
        assert np.all(np.diff(sep) >= 0)


@requires_available_yip
class TestRawContrastCurve:
    """Tests for raw contrast curve computation."""

    def test_positive_values(self, coro):
        """Raw contrast should be positive."""
        contrast = coro.curves().raw_contrast
        assert np.all(contrast >= 0)


@requires_available_yip
class TestTruncationThroughputCurve:
    """Tests for truncation throughput curve."""

    def test_values_bounded(self, coro):
        """Truncation throughput should be in [0, 1]."""
        rule = HalfMaximum(0.5)
        prepared = coro.with_rules((rule,))
        throughput = prepared.curves(rule).throughput
        assert np.all(throughput >= 0)
        assert np.all(throughput <= 1.0)
        constructed = Coronagraph(coro.yip_path, rules=(rule,)).curves(rule)
        np.testing.assert_array_equal(throughput, constructed.throughput)


@requires_available_yip
class TestTruncationCoreAreaCurve:
    """Tests for truncation core area curve."""

    def test_positive_areas(self, coro):
        """Core areas should be positive."""
        rule = HalfMaximum(0.5)
        prepared = coro.with_rules((rule,))
        core_area = prepared.curves(rule).core_area_lod2
        assert np.all(core_area > 0)


@requires_available_yip
class TestCoreAreaCurve:
    """Tests for fixed-aperture core area curve."""

    def test_constant_for_fixed_aperture(self, coro):
        """An in-image circle has area pi r^2 within fractional-cell roundoff."""
        record = coro.curves()
        x, y = coro.grid.coordinates()
        inside = (record.separation_lod - 0.7 >= np.min(x)) & (
            record.separation_lod + 0.7 <= np.max(x)
        )
        assert np.min(y) < -0.7 < 0.7 < np.max(y)
        core_area = record.core_area_lod2[inside]
        expected = np.pi * 0.7**2  # exact fractional disk area
        np.testing.assert_allclose(core_area, expected, rtol=1e-6)


class TestOccTransCurve:
    """Sky transmission retains its independent radial-bin measurement."""

    def test_values_bounded(self, syn_coro_1d):
        """Fails if radial bins, interpolation, or transmission units change.

        Independent NumPy bin means and scipy PCHIP allow 16 stored-dtype
        eps for radial geometry and fitted-coefficient rounding.
        """
        coro = syn_coro_1d
        image = np.asarray(coro.sky_trans.data)
        grid = coro.sky_trans.grid
        y, x = np.indices(grid.shape)
        radius = np.hypot(y - grid.center_y, x - grid.center_x)
        bins = np.linspace(0, radius.max(), max(grid.shape) // 2 + 1)
        index = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, len(bins) - 2)
        expected = np.bincount(index, weights=image.ravel()) / np.bincount(index)
        nodes = (bins[:-1] + bins[1:]) / 2 * grid.step
        queries = np.sort(np.r_[nodes, (nodes[1:] + nodes[:-1]) / 2])
        actual = coro.occulter_transmission(queries)
        tol = 16 * np.finfo(np.asarray(coro.sky_trans.data).dtype).eps
        np.testing.assert_allclose(
            actual, PchipInterpolator(nodes, expected)(queries), rtol=tol
        )
        assert np.all(actual >= 0)
        assert np.all(actual <= 1.0 + 1e-6)


class TestCoreMeanIntensityCurve:
    """All diameter profiles retain the analytic stellar pixel intensity."""

    def test_returns_all_diameters(self, performance_interpolation, synthetic_1d):
        """Fails if a diameter is lost or a profile is converted to area density.

        The synthetic stellar maps are constant at stellar_level per pixel.
        Every ring and its PCHIP interpolant have that same constant value;
        The log/exp round trip amplifies ordinate rounding by abs(log(level));
        four such errors allow spline arithmetic and ring summation.
        """
        with jax.enable_x64():
            coro, _ = performance_interpolation
            diameters = np.r_[0.0, np.asarray(coro._stellar_ln_interp.x)[1:]]
            assert len(diameters) == len(coro._stellar_ln_interp.x)
            for diameter in diameters:
                record = stellar_intensity_curves(coro, diameter)
                nodes = record.separation_lod
                queries = np.sort(np.r_[nodes, (nodes[1:] + nodes[:-1]) / 2])
                actual = record.evaluate("core_mean_intensity", queries)
                tol = (
                    4
                    * max(1.0, abs(np.log(synthetic_1d.stellar_level)))
                    * np.finfo(np.asarray(coro._stellar_ln_interp.c).dtype).eps
                )
                np.testing.assert_allclose(actual, synthetic_1d.stellar_level, rtol=tol)


@requires_available_yip
class TestPreparedPerformanceCurves:
    """Integration tests for prepared curve records."""

    def test_iwa_within_range(self, coro):
        """IWA should be within the separation range."""
        result = coro.curves()
        iwa = result.iwa_lod
        sep = result.separation_lod
        assert iwa >= sep[0]
        assert iwa <= sep[-1]

    def test_contrast_floor_applied(self, coro):
        """An explicit engineering floor bounds contrast without changing the record."""
        floor = 1e-9
        result = coro.curves()
        assert np.all(
            np.maximum(result.evaluate("raw_contrast", result.separation_lod), floor)
            >= floor
        )
        assert np.any(result.raw_contrast < floor)


@pytest.fixture(scope="module")
def performance_interpolation(synthetic_1d):
    """The one record evaluator on real measured tables."""
    with jax.enable_x64():
        coro = Coronagraph(synthetic_1d.path)
    return (coro, coro.curves())


def test_performance_interpolation_default_and_legacy(performance_interpolation):
    """Fails if a curve's PCHIP or declared interpolation space changes.

    Independent scipy evaluation allows 1e-13 relative polynomial roundoff.
    The removed configurable-order API raises rather than aliasing.
    """
    coro, record = performance_interpolation
    with pytest.raises(TypeError, match="interp_order"):
        Coronagraph(coro.yip_path, interp_order=1)
    for name in (
        "throughput",
        "core_area_lod2",
        "raw_contrast",
        "leakage_fraction",
        "leakage_density_lod2",
    ):
        values = getattr(record, name)
        is_log = record.interpolation[name] == "log"
        y = np.log10(np.maximum(values, 1e-300)) if is_log else values
        nodes = record.separation_lod
        grid = np.linspace(nodes[0], nodes[-1], 301)
        expected = PchipInterpolator(nodes, y)(grid)
        if is_log:
            expected = 10**expected
        np.testing.assert_allclose(
            record.evaluate(name, grid), expected, rtol=1e-13, atol=0
        )


def test_performance_interpolation_linear_tails(performance_interpolation):
    """Fails if tail extrapolation uses the end cubic (float64 bound 1e-13)."""
    coro, record = performance_interpolation
    for name in ("throughput", "core_area_lod2", "raw_contrast"):
        values, nodes = (
            np.asarray(getattr(record, name)),
            np.asarray(record.separation_lod),
        )
        is_log = record.interpolation[name] == "log"
        y = np.log10(np.maximum(values, 1e-300)) if is_log else values
        grid = np.array([nodes[0] - 0.1, nodes[-1] + 0.1, np.nan])
        expected = make_interp_spline(nodes, y, k=1)(grid)
        if is_log:
            expected = 10**expected
        np.testing.assert_allclose(record.evaluate(name, grid), expected, rtol=1e-13)
    with jax.enable_x64():
        assert np.isnan(
            stellar_intensity_curves(coro, 0.3, interpolation="keys").evaluate(
                "core_mean_intensity", 2.0
            )
        )


def test_default_preserves_a_steep_monotone_rise():
    """Fails if cubic overshoot replaces shape-preserving interpolation."""
    from hwoutils.grids import Frame

    nodes = np.r_[np.linspace(0, 4, 17), 6, 9, 14]
    values = 1 / (1 + np.exp(-10 * (nodes - 3)))
    grid = np.linspace(nodes[0], nodes[-1], 10001)
    record = PerformanceCurves(
        nodes,
        Frame(500, 6, "circumscribed"),
        Circular(0.7),
        values,
        np.ones_like(values),
        values,
        values,
        values,
        3.0,
        14.0,
    )
    got = record.evaluate("throughput", grid)
    wrong = make_interp_spline(nodes, values, k=3)(grid)
    assert np.all(got >= 0)
    assert np.all(np.diff(got) >= -np.finfo(float).eps)
    assert np.any(np.diff(wrong) < 0)
    assert np.any(wrong < 0)


def test_interpolation_does_not_change_cache_identity(performance_interpolation):
    """Off-grid evaluation cannot mutate the measured table or its identity."""
    coro, record = performance_interpolation
    host = _YIPData(coro.yip_path)
    identity, filename = host._auto_perf_identity(), host._perf_filename()
    original = record.throughput.copy()
    record.evaluate("throughput", np.linspace(0.1, 5.1, 101))
    assert host._auto_perf_identity() == identity
    assert host._perf_filename() == filename
    np.testing.assert_array_equal(original, record.throughput)


@pytest.mark.parametrize("rule", [Circular(0.7), HalfMaximum(0.3)])
def test_zero_on_axis_psf_builds_finite_curves(tmp_path, rule):
    """An exactly zero off-axis image gets the finite contrast sentinel.

    An ideal occulter can transmit nothing for a source on the optical axis.
    Before the sentinel, that offset produced an infinite raw contrast and the
    interpolator constructors in scipy rejected the table.
    """
    from synthetic_yip import write_synthetic_yip

    yip = write_synthetic_yip(tmp_path / "zero_floor", occulter_floor=0.0)
    coro = Coronagraph(yip.path, rules=(rule,))
    data = coro.curves(rule)

    zero = np.argmin(data.separation_lod)
    assert data.separation_lod[zero] == 0.0
    assert data.raw_contrast[zero] == 1e20
    assert np.all(np.isfinite(data.raw_contrast))
    assert float(coro.curves(rule).evaluate("throughput", 0.0)) == pytest.approx(
        0.0, abs=1e-12
    )
    far = 5.0
    # A Gaussian half-maximum core encloses (1-fraction) of its total flux;
    # the circle instead uses the package's analytic encircled-energy formula.
    expected = (
        yip.psf_flux * yip.occulter(5.0) * (1 - rule.fraction)
        if isinstance(rule, HalfMaximum)
        else yip.expected_throughput(5.0)
    )
    np.testing.assert_allclose(
        float(coro.curves(rule).evaluate("throughput", far)), expected, rtol=0.05
    )
    assert np.isfinite(float(coro.curves(rule).evaluate("raw_contrast", far)))
