"""Tests for yippy.performance -- coronagraph performance curve computation.

Uses a real coronagraph YIP via the session-scoped ``coro`` fixture from
conftest.py (downloaded via pooch).
"""

import numpy as np
import pytest
from conftest import requires_available_yip
from lod_unit import lod
from scipy.interpolate import PchipInterpolator, make_interp_spline

from yippy import Circular, Coronagraph, HalfMaximum, PerformanceCurves
from yippy.performance import (
    _collect_and_sort,
    _oversample_psf,
    _threshold_mask,
    compute_all_performance_curves,
    compute_core_area_curve,
    compute_core_mean_intensity_curve,
    compute_occ_trans_curve,
    compute_raw_contrast_curve,
    compute_throughput_curve,
    compute_truncation_core_area_curve,
    compute_truncation_throughput_curve,
)

# =============================================================================
# Helpers
# =============================================================================


class TestCollectAndSort:
    """Tests for _collect_and_sort helper."""

    def test_sorts_ascending(self):
        """Ascending sort by separation."""
        sep, val = _collect_and_sort([3, 1, 2], [30, 10, 20])
        np.testing.assert_array_equal(sep, [1, 2, 3])
        np.testing.assert_array_equal(val, [10, 20, 30])

    def test_empty_input(self):
        """Empty input returns empty arrays."""
        sep, val = _collect_and_sort([], [])
        assert len(sep) == 0
        assert len(val) == 0


class TestOversamplePSF:
    """Tests for _oversample_psf flux conservation."""

    def test_flux_conservation(self):
        """Total flux should be preserved after oversampling."""
        # Use a Gaussian PSF (compact, away from edges) for tight test
        N = 32
        c = (N - 1) / 2.0
        y, x = np.mgrid[:N, :N]
        psf = np.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * 3.0**2))
        psf /= psf.sum()

        psf_os = _oversample_psf(psf, pixel_scale_arcsec=0.3, oversample=3)

        # Shape should be 3x in each dimension
        assert psf_os.shape == (96, 96)
        # Gaussian is compact so flux conservation should be very tight
        np.testing.assert_allclose(psf_os.sum(), psf.sum(), rtol=0.01)

    def test_no_negative_values(self):
        """Output should have no negative values."""
        psf = np.random.RandomState(42).rand(16, 16) - 0.3  # has negatives
        psf_os = _oversample_psf(psf, pixel_scale_arcsec=0.5, oversample=2)
        assert np.all(psf_os >= 0)


class TestThresholdMask:
    """Tests for _threshold_mask helper."""

    def test_basic_thresholding(self):
        """Mask should include pixels above ratio * peak."""
        image = np.array([[0, 0, 0], [0, 10, 5], [0, 0, 0]], dtype=float)
        mask = _threshold_mask(image, 0.4)
        assert mask[1, 1]  # peak (10) > 0.4*10
        assert mask[1, 2]  # 5 > 4
        assert not mask[0, 0]  # 0 < 4

    def test_fallback_to_peak(self):
        """When threshold is too high, fallback to peak-only mask."""
        image = np.array([[1, 1], [1, 2]], dtype=float)
        mask = _threshold_mask(image, 0.99)
        # Only the peak pixel should be in the mask
        assert mask.sum() == 1
        assert mask[1, 1]


# =============================================================================
# Performance Curves (require real coronagraph data)
# =============================================================================


@requires_available_yip
class TestThroughputCurve:
    """Tests for throughput curve computation."""

    def test_values_bounded(self, coro):
        """Throughput values should be in [0, 1]."""
        _sep, throughput = compute_throughput_curve(coro)
        assert np.all(throughput >= 0)
        assert np.all(throughput <= 1.0)

    def test_separations_sorted(self, coro):
        """Separations should be sorted ascending."""
        sep, _ = compute_throughput_curve(coro)
        assert np.all(np.diff(sep) >= 0)


@requires_available_yip
class TestRawContrastCurve:
    """Tests for raw contrast curve computation."""

    def test_positive_values(self, coro):
        """Raw contrast should be positive."""
        _sep, contrast = compute_raw_contrast_curve(coro)
        assert np.all(contrast >= 0)


@requires_available_yip
class TestTruncationThroughputCurve:
    """Tests for truncation throughput curve."""

    def test_values_bounded(self, coro):
        """Truncation throughput should be in [0, 1]."""
        _sep, throughput = compute_truncation_throughput_curve(
            coro, rule=HalfMaximum(0.5)
        )
        assert np.all(throughput >= 0)
        assert np.all(throughput <= 1.0)


@requires_available_yip
class TestTruncationCoreAreaCurve:
    """Tests for truncation core area curve."""

    def test_positive_areas(self, coro):
        """Core areas should be positive."""
        _sep, core_area = compute_truncation_core_area_curve(
            coro, rule=HalfMaximum(0.5)
        )
        assert np.all(core_area > 0)


@requires_available_yip
class TestCoreAreaCurve:
    """Tests for fixed-aperture core area curve."""

    def test_constant_for_fixed_aperture(self, coro):
        """Without Gaussian fitting, core area should be constant."""
        _sep, core_area = compute_core_area_curve(coro, fit_gaussian=False)
        expected = np.pi * 0.7**2  # exact fractional disk area
        np.testing.assert_allclose(core_area, expected, rtol=1e-6)


@requires_available_yip
class TestOccTransCurve:
    """Tests for occulter transmission curve."""

    def test_values_bounded(self, coro):
        """Occulter transmission should be in [0, 1]."""
        _sep, occ_trans = compute_occ_trans_curve(coro)
        assert np.all(occ_trans >= 0)
        assert np.all(occ_trans <= 1.0 + 1e-6)


@requires_available_yip
class TestCoreMeanIntensityCurve:
    """Tests for core mean intensity curve."""

    def test_returns_all_diameters(self, coro):
        """Should return profiles for all available stellar diameters."""
        sep, intensities = compute_core_mean_intensity_curve(coro)
        assert len(intensities) == len(coro.stellar_intens.diams)
        for _diam, profile in intensities.items():
            assert len(profile) == len(sep)


@requires_available_yip
class TestComputeAllPerformanceCurves:
    """Integration tests for the orchestrator."""

    def test_iwa_within_range(self, coro):
        """IWA should be within the separation range."""
        result = compute_all_performance_curves(coro, save_to_fits=False)
        iwa = result.iwa_lod
        sep = result.separation_lod
        assert iwa >= sep[0]
        assert iwa <= sep[-1]

    def test_contrast_floor_applied(self, coro):
        """When contrast_floor is set, contrast should respect it."""
        floor = 1e-9
        coro.contrast_floor = floor
        result = compute_all_performance_curves(coro, save_to_fits=False)
        assert np.all(coro.raw_contrast(result.separation_lod) >= floor)
        assert np.any(result.raw_contrast < floor)
        coro.contrast_floor = None  # clean up


@pytest.fixture(scope="module")
def performance_interpolation(synthetic_1d):
    """The one record evaluator on real measured tables."""
    coro = Coronagraph(synthetic_1d.path)
    return coro, coro.performance


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
        values, nodes = getattr(record, name), record.separation_lod
        is_log = record.interpolation[name] == "log"
        y = np.log10(np.maximum(values, 1e-300)) if is_log else values
        grid = np.array([nodes[0] - 0.1, nodes[-1] + 0.1, np.nan])
        expected = make_interp_spline(nodes, y, k=1)(grid)
        if is_log:
            expected = 10**expected
        np.testing.assert_allclose(record.evaluate(name, grid), expected, rtol=1e-13)
    assert np.isnan(coro.core_mean_intensity(2.0, stellar_diam=0.3))


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
    identity, filename = coro._auto_perf_identity(), coro._perf_filename()
    original = record.throughput.copy()
    record.evaluate("throughput", np.linspace(0.1, 5.1, 101))
    assert coro._auto_perf_identity() == identity
    assert coro._perf_filename() == filename
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
    coro = Coronagraph(yip.path, rule=rule)
    data = coro.compute_all_performance_curves(save_to_fits=False)

    zero = np.argmin(data.separation_lod)
    assert data.separation_lod[zero] == 0.0
    assert data.raw_contrast[zero] == 1e20
    assert np.all(np.isfinite(data.raw_contrast))
    assert float(coro.throughput(0.0 * lod)) == pytest.approx(0.0, abs=1e-12)
    far = 5.0 * lod
    # A Gaussian half-maximum core encloses (1-fraction) of its total flux;
    # the circle instead uses the package's analytic encircled-energy formula.
    expected = (
        yip.psf_flux * yip.occulter(5.0) * (1 - rule.fraction)
        if isinstance(rule, HalfMaximum)
        else yip.expected_throughput(5.0)
    )
    np.testing.assert_allclose(float(coro.throughput(far)), expected, rtol=0.05)
    assert np.isfinite(float(coro.raw_contrast(far)))
