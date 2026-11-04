"""Tests for yippy.performance -- coronagraph performance curve computation.

Uses a real coronagraph YIP via the session-scoped ``coro`` fixture from
conftest.py (downloaded via pooch).
"""

from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_available_yip
from lod_unit import lod
from scipy.interpolate import BSpline, PchipInterpolator, make_interp_spline

from yippy import Coronagraph
from yippy.performance import (
    _build_perf_interps,
    _collect_and_sort,
    _oversample_psf,
    _threshold_mask,
    compute_all_performance_curves,
    compute_core_area_curve,
    compute_core_mean_intensity_curve,
    compute_occ_trans_curve,
    compute_radial_average,
    compute_raw_contrast_curve,
    compute_throughput_curve,
    compute_truncation_core_area_curve,
    compute_truncation_throughput_curve,
    core_mean_intensity_profile,
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
            coro, psf_trunc_ratio=0.5
        )
        assert np.all(throughput >= 0)
        assert np.all(throughput <= 1.0)


@requires_available_yip
class TestTruncationCoreAreaCurve:
    """Tests for truncation core area curve."""

    def test_positive_areas(self, coro):
        """Core areas should be positive."""
        _sep, core_area = compute_truncation_core_area_curve(coro, psf_trunc_ratio=0.5)
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
        result = compute_all_performance_curves(coro, save_to_fits=False, plot=False)
        iwa = result["IWA"].value
        sep = result["separations"]
        assert iwa >= sep[0]
        assert iwa <= sep[-1]

    def test_contrast_floor_applied(self, coro):
        """When contrast_floor is set, contrast should respect it."""
        floor = 1e-9
        coro.contrast_floor = floor
        result = compute_all_performance_curves(coro, save_to_fits=False, plot=False)
        assert np.all(result["raw_contrast"] >= floor)
        coro.contrast_floor = None  # clean up


class TestRadialAverage:
    """Tests for the backward-compatible compute_radial_average wrapper."""

    def test_uniform_image(self):
        """Uniform image should have constant radial profile."""
        image = np.ones((32, 32)) * 5.0
        _sep, profile = compute_radial_average(image, pixel_scale_value=0.3)
        np.testing.assert_allclose(profile, 5.0, atol=0.5)


@pytest.fixture(scope="module", params=[None, 1])
def performance_interpolation(request, synthetic_1d):
    """Real measured tables with the default or legacy interpolators."""
    kwargs = {} if request.param is None else {"interp_order": request.param}
    coro = Coronagraph(synthetic_1d.path, **kwargs)
    constructor_interp = coro.throughput_interp
    data = coro.compute_all_performance_curves(save_to_fits=False, **kwargs)
    curves = [
        (
            coro.throughput_interp,
            data["separations_core_throughput"],
            data["core_throughput"],
        ),
        (coro.raw_contrast_interp, data["separations"], data["raw_contrast"]),
        (
            coro._log_contrast_interp,
            data["separations"],
            np.log10(np.abs(data["raw_contrast"]) + 1e-20),
        ),
        (coro.core_area_interp, data["separations_core_area"], data["core_area"]),
        (coro.occ_trans_interp, data["separations_occ_trans"], data["occ_trans"]),
        (
            coro.core_intensity_interp,
            data["separations_core_intensity"],
            data["core_intensities"][coro.stellar_intens.diams[0]],
        ),
        (
            coro._core_intensity_interp_at(0.05),
            coro.core_intensity_separations,
            core_mean_intensity_profile(coro, 0.05 * lod),
        ),
    ]
    return request.param, coro, constructor_interp, curves


def test_performance_interpolation_default_and_legacy(performance_interpolation):
    """All builders select PCHIP by default and preserve explicit linear values."""
    order, _, constructor_interp, curves = performance_interpolation
    assert isinstance(
        constructor_interp.pchip if order is None else constructor_interp,
        PchipInterpolator if order is None else BSpline,
    )
    for interp, nodes, values in curves:
        assert isinstance(
            interp.pchip if order is None else interp,
            PchipInterpolator if order is None else BSpline,
        )
        grid = np.linspace(nodes[0], nodes[-1], 301)
        if order is None:
            expected = PchipInterpolator(nodes, values)(grid)
            np.testing.assert_allclose(interp(grid), expected, rtol=0, atol=0)
        else:
            expected = make_interp_spline(nodes, values, k=1)(grid)
            np.testing.assert_array_equal(interp(grid), expected)


def test_performance_interpolation_linear_tails(performance_interpolation):
    """PCHIP keeps the old end-segment extrapolation for every separation curve."""
    _, coro, _, curves = performance_interpolation
    for interp, nodes, values in curves:
        grid = np.array(
            [nodes[0] - 3, nodes[0] - 0.1, nodes[-1] + 0.1, nodes[-1] + 3, np.nan]
        )
        expected = make_interp_spline(nodes, values, k=1)(grid)
        np.testing.assert_array_equal(interp(grid), expected)
    assert np.isnan(coro.core_mean_intensity(2.0, stellar_diam=0.3))


def test_default_preserves_a_steep_monotone_rise():
    """A regression to cubic splines produces negative values and turning points."""
    nodes = np.r_[np.linspace(0, 4, 17), 6, 9, 14]
    values = 1 / (1 + np.exp(-10 * (nodes - 3)))
    grid = np.linspace(nodes[0], nodes[-1], 10001)
    curves = []
    for order in [None, 3]:
        coro = SimpleNamespace()
        _build_perf_interps(coro, nodes, values, nodes, values, nodes, values, order)
        curves.append(coro.throughput_interp(grid))
    assert np.all(curves[0] >= 0)
    assert np.all(np.diff(curves[0]) >= -np.finfo(float).eps)
    assert np.any(np.diff(curves[1]) < 0)
    assert np.any(curves[1] < 0)


def test_interpolation_does_not_change_cache_identity(performance_interpolation):
    """Changing only interpolation reuses exactly the same measured-table cache."""
    _, coro, _, _ = performance_interpolation
    identity, filename = coro._auto_perf_identity(), coro._perf_filename()
    for order in [None, 1, 3]:
        other = copy(coro)
        other.interp_order = order
        assert other._auto_perf_identity() == identity
        assert other._perf_filename() == filename


@pytest.mark.parametrize("psf_trunc_ratio", [None, 0.3])
def test_zero_on_axis_psf_builds_finite_curves(tmp_path, psf_trunc_ratio):
    """An exactly zero off-axis image gets the finite contrast sentinel.

    An ideal occulter can transmit nothing for a source on the optical axis.
    Before the sentinel, that offset produced an infinite raw contrast and the
    interpolator constructors in scipy rejected the table.
    """
    from synthetic_yip import write_synthetic_yip

    yip = write_synthetic_yip(tmp_path / "zero_floor", occulter_floor=0.0)
    coro = Coronagraph(yip.path, psf_trunc_ratio=psf_trunc_ratio)
    data = coro.compute_all_performance_curves(save_to_fits=False)

    zero = np.argmin(data["separations"])
    assert data["separations"][zero] == 0.0
    assert data["raw_contrast"][zero] == 1e20
    assert np.all(np.isfinite(data["raw_contrast"]))
    assert float(coro.throughput(0.0 * lod)) == pytest.approx(0.0, abs=1e-12)
    far = 5.0 * lod
    np.testing.assert_allclose(
        float(coro.throughput(far)), yip.expected_throughput(5.0), rtol=0.05
    )
    assert np.isfinite(float(coro.raw_contrast(far)))
