"""Performance quantities against a package whose answers are known in closed form.

Every assertion here computes its expected value from the constants in
``tests/synthetic_yip.py`` and elementary functions, never by calling the
reader's own conversion or by re-typing its implementation. A shipped package
cannot support tests like these: its correct answers are not known
independently of the code that produces them, so a test against one can only
pin current behavior.

The remaining aperture bias is midpoint intensity quadrature over fractional
boundary cells. An independent circle-square quadrature of the analytic
Gaussian predicts this small bias without calling the production integrator.
"""

import numpy as np
import pytest
from test_band_limited_apertures import reference_circle_weights

from yippy.performance import stellar_intensity_curves

APERTURE_RADIUS_LOD = 0.7

APERTURE_BIAS_TOL = 2e-5


@pytest.fixture(params=["1d", "2d"])
def kind(request):
    """Run the shared assertions over both package symmetries."""
    return request.param


@pytest.fixture
def package(kind, synthetic_1d, synthetic_2d):
    """The synthetic package of the requested symmetry."""
    return synthetic_1d if kind == "1d" else synthetic_2d


@pytest.fixture
def coro(kind, syn_coro_1d, syn_coro_2d):
    """The coronagraph built over the requested package."""
    return syn_coro_1d if kind == "1d" else syn_coro_2d


@pytest.fixture
def aperture_bias(package):
    """Fractional-circle midpoint error from the analytic Gaussian alone."""
    width = package.pixscale / 10
    half = int(np.ceil(APERTURE_RADIUS_LOD / width)) + 1
    rows, cols = np.mgrid[-half:half, -half:half] + 0.5
    weights = reference_circle_weights(rows, cols, APERTURE_RADIUS_LOD / width)
    sigma = package.psf_sigma_lod
    intensity = np.exp(-(rows**2 + cols**2) * width**2 / (2 * sigma**2)) / (
        2 * np.pi * sigma**2
    )
    measured = (weights * intensity).sum() * width**2
    analytic = 1 - np.exp(-(APERTURE_RADIUS_LOD**2) / (2 * sigma**2))
    return measured / analytic - 1


def test_the_aperture_bias_is_a_separation_independent_constant(
    package, coro, aperture_bias
):
    """Midpoint intensity quadrature error cannot vary with separation.

    The closed-form throughput is the occulter transmission times the encircled
    flux of a Gaussian, and the encircled-flux factor does not depend on
    separation. Any real separation dependence in the ratio would mean the
    reader is doing something to the separation that the closed form does not,
    which is the failure this test exists to catch. The size of the constant is
    the aperture-discretization bias every looser tolerance in this file is set
    from.
    """
    separations = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    got = np.array(
        [float(coro.curves().evaluate("throughput", s)) for s in separations]
    )
    expected = package.expected_throughput(separations, APERTURE_RADIUS_LOD)
    ratio = got / expected

    assert np.ptp(ratio) < 1e-4, f"the throughput bias varies with separation: {ratio}"
    assert abs(np.mean(ratio) - 1.0) < 1e-3
    assert abs(np.mean(ratio) - 1.0 - aperture_bias) < APERTURE_BIAS_TOL, (
        f"measured bias {np.mean(ratio) - 1.0:+.7f}, quadrature {aperture_bias:+.7f}"
    )


def test_throughput_matches_the_encircled_flux_closed_form(
    package, coro, aperture_bias
):
    """Throughput is the occulter transmission times a Gaussian's encircled flux."""
    for separation in (1.0, 2.5, 5.0):
        got = float(coro.curves().evaluate("throughput", separation))
        expected = package.expected_throughput(separation, APERTURE_RADIUS_LOD)
        assert got == pytest.approx(
            expected * (1.0 + aperture_bias), rel=APERTURE_BIAS_TOL
        ), f"throughput({separation})"


def test_core_mean_intensity_returns_the_constant_stellar_level(package, coro):
    """A constant stellar map has that constant as its mean over any aperture.

    This is the one quantity in the package with no discretization error at
    all: the mean of a constant is the constant however the aperture is drawn,
    so the tolerance is a floating-point one and a failure here is a real
    normalization error rather than a sampling artifact.
    """
    for separation in (1.0, 3.0, 5.0):
        got = float(
            stellar_intensity_curves(coro, 0.0, interpolation="keys").evaluate(
                "core_mean_intensity", separation
            )
        )
        assert got == pytest.approx(package.stellar_level, rel=1e-6), (
            f"core_mean_intensity({separation}, 0)"
        )


def test_raw_contrast_divides_two_aperture_sums_not_two_means(package, coro):
    """Contrast is a ratio of summed flux, so its numerator carries the aperture area.

    A constant stellar map makes this separable: the numerator is the stellar
    level times however many pixels the aperture holds, and the denominator is
    the throughput. Dividing the reported contrast by the throughput and by the
    level therefore recovers the aperture area in pixels, and nothing else.

    That number is the test. If the numerator were a per-pixel mean rather than
    a sum it would come back as one, not as about twenty-five, and the mistake
    would be a factor of the aperture area in every contrast the package
    reports. Checking the area also avoids reimplementing the pixel-membership
    mask here, which would only restate the code under test.
    """
    nominal_area_px = np.pi * (APERTURE_RADIUS_LOD / package.pixscale) ** 2
    separations = np.array([1.5, 3.0, 5.0])
    implied_area = np.array(
        [
            float(coro.curves().evaluate("raw_contrast", s))
            * float(coro.curves().evaluate("throughput", s))
            / package.stellar_level
            for s in separations
        ]
    )

    assert np.ptp(implied_area) / np.mean(implied_area) < 1e-3, (
        f"the implied aperture area varies with separation: {implied_area}"
    )
    assert np.mean(implied_area) == pytest.approx(nominal_area_px, rel=1e-6), (
        f"contrast implies an aperture of {np.mean(implied_area):.2f} pixels, "
        f"not the {nominal_area_px:.2f} a {APERTURE_RADIUS_LOD} lambda/D circle "
        "holds; a value near 1 would mean the numerator is a mean, not a sum"
    )


def test_occulter_transmission_follows_the_radial_function(package, coro):
    """Sky transmission at a radius is the function the package was written with."""
    for radius in (1.0, 2.0, 3.0):
        got = float(coro._occultation_record.evaluate("occulter_transmission", radius))
        expected = package.sky_trans_at(radius)
        assert got == pytest.approx(expected, rel=0.03), (
            f"occulter_transmission({radius})"
        )


def test_inner_working_angle_is_the_half_maximum_crossing(package, coro):
    """The reported angle is where throughput reaches half its maximum.

    The encircled-flux factor is common to the curve and its maximum, so the
    crossing is a property of the occulter alone and has a closed form. The
    The tolerance is one native pixel, which is far wider than the linear
    interpolation error over a half-lambda/D knot spacing and narrow enough
    that an off-by-one in the crossing search would fail.
    """
    got = coro.curves().iwa_lod
    expected = package.expected_iwa()
    assert abs(got - expected) < package.pixscale, (
        f"IWA {got:.4f} is more than one native pixel from {expected:.4f}"
    )


def test_data_extent_is_array_half_width(package, coro):
    """Fails if array extent is labeled as a physical outer working angle."""
    assert not hasattr(coro, "OWA")
    assert coro.curves().data_extent_lod == pytest.approx(package.max_offset_in_image)
    assert coro.curves().data_extent_lod > package.offsets[:, 0].max(), (
        "this package has data extent beyond its last offset, "
        "which is the regime the contrast sentinel guards"
    )


def test_contrast_does_not_blow_up_between_last_offset_and_data_extent(package, coro):
    """Synthetic contrast stays bounded through the array extent.

    Beyond the last tabulated offset the reader is extrapolating, and the
    quantity it extrapolates is a ratio whose denominator is heading toward
    zero. A contrast that climbs by orders of magnitude there is reported to a
    yield code as an unreachable region rather than as an extrapolation, so
    this is the sentinel: it bounds the extrapolated value by ten times the
    larger of the two in-image knots that bracket the tabulated range.
    """
    last_offset = float(package.offsets[:, 0].max())
    data_extent_lod = coro.curves().data_extent_lod
    in_image = [
        float(coro.curves().evaluate("raw_contrast", s))
        for s in (last_offset - 0.5, last_offset)
    ]
    ceiling = 10.0 * max(in_image)

    probes = np.linspace(last_offset, data_extent_lod, 9)
    checked = 0
    for separation in probes:
        value = float(coro.curves().evaluate("raw_contrast", separation))
        assert np.isfinite(value), f"raw_contrast({separation}) is not finite"
        assert value <= ceiling, (
            f"raw_contrast({separation:.3f}) = {value:.3e} exceeds ten times the "
            f"larger bracketing in-image knot ({ceiling:.3e})"
        )
        checked += 1
    assert checked == len(probes), "the sentinel loop skipped separations"


def test_the_stellar_diameter_list_is_not_mutated_by_construction(package, coro):
    """A zero diameter stays zero on the loaded object.

    The interpolator needs a positive abscissa, but substituting one in place
    edits the package's own record of what it contains, so a caller reading
    ``diams`` back sees a diameter the package does not have. The substitution
    belongs to the interpolator, not to the loaded list.
    """
    from astropy.io import fits

    diams = fits.getdata(package.path / "stellar_intens_diam_list.fits")
    assert diams[0] == 0.0, (
        f"the first stellar diameter is {diams[0]!r}, not 0; the epsilon "
        "substitution has leaked into the loaded list"
    )
