"""The contract the shipped package and its reader hold each other to.

These are the assertions that can only be made against real data: the header
keywords a package is required to carry, the normalization its PSFs are written
with, and the working angles the reader derives from them. They are data-gated.
A cold cache with no network skips with a reason naming the cause, setting
``YIPPY_REQUIRE_REFERENCE_DATA=1`` turns that skip into a failure, and a cache
that is present and disagrees fails either way.

Numbers pinned here are measured properties of the shipped package rather than
derivations, so each one records what it means and how far it is allowed to
move. Where a quantity has a closed form it is tested against the synthetic
package instead, in ``test_analytic_yip.py``.
"""

import astropy.io.fits as pyfits
import numpy as np
import pytest
from lod_unit import lod

# The off-axis PSFs are normalized so that a source far outside the mask
# carries this fraction of the light an unocculted source would deliver to the
# array. It is a property of how the package was written, not of the reader,
# and the reader has no keyword to read it from, so it is recorded here until
# a future package declares it. Measured on eac1_aavc_2d.
FAR_FIELD_PSF_SUM = 0.485
FAR_FIELD_TOL = 2e-2

# The separation at which the PSF has cleared the mask but is still comfortably
# inside the array. Beyond about 28 lambda/D the blob starts falling off the
# 256-pixel edge and the sum drops for a geometric reason.
FAR_FIELD_SEPARATIONS_LOD = (20.0, 25.0)

# Inner working angle of the shipped package, pinned so that a change in the
# reader's half-maximum crossing is visible in review. This is a regression
# pin, not a derivation: it catches unintended change and establishes nothing
# about correctness.
EAC1_AAVC_IWA_LOD = 4.258
EAC1_AAVC_IWA_TOL = 5e-3


def test_the_header_declares_a_usable_pixel_scale(shipped_coro):
    """The reader's pixel scale is the header's, and it is physically sensible.

    Everything downstream converts separations with this number, so a package
    that omitted it or carried it in the wrong unit would move every planet in
    every simulated frame. The range is wide on purpose: it excludes a scale
    given in arcseconds or in milliarcseconds without excluding any plausible
    lambda/D sampling.
    """
    header = pyfits.getheader(shipped_coro.yip_path / "stellar_intens.fits", 0)
    declared = header["PIXSCALE"]

    assert float(shipped_coro.pixel_scale_arcsec.value) == pytest.approx(declared), (
        "the reader's pixel scale is not the one the header declares"
    )
    assert 0.05 <= declared <= 1.0, (
        f"a pixel scale of {declared} lambda/D per pixel is outside the range a "
        "yield input package is sampled at; the value is probably in another unit"
    )


def test_the_off_axis_normalization_is_the_recorded_one(shipped_coro):
    """A far-field off-axis PSF sums to the package's normalization constant.

    This is the absolute scale every consumer inherits. A shape or ratio test
    cannot see it, because a package rescaled by a constant factor passes every
    one of them, and the factor propagates into every count rate built on the
    package.
    """
    for separation in FAR_FIELD_SEPARATIONS_LOD:
        total = float(np.asarray(shipped_coro.offax(separation * lod, 0.0 * lod)).sum())
        assert total == pytest.approx(FAR_FIELD_PSF_SUM, rel=FAR_FIELD_TOL), (
            f"the off-axis PSF at {separation} lambda/D sums to {total:.6f}, not "
            f"the recorded {FAR_FIELD_PSF_SUM} normalization"
        )


def test_the_far_field_normalization_has_actually_plateaued(shipped_coro):
    """The recorded constant is a plateau, not a point on a rising curve.

    Pinning a number from a curve that is still climbing would make the
    tolerance above meaningless, since any separation would give a different
    answer. Checking that two well-separated far-field points agree is what
    makes the constant a property of the package.
    """
    sums = [
        float(np.asarray(shipped_coro.offax(s * lod, 0.0 * lod)).sum())
        for s in FAR_FIELD_SEPARATIONS_LOD
    ]
    spread = (max(sums) - min(sums)) / np.mean(sums)
    assert spread < FAR_FIELD_TOL, (
        f"the far-field PSF sums {sums} still vary by {spread:.3f} across the "
        "separations the normalization is read from, so it has not plateaued"
    )


def test_the_inscribed_diameter_option_is_exactly_a_separation_rescaling(shipped_coro):
    """Asking for inscribed units rescales the separation and nothing else.

    The option exists for compatibility with a yield code that measures
    separations in inscribed lambda/D. If it did anything besides scale the
    argument, the two routes would disagree, and a consumer mixing them would
    get two different throughputs for one planet.
    """
    from yippy import Coronagraph

    plain = shipped_coro
    inscribed = Coronagraph(plain.yip_path, use_inscribed_diameter=True)

    ratio = float(plain.header.diameter / plain.header.diameter_inscribed)
    assert ratio > 1.0, "the inscribed diameter should be smaller than the full one"

    for separation in (5.0, 10.0):
        through_option = float(inscribed.throughput(separation))
        through_scaling = float(plain.throughput(separation * ratio))
        assert through_option == pytest.approx(through_scaling, rel=1e-10), (
            f"at {separation} inscribed lambda/D the option gives "
            f"{through_option:.10f} but rescaling the separation gives "
            f"{through_scaling:.10f}"
        )


def test_the_inner_working_angle_has_not_moved(shipped_coro):
    """A regression pin on the derived inner working angle.

    Every yield number depends on where the reader says the mask opens. This
    catches an unintended change in the half-maximum crossing or in the
    throughput curve beneath it; it is not evidence that the value is right.
    The closed-form check of the same derivation is in ``test_analytic_yip.py``.
    """
    got = float(shipped_coro.IWA.to_value(lod))
    assert got == pytest.approx(EAC1_AAVC_IWA_LOD, abs=EAC1_AAVC_IWA_TOL), (
        f"the inner working angle moved to {got:.4f} lambda/D from the pinned "
        f"{EAC1_AAVC_IWA_LOD}"
    )


def test_the_outer_working_angle_is_the_array_edge(shipped_coro):
    """OWA is geometric: it is where the PSF centre leaves the array.

    It is not the last tabulated offset and not a point on the throughput
    curve. A consumer that treats it as the edge of the sampled region will
    query separations the package never tabulated, which is the regime the
    contrast sentinel in the analytic suite guards.
    """
    expected = shipped_coro.npixels / 2 * float(shipped_coro.pixel_scale_arcsec.value)
    assert float(shipped_coro.OWA.to_value(lod)) == pytest.approx(expected), (
        "the outer working angle is not the half-width of the array"
    )
    assert float(shipped_coro.OWA.to_value(lod)) > float(
        shipped_coro.IWA.to_value(lod)
    ), "the outer working angle is inside the inner one"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The reported outer working angle is the geometric edge of the array, "
        "32 lambda/D, but the package's controlled dark hole ends near 26 to "
        "28. This is a property of the package rather than an artefact of the "
        "reader: the stellar intensity map's own radial mean rises from "
        "3.8e-13 at 24 lambda/D to 3.5e-8 at 31, five orders, while throughput "
        "over the same span falls by under twenty percent. Raw contrast "
        "follows it, 3.0e-11 at 25 lambda/D and 1.7e-3 at the reported angle. "
        "A yield code that trusts the reported angle therefore places planets "
        "in a band the coronagraph does not control. Separately, offsets "
        "tabulated beyond the array edge are given a 1e20 contrast sentinel, "
        "so a query above the reported angle returns a number climbing toward "
        "it rather than an error. Either the reported angle is derived from "
        "the contrast curve, or the package carries its dark hole edge as its "
        "own keyword; both change numbers in every consumer, so the contract "
        "is written here and the choice is escalated."
    ),
)
def test_the_contrast_sentinel_holds_on_the_shipped_package(shipped_coro):
    """Contrast inside the OWA stays near its bracketing tabulated values.

    The same sentinel the analytic suite applies to a package whose answers are
    known, applied here to the one a yield code actually consumes. Beyond the
    last tabulated offset the reader extrapolates a ratio whose denominator is
    heading toward zero, and a contrast that climbs by orders of magnitude
    there reaches a yield code as an unreachable region rather than as an
    extrapolation.
    """
    owa = float(shipped_coro.OWA.to_value(lod))
    iwa = float(shipped_coro.IWA.to_value(lod))
    reference = max(
        float(shipped_coro.raw_contrast(s))
        for s in np.linspace(iwa + 1.0, 0.75 * owa, 8)
    )
    ceiling = 10.0 * reference

    probes = np.linspace(0.75 * owa, owa, 9)
    checked = 0
    for separation in probes:
        value = float(shipped_coro.raw_contrast(separation))
        assert np.isfinite(value), f"raw_contrast({separation:.2f}) is not finite"
        assert value <= ceiling, (
            f"raw_contrast({separation:.2f}) = {value:.3e} exceeds ten times the "
            f"largest in-image value below it ({ceiling:.3e})"
        )
        checked += 1
    assert checked == len(probes), "the sentinel loop skipped separations"
