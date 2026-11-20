"""Tap-count planning from delivery metadata."""

import pytest

from yippy.sinc.evidence import PlanRefused, Reason
from yippy.sinc.kernel import PrefilteredKernelProfile
from yippy.sinc.planner import plan_prefilter
from yippy.sinc.prefilter import _design_axis

from .helpers import delivery, tone_truth


def plan(data, max_tone_error):
    return plan_prefilter(
        declared_lattice=data["declared_lattice"],
        products=data["products"],
        reflection=data["reflection"],
        domain=data["domain"],
        max_tone_error=max_tone_error,
    )


@pytest.mark.medium(reason="designs three tap counts on two axes, 9 s")
def test_plan_picks_fewest_taps_and_admission_reuses_it(tone_full):
    """Fails if the planner skips a cheaper tap count or disagrees with admission.

    At sampling ratios 0.6 and 0.56 the two-axis tone error is about 1.6e-5
    for 7 taps and 2.7e-6 for 8, so requests of 3e-5 and 4e-6 separate them.
    """
    data = tone_full["data"]
    seven = plan(data, 3e-5)
    assert seven.planned and seven.reason_bits == 0
    assert seven.require() == PrefilteredKernelProfile(taps=7)
    assert [taps for taps, _ in seven.candidates] == [5, 6, 7]
    errors = [error for _, error in seven.candidates]
    assert errors[0] > errors[1] > 3e-5 >= errors[2] == seven.tone_error_two_axis
    # Admission records the same design, taken from the design cache.
    assert seven.kernel_design == tone_full["receipt"].kernel_design
    hits = _design_axis.cache_info().hits
    eight = plan(data, 4e-6)
    assert _design_axis.cache_info().hits >= hits + 6
    assert eight.require().taps == 8
    assert dict(eight.kernel_design)["filter_half_width"] == 16


@pytest.mark.medium(reason="designs all six tap counts, 6 s")
def test_plan_refusals():
    """Fails if an unreachable request, a coarse lattice or a tiny table plans.

    The planner stops at 10 taps. A request below what 10 taps reach refuses
    with TOLERANCE_EXCEEDED and require() raises with the plan attached.
    """
    truth, _ = tone_truth((4, 4))
    small = delivery(half=18, shape=(4, 4), spacing=0.3, truth=truth)
    refused = plan(small, 1e-12)
    assert not refused.planned and refused.profile is None
    assert refused.reason_bits == Reason.TOLERANCE_EXCEEDED
    assert [taps for taps, _ in refused.candidates] == [5, 6, 7, 8, 9, 10]
    with pytest.raises(PlanRefused, match="TOLERANCE_EXCEEDED") as caught:
        refused.require()
    assert caught.value.plan is refused
    assert caught.value.reason_bits == Reason.TOLERANCE_EXCEEDED
    coarse = delivery(half=18, shape=(4, 4), spacing=0.33, truth=truth)
    assert plan(coarse, 1e-3).reason_bits == Reason.KERNEL_MARGIN_FAIL
    aliased = delivery(half=18, shape=(4, 4), spacing=0.55, truth=truth)
    assert plan(aliased, 1e-3).reason_bits == (
        Reason.NYQUIST_FAIL | Reason.KERNEL_MARGIN_FAIL
    )
    tiny = delivery(half=6, shape=(4, 4), spacing=0.3, truth=truth)
    unserved = plan(tiny, 1e-3)
    assert unserved.reason_bits == Reason.INCOMPLETE_STENCIL
    assert unserved.candidates == ()
    with pytest.raises(ValueError, match="positive"):
        plan(small, 0.0)
