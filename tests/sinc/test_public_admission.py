"""Host planning and selection share literal sampling thresholds."""

import math

import yippy.sinc as sc
from yippy.sinc.admission import plan_spacing, sampling_reasons, select_profile
from yippy.sinc.evidence import Reason
from yippy.sinc.kernel import PrefilteredKernelProfile


def test_plan_spacing_and_reader_selection():
    """Exact-algebra basis: 2*(lambda0/lambda_min)*L*h <= margin, no slack."""
    for minimum, support in ((1000.0, (0.982, 0.946)), (800.0, (1.0, 1.0))):
        h = plan_spacing(1000.0, minimum, support)
        scale = 2 * (1000.0 / minimum) * max(support)
        assert scale * h <= 0.76
        assert scale * (h + 1e-6) > 0.76
    assert isinstance(select_profile([0.64]), PrefilteredKernelProfile)
    assert isinstance(
        select_profile([math.nextafter(0.64, math.inf)]), sc.KernelProfile
    )
    assert select_profile([0.5], taps=8).taps == 8
    assert (
        sampling_reasons([math.nextafter(0.76, math.inf)]) == Reason.KERNEL_MARGIN_FAIL
    )
