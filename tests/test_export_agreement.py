"""Whether the contrast a consumer reads is the contrast yippy reports.

``Coronagraph.raw_contrast`` interpolates the logarithm of contrast and
exponentiates. ``export_exosims`` writes its table from a separate interpolant
built on the same knots in linear space. The two agree at the knots and diverge
between them, so a consumer that reads the exported table and interpolates it
gets a different number from one that calls the accessor at the same
separation.

That is a convention fork rather than an arithmetic error: both sides are
self-consistent and neither is wrong on its own terms. Until one space is
chosen for both, these tests pin both halves: the agreement at the knots,
which must be exact to rounding, and the size of the between-knot
disagreement, which must not grow without someone noticing.
"""

import numpy as np
import pytest

# Where the two interpolants are compared. The knots of the synthetic package
# are spaced half a lambda/D apart.
KNOT_SPACING_LOD = 0.5

# The largest relative disagreement between the two interpolants anywhere
# between knots, measured on the synthetic package before this test was
# written. It falls off steeply with separation: 18 percent across the
# innermost knot interval, 5 percent across the next, and under a part in a
# thousand beyond about three lambda/D. The innermost interval sits inside the
# inner working angle, where no consumer reads a contrast, so the number that
# matters operationally is the 5 percent just outside it. Both are bounded
# here. Tolerance basis: spike.
MAX_BETWEEN_KNOT_DISAGREEMENT = 0.20
MAX_DISAGREEMENT_ABOVE_IWA = 0.06


def _knots(coro, upper_lod):
    """Tabulated positive offsets inside the array, as plain floats."""
    offsets = np.asarray(coro.offax.x_offsets, dtype=float)
    return offsets[(offsets > 0) & (offsets <= upper_lod)]


def test_the_two_contrast_paths_agree_at_the_knots(syn_coro_1d):
    """At a tabulated offset both interpolants return the tabulated value.

    Interpolating a logarithm and exponentiating is a round trip at a knot, so
    the only difference allowed here is float32 rounding through ``log10`` and
    ``power``. A real disagreement at a knot would mean the two interpolants
    were built from different data, not merely in different spaces.
    """
    coro = syn_coro_1d
    knots = _knots(coro, 6.0)
    assert len(knots) >= 4, "too few knots to make this comparison meaningful"

    for separation in knots:
        accessor = float(coro.raw_contrast(float(separation)))
        exported = float(coro.raw_contrast_interp(np.array([separation]))[0])
        assert accessor == pytest.approx(exported, rel=1e-5), (
            f"at the tabulated offset {separation} the accessor gives "
            f"{accessor:.6e} and the exported interpolant {exported:.6e}"
        )


def test_the_between_knot_disagreement_is_bounded_and_real(syn_coro_1d):
    """Between knots the two paths differ, by a bounded and recorded amount.

    Both halves of this matter. The disagreement has to be real, because a
    version of this test that passed trivially would hide the fork rather than
    record it; and it has to be bounded, because the bound is what a consumer
    reading the exported table is entitled to assume about how far it sits from
    the accessor.
    """
    coro = syn_coro_1d
    knots = _knots(coro, 6.0)
    midpoints = (knots[:-1] + knots[1:]) / 2.0

    disagreements = []
    for separation in midpoints:
        accessor = float(coro.raw_contrast(float(separation)))
        exported = float(coro.raw_contrast_interp(np.array([separation]))[0])
        disagreements.append(abs(accessor - exported) / exported)
    disagreements = np.array(disagreements)

    assert disagreements.max() < MAX_BETWEEN_KNOT_DISAGREEMENT, (
        f"the two contrast paths now disagree by {disagreements.max():.3f} "
        f"between knots, above the recorded {MAX_BETWEEN_KNOT_DISAGREEMENT}"
    )

    # The operationally relevant bound: inside the inner working angle nothing
    # reads a contrast, so a large disagreement there costs nothing.
    usable = disagreements[midpoints >= float(coro.IWA.value)]
    assert len(usable) >= 3, "too few usable knot intervals to bound"
    assert usable.max() < MAX_DISAGREEMENT_ABOVE_IWA, (
        f"outside the inner working angle the two paths now disagree by "
        f"{usable.max():.3f}, above the recorded {MAX_DISAGREEMENT_ABOVE_IWA}"
    )
    assert disagreements.max() > 1e-3, (
        "the two paths no longer disagree between knots, which means either "
        "the fork was closed, in which case delete this test and assert "
        "equality instead, or this comparison stopped exercising it"
    )


def test_the_disagreement_is_worst_where_the_curve_is_steepest(syn_coro_1d):
    """The fork is largest across the innermost knot intervals.

    Interpolating a quantity and interpolating its logarithm differ in
    proportion to the curvature over a knot interval, so the disagreement is
    largest where contrast falls fastest, which is at small separations. The
    steep region continues a little way past the inner working angle, so the
    fork reaches into the band a yield code actually reads, which is why it is
    worth closing rather than merely bounding.
    """
    coro = syn_coro_1d
    knots = _knots(coro, 6.0)
    midpoints = (knots[:-1] + knots[1:]) / 2.0

    disagreements = np.array(
        [
            abs(
                float(coro.raw_contrast(float(s)))
                - float(coro.raw_contrast_interp(np.array([s]))[0])
            )
            / float(coro.raw_contrast_interp(np.array([s]))[0])
            for s in midpoints
        ]
    )

    worst = midpoints[np.argmax(disagreements)]
    assert worst < float(coro.IWA.value) + 2.0 * KNOT_SPACING_LOD, (
        f"the worst disagreement has moved to {worst} lambda/D, away from the "
        f"steep part of the curve near the inner working angle "
        f"({float(coro.IWA.value):.2f})"
    )
