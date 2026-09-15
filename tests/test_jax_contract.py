"""The transformations a caller is entitled to apply to a coronagraph.

``Coronagraph`` is the loader: it holds scipy interpolants and is evaluated
eagerly. ``EqxCoronagraph`` is the array-only view a consumer calls from inside
``jit``, ``vmap`` or ``grad``, and it is the object these transformation tests
target. Synthesis through ``Coronagraph.offax`` is already JAX-backed, so the
batching tests use it directly.

Consumers call these methods from inside ``jit``, from inside ``vmap``, and
through ``grad``. Each transformation is a separate code path in JAX, so
agreement between them is a property that has to be asserted rather than
assumed: a traced argument that silently takes a Python branch, or a batched
call that reuses one element's index, produces a wrong answer with no error.

Every expected value here is another evaluation of the same quantity under a
different transformation, which makes these consistency properties rather than
physics. The physics is checked in ``test_analytic_yip.py`` against closed
forms.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lod_unit import lod

# Positions spanning the tabulated range, including one in a negative quadrant
# so that the mirroring path is exercised alongside the direct one.
POSITIONS = ((2.0, 0.0), (3.5, 1.5), (-2.5, -1.0))


def test_single_and_batched_synthesis_agree(syn_coro_1d):
    """One PSF at a time equals a batch of one, and equals a batch of many.

    A batched implementation that reused the first element's interpolation
    weights would still return the right shape and still conserve flux; only a
    comparison against the single-position path can see it.
    """
    coro = syn_coro_1d
    xs = np.array([x for x, _ in POSITIONS])
    ys = np.array([y for _, y in POSITIONS])

    batched = np.asarray(coro.offax.create_psfs(xs, ys))
    assert batched.shape[0] == len(POSITIONS), "the batch lost positions"

    for index, (x, y) in enumerate(POSITIONS):
        single = np.asarray(coro.offax(x * lod, y * lod))
        assert np.allclose(single, batched[index], atol=1e-6 * single.max()), (
            f"batched synthesis disagrees with the single call at ({x}, {y})"
        )


def test_the_parallel_path_agrees_with_the_batched_one(syn_coro_1d):
    """The parallel helper is an optimization, so it must change no value."""
    coro = syn_coro_1d
    xs = np.array([x for x, _ in POSITIONS])
    ys = np.array([y for _, y in POSITIONS])

    batched = np.asarray(coro.offax.create_psfs(xs, ys))
    parallel = np.asarray(coro.offax.create_psfs_parallel(xs, ys))
    assert np.allclose(batched, parallel, atol=1e-6 * batched.max()), (
        "the parallel path returns different PSFs from the batched path"
    )


@pytest.mark.parametrize("diameter", [0.0, 0.1])
def test_a_traced_stellar_diameter_runs_and_matches_eager(syn_eqx_1d, diameter):
    """Core mean intensity stays correct when its diameter argument is traced.

    A diameter that reaches a Python ``if`` raises the moment a caller puts the
    call inside ``jit``, because a comparison against a tracer is itself a
    tracer. Both a zero and a non-zero diameter are checked, since they select
    different interpolants and a selection written with ``jnp.where`` has to
    give the eager answer for each.
    """
    eager = float(syn_eqx_1d.core_mean_intensity(5.0, diameter))
    traced = eqx.filter_jit(lambda d: syn_eqx_1d.core_mean_intensity(5.0, d))(
        jnp.asarray(diameter)
    )

    assert float(traced) == pytest.approx(eager, rel=1e-6), (
        "core_mean_intensity under filter_jit disagrees with its eager value"
    )


def test_a_traced_separation_runs_and_matches_eager(syn_eqx_1d):
    """The performance curves stay correct when the separation is traced."""
    for name in ("throughput", "raw_contrast", "core_area", "occulter_transmission"):
        method = getattr(syn_eqx_1d, name)
        eager = float(method(3.0))
        traced = float(eqx.filter_jit(lambda s: method(s))(jnp.asarray(3.0)))  # noqa: B023
        assert traced == pytest.approx(eager, rel=1e-6), f"{name} under filter_jit"


def test_vmap_over_separations_equals_a_loop(syn_eqx_1d):
    """A batched evaluation equals the loop it replaces.

    ``vmap`` and a Python loop take different paths through an interpolant, and
    a reduction written against a scalar shape can quietly broadcast under
    ``vmap`` instead of failing.
    """
    separations = jnp.asarray([1.5, 2.5, 3.5, 4.5])

    looped = np.array([float(syn_eqx_1d.throughput(float(s))) for s in separations])
    mapped = np.asarray(jax.vmap(lambda s: syn_eqx_1d.throughput(s))(separations))

    assert np.allclose(looped, mapped, rtol=1e-6), (
        "vmapped throughput disagrees with the loop it replaces"
    )


# Step for the central difference. Too small and a float32 subtraction of two
# nearly equal values loses the answer to cancellation; too large and the
# curvature of the interpolant shows up as truncation error. Measured across
# both curves and three separations before this test was written: at 1e-3 the
# agreement is 2.5e-3, at 0.05 it is between 3e-6 and 7e-4, and by 0.2 the
# curved contrast curve is drifting again. Tolerance basis: spike.
DIFFERENCE_STEP = 0.05
GRADIENT_TOL = 1e-3


@pytest.mark.parametrize("name", ["throughput", "raw_contrast"])
def test_gradients_agree_with_a_central_difference(syn_eqx_1d, name):
    """Autodiff of each curve matches a finite difference at interior points.

    A finiteness check would pass for a wrong gradient, so the comparison is
    against an independently computed central difference. The points are inside
    the tabulated range and away from the knots, where the interpolant is
    smooth; at a knot the derivative is genuinely two-valued and neither answer
    would be wrong.

    ``core_area`` is deliberately not in this list. Every off-axis PSF in the
    synthetic package has the same shape, so its core area does not vary with
    separation at all and both the autodiff and the difference are exactly
    zero. Asserting that they agree would be asserting nothing, which is the
    shape of test this suite exists to avoid; the constancy itself is checked
    separately below.
    """
    method = getattr(syn_eqx_1d, name)

    for separation in (2.25, 3.25, 4.25):
        analytic = float(
            jax.grad(lambda s: jnp.sum(method(s)))(jnp.asarray(separation))
        )
        forward = float(method(separation + DIFFERENCE_STEP))
        backward = float(method(separation - DIFFERENCE_STEP))
        numeric = (forward - backward) / (2.0 * DIFFERENCE_STEP)

        assert abs(numeric) > 1e-12, (
            f"{name} does not vary with separation near {separation}, so this "
            "comparison would hold for any gradient"
        )
        assert abs(analytic - numeric) / abs(numeric) < GRADIENT_TOL, (
            f"d {name} / d separation at {separation}: autodiff {analytic:.6e} "
            f"against central difference {numeric:.6e}"
        )


def test_core_area_is_constant_when_every_psf_has_the_same_shape(syn_eqx_1d):
    """A fixed aperture over identically shaped PSFs gives one core area.

    This is what makes the gradient of ``core_area`` uninformative for this
    package, and it is worth asserting in its own right: a core area that
    drifted with separation here would mean the aperture is being placed or
    sized differently at different separations.
    """
    areas = np.array([float(syn_eqx_1d.core_area(s)) for s in (1.5, 2.5, 3.5, 4.5)])
    assert np.ptp(areas) / np.mean(areas) < 1e-6, (
        f"core area varies with separation for identical PSFs: {areas}"
    )


def test_the_same_position_gives_identical_psfs_to_roundoff(syn_coro_1d):
    """Synthesis is repeatable, so a repeated call returns the same array.

    A cached intermediate mutated in place, or any state carried between calls,
    would make every downstream frame irreproducible from its manifest.

    Tolerance basis: XLA's multithreaded CPU kernels may partition the FFTs and
    the weighted neighbor sum differently between calls, which changes the
    order of floating-point additions and moves pixels by up to one unit
    roundoff (about 6e-8 of the peak in float32). ``1e-6`` of the peak leaves
    margin for that while any stateful error is far larger.
    """
    coro = syn_coro_1d
    first = np.asarray(coro.offax(2.0 * lod, 1.0 * lod))
    second = np.asarray(coro.offax(2.0 * lod, 1.0 * lod))
    assert np.allclose(first, second, rtol=0, atol=1e-6 * first.max()), (
        "repeated synthesis differs beyond floating-point roundoff"
    )
