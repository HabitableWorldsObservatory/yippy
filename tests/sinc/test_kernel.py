"""Independent exact-real stencil arithmetic and raw kernel checks."""

import math
import random
import sys
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from yippy.sinc.kernel import exact_quotient_floor_v1, flush_subnormal, weights


def truth(q, origin, h):
    if abs(q) < sys.float_info.min:
        q = math.copysign(0.0, q)
    return math.floor(Fraction(q - origin) / Fraction(h))


compiled_floor = eqx.filter_jit(exact_quotient_floor_v1)


@pytest.mark.float64_only(reason="Float64 floor boundaries and subnormal bit patterns")
def test_floor_edge_list():
    """Fails if reciprocal rounding changes cell ownership; integer truth is exact."""
    queries = []
    for k in range(-40, 41):
        x = k * 0.38
        queries.append(x)
        for direction in (-math.inf, math.inf):
            adjacent = x
            for _ in range(4):
                adjacent = math.nextafter(adjacent, direction)
                queries.append(adjacent)
    queries += [0.0, -0.0, 5e-324, -5e-324, 1e-310, -1e-310, 5.699999999999999]
    rng = random.Random(0)
    queries += [rng.uniform(-15, 15) for _ in range(20000)]
    values = jnp.asarray(queries)
    expected = [truth(q, 0.0, 0.38) for q in queries]
    for fn in (
        exact_quotient_floor_v1,
        compiled_floor,
        eqx.filter_jit(jax.vmap(exact_quotient_floor_v1, in_axes=(0, None, None))),
    ):
        np.testing.assert_array_equal(
            fn(values, jnp.array(0.0), jnp.array(0.38)), expected
        )
    zeros = flush_subnormal(jnp.array([5e-324, -5e-324, 0.0, -0.0]))
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(zeros, jnp.uint64), [0, 2**63, 0, 2**63]
    )


@given(
    q=st.floats(-100.0, 100.0, allow_subnormal=False),
    origin=st.floats(-10.0, 10.0, allow_subnormal=False),
    h=st.floats(0.05, 0.5, allow_subnormal=False),
)
def test_floor_fraction_property(q, origin, h):
    """Fails if dynamic pitches or shifted origins break exact-real floors."""
    expected = truth(q, origin, h)
    args = tuple(jnp.asarray(v) for v in (q, origin, h))
    assert float(compiled_floor(*args)) == expected
    assert float(exact_quotient_floor_v1(*args)) == expected


@pytest.mark.float64_only(reason="Finite differences at a 1e-5 step")
def test_raw_kernel_and_gradient():
    """Fails on DC normalization, endpoint inclusion or inactive sqrt singularities.

    Floating-point tolerance 2e-7 checks an independent central derivative.
    """
    u = jnp.array([-17.0, -16.0, -3.2, 0.0, 0.35, 15.9, 16.0, 17.0])
    actual = weights(u)
    np.testing.assert_allclose(actual[3], 1.0, atol=0, rtol=4 * np.finfo(float).eps)
    np.testing.assert_array_equal(actual[jnp.array([0, 1, 6, 7])], jnp.zeros(4))
    derivative = jax.vmap(jax.grad(weights))(u)
    delta = 1e-5
    finite_difference = (weights(u + delta) - weights(u - delta)) / (2 * delta)
    # The hard cutoff has unequal one-sided derivatives at exactly +/-16.
    smooth = jnp.array([0, 2, 3, 4, 5, 7])
    np.testing.assert_allclose(
        derivative[smooth], finite_difference[smooth], atol=2e-7, rtol=2e-7
    )
    np.testing.assert_array_equal(derivative[jnp.array([0, 1, 6, 7])], jnp.zeros(4))
    taps = jnp.arange(-15, 17)
    assert abs(float(jnp.sum(weights(0.37 - taps))) - 1) > 1e-8
