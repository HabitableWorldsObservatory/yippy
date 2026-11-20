"""Analytic intensity, reflected operator, batch and unit-scale verification."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yippy.sinc as sc
from yippy.sinc.evidence import BoundKind, Reason, Status
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import stencil
from yippy.sinc.reader import read_batch, require_read

from .helpers import analytic, delivery, make_reader

compiled_read = eqx.filter_jit(lambda reader, q, p: reader.read(q, p))
compiled_batch = eqx.filter_jit(read_batch)


def test_off_node_analytic_and_scale(full_reader):
    """Fails on tap/axis/scale mistakes; 3e-7 bounds low-occupancy truncation.

    The largest source frequency is .16 cycles/lod at pitch .25 (8% of
    Nyquist). The measured correct error is 2.912e-7; a shifted y stencil gives
    1.1681e-6. The 3e-7 allowance resolves that mutation.
    """
    q = jnp.array([0.137, -0.219])
    image = compiled_read(full_reader, q, jnp.int32(0))
    expected = analytic(q)
    np.testing.assert_allclose(image.intensity, expected, atol=3e-7, rtol=0)
    assert image.accepted and image.status == Status.SAMPLING_ADMITTED
    assert image.reason_bits == Reason.BOUND_UNKNOWN
    assert not image.bound_available and image.bound_kind == BoundKind.UNKNOWN
    assert math.isinf(float(image.abs_error_bound))
    assert not bool(image.ledger.applicable[-1])
    scaled = eqx.tree_at(
        lambda r: r.grid.intensity, full_reader, full_reader.grid.intensity * 7
    )
    np.testing.assert_allclose(
        compiled_read(scaled, q, jnp.int32(0)).intensity,
        7 * image.intensity,
        rtol=3e-15,
        atol=0,
    )


@pytest.mark.parametrize(
    "queries",
    [
        [[0.13, -0.21], [-0.2, 0.3]],
        [
            [0.0, -0.0],
            [-0.0, 0.0],
            [math.nextafter(0.0, -math.inf), math.nextafter(0.0, math.inf)],
        ],
        [[0.0, 0.25], [-0.25, 0.0]],
    ],
)
def test_folded_full_and_modes(full_reader, quadrant_reader, queries):
    """Fails on detector flip direction, axis ownership or mode-dependent taps.

    A 32-epsilon weighted absolute-sample bound allows rebased phase rounding.
    Eager/JIT returned intensity tolerance is 32 eps times its absolute scale.
    """
    queries = jnp.array(queries)
    mapped = eqx.filter_jit(
        jax.vmap(lambda r, q: r.read(q, jnp.int32(0)), in_axes=(None, 0))
    )
    folded = mapped(quadrant_reader, queries)
    full = mapped(full_reader, queries)
    for index, q in enumerate(queries):
        eager = quadrant_reader.read(q, jnp.int32(0))
        compiled = compiled_read(quadrant_reader, q, jnp.int32(0))
        for result in (eager, compiled):
            assert result.status == folded.status[index]
            np.testing.assert_allclose(
                result.intensity,
                folded.intensity[index],
                rtol=32 * np.finfo(float).eps,
                atol=0,
            )
        taps, _ = stencil(
            q,
            origin_xy(quadrant_reader.grid.source),
            spacing_xy(quadrant_reader.grid.source),
        )
        compiled_taps, _ = eqx.filter_jit(stencil)(
            q,
            origin_xy(quadrant_reader.grid.source),
            spacing_xy(quadrant_reader.grid.source),
        )
        np.testing.assert_array_equal(taps, compiled_taps)
        full_taps, full_weights = stencil(
            q,
            origin_xy(full_reader.grid.source),
            spacing_xy(full_reader.grid.source),
        )
        samples = full_reader.grid.intensity[
            0, full_taps[1, :, None], full_taps[0, None, :]
        ]
        weighted_absolute = jnp.sum(
            jnp.abs(
                samples
                * full_weights[1, :, None, None, None]
                * full_weights[0, None, :, None, None]
            ),
            axis=(0, 1),
        )
        allowance = 32 * np.finfo(float).eps * weighted_absolute
        assert bool(
            jnp.all(
                jnp.abs(folded.intensity[index] - full.intensity[index]) <= allowance
            )
        )
        if bool(jnp.all(q == jnp.round(q / 0.25) * 0.25)):
            np.testing.assert_array_equal(
                folded.intensity[index], full.intensity[index]
            )
    assert folded.ledger.term_ids == full.ledger.term_ids
    assert bool(jnp.all(folded.ledger.applicable[:, -1]))


def test_stencils_crossing_one_or_no_axes():
    """Fails if reflection is applied to a query instead of each signed tap."""
    quad = delivery(quadrant=True, half=34)
    full = delivery(half=34)
    domain = sc.Domain(
        "box", "closed", jnp.array([[-4.1, -4.1], [4.1, 4.1]]), jnp.zeros(2)
    )
    quad["domain"] = full["domain"] = domain
    qr, fr = make_reader(quad), make_reader(full)
    queries = jnp.array([[4.01, 0.2], [4.01, 4.01], [-4.01, 4.01]])
    a = compiled_batch(qr, queries, jnp.zeros(3, dtype=jnp.int32))
    b = compiled_batch(fr, queries, jnp.zeros(3, dtype=jnp.int32))
    assert bool(jnp.all(a.accepted & b.accepted))
    np.testing.assert_allclose(a.intensity, b.intensity, atol=4e-14, rtol=0)


def test_asymmetric_full_and_quadrant_refusal():
    """Fails if non-covariant responses are silently folded or full grids flipped."""
    full = make_reader(delivery(asymmetric=True))
    q = jnp.array([-0.27, 0.18])
    np.testing.assert_allclose(
        compiled_read(full, q, jnp.int32(0)).intensity,
        analytic(q, asymmetric=True),
        atol=9e-6,
        rtol=0,
    )
    refused = sc.admit_grid(**delivery(quadrant=True, asymmetric=True))
    assert not refused.admitted
    assert refused.reason_bits & Reason.SYMMETRY_CHECK_FAILED


def test_mixed_batch_and_empty(full_reader):
    """Fails if a bad query poisons another lane or empty batches execute a read."""
    queries = jnp.array(
        [[0.1, 0.2], [jnp.nan, 0.1], [2.0, 0.0], [0.2, 0.1], [1e308, 0.0]]
    )
    products = jnp.array([0, 0, 0, 99, 0], dtype=jnp.int32)
    result = compiled_batch(full_reader, queries, products)
    np.testing.assert_array_equal(result.accepted, [True, False, False, False, False])
    assert bool(jnp.all(jnp.isnan(result.intensity[1:])))
    for index, reason in (
        (1, Reason.NONFINITE_QUERY),
        (2, Reason.OUTSIDE_DOMAIN),
        (3, Reason.PRODUCT_UNAVAILABLE),
        (4, Reason.INCOMPLETE_STENCIL),
    ):
        assert int(result.reason_bits[index]) & reason
    with pytest.raises(sc.ReadRefused) as caught:
        require_read(result)
    assert caught.value.indices == (1, 2, 3, 4)
    empty = compiled_batch(
        full_reader, jnp.empty((0, 2)), jnp.empty(0, dtype=jnp.int32)
    )
    assert all(leaf.shape[0] == 0 for leaf in jax.tree_util.tree_leaves(empty))
    assert empty.ledger.term_ids == result.ledger.term_ids
    require_read(empty)


def test_bounded_refuses_sampling_remains_ok(full_reader):
    """Fails if empirical or unknown errors can issue a bounded image."""
    bounded = eqx.tree_at(
        lambda r: r.policy, full_reader, sc.ReadPolicy("bounded", jnp.array(1.0))
    )
    result = compiled_read(bounded, jnp.array([0.1, 0.2]), jnp.int32(0))
    assert result.sampling_ok and not result.accepted
    assert result.status == Status.REFUSED
    assert result.reason_bits == Reason.BOUND_UNKNOWN
    assert bool(jnp.all(jnp.isnan(result.intensity)))


def test_dynamic_arrays_do_not_recompile(full_reader):
    """Fails if samples become static metadata or a compiled closure captures them."""
    traces = []

    @eqx.filter_jit
    def run(reader, q):
        traces.append(1)
        return reader.read(q, jnp.int32(0))

    q = jnp.array([0.1, 0.2])
    first = run(full_reader, q)
    changed = eqx.tree_at(
        lambda r: r.grid.intensity, full_reader, full_reader.grid.intensity * 2
    )
    second = run(changed, q)
    assert len(traces) == 1
    np.testing.assert_allclose(second.intensity, first.intensity * 2, rtol=0, atol=0)
    assert all(eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(full_reader))
    assert all(eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(first))


def test_negative_excursions_preserved(full_reader):
    """Fails on hidden clipping; a positive isolated node has negative sinc lobes."""
    samples = jnp.zeros_like(full_reader.grid.intensity).at[0, 20, 19, :, :].set(1.0)
    reader = eqx.tree_at(lambda r: r.grid.intensity, full_reader, samples)
    image = compiled_read(reader, jnp.array([0.125, 0.0]), jnp.int32(0))
    assert bool(jnp.all(image.intensity < -0.1))


@pytest.mark.parametrize("profile", ["raw", "prefiltered"])
def test_stencil_weights_reproduce_the_read(profile, full_reader, tone_full):
    """Fails if the stencil is a copy that drifts from the one a read contracts."""
    reader = full_reader if profile == "raw" else tone_full["reader"]
    queries = (
        [0.137, -0.219] if profile == "raw" else [0.31, -0.17],
        [-0.2, 0.3],
    )
    for query in queries:
        q = jnp.asarray(query)
        found = reader.stencil(q)
        assert found.nodes.shape == found.weights.shape == found.valid.shape
        assert found.nodes.shape == (2, reader.profile.taps)
        assert bool(jnp.all(found.valid))
        nodes, w = found.nodes, found.weights
        samples = reader.grid.intensity[0, nodes[1, :, None], nodes[0, None, :]]
        value = jnp.einsum("j,i,jirc->rc", w[1], w[0], samples.astype(jnp.float64))
        np.testing.assert_allclose(
            value, reader.read(q, jnp.int32(0)).intensity, rtol=1e-12, atol=1e-14
        )
