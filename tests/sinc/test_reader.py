"""Analytic intensity, reflected operator, batch and unit-scale verification."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import eps, roundoff

import yippy.sinc as sc
from yippy.sinc.evidence import BoundKind, Reason, Status
from yippy.sinc.geometry import origin_xy, source_grid, spacing_xy
from yippy.sinc.kernel import stencil
from yippy.sinc.reader import read_batch, require_read

from .helpers import analytic, delivery, make_reader

compiled_read = eqx.filter_jit(lambda reader, q, p: reader.read(q, p))
compiled_batch = eqx.filter_jit(read_batch)


def test_off_node_analytic_and_scale(full_reader):
    """Fails on tap/axis/scale mistakes; 3e-7 bounds low-occupancy truncation.

    The largest source frequency is .16 cycles/lod at pitch .25 (8% of
    Nyquist). Check all tap indices against the host floor rule so a one-node
    shift fails in either precision. The intensity bound adds 48 operations
    for phase, kernel evaluation and two 32-tap sums.
    Scaling a table adds one operation to the two contractions (32 epsilon units).
    """
    q = jnp.array([0.137, -0.219])
    lattice = full_reader.grid.source
    cell = np.floor(
        (np.asarray(q, dtype=np.float64) - origin_xy(lattice)) / spacing_xy(lattice)
    )
    expected_nodes = cell.astype(np.int32)[:, None] + np.arange(-15, 17)
    for select in (full_reader.stencil, eqx.filter_jit(full_reader.stencil)):
        np.testing.assert_array_equal(select(q).nodes, expected_nodes)
    image = compiled_read(full_reader, q, jnp.int32(0))
    expected = analytic(q)
    rounding = float(roundoff(float(expected.max()), n_ops=48, dtype=expected.dtype))
    np.testing.assert_allclose(image.intensity, expected, atol=3e-7 + rounding, rtol=0)
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
        rtol=32 * eps(image.intensity.dtype),
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
                rtol=32 * eps(q.dtype),
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
        allowance = 32 * eps(q.dtype) * weighted_absolute
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
    """Compare folded and signed reads within 96 units for kernels and contractions."""
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
    np.testing.assert_allclose(
        a.intensity,
        b.intensity,
        rtol=0,
        atol=float(
            roundoff(
                float(np.max(np.abs(b.intensity))), n_ops=96, dtype=b.intensity.dtype
            )
        ),
    )


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
        [
            [0.1, 0.2],
            [jnp.nan, 0.1],
            [2.0, 0.0],
            [0.2, 0.1],
            [jnp.finfo(full_reader._safe_query.dtype).max, 0.0],
        ]
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
    """Compare a host float64 separable contraction to the read.

    Weight an absolute sample sum and allow 64 rounding operations for the
    different contraction orders and their two tap reductions.
    """
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
        wy, wx = (np.asarray(w[axis], dtype=np.float64) for axis in (1, 0))
        value = np.einsum("j,jrc->rc", wy, np.einsum("i,jirc->jrc", wx, samples))
        scale = np.einsum(
            "j,jrc->rc",
            np.abs(wy),
            np.einsum("i,jirc->jrc", np.abs(wx), np.abs(samples)),
        )
        actual = np.asarray(reader.read(q, jnp.int32(0)).intensity)
        assert np.all(np.abs(actual - value) <= 64 * eps(actual.dtype) * scale)


def test_concrete_coordinates_preserve_subpixel_offsets(precision):
    """The host oracle uses 64 epsilon units for sinc, window and contraction."""
    data = delivery(shape=(2, 2), half=24)
    grid = data["declared_lattice"]
    origin = np.array([999999.4625, 999999.4625])
    spacing = np.array([0.025, 0.025])
    lattice = source_grid(origin, spacing, grid.shape, grid.frame)
    indices = np.asarray(data["indices_xy"])
    samples = 2 + 0.25 * np.sin(0.3 * indices[:, 0])
    image = np.broadcast_to(samples[None, :, None, None], (1, len(samples), 2, 2))
    domain = sc.Domain("box", "closed", [[999999.98] * 2, [999999.99] * 2], [0, 0])
    reader = make_reader(
        dict(
            data,
            declared_lattice=lattice,
            domain=domain,
            offsets_xy_lod=origin + indices * spacing,
            intensity_nodes=image,
        )
    )
    points = np.array([[999999.98, 999999.985], [999999.99, 999999.985]])
    product = jnp.int32(0)
    batch = read_batch(reader, points, jnp.zeros(2, jnp.int32))
    for index, q in enumerate(points):
        phase = (q - origin_xy(lattice)) / spacing
        nodes = np.floor(phase).astype(int)[:, None] + np.arange(-15, 17)
        argument = phase[:, None] - nodes
        window = np.i0(12 * np.sqrt(1 - (argument / 16) ** 2)) / np.i0(12)
        wx, wy = np.sinc(argument) * window
        expected = np.sum(wy) * np.sum(wx * (2 + 0.25 * np.sin(0.3 * nodes[0])))
        result = reader.read(q, product)
        assert result.accepted and batch.accepted[index]
        np.testing.assert_array_equal(reader.stencil(q).nodes, nodes)
        np.testing.assert_allclose(
            [result.intensity[0, 0], batch.intensity[index, 0, 0]],
            expected,
            atol=64 * eps(precision) * abs(expected),
            rtol=0,
        )
    for edge, direction in ((points[0], -np.inf), (points[1], np.inf)):
        outside = edge.copy()
        outside[0] = np.nextafter(edge[0], direction)
        assert reader.read(outside, product).reason_bits & Reason.OUTSIDE_DOMAIN
