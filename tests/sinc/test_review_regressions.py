"""Admission and independent numerical regression controls."""

import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from optixstuff.grids import Frame

import yippy.sinc as sc
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import exact_quotient_floor_v1
from yippy.sinc.reader import estimate_memory

from .helpers import delivery, make_reader


@pytest.mark.parametrize("target", ["mirror", "axis"])
def test_relaxed_symmetry_tolerance_refused(quadrant_delivery, target):
    """Fails if caller tolerance bypasses corrupted mirror or axis evidence."""
    data = dict(quadrant_delivery)
    if target == "mirror":
        checks = data["mirror_checks"]
        data["mirror_checks"] = eqx.tree_at(
            lambda c: c.intensity, checks, checks.intensity.at[0, 0, 0].add(0.01)
        )
    else:
        data["intensity_nodes"] = data["intensity_nodes"].at[0, 0, 0, 0].add(0.01)
    data["reflection"] = eqx.tree_at(
        lambda r: r.tolerance, data["reflection"], jnp.array(1.0)
    )
    result = sc.admit_grid(**data)
    assert not result.admitted
    assert result.reason_bits & sc.Reason.SYMMETRY_UNSUPPORTED
    assert result.receipt.mirror_local_peak == ()
    assert result.receipt.axis_relative == ()


@pytest.mark.parametrize("half", [20, 50])
def test_source_count_memory_accounting(half):
    """Fails if source arrays escape the allocation gate on one-pixel stamps."""
    data = delivery(half=half, shape=(1, 1))
    data["intensity_nodes"] = data["intensity_nodes"].astype(jnp.float32)
    nodes = data["intensity_nodes"]
    n = nodes.shape[1]
    estimate = estimate_memory(nodes.nbytes, (1, 1), sc.Tiling(), 1, admission=True)
    # Independently count simultaneous source-sized buffers, excluding samples:
    # indices, offsets, cast indices, displacement, expected, residual workspace,
    # linear indices, occupancy, permutation and sorting workspace.
    required = n * (8 + 16 + 16 + 16 + 16 + 16 + 8 + 8 + 8 + 16)
    assert estimate["scratch"] >= required
    assert estimate["staging"] >= 2 * nodes.nbytes
    data["memory"] = sc.MemoryBudget(
        *(estimate[k] for k in ("resident", "output", "scratch", "total"))
    )
    result = sc.admit_grid(**data)
    assert result.admitted
    assert dict(result.receipt.allocation_estimate) == estimate
    # Hold sample bytes constant and vary the number of sources vs pixels.
    fewer_sources = estimate_memory(
        nodes.nbytes, (1, 4), sc.Tiling(), 1, admission=True
    )
    assert estimate["scratch"] > fewer_sources["scratch"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("pitch", 1e-300),
        ("pitch", 1e-7),
        ("pitch", 1e7),
        ("origin", 1e-300),
        ("origin", 1e7),
        ("annulus", 1e-200),
        ("annulus", 1e155),
        ("annulus", 1e-30),
        ("box", 1e7),
    ],
)
def test_supported_metadata_range(full_delivery, field, value):
    """Fails if unsupported lattice or domain scales can issue a reader."""
    data = dict(full_delivery)
    reason = sc.Reason.METADATA_CONFLICT
    if field in ("pitch", "origin"):
        lattice = data["declared_lattice"]
        origin = origin_xy(lattice) if field == "pitch" else [value, value]
        spacing = [value, value] if field == "pitch" else spacing_xy(lattice)
        if field == "pitch" and value < 1e-150:
            with pytest.raises(ValueError, match="nonsingular"):
                sc.source_grid(origin, spacing, lattice.shape, lattice.frame)
            return
        data["declared_lattice"] = sc.source_grid(
            origin, spacing, lattice.shape, lattice.frame
        )
        reason = sc.Reason.LAYOUT_UNSUPPORTED
    elif field == "annulus":
        data["domain"] = sc.Domain(
            "annulus", "closed", jnp.zeros((2, 2)), jnp.array([value, 2 * value])
        )
    else:
        data["domain"] = sc.Domain(
            "box", "closed", jnp.array([[-value, -value], [value, value]]), jnp.zeros(2)
        )
    result = sc.admit_grid(**data)
    assert not result.admitted
    assert result.reason_bits & reason


def test_annulus_inclusive_coverage_corner(full_delivery):
    """Fails if safe-point selection discards an inclusive coverage corner."""
    data = dict(full_delivery)
    iy, ix = np.indices((32, 32), dtype=np.int32)
    indices = np.stack((ix.ravel(), iy.ravel()), axis=1)
    data.update(
        declared_lattice=sc.source_grid(
            jnp.zeros(2),
            jnp.full(2, 0.25),
            (32, 32),
            Frame(500.0, 6.0, "circumscribed"),
        ),
        indices_xy=indices,
        offsets_xy_lod=indices * 0.25,
        intensity_nodes=jnp.ones((1, 1024, 8, 6)),
        domain=sc.Domain(
            "annulus",
            "closed",
            jnp.zeros((2, 2)),
            jnp.array([0.0, math.hypot(3.75, 3.75)]),
        ),
    )
    result = sc.admit_grid(**data)
    assert result.admitted
    reader = make_reader(data)
    read = eqx.filter_jit(lambda r, q: r.read(q, jnp.int32(0)))(
        reader, jnp.array([3.75, 3.75])
    )
    assert read.accepted


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_negative_subnormal_samples(full_delivery, dtype):
    """Fails if floating comparisons flush negative original samples to zero."""
    nodes = np.array(full_delivery["intensity_nodes"], dtype=dtype)
    nodes[0, 0, 0, 0] = -np.nextafter(dtype(0), dtype(1))
    result = sc.admit_grid(**{**full_delivery, "intensity_nodes": nodes})
    assert not result.admitted and result.reason_bits & sc.Reason.DATA_INVALID
    nodes[0, 0, 0, 0] = -0.0
    nodes[0, 0, 0, 1] = 0.0
    assert sc.admit_grid(**{**full_delivery, "intensity_nodes": nodes}).admitted


def test_original_offset_precision_after_conversion(full_delivery):
    """Fails if unit conversion loses the original four-ULP precision."""
    arcsec_per_lod = (500e-9 / 6.0) * (180.0 * 3600.0 / np.pi)
    original = (np.asarray(full_delivery["offsets_xy_lod"]) * arcsec_per_lod).astype(
        np.float32
    )
    data = {**full_delivery, "offsets_xy_lod": original.astype(float) / arcsec_per_lod}
    refused = sc.admit_grid(**data)
    assert not refused.admitted and refused.reason_bits & sc.Reason.GRID_NONUNIFORM
    admitted = sc.admit_grid(**data, offset_original_dtype=jnp.float32)
    assert admitted.admitted and admitted.receipt.offset_dtype == "float32"
    scale = np.max(np.abs(data["offsets_xy_lod"]), axis=0).astype(np.float32)
    tau = 4 * (np.nextafter(scale, np.float32(np.inf)) - scale)
    np.testing.assert_array_equal(admitted.receipt.offset_tolerance_xy, tau)


def test_provenance_retained_and_hashed(full_delivery):
    """Fails if full provenance text is lost or omitted from receipt identity."""
    provenance = (
        ("response", "full argument\nwith units"),
        ("quadrature", "nodes=..."),
    )
    a = sc.admit_grid(**full_delivery, provenance=provenance)
    b = sc.admit_grid(**full_delivery, provenance=(*provenance, ("response", "other")))
    assert a.admitted and b.admitted
    assert a.receipt.provenance == provenance
    assert a.receipt.identity != b.receipt.identity
    assert a.require().receipt_id == a.receipt.identity


def test_axis_exact_separate_from_relative(quadrant_delivery):
    """Fails if exact and relative evidence merge or off-frame nodes lose it."""
    result = sc.admit_grid(**quadrant_delivery)
    assert result.admitted
    exact = np.asarray(result.receipt.axis_exact)
    relative = np.asarray(result.receipt.axis_relative)
    assert exact.dtype == np.bool_ and exact.shape == relative.shape
    # Far axis images have no pixel within 1 lambda/D of their source; their
    # residual is relative to the image maximum instead of undefined.
    assert np.all(np.isfinite(relative))
    far = 20 * 21
    nodes = quadrant_delivery["intensity_nodes"].at[0, far, 0, 0].add(0.01)
    bad = sc.admit_grid(**{**quadrant_delivery, "intensity_nodes": nodes})
    assert not bad.admitted and bad.reason_bits & sc.Reason.SYMMETRY_CHECK_FAILED
    assert any(
        not e and r > 1e-6
        for e, r in zip(bad.receipt.axis_exact, bad.receipt.axis_relative, strict=True)
    )
    # A zero image has no positive denominator: only exact identity passes it.
    nodes = quadrant_delivery["intensity_nodes"].at[0, far].set(0.0)
    blank = sc.admit_grid(**{**quadrant_delivery, "intensity_nodes": nodes})
    assert blank.admitted
    assert any(
        e and math.isnan(r)
        for e, r in zip(
            blank.receipt.axis_exact, blank.receipt.axis_relative, strict=True
        )
    )


def test_off_frame_axis_node_admitted_within_tolerance(quadrant_delivery):
    """Fails if an off-frame axis image symmetric to rounding is refused."""
    far = 20 * 21
    image = quadrant_delivery["intensity_nodes"][0, far]
    scale = float(jnp.max(image))
    assert scale > 0
    # An asymmetry far below the tolerance, which bit identity alone refuses.
    nudged = image.at[0, 0].add(1e-9 * scale)
    nodes = quadrant_delivery["intensity_nodes"].at[0, far].set(nudged)
    result = sc.admit_grid(**{**quadrant_delivery, "intensity_nodes": nodes})
    assert result.admitted


floor_modes = (
    exact_quotient_floor_v1,
    eqx.filter_jit(exact_quotient_floor_v1),
    eqx.filter_jit(jax.vmap(exact_quotient_floor_v1)),
)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_supported_range_floor_fraction_property(seed):
    """Fails if supported scales or edge neighbours select an inexact floor."""
    rng = np.random.default_rng(seed)
    origins = 10.0 ** rng.uniform(-6, 6, 64) * rng.choice([-1, 1], 64)
    origins[:3] = [0.0, 1e-6, 1e6]
    pitches = 10.0 ** rng.uniform(-6, 6, 64)
    pitches[:3] = [1e-6, 1e6, 0.38]
    edges = origins + pitches * rng.integers(-100, 101, 64)
    queries = np.concatenate(
        [
            edges,
            np.nextafter(edges, -np.inf),
            np.nextafter(edges, np.inf),
            origins,
            np.nextafter(origins, -np.inf),
            np.nextafter(origins, np.inf),
            origins + rng.uniform(-100, 100, 64) * pitches,
        ]
    )
    origins = np.tile(origins, 7)
    pitches = np.tile(pitches, 7)
    expected = [
        math.floor(
            Fraction((0.0 if abs(q) < np.finfo(float).tiny else q) - o) / Fraction(h)
        )
        for q, o, h in zip(queries, origins, pitches, strict=True)
    ]
    args = tuple(jnp.asarray(x) for x in (queries, origins, pitches))
    for fn in floor_modes:
        np.testing.assert_array_equal(fn(*args), expected)


@pytest.mark.parametrize("quadrant", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independent_reference_random_and_edges(quadrant, dtype):
    """Fails if a shared stencil or detector-flip defect escapes the oracle."""
    from .reference_reader import reference_read

    data = delivery(quadrant=quadrant, half=34, shape=(6, 9))
    data["intensity_nodes"] = data["intensity_nodes"].astype(dtype)
    data["domain"] = sc.Domain(
        "box", "closed", jnp.array([[-4.1, -4.1], [4.1, 4.1]]), jnp.zeros(2)
    )
    reader = make_reader(data)
    queries = list(np.random.default_rng(0).uniform(-4.1, 4.1, (8, 2)))
    queries += [
        [0.0, -0.0],
        [-5e-324, 5e-324],
        [4.01, 0.2],
        [4.01, 4.01],
        [-4.01, 4.01],
        [-4.01, -4.01],
    ]
    queries += [
        [np.nextafter(x, direction), 0.125]
        for x in [-0.25, 0.0, 0.25]
        for direction in [-np.inf, np.inf]
    ]
    q = jnp.asarray(queries)
    result = eqx.filter_jit(sc.read_batch)(reader, q, jnp.zeros(len(q), jnp.int32))
    assert np.all(result.accepted)
    for i, query in enumerate(queries):
        expected, scale = reference_read(data, query)
        assert np.all(
            np.abs(result.intensity[i] - expected) <= 32 * np.finfo(float).eps * scale
        )


def test_independent_reference_detects_one_tap_shift(full_delivery, full_reader):
    """Fails if the analytic query cannot distinguish a one-tap y shift."""
    from .reference_reader import reference_read

    query = np.array([0.137, -0.219])
    expected, scale = reference_read(full_delivery, query)
    mutant, _ = reference_read(full_delivery, query, shift_y=-1)
    allowance = 32 * np.finfo(float).eps * scale
    actual = eqx.filter_jit(lambda r, q: r.read(q, jnp.int32(0)))(
        full_reader, jnp.asarray(query)
    ).intensity
    assert np.all(np.abs(actual - expected) <= allowance)
    assert np.any(np.abs(mutant - expected) > allowance)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_axis_exact_preserves_subnormal_discrepancy(quadrant_delivery, dtype):
    """Fails if exact axis identity flushes a nonzero sample discrepancy."""
    nodes = np.array(quadrant_delivery["intensity_nodes"], dtype=dtype)
    # This far axis image has no positive local reference. A lone positive
    # subnormal is not exactly invariant, even if float comparisons flush it.
    nodes[0, 20 * 21] = 0
    nodes[0, 20 * 21, 0, 0] = np.nextafter(dtype(0), dtype(1))
    result = sc.admit_grid(**{**quadrant_delivery, "intensity_nodes": nodes})
    assert not result.admitted
    assert result.reason_bits & sc.Reason.SYMMETRY_CHECK_FAILED
    assert not result.receipt.axis_exact[20]
    assert math.isnan(result.receipt.axis_relative[20])


def test_reflection_tolerance_range_on_full_grid(full_delivery):
    """Fails if an inactive reflection declaration can carry a relaxed tolerance."""
    reflection = eqx.tree_at(
        lambda r: r.tolerance, full_delivery["reflection"], jnp.array(1.0)
    )
    result = sc.admit_grid(**{**full_delivery, "reflection": reflection})
    assert not result.admitted
    assert result.reason_bits & sc.Reason.SYMMETRY_UNSUPPORTED


# Built under binary64 because collection runs outside the sinc fixtures.
with jax.enable_x64(True):
    UNSUPPORTED_DOMAINS = [
        sc.Domain("annulus", "closed", jnp.zeros((2, 2)), jnp.array([1e-200, 2e-200])),
        sc.Domain(
            "annulus",
            "closed",
            jnp.zeros((2, 2)),
            jnp.array([0.0, math.nextafter(1e6, math.inf)]),
        ),
        sc.Domain(
            "box",
            "closed",
            jnp.array([[-1.0, -1.0], [math.nextafter(1e6, math.inf), 1.0]]),
            jnp.zeros(2),
        ),
    ]


@pytest.mark.parametrize("domain", UNSUPPORTED_DOMAINS)
def test_reader_constructor_enforces_domain_range(full_reader, domain):
    reader = full_reader
    with pytest.raises(ValueError, match="supported numeric range"):
        sc.Interpolator(
            reader.grid,
            domain,
            reader.profile,
            reader.policy,
            reader.tiling,
            reader.memory,
        )
