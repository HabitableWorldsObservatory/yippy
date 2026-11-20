"""Prefiltered profile: filter design, coverage, reads, evidence and refusals."""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.grids import Frame

import yippy.sinc as sc
from yippy.sinc import prefilter
from yippy.sinc.domain import coverage
from yippy.sinc.evidence import BoundKind, Reason, Status
from yippy.sinc.geometry import origin_xy, source_grid
from yippy.sinc.grid import CheckSet
from yippy.sinc.kernel import PrefilteredKernelProfile, bump_stencil
from yippy.sinc.prefilter import (
    build_coefficients,
    bump_transform,
    design_axis,
    design_grid,
)
from yippy.sinc.reader import Interpolator, PrefilteredReader, read_batch, require_read

from .helpers import delivery, make_prefiltered, tone_truth
from .reference_reader import (
    _bump,
    reference_prefiltered_read,
    reference_start,
)

compiled_read = eqx.filter_jit(lambda reader, q, p: reader.read(q, p))
BUDGET = sc.MemoryBudget(100_000_000, 10_000_000, 60_000_000, 400_000_000)


def figures(record):
    return dict(record)


def oracle(data, reader, query):
    design = figures(reader.design)
    return reference_prefiltered_read(
        data,
        np.asarray(query),
        reader.profile.taps,
        reader.profile.filter_half_width,
        (design["band_x"], design["band_y"]),
        (design["beta_x"], design["beta_y"]),
    )


def admitted_samples(data):
    """The canonical samples a prefiltered reader of ``data`` was built from."""
    admission = sc.admit_grid(**data, profile=PrefilteredKernelProfile())
    return admission.require().intensity


@pytest.fixture(scope="module")
def full(tone_full):
    return tone_full


@pytest.fixture(scope="module")
def quadrant(tone_quadrant):
    return tone_quadrant


@pytest.mark.medium(reason="one filter design per tap count, up to 7 s")
@pytest.mark.parametrize(
    "taps,half_width,tone,beta,filter_l1",
    [
        (7, 12, 9.816e-6, 15.086, 27.49),
        (8, 16, 1.629e-6, 17.241, 52.95),
        (9, 16, 3.099e-7, 19.396, 86.85),
    ],
)
def test_design_figures_at_reference_band(taps, half_width, tone, beta, filter_l1):
    """Fails if the bump shape, the fit or the tone-error evaluation changes.

    Reference values are the interior figures of the least-squares in-band
    inverse of a Kaiser-Bessel bump at a band edge of 0.3 cycles per node (the
    deconvolved-kernel construction of nonuniform FFTs: Fessler and Sutton
    2003; Barnett, Magland and af Klinteberg 2019). They are re-baselinable:
    2 percent is far above solver noise and far below the factor of about six
    between neighbouring tap counts.
    """
    design = design_axis(taps, half_width, 0.3, 45, False, (20, 24))
    np.testing.assert_allclose(design.interior_tone_error, tone, rtol=2e-2)
    np.testing.assert_allclose(design.tone_error, tone, rtol=2e-2)
    np.testing.assert_allclose(design.beta, beta, rtol=1e-3)
    np.testing.assert_allclose(
        jnp.sum(jnp.abs(design.direct[22])), filter_l1, rtol=1e-2
    )
    # The closed-form transform agrees with quadrature of the cut bump.
    grid = np.linspace(-taps / 2, taps / 2, 40001)
    shape = _bump(grid, taps, design.beta)
    shape[0] = shape[-1] = 0.5 / np.i0(design.beta)
    frequency = np.array([0.0, 0.11, 0.3, 0.45])
    quadrature = (np.cos(2 * np.pi * frequency[:, None] * grid) * shape).sum(1) * (
        grid[1] - grid[0]
    )
    np.testing.assert_allclose(
        bump_transform(jnp.asarray(frequency), taps, design.beta),
        quadrature,
        rtol=1e-8,
    )


@pytest.mark.medium(reason="dense tone scan of an assembled design, 5 s")
@pytest.mark.parametrize(
    "taps,half_width,band,size,reflected",
    [
        (7, 12, 0.3, 27, False),
        (8, 16, 0.32, 18, True),
        (9, 16, 0.3, 36, False),
        (7, 12, 0.15, 27, False),
    ],
)
def test_assembled_filters_meet_recorded_tone_error(
    taps, half_width, band, size, reflected
):
    """Fails if a filter row is misplaced, mirrored wrongly or under-reported.

    The recorded figure is computed from per-node filters; this applies the
    assembled matrices to tones and reads them with an exact rational stencil
    on a denser phase grid that includes switch points, their neighbours and
    every edge cell. The recorded figure must not be exceeded (0.1 percent for
    rounding) and must not be loose by more than a factor of two. The band of
    0.15 is an oversampled lattice, where the fit is rank deficient and the
    pinned singular-value cutoff selects the filter.
    """
    design = design_axis(taps, half_width, band, size, reflected, None)
    direct = np.asarray(design.direct)
    stored = np.arange(size)
    low = 1 - size if reflected else 0
    odd = taps % 2
    start = (taps - 1) / 2 + 5.5 if odd else taps / 2 + 5.0
    stop = size - ((taps - 1) / 2 + 6.5 if odd else taps / 2 + 6.0)
    first = -stop if reflected else start
    phases = np.arange(first, stop, 1 / 16)
    switches = np.arange(math.ceil(first), stop, 1.0) + (0.5 if odd else 0.0)
    switches = switches[(switches > first) & (switches < stop)]
    phases = np.concatenate([phases, switches, switches - 1e-9, switches + 1e-9])
    worst = 0.0
    for frequency in np.linspace(0.0, band, 41):
        tone = np.exp(2j * np.pi * frequency * stored)
        coefficient = direct @ tone
        if reflected:
            coefficient = coefficient + np.asarray(design.mirrored) @ np.conj(tone)
        for phase in phases:
            begin = reference_start(float(phase), 0.0, 1.0, taps)
            nodes = np.arange(begin, begin + taps)
            assert nodes[0] - 6 >= low and nodes[-1] + 6 <= size - 1
            weight = _bump(phase - nodes, taps, design.beta) / design.scale
            value = np.where(
                nodes < 0,
                np.conj(coefficient[np.abs(nodes)]),
                coefficient[np.abs(nodes)],
            )
            response = np.sum(weight * value) * np.exp(-2j * np.pi * frequency * phase)
            worst = max(worst, abs(response - 1.0))
    assert worst <= design.tone_error * (1 + 1e-3)
    assert worst >= 0.5 * design.tone_error


@pytest.mark.parametrize("taps", [5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("reflected", [False, True])
def test_covered_interval_endpoints(taps, reflected):
    """Fails on a halo or stencil off-by-one: the covered phases are exact.

    Phases are dyadic so every endpoint is representable. Covered reads need
    their taps plus six stored nodes on each side: (taps - 1) / 2 + 5.5 <= t <
    size - (taps - 1) / 2 - 6.5 for odd taps and taps / 2 + 5 <= t <
    size - taps / 2 - 6 for even taps; a mirror-symmetric table extends the
    lower limit to minus the upper one, included.
    """
    size, spacing = 40, 0.25
    lattice = source_grid(
        jnp.zeros(2),
        jnp.full(2, spacing),
        (size, size),
        Frame(500.0, 6.0, "circumscribed"),
    )
    profile = PrefilteredKernelProfile(taps=taps)
    below = (taps - 1) / 2 + 5.5 if taps % 2 else taps / 2 + 5.0
    above = size - below - 1.0
    low = -above * spacing if reflected else below * spacing
    high = above * spacing
    inside = [low, math.nextafter(high, -math.inf), 0.5 * (low + high)]
    outside = [math.nextafter(low, -math.inf), high, math.inf, math.nan]
    middle = 0.5 * (low + high)
    for value, expected in [(v, True) for v in inside] + [(v, False) for v in outside]:
        for query in ([value, middle], [middle, value]):
            covered = coverage(jnp.array(query), lattice, reflected, profile)
            assert bool(covered) is expected, (value, query)


@pytest.mark.parametrize("taps", [7, 8])
def test_stencil_modes_agree_at_switch_points(full, taps):
    """Fails if eager, jit and vmap pick different taps where a stencil switches.

    Integer taps must equal exact rational arithmetic in every mode, including
    one-ulp neighbours and negative phases. The read jumps at a switch because
    the bump is cut at 1 / I0(beta): the tap-weight sum just below a switch
    exceeds the sum at it by exactly the edge weight.
    """
    profile = PrefilteredKernelProfile(taps=taps)
    origin, spacing = jnp.array([0.0, -1.5]), jnp.array([0.25, 0.375])
    beta, scale = jnp.array([15.1, 17.2]), jnp.array([2.2, 2.4])
    values = []
    for cell in (-7, -1, 0, 3):
        for fraction in (0.0, 0.5):
            x = (cell + fraction) * 0.25
            values += [x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)]
    queries = jnp.asarray([[v, v * 1.5 - 1.5] for v in values])

    def one(query):
        return bump_stencil(query, origin, spacing, profile, beta, scale)

    eager = [one(q) for q in queries]
    mapped = eqx.filter_jit(jax.vmap(one))(queries)
    compiled = eqx.filter_jit(one)
    for index, query in enumerate(queries):
        expected = [
            reference_start(float(query[a]), float(origin[a]), float(spacing[a]), taps)
            for a in range(2)
        ]
        for nodes, weight in (eager[index], compiled(query)):
            np.testing.assert_array_equal(nodes[:, 0], expected)
            np.testing.assert_array_equal(nodes, mapped[0][index])
            np.testing.assert_allclose(weight, mapped[1][index], rtol=1e-13, atol=0)
    switch = (0.5 if taps % 2 else 0.0) * 0.25
    at = jnp.sum(one(jnp.array([switch, 0.0]))[1][0])
    # One ulp below, the rounded tap argument is still the excluded edge value.
    before = jnp.sum(one(jnp.array([switch - 1e-9, 0.0]))[1][0])
    edge = 1.0 / (float(jnp.i0(beta[0])) * float(scale[0]))
    np.testing.assert_allclose(before - at, edge, rtol=1e-5)


QUERIES = [
    [0.137, -0.219],
    [-1.31, 0.77],
    [0.15, -0.14],
    [0.0, 0.0],
    [-0.3, 0.28],
    [1.999, -1.999],
    [-2.0, 2.0],
]


@pytest.mark.medium(reason="host quadrature oracle per query, 6 s")
@pytest.mark.parametrize("fixture", ["full", "quadrant"])
def test_reads_match_independent_oracle(fixture, request):
    """Fails on a wrong flip, axis swap, filter offset, half-width or stencil.

    The oracle unfolds the table, fits its own filters with a quadrature bump
    transform and sums taps by exact rational stencils. Table, detector and
    band differ between x and y. Two independent least-squares fits agree to
    about 1e-9 of the signal; a misplaced tap or flip costs at least 1e-6.
    """
    case = request.getfixturevalue(fixture)
    reader, data = case["reader"], case["data"]
    for query in QUERIES:
        result = compiled_read(reader, jnp.array(query), jnp.int32(0))
        assert bool(result.accepted)
        np.testing.assert_allclose(
            result.intensity,
            oracle(data, reader, query),
            rtol=0,
            atol=2e-8 * case["norm"],
        )


def test_reads_track_band_limited_truth(full, quadrant):
    """Fails if a read drifts from truth beyond the design figure's scale.

    The truth holds tones up to the declared band edge, so a read error is at
    most the two-axis tone error times the l1 norm of the spectrum, were the
    recorded figure a true supremum. It is a sampled maximum, so this is a
    consistency check with a factor 1.5, not a bound validation. The quadrant
    delivery of the same symmetric scene must give the full delivery's reads.
    """
    design = figures(full["reader"].design)
    allowance = 1.5 * design["tone_error_two_axis"] * full["norm"]
    worst = 0.0
    for query in QUERIES:
        q = jnp.array(query)
        whole = compiled_read(full["reader"], q, jnp.int32(0))
        folded = compiled_read(quadrant["reader"], q, jnp.int32(0))
        error = float(jnp.max(jnp.abs(whole.intensity - full["truth"](q))))
        worst = max(worst, error)
        assert error <= allowance
        np.testing.assert_allclose(
            folded.intensity, whole.intensity, rtol=0, atol=1e-11 * full["norm"]
        )
        assert whole.status == Status.SAMPLING_ADMITTED
        assert whole.reason_bits == Reason.BOUND_UNKNOWN
        assert not whole.bound_available and math.isinf(float(whole.abs_error_bound))
        assert int(whole.bound_kind) == BoundKind.UNKNOWN
        assert not np.any(np.asarray(whole.ledger.evidence))
        assert bool(folded.ledger.applicable[-1])
    # The kernel is approximate: errors sit near the design figure, not at zero.
    assert worst >= 0.02 * allowance


def test_receipt_records_the_kernel_design(full):
    """Fails if the design figures leave the receipt or masquerade as a bound."""
    record = figures(full["receipt"].kernel_design)
    assert full["receipt"].kernel_design == full["reader"].design
    assert record["taps"] == 7 and type(record["taps"]) is int
    assert record["filter_half_width"] == 12 and record["halo"] == 6
    np.testing.assert_allclose([record["band_x"], record["band_y"]], [0.3, 0.28])
    ex, ey = record["tone_error_x"], record["tone_error_y"]
    np.testing.assert_allclose(record["tone_error_two_axis"], ex + ey + ex * ey)
    assert 1e-6 < ey < ex < 1.2e-5
    assert 1.9 < record["sample_gain_x"] < 2.4
    profile = PrefilteredKernelProfile()
    assert full["receipt"].profile_id == profile.identity
    assert full["reader"].grid.profile_id == profile.identity
    assert profile.identity == (
        "kaiser_bessel_prefiltered_v1:taps7:filter12:halo6:exact_quotient_floor_v1"
    )
    raw = sc.admit_grid(**full["data"])
    assert raw.receipt.kernel_design == ()
    assert raw.grid.profile_id == sc.KernelProfile().identity


def test_profiles_do_not_mix(full):
    """Fails if a grid admitted under one profile is read with another."""
    data, grid = full["data"], full["reader"].grid
    arguments = (sc.ReadPolicy(), sc.Tiling(), data["memory"])
    with pytest.raises(ValueError, match="different kernel profile"):
        Interpolator(grid, data["domain"], sc.KernelProfile(), *arguments)
    with pytest.raises(ValueError, match="different kernel profile"):
        PrefilteredReader(
            grid, data["domain"], PrefilteredKernelProfile(taps=8), *arguments
        )
    raw = sc.admit_grid(**data).require()
    with pytest.raises(ValueError, match="different kernel profile"):
        PrefilteredReader(raw, data["domain"], PrefilteredKernelProfile(), *arguments)
    with pytest.raises(TypeError):
        Interpolator(raw, data["domain"], PrefilteredKernelProfile(), *arguments)
    for taps in (4, 11, 7.0):
        with pytest.raises(ValueError, match="5 to 10 taps"):
            PrefilteredKernelProfile(taps=taps)


@pytest.mark.medium(reason="admits and builds a table per case, 4 s")
@pytest.mark.parametrize("quadrant_table", [False, True])
def test_table_edge_reads_and_halo_refusal(quadrant_table):
    """Fails if an edge read uses unstored nodes or the halo limit moves.

    The domain is wider than coverage, so coverage alone decides. The last
    covered phase on each side uses one-sided edge filters and must still match
    the oracle; just beyond refuses with INCOMPLETE_STENCIL and a NaN image.
    """
    truth, norm = tone_truth((5, 4))
    data = delivery(
        quadrant_table, half=14, shape=(5, 4), spacing=0.25, truth=truth, limit=6.0
    )
    data["memory"] = BUDGET
    reader, _ = make_prefiltered(data)
    size = data["declared_lattice"].shape[0]
    origin = float(origin_xy(data["declared_lattice"])[0])
    # Dyadic spacing makes both limits exact: the lower is covered, the upper not.
    high = origin + 0.25 * (size - 9.5)
    low = -high if quadrant_table else origin + 0.25 * 8.5
    # A nonzero origin rounds one-ulp neighbours back onto the limit, so step 1e-9.
    inside = [high - 1e-9, low]
    outside = [high, low - 1e-9]
    for value in inside:
        query = [value, 0.11]
        result = reader.read(jnp.array(query), jnp.int32(0))
        assert bool(result.accepted)
        np.testing.assert_allclose(
            result.intensity, oracle(data, reader, query), rtol=0, atol=2e-8 * norm
        )
    for value in outside:
        result = reader.read(jnp.array([0.11, value]), jnp.int32(0))
        assert not bool(result.accepted) and not bool(result.sampling_ok)
        assert int(result.reason_bits) & Reason.INCOMPLETE_STENCIL
        assert bool(jnp.all(jnp.isnan(result.intensity)))


@pytest.mark.parametrize(
    "half,quadrant_table,admitted",
    [(11, False, False), (12, False, True), (11, True, False), (12, True, True)],
)
def test_minimum_table_size(half, quadrant_table, admitted):
    """Fails if a table too small for the filter half-width is admitted.

    Seven taps need 25 signed nodes (13 stored in a quadrant), so that every
    coefficient node has the full 12-node filter on at least one side.
    """
    truth, _ = tone_truth((4, 4))
    data = delivery(
        quadrant_table, half=half, shape=(4, 4), spacing=0.3, truth=truth, limit=0.2
    )
    data["memory"] = BUDGET
    admission = sc.admit_grid(**data, profile=PrefilteredKernelProfile())
    assert admission.admitted is admitted
    if not admitted:
        assert admission.reason_bits == Reason.INCOMPLETE_STENCIL


def test_margin_memory_policy_and_precision(full):
    """Fails if the 0.64 margin, byte budgets, bounded policy or x64 guard slip."""
    truth, _ = tone_truth((4, 4))
    coarse = delivery(half=20, shape=(4, 4), spacing=0.33, truth=truth)
    coarse["memory"] = BUDGET
    refused = sc.admit_grid(**coarse, profile=PrefilteredKernelProfile())
    assert refused.reason_bits == Reason.KERNEL_MARGIN_FAIL
    assert sc.admit_grid(**coarse).admitted
    data = full["data"]
    small = {**data, "memory": sc.MemoryBudget(100_000_000, 10_000_000, 10**6, 10**9)}
    assert (
        sc.admit_grid(**small, profile=PrefilteredKernelProfile()).reason_bits
        == Reason.MEMORY_LIMIT
    )
    estimate = full["reader"].memory_estimate(1)
    # The coefficients replace the samples: one resident table.
    assert estimate["resident"] == full["reader"].coefficients.nbytes
    assert full["reader"].grid.stored == "prefiltered_coefficients"
    bounded = eqx.tree_at(
        lambda r: r.policy, full["reader"], sc.ReadPolicy("bounded", jnp.array(1.0))
    )
    result = bounded.read(jnp.array([0.1, 0.2]), jnp.int32(0))
    assert not bool(result.accepted) and bool(result.sampling_ok)
    assert int(result.status) == Status.REFUSED
    # The design is cached by its arguments; the precision check is not.
    cached = (7, 12, 0.29, 31, False, (12, 14))
    design_axis(*cached)
    with jax.enable_x64(False):
        with pytest.raises(sc.PrecisionUnavailable):
            design_axis(*cached)
        with pytest.raises(sc.PrecisionUnavailable):
            build_coefficients(full["reader"].grid.intensity, ())


def test_batch_checks_and_node_reads(full):
    """Fails if batching, admission checks or node reads bypass the new reader.

    A prefiltered read does not interpolate: at a lattice node it differs from
    the stored sample, by no more than the design figure's scale.
    """
    reader, data, truth = full["reader"], full["data"], full["truth"]
    design = figures(reader.design)
    allowance = 1.5 * design["tone_error_two_axis"] * full["norm"]
    queries = jnp.array([[0.137, -0.219], [0.3, 0.56], [9.0, 0.0]])
    batch = read_batch(reader, queries, jnp.zeros(3, dtype=jnp.int32))
    assert batch.accepted.tolist() == [True, True, False]
    for index in range(2):
        single = reader.read(queries[index], jnp.int32(0))
        np.testing.assert_allclose(
            batch.intensity[index], single.intensity, rtol=1e-13, atol=0
        )
    with pytest.raises(sc.ReadRefused):
        require_read(batch)
    empty = read_batch(reader, jnp.zeros((0, 2)), jnp.zeros(0, dtype=jnp.int32))
    assert empty.intensity.shape == (0, *data["products"].pixels.shape)
    node = jnp.array([0.3, 0.56])
    stored = truth(node)
    deviation = float(jnp.max(jnp.abs(batch.intensity[1] - stored)))
    assert 0.0 < deviation <= allowance
    positions = jnp.array([[0.41, -0.77], [-1.2, 0.3]])
    images = jax.vmap(truth)(positions)
    checks = CheckSet(
        positions,
        images,
        jnp.zeros(2, dtype=jnp.int32),
        jnp.max(images, axis=(1, 2)),
        "independent",
        "closed_form",
        "synthetic",
    )
    profile = PrefilteredKernelProfile()
    admission = sc.admit_grid(**data, profile=profile, checks=checks)
    assert 0.0 < admission.receipt.empirical_max_abs <= allowance
    checked = PrefilteredReader(
        admission.require(),
        data["domain"],
        profile,
        sc.ReadPolicy(),
        sc.Tiling(),
        data["memory"],
    )
    result = checked.read(positions[0], jnp.int32(0))
    assert int(result.reason_bits) == (Reason.BOUND_UNKNOWN | Reason.EMPIRICAL_ONLY)
    np.testing.assert_allclose(
        result.empirical_max_abs, admission.receipt.empirical_max_abs
    )


def test_float32_storage_stays_near_float64(full):
    """Fails if float32 coefficients are filtered in float32 or lose their scale.

    The filter runs in float64 and rounds once to the storage dtype, so the
    stored coefficients equal, bit for bit, the float64 build of the same
    float32 samples rounded to float32. A float32 filter pass moves them by
    about 4e-6 of the signal. The read of a float32 delivery then differs from
    the float64 one by rounding of samples and coefficients only: a few 1e-7 of
    the signal, measured 2e-7.
    """
    data = full["data"]
    single = {
        **data,
        "intensity_nodes": data["intensity_nodes"].astype(jnp.float32),
        "mirror_checks": None,
    }
    reader, _ = make_prefiltered(single)
    assert reader.coefficients.dtype == jnp.float32
    designs = design_grid(
        reader.profile, reader.grid.products, reader.grid.source, False, data["domain"]
    )
    wide = build_coefficients(admitted_samples(single).astype(jnp.float64), designs)
    np.testing.assert_array_equal(reader.coefficients, wide.astype(jnp.float32))
    worst = 0.0
    for query in QUERIES[:4]:
        q = jnp.array(query)
        exact = compiled_read(full["reader"], q, jnp.int32(0)).intensity
        rounded = reader.read(q, jnp.int32(0)).intensity
        assert rounded.dtype == jnp.float64
        worst = max(worst, float(jnp.max(jnp.abs(rounded - exact))))
    assert worst <= 1e-6 * full["norm"]


@pytest.mark.parametrize("fixture", ["full", "quadrant"])
def test_chunked_build_matches_single_chunk(fixture, request, monkeypatch):
    """Fails if a detector-row chunk is flipped or joined in the wrong order.

    With reflection each chunk is filtered together with the mirrored rows of
    the opposite detector edge. A chunk limit of one byte forces one row per
    chunk on the 6-row detector; the result must equal the single-chunk build
    to a few units in the last place (the compiler may vectorize the two chunk
    shapes differently), where a flipped or misordered row differs at order one.
    """
    case = request.getfixturevalue(fixture)
    reader, grid = case["reader"], case["reader"].grid
    samples = admitted_samples(case["data"])
    designs = design_grid(
        reader.profile,
        grid.products,
        grid.source,
        grid.reflection.kind != "none",
        reader.domain,
    )
    nodes = samples.shape[1] * samples.shape[2]
    assert prefilter.build_chunk_rows(nodes, (6, 5)) == 6
    monkeypatch.setattr(prefilter, "BUILD_CHUNK_BYTES", 1)
    assert prefilter.build_chunk_rows(nodes, (6, 5)) == 1
    chunked = build_coefficients(samples, designs)
    scale = float(jnp.max(jnp.abs(reader.coefficients)))
    np.testing.assert_allclose(
        chunked, reader.coefficients, rtol=0, atol=16 * np.finfo(float).eps * scale
    )


def test_coefficient_grid_is_not_read_as_samples(full):
    """Fails if a reader accepts a grid that already holds coefficients.

    A prefiltered reader keeps its coefficients in place of the samples, so its
    grid must not be filtered a second time or read raw as if it were samples.
    """
    grid, data = full["reader"].grid, full["data"]
    assert grid.stored == "prefiltered_coefficients"
    profile = PrefilteredKernelProfile()
    with pytest.raises(ValueError, match="samples"):
        PrefilteredReader(
            grid, data["domain"], profile, sc.ReadPolicy(), sc.Tiling(), data["memory"]
        )
    raw = dataclasses.replace(grid, profile_id="")
    with pytest.raises(ValueError, match="samples"):
        Interpolator(
            raw,
            data["domain"],
            sc.KernelProfile(),
            sc.ReadPolicy(),
            sc.Tiling(),
            data["memory"],
        )
