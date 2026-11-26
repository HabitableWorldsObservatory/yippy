"""Extended-scene rendering against brute-force sums of reads."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import roundoff
from hwoutils.precision import PrecisionError
from jax.test_util import check_grads

import yippy.sinc as sc
from yippy.sinc import scene as scene_module
from yippy.sinc.evidence import Reason, Status
from yippy.sinc.geometry import origin_xy, source_grid, spacing_xy
from yippy.sinc.reader import Interpolator, read_batch

from .helpers import make_prefiltered

render_points = eqx.filter_jit(lambda renderer, q, f: renderer.render_points(q, f))
render_raster = eqx.filter_jit(
    lambda renderer, scene, x, y: renderer.render_raster(scene, x, y)
)


def brute(reader, points, flux):
    reads = read_batch(reader, points, jnp.zeros(points.shape[0], dtype=jnp.int32))
    assert bool(jnp.all(reads.accepted))
    return jnp.einsum("q,qrc->rc", flux, reads.intensity)


def scene_points():
    points = jax.random.uniform(jax.random.key(0), (9, 2), minval=-1.8, maxval=1.8)
    # Sources on and beside both mirror axes use nodes of both signs.
    points = points.at[0].set(jnp.array([0.0, 0.0])).at[1].set(jnp.array([0.05, -0.02]))
    flux = jax.random.uniform(jax.random.key(1), (9,), minval=0.5, maxval=2.0)
    return points, flux


@pytest.mark.medium(reason="compiles three scene paths with and without jit, 12 s")
def test_translated_host_scenes_keep_distinct_sources_and_flux_tangents(full_delivery):
    """Translation preserves the two source responses and their flux derivatives.

    The dyadic translation, offsets and pitch are exact in binary64, and the
    two sources are one binary32 unit apart at the translation. The signed
    difference of two nearby reads exposes a collapsed coordinate.
    Each scene fold can regroup the two tap contractions, budgeted at 128
    working epsilon units on the sum of the absolute source image peaks.
    """
    full_delivery = dict(
        full_delivery,
        memory=sc.MemoryBudget(100_000_000, 10_000_000, 60_000_000, 400_000_000),
    )
    local, _ = make_prefiltered(full_delivery)
    lattice = local.grid.source
    shift = 2.0**19
    axes = np.array([0.125, 0.15625])
    offsets = np.column_stack((axes, np.full(2, axes[0])))
    positions = offsets + shift
    domain = sc.Domain("box", "closed", [positions[0], positions[-1]], [0, 0])
    shifted_grid = source_grid(
        origin_xy(lattice) + shift, spacing_xy(lattice), lattice.shape, lattice.frame
    )
    reader, _ = make_prefiltered(
        dict(
            full_delivery,
            declared_lattice=shifted_grid,
            offsets_xy_lod=np.asarray(full_delivery["offsets_xy_lod"], np.float64)
            + shift,
            domain=domain,
        )
    )
    reads = [local.read(q, jnp.int32(0)).intensity for q in offsets]
    expected = reads[0] - reads[1]
    scale = sum(jnp.max(jnp.abs(image)) for image in reads)
    budget = float(roundoff(scale, n_ops=128, dtype=expected.dtype))
    assert jnp.max(jnp.abs(expected)) > budget
    x, y = positions[:, 0], positions[:1, 1]
    raster = sc.SceneRenderer.from_raster(reader, 0, x, y)
    points = sc.SceneRenderer(reader, 0, positions)
    raster_grid = source_grid(positions[0], (0.03125, 0.03125), (1, 2), lattice.frame)
    direction = jnp.array([1.0, -1.0])
    for render in (
        lambda f: points.render_points(positions, f).intensity,
        lambda f: raster.render_raster(f[None, :], x, y).intensity,
        lambda f: raster.render(f[None, :], raster_grid)[0],
    ):
        for run in (render, eqx.filter_jit(render)):
            np.testing.assert_allclose(run(direction), expected, rtol=0, atol=budget)
            _, tangent = jax.jvp(run, (jnp.zeros_like(direction),), (direction,))
            np.testing.assert_allclose(tangent, expected, rtol=0, atol=budget)


@pytest.fixture(scope="module", params=["tone_full", "tone_quadrant"])
def case(request):
    return request.getfixturevalue(request.param)


@pytest.mark.medium(reason="compiles the render and a batch read per table, 4 s")
def test_points_equal_summed_reads(case):
    """Fails if the fold drops, double counts or mis-flips a touched node.

    The render regroups the same products the reads sum, so it agrees with the
    flux-weighted sum of reads within 128 epsilon units of scene scale for
    two tap contractions. Use full and reflected tables and cross both axes.
    """
    reader = case["reader"]
    points, flux = scene_points()
    renderer = sc.SceneRenderer(reader, 0, points)
    result = render_points(renderer, points, flux)
    expected = brute(reader, points, flux)
    scale = case["norm"] * float(jnp.sum(flux))
    np.testing.assert_allclose(
        result.intensity,
        expected,
        rtol=0,
        atol=float(roundoff(scale, n_ops=128, dtype=result.intensity.dtype)),
    )
    assert bool(result.accepted) and bool(result.sampling_ok)
    assert int(result.status) == Status.SAMPLING_ADMITTED
    assert int(result.reason_bits) == Reason.BOUND_UNKNOWN
    assert not bool(result.bound_available)
    assert np.isinf(float(result.abs_error_bound))
    assert np.isnan(float(result.empirical_max_abs))
    assert not np.any(np.asarray(result.ledger.evidence))


@pytest.mark.medium(reason="compiles raster, point and batch paths per table, 4 s")
def test_raster_equals_points_and_summed_reads(case):
    """Fails if the separable raster fold swaps axes or loses a pixel.

    The raster is not square and crosses both mirror axes; one pixel is dark.
    No center sits on a half cell, where the stencil choice can depend on how
    each path compiles. At scene scale, 128 rounding operations allow for
    the two tap contractions and the alternative flux summation orders.
    """
    reader = case["reader"]
    x, y = jnp.linspace(-0.9, 1.3, 6), jnp.linspace(-1.1, 0.66, 4)
    scene = jax.random.uniform(jax.random.key(2), (4, 6), minval=0.1)
    scene = scene.at[1, 2].set(0.0)
    grid_x, grid_y = jnp.meshgrid(x, y)
    centers = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    renderer = sc.SceneRenderer.from_raster(reader, 0, x, y)
    raster = render_raster(renderer, scene, x, y)
    points = renderer.render_points(centers, scene.ravel())
    expected = brute(reader, centers, scene.ravel())
    scale = case["norm"] * float(jnp.sum(scene))
    for result in (raster, points):
        assert bool(result.accepted)
        np.testing.assert_allclose(
            result.intensity,
            expected,
            rtol=0,
            atol=float(roundoff(scale, n_ops=128, dtype=result.intensity.dtype)),
        )


@pytest.mark.medium(reason="eleven uncompiled renders on one table, 6 s")
def test_refusals_padding_and_margin(tone_quadrant):
    """Fails if an unreadable source is dropped silently or padding refuses.

    A contributing source outside the domain, off the touched nodes or
    non-finite refuses the whole image; the same entries with zero flux are
    padding. One node of margin lets positions move by a node spacing.
    """
    reader = tone_quadrant["reader"]
    points, flux = scene_points()
    renderer = sc.SceneRenderer(reader, 0, points)
    tight = sc.SceneRenderer(reader, 0, points, margin_nodes=0)
    assert tight.nodes.shape[0] < renderer.nodes.shape[0]
    # A signed node and its mirrors share one stored row.
    signed = np.asarray(renderer.nodes) + np.array(renderer._corner_xy[::-1])
    stored = np.unique(np.abs(signed), axis=0).shape[0]
    assert stored < renderer.nodes.shape[0]
    assert renderer.table.shape == (stored, 30)
    assert renderer.table.dtype == reader.coefficients.dtype
    good = renderer.render_points(points, flux).intensity
    # (3.0, y) is covered by the table but outside the 2.0 box: one reason only.
    outside = renderer.render_points(points.at[3, 0].set(3.0), flux)
    assert not bool(outside.accepted) and bool(jnp.all(jnp.isnan(outside.intensity)))
    assert int(outside.reason_bits) == (Reason.OUTSIDE_DOMAIN | Reason.BOUND_UNKNOWN)
    assert int(outside.status) == Status.REFUSED
    broken = renderer.render_points(points.at[3, 1].set(jnp.nan), flux)
    assert int(broken.reason_bits) & Reason.NONFINITE_QUERY
    for bad in (5.0, jnp.nan):
        padded = renderer.render_points(points.at[3, 0].set(bad), flux.at[3].set(0.0))
        without = renderer.render_points(points, flux.at[3].set(0.0))
        assert bool(padded.accepted)
        np.testing.assert_array_equal(padded.intensity, without.intensity)
    # A shift of one node spacing stays inside the margin; the tight set refuses.
    shift = jnp.array([0.3, -0.28])
    moved = renderer.render_points(points * 0.8 + shift, flux)
    assert bool(moved.accepted)
    np.testing.assert_allclose(
        moved.intensity,
        brute(reader, points * 0.8 + shift, flux),
        rtol=0,
        atol=float(roundoff(jnp.max(good), n_ops=128)),
    )
    far = tight.render_points(points * 0.8 + 3 * shift, flux)
    assert not bool(far.accepted)
    assert int(far.reason_bits) & Reason.INCOMPLETE_STENCIL
    # A raster moved by four node spacings stays inside the domain and the
    # table's coverage but leaves the node set that was gathered for it.
    x, y = jnp.linspace(-0.5, 0.5, 3), jnp.linspace(-0.5, 0.5, 3)
    grid_x, grid_y = jnp.meshgrid(x, y)
    centers = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    patch = sc.SceneRenderer(reader, 0, centers, margin_nodes=0)
    scene = jnp.ones((3, 3))
    assert bool(patch.render_raster(scene, x, y).accepted)
    moved_raster = patch.render_raster(scene, x + 1.2, y)
    assert not bool(moved_raster.accepted)
    assert int(moved_raster.reason_bits) == (
        Reason.INCOMPLETE_STENCIL | Reason.BOUND_UNKNOWN
    )
    dark = patch.render_raster(jnp.zeros((3, 3)), x + 1.2, y)
    assert bool(dark.accepted)
    # Gather a raster crossing the permission boundary; only lit pixels matter.
    mixed_x = jnp.array([0.0, 3.0])
    mixed = sc.SceneRenderer.from_raster(reader, 0, mixed_x, y)
    padded = jnp.ones((3, 2)).at[:, 1].set(0.0)
    good = render_raster(mixed, padded, mixed_x, y)
    assert good.accepted and jnp.isfinite(good.intensity).all()
    assert not render_raster(mixed, padded.at[0, 1].set(1.0), mixed_x, y).accepted
    bounded = eqx.tree_at(
        lambda r: r.reader.policy, renderer, sc.ReadPolicy("bounded", jnp.array(1.0))
    )
    refused = bounded.render_points(points, flux)
    assert not bool(refused.accepted) and bool(refused.sampling_ok)


@pytest.mark.float64_only(reason="Float64 reference accumulation from float32 storage")
@pytest.mark.medium(reason="admits and builds a float32 table, 5 s")
def test_float32_storage_renders_like_summed_reads(tone_full):
    """Fails if a float32 table is contracted in float32.

    A read contracts float32 coefficients in float64. The render must do the
    same: it then equals the summed reads to float64 rounding (1e-12 of the
    scene scale), where a float32 contraction differs by about 1e-6.
    """
    data = tone_full["data"]
    single = {
        **data,
        "intensity_nodes": data["intensity_nodes"].astype(jnp.float32),
        "mirror_checks": None,
    }
    reader, _ = make_prefiltered(single)
    points, flux = scene_points()
    renderer = sc.SceneRenderer(reader, 0, points)
    assert renderer.table.dtype == jnp.float32
    result = render_points(renderer, points, flux)
    assert bool(result.accepted) and result.intensity.dtype == jnp.float64
    scale = tone_full["norm"] * float(jnp.sum(flux))
    np.testing.assert_allclose(
        result.intensity, brute(reader, points, flux), rtol=0, atol=1e-12 * scale
    )
    x, y = jnp.linspace(-0.9, 1.3, 6), jnp.linspace(-1.1, 0.66, 4)
    scene = jax.random.uniform(jax.random.key(2), (4, 6), minval=0.1)
    grid_x, grid_y = jnp.meshgrid(x, y)
    centers = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    raster = sc.SceneRenderer(reader, 0, centers, margin_nodes=0).render_raster(
        scene, x, y
    )
    np.testing.assert_allclose(
        raster.intensity,
        brute(reader, centers, scene.ravel()),
        rtol=0,
        atol=1e-12 * tone_full["norm"] * float(jnp.sum(scene)),
    )


def test_construction_refusals(tone_full):
    """Fails if a renderer is built on unreadable positions or past a budget."""
    reader, data = tone_full["reader"], tone_full["data"]
    points, _ = scene_points()
    with pytest.raises(ValueError, match="finite, permitted and covered"):
        sc.SceneRenderer(reader, 0, points.at[0, 0].set(9.0))
    with pytest.raises(ValueError, match="product_index"):
        sc.SceneRenderer(reader, 1, points)
    with pytest.raises(ValueError, match="margin_nodes"):
        sc.SceneRenderer(reader, 0, points, margin_nodes=-1)
    raw = Interpolator(
        sc.admit_grid(**data).require(),
        data["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(),
        data["memory"],
    )
    with pytest.raises(TypeError, match="PrefilteredReader"):
        sc.SceneRenderer(raw, 0, points)
    resident = reader.memory_estimate(1)["resident"]
    lean = eqx.tree_at(
        lambda r: r.memory,
        reader,
        sc.MemoryBudget(resident + 100, 10_000_000, 60_000_000, 400_000_000),
    )
    with pytest.raises(sc.MemoryLimit, match="scene table"):
        sc.SceneRenderer(lean, 0, points)


@pytest.mark.float64_only(reason="Finite differences at a 1e-6 step")
@pytest.mark.medium(reason="reverse-mode compile of the render, 4 s")
def test_gradients_through_the_render(tone_quadrant):
    """Fails if flux or position gradients are wrong, or NaN on a refused render.

    Positions sit away from stencil switch points, where the read is smooth;
    reverse-mode gradients must match central differences. A refused render
    returns NaN values but finite (zero) gradients, because refused lanes are
    evaluated at a safe substitute position.
    """
    reader = tone_quadrant["reader"]
    points = jnp.array([[0.07, -0.05], [0.71, 0.52], [-1.13, 0.9]])
    flux = jnp.array([1.0, 0.6, 1.7])
    renderer = sc.SceneRenderer(reader, 0, points)
    weights = jax.random.uniform(jax.random.key(3), (6, 5))

    @eqx.filter_jit
    def score(positions, amounts):
        return jnp.sum(weights * renderer.render_points(positions, amounts).intensity)

    check_grads(score, (points, flux), order=1, modes=["rev"], eps=1e-6, rtol=1e-5)
    gradient = jax.grad(score, argnums=(0, 1))(points.at[0, 0].set(7.0), flux)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in gradient)


@pytest.mark.float64_only(reason="Float64 reference accumulation from float32 storage")
@pytest.mark.medium(reason="admits a float32 quadrant and renders it twice, 5 s")
def test_float32_table_accumulation(tone_quadrant, monkeypatch):
    """Fails if a float32 table is contracted in the wrong precision or block order.

    With binary64 accumulation the render equals the summed reads, which widen
    each coefficient exactly, to binary64 rounding; seven-column blocks and a
    two-column tail cover the 30-pixel detector. Accumulating in float32 departs
    from them by float32 rounding of the amplitudes and the sums only.
    """
    data = tone_quadrant["data"]
    narrow = {**data, "intensity_nodes": data["intensity_nodes"].astype(jnp.float32)}
    reader, _ = make_prefiltered(narrow)
    monkeypatch.setattr(scene_module, "CONTRACT_COLUMNS", 7)
    points, flux = scene_points()
    expected = brute(reader, points, flux)
    scale = tone_quadrant["norm"] * float(jnp.sum(flux))
    wide = sc.SceneRenderer(reader, 0, points)
    assert wide.table.dtype == jnp.float32
    result = wide.render_points(points, flux)
    assert bool(result.accepted)
    np.testing.assert_allclose(result.intensity, expected, rtol=0, atol=1e-12 * scale)
    storage = sc.SceneRenderer(reader, 0, points, accumulate="storage")
    result = storage.render_points(points, flux)
    assert bool(result.accepted) and result.intensity.dtype == jnp.float64
    error = float(jnp.max(jnp.abs(result.intensity - expected))) / scale
    assert error <= 1e-6
    with pytest.raises(ValueError, match="accumulate"):
        sc.SceneRenderer(reader, 0, points, accumulate="float32")


@pytest.mark.float32_only(
    reason="Float64 absence must raise the shared precision error"
)
def test_explicit_float64_accumulation_requires_x64(tone_full):
    """A float32 scene can render, while an explicit float64 request refuses."""
    points, flux = scene_points()
    reader = tone_full["reader"]
    result = sc.SceneRenderer(reader, 0, points).render_points(points, flux)
    assert result.accepted and result.intensity.dtype == jnp.float32
    with pytest.raises(PrecisionError, match=r"float64 accumulation.*Enable x64"):
        sc.SceneRenderer(reader, 0, points, accumulate="float64")
