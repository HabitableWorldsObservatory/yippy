"""The blend scene operator renders an extended source from the stored PSF nodes."""

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.memory import MemoryLimit
from hwoutils.numerics import roundoff
from optixstuff.grids import Image, Measure
from synthetic_yip import write_synthetic_yip

import yippy.sinc
from yippy import (
    BlendSceneOperator,
    MaskedSceneOperator,
    SceneCoronagraph,
    fetch_yip,
    open_yip,
)
from yippy.scene import AbstractSceneOperator


def _dense_image(backend, grid, flux):
    """Contract the flux with one blend PSF per lit source inside the node hull.

    A source outside the hull of the tabulated nodes has no data and
    contributes zero (an unsymmetric table that stores one quadrant has no
    nodes in the other three).
    """
    x, y = (np.asarray(c, np.float64).ravel() for c in grid.coordinates())
    inside = np.asarray(backend.in_source_domain(x, y))
    lit = np.flatnonzero((flux.ravel() != 0) & inside)
    image = np.zeros(tuple(backend.flat_psfs.shape[-2:]))
    for i in range(0, lit.size, 16):
        block = lit[i : i + 16]
        psfs = backend._create_psfs(jnp.asarray(x[block]), jnp.asarray(y[block]))
        image += np.einsum("s,syx->yx", flux.ravel()[block], np.asarray(psfs))
    return image


def _agreement(model, flux):
    """Peak-normalized maximum difference and relative flux difference."""
    grid = model.scene_grid
    image, covered = model.operators[0].render(jnp.asarray(flux), grid)
    reference = _dense_image(model._offaxis_backend, grid, flux)
    x, y = (np.asarray(c, np.float64) for c in grid.coordinates())
    inside = np.asarray(model._offaxis_backend.in_source_domain(x, y))
    np.testing.assert_array_equal(np.asarray(covered), inside)
    peak = reference.max()
    return (
        float(np.abs(np.asarray(image) - reference).max() / peak),
        float(abs(float(image.sum()) - reference.sum()) / reference.sum()),
    )


@pytest.mark.parametrize(
    ("kind", "symmetric"), [("1d", True), ("2dq", True), ("2dq", False)]
)
def test_render_matches_the_dense_contraction(tmp_path, kind, symmetric):
    """Fails if the node regrouping disagrees with one blend PSF per source.

    The synthetic packages put nodes between source centers, so every read blends
    shifted neighbors; the unsymmetric case has no reflection classes and sources
    in all four quadrants. Sources stay clear of the window edge: there the
    sub-pixel shift of a truncated PSF rings below zero inside its window, which
    each dense read clips and the linear render keeps (1e-4 of the peak with the
    whole grid lit). Away from it they differ by the smaller negative ringing
    the reads clip and, for the radial table, by the sub-pixel remainder a
    stored PSF shares across sources: measured 9e-6 of the peak and 4e-6 of the
    flux, in float32 and float64 alike.
    """
    package = write_synthetic_yip(
        tmp_path / kind,
        kind=kind,
        npix=32,
        max_offset_lod=3.0,
        offset_step_lod=0.5,
        psf_sigma_lod=0.4,
        occulter_floor=1.0,
    )
    model = open_yip(package.path).prepare(x_symmetric=symmetric, y_symmetric=symmetric)
    model = model.with_scene_operator()
    flux = np.zeros(model.grid.shape)
    flux[8:24, 8:24] = np.random.default_rng(0).uniform(0.5, 1.5, (16, 16))
    peak_error, flux_error = _agreement(model, flux)
    assert peak_error < 3e-5
    assert flux_error < 1e-5


@pytest.mark.slow(
    reason="1000 dense reference PSFs and one 256-pixel build, 26-42 s run-bound"
)
def test_shipped_package_blocks_size_and_estimate(coro):
    """Fails if the node form drifts from the per-source blend on real 2D PSFs.

    Three blocks of sources (core, mid field, far field) are lit at random flux;
    agreement must hold to 1e-5 of the image peak and of the total flux, the
    float32 rounding of the two paths (measured 1.1e-6 and 1.9e-6). The built
    arrays must match the reported resident bytes and stay far below the
    per-source cube (8.6 GB).
    """
    model = coro.with_scene_operator()
    rng = np.random.default_rng(0)
    for lo, hi in ((0, 12), (8, 20), (80, 92)):
        side = np.arange(lo, hi)
        rows, cols = (a.ravel() for a in np.meshgrid(side, side, indexing="ij"))
        flux = np.zeros(model.scene_grid.shape)
        cy, cx = (n // 2 for n in flux.shape)
        flux[cy + rows, cx + cols] = rng.uniform(0.5, 1.5, rows.size)
        peak_error, flux_error = _agreement(model, flux)
        assert peak_error < 1e-5
        assert flux_error < 1e-5
    estimate = coro.scene_operator_bytes()
    assert model.operators[0].nbytes == estimate["resident"] < 400e6
    assert estimate["total"] == estimate["resident"] + estimate["scratch"]


@pytest.mark.float64_only(reason="Central finite differences at a 1e-6 step")
def test_render_is_linear_mirrored_and_differentiable(tmp_path):
    """Fails if the render stops being exactly linear in flux or loses reflection.

    Runs in float64: linearity and the reflected image hold to rounding, and the
    gradient must match central differences.
    """
    package = write_synthetic_yip(
        tmp_path / "linear",
        kind="2dq",
        npix=32,
        max_offset_lod=3.0,
        offset_step_lod=0.5,
        psf_sigma_lod=0.4,
        occulter_floor=1.0,
    )
    model = open_yip(package.path).prepare().with_scene_operator()
    render = lambda flux: model.operators[0].render(flux, model.grid)[0]  # noqa: E731
    rng = np.random.default_rng(0)
    f, g = (jnp.asarray(rng.uniform(0.5, 1.5, model.grid.shape)) for _ in range(2))
    np.testing.assert_allclose(
        render(2.0 * f - 3.0 * g),
        2.0 * render(f) - 3.0 * render(g),
        rtol=0,
        atol=1e-12,
    )
    # A lone source and its mirror image across x = 0 give mirrored images.
    lone = jnp.zeros(model.grid.shape).at[5, 12].set(1.0)
    mirror = jnp.zeros(model.grid.shape).at[5, 19].set(1.0)
    np.testing.assert_allclose(
        render(mirror), render(lone)[:, ::-1], rtol=0, atol=1e-12
    )
    loss = lambda flux: jnp.sum(render(flux) ** 2)  # noqa: E731
    direction = jnp.asarray(rng.normal(size=model.grid.shape))
    step = 1e-6
    numeric = (loss(f + step * direction) - loss(f - step * direction)) / (2 * step)
    analytic = jnp.vdot(jax.grad(loss)(f), direction)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7)


def test_estimate_allocates_nothing_and_matches_the_build(syn_coro_2d):
    """Fails if the size estimate reads PSF pixels or drifts from the built bytes.

    The stand-in backend holds the PSF cube as a shape-only placeholder, so any
    read of pixel data raises; the estimate must still match the built operator.
    """
    backend = copy.copy(syn_coro_2d._offaxis_backend)
    backend.flat_psfs = jax.ShapeDtypeStruct(
        backend.flat_psfs.shape, backend.flat_psfs.dtype
    )
    estimate = BlendSceneOperator.size_bytes(backend, syn_coro_2d.grid)
    built = syn_coro_2d.with_scene_operator().operators[0].nbytes
    assert abs(estimate["resident"] - built) <= 0.1 * built
    assert syn_coro_2d.scene_operator_bytes() == estimate


def test_budget_refuses_before_building_and_none_disables_it(syn_coro_2d):
    """Fails if an over-budget build starts, or the refusal omits what to change."""
    with pytest.raises(MemoryLimit) as refused:
        syn_coro_2d.with_scene_operator(max_bytes=1)
    total = syn_coro_2d.scene_operator_bytes()["total"]
    assert f"({total} bytes)" in str(refused.value)
    assert "max_bytes" in str(refused.value)
    assert syn_coro_2d.with_scene_operator(max_bytes=None).operators[0].nbytes > 0


def test_refusal_is_the_shared_memory_limit():
    """Fails if the sinc refusal stops being a hwoutils MemoryLimit or loses routing."""
    assert issubclass(yippy.sinc.MemoryLimit, MemoryLimit)
    assert issubclass(yippy.sinc.MemoryLimit, yippy.sinc.AdmissionRefused)
    assert issubclass(MemoryLimit, ValueError)


class _Gain(AbstractSceneOperator):
    """A stand-in operator: the flux scaled, valid or not as configured."""

    gain: float
    ok: bool = eqx.field(static=True, default=True)

    def render(self, flux, grid):
        return self.gain * flux, jnp.full(flux.shape, self.ok)


def test_scene_coronagraph_sums_its_operators(syn_coro_2d):
    """Fails if the render is not the sum of the operators, or drops a refusal.

    A cell is refused only when no operator covers it: a covered cell is valid
    whatever another operator reports for it.
    """
    model = syn_coro_2d.with_scene_operator()
    blend = model.operators[0]
    assert isinstance(blend, AbstractSceneOperator)
    flux = np.zeros(model.grid.shape)
    flux[10:14, 10:14] = 1.0
    scene = Image(jnp.asarray(flux), model.grid, (), Measure.integral(), "flux")
    single = model.render_scene(scene, None, grid=model.grid)
    gain = _Gain(0.5)
    both = SceneCoronagraph(model, (blend, gain)).render_scene(
        scene, None, grid=model.grid
    )
    assert single.valid and both.valid
    alone = SceneCoronagraph(model, (gain,)).render_scene(scene, None, grid=model.grid)
    expected = single.image.data + alone.image.data
    assert float(jnp.abs(alone.image.data).max()) > 0
    np.testing.assert_allclose(both.image.data, expected, rtol=1e-6, atol=1e-9)
    covered_elsewhere = SceneCoronagraph(
        model, (blend, _Gain(1.0, ok=False))
    ).render_scene(scene, None, grid=model.grid)
    assert covered_elsewhere.valid
    refused = SceneCoronagraph(model, (_Gain(1.0, ok=False),)).render_scene(
        scene, None, grid=model.grid
    )
    assert not refused.valid
    assert bool(jnp.all(jnp.isfinite(refused.image.data)))


def _disjoint_masks(grid):
    """A checkerboard of 4-cell tiles and its complement on a source grid."""
    row, col = np.indices(grid.shape)
    inside = ((row // 4 + col // 4) % 2) == 0
    return inside, ~inside


def test_disjoint_masked_operators_sum_to_the_unmasked_render(tmp_path):
    """Fails if masked operators do not compose into the unmasked operator.

    Two operators over complementary masks must add to the single operator over
    every source, to rounding, with the scene valid: each reads only its own
    cells and the union covers every lit cell (before, an operator that was
    handed a source outside its mask turned the whole image to NaN).

    The reference image peak scales a 128-operation budget for shifting and
    the separate operator contractions and sum.
    """
    package = write_synthetic_yip(
        tmp_path / "masked",
        kind="2dq",
        npix=32,
        max_offset_lod=3.0,
        offset_step_lod=0.5,
        psf_sigma_lod=0.4,
        occulter_floor=1.0,
    )
    model = open_yip(package.path).prepare()
    whole = model.with_scene_operator()
    grid = whole.scene_grid
    keep, rest = _disjoint_masks(grid)
    halves = model.with_scene_operator(mask=keep).with_operators(
        *model.with_scene_operator(mask=rest).operators
    )
    assert len(halves.operators) == 2
    flux = np.zeros(grid.shape)
    flux[8:24, 8:24] = np.random.default_rng(3).uniform(0.5, 1.5, (16, 16))
    scene = Image(jnp.asarray(flux), grid, (), Measure.integral(), "flux")
    reference = whole.render_scene(scene, None, grid=model.grid)
    composed = halves.render_scene(scene, None, grid=model.grid)
    assert reference.valid and composed.valid
    assert float(jnp.abs(reference.image.data).max()) > 0
    np.testing.assert_allclose(
        composed.image.data,
        reference.image.data,
        rtol=0,
        atol=float(
            roundoff(
                np.max(np.abs(reference.image.data)),
                n_ops=128,
                dtype=reference.image.data.dtype,
            )
        ),
    )
    # One half alone leaves lit cells uncovered: finite image, not valid.
    alone = model.with_scene_operator(mask=keep).render_scene(
        scene, None, grid=model.grid
    )
    assert not alone.valid
    assert bool(jnp.all(jnp.isfinite(alone.image.data)))
    assert float(jnp.abs(alone.image.data).sum()) < float(
        jnp.abs(reference.image.data).sum()
    )


def test_masked_wrapper_matches_the_native_mask(tmp_path):
    """Fails if MaskedSceneOperator reads or reports a cell outside its mask.

    At the reference peak, 128 epsilon units cover filtering and contraction.
    """
    package = write_synthetic_yip(
        tmp_path / "wrapper",
        kind="2dq",
        npix=32,
        max_offset_lod=3.0,
        offset_step_lod=0.5,
        psf_sigma_lod=0.4,
        occulter_floor=1.0,
    )
    model = open_yip(package.path).prepare().with_scene_operator()
    keep, _ = _disjoint_masks(model.scene_grid)
    native = BlendSceneOperator.from_backend(
        model._offaxis_backend, model.scene_grid, mask=keep
    )
    wrapped = MaskedSceneOperator(model.operators[0], keep)
    flux = jnp.asarray(
        np.random.default_rng(4).uniform(0.5, 1.5, model.scene_grid.shape)
    )
    image_n, covered_n = native.render(flux, model.scene_grid)
    image_w, covered_w = wrapped.render(flux, model.scene_grid)
    np.testing.assert_allclose(
        image_w,
        image_n,
        rtol=0,
        atol=float(roundoff(np.max(np.abs(image_n)), n_ops=128, dtype=image_n.dtype)),
    )
    assert bool(jnp.all(covered_w == covered_n & keep))


def test_nonfinite_flux_is_zero_and_invalid(syn_coro_2d):
    """Fails if a nonfinite source cell reaches the image or passes as valid."""
    model = syn_coro_2d.with_scene_operator()
    grid = model.scene_grid
    flux = np.zeros(grid.shape)
    flux[10:14, 10:14] = 1.0
    clean = Image(jnp.asarray(flux), grid, (), Measure.integral(), "flux")
    flux[12, 12] = np.nan
    broken = Image(jnp.asarray(flux), grid, (), Measure.integral(), "flux")
    good = model.render_scene(clean, None, grid=model.grid)
    bad = model.render_scene(broken, None, grid=model.grid)
    assert good.valid and not bad.valid
    assert bool(jnp.all(jnp.isfinite(bad.image.data)))


@pytest.fixture(scope="module")
def wide_coro(tmp_path_factory):
    """A package whose nodes reach 3 times the stamp half-width.

    Its Gaussian PSFs (sigma 2 lambda/D on a 16 lambda/D stamp) are nonzero at
    the stamp edge, so a translated node carries its truncation into the image.
    """
    package = write_synthetic_yip(
        tmp_path_factory.mktemp("wide") / "wide",
        kind="2dq",
        npix=64,
        pixscale=0.25,
        max_offset_lod=24.0,
        offset_step_lod=1.0,
        psf_sigma_lod=2.0,
        occulter_floor=1.0,
    )
    return open_yip(package.path).prepare().with_scene_operator()


@pytest.fixture(scope="module")
def eac1_x64():
    """The shipped 2D package in float64 with its scene operator."""
    return open_yip(fetch_yip("eac1_aavc_2d")).prepare().with_scene_operator()


def test_scene_grid_reaches_the_node_hull(wide_coro):
    """Fails if sources between the stamp edge and the outermost node are dropped.

    The nodes reach 3 times the stamp half-width. The scene grid must hold the
    cells out to the hull, and the operator covers exactly the cells inside the
    hull.
    """
    model = wide_coro
    stamp = model.grid
    grid = model.scene_grid
    assert grid.shape[0] > stamp.shape[0] and grid.shape[1] > stamp.shape[1]
    x, y = (np.asarray(c) for c in grid.coordinates())
    assert np.abs(x).max() >= 24.0 - 0.25 and np.abs(y).max() >= 24.0 - 0.25
    # The stamp sits at the middle of the scene grid, cell for cell.
    ny, nx = stamp.shape
    top, left = (grid.shape[0] - ny) // 2, (grid.shape[1] - nx) // 2
    np.testing.assert_allclose(
        x[top, left : left + nx], np.asarray(stamp.coordinates()[0])[0]
    )
    inside = np.asarray(model._offaxis_backend.in_source_domain(x, y))
    np.testing.assert_array_equal(np.asarray(model.operators[0].covered), inside)


_EAC1 = pytest.mark.medium(reason="float64 load and scene operator of eac1_aavc_2d")


@pytest.mark.parametrize(
    ("model_name", "cell_lod"),
    [
        ("wide_coro", (2.1, 1.4)),
        ("wide_coro", (-7.9, 3.1)),
        ("wide_coro", (23.9, -0.9)),
        pytest.param("eac1_x64", (10.125, 5.125), marks=_EAC1),
        pytest.param("eac1_x64", (-31.875, 0.125), marks=_EAC1),
        pytest.param("eac1_x64", (61.875, 0.125), marks=_EAC1),
    ],
    ids=[
        "synthetic-interior",
        "synthetic-stamp-edge",
        "synthetic-hull-edge",
        "eac1-interior",
        "eac1-stamp-edge",
        "eac1-hull-edge",
    ],
)
def test_a_point_read_is_the_clipped_render_of_its_cell(model_name, cell_lod, request):
    """Fails if a point read leaves light where its nodes' stored windows end.

    One lit cell renders the same blend as the point read of that source: the
    same nodes, weights and shifts, each node zero beyond its stored window
    translated by the whole-pixel part of its shift. The read is that render
    clipped at zero, to 1e-12 of its peak (basis: exact-algebra, float64; the
    two differ only in the order of the Fourier transforms). Cells sit inside
    the stamp, at its edge (the mirrored quadrant) and at the edge of the node
    hull. Before the read cut each node to its window it differed by up to 0.2
    of the peak on the synthetic table and 3e-2 on eac1_aavc_2d at the hull
    edge, and by 9e-4 and 8e-6 inside the stamp.

    At peak scale, 128 epsilon units cover FFT shifting and dyadic blending.
    """
    model = request.getfixturevalue(model_name)
    grid = model.scene_grid
    x, y = (np.asarray(c, np.float64) for c in grid.coordinates())
    cell = np.unravel_index(
        np.argmin((x - cell_lod[0]) ** 2 + (y - cell_lod[1]) ** 2), x.shape
    )
    flux = jnp.zeros(grid.shape).at[cell].set(1.0)
    image, covered = (np.asarray(a) for a in model.operators[0].render(flux, grid))
    read = np.asarray(
        model._offaxis_backend._create_psfs(
            jnp.asarray([x[cell]]), jnp.asarray([y[cell]])
        )[0]
    )
    assert covered[cell]
    peak = read.max()
    assert peak > 0
    np.testing.assert_allclose(
        read,
        np.maximum(image, 0.0),
        rtol=0,
        atol=float(roundoff(peak, n_ops=128, dtype=read.dtype)),
    )


@pytest.mark.parametrize(
    ("kind", "source_lod", "outside"),
    [
        # Nodes on a 4 lambda/D grid; the query blends (4, 0) and (4, 4), both
        # shifted +5.5 pixels in x, so the first columns lie beyond both.
        ("2dq", (5.375, 0.0), np.s_[:, :5]),
        ("2dq", (0.0, 5.375), np.s_[:5, :]),
        # Below the first radial node (4 lambda/D) the read is that node alone,
        # shifted -5.5 pixels: the last columns lie beyond it.
        ("1d", (2.625, 0.0), np.s_[:, -5:]),
    ],
    ids=["polar-x", "polar-y", "radial"],
)
def test_a_read_is_zero_beyond_the_stored_windows(tmp_path, kind, source_lod, outside):
    """Fails if a read puts light where no stored PSF of its blend reaches.

    Every node PSF is a Gaussian of sigma 4 lambda/D on a 8 lambda/D stamp, so
    it is far from zero at the stamp edge. A read whose every neighbor is
    translated by 5.5 pixels the same way has no data in the 5 rows or columns
    shifted in from beyond the stored arrays: they must be 0 to 1e-12 of the
    read's peak (basis: exact-algebra, the window cut leaves exact zeros).
    Summing the shifted spectra before one inverse transform left the Fourier
    tail of each truncated PSF there, 7e-3 to 7e-2 of the peak.

    At peak scale, 128 epsilon units cover the FFT and blend before masking.
    """
    package = write_synthetic_yip(
        tmp_path / kind,
        kind=kind,
        npix=32,
        psf_sigma_lod=4.0,
        occulter_floor=1.0,
        min_offset_lod=4.0 if kind == "1d" else 0.0,
        max_offset_lod=8.0,
        offset_step_lod=4.0,
    )
    backend = open_yip(package.path).prepare()._offaxis_backend
    read = np.asarray(
        backend._create_psfs(
            jnp.asarray([source_lod[0]]), jnp.asarray([source_lod[1]])
        )[0]
    )
    peak = read.max()
    assert peak > 0
    np.testing.assert_allclose(
        read[outside],
        0.0,
        rtol=0,
        atol=float(roundoff(peak, n_ops=128, dtype=read.dtype)),
    )
