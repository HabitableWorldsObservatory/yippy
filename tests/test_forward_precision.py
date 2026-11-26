"""Verify that prepared reads follow their array precision under x64."""

import re
from dataclasses import replace
from hashlib import sha256

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff.grids import Image, Measure
from optixstuff.performance import azimuthal_mean_profile, profile_radii
from sinc.helpers import delivery, make_prefiltered, make_reader, tone_delivery
from sinc_fixture import save_manifest, write_delivery
from synthetic_yip import synthetic_coronagraph, write_synthetic_yip

from yippy import MaskedSceneOperator, SceneCoronagraph, open_yip
from yippy.performance import stellar_intensity_curves
from yippy.sinc import Domain, SceneRenderer
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import read_stencil
from yippy.sinc.reader import read_batch

pytestmark = pytest.mark.float64_only(reason="Trace float32 models with x64 available")


@pytest.fixture(scope="module")
def delivery_models(tmp_path_factory):
    """Use a full rectangle so auto traces both the sinc and blend routes."""
    path = tmp_path_factory.mktemp("precision_delivery")
    manifest = write_delivery(path, full=True)
    manifest["checks"] = []
    save_manifest(path, manifest)
    package = open_yip(path)
    return {mode: package.prepare(interpolation=mode) for mode in ("blend", "auto")}


@pytest.fixture(scope="module")
def fits_models(tmp_path_factory):
    """Use compact stamps whose Gaussian wings survive canonical narrowing."""
    path = tmp_path_factory.mktemp("precision_fits")
    radial = write_synthetic_yip(
        path / "radial", npix=12, psf_sigma_lod=1.0, max_offset_lod=2.0
    )
    zeros = write_synthetic_yip(
        path / "zeros", npix=12, psf_sigma_lod=1.0, max_offset_lod=2.0
    )
    with fits.open(zeros.path / "stellar_intens.fits", mode="update") as hdus:
        hdus[0].data[0, 2, 2] = 0
    return {
        "1d": open_yip(radial.path).prepare(),
        "zeros": open_yip(zeros.path).prepare(),
        "2d": synthetic_coronagraph(path / "quadrant", n=12),
    }


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_fits_source_boundary_with_float64_coordinates(fits_models, compiled):
    """The radius-2 disk admits through 2.000002, independent of the table dtype."""
    full = fits_models["1d"]
    x = np.array([2.0, 2.000001975, 2.000002025, 2.01])
    y = np.zeros_like(x)
    assert np.float32(x[1]) == np.float32(x[2])

    def read(model, x, y):
        return model.source_psfs(None, x, y)

    run = eqx.filter_jit(read) if compiled else read
    for model in (full, canonical(full, precision="float32")):
        result = run(model, x, y)
        np.testing.assert_array_equal(result.valid, [True, True, False, False])
        np.testing.assert_array_equal(result.image.data[2:], 0)
        assert np.all(np.max(result.image.data[:2], axis=(1, 2)) > 0)
        assert result.image.data.dtype == np.float64


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_delivery_blend_angular_wrap_with_float32_offsets(tmp_path, compiled):
    """At the negative x-axis, a float64 read uses the full angular period.

    The reference pixel is from a float64 blend of the same synthetic FITS
    table with a full-width period. Allow 128 float64 epsilons at its peak
    for four blended windows with length-16 FFT shifts on each axis.
    """
    manifest = write_delivery(tmp_path, full=True)
    offsets = tmp_path / "offsets.fits"
    with fits.open(offsets, mode="update") as hdus:
        hdus[0].data = hdus[0].data.astype(np.float32)
    manifest["planes"][0]["offsets_sha256"] = sha256(offsets.read_bytes()).hexdigest()
    manifest["checks"] = []
    save_manifest(tmp_path, manifest)
    model = open_yip(tmp_path).prepare(interpolation="blend")
    assert model._offaxis_backend.blend_offsets.dtype == np.float32
    query = np.array([[-0.443123456789, -1.678123456789e-9]])

    def read(query):
        return model.source_psfs(None, *query.T)

    result = (jax.jit(read) if compiled else read)(query)
    np.testing.assert_array_equal(result.valid, [True])
    assert result.image.data.dtype == np.float64
    peak = 0.02014727633478523
    np.testing.assert_allclose(
        result.image.data[0, 1, 0],
        peak,
        rtol=0,
        atol=float(ulps(128, np.float64(peak))),
    )


@pytest.mark.parametrize(
    ("kind", "read", "operations"),
    [
        pytest.param(
            "2d", lambda m, q: m.stellar_map(None, q[0]), 64, id="stellar-native"
        ),
        pytest.param(
            "zeros", lambda m, q: m.stellar_map(None, q[0]), 64, id="stellar-zeros"
        ),
        pytest.param(
            "2d",
            lambda m, q: m.stellar_map(None, q[0], grid=m.stellar_grid),
            128,
            id="stellar-footprint",
        ),
        pytest.param(
            "2d", lambda m, q: m.on_axis_psf(q[0], m.stellar_grid), 128, id="on-axis"
        ),
        pytest.param(
            "1d",
            lambda m, q: m.source_psfs(None, q[:1], q[1:]),
            128,
            id="source-radial",
        ),
        pytest.param(
            "2d", lambda m, q: m.source_psfs(None, q[:1], q[1:]), 128, id="source-2d"
        ),
        pytest.param(
            "2d",
            lambda m, q: m.source_psfs(None, q[:1], q[1:], grid=m.grid),
            256,
            id="source-detector",
        ),
        pytest.param(
            "2d", lambda m, q: m.uniform_sky_response(q[0]), 16, id="sky-profile"
        ),
        pytest.param(
            "2d",
            lambda m, q: m.off_axis_psf(None, q[0], m.grid).data,
            256,
            id="off-axis",
        ),
        pytest.param(
            "2d",
            lambda m, q: m.uniform_sky_map(None, grid=m.sky_trans.grid).data,
            32,
            id="sky-native",
        ),
        pytest.param(
            "2d",
            lambda m, q: m.curves().evaluate("throughput", q[0]),
            32,
            id="throughput",
        ),
    ],
)
def test_coronagraph_read_closure(fits_models, kind, read, operations):
    """Use a peak-scaled roundoff budget for each table path.

    A profile reads a cubic plus a logarithm or exponential (16-32
    operations). Allow 64 for the log stellar spline, including knot
    subtraction and the relative conditioning of exp at log intensity -18.
    Budget 128 for the two separable footprint contractions or the blend
    (four shifted windows with length-48 FFTs). Chaining a detector read
    and a blend allows 256. The native sky map returns the stored samples.
    """
    model = fits_models[kind]
    query = jnp.array([0.0625, 0.09375])
    args = (model, query)
    assert_precision_closed(read, *canonical(args, precision="float32"))

    def values(model, query):
        result = read(model, query)
        if hasattr(result, "image"):
            return result.image.data, result.valid
        return result.data if hasattr(result, "grid") else result

    precision_pair(values, args, bound=float(ulps(operations, np.float32(1))))


def test_sky_profile_closure_across_frames(delivery_models):
    """The reference separation converts to sky lambda/D before the cubic read.

    Allow 16 peak-scaled epsilons for the ratio, coordinate and cubic.
    """
    model = delivery_models["blend"]
    profile = model.uniform_sky_profile
    assert model.frame != profile.frame
    query = jnp.array([0.0625, 0.09375])

    def read(model, query):
        return model.uniform_sky_response(query)

    args = (model, query)
    assert_precision_closed(read, *canonical(args, precision="float32"))
    precision_pair(read, args, bound=float(ulps(16, np.float32(1))))
    # The synthetic versioned product is at 500 nm and the sky at 400 nm,
    # with the same pupil diameter, so its native separation is 5/4 larger.
    expected = profile.evaluate(query * 1.25)
    np.testing.assert_allclose(
        read(*args), expected, rtol=0, atol=float(ulps(16, jnp.max(jnp.abs(expected))))
    )


class _DependencyPrecisionError(AssertionError):
    """Identify a closure failure that matches the known upstream operation."""


def _assert_dependency_closed(read, *args, error=AssertionError, match):
    try:
        assert_precision_closed(read, *args)
    except error as exc:
        if re.fullmatch(match, str(exc)):
            raise _DependencyPrecisionError(str(exc)) from exc
        raise


_GRID_WIDENING = (
    r"[^\n]*/optixstuff/grids\.py:\d+:\d+ \(PixelGrid\.coordinates\): "
    r"precision widened to float64"
)


@pytest.mark.parametrize(
    ("read", "match", "operations"),
    [
        pytest.param(
            lambda m, q: (
                m.uniform_sky_map(
                    None,
                    grid=replace(
                        m.sky_trans.grid, center_x=m.sky_trans.grid.center_x + 0.25
                    ),
                ).data
            ),
            _GRID_WIDENING,
            64,
            id="sky-shifted",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_DependencyPrecisionError,
                reason="optixstuff PixelGrid.coordinates widens shifted sky to float64",
            ),
        ),
        pytest.param(
            lambda m, q: m.curves().evaluate("leakage_density_lod2", q[0], q[1]),
            r"[^\n]*/optixstuff/performance\.py:\d+:\d+ "
            r"\(PerformanceCurves\._evaluate_by_diameter\): "
            r"precision widened to float64",
            128,
            id="leakage",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_DependencyPrecisionError,
                reason="optixstuff diameter-dependent leakage widens to float64",
            ),
        ),
    ],
)
def test_coronagraph_read_closure_waiting_on_dependencies(
    fits_models, read, match, operations
):
    """Check dependency promotions and compare the callable reads to float64.

    Allow 64 peak-scaled epsilons for shifted sky coordinates and bilinear
    interpolation. Allow 128 for nested leakage polynomials in log space.
    """
    args = (fits_models["2d"], jnp.array([0.0625, 0.09375]))
    precision_pair(read, args, bound=float(ulps(operations, np.float32(1))))
    _assert_dependency_closed(read, *canonical(args, precision="float32"), match=match)


@pytest.mark.xfail(
    strict=True,
    raises=_DependencyPrecisionError,
    reason="interpax PCHIP slope branches disagree on float32 and float64 output",
)
def test_stellar_ring_keys_closure(fits_models):
    """Pin both mismatched interpax slopes at the public Keys entry point."""

    def read(model, query):
        return stellar_intensity_curves(model, query[0]).evaluate(query[1])

    args = (fits_models["2d"], jnp.array([0.0625, 0.09375]))
    _assert_dependency_closed(
        read,
        *canonical(args, precision="float32"),
        error=TypeError,
        match=(
            r"cond branches must have equal output types but they differ\.\n\n"
            r"true_fun is slope_zero at [^\n]*/interpax/_fd_derivs\.py:\d+\n"
            r"false_fun is slope_nonzero at [^\n]*/interpax/_fd_derivs\.py:\d+\n\n"
            r"  \* the output of true_fun at path \[0\] has type float32\[1,1\] "
            r"but the corresponding output of false_fun has type float64\[1,1\], "
            r"so the dtypes do not match;\n"
            r"  \* the output of true_fun at path \[1\] has type float32\[1,1\] "
            r"but the corresponding output of false_fun has type float64\[1,1\], "
            r"so the dtypes do not match\.\n\n"
            r"Revise true_fun and/or false_fun so that all output types match\."
        ),
    )


@pytest.mark.xfail(
    strict=True,
    raises=_DependencyPrecisionError,
    reason="optixstuff azimuthal_mean_profile widens quintic samples to float64",
)
def test_stellar_ring_quintic_closure(fits_models):
    """Compare the full read and pin the closure failure to the isolated sampler.

    Allow 256 epsilons at peak for the stellar spline, azimuthal geometry,
    degree-five prefilter and stencil, arc sums and final radial interpolation.
    """

    def read(model, query):
        return stellar_intensity_curves(
            model, query[0], interpolation="quintic"
        ).evaluate(query[1])

    args = (fits_models["2d"], jnp.array([0.0625, 0.09375]))
    precision_pair(read, args, bound=float(ulps(256, np.float32(1))))
    model, query = canonical(args, precision="float32")
    grid = model.stellar_grid
    image = model.stellar_map(None, query[0]).integrated().data
    assert image.dtype == np.float32
    center = (grid.center_y, grid.center_x)
    radii = jnp.asarray(profile_radii(grid.shape, center), dtype=image.dtype)
    sampler_line = azimuthal_mean_profile.__code__.co_firstlineno
    _assert_dependency_closed(
        azimuthal_mean_profile,
        image,
        center,
        radii,
        None,
        "quintic",
        match=(
            rf"[^\n]*/optixstuff/performance\.py:{sampler_line}: "
            r"precision widened to float64"
        ),
    )


@pytest.mark.parametrize("mode", ["blend", "auto"])
@pytest.mark.parametrize(
    "method",
    [
        "create_psf_with_info",
        "create_psfs_with_info",
        "create_psf",
        "create_psfs",
        "create_psfs_parallel",
        "source_psfs",
    ],
)
def test_delivery_read_closure(delivery_models, mode, method):
    """Allow 128 peak-scaled epsilons for either backend in the synthetic read.

    Raw Kaiser sinc uses two separable 32-tap contractions. Budget 32
    roundings per contraction for a term's multiply and up to 31 additions,
    leaving 64 for local coordinates and Kaiser-sinc weight evaluation.
    Blend combines four windows, each shifted with length-16 real x and
    complex y forward/inverse FFTs (four radix-2 stages per transform),
    including the phasors, window weights and accumulation.
    """
    model = delivery_models[mode]
    query = jnp.array([[0.0625, 0.09375], [0.15625, -0.09375]])

    def read(model, query):
        if method == "source_psfs":
            return model.source_psfs(None, *query.T)
        reader = model._offaxis_backend
        x, y = query.T if method.startswith("create_psfs") else query[0]
        return getattr(reader, method)(x, y)

    args = (model, query)
    assert_precision_closed(read, *canonical(args, precision="float32"))
    if mode == "auto" and method == "create_psfs_with_info":
        assert model._offaxis_backend.reader.profile.taps == 32
        # The first source is in the served box. The second uses blend fallback.
        np.testing.assert_array_equal(read(*args).selected_backend, [1, 0])

    def values(model, query):
        result = read(model, query)
        if method == "source_psfs":
            return result.image.data, result.valid
        return (
            (result.intensity, result.accepted, result.selected_backend)
            if method.endswith("with_info")
            else result.data
        )

    precision_pair(values, args, bound=float(ulps(128, np.float32(1))))


@pytest.fixture(scope="module", params=["raw", "prefiltered"])
def sinc_reader(request):
    """Build readers over the existing analytic sinc test tables."""
    if request.param == "raw":
        return make_reader(delivery())
    data, _, _ = tone_delivery()
    reader, _ = make_prefiltered(data)
    return reader


def _kernel_stencil(reader, query, _products):
    grid = reader.grid.source
    beta, scale = reader._design()
    return read_stencil(
        reader.profile,
        query[0],
        origin_xy(grid),
        spacing_xy(grid),
        beta=beta,
        scale=scale,
    )


@pytest.mark.parametrize(
    "read",
    [
        lambda r, q, p: r.read(q[0], p[0]),
        read_batch,
        lambda r, q, p: read_batch(r, q[:0], p[:0]),
        lambda r, q, p: r.stencil(q[0]),
        _kernel_stencil,
    ],
    ids=["scalar", "batch", "empty", "stencil", "kernel-stencil"],
)
def test_sinc_read_closure(sinc_reader, read):
    """Allow for 32 taps on each of two axes plus kernel evaluation.

    Use 128 peak-scaled epsilons for the two 32-term sums, local phase
    arithmetic and the sinc or Kaiser-Bessel weights. The two queries are
    away from stencil boundaries, and both readers contract admitted samples.
    """
    query = jnp.array([[0.0625, 0.09375], [0.15625, -0.09375]])
    args = (sinc_reader, query, jnp.zeros(2, dtype=jnp.int32))
    assert_precision_closed(read, *canonical(args, precision="float32"))

    def values(*args):
        result = read(*args)
        if hasattr(result, "intensity"):
            return result.intensity, result.accepted
        return result.nodes, result.weights, result.valid

    precision_pair(values, args, bound=float(ulps(128, np.float32(1))))


@pytest.mark.parametrize(
    "method", ["create_psf", "create_psfs", "create_psfs_parallel"]
)
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_fits_backend_closure(fits_models, kind, method):
    """Allow 128 peak-scaled epsilons for the shifted Fourier blends."""
    query = jnp.array([[0.3125, 0.59375], [0.15625, -0.09375]])

    def read(backend, query):
        x, y = query[0] if method == "create_psf" else query.T
        return getattr(backend, method)(x, y).data

    args = (fits_models[kind]._offaxis_backend, query)
    assert_precision_closed(read, *canonical(args, precision="float32"))
    precision_pair(read, args, bound=float(ulps(128, np.float32(1))))


@pytest.fixture(scope="module")
def sinc_scene():
    """Use a raster whose four nonzero cells share a small coefficient box."""
    data, _, _ = tone_delivery()
    reader, _ = make_prefiltered(data)
    x = np.array([-0.09375, 0.09375])
    y = np.array([-0.0625, 0.0625])
    return SceneRenderer.from_raster(reader, 0, x, y)


@pytest.mark.parametrize("method", ["render_points", "render_raster", "render"])
def test_sinc_scene_closure(sinc_scene, method):
    """Allow 128 peak-scaled epsilons for spread, contraction and image summation.

    Four sources put at most four contributions into each amplitude and a
    seven-tap stencil per axis bounds the number of terms per source.
    """
    x = jnp.array([-0.09375, 0.09375])
    y = jnp.array([-0.0625, 0.0625])
    flux = jnp.array([[1.0, 0.5], [0.25, 0.125]])
    grid = replace(
        sinc_scene.reader.grid.source,
        shape=(2, 2),
        center_x=0.5,
        center_y=0.5,
        basis=((0.1875, 0.0), (0.0, 0.125)),
    )

    def read(scene, flux, x, y):
        if method == "render_points":
            points = jnp.stack(jnp.meshgrid(x, y), axis=-1).reshape(-1, 2)
            return scene.render_points(points, flux.ravel())
        if method == "render_raster":
            return scene.render_raster(flux, x, y)
        return scene.render(flux, grid)

    def values(*args):
        result = read(*args)
        return (
            (result.intensity, result.accepted)
            if hasattr(result, "intensity")
            else result
        )

    args = (sinc_scene, flux, x, y)
    assert_precision_closed(read, *canonical(args, precision="float32"))
    precision_pair(values, args, bound=float(ulps(128, np.float32(1))))


def test_annulus_membership_closure():
    """Require exact membership for ordinary, boundary and rejected queries."""
    domain = Domain("annulus", "closed", ((0, 0), (0, 0)), (0.25, 1.0))
    points = jnp.array([[0.0, 0.0], [0.25, 0.0], [0.5, -0.25], [0, 1.25]])
    assert_precision_closed(domain.contains, points.astype(jnp.float32))
    precision_pair(domain.contains, (points,), bound=0)


@pytest.mark.parametrize("kind", ["blend", "masked", "dense"])
@pytest.mark.parametrize("entry_point", ["operator", "coronagraph"])
def test_scene_coronagraph_closure(fits_models, delivery_models, kind, entry_point):
    """Allow 256 peak-scaled epsilons for Fourier blending plus detector reads.

    The dense route uses the same budget for a sum of stored source images.
    """
    model = delivery_models["auto"] if kind == "dense" else fits_models["2d"]
    model = model.with_scene_operator()
    grid = model.scene_grid
    if kind == "masked":
        operator = MaskedSceneOperator(model.operators[0], jnp.ones(grid.shape, bool))
        model = SceneCoronagraph(model, (operator,), grid)
    row, col = grid.shape[0] // 2, grid.shape[1] // 2
    flux = jnp.zeros(grid.shape).at[row, col].set(1.0)

    def read(model, flux):
        if entry_point == "operator":
            return model.operators[0].render(flux, grid)
        scene = Image(flux, grid, (), Measure.integral(), "flux")
        return model.render_scene(scene, None, grid=model.grid)

    args = (model, flux)
    assert_precision_closed(read, *canonical(args, precision="float32"))

    def values(model, flux):
        result = read(model, flux)
        return (
            (result.image.data, result.valid)
            if entry_point == "coronagraph"
            else result
        )

    precision_pair(values, args, bound=float(ulps(256, np.float32(1))))


@pytest.mark.xfail(
    strict=True,
    raises=_DependencyPrecisionError,
    reason="optixstuff PixelGrid.coordinates widens scene resampling to float64",
)
def test_render_scene_with_resampled_input_closure(fits_models):
    """Allow 256 peak-scaled epsilons for coverage, overlap and detector reads.

    A quarter-pixel shift splits the lit source across two native cells before
    the Fourier blend. The flux remains inside the scene and admitted.
    """
    model = fits_models["2d"].with_scene_operator()
    source = model.scene_grid
    shifted = replace(source, center_x=source.center_x + 0.25)
    row, col = (length // 2 for length in source.shape)
    flux = jnp.zeros(source.shape).at[row, col].set(1)

    def read(model, flux):
        scene = Image(flux, shifted, (), Measure.integral(), "flux")
        result = model.render_scene(scene, None, grid=model.grid)
        return result.image.data, result.valid

    args = (model, flux)
    data, valid = read(*args)
    assert valid and np.max(data) > 0
    precision_pair(read, args, bound=float(ulps(256, np.float32(1))))
    _assert_dependency_closed(
        read, *canonical(args, precision="float32"), match=_GRID_WIDENING
    )
