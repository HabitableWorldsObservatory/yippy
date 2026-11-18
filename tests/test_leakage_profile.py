"""The core mean intensity is the azimuthal mean of the stellar map.

``stellar_intensity_curves(backend, d)`` records the leaked-starlight intensity
an exposure-time calculator reads; ``Coronagraph.stellar_intens(d)`` is the
map an image simulator renders. The two describe the same star, so at every
separation and every stellar diameter, tabulated or not, the curve must equal
the mean of the map on the circle of that radius about the optical axis.

The reference here is independent of the code under test: the map is
interpolated with a bivariate quintic spline (scipy) and averaged over a dense
ring of position angles. On a well-sampled map that mean converges to the exact
azimuthal mean; on the shipped vortex package it agrees with the Fourier-Bessel
series of the map to 0.4 percent everywhere tested.
"""

import jax
import numpy as np
import pytest
from conftest import REFERENCE_DATA_REQUIRED
from scipy.interpolate import RectBivariateSpline

from yippy import open_yip
from yippy.performance import stellar_intensity_curves

# Separations (lambda/D) and stellar diameters (lambda/D) probed on the
# shipped packages. The diameters fall on, and between, the tabulated ones.
SEPARATIONS_LOD = np.linspace(2.0, 25.0, 93)
DIAMETERS_LOD = (0.0, 0.0316, 0.038, 0.045, 0.09, 0.2)

# Tolerances on |curve / reference - 1| over the separation grid at one
# diameter. The median and 90th percentile bound the curve where it carries
# most of the leakage; the maximum allows for the deepest nulls of the
# profile, where a few-percent relative error is a tiny absolute one. The
# radial-bin profile this replaced reached medians of 1.2 to 140 percent and
# maxima of 18 to 330 percent on the same grid.
MEDIAN_TOL = 5e-3
P90_TOL = 1e-2
MAX_TOL = 5e-2


def ring_mean(image, center, radii_pix, n_theta=2048):
    """Mean of a quintic-spline interpolant of ``image`` on circles."""
    spline = RectBivariateSpline(
        np.arange(image.shape[0]), np.arange(image.shape[1]), image, kx=5, ky=5
    )
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    return np.array(
        [
            spline.ev(
                center[0] + r * np.sin(theta), center[1] + r * np.cos(theta)
            ).mean()
            for r in radii_pix
        ]
    )


def relative_errors(coro, diameter_lod, separations_lod):
    """|curve / reference - 1| and the reference, at each separation."""
    pixel_lod = float(coro.grid.step)
    image = np.asarray(
        coro.stellar_map(None, diameter_lod, grid=None).data, dtype=np.float64
    )
    reference = ring_mean(
        image,
        (coro.stellar_grid.center_y, coro.stellar_grid.center_x),
        separations_lod / pixel_lod,
    )
    curve = np.asarray(
        stellar_intensity_curves(coro, diameter_lod, interpolation="keys").evaluate(
            separations_lod
        ),
        dtype=np.float64,
    )
    return np.abs(curve / reference - 1.0), reference


@pytest.fixture(scope="module", params=["usort_offaxis_ovc", "eac1_aavc_2d"])
def shipped(request):
    """A shipped package, or a skip that says why it is absent."""
    from yippy import fetch_yip

    try:
        path = fetch_yip(request.param)
    except Exception as exc:
        message = (
            f"data-absent: {request.param} is not cached and could not be "
            f"fetched ({exc})"
        )
        if REFERENCE_DATA_REQUIRED:
            pytest.fail(message)
        pytest.skip(message)
    return open_yip(path).prepare()


@pytest.mark.parametrize("diameter_lod", DIAMETERS_LOD)
def test_curve_is_the_azimuthal_mean_of_the_map(shipped, diameter_lod):
    """At every diameter the curve is the ring mean of the map at that diameter."""
    err, _ = relative_errors(shipped, diameter_lod, SEPARATIONS_LOD)
    worst = SEPARATIONS_LOD[np.argmax(err)]
    summary = (
        f"{shipped.package.source.path.name} at {diameter_lod} lambda/D: "
        f"median {np.median(err):.4f}, "
        f"90th percentile {np.percentile(err, 90):.4f}, max {err.max():.4f} "
        f"at {worst:.2f} lambda/D"
    )
    assert np.median(err) <= MEDIAN_TOL, summary
    assert np.percentile(err, 90) <= P90_TOL, summary
    assert err.max() <= MAX_TOL, summary


# A synthetic package needs no data: one smooth ring per tabulated diameter,
# its radius and brightness growing with the diameter, so a diameter between
# two nodes gives a map that is neither of theirs.
RING_DIAMS_LOD = (0.0, 0.05, 0.1, 0.2)
RING_RADIUS_PIX = (8.0, 9.0, 11.0, 14.0)
RING_LEVEL = (1e-10, 3e-10, 2e-9, 3e-8)
RING_WIDTH_PIX = 3.0
RING_FLOOR = 1e-13


@pytest.fixture(scope="module")
def ring_coro(tmp_path_factory):
    """A package whose stellar maps are smooth rings about the optical axis."""
    import astropy.io.fits as pyfits
    from synthetic_yip import write_synthetic_yip

    package = write_synthetic_yip(tmp_path_factory.mktemp("rings") / "rings")
    with pyfits.open(package.path / "stellar_intens.fits") as hdul:
        header = hdul[0].header.copy()
    axis = (package.npix - 1) / 2
    yy, xx = np.mgrid[: package.npix, : package.npix]
    r = np.hypot(yy - axis, xx - axis)
    maps = np.stack(
        [
            RING_FLOOR + level * np.exp(-0.5 * ((r - radius) / RING_WIDTH_PIX) ** 2)
            for radius, level in zip(RING_RADIUS_PIX, RING_LEVEL, strict=True)
        ]
    )
    header["NAXIS3"] = len(RING_DIAMS_LOD)
    pyfits.PrimaryHDU(maps, header=header).writeto(
        package.path / "stellar_intens.fits", overwrite=True
    )
    pyfits.PrimaryHDU(np.asarray(RING_DIAMS_LOD)).writeto(
        package.path / "stellar_intens_diam_list.fits", overwrite=True
    )
    return open_yip(package.path).prepare()


@pytest.mark.parametrize("diameter_lod", (0.0, 0.03, 0.05, 0.075, 0.15, 0.2))
def test_curve_follows_the_map_between_tabulated_diameters(ring_coro, diameter_lod):
    """Between nodes the curve is the ring mean of the interpolated map.

    A curve interpolated in diameter on its own, rather than taken from the
    map, departs from the map's ring mean by tens of percent here, because
    the map interpolates in log intensity and the ring moves outward. The
    check covers the ring itself, where the map is at least a tenth of its
    crest; far down the Gaussian tails the cubic sampling kernel's error
    relative to the local value grows to about one percent.
    """
    pixel_lod = float(ring_coro.grid.step)
    separations = np.arange(4.0, 22.0, 0.5) * pixel_lod
    err, reference = relative_errors(ring_coro, diameter_lod, separations)
    err = np.where(reference >= 0.1 * reference.max(), err, 0.0)
    assert err.max() <= 3e-3, (
        f"curve departs from the map's ring mean by {err.max():.4f} at "
        f"{separations[np.argmax(err)] / pixel_lod:.1f} px"
    )


@pytest.mark.parametrize("leakage_rule", ["aperture_mean", "azimuthal_ring_mean"])
def test_jax_view_matches_the_loader_between_diameters(ring_coro, leakage_rule):
    """A nonconstant radial profile agrees in float64, including linear tails."""
    from optixstuff.performance import azimuthal_mean_profile, profile_radii
    from scipy.interpolate import PchipInterpolator

    with jax.enable_x64():
        coro = ring_coro.package.prepare(leakage_rule=leakage_rule)
        grid = coro.stellar_grid
        radii = profile_radii(grid.shape, (grid.center_y, grid.center_x))
        nodes = radii * grid.step
        separations = np.array(
            [
                nodes[0] - 0.3,
                nodes[0],
                1.39,
                2.03,
                3.37,
                nodes[-1],
                nodes[-1] + 0.3,
                nodes[-1] + 2.0,
            ]
        ).reshape(2, 4)
        for diameter in (0.0, 0.075):
            image = coro.stellar_map(None, diameter, grid=None).data
            values = np.asarray(
                azimuthal_mean_profile(image, (grid.center_y, grid.center_x), radii)
            )
            expected = PchipInterpolator(nodes, values)(separations)
            expected = np.where(
                separations < nodes[0],
                values[0]
                + (separations - nodes[0])
                * (values[1] - values[0])
                / (nodes[1] - nodes[0]),
                expected,
            )
            expected = np.where(
                separations > nodes[-1],
                values[-1]
                + (separations - nodes[-1])
                * (values[-1] - values[-2])
                / (nodes[-1] - nodes[-2]),
                expected,
            )

            def method(s, d):
                return stellar_intensity_curves(coro, d).evaluate(s)

            np.testing.assert_allclose(
                method(separations, diameter), expected, rtol=1e-12, atol=1e-24
            )
            np.testing.assert_allclose(
                jax.jit(method)(separations, diameter), expected, rtol=1e-12, atol=1e-24
            )


def test_diameter_outside_the_tabulated_range_is_nan(ring_coro):
    """The loader does not extrapolate the curve past its largest diameter."""
    assert np.isnan(
        stellar_intensity_curves(ring_coro, 0.3, interpolation="keys").evaluate(2.5)
    )


@pytest.mark.parametrize("diameter", [0.0, 0.075])
@pytest.mark.parametrize("shape", [(3,), (2, 2)])
def test_eqx_separation_arrays_keep_shape(ring_coro, diameter, shape):
    """Array evaluation agrees with scalar vmap, eagerly and under JIT."""
    import jax
    import jax.numpy as jnp

    coro = ring_coro
    separations = jnp.linspace(1.5, 3.5, np.prod(shape)).reshape(shape)
    expected = jax.vmap(
        lambda s: stellar_intensity_curves(
            coro, getattr(diameter, "value", diameter), interpolation="keys"
        ).evaluate(s)
    )(separations.ravel()).reshape(shape)
    for evaluate in (
        (
            lambda separation, diameter=0.0, interpolation="keys": (
                stellar_intensity_curves(
                    coro, diameter, interpolation=interpolation
                ).evaluate(separation)
            )
        ),
        jax.jit(
            lambda s, d: stellar_intensity_curves(
                coro, d, interpolation="keys"
            ).evaluate(s)
        ),
    ):
        result = evaluate(separations, diameter)
        assert result.shape == shape
        np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("diameter", [-0.01, 0.3, np.nan])
def test_public_diameter_bounds_under_jit(ring_coro, diameter):
    """The host loader and traced array view both reject invalid diameters."""
    import jax
    import jax.numpy as jnp

    coro = ring_coro
    # The scipy loader is host-only; its inputs must be static inside JIT.
    host = jax.jit(
        lambda: jnp.asarray(
            stellar_intensity_curves(
                coro,
                getattr(diameter, "value", diameter),
                interpolation="keys",
            ).evaluate(2.5)
        )
    )()
    traced = jax.jit(
        lambda d: stellar_intensity_curves(coro, d, interpolation="keys").evaluate(2.5)
    )(diameter)
    assert np.isnan(host)
    assert np.isnan(traced)


@pytest.mark.parametrize(
    "shape,center", [((65, 65), (32, 32)), ((65, 61), (29.3, 34.7))]
)
@pytest.mark.parametrize("dtype", [float, int])
def test_corner_arcs_of_constant_maps(shape, center, dtype):
    """Every intersecting circle has mean one, even near a far corner."""
    import jax
    import jax.numpy as jnp
    from optixstuff.performance import azimuthal_mean_profile

    corner = max(
        np.hypot(y - center[0], x - center[1])
        for y in (0, shape[0] - 1)
        for x in (0, shape[1] - 1)
    )
    radii = np.r_[44.75, 45.0, 45.25, corner - 0.001, corner]
    with jax.enable_x64():
        evaluate = jax.jit(
            lambda r: azimuthal_mean_profile(jnp.ones(shape, dtype=dtype), center, r)
        )
        np.testing.assert_allclose(evaluate(radii), 1.0, atol=1e-12)
        np.testing.assert_allclose(evaluate(np.array([corner + 0.01])), 0.0)
