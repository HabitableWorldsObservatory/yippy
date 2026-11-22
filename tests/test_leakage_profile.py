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
        coro.stellar_map(None, diameter_lod, grid=None).integrated().data,
        dtype=np.float64,
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
            image = coro.stellar_map(None, diameter, grid=None).integrated().data
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


# A modulated map I(r, phi) = I0(r) (1 + m cos 2 phi) has closed forms for every
# leakage rule. The +x aperture picks up the modulation, the ring-averaged
# aperture removes it, and the point-sample ring mean times the area ignores the
# curvature of I0 across the aperture.
MOD_AMPLITUDE = 0.8
MOD_SCALE_LOD = 4.0
MOD_LEVEL = 1e-9
MOD_SEPARATIONS_LOD = (2.0, 3.0)


def _i0(r_lod):
    return MOD_LEVEL * np.exp(-r_lod / MOD_SCALE_LOD)


def _disk_integral(f, center_x, radius=0.7, n_r=48, n_phi=96):
    """Integral of f(x, y) over a disk by midpoint polar quadrature, in lambda/D^2."""
    r = (np.arange(n_r) + 0.5) / n_r * radius
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    area = rr * (radius / n_r) * (2 * np.pi / n_phi)
    return float(np.sum(f(center_x + rr * np.cos(pp), rr * np.sin(pp)) * area))


@pytest.fixture(scope="module")
def modulated_package(tmp_path_factory):
    """A quarterly symmetric package whose stellar map carries a cos 2 phi term."""
    import astropy.io.fits as pyfits
    from synthetic_yip import write_synthetic_yip

    package = write_synthetic_yip(
        tmp_path_factory.mktemp("modulated"), kind="2dq", max_offset_lod=4.0
    )
    axis = (package.npix - 1) / 2
    yy, xx = np.mgrid[: package.npix, : package.npix]
    x, y = (xx - axis) * package.pixscale, (yy - axis) * package.pixscale
    cos2 = (x**2 - y**2) / np.maximum(x**2 + y**2, 1e-30)
    image = _i0(np.hypot(x, y)) * (1 + MOD_AMPLITUDE * cos2)
    with pyfits.open(package.path / "stellar_intens.fits", mode="update") as hdul:
        hdul[0].data[:] = image
    return package


@pytest.fixture(scope="module")
def modulated(modulated_package):
    """The three leakage rules prepared on the modulated package."""
    from yippy import open_yip

    package = open_yip(modulated_package.path)
    with pytest.warns(DeprecationWarning, match="azimuthal_ring_mean"):
        point = package.prepare(leakage_rule="azimuthal_ring_mean")
    return {
        "default": package.prepare(),
        "aperture_mean": package.prepare(leakage_rule="aperture_mean"),
        "aperture_ring_mean": package.prepare(leakage_rule="aperture_ring_mean"),
        "azimuthal_ring_mean": point,
        "pixel_area": modulated_package.pixscale**2,
    }


@pytest.mark.parametrize("separation", MOD_SEPARATIONS_LOD)
def test_leakage_rules_on_a_modulated_map(modulated, separation):
    """Each rule reads its documented quantity from I0(r) (1 + m cos 2 phi).

    Fails if aperture_mean stops reading the +x axis, if the ring rule keeps
    any of the modulation (the +x reading would then pass it) or averages a
    point instead of the aperture, or if azimuthal_ring_mean gains the
    finite-aperture term that distinguishes it from the ring rule.
    """
    pixel_area = modulated["pixel_area"]
    s = separation

    def leak(rule):
        record = modulated[rule].curves(leakage_rule=rule)
        return float(record.evaluate("leakage_fraction", s)), record

    along_x, _ = leak("aperture_mean")
    ring, ring_record = leak("aperture_ring_mean")
    _, point_record = leak("azimuthal_ring_mean")
    area = float(point_record.evaluate("core_area_lod2", s))
    # The point rule stores its ring density per unit area; the leakage read by a
    # calculator is that density times the aperture area.
    point = float(point_record.evaluate("leakage_density_lod2", s)) * area

    x_axis = _disk_integral(
        lambda x, y: (
            _i0(np.hypot(x, y)) * (1 + MOD_AMPLITUDE * (x**2 - y**2) / (x**2 + y**2))
        ),
        s,
    )
    ring_ref = _disk_integral(lambda x, y: _i0(np.hypot(x, y)), s)
    np.testing.assert_allclose(along_x, x_axis / pixel_area, rtol=1e-3)
    np.testing.assert_allclose(ring, ring_ref / pixel_area, rtol=1e-3)
    # The two defaults differ by the modulation, so neither passes the other's test.
    assert abs(along_x / ring - 1) > 0.05
    # The point-sample rule is I0(s) times the aperture area; the ring rule adds
    # the curvature of I0 across the aperture.
    np.testing.assert_allclose(point * pixel_area / area / _i0(s), 1.0, rtol=1e-3)
    finite_aperture = ring - point
    expected = (ring_ref - _i0(s) * area) / pixel_area  # integral of I0 - I0(s)
    np.testing.assert_allclose(finite_aperture, expected, rtol=0.05)
    assert ring_record.leakage_rule == "aperture_mean"
    assert modulated["aperture_ring_mean"].leakage_rule == "aperture_ring_mean"


def test_ring_rule_is_the_default_for_a_two_dimensional_product(
    modulated, tmp_path_factory
):
    """A 2D product defaults to the ring rule and a 1D one keeps the +x cut.

    Fails if a 2D product reads the +x axis by default, or a 1D product pays for
    the ring average (on a symmetric map the two agree, so the label is the test).
    """
    from synthetic_yip import write_synthetic_yip

    from yippy import open_yip

    assert modulated["default"].leakage_rule == "aperture_ring_mean"
    one_d = write_synthetic_yip(tmp_path_factory.mktemp("one-d"), kind="1d")
    assert open_yip(one_d.path).prepare().leakage_rule == "aperture_mean"


def test_ring_azimuth_ladder_converges_on_a_modulated_map(modulated_package):
    """A cos 2 phi map is averaged exactly by any ladder rung above four angles.

    Fails if the ring mean depends on the number of angles once it exceeds the
    highest harmonic of the map, or if a map carrying harmonic n needs fewer than
    n + 1 angles (the 4-angle rung then still aliases cos 4 phi).
    """
    from yippy import open_yip
    from yippy.performance import ring_azimuth_samples, ring_symmetrized

    coro = open_yip(modulated_package.path).prepare()
    image = coro.stellar_map(None, 0.0, grid=None)
    grid = coro.stellar_grid
    default = ring_symmetrized(image).data
    assert ring_azimuth_samples(grid.shape, (grid.center_y, grid.center_x)) > 64
    for n_theta in (8, 16, 32, 64):
        rung = ring_symmetrized(image, n_theta).data
        np.testing.assert_allclose(rung, default, rtol=1e-3)


def test_curves_of_a_2d_product_warn_once(modulated_package, caplog):
    """Reading a 2D product's curves logs one warning per product, not per read.

    Fails if the warning repeats on every read, is missing on a 2D product, or is
    raised for a rotationally symmetric product.
    """
    from yippy import open_yip
    from yippy.coronagraph import _WARNED_2D

    _WARNED_2D.clear()
    coro = open_yip(modulated_package.path).prepare()
    with caplog.at_level("WARNING", logger="yippy"):
        coro.curves()
        coro.curves()
    messages = [
        r.getMessage() for r in caplog.records if "2D product" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "position angle" in messages[0]
