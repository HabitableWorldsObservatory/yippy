"""Absolute photometry and the shared curve record."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.grids import Frame, Image, PixelGrid

from yippy import Circular, HalfMaximum, Matched, PerformanceCurves, performance_curves


def test_circular_centre_offset_gaussian(displaced_gaussian_source):
    """Fails if source photometry searches for the displaced PSF peak.

    The offset-circle integral is a 1-D polar quadrature using I0; peak
    photometry is T*(1-exp(-R**2/(2*sigma**2))). The Gaussian's alias/edge
    errors are below 1e-18. A 1e-4 absolute bound covers the retained midpoint
    intensity quadrature at h=0.02 lod; the two estimands differ by >0.0005.
    Basis: mutation (swapping the centre fails), quadrature checked below.
    """
    from scipy.integrate import quad
    from scipy.special import i0

    total, sigma, radius = 0.31, 0.6, 0.7
    displacement = np.hypot(0.08, 0.06)
    offset, error = quad(
        lambda r: (
            total
            * r
            / sigma**2
            * np.exp(-(r * r + displacement**2) / (2 * sigma**2))
            * i0(r * displacement / sigma**2)
        ),
        0,
        radius,
        epsabs=1e-14,
        epsrel=1e-14,
    )
    assert error < 1e-14
    centred = total * (1 - np.exp(-(radius**2) / (2 * sigma**2)))
    assert centred - offset > 0.0005
    with jax.enable_x64():
        source, star = displaced_gaussian_source(0.2)
        for centre, expected in (("source", offset), ("peak", centred)):
            rule = Circular(radius, centre=centre)
            record = performance_curves(source, star, rule, separations_lod=[0.375])
            assert record.rule == rule
            np.testing.assert_allclose(record.throughput, expected, atol=1e-4, rtol=0)
        assert Circular(radius).centre == "source"
        assert len({Circular(radius), Circular(radius, centre="peak")}) == 2
        for invalid in ("centroid", "SOURCE", None, 1):
            with pytest.raises(ValueError, match="centre"):
                Circular(radius, centre=invalid)


def test_source_centre_exact_coverage_and_quadrature(displaced_gaussian_source):
    """Fails if pixelization or oversampling moves the source-centred disk.

    Basis: floating-point, 256 eps for exact native coverage and pi R**2.
    Equal physical subpixel pitch (0.02 lod) also preserves throughput to
    256 eps across two detector pitches; Gaussian alias error <1e-18.
    Fixed detector pitch with factors 5,10,20 retains a finite quadrature
    residual: require refinement toward the independent offset integral,
    rather than falsely calling midpoint intensity samples exact integrals.
    """
    from scipy.integrate import quad
    from scipy.special import i0

    from yippy.performance import _performance_curves

    radius, sigma, total = 0.7, 0.6, 0.31
    d2 = 0.08**2 + 0.06**2
    expected = quad(
        lambda r: (
            total
            * r
            / sigma**2
            * np.exp(-(r * r + d2) / (2 * sigma**2))
            * i0(r * np.sqrt(d2) / sigma**2)
        ),
        0,
        radius,
        epsabs=1e-14,
    )[0]
    tolerance = 256 * np.finfo(float).eps
    fluxes = []
    with jax.enable_x64():
        rule = Circular(radius)
        for pitch, factor in ((0.2, 10), (0.1, 5)):
            source, star = displaced_gaussian_source(pitch)
            psf = source(0.375)
            native = rule.weights(psf, (0.375, 0.0), oversample=1)
            for oversample in (factor, 20):
                weights = rule.weights(psf, (0.375, 0.0), oversample=oversample)
                np.testing.assert_allclose(weights, native, atol=tolerance, rtol=0)
                np.testing.assert_allclose(
                    float(weights.sum()) * pitch**2, np.pi * radius**2, rtol=tolerance
                )
            record = _performance_curves(
                source, star, rule, separations_lod=[0.375], oversample=factor
            )
            fluxes.append(record.throughput[0])
        np.testing.assert_allclose(fluxes[0], fluxes[1], rtol=tolerance)
        source, star = displaced_gaussian_source(0.2)
        errors = [
            abs(
                _performance_curves(
                    source, star, rule, separations_lod=[0.375], oversample=f
                ).throughput[0]
                - expected
            )
            for f in (5, 10, 20)
        ]
        assert errors[2] < errors[1] < errors[0] < 1e-4


def test_record_interpolation_spaces():
    """Fails if contrast is interpolated linearly or throughput logarithmically.

    The two-point midpoint is exactly an arithmetic/geometric mean; 2e-14
    permits float64 log/power roundoff only.
    """
    record = PerformanceCurves(
        separation_lod=np.array([0.0, 2.0]),
        frame=Frame(500, 6, "circumscribed"),
        rule=Circular(0.7),
        throughput=np.array([0.1, 0.9]),
        core_area_lod2=np.ones(2),
        leakage_fraction=np.array([1e-8, 1e-4]),
        leakage_density_lod2=np.array([1e-8, 1e-4]),
        raw_contrast=np.array([1e-8, 1e-4]),
        iwa_lod=1.0,
        data_extent_lod=4.0,
    )
    for name in ("leakage_fraction", "leakage_density_lod2", "raw_contrast"):
        np.testing.assert_allclose(record.evaluate(name, 1.0), 1e-6, rtol=2e-14)
    np.testing.assert_allclose(record.evaluate("throughput", 1.0), 0.5, rtol=2e-14)
    with jax.enable_x64():
        slope = np.log(1e4) / 2
        for point in (0.0, 1.0, 2.0):
            derivative = jax.grad(lambda q: record.evaluate("raw_contrast", q))(
                jnp.asarray(point)
            )
            np.testing.assert_allclose(
                derivative, slope * 1e-8 * np.exp(slope * point), rtol=2e-14
            )
        assert np.isposinf(record.evaluate("raw_contrast", jnp.asarray(10000.0)))


def test_matched_gaussian_definition():
    """Fails if matched weights or noise-equivalent area use geometric area.

    For p=T exp(-r^2/(2 sigma^2))/(2 pi sigma^2), peak-centered samples give
    tau=T/2 and A=4 pi sigma^2. At sigma/step=4 the stricter p^2 lattice
    error is O(exp(-pi^2 * 16)) < 1e-68; the border is over ten sigma away
    (omitted p flux < 1e-20). The 2e-12 bound permits float64 arithmetic
    and summation, comfortably above both pixelization and finite-box errors.
    """
    with jax.enable_x64():
        grid = PixelGrid.focal(81, 0.1, Frame(500, 6, "circumscribed"))
        x, y = grid.coordinates()
        sigma, total = 0.4, 0.3
        p = (
            total
            * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            / (2 * np.pi * sigma * sigma)
            * 0.1**2
        )
        psf = Image(p, grid, (), "per_pixel", "stellar_flux_fraction")
        star = Image(
            np.full(grid.shape, 1e-8), grid, (), "per_pixel", "stellar_flux_fraction"
        )
        rule = Matched()
        record = performance_curves(
            lambda s: psf, star, rule, separations_lod=[0.0, 1.0]
        )
        np.testing.assert_allclose(
            rule.weights(psf, (0.0, 0.0)), p / p.max(), rtol=2e-14
        )
        np.testing.assert_allclose(record.throughput, total / 2, rtol=2e-12)
        np.testing.assert_allclose(
            record.core_area_lod2, 4 * np.pi * sigma * sigma, rtol=2e-12
        )
        assert hash(rule) == hash(Matched())
        assert not hasattr(record, "OWA")
        assert record.data_extent_lod == 81 * 0.1 / 2


def _gaussian_source():
    """Small analytic source; the nearest border is over eight sigma away."""
    grid = PixelGrid.focal(49, 0.25, Frame(500, 6, "circumscribed"))
    x, y = grid.coordinates()

    def source(separation):
        p = (
            0.31
            * np.exp(-((x - separation) ** 2 + y * y) / (2 * 0.4**2))
            / (2 * np.pi * 0.4**2)
            * 0.25**2
        )
        return Image(p, grid, (), "per_pixel", "stellar_flux_fraction")

    return source, x, y, grid


def test_geometric_rules_preserve_primitive_curves():
    """Fails if the record moves a core or renormalizes its planet flux.

    Parent values are computed here from retained low-level measurements;
    2e-12 is float64 summation roundoff for identical quadrature. The ring
    mean uses the parent's spatial reader, divided by typed pixel area.
    """
    from yippy.performance import (
        azimuthal_mean_profile,
        circular_aperture,
        truncation_core,
    )

    with jax.enable_x64():
        source, x, y, grid = _gaussian_source()
        star = Image(
            1e-8 * (2 + 0.03 * (x * x + y * y) + 0.2 * x) * 0.25**2,
            grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
        )
        seps = np.array([1.0, 2.0])
        for rule in (Circular(0.7, centre="peak"), HalfMaximum(0.5)):
            record = performance_curves(
                source,
                star,
                rule,
                separations_lod=seps,
                leakage_rule="azimuthal_ring_mean",
            )
            for i, sep in enumerate(seps):
                nominal = (24.0, 24.0 + sep / 0.25)
                if isinstance(rule, Circular):
                    tau, area, _ = circular_aperture(
                        source(sep).data,
                        0.7 / 0.25,
                        0.25,
                        oversample=10,
                        half_width_pix=8,
                        nominal_center=nominal,
                    )
                else:
                    tau, area = truncation_core(
                        source(sep).data,
                        0.5,
                        0.25,
                        oversample=10,
                        half_width_pix=12,
                        nominal_center=nominal,
                    )
                np.testing.assert_allclose(record.throughput[i], tau, rtol=2e-12)
                np.testing.assert_allclose(record.core_area_lod2[i], area, rtol=2e-12)
            parent_mean = azimuthal_mean_profile(star.data, (24.0, 24.0), seps / 0.25)
            np.testing.assert_allclose(
                record.leakage_density_lod2, parent_mean / 0.25**2, rtol=2e-12
            )


def test_absolute_leakage_sampling():
    """Fails if leakage is density, contrast, or normalized by PSF throughput.

    A varying per-pixel stellar map has an exact aperture sum at either
    oversampling. The difference is bounded by the independent native-weight
    sampling gap plus 2e-12 relative float64 DFT/summation roundoff.
    """
    from yippy.performance import _performance_curves

    with jax.enable_x64():
        source, x, y, grid = _gaussian_source()
        star = Image(
            3e-9 * (2 + np.sin(x) + 0.03 * y**2),
            grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
        )
        rule, sep = Circular(0.7), 1.0
        measured, primitive = [], []
        for factor in (5, 10):
            record = _performance_curves(
                source, star, rule, separations_lod=[sep], oversample=factor
            )
            weights = rule.weights(source(sep), (sep, 0.0), oversample=factor)
            expected = float(np.sum(np.asarray(star.data) * np.asarray(weights)))
            measured.append(record.leakage_fraction[0])
            primitive.append(expected)
            np.testing.assert_allclose(measured[-1], expected, rtol=2e-12, atol=0)
        assert (
            abs(measured[1] - measured[0])
            <= abs(primitive[1] - primitive[0]) + 2e-12 * primitive[1]
        )
        small_grid = PixelGrid.focal(5, 0.25, grid.frame)
        small_star = Image(
            np.full(small_grid.shape, 3e-9),
            small_grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
        )
        outside = performance_curves(source, small_star, rule, separations_lod=[2.0])
        # The disk's nearest edge is 1.3 lod, beyond the 0.625 lod stellar edge.
        # Zero extension must not wrap the opposite side of the stellar image.
        assert outside.leakage_fraction[0] == 0.0


def test_leakage_mean_conventions_and_disconnected_core():
    """Fails if a ring mean replaces a local mean or islands join the peak core.

    Constant density is symmetric and agrees at the 2e-12 DFT floor. An
    azimuthal cosine modulation has exactly zero ring mean but nonzero local
    mean. The 1 percent gap is far below its 10 percent imposed modulation.
    """
    with jax.enable_x64():
        source, x, _, grid = _gaussian_source()
        for asymmetric in (False, True):
            density = (
                1e-8 * (1 + 0.1 * np.sin(x))
                if asymmetric
                else np.full(grid.shape, 1e-8)
            )
            star = Image(
                density * 0.25**2, grid, (), "per_pixel", "stellar_flux_fraction"
            )
            local = performance_curves(
                source, star, Circular(0.7), separations_lod=[1.0, 2.0]
            )
            ring = performance_curves(
                source,
                star,
                Circular(0.7),
                separations_lod=[1.0, 2.0],
                leakage_rule="azimuthal_ring_mean",
            )
            assert local.leakage_rule == "aperture_mean"
            assert ring.leakage_rule == "azimuthal_ring_mean"
            np.testing.assert_allclose(
                local.leakage_fraction, ring.leakage_fraction, rtol=2e-12
            )
            if asymmetric:
                assert np.all(
                    abs(local.leakage_density_lod2 / ring.leakage_density_lod2 - 1)
                    > 0.01
                )
            else:
                np.testing.assert_allclose(
                    local.leakage_density_lod2, ring.leakage_density_lod2, rtol=2e-12
                )
        # A brighter, disconnected second core inside the local window
        # must not become part of the nominal source's aperture.
        p = source(1.0).data + source(3.0).data * 1.1
        image = Image(p, grid, (), "per_pixel", "stellar_flux_fraction")
        weights = np.asarray(HalfMaximum().weights(image, (1.0, 0.0)))
        assert weights[24, 28] > 0.99
        assert weights[24, 36] == 0.0
