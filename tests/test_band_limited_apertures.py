"""Band-limited aperture sums against the clear circular pupil's Airy pattern.

The intensity and encircled energy follow Born and Wolf, Principles of Optics,
7th edition, section 8.5.2 (diffraction at a circular aperture):
I(r) = (pi / 4) [2 J1(pi r) / (pi r)]^2 and
E(r) = 1 - J0(pi r)^2 - J1(pi r)^2, with r in lambda/D.
The intensity integrates to one over the infinite image; do not normalize the
finite crop, which would fold the missing Airy wings into the core.
"""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.apertures import Circular
from optixstuff.grids import Frame, Image, Measure, PixelGrid
from optixstuff.performance import (
    _circle_square_weights,
    circular_aperture,
    core_subpixel_window,
    core_window_half_width_pix,
    performance_curves,
    truncation_core,
)
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0, j1, jvp

from yippy._yip import _YIPData
from yippy.performance import _iter_xaxis_positions


def airy_psf(pixel_lod, shift_pix=0.0, size=256):
    """Sample a unit-flux Airy intensity without renormalizing its finite crop."""
    rows, cols = np.indices((size, size))
    radius = np.hypot(rows - size // 2, cols - size // 2 - shift_pix) * pixel_lod
    x = np.pi * radius
    amplitude = 2 * j1(x) / np.where(x == 0, 1.0, x)
    amplitude[x == 0] = 1.0
    return np.pi / 4 * amplitude**2 * pixel_lod**2


def image_source(psf, pixel_lod, star=None):
    """Return an explicit PSF callable and stellar Image on one physical grid."""
    center_y, center_x = np.array(psf.shape) // 2
    grid = PixelGrid.regular(
        psf.shape,
        pixel_lod,
        "lod",
        center=(center_x, center_y),
        frame=Frame(500, 6, "circumscribed"),
    )
    stellar = Image(
        psf if star is None else star,
        grid,
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    return lambda separation: Image(
        psf, grid, (), Measure.integral(), "stellar_flux_fraction"
    ), stellar


def encircled_energy(radius_lod):
    """Infinite-image Airy encircled energy from Born and Wolf, section 8.5.2."""
    x = np.pi * radius_lod
    return 1 - j0(x) ** 2 - j1(x) ** 2


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize("radius_lod", [0.7, 1.0, 1.22])
def test_circle_airy_encircled_energy(pixel_lod, shift_pix, radius_lod):
    """A circular sum recovers the Airy encircled energy at each sampling phase."""
    source, stellar = image_source(airy_psf(pixel_lod, shift_pix), pixel_lod)
    measured = performance_curves(
        source, stellar, Circular(radius_lod, centre="peak"), separations_lod=[0.0]
    ).throughput[0]
    assert measured == pytest.approx(encircled_energy(radius_lod), rel=1e-3)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize("metric", ["area", "throughput"])
def test_truncation_airy_half_maximum(pixel_lod, shift_pix, metric):
    """The half-maximum core has the Airy FWHM area and encircled energy."""
    radius = brentq(lambda r: (2 * j1(np.pi * r) / (np.pi * r)) ** 2 - 0.5, 0.1, 1)
    throughput, area = truncation_core(
        airy_psf(pixel_lod, shift_pix),
        0.5,
        pixel_lod,
        half_width_pix=core_window_half_width_pix(pixel_lod),
    )
    if metric == "area":
        assert float(area) == pytest.approx(np.pi * radius**2, rel=1e-3)
    else:
        assert float(throughput) == pytest.approx(encircled_energy(radius), rel=1e-3)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
def test_circle_is_invariant_to_subpixel_shift(pixel_lod):
    """A 0.3-pixel source shift preserves the peak-centered encircled flux."""
    fluxes = [
        performance_curves(
            *image_source(airy_psf(pixel_lod, shift), pixel_lod),
            Circular(0.7, centre="peak"),
            separations_lod=[0.0],
        ).throughput[0]
        for shift in (0.0, 0.3)
    ]
    assert fluxes[0] == pytest.approx(fluxes[1], rel=1e-3)


def test_flat_image_flux_is_the_counted_circle_area():
    """A flat stellar image carries intensity times the measured aperture area."""
    psf = airy_psf(0.25, 0.3, size=64)
    source, stellar = image_source(psf, 0.25, star=np.full_like(psf, 3.0))
    record = performance_curves(source, stellar, Circular(0.7), separations_lod=[0.0])
    flux, contrast, area = (
        record.throughput[0],
        record.raw_contrast[0],
        record.core_area_lod2[0],
    )
    assert contrast * flux == pytest.approx(3 * area / 0.25**2, rel=1e-6)


def test_circle_fractional_boundary_has_analytic_area():
    """Fractions of boundary cells tile the disk to one part per million."""
    source, stellar = image_source(airy_psf(0.25, size=64), 0.25)
    # Boundary fractions must tile the analytic disk, including axial cells.
    area = performance_curves(
        source, stellar, Circular(0.7), separations_lod=[0.0]
    ).core_area_lod2[0]
    assert area == pytest.approx(np.pi * 0.7**2, rel=1e-6)


@pytest.mark.float64_only(
    reason="Float64 reference roundoff for geometry and photometry"
)
@pytest.mark.parametrize("radius_lod", [0.7 * 575 / 553.4715922101486, 0.7, 1.3])
@pytest.mark.parametrize("centre", [(0.0, 0.0), (0.5, 0.5), (0.3, -0.1)])
def test_circle_weights_tile_the_disk_to_float64_rounding(radius_lod, centre):
    """Exact overlap weights sum to pi r^2 at float64 rounding (64 eps).

    Basis: floating-point. The previous arcsin(z / r) form lost about
    sqrt(eps) relative (5e-9) where a chord meets the radius; this test
    fails it by a factor of about 3e5.
    """
    step = 0.05
    radius_pix = radius_lod / step
    n = 2 * int(radius_pix) + 8
    rows, cols = np.indices((n, n), dtype=np.float64) - (n - 1) / 2
    weights = _circle_square_weights(
        jnp.asarray(rows - centre[1]), jnp.asarray(cols - centre[0]), radius_pix
    )
    area = float(jnp.sum(weights)) * step**2
    assert area == pytest.approx(np.pi * radius_lod**2, rel=64 * np.finfo(float).eps)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize("radius_lod", [0.7, 1.0, 1.22])
def test_circle_matches_independently_sampled_airy(pixel_lod, shift_pix, radius_lod):
    """The DFT sum agrees with exact Airy values at the same subpixel centers."""
    half = int(np.ceil(radius_lod / pixel_lod * 10)) + 1
    rows, cols = np.mgrid[-half:half, -half:half] + 0.5
    h = pixel_lod / 10
    radius = np.hypot(rows, cols) * h
    x = np.pi * radius
    intensity = np.pi / 4 * (2 * j1(x) / x) ** 2
    weights = reference_circle_weights(rows, cols, radius_lod / h)
    expected = (weights * intensity).sum() * h**2
    source, stellar = image_source(airy_psf(pixel_lod, shift_pix), pixel_lod)
    measured = performance_curves(
        source, stellar, Circular(radius_lod, centre="peak"), separations_lod=[0.0]
    ).throughput[0]
    assert measured == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.float64_only(reason="Float64 independently sampled Airy reference")
def test_truncation_matches_independently_sampled_airy(pixel_lod, shift_pix):
    """Threshold membership and flux agree with directly sampled Airy values."""
    rows, cols = np.mgrid[-100:100, -100:100] + 0.5
    radius = np.hypot(rows, cols) * pixel_lod / 10
    x = np.pi * radius
    intensity = np.pi / 4 * (2 * j1(x) / x) ** 2
    h = pixel_lod / 10
    amplitude = 2 * j1(x) / x
    derivative = np.pi**2 / 2 * amplitude * (2 * jvp(1, x) / x - 2 * j1(x) / x**2)
    grad_y = derivative * rows * h / radius
    grad_x = derivative * cols * h / radius
    weights = reference_level_weights(intensity - np.pi / 8, grad_y * h, grad_x * h)
    expected_area = weights.sum() * h**2
    expected_flux = (intensity * weights).sum() * h**2
    flux, area = truncation_core(
        airy_psf(pixel_lod, shift_pix),
        0.5,
        pixel_lod,
        half_width_pix=core_window_half_width_pix(pixel_lod),
    )
    assert float(area) == pytest.approx(expected_area, rel=1e-6)
    assert float(flux) == pytest.approx(expected_flux, rel=2e-6)


@pytest.mark.float64_only(reason="Float64 shipped subpixel quadrature benchmark")
@pytest.mark.parametrize(
    "aperture",
    [
        "circle",
        # The 40x truncation reference costs over 40 s per metric in CI.
        pytest.param("truncation", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("metric", ["throughput", "area"])
def test_shipped_ten_against_forty_subpixels(shipped_coro, aperture, metric):
    """Quantify 10x boundary quadrature against 40x on a native shipped PSF."""
    coro = shipped_coro
    pixel_lod = float(coro.grid.step)
    for pos in _iter_xaxis_positions(_YIPData(coro.package)):
        values = []
        for factor in (10, 40):
            psf = jnp.asarray(pos.image.integrated().data, dtype=jnp.float64)
            if aperture == "circle":
                result = circular_aperture(
                    psf,
                    0.7 / pixel_lod,
                    pixel_lod,
                    oversample=factor,
                    half_width_pix=int(np.ceil(1.7 / pixel_lod)) + 1,
                    nominal_center=(pos.py, pos.px),
                )
            else:
                result = truncation_core(
                    psf,
                    0.3,
                    pixel_lod,
                    oversample=factor,
                    half_width_pix=core_window_half_width_pix(pixel_lod),
                    nominal_center=(pos.py, pos.px),
                )
            values.append(float(result[0 if metric == "throughput" else 1]))
        assert values[0] == pytest.approx(values[1], rel=1e-3), (
            f"offset={pos.separation}, {aperture} {metric}: "
            f"10x={values[0]}, 40x={values[1]}"
        )


def test_shipped_contrast_is_repeatable_for_faint_stellar_sums(shipped_coro):
    """Faint stellar DFT sums are stable despite cancellation of bright modes."""
    coro = shipped_coro
    positions = list(_iter_xaxis_positions(_YIPData(coro.package)))
    images = {pos.separation: pos.image for pos in positions}
    stellar = coro.stellar_map(None, 0.0, grid=None)
    values = np.stack(
        [
            performance_curves(
                images.__getitem__,
                stellar,
                Circular(0.7),
                separations_lod=sorted(images),
            ).raw_contrast
            for _ in range(20)
        ]
    )
    np.testing.assert_allclose(
        values, np.broadcast_to(values[0], values.shape), rtol=1e-6
    )


def test_circular_aperture_composes_under_jit():
    """The numerical kernel supports enclosing JIT with the active JAX precision."""
    flux, area, star_flux = jax.jit(
        lambda psf: circular_aperture(psf, 2.0, 0.25, half_width_pix=3)
    )(jnp.zeros((8, 8)))
    assert float(flux) == 0
    assert float(star_flux) == 0
    # The first subpixel is half a subpixel inside each image edge.
    expected = quad(
        lambda x: np.sqrt(20**2 - x**2) + 0.5, -0.5, np.sqrt(20**2 - 0.5**2)
    )[0]
    expected += quad(lambda x: 2 * np.sqrt(20**2 - x**2), np.sqrt(20**2 - 0.5**2), 20)[
        0
    ]
    assert float(area) == pytest.approx(expected * 0.025**2, rel=1e-6)


def test_shipped_circle_stays_near_nominal_source(shipped_coro):
    """A dark on-axis source must not jump to a remote leakage speckle."""
    flux = shipped_coro.curves().throughput
    assert flux[0] < 1e-8
    first_max = next(
        (i for i in range(1, len(flux) - 1) if flux[i] > flux[i + 1]), len(flux) - 1
    )
    assert np.all(np.diff(flux[: first_max + 1]) >= 0)


def reference_circle_weights(rows, cols, radius):
    """Independently integrate circle-square overlaps with adaptive quadrature."""
    weights = np.zeros_like(rows)
    inner = (np.abs(rows) + 0.5) ** 2 + (np.abs(cols) + 0.5) ** 2 <= radius**2
    outer = (
        np.maximum(np.abs(rows) - 0.5, 0) ** 2 + np.maximum(np.abs(cols) - 0.5, 0) ** 2
        >= radius**2
    )
    weights[inner] = 1
    for i, j in zip(*np.where(~inner & ~outer), strict=True):
        y, x = rows[i, j], cols[i, j]
        lo, hi = max(-radius, x - 0.5), min(radius, x + 0.5)
        points = []
        for edge in (y - 0.5, y + 0.5):
            if abs(edge) < radius:
                crossing = np.sqrt(radius**2 - edge**2)
                points.extend(v for v in (-crossing, crossing) if lo < v < hi)

        def height(xx, y=y):
            bound = np.sqrt(max(radius**2 - xx**2, 0))
            return max(0, min(y + 0.5, bound) - max(y - 0.5, -bound))

        weights[i, j] = quad(height, lo, hi, points=points, epsabs=1e-11)[0]
    return weights


def reference_level_weights(level, gy, gx):
    """Clip each boundary square to an analytic tangent half-plane."""
    extent = (np.abs(gy) + np.abs(gx)) / 2
    weights = (level >= extent).astype(float)
    square = [np.array(p) for p in [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]]
    for i, j in zip(*np.where(np.abs(level) < extent), strict=True):
        polygon = []
        gradient = np.array([gy[i, j], gx[i, j]])
        for a, b in zip(square, square[1:] + square[:1], strict=True):
            va, vb = level[i, j] + gradient @ a, level[i, j] + gradient @ b
            if va >= 0:
                polygon.append(a)
            if (va >= 0) != (vb >= 0):
                polygon.append(a + va / (va - vb) * (b - a))
        vertices = np.array(polygon)
        weights[i, j] = (
            abs(
                np.sum(
                    vertices[:, 0] * np.roll(vertices[:, 1], 1)
                    - vertices[:, 1] * np.roll(vertices[:, 0], 1)
                )
            )
            / 2
        )
    return weights


@pytest.mark.float64_only(
    reason="Float64 reference roundoff for geometry and photometry"
)
def test_dft_gradients_match_periodic_modes():
    """The level-set normal is the spectral derivative, including both axes."""
    y, x = np.indices((24, 24))
    psf = 2 + np.sin(2 * np.pi * y / 24) + 0.3 * np.cos(4 * np.pi * x / 24)
    fine, sy, sx, gy, gx = core_subpixel_window(
        psf, half_width_pix=3, return_gradient=True
    )
    yy = float(sy) + (np.arange(fine.shape[0]) + 0.5) / 10 - 0.5
    xx = float(sx) + (np.arange(fine.shape[1]) + 0.5) / 10 - 0.5
    np.testing.assert_allclose(
        gy,
        np.broadcast_to(
            (2 * np.pi / 24 * np.cos(2 * np.pi * yy / 24))[:, None], fine.shape
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gx,
        np.broadcast_to(
            (-0.3 * 4 * np.pi / 24 * np.sin(4 * np.pi * xx / 24))[None, :], fine.shape
        ),
        atol=1e-12,
    )


@pytest.mark.parametrize("value,ratio", [(0.0, 0.5), (2.0, 1.0), (2.0, 1.1)])
def test_truncation_peak_only_safeguard(value, ratio):
    """When nothing exceeds threshold, every tied fine-grid peak survives."""
    flux, area = truncation_core(np.full((5, 5), value), ratio, 0.25, half_width_pix=2)
    assert float(area) == pytest.approx(25 * 0.25**2)
    assert float(flux) == pytest.approx(25 * value)


def test_shipped_truncation_stays_near_nominal_source(shipped_coro, precision):
    """The on-axis core excludes unrelated bright leakage outside its window."""
    pos = next(_iter_xaxis_positions(_YIPData(shipped_coro.package)))
    flux, _ = truncation_core(
        jnp.asarray(pos.image.integrated().data, dtype=precision),
        0.3,
        0.25,
        nominal_center=(pos.py, pos.px),
    )
    assert float(flux) < 1e-8


def test_nominal_search_preserves_fractional_pixel_position():
    """Native offsets use the same unrounded coordinates as synthesized PSFs."""
    source, _ = image_source(airy_psf(0.25, 0.3, size=32), 0.25)
    image = source(0.0)
    coro = SimpleNamespace(
        offax=SimpleNamespace(
            x_offsets=np.array([0.075]),
            y_offsets=np.array([0.0]),
            grid=replace(image.grid, center_y=16.2),
            get_psf_by_offset_idx=lambda i, j: image,
        )
    )
    pos = next(_iter_xaxis_positions(coro))
    assert pos.px == pytest.approx(16.3)
    assert pos.py == pytest.approx(16.2)


@pytest.mark.parametrize("column", [31.4, 31.6])
def test_fractional_edge_search_excludes_competing_lobe(column):
    """An in-image source keeps its search disk after window conversion."""
    y, x = np.indices((32, 32))
    psf = 0.001 * np.exp(-((y - 16) ** 2 + (x - 31.4) ** 2) / 4)
    psf += np.exp(-((y - 16) ** 2 + (x - 19) ** 2) / 4)
    flux, _, _ = circular_aperture(psf, 2.8, 0.25, nominal_center=(16.0, column))
    assert float(flux) < 0.02


@pytest.mark.float64_only(
    reason="Float64 reference roundoff for geometry and photometry"
)
@pytest.mark.parametrize("nominal", [None, (16.0, 40.0)])
def test_global_truncation_is_independent_of_window(nominal):
    """Retained off-image cores integrate every tile with fractional weights."""
    y, x = np.indices((32, 32))
    psf = (
        2 + np.cos(2 * np.pi * (y - 16) / 32) + 0.7 * np.cos(2 * np.pi * (x - 16) / 32)
    )
    values = [
        np.array(
            truncation_core(psf, 0.65, 0.25, half_width_pix=h, nominal_center=nominal)
        )
        for h in (3, 8, 16)
    ]
    reference = np.array(
        truncation_core(psf, 0.65, 0.25, half_width_pix=16, nominal_center=(16.0, 16.0))
    )
    np.testing.assert_allclose(values, np.broadcast_to(reference, (3, 2)), rtol=1e-10)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
def test_fixed_center_airy_photometry(pixel_lod):
    """A known centroid measures Airy energy without seeking another peak."""
    psf = airy_psf(pixel_lod, shift_pix=0.3)
    flux, area, _ = circular_aperture(
        psf, 0.7 / pixel_lod, pixel_lod, fixed_center=(128.0, 128.3)
    )
    assert float(flux) == pytest.approx(encircled_energy(0.7), rel=1e-3)
    assert float(area) == pytest.approx(np.pi * 0.7**2, rel=1e-6)


@pytest.mark.float64_only(
    reason="Float64 reference roundoff for geometry and photometry"
)
def test_fixed_center_ignores_a_brighter_remote_source():
    """Known-centroid photometry also works on arbitrary, non-PSF images."""
    y, x = np.indices((32, 32))
    image = 2 + np.exp(-((y - 24) ** 2 + (x - 24) ** 2) / 4) * 1e3
    flux, area, _ = circular_aperture(
        image, 2.0, 1.0, half_width_pix=3, fixed_center=(8.2, 7.8)
    )
    assert float(flux) == pytest.approx(2 * np.pi * 4, rel=1e-5)
    assert float(area) == pytest.approx(np.pi * 4, rel=1e-10)


@pytest.mark.parametrize("value,ratio", [(0.0, 0.5), (2.0, 1.0), (2.0, 1.1)])
@pytest.mark.parametrize("nominal", [None, (16.0, 40.0)])
@pytest.mark.parametrize("dtype", [float, int])
def test_global_peak_only_safeguard_covers_the_image(value, ratio, nominal, dtype):
    """Degenerate global cores retain tied maxima across all integration tiles."""
    for half in (3, 8):
        flux, area = truncation_core(
            np.full((32, 32), value, dtype=dtype),
            ratio,
            0.25,
            half_width_pix=half,
            nominal_center=nominal,
        )
        assert float(area) == pytest.approx(32**2 * 0.25**2)
        assert float(flux) == pytest.approx(32**2 * value)
