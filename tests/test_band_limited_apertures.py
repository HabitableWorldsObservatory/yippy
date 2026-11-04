"""Band-limited aperture sums against the clear circular pupil's Airy pattern.

The intensity and encircled energy follow Born and Wolf, Principles of Optics,
7th edition, section 8.5.2 (diffraction at a circular aperture):
I(r) = (pi / 4) [2 J1(pi r) / (pi r)]^2 and
E(r) = 1 - J0(pi r)^2 - J1(pi r)^2, with r in lambda/D.
The intensity integrates to one over the infinite image; do not normalize the
finite crop, which would fold the missing Airy wings into the core.
"""

from types import SimpleNamespace

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lod_unit import lod
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0, j1, jvp

from yippy.performance import (
    _iter_xaxis_positions,
    circular_aperture,
    compute_core_area_curve,
    compute_raw_contrast_curve,
    compute_throughput_curve,
    core_subpixel_window,
    core_window_half_width_pix,
    truncation_core,
)


def airy_psf(pixel_lod, shift_pix=0.0, size=256):
    """Sample a unit-flux Airy intensity without renormalizing its finite crop."""
    rows, cols = np.indices((size, size))
    radius = np.hypot(rows - size // 2, cols - size // 2 - shift_pix) * pixel_lod
    x = np.pi * radius
    amplitude = 2 * j1(x) / np.where(x == 0, 1.0, x)
    amplitude[x == 0] = 1.0
    return np.pi / 4 * amplitude**2 * pixel_lod**2


def image_coro(psf, pixel_lod, star=None):
    """Expose one image through the same data interface as a loaded package."""
    center_y, center_x = np.array(psf.shape) // 2
    return SimpleNamespace(
        pixel_scale_arcsec=pixel_lod * lod / u.pix,
        offax=SimpleNamespace(
            x_offsets=np.array([0.0]) * lod,
            y_offsets=np.array([0.0]) * lod,
            center_x=center_x * u.pix,
            center_y=center_y * u.pix,
            get_psf_by_offset_idx=lambda i, j: psf,
        ),
        stellar_intens=lambda diam: psf if star is None else star,
    )


def encircled_energy(radius_lod):
    """Infinite-image Airy encircled energy from Born and Wolf, section 8.5.2."""
    x = np.pi * radius_lod
    return 1 - j0(x) ** 2 - j1(x) ** 2


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize("radius_lod", [0.7, 1.0, 1.22])
def test_circle_airy_encircled_energy(pixel_lod, shift_pix, radius_lod):
    """A circular sum recovers the Airy encircled energy at each sampling phase."""
    coro = image_coro(airy_psf(pixel_lod, shift_pix), pixel_lod)
    measured = compute_throughput_curve(coro, aperture_radius_lod=radius_lod)[1][0]
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
        compute_throughput_curve(image_coro(airy_psf(pixel_lod, shift), pixel_lod))[1][
            0
        ]
        for shift in (0.0, 0.3)
    ]
    assert fluxes[0] == pytest.approx(fluxes[1], rel=1e-3)


def test_flat_image_flux_is_the_counted_circle_area():
    """A flat stellar image carries intensity times the measured aperture area."""
    psf = airy_psf(0.25, 0.3, size=64)
    coro = image_coro(psf, 0.25, star=np.full_like(psf, 3.0))
    flux = compute_throughput_curve(coro)[1][0]
    contrast = compute_raw_contrast_curve(coro)[1][0]
    area = compute_core_area_curve(coro)[1][0]
    assert contrast * flux == pytest.approx(3 * area / 0.25**2, rel=1e-6)


def test_circle_fractional_boundary_has_analytic_area():
    """Fractions of boundary cells tile the disk to one part per million."""
    coro = image_coro(airy_psf(0.25, size=64), 0.25)
    # Boundary fractions must tile the analytic disk, including axial cells.
    area = compute_core_area_curve(coro)[1][0]
    assert area == pytest.approx(np.pi * 0.7**2, rel=1e-6)


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
    coro = image_coro(airy_psf(pixel_lod, shift_pix), pixel_lod)
    measured = compute_throughput_curve(coro, aperture_radius_lod=radius_lod)[1][0]
    assert measured == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
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
    with jax.enable_x64():
        flux, area = truncation_core(
            airy_psf(pixel_lod, shift_pix),
            0.5,
            pixel_lod,
            half_width_pix=core_window_half_width_pix(pixel_lod),
        )
    assert float(area) == pytest.approx(expected_area, rel=1e-6)
    assert float(flux) == pytest.approx(expected_flux, rel=2e-6)


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
    pixel_lod = float(coro.pixel_scale_arcsec.value)
    for pos in _iter_xaxis_positions(coro):
        values = []
        for factor in (10, 40):
            with jax.enable_x64():
                psf = jnp.asarray(pos.psf, dtype=jnp.float64)
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
    values = np.stack([compute_raw_contrast_curve(shipped_coro)[1] for _ in range(20)])
    np.testing.assert_allclose(
        values, np.broadcast_to(values[0], values.shape), rtol=1e-6
    )


def test_circular_aperture_composes_under_jit():
    """The numerical kernel supports enclosing JIT with the active JAX precision."""
    with jax.enable_x64(False):
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
    _, flux = compute_throughput_curve(shipped_coro)
    assert flux[0] < 1e-8
    first_max = next(i for i in range(1, len(flux) - 1) if flux[i] > flux[i + 1])
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


def test_dft_gradients_match_periodic_modes():
    """The level-set normal is the spectral derivative, including both axes."""
    y, x = np.indices((24, 24))
    psf = 2 + np.sin(2 * np.pi * y / 24) + 0.3 * np.cos(4 * np.pi * x / 24)
    with jax.enable_x64():
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


def test_shipped_truncation_stays_near_nominal_source(shipped_coro):
    """The on-axis core excludes unrelated bright leakage outside its window."""
    pos = next(_iter_xaxis_positions(shipped_coro))
    with jax.enable_x64():
        flux, _ = truncation_core(
            jnp.asarray(pos.psf, dtype=jnp.float64),
            0.3,
            0.25,
            nominal_center=(pos.py, pos.px),
        )
    assert float(flux) < 1e-8


def test_nominal_search_preserves_fractional_pixel_position():
    """Native offsets use the same unrounded coordinates as synthesized PSFs."""
    coro = image_coro(airy_psf(0.25, 0.3, size=32), 0.25)
    coro.offax.x_offsets = np.array([0.075]) * lod
    coro.offax.center_y = 16.2 * u.pix
    pos = next(_iter_xaxis_positions(coro))
    assert pos.px == pytest.approx(16.3)
    assert pos.py == pytest.approx(16.2)


@pytest.mark.parametrize("column", [31.4, 31.6])
def test_fractional_edge_search_excludes_competing_lobe(column):
    """An in-image source keeps its search disk after window conversion."""
    y, x = np.indices((32, 32))
    psf = 0.001 * np.exp(-((y - 16) ** 2 + (x - 31.4) ** 2) / 4)
    psf += np.exp(-((y - 16) ** 2 + (x - 19) ** 2) / 4)
    with jax.enable_x64():
        flux, _, _ = circular_aperture(psf, 2.8, 0.25, nominal_center=(16.0, column))
    assert float(flux) < 0.02


@pytest.mark.parametrize("nominal", [None, (16.0, 40.0)])
def test_global_truncation_is_independent_of_window(nominal):
    """Retained off-image cores integrate every tile with fractional weights."""
    y, x = np.indices((32, 32))
    psf = (
        2 + np.cos(2 * np.pi * (y - 16) / 32) + 0.7 * np.cos(2 * np.pi * (x - 16) / 32)
    )
    with jax.enable_x64():
        values = [
            np.array(
                truncation_core(
                    psf, 0.65, 0.25, half_width_pix=h, nominal_center=nominal
                )
            )
            for h in (3, 8, 16)
        ]
        reference = np.array(
            truncation_core(
                psf, 0.65, 0.25, half_width_pix=16, nominal_center=(16.0, 16.0)
            )
        )
    np.testing.assert_allclose(values, np.broadcast_to(reference, (3, 2)), rtol=1e-10)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
def test_fixed_center_airy_photometry(pixel_lod):
    """A known centroid measures Airy energy without seeking another peak."""
    psf = airy_psf(pixel_lod, shift_pix=0.3)
    with jax.enable_x64():
        flux, area, _ = circular_aperture(
            psf, 0.7 / pixel_lod, pixel_lod, fixed_center=(128.0, 128.3)
        )
    assert float(flux) == pytest.approx(encircled_energy(0.7), rel=1e-3)
    assert float(area) == pytest.approx(np.pi * 0.7**2, rel=1e-6)


def test_fixed_center_ignores_a_brighter_remote_source():
    """Known-centroid photometry also works on arbitrary, non-PSF images."""
    y, x = np.indices((32, 32))
    image = 2 + np.exp(-((y - 24) ** 2 + (x - 24) ** 2) / 4) * 1e3
    with jax.enable_x64():
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
    with jax.enable_x64():
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
