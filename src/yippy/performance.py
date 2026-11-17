"""Shared aperture quadrature and frozen, single-axis performance records.

Photometry uses one rule and one separation axis. All public curve interpolation
passes through PerformanceCurves.evaluate; the low-level band-limited circle
and level-set kernels remain available for convergence checks.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Frame
from hwoutils.map_coordinates import map_coordinates
from hwoutils.radial import radial_profile
from hwoutils.transforms import downsample_psf, resample_flux
from lod_unit import lod
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar

from .apertures import DEFAULT_APERTURE_RULE, ApertureRule, Circular, HalfMaximum
from .logger import logger

if TYPE_CHECKING:
    from .coronagraph import Coronagraph


CORE_OVERSAMPLE = 10
"""Subpixels per pixel side used to measure core areas and aperture fluxes."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _OffAxisPosition:
    """Data for a single off-axis PSF position along the performance axis."""

    separation: float  # |x| in lam/D
    psf: np.ndarray  # the off-axis PSF image
    px: float  # nominal x position in the image, including subpixel phase
    py: float  # nominal y position in the image, including subpixel phase


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_xaxis_positions(coro: Coronagraph):
    """Yield :class:`_OffAxisPosition` for each valid x-offset (y ~= 0).

    Iterates over each valid x-offset position along the performance-curve
    axis. All native YIP offsets are included, even those where the nominal
    point-source location lies outside the image, because the truncation-ratio
    throughput remains meaningful as long as the translated PSF retains
    significant in-image power.
    """
    y_offsets = np.array(coro.offax.y_offsets)
    y_idx = int(np.argmin(np.abs(y_offsets)))
    y_lod_val = y_offsets[y_idx]

    if y_lod_val != 0:
        logger.warning(f"No PSF at y=0, using closest y offset: {y_lod_val:.3f} lam/D")

    for i, x_lod_val in enumerate(np.array(coro.offax.x_offsets)):
        if x_lod_val < 0:
            continue
        r = abs(x_lod_val)
        planet_image = coro.offax.get_psf_by_offset_idx(i, y_idx)
        planet_psf = None if planet_image is None else np.asarray(planet_image.data)
        if planet_psf is None:
            continue
        with jax.enable_x64():
            px, py = coro.offax.grid.unit_to_pixel(x_lod_val, y_lod_val)
        px, py = float(px), float(py)
        yield _OffAxisPosition(separation=r, psf=planet_psf, px=px, py=py)


def _collect_and_sort(separations: list, values: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert lists to sorted arrays by separation."""
    sep = np.array(separations)
    val = np.array(values)
    order = np.argsort(sep)
    return sep[order], val[order]


def _point_source_in_image(pos: _OffAxisPosition) -> bool:
    """Return True if the nominal point-source pixel lies inside the PSF image."""
    ny, nx = pos.psf.shape
    return 0 <= pos.px < nx and 0 <= pos.py < ny


def _oversample_psf(
    psf: np.ndarray, pixel_scale_arcsec: float, oversample: int
) -> np.ndarray:
    """Oversample a PSF using area-scaled cubic interpolation.

    Uses ``resample_flux`` from hwoutils which converts to surface brightness
    before interpolation and back to flux units afterward. This is smooth
    center sampling, not exact target-pixel integration. Aperture estimates
    require an oversampling convergence check; clipping can also change flux.

    Args:
        psf: 2D PSF image.
        pixel_scale_arcsec: Pixel scale of the input PSF (lam/D per pixel).
        oversample: Oversampling factor.

    Returns:
        Oversampled PSF in target-pixel flux units, with negative values clamped.
    """
    os_pix = pixel_scale_arcsec / oversample
    ny_os = psf.shape[0] * oversample
    nx_os = psf.shape[1] * oversample
    psf_os = np.asarray(
        resample_flux(
            jnp.asarray(np.asarray(psf, dtype=np.float64)),
            pixel_scale_arcsec,
            os_pix,
            (ny_os, nx_os),
        )
    )
    return np.maximum(psf_os, 0.0)


def _resolve_oversample(coro: Coronagraph, oversample: int | None) -> int:
    """Return the effective oversampling factor for a calculation.

    ``None`` selects AYO's rule, ``ceil(pixscale / 0.05)`` for the realized
    pixel scale in lambda/D; any other value is used as given.
    """
    if oversample is None:
        return int(np.ceil(coro.pixel_scale_arcsec.value / 0.05))
    return int(oversample)


def _threshold_mask(psf_os: np.ndarray, trunc_ratio: float) -> np.ndarray:
    """Create boolean mask of pixels exceeding ``trunc_ratio * peak``.

    Falls back to peak-only mask if no pixels exceed the threshold.
    """
    peak = psf_os.max()
    mask = psf_os > trunc_ratio * peak
    if not mask.any():
        mask = psf_os == peak
    return mask


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_throughput_curve(
    coro, rule=DEFAULT_APERTURE_RULE, oversample=CORE_OVERSAMPLE
):
    """Throughput from the common record builder."""
    record = _performance_curves(
        coro,
        coro.stellar_intens(coro.stellar_intens.diams[0]),
        rule,
        oversample=oversample,
    )
    return record.separation_lod, record.throughput


def compute_raw_contrast_curve(
    coro,
    stellar_diam=0.0 * lod,
    rule=DEFAULT_APERTURE_RULE,
    oversample=CORE_OVERSAMPLE,
):
    """Stellar aperture fraction divided by planet aperture throughput."""
    record = _performance_curves(
        coro, coro.stellar_intens(stellar_diam), rule, oversample=oversample
    )
    return record.separation_lod, record.raw_contrast


def compute_core_area_curve(
    coro,
    rule=DEFAULT_APERTURE_RULE,
    fit_gaussian=False,
    oversample=CORE_OVERSAMPLE,
):
    """Area from the common record builder; Gaussian fitting is unsupported."""
    if fit_gaussian:
        raise ValueError("select an ApertureRule; Gaussian-fit areas are not supported")
    record = _performance_curves(
        coro,
        coro.stellar_intens(coro.stellar_intens.diams[0]),
        rule,
        oversample=oversample,
    )
    return record.separation_lod, record.core_area_lod2


PEAK_SEARCH_RADIUS_LOD = 1.0
"""Radius about the nominal source used to locate its peak, in lambda/D."""


CORE_WINDOW_LOD = 3.0
"""Half-width, in lambda/D, of the window about the PSF peak holding the core."""


def core_window_half_width_pix(pixel_scale_lod: float) -> int:
    """Half-width in pixels of the core-area window for a pixel scale."""
    return int(np.ceil(CORE_WINDOW_LOD / pixel_scale_lod))


def _band_limited_samples(spectrum, start, n_pix, oversample):
    """Inverse-DFT basis onto the subpixel centers of ``n_pix`` pixels.

    Row ``j`` evaluates the image's trigonometric interpolant at pixel
    coordinate ``start + (j + 0.5) / oversample - 0.5``, the center of
    subpixel ``j`` counted from pixel ``start``.
    """
    size = spectrum.shape[0]
    coords = start + (jnp.arange(n_pix * oversample) + 0.5) / oversample - 0.5
    freqs = jnp.fft.fftfreq(size)
    return jnp.exp(2j * jnp.pi * coords[:, None] * freqs[None, :]) / size


@functools.partial(
    jax.jit, static_argnames=("oversample", "half_width_pix", "return_gradient")
)
def core_subpixel_window(
    psf,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    nominal_center=None,
    search_radius_pix=None,
    return_gradient=False,
):
    """The PSF's band-limited interpolant on subpixels about the source peak.

    The window is ``2 * half_width_pix + 1`` pixels on a side (clipped to the
    image) about the brightest pixel, and each pixel is split into
    ``oversample`` x ``oversample`` subpixels. The values are the PSF's
    trigonometric interpolant, a matrix DFT of the whole image evaluated at
    the subpixel centers, in the PSF's units (flux per pixel at that point).
    This is the image :func:`truncation_core` thresholds.

    Args:
        psf: 2-D PSF image.
        oversample: Subpixels per pixel side.
        half_width_pix: Half-width in pixels of the window.
        nominal_center: Optional source (row, column) in original pixels.
        search_radius_pix: Peak-search radius about ``nominal_center``.
        return_gradient: Also return the two spectral derivatives, in
            intensity per original pixel coordinate.

    Returns:
        ``(fine, start_y, start_x)``: the subpixel image, and the row and
        column of the window's first pixel in ``psf``. With
        ``return_gradient=True``, append ``grad_y, grad_x``.
    """
    center = _peak_index(psf, nominal_center, search_radius_pix)
    return _subpixel_window_at(psf, center, oversample, half_width_pix, return_gradient)


def _subpixel_window_at(psf, center, oversample, half_width_pix, return_gradient=False):
    """Evaluate a window about a supplied center without searching for a peak."""
    ny, nx = psf.shape
    n_y = min(2 * half_width_pix + 1, ny)
    n_x = min(2 * half_width_pix + 1, nx)
    row, col = jnp.rint(jnp.asarray(center)).astype(int)

    start_y = jnp.clip(row - half_width_pix, 0, ny - n_y)
    start_x = jnp.clip(col - half_width_pix, 0, nx - n_x)
    spectrum = jnp.fft.fft2(psf)
    basis_y = _band_limited_samples(spectrum, start_y, n_y, oversample)
    basis_x = _band_limited_samples(spectrum.T, start_x, n_x, oversample)
    fine = jnp.real(basis_y @ spectrum @ basis_x.T)
    if return_gradient:
        grad_y = jnp.real(
            (basis_y * (2j * jnp.pi * jnp.fft.fftfreq(ny))) @ spectrum @ basis_x.T
        )
        grad_x = jnp.real(
            basis_y @ spectrum @ (basis_x * (2j * jnp.pi * jnp.fft.fftfreq(nx))).T
        )
        return fine, start_y, start_x, grad_y, grad_x
    return fine, start_y, start_x


def _peak_index(image, nominal_center=None, search_radius_pix=None, valid=None):
    """Locate the brightest sample within the nominal source's search disk."""
    if nominal_center is None:
        return jnp.unravel_index(jnp.argmax(image), image.shape)
    rows = jnp.arange(image.shape[0])[:, None] - nominal_center[0]
    cols = jnp.arange(image.shape[1])[None, :] - nominal_center[1]
    inside = rows**2 + cols**2 <= search_radius_pix**2
    # Native offsets beyond the image retain the global, in-image core.
    valid = _nominal_in_image(nominal_center, image.shape) if valid is None else valid
    candidates = jnp.where(inside | ~valid, image, -jnp.inf)
    return jnp.unravel_index(jnp.argmax(candidates), image.shape)


def _nominal_in_image(nominal_center, shape):
    """Keep source validity in original image coordinates."""
    return (
        (nominal_center[0] >= 0)
        & (nominal_center[0] < shape[0])
        & (nominal_center[1] >= 0)
        & (nominal_center[1] < shape[1])
    )


def _window_peak(
    fine, start_y, start_x, oversample, nominal_center, search_radius_pix, image_shape
):
    """Peak sample in a window, restricted in global pixel coordinates."""
    center = (
        None
        if nominal_center is None
        else (
            (nominal_center[0] - start_y + 0.5) * oversample - 0.5,
            (nominal_center[1] - start_x + 0.5) * oversample - 0.5,
        )
    )
    row, col = _peak_index(
        fine,
        center,
        search_radius_pix * oversample,
        valid=None
        if nominal_center is None
        else _nominal_in_image(nominal_center, image_shape),
    )
    return jnp.array(
        [
            start_y + (row + 0.5) / oversample - 0.5,
            start_x + (col + 0.5) / oversample - 0.5,
        ]
    )


def _refine_peak(psf, center, oversample, nominal_center=None, search_radius_pix=None):
    """Refine an interior maximum using derivatives of the same DFT basis.

    Interior Newton steps require a negative-definite Hessian and stay
    within one subpixel of the initial sample. Near the search-disk boundary,
    also optimize along its circumference; flat maxima stay put.
    """
    spectrum = jnp.fft.fft2(psf)
    fy = 2j * jnp.pi * jnp.fft.fftfreq(psf.shape[0])
    fx = 2j * jnp.pi * jnp.fft.fftfreq(psf.shape[1])

    def evaluate(point):
        modes = (
            jnp.exp(fy * point[0])[:, None]
            * spectrum
            * jnp.exp(fx * point[1])[None, :]
            / psf.size
        )
        value = jnp.real(jnp.sum(modes))
        gy = jnp.real(jnp.sum(modes * fy[:, None]))
        gx = jnp.real(jnp.sum(modes * fx[None, :]))
        yy = jnp.real(jnp.sum(modes * fy[:, None] ** 2))
        xx = jnp.real(jnp.sum(modes * fx[None, :] ** 2))
        yx = jnp.real(jnp.sum(modes * fy[:, None] * fx[None, :]))
        det = yy * xx - yx**2
        safe_det = jnp.where(det > 0, det, 1.0)
        step = jnp.array([xx * gy - yx * gx, yy * gx - yx * gy]) / safe_det
        return (
            value,
            step,
            (yy < 0) & (det > 0),
            jnp.array([gy, gx]),
            jnp.array([[yy, yx], [yx, xx]]),
        )

    def refine(_, point):
        _, step, maximum, _, _ = evaluate(point)
        candidate = point - step
        accept = maximum & jnp.all(jnp.abs(candidate - center) <= 1 / oversample)
        return jnp.where(accept, candidate, point)

    point = jax.lax.fori_loop(0, 3, refine, center)
    if nominal_center is not None:
        nominal = jnp.asarray(nominal_center)
        in_image = jnp.all(nominal >= 0) & jnp.all(nominal < jnp.array(psf.shape))
        distance = jnp.linalg.norm(center - nominal)
        inside = jnp.linalg.norm(point - nominal) <= search_radius_pix
        point = jnp.where(in_image & ~inside, center, point)

        def boundary_peak(point):
            def on_circle(angle):
                return nominal + search_radius_pix * jnp.array(
                    [jnp.sin(angle), jnp.cos(angle)]
                )

            def refine_angle(_, angle):
                candidate = on_circle(angle)
                _, _, _, gradient, hessian = evaluate(candidate)
                radial = candidate - nominal
                tangent = jnp.array([radial[1], -radial[0]])
                first = gradient @ tangent
                second = tangent @ hessian @ tangent - gradient @ radial
                step = first / jnp.where(second < 0, second, 1.0)
                return angle - jnp.where(second < 0, jnp.clip(step, -0.2, 0.2), 0.0)

            delta = center - nominal
            angle = jnp.arctan2(delta[0], delta[1])
            angle = jax.lax.fori_loop(0, 5, refine_angle, angle)
            candidate = on_circle(angle)
            valid = jnp.all(candidate >= -0.5) & jnp.all(
                candidate <= jnp.array(psf.shape) - 0.5
            )
            better = evaluate(candidate)[0] > evaluate(point)[0]
            return jnp.where(valid & better, candidate, point)

        point = jax.lax.cond(
            in_image & (distance >= search_radius_pix - 2 / oversample),
            boundary_peak,
            lambda p: p,
            point,
        )
    return point, evaluate(point)[0]


def _circle_square_weights(rows, cols, radius):
    """Exact disk overlap of unit squares centered at rows and cols.

    Integrate sqrt(radius**2 - x**2) analytically in each quadrant,
    then difference its rectangle primitive at the four square corners.
    The angle is atan2(z, w) with w = sqrt(radius**2 - z**2) supplied by the
    caller where it is known exactly: arcsin(z / radius) loses about
    sqrt(eps) relative near z = radius, where its slope diverges.
    """

    def primitive(x, y):
        a = jnp.minimum(jnp.abs(x), radius)
        b = jnp.minimum(jnp.abs(y), radius)
        w_a = jnp.sqrt(jnp.maximum(radius**2 - a**2, 0.0))
        chord = jnp.sqrt(jnp.maximum(radius**2 - b**2, 0.0))
        clipped = chord < a
        split = jnp.where(clipped, chord, a)
        # At split = chord the complementary leg is b exactly.
        w_split = jnp.where(clipped, b, w_a)

        def integral(z, w):
            return 0.5 * (z * w + radius**2 * jnp.arctan2(z, w))

        return (
            jnp.sign(x)
            * jnp.sign(y)
            * (b * split + integral(a, w_a) - integral(split, w_split))
        )

    fraction = (
        primitive(cols + 0.5, rows + 0.5)
        - primitive(cols - 0.5, rows + 0.5)
        - primitive(cols + 0.5, rows - 0.5)
        + primitive(cols - 0.5, rows - 0.5)
    )
    inner = (jnp.abs(rows) + 0.5) ** 2 + (jnp.abs(cols) + 0.5) ** 2 <= radius**2
    outer = (
        jnp.maximum(jnp.abs(rows) - 0.5, 0) ** 2
        + jnp.maximum(jnp.abs(cols) - 0.5, 0) ** 2
        >= radius**2
    )
    return jnp.where(inner, 1.0, jnp.where(outer, 0.0, jnp.clip(fraction, 0, 1)))


def _measure_circular_aperture(
    psf,
    radius_pix,
    pixel_scale_lod,
    oversample,
    half_width_pix,
    star_psf=None,
    nominal_center=None,
    star_frame=None,
):
    """Host measurement boundary using float64 to resolve faint stellar leakage."""
    with jax.enable_x64():
        values = circular_aperture(
            jnp.asarray(psf, dtype=jnp.float64),
            radius_pix,
            pixel_scale_lod,
            oversample=oversample,
            half_width_pix=half_width_pix,
            nominal_center=nominal_center,
            star_frame=star_frame,
            star_psf=None
            if star_psf is None
            else jnp.asarray(star_psf, dtype=jnp.float64),
        )
        return tuple(float(value) for value in values)


@functools.partial(
    jax.jit, static_argnames=("oversample", "half_width_pix", "return_samples")
)
def circular_aperture(
    psf,
    radius_pix,
    pixel_scale_lod,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    star_psf=None,
    nominal_center=None,
    fixed_center=None,
    star_frame=None,
    return_samples=False,
):
    """Flux and exact area of a circle on the PSF's trigonometric interpolant.

    Each subpixel carries its exact area fraction inside ``radius_pix``.
    Flux is the sum of these weights times the interpolated center intensity,
    divided by ``oversample**2``; area is their sum times the subpixel solid
    angle. The circle is centered on the refined interpolated peak. An optional
    ``nominal_center=(row, column)`` restricts the peak search to a disk of
    radius :data:`PEAK_SEARCH_RADIUS_LOD` in lambda/D about the source.
    Set ``fixed_center=(row, column)`` for photometry at a known centroid
    on any image, without peak search. It takes precedence over
    ``nominal_center``. No flux renormalization or clipping is applied.

    The window must contain the circle plus a margin for locating the peak;
    it is clipped to the image as in :func:`core_subpixel_window`. When
    ``star_psf`` is supplied, its interpolant is measured at the same global
    coordinates through the same weights as the planet. An optional
    ``star_frame=(star_cy,star_cx,off_cy,off_cx,pitch_ratio)`` maps detector
    coordinates to a distinct stellar frame and converts pixel areas.
    Sampling at 0.5 lambda/D per pixel or finer is required for an unaliased
    reconstruction; finite-image periodic boundaries can still affect edges.
    This kernel uses the active JAX precision and supports ``jit`` and
    ``vmap``. Performance curves evaluate it in float64 to avoid cancellation
    in faint stellar sums, without changing the global precision setting.

    Returns:
        ``(planet_flux, area, star_flux)``; ``star_flux`` is zero when omitted.
    """
    search_radius = PEAK_SEARCH_RADIUS_LOD / pixel_scale_lod
    if fixed_center is None:
        fine, start_y, start_x = core_subpixel_window(
            psf, oversample, half_width_pix, nominal_center, search_radius
        )
        center = _window_peak(
            fine, start_y, start_x, oversample, nominal_center, search_radius, psf.shape
        )
        refined, _ = _refine_peak(
            psf, center, oversample, nominal_center, search_radius
        )
    else:
        refined = jnp.asarray(fixed_center)
        fine, start_y, start_x = _subpixel_window_at(
            psf, refined, oversample, half_width_pix
        )
    # The refined peak can lie across the search disk from the native peak.
    # Recenter the integration window so its edge cannot clip an interior disk.
    fine, start_y, start_x = _subpixel_window_at(
        psf, refined, oversample, half_width_pix
    )
    rows = (
        start_y
        + (jnp.arange(fine.shape[0])[:, None] + 0.5) / oversample
        - 0.5
        - refined[0]
    ) * oversample
    cols = (
        start_x
        + (jnp.arange(fine.shape[1])[None, :] + 0.5) / oversample
        - 0.5
        - refined[1]
    ) * oversample
    circle = _circle_square_weights(rows, cols, radius_pix * oversample)
    if return_samples:
        return fine, circle, start_y, start_x
    flux = jnp.sum(circle * fine) / oversample**2
    area = jnp.sum(circle) * (pixel_scale_lod / oversample) ** 2
    star_flux = 0.0
    if star_psf is not None:
        spectrum = jnp.fft.fft2(star_psf)
        if star_frame is None:
            basis_y = _band_limited_samples(
                spectrum, start_y, fine.shape[0] // oversample, oversample
            )
            basis_x = _band_limited_samples(
                spectrum.T, start_x, fine.shape[1] // oversample, oversample
            )
            area_ratio = 1.0
        else:
            star_cy, star_cx, off_cy, off_cx, pitch_ratio = star_frame
            row_coords = (
                star_cy
                + (
                    start_y
                    + (jnp.arange(fine.shape[0]) + 0.5) / oversample
                    - 0.5
                    - off_cy
                )
                * pitch_ratio
            )
            col_coords = (
                star_cx
                + (
                    start_x
                    + (jnp.arange(fine.shape[1]) + 0.5) / oversample
                    - 0.5
                    - off_cx
                )
                * pitch_ratio
            )
            basis_y = (
                jnp.exp(
                    2j
                    * jnp.pi
                    * row_coords[:, None]
                    * jnp.fft.fftfreq(star_psf.shape[0])
                )
                / star_psf.shape[0]
            )
            basis_x = (
                jnp.exp(
                    2j
                    * jnp.pi
                    * col_coords[:, None]
                    * jnp.fft.fftfreq(star_psf.shape[1])
                )
                / star_psf.shape[1]
            )
            area_ratio = pitch_ratio**2
        star_fine = jnp.real(basis_y @ spectrum @ basis_x.T) * area_ratio
        star_flux = jnp.sum(circle * star_fine) / oversample**2
    return flux, area, star_flux


def _level_set_weights(fine, grad_y, grad_x, threshold, oversample):
    """Fraction of each subpixel above the tangent-plane threshold."""
    # Exact area under the local tangent line in a square. The sum of two
    # independent uniforms has a trapezoidal CDF; this also handles an
    # axis-aligned normal without dividing by its vanishing component.
    a = jnp.maximum(jnp.abs(grad_y), jnp.abs(grad_x)) / oversample
    b = jnp.minimum(jnp.abs(grad_y), jnp.abs(grad_x)) / oversample
    z = jnp.abs(fine - threshold)
    safe_a = jnp.where(a > 0, a, 1.0)
    safe_b = jnp.where(b > 0, b, 1.0)
    tail = jnp.where(
        z <= (a - b) / 2,
        0.5 - z / safe_a,
        jnp.maximum((a + b) / 2 - z, 0) ** 2 / (2 * safe_a * safe_b),
    )
    core = jnp.where(fine >= threshold, 1 - tail, tail)
    return jnp.where(a > 0, core, (fine > threshold).astype(fine.dtype))


def _global_truncation(psf, threshold, pixel_scale_lod, oversample):
    """Integrate the whole interpolant in strips with bounded working memory."""
    ny, nx = psf.shape
    spectrum = jnp.fft.fft2(psf)
    fx = jnp.fft.fftfreq(nx)
    # Zero-pad Fourier modes onto the same subpixel centers as the matrix
    # DFT window. Keep the signed Nyquist mode, then take the real part only
    # after both axes have been evaluated, as in core_subpixel_window.
    modes = jnp.mod(jnp.rint(fx * nx).astype(int), nx * oversample)
    phase = jnp.exp(2j * jnp.pi * fx * (0.5 / oversample - 0.5))
    padded = jnp.zeros((ny, nx * oversample), dtype=spectrum.dtype)
    along_x = (
        jnp.fft.ifft(padded.at[:, modes].set(spectrum * phase), axis=1) * oversample
    )
    dx = (
        jnp.fft.ifft(
            padded.at[:, modes].set(spectrum * phase * (2j * jnp.pi * fx)), axis=1
        )
        * oversample
    )
    strip_height = min(4, ny)

    def integrate(index, state):
        total, highest, count = state
        start = index * strip_height
        basis = _band_limited_samples(spectrum, start, strip_height, oversample)
        fine = jnp.real(basis @ along_x)
        gy = jnp.real((basis * (2j * jnp.pi * jnp.fft.fftfreq(ny))) @ along_x)
        gx = jnp.real(basis @ dx)
        weights = _level_set_weights(fine, gy, gx, threshold, oversample)
        inside = start + jnp.arange(strip_height * oversample) / oversample < ny
        weights = jnp.where(inside[:, None], weights, 0.0)
        maximum = jnp.max(jnp.where(inside[:, None], fine, -jnp.inf))
        ties = jnp.sum((fine == maximum) & inside[:, None], dtype=jnp.int32)
        count = jnp.where(
            maximum > highest, ties, jnp.where(maximum == highest, count + ties, count)
        )
        return (
            total + jnp.array([jnp.sum(weights * fine), jnp.sum(weights)]),
            jnp.maximum(highest, maximum),
            count,
        )

    sums, highest, count = jax.lax.fori_loop(
        0,
        (ny + strip_height - 1) // strip_height,
        integrate,
        (
            jnp.zeros(2, dtype=along_x.real.dtype),
            jnp.array(-jnp.inf, dtype=along_x.real.dtype),
            jnp.array(0, dtype=jnp.int32),
        ),
    )
    sums = jnp.where(highest > threshold, sums, jnp.array([highest * count, count]))
    return sums[0] / oversample**2, sums[1] * (pixel_scale_lod / oversample) ** 2


@functools.partial(jax.jit, static_argnames=("oversample", "half_width_pix"))
def _truncation_samples(
    psf, fraction, pixel_scale_lod, oversample, half_width_pix, nominal_center
):
    search_radius = PEAK_SEARCH_RADIUS_LOD / pixel_scale_lod
    fine, start_y, start_x, grad_y, grad_x = core_subpixel_window(
        psf,
        oversample,
        half_width_pix,
        nominal_center,
        search_radius,
        return_gradient=True,
    )
    center = _window_peak(
        fine, start_y, start_x, oversample, nominal_center, search_radius, psf.shape
    )
    _, peak = _refine_peak(psf, center, oversample, nominal_center, search_radius)
    threshold = fraction * peak
    core = _level_set_weights(fine, grad_y, grad_x, threshold, oversample)
    core = jnp.where(
        jnp.any(fine > threshold), core, (fine == jnp.max(fine)).astype(fine.dtype)
    )
    return fine, core, start_y, start_x, threshold


@functools.partial(jax.jit, static_argnames=("oversample", "half_width_pix"))
def truncation_core(
    psf,
    fraction,
    pixel_scale_lod,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    nominal_center=None,
):
    """Throughput and area of the region where a PSF exceeds a fraction of its peak.

    A subpixel's weight is the exact square fraction on the bright side of
    the tangent to the level set at its center. Both the intensity and its
    gradient come from the same trigonometric interpolant (derivatives of
    the DFT basis). Area and throughput use the same weights. This local
    linear boundary approximation is second order in subpixel width.

    The threshold is ``fraction`` times the refined interpolated peak.
    If no subpixel exceeds it, the original ``fine == fine.max()`` safeguard
    selects all tied peak samples across the measurement domain: the local
    window for an in-image source, otherwise the whole image.

    With an in-image ``nominal_center``, the peak search stays within
    :data:`PEAK_SEARCH_RADIUS_LOD` of that position and only the local core
    window contributes. This excludes unrelated leakage elsewhere inside the
    IWA. Without a nominal position, or for a native offset outside the image,
    retain the global peak and integrate the whole image in strips with the
    same interpolant and fractional weights, preserving in-image power.

    An intensity PSF sampled at 0.5 lambda/D per pixel or finer supports
    band-limited reconstruction. The default ten subpixels per side resolve
    the tested Airy boundaries to 0.1%; finite-image interpolation and more
    complicated level sets still require convergence checks.

    The shapes are fixed by the static arguments, so the function is
    ``jit``-compiled once and can be ``vmap``-ped over a stack of PSFs.

    Args:
        psf: 2-D PSF image.
        fraction: Fraction of the peak that bounds the core.
        pixel_scale_lod: Pixel scale in lambda/D per pixel.
        oversample: Subpixels per pixel side.
        half_width_pix: Half-width in pixels of the subpixel window, clipped
            to the image.
        nominal_center: Optional source (row, column) in original pixels.

    Returns:
        ``(throughput, area)``: the core throughput in the PSF's flux units
        and the core area in (lambda/D)^2.
    """
    fine, core, _start_y, _start_x, threshold = _truncation_samples(
        psf,
        fraction,
        pixel_scale_lod,
        oversample,
        half_width_pix,
        nominal_center,
    )
    local = (
        jnp.sum(core * fine) / oversample**2,
        jnp.sum(core) * (pixel_scale_lod / oversample) ** 2,
    )
    in_image = (
        False
        if nominal_center is None
        else _nominal_in_image(nominal_center, psf.shape)
    )
    return jax.lax.cond(
        in_image,
        lambda: local,
        lambda: _global_truncation(psf, threshold, pixel_scale_lod, oversample),
    )


CORE_STEP_LOD = 0.25
"""Opt-in spacing, in lambda/D, for core curves measured on synthesized PSFs."""

_PSF_BATCH = 16
"""Off-axis PSFs synthesized per call when tabulating the core curves."""


def core_separations(coro: Coronagraph, step_lod: float = CORE_STEP_LOD) -> np.ndarray:
    """Separations, in lambda/D, at which the PSF is synthesized for the core curves.

    This grid is used only when a caller opts into synthesized PSFs through
    ``separation_step_lod``. Default curves use the shipped PSFs at their own
    offsets. The optional uniform grid steps by ``step_lod`` from the smallest
    non-negative x offset to the
    largest one whose PSF center still lies in the image
    (``max_offset_in_image``), less one core window (:data:`CORE_WINDOW_LOD`).
    Closer to the edge the core is cut off by the image border and the
    band-limited interpolant wraps light from the opposite border into it,
    which changes the core throughput by up to 7 percent and the area by up
    to 11 percent; a synthesized PSF beyond the image is blank. Tabulated
    offsets beyond the end of the grid keep their own PSFs.

    The tabulated offsets inside that range are added to the grid: yippy
    mixes the two neighboring PSFs with weights linear in separation, so the
    synthesized PSF, and its core, change slope at each offset.

    Args:
        coro: Coronagraph instance.
        step_lod: Grid spacing in lambda/D.

    Returns:
        1-D array of separations in lambda/D.
    """
    x_offsets = np.asarray(coro.offax.x_offsets, dtype=float)
    x_offsets = x_offsets[x_offsets >= 0]
    edge = coro.offax.max_offset_in_image.to(lod).value - CORE_WINDOW_LOD
    s_max = max(min(x_offsets.max(), edge), x_offsets.min())
    n_steps = int(np.floor((s_max - x_offsets.min()) / step_lod + 1e-9))
    grid = x_offsets.min() + step_lod * np.arange(n_steps + 1)
    if s_max - grid[-1] > 1e-9:
        grid = np.append(grid, s_max)
    # The synthesized PSF changes slope at every tabulated offset, so the
    # offsets themselves are knots.
    return np.union1d(grid, x_offsets[x_offsets <= s_max])


def _xaxis_y_offset(coro: Coronagraph) -> float:
    """The tabulated y offset closest to zero, where the x-axis curves are read."""
    y_offsets = np.asarray(coro.offax.y_offsets, dtype=float)
    return float(y_offsets[np.argmin(np.abs(y_offsets))])


def compute_truncation_curves(
    coro,
    rule=HalfMaximum(),  # noqa: B008 - frozen rule
    oversample=None,
    separation_step_lod=None,
):
    """Throughput and area of a HalfMaximum rule through the common builder."""
    if not isinstance(rule, HalfMaximum):
        raise TypeError("compute_truncation_curves requires HalfMaximum")
    separations = None
    if separation_step_lod is not None:
        dense = core_separations(coro, separation_step_lod)
        native = np.asarray(coro.offax.x_offsets, dtype=float)
        separations = np.unique(np.r_[dense, native[native > dense[-1]]])
    record = _performance_curves(
        coro,
        coro.stellar_intens(coro.stellar_intens.diams[0]),
        rule,
        separations_lod=separations,
        oversample=CORE_OVERSAMPLE if oversample is None else oversample,
    )
    return record.separation_lod, record.throughput, record.core_area_lod2


def compute_truncation_throughput_curve(
    coro,
    rule=HalfMaximum(),  # noqa: B008 - frozen rule
    oversample=None,
    separation_step_lod=None,
):
    """Throughput of the requested half-maximum rule."""
    sep, tau, _ = compute_truncation_curves(coro, rule, oversample, separation_step_lod)
    return sep, tau


def compute_truncation_core_area_curve(
    coro,
    rule=HalfMaximum(),  # noqa: B008 - frozen rule
    oversample=None,
    separation_step_lod=None,
):
    """Geometric area of the requested half-maximum rule."""
    sep, _, area = compute_truncation_curves(
        coro, rule, oversample, separation_step_lod
    )
    return sep, area


def compute_occ_trans_curve(
    coro: Coronagraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute occulter (sky) transmission curve.

    This is the radial profile of the sky transmission mask.

    Args:
        coro: Coronagraph instance.

    Returns:
        ``(separations_lod, occ_trans)`` - 1-D arrays.
    """
    sky_trans_data = coro.sky_trans.data
    grid = coro.sky_trans.grid
    separations, profile = radial_profile(
        jnp.asarray(np.asarray(sky_trans_data, dtype=np.float64)),
        grid.step,
        center=(grid.center_y, grid.center_x),
    )
    return np.asarray(separations), np.asarray(profile)


PROFILE_STEP_PIX = 0.25
"""Radial spacing, in pixels, of the tabulated core mean intensity profile."""


def profile_radii(shape: tuple[int, int], center: tuple[float, float]) -> np.ndarray:
    """Radii, in pixels, at which the core mean intensity profile is tabulated.

    The grid starts on the axis and steps by :data:`PROFILE_STEP_PIX` out to
    the farthest pixel center of an image of ``shape`` measured from
    ``center``, so the whole map is covered.

    Args:
        shape: ``(ny, nx)`` of the stellar intensity map.
        center: ``(row, column)`` of the optical axis as pixel-center indices.

    Returns:
        1-D array of radii in pixels.
    """
    ny, nx = shape
    cy, cx = center
    r_max = max(np.hypot(y - cy, x - cx) for y in (0, ny - 1) for x in (0, nx - 1))
    return np.arange(int(np.floor(r_max / PROFILE_STEP_PIX)) + 1) * PROFILE_STEP_PIX


_QUINTIC_PAD = 48


@eqx.filter_jit
def _quintic_coefficients(image):
    """Prefilter nearest-extended samples into cardinal quintic coefficients.

    Uses the causal/anticausal inverse B-spline filter of P. Thevenaz,
    T. Blu and M. Unser, "Interpolation Revisited", IEEE Transactions on
    Medical Imaging 19(7), 739-758 (2000), doi:10.1109/42.875199.
    The two stable poles factor the degree-5 interpolation system; each
    scan solves one triangular recurrence for every line simultaneously.

    A 48-sample nearest-edge halo makes the finite initialization error
    smaller than float64 roundoff at the image (largest pole magnitude
    0.431, so its tail is below 3e-18). Keep the halo for stencil reads:
    extending coefficients directly would not extend the original samples.
    """
    poles = (-0.4305753470999738, -0.04309628820326465)
    coefficients = jnp.pad(image, _QUINTIC_PAD, mode="edge")
    for axis in (0, 1):
        lines = jnp.moveaxis(coefficients, axis, 0)
        for pole in poles:
            lines = lines * ((1 - pole) * (1 - 1 / pole))

            def causal(previous, sample, pole=pole):
                value = sample + pole * previous
                return value, value

            first = lines[0] / (1 - pole)
            _, forward = jax.lax.scan(causal, first, lines[1:])
            forward = jnp.concatenate([first[None], forward])

            def anticausal(following, sample, pole=pole):
                value = pole * (following - sample)
                return value, value

            last = -pole * forward[-1] / (1 - pole)
            _, backward = jax.lax.scan(anticausal, last, forward[:-1], reverse=True)
            lines = jnp.concatenate([backward, last[None]])
        coefficients = jnp.moveaxis(lines, 0, axis)
    return coefficients


def _quintic_weights(distance):
    """Evaluate the centered degree-5 B-spline on its six-sample support."""
    t = jnp.abs(distance)
    inner = (66 + t**2 * (-60 + t**2 * (30 - 10 * t))) / 120
    middle = ((3 - t) ** 5 - 6 * (2 - t) ** 5) / 120
    outer = jnp.maximum(3 - t, 0) ** 5 / 120
    return jnp.where(t < 1, inner, jnp.where(t < 2, middle, outer))


def _quintic_sample(coefficients, coordinates):
    """Read a prefiltered image with a separable six-by-six stencil."""
    rows, cols = coordinates
    offsets = jnp.arange(-2, 4)
    iy = jnp.floor(rows).astype(jnp.int32)[..., None] + offsets
    ix = jnp.floor(cols).astype(jnp.int32)[..., None] + offsets
    wy = _quintic_weights(rows[..., None] - iy)
    wx = _quintic_weights(cols[..., None] - ix)
    samples = coefficients[
        jnp.clip(iy + _QUINTIC_PAD, 0, coefficients.shape[0] - 1)[..., :, None],
        jnp.clip(ix + _QUINTIC_PAD, 0, coefficients.shape[1] - 1)[..., None, :],
    ]
    return jnp.sum(jnp.sum(samples * wx[..., None, :], axis=-1) * wy, axis=-1)


def azimuthal_mean_profile(
    image, center, radii_pix, n_theta=None, interpolation="keys"
):
    """Azimuthal mean of an image on circles about ``center``.

    Each circle is sampled at ``n_theta`` equally spaced positions along its
    in-image angular arcs with the selected spatial interpolant
    and the samples that fall inside the image are averaged. This is the mean
    intensity at exactly that radius. Averaging the pixels whose centers fall
    in an annulus instead mixes a range of radii, which biases the profile by
    several percent wherever it is curved on the scale of the annulus width.

    Args:
        image: 2-D array.
        center: ``(row, column)`` of the circle centers as pixel-center
            indices.
        radii_pix: 1-D array of circle radii in pixels.
        n_theta: Position angles per circle. Defaults to twice the larger
            image dimension, more than the highest azimuthal harmonic a map
            sampled at or above Nyquist carries at its corner radius.
        interpolation: ``"keys"`` (default, unchanged Keys cubic) or
            ``"quintic"`` (cardinal degree-5 B-spline). Quintic coefficients
            are prefiltered once per image per call, shared by all radii.
            Both extend samples beyond the edge with the nearest pixel and
            average only in-image arcs; tangencies use endpoint values.
            This string and ``n_theta`` must be static under JIT (for example,
            use ``eqx.filter_jit``). Quintic follows P. Thevenaz, T. Blu and
            M. Unser, "Interpolation Revisited", IEEE Transactions on Medical
            Imaging 19(7), 739-758 (2000), doi:10.1109/42.875199; the prefilter
            and evaluation use only JAX and differentiate through the image.

    Returns:
        1-D JAX array, the mean of each circle's in-image samples (0 for a
        circle with none).
    """
    image = jnp.asarray(image, dtype=jnp.result_type(image, 1.0))
    if interpolation not in ("keys", "quintic"):
        raise ValueError("interpolation must be 'keys' or 'quintic'")
    if interpolation == "quintic":
        coefficients = _quintic_coefficients(image)
    ny, nx = image.shape
    if n_theta is None:
        n_theta = 2 * max(ny, nx)
    radii = jnp.asarray(radii_pix, dtype=image.dtype)[:, None]
    safe_radii = jnp.where(radii > 0, radii, 1.0)
    # Circle/edge intersections split the circumference into wholly inside
    # or outside arcs. Sampling arc length cannot miss a short corner arc.
    y_edges = (jnp.array([0, ny - 1]) - center[0]) / safe_radii
    x_edges = (jnp.array([0, nx - 1]) - center[1]) / safe_radii
    if interpolation == "keys":
        y_angles = jnp.arcsin(jnp.clip(y_edges, -1, 1))
        x_angles = jnp.arccos(jnp.clip(x_edges, -1, 1))
    else:
        # Clipped inverse trig has 0 * inf derivatives outside [-1, 1].
        # Sanitize the unused branch before evaluation. Preserve the Keys
        # path exactly; quintic has finite coordinate gradients off tangency.
        y_crosses = jnp.abs(y_edges) < 1
        x_crosses = jnp.abs(x_edges) < 1
        y_angles = jnp.where(
            y_crosses,
            jnp.arcsin(jnp.where(y_crosses, y_edges, 0.0)),
            jnp.sign(y_edges) * (jnp.pi / 2),
        )
        x_angles = jnp.where(
            x_crosses,
            jnp.arccos(jnp.where(x_crosses, x_edges, 0.0)),
            jnp.where(x_edges < 0, jnp.pi, 0.0),
        )
    crossings = jnp.mod(
        jnp.concatenate([y_angles, jnp.pi - y_angles, x_angles, -x_angles], axis=1),
        2 * jnp.pi,
    )
    ends = jnp.broadcast_to(jnp.array([0.0, 2 * jnp.pi]), (radii.shape[0], 2))
    edges = jnp.sort(jnp.concatenate([ends, crossings], axis=1), axis=1)

    def coordinates(theta):
        return center[0] + radii * jnp.sin(theta), center[1] + radii * jnp.cos(theta)

    def in_image(rows, cols, tolerance=0.0):
        return (
            (rows >= -tolerance)
            & (rows <= ny - 1 + tolerance)
            & (cols >= -tolerance)
            & (cols <= nx - 1 + tolerance)
        )

    mid = (edges[:, :-1] + edges[:, 1:]) / 2
    lengths = jnp.where(in_image(*coordinates(mid)), jnp.diff(edges, axis=1), 0.0)
    cumulative = jnp.concatenate(
        [jnp.zeros_like(radii), jnp.cumsum(lengths, axis=1)], axis=1
    )
    total_length = cumulative[:, -1:]
    positions = total_length * ((jnp.arange(n_theta) + 0.5) / n_theta)
    index = (
        jax.vmap(lambda c, t: jnp.searchsorted(c, t, side="right"))(
            cumulative, positions
        )
        - 1
    )
    index = jnp.clip(index, 0, lengths.shape[1] - 1)
    theta = (
        jnp.take_along_axis(edges, index, axis=1)
        + positions
        - jnp.take_along_axis(cumulative, index, axis=1)
    )
    if interpolation == "keys":
        values = map_coordinates(image, coordinates(theta), order=3, mode="nearest")
    else:
        values = _quintic_sample(coefficients, coordinates(theta))
    # A tangent circle has zero arc length but can touch one or more corners.
    rows, cols = coordinates(edges)
    inside = in_image(rows, cols, 8 * jnp.finfo(image.dtype).eps * max(ny, nx))
    if interpolation == "keys":
        endpoints = map_coordinates(image, [rows, cols], order=3, mode="nearest")
    else:
        endpoints = _quintic_sample(coefficients, (rows, cols))
    count = jnp.sum(inside, axis=1)
    tangent_mean = jnp.sum(jnp.where(inside, endpoints, 0.0), axis=1) / jnp.maximum(
        count, 1
    )
    return jnp.where(total_length[:, 0] > 0, jnp.mean(values, axis=1), tangent_mean)


def core_mean_intensity_profile(
    coro: Coronagraph, stellar_diam, interpolation="keys"
) -> np.ndarray:
    """Core mean intensity profile of the stellar map at ``stellar_diam``.

    The profile is the azimuthal mean of ``coro.stellar_intens(stellar_diam)``
    about the optical axis, on the radii of :func:`profile_radii`. Because it
    is computed from the map itself, the curve at any diameter, tabulated or
    not, is the azimuthal mean of the map an image simulator renders at that
    diameter.

    Args:
        coro: Coronagraph instance.
        stellar_diam: Stellar angular diameter (Quantity in lambda/D).
        interpolation: Spatial reader, ``"keys"`` or ``"quintic"``; see
            :func:`azimuthal_mean_profile`.

    Returns:
        1-D array, per pixel, on the radii of :func:`profile_radii`.
    """
    psf = np.asarray(
        coro.stellar_intens._evaluate_array(stellar_diam), dtype=np.float64
    )
    center = (coro.stellar_intens.grid.center_y, coro.stellar_intens.grid.center_x)
    radii = profile_radii(psf.shape, center)
    return np.asarray(
        azimuthal_mean_profile(
            jnp.asarray(psf), center, radii, interpolation=interpolation
        )
    )


def compute_core_mean_intensity_curve(
    coro: Coronagraph,
    stellar_diam_values=None,
    interpolation="keys",
) -> tuple[np.ndarray, dict]:
    """Compute core mean intensity curves for different stellar diameters.

    Args:
        coro: Coronagraph instance.
        stellar_diam_values:
            List of stellar diameters.  ``None`` -> use all available.
        interpolation: Spatial reader, ``"keys"`` or ``"quintic"``; see
            :func:`azimuthal_mean_profile`.

    Returns:
        ``(separations_lod, intensities_dict)`` where *intensities_dict*
        maps each stellar diameter to its radial intensity profile: the
        azimuthal mean, about the optical axis (the stellar pixel grid),
        of the fraction of the star's light falling on each pixel, at each
        separation (:func:`core_mean_intensity_profile`). The values are per
        pixel, not per unit solid angle; divide by the squared pixel scale in
        lambda/D for an intensity per (lambda/D)^2.
    """
    available_diams = coro.stellar_intens.diams
    if stellar_diam_values is None:
        stellar_diam_values = available_diams
    else:
        for diam in stellar_diam_values:
            if diam not in available_diams:
                raise ValueError(
                    f"Requested stellar diameter {diam}"
                    f" not found in available diameters"
                )

    shape = coro.stellar_intens.psfs.shape[-2:]
    radii = profile_radii(
        shape, (coro.stellar_intens.grid.center_y, coro.stellar_intens.grid.center_x)
    )
    separations = (
        radii
        * getattr(coro, "stellar_pixel_scale_arcsec", coro.pixel_scale_arcsec).value
    )
    intensities = {
        diam: core_mean_intensity_profile(coro, diam, interpolation=interpolation)
        for diam in stellar_diam_values
    }
    return separations, intensities


def compute_all_performance_curves(coro, rule=None, **kwargs):
    """Build/select one record through Coronagraph's cache-aware boundary."""
    return coro.compute_all_performance_curves(rule=rule, **kwargs)


# ---------------------------------------------------------------------------
# Aperture records and common photometric integration
# ---------------------------------------------------------------------------


def _validate_photometry_image(image):
    from hwoutils.grids import Image

    if not isinstance(image, Image) or image.data.ndim != 2:
        raise ValueError("photometry requires a two-dimensional Image")
    if image.grid.unit != "lod" or image.measure != "per_pixel":
        raise ValueError("photometry requires per_pixel flux fractions on a lod grid")
    # The retained spectral kernels require positive, axis-aligned square pixels.
    _ = image.grid.step


@jax.jit
def _connected_core(weights, seed_index):
    """Eight-connected positive-weight component containing the source-local peak."""
    support = weights > 0
    seed = jnp.zeros_like(support).at[seed_index[0], seed_index[1]].set(True)

    def grow(state):
        old, _ = state
        padded = jnp.pad(old, 1)
        neighbors = jnp.zeros_like(old)
        for row in range(3):
            for col in range(3):
                neighbors = (
                    neighbors
                    | padded[row : row + old.shape[0], col : col + old.shape[1]]
                )
        new = support & neighbors
        return new, jnp.any(new != old)

    component, _ = jax.lax.while_loop(lambda state: state[1], grow, (seed, True))
    return jnp.where(component, weights, 0)


def _rule_samples(rule, psf, source_position, oversample=CORE_OVERSAMPLE):
    """Shared quadrature: per-native-pixel samples, weights, window origin, factor."""
    from .apertures import Circular, HalfMaximum, Matched

    _validate_photometry_image(psf)
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    grid = psf.grid
    pixel = grid.unit_to_pixel(*source_position)
    nominal = (pixel.y, pixel.x)
    data = jnp.asarray(psf.data, dtype=jnp.float64)
    if isinstance(rule, Circular):
        half_width = (
            int(np.ceil((rule.radius_lod + PEAK_SEARCH_RADIUS_LOD) / grid.step)) + 1
        )
        fine, weights, sy, sx = circular_aperture(
            data,
            rule.radius_lod / grid.step,
            grid.step,
            oversample=oversample,
            half_width_pix=half_width,
            nominal_center=nominal,
            fixed_center=None
            if rule.centre == "peak" and bool(_nominal_in_image(nominal, grid.shape))
            else nominal,
            return_samples=True,
        )
    elif isinstance(rule, HalfMaximum):
        half_width = core_window_half_width_pix(grid.step)
        if not bool(_nominal_in_image(nominal, grid.shape)):
            half_width = max(grid.shape)
        fine, weights, sy, sx, _ = _truncation_samples(
            data,
            rule.fraction,
            grid.step,
            oversample,
            half_width,
            nominal,
        )
        center = _window_peak(
            fine,
            sy,
            sx,
            oversample,
            nominal,
            PEAK_SEARCH_RADIUS_LOD / grid.step,
            grid.shape,
        )
        seed = jnp.rint((center - jnp.array([sy, sx]) + 0.5) * oversample - 0.5)
        weights = _connected_core(weights, seed.astype(jnp.int32))
        weights = jnp.where(jnp.max(data) > 0, weights, 0)
    elif isinstance(rule, Matched):
        return data, rule.weights(psf, source_position), 0, 0, 1
    else:
        raise TypeError("rule must be Circular, HalfMaximum or Matched")
    return fine, weights, sy, sx, oversample


def _rule_weights(rule, psf, source_position, oversample):
    """Aggregate the common subpixel coverage to native fractional pixel weights."""
    with jax.enable_x64():
        _, weights, sy, sx, factor = _rule_samples(
            rule, psf, source_position, oversample
        )
        ny, nx = weights.shape[0] // factor, weights.shape[1] // factor
        # Coverage is a fraction, so convert integrated subpixel weights to a mean.
        native, _ = downsample_psf(weights, 1.0, (ny, nx))
        native = native / factor**2
        return jax.lax.dynamic_update_slice(jnp.zeros(psf.grid.shape), native, (sy, sx))


def _stellar_samples(stellar, planet_grid, sy, sx, shape, factor):
    """Band-limited stellar per-pixel fractions in the planet's pixel units."""
    sg = stellar.grid
    sf, pf = sg.frame, planet_grid.frame
    ratio = (
        planet_grid.step
        * pf.lod_to_arcsec(pf.reference_wavelength_nm)
        / (sg.step * sf.lod_to_arcsec(sf.reference_wavelength_nm))
    )
    rows = (
        sg.center_y
        + (sy + (jnp.arange(shape[0]) + 0.5) / factor - 0.5 - planet_grid.center_y)
        * ratio
    )
    cols = (
        sg.center_x
        + (sx + (jnp.arange(shape[1]) + 0.5) / factor - 0.5 - planet_grid.center_x)
        * ratio
    )
    spectrum = jnp.fft.fft2(jnp.asarray(stellar.data, dtype=jnp.float64))
    by = (
        jnp.exp(2j * jnp.pi * rows[:, None] * jnp.fft.fftfreq(sg.shape[0]))
        / sg.shape[0]
    )
    bx = (
        jnp.exp(2j * jnp.pi * cols[:, None] * jnp.fft.fftfreq(sg.shape[1]))
        / sg.shape[1]
    )
    inside = (
        (rows[:, None] >= -0.5)
        & (rows[:, None] < sg.shape[0] - 0.5)
        & (cols[None, :] >= -0.5)
        & (cols[None, :] < sg.shape[1] - 0.5)
    )
    return jnp.where(inside, jnp.real(by @ spectrum @ bx.T) * ratio**2, 0)


def leakage_weights(offaxis, stellar, record, *, native_offaxis=None):
    """Prepare linear native-image operators for a performance record.

    Returns aperture weights and, for ring averaging, density weights, each
    with shape (separation, stellar_y, stellar_x). All aperture construction
    and image-coordinate transforms happen here, before runtime queries.
    For retained native records, native_offaxis supplies shipped samples and
    their nearest-to-zero y coordinate, just as the record measurement does.
    Explicit synthesized axes use the offaxis callable at (separation, 0).
    """
    from dataclasses import replace

    with jax.enable_x64():
        aperture = []
        native = {}
        if native_offaxis is not None:
            y_offsets = np.asarray(native_offaxis.y_offsets)
            iy = int(np.argmin(np.abs(y_offsets)))
            x_offsets = np.asarray(
                native_offaxis.x_offsets, dtype=record.separation_lod.dtype
            )
            for ix, x in enumerate(x_offsets):
                native[float(x)] = (ix, iy, float(y_offsets[iy]))
        for separation in np.asarray(record.separation_lod):
            position = (float(separation), 0.0)
            if float(separation) in native:
                ix, iy, y = native[float(separation)]
                psf = native_offaxis.get_psf_by_offset_idx(ix, iy)
                position = (float(separation), y)
            else:
                psf = offaxis(float(separation))
            weights = _rule_weights(record.rule, psf, position, CORE_OVERSAMPLE)
            if stellar.grid == psf.grid:
                aperture.append(weights)
            else:

                def integrate(data, psf=psf, weights=weights):
                    samples = _stellar_samples(
                        replace(stellar, data=data), psf.grid, 0, 0, psf.grid.shape, 1
                    )
                    return jnp.sum(samples * weights)

                aperture.append(jax.grad(integrate)(jnp.zeros(stellar.grid.shape)))
        aperture = jnp.stack(aperture)
        ring = None
        if record.leakage_rule == "azimuthal_ring_mean":
            sg, pf = stellar.grid, record.frame
            ratio = pf.lod_to_arcsec(
                pf.reference_wavelength_nm
            ) / sg.frame.lod_to_arcsec(sg.frame.reference_wavelength_nm)
            radii = jnp.asarray(record.separation_lod) * ratio / sg.step

            def profile(data):
                return (
                    azimuthal_mean_profile(data, (sg.center_y, sg.center_x), radii)
                    / sg.pixel_area
                    * ratio**2
                )

            ring = jax.jacrev(profile)(jnp.zeros(sg.shape))
    return aperture, ring


class PerformanceCurves(eqx.Module):
    """Frozen photometry on one separation axis; all fractions are absolute.

    ``separation_lod``: nonnegative increasing sample offsets, shape (S,).
    ``frame``: wavelength and circumscribed/inscribed diameter defining lambda/D.
    ``rule``: the static, hashable aperture or matched weighting rule.
    ``throughput``: weighted fraction of unocculted planet flux, tau.
    ``core_area_lod2``: geometric area or Matched noise-equivalent area, A.
    ``leakage_fraction``: stellar per-pixel fractions integrated with the weights.
    ``leakage_density_lod2``: selected mean, ell/A for aperture_mean; the radial
    azimuthal mean per unit area for azimuthal_ring_mean. These differ in general.
    ``raw_contrast``: leakage_density_lod2 * A / tau, with 1e20 when tau <= 0.
    ``iwa_lod``: first half-maximum-throughput crossing of PCHIP; if no crossing
    exists, the first positive-throughput separation (NaN for an all-zero curve).
    ``data_extent_lod``: minimum array half-width, not an outer working angle.
    ``leakage_rule``: aperture_mean or azimuthal_ring_mean, identifying the density.
    ``stellar_diameter``: optional provenance, a diameter in the stellar lod frame.
    ``interpolation``: immutable quantity-to-space mapping. PCHIP within the
    sampled interval and linear end-segment tails in that same space. Exact
    zeros use a 1e-300 floor for log interpolation and remain zero at their knots.

    Host NumPy inputs retain float64. JAX inputs use the same stored polynomial
    coefficients and evaluator, and support jit/vmap/grad. An Equinox consumer
    can convert the array leaves to its active JAX dtype once at construction.
    """

    separation_lod: np.ndarray
    frame: Frame = eqx.field(static=True)
    rule: ApertureRule = eqx.field(static=True)
    throughput: np.ndarray
    core_area_lod2: np.ndarray
    leakage_fraction: np.ndarray
    leakage_density_lod2: np.ndarray
    raw_contrast: np.ndarray
    iwa_lod: float
    data_extent_lod: float = eqx.field(static=True)
    leakage_rule: str = eqx.field(static=True)
    stellar_diameter: float | None = eqx.field(static=True)
    _coefficients: dict
    _tails: dict
    _extra: dict

    interpolation: ClassVar = MappingProxyType(
        {
            "throughput": "linear",
            "core_area_lod2": "linear",
            "leakage_fraction": "log",
            "leakage_density_lod2": "log",
            "raw_contrast": "log",
            "occulter_transmission": "linear",
            "core_mean_intensity": "linear",
        }
    )

    def __init__(
        self,
        separation_lod,
        frame,
        rule,
        throughput,
        core_area_lod2,
        leakage_fraction,
        leakage_density_lod2,
        raw_contrast,
        iwa_lod,
        data_extent_lod,
        leakage_rule="aperture_mean",
        stellar_diameter=None,
        *,
        extra=None,
        _coefficients=None,
        _tails=None,
        _extra=None,
    ):
        """Validate measured arrays and prepare one set of PCHIP coefficients."""
        from .apertures import Circular, HalfMaximum, Matched

        if not isinstance(rule, (Circular, HalfMaximum, Matched)):
            raise TypeError("rule must be Circular, HalfMaximum or Matched")
        if leakage_rule not in ("aperture_mean", "azimuthal_ring_mean"):
            raise ValueError("unknown leakage_rule")
        x = np.array(separation_lod, dtype=float, copy=True)
        if (
            x.ndim != 1
            or len(x) < 1
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)
            or x[0] < 0
        ):
            raise ValueError(
                "separation_lod must contain increasing nonnegative finite values"
            )
        x.flags.writeable = False
        self.separation_lod = x
        self.frame, self.rule = frame, rule
        self.iwa_lod, self.data_extent_lod = float(iwa_lod), float(data_extent_lod)
        self.leakage_rule, self.stellar_diameter = leakage_rule, stellar_diameter
        values = dict(
            throughput=throughput,
            core_area_lod2=core_area_lod2,
            leakage_fraction=leakage_fraction,
            leakage_density_lod2=leakage_density_lod2,
            raw_contrast=raw_contrast,
        )
        self._extra = {}
        self._coefficients, self._tails = {}, {}
        for name, value in (
            values | (extra if extra is not None else (_extra or {}))
        ).items():
            a = np.array(value, dtype=float, copy=True)
            if a.shape != x.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be finite and have shape {x.shape}")
            a.flags.writeable = False
            if name in values:
                setattr(self, name, a)
            else:
                self._extra[name] = a
            y = (
                np.log10(np.maximum(a, 1e-300))
                if self.interpolation[name] == "log"
                else a
            )
            if len(x) == 1:
                self._coefficients[name] = np.array([[0.0], [0.0], [0.0], [y[0]]])
                self._tails[name] = np.array([y[0], y[0], 0.0, 0.0])
                continue
            self._coefficients[name] = PchipInterpolator(x, y).c
            self._tails[name] = np.array(
                [
                    y[0],
                    y[-1],
                    (y[1] - y[0]) / (x[1] - x[0]),
                    (y[-1] - y[-2]) / (x[-1] - x[-2]),
                ]
            )

    def evaluate_samples(self, name, separation_lod, values):
        """Evaluate new knot values with this record's interpolation and tails.

        Values may depend on traced inputs. Aperture geometry stays prepared;
        only the scalar PCHIP coefficients are rebuilt.
        """
        x = jnp.asarray(self.separation_lod)
        values = jnp.asarray(values)
        if len(x) == 1:
            return jnp.full_like(
                jnp.asarray(separation_lod), values[0], dtype=values.dtype
            )
        y = (
            jnp.where(
                values > 0,
                jnp.maximum(jnp.log10(jnp.where(values > 0, values, 1)), -300),
                -300,
            )
            if self.interpolation[name] == "log"
            else values
        )
        coefficients = interpax.PchipInterpolator(x, y, check=False).c
        tails = jnp.stack(
            (
                y[0],
                y[-1],
                (y[1] - y[0]) / (x[1] - x[0]),
                (y[-1] - y[-2]) / (x[-1] - x[-2]),
            )
        )
        record = eqx.tree_at(
            lambda r: (getattr(r, name), r._coefficients[name], r._tails[name]),
            self,
            (values, coefficients, tails),
        )
        return record.evaluate(name, jnp.asarray(separation_lod))

    def evaluate(self, name, separation_lod):
        """Evaluate one named quantity using its declared interpolation space."""
        if name not in self._coefficients:
            raise ValueError(f"quantity {name!r} is not present in this record")
        xp = (
            jnp
            if isinstance(separation_lod, (jax.Array, jax.core.Tracer))
            or isinstance(self._coefficients[name], jax.core.Tracer)
            else np
        )
        q = xp.asarray(separation_lod)
        x = xp.asarray(self.separation_lod)
        if len(x) == 1:
            value = self._extra[name] if name in self._extra else getattr(self, name)
            return xp.full_like(q, xp.asarray(value)[0], dtype=x.dtype)
        c = xp.asarray(self._coefficients[name])
        t = xp.asarray(self._tails[name])
        i = xp.clip(xp.searchsorted(x, q, side="right") - 1, 0, len(x) - 2)
        dx = q - x[i]
        value = ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]
        value = xp.where(
            q < x[0],
            t[0] + (q - x[0]) * t[2],
            xp.where(q > x[-1], t[1] + (q - x[-1]) * t[3], value),
        )
        if self.interpolation[name] == "log":
            value = xp.power(10.0, value)
            values = xp.asarray(getattr(self, name))
            k = xp.clip(xp.searchsorted(x, q), 0, len(x) - 1)
            exact = xp.where(q == x[k], values[k], value)
            # Restore measured knot values without erasing the PCHIP derivative.
            if xp is jnp:
                correction = xp.where(q == x[k], values[k] - value, 0)
                value = value + jax.lax.stop_gradient(correction)
            else:
                value = exact
        return value


def _iwa(separation, throughput):
    positive = throughput > 0
    if not np.any(positive):
        return np.nan
    if len(separation) == 1:
        return float(separation[0])
    half = np.max(throughput) / 2
    f = PchipInterpolator(separation, throughput)
    for left, right, yl, yr in zip(
        separation[:-1], separation[1:], throughput[:-1], throughput[1:], strict=True
    ):
        if yl == half:
            return float(left)
        if (yl - half) * (yr - half) < 0:
            return root_scalar(lambda s: f(s) - half, bracket=(left, right)).root
    return float(separation[np.argmax(positive)])


def performance_curves(
    offaxis,
    stellar,
    rule,
    *,
    separations_lod=None,
    leakage_rule="aperture_mean",
    stellar_diameter=None,
) -> PerformanceCurves:
    """Tabulate one aperture rule on one axis.

    Minimal protocol: ``offaxis(separation_lod: float) -> Image`` on the +x
    axis. Supply ``separations_lod`` or an object attribute of that name.
    Objects with ``create_psf(x_lod, y_lod) -> Image`` are also accepted;
    Coronagraph implements this interface. Its default axis uses its shipped
    nonnegative x-offset PSFs at the nearest available y to zero. Circular
    rules omit native offsets outside the image; explicit axes retain clipped
    apertures (possibly empty). Explicit
    separations synthesize PSFs at (s, 0). The returned images must share one
    positive axis-aligned square lod grid, carrying per-pixel source fractions.
    Stellar is a per-pixel stellar-flux Image, potentially on a distinct grid
    and frame; its samples are mapped through the frames' angular scales.
    Leakage is the native per-pixel stellar map times the rule's fractional
    native weights; distinct grids use band-limited stellar samples at the
    planet pixel centers, zero outside the stellar footprint. Planet throughput
    retains subpixel band-limited integration for encircled-energy accuracy.
    Circular's ``centre`` chooses the supplied source position (default) or
    its refined local peak; every field, including IWA, uses that same rule.

    ``stellar_diameter`` records the provenance of the already evaluated map;
    it does not broaden or reevaluate that map. All integration uses the
    existing band-limited aperture machinery at CORE_OVERSAMPLE subpixels.
    """
    return _performance_curves(
        offaxis,
        stellar,
        rule,
        separations_lod=separations_lod,
        leakage_rule=leakage_rule,
        stellar_diameter=stellar_diameter,
    )


def _performance_curves(
    offaxis,
    stellar,
    rule,
    *,
    separations_lod=None,
    leakage_rule="aperture_mean",
    stellar_diameter=None,
    oversample=CORE_OVERSAMPLE,
):
    from hwoutils.grids import Image

    from .apertures import Matched

    _validate_photometry_image(stellar)
    if leakage_rule not in ("aperture_mean", "azimuthal_ring_mean"):
        raise ValueError("unknown leakage_rule")
    if separations_lod is None and hasattr(offaxis, "offax"):
        positions = list(_iter_xaxis_positions(offaxis))
        if isinstance(rule, Circular):
            # A source outside the detector has no local geometric aperture.
            # Do not treat a global leakage peak as its photometric center.
            positions = [p for p in positions if _point_source_in_image(p)]
        separations = np.array([p.separation for p in positions])
        images = [
            Image(
                p.psf,
                offaxis.grid,
                (),
                "per_pixel",
                "stellar_flux_fraction",
                reconstruction="band_limited",
            )
            for p in positions
        ]
        locations = [
            (p.px - offaxis.grid.center_x, p.py - offaxis.grid.center_y)
            for p in positions
        ]
        locations = [
            (x * offaxis.grid.step, y * offaxis.grid.step) for x, y in locations
        ]
    else:
        if separations_lod is None:
            separations_lod = getattr(offaxis, "separation_lod", None)
        if separations_lod is None:
            raise ValueError("separations_lod is required for a callable source")
        separations = np.asarray(separations_lod, dtype=float)
        if (
            separations.ndim != 1
            or len(separations) < 1
            or np.any(np.diff(separations) <= 0)
            or np.any(separations < 0)
            or not np.all(np.isfinite(separations))
        ):
            raise ValueError(
                "separations_lod must contain increasing nonnegative finite values"
            )
        get = (
            (lambda s: offaxis.create_psf(float(s), 0.0))
            if hasattr(offaxis, "create_psf")
            else offaxis
        )
        images = [get(float(s)) for s in separations]
        locations = [(float(s), 0.0) for s in separations]
    if not images:
        raise ValueError("no off-axis source positions lie inside the image")
    grid = images[0].grid
    tau, area, leakage = [], [], []
    with jax.enable_x64():
        for psf, position in zip(images, locations, strict=True):
            if psf.grid != grid:
                raise ValueError("off-axis Images must share one grid")
            fine, weights, sy, sx, factor = _rule_samples(
                rule, psf, position, oversample
            )
            ny, nx = weights.shape[0] // factor, weights.shape[1] // factor
            # Coverage is a fraction, not the integrated weight of a detector bin.
            native_weights, _ = downsample_psf(weights, 1.0, (ny, nx))
            native_weights = native_weights / factor**2
            if stellar.grid == grid:
                star = jax.lax.dynamic_slice(
                    jnp.asarray(stellar.data, dtype=jnp.float64), (sy, sx), (ny, nx)
                )
            else:
                star = _stellar_samples(stellar, grid, sy, sx, (ny, nx), 1)
            tau.append(float(jnp.sum(fine * weights) / factor**2))
            leakage.append(float(jnp.sum(star * native_weights)))
            if isinstance(rule, Matched):
                w2 = float(jnp.sum(weights**2))
                area.append(
                    float(jnp.sum(weights)) ** 2 / w2 * grid.pixel_area
                    if w2 > 0
                    else 0.0
                )
            else:
                area.append(float(jnp.sum(weights)) * grid.pixel_area / factor**2)
        tau, area, leakage = map(np.asarray, (tau, area, leakage))
        density = np.divide(leakage, area, out=np.zeros_like(leakage), where=area > 0)
        if leakage_rule == "azimuthal_ring_mean":
            sf, pf = stellar.grid.frame, grid.frame
            angular_ratio = pf.lod_to_arcsec(
                pf.reference_wavelength_nm
            ) / sf.lod_to_arcsec(sf.reference_wavelength_nm)
            density = (
                np.asarray(
                    azimuthal_mean_profile(
                        jnp.asarray(stellar.data, dtype=jnp.float64),
                        (stellar.grid.center_y, stellar.grid.center_x),
                        jnp.asarray(separations * angular_ratio / stellar.grid.step),
                    )
                )
                / stellar.grid.pixel_area
                * angular_ratio**2
            )
    contrast = np.divide(
        density * area, tau, out=np.full_like(tau, 1e20), where=tau > 0
    )
    return PerformanceCurves(
        separations,
        grid.frame,
        rule,
        tau,
        area,
        leakage,
        density,
        contrast,
        _iwa(separations, tau),
        min(grid.shape) * grid.step / 2,
        leakage_rule,
        stellar_diameter,
    )


def stellar_intensity_curves(backend, stellar_diam_lod=0.0, *, interpolation="keys"):
    """Return a radial per-stellar-pixel intensity record from an Eqx backend.

    This measurement is the ring mean of the stellar map, independent of an
    aperture rule's local leakage density. A traced diameter recomputes the
    profile from the interpolated map. A single supplied profile has no
    diameter dependence; otherwise out-of-range diameters yield NaN.
    Evaluate ``core_mean_intensity`` on the returned PerformanceCurves.
    The spatial reader is static under JIT; use ``eqx.filter_jit``.
    """
    if interpolation not in ("keys", "quintic"):
        raise ValueError("interpolation must be 'keys' or 'quintic'")
    grid = backend.stellar_grid
    radii = profile_radii(grid.shape, (grid.center_y, grid.center_x))
    nodes = radii * grid.step
    zeros = np.zeros_like(nodes)
    base = PerformanceCurves(
        nodes,
        grid.frame,
        DEFAULT_APERTURE_RULE,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        0.0,
        min(grid.shape) * grid.step / 2,
        extra={"core_mean_intensity": zeros},
    )
    knots = backend._stellar_ln_interp.x
    diameter = jnp.asarray(stellar_diam_lod)
    valid = (diameter >= 0) & (diameter <= knots[-1])
    image = backend.stellar_intens(jnp.clip(diameter, 0.0, knots[-1])).data
    profile = azimuthal_mean_profile(
        image,
        (grid.center_y, grid.center_x),
        jnp.asarray(radii),
        interpolation=interpolation,
    )
    coefficients = interpax.PchipInterpolator(
        jnp.asarray(nodes), profile, extrapolate=False, check=False
    ).c
    tails = jnp.stack(
        (
            profile[0],
            profile[-1],
            (profile[1] - profile[0]) / (nodes[1] - nodes[0]),
            (profile[-1] - profile[-2]) / (nodes[-1] - nodes[-2]),
        )
    )
    values = tuple(
        jnp.where(valid, value, jnp.nan) for value in (profile, coefficients, tails)
    )
    return eqx.tree_at(
        lambda record: (
            record._extra["core_mean_intensity"],
            record._coefficients["core_mean_intensity"],
            record._tails["core_mean_intensity"],
        ),
        base,
        values,
    )
