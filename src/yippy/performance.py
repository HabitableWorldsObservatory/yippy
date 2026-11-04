"""Coronagraph performance metric computation.

This module provides standalone functions for computing coronagraph performance
curves: throughput, raw contrast, core area, occulter transmission, and core
mean intensity. Each metric has its own compute function for clarity, plus a
``compute_all_performance_curves`` orchestrator.

All functions operate on a Coronagraph instance passed as the first argument,
keeping the Coronagraph class itself slim while preserving backward-compatible
delegation methods on it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import GAUSSIAN_FWHM_FACTOR
from hwoutils.map_coordinates import map_coordinates
from hwoutils.radial import radial_profile
from hwoutils.transforms import resample_flux
from lod_unit import lod
from scipy.interpolate import PchipInterpolator, make_interp_spline
from scipy.optimize import root_scalar

from .logger import logger
from .util import (
    convert_to_pix,
    load_core_truncation_from_fits,
    load_coro_performance_from_fits,
    save_coro_performance_to_fits,
)

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
        planet_psf = coro.offax.get_psf_by_offset_idx(i, y_idx)
        if planet_psf is None:
            continue
        px = convert_to_pix(
            x_lod_val, coro.offax.center_x, coro.pixel_scale_arcsec
        ).value
        py = convert_to_pix(
            y_lod_val, coro.offax.center_y, coro.pixel_scale_arcsec
        ).value
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


def _compute_iwa_owa(
    coro: Coronagraph, sep: np.ndarray, throughput: np.ndarray
) -> None:
    """Compute IWA and OWA from throughput curve, storing on *coro*."""
    valid_mask = throughput > 0
    half_max_throughput = max(throughput[valid_mask]) / 2

    def iwa_func(x):
        return coro.throughput_interp(x) - half_max_throughput

    iwa_bracket = None
    for i in range(1, len(sep)):
        if iwa_func(sep[i - 1]) * iwa_func(sep[i]) < 0:
            iwa_bracket = [sep[i - 1], sep[i]]
            break

    if iwa_bracket is not None:
        coro.IWA = root_scalar(iwa_func, bracket=iwa_bracket).root * lod
    else:
        first_valid_idx = np.argmax(valid_mask)
        coro.IWA = sep[first_valid_idx] * lod
        logger.warning(f"Could not find IWA bracket, using first valid sep: {coro.IWA}")

    if hasattr(coro.offax, "max_offset_in_image"):
        coro.OWA = coro.offax.max_offset_in_image
        logger.info(
            f"OWA set to max_offset_in_image: {coro.OWA.to(lod).value:.2f} lam/D"
        )
    else:
        coro.OWA = np.max(sep) * lod
        logger.warning(
            "max_offset_in_image not available, using maximum separation as OWA"
        )


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------


def compute_radial_average(
    image: np.ndarray,
    pixel_scale_value: float,
    center: tuple[float, float] | None = None,
    nbins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial average of a 2D image.

    Thin wrapper around :func:`hwoutils.radial.radial_profile` for
    backward compatibility.

    Args:
        image: 2D numpy array.
        pixel_scale_value: Pixel scale in lambda/D per pixel (float).
        center: ``(cy, cx)`` pixel coordinates of center. Defaults to
            image center.
        nbins: Number of radial bins. Defaults to ``floor(max_dim / 2)``.

    Returns:
        ``(separations_lod, radial_profile)`` - 1-D arrays.
    """
    kwargs = {}
    if center is not None:
        # Incoming convention is [x, y]; radial_profile expects (cy, cx)
        if isinstance(center, list):
            kwargs["center"] = (center[1], center[0])
        else:
            kwargs["center"] = center
    if nbins is not None:
        kwargs["nbins"] = nbins
    seps, prof = radial_profile(
        jnp.asarray(np.asarray(image, dtype=np.float64)),
        pixel_scale_arcsec=pixel_scale_value,
        **kwargs,
    )
    return np.asarray(seps), np.asarray(prof)


def plot_performance_curve(
    x, y, title, xlabel, ylabel, marker="o-", log_scale=False, ms=4
):
    """Plot a single performance curve."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y, marker, ms=ms)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_throughput_curve(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    oversample: int = CORE_OVERSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coronagraph throughput vs separation.

    Throughput is the fraction of the total flux (planet PSF normalized to 1)
    that lands inside a photometric aperture at each off-axis position.
    The circle is centered on the interpolated peak within
    :data:`PEAK_SEARCH_RADIUS_LOD` of the nominal source. Boundary subpixels
    carry exact circle-square overlap fractions (:func:`circular_aperture`).

    Args:
        coro: Coronagraph instance.
        aperture_radius_lod: Aperture radius in lam/D.
        oversample: Oversampling factor.

    Returns:
        ``(separations, throughputs)`` - sorted 1-D arrays.
    """
    separations, throughputs = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale_arcsec.value

    for pos in _iter_xaxis_positions(coro):
        if not _point_source_in_image(pos):
            separations.append(pos.separation)
            throughputs.append(0.0)
            continue
        flux, _, _ = _measure_circular_aperture(
            pos.psf,
            radius_pix,
            coro.pixel_scale_arcsec.value,
            oversample=oversample,
            half_width_pix=int(
                np.ceil(
                    radius_pix + PEAK_SEARCH_RADIUS_LOD / coro.pixel_scale_arcsec.value
                )
            )
            + 1,
            nominal_center=(pos.py, pos.px),
        )
        separations.append(pos.separation)
        throughputs.append(float(flux))

    return _collect_and_sort(separations, throughputs)


def compute_raw_contrast_curve(
    coro: Coronagraph,
    stellar_diam=0.0 * lod,
    aperture_radius_lod: float = 0.7,
    oversample: int = CORE_OVERSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw contrast curve vs separation.

    Contrast is the ratio of stellar flux to planet flux within the same
    circle centered on the interpolated planet peak near its nominal
    position. Both images use identical coordinates and fractional weights
    (:func:`circular_aperture`).

    Args:
        coro: Coronagraph instance.
        stellar_diam: Stellar angular diameter for the on-axis PSF.
        aperture_radius_lod: Aperture radius in lam/D.
        oversample: Oversampling factor.

    Returns:
        ``(separations, contrasts)`` - sorted 1-D arrays.
    """
    star_psf = coro.stellar_intens(stellar_diam)
    separations, contrasts = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale_arcsec.value

    for pos in _iter_xaxis_positions(coro):
        if not _point_source_in_image(pos):
            # Raw contrast is undefined outside the image. Use a large finite
            # sentinel so downstream spline construction stays well-defined;
            # the point lies beyond OWA (max_offset_in_image) so it will not
            # be queried in physically meaningful regimes.
            separations.append(pos.separation)
            contrasts.append(1e20)
            continue
        planet_flux, _, star_flux = _measure_circular_aperture(
            pos.psf,
            radius_pix,
            coro.pixel_scale_arcsec.value,
            oversample=oversample,
            half_width_pix=int(
                np.ceil(
                    radius_pix + PEAK_SEARCH_RADIUS_LOD / coro.pixel_scale_arcsec.value
                )
            )
            + 1,
            nominal_center=(pos.py, pos.px),
            star_psf=star_psf,
        )
        # A zero off-axis PSF (e.g. a source on the optical axis behind an
        # ideal mask) has undefined contrast; use the same finite sentinel as
        # out-of-image points so the interpolators stay well-defined.
        contrast_val = star_flux / planet_flux if planet_flux > 0 else 1e20
        separations.append(pos.separation)
        contrasts.append(float(contrast_val))

    return _collect_and_sort(separations, contrasts)


def compute_core_area_curve(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    fit_gaussian: bool = False,
    use_phot_aperture_as_min: bool = False,
    oversample: int = CORE_OVERSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute core area vs separation.

    If *fit_gaussian* is True a 2-D Gaussian is fitted to each PSF; otherwise
    exact circle-square overlaps give ``pi * aperture_radius_lod**2`` when
    the aperture fits inside the image (:func:`circular_aperture`). Apertures
    crossing an image edge include only their in-image area.

    Args:
        coro: Coronagraph instance.
        aperture_radius_lod: Aperture radius in lam/D.
        fit_gaussian: Whether to fit a 2-D Gaussian.
        use_phot_aperture_as_min: Use aperture area as a floor when fitting.
        oversample: Oversampling factor.

    Returns:
        ``(separations, core_areas)`` - sorted 1-D arrays, area in (lam/D)**2.
    """
    if fit_gaussian:
        from scipy.optimize import curve_fit

        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y):
            x, y = coords
            return (
                amplitude
                * np.exp(-(((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2) / 2)
            ).ravel()

    separations, core_areas = [], []

    for pos in _iter_xaxis_positions(coro):
        if fit_gaussian and not _point_source_in_image(pos):
            separations.append(pos.separation)
            core_areas.append(np.pi * aperture_radius_lod**2)
            continue
        if fit_gaussian:
            radius_pix = aperture_radius_lod / coro.pixel_scale_arcsec.value * 3
            margin = int(np.ceil(radius_pix * 3))
            ny, nx = pos.psf.shape
            px, py = int(pos.px), int(pos.py)
            xmin, xmax = max(0, px - margin), min(nx - 1, px + margin)
            ymin, ymax = max(0, py - margin), min(ny - 1, py + margin)
            subarr = np.asarray(pos.psf[ymin : ymax + 1, xmin : xmax + 1])
            sub_os = np.asarray(
                resample_flux(
                    jnp.asarray(np.asarray(subarr, dtype=np.float64)),
                    1.0,
                    1.0 / oversample,
                    tuple(n * oversample for n in subarr.shape),
                )
            )
            px_os, py_os = (pos.px - xmin) * oversample, (pos.py - ymin) * oversample
            y_grid, x_grid = np.indices(sub_os.shape)
            amplitude = sub_os.max()
            sigma_init = 1.0 * oversample
            popt, _ = curve_fit(
                gaussian_2d,
                (x_grid, y_grid),
                sub_os.ravel(),
                p0=[amplitude, px_os, py_os, sigma_init, sigma_init],
            )
            _, _, _, sigma_x, sigma_y = popt
            sigma_x_lod = sigma_x / oversample * coro.pixel_scale_arcsec.value
            sigma_y_lod = sigma_y / oversample * coro.pixel_scale_arcsec.value
            fwhm_x = GAUSSIAN_FWHM_FACTOR * sigma_x_lod
            fwhm_y = GAUSSIAN_FWHM_FACTOR * sigma_y_lod
            area = np.pi * fwhm_x * fwhm_y / 4
            if use_phot_aperture_as_min:
                area = max(area, np.pi * aperture_radius_lod**2)
        else:
            radius_pix = aperture_radius_lod / coro.pixel_scale_arcsec.value
            _, area, _ = _measure_circular_aperture(
                pos.psf,
                radius_pix,
                coro.pixel_scale_arcsec.value,
                oversample=oversample,
                half_width_pix=int(
                    np.ceil(
                        radius_pix
                        + PEAK_SEARCH_RADIUS_LOD / coro.pixel_scale_arcsec.value
                    )
                )
                + 1,
                nominal_center=(pos.py, pos.px),
            )
            area = float(area)

        separations.append(pos.separation)
        core_areas.append(area)

    return _collect_and_sort(separations, core_areas)


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
    """

    def primitive(x, y):
        a = jnp.minimum(jnp.abs(x), radius)
        b = jnp.minimum(jnp.abs(y), radius)
        split = jnp.minimum(a, jnp.sqrt(jnp.maximum(radius**2 - b**2, 0.0)))

        def integral(z):
            return 0.5 * (
                z * jnp.sqrt(jnp.maximum(radius**2 - z**2, 0.0))
                + radius**2 * jnp.arcsin(jnp.clip(z / radius, 0, 1))
            )

        return jnp.sign(x) * jnp.sign(y) * (b * split + integral(a) - integral(split))

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
            star_psf=None
            if star_psf is None
            else jnp.asarray(star_psf, dtype=jnp.float64),
        )
        return tuple(float(value) for value in values)


@functools.partial(jax.jit, static_argnames=("oversample", "half_width_pix"))
def circular_aperture(
    psf,
    radius_pix,
    pixel_scale_lod,
    oversample: int = CORE_OVERSAMPLE,
    half_width_pix: int = 12,
    star_psf=None,
    nominal_center=None,
    fixed_center=None,
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
    coordinates through the same weights as the planet.
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
    flux = jnp.sum(circle * fine) / oversample**2
    area = jnp.sum(circle) * (pixel_scale_lod / oversample) ** 2
    star_flux = 0.0
    if star_psf is not None:
        spectrum = jnp.fft.fft2(star_psf)
        basis_y = _band_limited_samples(
            spectrum, start_y, fine.shape[0] // oversample, oversample
        )
        basis_x = _band_limited_samples(
            spectrum.T, start_x, fine.shape[1] // oversample, oversample
        )
        star_fine = jnp.real(basis_y @ spectrum @ basis_x.T)
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
def truncation_core(
    psf,
    psf_trunc_ratio,
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

    The threshold is ``psf_trunc_ratio`` times the refined interpolated peak.
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
        psf_trunc_ratio: Fraction of the peak that bounds the core.
        pixel_scale_lod: Pixel scale in lambda/D per pixel.
        oversample: Subpixels per pixel side.
        half_width_pix: Half-width in pixels of the subpixel window, clipped
            to the image.
        nominal_center: Optional source (row, column) in original pixels.

    Returns:
        ``(throughput, area)``: the core throughput in the PSF's flux units
        and the core area in (lambda/D)^2.
    """
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
    threshold = psf_trunc_ratio * peak
    core = _level_set_weights(fine, grad_y, grad_x, threshold, oversample)
    core = jnp.where(
        jnp.any(fine > threshold), core, (fine == jnp.max(fine)).astype(fine.dtype)
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
    coro: Coronagraph,
    psf_trunc_ratio: float = 0.5,
    oversample: int | None = CORE_OVERSAMPLE,
    separation_step_lod: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core throughput and core area with a PSF-truncation-ratio aperture.

    The core is the region where the PSF exceeds ``psf_trunc_ratio * peak``;
    the throughput is the fraction of the source's light inside it (AYO's
    ``photap_frac``) and the area its solid angle in (lam/D)**2 (AYO's
    ``omega_lod``), both measured on ``oversample`` subpixels per pixel side
    (:func:`truncation_core`). By default only the package's own non-negative
    off-axis offsets are measured. The curves interpolate those measurements
    between offsets. Passing ``separation_step_lod=CORE_STEP_LOD`` opts into
    measuring synthesized PSFs on a quarter-lambda/D grid as well
    (:func:`core_separations`); their optical accuracy between shipped offsets
    cannot be established from the package alone.

    Args:
        coro: Coronagraph instance.
        psf_trunc_ratio: Fraction of PSF peak used as threshold.
        oversample: Subpixels per pixel side at which the band-limited
            interpolant is evaluated (:func:`truncation_core`); ``None``
            selects :data:`CORE_OVERSAMPLE`. This sets the resolution of the
            core boundary, as for circular apertures.
        separation_step_lod: Spacing of the separation grid in lambda/D.
            Defaults to ``None`` for the package's own off-axis offsets.
            Pass :data:`CORE_STEP_LOD` to opt into synthesized PSFs.

    Returns:
        ``(separations, throughputs, core_areas)`` - 1-D arrays sorted by
        separation, area in (lam/D)^2.
    """
    pix_lod = float(coro.pixel_scale_arcsec.value)
    half_width = core_window_half_width_pix(pix_lod)
    oversample = CORE_OVERSAMPLE if oversample is None else int(oversample)

    def cores_of(psfs, centers):
        throughputs, areas = jax.lax.map(
            lambda pair: truncation_core(
                pair[0],
                psf_trunc_ratio,
                pix_lod,
                oversample=oversample,
                half_width_pix=half_width,
                nominal_center=pair[1],
            ),
            (jnp.asarray(psfs, dtype=jnp.float64), jnp.asarray(centers)),
        )
        return np.column_stack([np.asarray(throughputs), np.asarray(areas)])

    separations, cores = [], []
    positions = list(_iter_xaxis_positions(coro))
    if separation_step_lod is not None:
        grid = core_separations(coro, separation_step_lod)
        y0 = _xaxis_y_offset(coro)
        for start in range(0, len(grid), _PSF_BATCH):
            chunk = grid[start : start + _PSF_BATCH]
            # A full batch every call, so the synthesis compiles once.
            padded = jnp.asarray(np.pad(chunk, (0, _PSF_BATCH - len(chunk)), "edge"))
            psfs = coro.offax.create_psfs(padded, jnp.full_like(padded, y0))
            separations.extend(chunk.tolist())
            centers = np.column_stack(
                [
                    np.full(
                        len(padded),
                        convert_to_pix(
                            y0 * lod, coro.offax.center_y, coro.pixel_scale_arcsec
                        ).value,
                    ),
                    convert_to_pix(
                        np.asarray(padded) * lod,
                        coro.offax.center_x,
                        coro.pixel_scale_arcsec,
                    ).value,
                ]
            )
            with jax.enable_x64():
                cores.append(cores_of(psfs, centers)[: len(chunk)])
        positions = [pos for pos in positions if pos.separation > grid[-1]]

    if positions:
        separations.extend(pos.separation for pos in positions)
        with jax.enable_x64():
            cores.append(
                cores_of(
                    np.stack([pos.psf for pos in positions]),
                    np.array([(pos.py, pos.px) for pos in positions]),
                )
            )

    cores = np.concatenate(cores)
    order = np.argsort(separations)
    return np.asarray(separations)[order], cores[order, 0], cores[order, 1]


def compute_truncation_throughput_curve(
    coro: Coronagraph,
    psf_trunc_ratio: float = 0.5,
    oversample: int | None = CORE_OVERSAMPLE,
    separation_step_lod: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Core throughput with a PSF-truncation-ratio aperture.

    The throughput half of :func:`compute_truncation_curves`, which documents
    the arguments.

    Returns:
        ``(separations, throughputs)`` - sorted 1-D arrays.
    """
    sep, throughput, _ = compute_truncation_curves(
        coro, psf_trunc_ratio, oversample, separation_step_lod
    )
    return sep, throughput


def compute_truncation_core_area_curve(
    coro: Coronagraph,
    psf_trunc_ratio: float = 0.5,
    oversample: int | None = CORE_OVERSAMPLE,
    separation_step_lod: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Core area with a PSF-truncation-ratio aperture.

    The area half of :func:`compute_truncation_curves`, which documents the
    arguments.

    Returns:
        ``(separations, core_areas)`` - sorted 1-D arrays, area in (lam/D)^2.
    """
    sep, _, area = compute_truncation_curves(
        coro, psf_trunc_ratio, oversample, separation_step_lod
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
    sky_trans_data = coro.sky_trans()
    return compute_radial_average(sky_trans_data, coro.pixel_scale_arcsec.value)


def stellar_axis_index(coro: Coronagraph) -> tuple[float, float]:
    """Pixel-center index ``(row, column)`` of the optical axis in the stellar maps.

    A package's ``XCENTER`` and ``YCENTER`` (or ``npix / 2`` when they are
    absent) count pixels from the array's outer corner. With integer indices
    labeling pixel centers the axis therefore sits half a pixel earlier: a
    256-pixel map with ``XCENTER = 128`` is point-symmetric about index 127.5,
    between the four central pixels.

    Args:
        coro: Coronagraph instance.

    Returns:
        ``(row, column)`` of the optical axis as 0-based pixel-center indices.
    """
    return (
        coro.stellar_intens.center_y - 0.5,
        coro.stellar_intens.center_x - 0.5,
    )


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


def azimuthal_mean_profile(image, center, radii_pix, n_theta=None):
    """Azimuthal mean of an image on circles about ``center``.

    Each circle is sampled at ``n_theta`` equally spaced positions along its
    in-image angular arcs with
    Keys cubic interpolation (:func:`hwoutils.map_coordinates.map_coordinates`)
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

    Returns:
        1-D JAX array, the mean of each circle's in-image samples (0 for a
        circle with none).
    """
    image = jnp.asarray(image, dtype=jnp.result_type(image, 1.0))
    ny, nx = image.shape
    if n_theta is None:
        n_theta = 2 * max(ny, nx)
    radii = jnp.asarray(radii_pix, dtype=image.dtype)[:, None]
    safe_radii = jnp.where(radii > 0, radii, 1.0)
    # Circle/edge intersections split the circumference into wholly inside
    # or outside arcs. Sampling arc length cannot miss a short corner arc.
    y_edges = (jnp.array([0, ny - 1]) - center[0]) / safe_radii
    x_edges = (jnp.array([0, nx - 1]) - center[1]) / safe_radii
    y_angles = jnp.arcsin(jnp.clip(y_edges, -1, 1))
    x_angles = jnp.arccos(jnp.clip(x_edges, -1, 1))
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
    values = map_coordinates(image, coordinates(theta), order=3, mode="nearest")
    # A tangent circle has zero arc length but can touch one or more corners.
    rows, cols = coordinates(edges)
    inside = in_image(rows, cols, 8 * jnp.finfo(image.dtype).eps * max(ny, nx))
    endpoints = map_coordinates(image, [rows, cols], order=3, mode="nearest")
    count = jnp.sum(inside, axis=1)
    tangent_mean = jnp.sum(jnp.where(inside, endpoints, 0.0), axis=1) / jnp.maximum(
        count, 1
    )
    return jnp.where(total_length[:, 0] > 0, jnp.mean(values, axis=1), tangent_mean)


def core_mean_intensity_profile(coro: Coronagraph, stellar_diam) -> np.ndarray:
    """Core mean intensity profile of the stellar map at ``stellar_diam``.

    The profile is the azimuthal mean of ``coro.stellar_intens(stellar_diam)``
    about the optical axis, on the radii of :func:`profile_radii`. Because it
    is computed from the map itself, the curve at any diameter, tabulated or
    not, is the azimuthal mean of the map an image simulator renders at that
    diameter.

    Args:
        coro: Coronagraph instance.
        stellar_diam: Stellar angular diameter (Quantity in lambda/D).

    Returns:
        1-D array, per pixel, on the radii of :func:`profile_radii`.
    """
    psf = np.asarray(coro.stellar_intens(stellar_diam), dtype=np.float64)
    center = stellar_axis_index(coro)
    radii = profile_radii(psf.shape, center)
    return np.asarray(azimuthal_mean_profile(jnp.asarray(psf), center, radii))


def compute_core_mean_intensity_curve(
    coro: Coronagraph,
    stellar_diam_values=None,
) -> tuple[np.ndarray, dict]:
    """Compute core mean intensity curves for different stellar diameters.

    Args:
        coro: Coronagraph instance.
        stellar_diam_values:
            List of stellar diameters.  ``None`` -> use all available.

    Returns:
        ``(separations_lod, intensities_dict)`` where *intensities_dict*
        maps each stellar diameter to its radial intensity profile: the
        azimuthal mean, about the optical axis (:func:`stellar_axis_index`),
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
    radii = profile_radii(shape, stellar_axis_index(coro))
    separations = radii * coro.pixel_scale_arcsec.value
    intensities = {
        diam: core_mean_intensity_profile(coro, diam) for diam in stellar_diam_values
    }
    return separations, intensities


# ---------------------------------------------------------------------------
# Interpolator builder (shared by compute_all_performance_curves and
# Coronagraph.set_psf_trunc_ratio)
# ---------------------------------------------------------------------------


class _PchipWithLinearTails:
    """Shape-preserving interior with the legacy end-segment extrapolation."""

    def __init__(self, x, y):
        """Keep the original nodes and values for the Equinox conversion."""
        self.pchip = PchipInterpolator(x, y, extrapolate=False)
        self.linear = make_interp_spline(x, y, k=1)

    def __call__(self, x):
        """Evaluate PCHIP inside the table and the linear segments outside."""
        x = np.asarray(x)
        outside = (x < self.pchip.x[0]) | (x > self.pchip.x[-1])
        return np.where(outside, self.linear(x), self.pchip(x))


def _build_curve_interp(x, y, interp_order):
    """Use PCHIP by default, retaining explicit B-spline orders unchanged."""
    if interp_order is None:
        return _PchipWithLinearTails(x, y)
    return make_interp_spline(x, y, k=interp_order)


def _build_perf_interps(
    coro, sep_t, throughput, sep, raw_contrast, sep_ca, core_area, interp_order
):
    """Assign throughput, contrast, and core area spline interpolators to *coro*."""
    coro.interp_order = interp_order
    coro.throughput_interp = _build_curve_interp(sep_t, throughput, interp_order)

    log_contrast = np.log10(np.abs(raw_contrast) + 1e-20)
    coro._log_contrast_interp = _build_curve_interp(sep, log_contrast, interp_order)
    coro.raw_contrast_interp = _build_curve_interp(sep, raw_contrast, interp_order)

    coro.core_area_interp = _build_curve_interp(sep_ca, core_area, interp_order)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_performance_curves(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    stellar_diam=None,
    fit_gaussian_for_core_area: bool = False,
    use_phot_aperture_as_min: bool = False,
    oversample: int | None = CORE_OVERSAMPLE,
    save_to_fits: bool = True,
    performance_file: str = "coro_perf.fits",
    load_from_file: str | None = None,
    cache_dir: Path | None = None,
    plot: bool = False,
    psf_trunc_ratio: float | None = None,
    interp_order: int | None = None,
    _validated_cache: bool = False,
    separation_step_lod: float | None = None,
) -> dict:
    """Compute (or load) all coronagraph performance curves.

    Stores interpolators on *coro*. ``interp_order=None`` (the default) uses
    shape-preserving PCHIP between nodes and linear end-segment extrapolation.
    An integer uses ``make_interp_spline(k=interp_order)`` unchanged, so
    ``interp_order=1`` restores the previous linear behavior.
    This is the main entrypoint called during ``Coronagraph.__init__``.  It
    delegates to the individual ``compute_*`` helpers above, builds spline
    interpolators, and computes IWA / OWA.

    ``load_from_file`` is an explicit import: the named table supplies the
    separation, throughput and raw contrast as-is, whatever settings produced
    it, and nothing is written. With a truncation ratio the core throughput
    and core area come from the file's core table as-is (and are computed
    when it has none); an explicit import in that mode logs a warning,
    because the file's truncation ratio is not checked. Automatic cache
    reuse in ``Coronagraph`` validates a table's identity metadata before
    passing it here (``_validated_cache=True``). A table written with
    ``save_to_fits`` records the identity of the settings that actually
    produced it (when *coro* provides ``_perf_identity``), so a table saved
    under non-default settings is never accepted as an automatic entry for
    different ones. The stored contrast is always unfloored; the floor is
    applied after loading.

    Circular fluxes and counted areas use the band-limited interpolant at
    ``oversample`` subpixels per pixel side (default :data:`CORE_OVERSAMPLE`).
    Truncation cores use :data:`CORE_OVERSAMPLE` and are stored in the table's
    core table at the package's own off-axis offsets.
    ``separation_step_lod=CORE_STEP_LOD`` opts into a grid of synthesized PSFs
    (:func:`compute_truncation_curves`); the default is ``None``.

    Returns a dict of all curve data for convenience. ``throughput`` and
    ``raw_contrast`` are on ``separations``; in truncation mode that
    throughput is the core curve read at those separations. The curve the
    throughput interpolator is built from is ``core_throughput`` on
    ``separations_core_throughput``, which are ``separations`` without a
    truncation ratio.
    """
    if stellar_diam is None:
        stellar_diam = coro.stellar_intens.diams[0]
    oversample = _resolve_oversample(coro, oversample)
    truncation = psf_trunc_ratio is not None

    # Resolve the directory for save/load operations
    io_dir = cache_dir if cache_dir is not None else coro.yip_path

    # ------------------------------------------------------------------
    # Throughput + contrast (+ truncation core): load from file or compute
    # ------------------------------------------------------------------
    loaded = False
    core = None
    if load_from_file is not None:
        logger.info(f"Loading throughput and contrast from {load_from_file}")
        try:
            sep, throughput, raw_contrast = load_coro_performance_from_fits(
                load_from_file, io_dir
            )
            if truncation:
                core = load_core_truncation_from_fits(load_from_file, io_dir)
            loaded = True
            logger.info(f"Successfully loaded performance data from {load_from_file}")
        except Exception as e:
            logger.warning(f"Error loading from {load_from_file}: {e}")
            logger.info("Computing throughput and contrast from scratch")
        if loaded and truncation and not _validated_cache:
            logger.warning(
                f"Explicit import of {load_from_file} with psf_trunc_ratio="
                f"{psf_trunc_ratio}: the core throughput and area are "
                + (
                    "taken from the file as-is; its truncation ratio is not checked"
                    if core is not None
                    else "not in the file and are computed"
                )
            )

    if truncation and core is None:
        logger.info(
            f"Computing core throughput and area (trunc ratio = {psf_trunc_ratio})..."
        )
        core = compute_truncation_curves(
            coro,
            psf_trunc_ratio=psf_trunc_ratio,
            separation_step_lod=separation_step_lod,
        )

    if not loaded:
        logger.info("Computing raw contrast curve...")
        sep, raw_contrast = compute_raw_contrast_curve(
            coro,
            stellar_diam=stellar_diam,
            aperture_radius_lod=aperture_radius_lod,
            oversample=oversample,
        )
        if truncation:
            throughput = np.interp(sep, core[0], core[1])
        else:
            logger.info("Computing throughput curve...")
            sep_t, throughput = compute_throughput_curve(
                coro, aperture_radius_lod=aperture_radius_lod, oversample=oversample
            )
            # Both iterate the same positions
            assert np.allclose(sep, sep_t), "Throughput and contrast separations differ"

        if save_to_fits:
            identity = None
            if hasattr(coro, "_perf_identity"):
                identity = coro._perf_identity(
                    aperture_radius_lod=aperture_radius_lod,
                    psf_trunc_ratio=psf_trunc_ratio,
                    oversample=oversample,
                    stellar_diam=stellar_diam,
                    separation_step_lod=separation_step_lod,
                )
            save_coro_performance_to_fits(
                sep,
                throughput,
                raw_contrast,
                performance_file,
                io_dir,
                identity=identity,
                core=core,
            )

    # Apply contrast floor
    if coro.contrast_floor is not None:
        raw_contrast = np.maximum(np.abs(raw_contrast), coro.contrast_floor)
        logger.info(f"Applied contrast floor of {coro.contrast_floor:.1e}")

    # ------------------------------------------------------------------
    # Core area (and, with a truncation ratio, core throughput)
    # ------------------------------------------------------------------
    if truncation:
        sep_t, throughput_t, core_area = core
        sep_ca = sep_t
    else:
        sep_t, throughput_t = sep, throughput
        logger.info("Computing core area curve...")
        sep_ca, core_area = compute_core_area_curve(
            coro,
            aperture_radius_lod=aperture_radius_lod,
            fit_gaussian=fit_gaussian_for_core_area,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            oversample=oversample,
        )

    _build_perf_interps(
        coro,
        sep_t,
        throughput_t,
        sep,
        raw_contrast,
        sep_ca,
        core_area,
        interp_order,
    )

    # ------------------------------------------------------------------
    # Occulter transmission
    # ------------------------------------------------------------------
    logger.info("Computing occulter transmission curve...")
    sep_occ_trans, occ_trans = compute_occ_trans_curve(coro)
    coro.occ_trans_interp = _build_curve_interp(sep_occ_trans, occ_trans, interp_order)

    # ------------------------------------------------------------------
    # Core mean intensity
    # ------------------------------------------------------------------
    logger.info("Computing core mean intensity curve...")
    sep_core_intensity, core_intensities = compute_core_mean_intensity_curve(
        coro, stellar_diam_values=None
    )
    coro.core_intensity_dict = core_intensities
    coro.core_intensity_separations = sep_core_intensity
    # Profiles at other diameters are computed from the interpolated map on
    # first use, so they always agree with it (Coronagraph.core_mean_intensity).
    coro._core_intensity_cache = {}

    # 1D spline for the default stellar diameter (backward compatibility
    # and fast path for the common single-diameter case)
    coro.core_intensity_interp = _build_curve_interp(
        sep_core_intensity,
        core_intensities[stellar_diam],
        interp_order,
    )

    # ------------------------------------------------------------------
    # IWA / OWA
    # ------------------------------------------------------------------
    _compute_iwa_owa(coro, sep_t, throughput_t)

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if plot:
        plot_performance_curve(
            sep_t,
            throughput_t,
            title=f"{coro.name} Throughput",
            xlabel="Separation [lam/D]",
            ylabel="Throughput",
            ms=6,
        )
        plot_performance_curve(
            sep,
            raw_contrast,
            title=f"{coro.name} Raw Contrast",
            xlabel="Separation [lam/D]",
            ylabel="Raw Contrast",
            log_scale=True,
        )
        suffix = (
            " (Gaussian fit)" if fit_gaussian_for_core_area else " (fixed aperture)"
        )
        plot_performance_curve(
            sep_ca,
            core_area,
            title=f"{coro.name} Core Area{suffix}",
            xlabel="Separation [lam/D]",
            ylabel="Core Area [(lam/D)**2]",
        )

    return {
        "separations": sep,
        "throughput": throughput,
        "separations_core_throughput": sep_t,
        "core_throughput": throughput_t,
        "raw_contrast": raw_contrast,
        "separations_occ_trans": sep_occ_trans,
        "occ_trans": occ_trans,
        "separations_core_area": sep_ca,
        "core_area": core_area,
        "separations_core_intensity": sep_core_intensity,
        "core_intensities": core_intensities,
        "IWA": coro.IWA,
        "OWA": coro.OWA,
    }
