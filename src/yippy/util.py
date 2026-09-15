"""Utility functions for the yippy package."""

import hashlib
import json
from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.units import Quantity
from hwoutils.transforms import resample_flux
from lod_unit import lod, lod_eq


def convert_to_lod(
    x: Quantity, center_pix=None, pixel_scale_arcsec=None, lam=None, D=None, dist=None
) -> Quantity:
    """Convert the x/y position to lambda/D.

    This function has the following assumptions on the x/y values provided:
        - If units are pixels, they follow the 00LL convention. As in the (0,0)
          point is the lower left corner of the image.
        - If the x/y values are in lambda/D, angular, or length units the
            (0,0) point is the center of the image, where the star is
            (hopefully) located.

    Args:
        x (astropy.units.Quantity):
            Position. Can be units of pixel, an angular unit (e.g. arcsec),
            or a length unit (e.g. AU)
        center_pix (astropy.units.Quantity):
            Center of the image in pixels (for the relevant axis)
        pixel_scale_arcsec (astropy.units.Quantity):
            Pixel scale of in
        lam (astropy.units.Quantity):
            Wavelength of the observation
        D (astropy.units.Quantity):
            Diameter of the telescope
        dist (astropy.units.Quantity):
            Distance to the system
    """
    if x.unit == "pixel":
        assert center_pix is not None, (
            "Center pixel must be provided to convert pixel to lod."
        )
        assert pixel_scale_arcsec is not None, (
            "Pixel scale must be provided to convert pixel to lod."
        )
        assert pixel_scale_arcsec.unit == (lod / u.pix), (
            f"Pixel scale must be in units of lod/pix, not {pixel_scale_arcsec.unit}."
        )

        x = x - center_pix
        x = x * pixel_scale_arcsec
        # Center the x position
    elif x.unit.physical_type == "angle":
        assert lam is not None, (
            f"Wavelength must be provided to convert {x.unit.physical_type} to lod."
        )
        assert D is not None, (
            f"Telescope diameter must be provided to convert {x.unit.physical_type}"
            f" to lod."
        )
        x = x.to(lod, lod_eq(lam, D))
    elif x.unit.physical_type == "length":
        # If the distance to the system is not provided, raise an error
        assert dist is not None, (
            f"Distance to system must be provided to convert {x.unit.physical_type}"
            f" to {lod}."
        )
        x_angular = np.arctan(x.to(u.m).value / dist.to(u.m).value) * u.rad
        x = x_angular.to(lod, lod_eq(lam, D))
    else:
        raise ValueError(f"No conversion implemented for {x.unit.physical_type}")
    return x


def convert_to_pix(
    x: Quantity, center_pix, pixel_scale_arcsec, lam=None, D=None, dist=None
) -> Quantity:
    """Convert the x/y position from lambda/D to pixel units.

    This function has the following assumptions on the x/y values provided:
        - If the desired output is in pixels, the (0,0) point is the lower left
          corner of the image.
        - If the x/y values are in lambda/D, angular, or length units, the
          (0,0) point is the center of the image, where the star is
          (hopefully) located.

    Args:
        x (astropy.units.Quantity):
            Position to convert. Should be in units of lambda/D, an angular unit
            (e.g., arcsec), or a length unit (e.g., AU).
        center_pix (astropy.units.Quantity, optional):
            Center of the image in pixels (for the relevant axis). Required if
            converting to pixel units.
        pixel_scale_arcsec (astropy.units.Quantity, optional):
            Pixel scale in units of lambda/D per pixel. Required if converting to
            pixel units.
        lam (astropy.units.Quantity, optional):
            Wavelength of the observation. Required if converting from angular or
            length units to pixel units.
        D (astropy.units.Quantity, optional):
            Diameter of the telescope. Required if converting from angular or
            length units to pixel units.
        dist (astropy.units.Quantity, optional):
            Distance to the system. Required if converting from length units to
            pixel units.

    Returns:
        astropy.units.Quantity:
            Position in pixel units.

    Raises:
        AssertionError:
            If required parameters for the conversion are not provided or have
            incorrect units.
        ValueError:
            If the input unit type is not supported for conversion.
    """
    if isinstance(x, float) or isinstance(x, np.floating):
        # Assume x is a float in lambda/D
        x_pixels = x * lod / pixel_scale_arcsec + center_pix
    elif x.unit == lod:
        # Center the x position
        x_pixels = x / pixel_scale_arcsec + center_pix

    elif x.unit.physical_type == "angle":
        # Conversion from angle to pixels
        assert lam is not None, (
            "Wavelength must be provided to convert angle to pixels."
        )
        assert D is not None, (
            "Telescope diameter must be provided to convert angle to pixels."
        )

        # Convert angle to lambda/D
        x_lod = x.to(u.rad, lod_eq(lam, D))
        # Now convert lambda/D to pixels
        x_pixels = x_lod / pixel_scale_arcsec + center_pix

    elif x.unit.physical_type == "length":
        # Conversion from length to pixels
        assert lam is not None, (
            "Wavelength must be provided to convert length to pixels."
        )
        assert D is not None, (
            "Telescope diameter must be provided to convert length to pixels."
        )
        assert dist is not None, (
            "Distance to system must be provided to convert length to pixels."
        )

        # Convert length to angle
        x_angle = np.arctan(x.to(u.m).value / dist.to(u.m).value) * u.rad
        # Convert angle to lambda/D
        x_lod = x_angle.to(lod, lod_eq(lam, D))
        # Now convert lambda/D to pixels
        x_pixels = x_lod / pixel_scale_arcsec + center_pix

    else:
        raise ValueError(f"No conversion implemented for {x.unit.physical_type}")

    return x_pixels


def _canonical_cache_value(value):
    """Normalize a cache identity value to a JSON-stable form.

    Real numbers of any Python or NumPy type become ``float.hex`` strings, so
    ``1``, ``1.0`` and ``np.float32(1.0)`` agree while values that differ in any
    bit of their double representation stay distinct. Booleans are kept apart
    from numbers.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value).hex()
    if isinstance(value, dict):
        return {str(k): _canonical_cache_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_cache_value(v) for v in value]
    raise TypeError(f"Unsupported cache identity value: {value!r}")


def cache_identity_json(identity: dict) -> str:
    """Deterministic serialization of a cache identity mapping."""
    return json.dumps(
        _canonical_cache_value(identity), sort_keys=True, separators=(",", ":")
    )


def cache_identity_digest(identity: dict) -> str:
    """SHA-256 hex digest of the canonical serialization of ``identity``."""
    return hashlib.sha256(cache_identity_json(identity).encode()).hexdigest()


def save_coro_performance_to_fits(
    sep: np.ndarray,
    throughput: np.ndarray,
    raw_contrast: np.ndarray,
    filename: str,
    outdir: Path,
    overwrite=True,
    identity: dict | None = None,
):
    """Save coronagraph performance (throughput, raw_contrast) to a FITS file.

    Args:
        sep: Separations in lambda/D.
        throughput: Throughput at each separation.
        raw_contrast: Unfloored raw contrast at each separation.
        filename: Output file name.
        outdir: Output directory.
        overwrite: Whether to overwrite an existing file.
        identity: Optional mapping describing the settings that produced the
            table. Its digest (``CACHEKEY``) and canonical JSON (``CACHEID``)
            are written to the primary header, which is what automatic cache
            reuse validates against.
    """
    sort_idx = np.argsort(sep)
    col_sep = fits.Column(name="separation_lamD", format="E", array=sep[sort_idx])
    col_thr = fits.Column(name="throughput", format="E", array=throughput[sort_idx])
    col_con = fits.Column(name="raw_contrast", format="E", array=raw_contrast[sort_idx])
    cols = fits.ColDefs([col_sep, col_thr, col_con])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = "CORO_PERFORMANCE"

    primary_hdu = fits.PrimaryHDU()
    if identity is not None:
        primary_hdu.header["CACHEKEY"] = cache_identity_digest(identity)
        primary_hdu.header["CACHEID"] = cache_identity_json(identity)
    hdul = fits.HDUList([primary_hdu, tbhdu])
    outpath = outdir / filename
    hdul.writeto(outpath, overwrite=overwrite)


def read_performance_cache_key(path: Path) -> str | None:
    """Return the ``CACHEKEY`` digest of a performance table, or None.

    Unreadable files and tables written without identity metadata return
    None, so a caller comparing against an expected digest treats them as a
    cache miss.
    """
    try:
        return fits.getheader(path, 0).get("CACHEKEY")
    except (OSError, ValueError):
        return None


def load_coro_performance_from_fits(filename: str, indir: Path):
    """Load (separation, throughput, raw_contrast) from a FITS file."""
    inpath = indir / filename
    with fits.open(inpath) as hdul:
        data = hdul["CORO_PERFORMANCE"].data
        sep = data["separation_lamD"]
        thr = data["throughput"]
        con = data["raw_contrast"]
    return sep, thr, con


def extract_and_oversample_subarray(
    psf_img: np.ndarray,
    center_x: float,
    center_y: float,
    radius_pix: float,
    oversample: int,
):
    """Get oversampled subarray of the PSF image around a given center.

    Extract a subarray of `psf_img` around (center_x, center_y),
    then oversample that subarray by the specified factor.

    Args:
        psf_img (np.ndarray):
            The input PSF image
        center_x (float):
            Position of the center in the x direction
        center_y (float):
            Position of the center in the y direction
        radius_pix (float):
            The radius of the subarray in pixels
        oversample (int):
            The oversampling factor

    Returns:
        subarr_oversamp (np.ndarray):
            The oversampled subarray
        center_x_os (float):
            center_x in oversampled subarray coords
        center_y_os (float):
            center_y in oversampled subarray coords
        radius_os (float):
            radius_pix * oversample
        subarr (np.ndarray):
            the original subarray (for flux renormalization)
    """
    ny, nx = psf_img.shape
    margin = int(np.ceil(radius_pix * 3))
    xmin = max(0, int(np.floor(center_x - margin)))
    xmax = min(nx - 1, int(np.ceil(center_x + margin)))
    ymin = max(0, int(np.floor(center_y - margin)))
    ymax = min(ny - 1, int(np.ceil(center_y + margin)))

    subarr = psf_img[ymin : ymax + 1, xmin : xmax + 1]

    # Flux-conserving oversample using resample_flux
    ny_os = subarr.shape[0] * oversample
    nx_os = subarr.shape[1] * oversample
    subarr_oversamp = np.asarray(
        resample_flux(
            jnp.asarray(np.asarray(subarr, dtype=np.float64)),
            1.0,
            1.0 / oversample,
            (ny_os, nx_os),
        )
    )

    center_x_os = (center_x - xmin) * oversample
    center_y_os = (center_y - ymin) * oversample
    radius_os = radius_pix * oversample

    return subarr_oversamp, center_x_os, center_y_os, radius_os, subarr


def measure_flux_in_oversampled_aperture(
    subarr_oversamp: np.ndarray,
    center_x_os: float,
    center_y_os: float,
    radius_os: float,
    subarr_original: np.ndarray,
) -> float:
    """Get flux in a circular aperture of radius `radius_os` in the oversampled array.

    Returns:
        flux_in_ap (float): total flux inside the circular mask
    """
    yy_os, xx_os = np.indices(subarr_oversamp.shape)
    rr_os = np.sqrt((xx_os - center_x_os) ** 2 + (yy_os - center_y_os) ** 2)
    ap_mask = rr_os <= radius_os

    flux_in_ap = subarr_oversamp[ap_mask].sum()

    return flux_in_ap


def crop_around_peak(arr, radius):
    """Crop a 2D array to a square region centered on the peak pixel.

    The output is always square with side length ``2 * r`` where ``r``
    is the largest feasible radius that fits within the array bounds
    (capped at the requested *radius*). This function is mostly used
    in the documentation animations.

    Args:
        arr (np.ndarray): 2D input array.
        radius (int): Desired half-width of the output crop in pixels.

    Returns:
        np.ndarray: Square cropped subarray centered on the peak.
    """
    peak_y, peak_x = np.unravel_index(arr.argmax(), arr.shape)
    ny, nx = arr.shape
    # Feasible half-widths in each direction from the peak
    r = min(radius, peak_y, ny - peak_y, peak_x, nx - peak_x)
    return arr[peak_y - r : peak_y + r, peak_x - r : peak_x + r]


def create_shift_mask(psf, shift_x, shift_y, fill_val=1):
    """Create a mask to identify valid pixels to average.

    This function is useful because when the PSF is shifted there are empty
    pixels, since they were outside the initial image, and should not be
    included in the final average.

    Args:
        psf (np.ndarray):
            The PSF image to shift.
        shift_x (float):
            The shift in the x direction.
        shift_y (float):
            The shift in the y direction.
        fill_val (float, optional):
            The value to fill the mask with.

    Returns:
        np.ndarray:
            The mask to identify valid pixels to average.
    """
    mask = np.full_like(psf, fill_val)

    # Handle x-direction shifting
    if shift_x > 0:
        # Zero out the left side
        mask[:, : int(np.ceil(shift_x))] = 0
    elif shift_x < 0:
        # Zero out the right side
        mask[:, int(np.floor(shift_x)) :] = 0

    # Handle y-direction shifting
    if shift_y > 0:
        # Zero out the bottom side
        mask[: int(np.ceil(shift_y)), :] = 0
    elif shift_y < 0:
        # Zero out the top side
        mask[int(np.floor(shift_y)) :, :] = 0

    return mask
