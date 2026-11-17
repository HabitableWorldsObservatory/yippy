"""Utility functions for the yippy package."""

import hashlib
import json
from pathlib import Path

import astropy.io.fits as fits
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Frame
from hwoutils.transforms import resample_flux


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
    record, filename, outdir, overwrite=True, identity=None
):
    """Atomically store the complete single-axis record as float64 columns."""
    import os
    from dataclasses import asdict

    header = fits.Header()
    header["PERFVER"] = 4
    header["APERRULE"] = type(record.rule).__name__
    header["RULEJSON"] = json.dumps(asdict(record.rule), sort_keys=True)
    header["LEAKRULE"] = record.leakage_rule
    header["FRAME"] = json.dumps(asdict(record.frame), sort_keys=True)
    header["IWA_LOD"] = record.iwa_lod if np.isfinite(record.iwa_lod) else "undefined"
    header["EXTENT"] = record.data_extent_lod
    if record.stellar_diameter is not None:
        header["STARDIAM"] = record.stellar_diameter
    if identity is not None:
        header["CACHEKEY"] = cache_identity_digest(identity)
        header["CACHEID"] = cache_identity_json(identity)
    names = (
        "separation_lod",
        "throughput",
        "core_area_lod2",
        "leakage_fraction",
        "leakage_density_lod2",
        "raw_contrast",
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=name, format="D", array=getattr(record, name))
            for name in names
        ],
        name="PERFORMANCE",
    )
    outpath = Path(outdir) / filename
    if outpath.exists() and not overwrite:
        raise FileExistsError(outpath)
    temporary = outpath.with_name(f"{outpath.name}.{os.getpid()}.tmp")
    try:
        fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(
            temporary, overwrite=True
        )
        os.replace(temporary, outpath)
    finally:
        temporary.unlink(missing_ok=True)


def read_performance_cache_key(path: Path) -> str | None:
    """Identity digest, or None for missing/unreadable/old-format records."""
    try:
        header = fits.getheader(path, 0)
        return header.get("CACHEKEY") if header.get("PERFVER") == 4 else None
    except (OSError, ValueError):
        return None


def load_coro_performance_from_fits(filename: str, indir: Path):
    """Read a complete record; legacy partial tables are not supported."""
    from .apertures import Circular, HalfMaximum, Matched
    from .performance import PerformanceCurves

    with fits.open(Path(indir) / filename) as hdus:
        header = hdus[0].header
        if header.get("PERFVER") != 4:
            raise ValueError("performance cache requires PERFVER=4")
        rule_type = {
            "Circular": Circular,
            "HalfMaximum": HalfMaximum,
            "Matched": Matched,
        }[header["APERRULE"]]
        rule = rule_type(**json.loads(header["RULEJSON"]))
        fields = {
            name: np.array(hdus["PERFORMANCE"].data[name], dtype=float)
            for name in (
                "separation_lod",
                "throughput",
                "core_area_lod2",
                "leakage_fraction",
                "leakage_density_lod2",
                "raw_contrast",
            )
        }
        iwa = header["IWA_LOD"]
        return PerformanceCurves(
            **fields,
            frame=Frame(**json.loads(header["FRAME"])),
            rule=rule,
            leakage_rule=header["LEAKRULE"],
            iwa_lod=np.nan if iwa == "undefined" else float(iwa),
            data_extent_lod=header["EXTENT"],
            stellar_diameter=header.get("STARDIAM"),
        )


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
        tuple: The oversampled subarray, its center x/y coordinates and aperture
            radius in oversampled pixels, and the original subarray for flux
            renormalization.
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
