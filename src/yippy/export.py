"""Export coronagraph performance curves to external formats.

Provides functions to export coronagraph performance data in EXOSIMS FITS
format and AYO's CSV format.

All functions operate on a Coronagraph instance passed as the first argument.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from lod_unit import lod

from .logger import logger
from .performance import compute_radial_average

if TYPE_CHECKING:
    from .coronagraph import Coronagraph


def _save_to_exosims_format(
    coro: Coronagraph,
    sep,
    throughput,
    raw_contrast,
    core_area,
    sep_occ_trans,
    occ_trans,
    sep_core_intensity,
    core_intensities,
    aperture_radius_lod,
    fit_gaussian_for_core_area,
    use_phot_aperture_as_min,
    units,
):
    """Write individual EXOSIMS FITS files to ``<yip_path>/exosims/``."""
    exosims_dir = Path(coro.yip_path, "exosims")
    exosims_dir.mkdir(exist_ok=True)

    # Base header from coronagraph parameters
    base_header = pyfits.Header()
    base_header["PIXSCALE"] = (coro.pixel_scale_arcsec.value, "Angular pixel scale")
    base_header["LAMBDA"] = (coro.header.lambda0.value, "Wavelength in micrometers")
    base_header["D"] = (coro.header.diameter.value, "Telescope diameter in meters")
    base_header["OBSCURED"] = (coro.header.obscured, "Obscuration fraction")
    if coro.header.maxlam is not None and coro.header.minlam is not None:
        base_header["DELTALAM"] = (
            (coro.header.maxlam - coro.header.minlam).value,
            "Bandpass width in micrometers",
        )
    base_header["UNITS"] = (units, "Angular units")

    # Occulter transmission
    hdul = pyfits.HDUList(
        [
            pyfits.PrimaryHDU(
                np.vstack((sep_occ_trans, occ_trans)).T,
                header=base_header.copy(),
            )
        ]
    )
    hdul.writeto(exosims_dir / "occ_trans.fits", overwrite=True)

    # Core throughput
    thruput_header = base_header.copy()
    if fit_gaussian_for_core_area:
        thruput_header["PHOTAPER"] = "Gaussian"
        thruput_header["MINAPER"] = (
            aperture_radius_lod if use_phot_aperture_as_min else 0
        )
    else:
        thruput_header["PHOTAPER"] = aperture_radius_lod

    hdul = pyfits.HDUList(
        [
            pyfits.PrimaryHDU(
                np.vstack((sep, throughput)).T,
                header=thruput_header,
            )
        ]
    )
    hdul.writeto(exosims_dir / "core_thruput.fits", overwrite=True)

    # Core area (only for Gaussian fitting)
    if fit_gaussian_for_core_area:
        hdul = pyfits.HDUList(
            [
                pyfits.PrimaryHDU(
                    np.vstack((sep, core_area)).T,
                    header=thruput_header.copy(),
                )
            ]
        )
        hdul.writeto(exosims_dir / "core_area.fits", overwrite=True)
    else:
        logger.info(
            f"Fixed aperture core area: {aperture_radius_lod**2 * np.pi:.6f} (lam/D)**2"
        )

    # Core mean intensity
    ci_header = base_header.copy()
    stellar_diams = list(core_intensities.keys())
    for j, diam in enumerate(stellar_diams):
        ci_header[f"DIAM{j:03d}"] = (
            diam.value,
            f"Stellar diameter {j} in lambda/D",
        )
    intensity_array = np.zeros((len(stellar_diams), len(sep_core_intensity)))
    for j, diam in enumerate(stellar_diams):
        intensity_array[j] = core_intensities[diam]
    hdul = pyfits.HDUList(
        [
            pyfits.PrimaryHDU(
                np.vstack((sep_core_intensity, intensity_array)).T,
                header=ci_header,
            )
        ]
    )
    hdul.writeto(exosims_dir / "core_mean_intensity.fits", overwrite=True)

    # Raw contrast
    hdul = pyfits.HDUList(
        [
            pyfits.PrimaryHDU(
                np.vstack((sep, raw_contrast)).T,
                header=thruput_header.copy(),
            )
        ]
    )
    hdul.writeto(exosims_dir / "raw_contrast.fits", overwrite=True)

    logger.info(f"EXOSIMS format files saved to {exosims_dir}/")
    logger.info("Files created:")
    logger.info("  - occ_trans.fits (occulter transmission)")
    logger.info("  - core_thruput.fits (throughput)")
    if fit_gaussian_for_core_area:
        logger.info("  - core_area.fits (core area from Gaussian fits)")
    logger.info("  - core_mean_intensity.fits (stellar intensity)")
    logger.info("  - raw_contrast.fits (raw contrast)")
    logger.info("  - specs.json (EXOSIMS specification file)")


def export_exosims(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    fit_gaussian_for_core_area: bool = False,
    use_phot_aperture_as_min: bool = False,
    units: str = "LAMBDA/D",
) -> dict:
    """Export performance curves in EXOSIMS format.

    Writes individual FITS files and a ``specs.json`` to
    ``<yip_path>/exosims/``.

    Returns the EXOSIMS specs dictionary.
    """
    # Validate that interpolators exist
    required = [
        "throughput_interp",
        "raw_contrast_interp",
        "occ_trans_interp",
        "core_area_interp",
        "core_intensity_interp",
    ]
    missing = [a for a in required if not hasattr(coro, a)]
    if missing:
        raise ValueError(
            f"Performance curves not computed. Missing: {missing}. "
            "Call compute_all_performance_curves() first."
        )

    # Build separations
    x_offsets = np.array(coro.offax.x_offsets)
    separations = np.abs(x_offsets)
    if hasattr(coro.offax, "max_offset_in_image"):
        max_sep = coro.offax.max_offset_in_image.to(lod).value
        separations = separations[separations <= max_sep]
    separations = np.sort(np.unique(separations))

    # Evaluate interpolators
    throughput = coro.throughput_interp(separations)
    raw_contrast = coro.raw_contrast_interp(separations)
    core_area = coro.core_area_interp(separations)

    # Clip small negatives from spline artifacts
    if np.any(throughput < 0):
        n_neg = np.sum(throughput < 0)
        min_val = np.min(throughput)
        if min_val < -0.01:
            raise ValueError(
                f"Found {n_neg} negative throughput values. Min: {min_val:.6f}"
            )
        logger.debug(
            f"Clipping {n_neg} small negative throughput values (min: {min_val:.6f})"
        )
        throughput = np.clip(throughput, 0, None)

    if np.any(throughput > 1):
        raise ValueError(f"Throughput > 1: max = {np.max(throughput):.3f}")

    # Occulter transmission
    sky_trans_data = coro.sky_trans.data
    sep_occ_trans, occ_trans = compute_radial_average(
        sky_trans_data,
        coro.sky_trans.grid.step,
        center=(coro.sky_trans.grid.center_y, coro.sky_trans.grid.center_x),
    )

    # Core mean intensity
    if hasattr(coro, "core_intensity_dict"):
        sep_core_intensity = coro.core_intensity_separations
        core_intensities = coro.core_intensity_dict
    else:
        sep_core_intensity, core_intensities = coro.core_mean_intensity_curve(
            plot=False
        )

    # Write FITS files
    _save_to_exosims_format(
        coro,
        sep=separations,
        throughput=throughput,
        raw_contrast=raw_contrast,
        core_area=core_area,
        sep_occ_trans=sep_occ_trans,
        occ_trans=occ_trans,
        sep_core_intensity=sep_core_intensity,
        core_intensities=core_intensities,
        aperture_radius_lod=aperture_radius_lod,
        fit_gaussian_for_core_area=fit_gaussian_for_core_area,
        use_phot_aperture_as_min=use_phot_aperture_as_min,
        units=units,
    )

    # IWA / OWA
    IWA = coro.IWA
    OWA = coro.OWA

    to_arcsec = units.lower() == "arcsec"
    if to_arcsec:
        frame = coro.grid.frame
        angunit = frame.lod_to_arcsec(coro.header.lambda0.to_value(u.nm))
        IWA_output = float(IWA.to_value(lod) * angunit)
        OWA_output = float(OWA.to_value(lod) * angunit)
    else:
        IWA_output = IWA.to_value(lod)
        OWA_output = OWA.to_value(lod)

    # Core area filename or scalar
    core_area_fname = (
        "core_area.fits"
        if fit_gaussian_for_core_area
        else aperture_radius_lod**2 * np.pi
    )

    # deltaLam
    deltaLam = None
    if coro.header.maxlam is not None and coro.header.minlam is not None:
        deltaLam = (coro.header.maxlam - coro.header.minlam).to(u.nm).value

    # Specs dict
    outdict = {
        "pupilDiam": coro.header.diameter.to(u.m).value,
        "obscurFac": coro.header.obscured,
        "starlightSuppressionSystems": [
            {
                "name": coro.name,
                "lam": coro.header.lambda0.to(u.nm).value,
                "deltaLam": deltaLam,
                "occ_trans": "occ_trans.fits",
                "core_thruput": "core_thruput.fits",
                "core_mean_intensity": "core_mean_intensity.fits",
                "core_area": core_area_fname,
                "IWA": IWA_output,
                "OWA": OWA_output,
                "input_angle_units": units,
            }
        ],
    }

    exosims_dir = Path(coro.yip_path, "exosims")
    specs_file = exosims_dir / "specs.json"
    with open(specs_file, "w") as f:
        json.dump(outdict, f, indent=2)

    logger.info(f"EXOSIMS specs saved to {specs_file}")
    return outdict


def export_ayo_csv(
    coro: Coronagraph,
    output_path,
    sep_min: float = 0.125,
    sep_max: float = 32.0,
    sep_step: float = 0.25,
    contrast_floor: float = 1e-10,
    ppf: float = 30.0,
) -> Path:
    """Export performance curves in AYO CSV format.

    Returns the path to the saved CSV file.
    """
    output_path = Path(output_path)
    separations = np.arange(sep_min, sep_max + sep_step / 2, sep_step)

    raw_contrast = np.abs(coro.raw_contrast(separations))
    contrast = np.maximum(raw_contrast, contrast_floor)
    noise_floor_val = contrast / ppf
    throughput = coro.throughput(separations)
    occ_trans = coro.occulter_transmission(separations)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Sep (l/D)",
                "Contrast (for a point source)",
                "Noise floor (point source 1-sigma)",
                "Core throughput",
                "Skytrans",
            ]
        )
        for i, sep in enumerate(separations):
            writer.writerow(
                [
                    f"{sep:.6f}",
                    f"{contrast[i]:.6e}",
                    f"{noise_floor_val[i]:.6e}",
                    f"{throughput[i]:.6e}",
                    f"{occ_trans[i]:.6f}",
                ]
            )

    logger.info(f"AYO-format CSV saved to {output_path}")
    return output_path
