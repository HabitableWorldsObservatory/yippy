"""Export the shared performance record to EXOSIMS FITS and AYO CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np

from .logger import logger

if TYPE_CHECKING:
    from .coronagraph import Coronagraph


def export_exosims(
    coro: Coronagraph,
    units: str = "LAMBDA/D",
    *,
    separations_lod=None,
) -> dict:
    """Write the selected performance record to ``<yip_path>/exosims/``.

    All performance quantities use the record's interpolation policy at the
    requested separations (its knots by default). ``arcsec`` converts both
    separation and aperture area. The EXOSIMS core-mean-intensity file retains
    per-stellar-pixel flux fractions; LEAKRULE identifies the averaging rule.
    Array extent is not an outer working angle and is not exported as one.
    """
    if units.upper() not in ("LAMBDA/D", "ARCSEC"):
        raise ValueError("units must be 'LAMBDA/D' or 'arcsec'")
    performance = coro._records[0]
    separations = np.asarray(
        performance.separation_lod if separations_lod is None else separations_lod,
        dtype=float,
    )
    angle_scale = (
        float(performance.frame.lod_to_arcsec(coro.header.lambda0.to_value(u.nm)))
        if units.lower() == "arcsec"
        else 1.0
    )
    output_separations = separations * angle_scale
    exosims_dir = Path(coro.yip_path, "exosims")
    exosims_dir.mkdir(exist_ok=True)

    header = pyfits.Header()
    header["PIXSCALE"] = (coro.grid.step, "Angular pixel scale")
    header["LAMBDA"] = (coro.header.lambda0.to_value(u.um), "Wavelength in micrometers")
    header["D"] = (coro.header.diameter.to_value(u.m), "Telescope diameter in meters")
    header["OBSCURED"] = (coro.header.obscured, "Obscuration fraction")
    header["UNITS"] = (units, "Angular units")
    header["APERRULE"] = (type(performance.rule).__name__, "Photometric weighting rule")
    header["LEAKRULE"] = (performance.leakage_rule, "Stellar leakage averaging rule")
    if hasattr(performance.rule, "radius_lod"):
        header["RADLOD"] = (performance.rule.radius_lod, "Aperture radius in lambda/D")
        header["CENTRE"] = (
            performance.rule.centre,
            "Circular aperture centre convention",
        )
    if hasattr(performance.rule, "fraction"):
        header["FRACPEAK"] = (performance.rule.fraction, "Fraction of peak intensity")
    delta_lambda_nm = None
    if coro.header.maxlam is not None and coro.header.minlam is not None:
        bandwidth = coro.header.maxlam - coro.header.minlam
        header["DELTALAM"] = (bandwidth.to_value(u.um), "Bandpass width in micrometers")
        delta_lambda_nm = bandwidth.to_value(u.nm)

    stellar_grid = coro.stellar_grid
    stellar_frame = stellar_grid.frame
    angular_ratio = performance.frame.lod_to_arcsec(
        performance.frame.reference_wavelength_nm
    ) / stellar_frame.lod_to_arcsec(stellar_frame.reference_wavelength_nm)
    stellar_pixel_area_lod2 = stellar_grid.pixel_area / angular_ratio**2
    curves = {
        "core_thruput": performance.evaluate("throughput", separations),
        "core_area": np.asarray(performance.evaluate("core_area_lod2", separations))
        * angle_scale**2,
        "raw_contrast": performance.evaluate("raw_contrast", separations),
        "core_mean_intensity": np.asarray(
            performance.evaluate("leakage_density_lod2", separations)
        )
        * stellar_pixel_area_lod2,
        "occ_trans": coro._occultation_record.evaluate(
            "occulter_transmission", separations
        ),
    }
    for name, values in curves.items():
        curve_header = header.copy()
        if name == "core_mean_intensity":
            curve_header["BUNIT"] = "stellar flux fraction/pixel"
        elif name == "core_area":
            curve_header["BUNIT"] = "arcsec2" if units.lower() == "arcsec" else "lod2"
        pyfits.PrimaryHDU(
            np.column_stack((output_separations, values)), header=curve_header
        ).writeto(exosims_dir / f"{name}.fits", overwrite=True)

    specs = {
        "pupilDiam": coro.header.diameter.to_value(u.m),
        "obscurFac": coro.header.obscured,
        "starlightSuppressionSystems": [
            {
                "name": coro.yip_path.name,
                "lam": coro.header.lambda0.to_value(u.nm),
                "deltaLam": delta_lambda_nm,
                "occ_trans": "occ_trans.fits",
                "core_thruput": "core_thruput.fits",
                "core_mean_intensity": "core_mean_intensity.fits",
                "core_area": "core_area.fits",
                "IWA": float(performance.iwa_lod) * angle_scale,
                "input_angle_units": units,
            }
        ],
    }
    with open(exosims_dir / "specs.json", "w") as stream:
        json.dump(specs, stream, indent=2)
    logger.info(f"EXOSIMS files saved to {exosims_dir}")
    return specs


def export_ayo_csv(
    coro: Coronagraph,
    output_path,
    sep_min: float = 0.125,
    sep_max: float = 32.0,
    sep_step: float = 0.25,
    contrast_floor: float | None = None,
    ppf: float = 30.0,
    *,
    separations_lod=None,
) -> Path:
    """Write AYO columns using the selected record's interpolation policy.

    Point-source contrast is the selected mean leakage density times the
    mean collecting area divided by throughput. The noise-floor column is
    this contrast divided by ppf, including any requested contrast floor.
    Skytrans is the radial sky transmission.

    Separations are in lambda/D. A contrast floor applies only when explicitly
    requested and does not change the performance record. Seventeen significant
    digits preserve float64 values through a CSV round trip.
    """
    output_path = Path(output_path)
    separations = np.asarray(
        np.arange(sep_min, sep_max + sep_step / 2, sep_step)
        if separations_lod is None
        else separations_lod,
        dtype=float,
    )
    contrast = np.asarray(coro._records[0].evaluate("raw_contrast", separations))
    if contrast_floor is not None:
        contrast = np.maximum(contrast, contrast_floor)
    throughput = coro._records[0].evaluate("throughput", separations)
    occ_trans = coro._occultation_record.evaluate("occulter_transmission", separations)
    columns = np.column_stack(
        (separations, contrast, contrast / ppf, throughput, occ_trans)
    )
    with open(output_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Sep (l/D)",
                "Contrast (for a point source)",
                "Noise floor (point source 1-sigma)",
                "Core throughput",
                "Skytrans",
            ]
        )
        writer.writerows([format(value, ".17g") for value in row] for row in columns)
    logger.info(f"AYO-format CSV saved to {output_path}")
    return output_path
