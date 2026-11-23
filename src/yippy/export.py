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


def _smallest_diameter(record):
    """The smallest stellar diameter a record tabulates in lambda/D, or None."""
    diameters = record.stellar_diameters_lod
    return None if diameters is None else float(diameters[0])


def export_exosims(
    coro: Coronagraph,
    units: str = "LAMBDA/D",
    *,
    separations_lod=None,
    output_dir=None,
) -> dict:
    """Write the selected record to output_dir (default: the package's exosims folder).

    All performance quantities use the record's interpolation policy at the
    requested separations (its knots by default). ``arcsec`` converts both
    separation and aperture area. The EXOSIMS core-mean-intensity file retains
    per-stellar-pixel flux fractions; LEAKRULE identifies the averaging rule.
    The leakage and raw contrast are those of the smallest tabulated stellar
    diameter, a point star for the shipped products.
    Array extent is not an outer working angle and is not exported as one.

    Args:
        coro: Coronagraph whose first prepared record is exported.
        units: Angle units of the files, ``"LAMBDA/D"`` or ``"arcsec"``
            (case-insensitive).
        separations_lod: Separations in lambda/D at which to evaluate; None uses the
            knots of the record.
        output_dir: Directory to write into; None uses the ``exosims`` folder of the
            package.

    Returns:
        The ``specs.json`` content: pupil diameter in m, obscuration fraction and the
        starlight suppression system entry.
    """
    if units.upper() not in ("LAMBDA/D", "ARCSEC"):
        raise ValueError("units must be 'LAMBDA/D' or 'arcsec'")
    performance = coro._records[0]
    point_star = _smallest_diameter(performance)
    separations = np.asarray(
        performance.separation_lod if separations_lod is None else separations_lod,
        dtype=float,
    )
    offaxis = coro.package.header.offaxis_psfs
    angle_scale = (
        float(
            performance.frame.lod_to_arcsec(performance.frame.reference_wavelength_nm)
        )
        if units.lower() == "arcsec"
        else 1.0
    )
    output_separations = separations * angle_scale
    exosims_dir = (
        coro.package.source.path / "exosims" if output_dir is None else Path(output_dir)
    )
    exosims_dir.mkdir(parents=True, exist_ok=True)

    header = pyfits.Header()
    header["PIXSCALE"] = (coro.grid.step, "Angular pixel scale")
    header["LAMBDA"] = (offaxis.lambda0.to_value(u.um), "Wavelength in micrometers")
    header["D"] = (offaxis.diameter.to_value(u.m), "Telescope diameter in meters")
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
    if offaxis.maxlam is not None and offaxis.minlam is not None:
        bandwidth = offaxis.maxlam - offaxis.minlam
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
        "raw_contrast": performance.evaluate("raw_contrast", separations, point_star),
        "core_mean_intensity": np.asarray(
            performance.evaluate("leakage_density_lod2", separations, point_star)
        )
        * stellar_pixel_area_lod2,
        "occ_trans": coro.uniform_sky_response(separations),
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
        "pupilDiam": offaxis.diameter.to_value(u.m),
        "obscurFac": coro.header.obscured,
        "starlightSuppressionSystems": [
            {
                "name": coro.package.source.path.name,
                "lam": offaxis.lambda0.to_value(u.nm),
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
    ppf: float,
    *,
    separation_min_lod: float = 0.125,
    separation_max_lod: float = 32.0,
    separation_step_lod: float = 0.25,
    contrast_floor: float | None = None,
    separations_lod=None,
) -> Path:
    """Write AYO columns using the selected record's interpolation policy.

    Point-source contrast is the selected mean leakage density times the
    mean collecting area divided by throughput. The noise-floor column is
    this contrast divided by ``ppf``, the post-processing factor the caller
    assumes, including any requested contrast floor. The leakage is that of the
    smallest tabulated stellar diameter, a point star for the shipped products.
    Skytrans is the radial sky transmission.

    Separations are in lambda/D. A contrast floor applies only when explicitly
    requested and does not change the performance record. Seventeen significant
    digits preserve float64 values through a CSV round trip.

    Args:
        coro: Coronagraph whose first prepared record is exported.
        output_path: Path of the CSV file to write.
        ppf: Post-processing factor by which the noise-floor column divides the
            contrast.
        separation_min_lod: First separation of the default grid, in lambda/D.
        separation_max_lod: Last separation of the default grid, in lambda/D.
        separation_step_lod: Spacing of the default grid, in lambda/D.
        contrast_floor: Optional lower bound applied to the contrast column; None
            applies none.
        separations_lod: Explicit separations in lambda/D; None uses the default grid.

    Returns:
        The path of the CSV file.
    """
    output_path = Path(output_path)
    separations = np.asarray(
        np.arange(
            separation_min_lod,
            separation_max_lod + separation_step_lod / 2,
            separation_step_lod,
        )
        if separations_lod is None
        else separations_lod,
        dtype=float,
    )
    contrast = np.asarray(
        coro._records[0].evaluate(
            "raw_contrast", separations, _smallest_diameter(coro._records[0])
        )
    )
    if contrast_floor is not None:
        contrast = np.maximum(contrast, contrast_floor)
    throughput = coro._records[0].evaluate("throughput", separations)
    occ_trans = coro.uniform_sky_response(separations)
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
