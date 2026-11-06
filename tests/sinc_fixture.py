"""Small symmetric sampled-intensity delivery with independent signed images."""

import csv
import hashlib
import json

import numpy as np
from astropy.io import fits
from synthetic_yip import _header, write_synthetic_yip

_PREMISES = (
    "entrance_tilt_model",
    "fixed_linear_optics",
    "fixed_detector_registration",
    "fixed_pixel_response",
    "source_independent_normalization",
    "source_independent_spectral_weights",
)


def write_delivery(root, *, full=False):
    """Write native float32 band-limited samples and a separate legacy package."""
    root.mkdir(parents=True, exist_ok=True)
    write_synthetic_yip(
        root / "legacy", npix=4, pixscale=0.4, max_offset_lod=1.0, offset_step_lod=0.5
    )
    for name in ("stellar_intens.fits", "sky_trans.fits"):
        with fits.open(root / "legacy" / name, mode="update") as hdus:
            for key in ("LAMBDA", "MINLAM", "MAXLAM"):
                hdus[0].header[key] = (400.0, "nm")
    axis = np.arange(-16, 17) if full else np.arange(18)
    ix, iy = np.meshgrid(axis, axis, indexing="xy")
    indices = np.column_stack((ix.ravel(), iy.ravel()))
    offsets = indices * 0.38
    detector_y, detector_x = np.mgrid[:4, :4]
    x, y = (detector_x - 1.5) * 0.25, (detector_y - 1.5) * 0.25

    def images(coords):
        return np.array(
            [
                (1 + 0.2 * np.cos(2 * np.pi * 0.2 * (x - qx)))
                * (1 + 0.1 * np.cos(2 * np.pi * 0.3 * (y - qy)))
                * 1e-3
                for qx, qy in coords
            ],
            np.float32,
        )

    header = _header(4, 1, 0.25, 1.5, "symmetric")
    header["MINLAM"] = (500.0, "nm")
    header["MAXLAM"] = (500.0, "nm")
    fits.writeto(root / "nodes.fits", images(offsets), header)
    fits.writeto(root / "offsets.fits", offsets)
    with (root / "training.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "ix", "iy", "x_lod", "y_lod"])
        for index, (k, q) in enumerate(zip(indices, offsets, strict=True)):
            writer.writerow([index, *k, *q])
    signed = np.array([[-0.38, 0.38], [0.38, -0.38], [-0.38, -0.38]])
    fits.writeto(root / "mirror.fits", images(signed))
    (root / "mirror.json").write_text('{"status":"pass"}')
    checks = np.array([[0.07, -0.04]])
    fits.writeto(root / "checks.fits", images(checks))
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()  # noqa: E731
    base = int(np.flatnonzero(np.all(indices == [1, 1], axis=1))[0])
    m = {
        "schema": "sinc-delivery-0.1",
        "delivery_id": "synthetic",
        "design_id": "symmetric",
        "lambda0_m": 500e-9,
        "diameter_m": 6.0,
        "diameter_convention": "entrance_diameter",
        "coordinate_model": "paraxial_entrance_tilt",
        "source_unit": "lambda0/D",
        "offset_order": "xy",
        "array_order": "source,y,x",
        "source_origin_lod": [0.0, 0.0],
        "source_spacing_lod": [0.38, 0.38],
        "offset_tolerance_rule": "four_ulp_pitch_cap",
        "support_extent_D": [1.0, 1.0],
        "support_evidence": "analytic-band-limited",
        "premises": {
            key: {"value": True, "evidence": 1, "source_id": "analytic"}
            for key in _PREMISES
        },
        "kernel": {
            "family": "kaiser_sinc_raw",
            "half_width": 16,
            "beta": 12,
            "max_occupancy": 0.76,
        },
        "served": {
            "kind": "box",
            "boundary": "closed",
            "bounds_lod": [[-0.1, 0.1], [-0.1, 0.1]],
            "radii_lod": None,
        },
        "layout": "full_rectangle" if full else "quadrant_rectangle",
        "reflection": None
        if full
        else {
            "kind": "two_axis_pixel_flip_v1",
            "x_mirror": True,
            "y_mirror": True,
            "evidence": "designer_declared",
            "declaration_id": "analytic",
            "scope": "fixed symmetric optics",
            "response_argument_kind": "component_reflections",
            "response_argument": "even cosine factors",
            "optical_state_id": "analytic",
            "check_tolerance_local_peak": 1e-6,
            "check_status": "pass",
            "report_path": "mirror.json",
            "report_sha256": digest("mirror.json"),
        },
        "detector_shape_yx": [4, 4],
        "detector_center_xy": [1.5, 1.5],
        "detector_convention": "pixel_midpoint_v1",
        "pixel_scale_lod_xy": [0.25, 0.25],
        "pixel_scale_arcsec_xy": [(0.25 * (500e-9 / 6.0)) * (180 * 3600 / np.pi)] * 2,
        "sample_kind": "integral",
        "pixel_response": "uniform square",
        "instrument_to_sky": None,
        "normalization": {"kind": "incident_photon", "source_independent": True},
        "optical_state": {"state_id": "analytic"},
        "storage_dtype": "float32",
        "spectral_sampling": {
            "node_count": 1,
            "shortest_contributing_m": 500e-9,
            "campaign_sampling_min_m": None,
        },
        "legacy_package_path": "legacy",
        "training_coordinates_path": "training.csv",
        "planes": [
            {
                "id": "mono",
                "kind": "monochromatic",
                "wavelength_m": 500e-9,
                "band_edges_m": [500e-9, 500e-9],
                "psf_path": "nodes.fits",
                "offsets_path": "offsets.fits",
                "psf_sha256": digest("nodes.fits"),
                "offsets_sha256": digest("offsets.fits"),
                "photon_conversion": 1.0,
                "reference_uncertainty": None,
                "sampling_min_wavelength_m": 500e-9,
                "shortest_contributing_m": 500e-9,
            }
        ],
        "mirror_checks": []
        if full
        else [
            {
                "id": f"mirror-{i}",
                "plane_id": "mono",
                "base_index": base,
                "source_lod_xy": q.tolist(),
                "image_index": i,
                "psf_path": "mirror.fits",
                "psf_sha256": digest("mirror.fits"),
                "independent_propagation": True,
            }
            for i, q in enumerate(signed)
        ],
        "checks": [
            {
                "id": "check-0",
                "plane_id": "mono",
                "source_lod_xy": checks[0].tolist(),
                "psf_path": "checks.fits",
                "psf_sha256": digest("checks.fits"),
                "image_index": 0,
                "independent_propagation": True,
                "reference_units": "incident_fraction_per_pixel",
                "local_peak_reference": float(images(checks).max()),
            }
        ],
    }
    save_manifest(root, m)
    return m


def save_manifest(root, manifest):
    """Update only the fixture's manifest for a distinct input failure."""
    (root / "sinc-delivery.json").write_text(json.dumps(manifest))
