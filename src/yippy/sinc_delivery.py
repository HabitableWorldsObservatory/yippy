"""Host-only intake for versioned sampled-intensity deliveries.

FITS images retain their stored floating precision; byte order alone is changed.
The training CSV supplies authoritative integer nodes, never fitted offsets.
"""

import csv
import hashlib
import json
import re
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from hwoutils.constants import rad2arcsec
from lod_unit import lod

from .header import HeaderData

_PREMISES = (
    "entrance_tilt_model",
    "fixed_linear_optics",
    "fixed_detector_registration",
    "fixed_pixel_response",
    "source_independent_normalization",
    "source_independent_spectral_weights",
)


def file_sha256(path):
    """Hash a file in bounded blocks without retaining a second cube."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Delivery:
    """Validated host input, including original metadata and native arrays."""

    root: Path
    manifest: dict
    manifest_sha256: str
    plane: dict
    header: HeaderData
    nodes: np.ndarray
    offsets: np.ndarray
    indices: np.ndarray
    origin_lod: np.ndarray
    shape_yx: tuple[int, int]
    tolerance_lod: np.ndarray
    coordinate_residual_lod: np.ndarray
    conversion: float
    independent_images: dict
    missing_metadata: tuple[str, ...]

    def read_images(self, records):
        """Read independent images from exact file/index pairs in record order."""
        images = []
        for record in records:
            key = (record["psf_path"], record["psf_sha256"], record["image_index"])
            if key not in self.independent_images:
                raise ValueError(
                    "Independent image_index must identify a validated cube row"
                )
            images.append(self.independent_images[key])
        return np.stack(images)


def detect_delivery(path):
    """Recognize a manifest before invoking any legacy layout inference."""
    manifest_path = Path(path) / "sinc-delivery.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "sinc-delivery-0.1":
        raise ValueError("Unsupported sinc delivery schema")
    if manifest.get("layout") not in ("quadrant_rectangle", "full_rectangle"):
        raise ValueError("Unsupported declared sinc delivery layout")
    return manifest


def _check_hash(path, expected, *, required=False):
    if not expected:
        if required:
            raise ValueError(f"Missing sha256 for {path.name}")
        return
    if file_sha256(path) != expected:
        raise ValueError(f"sha256 mismatch: {path.name}")


def _native(array):
    return np.array(array, dtype=array.dtype.newbyteorder("="), copy=True)


def _valid_intensity(array):
    """Check signs by bits so negative subnormals cannot flush to accepted zero."""
    integer = np.int32 if array.dtype.itemsize == 4 else np.int64
    bits = array.view(integer)
    negative = (bits < 0) & ((bits & np.iinfo(integer).max) != 0)
    return np.all(np.isfinite(array) & ~negative)


def _agree(actual, expected, label, tolerance):
    if not np.all(np.isfinite(actual)) or np.any(np.abs(actual - expected) > tolerance):
        raise ValueError(f"Conflicting {label} in delivery metadata or FITS header")


def _header_checks(header, manifest, plane, tolerance_lod):
    angular_arcsec = (float(manifest["lambda0_m"]) / float(manifest["diameter_m"])) * (
        rad2arcsec
    )
    for key, expected in (
        ("LAMBDA", manifest["lambda0_m"]),
        ("MINLAM", plane["wavelength_m"]),
        ("MAXLAM", plane["wavelength_m"]),
        ("D", manifest["diameter_m"]),
    ):
        if key not in header:
            raise ValueError(f"Missing FITS {key}")
        unit = HeaderData.extract_unit(header.comments[key], u.m, key)
        actual = (float(header[key]) * unit).to_value(u.m)
        _agree(actual, float(expected), key, 8 * np.spacing(float(expected)))
    if header.get("N_LAM") != 1:
        raise ValueError("Conflicting FITS N_LAM for monochromatic product")
    pitch = np.asarray(manifest["pixel_scale_lod_xy"], dtype=np.float64)
    if pitch.shape != (2,) or not np.all(pitch > 0) or pitch[0] != pitch[1]:
        raise ValueError("Only square positive aligned detector pixels are supported")
    if "PIXSCALE" not in header:
        raise ValueError("Missing FITS PIXSCALE")
    comment = header.comments["PIXSCALE"].strip().lower()
    actual_pitch = float(header["PIXSCALE"])
    if not comment or re.search(r"\b(?:lambda0?|lam)\s*/\s*d\b|\blod\b", comment):
        pass
    else:
        match = re.search(r"\b(arcsec|mas|deg|rad)\b", comment)
        if match is None:
            raise ValueError("Unknown PIXSCALE angular unit")
        factor = {"arcsec": 1.0, "mas": 0.001, "deg": 3600.0, "rad": rad2arcsec}[
            match[1]
        ]
        actual_pitch = (actual_pitch * factor) / angular_arcsec
    _agree(actual_pitch, pitch, "PIXSCALE", tolerance_lod)
    _agree(
        np.asarray(manifest["pixel_scale_arcsec_xy"]) / angular_arcsec,
        pitch,
        "angular pixel scale",
        tolerance_lod,
    )
    shape = tuple(manifest["detector_shape_yx"])
    if shape != (header["NAXIS2"], header["NAXIS1"]) or shape[0] != shape[1]:
        raise ValueError("Conflicting detector shape; square images required")
    center = np.asarray(manifest["detector_center_xy"], dtype=np.float64)
    _agree(
        np.array([header["XCENTER"], header["YCENTER"]]), center, "detector center", 0.0
    )
    if manifest.get("detector_convention") != "pixel_midpoint_v1" or not np.array_equal(
        center, (np.array(shape[::-1]) - 1) / 2
    ):
        raise ValueError("Incompatible midpoint detector frame")


def load_delivery(root, manifest, options):
    """Validate selected metadata, hashes, coordinates and native FITS arrays.

    Args:
        root: Delivery directory.
        manifest: Recognized versioned manifest.
        options: SincOptions selecting exactly one monochromatic plane.

    Returns:
        A Delivery ready for sinchole construction, with no legacy transforms.
    """
    root = Path(root)
    for key in (
        "detector_shape_yx",
        "detector_center_xy",
        "pixel_scale_lod_xy",
        "pixel_scale_arcsec_xy",
        "sample_kind",
        "normalization",
    ):
        if manifest.get(key) is None:
            raise ValueError("Missing required geometry or output units: " + key)
    planes = manifest["planes"]
    ids = [p["id"] for p in planes]
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Delivery plane IDs must be nonempty and unique")
    selected = options.product_id
    if selected is None:
        if len(planes) != 1:
            raise ValueError("sinc_options.product_id is required for multiple planes")
        selected = ids[0]
    if selected not in ids:
        raise ValueError(f"Unknown product_id: {selected}")
    plane = planes[ids.index(selected)]
    if plane.get("photon_conversion") is None:
        raise ValueError("Missing required photon_conversion")
    if plane["kind"] != "monochromatic":
        raise ValueError("Only one selected monochromatic product is supported")
    for key, value in (
        ("source_unit", "lambda0/D"),
        ("offset_order", "xy"),
        ("array_order", "source,y,x"),
        ("offset_tolerance_rule", "four_ulp_pitch_cap"),
    ):
        if manifest.get(key) != value:
            raise ValueError(f"Unsupported {key}")
    if "offset_tolerance_lod" in manifest:
        raise ValueError("Absolute offset tolerance overrides are unsupported")
    kernel = manifest["kernel"]
    if kernel != {
        "family": "kaiser_sinc_raw",
        "half_width": 16,
        "beta": 12,
        "max_occupancy": 0.76,
    }:
        raise ValueError("Unsupported sinc kernel profile")
    spectral = manifest["spectral_sampling"]
    if spectral["node_count"] != len(planes):
        raise ValueError("Conflicting spectral product count")
    wavelengths = [p["wavelength_m"] for p in planes]
    if any(a < b for a, b in pairwise(wavelengths)):
        raise ValueError("Wavelength order must be longest to shortest")
    if spectral["shortest_contributing_m"] != min(
        p["shortest_contributing_m"] for p in planes
    ):
        raise ValueError("Conflicting campaign shortest contributor")
    for p in planes:
        if (
            p["kind"] != "monochromatic"
            or not np.isfinite(float(p["wavelength_m"]) * 1e9)
            or p["wavelength_m"] <= 0
            or p["shortest_contributing_m"] != p["wavelength_m"]
            or (
                p.get("sampling_min_wavelength_m") is not None
                and not (0 < p["sampling_min_wavelength_m"] <= p["wavelength_m"])
            )
            or p["band_edges_m"] != [p["wavelength_m"]] * 2
        ):
            raise ValueError("METADATA_CONFLICT: monochromatic wavelength metadata")
    campaign = spectral.get("campaign_sampling_min_m")
    if campaign is not None and not (
        0
        < campaign
        <= min(p.get("sampling_min_wavelength_m") or p["wavelength_m"] for p in planes)
    ):
        raise ValueError("Conflicting campaign sampling minimum")
    if spectral.get("band_ready") and campaign is None:
        raise ValueError("Band-ready campaign requires a sampling minimum")
    _validate_declarations(manifest)
    _validate_domain(manifest.get("served"), options.domain)
    if (
        manifest["layout"] == "full_rectangle"
        and manifest.get("reflection") is not None
    ):
        raise ValueError("Full rectangle requires reflection=null")
    for product in planes:
        for field in ("psf", "offsets"):
            _check_hash(
                root / product[field + "_path"],
                product.get(field + "_sha256"),
                required=True,
            )
    for name, digest in manifest.get("extensions", {}).get("file_sha256", {}).items():
        _check_hash(root / name, digest, required=True)
    psf_path, offsets_path = root / plane["psf_path"], root / plane["offsets_path"]
    header = fits.getheader(psf_path)
    if header["NAXIS"] != 3 or header["BITPIX"] not in (-32, -64):
        raise ValueError("PSF cube requires native float32/float64 [source,y,x]")
    count = header["NAXIS3"]
    offsets_header = fits.getheader(offsets_path)
    if (
        offsets_header.get("NAXIS"),
        offsets_header.get("NAXIS1"),
        offsets_header.get("NAXIS2"),
    ) != (2, 2, count):
        raise ValueError("Offsets require [source,2] matching PSF cube")
    hashes = manifest.get("extensions", {}).get("file_sha256", {})
    training_path = manifest["training_coordinates_path"]
    _check_hash(root / training_path, hashes.get(training_path))
    with (root / training_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count:
        raise ValueError("Training CSV row count differs from cube")
    row_ids = np.array([int(row["index"]) for row in rows])
    if not np.array_equal(np.sort(row_ids), np.arange(count)):
        raise ValueError("Training index must be a unique permutation of FITS rows")
    order = np.argsort(row_ids)
    indices = np.array([[int(row["ix"]), int(row["iy"])] for row in rows], np.int64)[
        order
    ]
    csv_offsets = np.array(
        [[float(row["x_lod"]), float(row["y_lod"])] for row in rows]
    )[order]
    if len(np.unique(indices, axis=0)) != count:
        raise ValueError("Duplicate training nodes")
    lower, upper = indices.min(axis=0), indices.max(axis=0)
    size = upper - lower + 1
    if int(np.prod(size)) != count:
        raise ValueError("Holes in declared complete rectangle")
    if np.any(size > np.iinfo(np.int32).max):
        raise ValueError("Source rectangle exceeds int32 indices")
    origin = np.asarray(manifest["source_origin_lod"], np.float64)
    spacing = np.asarray(manifest["source_spacing_lod"], np.float64)
    if (
        origin.shape != (2,)
        or spacing.shape != (2,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(spacing) & (spacing > 0))
    ):
        raise ValueError("Invalid declared source lattice")
    if manifest["layout"] == "quadrant_rectangle" and (
        np.any(origin != 0) or np.any(lower != 0)
    ):
        raise ValueError("Quadrant lattice must be anchored at zero with both axes")
    if not np.all((spacing >= 1e-6) & (spacing <= 1e6)) or not np.all(
        (origin == 0) | ((np.abs(origin) >= 1e-6) & (np.abs(origin) <= 1e6))
    ):
        raise ValueError("Source lattice outside supported numeric range")
    offsets = _native(fits.getdata(offsets_path))
    if offsets.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Offsets require float32 or float64")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("Nonfinite source offsets")
    scale = np.maximum(
        np.max(np.abs(offsets.astype(np.float64)), axis=0), np.abs(spacing)
    )
    rounded = scale.astype(offsets.dtype)
    tolerance = 4 * (
        np.nextafter(rounded, np.array(np.inf, offsets.dtype)) - rounded
    ).astype(np.float64)
    if np.any(tolerance >= 1e-4 * np.abs(spacing)):
        raise ValueError("Offset uncertainty exceeds pitch cap")
    expected = origin + spacing * indices
    canonical_origin = origin + spacing * lower
    canonical = canonical_origin + spacing * (indices - lower)
    for actual in (csv_offsets, offsets.astype(np.float64)):
        _agree(actual, expected, "authoritative training coordinates", tolerance)
        _agree(actual, canonical, "canonical lattice", tolerance)
    if not np.all(
        (canonical_origin == 0)
        | ((np.abs(canonical_origin) >= 1e-6) & (np.abs(canonical_origin) <= 1e6))
    ):
        raise ValueError("Canonical lattice outside supported numeric range")
    residual = np.max(np.abs(offsets - canonical), axis=0)
    _header_checks(header, manifest, plane, tolerance)
    measure = manifest["sample_kind"]
    if measure not in ("integral", "mean"):
        raise ValueError("Only integral and mean pixel samples are supported")
    conversion = float(plane["photon_conversion"])
    if not np.isfinite(conversion) or conversion <= 0:
        raise ValueError("Photon conversion must be positive and known")
    if manifest["normalization"]["kind"] == "incident_photon" and conversion != 1:
        raise ValueError("Incident photon normalization requires photon_conversion=1")
    if measure == "mean":
        conversion *= float(np.prod(manifest["pixel_scale_lod_xy"]))
    if not np.isfinite(conversion) or conversion <= 0:
        raise ValueError("Invalid pixel-area photon conversion")
    nodes = _native(fits.getdata(psf_path))
    if nodes.dtype != np.dtype(manifest["storage_dtype"]):
        raise ValueError("Conflicting storage_dtype")
    for start in range(0, nodes.size, 65536):
        block = nodes.reshape(-1)[start : start + 65536]
        if not _valid_intensity(block):
            raise ValueError("Invalid negative or nonfinite intensity data")
    for record in (
        manifest.get("reflection"),
        manifest.get("validation"),
        manifest.get("stamp_validation"),
    ):
        if record:
            status = record.get("status", record.get("check_status"))
            if status in ("pass", "fail") and not all(
                record.get(key) for key in ("report_path", "report_sha256")
            ):
                raise ValueError("Completed validation requires a hashed report")
            for field in ("report", "refinement_report"):
                if record.get(field + "_path"):
                    _check_hash(
                        root / record[field + "_path"],
                        record.get(field + "_sha256"),
                        required=True,
                    )
    independent = _independent_images(root, manifest, indices, tuple(nodes.shape[1:]))
    _validate_check_positions(manifest, canonical_origin, spacing, size, options.domain)
    for record in manifest.get("checks", []):
        reference = record.get("local_peak_reference")
        if reference is not None and not (0 < reference / conversion < np.inf):
            raise ValueError("DATA_INVALID: unrepresentable local_peak_reference")
    return Delivery(
        root,
        manifest,
        file_sha256(root / "sinc-delivery.json"),
        plane,
        replace(
            HeaderData.from_fits_header(header),
            pixscale=float(manifest["pixel_scale_lod_xy"][0]) * lod / u.pix,
            lambda0=float(manifest["lambda0_m"]) * u.m,
            diameter=float(manifest["diameter_m"]) * u.m,
            minlam=float(plane["wavelength_m"]) * u.m,
            maxlam=float(plane["wavelength_m"]) * u.m,
        ),
        nodes,
        offsets,
        (indices - lower).astype(np.int32),
        canonical_origin,
        tuple(int(n) for n in size[::-1]),
        tolerance,
        residual,
        conversion,
        independent,
        _missing_metadata(manifest, plane),
    )


def admit_delivery(delivery, options, sc):
    """Map validated delivery declarations to the installed sinchole API."""
    d, p = delivery.manifest, delivery.plane
    f64 = lambda value: jnp.asarray(value, dtype=jnp.float64)  # noqa: E731
    pixels = sc.PixelGrid(
        tuple(d["detector_shape_yx"]),
        f64(d["detector_center_xy"]),
        jnp.diag(f64(d["pixel_scale_lod_xy"])),
        "pixel_integral" if d["sample_kind"] == "integral" else "pixel_mean_density",
    )
    premises = [d.get("premises", {}).get(key, {}) for key in _PREMISES]
    norm = d["normalization"]
    if not norm.get("source_independent", False):
        premises[4] = dict(premises[4], value=False)
    products = sc.Products(
        sc.Frame(
            f64(float(d["lambda0_m"]) * 1e9),
            f64(d["diameter_m"]),
            d["diameter_convention"],
            d["coordinate_model"],
        ),
        pixels,
        sc.Normalization(
            norm["kind"],
            f64([p["photon_conversion"]]),
            f64(
                [
                    p.get("reference_uncertainty")
                    if p.get("reference_uncertainty") is not None
                    else np.nan
                ]
            ),
            (norm.get("included_losses", ""),),
            json.dumps(norm, sort_keys=True),
        ),
        p["kind"],
        f64([p["wavelength_m"] * 1e9]),
        f64([[p["wavelength_m"] * 1e9] * 2]),
        f64([p["sampling_min_wavelength_m"] * 1e9]),
        f64(
            np.nan
            if d["spectral_sampling"]["campaign_sampling_min_m"] is None
            else d["spectral_sampling"]["campaign_sampling_min_m"] * 1e9
        ),
        f64([d.get("support_extent_D", [np.nan, np.nan])]),
        jnp.array([1 if d.get("support_evidence") else 0], jnp.int32),
        (d.get("support_evidence", ""),),
        sc.Premises(
            jnp.array([r.get("value", False) for r in premises], bool),
            jnp.array([r.get("evidence", 0) for r in premises], jnp.int32),
            tuple(r.get("source_id", "") for r in premises),
        ),
        hashlib.sha256(
            json.dumps(d["optical_state"], sort_keys=True).encode()
        ).hexdigest(),
        "monochromatic",
        delivery.manifest_sha256 + ":" + p["id"],
    )
    lattice = sc.Lattice(
        f64(delivery.origin_lod), f64(d["source_spacing_lod"]), delivery.shape_yx
    )
    served = d["served"]
    domain = sc.Domain(
        served["kind"],
        served["boundary"],
        f64(
            np.array(served["bounds_lod"]).T
            if served["kind"] == "box"
            else np.zeros((2, 2))
        ),
        f64(served["radii_lod"] if served["kind"] == "annulus" else [0.0, 0.0]),
    )
    if options.domain is not None:
        domain = options.domain
    reflection = d.get("reflection")
    mirror = None
    if d["layout"] == "quadrant_rectangle":
        if (
            not reflection
            or not all(
                reflection.get(key)
                for key in (
                    "x_mirror",
                    "y_mirror",
                    "declaration_id",
                    "response_argument",
                    "optical_state_id",
                    "scope",
                    "report_sha256",
                )
            )
            or reflection.get("response_argument_kind")
            not in (
                "component_reflections",
                "mirror_plus_conjugation",
                "other_response",
            )
            or reflection.get("check_tolerance_local_peak") != 1e-6
            or reflection.get("check_status") != "pass"
            or reflection.get("optical_state_id") != d["optical_state"].get("state_id")
        ):
            raise sc.AdmissionRefused(
                sc.Admission(None, sc.Receipt(), int(sc.Reason.SYMMETRY_UNSUPPORTED))
            )
        reflection = sc.Reflection(
            reflection["kind"],
            jnp.array([True, True]),
            reflection["declaration_id"],
            hashlib.sha256(
                json.dumps(
                    {
                        key: reflection[key]
                        for key in (
                            "response_argument_kind",
                            "response_argument",
                            "optical_state_id",
                        )
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            jnp.int32(1 if reflection.get("evidence") == "designer_declared" else 0),
            reflection["report_sha256"],
            jnp.array([False]),
            f64(reflection["check_tolerance_local_peak"]),
        )
        records = [r for r in d.get("mirror_checks", []) if r["plane_id"] == p["id"]]
        if not records or not all(r.get("independent_propagation") for r in records):
            raise sc.AdmissionRefused(
                sc.Admission(None, sc.Receipt(), int(sc.Reason.SYMMETRY_CHECK_MISSING))
            )
        coords = np.array([r["source_lod_xy"] for r in records], np.float64)
        signed_indices = []
        for record, coord in zip(records, coords, strict=True):
            index = record["base_index"]
            if type(index) is not int or not 0 <= index < len(delivery.indices):
                raise ValueError("Mirror base_index must identify a training row")
            base = delivery.indices[index].astype(np.int64)
            signed_indices.append(base * np.where(coord < 0, -1, 1))
        mirror = sc.MirrorCheckSet(
            f64(coords),
            jnp.array(signed_indices, jnp.int32),
            jnp.asarray(delivery.read_images(records)),
            jnp.zeros(len(records), jnp.int32),
            delivery.manifest_sha256,
        )
    else:
        if reflection is not None:
            raise ValueError("Full rectangle requires reflection=null")
        reflection = sc.Reflection(
            "none",
            jnp.array([False, False]),
            "",
            "",
            jnp.int32(0),
            "",
            jnp.array([False]),
        )
    records = [r for r in d.get("checks", []) if r["plane_id"] == p["id"]]
    checks = None
    if records:
        if not all(
            r.get("independent_propagation") and r.get("reference_units")
            for r in records
        ):
            raise ValueError(
                "Checks require independent images and explicit reference_units"
            )
        peaks = []
        for record in records:
            reference = record.get("local_peak_reference")
            if record["reference_units"] == "incident_fraction_per_pixel":
                peaks.append(
                    np.nan if reference is None else reference / delivery.conversion
                )
            else:
                raise ValueError("Unsupported independent check reference_units")
        checks = sc.CheckSet(
            f64([r["source_lod_xy"] for r in records]),
            jnp.asarray(delivery.read_images(records)),
            jnp.zeros(len(records), jnp.int32),
            f64(peaks),
            delivery.manifest_sha256,
            "independent_images",
            p["id"],
        )
    memory = options.memory
    if memory is None:
        memory = delivery_memory(delivery, sc)
    admission = sc.admit_grid(
        delivery.nodes[None],
        delivery.offsets,
        declared_lattice=lattice,
        indices_xy=jnp.asarray(delivery.indices),
        products=products,
        reflection=reflection,
        domain=domain,
        checks=checks,
        mirror_checks=mirror,
        memory=memory,
        offset_original_dtype=delivery.offsets.dtype,
        provenance=(("manifest", json.dumps(d, sort_keys=True)),),
    )
    return admission, domain, memory


def _validate_declarations(d):
    """Reject malformed physical declarations before optional admission gates."""
    norm = d["normalization"]
    if not isinstance(norm, dict) or norm.get("kind") not in (
        "incident_photon",
        "unocculted_peak",
    ):
        raise ValueError("Unknown or missing normalization reference")
    for key in ("lambda0_m", "diameter_m"):
        value = float(d[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid physical geometry: " + key)
    support = d.get("support_extent_D")
    if support is not None:
        support = np.asarray(support, np.float64)
        if support.shape != (2,) or not np.all(np.isfinite(support) & (support > 0)):
            raise ValueError("Support extents require two finite positive values")
    for entry in (d.get("premises") or {}).values():
        if not isinstance(entry, dict):
            raise ValueError("Premises require keyed evidence records")
        if entry.get("value") is False:
            raise ValueError("METADATA_CONFLICT: explicitly false physical premise")
        if entry.get("evidence") == 2:
            raise ValueError(
                "Externally established premises require verified external evidence"
            )
    reflection = d.get("reflection")
    if (
        d["layout"] == "quadrant_rectangle"
        and reflection is not None
        and reflection.get("kind") != "two_axis_pixel_flip_v1"
    ):
        raise ValueError("Unsupported quadrant reflection kind")


def _missing_metadata(d, p):
    """List absent physical claims without substituting geometry or units."""
    missing = [
        key
        for key in (
            "coordinate_model",
            "diameter_convention",
            "optical_state",
            "support_extent_D",
            "support_evidence",
            "served",
        )
        if d.get(key) is None or d.get(key) == ""
    ]
    for key in _PREMISES:
        entry = (d.get("premises") or {}).get(key) or {}
        if any(
            field not in entry or entry[field] is None
            for field in ("value", "evidence", "source_id")
        ):
            missing.append("premises." + key)
    if p.get("sampling_min_wavelength_m") is None:
        missing.append("sampling_min_wavelength_m")
    if "campaign_sampling_min_m" not in d["spectral_sampling"]:
        missing.append("campaign_sampling_min_m")
    return tuple(missing)


def _validate_domain(served, override):
    """Validate permission declarations and restrictions without core objects."""

    def values(kind, boundary, limits, radii):
        if boundary not in ("closed", "lower_closed_upper_open"):
            raise ValueError("Unsupported domain boundary")
        data = np.asarray(limits if kind == "box" else radii, np.float64)
        if kind not in ("box", "annulus") or data.shape != (
            (2, 2) if kind == "box" else (2,)
        ):
            raise ValueError("Unsupported served domain geometry")
        normal = (data == 0) | (np.abs(data) >= np.finfo(np.float64).tiny)
        if not np.all(np.isfinite(data) & normal) or np.any(np.abs(data) > 1e6):
            raise ValueError("Domain outside supported numeric range")
        if kind == "box":
            ordered = data[0] <= data[1] if boundary == "closed" else data[0] < data[1]
            if not np.all(ordered):
                raise ValueError("Box limits must be ordered and nonempty")
        elif not (
            0 <= data[0] < data[1]
            and data[1] >= 1e-6
            and (data[0] == 0 or data[0] >= 1e-6)
        ):
            raise ValueError(
                "Annulus requires ordered radii in supported numeric range"
            )
        return data

    if served is None:
        if override is not None:
            values(
                override.kind,
                override.boundary,
                override.limits_xy_lod,
                override.radii_lod,
            )
        return
    kind, boundary = served["kind"], served["boundary"]
    declared = values(
        kind,
        boundary,
        np.array(served["bounds_lod"]).T if kind == "box" else None,
        served.get("radii_lod"),
    )
    if override is not None:
        if override.kind != kind or override.boundary != boundary:
            raise ValueError(
                "Domain override must restrict declared geometry and boundary"
            )
        restricted = values(kind, boundary, override.limits_xy_lod, override.radii_lod)
        subset = (
            (
                np.all(restricted[0] >= declared[0])
                and np.all(restricted[1] <= declared[1])
            )
            if kind == "box"
            else (restricted[0] >= declared[0] and restricted[1] <= declared[1])
        )
        if not subset:
            raise ValueError("Domain override cannot expand served permission")


def _independent_images(root, manifest, indices, shape):
    """Hash/open each independent file once, validating exact authoritative rows."""
    records = [*manifest.get("checks", []), *manifest.get("mirror_checks", [])]
    grouped = {}
    ids = {p["id"] for p in manifest["planes"]}
    for record in records:
        if record["plane_id"] not in ids:
            raise ValueError("Independent image references unknown plane_id")
        coord = np.asarray(record["source_lod_xy"], np.float64)
        if coord.shape != (2,) or not np.all(np.isfinite(coord)):
            raise ValueError("Independent coordinates require finite xy")
        if "base_index" in record:
            index = record["base_index"]
            if type(index) is not int or not 0 <= index < len(indices):
                raise ValueError("Mirror base_index must identify a training row")
        grouped.setdefault(record["psf_path"], []).append(record)
    images = {}
    for name, entries in grouped.items():
        hashes = {record.get("psf_sha256") for record in entries}
        if len(hashes) != 1:
            raise ValueError("Conflicting independent image hashes")
        digest = next(iter(hashes))
        _check_hash(root / name, digest, required=True)
        with fits.open(root / name, memmap=True) as hdus:
            cube = hdus[0].data
            if cube.ndim != 3 or cube.shape[1:] != shape or cube.dtype.kind != "f":
                raise ValueError("Independent images require matching floating cube")
            for record in entries:
                index = record["image_index"]
                if type(index) is not int or not 0 <= index < len(cube):
                    raise ValueError("Independent image_index must identify a cube row")
                image = _native(cube[index])
                if not _valid_intensity(image):
                    raise ValueError("Invalid independent intensity data")
                images[(name, digest, index)] = image
    for record in manifest.get("checks", []):
        reference = record.get("local_peak_reference")
        if reference is not None and not (0 < reference < np.inf):
            raise ValueError(
                "DATA_INVALID: local_peak_reference must be positive and finite"
            )
        if (
            not record.get("independent_propagation")
            or record.get("reference_units") != "incident_fraction_per_pixel"
        ):
            raise ValueError(
                "Checks require independent images and explicit reference_units"
            )
    return images


def _validate_check_positions(manifest, origin, spacing, size, override):
    """Reject empirical checks that the core would refuse as DATA_INVALID."""
    served = manifest.get("served")
    for record in manifest.get("checks", []):
        query = np.asarray(record["source_lod_xy"], np.float64)
        query = np.where(np.abs(query) < np.finfo(np.float64).tiny, 0.0, query)
        if served is not None or override is not None:
            kind = override.kind if override is not None else served["kind"]
            boundary = override.boundary if override is not None else served["boundary"]
            if kind == "box":
                bounds = (
                    np.asarray(override.limits_xy_lod)
                    if override is not None
                    else np.asarray(served["bounds_lod"]).T
                )
                lower, upper = query >= bounds[0], query <= bounds[1]
                if boundary != "closed":
                    upper = query < bounds[1]
            else:
                radii = (
                    np.asarray(override.radii_lod)
                    if override is not None
                    else np.asarray(served["radii_lod"])
                )
                radius2 = np.sum(query * query)
                lower, upper = radius2 >= radii[0] ** 2, radius2 <= radii[1] ** 2
                if boundary != "closed":
                    upper = radius2 < radii[1] ** 2
            if not (np.all(lower) and np.all(upper)):
                raise ValueError(
                    "DATA_INVALID: independent check outside served domain"
                )
        # Same exact-quotient floor as the core, after the rounded subtraction.
        difference = query - origin
        remainder = np.fmod(difference, spacing)
        floor = np.rint((difference - remainder) / spacing) - (
            (remainder != 0) & (difference < 0)
        )
        lower = 16 - size if manifest["layout"] == "quadrant_rectangle" else 15
        if not np.all(np.isfinite(floor) & (floor >= lower) & (floor <= size - 17)):
            raise ValueError("DATA_INVALID: independent check outside stencil coverage")


def campaign_sampling_failed(delivery):
    """Evaluate the explicit campaign test with the core's separate binary64 steps."""
    d = delivery.manifest
    minimum = d["spectral_sampling"].get("campaign_sampling_min_m")
    if minimum is None or d.get("support_extent_D") is None:
        return False
    # Match the core's nm frame and operation order, including rounded conversions.
    sigma = float(d["lambda0_m"] * 1e9) / float(minimum * 1e9)
    for extent, pitch in zip(
        d["support_extent_D"], d["source_spacing_lod"], strict=True
    ):
        bandwidth = sigma * float(extent)
        rho = (2.0 * bandwidth) * float(pitch)
        if not rho < 1.0 or not rho <= 0.76:
            return True
    return False


def delivery_memory(delivery, sc, supplied=None):
    """Isolate the private core allocation-model coupling for admission and reads."""
    from sinchole.reader import estimate_memory

    estimates = [
        estimate_memory(
            delivery.nodes.nbytes,
            delivery.nodes.shape[1:],
            sc.Tiling(),
            1,
            admission=True,
        )
    ]
    if supplied is not None:
        for key in ("resident", "output", "scratch", "total"):
            if estimates[0][key] > getattr(supplied, "max_" + key + "_bytes"):
                raise sc.MemoryLimit(key + " estimate exceeds supplied budget")
        return supplied
    estimates.append(
        estimate_memory(
            delivery.nodes.nbytes, delivery.nodes.shape[1:], sc.Tiling(), 1024
        )
    )
    ceilings = {key: max(item[key] for item in estimates) for key in estimates[0]}
    ceilings["output"] += 1024 * 4
    ceilings["total"] += 1024 * 4
    return sc.MemoryBudget(
        *(int(ceilings[key]) for key in ("resident", "output", "scratch", "total"))
    )
