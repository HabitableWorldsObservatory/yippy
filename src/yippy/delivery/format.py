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
import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from hwoutils.constants import m2nm, rad2arcsec
from lod_unit import lod
from optixstuff.grids import Frame, Measure, PixelGrid

from yippy.sinc.admission import (
    MIRROR_TOLERANCE,
    index_reasons,
    lattice_reasons,
    offset_evidence,
    sampling_ratio,
    sampling_reasons,
    select_profile,
    spectral_reasons,
    valid_intensity,
    validate_declarations,
)
from yippy.sinc.domain import Domain, restrict_domain, validate_check_positions
from yippy.sinc.evidence import Admission, AdmissionRefused, Reason, Receipt
from yippy.sinc.geometry import source_grid
from yippy.sinc.grid import (
    CheckSet,
    MirrorCheckSet,
    Normalization,
    Premises,
    Products,
    Reflection,
    admit_grid,
)
from yippy.sinc.kernel import KernelProfile, PrefilteredKernelProfile
from yippy.sinc.reader import MemoryBudget

from ..header import HeaderData
from ..sampling import sample_units

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
    grid: PixelGrid
    nodes: np.ndarray
    offsets: np.ndarray
    indices: np.ndarray
    origin_lod: np.ndarray
    shape_yx: tuple[int, int]
    tolerance_lod: np.ndarray
    coordinate_residual_lod: np.ndarray
    measure: Measure
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


def _agree(actual, expected, label, tolerance):
    if not np.all(np.isfinite(actual)) or np.any(np.abs(actual - expected) > tolerance):
        raise ValueError(f"Conflicting {label} in delivery metadata or FITS header")


def _header_checks(header, manifest, plane, tolerance_lod):
    frame = Frame(
        float(manifest["lambda0_m"]) * m2nm,
        float(manifest["diameter_m"]),
        "circumscribed",
    )
    angular_arcsec = frame.lod_to_arcsec(frame.reference_wavelength_nm)
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
    grid = HeaderData.from_fits_header(header).pixel_grid(shape)
    _agree(np.array([grid.center_x, grid.center_y]), center, "detector center", 0.0)
    if manifest.get("detector_convention") != "pixel_midpoint_v1" or not np.array_equal(
        center, (np.array(shape[::-1]) - 1) / 2
    ):
        raise ValueError("Incompatible midpoint detector frame")


def select_plane(manifest, selected):
    """Resolve exactly one named monochromatic plane in a versioned delivery."""
    planes = manifest["planes"]
    ids = [p["id"] for p in planes]
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Delivery plane IDs must be nonempty and unique")
    if selected is None:
        if len(planes) != 1:
            raise ValueError("sinc_options.product_id is required for multiple planes")
        selected = ids[0]
    if selected not in ids:
        raise ValueError(f"Unknown product_id: {selected}")
    return planes[ids.index(selected)]


def delivery_header(header, manifest, plane):
    """Resolve the selected plane's geometry in the manifest reference frame."""
    return replace(
        HeaderData.from_fits_header(header),
        sample_kind=manifest["sample_kind"],
        pixscale=float(manifest["pixel_scale_lod_xy"][0]) * lod / u.pix,
        lambda0=float(manifest["lambda0_m"]) * m2nm * u.nm,
        diameter_convention=manifest.get("diameter_convention") or "circumscribed",
        coordinate_model=manifest.get("coordinate_model") or "paraxial_entrance_tilt",
        diameter=float(manifest["diameter_m"]) * u.m,
        minlam=float(plane["wavelength_m"]) * u.m,
        maxlam=float(plane["wavelength_m"]) * u.m,
    )


def load_delivery(root, manifest, options, *, package=None):
    """Validate selected metadata, hashes, coordinates and native FITS arrays.

    Args:
        root: Delivery directory.
        manifest: Recognized versioned manifest.
        options: SincOptions selecting exactly one monochromatic plane.
        package: Open native package providing its measured arrays and header cards.

    Returns:
        A Delivery ready for yippy.sinc construction, with no legacy transforms.
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
    plane = select_plane(manifest, options.product_id)
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
        "half_width": KernelProfile().taps // 2,
        "beta": 12,
        "max_occupancy": KernelProfile().margin_limit,
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
    campaign = spectral.get("campaign_sampling_min_m")
    for p in planes:
        minimum = p.get("sampling_min_wavelength_m")
        reasons = spectral_reasons(
            p["kind"],
            p["wavelength_m"] * m2nm,
            [v * m2nm for v in p["band_edges_m"]],
            np.nan if minimum is None else minimum * m2nm,
            np.nan if campaign is None else campaign * m2nm,
        )
        if (
            p["kind"] != "monochromatic"
            or p["shortest_contributing_m"] != p["wavelength_m"]
            or reasons & Reason.METADATA_CONFLICT
        ):
            raise ValueError("METADATA_CONFLICT: monochromatic wavelength metadata")
    if spectral.get("band_ready") and campaign is None:
        raise ValueError("Band-ready campaign requires a sampling minimum")
    validate_declarations(
        manifest["normalization"],
        manifest["lambda0_m"] * m2nm,
        manifest["diameter_m"],
        manifest.get("support_extent_D"),
        manifest.get("premises"),
        (manifest.get("reflection") or {}).get("kind"),
        manifest["layout"] == "quadrant_rectangle",
    )
    _domain(manifest.get("served"), options.domain)
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
    header = (
        package.fits_header("offaxis_psfs") if package else fits.getheader(psf_path)
    )
    if header["NAXIS"] != 3 or header["BITPIX"] not in (-32, -64):
        raise ValueError("PSF cube requires native float32/float64 [source,y,x]")
    count = header["NAXIS3"]
    offsets_header = (
        package.fits_header("offsets_xy_lod")
        if package
        else fits.getheader(offsets_path)
    )
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
    lower, upper = indices.min(axis=0), indices.max(axis=0)
    size = upper - lower + 1
    flags = index_reasons(indices - lower, tuple(size[::-1]))
    if flags & Reason.GRID_DUPLICATE:
        raise ValueError("Duplicate training nodes")
    if flags & Reason.GRID_INCOMPLETE:
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
    if lattice_reasons(origin, spacing):
        raise ValueError("Source lattice outside supported numeric range")
    offsets = _native(package.offsets_xy_lod if package else fits.getdata(offsets_path))
    if offsets.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Offsets require float32 or float64")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("Nonfinite source offsets")
    canonical_origin = origin + spacing * lower
    if lattice_reasons(canonical_origin, spacing):
        raise ValueError("Canonical lattice outside supported numeric range")
    for actual in (csv_offsets, offsets):
        for start, index in ((origin, indices), (canonical_origin, indices - lower)):
            valid, tolerance, residual = offset_evidence(
                actual, start, spacing, index, offsets.dtype
            )
            if not valid:
                raise ValueError(
                    "Conflicting authoritative training coordinates "
                    "or canonical lattice"
                )
    tolerance, residual = np.asarray(tolerance), np.asarray(residual)
    _header_checks(header, manifest, plane, tolerance)
    kind = manifest["sample_kind"]
    if "SAMPKIND" in header and header["SAMPKIND"] != kind:
        raise ValueError("Conflicting FITS SAMPKIND and manifest sample_kind")
    metadata = delivery_header(header, manifest, plane)
    grid = metadata.pixel_grid()
    measure, factor = sample_units(kind, grid)
    conversion = float(plane["photon_conversion"])
    if not np.isfinite(conversion) or conversion <= 0:
        raise ValueError("Photon conversion must be positive and known")
    if manifest["normalization"]["kind"] == "incident_photon" and conversion != 1:
        raise ValueError("Incident photon normalization requires photon_conversion=1")
    # The status ledger and error policy are expressed in fractions per pixel.
    if measure == Measure.density("grid"):
        conversion *= grid.cell_area("grid")
    if not np.isfinite(conversion) or conversion <= 0:
        raise ValueError("Invalid pixel-area photon conversion")
    nodes = _native(package.offaxis_psfs if package else fits.getdata(psf_path))
    if nodes.dtype != np.dtype(manifest["storage_dtype"]):
        raise ValueError("Conflicting storage_dtype")
    for start in range(0, nodes.size, 65536):
        block = nodes.reshape(-1)[start : start + 65536]
        if not valid_intensity(block):
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
    if factor != 1.0:
        # _native makes private, native-endian copies of the FITS data.
        with np.errstate(over="raise", invalid="raise"):
            try:
                nodes *= factor
                for image in independent.values():
                    image *= factor
            except FloatingPointError as err:
                raise ValueError(
                    "Decoded intensity exceeds the storage precision"
                ) from err
    _validate_check_positions(manifest, canonical_origin, spacing, size)
    for record in manifest.get("checks", []):
        reference = record.get("local_peak_reference")
        if reference is not None and not (0 < reference / conversion < np.inf):
            raise ValueError("DATA_INVALID: unrepresentable local_peak_reference")
    return Delivery(
        root,
        manifest,
        file_sha256(root / "sinc-delivery.json"),
        plane,
        metadata,
        grid,
        nodes,
        offsets,
        (indices - lower).astype(np.int32),
        canonical_origin,
        tuple(int(n) for n in size[::-1]),
        tolerance,
        residual,
        measure,
        conversion,
        independent,
        _missing_metadata(manifest, plane),
    )


def admit_delivery(delivery, options):
    """Map validated delivery declarations to the installed yippy.sinc API."""
    d, p = delivery.manifest, delivery.plane
    f64 = lambda value: jnp.asarray(value, dtype=jnp.float64)  # noqa: E731
    premises = [d.get("premises", {}).get(key, {}) for key in _PREMISES]
    norm = d["normalization"]
    if not norm.get("source_independent", False):
        premises[4] = dict(premises[4], value=False)
    products = Products(
        delivery.grid,
        Normalization(
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
        f64([p["wavelength_m"] * m2nm]),
        f64([[p["wavelength_m"] * m2nm] * 2]),
        f64([p["sampling_min_wavelength_m"] * m2nm]),
        f64(
            np.nan
            if d["spectral_sampling"]["campaign_sampling_min_m"] is None
            else d["spectral_sampling"]["campaign_sampling_min_m"] * m2nm
        ),
        f64([d.get("support_extent_D", [np.nan, np.nan])]),
        jnp.array([1 if d.get("support_evidence") else 0], jnp.int32),
        (d.get("support_evidence", ""),),
        Premises(
            jnp.array([r.get("value", False) for r in premises], bool),
            jnp.array([r.get("evidence", 0) for r in premises], jnp.int32),
            tuple(r.get("source_id", "") for r in premises),
        ),
        hashlib.sha256(
            json.dumps(d["optical_state"], sort_keys=True).encode()
        ).hexdigest(),
        "monochromatic",
        delivery.manifest_sha256 + ":" + p["id"],
        measure=delivery.measure,
    )
    lattice = source_grid(
        delivery.origin_lod,
        d["source_spacing_lod"],
        delivery.shape_yx,
        delivery.grid.frame,
    )
    domain = _domain(d["served"], options.domain)
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
            or reflection.get("check_tolerance_local_peak") != MIRROR_TOLERANCE
            or reflection.get("check_status") != "pass"
            or reflection.get("optical_state_id") != d["optical_state"].get("state_id")
        ):
            raise AdmissionRefused(
                Admission(None, Receipt(), int(Reason.SYMMETRY_UNSUPPORTED))
            )
        reflection = Reflection(
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
            raise AdmissionRefused(
                Admission(None, Receipt(), int(Reason.SYMMETRY_CHECK_MISSING))
            )
        coords = np.array([r["source_lod_xy"] for r in records], np.float64)
        signed_indices = []
        for record, coord in zip(records, coords, strict=True):
            index = record["base_index"]
            if type(index) is not int or not 0 <= index < len(delivery.indices):
                raise ValueError("Mirror base_index must identify a training row")
            base = delivery.indices[index].astype(np.int64)
            signed_indices.append(base * np.where(coord < 0, -1, 1))
        mirror = MirrorCheckSet(
            f64(coords),
            jnp.array(signed_indices, jnp.int32),
            jnp.asarray(delivery.read_images(records)),
            jnp.zeros(len(records), jnp.int32),
            delivery.manifest_sha256,
        )
    else:
        if reflection is not None:
            raise ValueError("Full rectangle requires reflection=null")
        reflection = Reflection(
            "none",
            jnp.array([False, False]),
            "",
            "",
            jnp.int32(0),
            "",
            jnp.array([False]),
        )
    records = [r for r in d.get("checks", []) if r["plane_id"] == p["id"]]
    # Checks outside a caller's narrower domain cannot be read; exclude them
    # from the empirical comparison and keep their identifiers in provenance.
    excluded = [
        r["id"] for r in records if not bool(domain.contains(f64(r["source_lod_xy"])))
    ]
    records = [r for r in records if r["id"] not in excluded]
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
        checks = CheckSet(
            f64([r["source_lod_xy"] for r in records]),
            jnp.asarray(delivery.read_images(records)),
            jnp.zeros(len(records), jnp.int32),
            f64(peaks),
            delivery.manifest_sha256,
            "independent_images",
            p["id"],
        )
    profile = delivery_profile(delivery, options)
    memory = delivery_memory(delivery, options.memory, profile=profile)
    admission = admit_grid(
        delivery.nodes[None],
        delivery.offsets,
        declared_lattice=lattice,
        indices_xy=jnp.asarray(delivery.indices),
        products=products,
        reflection=reflection,
        domain=domain,
        profile=profile,
        checks=checks,
        mirror_checks=mirror,
        memory=memory,
        offset_original_dtype=delivery.offsets.dtype,
        provenance=(
            ("manifest", json.dumps(d, sort_keys=True)),
            ("excluded_checks", json.dumps(excluded)),
        ),
    )
    return admission, domain, memory


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


def _domain(served, override=None):
    """Translate file-domain fields and delegate numeric permission checks."""
    with jax.enable_x64():
        declared = (
            None
            if served is None
            else Domain(
                served["kind"],
                served["boundary"],
                np.asarray(served["bounds_lod"]).T
                if served["kind"] == "box"
                else np.zeros((2, 2)),
                served["radii_lod"] if served["kind"] == "annulus" else np.zeros(2),
            )
        )
        return restrict_domain(declared, override)


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
                if not valid_intensity(image):
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


def _validate_check_positions(manifest, origin, spacing, size):
    """Check delivered references against the declared raw-table permission."""
    with jax.enable_x64():
        frame = Frame(
            manifest["lambda0_m"] * m2nm,
            manifest["diameter_m"],
            manifest.get("diameter_convention") or "circumscribed",
        )
        lattice = source_grid(origin, spacing, tuple(size[::-1]), frame)
        validate_check_positions(
            [r["source_lod_xy"] for r in manifest.get("checks", [])],
            _domain(manifest.get("served")),
            lattice,
            manifest["layout"] == "quadrant_rectangle",
        )


def delivery_ratios(delivery, *, campaign_only=False):
    """Translate manifest units for the engine's sampling-ratio calculation."""
    d, p = delivery.manifest, delivery.plane
    minima = [d["spectral_sampling"].get("campaign_sampling_min_m")]
    if not campaign_only:
        minima.append(p.get("sampling_min_wavelength_m"))
    support = d.get("support_extent_D")
    if support is None:
        return ()
    return tuple(
        sampling_ratio(d["lambda0_m"] * m2nm, minimum * m2nm, extent, pitch)
        for minimum in minima
        if minimum is not None
        for extent, pitch in zip(support, d["source_spacing_lod"], strict=True)
    )


def campaign_sampling_failed(delivery):
    """Ask the engine whether the explicit campaign claim meets admission."""
    return bool(sampling_reasons(delivery_ratios(delivery, campaign_only=True)))


def delivery_profile(delivery, options):
    """Pass the declared band and requested tap count to engine selection."""
    return select_profile(delivery_ratios(delivery), taps=options.taps)


def delivery_memory(delivery, supplied=None, *, profile=None):
    """Pass table sizes and yippy's output tag allocation to the engine."""
    return MemoryBudget.for_table(
        delivery.nodes.nbytes,
        delivery.nodes.shape[1:],
        profile=profile,
        supplied=supplied,
        extra_output_bytes=1024 * 4,
        extra_resident_bytes=delivery.nodes.nbytes
        if isinstance(profile, PrefilteredKernelProfile)
        else 0,
    )
