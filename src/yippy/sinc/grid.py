"""Immutable intensity products and host admission of complete source lattices."""

import hashlib
import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf
from hwoutils.precision import float_dtype
from optixstuff.grids import Measure, PixelGrid

from yippy.sinc._types import (
    FLOAT_DTYPES,
    I32,
    Boolean,
    RealArray,
    check_array,
    check_static,
)
from yippy.sinc.admission import (
    MIRROR_TOLERANCE,
    index_reasons,
    lattice_reasons,
    offset_evidence,
    sampling_reasons,
    spectral_reasons,
    valid_intensity,
)
from yippy.sinc.domain import in_supported_range, safe_query
from yippy.sinc.evidence import Admission, Reason, Receipt
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import KernelProfile, PrefilteredKernelProfile
from yippy.sinc.prefilter import design_grid, design_record, sampling_ratios


def _host_float(value):
    """Keep host declarations in binary64 independently of JAX availability."""
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


class Normalization(eqx.Module):
    """Source-independent product conversions; reads retain delivered units.

    Attributes:
        reference: Name of the normalization reference.
        photon_conversion: Conversion from delivered units to photons, one value
            per product, shape (n_products,).
        reference_uncertainty_fraction: Fractional uncertainty of the reference,
            shape (n_products,).
        included_factors: Names of the factors included in the conversion.
        provenance_id: Identifier of the provenance record.
    """

    reference: str = eqx.field(static=True)
    photon_conversion: np.ndarray = eqx.field(converter=_host_float)
    reference_uncertainty_fraction: np.ndarray = eqx.field(converter=_host_float)
    included_factors: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(
            self.photon_conversion, (self.photon_conversion.size,), (jnp.float64,)
        )
        check_array(
            self.reference_uncertainty_fraction,
            self.photon_conversion.shape,
            (jnp.float64,),
        )


class Premises(eqx.Module):
    """Six ordered physical premises, evidence codes and provenance sources.

    Attributes:
        values: Truth value of each premise, shape (6,).
        evidence: Evidence code of each premise, shape (6,).
        source_ids: Six provenance identifiers, one per premise.
    """

    values: Boolean = eqx.field(converter=jnp.asarray)
    evidence: I32 = eqx.field(converter=jnp.asarray)
    source_ids: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.values, (6,), (jnp.bool_,))
        check_array(self.evidence, (6,), (jnp.int32,))
        if not isinstance(self.source_ids, tuple) or len(self.source_ids) != 6:
            raise ValueError("Premises require six provenance IDs")


class Products(eqx.Module):
    """Exact monochromatic or fixed band-mean products with explicit support.

    Attributes:
        pixels: Detector pixel grid, in lambda/D with a frame.
        normalization: Product conversions.
        kind: Spectral kind, "monochromatic" or "band_mean".
        wavelength_nm: Central wavelength per product, in nm, shape (n_products,).
        limits_nm: Lower and upper wavelength limit per product, in nm, shape
            (n_products, 2).
        sampling_min_nm: Shortest wavelength used for sampling checks per
            product, in nm, shape (n_products,).
        campaign_sampling_min_nm: Campaign minimum wavelength, in nm; NaN if
            none was declared. Scalar.
        support_extent_xy: Support extents (x, y) per product, shape
            (n_products, 2).
        support_evidence: Evidence code of the support per product, shape
            (n_products,).
        support_source_ids: Provenance identifier of the support per product.
        premises: Physical premises of the delivery.
        optical_id: Identifier of the optical model.
        spectral_weight_id: Identifier of the spectral weighting.
        identity: Identity string of the delivery.
        measure: Measure of the intensity values, integral by default.
    """

    pixels: PixelGrid
    normalization: Normalization
    kind: str = eqx.field(static=True)
    wavelength_nm: np.ndarray = eqx.field(converter=_host_float)
    limits_nm: np.ndarray = eqx.field(converter=_host_float)
    sampling_min_nm: np.ndarray = eqx.field(converter=_host_float)
    campaign_sampling_min_nm: np.ndarray = eqx.field(converter=_host_float)
    support_extent_xy: np.ndarray = eqx.field(converter=_host_float)
    support_evidence: I32 = eqx.field(converter=jnp.asarray)
    support_source_ids: tuple[str, ...] = eqx.field(static=True)
    premises: Premises
    optical_id: str = eqx.field(static=True)
    spectral_weight_id: str = eqx.field(static=True)
    identity: str = eqx.field(static=True)
    measure: Measure = eqx.field(static=True, default=Measure.integral())

    @property
    def frame(self):
        """Reference frame carried by the detector grid."""
        return self.pixels.frame

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if not isinstance(self.measure, Measure):
            raise TypeError("products require a Measure")
        if self.measure.kind == "density":
            self.pixels.cell_area(self.measure.area)
        if self.pixels.unit != "lod" or self.pixels.frame is None:
            raise ValueError("Products require a lod grid with a frame")
        count = self.wavelength_nm.size
        if count < 1:
            raise ValueError("At least one product is required")
        for value in (
            self.wavelength_nm,
            self.sampling_min_nm,
            self.normalization.photon_conversion,
        ):
            check_array(value, (count,), (jnp.float64,))
        check_array(self.limits_nm, (count, 2), (jnp.float64,))
        check_array(self.support_extent_xy, (count, 2), (jnp.float64,))
        check_array(self.support_evidence, (count,), (jnp.int32,))
        check_array(self.campaign_sampling_min_nm, (), (jnp.float64,))
        if len(self.support_source_ids) != count:
            raise ValueError("Support source count differs from product count")


class Reflection(eqx.Module):
    """Declared two-axis detector flips; admission computes check_passed itself.

    Reflection requires 0 < tolerance <= MIRROR_TOLERANCE of the local peak.
    Axis evidence records exact native-value identity separately from relative
    residuals. An axis image whose source lies more than 1 lambda/D from every
    pixel center (a halo node beyond the detector frame) has no local peak, so
    its residual is relative to the image's own maximum, which is below the
    off-frame peak and therefore stricter. Without a positive denominator, only
    exact identity passes.

    Attributes:
        kind: Reflection kind, "none" or "two_axis_pixel_flip_v1".
        axes: Which source axes are reflected, shape (2,) as (x, y).
        declaration_id: Identifier of the reflection declaration.
        argument_id: Identifier of the supporting argument.
        evidence: Evidence code of the declaration, scalar.
        check_report_id: Identifier of the check report.
        check_passed: Whether the mirror checks passed, one per product.
        tolerance: Largest allowed mirror residual relative to the local peak,
            dimensionless; defaults to MIRROR_TOLERANCE.
    """

    kind: str = eqx.field(static=True)
    axes: Boolean = eqx.field(converter=jnp.asarray)
    declaration_id: str = eqx.field(static=True)
    argument_id: str = eqx.field(static=True)
    evidence: I32 = eqx.field(converter=jnp.asarray)
    check_report_id: str = eqx.field(static=True)
    check_passed: Boolean = eqx.field(converter=jnp.asarray)
    tolerance: np.ndarray = eqx.field(
        default_factory=lambda: _host_float(MIRROR_TOLERANCE), converter=_host_float
    )

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.axes, (2,), (jnp.bool_,))
        check_array(self.evidence, (), (jnp.int32,))
        check_array(self.check_passed, (self.check_passed.size,), (jnp.bool_,))
        check_array(self.tolerance, (), (jnp.float64,))


class SampledGrid(eqx.Module):
    """Canonical native float32/64 samples [product, sy, sx, row, column].

    ``profile_id`` names the kernel profile the grid was admitted under; its
    sampling margin, coverage and empirical checks hold for that profile only.
    An empty value marks a grid built outside admission.

    ``stored`` names what ``intensity`` holds: ``"samples"``, the admitted node
    images, or ``"prefiltered_coefficients"``, the coefficient table a
    PrefilteredReader builds from them and keeps in their place. Readers are
    constructed only from samples.

    Attributes:
        intensity: Stored values, shape (product, sy, sx, row, column), in the
            delivered units, float32 or float64. Samples for a prefiltered
            reader stay on the host at delivered precision until filtering.
        source: Source-node pixel grid, in lambda/D.
        products: Products the samples belong to.
        reflection: Reflection declaration.
        receipt_id: Identifier of the admission receipt.
        profile_id: Identifier of the kernel profile admitted under; empty if
            built outside admission.
        stored: What ``intensity`` holds, "samples" or
            "prefiltered_coefficients".
    """

    intensity: jax.Array | np.ndarray
    source: PixelGrid
    products: Products
    reflection: Reflection
    receipt_id: str = eqx.field(static=True)
    _empirical_max_abs: RealArray = eqx.field(
        default_factory=lambda: jnp.asarray(jnp.nan)
    )
    _has_checks: bool = eqx.field(static=True, default=False)
    profile_id: str = eqx.field(static=True, default="")
    stored: str = eqx.field(static=True, default="samples")

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if self.stored not in ("samples", "prefiltered_coefficients"):
            raise ValueError("stored must be 'samples' or 'prefiltered_coefficients'")
        count = self.products.wavelength_nm.size
        check_array(
            self.intensity,
            (count, *self.source.shape, *self.products.pixels.shape),
            (jnp.float32, jnp.float64),
        )
        check_array(self.reflection.check_passed, (count,), (jnp.bool_,))
        check_array(self._empirical_max_abs, (), FLOAT_DTYPES)


class CheckSet(eqx.Module):
    """Independent off-node images and optional independently supplied core peaks.

    Attributes:
        source_xy_lod: Check source coordinates, shape (n_checks, 2), in lambda/D.
        intensity: Independent check images, shape (n_checks, row, column), in
            the delivered units.
        product_index: Product of each check, shape (n_checks,).
        core_peak: Independently supplied core peak per check, shape
            (n_checks,), in the delivered units.
        provenance_id: Identifier of the provenance record.
        reference_method_id: Identifier of the reference method.
        target_id: Identifier of the error target.
    """

    source_xy_lod: np.ndarray = eqx.field(converter=_host_float)
    intensity: np.ndarray = eqx.field(converter=np.asarray)
    product_index: I32 = eqx.field(converter=jnp.asarray)
    core_peak: np.ndarray = eqx.field(converter=_host_float)
    provenance_id: str = eqx.field(static=True)
    reference_method_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.product_index.size
        check_array(self.source_xy_lod, (count, 2), (jnp.float64,))
        check_array(self.product_index, (count,), (jnp.int32,))
        check_array(self.core_peak, (count,), (jnp.float64,))
        if self.intensity.ndim != 3:
            raise ValueError("Check images require [query, row, column]")
        check_array(
            self.intensity,
            (count, *self.intensity.shape[1:]),
            (jnp.float32, jnp.float64),
        )


class MirrorCheckSet(eqx.Module):
    """Independent signed node images; indices are signed about the zero anchor.

    Each product must include x-negative, y-negative and both-negative checks.
    These finite comparisons are empirical evidence, never a global bound.

    Attributes:
        source_xy_lod: Check source coordinates, shape (n_checks, 2), in lambda/D.
        indices_xy: Signed node indices (x, y), shape (n_checks, 2).
        intensity: Independent node images, shape (n_checks, row, column), in
            the delivered units.
        product_index: Product of each check, shape (n_checks,).
        provenance_id: Identifier of the provenance record.
    """

    source_xy_lod: np.ndarray = eqx.field(converter=_host_float)
    indices_xy: I32 = eqx.field(converter=jnp.asarray)
    intensity: np.ndarray = eqx.field(converter=np.asarray)
    product_index: I32 = eqx.field(converter=jnp.asarray)
    provenance_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.product_index.size
        check_array(self.source_xy_lod, (count, 2), (jnp.float64,))
        check_array(self.indices_xy, (count, 2), (jnp.int32,))
        check_array(self.product_index, (count,), (jnp.int32,))
        if self.intensity.ndim != 3:
            raise ValueError("Mirror images require [query, row, column]")
        check_array(
            self.intensity,
            (count, *self.intensity.shape[1:]),
            (jnp.float32, jnp.float64),
        )


def _positive(values):
    return bool(np.all(np.isfinite(values) & (values > 0)))


def _metadata_reasons(products, reflection):
    reasons = Reason(0)
    pixels, norm = products.pixels, products.normalization
    premises = products.premises
    if (
        not bool(np.all((premises.evidence >= 1) & (premises.evidence <= 2)))
        or not all(premises.source_ids)
        or not all(
            (products.identity, products.optical_id, products.spectral_weight_id)
        )
    ):
        reasons |= Reason.METADATA_MISSING
    if not bool(np.all(premises.values[:4])):
        reasons |= Reason.FRAME_UNSUPPORTED
    if not bool(premises.values[5]):
        reasons |= Reason.METADATA_CONFLICT
    if not bool(premises.values[4]):
        reasons |= Reason.NORMALIZATION_UNSUPPORTED
    basis = np.asarray(pixels.basis)
    norms = np.sum(basis * basis, axis=0)
    dot = float(np.sum(basis[:, 0] * basis[:, 1]))
    orthogonal = abs(dot) <= 8 * math.ulp(1.0) * math.sqrt(float(np.prod(norms)))
    if (
        pixels.unit != "lod"
        or not _positive(norms)
        or not bool(np.all(np.isfinite(basis)))
        or not bool(np.all(np.isfinite(np.asarray((pixels.center_x, pixels.center_y)))))
        or not orthogonal
    ):
        reasons |= Reason.FRAME_UNSUPPORTED
    if reflection.kind == "two_axis_pixel_flip_v1":
        expected = np.asarray([(n - 1) / 2 for n in pixels.shape[::-1]])
        if (
            not bool(np.all(np.asarray((pixels.center_x, pixels.center_y)) == expected))
            or float(basis[0, 1]) != 0
            or float(basis[1, 0]) != 0
        ):
            reasons |= Reason.FRAME_UNSUPPORTED
    if (
        norm.reference not in ("incident_photon", "unocculted_peak")
        or not _positive(norm.photon_conversion)
        or not norm.provenance_id
    ):
        reasons |= Reason.NORMALIZATION_UNSUPPORTED
    uncertainty = norm.reference_uncertainty_fraction
    if not bool(
        np.all(np.isnan(uncertainty) | (np.isfinite(uncertainty) & (uncertainty >= 0)))
    ):
        reasons |= Reason.NORMALIZATION_UNSUPPORTED
    if (
        not _positive(products.support_extent_xy)
        or not bool(np.all(products.support_evidence == 1))
        or not all(products.support_source_ids)
    ):
        reasons |= Reason.UNKNOWN_BANDWIDTH
    for wavelength, limits, minimum in zip(
        products.wavelength_nm.tolist(),
        products.limits_nm.tolist(),
        products.sampling_min_nm.tolist(),
        strict=True,
    ):
        reasons |= spectral_reasons(
            products.kind,
            wavelength,
            limits,
            minimum,
            float(products.campaign_sampling_min_nm),
        )
    return reasons


def _local_peak(image, source, coordinates):
    selected = np.sum((coordinates - source[:, None, None]) ** 2, axis=0) <= 1.0
    return float(np.max(np.where(selected, image, -np.inf)))


def _symmetry_checks(cube, lattice, products, reflection, checks):
    reasons = Reason(0)
    count = cube.shape[0]
    passed = [True] * count
    residuals, axis_residuals, axis_exact = [], [], []
    tolerance = float(reflection.tolerance)
    with jax.enable_x64():  # precision-ok: host-only detector coordinates
        coordinates = np.asarray(products.pixels.coordinates())
    declared = (
        bool(np.all(reflection.axes))
        and int(reflection.evidence) in (1, 2)
        and all(
            (
                reflection.declaration_id,
                reflection.argument_id,
                reflection.check_report_id,
            )
        )
        and math.isfinite(tolerance)
        and 0 < tolerance <= MIRROR_TOLERANCE
    )
    if not declared:
        return Reason.SYMMETRY_UNSUPPORTED, jnp.zeros(count, dtype=bool), (), (), ()
    if checks is None or checks.product_index.size == 0:
        reasons |= Reason.SYMMETRY_CHECK_MISSING
    else:
        if checks.intensity.shape[1:] != products.pixels.shape:
            raise ValueError("Mirror detector shape differs from samples")
        categories = [set() for _ in range(count)]
        valid_offsets, _, _ = offset_evidence(
            checks.source_xy_lod,
            origin_xy(lattice),
            spacing_xy(lattice),
            checks.indices_xy,
        )
        valid = (
            valid_offsets
            and bool(np.all(np.isfinite(checks.intensity) & (checks.intensity >= 0)))
            and bool(np.all(np.isfinite(checks.source_xy_lod)))
            and bool(checks.provenance_id)
        )
        if not valid:
            reasons |= Reason.SYMMETRY_REFERENCE_INVALID
        for index in range(checks.product_index.size):
            product = int(checks.product_index[index])
            ix, iy = checks.indices_xy[index].tolist()
            if not (
                0 <= product < count
                and abs(ix) < cube.shape[2]
                and abs(iy) < cube.shape[1]
                and (ix < 0 or iy < 0)
            ):
                reasons |= Reason.SYMMETRY_REFERENCE_INVALID
                continue
            categories[product].add((ix < 0, iy < 0))
            reference = np.asarray(checks.intensity[index], dtype=np.float64)
            predicted = np.asarray(cube[product, abs(iy), abs(ix)], dtype=np.float64)
            if ix < 0:
                predicted = predicted[:, ::-1]
            if iy < 0:
                predicted = predicted[::-1, :]
            peak = _local_peak(reference, checks.source_xy_lod[index], coordinates)
            if not math.isfinite(peak) or peak <= 0:
                reasons |= Reason.SYMMETRY_REFERENCE_INVALID
                passed[product] = False
                residuals.append(math.inf)
                continue
            residual = float(np.max(np.abs(reference - predicted))) / peak
            residuals.append(residual)
            if not residual <= tolerance:
                reasons |= Reason.SYMMETRY_CHECK_FAILED
                passed[product] = False
        if any(
            not {(True, False), (False, True), (True, True)} <= group
            for group in categories
        ):
            reasons |= Reason.SYMMETRY_CHECK_MISSING
    for product in range(count):
        for axis, size in ((0, cube.shape[1]), (1, cube.shape[2])):
            chunk = max(1, min(32, 65536 // math.prod(products.pixels.shape)))
            for start in range(0, size, chunk):
                stop = min(size, start + chunk)
                images = (
                    cube[product, start:stop, 0]
                    if axis == 0
                    else cube[product, 0, start:stop]
                )
                indices = np.arange(start, stop, dtype=np.float64)
                zeros = np.zeros_like(indices)
                source = np.stack(
                    (zeros, indices) if axis == 0 else (indices, zeros), axis=1
                )
                source = source * spacing_xy(lattice)
                exact, relative = _axis_residuals(images, source, coordinates, axis)
                values = relative.tolist()
                axis_residuals.extend(values)
                identities = exact.tolist()
                axis_exact.extend(identities)
                if any(
                    not (same or value <= tolerance)
                    for same, value in zip(identities, values, strict=True)
                ):
                    reasons |= Reason.SYMMETRY_CHECK_FAILED
                    passed[product] = False
    return (
        reasons,
        jnp.asarray(passed),
        tuple(residuals),
        tuple(axis_residuals),
        tuple(axis_exact),
    )


def _axis_residuals(images, sources, coordinates, axis):
    integer = np.int32 if images.dtype == np.float32 else np.int64
    bits = images.view(integer)
    # Numeric identity allows both signed zeros but preserves subnormal samples.
    bits = np.where((bits & np.iinfo(integer).max) == 0, 0, bits)
    flipped_bits = bits[:, :, ::-1] if axis == 0 else bits[:, ::-1, :]
    exact = np.all(bits == flipped_bits, axis=(1, 2))
    minimum_normal = np.finfo(images.dtype).tiny
    images = np.asarray(images, dtype=np.float64)
    selected = (
        np.sum((coordinates[None] - sources[:, :, None, None]) ** 2, axis=1) <= 1.0
    )
    peaks = np.where(
        np.any(selected, axis=(1, 2)),
        np.max(np.where(selected, images, -np.inf), axis=(1, 2)),
        # Sources beyond the frame have no in-stamp peak; the image maximum is
        # below theirs, so the relative residual is the stricter one.
        np.max(images, axis=(1, 2)),
    )
    flipped = images[:, :, ::-1] if axis == 0 else images[:, ::-1, :]
    differences = np.max(np.abs(images - flipped), axis=(1, 2))
    positive = peaks >= minimum_normal
    relative = differences / np.where(positive, peaks, 1.0)
    # Exact invariance does not require an in-stamp relative denominator.
    return exact, np.where(positive, relative, np.nan)


def _digest(*values):
    digest = hashlib.sha256()
    for value in values:
        leaves, tree = jax.tree_util.tree_flatten(value)
        digest.update(str(tree).encode())
        for leaf in leaves:
            if eqx.is_array(leaf):
                digest.update(str((leaf.shape, leaf.dtype)).encode())
                flat = leaf.reshape(-1)
                for start in range(0, flat.size, 65536):
                    digest.update(jax.device_get(flat[start : start + 65536]).tobytes())
            else:
                digest.update(repr(leaf).encode())
    return digest.hexdigest()


_DEFAULT_PROFILE = KernelProfile()


def admit_grid(
    intensity_nodes,
    offsets_xy_lod,
    *,
    declared_lattice,
    indices_xy,
    products,
    reflection,
    domain,
    profile=_DEFAULT_PROFILE,
    checks=None,
    memory,
    mirror_checks=None,
    offset_original_dtype=None,
    provenance=(),
):
    """Admit complete canonical samples, accumulating semantic refusal reasons.

    Args:
        intensity_nodes: Native float32/64 array [P,N,Py,Px].
        offsets_xy_lod: Converted float32/64 source offsets [N,2].
        declared_lattice: Authoritative origin, spacing and shape; never fitted.
        indices_xy: Rebased int32 source indices [N,2], in arbitrary row order.
        products: Physical frame, support, premises and normalization.
        reflection: Explicit full-grid or two-axis reflection declaration.
        domain: Signed served permission region intersected with stencil coverage.
        profile: Raw Kaiser-sinc profile (default) or a prefiltered profile. The
            profile sets the sampling margin, the smallest table, coverage and
            the reader used for ``checks``; the samples stored are raw either way.
        checks: Optional independent off-node empirical comparisons.
        memory: Explicit resident, output, scratch and total byte limits.
        mirror_checks: Required independent signed node checks for quadrants.
        offset_original_dtype: Original float32/64 precision before conversion;
            None uses offsets_xy_lod.dtype. Four ULP are evaluated in this dtype
            at the converted coordinate scale, without recasting the offsets.
        provenance: Tuple of (key, text) string pairs retained verbatim in the
            host receipt and hashed into its identity; never traced.

    Returns:
        An Admission with grid=None on semantic refusal. Admission.require()
        raises AdmissionRefused (or a resource-specific subclass) on refusal.
        Structural Python shape/dtype mistakes raise TypeError or ValueError.
    """
    # Local import breaks the intentional admission/reader construction cycle.
    from yippy.sinc.reader import (
        Interpolator,
        PrefilteredReader,
        ReadPolicy,
        Tiling,
        estimate_memory,
    )

    if not isinstance(provenance, tuple) or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(value, str) for value in pair)
        for pair in provenance
    ):
        raise TypeError("Provenance requires a tuple of (key, text) string pairs")
    if not hasattr(intensity_nodes, "shape") or len(intensity_nodes.shape) != 4:
        raise ValueError("Nodes require [product, source, row, column]")
    allocations = estimate_memory(
        intensity_nodes.nbytes,
        products.pixels.shape,
        Tiling(),
        1,
        admission=True,
        profile=profile,
    )
    if any(
        allocations[key] > getattr(memory, f"max_{key}_bytes")
        for key in ("resident", "output", "scratch", "total")
    ):
        receipt = Receipt(
            allocation_estimate=tuple(allocations.items()), provenance=provenance
        )
        return Admission(None, receipt, int(Reason.MEMORY_LIMIT))
    nodes = np.asarray(intensity_nodes)
    offsets = np.asarray(offsets_xy_lod)
    indices = np.asarray(indices_xy)
    offset_dtype = (
        offsets.dtype
        if offset_original_dtype is None
        else np.dtype(offset_original_dtype)
    )
    if offset_dtype not in FLOAT_DTYPES:
        raise TypeError("Original offsets require float32 or float64 precision")
    if nodes.ndim != 4:
        raise ValueError("Nodes require [product, source, row, column]")
    count, rows = nodes.shape[:2]
    check_array(
        nodes,
        (products.wavelength_nm.size, rows, *products.pixels.shape),
        (jnp.float32, jnp.float64),
    )
    check_array(offsets, (rows, 2), (jnp.float32, jnp.float64))
    check_array(indices, (rows, 2), (jnp.int32,))
    check_array(reflection.check_passed, (count,), (jnp.bool_,))
    reasons = _metadata_reasons(products, reflection)
    lattice = declared_lattice
    origin, spacing = origin_xy(lattice).tolist(), spacing_xy(lattice).tolist()
    geometry_reasons = lattice_reasons(origin, spacing, lattice.basis)
    if lattice.unit != "lod" or lattice.frame != products.frame:
        geometry_reasons |= Reason.FRAME_UNSUPPORTED
    reasons |= geometry_reasons
    geometry_valid = not geometry_reasons
    domain_valid = in_supported_range(domain)
    if not domain_valid:
        reasons |= Reason.METADATA_CONFLICT
    reflected = reflection.kind == "two_axis_pixel_flip_v1"
    if not 0 < float(reflection.tolerance) <= MIRROR_TOLERANCE:
        reasons |= Reason.SYMMETRY_UNSUPPORTED
    if reflection.kind not in ("none", "two_axis_pixel_flip_v1"):
        reasons |= Reason.LAYOUT_UNSUPPORTED | Reason.SYMMETRY_UNSUPPORTED
    if reflected and any(v != 0 for v in origin):
        reasons |= Reason.GRID_NONUNIFORM
    if not reflected and (bool(jnp.any(reflection.axes)) or mirror_checks is not None):
        reasons |= Reason.SYMMETRY_UNSUPPORTED
    sy, sx = lattice.shape
    # Numerical buffers keep occupancy validation bounded by the delivery size.
    # Never allocate the declared rectangle when its size exceeds delivered rows.
    linear = indices[:, 1].astype(np.int64) * sx + indices[:, 0]
    reasons |= index_reasons(indices, lattice.shape)
    if min(sy, sx) < profile.minimum_size(reflected):
        reasons |= Reason.INCOMPLETE_STENCIL
    tau, residual = (), ()
    if rows and geometry_valid:
        uniform, tau, residual = offset_evidence(
            offsets, origin_xy(lattice), spacing_xy(lattice), indices, offset_dtype
        )
        if not uniform:
            reasons |= Reason.GRID_NONUNIFORM
        reasons |= sampling_reasons(
            (rho for _, rho in sampling_ratios(products, spacing)), profile
        )
    flat_nodes = nodes.reshape(-1)
    for start in range(0, flat_nodes.size, 65536):
        block = flat_nodes[start : start + 65536]
        if not valid_intensity(block):
            reasons |= Reason.DATA_INVALID
    allocations = estimate_memory(
        nodes.nbytes,
        products.pixels.shape,
        Tiling(),
        1,
        admission=True,
        profile=profile,
    )
    if any(
        allocations[key] > getattr(memory, f"max_{key}_bytes")
        for key in ("resident", "output", "scratch", "total")
    ):
        reasons |= Reason.MEMORY_LIMIT
    if (
        geometry_valid
        and domain_valid
        and safe_query(domain, lattice, reflected, profile) is None
    ):
        reasons |= Reason.INCOMPLETE_STENCIL
    receipt = Receipt(
        profile_id=profile.identity,
        execution_profile=f"jax-{jax.__version__}:{jax.default_backend()}:{float_dtype()}",
        offset_dtype=str(offset_dtype),
        provenance=provenance,
        offset_tolerance_xy=tau,
        canonicalization_residual_xy=residual,
        stored_bounds_xy=(
            tuple(origin),
            tuple(
                o + (n - 1) * h
                for o, n, h in zip(origin, (sx, sy), spacing, strict=True)
            ),
        ),
        served_geometry=(
            domain.kind,
            domain.boundary,
            tuple(map(tuple, domain.limits_xy_lod.tolist())),
            tuple(domain.radii_lod.tolist()),
        ),
        premise_evidence=tuple(
            zip(
                products.premises.values.tolist(),
                products.premises.evidence.tolist(),
                products.premises.source_ids,
                strict=True,
            )
        ),
        provenance_ids=(
            products.identity,
            products.optical_id,
            products.spectral_weight_id,
            reflection.declaration_id,
            reflection.argument_id,
            reflection.check_report_id,
        ),
        allocation_estimate=tuple(allocations.items()),
    )
    if reasons:
        return Admission(None, receipt, int(reasons))
    prefiltered = isinstance(profile, PrefilteredKernelProfile)
    if prefiltered:
        designs = design_grid(profile, products, lattice, reflected, domain)
        receipt = replace(receipt, kernel_design=design_record(designs))
    order = np.argsort(linear)
    cube = nodes[:, order].reshape(count, sy, sx, *products.pixels.shape)
    if reflected:
        symmetry_reasons, passed, mirror, axes, exact = _symmetry_checks(
            cube, lattice, products, reflection, mirror_checks
        )
        reasons |= symmetry_reasons
        reflection = eqx.tree_at(lambda r: r.check_passed, reflection, passed)
        receipt = replace(
            receipt,
            mirror_local_peak=mirror,
            axis_relative=axes,
            axis_exact=exact,
            provenance_ids=(
                *receipt.provenance_ids,
                mirror_checks.provenance_id if mirror_checks else "",
            ),
        )
    if reasons:
        return Admission(None, receipt, int(reasons))
    identity = _digest(
        cube,
        offsets,
        indices,
        lattice,
        products,
        reflection,
        domain,
        profile,
        checks,
        mirror_checks,
        str(offset_dtype),
        provenance,
    )
    receipt = replace(receipt, identity=identity)
    if prefiltered:
        cube.setflags(write=False)
        resident = cube
    elif cube.dtype == np.float64 and float_dtype() == np.float32:
        resident = as_float_leaf(cube)
    else:
        resident = jnp.asarray(cube)
    grid = SampledGrid(
        resident, lattice, products, reflection, identity, profile_id=profile.identity
    )
    if checks is not None:
        if checks.intensity.shape[1:] != products.pixels.shape:
            raise ValueError("Check detector shape differs from samples")
        valid = (
            checks.product_index.size > 0
            and all(
                (checks.provenance_id, checks.reference_method_id, checks.target_id)
            )
            and bool(np.all(np.isfinite(checks.intensity) & (checks.intensity >= 0)))
            and bool(
                np.all(
                    np.isnan(checks.core_peak)
                    | (np.isfinite(checks.core_peak) & (checks.core_peak > 0))
                )
            )
        )
        if not valid:
            return Admission(None, receipt, int(Reason.DATA_INVALID))
        reader_type = PrefilteredReader if prefiltered else Interpolator
        reader = reader_type(grid, domain, profile, ReadPolicy(), Tiling(), memory)
        absolute, relative = [], []
        for index in range(checks.product_index.size):
            result = reader.read(
                checks.source_xy_lod[index], checks.product_index[index]
            )
            if not bool(result.accepted):
                return Admission(
                    None, receipt, int(result.reason_bits) | int(Reason.DATA_INVALID)
                )
            error = float(
                np.max(
                    np.abs(
                        np.asarray(result.intensity, dtype=np.float64)
                        - checks.intensity[index]
                    )
                )
            )
            absolute.append(error)
            relative.append(error / float(checks.core_peak[index]))
        empirical = max(absolute)
        receipt = replace(
            receipt,
            empirical_absolute=tuple(absolute),
            empirical_local_peak=tuple(relative),
            empirical_max_abs=empirical,
            provenance_ids=(
                *receipt.provenance_ids,
                checks.provenance_id,
                checks.reference_method_id,
                checks.target_id,
            ),
        )
        grid = SampledGrid(
            resident,
            lattice,
            products,
            reflection,
            identity,
            jnp.asarray(empirical),
            True,
            profile.identity,
        )
    return Admission(grid, receipt, 0)
