"""File-free host checks shared by loaders, planners and grid admission."""

import math
from decimal import Decimal

import numpy as np

from yippy.sinc.evidence import Reason
from yippy.sinc.kernel import KernelProfile, PrefilteredKernelProfile

MIRROR_TOLERANCE = 1e-6


def offset_evidence(offsets, origin, spacing, indices, original_dtype=None):
    """Return validity, four-ULP tolerances and canonical residuals per axis.

    The tolerance is evaluated in the original float32/64 coordinate precision
    and must be strictly below 1e-4 of a pitch. Separate binary64 operations
    define canonical coordinates, without fused multiply-add.

    Args:
        offsets: Delivered node offsets, shape (n_nodes, 2), in the source
            coordinate unit.
        origin: Declared coordinate (x, y) of the first node.
        spacing: Declared node pitches (x, y).
        indices: Integer node indices (x, y), shape (n_nodes, 2).
        original_dtype: Precision the offsets were stored in. If None, the
            dtype of ``offsets`` is used. Must be float32 or float64.

    Returns:
        Tuple of the overall validity flag, the per-axis tolerances (tuple of
        two floats, same unit as offsets) and the per-axis worst residuals
        between delivered and canonical coordinates (tuple of two floats).

    Raises:
        TypeError: If the original precision is not float32 or float64.
    """
    offsets = np.asarray(offsets)
    dtype = offsets.dtype if original_dtype is None else np.dtype(original_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Original offsets require float32 or float64 precision")
    spacing = np.asarray(spacing, np.float64)
    scale = np.maximum(
        np.max(np.abs(offsets.astype(np.float64)), axis=0), np.abs(spacing)
    ).astype(dtype)
    tolerance = 4 * (np.nextafter(scale, np.array(np.inf, dtype)) - scale).astype(
        np.float64
    )
    expected = (
        np.asarray(origin, np.float64) + np.asarray(indices, np.float64) * spacing
    )
    residual = np.max(np.abs(offsets.astype(np.float64) - expected), axis=0)
    valid = (
        np.all(np.isfinite(offsets))
        and np.all(residual <= tolerance)
        and np.all(tolerance < 1e-4 * np.abs(spacing))
    )
    return bool(valid), tuple(tolerance), tuple(residual)


def sampling_ratio(reference_nm, minimum_nm, support_extent, spacing_lod):
    """Compute 2 B h with separate binary64 wavelength and bandwidth rounding.

    Args:
        reference_nm: Reference wavelength, in nm.
        minimum_nm: Shortest wavelength in the band, in nm.
        support_extent: Support width of the product along one axis, in
            cycles per (lambda/D) at the reference wavelength.
        spacing_lod: Node pitch along the same axis, in lambda/D.

    Returns:
        Sampling ratio 2 B h, dimensionless; values below 1 satisfy Nyquist.
    """
    sigma = float(reference_nm) / float(minimum_nm)
    bandwidth = sigma * float(support_extent)
    return (2.0 * bandwidth) * float(spacing_lod)


def sampling_reasons(ratios, profile=None):
    """Return literal Nyquist and profile-margin failures, without tolerance.

    Args:
        ratios: Sampling ratios (dimensionless), one per axis.
        profile: Kernel profile supplying the margin limit. If None, the
            default raw profile is used.

    Returns:
        Reason flags with the Nyquist and kernel-margin failure bits set where
        any ratio fails; empty when all pass.
    """
    profile = KernelProfile() if profile is None else profile
    reasons = Reason(0)
    for ratio in ratios:
        if not ratio < 1.0:
            reasons |= Reason.NYQUIST_FAIL
        if not ratio <= profile.margin_limit:
            reasons |= Reason.KERNEL_MARGIN_FAIL
    return reasons


def select_profile(ratios, *, taps=None):
    """Choose prefiltering at load time when its sampling margin permits it.

    None uses the default prefilter tap count, falling back to raw for bands
    above the prefilter margin. An explicit 5..10 tap count requests that
    prefiltered profile; admission then reports any margin failure.

    Args:
        ratios: Sampling ratios (dimensionless), one per axis.
        taps: Prefilter tap count, 5 to 10, or None for the default behavior
            described above.

    Returns:
        The kernel profile to use, prefiltered or raw.
    """
    candidate = (
        PrefilteredKernelProfile()
        if taps is None
        else PrefilteredKernelProfile(taps=taps)
    )
    return (
        KernelProfile()
        if taps is None and sampling_reasons(ratios, candidate)
        else candidate
    )


def plan_spacing(reference_nm, minimum_nm, support_extent, decimals=6, *, profile=None):
    """Largest decimal-rounded isotropic spacing allowed by the profile margin.

    Args:
        reference_nm: Reference wavelength, in nm.
        minimum_nm: Shortest wavelength in the band, in nm; must satisfy
            0 < minimum_nm <= reference_nm.
        support_extent: Support widths along the two axes; all must be
            positive and finite. The longest one sets the spacing.
        decimals: Number of decimal places the spacing is rounded down to.
        profile: Kernel profile supplying the margin limit. If None, the
            default raw profile is used.

    Returns:
        Node pitch in lambda/D.

    Raises:
        ValueError: If the wavelengths or support widths are invalid, or the
            spacing rounds to zero.
    """
    profile = KernelProfile() if profile is None else profile
    if not (
        math.isfinite(reference_nm)
        and math.isfinite(minimum_nm)
        and 0 < minimum_nm <= reference_nm
    ):
        raise ValueError("need 0 < minimum <= reference wavelength")
    if not all(math.isfinite(v) and v > 0 for v in support_extent):
        raise ValueError("support widths must be positive")
    longest = max(support_extent)
    limit = profile.margin_limit / sampling_ratio(
        reference_nm, minimum_nm, longest, 1.0
    )
    unit = Decimal(1).scaleb(-decimals)
    spacing = float((Decimal(repr(limit)) // unit) * unit)
    while (
        sampling_ratio(reference_nm, minimum_nm, longest, spacing)
        > profile.margin_limit
    ):
        spacing = float(Decimal(repr(spacing)) - unit)
    if spacing <= 0:
        raise ValueError("spacing underflows the requested decimals")
    return spacing


def judge_checks(
    interior,
    mirror,
    axis,
    exact,
    *,
    reflected,
    acceptance_level,
    mirror_tolerance=MIRROR_TOLERANCE,
):
    """Return empirical-check failures; these residuals are not image bounds.

    Args:
        interior: Worst relative interior residual per check.
        mirror: Worst relative mirror residual per check.
        axis: Worst relative axis residual per axis.
        exact: Flags, one per axis, marking axes whose residual is exactly
            satisfied by construction and so is not tested.
        reflected: Whether the product is a reflected quadrant, which enables
            the mirror and axis tests.
        acceptance_level: Largest allowed interior residual.
        mirror_tolerance: Largest allowed mirror and axis residual.

    Returns:
        Tuple of failure messages; empty when every check passes.
    """
    failures = []
    if not interior:
        failures.append("no interior checks were compared")
    elif not all(math.isfinite(v) for v in interior):
        failures.append("an interior check has no positive local peak reference")
    elif max(interior) > acceptance_level:
        failures.append(
            f"worst interior residual {max(interior):.3e} "
            f"exceeds {acceptance_level:.0e}"
        )
    if reflected:
        if not mirror:
            failures.append("no mirror checks were compared")
        elif not all(math.isfinite(v) and v <= mirror_tolerance for v in mirror):
            worst = max((v for v in mirror if math.isfinite(v)), default=math.nan)
            failures.append(
                f"mirror residual {worst:.3e} exceeds {mirror_tolerance:.0e}"
            )
        for value, is_exact in zip(axis, exact, strict=True):
            if not is_exact and not (
                math.isfinite(value) and value <= mirror_tolerance
            ):
                failures.append(
                    f"axis residual {value:.3e} exceeds {mirror_tolerance:.0e}"
                )
                break
    return tuple(failures)


def lattice_reasons(origin, spacing, basis=None):
    """Check canonical source geometry and the kernel's supported numeric range.

    Args:
        origin: Coordinate (x, y) of the first node, in lambda/D.
        spacing: Node pitches (x, y), in lambda/D.
        basis: Optional 2 x 2 grid basis; off-diagonal terms mark a rotated or
            sheared grid. If None, no basis check is made.

    Returns:
        Reason flags for nonuniform or unsupported layout; empty when valid.
    """
    from yippy.sinc.domain import normal_or_zero

    reasons = Reason(0)
    if not all(normal_or_zero(v) for v in (*origin, *spacing)) or not all(
        h > 0 for h in spacing
    ):
        reasons |= Reason.GRID_NONUNIFORM
    if basis is not None and (basis[0][1] != 0 or basis[1][0] != 0):
        reasons |= Reason.GRID_NONUNIFORM
    if not (
        all(1e-6 <= h <= 1e6 for h in spacing)
        and all(v == 0 or 1e-6 <= abs(v) <= 1e6 for v in origin)
    ):
        reasons |= Reason.LAYOUT_UNSUPPORTED
    return reasons


def validate_declarations(
    normalization,
    reference_nm,
    diameter_m,
    support,
    premises,
    reflection_kind,
    reflected,
):
    """Reject malformed physical declarations before optional admission gates.

    Args:
        normalization: Mapping whose ``kind`` is ``incident_photon`` or
            ``unocculted_peak``.
        reference_nm: Reference wavelength, in nm; must be positive and finite.
        diameter_m: Telescope aperture diameter, in m; must be positive and
            finite.
        support: Optional support extents (two finite positive values).
        premises: Optional mapping of keyed evidence records for physical
            premises.
        reflection_kind: Declared quadrant reflection kind, or None.
        reflected: Whether the product is stored as a reflected quadrant.

    Raises:
        ValueError: If any declaration is malformed, conflicting or unverified.
    """
    norm = normalization
    if not isinstance(norm, dict) or norm.get("kind") not in (
        "incident_photon",
        "unocculted_peak",
    ):
        raise ValueError("Unknown or missing normalization reference")
    for key, raw in (("reference_nm", reference_nm), ("diameter_m", diameter_m)):
        value = float(raw)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid physical geometry: " + key)
    if support is not None:
        support = np.asarray(support, np.float64)
        if support.shape != (2,) or not np.all(np.isfinite(support) & (support > 0)):
            raise ValueError("Support extents require two finite positive values")
    for entry in (premises or {}).values():
        if not isinstance(entry, dict):
            raise ValueError("Premises require keyed evidence records")
        if entry.get("value") is False:
            raise ValueError("METADATA_CONFLICT: explicitly false physical premise")
        if entry.get("evidence") == 2:
            raise ValueError(
                "Externally established premises require verified external evidence"
            )
    if reflected and reflection_kind not in (None, "two_axis_pixel_flip_v1"):
        raise ValueError("Unsupported quadrant reflection kind")


def spectral_reasons(kind, wavelength_nm, limits_nm, minimum_nm, campaign_nm=math.nan):
    """Validate one product's spectral support and sampling declarations.

    Args:
        kind: Spectral kind, ``monochromatic`` or ``band_mean``.
        wavelength_nm: Central wavelength, in nm.
        limits_nm: Lower and upper wavelength limits of the product, in nm.
        minimum_nm: Shortest wavelength used for sampling checks, in nm.
        campaign_nm: Campaign minimum wavelength, in nm; NaN means none was
            declared.

    Returns:
        Reason flags for missing or conflicting metadata; empty when valid.
    """
    reasons = Reason(0)
    if not math.isfinite(minimum_nm) or minimum_nm <= 0:
        reasons |= Reason.METADATA_MISSING
    low, high = limits_nm
    if (
        kind not in ("monochromatic", "band_mean")
        or not all(math.isfinite(v) and v > 0 for v in (wavelength_nm, low, high))
        or not low <= wavelength_nm <= high
        or minimum_nm > wavelength_nm
        or (kind == "monochromatic" and (low != wavelength_nm or high != wavelength_nm))
        or (
            not math.isnan(campaign_nm)
            and (
                not math.isfinite(campaign_nm)
                or campaign_nm <= 0
                or campaign_nm > minimum_nm
            )
        )
    ):
        reasons |= Reason.METADATA_CONFLICT
    return reasons


def valid_intensity(array):
    """Check finite nonnegative native samples, including subnormal sign bits.

    Args:
        array: Intensity samples in their native float32 or float64 dtype.

    Returns:
        True if every sample is finite and not negative.
    """
    array = np.asarray(array)
    integer = np.int32 if array.dtype.itemsize == 4 else np.int64
    bits = array.view(integer)
    negative = (bits < 0) & ((bits & np.iinfo(integer).max) != 0)
    return bool(np.all(np.isfinite(array) & ~negative))


def index_reasons(indices_xy, shape_yx):
    """Check a complete rectangle without allocating its declared occupancy grid.

    Args:
        indices_xy: Integer node indices (x, y), shape (n_nodes, 2).
        shape_yx: Declared grid shape as (rows, columns).

    Returns:
        Reason flags for duplicate or incomplete indices; empty when the
        indices fill the rectangle exactly once.
    """
    indices = np.asarray(indices_xy)
    shape_xy = np.asarray(shape_yx[::-1])
    reasons = Reason(0)
    if len(np.unique(indices, axis=0)) != len(indices):
        reasons |= Reason.GRID_DUPLICATE
    if (
        reasons
        or len(indices) != math.prod(shape_yx)
        or not np.all((indices >= 0) & (indices < shape_xy))
    ):
        reasons |= Reason.GRID_INCOMPLETE
    return reasons
