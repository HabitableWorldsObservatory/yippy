"""Signed permission domains and independent full-stencil coverage."""

import math
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.precision import like

from yippy.sinc._types import check_static
from yippy.sinc.geometry import array_namespace, origin_xy, spacing_xy
from yippy.sinc.kernel import (
    KernelProfile,
    _float32_ceiling,
    flush_subnormal,
    lattice_phase_start,
)

_RAW_PROFILE = KernelProfile()
MIN_GEOMETRY_LOD = 1e-6  # tol: algorithm, 1e-6 lambda/D minimum nonzero geometry
MAX_GEOMETRY_LOD = 1e6


class Domain(eqx.Module):
    """A box or origin-centered annulus, without tolerance-expanded boundaries.

    Attributes:
        kind: Either "box" or "annulus".
        boundary: Either "closed" or "lower_closed_upper_open".

    The box limits (``limits_xy_lod``, shape (2, 2): row 0 the lower (x, y),
    row 1 the upper, in lambda/D, zero for an annulus) and the annulus radii
    (``radii_lod``, shape (2,), inner and outer in lambda/D, zero for a box)
    are host properties stored as immutable tuples.
    """

    kind: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    _limits: tuple = eqx.field(static=True)
    _radii: tuple = eqx.field(static=True)

    def __init__(self, kind, boundary, limits_xy_lod, radii_lod):
        """Preserve the declared boundaries as immutable host floats.

        Args:
            kind: Either "box" or "annulus".
            boundary: Either "closed" or "lower_closed_upper_open".
            limits_xy_lod: Box limits, shape (2, 2): row 0 holds the lower (x, y)
                and row 1 the upper (x, y), in lambda/D. Zero for an annulus.
            radii_lod: Inner and outer radius, shape (2,), in lambda/D. Zero for
                a box.
        """
        self.kind, self.boundary = kind, boundary
        limits = np.asarray(limits_xy_lod, dtype=np.float64)
        radii = np.asarray(radii_lod, dtype=np.float64)
        if limits.shape != (2, 2) or radii.shape != (2,):
            raise ValueError("Domain requires box limits [2, 2] and radii [2]")
        self._limits = tuple(map(tuple, limits.tolist()))
        self._radii = tuple(radii.tolist())

    @property
    def limits_xy_lod(self):
        """Return the host box limits."""
        return np.asarray(self._limits)

    @property
    def radii_lod(self):
        """Return the host annular radii."""
        return np.asarray(self._radii)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if self.kind not in ("box", "annulus"):
            raise ValueError("Domain kind must be box or annulus")
        if self.boundary not in ("closed", "lower_closed_upper_open"):
            raise ValueError("Unsupported domain boundary")
        values = [*self.limits_xy_lod.tolist(), self.radii_lod.tolist()]
        if any(not normal_or_zero(v) for row in values for v in row):
            raise ValueError("Domain limits must be finite, normal or zero")
        if self.kind == "box":
            ordered = self.limits_xy_lod[0] <= self.limits_xy_lod[1]
            if self.boundary != "closed":
                ordered = self.limits_xy_lod[0] < self.limits_xy_lod[1]
            if not bool(np.all(ordered)):
                raise ValueError("Box limits must be ordered and nonempty")
            if bool(np.any(self.radii_lod != 0)):
                raise ValueError("Unused radii must be zero")
        else:
            inner, outer = self.radii_lod.tolist()
            if not 0 <= inner < outer or bool(np.any(self.limits_xy_lod != 0)):
                raise ValueError("Annulus requires ordered radii and zero box limits")

    def contains(self, source_xy_lod):
        """Test original signed coordinates after explicit subnormal flushing.

        Args:
            source_xy_lod: Query source coordinate (x, y), in lambda/D.

        Returns:
            Scalar boolean, True if the query is inside the domain.
        """
        if array_namespace(source_xy_lod) is np:
            return self._contains_host(source_xy_lod)
        if source_xy_lod.ndim > 1:
            return jax.vmap(self.contains)(source_xy_lod.reshape(-1, 2)).reshape(
                source_xy_lod.shape[:-1]
            )
        query = flush_subnormal(
            jax.lax.stop_gradient(source_xy_lod).astype(jnp.float32)
        )
        if self.kind == "box":
            lower, upper = self.limits_xy_lod
            # Round a boundary toward the first allowed representable value.
            # This preserves membership for the same represented query.
            low = query >= like(_float32_ceiling(lower), query)
            if self.boundary == "closed":
                high = query <= like(-_float32_ceiling(-upper), query)
            else:
                high = query < like(_float32_ceiling(upper), query)
            return jnp.all(low & high)
        lower, upper = self.radii_lod**2
        below_lower = _rounded_sum_below(query, lower, inclusive=False)
        below_upper = _rounded_sum_below(
            query, upper, inclusive=self.boundary == "closed"
        )
        return jnp.all(jnp.isfinite(query)) & ~below_lower & below_upper

    def _within_radii(self, radius2, *, xp):
        inner, outer = self.radii_lod**2
        lower = radius2 >= xp.asarray(inner, dtype=radius2.dtype)
        upper = xp.asarray(outer, dtype=radius2.dtype)
        return lower & (
            radius2 <= upper if self.boundary == "closed" else radius2 < upper
        )

    def _contains_host(self, points):
        points = np.asarray(points, dtype=np.float64)
        points = np.where(np.abs(points) < sys.float_info.min, 0.0, points)
        if self.kind == "box":
            lower, upper = self.limits_xy_lod
            high = points <= upper if self.boundary == "closed" else points < upper
            return np.all((points >= lower) & high, axis=-1)
        return self._within_radii(np.sum(points**2, axis=-1), xp=np)


def _rounded_sum_below(query, limit, *, inclusive):
    """Compare two binary32 squares using the binary64 sum's rounding interval.

    Binary32 squares have at most 48 significand bits. Two unsigned 24-bit
    limbs store each product exactly. Compare their sum to the binary64
    midpoint just above (inclusive) or below (exclusive) the limit, breaking
    an exact midpoint tie toward the even significand.
    """
    if limit == 0:
        return jnp.all(query == 0) if inclusive else jnp.asarray(False)
    _, exponent = math.frexp(limit)
    adjacent = math.nextafter(limit, math.inf if inclusive else -math.inf)
    # Three 20-bit radix digits suffice for a 54-significant-bit midpoint.
    midpoint = int(math.ldexp(limit, 60 - exponent)) + int(
        math.ldexp(adjacent - limit, 59 - exponent)
    )
    even = (int(np.float64(limit).view(np.uint64)) & 1) == 0
    tie_passes = even if inclusive else not even

    # Clipping outside the largest possible boundary keeps all digit
    # conversions in range even for a query near the binary32 maximum.
    clipped = jnp.minimum(jnp.abs(query), math.ldexp(1.0, (exponent + 1) // 2 + 1))
    mantissa, power = jnp.frexp(clipped)
    significand = (mantissa * 2**24).astype(jnp.uint32)
    high, low = significand >> 12, significand & 4095
    bottom = low * low
    middle = 2 * high * low + (bottom >> 12)
    limbs = jnp.stack(
        (high * high + (middle >> 12), ((middle & 4095) << 12) + (bottom & 4095))
    )
    terms = jnp.ldexp(
        limbs.astype(query.dtype),
        2 * power - exponent - jnp.array([[24], [48]], jnp.int32),
    )
    difference = jnp.int32(0)
    for shift in (40, 20, 0):
        scaled = terms * 2**20
        digit = jnp.floor(scaled)
        terms = scaled - digit
        target = midpoint >> shift
        if shift != 40:
            target &= 2**20 - 1
        difference = jnp.clip(
            difference * 2**20 + jnp.sum(digit.astype(jnp.int32)) - target, -4, 4
        )
    # At a tie the larger square fits in the 60 bits compared above.
    # Only the smaller square can leave a tail; its two limbs are disjoint.
    equal = (difference == 0) & ~jnp.any(terms > 0)
    return (difference < 0) | (equal & tie_passes)


def normal_or_zero(value):
    """Check host binary64 values without backend denormal-mode dependence.

    Args:
        value: Host float to test.

    Returns:
        True if the value is finite and either zero or a normal number.
    """
    return math.isfinite(value) and (value == 0 or abs(value) >= sys.float_info.min)


def in_supported_range(domain):
    """Whether a domain lies in the profile's supported numeric range.

    Box limits must have magnitude <= 1e6; annulus radii must be 0 (inner only)
    or in [1e-6, 1e6], so the squared membership comparisons neither underflow
    nor overflow. Admission and reader construction share this gate.

    Args:
        domain: Permission domain to test.

    Returns:
        True if the domain limits or radii are in range.
    """
    if domain.kind == "box":
        return all(
            abs(v) <= MAX_GEOMETRY_LOD
            for row in domain.limits_xy_lod.tolist()
            for v in row
        )
    inner, outer = domain.radii_lod.tolist()
    return (
        inner == 0 or MIN_GEOMETRY_LOD <= inner <= MAX_GEOMETRY_LOD
    ) and MIN_GEOMETRY_LOD <= outer <= MAX_GEOMETRY_LOD


def coverage(query, lattice, reflected, profile=_RAW_PROFILE):
    """Check float stencil ranges before any potentially overflowing int cast.

    A query is covered when its taps, and the profile's halo of stored nodes
    beyond the first and the last tap, all lie inside the signed table.

    Args:
        query: Source coordinate (x, y), in lambda/D.
        lattice: Source pixel grid of the stored table.
        reflected: Whether the table stores one reflected quadrant.
        profile: Kernel profile giving the stencil and halo size. Defaults to
            the raw profile.

    Returns:
        Scalar boolean, True if the full stencil is covered.
    """
    _, start = lattice_phase_start(
        profile,
        query,
        origin_xy(lattice),
        spacing_xy(lattice),
        lattice.shape[::-1],
        reflected,
    )
    return covered_start(start, lattice.shape[::-1], reflected, profile)


def covered_start(start, size, reflected, profile):
    """Test the floating first tap before converting covered indices to int32.

    Args:
        start: First tap per axis (x first), shape (..., n_axes).
        size: Stored node count per axis (x first).
        reflected: Whether the table stores one reflected quadrant.
        profile: Kernel profile giving the stencil and halo size.

    Returns:
        Boolean, True where every axis has its full stencil covered.
    """
    xp = array_namespace(start)
    size = xp.asarray(size)
    lower = profile.halo + 1 - size if reflected else profile.halo
    upper = size - profile.halo - profile.taps
    return xp.all(xp.isfinite(start) & (start >= lower) & (start <= upper), axis=-1)


def safe_query(domain, lattice, reflected, profile=_RAW_PROFILE):
    """Find and verify a point in the domain/coverage intersection.

    Test inclusive coverage endpoints before their inward neighbours, box
    corners and centers, and annular candidates on corner rays and box sides
    (including axis crossings). Every candidate uses the read-time membership
    and exact floor rule. This finite list is not a proof of nonemptiness for
    every floating-point geometry: an unresolved intersection returns None,
    causing admission to refuse with INCOMPLETE_STENCIL.

    Args:
        domain: Permission domain.
        lattice: Source pixel grid of the stored table.
        reflected: Whether the table stores one reflected quadrant.
        profile: Kernel profile giving the stencil and halo size. Defaults to
            the raw profile.

    Returns:
        Source coordinate (x, y) in lambda/D, shape (2,), that is both
        permitted and fully covered, or None if no candidate qualifies.
    """
    origin = origin_xy(lattice).tolist()
    spacing = spacing_xy(lattice).tolist()
    size = lattice.shape[::-1]
    below, above = profile.coverage_margins
    low = [
        o + h * (above - n if reflected else below)
        for o, h, n in zip(origin, spacing, size, strict=True)
    ]
    high = [o + h * (n - above) for o, h, n in zip(origin, spacing, size, strict=True)]
    if domain.kind == "box":
        bounds = domain.limits_xy_lod.tolist()
        low = [max(a, b) for a, b in zip(low, bounds[0], strict=True)]
        high = [min(a, b) for a, b in zip(high, bounds[1], strict=True)]
    if any(a > b for a, b in zip(low, high, strict=True)):
        return None
    axes = [
        (
            a,
            b,
            math.nextafter(a, math.inf),
            math.nextafter(b, -math.inf),
            a / 2 + b / 2,
            max(a, min(0.0, b)),
        )
        for a, b in zip(low, high, strict=True)
    ]
    candidates = [[x, y] for x in axes[0] for y in axes[1]]
    if domain.kind == "annulus":
        inner, outer = domain.radii_lod.tolist()
        radii = (
            inner,
            outer,
            (inner + outer) / 2,
            math.nextafter(inner, math.inf),
            math.nextafter(outer, -math.inf),
        )
        directions = [[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])]
        directions += [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        for radius in radii:
            for x, y in directions:
                norm = math.hypot(x, y)
                if norm:
                    candidates.append([radius * (x / norm), radius * (y / norm)])
            # Circle/box-side intersections also cover narrow off-axis regions.
            for axis in range(2):
                for value in (low[axis], high[axis], 0.0):
                    if abs(value) <= radius:
                        other = math.sqrt((radius - abs(value)) * (radius + abs(value)))
                        for sign in (-1, 1):
                            point = [value, sign * other]
                            candidates.append(point if axis == 0 else point[::-1])
    points = np.asarray(candidates, dtype=np.float64)
    valid = domain._contains_host(points) & coverage(
        points, lattice, reflected, profile
    )
    if np.any(valid):
        return points[np.argmax(valid)]
    return None


def restrict_domain(declared, override=None):
    """Validate numeric range and require an override to restrict permission.

    Args:
        declared: Domain declared by the delivery, or None.
        override: Optional restricting domain of the same kind and boundary.

    Returns:
        The override if given, else the declared domain (possibly None).

    Raises:
        ValueError: If a domain is out of range or the override expands or
            changes the declared geometry.
    """
    for domain in (declared, override):
        if domain is not None and not in_supported_range(domain):
            raise ValueError("Domain outside supported numeric range")
    if declared is None:
        return override
    if override is None:
        return declared
    if override.kind != declared.kind or override.boundary != declared.boundary:
        raise ValueError("Domain override must restrict declared geometry and boundary")
    if declared.kind == "box":
        subset = np.all(
            override.limits_xy_lod[0] >= declared.limits_xy_lod[0]
        ) & np.all(override.limits_xy_lod[1] <= declared.limits_xy_lod[1])
    else:
        subset = (
            override.radii_lod[0] >= declared.radii_lod[0]
            and override.radii_lod[1] <= declared.radii_lod[1]
        )
    if not bool(subset):
        raise ValueError("Domain override cannot expand served permission")
    return override


def validate_check_positions(positions, domain, lattice, reflected, profile=None):
    """Reject independent checks outside permission or full stencil coverage.

    Args:
        positions: Check coordinates (x, y), in lambda/D, one per check.
        domain: Permission domain, or None to skip the permission test.
        lattice: Source pixel grid of the stored table.
        reflected: Whether the table stores one reflected quadrant.
        profile: Kernel profile giving the stencil and halo size. If None, the
            default raw profile is used.

    Raises:
        ValueError: If a check lies outside the domain or stencil coverage.
    """
    profile = KernelProfile() if profile is None else profile
    for point in np.asarray(positions, dtype=np.float64):
        if domain is not None and not domain._contains_host(point):
            raise ValueError("DATA_INVALID: independent check outside served domain")
        if not coverage(point, lattice, reflected, profile):
            raise ValueError("DATA_INVALID: independent check outside stencil coverage")
