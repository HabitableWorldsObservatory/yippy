"""Signed permission domains and independent full-stencil coverage."""

import math
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

from yippy.sinc._types import F64, check_array, check_static
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import KernelProfile, flush_subnormal

_RAW_PROFILE = KernelProfile()


class Domain(eqx.Module):
    """A box or origin-centered annulus, without tolerance-expanded boundaries."""

    kind: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    limits_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    radii_lod: F64 = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.limits_xy_lod, (2, 2), (jnp.float64,))
        check_array(self.radii_lod, (2,), (jnp.float64,))
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
            if not bool(jnp.all(ordered)):
                raise ValueError("Box limits must be ordered and nonempty")
            if bool(jnp.any(self.radii_lod != 0)):
                raise ValueError("Unused radii must be zero")
        else:
            inner, outer = self.radii_lod.tolist()
            if not 0 <= inner < outer or bool(jnp.any(self.limits_xy_lod != 0)):
                raise ValueError("Annulus requires ordered radii and zero box limits")

    def contains(self, source_xy_lod):
        """Test original signed coordinates after explicit subnormal flushing."""
        query = flush_subnormal(source_xy_lod)
        if self.kind == "box":
            low = jnp.all(query >= self.limits_xy_lod[0])
            upper = query <= self.limits_xy_lod[1]
            if self.boundary != "closed":
                upper = query < self.limits_xy_lod[1]
            return low & jnp.all(upper)
        radius2 = jnp.sum(query * query)
        lower = radius2 >= self.radii_lod[0] ** 2
        upper = radius2 <= self.radii_lod[1] ** 2
        if self.boundary != "closed":
            upper = radius2 < self.radii_lod[1] ** 2
        return lower & upper


def normal_or_zero(value):
    """Check host binary64 values without backend denormal-mode dependence."""
    return math.isfinite(value) and (value == 0 or abs(value) >= sys.float_info.min)


def in_supported_range(domain):
    """Whether a domain lies in the profile's supported numeric range.

    Box limits must have magnitude <= 1e6; annulus radii must be 0 (inner only)
    or in [1e-6, 1e6], so the squared membership comparisons neither underflow
    nor overflow. Admission and reader construction share this gate.
    """
    if domain.kind == "box":
        return all(abs(v) <= 1e6 for row in domain.limits_xy_lod.tolist() for v in row)
    inner, outer = domain.radii_lod.tolist()
    return (inner == 0 or 1e-6 <= inner <= 1e6) and 1e-6 <= outer <= 1e6


def coverage(query, lattice, reflected, profile=_RAW_PROFILE):
    """Check float stencil ranges before any potentially overflowing int cast.

    A query is covered when its taps, and the profile's halo of stored nodes
    beyond the first and the last tap, all lie inside the signed table.
    """
    start = profile.stencil_start(query, origin_xy(lattice), spacing_xy(lattice))
    size = jnp.asarray(lattice.shape[::-1], dtype=jnp.float64)
    halo = float(profile.halo)
    lower = halo + 1 - size if reflected else jnp.full((2,), halo)
    upper = size - halo - profile.taps
    return jnp.all(jnp.isfinite(start) & (start >= lower) & (start <= upper))


@eqx.filter_jit
def _candidate_membership(points, domain, lattice, reflected, profile):
    return jax.vmap(
        lambda q: domain.contains(q) & coverage(q, lattice, reflected, profile)
    )(points)


def safe_query(domain, lattice, reflected, profile=_RAW_PROFILE):
    """Find and verify a point in the domain/coverage intersection.

    Test inclusive coverage endpoints before their inward neighbours, box
    corners and centers, and annular candidates on corner rays and box sides
    (including axis crossings). Every candidate uses the read-time membership
    and exact floor rule. This finite list is not a proof of nonemptiness for
    every floating-point geometry: an unresolved intersection returns None,
    causing admission to refuse with INCOMPLETE_STENCIL.
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
    points = jnp.asarray(candidates, dtype=jnp.float64)
    valid = _candidate_membership(points, domain, lattice, reflected, profile).tolist()
    for index, permitted in enumerate(valid):
        if permitted:
            return points[index]
    return None


def restrict_domain(declared, override=None):
    """Validate numeric range and require an override to restrict permission."""
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
        subset = jnp.all(
            override.limits_xy_lod[0] >= declared.limits_xy_lod[0]
        ) & jnp.all(override.limits_xy_lod[1] <= declared.limits_xy_lod[1])
    else:
        subset = (
            override.radii_lod[0] >= declared.radii_lod[0]
            and override.radii_lod[1] <= declared.radii_lod[1]
        )
    if not bool(subset):
        raise ValueError("Domain override cannot expand served permission")
    return override


def validate_check_positions(positions, domain, lattice, reflected, profile=None):
    """Reject independent checks outside permission or full stencil coverage."""
    profile = KernelProfile() if profile is None else profile
    for position in positions:
        query = jnp.asarray(position, dtype=jnp.float64)
        if domain is not None and not bool(domain.contains(query)):
            raise ValueError("DATA_INVALID: independent check outside served domain")
        if not bool(coverage(query, lattice, reflected, profile)):
            raise ValueError("DATA_INVALID: independent check outside stencil coverage")
