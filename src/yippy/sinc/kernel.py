"""Kernel profiles, their weights and reproducible stencil selection."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from yippy.sinc._types import Boolean, check_static

# Stored nodes a prefiltered read needs beyond its outermost tap on each side.
PREFILTER_HALO = 6


class AbstractKernelProfile(eqx.Module):
    """Stencil geometry and admission limits that every reader profile declares.

    A read at query phase t = (q - origin) / spacing uses the ``taps`` consecutive
    nodes that start at ``stencil_start``. Coverage additionally requires ``halo``
    stored nodes beyond the first and the last tap.
    """

    name: eqx.AbstractVar[str]
    floor_rule: eqx.AbstractVar[str]
    # Taps per source axis in one read: a field or a property of the subclass.
    taps: eqx.AbstractVar[int]

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return the joint weight and stencil arithmetic identity."""

    @property
    @abstractmethod
    def halo(self) -> int:
        """Stored nodes required beyond the outermost tap on each side."""

    @property
    @abstractmethod
    def margin_limit(self) -> float:
        """Largest admitted sampling ratio 2 B h."""

    @abstractmethod
    def stencil_start(self, query, origin, spacing):
        """Return the float first tap index of a prevalidated query, per axis."""

    @abstractmethod
    def minimum_size(self, reflected: bool) -> int:
        """Smallest stored node count per axis that can serve any read."""

    @property
    def coverage_margins(self):
        """Phase margins (low, high): covered phases t satisfy low <= t < size - high.

        With reflection the lower limit is -(size - high) instead.
        """
        first = self.halo + (self.taps - 1) // 2
        if self.taps % 2:
            return first - 0.5, first + 0.5
        return float(first), float(first + 1)


class KernelProfile(AbstractKernelProfile):
    """Fixed unnormalized W=16, beta=12 kernel with versioned floor semantics.

    Admission supports spacings and nonzero origin magnitudes in [1e-6, 1e6],
    box limit magnitudes <=1e6, and annular radii in [1e-6, 1e6] (inner zero
    allowed). Query subnormal flushing then leaves nonzero differences and
    fmod remainders normal. Permitted quotients have magnitude <=2e12, below
    2**53, so the integer quotient and its negative-side correction are exact.
    Squared radius limits and boundary-scale coordinates neither underflow nor
    overflow; negligible coordinate squares may round to zero without changing
    membership. Queries far outside the lattice fail coverage before gathering.
    """

    name: str = eqx.field(static=True, default="kaiser_w16_beta12_raw_v1")
    floor_rule: str = eqx.field(static=True, default="exact_quotient_floor_v1")

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if self.name != "kaiser_w16_beta12_raw_v1":
            raise ValueError("Unsupported kernel profile")
        if self.floor_rule != "exact_quotient_floor_v1":
            raise ValueError("Unsupported floor rule")

    @property
    def identity(self):
        """Return the joint weight and stencil arithmetic identity."""
        return f"{self.name}:{self.floor_rule}"

    @property
    def taps(self):
        """Taps per source axis in one read."""
        return 32

    @property
    def halo(self):
        """Stored nodes required beyond the outermost tap on each side."""
        return 0

    @property
    def margin_limit(self):
        """Largest admitted sampling ratio 2 B h."""
        return 0.76

    def stencil_start(self, query, origin, spacing):
        """Return the float first tap index, 15 nodes below the exact floor."""
        return exact_quotient_floor_v1(query, origin, spacing) - 15.0

    def minimum_size(self, reflected):
        """Smallest stored node count per axis that can serve any read."""
        return 17 if reflected else 32


class PrefilteredKernelProfile(AbstractKernelProfile):
    """Kaiser-Bessel bump read of coefficients filtered once at load.

    The read sums ``taps`` coefficient images per axis with the bump weights of
    ``bump``. The coefficients are the stored samples convolved, along both source
    axes, with a deapodization filter of half-width ``filter_half_width`` designed
    for the admitted band (see ``yippy.sinc.prefilter``). The tap count is chosen
    when a delivery is loaded: the delivery itself always holds the raw samples.
    Odd tap counts center the stencil on the nearest node (halves round up), even
    counts on the cell that holds the query. The filter half-width (12 nodes at
    7 taps, 16 otherwise) and the halo are fixed for this profile name.
    """

    name: str = eqx.field(static=True, default="kaiser_bessel_prefiltered_v1")
    floor_rule: str = eqx.field(static=True, default="exact_quotient_floor_v1")
    taps: int = eqx.field(static=True, default=7)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if self.name != "kaiser_bessel_prefiltered_v1":
            raise ValueError("Unsupported kernel profile")
        if self.floor_rule != "exact_quotient_floor_v1":
            raise ValueError("Unsupported floor rule")
        if type(self.taps) is not int or not 5 <= self.taps <= 10:
            raise ValueError("Prefiltered reads support 5 to 10 taps per axis")

    @property
    def identity(self):
        """Return the joint weight, filter and stencil arithmetic identity."""
        return (
            f"{self.name}:taps{self.taps}:filter{self.filter_half_width}:"
            f"halo{self.halo}:{self.floor_rule}"
        )

    @property
    def filter_half_width(self):
        """Half-width, in nodes, of the interior deapodization filter."""
        return 12 if self.taps == 7 else 16

    @property
    def halo(self):
        """Stored nodes required beyond the outermost tap on each side."""
        return PREFILTER_HALO

    @property
    def margin_limit(self):
        """Largest admitted sampling ratio 2 B h."""
        return 0.64

    def stencil_start(self, query, origin, spacing):
        """Return the float first tap index from exact floor and half-cell tests."""
        floor = exact_quotient_floor_v1(query, origin, spacing)
        if self.taps % 2 == 0:
            return floor - (self.taps // 2 - 1)
        return floor + upper_half_cell(query, origin, spacing) - (self.taps - 1) // 2

    def minimum_size(self, reflected):
        """Smallest stored node count per axis that can serve any read.

        Every coefficient node needs the full filter half-width on at least one
        side, so the signed table spans at least 2 * filter_half_width + 1 nodes.
        """
        signed = max(2 * self.filter_half_width + 1, self.taps + 2 * self.halo)
        return (signed + 2) // 2 if reflected else signed


def flush_subnormal(value):
    """Replace binary64 subnormals by signed zero using integer bits."""
    bits = jax.lax.bitcast_convert_type(value, jnp.int64)
    exponent = (bits >> 52) & 0x7FF
    zero = jax.lax.bitcast_convert_type(bits & jnp.int64(-(2**63)), jnp.float64)
    return jnp.where(exponent == 0, zero, value)


def exact_quotient_floor_v1(query, origin, spacing):
    """Return floating integer floors of the exact quotient of rounded d by h.

    Query subnormals become signed zero before the single subtraction d=q-o.
    Callers must check the resulting float index range before integer conversion.
    The remainder determines the side of a cell edge without reciprocal rounding.
    """
    # For nonzero origins >=1e-6, adjacent cancellation is >=2**-73;
    # with zero origin, flushed nonzero q is normal. If |d| < h, r=d;
    # otherwise h>=1e-6 gives a binary remainder quantum >=2**-73.
    # Permitted |q-o|/h <=2e12 keeps integer recovery below 2**53.
    difference = flush_subnormal(query) - origin
    remainder = jnp.fmod(difference, spacing)
    quotient = jnp.round((difference - remainder) / spacing)
    return quotient - jnp.where((remainder != 0) & (difference < 0), 1.0, 0.0)


def upper_half_cell(query, origin, spacing):
    """Return 1.0 where the exact phase fraction is at least one half, else 0.0.

    fmod is exact and doubling is exact, so the comparison has no rounding. For
    negative differences the remainder lies in (-h, 0] and the fraction is
    (r + h) / h unless r is zero.
    """
    difference = flush_subnormal(query) - origin
    remainder = jnp.fmod(difference, spacing)
    doubled = 2.0 * remainder
    upper = jnp.where(
        difference >= 0, doubled >= spacing, (remainder != 0) & (doubled >= -spacing)
    )
    return jnp.where(upper, 1.0, 0.0)


def weights(argument):
    """Evaluate raw Kaiser-sinc weights, with safe inactive derivative lanes."""
    active = jnp.abs(argument) < 16.0
    inside = jnp.where(active, 1.0 - (argument / 16.0) ** 2, 1.0)
    window = jnp.i0(12.0 * jnp.sqrt(inside)) / jnp.i0(jnp.float64(12.0))
    return jnp.where(active, jnp.sinc(argument) * window, 0.0)


def stencil(query, origin, spacing):
    """Return signed int64 taps and float64 weights for a prevalidated query."""
    query = flush_subnormal(query)
    floor = exact_quotient_floor_v1(query, origin, spacing).astype(jnp.int64)
    taps = floor[:, None] + jnp.arange(-15, 17, dtype=jnp.int64)
    phase = (query - origin) / spacing
    return taps, weights(phase[:, None] - taps)


def bump(argument, taps, beta):
    """Evaluate the Kaiser-Bessel bump of support ``taps``.

    The bump is I0(beta * sqrt(1 - (2 u / taps)**2)) / I0(beta) for
    abs(u) < taps / 2 and zero elsewhere, so it is cut at 1 / I0(beta) rather
    than falling to zero: a read jumps by about that fraction where its stencil
    moves to the next cell. It does not interpolate: reads apply it to
    coefficients, never to raw samples, and do not reproduce a stored sample
    exactly. Inactive lanes are sanitized before the square root so derivatives
    stay finite.
    """
    half = taps / 2.0
    active = jnp.abs(argument) < half
    inside = jnp.where(active, 1.0 - (argument / half) ** 2, 1.0)
    return jnp.where(active, jnp.i0(beta * jnp.sqrt(inside)) / jnp.i0(beta), 0.0)


def bump_stencil(query, origin, spacing, profile, beta, scale):
    """Return signed int64 taps and float64 bump weights for a prevalidated query.

    ``beta`` and ``scale`` are the per-axis shape and normalization of the design
    that built the coefficients; the weights are bump / scale.
    """
    query = flush_subnormal(query)
    start = profile.stencil_start(query, origin, spacing).astype(jnp.int64)
    taps = start[:, None] + jnp.arange(profile.taps, dtype=jnp.int64)
    phase = (query - origin) / spacing
    value = bump(phase[:, None] - taps, profile.taps, beta[:, None])
    return taps, value / scale[:, None]


class Stencil(eqx.Module):
    """The taps of one read: per-axis nodes, weights and storage validity.

    All three arrays are [2, taps], x axis first. ``nodes`` are signed int64
    node indices, ``weights`` the float64 one-dimensional weights (a read
    weights the node pair (i, j) by ``weights[0, i] * weights[1, j]``), and
    ``valid`` marks nodes a table can serve: inside the stored extent, or with
    reflection inside it after taking the absolute index.
    """

    nodes: jax.Array
    weights: jax.Array
    valid: Boolean


def read_stencil(
    profile,
    query,
    origin,
    spacing,
    size=None,
    reflected=False,
    beta=None,
    scale=None,
):
    """Return the Stencil a read of ``profile`` uses at a prevalidated query.

    This is the one implementation of tap selection and weights: readers
    contract their tables with it and plots draw it. ``size`` is the stored
    node count per axis (x first); None marks every node valid. A prefiltered
    profile also needs the ``beta`` and ``scale`` of its designed band.
    """
    if isinstance(profile, PrefilteredKernelProfile):
        if beta is None or scale is None:
            raise ValueError(
                "A prefiltered stencil needs the beta and scale of its design"
            )
        nodes, weights_ = bump_stencil(query, origin, spacing, profile, beta, scale)
    else:
        nodes, weights_ = stencil(query, origin, spacing)
    if size is None:
        valid = jnp.ones(nodes.shape, dtype=bool)
    else:
        reach = jnp.abs(nodes) if reflected else nodes
        valid = (reach >= 0) & (reach < jnp.asarray(size)[:, None])
    return Stencil(nodes, weights_, valid)
