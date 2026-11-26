"""Kernel profiles, their weights and reproducible stencil selection."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.precision import float_dtype, like

from yippy.sinc._types import Boolean, check_static

# Stored nodes a prefiltered read needs beyond its outermost tap on each side.
PREFILTER_HALO = 6


class AbstractKernelProfile(eqx.Module):
    """Stencil geometry and admission limits that every reader profile declares.

    A read at query phase t = (q - origin) / spacing uses the ``taps`` consecutive
    nodes that start at ``stencil_start``. Coverage additionally requires ``halo``
    stored nodes beyond the first and the last tap.

    Attributes:
        name: Profile name, part of the arithmetic identity.
        floor_rule: Name of the floor rule used to select the first tap.
        taps: Taps per source axis in one read.
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
        """Return the float first tap index of a prevalidated query, per axis.

        Args:
            query: Query coordinates, one per axis, in the unit of ``origin``.
            origin: Coordinate of node zero, per axis.
            spacing: Node spacing, per axis, in the unit of ``origin``.

        Returns:
            Float index of the first tap, per axis.
        """

    @abstractmethod
    def minimum_size(self, reflected: bool) -> int:
        """Smallest stored node count per axis that can serve any read.

        Args:
            reflected: Whether the table is stored for non-negative positions
                only and read by reflection.

        Returns:
            The smallest stored node count per axis.
        """

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

    Attributes:
        name: Profile name; only ``kaiser_w16_beta12_raw_v1`` is supported.
        floor_rule: Floor rule name; only ``exact_quotient_floor_v1`` is
            supported.
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
        """Return the float first tap index, 15 nodes below the exact floor.

        Args:
            query: Query coordinates, one per axis, in the unit of ``origin``.
            origin: Coordinate of node zero, per axis.
            spacing: Node spacing, per axis, in the unit of ``origin``.

        Returns:
            Float index of the first tap, per axis.
        """
        return exact_quotient_floor_v1(query, origin, spacing) - 15.0

    def minimum_size(self, reflected):
        """Smallest stored node count per axis that can serve any read.

        Args:
            reflected: Whether the table is stored for non-negative positions
                only and read by reflection.

        Returns:
            The smallest stored node count per axis: 17 with reflection, else
            32.
        """
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

    Attributes:
        name: Profile name; only ``kaiser_bessel_prefiltered_v1`` is supported.
        floor_rule: Floor rule name; only ``exact_quotient_floor_v1`` is
            supported.
        taps: Taps per source axis in one read, an integer from 5 to 10.
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
        """Return the float first tap index from exact floor and half-cell tests.

        Args:
            query: Query coordinates, one per axis, in the unit of ``origin``.
            origin: Coordinate of node zero, per axis.
            spacing: Node spacing, per axis, in the unit of ``origin``.

        Returns:
            Float index of the first tap, per axis.
        """
        floor = exact_quotient_floor_v1(query, origin, spacing)
        if self.taps % 2 == 0:
            return floor - (self.taps // 2 - 1)
        return floor + upper_half_cell(query, origin, spacing) - (self.taps - 1) // 2

    def minimum_size(self, reflected):
        """Smallest stored node count per axis that can serve any read.

        Every coefficient node needs the full filter half-width on at least one
        side, so the signed table spans at least 2 * filter_half_width + 1 nodes.

        Args:
            reflected: Whether the table is stored for non-negative positions
                only and read by reflection.

        Returns:
            The smallest stored node count per axis.
        """
        signed = max(2 * self.filter_half_width + 1, self.taps + 2 * self.halo)
        return (signed + 2) // 2 if reflected else signed


def flush_subnormal(value):
    """Replace subnormals by signed zero using integer bits.

    Args:
        value: Float32 or float64 array.

    Returns:
        The array with each subnormal replaced by a zero of the same sign.
    """
    if value.dtype == jnp.float32:
        integer, fraction, exponent_mask = jnp.int32, 23, 0xFF
    else:
        integer, fraction, exponent_mask = jnp.int64, 52, 0x7FF
    bits = jax.lax.bitcast_convert_type(value, integer)
    exponent = (bits >> fraction) & exponent_mask
    sign = jnp.iinfo(integer).min
    zero = jax.lax.bitcast_convert_type(bits & sign, value.dtype)
    return jnp.where(exponent == 0, zero, value)


def exact_quotient_floor_v1(query, origin, spacing):
    """Return floating integer floors of the exact quotient of rounded d by h.

    Query subnormals become signed zero before the single subtraction d=q-o.
    Callers must check the resulting float index range before integer conversion.
    The remainder determines the side of a cell edge without reciprocal rounding.

    Args:
        query: Query coordinates, in the unit of ``origin``.
        origin: Coordinate of node zero.
        spacing: Node spacing, in the unit of ``origin``.

    Returns:
        Float array, shaped like ``query``, of the integer floor of
        (query - origin) / spacing.
    """
    # For nonzero origins >=1e-6, adjacent cancellation is >=2**-73;
    # with zero origin, flushed nonzero q is normal. If |d| < h, r=d;
    # otherwise h>=1e-6 gives a binary remainder quantum >=2**-73.
    # Permitted |q-o|/h <=2e12 keeps integer recovery below 2**53.
    difference = flush_subnormal(query) - origin
    remainder = jnp.fmod(difference, spacing)
    quotient = jnp.round((difference - remainder) / spacing)
    below = (remainder != 0) & (difference < 0)
    return quotient - below.astype(quotient.dtype)


def upper_half_cell(query, origin, spacing):
    """Return 1.0 where the exact phase fraction is at least one half, else 0.0.

    fmod is exact and doubling is exact, so the comparison has no rounding. For
    negative differences the remainder lies in (-h, 0] and the fraction is
    (r + h) / h unless r is zero.

    Args:
        query: Query coordinates, in the unit of ``origin``.
        origin: Coordinate of node zero.
        spacing: Node spacing, in the unit of ``origin``.

    Returns:
        Float array, shaped like ``query``, of 1.0 or 0.0.
    """
    difference = flush_subnormal(query) - origin
    remainder = jnp.fmod(difference, spacing)
    doubled = 2.0 * remainder
    upper = jnp.where(
        difference >= 0, doubled >= spacing, (remainder != 0) & (doubled >= -spacing)
    )
    return upper.astype(difference.dtype)


def _float32_ceiling(value):
    """First binary32 value at or above a host binary64 boundary."""
    rounded = np.asarray(value, dtype=np.float32)
    return np.where(rounded < value, np.nextafter(rounded, np.float32(np.inf)), rounded)


def _relative_phase(query, origin, spacing):
    """Subtract the host reference before rounding away its low bits."""
    high = origin.astype(np.float32)
    low = origin - np.asarray(high, dtype=np.float64)
    return ((query - like(high, query)) - like(low, query)) / like(spacing, query)


def _host_start(profile, difference, h):
    """Evaluate the binary64 quotient floor and odd-tap half-cell rule."""
    remainder = np.fmod(difference, h)
    quotient = np.rint((difference - remainder) / h)
    floor = quotient - ((remainder != 0) & (difference < 0))
    if profile.taps % 2:
        upper = np.where(
            difference >= 0,
            2 * remainder >= h,
            (remainder != 0) & (2 * remainder >= -h),
        )
        return floor + upper - (profile.taps - 1) // 2
    return floor - (profile.taps // 2 - 1)


def float32_stencil_start(profile, query, origin, spacing, size, reflected):
    """Select taps at host-computed edges for represented float32 queries.

    Args:
        profile: Kernel profile giving the stencil and coverage margins.
        query: Float32 query coordinates, one per axis (x first).
        origin: Host coordinate of node zero, per axis.
        spacing: Host node spacing, per axis.
        size: Stored node count per axis (x first).
        reflected: Whether the table is read by reflection.

    Returns:
        First tap per axis as a float array shaped like ``query``.
    """
    below, _ = profile.coverage_margins
    starts = []
    for axis, (o, h, n) in enumerate(zip(origin, spacing, size, strict=True)):
        low = 1 - n if reflected else 0
        # Include the immediate uncovered tap starts for stencil inspection.
        first = low - profile.taps
        stop = n + profile.taps
        tap = np.arange(first, stop, dtype=np.float64)
        shift = below - profile.halo
        boundary = (tap + shift) * h

        # First binary64 difference in the new cell. The rounded difference
        # crosses at the midpoint with its predecessor. Form that midpoint
        # after adding the origin so cancellation near zero keeps the gap.
        boundary = np.where(
            _host_start(profile, boundary, h) < tap,
            np.nextafter(boundary, np.inf),
            boundary,
        )
        gap = boundary - np.nextafter(boundary, -np.inf)
        edges = _float32_ceiling((o + boundary) - gap / 2)
        previous = np.nextafter(edges, np.float32(-np.inf))
        earlier = (
            _host_start(profile, np.asarray(previous, dtype=np.float64) - o, h) >= tap
        )
        later = _host_start(profile, np.asarray(edges, dtype=np.float64) - o, h) < tap
        edges = np.where(
            earlier,
            previous,
            np.where(later, np.nextafter(edges, np.float32(np.inf)), edges),
        )
        index = jnp.searchsorted(like(edges, query), query[..., axis], side="right")
        tail = jnp.floor(_relative_phase(query[..., axis], np.asarray(o), h) - shift)
        interior = (first + index - 1).astype(tail.dtype)
        starts.append(jnp.where((index > 0) & (index < edges.size), interior, tail))
    return jnp.stack(starts, axis=-1)


def weights(argument):
    """Evaluate raw Kaiser-sinc weights, with safe inactive derivative lanes.

    Args:
        argument: Distance from the query to each tap, in nodes. The weight is
            zero where the magnitude is 16 or more.

    Returns:
        Weights, shaped like ``argument``.
    """
    active = jnp.abs(argument) < 16.0
    inside = jnp.where(active, 1.0 - (argument / 16.0) ** 2, like(1, argument))
    window = jnp.i0(12.0 * jnp.sqrt(inside)) / jnp.i0(like(12.0, argument))
    return jnp.where(active, jnp.sinc(argument) * window, like(0, argument))


def stencil(query, origin, spacing):
    """Return signed int32 taps and working-precision weights for a prevalidated query.

    Args:
        query: Query coordinates, one per axis, in the unit of ``origin``.
        origin: Coordinate of node zero, per axis.
        spacing: Node spacing, per axis, in the unit of ``origin``.

    Returns:
        A tuple ``(taps, weights)``, each of shape (n_axes, 32): the node
        indices and the one-dimensional weights.
    """
    query = flush_subnormal(query)
    floor = exact_quotient_floor_v1(query, origin, spacing).astype(jnp.int32)
    taps = floor[:, None] + jnp.arange(-15, 17, dtype=jnp.int32)
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

    Args:
        argument: Distance from the query to each tap, in nodes.
        taps: Support of the bump, in nodes.
        beta: Kaiser-Bessel shape parameter; broadcastable against ``argument``.

    Returns:
        Bump values, shaped like ``argument``, equal to 1 at the center.
    """
    half = taps / 2.0
    active = jnp.abs(argument) < half
    inside = jnp.where(active, 1.0 - (argument / half) ** 2, like(1, argument))
    return jnp.where(
        active, jnp.i0(beta * jnp.sqrt(inside)) / jnp.i0(beta), like(0, argument)
    )


def bump_stencil(query, origin, spacing, profile, beta, scale):
    """Return signed int32 taps and bump weights for a prevalidated query.

    ``beta`` and ``scale`` are the per-axis shape and normalization of the design
    that built the coefficients; the weights are bump / scale.

    Args:
        query: Query coordinates, one per axis, in the unit of ``origin``.
        origin: Coordinate of node zero, per axis.
        spacing: Node spacing, per axis, in the unit of ``origin``.
        profile: The prefiltered kernel profile that selects the taps.
        beta: Per-axis Kaiser-Bessel shape parameter, shape (n_axes,).
        scale: Per-axis normalization, shape (n_axes,).

    Returns:
        A tuple ``(taps, weights)``, each of shape (n_axes, profile.taps): the
        node indices and the bump weights divided by ``scale``.
    """
    query = flush_subnormal(query)
    start = profile.stencil_start(query, origin, spacing).astype(jnp.int32)
    taps = start[:, None] + jnp.arange(profile.taps, dtype=jnp.int32)
    phase = (query - origin) / spacing
    value = bump(phase[:, None] - taps, profile.taps, beta[:, None])
    return taps, value / scale[:, None]


class Stencil(eqx.Module):
    """The taps of one read: per-axis nodes, weights and storage validity.

    All three arrays are [2, taps], x axis first. ``nodes`` are signed int32
    node indices, ``weights`` the working-precision one-dimensional weights (a read
    weights the node pair (i, j) by ``weights[0, i] * weights[1, j]``), and
    ``valid`` marks nodes a table can serve: inside the stored extent, or with
    reflection inside it after taking the absolute index.

    Attributes:
        nodes: Signed int32 node indices, shape (2, taps).
        weights: Working-precision one-dimensional weights, shape (2, taps).
        valid: Boolean mask of nodes a table can serve, shape (2, taps).
    """

    nodes: jax.Array
    weights: jax.Array
    valid: Boolean


def lattice_phase_start(profile, query, origin, spacing, size, reflected):
    """Compute local phase and first tap before narrowing a host coordinate.

    Args:
        profile: Kernel profile that selects the taps.
        query: Query coordinates, one per axis (x first); host or traced.
        origin: Host coordinate of node zero, per axis.
        spacing: Host node spacing, per axis.
        size: Stored node count per axis (x first), or None.
        reflected: Whether the table is read by reflection.

    Returns:
        Tuple of the local phase and the first tap per axis.
    """
    if not isinstance(query, (jax.Array, jax.core.Tracer)):
        points = np.asarray(query, dtype=np.float64)
        clean = np.where(np.abs(points) < np.finfo(np.float64).tiny, 0.0, points)
        difference = clean - origin
        with np.errstate(over="ignore"):
            return difference / spacing, _host_start(profile, difference, spacing)
    clean = flush_subnormal(query)
    if isinstance(origin, np.ndarray) and size is not None:
        # Permission and first-tap decisions share one represented grid under
        # JIT. Phase and its derivative retain the input coordinate precision.
        represented = flush_subnormal(jax.lax.stop_gradient(clean).astype(jnp.float32))
        first = float32_stencil_start(
            profile, represented, origin, spacing, size, reflected
        )
        if query.dtype == jnp.float32:
            phase = _relative_phase(clean, origin, spacing)
        else:
            phase = (clean - like(origin, clean)) / like(spacing, clean)
        return phase, first
    origin, spacing = like(origin, query), like(spacing, query)
    return (clean - origin) / spacing, profile.stencil_start(clean, origin, spacing)


def stencil_at(profile, phase, first, size, reflected, beta, scale):
    """Build weights and signed nodes from the selected local lattice cell.

    Args:
        profile: Kernel profile that selects the taps and weights.
        phase: Local phase per axis, in node spacings from node zero.
        first: First tap per axis.
        size: Stored node count per axis (x first), or None to mark every
            node valid.
        reflected: Whether node validity uses the absolute index.
        beta: Per-axis Kaiser-Bessel shape of a prefiltered profile.
        scale: Per-axis normalization of a prefiltered profile.

    Returns:
        The ``Stencil`` of nodes, weights and validity.

    Raises:
        ValueError: If a prefiltered profile lacks ``beta`` or ``scale``.
    """
    nodes = first.astype(jnp.int32)[:, None] + jnp.arange(profile.taps, dtype=jnp.int32)
    argument = phase[:, None] - nodes
    if isinstance(profile, PrefilteredKernelProfile):
        if beta is None or scale is None:
            raise ValueError(
                "A prefiltered stencil needs the beta and scale of its design"
            )
        weights_ = bump(argument, profile.taps, beta[:, None]) / scale[:, None]
    else:
        weights_ = weights(argument)
    if size is None:
        valid = jnp.ones(nodes.shape, dtype=bool)
    else:
        reach = jnp.abs(nodes) if reflected else nodes
        valid = (reach >= 0) & (reach < jnp.asarray(size)[:, None])
    return Stencil(nodes, weights_, valid)


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

    Args:
        profile: The kernel profile that selects the taps and weights.
        query: Query coordinates, one per axis (x first), in the unit of
            ``origin``.
        origin: Coordinate of node zero, per axis.
        spacing: Node spacing, per axis, in the unit of ``origin``.
        size: Stored node count per axis (x first). None marks every node
            valid.
        reflected: Whether the table is read by reflection, so a node index is
            validated by its absolute value.
        beta: Per-axis Kaiser-Bessel shape parameter of a prefiltered profile;
            required for one and ignored otherwise.
        scale: Per-axis normalization of a prefiltered profile; required for
            one and ignored otherwise.

    Returns:
        The ``Stencil`` of nodes, weights and validity for the read.
    """
    phase, first = lattice_phase_start(profile, query, origin, spacing, size, reflected)
    dtype = (
        query.dtype
        if isinstance(query, (jax.Array, jax.core.Tracer))
        else float_dtype()
    )
    return stencil_at(
        profile,
        jnp.asarray(phase, dtype),
        jnp.asarray(first, dtype),
        size,
        reflected,
        beta,
        scale,
    )
