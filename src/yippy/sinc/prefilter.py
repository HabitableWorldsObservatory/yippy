"""Deapodization filter design and the coefficient build for prefiltered reads.

A prefiltered read is the interpolation step of a nonuniform fast Fourier
transform moved into the sample domain. The read applies a compact
Kaiser-Bessel bump to a coefficient table; the droop of the bump's transform
inside the signal band is divided out once, when the table is loaded, by a short
filter along each source axis (Dutt and Rokhlin 1993; Fessler and Sutton 2003;
Barnett, Magland and af Klinteberg 2019). The composite of filter and bump is
a wide interpolating kernel, and only its narrow factor is paid per read.

On a tone exp(2 pi i f n), f in cycles per node, a read at fractional
registration tau returns R(tau, f) times the tone, with

    R(tau, f) = sum_k w(tau - k) D_k(f) exp(2 pi i f (k - tau)),

where w is the bump over its scale and D_k is the transfer function of the
filter that built coefficient node k. The tone error ``max |R - 1|`` over the
admitted band and over every covered read is the design figure recorded with
an admission. It is a property of the kernel under the declared band limit,
not an error bound on a delivered image.

Everything here runs eagerly in float64 on the host, once per load.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp

from yippy.sinc.admission import sampling_ratio
from yippy.sinc.evidence import PrecisionUnavailable
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.kernel import PREFILTER_HALO, bump

# Trial shapes: beta = factor * pi * taps * (1 - band).
_BETA_FACTORS = tuple(0.5 + 0.04 * step for step in range(16))
_FIT_FREQUENCIES = 400
_CHECK_FREQUENCIES = 301
# Float64 working-memory target per detector-row chunk of the coefficient build,
# and the working bytes one stored sample needs while its chunk is filtered.
BUILD_CHUNK_BYTES = 64 * 2**20
BUILD_SAMPLE_BYTES = 56


def _require_precision():
    if not jax.config.x64_enabled:
        raise PrecisionUnavailable(
            "Enable jax_enable_x64 before designing or building a prefilter"
        )


def _registrations():
    """Registrations in [0, 1] clustered at the stencil switch points.

    The bump is cut at a nonzero value, so the tone error jumps where the
    stencil changes (registration 0 for even tap counts, one half for odd) and
    is largest there. The set holds both points, their one-sided limits and
    Chebyshev clustering toward them.
    """
    lobatto = 0.25 * (1.0 - jnp.cos(jnp.pi * jnp.arange(65) / 64))
    limits = jnp.asarray([1e-9, 0.5 - 1e-9, 0.5 + 1e-9, 1.0 - 1e-9])
    return jnp.unique(jnp.concatenate([lobatto, 0.5 + lobatto, limits]))


def bump_transform(frequency, taps, beta):
    """Fourier transform of the bump at ``frequency`` in cycles per node.

    The compactly supported Kaiser-Bessel function has the closed transform
    taps * sinh(z) / (z * I0(beta)) with z**2 = beta**2 - (pi * taps * f)**2,
    continued as sin(z) / z beyond the main lobe.
    """
    square = beta**2 - (jnp.pi * taps * frequency) ** 2
    root = jnp.sqrt(jnp.abs(square))
    safe = jnp.where(root > 1e-8, root, 1.0)
    lobe = jnp.where(square >= 0, jnp.sinh(safe), jnp.sin(safe)) / safe
    return taps * jnp.where(root > 1e-8, lobe, 1.0) / jnp.i0(beta)


def bump_scale(taps, beta):
    """Mean tap sum of the bump over registrations, so read weights sum to about 1."""
    registration = (8.0 * jnp.arange(64) + 0.5) / 512.0
    offsets = jnp.arange(-taps, taps + 1)
    return float(
        jnp.mean(jnp.sum(bump(registration[:, None] - offsets, taps, beta), 1))
    )


def _fit(taps, beta, scale, band, left, right):
    """Minimum-norm least-squares filter on offsets -left..right.

    The fit asks D(f) * bump_transform(f) = scale on the band only and imposes
    nothing outside it, which keeps the filter short and its gain small. The
    stacked system has full column rank only near the widest admitted bands
    (about 0.3 cycles per node and above). On a narrower band it is rank
    deficient to working precision, and the solver returns the minimum-norm
    solution among the least-squares ones after discarding singular values
    below ``rcond`` times the largest. The cutoff is pinned to the binary64
    epsilon times the larger dimension of the system, so the choice does not
    depend on a library default.
    """
    offsets = jnp.arange(-left, right + 1)
    frequency = jnp.linspace(0.0, band, _FIT_FREQUENCIES)
    phase = 2 * jnp.pi * frequency[:, None] * offsets[None, :]
    system = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])
    target = jnp.concatenate(
        [scale / bump_transform(frequency, taps, beta), jnp.zeros(_FIT_FREQUENCIES)]
    )
    rcond = float(jnp.finfo(jnp.float64).eps) * max(system.shape)
    return jnp.linalg.lstsq(system, target, rcond=rcond)[0]


class _Filters:
    """Filters of one axis design, keyed by the stored nodes on each side."""

    def __init__(self, taps, beta, scale, band, half_width):
        self.taps, self.beta, self.scale = taps, beta, scale
        self.band, self.half_width = band, half_width
        self._cache = {}

    def get(self, left, right):
        """Return the filter on offsets -left..right, mirrored where left < right."""
        key = (left, right)
        if key not in self._cache:
            if left < right:
                value = self.get(right, left)[::-1]
            else:
                value = _fit(self.taps, self.beta, self.scale, self.band, left, right)
                if left == right:
                    value = 0.5 * (value + value[::-1])
            self._cache[key] = value
        return self._cache[key]

    def transfer(self, left, right, frequency):
        """Transfer function D(f) of one filter, relative to its own node."""
        offsets = jnp.arange(-left, right + 1)
        phase = 2j * jnp.pi * frequency[:, None] * offsets[None, :]
        return jnp.exp(phase) @ self.get(left, right).astype(jnp.complex128)


def _tap_terms(taps, beta, scale, registration, frequency):
    """Relative tap offsets, bump weights [tau, k] and tone terms [tau, k, f]."""
    offsets = jnp.arange(-(taps // 2) - 1, taps // 2 + 2)
    weight = bump(registration[:, None] - offsets, taps, beta) / scale
    shift = offsets[None, :, None] - registration[:, None, None]
    tone = jnp.exp(2j * jnp.pi * shift * frequency[None, None, :])
    return offsets, weight, weight[:, :, None] * tone


def _interior_error(taps, beta, band, half_width):
    """Worst in-band tone error of an interior read on a coarse trial grid."""
    scale = bump_scale(taps, beta)
    filters = _Filters(taps, beta, scale, band, half_width)
    frequency = jnp.linspace(0.0, band, _CHECK_FREQUENCIES)
    registration = (8.0 * jnp.arange(64) + 0.5) / 512.0
    _, _, terms = _tap_terms(taps, beta, scale, registration, frequency)
    response = terms.sum(1) * filters.transfer(half_width, half_width, frequency)
    return float(jnp.max(jnp.abs(response - 1.0))), filters


@dataclass(frozen=True)
class AxisDesign:
    """Host design of one source axis: filter matrices and their figures.

    ``direct`` [size, size] multiplies the stored samples and ``mirrored``
    [size, size] multiplies their detector-flipped images (None without
    reflection), so that coefficients = direct @ samples + mirrored @ flipped.
    ``tone_error`` is the worst in-band tone error over every covered read the
    served cells can make, edge reads included; ``interior_tone_error`` is the
    figure of reads whose coefficient nodes all have the full filter half-width
    (equal to ``tone_error`` when the served cells hold no such read).
    ``sample_gain`` is the worst absolute weight sum that carries a stored-sample
    error into a read. All three are maxima over a finite set of frequencies and
    registrations: estimates of the kernel's worst case, not outward bounds.
    """

    taps: int
    filter_half_width: int
    band: float
    beta: float
    scale: float
    tone_error: float
    interior_tone_error: float
    sample_gain: float
    direct: jax.Array
    mirrored: jax.Array | None


def design_axis(taps, filter_half_width, band, size, reflected, cells=None):
    """Design the bump shape and every filter one source axis needs.

    Designs are cached by their arguments; the precision check runs on every
    call, cached or not.

    Args:
        taps: Bump taps per axis.
        filter_half_width: Half-width of the interior filter in nodes.
        band: Band edge in cycles per node (half the sampling ratio), below 0.5.
        size: Stored node count along the axis.
        reflected: Whether the stored nodes are the nonnegative half of a
            mirror-symmetric signed table.
        cells: Optional (first, last) signed cell indices floor(t) that reads
            may fall in; None means every covered cell. It only narrows the
            reads that the recorded figures cover.

    Returns:
        The AxisDesign. A coefficient node closer than the half-width to an
        end of the signed table gets a one-sided minimum-norm filter that uses
        only stored nodes; across a mirror axis the filter runs over reflected
        samples and needs no edge treatment.
    """
    _require_precision()
    return _design_axis(taps, filter_half_width, band, size, reflected, cells)


@lru_cache(maxsize=32)
def _design_axis(taps, filter_half_width, band, size, reflected, cells):
    """Uncached body of ``design_axis``; arguments are hashable host values."""
    if not 0.0 < band < 0.5:
        raise ValueError("The design band must lie strictly inside (0, 0.5)")
    trials = []
    for factor in _BETA_FACTORS:
        beta = factor * math.pi * taps * (1.0 - band)
        error, filters = _interior_error(taps, beta, band, filter_half_width)
        trials.append((error, beta, filters))
    _, beta, filters = min(trials, key=lambda trial: trial[0])
    scale = filters.scale
    low, high = (1 - size, size - 1) if reflected else (0, size - 1)
    signed = high - low + 1
    sides = [
        (min(filter_half_width, node - low), min(filter_half_width, high - node))
        for node in range(low, high + 1)
    ]
    matrix = jnp.zeros((signed, signed))
    for row, (left, right) in enumerate(sides):
        matrix = matrix.at[row, row - left : row + right + 1].set(
            filters.get(left, right)
        )
    registration = _registrations()
    frequency = jnp.linspace(0.0, band, _CHECK_FREQUENCIES)
    offsets, weight, terms = _tap_terms(taps, beta, scale, registration, frequency)
    # First tap of the stencil relative to the cell base, per registration.
    if taps % 2:
        first = jnp.where(registration >= 0.5, 1, 0) - (taps - 1) // 2
    else:
        first = jnp.full(registration.shape, 1 - taps // 2)
    worst, interior, gain = {}, 0.0, 0.0
    first_cell, last_cell = (low, high) if cells is None else cells
    for base in range(max(low, first_cell), min(high, last_cell) + 1):
        covered = (base + first >= low + PREFILTER_HALO) & (
            base + first + taps - 1 <= high - PREFILTER_HALO
        )
        if not bool(jnp.any(covered)):
            continue
        rows = [min(max(base + int(k), low), high) - low for k in offsets]
        pattern = tuple(sides[row] for row in rows)
        key = (pattern, tuple(covered.tolist()))
        if key not in worst:
            transfer = jnp.stack(
                [filters.transfer(left, right, frequency) for left, right in pattern]
            )
            response = jnp.einsum("tkf,kf->tf", terms, transfer)
            error = jnp.max(jnp.abs(response - 1.0), axis=1)
            composite = jnp.zeros((registration.size, offsets.size + 2 * signed))
            for column, (left, right) in enumerate(pattern):
                start = signed + column - left
                composite = composite.at[:, start : start + left + right + 1].add(
                    weight[:, column : column + 1] * filters.get(left, right)[None, :]
                )
            l1 = jnp.sum(jnp.abs(composite), axis=1)
            worst[key] = (
                float(jnp.max(jnp.where(covered, error, 0.0))),
                float(jnp.max(jnp.where(covered, l1, 0.0))),
            )
            if all(side == (filter_half_width,) * 2 for side in pattern):
                interior = max(interior, worst[key][0])
        gain = max(gain, worst[key][1])
    tone_error = max((value[0] for value in worst.values()), default=math.inf)
    if reflected:
        direct = matrix[size - 1 :, size - 1 :]
        mirrored = matrix[size - 1 :, size - 1 :: -1].at[:, 0].set(0.0)
    else:
        direct, mirrored = matrix, None
    return AxisDesign(
        taps,
        filter_half_width,
        band,
        beta,
        scale,
        tone_error,
        interior or tone_error,
        gain,
        direct,
        mirrored,
    )


def sampling_ratios(products, spacing):
    """Yield (axis, ratio) for every declared minimum wavelength of every product.

    The ratio is 2 B h with B = (reference / wavelength) * support extent, the
    quantity admission compares against Nyquist and the kernel margin.
    Non-finite and non-positive minima are skipped; admission refuses those
    through its metadata checks.
    """
    reference = float(products.frame.reference_wavelength_nm)
    campaign = float(products.campaign_sampling_min_nm)
    for minimum, extents in zip(
        products.sampling_min_nm.tolist(),
        products.support_extent_xy.tolist(),
        strict=True,
    ):
        minima = [minimum] if math.isnan(campaign) else [minimum, campaign]
        for value in minima:
            if not math.isfinite(value) or value <= 0:
                continue
            for axis, (extent, pitch) in enumerate(zip(extents, spacing, strict=True)):
                # Python binary64 operations preserve the declared separate rounding.
                yield axis, sampling_ratio(reference, value, extent, pitch)


def design_band(products, spacing):
    """Band edge per source axis in cycles per node, at the shortest wavelength.

    The value is half the largest sampling ratio 2 B h over products and over
    their declared minimum wavelengths, so the filter is designed for the blue
    edge of the band and covers every longer wavelength.
    """
    edges = [0.0, 0.0]
    for axis, ratio in sampling_ratios(products, spacing):
        edges[axis] = max(edges[axis], ratio / 2.0)
    return tuple(edges)


def served_cells(domain, lattice):
    """Signed cell range per axis, (x, y), that the domain's bounding box spans.

    One extra cell on each side absorbs coordinate rounding. Coverage, not this
    range, decides which reads are served; the range only keeps unreachable
    table ends out of the recorded design figures.
    """
    origin = origin_xy(lattice).tolist()
    spacing = spacing_xy(lattice).tolist()
    if domain.kind == "box":
        limits = domain.limits_xy_lod.tolist()
    else:
        outer = float(domain.radii_lod[1])
        limits = [[-outer, -outer], [outer, outer]]
    return tuple(
        (
            math.floor((limits[0][axis] - origin[axis]) / spacing[axis]) - 1,
            math.floor((limits[1][axis] - origin[axis]) / spacing[axis]) + 1,
        )
        for axis in range(2)
    )


def design_grid(profile, products, lattice, reflected, domain=None):
    """Return the (x, y) AxisDesign pair for an admitted lattice and profile."""
    spacing = spacing_xy(lattice).tolist()
    band = design_band(products, spacing)
    sy, sx = lattice.shape
    cells = (None, None) if domain is None else served_cells(domain, lattice)
    return tuple(
        design_axis(
            profile.taps,
            profile.filter_half_width,
            edge,
            size,
            bool(reflected),
            cell,
        )
        for edge, size, cell in zip(band, (sx, sy), cells, strict=True)
    )


def design_record(designs):
    """Flatten an (x, y) design pair into the (key, value) pairs of a receipt."""
    x, y = designs
    two_axis = x.tone_error + y.tone_error + x.tone_error * y.tone_error
    return (
        ("taps", x.taps),
        ("filter_half_width", x.filter_half_width),
        ("halo", PREFILTER_HALO),
        ("band_x", x.band),
        ("band_y", y.band),
        ("beta_x", x.beta),
        ("beta_y", y.beta),
        ("tone_error_x", x.tone_error),
        ("tone_error_y", y.tone_error),
        ("tone_error_two_axis", two_axis),
        ("interior_tone_error_x", x.interior_tone_error),
        ("interior_tone_error_y", y.interior_tone_error),
        ("sample_gain_x", x.sample_gain),
        ("sample_gain_y", y.sample_gain),
    )


def build_chunk_rows(nodes, shape_yx):
    """Detector rows filtered per chunk under the float64 working-memory target."""
    row_bytes = BUILD_SAMPLE_BYTES * nodes * shape_yx[1]
    return max(1, min(shape_yx[0], BUILD_CHUNK_BYTES // row_bytes))


@eqx.filter_jit
def _filter_rows(samples, flipped, direct, mirrored, dtype):
    """Filter one detector-row chunk [sy, sx, rows, px] along both source axes."""
    direct_x, direct_y = direct
    samples = samples.astype(jnp.float64)
    along_x = jnp.einsum("km,jmrc->jkrc", direct_x, samples)
    if mirrored is None:
        return jnp.einsum("kj,jmrc->kmrc", direct_y, along_x).astype(dtype)
    mirrored_x, mirrored_y = mirrored
    flipped = flipped.astype(jnp.float64)
    along_x = along_x + jnp.einsum("km,jmrc->jkrc", mirrored_x, samples[..., ::-1])
    flipped_x = jnp.einsum("km,jmrc->jkrc", direct_x, flipped) + jnp.einsum(
        "km,jmrc->jkrc", mirrored_x, flipped[..., ::-1]
    )
    return (
        jnp.einsum("kj,jmrc->kmrc", direct_y, along_x)
        + jnp.einsum("kj,jmrc->kmrc", mirrored_y, flipped_x)
    ).astype(dtype)


def build_coefficients(intensity, designs):
    """Filter stored samples [P, sy, sx, py, px] into coefficients of the same shape.

    The filter arithmetic is float64 whatever the storage dtype; the result is
    rounded once to the storage dtype. Detector rows are processed in chunks so
    the float64 working set stays near a fixed size. With reflection, samples at
    negative node indices are the detector-flipped stored images, which lets the
    filter run across both mirror axes without an edge.
    """
    _require_precision()
    design_x, design_y = designs
    direct = (design_x.direct, design_y.direct)
    mirrored = None
    if design_x.mirrored is not None:
        mirrored = (design_x.mirrored, design_y.mirrored)
    count, sy, sx, py, px = intensity.shape
    rows = build_chunk_rows(sy * sx, (py, px))
    products = []
    for product in range(count):
        pieces = []
        for start in range(0, py, rows):
            stop = min(py, start + rows)
            samples = intensity[product, :, :, start:stop]
            flipped = None
            if mirrored is not None:
                flipped = intensity[product, :, :, py - stop : py - start][:, :, ::-1]
            pieces.append(
                _filter_rows(samples, flipped, direct, mirrored, intensity.dtype)
            )
        products.append(jnp.concatenate(pieces, axis=2))
    return jnp.stack(products)
