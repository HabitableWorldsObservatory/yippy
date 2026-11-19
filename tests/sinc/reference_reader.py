"""Independent finite-sum oracle using only NumPy and rational floor truth."""

import math
from fractions import Fraction
from functools import cache

import numpy as np

from yippy.sinc.geometry import origin_xy, spacing_xy


def reference_read(data, query, product=0, shift_y=0):
    """Return the raw finite sum and absolute weighted sum per detector pixel."""
    query = np.asarray(query, dtype=float).copy()
    query[np.abs(query) < np.finfo(float).tiny] = 0.0
    origin = np.asarray(origin_xy(data["declared_lattice"]))
    spacing = np.asarray(spacing_xy(data["declared_lattice"]))
    taps, weights = [], []
    for axis in range(2):
        difference = float(query[axis] - origin[axis])
        floor = math.floor(Fraction(difference) / Fraction(float(spacing[axis])))
        shift = shift_y if axis == 1 else 0
        indices = np.arange(floor - 15 + shift, floor + 17 + shift)
        phase = difference / spacing[axis] - indices
        active = np.abs(phase) < 16
        weight = np.zeros(32)
        u = phase[active]
        weight[active] = (
            np.sinc(u) * np.i0(12 * np.sqrt(1 - (u / 16) ** 2)) / np.i0(12.0)
        )
        taps.append(indices)
        weights.append(weight)
    shape = data["products"].pixels.shape
    nodes = np.asarray(data["intensity_nodes"])
    cube = nodes.reshape(nodes.shape[0], *data["declared_lattice"].shape, *shape)
    reflected = data["reflection"].kind == "two_axis_pixel_flip_v1"
    result, absolute = np.zeros(shape), np.zeros(shape)
    for j, iy in enumerate(taps[1]):
        row = np.zeros(shape)
        for i, ix in enumerate(taps[0]):
            sample = cube[
                product, abs(iy) if reflected else iy, abs(ix) if reflected else ix
            ].astype(float)
            if reflected and ix < 0:
                sample = sample[:, ::-1]
            if reflected and iy < 0:
                sample = sample[::-1, :]
            row += weights[0][i] * sample
            absolute += np.abs(weights[1][j] * weights[0][i] * sample)
        result += weights[1][j] * row
    return result, absolute


def _bump(u, taps, beta):
    """Kaiser-Bessel bump of support ``taps``, cut (not zero) at its edge."""
    u = np.asarray(u, dtype=float)
    value = np.zeros(u.shape)
    active = np.abs(u) < taps / 2
    root = np.sqrt(1 - (2 * u[active] / taps) ** 2)
    value[active] = np.i0(beta * root) / np.i0(beta)
    return value


@cache
def reference_filter(taps, beta, band, left, right):
    """Least-squares in-band inverse of the bump transform on offsets -left..right.

    The bump transform is integrated numerically, so this shares no closed form
    with the library. The read normalization cancels between filter and weights
    and is left at one.
    """
    grid = np.linspace(-taps / 2, taps / 2, 8001)
    step = grid[1] - grid[0]
    shape = _bump(grid, taps, beta)
    shape[0] = shape[-1] = 0.5 / np.i0(beta)
    frequency = np.linspace(0.0, band, 400)
    transform = (np.cos(2 * np.pi * frequency[:, None] * grid[None, :]) * shape).sum(
        1
    ) * step
    offsets = np.arange(-left, right + 1)
    phase = 2 * np.pi * frequency[:, None] * offsets[None, :]
    system = np.concatenate([np.cos(phase), np.sin(phase)])
    target = np.concatenate([1.0 / transform, np.zeros(frequency.size)])
    return np.linalg.lstsq(system, target, rcond=None)[0]


def reference_start(query, origin, spacing, taps):
    """First tap of a prefiltered stencil from exact rational arithmetic."""
    query = 0.0 if abs(query) < np.finfo(float).tiny else float(query)
    phase = Fraction(query - origin) / Fraction(spacing)
    if taps % 2:
        return math.floor(phase + Fraction(1, 2)) - (taps - 1) // 2
    return math.floor(phase) - taps // 2 + 1


def reference_prefiltered_read(data, query, taps, half_width, bands, betas, product=0):
    """Brute-force prefiltered read: unfold, filter node by node, then sum taps.

    A mirror-symmetric delivery is first unfolded into its full signed table by
    explicit detector flips (columns for negative x nodes, rows for negative y),
    so no folded index arithmetic is shared with the library. Every coefficient
    node the read touches is filtered with the offsets that exist on each side,
    up to ``half_width``. ``bands`` and ``betas`` are (x, y).
    """
    lattice = data["declared_lattice"]
    origin = np.asarray(origin_xy(lattice))
    spacing = np.asarray(spacing_xy(lattice))
    shape = data["products"].pixels.shape
    nodes = np.asarray(data["intensity_nodes"], dtype=float)
    sy, sx = lattice.shape
    cube = np.zeros((sy, sx, *shape))
    for row, (ix, iy) in enumerate(np.asarray(data["indices_xy"])):
        cube[iy, ix] = nodes[product, row]
    low = np.zeros(2, dtype=int)
    if data["reflection"].kind == "two_axis_pixel_flip_v1":
        signed = np.zeros((2 * sy - 1, 2 * sx - 1, *shape))
        for ky in range(1 - sy, sy):
            for kx in range(1 - sx, sx):
                image = cube[abs(ky), abs(kx)]
                if kx < 0:
                    image = image[:, ::-1]
                if ky < 0:
                    image = image[::-1, :]
                signed[ky + sy - 1, kx + sx - 1] = image
        cube, low = signed, np.array([1 - sx, 1 - sy])
    high = low + np.array(cube.shape[1::-1]) - 1
    starts, weights, filters = [], [], []
    for axis in range(2):
        start = reference_start(query[axis], origin[axis], spacing[axis], taps)
        indices = np.arange(start, start + taps)
        phase = (float(query[axis]) - origin[axis]) / spacing[axis]
        starts.append(indices)
        weights.append(_bump(phase - indices, taps, betas[axis]))
        filters.append(
            [
                (
                    int(min(half_width, k - low[axis])),
                    reference_filter(
                        taps,
                        float(betas[axis]),
                        float(bands[axis]),
                        int(min(half_width, k - low[axis])),
                        int(min(half_width, high[axis] - k)),
                    ),
                )
                for k in indices
            ]
        )
    result = np.zeros(shape)
    for j, ky in enumerate(starts[1]):
        left_y, filter_y = filters[1][j]
        for i, kx in enumerate(starts[0]):
            left_x, filter_x = filters[0][i]
            block = cube[
                ky - left_y - low[1] : ky - left_y - low[1] + filter_y.size,
                kx - left_x - low[0] : kx - left_x - low[0] + filter_x.size,
            ]
            coefficient = np.einsum("a,b,abrc->rc", filter_y, filter_x, block)
            result += weights[1][j] * weights[0][i] * coefficient
    return result
