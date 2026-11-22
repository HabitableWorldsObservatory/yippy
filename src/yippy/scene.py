"""Scene operators: an extended source rendered through the stored PSF nodes."""

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

from ._jax_funcs import blend_pad, idw_read_plan, separable_read_plan
from ._precision import float_dtype
from .logger import logger
from .sinc.scene import AbstractSceneOperator


class _DenseSceneOperator(AbstractSceneOperator):
    """Native full or mirror-quadrant PSFs and their source admission mask.

    One PSF per source position: the route for a sinc delivery that falls back to
    blend per source, and the reference the blend operator is tested against.
    """

    psfs: jax.Array
    valid: jax.Array

    def __init__(self, psfs, valid):
        """Drop refused and nonfinite planes before any flux contraction."""
        psfs, valid = jnp.asarray(psfs), jnp.asarray(valid)
        if psfs.ndim != 4 or valid.shape != psfs.shape[:2] or valid.dtype != bool:
            raise ValueError("expected PSFs [sy,sx,y,x] and boolean validity [sy,sx]")
        self.valid = valid & jnp.all(jnp.isfinite(psfs), axis=(-1, -2))
        self.psfs = jnp.where(self.valid[..., None, None], psfs, 0.0)

    def render(self, flux, grid):
        """Return the native detector array and whether every lit source is valid."""
        if flux.shape != grid.shape:
            raise ValueError("source flux must match the native grid")
        if self.valid.shape == grid.shape:
            valid = jnp.all((flux == 0) | (jnp.isfinite(flux) & self.valid))
            image = jnp.einsum("ij,ijyx->yx", flux, self.psfs)
        else:
            ny, nx = grid.shape
            if not grid.is_geometric or self.valid.shape != (
                (ny + 1) // 2,
                (nx + 1) // 2,
            ):
                raise ValueError(
                    "quarter PSFs must match the positive quadrant of a geometric grid"
                )
            # Shared zero-offset axes belong to the positive halves on odd grids.
            sx, sy = math.ceil(grid.center_x), math.ceil(grid.center_y)
            padx, pady = nx % 2, ny % 2
            quadrants = jnp.stack(
                (
                    flux[sy:, sx:],
                    jnp.pad(flux[sy:, :sx][:, ::-1], ((0, 0), (padx, 0))),
                    jnp.pad(flux[:sy, :sx][::-1, ::-1], ((pady, 0), (padx, 0))),
                    jnp.pad(flux[:sy, sx:][::-1, :], ((pady, 0), (0, 0))),
                )
            )
            valid = jnp.all((quadrants == 0) | (jnp.isfinite(quadrants) & self.valid))
            parts = jnp.einsum("qij,ijyx->qyx", quadrants, self.psfs)
            image = (
                parts[0] + parts[1, :, ::-1] + parts[2, ::-1, ::-1] + parts[3, ::-1, :]
            )
        return jnp.where(valid, image, jnp.nan), valid


# Stored PSFs pre-shifted per build step: bounds the transient spectra.
_PRESHIFT_CHUNK = 8
# Sources per host-side neighbor search.
_PLAN_CHUNK = 8192
# Fractional shifts that agree within this many pixels share one table row.
_FRACTION_TOL = 1e-5
# Sources per batch of the dense per-source build.
DENSE_BATCH = 128


@dataclass(frozen=True)
class _BlendPlan:
    """Host-side regrouping of every source read by stored PSF node."""

    node: np.ndarray  # [term] stored PSF index
    frac_yx: np.ndarray  # [term, 2] sub-pixel shift of the stored PSF
    source: np.ndarray  # [term, S] flat source index, 0 on padding
    weight: np.ndarray  # [term, S] blend weight, 0 on padding
    reflection: np.ndarray  # [term, S] reflection class of the source
    shift_yx: np.ndarray  # [term, S, 2] integer displacement modulo n_fft
    flips: tuple
    n_fft: int
    shape: tuple
    itemsize: int

    def resident(self):
        """Bytes of every array the operator holds."""
        terms, per_term = self.source.shape
        half = self.n_fft // 2 + 1
        table = terms * self.n_fft * half * 2 * self.itemsize
        entries = terms * per_term * (4 + self.itemsize + 4 + 8)
        return table + entries + self.shape[0] * self.shape[1]

    def scratch(self, npix):
        """Peak transient bytes of a build or a render, whichever is larger."""
        padded = npix + 2 * blend_pad(npix)
        build = _PRESHIFT_CHUNK * padded * padded * self.itemsize * 4
        planes = len(self.flips) * self.n_fft * self.n_fft * self.itemsize
        return max(build, 4 * planes)


def _read_plan(backend):
    """The blend reader of a backend as a function of one source position.

    Call it inside ``jax.enable_x64()`` so the offsets keep double precision.
    """
    scale = float(backend.pixel_scale_arcsec.value)
    xsym, ysym = bool(backend.x_symmetric), bool(backend.y_symmetric)
    if backend.type == "1d":
        args = (
            *(
                jnp.asarray(a, jnp.float64)
                for a in (backend.x_offsets, backend.y_offsets)
            ),
            jnp.asarray(backend.offset_to_flat_idx),
        )

        def read(x, y):
            return separable_read_plan(x, y, scale, *args, xsym, ysym, "1d")

    else:
        args = tuple(
            jnp.asarray(a, jnp.float64)
            for a in (backend.flat_x_offsets, backend.flat_y_offsets)
        )

        def read(x, y):
            return idw_read_plan(x, y, scale, *args, xsym, ysym, backend.type)

    return read


def _blend_plan(backend, grid, valid, itemsize):
    """Neighbors, weights and shifts of every source read, grouped by table row.

    Runs on the host from the node offsets alone; no PSF pixel is touched.
    """
    ny, nx = grid.shape
    npix = backend.flat_psfs.shape[-1]
    if tuple(backend.flat_psfs.shape[-2:]) != (ny, nx):
        raise ValueError("source grid must match the stored PSF grid")
    xsym, ysym = bool(backend.x_symmetric), bool(backend.y_symmetric)
    with jax.enable_x64():
        xg, yg = grid.coordinates()
        xs, ys = np.asarray(xg, np.float64).ravel(), np.asarray(yg, np.float64).ravel()
        read = jax.jit(jax.vmap(_read_plan(backend)))
        parts = [
            [
                np.asarray(a)
                for a in read(
                    jnp.asarray(xs[i : i + _PLAN_CHUNK]),
                    jnp.asarray(ys[i : i + _PLAN_CHUNK]),
                )
            ]
            for i in range(0, xs.size, _PLAN_CHUNK)
        ]
    node, weight, shift_x, shift_y = (
        np.concatenate(c) for c in zip(*parts, strict=True)
    )
    admitted = np.asarray(valid, bool).ravel().copy()
    # The operator reads a source only inside the hull of the tabulated nodes,
    # the same domain the point reader has.
    admitted &= np.asarray(backend.in_source_domain(xs, ys))
    weight = np.where(admitted[:, None], weight, 0.0)
    flips = tuple(
        (fx, fy)
        for fy in ((0, 1) if ysym else (0,))
        for fx in ((0, 1) if xsym else (0,))
    )
    reflection = (xs < 0) * xsym + ((ys < 0) * ysym) * (2 if xsym else 1)
    source = np.broadcast_to(np.arange(xs.size)[:, None], node.shape)
    reflection = np.broadcast_to(reflection[:, None], node.shape)
    live = weight > 0
    node, weight, source = node[live], weight[live], source[live]
    reflection = reflection[live]
    shift_x, shift_y = shift_x[live], shift_y[live]
    if not node.size:
        node, source, reflection = (np.zeros(1, int) for _ in range(3))
        weight, shift_x, shift_y = (np.zeros(1) for _ in range(3))
    # Integer part of each shift; the remainder is shared by a node's reads.
    whole_x = np.floor(shift_x + 0.5 + _FRACTION_TOL)
    whole_y = np.floor(shift_y + 0.5 + _FRACTION_TOL)
    frac = np.stack((shift_y - whole_y, shift_x - whole_x), axis=1)
    keys = np.column_stack((node, np.round(frac / (4 * _FRACTION_TOL)).astype(int)))
    rows, term = np.unique(keys, axis=0, return_inverse=True)
    term = term.reshape(-1)
    count = np.bincount(term, minlength=rows.shape[0])
    mean_frac = np.stack(
        [np.bincount(term, frac[:, a]) / count for a in range(2)], axis=1
    )
    disp = np.stack((whole_y, whole_x), axis=1).astype(int)
    span = int((disp.max(axis=0) - disp.min(axis=0)).max())
    n_fft = int(next_fast_len(npix + span, real=True))
    order = np.argsort(term, kind="stable")
    start = np.concatenate(([0], np.cumsum(count)[:-1]))
    slot = np.arange(order.size) - start[term[order]]
    width = int(count.max())

    def pack(values, fill, dtype):
        out = np.full((rows.shape[0], width, *values.shape[1:]), fill, dtype)
        out[term[order], slot] = values[order]
        return out

    return _BlendPlan(
        node=rows[:, 0],
        frac_yx=mean_frac,
        source=pack(source, 0, np.int32),
        weight=pack(weight, 0.0, np.float64),
        reflection=pack(reflection, 0, np.int32),
        shift_yx=pack(disp % n_fft, 0, np.int32),
        flips=flips,
        n_fft=n_fft,
        shape=(ny, nx),
        itemsize=itemsize,
    )


@eqx.filter_jit
def _preshift(psfs, frac_yx, kx, ky, n_pad, n_fft):
    """Translate stored PSFs by their sub-pixel remainder; return their rfft2."""
    ny, nx = psfs.shape[-2:]
    pad = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    spec = jnp.fft.rfft2(jnp.pad(psfs, pad))
    phase = frac_yx[:, 1, None, None] * kx[None, None, :]
    phase = phase + frac_yx[:, 0, None, None] * ky[None, :, None]
    spec = spec * jnp.exp(-2j * jnp.pi * phase)
    shifted = jnp.fft.irfft2(spec, s=(ny + 2 * n_pad, nx + 2 * n_pad))
    window = shifted[:, n_pad : n_pad + ny, n_pad : n_pad + nx]
    return jnp.fft.rfft2(window, s=(n_fft, n_fft))


class BlendSceneOperator(AbstractSceneOperator):
    """A blend-interpolated scene render, regrouped by stored PSF node.

    A blend read is a weighted sum of the four nearest stored PSFs, each
    translated to the source. On the native pixel grid a translation is a whole
    number of pixels plus a remainder that depends on the stored PSF alone, so
    the image is one FFT product per stored PSF: the stored PSF shifted once by
    its remainder, times the map of source flux weighted at each whole-pixel
    displacement. The table holds one spectrum per stored PSF (not one PSF per
    source), and the render is exactly linear in the source flux.
    """

    table: jax.Array  # [term, n_fft, n_fft // 2 + 1] complex rfft2 of shifted PSFs
    source: jax.Array  # [term, S] int32 flat index into the source raster
    weight: jax.Array  # [term, S] blend weight, zero on padding
    reflection: jax.Array  # [term, S] int32 reflection class of the source
    shift_yx: jax.Array  # [term, S, 2] int32 whole-pixel displacement modulo n_fft
    valid: jax.Array  # [ny, nx] bool per-source admission
    n_fft: int = eqx.field(static=True)
    flips: tuple = eqx.field(static=True)  # (flip_x, flip_y) of each class
    shape: tuple = eqx.field(static=True)

    @classmethod
    def from_backend(cls, backend, grid, *, valid=None):
        """Regroup the reads of every source on ``grid`` by stored PSF.

        Args:
            backend: The blend backend holding the stored PSFs and offsets.
            grid: Native source grid, equal to the stored PSF grid.
            valid: Optional boolean [ny, nx] mask; sources outside it render
                nothing and mark the render invalid when lit.
        """
        dtype = float_dtype()
        valid = np.ones(grid.shape, bool) if valid is None else np.asarray(valid, bool)
        if valid.shape != tuple(grid.shape):
            raise ValueError("valid must match the source grid")
        if backend.type == "1d":
            logger.warning(
                "Building a scene operator from a 1D YIP: off-axis PSFs are "
                "translated to each source position, not rotated to its "
                "position angle, so azimuthal PSF structure keeps its +x "
                "orientation. Use a 2D YIP for spatially resolved scenes."
            )
        plan = _blend_plan(backend, grid, valid, jnp.dtype(dtype).itemsize)
        n_pad = blend_pad(backend.flat_psfs.shape[-1])
        psfs = jnp.asarray(backend.flat_psfs, dtype)
        rows = []
        for i in range(0, plan.node.size, _PRESHIFT_CHUNK):
            block = slice(i, i + _PRESHIFT_CHUNK)
            rows.append(
                _preshift(
                    psfs[plan.node[block]],
                    jnp.asarray(plan.frac_yx[block], dtype),
                    jnp.asarray(backend.kx, dtype),
                    jnp.asarray(backend.ky, dtype),
                    n_pad,
                    plan.n_fft,
                )
            )
        return cls(
            table=jnp.concatenate(rows),
            source=jnp.asarray(plan.source),
            weight=jnp.asarray(plan.weight, dtype),
            reflection=jnp.asarray(plan.reflection),
            shift_yx=jnp.asarray(plan.shift_yx),
            valid=jnp.asarray(valid),
            n_fft=plan.n_fft,
            flips=plan.flips,
            shape=plan.shape,
        )

    @staticmethod
    def size_bytes(backend, grid, valid=None):
        """Resident, scratch and total bytes of the operator, from shapes alone."""
        itemsize = jnp.dtype(float_dtype()).itemsize
        valid = np.ones(grid.shape, bool) if valid is None else valid
        plan = _blend_plan(backend, grid, valid, itemsize)
        resident = plan.resident()
        scratch = plan.scratch(backend.flat_psfs.shape[-1])
        return {"resident": resident, "scratch": scratch, "total": resident + scratch}

    @property
    def nbytes(self):
        """Bytes held by the operator's arrays."""
        return sum(leaf.nbytes for leaf in jax.tree.leaves(self))

    def render(self, flux, grid):
        """Return the native detector array and whether every lit source is valid."""
        if flux.shape != grid.shape or tuple(grid.shape) != self.shape:
            raise ValueError("source flux must match the native grid")
        ny, nx = self.shape
        dtype = self.weight.dtype
        n, planes = self.n_fft, len(self.flips)
        valid = jnp.all((flux == 0) | (jnp.isfinite(flux) & self.valid))
        lit = jnp.where(jnp.isfinite(flux), flux, 0.0).astype(dtype).ravel()

        def add_term(acc, term):
            table, source, weight, reflection, shift = term
            amp = jnp.zeros((planes, n, n), dtype)
            amp = amp.at[reflection, shift[:, 0], shift[:, 1]].add(lit[source] * weight)
            return acc + jnp.fft.rfft2(amp) * table, None

        acc = jnp.zeros((planes, n, n // 2 + 1), self.table.dtype)
        terms = (self.table, self.source, self.weight, self.reflection, self.shift_yx)
        acc, _ = jax.lax.scan(add_term, acc, terms)
        parts = jnp.fft.irfft2(acc, s=(n, n))[:, :ny, :nx]
        image = sum(
            part[:: -1 if fy else 1, :: -1 if fx else 1]
            for part, (fx, fy) in zip(parts, self.flips, strict=True)
        )
        return jnp.where(valid, image, jnp.nan), valid
