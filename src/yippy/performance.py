"""YIP offset selection and stellar-product radial diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.grids import Image
from optixstuff.performance import (
    CORE_WINDOW_LOD,
    SampledProfile,
    azimuthal_mean_profile,
    profile_radii,
)

from .logger import logger

if TYPE_CHECKING:
    from ._yip import _YIPData
    from .coronagraph import Coronagraph
CORE_STEP_LOD = 0.25


@dataclass
class _OffAxisPosition:
    """Data for a single off-axis PSF position along the performance axis."""

    separation: float  # |x| in lam/D
    image: Image  # the native off-axis PSF
    px: float  # nominal x position in the image, including subpixel phase
    py: float  # nominal y position in the image, including subpixel phase


def _iter_xaxis_positions(coro: _YIPData):
    """Yield :class:`_OffAxisPosition` for each valid x-offset (y ~= 0).

    Iterates over each valid x-offset position along the performance-curve
    axis. All native YIP offsets are included, even those where the nominal
    point-source location lies outside the image, because the truncation-ratio
    throughput remains meaningful as long as the translated PSF retains
    significant in-image power.
    """
    y_offsets = np.array(coro.offax.y_offsets)
    y_idx = int(np.argmin(np.abs(y_offsets)))
    y_lod_val = y_offsets[y_idx]

    if y_lod_val != 0:
        logger.warning(f"No PSF at y=0, using closest y offset: {y_lod_val:.3f} lam/D")

    for i, x_lod_val in enumerate(np.array(coro.offax.x_offsets)):
        if x_lod_val < 0:
            continue
        r = abs(x_lod_val)
        planet_image = coro.offax.get_psf_by_offset_idx(i, y_idx)
        if planet_image is None:
            continue
        with jax.enable_x64():
            px, py = coro.offax.grid.unit_to_pixel(x_lod_val, y_lod_val)
        px, py = float(px), float(py)
        yield _OffAxisPosition(separation=r, image=planet_image, px=px, py=py)


def core_separations(coro: Coronagraph, step_lod: float = CORE_STEP_LOD) -> np.ndarray:
    """Separations, in lambda/D, at which the PSF is synthesized for the core curves.

    This grid is used only when a caller opts into synthesized PSFs through
    ``separation_step_lod``. Default curves use the shipped PSFs at their own
    offsets. The optional uniform grid steps by ``step_lod`` from the smallest
    non-negative x offset to the
    largest one whose PSF center still lies in the image
    (``max_offset_in_image``), less one core window (:data:`CORE_WINDOW_LOD`).
    Closer to the edge the core is cut off by the image border and the
    band-limited interpolant wraps light from the opposite border into it,
    which changes the core throughput by up to 7 percent and the area by up
    to 11 percent; a synthesized PSF beyond the image is blank. Tabulated
    offsets beyond the end of the grid keep their own PSFs.

    The tabulated offsets inside that range are added to the grid: yippy
    mixes the two neighboring PSFs with weights linear in separation, so the
    synthesized PSF, and its core, change slope at each offset.

    Args:
        coro: Loaded public Coronagraph.
        step_lod: Grid spacing in lambda/D.

    Returns:
        1-D array of separations in lambda/D.
    """
    x_offsets = np.asarray(coro._offaxis_backend.x_offsets, dtype=float)
    x_offsets = x_offsets[x_offsets >= 0]
    edge = coro.grid.shape[1] * coro.grid.step / 2 - CORE_WINDOW_LOD
    s_max = max(min(x_offsets.max(), edge), x_offsets.min())
    n_steps = int(np.floor((s_max - x_offsets.min()) / step_lod + 1e-9))
    grid = x_offsets.min() + step_lod * np.arange(n_steps + 1)
    if s_max - grid[-1] > 1e-9:
        grid = np.append(grid, s_max)
    # The synthesized PSF changes slope at every tabulated offset, so the
    # offsets themselves are knots.
    return np.union1d(grid, x_offsets[x_offsets <= s_max])


def stellar_intensity_curves(
    backend, stellar_diameter_lod=0.0, *, interpolation="keys"
):
    """Measure a radial stellar ring mean in flux fraction per stellar pixel.

    This measurement is the ring mean of the stellar map, independent of an
    aperture rule's local leakage density. A traced diameter recomputes the
    profile from the interpolated map. A single supplied profile has no
    diameter dependence; otherwise out-of-range diameters yield NaN.
    The profile interpolates in linear intensity between sampled radii.
    The spatial reader is static under JIT; use ``eqx.filter_jit``.
    """
    if interpolation not in ("keys", "quintic"):
        raise ValueError("interpolation must be 'keys' or 'quintic'")
    grid = backend.stellar_grid
    radii = profile_radii(grid.shape, (grid.center_y, grid.center_x))
    nodes = radii * grid.step
    knots = backend._stellar_ln_interp.x
    diameter = jnp.asarray(stellar_diameter_lod)
    valid = (diameter >= 0) & (diameter <= knots[-1])
    image = (
        backend.stellar_map(None, jnp.clip(diameter, 0.0, knots[-1]), grid=None)
        .integrated()
        .data
    )
    image = jnp.asarray(image, dtype=jnp.asarray(nodes).dtype)
    profile = azimuthal_mean_profile(
        image,
        (grid.center_y, grid.center_x),
        jnp.asarray(radii),
        interpolation=interpolation,
    )
    return SampledProfile(
        nodes,
        jnp.where(valid, profile, jnp.nan),
        quantity="stellar_ring_mean_per_pixel",
        frame=grid.frame,
        interpolation="linear",
    )
