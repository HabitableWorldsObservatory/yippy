"""YIP offset selection and stellar-product radial diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.grids import Image
from optixstuff.performance import (
    CORE_WINDOW_LOD,
    PerformanceCurves,
    SampledProfile,
    azimuthal_mean_profile,
    profile_radii,
)

from .logger import logger

if TYPE_CHECKING:
    from ._yip import _YIPData
    from .coronagraph import Coronagraph
CORE_STEP_LOD = 0.25

#: The star-leakage rules a Coronagraph records.
#:
#: ``aperture_ring_mean`` is the aperture leakage averaged over position angle:
#: at each separation s, the mean over angle of the exact-coverage aperture sum
#: with its centre at (s cos(phi), s sin(phi)). It is the default for a product
#: whose stellar map is not rotationally symmetric (a "2D" YIP).
#: ``aperture_mean`` reads the aperture on the +x axis only, which is the same
#: number at every angle for a rotationally symmetric ("1D") product and the
#: default there. ``azimuthal_ring_mean`` is the point-sample ring mean times the
#: aperture area; it is deprecated (see :func:`resolve_leakage_rule`).
RING_RULE = "aperture_ring_mean"
LEAKAGE_RULES = (RING_RULE, "aperture_mean", "azimuthal_ring_mean")
_RECORD_LABEL = {RING_RULE: "aperture_mean"}


def resolve_leakage_rule(leakage_rule, product_type):
    """Return the leakage rule to use, applying the default for the product.

    ``product_type`` is the off-axis backend type: ``"1d"`` for a rotationally
    symmetric product, anything else for a 2D product. ``None`` selects
    ``aperture_ring_mean`` for a 2D product and ``aperture_mean`` for a 1D one.
    ``azimuthal_ring_mean`` still works but is deprecated: the ring rule is its
    finite-aperture generalization (the two agree as the aperture shrinks to a
    point), so it is subsumed. It warns with ``DeprecationWarning``.
    """
    if leakage_rule is None:
        return "aperture_mean" if product_type == "1d" else RING_RULE
    if leakage_rule not in LEAKAGE_RULES:
        raise ValueError(
            f"unknown leakage_rule {leakage_rule!r}; expected one of {LEAKAGE_RULES}"
        )
    if leakage_rule == "azimuthal_ring_mean":
        warnings.warn(
            "leakage_rule 'azimuthal_ring_mean' is deprecated: it multiplies a "
            "point-sample ring mean by the aperture area and ignores the aperture "
            "extent; use 'aperture_ring_mean'",
            DeprecationWarning,
            stacklevel=3,
        )
    return leakage_rule


def record_label(leakage_rule):
    """The ``PerformanceCurves.leakage_rule`` label a yippy rule is stored under.

    A ring-mean record holds the aperture density of the azimuthally
    symmetrized stellar map, which is an ``aperture_mean`` record of that map,
    so it carries that label; the Coronagraph keeps the yippy rule.
    """
    return _RECORD_LABEL.get(leakage_rule, leakage_rule)


def ring_azimuth_samples(shape, center):
    """Default position angles per ring: one per pixel at the farthest ring.

    The map can carry harmonics up to half its circumference in pixels at its
    corner radius, so ``ceil(2 pi r_max)`` equally spaced angles (1131 for a
    256 by 256 map) average every ring without aliasing. Fewer angles leave
    structure that is narrower than their spacing unaveraged; the convergence
    ladder is in the tests and the documentation.
    """
    return int(np.ceil(2 * np.pi * np.max(profile_radii(shape, center))))


def ring_symmetrized(image, n_theta=None, grid=None):
    """Return ``image`` with every pixel replaced by the mean of its ring.

    The mean over position angle of the aperture sum at distance s from the
    axis equals the aperture sum at +x of the azimuthally averaged map, because
    the angle average of an aperture's indicator is rotationally symmetric. So
    weighting this map with the ordinary aperture pixel weights at +x gives the
    ring-averaged aperture leakage with the same circle and pixel weights.

    The ring mean at radius r is the mean of the map at ``n_theta`` equally
    spaced angles (in-image arcs only), evaluated at the profile radii of
    :func:`optixstuff.performance.profile_radii` and linear in r between them.
    ``n_theta`` defaults to :func:`ring_azimuth_samples`.

    ``grid`` selects the pixel grid of the result, the PSF grid when it differs
    from the stellar one: each pixel takes the ring mean at its own angular
    radius, in flux fraction per pixel of ``grid``, and zero beyond the stellar
    footprint. The default is the image's own grid.
    """
    image = image.integrated()
    sg = image.grid
    grid = sg if grid is None else grid
    center = (sg.center_y, sg.center_x)
    radii = profile_radii(sg.shape, center)
    if n_theta is None:
        n_theta = ring_azimuth_samples(sg.shape, center)
    sf, pf = sg.frame, grid.frame
    # Stellar pixels per pixel of the target grid, along a radius.
    ratio = (grid.step * pf.lod_to_arcsec(pf.reference_wavelength_nm)) / (
        sg.step * sf.lod_to_arcsec(sf.reference_wavelength_nm)
    )
    with jax.enable_x64():
        data = jnp.asarray(image.data, dtype=jnp.float64)
        profile = azimuthal_mean_profile(data, center, jnp.asarray(radii), n_theta)
        rows, cols = jnp.indices(grid.shape)
        stellar_radius = jnp.hypot(rows - grid.center_y, cols - grid.center_x) * ratio
        symmetric = jnp.interp(stellar_radius, jnp.asarray(radii), profile)
        symmetric = jnp.where(stellar_radius <= radii[-1], symmetric, 0.0) * ratio**2
    return replace(image, data=symmetric, grid=grid)


def ring_leakage_record(record, rule, offaxis, locations, stellar, oversample):
    """Return ``record`` with its leakage fields replaced by the ring-averaged aperture.

    ``record`` is the ordinary (+x) record of ``rule``: its throughput, area
    and working angle are kept. At each of its separations the leakage fraction
    becomes the aperture's pixel weights (``rule.weights`` on the PSF there, the
    weights that give the throughput) summed over the azimuthally symmetrized
    stellar map, and the density and raw contrast follow from it. The stellar
    map may be on a different grid from the PSFs (see :func:`ring_symmetrized`).
    """
    grid = offaxis(float(record.separation_lod[0])).grid
    sym = np.asarray(ring_symmetrized(stellar, grid=grid).data, dtype=float)
    leakage = np.empty(len(record.separation_lod))
    for i, s in enumerate(record.separation_lod):
        psf = offaxis(float(s))
        weights = np.asarray(rule.weights(psf, locations[i], oversample=oversample))
        leakage[i] = float(np.sum(weights * sym))
    area = np.asarray(record.core_area_lod2)
    density = np.divide(leakage, area, out=np.zeros_like(leakage), where=area > 0)
    tau = np.asarray(record.throughput)
    contrast = np.divide(
        density * area, tau, out=np.full_like(tau, 1e20), where=tau > 0
    )
    return PerformanceCurves(
        np.asarray(record.separation_lod),
        record.frame,
        record.rule,
        tau,
        area,
        leakage,
        density,
        contrast,
        record.iwa_lod,
        record.data_extent_lod,
        record.leakage_rule,
        record.stellar_diameter_lod,
        throughput_variance=np.asarray(record.throughput_variance),
        core_area_variance_lod2=np.asarray(record.core_area_variance_lod2),
        leakage_fraction_variance=leakage,
        leakage_density_variance_lod2=density,
    )


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
