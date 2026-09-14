"""Pure JAX/Equinox coronagraph module.

This module provides ``EqxCoronagraph``, a first-class ``eqx.Module`` that
wraps the data loaded by :class:`yippy.Coronagraph` into a form that is fully
compatible with ``jax.jit``, ``jax.vmap``, and other JAX transformations.

Usage::

    from yippy import EqxCoronagraph

    # One-liner: pass a YIP path directly
    coro = EqxCoronagraph("/path/to/yip")

    # Or from an existing yippy Coronagraph
    from yippy import Coronagraph
    yippy_coro = Coronagraph("/path/to/yip")
    coro = EqxCoronagraph(yippy_coro=yippy_coro)

All methods on ``EqxCoronagraph`` are JIT-traceable.  Downstream code should
use ``eqx.filter_jit`` (not ``jax.jit``) when JIT-compiling functions that
accept an ``EqxCoronagraph`` as input::

    import equinox as eqx

    @eqx.filter_jit
    def simulate(coro, x, y):
        psf = coro.create_psf(x, y)
        stellar = coro.stellar_intens(0.01)
        return psf + stellar
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import interpax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._precision import float_dtype
from .coronagraph import Coronagraph


class EqxCoronagraph(eqx.Module):
    """Pure JAX/Equinox coronagraph -- no astropy, no scipy, no I/O at runtime.

    This module stores all coronagraph data as JAX arrays and interpax
    interpolators.  It is a valid pytree and can be passed through any JAX
    transformation.

    Fields fall into two categories when processed by ``eqx.filter_jit``:

    **Dynamic** (JAX arrays / eqx.Module leaves -- values can change without
    recompiling, *provided shapes stay the same*):

    - ``sky_trans``, ``psf_datacube``
    - All ``interpax.CubicSpline`` interpolators (they are ``eqx.Module``
      instances whose leaves are JAX arrays)

    **Static** (non-array Python objects -- changing triggers recompilation,
    but ``filter_jit`` handles this automatically):

    - ``create_psf``, ``create_psfs`` (callables / closures)
    - Scalar metadata (``pixel_scale_lod``, ``IWA``, ``OWA``, etc.)
    - ``psf_shape`` (tuple)

    Switching between different ``EqxCoronagraph`` instances inside a
    ``filter_jit``-compiled function **will** cause recompilation (different
    callable closures and likely different interpolator shapes).  This is
    expected and unavoidable.
    """

    # -- Scalar metadata (auto-static in filter_jit) -----------------------
    pixel_scale_lod: float
    psf_shape: tuple[int, int]
    center_x: float
    center_y: float
    IWA: float  # inner working angle in lam/D
    OWA: float  # outer working angle in lam/D
    frac_obscured: float
    contrast_floor: float | None

    # -- Off-axis PSF synthesis (auto-static: callables) ------------------
    create_psf: callable
    create_psfs: callable

    # -- Stellar intensity interpolation (dynamic: eqx.Module) -----------
    _stellar_ln_interp: interpax.CubicSpline

    # -- Performance curves (dynamic: eqx.Module) ------------------------
    _throughput_interp: interpax.CubicSpline
    _log_contrast_interp: interpax.CubicSpline
    _occ_trans_interp: interpax.CubicSpline
    _core_area_interp: interpax.CubicSpline
    _core_mean_intensity_interp: interpax.CubicSpline
    _core_mean_intensity_interp_2d: interpax.Interpolator2D | None
    _has_2d_core_intensity: bool
    _diam_knot_range: tuple[float, float]

    # -- Static arrays (dynamic) -----------------------------------------
    sky_trans: Array
    psf_datacube: Array | None

    # -- Construction -----------------------------------------------------

    def __init__(
        self,
        yip_path: str | Path | None = None,
        *,
        yippy_coro: Coronagraph | None = None,
        ensure_psf_datacube: bool = False,
        # Forwarded to yippy.Coronagraph when building from yip_path
        downsample_shape: tuple[int, int] | None = None,
        aperture_radius_lod: float = 0.7,
        contrast_floor: float | None = None,
        use_inscribed_diameter: bool = False,
        # Extra Coronagraph kwargs
        x_symmetric: bool = True,
        y_symmetric: bool = True,
        **kwargs,
    ):
        """Create a pure-JAX coronagraph from a YIP directory or existing Coronagraph.

        Args:
            yip_path:
                Path to a Yield Input Package directory.  If provided (and
                ``yippy_coro`` is not), a temporary ``yippy.Coronagraph`` is
                built internally.
            yippy_coro:
                An already-initialised ``yippy.Coronagraph`` instance.  Takes
                precedence over ``yip_path`` if both are given.
            ensure_psf_datacube:
                If ``True``, generate/load the 4-D PSF datacube and store it.
                The datacube can be very large; default is ``False``.
            downsample_shape:
                Optional ``(ny, nx)`` to downsample PSFs (forwarded).
            aperture_radius_lod:
                Aperture radius in lam/D for performance curves (forwarded).
            contrast_floor:
                Minimum contrast value for engineering stability floor (forwarded).
            use_inscribed_diameter:
                Whether to use inscribed diameter for lam/D calcs (forwarded).
            x_symmetric:
                Whether off-axis PSFs are symmetric about the x-axis (forwarded).
            y_symmetric:
                Whether off-axis PSFs are symmetric about the y-axis (forwarded).
            **kwargs:
                Additional keyword arguments to pass to ``yippy.Coronagraph``.

        Raises:
            ValueError: If neither ``yip_path`` nor ``yippy_coro`` is provided.
        """
        if yippy_coro is None and yip_path is None:
            raise ValueError("Provide either yip_path or yippy_coro")

        if yippy_coro is None:
            yippy_coro = Coronagraph(
                yip_path,
                downsample_shape=downsample_shape,
                aperture_radius_lod=aperture_radius_lod,
                contrast_floor=contrast_floor,
                x_symmetric=x_symmetric,
                y_symmetric=y_symmetric,
                use_inscribed_diameter=use_inscribed_diameter,
                **kwargs,
            )

        # -- Scalar metadata ---------------------------------------------
        self.pixel_scale_lod = float(yippy_coro.pixel_scale_arcsec.value)
        self.psf_shape = tuple(map(int, yippy_coro.psf_shape))
        self.center_x = float(yippy_coro.offax.center_x.value)
        self.center_y = float(yippy_coro.offax.center_y.value)
        self.IWA = float(yippy_coro.IWA.value)
        self.OWA = float(yippy_coro.OWA.value)
        self.frac_obscured = float(yippy_coro.frac_obscured)
        self.contrast_floor = (
            float(contrast_floor) if contrast_floor is not None else None
        )

        # -- PSF creation callables --------------------------------------
        self.create_psf = yippy_coro.offax.create_psf
        self.create_psfs = yippy_coro.offax.create_psfs

        # -- Stellar intensity interpolation -----------------------------
        # FITS-sourced arrays are big-endian (>f8); jnp cannot ingest those, so
        # an explicit dtype both byte-swaps to native and follows the x64 flag.
        stellar = yippy_coro.stellar_intens
        stellar_diams = jnp.asarray(stellar.diams.value, dtype=float_dtype())
        # Convert stellar PSFs to JAX arrays and build log-space interpolator
        stellar_psfs = jnp.asarray(stellar.psfs, dtype=float_dtype())
        self._stellar_ln_interp = interpax.CubicSpline(
            stellar_diams, jnp.log(stellar_psfs)
        )

        # -- Performance curve interpolators -----------------------------
        self._throughput_interp = _scipy_to_interpax(yippy_coro.throughput_interp)
        self._log_contrast_interp = _scipy_to_interpax(yippy_coro._log_contrast_interp)
        self._occ_trans_interp = _scipy_to_interpax(yippy_coro.occ_trans_interp)
        self._core_area_interp = _scipy_to_interpax(yippy_coro.core_area_interp)
        self._core_mean_intensity_interp = _scipy_to_interpax(
            yippy_coro.core_intensity_interp
        )

        # 2D core mean intensity (separation x stellar_diam) when available
        if yippy_coro.core_intensity_interp_2d is not None:
            rgi = yippy_coro.core_intensity_interp_2d
            # RegularGridInterpolator stores grid points in .grid
            sep_knots = jnp.asarray(rgi.grid[0], dtype=float_dtype())
            diam_knots = jnp.asarray(rgi.grid[1], dtype=float_dtype())
            values_2d = jnp.asarray(rgi.values, dtype=float_dtype())
            self._core_mean_intensity_interp_2d = interpax.Interpolator2D(
                sep_knots,
                diam_knots,
                values_2d,
                method="linear",
                extrap=False,  # returns NaN out-of-bounds
            )
            self._has_2d_core_intensity = True
            self._diam_knot_range = (
                float(jnp.min(diam_knots)),
                float(jnp.max(diam_knots)),
            )
        else:
            self._core_mean_intensity_interp_2d = None
            self._has_2d_core_intensity = False
            self._diam_knot_range = (0.0, 0.0)

        # -- Sky transmission --------------------------------------------
        self.sky_trans = jnp.asarray(yippy_coro.sky_trans(), dtype=float_dtype())

        # -- Optional PSF datacube ---------------------------------------
        if ensure_psf_datacube:
            if not yippy_coro.has_psf_datacube:
                yippy_coro.create_psf_datacube()
            datacube = yippy_coro.psf_datacube
            # jnp.asarray canonicalizes to the active float dtype and is a no-op
            # when the cube already matches (no copy).
            self.psf_datacube = jnp.asarray(datacube)
            # Release reference in yippy to avoid duplicate storage
            yippy_coro.psf_datacube = None
        else:
            self.psf_datacube = None

    # -- Public methods (all JIT-traceable) -------------------------------

    def stellar_intens(self, stellar_diam_lod: float) -> Array:
        """Interpolate the stellar intensity map for a given stellar diameter.

        Args:
            stellar_diam_lod: Stellar diameter in lam/D (unitless float).

        Returns:
            2-D JAX array containing the stellar intensity map.
        """
        return jnp.exp(self._stellar_ln_interp(stellar_diam_lod))

    def throughput(self, separation_lod: float) -> Array:
        """Evaluate coronagraph throughput at the given separation.

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar throughput value.
        """
        return self._throughput_interp(separation_lod)

    def raw_contrast(self, separation_lod: float) -> Array:
        """Evaluate raw contrast at the given separation (log-space interpolation).

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar raw contrast value.
        """
        result = jnp.power(10.0, self._log_contrast_interp(separation_lod))
        if self.contrast_floor is not None:
            result = jnp.maximum(result, self.contrast_floor)
        return result

    def noise_floor_exosims(
        self,
        separation_lod: float,
        contrast_floor: float = 1e-10,
        ppf: float = 30.0,
    ) -> Array:
        """Noise floor in EXOSIMS contrast convention.

        Computed as ``max(|raw_contrast|, contrast_floor) / ppf``.

        Args:
            separation_lod: Separation from the star in lambda/D.
            contrast_floor: Minimum contrast value.
            ppf: Post-processing noise suppression factor.

        Returns:
            Scalar noise floor value (EXOSIMS convention).
        """
        rc = jnp.abs(self.raw_contrast(separation_lod))
        return jnp.maximum(rc, contrast_floor) / ppf

    def noise_floor_ayo(
        self,
        separation_lod: float,
        ppf: float = 30.0,
    ) -> Array:
        """Noise floor in AYO/pyEDITH per-pixel convention.

        Computed as ``core_mean_intensity(sep) / ppf``.

        Args:
            separation_lod: Separation from the star in lambda/D.
            ppf: Post-processing noise suppression factor.

        Returns:
            Scalar noise floor value (AYO/pyEDITH convention).
        """
        return self.core_mean_intensity(separation_lod) / ppf

    def occulter_transmission(self, separation_lod: float) -> Array:
        """Evaluate occulter transmission at the given separation.

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar occulter transmission value.
        """
        return self._occ_trans_interp(separation_lod)

    def core_area(self, separation_lod: float) -> Array:
        """Evaluate core area at the given separation.

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar core area value in (lam/D)**2.
        """
        return self._core_area_interp(separation_lod)

    def core_mean_intensity(
        self, separation_lod: float, stellar_diam_lod: float = 0.0
    ) -> Array:
        """Evaluate core mean intensity at the given separation.

        Uses the 1D spline for the default diameter (point source) and
        the 2D interpolant for non-default stellar diameters when
        available.

        Args:
            separation_lod: Separation from the star in lambda/D.
            stellar_diam_lod: Stellar angular diameter in lambda/D.
                Default is 0.0 (point source).

        Returns:
            Scalar core mean intensity value.
        """
        point_source = self._core_mean_intensity_interp(separation_lod)
        if not self._has_2d_core_intensity:
            return point_source

        # Select with jnp.where rather than a Python if, so that a traced
        # stellar diameter works: a comparison against a tracer is itself a
        # tracer and cannot be used in a branch. Both interpolants are
        # therefore evaluated. The two-dimensional one returns NaN outside its
        # diameter knots and a NaN in the branch that is not selected still
        # propagates through a gradient, so the diameter is clamped into the
        # tabulated range before it is evaluated.
        low, high = self._diam_knot_range
        clamped = jnp.clip(stellar_diam_lod, low, high)
        extended = self._core_mean_intensity_interp_2d(separation_lod, clamped)
        return jnp.where(stellar_diam_lod == 0.0, point_source, extended)


# -- Helpers ------------------------------------------------------------------


def _scipy_to_interpax(scipy_spline):
    """Convert a ``scipy.interpolate.BSpline`` / ``make_interp_spline`` to interpax.

    The scipy spline stores knots (``t``) and coefficients (``c``).  We
    re-evaluate it on its interior knots (the original data x-values) and
    build a fresh interpax interpolator from those (x, y) pairs.

    For linear splines (k=1) we use ``interpax.Interpolator1D(method='linear')``.
    For cubic splines (k=3) we use ``interpax.CubicSpline``.

    Args:
        scipy_spline: A scipy BSpline or result of ``make_interp_spline``.

    Returns:
        An interpax interpolator that approximates the same function.
    """
    # Extract the unique interior knots (stripping the k+1 padded boundary
    # knots from each end).  For make_interp_spline the interior knots
    # exactly equal the original x data.
    k = scipy_spline.k
    t = scipy_spline.t
    x_np = np.unique(t[k:-k])

    # Evaluate the scipy spline on those x values
    y_np = scipy_spline(x_np)

    # Convert to JAX arrays
    x_jax = jnp.asarray(x_np, dtype=float_dtype())
    y_jax = jnp.asarray(y_np, dtype=float_dtype())

    if k <= 1:
        return interpax.Interpolator1D(x_jax, y_jax, method="linear", extrap=True)
    return interpax.CubicSpline(x_jax, y_jax)
