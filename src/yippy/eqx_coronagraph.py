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
        return psf.data + stellar.data
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image, PixelGrid
from jaxtyping import Array

from ._precision import float_dtype
from .apertures import DEFAULT_APERTURE_RULE, ApertureRule
from .coronagraph import Coronagraph
from .performance import (
    PerformanceCurves,
    profile_radii,
)


class EqxCoronagraph(eqx.Module):
    """Pure JAX/Equinox coronagraph -- no astropy, no scipy, no I/O at runtime.

    This module stores all coronagraph data as JAX arrays and interpax
    interpolators.  It is a valid pytree and can be passed through any JAX
    transformation.

    Fields fall into two categories when processed by ``eqx.filter_jit``:

    **Dynamic** (JAX arrays / eqx.Module leaves -- values can change without
    recompiling, *provided shapes stay the same*):

    - ``sky_trans``, ``psf_datacube``
    - All performance and stellar interpolators (they are ``eqx.Module``
      instances whose leaves are JAX arrays)

    **Static** (non-array Python objects -- changing triggers recompilation,
    but ``filter_jit`` handles this automatically):

    - ``create_psf``, ``create_psfs`` (callables / closures)
    - Scalar metadata (``pixel_scale_lod``, ``frac_obscured``, etc.)
    - ``psf_shape`` (tuple)

    Switching between different ``EqxCoronagraph`` instances inside a
    ``filter_jit``-compiled function **will** cause recompilation (different
    callable closures and likely different interpolator shapes).  This is
    expected and unavoidable.
    """

    # -- Scalar metadata (auto-static in filter_jit) -----------------------
    pixel_scale_lod: float
    psf_shape: tuple[int, int]
    grid: PixelGrid
    stellar_grid: PixelGrid
    sky_grid: PixelGrid
    frac_obscured: float

    # -- Off-axis PSF synthesis (auto-static: callables) ------------------
    _offaxis_backend: object
    stellar_pixel_scale_lod: float
    sky_pixel_scale_lod: float
    stellar_frame: tuple[float, float]
    sky_frame: tuple[float, float]

    # -- Stellar intensity interpolation (dynamic: eqx.Module) -----------
    _stellar_ln_interp: interpax.CubicSpline

    # -- Performance curves (dynamic: eqx.Module) ------------------------
    performance: PerformanceCurves
    _performance_records: tuple[PerformanceCurves, ...]
    occultation_record: PerformanceCurves
    _intensity_record: PerformanceCurves
    _core_intensity_varies_with_diameter: bool = eqx.field(static=True)
    _core_intensity_interpolation: str = eqx.field(static=True)
    _core_intensity_diameter: float = eqx.field(static=True)
    _diam_knot_range: tuple[float, float]
    _profile_radii_pix: Array
    _stellar_axis: tuple[float, float] = eqx.field(static=True)

    # -- Static arrays (dynamic) -----------------------------------------
    sky_trans: Image
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
        rule: ApertureRule = DEFAULT_APERTURE_RULE,
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
            rule:
                Static aperture rule for performance curves (forwarded).
            x_symmetric:
                Whether off-axis PSFs are symmetric about the x-axis (forwarded).
            y_symmetric:
                Whether off-axis PSFs are symmetric about the y-axis (forwarded).
            **kwargs:
                Additional keyword arguments to pass to ``yippy.Coronagraph``.

        Raises:
            ValueError: If neither ``yip_path`` nor ``yippy_coro`` is provided.
        """
        import inspect

        inspect.signature(Coronagraph).bind_partial(yip_path, **kwargs)
        if yippy_coro is None and yip_path is None:
            raise ValueError("Provide either yip_path or yippy_coro")

        if yippy_coro is None:
            yippy_coro = Coronagraph(
                yip_path,
                downsample_shape=downsample_shape,
                rule=rule,
                x_symmetric=x_symmetric,
                y_symmetric=y_symmetric,
                **kwargs,
            )

        if yippy_coro is not None:
            requested = kwargs.get("interpolation")
            if (
                requested is not None
                and requested != yippy_coro.interpolation_info.requested
            ):
                raise ValueError("Conflicting interpolation for existing Coronagraph")
            supplied = kwargs.get("sinc_options")
            if supplied is not None:
                inherited = yippy_coro._sinc_options
                if supplied.product_id != inherited.product_id or any(
                    eqx.tree_equal(getattr(supplied, key), getattr(inherited, key))
                    is not True
                    for key in ("domain", "memory", "policy")
                ):
                    raise ValueError(
                        "Conflicting sinc_options for existing Coronagraph"
                    )

        # -- Scalar metadata ---------------------------------------------
        self.pixel_scale_lod = float(yippy_coro.pixel_scale_arcsec.value)
        self.stellar_pixel_scale_lod = yippy_coro.stellar_intens.grid.step
        self.sky_pixel_scale_lod = yippy_coro.sky_trans.grid.step
        sky_header = getattr(yippy_coro, "sky_header", None)

        def frame_metadata(header):
            return (
                float(header.lambda0.to_value("nm"))
                if header.lambda0 is not None
                else float("nan"),
                float(header.diameter.to_value("m"))
                if header.diameter is not None
                else float("nan"),
            )

        self.stellar_frame = frame_metadata(yippy_coro.header)
        self.sky_frame = frame_metadata(sky_header or yippy_coro.header)
        self.psf_shape = tuple(map(int, yippy_coro.psf_shape))
        self.grid = yippy_coro.grid
        self.stellar_grid = yippy_coro.stellar_intens.grid
        self.sky_grid = yippy_coro.sky_trans.grid
        self.frac_obscured = float(yippy_coro.frac_obscured)

        # -- PSF creation callables --------------------------------------
        self._offaxis_backend = yippy_coro.offax

        # -- Stellar intensity interpolation -----------------------------
        # FITS-sourced arrays are big-endian (>f8); jnp cannot ingest those, so
        # an explicit dtype both byte-swaps to native and follows the x64 flag.
        stellar = yippy_coro.stellar_intens
        # Match the loader's zero-diameter nudge, including for curved profiles.
        stellar_diams = jnp.asarray(stellar.ln_interp.x, dtype=float_dtype())
        # Convert stellar PSFs to JAX arrays and build log-space interpolator
        stellar_psfs = jnp.asarray(stellar.psfs, dtype=float_dtype())
        self._stellar_ln_interp = interpax.CubicSpline(
            stellar_diams, jnp.log(stellar_psfs)
        )

        # Store the record's polynomial coefficients without fitting again.
        def jax_record(record):
            return jax.tree_util.tree_map(
                lambda value: (
                    jnp.asarray(value, dtype=float_dtype())
                    if isinstance(value, np.ndarray)
                    and np.issubdtype(value.dtype, np.floating)
                    else value
                ),
                record,
            )

        self.performance = jax_record(yippy_coro.performance)
        self._performance_records = tuple(
            self.performance if record is yippy_coro.performance else jax_record(record)
            for record in yippy_coro.curves_by_rule.values()
        )
        self.occultation_record = jax_record(yippy_coro.occultation_record)
        self._intensity_record = jax_record(yippy_coro._intensity_record)

        # Core mean intensity at other diameters: the azimuthal mean of the
        # interpolated stellar map on the profile radii, computed on demand.
        diam_values = np.asarray(stellar.diams.value, dtype=float)
        self._core_intensity_varies_with_diameter = len(diam_values) > 1
        self._core_intensity_interpolation = getattr(
            yippy_coro, "_core_intensity_interpolation", "keys"
        )
        self._core_intensity_diameter = getattr(
            yippy_coro, "_core_intensity_diameter", float(diam_values[0])
        )
        self._diam_knot_range = (float(diam_values.min()), float(diam_values.max()))
        self._stellar_axis = (self.stellar_grid.center_y, self.stellar_grid.center_x)
        self._profile_radii_pix = jnp.asarray(
            profile_radii(stellar.psfs.shape[-2:], self._stellar_axis),
            dtype=float_dtype(),
        )

        # -- Sky transmission --------------------------------------------
        self.sky_trans = yippy_coro.sky_trans()

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

    @property
    def curves_by_rule(self) -> dict[ApertureRule, PerformanceCurves]:
        """Performance records indexed by their static aperture rules."""
        return {record.rule: record for record in self._performance_records}

    # -- Public methods (all JIT-traceable) -------------------------------

    def create_psf(self, x_lod, y_lod):
        """Return one off-axis PSF, delegating dynamic delivery arrays."""
        return self._offaxis_backend.create_psf(x_lod, y_lod)

    def create_psfs(self, x, y):
        """Return off-axis PSFs at equal-length coordinate vectors."""
        return self._offaxis_backend.create_psfs(x, y)

    def create_psf_with_info(self, x_lod, y_lod):
        """Return fixed-product diagnostics for a versioned delivery."""
        if not hasattr(self._offaxis_backend, "create_psf_with_info"):
            raise ValueError(
                "with-info on EqxCoronagraph requires a versioned delivery"
            )
        return self._offaxis_backend.create_psf_with_info(x_lod, y_lod)

    def create_psfs_with_info(self, x, y):
        """Return fixed-product batch diagnostics for a versioned delivery."""
        if not hasattr(self._offaxis_backend, "create_psf_with_info"):
            raise ValueError(
                "with-info on EqxCoronagraph requires a versioned delivery"
            )
        return self._offaxis_backend.create_psfs_with_info(x, y)

    def stellar_intens(self, stellar_diam_lod: float) -> Image:
        """Interpolate the stellar intensity map for a given stellar diameter.

        Args:
            stellar_diam_lod: Stellar diameter in lam/D (unitless float).

        Returns:
            Image of per-pixel unocculted stellar flux fractions.
        """
        data = jnp.exp(self._stellar_ln_interp(stellar_diam_lod))
        axes = tuple(f"diameter_{i}" for i in range(data.ndim - 2))
        return Image(
            data,
            self.stellar_grid,
            axes,
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
            # The residual halo continues past the tabulated field.
            outside="reflect",
        )
