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
from hwoutils.grids import Image, PixelGrid, resample
from jaxtyping import Array

from ._precision import float_dtype
from .apertures import ApertureRule, Circular
from .coronagraph import Coronagraph
from .performance import (
    PerformanceCurves,
    azimuthal_mean_profile,
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
    - Scalar metadata (``pixel_scale_lod``, ``data_extent_lod``, etc.)
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
    contrast_floor: float | None

    # -- Off-axis PSF synthesis (auto-static: callables) ------------------
    _legacy_create_psf: callable
    _legacy_create_psfs: callable
    _offaxis_backend: eqx.Module | None
    stellar_pixel_scale_lod: float
    sky_pixel_scale_lod: float
    stellar_frame: tuple[float, float]
    sky_frame: tuple[float, float]

    # -- Stellar intensity interpolation (dynamic: eqx.Module) -----------
    _stellar_ln_interp: interpax.CubicSpline

    # -- Performance curves (dynamic: eqx.Module) ------------------------
    performance: PerformanceCurves
    _performance_records: tuple[PerformanceCurves, ...]
    _occultation_record: PerformanceCurves
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
        rule: ApertureRule = Circular(0.7),  # noqa: B008 - frozen rule
        contrast_floor: float | None = None,
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
            contrast_floor:
                Minimum contrast value for engineering stability floor (forwarded).
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
                contrast_floor=contrast_floor,
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
        self.contrast_floor = (
            float(contrast_floor) if contrast_floor is not None else None
        )

        # -- PSF creation callables --------------------------------------
        self._offaxis_backend = (
            yippy_coro.offax if yippy_coro._sinc_delivery is not None else None
        )
        self._legacy_create_psf = (
            yippy_coro.offax.create_psf if self._offaxis_backend is None else None
        )
        self._legacy_create_psfs = (
            yippy_coro.offax.create_psfs if self._offaxis_backend is None else None
        )

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
        self._occultation_record = jax_record(yippy_coro._occultation_record)
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

    @property
    def iwa_lod(self) -> float:
        """First half-maximum-throughput crossing in the record's frame."""
        return self.performance.iwa_lod

    @property
    def data_extent_lod(self) -> float:
        """Array half-width in the record's frame."""
        return self.performance.data_extent_lod

    # -- Public methods (all JIT-traceable) -------------------------------

    def create_psf(self, x_lod, y_lod):
        """Return one off-axis PSF, delegating dynamic delivery arrays."""
        if self._offaxis_backend is not None:
            return self._offaxis_backend.create_psf(x_lod, y_lod)
        return self._legacy_create_psf(x_lod, y_lod)

    def create_psfs(self, x, y):
        """Return off-axis PSFs at equal-length coordinate vectors."""
        if self._offaxis_backend is not None:
            return self._offaxis_backend.create_psfs(x, y)
        return self._legacy_create_psfs(x, y)

    def create_psf_with_info(self, x_lod, y_lod):
        """Return fixed-product diagnostics for a versioned delivery."""
        if self._offaxis_backend is None:
            raise ValueError(
                "with-info on EqxCoronagraph requires a versioned delivery"
            )
        return self._offaxis_backend.create_psf_with_info(x_lod, y_lod)

    def create_psfs_with_info(self, x, y):
        """Return fixed-product batch diagnostics for a versioned delivery."""
        if self._offaxis_backend is None:
            raise ValueError(
                "with-info on EqxCoronagraph requires a versioned delivery"
            )
        return self._offaxis_backend.create_psfs_with_info(x, y)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, pixel_scale_lod, shape):
        """Resample the fixed off-axis product in its reference lambda/D frame.

        Args:
            wavelength_nm: Accepted for fixed-product table compatibility;
                it does not select or rescale another spectral product.
            x_lod: Source x coordinates in the off-axis reference frame.
            y_lod: Source y coordinates in the off-axis reference frame.
            pixel_scale_lod: Requested detector pitch in that same frame.
            shape: Requested detector shape (row, column).

        Returns:
            Photon fraction per target pixel for each source, using the PSF's
            declared reconstruction. Coarsening can downgrade the declaration.
        """
        target = PixelGrid.regular(shape, pixel_scale_lod, "lod", frame=self.grid.frame)
        return resample(self.create_psfs(x_lod, y_lod), target)

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, pixel_scale_lod, shape):
        """Resample stellar photon fractions using the stellar reference pitch.

        Args:
            wavelength_nm: Accepted for fixed-product table compatibility.
            stellar_diam_lod: Diameter in the stellar reference frame.
            pixel_scale_lod: Target pitch in the stellar reference frame.
            shape: Requested detector shape (row, column).

        Returns:
            Stellar photon fractions per target pixel, using the intensity's
            declared reconstruction. Coarsening can downgrade the declaration.
        """
        target = PixelGrid.regular(
            shape, pixel_scale_lod, "lod", frame=self.stellar_grid.frame
        )
        return resample(self.stellar_intens(stellar_diam_lod), target)

    def background_transmission(self, wavelength_nm, *, pixel_scale_lod, shape):
        """Resample the dimensionless sky mask with its own reference pitch.

        Args:
            wavelength_nm: Accepted for fixed-product table compatibility.
            pixel_scale_lod: Target pitch in the sky reference frame.
            shape: Requested detector shape (row, column).

        Returns:
            Dimensionless transmission using the delivered map's reconstruction.
        """
        target = PixelGrid.regular(
            shape, pixel_scale_lod, "lod", frame=self.sky_grid.frame
        )
        return resample(self.sky_trans, target)

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
        )

    def throughput(self, separation_lod: float) -> Array:
        """Evaluate coronagraph throughput at the given separation.

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar throughput value.
        """
        return self.performance.evaluate("throughput", jnp.asarray(separation_lod))

    def raw_contrast(self, separation_lod: float) -> Array:
        """Evaluate raw contrast at the given separation (log-space interpolation).

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar raw contrast value.
        """
        result = self.performance.evaluate("raw_contrast", jnp.asarray(separation_lod))
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
        interpolation: str = "keys",
    ) -> Array:
        """Noise floor in AYO/pyEDITH per-pixel convention.

        Computed as ``core_mean_intensity(sep) / ppf``.

        Args:
            separation_lod: Separation from the star in lambda/D.
            ppf: Post-processing noise suppression factor.
            interpolation: Spatial reader, ``"keys"`` or ``"quintic"``.

        Returns:
            Scalar noise floor value (AYO/pyEDITH convention).
        """
        return (
            self.core_mean_intensity(separation_lod, interpolation=interpolation) / ppf
        )

    def occulter_transmission(self, separation_lod: float) -> Array:
        """Evaluate occulter transmission at the given separation.

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar occulter transmission value.
        """
        return self._occultation_record.evaluate(
            "occulter_transmission", jnp.asarray(separation_lod)
        )

    def core_area(self, separation_lod: float) -> Array:
        """Evaluate core area at the given separation.

        Args:
            separation_lod: Separation from the star in lam/D.

        Returns:
            Scalar core area value in (lam/D)**2.
        """
        return self.performance.evaluate("core_area_lod2", jnp.asarray(separation_lod))

    def core_mean_intensity(
        self,
        separation_lod: float,
        stellar_diam_lod: float = 0.0,
        interpolation: str = "keys",
    ) -> Array:
        """Evaluate core mean intensity at the given separation.

        Uses the 1D spline for the default diameter (point source). At any
        other diameter the value is the azimuthal mean of
        :meth:`stellar_intens` at that diameter, so the curve and the map
        describe the same star.

        Each call at a nonzero diameter interpolates one stellar map and
        computes its azimuthal profile. Separation interpolation uses PCHIP
        with linear end-segment tails, as for the precomputed record.

        Args:
            separation_lod: Separation from the star in lambda/D.
            stellar_diam_lod: Stellar angular diameter in lambda/D.
                Default is 0.0 (point source).
            interpolation: Spatial reader, ``"keys"`` (default) or
                ``"quintic"``. Static under JIT; use ``eqx.filter_jit``.
                A reader differing from the precomputed table recomputes from
                the map, including for a point source.

        Returns:
            Core mean intensity with the shape of separation_lod: the azimuthal
            mean, about the optical axis, of the fraction of the star's light falling on
            one pixel. The unit is per pixel of ``pixel_scale_lod``, not per
            unit solid angle; divide by ``pixel_scale_lod**2`` for
            (lambda/D)^-2.
        """
        separation_lod = jnp.asarray(separation_lod)
        if interpolation not in ("keys", "quintic"):
            raise ValueError("interpolation must be 'keys' or 'quintic'")
        if interpolation != self._core_intensity_interpolation:
            low, high = self._diam_knot_range
            diameter = jnp.where(
                stellar_diam_lod == 0.0,
                self._core_intensity_diameter,
                stellar_diam_lod,
            )
            if not self._core_intensity_varies_with_diameter:
                diameter = self._core_intensity_diameter
            result = self._core_mean_intensity_at(
                separation_lod, jnp.clip(diameter, low, high), interpolation
            )
            return jnp.where((diameter >= low) & (diameter <= high), result, jnp.nan)
        point_source = self._intensity_record.evaluate(
            "core_mean_intensity", separation_lod
        )
        if not self._core_intensity_varies_with_diameter:
            return point_source

        # lax.cond rather than a Python if, so that a traced stellar diameter
        # works, and rather than jnp.where, so that the point-source call does
        # not pay for a profile. The diameter is clamped into the tabulated
        # range, where the stellar map interpolant is defined.
        low, high = self._diam_knot_range
        clamped = jnp.clip(stellar_diam_lod, low, high)
        result = jax.lax.cond(
            stellar_diam_lod == 0.0,
            lambda: jnp.asarray(point_source),
            lambda: self._core_mean_intensity_at(
                separation_lod, clamped, interpolation
            ),
        )
        valid = (stellar_diam_lod == 0.0) | (
            (stellar_diam_lod >= low) & (stellar_diam_lod <= high)
        )
        return jnp.where(valid, result, jnp.nan)

    def _core_mean_intensity_at(
        self, separation_lod, stellar_diam_lod, interpolation="keys"
    ) -> Array:
        """Azimuthal mean of the interpolated stellar map at one diameter."""
        stellar_map = self.stellar_intens(stellar_diam_lod).data
        radii = self._profile_radii_pix
        profile = azimuthal_mean_profile(
            stellar_map, self._stellar_axis, radii, interpolation=interpolation
        )
        nodes = self._intensity_record.separation_lod
        coefficients = interpax.PchipInterpolator(
            nodes, profile, extrapolate=False, check=False
        ).c
        tails = jnp.stack(
            (
                profile[0],
                profile[-1],
                (profile[1] - profile[0]) / (nodes[1] - nodes[0]),
                (profile[-1] - profile[-2]) / (nodes[-1] - nodes[-2]),
            )
        )
        # A traced stellar diameter changes the sampled profile. Replace its
        # dynamic leaves while keeping the ancillary record's axis and metadata.
        record = eqx.tree_at(
            lambda curves: (
                curves._extra["core_mean_intensity"],
                curves._coefficients["core_mean_intensity"],
                curves._tails["core_mean_intensity"],
            ),
            self._intensity_record,
            (profile, coefficients, tails),
        )
        return record.evaluate("core_mean_intensity", jnp.asarray(separation_lod))
