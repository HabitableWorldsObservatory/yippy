"""Immutable YIP tables implementing the shared coronagraph contract."""

import math
import warnings
from copy import copy
from dataclasses import fields, replace
from math import pi

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import is_traced
from hwoutils.memory import DEFAULT_MAX_BYTES, admit
from hwoutils.precision import like
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from optixstuff.coronagraph import (
    AbstractSceneCoronagraph,
    AbstractTableCoronagraph,
    ProductGrids,
    SceneRender,
    SourcePSFs,
    on_grid,
)
from optixstuff.grids import Image, Measure, PixelGrid, coverage
from optixstuff.performance import RING_RULE, PerformanceCurves, SampledProfile
from optixstuff.primary import SimplePrimary
from optixstuff.spectral import Band, same_wavelength

from ._jax_funcs import blend_pad
from ._precision import float_dtype
from ._stellar_intens import bounded_stellar_pchip, stellar_knot_index
from ._yip import _YIPData
from .header import HeaderData
from .interpolation import SincOptions
from .logger import logger
from .performance import resolve_leakage_rule
from .sampling import STELLAR_RECONSTRUCTION, fraction_image, stellar_on_grid
from .scene import (
    DENSE_BATCH,
    AbstractSceneOperator,
    BlendSceneOperator,
    MaskedSceneOperator,
    _DenseSceneOperator,
)
from .sinc import SceneRenderer
from .sinc.geometry import array_namespace
from .sinc.reader import PrefilteredReader

_WARNED_2D: set = set()
# Flux magnitude in the scene's units below the normal binary32 range is dark.
_SCENE_DARK_FLUX = np.finfo(np.float32).tiny


@jax.custom_jvp
def _scene_flux(flux):
    """Flush dark cells while retaining the linear operator's flux derivative."""
    dark = (flux > -_SCENE_DARK_FLUX) & (flux < _SCENE_DARK_FLUX)
    return jnp.where(dark, 0, flux)


_scene_flux.defjvps(lambda tangent, result, flux: tangent)


def _jax_record(record):
    """Preserve fitted coefficients while adopting the active floating dtype."""
    return jax.tree.map(
        lambda value: (
            jnp.asarray(value, dtype=float_dtype())
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating)
            else value
        ),
        record,
    )


def _stellar_exp(log_values):
    """Exponentiate with a zero tangent for the overflowing entries."""
    dtype = log_values.dtype
    limit = np.log(float(np.finfo(dtype).max))
    max_log = dtype.type(limit)
    if float(max_log) > limit:
        max_log = np.nextafter(max_log, dtype.type(-np.inf))
    overflow = log_values > max_log
    safe_log = jnp.where(overflow, 0, log_values)
    return jnp.where(overflow, like(jnp.inf, log_values), jnp.exp(safe_log))


def _declared_band(offaxis_header, stellar_header):
    """Read the off-axis band after comparing both intensity headers in nm."""

    def edges_nm(header):
        return tuple(
            None if edge is None else float(edge.to_value("nm"))
            for edge in (header.minlam, header.maxlam)
        )

    offaxis, stellar = edges_nm(offaxis_header), edges_nm(stellar_header)
    if offaxis == stellar == (None, None):
        return None
    if (
        None in offaxis
        or None in stellar
        or not all(same_wavelength(a, b) for a, b in zip(offaxis, stellar, strict=True))
    ):
        raise ValueError(
            "Both intensity headers must declare matching (MINLAM, MAXLAM) in nm: "
            f"offax_psf.fits has {offaxis}, stellar_intens.fits has {stellar}"
        )
    return None if same_wavelength(*offaxis) else Band(*offaxis)


class _StellarLogSpline(interpax.CubicSpline):
    """Expose the fitted knot and coefficient arrays as model-owned leaves."""


class _StellarLinearSpline(interpax.PchipInterpolator):
    """Expose the bounded PCHIP arrays as model-owned leaves."""


class Coronagraph(AbstractTableCoronagraph):
    """A frozen, JAX-compatible coronagraph loaded from a Yield Input Package.

    Coordinates and stellar diameters are floats in lambda/D. The
    characterization record for each aperture rule is prepared at construction,
    before JIT queries.
    The default photometric rule is a circle of radius 0.7 lambda/D.

    In offax_psf.fits, MINLAM below MAXLAM declares a flat average over that
    interval on a lambda/D grid at LAMBDA. Place this grid once per observing
    band at its centre (Stark and Krist 2019, "Standardized Coronagraph
    Parameters for Input into Yield Calculations"). Producers computing one
    wavelength must write equal limits. Missing limits also indicate a
    single wavelength. The limits in stellar_intens.fits must agree with
    offax_psf.fits, while sky_trans.fits may differ. N_LAM is an optional
    consistency check outside the specification.

    Attributes:
        grid: Native pixel grid of the off-axis PSFs, in lambda/D.
        stellar_grid: Native pixel grid of the stellar intensity maps, in lambda/D.
        sky_trans: Sky transmission image on its native grid.
        header: Parsed YIP header metadata.
        interpolation_info: Routing and provenance of the off-axis interpolation.
        bandpass: Spectral band recorded by the YIP in nm, or None when it holds a
            single wavelength.
        band_averaged: True for a flat band average placed at the observing band
            centre. A single-wavelength product is placed at each node wavelength.
        uniform_sky_profile: Radial profile of the uniform-sky response.
    """

    grid: PixelGrid
    stellar_grid: PixelGrid
    sky_trans: Image
    header: HeaderData = eqx.field(static=True)
    interpolation_info: object = eqx.field(static=True)
    bandpass: Band | None
    band_averaged: bool = eqx.field(static=True)
    _offaxis_backend: object
    _stellar_ln_interp: _StellarLogSpline
    _stellar_linear_interp: _StellarLinearSpline | None
    _stellar_samples: jax.Array
    _records: tuple[PerformanceCurves, ...]
    uniform_sky_profile: SampledProfile
    _host: _YIPData = eqx.field(static=True)

    @property
    def package(self):
        """Native package from which this model was prepared."""
        return self._host.package

    @property
    def leakage_rule(self) -> str:
        """The leakage rule of the prepared records (the default for ``curves``)."""
        return self._records[0].leakage_rule

    def __init__(
        self,
        package,
        *,
        x_symmetric=True,
        y_symmetric=True,
        use_quarter_psf_datacube=True,
        downsample_shape=None,
        rules=(DEFAULT_APERTURE_RULE,),
        leakage_rule=None,
        interpolation="blend",
        sinc_options: SincOptions | None = None,
    ):
        """Load host products and freeze their tables and prepared aperture rules.

        Args:
            package: Open YIP package supplying the native products.
            x_symmetric: Whether the response is mirror symmetric about the x axis.
            y_symmetric: Whether the response is mirror symmetric about the y axis.
            use_quarter_psf_datacube: Whether a generated PSF datacube stores only the
                positive quarter of source offsets.
            downsample_shape: Optional (ny, nx) target shape for the stored PSFs; None
                keeps the native shape.
            rules: Aperture rules to prepare a record for; the first is the
                coronagraph's own rule. Duplicates are dropped and at least one is
                required.
            leakage_rule: Name of the leakage rule, or None for the product default.
            interpolation: Interpolation mode requested for off-axis reads, for example
                ``"blend"`` or ``"auto"``.
            sinc_options: Optional sinc construction controls; None uses defaults.
        """
        rules = tuple(dict.fromkeys(rules))
        if not rules:
            raise ValueError("rules must contain at least one aperture rule")
        package._check_source()
        host = _YIPData(
            package,
            x_symmetric,
            y_symmetric,
            use_quarter_psf_datacube,
            downsample_shape,
            rule=rules[0],
            leakage_rule=leakage_rule,
            interpolation=interpolation,
            sinc_options=sinc_options,
        )
        package._check_source()
        self._host = host
        self.header = host.header
        self.grid = host.grid
        self.stellar_grid = host.stellar_intens.grid
        self.interpolation_info = host.interpolation_info
        spectral_header = host.offaxis_header
        self.bandpass = _declared_band(spectral_header, host.header)
        self.band_averaged = self.bandpass is not None
        if self.band_averaged and spectral_header.n_lam == 1:
            warnings.warn(
                "N_LAM = 1 contradicts the declared band in offax_psf.fits. "
                "The band declaration is used.",
                UserWarning,
                stacklevel=2,
            )
        self._offaxis_backend = host.offax
        stellar = host.stellar_intens
        diams_lod = jnp.asarray(stellar.diams.value, dtype=float_dtype())
        samples = jnp.asarray(stellar.spline_samples, dtype=float_dtype())
        self._stellar_samples = samples
        self._stellar_linear_interp = None
        log_samples = samples
        if stellar.zero_pixels.any():
            self._stellar_linear_interp = _StellarLinearSpline(diams_lod, samples)
            log_samples = jnp.where(jnp.asarray(stellar.zero_pixels), 1, samples)
        self._stellar_ln_interp = _StellarLogSpline(diams_lod, jnp.log(log_samples))
        records = tuple(host._prepare_record(rule) for rule in rules)
        self._records = tuple(_jax_record(record) for record in records)
        self.uniform_sky_profile = _jax_record(host.uniform_sky_profile)
        self.sky_trans = host.sky_trans

    @property
    def primary(self):
        """Primary geometry recorded by the YIP; missing required metadata raises."""
        for name in ("diameter", "diameter_inscribed", "obscured"):
            if getattr(self.header, name) is None:
                raise ValueError(f"YIP metadata is missing {name}")
        diameter_m = self.header.diameter.to_value("m")
        return SimplePrimary(
            circumscribed_diameter_m=diameter_m,
            inscribed_diameter_m=self.header.diameter_inscribed.to_value("m"),
            collecting_area_m2=pi * diameter_m**2 / 4 * (1 - self.header.obscured),
        )

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule=None):
        """Select one prepared record; requesting an unprepared rule raises.

        Every field is a function of separation alone. For a 2D product (a
        stellar map that is not rotationally symmetric) the star leakage is
        therefore an average over position angle, or a single cut, and is not
        the leakage at a particular planet position angle; reading it as one
        logs a warning once per product. ``leakage_rule`` defaults to the
        coronagraph's own rule: ``aperture_ring_mean`` for a 2D product (the
        exact-coverage aperture of ``rule`` averaged over position angle),
        ``aperture_mean`` for a 1D product (the aperture at +x, equal to every
        angle there). The leakage at a planet's own angle is the image or
        aperture-record route of ``coronagraphoto``.

        Args:
            rule: Aperture rule of the record to select; it must have been prepared.
            leakage_rule: Leakage rule of the record; None uses the coronagraph's own.

        Returns:
            The prepared performance record for the rule and leakage rule, with every
            field a function of separation.
        """
        leakage_rule = self.leakage_rule if leakage_rule is None else leakage_rule
        record = self._find_record(rule, leakage_rule)
        if record is None:
            raise ValueError(
                "No prepared record; supply rules and leakage_rule at construction"
            )
        self._warn_if_azimuthal()
        return record

    def _warn_if_azimuthal(self):
        """Log once per product that a 2D product's curves are not a planet value."""
        key = (self.package.source.digest, self.leakage_rule)
        if self._offaxis_backend.type == "1d" or key in _WARNED_2D:
            return
        _WARNED_2D.add(key)
        what = {
            RING_RULE: "averaged over position angle",
            "aperture_mean": "read on the +x axis only",
        }.get(self.leakage_rule, "a point-sample ring mean")
        logger.warning(
            f"{self.package.source.path.name} is a 2D product: its curves give the "
            f"star leakage by separation only ({self.leakage_rule}, {what}). "
            "The leakage at a planet's own position angle comes from the image or "
            "aperture-record route (coronagraphoto), not from the curves."
        )

    def uniform_sky_response(self, separation_lod):
        """Annular mean of the uniform-sky map at separations in lambda/D.

        Args:
            separation_lod: Separations in lambda/D of this coronagraph's frame; any
                array shape.

        Returns:
            Annular mean of the uniform-sky map at each separation, with the shape of
            ``separation_lod``.
        """
        profile = self.uniform_sky_profile
        frame = self.frame
        if frame != profile.frame:
            # A single angular separation spans different native lambda/D counts.
            ratio = frame.lod_to_arcsec(
                frame.reference_wavelength_nm
            ) / profile.frame.lod_to_arcsec(profile.frame.reference_wavelength_nm)
            dtype = jnp.result_type(
                profile.values, profile.separation_lod, *jax.tree.leaves(separation_lod)
            )
            separation_lod = jnp.asarray(separation_lod, dtype=dtype)
            separation_lod = separation_lod * like(ratio, separation_lod)
        return profile.evaluate(separation_lod)

    def uniform_sky_map(self, wavelength_nm, *, grid: PixelGrid) -> Image:
        """Return dimensionless transmission at the requested pixel centers.

        Args:
            wavelength_nm: Wavelength in nm. The fixed table ignores it.
            grid: Grid on which to sample the transmission.

        Returns:
            The intensive transmission image on ``grid``.
        """
        sky = self.sky_trans
        geometry = (grid.basis, getattr(grid.orientation, "pa_y_deg", None))
        if grid is sky.grid or (
            not any(map(is_traced, jax.tree.leaves(geometry))) and grid == sky.grid
        ):
            return sky
        return super().uniform_sky_map(wavelength_nm, grid=grid)

    def off_axis_psf(self, wavelength_nm, separation_lod, grid: PixelGrid) -> Image:
        """Return cell-integrated source fractions for a source along +x.

        Args:
            wavelength_nm: Wavelength in nm. The fixed table ignores it.
            separation_lod: Source separation along +x in lambda/D.
            grid: Grid of the returned image.

        Returns:
            Flux fraction per cell, or zero for a source outside the domain.
        """
        xp = array_namespace(separation_lod)
        offsets = self._offaxis_backend.x_offsets
        dtype = jnp.result_type(offsets, *jax.tree.leaves(separation_lod))
        result = self.source_psfs(
            wavelength_nm, separation_lod, xp.zeros(1, dtype=dtype), grid=grid
        )
        data = result.image.data[0]
        return replace(
            result.image,
            data=jnp.where(result.valid[0], data, like(0, data)),
            leading_axes=(),
        )

    def _find_record(self, rule, leakage_rule):
        return next(
            (
                record
                for record in self._records
                if record.rule == rule and record.leakage_rule == leakage_rule
            ),
            None,
        )

    def at_wavelength(self, wavelength_nm) -> ProductGrids:
        """Return the three recorded native grids for this fixed YIP product.

        Args:
            wavelength_nm: Wavelength in nm. It is not used: the product is fixed and
                its grids do not depend on wavelength.

        Returns:
            The native image, stellar and sky-transmission grids of the product.
        """
        return ProductGrids(self.grid, self.stellar_grid, self.sky_trans.grid)

    def _native_stellar_map(self, stellar_diameter_lod):
        if hasattr(stellar_diameter_lod, "unit"):
            raise TypeError("stellar_diameter_lod must be floats in lambda/D")
        axis = self._stellar_ln_interp.x
        diameter = jnp.asarray(stellar_diameter_lod, dtype=axis.dtype)
        knot = stellar_knot_index(axis, diameter, jnp)
        at_knot = (diameter == axis[knot])[..., None, None]
        samples = self._stellar_samples
        data = _stellar_exp(self._stellar_ln_interp(diameter))
        stellar = self._host.stellar_intens
        if self._stellar_linear_interp is not None:
            zero_pixels = jnp.asarray(stellar.zero_pixels)
            linear = bounded_stellar_pchip(
                self._stellar_linear_interp, diameter, samples, knot, jnp
            )
            clamped = (diameter <= axis[0]) | (diameter >= axis[-1])
            linear = jnp.where(clamped[..., None, None], samples[knot], linear)
            data = jnp.where(zero_pixels, linear, data)
        # Preserve the spline's diameter derivative while returning the sample.
        knot_value = jnp.where(at_knot & jnp.isfinite(data), data, 0)
        sample = samples[knot]
        correction = knot_value - jax.lax.stop_gradient(knot_value)
        exact = jnp.where(sample == 0, sample, sample + correction)
        data = jnp.where(at_knot, exact, data)
        return fraction_image(
            data,
            self.stellar_grid,
            tuple(f"diameter_{i}" for i in range(data.ndim - 2)),
            stellar.measure,
            reconstruction=STELLAR_RECONSTRUCTION,
        )

    def stellar_map(self, wavelength_nm, stellar_diameter_lod, *, grid=None):
        """Per-pixel stellar fraction on grid, using the stellar product's frame.

        The native map holds point samples of the intensity. On a grid each
        pixel is the integral of their cubic spline over the pixel footprint,
        floored at a tenth of the linear footprint integral for nonnegative
        samples.

        Args:
            wavelength_nm: Wavelength in nm. This product returns its fixed table.
            stellar_diameter_lod: Angular diameter of the star in lambda/D.
            grid: Grid the Image is returned on; None keeps the native samples.

        Returns:
            Stellar flux fraction per pixel, or the native density when ``grid``
            is None.
        """
        image = self._native_stellar_map(stellar_diameter_lod)
        return image if grid is None else stellar_on_grid(image, grid)

    def _native_source_psfs(self, x_lod, y_lod):
        if hasattr(x_lod, "unit") or hasattr(y_lod, "unit"):
            raise TypeError("source coordinates must be floats in lambda/D")
        backend = self._offaxis_backend
        xp_x, xp_y = array_namespace(x_lod), array_namespace(y_lod)
        x = xp_x.atleast_1d(xp_x.asarray(x_lod))
        y = xp_y.atleast_1d(xp_y.asarray(y_lod))
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Coordinates must be equal-length vectors")
        if hasattr(backend, "create_psfs_with_info"):
            # The delivery keeps its own admission as the source domain; the
            # image of a source it refuses is zero here, not the NaN of its read.
            read = backend.create_psfs_with_info(x, y)
            accepted = jnp.asarray(read.accepted, bool)
            intensity = jnp.where(
                accepted[:, None, None], read.intensity, like(0, read.intensity)
            )
            image = fraction_image(intensity, self.grid, ("source",), backend.measure)
            return SourcePSFs(image, accepted)
        # Admission uses the supplied coordinates before entering the working
        # precision of the numerical read.
        inside = backend.in_source_domain(x, y)
        dtype = backend._coordinate_dtype(x_lod, y_lod)
        x, y = jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)
        x, y = jnp.where(inside, x, 0), jnp.where(inside, y, 0)
        image = backend.create_psfs(x, y)
        valid = inside & jnp.all(jnp.isfinite(image.data), axis=(-1, -2))
        return SourcePSFs(
            replace(
                image,
                data=jnp.where(valid[:, None, None], image.data, like(0, image.data)),
            ),
            valid,
        )

    def with_rules(self, rules, *, leakage_rule=None):
        """Return this coronagraph with a different set of prepared aperture rules.

        Every table and field except the prepared records is kept, so a
        modified module keeps its modifications. Preparation runs on the host,
        outside JAX transformations. ``leakage_rule`` defaults to the current one.

        Args:
            rules: Aperture rules to prepare; the first is the coronagraph's own rule.
                Duplicates are dropped and at least one is required.
            leakage_rule: Leakage rule name; None keeps the current one.

        Returns:
            A copy of this coronagraph whose prepared records are those of ``rules``.
        """
        rules = tuple(dict.fromkeys(rules))
        if not rules:
            raise ValueError("rules must contain at least one aperture rule")
        if leakage_rule is None:
            leakage_rule = self.leakage_rule
        leakage_rule = resolve_leakage_rule(leakage_rule, self._offaxis_backend.type)
        host = self._host_for_rule(rules[0], leakage_rule)
        records = tuple(_jax_record(host._prepare_record(rule)) for rule in rules)
        return eqx.tree_at(lambda c: c._records, self, records)

    def _host_for_rule(self, rule, leakage_rule):
        """Use the loaded host tables with independent per-request state."""
        self.package._check_source()
        host = copy(self._host)
        host.curves_by_rule = {}
        host.rule, host.leakage_rule = rule, leakage_rule
        host.offax = self._offaxis_backend
        return host

    def _dense_delivery(self):
        """Whether scene reads mix sinc and blend per source (one PSF per source)."""
        return self._host._sinc_delivery is not None

    def _prefiltered_scene(self):
        """Whether the scene renders from a prefiltered coefficient table."""
        backend = self._offaxis_backend
        reader = getattr(backend, "reader", None)
        return isinstance(reader, PrefilteredReader) and backend.blend_nodes is None

    @property
    def scene_grid(self):
        """The grid of source cells a scene operator reads, in lambda/D.

        Cells of the stored PSF grid, extended symmetrically to the hull of the
        tabulated off-axis nodes when the nodes reach beyond the stamp, so a
        source between the stamp edge and the outermost node is rendered, not
        dropped. ``render_scene`` takes the scene on this grid; its image is on
        ``grid``. Routes that render from a per-source cube or a prefiltered
        table read the stored PSF grid.
        """
        backend = self._offaxis_backend
        grid = self.grid
        if (
            self._prefiltered_scene()
            or self._dense_delivery()
            or not grid.is_geometric
            or not hasattr(backend, "source_extent_lod")
        ):
            return grid
        if backend.type == "1d":
            reach = (backend.source_radius_lod,) * 2
        else:
            reach = backend.source_extent_lod
        ny, nx = grid.shape
        step = grid.step
        # Cell centers out to the cell that holds the outermost node.
        pad_x, pad_y = (
            max(0, math.ceil(r / step - 0.5 - (n - 1) / 2 - 1e-9))
            for r, n in zip(reach, (nx, ny), strict=True)
        )
        if not (pad_x or pad_y):
            return grid
        return replace(
            grid,
            shape=(ny + 2 * pad_y, nx + 2 * pad_x),
            center_x=grid.center_x + pad_x,
            center_y=grid.center_y + pad_y,
        )

    def scene_operator_bytes(self, mask=None):
        """Estimate the bytes of `with_scene_operator` without allocating them.

        Returns a dict with the keys ``resident`` (the operator), ``scratch``
        (the transient peak of building or rendering) and ``total``. The
        estimate comes from shapes and node offsets only. A prefiltered
        delivery reports the upper bound of its coefficient table; its own
        memory budget governs the build.

        Args:
            mask: Optional boolean mask on ``scene_grid`` restricting the sources read,
                as for ``with_scene_operator``.

        Returns:
            Dict with the bytes ``resident``, ``scratch`` and ``total``.
        """
        backend = self._offaxis_backend
        itemsize = jnp.dtype(float_dtype()).itemsize
        if self._prefiltered_scene():
            coefficients = backend.reader.coefficients
            resident = (
                int(np.prod(coefficients.shape[1:])) * coefficients.dtype.itemsize
            )
            scratch = 5 * int(np.prod(coefficients.shape[-2:])) * 8
        elif self._dense_delivery():
            host = self._host_for_scene()
            sy, sx = host._datacube_offsets()[0].shape
            npix = int(self._host.npixels)
            resident = sy * sx * npix * npix * itemsize
            padded = npix + 2 * blend_pad(npix)
            # Per source: the four shifted neighbor spectra and their inverse
            # along y, taken per neighbor.
            batch = DENSE_BATCH * 2 * 4 * padded * (padded // 2 + 1) * 2 * itemsize
            # The cube is gathered on the host, then copied to the device.
            scratch = resident + batch
        else:
            return BlendSceneOperator.size_bytes(backend, self.scene_grid, mask)
        return {"resident": resident, "scratch": scratch, "total": resident + scratch}

    def _host_for_scene(self):
        """The host tables with the symmetric quarter storage the dense route needs."""
        record = self._records[0]
        host = self._host_for_rule(record.rule, self.leakage_rule)
        backend = self._offaxis_backend
        # Signed tables and permission regions can break mirror symmetry.
        host.use_quarter_psf_datacube = (
            host.use_quarter_psf_datacube
            and backend.type in ("1d", "2dq")
            and host.x_symmetric
            and host.y_symmetric
            and self.grid.is_geometric
        )
        return host

    def with_scene_operator(self, *, max_bytes=DEFAULT_MAX_BYTES, mask=None):
        """Prepare a scene renderer for this model on its ``scene_grid``.

        Prefiltered products render from their coefficient table. A blend model
        renders through ``BlendSceneOperator``, one FFT product per stored PSF. A
        delivery that mixes sinc and blend reads builds one PSF per source.

        Args:
            max_bytes: Refuse, before allocating, an operator whose estimated
                total (see ``scene_operator_bytes``) exceeds this many bytes, on
                every route. ``None`` disables the check.
            mask: Optional boolean mask on ``scene_grid``: the operator reads
                only the sources inside it and reports the rest uncovered.
                Combine masked operators with ``SceneCoronagraph.with_operators``.

        Returns:
            A ``SceneCoronagraph`` holding the prepared scene operator.

        Raises:
            hwoutils.memory.MemoryLimit: The estimate exceeds ``max_bytes``.
        """
        if max_bytes is not None:
            estimate = self.scene_operator_bytes(mask)
            logger.info(
                f"Scene operator estimate: {estimate['resident'] / 1e9:.2f} GB "
                f"resident, {estimate['scratch'] / 1e9:.2f} GB scratch."
            )
            admit(estimate["total"], max_bytes, "scene operator")
        backend = self._offaxis_backend
        if self._prefiltered_scene():
            x, y = self.grid.coordinates()
            operator = SceneRenderer.from_raster(
                backend.reader, 0, x[0], y[:, 0], scale=backend.conversion
            )
        elif self._dense_delivery():
            host = self._host_for_scene()
            host._create_delivery_datacube(DENSE_BATCH)
            cube = jnp.asarray(host.psf_datacube)
            valid = host.psf_datacube_info["accepted"]
            operator = _DenseSceneOperator(cube, valid)
        else:
            return SceneCoronagraph(
                self,
                (BlendSceneOperator.from_backend(backend, self.scene_grid, mask=mask),),
            )
        if mask is not None:
            operator = MaskedSceneOperator(operator, mask)
        return SceneCoronagraph(self, (operator,))

    def __repr__(self):
        """Summarize the native grid and prepared rules.

        Returns:
            Text naming the YIP folder, the native grid shape and the number of prepared
            rules.
        """
        return (
            f"Coronagraph({self.package.source.path.name!r}, grid={self.grid.shape}, "
            f"rules={len(self._records)})"
        )


class SceneCoronagraph(Coronagraph, AbstractSceneCoronagraph):
    """YIP imaging model with prepared native scene operators.

    The rendered image is the sum of the operators' images, so operators should
    cover disjoint sources. A valid scene has finite lit cells wholly inside
    ``scene_grid``, with the resampled flux covered by the operators.

    Attributes:
        operators: Scene operators whose images are summed.
    """

    operators: tuple[AbstractSceneOperator, ...]
    _source_grid: PixelGrid = eqx.field(static=True)

    def __init__(self, model, operators, scene_grid=None):
        """Reuse the prepared imaging tables and bind the scene operators.

        Args:
            model: The imaging model whose tables the operators sit beside.
            operators: Scene operators built for ``scene_grid``.
            scene_grid: The source grid the operators read; the model's own
                ``scene_grid`` by default. Pass the stored PSF grid for an
                operator built on it.
        """
        for field in fields(Coronagraph):
            setattr(self, field.name, getattr(model, field.name))
        self.operators = tuple(operators)
        self._source_grid = model.scene_grid if scene_grid is None else scene_grid

    @property
    def scene_grid(self):
        """The grid of source cells the operators read."""
        return self._source_grid

    def with_operators(self, *operators):
        """Return this model with more scene operators added to the sum.

        Args:
            *operators: Scene operators to add to the sum.

        Returns:
            A new ``SceneCoronagraph`` with the added operators.
        """
        return SceneCoronagraph(self, (*self.operators, *operators))

    def render_scene(self, scene, wavelength_nm, *, grid):
        """Sum native flux through the operators and sample the detector.

        ``scene`` is a cell-integral image, on ``scene_grid`` or resampled onto
        it. The result is finite. Its ``valid`` is False when a lit input cell
        extends beyond ``scene_grid`` or when resampled flux is nonfinite or
        falls on a cell covered by no operator.
        Flux magnitudes below the smallest normal float32 count as zero.
        The flux derivative on covered cells is the linear operator response,
        including at zero flux.

        Args:
            scene: Cell-integral two-dimensional image of the source flux, on
                ``scene_grid`` or resampled onto it.
            wavelength_nm: Wavelength in nm. It is not used by this fixed product.
            grid: Detector grid onto which the summed image is sampled.

        Returns:
            The finite image on ``grid`` and a flag indicating whether all lit
            cells are supported.
        """
        if scene.measure != Measure.integral() or scene.data.ndim != 2:
            raise ValueError("scene must be a two-dimensional cell-integral Image")
        source = self.scene_grid
        input_supported = True
        flux = _scene_flux(scene.data)
        if scene.grid != source:
            source_footprint, input_grid = source, scene.grid
            if source.unit != input_grid.unit or (
                source.unit == "lod" and source.frame != input_grid.frame
            ):
                if source.unit == "lod":
                    source_footprint = source.to_arcsec(
                        source.frame.reference_wavelength_nm
                    )
                if input_grid.unit == "lod":
                    input_grid = input_grid.to_arcsec(
                        input_grid.frame.reference_wavelength_nm
                    )
            inside = coverage(source_footprint, input_grid)
            input_supported = ~jnp.any((flux != 0) & ~inside)
            flux = on_grid(replace(scene, data=flux), source).data
            flux = _scene_flux(flux)
        parts = [operator.render(flux, source) for operator in self.operators]
        data = sum(part[0] for part in parts)
        covered = jnp.any(jnp.stack([part[1] for part in parts]), axis=0)
        valid = input_supported & jnp.all((flux == 0) | (jnp.isfinite(flux) & covered))
        image = fraction_image(
            data, self.grid, (), self._offaxis_backend.measure, quantity=scene.quantity
        )
        return SceneRender(on_grid(image, grid), valid)
