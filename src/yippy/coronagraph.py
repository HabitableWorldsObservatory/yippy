"""Immutable YIP tables implementing the shared coronagraph contract."""

from copy import copy
from dataclasses import fields, replace
from math import pi

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.memory import DEFAULT_MAX_BYTES, admit
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from optixstuff.coronagraph import (
    AbstractSceneCoronagraph,
    AbstractTableCoronagraph,
    ProductGrids,
    SceneRender,
    SourcePSFs,
    on_grid,
)
from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.performance import RING_RULE, PerformanceCurves, SampledProfile
from optixstuff.primary import SimplePrimary
from optixstuff.spectral import Band

from ._jax_funcs import blend_pad
from ._precision import float_dtype
from ._yip import _YIPData
from .header import HeaderData
from .interpolation import SincOptions
from .logger import logger
from .performance import resolve_leakage_rule
from .sampling import STELLAR_RECONSTRUCTION, fraction_image
from .scene import (
    DENSE_BATCH,
    AbstractSceneOperator,
    BlendSceneOperator,
    MaskedSceneOperator,
    _DenseSceneOperator,
)
from .sinc import SceneRenderer
from .sinc.reader import PrefilteredReader

_WARNED_2D: set = set()


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


class Coronagraph(AbstractTableCoronagraph):
    """A frozen, JAX-compatible coronagraph loaded from a Yield Input Package.

    Coordinates and stellar diameters are floats in lambda/D. The
    characterization record for each aperture rule is prepared at construction,
    before JIT queries.
    The default photometric rule is a circle of radius 0.7 lambda/D.
    """

    grid: PixelGrid
    stellar_grid: PixelGrid
    sky_trans: Image
    header: HeaderData = eqx.field(static=True)
    interpolation_info: object = eqx.field(static=True)
    bandpass: Band | None
    _offaxis_backend: object
    _stellar_ln_interp: interpax.CubicSpline
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
        """Load host products and freeze their tables and prepared aperture rules."""
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
        self.bandpass = None
        if spectral_header.minlam is not None and spectral_header.maxlam is not None:
            low_nm, high_nm = (
                spectral_header.minlam.to_value("nm"),
                spectral_header.maxlam.to_value("nm"),
            )
            # A single recorded wavelength has no finite-width spectral band.
            if low_nm != high_nm:
                self.bandpass = Band(low_nm, high_nm)
        self._offaxis_backend = host.offax
        self._stellar_ln_interp = interpax.CubicSpline(
            jnp.asarray(host.stellar_intens.ln_interp.x, dtype=float_dtype()),
            jnp.log(jnp.asarray(host.stellar_intens.psfs, dtype=float_dtype())),
        )
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
        """Annular mean of the uniform-sky map at separations in lambda/D."""
        profile = self.uniform_sky_profile
        frame = self.frame
        if frame != profile.frame:
            # A single angular separation spans different native lambda/D counts.
            ratio = frame.lod_to_arcsec(
                frame.reference_wavelength_nm
            ) / profile.frame.lod_to_arcsec(profile.frame.reference_wavelength_nm)
            separation_lod = jnp.asarray(separation_lod) * ratio
        return profile.evaluate(separation_lod)

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
        """Return the three recorded native grids for this fixed YIP product."""
        return ProductGrids(self.grid, self.stellar_grid, self.sky_trans.grid)

    def _native_stellar_map(self, stellar_diameter_lod):
        if hasattr(stellar_diameter_lod, "unit"):
            raise TypeError("stellar_diameter_lod must be floats in lambda/D")
        data = jnp.exp(self._stellar_ln_interp(stellar_diameter_lod))
        return fraction_image(
            data,
            self.stellar_grid,
            tuple(f"diameter_{i}" for i in range(data.ndim - 2)),
            self._host.stellar_intens.measure,
            reconstruction=STELLAR_RECONSTRUCTION,
        )

    def _native_source_psfs(self, x_lod, y_lod):
        if hasattr(x_lod, "unit") or hasattr(y_lod, "unit"):
            raise TypeError("source coordinates must be floats in lambda/D")
        x, y = jnp.atleast_1d(jnp.asarray(x_lod)), jnp.atleast_1d(jnp.asarray(y_lod))
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Coordinates must be equal-length vectors")
        backend = self._offaxis_backend
        if hasattr(backend, "create_psfs_with_info"):
            # The delivery keeps its own admission as the source domain; the
            # image of a source it refuses is zero here, not the NaN of its read.
            read = backend.create_psfs_with_info(x, y)
            accepted = jnp.asarray(read.accepted, bool)
            intensity = jnp.where(accepted[:, None, None], read.intensity, 0.0)
            image = fraction_image(intensity, self.grid, ("source",), backend.measure)
            return SourcePSFs(image, accepted)
        finite = jnp.isfinite(x) & jnp.isfinite(y)
        x, y = jnp.where(finite, x, 0.0), jnp.where(finite, y, 0.0)
        # The read clamps to the outermost ring, so it is finite and
        # differentiable for any position; the source domain is the node hull.
        inside = backend.in_source_domain(x, y)
        image = backend.create_psfs(x, y)
        valid = finite & inside & jnp.all(jnp.isfinite(image.data), axis=(-1, -2))
        return SourcePSFs(
            replace(image, data=jnp.where(valid[:, None, None], image.data, 0.0)),
            valid,
        )

    def with_rules(self, rules, *, leakage_rule=None):
        """Return this coronagraph with a different set of prepared aperture rules.

        Every table and field except the prepared records is kept, so a
        modified module keeps its modifications. Preparation runs on the host,
        outside JAX transformations. ``leakage_rule`` defaults to the current one.
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

        ``render_scene`` takes the scene on this grid; its image is on ``grid``.
        """
        return self.grid

    def scene_operator_bytes(self, mask=None):
        """Estimate the bytes of `with_scene_operator` without allocating them.

        Returns a dict with the keys ``resident`` (the operator), ``scratch``
        (the transient peak of building or rendering) and ``total``. The
        estimate comes from shapes and node offsets only. A prefiltered
        delivery reports the upper bound of its coefficient table; its own
        memory budget governs the build.
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
            batch = DENSE_BATCH * 4 * padded * (padded // 2 + 1) * 2 * itemsize
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
        """Summarize the native grid and prepared rules."""
        return (
            f"Coronagraph({self.package.source.path.name!r}, grid={self.grid.shape}, "
            f"rules={len(self._records)})"
        )


class SceneCoronagraph(Coronagraph, AbstractSceneCoronagraph):
    """YIP imaging model with prepared native scene operators.

    The rendered image is the sum of the operators' images, so operators should
    cover disjoint sources. It is valid when every lit source cell is covered by
    some operator; a lit cell no operator covers contributes zero.
    """

    operators: tuple[AbstractSceneOperator, ...]

    def __init__(self, model, operators):
        """Reuse the prepared imaging tables and bind the scene operators."""
        for field in fields(Coronagraph):
            setattr(self, field.name, getattr(model, field.name))
        self.operators = tuple(operators)

    def with_operators(self, *operators):
        """Return this model with more scene operators added to the sum."""
        return SceneCoronagraph(self, (*self.operators, *operators))

    def render_scene(self, scene, wavelength_nm, *, grid):
        """Sum native flux through the operators and sample the detector.

        ``scene`` is a cell-integral image, on ``scene_grid`` or resampled onto
        it. The result is finite; its ``valid`` is False when a lit cell is
        nonfinite or covered by no operator.
        """
        if scene.measure != Measure.integral() or scene.data.ndim != 2:
            raise ValueError("scene must be a two-dimensional cell-integral Image")
        source = self.scene_grid
        flux = scene.data if scene.grid == source else on_grid(scene, source).data
        parts = [operator.render(flux, source) for operator in self.operators]
        data = sum(part[0] for part in parts)
        covered = jnp.any(jnp.stack([part[1] for part in parts]), axis=0)
        valid = jnp.all((flux == 0) | (jnp.isfinite(flux) & covered))
        image = fraction_image(
            data, self.grid, (), self._offaxis_backend.measure, quantity=scene.quantity
        )
        return SceneRender(on_grid(image, grid), valid)
