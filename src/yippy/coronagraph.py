"""Immutable YIP tables implementing the shared coronagraph contract."""

from copy import copy
from dataclasses import fields, replace
from math import pi

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
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
from optixstuff.performance import PerformanceCurves, SampledProfile
from optixstuff.primary import SimplePrimary
from optixstuff.spectral import Band

from ._precision import float_dtype
from ._yip import _YIPData
from .header import HeaderData
from .interpolation import SincOptions
from .sampling import fraction_image
from .scene import DenseSceneOperator
from .sinc import SceneRenderer
from .sinc.reader import PrefilteredReader


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

    def __init__(
        self,
        package,
        *,
        x_symmetric=True,
        y_symmetric=True,
        use_quarter_psf_datacube=True,
        downsample_shape=None,
        rules=(DEFAULT_APERTURE_RULE,),
        leakage_rule="aperture_mean",
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

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Select one prepared record; requesting an unprepared rule raises."""
        record = self._find_record(rule, leakage_rule)
        if record is None:
            raise ValueError(
                "No prepared record; supply rules and leakage_rule at construction"
            )
        return record

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
            outside="reflect",
        )

    def _native_source_psfs(self, x_lod, y_lod):
        if hasattr(x_lod, "unit") or hasattr(y_lod, "unit"):
            raise TypeError("source coordinates must be floats in lambda/D")
        x, y = jnp.atleast_1d(jnp.asarray(x_lod)), jnp.atleast_1d(jnp.asarray(y_lod))
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Coordinates must be equal-length vectors")
        backend = self._offaxis_backend
        if hasattr(backend, "create_psfs_with_info"):
            read = backend.create_psfs_with_info(x, y)
            image = fraction_image(
                read.intensity,
                self.grid,
                ("source",),
                backend.measure,
            )
            return SourcePSFs(image, read.accepted)
        finite = jnp.isfinite(x) & jnp.isfinite(y)
        image = backend.create_psfs(
            jnp.where(finite, x, 0.0), jnp.where(finite, y, 0.0)
        )
        valid = finite & jnp.all(jnp.isfinite(image.data), axis=(-1, -2))
        return SourcePSFs(
            replace(image, data=jnp.where(valid[:, None, None], image.data, jnp.nan)),
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
            leakage_rule = self._records[0].leakage_rule
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

    def with_scene_operator(self, *, batch_size=128):
        """Prepare a scene renderer for this model on its native source grid.

        Admitted prefiltered products render from their coefficient table.
        Other interpolation modes cache dense PSFs with per-source admission.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        backend = self._offaxis_backend
        reader = getattr(backend, "reader", None)
        if isinstance(reader, PrefilteredReader) and backend.blend_nodes is None:
            x, y = self.grid.coordinates()
            operator = SceneRenderer.from_raster(reader, 0, x[0], y[:, 0])
        else:
            record = self._records[0]
            host = self._host_for_rule(record.rule, record.leakage_rule)
            # Signed tables and permission regions can break mirror symmetry.
            host.use_quarter_psf_datacube = (
                host.use_quarter_psf_datacube
                and backend.type in ("1d", "2dq")
                and host.x_symmetric
                and host.y_symmetric
                and self.grid.is_geometric
            )
            host._prepare_dense_psfs(batch_size)
            cube = jnp.asarray(host.psf_datacube)
            valid = (
                host.psf_datacube_info["accepted"]
                if host._sinc_delivery is not None
                else jnp.ones(cube.shape[:2], dtype=bool)
            )
            operator = DenseSceneOperator(cube, valid)
        return SceneCoronagraph(self, operator)

    def __repr__(self):
        """Summarize the native grid and prepared rules."""
        return (
            f"Coronagraph({self.package.source.path.name!r}, grid={self.grid.shape}, "
            f"rules={len(self._records)})"
        )


class SceneCoronagraph(Coronagraph, AbstractSceneCoronagraph):
    """YIP imaging model with a prepared native scene operator."""

    operator: DenseSceneOperator | SceneRenderer

    def __init__(self, model, operator):
        """Reuse the prepared imaging tables and bind the chosen operator."""
        for field in fields(Coronagraph):
            setattr(self, field.name, getattr(model, field.name))
        self.operator = operator

    def render_scene(self, scene, wavelength_nm, *, grid):
        """Sum native source-pixel flux through the operator and sample the detector."""
        if scene.measure != Measure.integral() or scene.data.ndim != 2:
            raise ValueError("scene must be a two-dimensional cell-integral Image")
        flux = scene.data if scene.grid == self.grid else on_grid(scene, self.grid).data
        if isinstance(self.operator, SceneRenderer):
            x, y = self.grid.coordinates()
            read = self.operator.render_raster(
                jnp.asarray(flux, jnp.float64), x[0], y[:, 0]
            )
            data, valid = (
                read.intensity * self._offaxis_backend.conversion,
                read.accepted,
            )
            if self.operator.reader.policy.mode == "bounded":
                # A dark raster is exactly zero without making a bounded PSF read.
                dark = jnp.all(flux == 0)
                data, valid = jnp.where(dark, 0.0, data), valid | dark
        else:
            data, valid = self.operator.render(flux, self.grid)
        image = fraction_image(
            data, self.grid, (), self._offaxis_backend.measure, quantity=scene.quantity
        )
        return SceneRender(on_grid(image, grid), valid)
