"""Immutable YIP tables implementing the shared coronagraph contract."""

from math import pi
from pathlib import Path

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Frame, Image, PixelGrid
from hwoutils.spectral import Band
from optixstuff.apertures import DEFAULT_APERTURE_RULE, ApertureRule
from optixstuff.coronagraph import AbstractTableCoronagraph
from optixstuff.performance import PerformanceCurves, leakage_weights
from optixstuff.primary import SimplePrimary

from ._precision import float_dtype
from ._yip import _YIPData
from .export import export_ayo_csv, export_exosims
from .header import HeaderData
from .interpolation import SincOptions, blend_read


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


def _native_samples(host, record):
    """Retain the shipped PSFs and nearest-to-zero YIP row for native weights."""
    offaxis = host.offax
    y_offsets = np.asarray(offaxis.y_offsets)
    iy = int(np.argmin(np.abs(y_offsets)))
    offsets = np.asarray(offaxis.x_offsets, dtype=record.separation_lod.dtype)
    indices = {float(x): ix for ix, x in enumerate(offsets)}
    return tuple(
        (
            offaxis.get_psf_by_offset_idx(indices[float(s)], iy),
            (float(s), float(y_offsets[iy])),
        )
        if float(s) in indices
        else (host.create_psf(float(s), 0.0), (float(s), 0.0))
        for s in np.asarray(record.separation_lod)
    )


class Coronagraph(AbstractTableCoronagraph):
    """A frozen, JAX-compatible coronagraph loaded from a Yield Input Package.

    Coordinates and stellar diameters are floats in lambda/D. Aperture rules
    and leakage geometry are prepared at construction, before JIT queries.
    The default photometric rule is a circle of radius 0.7 lambda/D.
    """

    grid: PixelGrid
    stellar_grid: PixelGrid
    sky_trans: Image
    psf_datacube: object
    header: HeaderData = eqx.field(static=True)
    yip_path: Path
    interpolation_info: object = eqx.field(static=True)
    bandpass: Band | None
    outer_working_angle_lod: None
    _offaxis_backend: object
    _stellar_ln_interp: interpax.CubicSpline
    _records: tuple[PerformanceCurves, ...]
    _leakage_operators: tuple
    _occultation_record: PerformanceCurves
    _load_options: tuple = eqx.field(static=True)
    _sinc_configuration: tuple
    _source_signature: str = eqx.field(static=True)

    def __init__(
        self,
        yip_path,
        stellar_intens_file="stellar_intens.fits",
        stellar_diam_file="stellar_intens_diam_list.fits",
        offax_data_file="offax_psf.fits",
        offax_offsets_file="offax_psf_offset_list.fits",
        sky_trans_file="sky_trans.fits",
        x_symmetric=True,
        y_symmetric=True,
        use_quarter_psf_datacube=True,
        downsample_shape=None,
        *,
        rules=(DEFAULT_APERTURE_RULE,),
        leakage_rule="aperture_mean",
        interpolation="blend",
        sinc_options: SincOptions | None = None,
    ):
        """Load host products and freeze their tables and prepared aperture rules."""
        rules = tuple(dict.fromkeys(rules))
        if not rules:
            raise ValueError("rules must contain at least one aperture rule")
        options = (
            stellar_intens_file,
            stellar_diam_file,
            offax_data_file,
            offax_offsets_file,
            sky_trans_file,
            x_symmetric,
            y_symmetric,
            use_quarter_psf_datacube,
            downsample_shape,
        )
        host = _YIPData(
            yip_path,
            *options,
            rule=rules[0],
            leakage_rule=leakage_rule,
            interpolation=interpolation,
            sinc_options=sinc_options,
        )
        self.yip_path = Path(yip_path)
        self._load_options = (options, interpolation)
        config = host._sinc_options
        self._sinc_configuration = (
            config.product_id,
            config.domain,
            config.memory,
            config.policy,
            config.taps,
        )
        self._source_signature = host._source_sig
        self.header = host.header
        self.grid = host.grid
        self.stellar_grid = host.stellar_intens.grid
        self.interpolation_info = host.interpolation_info
        self.outer_working_angle_lod = None
        spectral_header = host.offaxis_header
        self.bandpass = None
        if spectral_header.minlam is not None and spectral_header.maxlam is not None:
            lo_nm, hi_nm = (
                spectral_header.minlam.to_value("nm"),
                spectral_header.maxlam.to_value("nm"),
            )
            # A single recorded wavelength has no finite-width spectral band.
            if lo_nm != hi_nm:
                self.bandpass = Band(lo_nm, hi_nm)
        self._offaxis_backend = host.offax
        self._stellar_ln_interp = interpax.CubicSpline(
            jnp.asarray(host.stellar_intens.ln_interp.x, dtype=float_dtype()),
            jnp.log(jnp.asarray(host.stellar_intens.psfs, dtype=float_dtype())),
        )
        records = tuple(host._prepare_record(rule) for rule in rules)
        self._records = tuple(_jax_record(record) for record in records)
        self._leakage_operators = tuple(
            jax.tree.map(
                jnp.asarray,
                leakage_weights(
                    lambda s: host.create_psf(s, 0.0),
                    self._native_stellar_map(0.0),
                    record,
                    samples=_native_samples(host, record),
                ),
            )
            for record in self._records
        )
        self._occultation_record = _jax_record(host.occultation_record)
        self.sky_trans = host.sky_trans()
        self.psf_datacube = None

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

    def prepared(
        self, rule: ApertureRule, *, leakage_rule="aperture_mean"
    ) -> tuple["Coronagraph", PerformanceCurves]:
        """Return this model if ready, or prepare the requested record on the host."""
        record = self._find_record(rule, leakage_rule)
        if record is not None:
            return self, record
        model = self.with_rules((rule,), leakage_rule=leakage_rule)
        return model, model.curves(rule, leakage_rule=leakage_rule)

    def _find_record(self, rule, leakage_rule):
        return next(
            (
                record
                for record in self._records
                if record.rule == rule and record.leakage_rule == leakage_rule
            ),
            None,
        )

    def _prepared_leakage_weights(self, rule, leakage_rule):
        for record, weights in zip(self._records, self._leakage_operators, strict=True):
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return weights
        raise ValueError("No prepared stellar-map weights for this rule")

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Dimensionless transmission measured from the ancillary sky product."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be floats in lambda/D")
        return self._occultation_record.evaluate(
            "occulter_transmission", separation_lod
        )

    def stellar_frame(self, wavelength_nm) -> Frame:
        """Return the frame of the native stellar intensity product."""
        return self.stellar_grid.frame

    def _native_stellar_map(self, stellar_diam_lod):
        if hasattr(stellar_diam_lod, "unit"):
            raise TypeError("stellar_diam_lod must be floats in lambda/D")
        data = jnp.exp(self._stellar_ln_interp(stellar_diam_lod))
        return Image(
            data,
            self.stellar_grid,
            tuple(f"diameter_{i}" for i in range(data.ndim - 2)),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
            outside="reflect",
        )

    def _native_source_psfs(self, x_lod, y_lod):
        if hasattr(x_lod, "unit") or hasattr(y_lod, "unit"):
            raise TypeError("source coordinates must be floats in lambda/D")
        if (
            not self.interpolation_info.manifest_sha256
            and self.interpolation_info.requested == "auto"
        ):
            return Image(
                self.source_psfs_with_info(x_lod, y_lod).intensity,
                self.grid,
                ("source",),
                "per_pixel",
                "stellar_flux_fraction",
                reconstruction="band_limited",
            )
        return self._offaxis_backend.create_psfs(
            jnp.atleast_1d(jnp.asarray(x_lod)), jnp.atleast_1d(jnp.asarray(y_lod))
        )

    def source_psfs_with_info(self, x_lod, y_lod):
        """Native source samples with admission status and backend diagnostics.

        Unlike source_psfs, this returns InterpolationRead diagnostics, whose
        intensity has shape (source, y, x), on the native grid only.
        """
        if hasattr(x_lod, "unit") or hasattr(y_lod, "unit"):
            raise TypeError("source coordinates must be floats in lambda/D")
        x_lod, y_lod = (
            jnp.atleast_1d(jnp.asarray(x_lod)),
            jnp.atleast_1d(jnp.asarray(y_lod)),
        )
        backend = self._offaxis_backend
        if hasattr(backend, "create_psfs_with_info"):
            return backend.create_psfs_with_info(x_lod, y_lod)

        def one(x, y):
            finite = jnp.isfinite(x) & jnp.isfinite(y)
            image = backend.create_psf(
                jnp.where(finite, x, 0.0), jnp.where(finite, y, 0.0)
            )
            info = self.interpolation_info
            reasons = (
                (1 << 21)
                if "EXTRA_MISSING" in info.reasons
                else ((1 << 0) if "LAYOUT_UNSUPPORTED" in info.reasons else 0)
            )
            return blend_read(
                image.data, finite, reasons, fallback=info.requested == "auto"
            )

        return jax.vmap(one)(x_lod, y_lod)

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
        host = self._reopen_host(rules[0], leakage_rule)
        records = tuple(_jax_record(host._prepare_record(rule)) for rule in rules)
        operators = tuple(
            jax.tree.map(
                jnp.asarray,
                leakage_weights(
                    lambda s: host.create_psf(s, 0.0),
                    self._native_stellar_map(0.0),
                    record,
                    samples=_native_samples(host, record),
                ),
            )
            for record in records
        )
        return eqx.tree_at(
            lambda c: (c._records, c._leakage_operators), self, (records, operators)
        )

    def _reopen_host(self, rule, leakage_rule):
        """Reopen the YIP on the host with this coronagraph's load options."""
        options, interpolation = self._load_options
        host = _YIPData(
            self.yip_path,
            *options,
            rule=rule,
            leakage_rule=leakage_rule,
            interpolation=interpolation,
            sinc_options=SincOptions(*self._sinc_configuration),
        )
        if host._source_sig != self._source_signature:
            raise ValueError("YIP source files changed; construct a new Coronagraph")
        host.offax = self._offaxis_backend
        return host

    def create_psf_datacube(self, batch_size=128):
        """Return a new coronagraph with a cached PSF cube for extended scenes.

        Assign the returned module: coro = coro.create_psf_datacube(). Loading
        and cache writes happen on the host, outside JAX transformations.
        """
        record = self._records[0]
        host = self._reopen_host(record.rule, record.leakage_rule)
        host.create_psf_datacube(batch_size)
        return eqx.tree_at(
            lambda c: c.psf_datacube,
            self,
            jnp.asarray(host.psf_datacube),
            is_leaf=lambda x: x is None,
        )

    def to_exosims(self, units="LAMBDA/D", *, separations_lod=None):
        """Export the first prepared record to EXOSIMS FITS tables."""
        return export_exosims(self, units=units, separations_lod=separations_lod)

    def dump_ayo_csv(
        self,
        output_path,
        sep_min=0.125,
        sep_max=32.0,
        sep_step=0.25,
        contrast_floor=None,
        ppf=30.0,
        *,
        separations_lod=None,
    ):
        """Export the first prepared record to an AYO CSV table."""
        return export_ayo_csv(
            self,
            output_path,
            sep_min=sep_min,
            sep_max=sep_max,
            sep_step=sep_step,
            contrast_floor=contrast_floor,
            ppf=ppf,
            separations_lod=separations_lod,
        )

    def __repr__(self):
        """Summarize the native grid and prepared rules."""
        return (
            f"Coronagraph({self.yip_path.name!r}, grid={self.grid.shape}, "
            f"rules={len(self._records)})"
        )
