"""Dynamic fixed-product backend for versioned intensity deliveries."""

from typing import Any

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from lod_unit import lod
from optixstuff.grids import Measure, PixelGrid

from .._jax_funcs import synthesize_psf_idw
from ..interpolation import InterpolationRead, ReadLedger, blend_read
from ..sampling import fraction_image
from ..sinc.evidence import Reason
from ..sinc.reader import require_read

# Construction/query fallbacks allowed for declared complete full rectangles.
_FALLBACK_MASK = int(
    Reason.LAYOUT_UNSUPPORTED
    | Reason.METADATA_MISSING
    | Reason.NORMALIZATION_UNSUPPORTED
    | Reason.FRAME_UNSUPPORTED
    | Reason.UNKNOWN_BANDWIDTH
    | Reason.NYQUIST_FAIL
    | Reason.KERNEL_MARGIN_FAIL
    | Reason.INCOMPLETE_STENCIL
    | Reason.OUTSIDE_DOMAIN
    | Reason.SYMMETRY_UNSUPPORTED
    | Reason.PRECISION_UNAVAILABLE
    | Reason.MEMORY_LIMIT
)


class DeliveryBackend(eqx.Module):
    """Own the optional sinc reader and signed full-grid blend arrays dynamically.

    Quadrant products never create a blend representation. For these products,
    auto refusal returns statuses/NaNs because midpoint-aware blend is unavailable.

    Attributes:
        training_nodes: Stored PSFs of a prefiltered reader in row-major lattice order,
            or None.
        reader: The sinc reader, or None when only blend reads are used.
        blend_nodes: Signed full-grid stored PSFs [node, y, x] for blend reads, or None.
        blend_offsets: Source offsets of ``blend_nodes`` [node, 2] in lambda0/D, or
            None.
        x_offsets: Lattice x positions of the stored sources, in lambda0/D.
        y_offsets: Lattice y positions of the stored sources, in lambda0/D.
        conversion: Factor that turns stored intensity into photon fractions per pixel.
        pitch_lod: Detector pixel pitch in reference lambda/D per pixel.
        measure: Measure of the stored samples.
        grid: Native detector pixel grid.
        psf_shape: Detector array shape (ny, nx) of one stored PSF.
        requested: Interpolation mode that was requested, such as ``"sinc"`` or
            ``"auto"``.
        fallback_reasons: Bit mask of the reasons that sent the product to the blend
            route.
        n_psfs: Number of stored PSFs.
        product_identity: Manifest digest and plane ID that identify the product.
        type: Backend type tag; ``"sinc"`` for this backend.
    """

    training_nodes: jax.Array | None
    reader: Any
    blend_nodes: jax.Array | None
    blend_offsets: jax.Array | None
    x_offsets: jax.Array
    y_offsets: jax.Array
    conversion: float
    pitch_lod: float
    measure: Measure = eqx.field(static=True)
    grid: PixelGrid
    psf_shape: tuple[int, int]
    requested: str = eqx.field(static=True)
    fallback_reasons: int = eqx.field(static=True)
    n_psfs: int = eqx.field(static=True)
    product_identity: str = eqx.field(static=True)
    type: str = eqx.field(static=True, default="sinc")

    def __init__(self, delivery, reader, requested, fallback_reasons=0):
        """Bind one selected native plane and its photon-unit conversion.

        Args:
            delivery: Validated delivery whose selected plane is bound.
            reader: Sinc reader built from the delivery, or None for blend-only reads.
            requested: Interpolation mode that was requested, such as ``"sinc"``,
                ``"blend"`` or ``"auto"``.
            fallback_reasons: Bit mask of the reasons that sent the product to the blend
                route; 0 when none.
        """
        self.reader = reader
        self.training_nodes = None
        if reader is not None and reader.grid.stored == "prefiltered_coefficients":
            order = np.argsort(
                delivery.indices[:, 1] * delivery.shape_yx[1] + delivery.indices[:, 0]
            )
            self.training_nodes = jnp.asarray(delivery.nodes)[order]
        full = delivery.manifest["layout"] == "full_rectangle" and (
            reader is None or requested == "auto"
        )
        self.blend_nodes = jnp.asarray(delivery.nodes) if full else None
        self.blend_offsets = jnp.asarray(delivery.offsets) if full else None
        if full and self.training_nodes is not None:
            self.blend_nodes = self.training_nodes
            self.blend_offsets = self.blend_offsets[order]
        origin, spacing = (
            delivery.origin_lod,
            np.array(delivery.manifest["source_spacing_lod"]),
        )
        self.x_offsets = jnp.asarray(
            origin[0] + spacing[0] * np.arange(delivery.shape_yx[1])
        )
        self.y_offsets = jnp.asarray(
            origin[1] + spacing[1] * np.arange(delivery.shape_yx[0])
        )
        self.conversion = delivery.conversion
        self.measure = delivery.measure
        self.pitch_lod = float(delivery.manifest["pixel_scale_lod_xy"][0])
        self.grid = delivery.grid
        self.psf_shape = tuple(delivery.manifest["detector_shape_yx"])
        self.requested = requested
        self.fallback_reasons = fallback_reasons
        self.n_psfs = len(delivery.nodes)
        self.product_identity = delivery.manifest_sha256 + ":" + delivery.plane["id"]

    @property
    def pixel_scale_arcsec(self):
        """Compatibility pitch in reference lambda/D per pixel."""
        return self.pitch_lod * lod / u.pix

    @property
    def max_offset_in_image(self):
        """Largest positive detector pixel-center offset in reference units."""
        return self.grid.center_x * self.pitch_lod * lod

    def get_psf_by_offset_idx(self, x_idx, y_idx):
        """Return a measured training image for performance diagnostics.

        Args:
            x_idx: Index of the source along the x lattice axis.
            y_idx: Index of the source along the y lattice axis.

        Returns:
            The measured training image in photon fractions on the native detector grid.
        """
        if self.reader is not None:
            index = y_idx * self.x_offsets.size + x_idx
            image = (
                self.reader.grid.intensity[0, y_idx, x_idx]
                if self.training_nodes is None
                else self.training_nodes[index]
            )
        else:
            offset = jnp.array([self.x_offsets[x_idx], self.y_offsets[y_idx]])
            index = jnp.argmin(jnp.sum((self.blend_offsets - offset) ** 2, axis=1))
            image = self.blend_nodes[index]
        return fraction_image(
            np.asarray(image) * self.conversion,
            self.grid,
            (),
            self.measure,
        )

    def _blend_image(self, query):
        size = self.psf_shape[-1]
        pad = int(1.5 * size)
        if (size + 2 * pad) % 2:
            pad += 1
        fft_size = size + 2 * pad
        return (
            synthesize_psf_idw(
                query[0],
                query[1],
                self.pitch_lod,
                self.blend_nodes,
                self.blend_offsets[:, 0],
                self.blend_offsets[:, 1],
                jnp.fft.rfftfreq(fft_size),
                jnp.fft.fftfreq(fft_size),
                pad,
                False,
                False,
                "2d",
            )
            * self.conversion
        )

    @eqx.filter_jit
    def create_psf_with_info(self, x_lod, y_lod):
        """Read a fixed product and return photon-unit image and diagnostics.

        Args:
            x_lod: Source x position in lambda0/D.
            y_lod: Source y position in lambda0/D.

        Returns:
            The interpolation read: the photon-unit image with its acceptance, status,
            error evidence and ledger.
        """
        dtype = (
            jnp.float64
            if self.reader is not None
            else jax.dtypes.canonicalize_dtype(jnp.float64)
        )
        query = jnp.array([x_lod, y_lod], dtype=dtype)
        finite = jnp.all(jnp.isfinite(query))
        if self.reader is None:
            safe = jnp.where(finite, query, jnp.zeros(2))
            return blend_read(
                self._blend_image(safe),
                finite,
                self.fallback_reasons,
                fallback=self.requested == "auto",
                target_id="incident_photon_per_pixel:" + self.product_identity,
                profile_id=f"blend:conversion-{self.conversion}",
            )
        result = self.reader.read(query, jnp.int32(0))
        ledger = result.ledger
        factor = self.conversion
        converted = InterpolationRead(
            result.intensity * factor,
            result.accepted,
            result.sampling_ok,
            result.status,
            result.reason_bits,
            result.bound_available,
            result.bound_kind,
            result.abs_error_bound * factor,
            result.empirical_max_abs * factor,
            ReadLedger(
                ledger.bound * factor,
                ledger.evidence,
                ledger.applicable,
                ledger.term_ids,
                "incident_photon_per_pixel:" + self.product_identity,
                "absolute_max_pixel",
                ledger.profile_id + f":photon-per-pixel:conversion-{factor}",
                ledger.bound_kind,
            ),
            jnp.int32(1),
        )
        if (
            self.requested == "auto"
            and self.blend_nodes is not None
            and self.reader.policy.mode == "sampling"
        ):
            bits = converted.reason_bits
            eligible = (
                finite
                & ~converted.accepted
                & (
                    (
                        bits
                        & jnp.uint32(
                            ~(
                                _FALLBACK_MASK
                                | int(Reason.BOUND_UNKNOWN | Reason.EMPIRICAL_ONLY)
                            )
                            & 0xFFFFFFFF
                        )
                    )
                    == 0
                )
            )

            def fallback():
                fallback_result = blend_read(
                    self._blend_image(jnp.where(finite, query, jnp.zeros(2))),
                    finite,
                    fallback=True,
                )
                # Keep the sinc ledger shape and original refusal reasons.
                return eqx.tree_at(
                    lambda r: (
                        r.intensity,
                        r.accepted,
                        r.status,
                        r.reason_bits,
                        r.selected_backend,
                        r.empirical_max_abs,
                    ),
                    converted,
                    (
                        fallback_result.intensity,
                        fallback_result.accepted,
                        fallback_result.status,
                        bits | jnp.uint32(Reason.FALLBACK_USED),
                        jnp.int32(0),
                        jnp.asarray(jnp.nan),
                    ),
                )

            return jax.lax.cond(eligible, fallback, lambda: converted)
        return converted

    def create_psfs_with_info(self, x_lod, y_lod):
        """Read equal-length coordinate vectors with serial bounded scratch.

        Args:
            x_lod: Source x positions in lambda0/D, a vector of length N.
            y_lod: Source y positions in lambda0/D, a vector of the same length.

        Returns:
            The interpolation reads of the N sources, stacked on a leading axis of
            length N.
        """
        dtype = (
            jnp.float64
            if self.reader is not None
            else jax.dtypes.canonicalize_dtype(jnp.float64)
        )
        x, y = jnp.asarray(x_lod, dtype), jnp.asarray(y_lod, dtype)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Coordinates must be equal-length vectors")
        if self.reader is not None:
            estimates = self.reader.memory_estimate(x.size)
            if self.training_nodes is not None:
                for key in ("resident", "total"):
                    estimates[key] += self.training_nodes.nbytes
            estimates["output"] += x.size * 4
            estimates["total"] += x.size * 4
            self.reader.memory.check(estimates)
        if not x.size:
            shape = eqx.filter_eval_shape(self.create_psf_with_info, 0.0, 0.0)
            return jax.tree.map(lambda a: jnp.empty((0, *a.shape), a.dtype), shape)
        return jax.lax.map(lambda xy: self.create_psf_with_info(*xy), (x, y))

    def _require(self, result):
        if self.reader is not None and not isinstance(result.accepted, jax.core.Tracer):
            require_read(result)
        return result.intensity

    def create_psf(self, x_lod, y_lod):
        """Return photon fractions, raising eagerly on a refused query.

        Args:
            x_lod: Source x position in lambda0/D.
            y_lod: Source y position in lambda0/D.

        Returns:
            The PSF image of the source in photon fractions on the native detector grid.
        """
        return fraction_image(
            self._require(self.create_psf_with_info(x_lod, y_lod)),
            self.grid,
            (),
            self.measure,
        )

    def create_psfs(self, x_lod, y_lod):
        """Return photon fractions for a vector of source coordinates.

        Args:
            x_lod: Source x positions in lambda0/D, a vector of length N.
            y_lod: Source y positions in lambda0/D, a vector of the same length.

        Returns:
            PSF images in photon fractions on the native detector grid, with a leading
            source axis of length N.
        """
        return fraction_image(
            self._require(self.create_psfs_with_info(x_lod, y_lod)),
            self.grid,
            ("source",),
            self.measure,
        )

    def create_psfs_parallel(self, x_vals, y_vals):
        """Use the same bounded batch route; parallel tuning is not provided.

        Args:
            x_vals: Source x positions in lambda0/D, a vector of length N.
            y_vals: Source y positions in lambda0/D, a vector of the same length.

        Returns:
            The same images as ``create_psfs``.
        """
        return self.create_psfs(x_vals, y_vals)

    def __call__(self, x, y):
        """Accept reference-unit quantities at the host compatibility boundary.

        Args:
            x: Source x position in reference lambda/D; a quantity or a float.
            y: Source y position in reference lambda/D; a quantity or a float.

        Returns:
            The PSF image of the source, as returned by ``create_psf``.
        """
        return self.create_psf(getattr(x, "value", x), getattr(y, "value", y))
