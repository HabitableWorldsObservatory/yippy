"""Float64 tiled interpolation and bounded serial query mapping."""

import dataclasses
import math
from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from yippy.sinc._types import F64, I32, U32, Boolean, check_array, check_static
from yippy.sinc.domain import Domain, coverage, in_supported_range, safe_query
from yippy.sinc.evidence import (
    BoundKind,
    EvidenceBinding,
    Ledger,
    MemoryLimit,
    PrecisionUnavailable,
    ReadRefused,
    Reason,
    Status,
    UnknownErrorModel,
)
from yippy.sinc.geometry import origin_xy, spacing_xy
from yippy.sinc.grid import Products, SampledGrid
from yippy.sinc.kernel import (
    AbstractKernelProfile,
    KernelProfile,
    PrefilteredKernelProfile,
    flush_subnormal,
    read_stencil,
)
from yippy.sinc.prefilter import (
    BUILD_CHUNK_BYTES,
    BUILD_SAMPLE_BYTES,
    build_coefficients,
    design_grid,
    design_record,
)


def _require_precision():
    if not jax.config.x64_enabled:
        raise PrecisionUnavailable(
            "Enable jax_enable_x64 before construction or tracing"
        )


def _default_budget():
    _require_precision()
    return jnp.asarray(jnp.inf, dtype=jnp.float64)


class ReadPolicy(eqx.Module):
    """Sampling permits unknown errors; bounded always refuses in this version.

    Attributes:
        mode: Policy mode, "sampling" or "bounded".
        max_abs_error: Largest accepted absolute error, in delivered sample
            units, as a float64 scalar. Must be infinite for "sampling" and
            finite and nonnegative for "bounded". Defaults to infinity.
    """

    mode: str = eqx.field(static=True, default="sampling")
    max_abs_error: F64 = eqx.field(
        default_factory=_default_budget, converter=jnp.asarray
    )

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.max_abs_error, (), (jnp.float64,))
        budget = float(self.max_abs_error)
        if self.mode not in ("sampling", "bounded"):
            raise ValueError("Policy must be sampling or bounded")
        if self.mode == "sampling" and budget != math.inf:
            raise ValueError("Sampling policy requires an infinite budget")
        if self.mode == "bounded" and (not math.isfinite(budget) or budget < 0):
            raise ValueError("Bounded policy requires a finite nonnegative budget")


class Tiling(eqx.Module):
    """Positive static query chunk and flattened detector tile sizes.

    Attributes:
        query_chunk: Number of queries read per serial chunk. Defaults to 1.
        pixel_tile: Number of flattened detector pixels per tile in a raw
            read. Defaults to 256.
    """

    query_chunk: int = eqx.field(static=True, default=1)
    pixel_tile: int = eqx.field(static=True, default=256)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if any(
            type(n) is not int or n <= 0 for n in (self.query_chunk, self.pixel_tile)
        ):
            raise ValueError("Tiling counts must be positive integers")


class MemoryBudget(eqx.Module):
    """Explicit byte ceilings; no machine-RAM assumptions or out-of-core mode.

    Attributes:
        max_resident_bytes: Ceiling on the resident table estimate, in bytes.
        max_output_bytes: Ceiling on the result estimate, in bytes.
        max_scratch_bytes: Ceiling on the scratch estimate, in bytes.
        max_total_bytes: Ceiling on the total estimate, in bytes.
    """

    max_resident_bytes: int = eqx.field(static=True)
    max_output_bytes: int = eqx.field(static=True)
    max_scratch_bytes: int = eqx.field(static=True)
    max_total_bytes: int = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        values = (
            self.max_resident_bytes,
            self.max_output_bytes,
            self.max_scratch_bytes,
            self.max_total_bytes,
        )
        if any(type(n) is not int or n <= 0 for n in values):
            raise ValueError("Memory budgets must be positive integer byte limits")

    def check(self, estimates):
        """Raise MemoryLimit when any logical allocation exceeds its ceiling.

        Args:
            estimates: Mapping with the keys "resident", "output", "scratch" and
                "total", in bytes, as returned by ``estimate_memory``.
        """
        for key in ("resident", "output", "scratch", "total"):
            if estimates[key] > getattr(self, f"max_{key}_bytes"):
                raise MemoryLimit(
                    f"{key} estimate {estimates[key]} bytes exceeds budget"
                )

    @classmethod
    def for_table(
        cls,
        resident_bytes,
        shape_yx,
        *,
        profile=None,
        supplied=None,
        queries=1024,
        extra_output_bytes=0,
        extra_resident_bytes=0,
    ):
        """Check admission memory or size a budget for admission and batched reads.

        Args:
            resident_bytes: Size of the stored samples, in bytes.
            shape_yx: Detector shape as (rows, columns), in pixels.
            profile: Kernel profile; None means the raw profile.
            supplied: Budget to check against the admission estimate. When
                given, it is returned unchanged if it passes.
            queries: Number of queries in the batched read the budget must
                also cover.
            extra_output_bytes: Bytes added to the output and total ceilings.
            extra_resident_bytes: Bytes added to the resident and total
                estimates.

        Returns:
            ``supplied`` when it is given, else a new budget whose ceilings are
            the larger of the admission and batched-read estimates.
        """
        admission = estimate_memory(
            resident_bytes, shape_yx, Tiling(), 1, admission=True, profile=profile
        )
        for key in ("resident", "total"):
            admission[key] += extra_resident_bytes
        if supplied is not None:
            supplied.check(admission)
            return supplied
        batch = estimate_memory(
            resident_bytes, shape_yx, Tiling(), queries, profile=profile
        )
        for key in ("resident", "total"):
            batch[key] += extra_resident_bytes
        ceilings = {key: max(admission[key], batch[key]) for key in admission}
        for key in ("output", "total"):
            ceilings[key] += extra_output_bytes
        return cls(
            *(int(ceilings[key]) for key in ("resident", "output", "scratch", "total"))
        )


def estimate_memory(
    resident_bytes, shape_yx, tiling, queries, *, admission=False, profile=None
):
    """Estimate logical allocations, including all result leaves and padding.

    ``resident_bytes`` is the size of the stored samples. ``profile`` None means
    the raw profile. A prefiltered reader keeps a coefficient table of the same
    size in place of the samples, so its resident estimate equals the samples;
    its scratch counts one taps x taps block of images in the storage dtype and
    in float64, and the load-time build (one float64 detector-row chunk and two
    transient copies of the coefficients). Detector tiling does not apply to a
    prefiltered read, so ``tiling.pixel_tile`` is ignored for that profile.

    Admission budgets three native copies (input staging, reorder, resident).
    Source count is bounded by resident_bytes / (4 * pixels): at least one
    product and at least four bytes per sample. This deliberately overestimates
    multi-product or float64 deliveries without changing the estimator API.
    Source scratch reserves 192 bytes per possible node for indices (8), offsets
    (16), float64 index/displacement/expected/residual buffers (64), linear keys
    (8), occupancy (8), permutation (8), sort workspace (32), masks/casts (48).
    Occupancy is allocated only when the declared cell count equals node count.
    Read scratch includes the tile, contractions, padded output carry, and one
    serial query chunk. Compiler workspace and caller-owned tables are external.

    Args:
        resident_bytes: Size of the stored samples, in bytes.
        shape_yx: Detector shape as (rows, columns), in pixels.
        tiling: Query chunk and pixel tile sizes.
        queries: Number of queries whose results are held at once.
        admission: Whether to include the admission-time staging and source
            checking allocations.
        profile: Kernel profile; None means the raw profile.

    Returns:
        A dict of byte estimates with the keys "resident", "output", "scratch",
        "staging" and "total".
    """
    pixels = math.prod(shape_yx)
    tile = min(tiling.pixel_tile, pixels)
    # Eight ledger rows plus all scalar diagnostics accompany each image.
    result_bytes = pixels * 8 + 8 * (8 + 4 + 1) + 39
    output = queries * result_bytes
    resident = resident_bytes
    if isinstance(profile, PrefilteredKernelProfile):
        taps = profile.taps
        scratch = 3 * taps * taps * pixels * 8 + 2 * taps * 16 + pixels * 8
        # The sample count is at most resident_bytes / 4 (float32 storage).
        chunk = BUILD_SAMPLE_BYTES * (resident_bytes // 4)
        scratch += max(min(BUILD_CHUNK_BYTES, chunk), chunk // shape_yx[0])
        # Joining row chunks and then products holds the coefficients twice more.
        scratch += 2 * resident_bytes
    else:
        scratch = 32 * 32 * tile * 8 + 32 * tile * 8 + 2 * 32 * 16
        scratch += ((pixels + tile - 1) // tile) * tile * 8
    scratch += tiling.query_chunk * result_bytes
    staging = 2 * resident_bytes if admission else 0
    if admission:
        # Host checking uses at most 65536 samples and one detector geometry.
        scratch += min(resident_bytes // 4, 65536) * 24 + pixels * 64
        scratch += (resident_bytes // (4 * pixels)) * 192
        # Bounded safe-point candidate coordinates, floor and membership work.
        scratch += 65536
    return dict(
        resident=resident,
        output=output,
        scratch=scratch,
        staging=staging,
        total=resident + output + scratch + staging,
    )


class ImageRead(eqx.Module):
    """Image and numerical diagnostics; bounds use delivered sample units.

    Attributes:
        intensity: Float64 image, shape (rows, columns), in delivered sample
            units. NaN where the read was refused.
        accepted: Whether the read was accepted, a scalar.
        sampling_ok: Whether the query passed every sampling check, a scalar.
        status: Int32 scalar status code from ``Status``.
        reason_bits: Uint32 scalar bit mask of ``Reason`` flags.
        bound_available: Whether an error bound is available, a scalar.
        bound_kind: Int32 scalar kind code from ``BoundKind``.
        abs_error_bound: Float64 scalar absolute error bound, in delivered
            sample units; infinity when unknown.
        empirical_max_abs: Float64 scalar empirical maximum absolute error of
            the grid checks, in delivered sample units.
        ledger: Per-source evidence ledger of the read.
    """

    intensity: F64
    accepted: Boolean
    sampling_ok: Boolean
    status: I32
    reason_bits: U32
    bound_available: Boolean
    bound_kind: I32
    abs_error_bound: F64
    empirical_max_abs: F64
    ledger: Ledger

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if self.intensity.ndim != 2:
            raise ValueError("Scalar read images require two detector axes")
        check_array(self.intensity, self.intensity.shape, (jnp.float64,))
        for value in (self.accepted, self.sampling_ok, self.bound_available):
            check_array(value, (), (jnp.bool_,))
        for value in (self.status, self.bound_kind):
            check_array(value, (), (jnp.int32,))
        check_array(self.reason_bits, (), (jnp.uint32,))
        for value in (self.abs_error_bound, self.empirical_max_abs):
            check_array(value, (), (jnp.float64,))


class AbstractImageReader(eqx.Module):
    """Nominal fixed-detector image reader without a storage-layout contract.

    Attributes:
        products: Physical metadata of the delivered products.
        memory: Byte ceilings the reader checks its estimates against.
    """

    products: eqx.AbstractVar[Products]
    memory: eqx.AbstractVar[MemoryBudget]

    @abstractmethod
    def read(self, source_xy_lod: F64, product_index: I32) -> ImageRead:
        """Read one source [2] and int32 product [], returning delivered units.

        Args:
            source_xy_lod: Float64 source position (x, y), shape (2,), in
                lambda/D.
            product_index: Int32 scalar index of the product to read.

        Returns:
            The ``ImageRead`` of the source for that product.
        """


def _bind_evidence(grid, domain, profile):
    """Return a verified covered point and the unknown-error model of a reader."""
    if not in_supported_range(domain):
        raise ValueError("Domain lies outside the supported numeric range")
    point = safe_query(domain, grid.source, grid.reflection.kind != "none", profile)
    if point is None:
        raise ValueError("Domain has no covered query")
    binding = EvidenceBinding(
        grid.receipt_id,
        grid.products.identity,
        "absolute_max_pixel_delivered",
        f"{profile.identity}:storage-{grid.intensity.dtype}:compute-float64:"
        f"jax-{jax.__version__}:{jax.default_backend()}:delivered",
        domain,
    )
    return point, UnknownErrorModel(binding)


class AbstractGridReader(AbstractImageReader):
    """Shared refusal and evidence logic of readers over one admitted grid.

    A concrete reader supplies ``_image``, the contraction of its own table at a
    prevalidated query. Status, reasons and the all-unknown ledger are the same
    for every profile: no profile turns a delivered grid into a bounded read.

    Attributes:
        grid: The admitted sampled grid the reader reads.
        domain: Region of source positions the reader accepts.
        profile: Kernel profile that selects taps and weights.
        policy: Read policy.
        errors: Unknown-error model that builds the evidence ledger.
        tiling: Query chunk and pixel tile sizes.
        memory: Byte ceilings the reader checks its estimates against.
    """

    # What the bound grid's intensity holds once the reader is built.
    _stored: ClassVar[str] = "samples"
    grid: eqx.AbstractVar[SampledGrid]
    domain: eqx.AbstractVar[Domain]
    profile: eqx.AbstractVar[AbstractKernelProfile]
    policy: eqx.AbstractVar[ReadPolicy]
    errors: eqx.AbstractVar[UnknownErrorModel]
    tiling: eqx.AbstractVar[Tiling]
    memory: eqx.AbstractVar[MemoryBudget]
    _safe_query: eqx.AbstractVar[F64]

    @abstractmethod
    def _image(self, query, product):
        """Return the float64 image and per-axis weight l1 of a prevalidated query."""

    def _design(self):
        """Return the (beta, scale) of the kernel design, or None for raw reads."""
        return None, None

    def stencil(self, query):
        """Return the Stencil of a prevalidated query, as ``read`` contracts it.

        Nodes are signed indices, so a reflected quadrant table shows negative
        nodes; ``valid`` marks the nodes the table serves.

        Args:
            query: Float64 source position (x, y), shape (2,), in lambda/D.

        Returns:
            The ``Stencil`` of nodes, weights and validity for the query.
        """
        beta, scale = self._design()
        return read_stencil(
            self.profile,
            query,
            origin_xy(self.grid.source),
            spacing_xy(self.grid.source),
            self.grid.source.shape[::-1],
            self.grid.reflection.kind != "none",
            beta,
            scale,
        )

    def _check_grid(self):
        if self.grid.stored != self._stored:
            raise ValueError(f"Reader requires a grid that stores {self._stored}")
        if not self.grid.receipt_id:
            raise ValueError("Reader requires an admission receipt identity")
        if self.grid.reflection.kind not in ("none", "two_axis_pixel_flip_v1"):
            raise ValueError("Unsupported reflection")
        if self.grid.reflection.kind != "none" and not bool(
            jnp.all(self.grid.reflection.check_passed)
        ):
            raise ValueError("Reflection checks have not passed")
        self.memory.check(self.memory_estimate(1))

    def memory_estimate(self, queries):
        """Estimate this reader's allocations for ``queries`` serial reads.

        Args:
            queries: Number of queries whose results are held at once.

        Returns:
            A dict of byte estimates, as returned by ``estimate_memory``.
        """
        return estimate_memory(
            self.grid.intensity.nbytes,
            self.products.pixels.shape,
            self.tiling,
            queries,
            profile=self.profile,
        )

    @property
    def products(self):
        """Delegate physical metadata to its sole owner, the sampled grid."""
        return self.grid.products

    def _validate(self, query, product):
        """Return sanitized inputs, per-condition flags and the reason mask."""
        finite = jnp.all(jnp.isfinite(query))
        sanitized = flush_subnormal(jnp.where(jnp.isfinite(query), query, 0.0))
        available = (product >= 0) & (product < self.products.wavelength_nm.size)
        member = self.domain.contains(sanitized)
        covered = coverage(
            sanitized,
            self.grid.source,
            self.grid.reflection.kind != "none",
            self.profile,
        )
        reason = jnp.asarray(Reason.BOUND_UNKNOWN, dtype=jnp.uint32)
        for ok, bit in (
            (finite, Reason.NONFINITE_QUERY),
            (available, Reason.PRODUCT_UNAVAILABLE),
            (member, Reason.OUTSIDE_DOMAIN),
            (covered, Reason.INCOMPLETE_STENCIL),
        ):
            reason = reason | jnp.where(ok, jnp.uint32(0), jnp.uint32(bit))
        return sanitized, finite & available & member & covered, reason

    def read(self, source_xy_lod: F64, product_index: I32) -> ImageRead:
        """Return one image; invalid runtime values refuse only their own lane.

        Coordinates and weights use binary64. Exact quotient floors determine
        every tap, including zero-weight endpoints. Bounded policy leaves
        sampling_ok independent of its unknown-bound refusal.

        Args:
            source_xy_lod: Float64 source position (x, y), shape (2,), in
                lambda/D.
            product_index: Int32 scalar index of the product to read.

        Returns:
            The ``ImageRead`` of the source for that product.
        """
        _require_precision()
        query, product = jnp.asarray(source_xy_lod), jnp.asarray(product_index)
        check_array(query, (2,), (jnp.float64,))
        check_array(product, (), (jnp.int32,))
        self.memory.check(self.memory_estimate(1))
        sanitized, sampling_ok, reason = self._validate(query, product)
        temporary_product = jnp.where(
            (product >= 0) & (product < self.products.wavelength_nm.size), product, 0
        )
        accepted = sampling_ok & (self.policy.mode == "sampling")
        if self.grid._has_checks:
            reason = reason | jnp.uint32(Reason.EMPIRICAL_ONLY)
        safe = jnp.where(accepted, sanitized, self._safe_query)
        image, gain = self._image(safe, temporary_product)
        ledger = self.errors.evaluate(safe, temporary_product, gain, grid=self.grid)
        return ImageRead(
            jnp.where(accepted, image, jnp.nan),
            accepted,
            sampling_ok,
            jnp.asarray(
                jnp.where(accepted, Status.SAMPLING_ADMITTED, Status.REFUSED),
                dtype=jnp.int32,
            ),
            reason,
            jnp.asarray(False),
            jnp.asarray(BoundKind.UNKNOWN, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            self.grid._empirical_max_abs,
            ledger,
        )


class Interpolator(AbstractGridReader):
    """Resident raw Kaiser-sinc reader with internally bound unknown evidence.

    Attributes:
        grid: The admitted sampled grid the reader reads.
        domain: Region of source positions the reader accepts.
        profile: The raw kernel profile.
        policy: Read policy.
        errors: Unknown-error model that builds the evidence ledger.
        tiling: Query chunk and pixel tile sizes.
        memory: Byte ceilings the reader checks its estimates against.
    """

    grid: SampledGrid
    domain: Domain
    profile: KernelProfile
    policy: ReadPolicy
    errors: UnknownErrorModel
    tiling: Tiling
    memory: MemoryBudget
    _safe_query: F64

    def __init__(self, grid, domain, profile, policy, tiling, memory):
        """Bind an admitted grid and a nonempty covered domain on the host.

        Args:
            grid: The admitted sampled grid to read.
            domain: Region of source positions the reader accepts.
            profile: Kernel profile of the grid.
            policy: Read policy.
            tiling: Query chunk and pixel tile sizes.
            memory: Byte ceilings the reader checks its estimates against.
        """
        _require_precision()
        if not isinstance(profile, KernelProfile):
            raise TypeError("Interpolator reads with the raw KernelProfile")
        if grid.profile_id not in ("", profile.identity):
            raise ValueError("Grid was admitted under a different kernel profile")
        self.grid, self.domain, self.profile = grid, domain, profile
        self.policy, self.tiling, self.memory = policy, tiling, memory
        self._safe_query, self.errors = _bind_evidence(grid, domain, profile)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        self._check_grid()

    def _image(self, query, product):
        stencil = self.stencil(query)
        taps, weight = stencil.nodes, stencil.weights
        kx, ky = taps[0], taps[1]
        wx, wy = weight[0], weight[1]
        ny, nx = self.products.pixels.shape
        pixels = ny * nx
        tile_size = min(self.tiling.pixel_tile, pixels)
        tile_count = (pixels + tile_size - 1) // tile_size
        reflected = self.grid.reflection.kind != "none"
        source_x = jnp.abs(kx) if reflected else kx
        source_y = jnp.abs(ky) if reflected else ky

        def pixel_tile(tile, output):
            flat = tile * tile_size + jnp.arange(tile_size)
            mask = flat < pixels
            # Padding has its own safe pixel index, independent of source geometry.
            safe_flat = jnp.where(mask, flat, 0)
            rows, columns = safe_flat // nx, safe_flat % nx
            detector_x = jnp.broadcast_to(columns, (32, tile_size))
            if reflected:
                detector_x = jnp.where(kx[:, None] < 0, nx - 1 - detector_x, detector_x)

            def source_row(index, total):
                detector_y = rows
                if reflected:
                    detector_y = jnp.where(ky[index] < 0, ny - 1 - rows, rows)
                samples = self.grid.intensity[
                    product,
                    source_y[index],
                    source_x[:, None],
                    detector_y[None, :],
                    detector_x,
                ].astype(jnp.float64)
                inner = jnp.sum(wx[:, None] * samples, axis=0)
                return total + wy[index] * inner

            values = jax.lax.fori_loop(
                0, 32, source_row, jnp.zeros(tile_size, dtype=jnp.float64)
            )
            values = jnp.where(mask, values, 0.0)
            return jax.lax.dynamic_update_slice(output, values, (tile * tile_size,))

        output = jax.lax.fori_loop(
            0,
            tile_count,
            pixel_tile,
            jnp.zeros(tile_count * tile_size, dtype=jnp.float64),
        )
        return output[:pixels].reshape(ny, nx), jnp.sum(jnp.abs(weight), axis=1)


class PrefilteredReader(AbstractGridReader):
    """Kaiser-Bessel bump reader of coefficients filtered once at construction.

    Construction is the load step: it designs the deapodization filter for the
    grid's declared band in float64, filters the stored samples along both
    source axes and keeps the coefficients, in the storage dtype, in place of
    the samples: the reader's ``grid`` carries the coefficient table as its
    ``intensity`` (``grid.stored`` says so, and ``coefficients`` names the same
    array), so the reader holds one table, not two. A read then sums
    ``profile.taps`` squared coefficient images.

    ``design`` holds the same ordered (key, value) pairs as the admission
    receipt's ``kernel_design``, evaluated for this reader's domain. They
    describe the kernel on the declared band. Reads carry the same unknown
    error bound as raw reads.

    Attributes:
        grid: The admitted grid, carrying the coefficient table as its
            ``intensity``.
        domain: Region of source positions the reader accepts.
        profile: The prefiltered kernel profile.
        policy: Read policy.
        errors: Unknown-error model that builds the evidence ledger.
        tiling: Query chunk and pixel tile sizes.
        memory: Byte ceilings the reader checks its estimates against.
        design: Ordered (key, value) pairs describing the kernel design on the
            declared band.
    """

    grid: SampledGrid
    domain: Domain
    profile: PrefilteredKernelProfile
    policy: ReadPolicy
    errors: UnknownErrorModel
    tiling: Tiling
    memory: MemoryBudget
    _safe_query: F64
    design: tuple[tuple[str, int | float], ...] = eqx.field(static=True)
    _beta_xy: F64
    _scale_xy: F64
    _gain_xy: F64

    def __init__(self, grid, domain, profile, policy, tiling, memory):
        """Bind an admitted grid and build its coefficient table on the host.

        Args:
            grid: The admitted sampled grid to read.
            domain: Region of source positions the reader accepts.
            profile: Kernel profile of the grid.
            policy: Read policy.
            tiling: Query chunk and pixel tile sizes.
            memory: Byte ceilings the reader checks its estimates against.
        """
        _require_precision()
        if not isinstance(profile, PrefilteredKernelProfile):
            raise TypeError("PrefilteredReader needs a PrefilteredKernelProfile")
        if grid.profile_id != profile.identity:
            raise ValueError("Grid was admitted under a different kernel profile")
        if grid.stored != "samples":
            raise ValueError("PrefilteredReader builds from samples, not coefficients")
        self.grid, self.domain, self.profile = grid, domain, profile
        self.policy, self.tiling, self.memory = policy, tiling, memory
        self._safe_query, self.errors = _bind_evidence(grid, domain, profile)
        memory.check(self.memory_estimate(1))
        designs = design_grid(
            profile,
            grid.products,
            grid.source,
            grid.reflection.kind != "none",
            domain,
        )
        self.design = design_record(designs)
        coefficients = build_coefficients(grid.intensity, designs)
        self.grid = dataclasses.replace(
            grid, intensity=coefficients, stored="prefiltered_coefficients"
        )
        self._beta_xy = jnp.asarray([d.beta for d in designs], dtype=jnp.float64)
        self._scale_xy = jnp.asarray([d.scale for d in designs], dtype=jnp.float64)
        self._gain_xy = jnp.asarray([d.sample_gain for d in designs], dtype=jnp.float64)

    _stored: ClassVar[str] = "prefiltered_coefficients"

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        self._check_grid()

    @property
    def coefficients(self):
        """Coefficient table [product, sy, sx, row, column] in the storage dtype."""
        return self.grid.intensity

    def _design(self):
        return self._beta_xy, self._scale_xy

    def _image(self, query, product):
        stencil = self.stencil(query)
        taps, weight = stencil.nodes, stencil.weights
        kx, ky = taps[0], taps[1]
        table = self.coefficients[product]
        if self.grid.reflection.kind == "none":
            count = self.profile.taps
            block = jax.lax.dynamic_slice(
                table, (ky[0], kx[0], 0, 0), (count, count, *table.shape[2:])
            )
        else:
            block = table[jnp.abs(ky)[:, None], jnp.abs(kx)[None, :]]
            block = jnp.where((kx < 0)[None, :, None, None], block[..., ::-1], block)
            block = jnp.where((ky < 0)[:, None, None, None], block[..., ::-1, :], block)
        image = jnp.einsum(
            "j,i,jirc->rc", weight[1], weight[0], block.astype(jnp.float64)
        )
        # Stored-sample errors reach a read through the composite of filter and
        # bump, so the gain reported to the error model is the design figure.
        return image, self._gain_xy


def read_batch(
    reader: AbstractImageReader, source_xy_lod: F64, product_index: I32
) -> ImageRead:
    """Read [Q,2] sources and int32 [Q] products using bounded serial chunks.

    Every array leaf gains Q; static metadata remains unbatched. Q=0 allocates
    empty leaves from an abstract shape trace without executing a read. Additional
    outer vmap dimensions require separate caller memory accounting.

    Args:
        reader: The reader to call for each query.
        source_xy_lod: Float64 source positions, shape (Q, 2), in lambda/D.
        product_index: Int32 product indices, shape (Q,).

    Returns:
        An ``ImageRead`` whose array leaves have a leading axis of length Q.
    """
    _require_precision()
    queries, products = jnp.asarray(source_xy_lod), jnp.asarray(product_index)
    if queries.ndim != 2:
        raise ValueError("Batch coordinates require [query, 2]")
    count = queries.shape[0]
    check_array(queries, (count, 2), (jnp.float64,))
    check_array(products, (count,), (jnp.int32,))
    gridded = isinstance(reader, AbstractGridReader)
    tiling = reader.tiling if gridded else Tiling()
    estimates = (
        reader.memory_estimate(count)
        if gridded
        else estimate_memory(0, reader.products.pixels.shape, tiling, count)
    )
    reader.memory.check(estimates)
    if count == 0:
        if gridded:
            ledger = reader.errors.evaluate(
                reader._safe_query, jnp.int32(0), jnp.ones(2), grid=reader.grid
            )
            shape = ImageRead(
                jnp.empty(reader.products.pixels.shape, dtype=jnp.float64),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.int32(0),
                jnp.uint32(0),
                jnp.asarray(False),
                jnp.int32(0),
                jnp.asarray(jnp.inf),
                reader.grid._empirical_max_abs,
                ledger,
            )
        else:
            # Future nominal readers determine their own ledger row count.
            shape = eqx.filter_eval_shape(
                reader.read,
                jax.ShapeDtypeStruct((2,), jnp.float64),
                jax.ShapeDtypeStruct((), jnp.int32),
            )
        return jax.tree_util.tree_map(
            lambda leaf: jnp.empty((0, *leaf.shape), leaf.dtype), shape
        )
    chunk = min(tiling.query_chunk, count)
    padded = ((count + chunk - 1) // chunk) * chunk
    queries = jnp.pad(queries, ((0, padded - count), (0, 0)))
    products = jnp.pad(products, ((0, padded - count),))

    def one_chunk(values):
        return jax.lax.map(lambda pair: reader.read(*pair), values)

    result = jax.lax.map(
        one_chunk, (queries.reshape(-1, chunk, 2), products.reshape(-1, chunk))
    )
    return jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((padded, *leaf.shape[2:]))[:count], result
    )


def require_read(result: ImageRead) -> ImageRead:
    """Return an eager result or raise with every refused flattened query index.

    Args:
        result: A concrete (not traced) read result, scalar or batched.

    Returns:
        ``result``, unchanged, when every query was accepted.
    """
    accepted = result.accepted.reshape(-1).tolist()
    statuses = result.status.reshape(-1).tolist()
    reasons = result.reason_bits.reshape(-1).tolist()
    indices = [index for index, ok in enumerate(accepted) if not ok]
    if indices:
        raise ReadRefused(
            indices, [statuses[i] for i in indices], [reasons[i] for i in indices]
        )
    return result


def make_reader(grid, domain, *, profile, policy=None, tiling=None, memory):
    """Construct the reader for the profile selected when the grid was admitted.

    Args:
        grid: The admitted sampled grid to read.
        domain: Region of source positions the reader accepts.
        profile: Kernel profile; a prefiltered profile selects the prefiltered
            reader, any other the raw interpolator.
        policy: Read policy. None means the default ``ReadPolicy``.
        tiling: Query chunk and pixel tile sizes. None means the default
            ``Tiling``.
        memory: Byte ceilings the reader checks its estimates against.

    Returns:
        The reader for the profile.
    """
    reader = (
        PrefilteredReader
        if isinstance(profile, PrefilteredKernelProfile)
        else Interpolator
    )
    return reader(
        grid,
        domain,
        profile,
        ReadPolicy() if policy is None else policy,
        Tiling() if tiling is None else tiling,
        memory,
    )
