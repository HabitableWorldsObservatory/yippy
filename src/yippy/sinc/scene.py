"""Extended-scene rendering from a fixed set of prefiltered coefficient images."""

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from yippy.sinc._types import F64, I32, check_array, check_static
from yippy.sinc.domain import coverage
from yippy.sinc.evidence import BoundKind, MemoryLimit, Reason, Status
from yippy.sinc.kernel import flush_subnormal
from yippy.sinc.reader import ImageRead, PrefilteredReader, _require_precision

# Detector columns widened to binary64 at a time when a narrower table is
# contracted with binary64 accumulation; the whole table is never widened.
CONTRACT_COLUMNS = 1024


class AbstractSceneOperator(eqx.Module):
    """A linear map from native source-pixel flux to a detector image.

    A scene coronagraph holds a tuple of operators and adds their images, which
    is exact because rendering is linear in the flux. Each operator renders the
    source cells it covers and reports which they are; a lit cell it does not
    cover contributes zero and is neither flagged nor an error here. The scene
    is valid when every lit cell is covered by some operator, so operators over
    disjoint masks compose into one valid render.
    ``yippy.BlendSceneOperator`` renders from stored PSF nodes and
    ``SceneRenderer`` from prefiltered coefficient images.
    """

    @abstractmethod
    def render(self, flux, grid):
        """Return the image and the source cells this operator covers.

        Args:
            flux: Source flux [sy, sx] on the source grid ``grid``, one value per
                pixel. A nonfinite value counts as zero.
            grid: The source grid of ``flux``.

        Returns:
            ``(image, covered)``: the finite native detector array, linear in
            ``flux``, and a boolean [sy, sx] mask of the cells the operator
            renders. The image holds no light from a cell outside the mask.
        """


class SceneRenderer(AbstractSceneOperator):
    """Render many sources at once as one contraction of coefficient images.

    A prefiltered read is linear in the coefficient images, so the image of a
    scene, the flux-weighted sum of the reads of its sources, can be regrouped
    by lattice node: every source spreads its flux onto its taps with the
    read's own weights, and the folded node amplitudes multiply the coefficient
    images once. The result equals the sum of the individual reads up to
    floating-point rounding, at the cost of one pass over the touched nodes.

    The touched nodes are fixed at construction, so every shape is static under
    jit. ``nodes`` lists them as signed lattice nodes relative to the touched
    rectangle. ``table`` holds the coefficient images [rows, pixels] of the
    stored nodes they map to, in the storage dtype and unflipped, and
    ``lookup`` maps a signed node to its row (the row count marks an absent
    node). On a mirror-symmetric grid a signed node and its mirrors share one
    stored row, so the table holds a quarter of the touched nodes: amplitudes
    fold per sign quadrant, one product contracts the four quadrants with the
    table, and the four partial images are flipped and added.

    ``accumulate`` sets the contraction arithmetic. ``"float64"`` accumulates
    in binary64 whatever the storage dtype, as a single read does; a narrower
    table is widened a block of detector columns at a time. ``"storage"``
    rounds the amplitudes to the storage dtype and accumulates in it, about
    twice as fast for a float32 table at the cost of float32 accumulation
    (relative rounding of order 1e-7 times the square root of the node count).

    Rendering refuses the whole image when any contributing source cannot be
    read. A source contributes when its flux, or its raster value, is nonzero;
    zero-flux entries are padding and may hold any position.
    """

    reader: PrefilteredReader
    table: jax.Array
    lookup: I32
    nodes: I32
    product_index: int = eqx.field(static=True)
    margin_nodes: int = eqx.field(static=True)
    accumulate: str = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    _corner_xy: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        reader,
        product_index,
        source_xy_lod,
        *,
        margin_nodes=1,
        accumulate="float64",
        scale=1.0,
    ):
        """Gather the coefficient images that reads near the given positions touch.

        Args:
            reader: A PrefilteredReader; its policy, domain and coverage apply.
            product_index: Python int, the one product this renderer serves.
            source_xy_lod: Host float64 positions [Q, 2] whose stencils define the
                touched nodes: the sources themselves, or the centers of the
                raster pixels that can be lit. Every position must be readable.
            margin_nodes: Lattice nodes added around the touched set in every
                direction, so positions may move by that many node spacings (in
                a fit, for example) before a render refuses.
            accumulate: ``"float64"`` (binary64 accumulation) or ``"storage"``
                (accumulation in the table's storage dtype).
            scale: Constant that ``render`` multiplies into the image, for
                example a photon-unit conversion.
        """
        _require_precision()
        if not isinstance(reader, PrefilteredReader):
            raise TypeError("SceneRenderer needs a PrefilteredReader")
        count = reader.grid.intensity.shape[0]
        if type(product_index) is not int or not 0 <= product_index < count:
            raise ValueError("product_index must be an available Python int")
        if type(margin_nodes) is not int or margin_nodes < 0:
            raise ValueError("margin_nodes must be a nonnegative integer")
        if accumulate not in ("float64", "storage"):
            raise ValueError("accumulate must be 'float64' or 'storage'")
        self.accumulate = accumulate
        self.scale = float(scale)
        positions = jnp.asarray(source_xy_lod)
        if positions.ndim != 2 or positions.shape[0] == 0:
            raise ValueError("Scene positions require a nonempty [source, 2] array")
        check_array(positions, (positions.shape[0], 2), (jnp.float64,))
        self.reader, self.product_index = reader, product_index
        self.margin_nodes = margin_nodes
        product = jnp.int32(product_index)
        readable = jax.vmap(lambda q: reader._validate(q, product)[1])(positions)
        if not bool(jnp.all(readable)):
            raise ValueError("Scene positions must be finite, permitted and covered")
        taps = jax.vmap(lambda q: reader.stencil(flush_subnormal(q)).nodes)(positions)
        # Nodes that some covered read can use, as signed (low, high) per axis.
        size = reader.grid.source.shape[::-1]
        halo = reader.profile.halo
        reflected = reader.grid.reflection.kind != "none"
        usable = [(halo - (n - 1) if reflected else halo, n - 1 - halo) for n in size]
        low = [
            max(int(jnp.min(taps[:, axis])) - margin_nodes, usable[axis][0])
            for axis in range(2)
        ]
        high = [
            min(int(jnp.max(taps[:, axis])) + margin_nodes, usable[axis][1])
            for axis in range(2)
        ]
        span_x, span_y = (high[axis] - low[axis] + 1 for axis in range(2))
        touched = jnp.zeros((span_y, span_x), dtype=jnp.bool_)
        touched = touched.at[
            (taps[:, 1, :, None] - low[1]), (taps[:, 0, None, :] - low[0])
        ].set(True)
        for _ in range(margin_nodes):
            padded = jnp.pad(touched, 1)
            touched = jnp.zeros_like(touched)
            for shift_y in range(3):
                for shift_x in range(3):
                    touched = (
                        touched
                        | padded[shift_y : shift_y + span_y, shift_x : shift_x + span_x]
                    )
        self.nodes = jnp.argwhere(touched).astype(jnp.int32)
        self._corner_xy = (low[0], low[1])
        # Signed nodes and their mirrors share one stored row.
        signed = np.asarray(self.nodes) + np.array([low[1], low[0]])
        stored, inverse = np.unique(np.abs(signed), axis=0, return_inverse=True)
        rows = stored.shape[0]
        self.lookup = (
            jnp.full((span_y, span_x), rows, dtype=jnp.int32)
            .at[self.nodes[:, 0], self.nodes[:, 1]]
            .set(jnp.asarray(inverse.reshape(-1), dtype=jnp.int32))
        )
        coefficients = reader.coefficients
        pixels = coefficients.shape[3] * coefficients.shape[4]
        estimate = reader.memory_estimate(1)
        table_bytes = rows * pixels * coefficients.dtype.itemsize
        if (
            estimate["resident"] + table_bytes > reader.memory.max_resident_bytes
            or estimate["total"] + table_bytes > reader.memory.max_total_bytes
        ):
            raise MemoryLimit(
                f"scene table of {table_bytes} bytes exceeds the resident budget"
            )
        self.table = coefficients[product_index][stored[:, 0], stored[:, 1]].reshape(
            rows, pixels
        )

    @classmethod
    def from_raster(cls, reader, product_index, x_lod, y_lod, *, scale=1.0):
        """Gather the covered, permitted centers of a fixed raster.

        Unsupported centers may stay dark during a render. If no center is
        supported, one covered stencil suffices to represent the all-dark image.
        """
        x_lod, y_lod = jnp.asarray(x_lod), jnp.asarray(y_lod)
        if x_lod.ndim != 1 or y_lod.ndim != 1 or not x_lod.size or not y_lod.size:
            raise ValueError("Raster axes must be nonempty coordinate vectors")
        x, y = jnp.meshgrid(x_lod, y_lod, indexing="xy")
        centers = jnp.stack((x.ravel(), y.ravel()), axis=1)
        readable = jax.vmap(lambda q: reader._validate(q, product_index)[1])(centers)
        positions = centers[readable]
        if not positions.shape[0]:
            positions = reader._safe_query[None, :]
        return cls(reader, product_index, positions, margin_nodes=0, scale=scale)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.nodes.shape[0]
        check_array(self.nodes, (count, 2), (jnp.int32,))
        check_array(self.lookup, self.lookup.shape, (jnp.int32,))
        if self.table.ndim != 2 or not 0 < self.table.shape[0] <= count:
            raise ValueError("Scene table requires one row per stored touched node")
        if self.accumulate not in ("float64", "storage"):
            raise ValueError("accumulate must be 'float64' or 'storage'")

    @property
    def _groups(self):
        # One amplitude row per sign quadrant on a mirror-symmetric grid.
        return 4 if self.reader.grid.reflection.kind != "none" else 1

    def _quadrant(self, node_x, node_y):
        """Sign quadrant of signed nodes: bit 0 for x < 0, bit 1 for y < 0."""
        if self._groups == 1:
            return jnp.zeros(jnp.broadcast_shapes(node_x.shape, node_y.shape), int)
        return (node_x < 0).astype(int) + 2 * (node_y < 0).astype(int)

    def _scratch(self, working_bytes):
        rows, pixels = self.table.shape
        # Partial images per quadrant, then the image.
        working_bytes += (self._groups + 1) * pixels * 8
        # A narrower table is widened one block of detector columns at a time.
        if self.table.dtype != jnp.float64 and self.accumulate == "float64":
            working_bytes += rows * min(CONTRACT_COLUMNS, pixels) * 8
        limit = self.reader.memory.max_scratch_bytes
        if working_bytes > limit:
            raise MemoryLimit(f"scene scratch {working_bytes} bytes exceeds budget")

    def _axis_ok(self, values, axis):
        """Sanitize coordinates of one axis and flag the finite, covered ones."""
        finite = jnp.isfinite(values)
        clean = flush_subnormal(jnp.where(finite, values, 0.0))
        safe = self.reader._safe_query
        reflected = self.reader.grid.reflection.kind != "none"
        covered = jax.vmap(
            lambda v: coverage(
                safe.at[axis].set(v),
                self.reader.grid.source,
                reflected,
                self.reader.profile,
            )
        )(clean)
        return clean, finite, covered

    def _contract(self, amplitudes):
        """Float64 partial images [groups, pixels] of amplitudes [groups, rows]."""
        table = self.table
        # Full precision keeps accelerator backends from a reduced-precision product.
        highest = jax.lax.Precision.HIGHEST
        if table.dtype == jnp.float64 or self.accumulate == "storage":
            product = jnp.matmul(
                amplitudes.astype(table.dtype), table, precision=highest
            )
            return product.astype(jnp.float64)
        pixels = table.shape[1]
        width = min(CONTRACT_COLUMNS, pixels)
        blocks = pixels // width

        def block(index, partial):
            columns = jax.lax.dynamic_slice_in_dim(table, index * width, width, 1)
            value = jnp.matmul(
                amplitudes, columns.astype(jnp.float64), precision=highest
            )
            return jax.lax.dynamic_update_slice_in_dim(partial, value, index * width, 1)

        partial = jax.lax.fori_loop(
            0, blocks, block, jnp.zeros((amplitudes.shape[0], pixels))
        )
        if blocks * width < pixels:
            tail = table[:, blocks * width :].astype(jnp.float64)
            partial = partial.at[:, blocks * width :].set(
                jnp.matmul(amplitudes, tail, precision=highest)
            )
        return partial

    def _image(self, amplitudes):
        """Finite native image [ny, nx] of node amplitudes [groups, rows]."""
        partial = self._contract(amplitudes).reshape(
            amplitudes.shape[0], *self.reader.products.pixels.shape
        )
        image = partial[0]
        if self._groups == 4:
            # A node with x < 0 reads its mirror's image flipped across columns,
            # one with y < 0 flipped across rows.
            image = (
                image
                + partial[1][:, ::-1]
                + partial[2][::-1, :]
                + partial[3][::-1, ::-1]
            )
        return image

    def _finish(self, amplitudes, accepted, sampling_ok, failures):
        """Contract node amplitudes with the table and attach read diagnostics."""
        reader = self.reader
        image = self._image(amplitudes)
        reason = jnp.asarray(Reason.BOUND_UNKNOWN, dtype=jnp.uint32)
        for failed, bit in failures:
            reason = reason | jnp.where(failed, jnp.uint32(bit), jnp.uint32(0))
        product = jnp.int32(self.product_index)
        ledger = reader.errors.evaluate(
            reader._safe_query, product, reader._gain_xy, grid=reader.grid
        )
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
            jnp.asarray(jnp.nan, dtype=jnp.float64),
            ledger,
        )

    def render_points(self, source_xy_lod: F64, flux: F64) -> ImageRead:
        """Render point sources [Q, 2] with fluxes [Q] as one image.

        The image is the sum over sources of flux times that source's read, in
        delivered units per unit flux. Node amplitudes are folded and contracted
        with the coefficient images; with ``accumulate="float64"`` the image
        equals the summed reads to binary64 rounding. The error bound
        is unknown, as for a single read, and a point-read empirical check does
        not transfer, so ``empirical_max_abs``
        is NaN.

        Any contributing source that is non-finite, outside the domain, not
        covered or outside the touched nodes refuses the whole image. Reads
        jump by the bump's edge value where a stencil switches cell, so
        position gradients are piecewise.
        """
        _require_precision()
        queries, flux = jnp.asarray(source_xy_lod), jnp.asarray(flux)
        if queries.ndim != 2:
            raise ValueError("Scene coordinates require [source, 2]")
        count = queries.shape[0]
        check_array(queries, (count, 2), (jnp.float64,))
        check_array(flux, (count,), (jnp.float64,))
        reader, taps = self.reader, self.reader.profile.taps
        rows = self.table.shape[0]
        groups = self._groups
        self._scratch(count * taps * taps * 40 + groups * (rows + 1) * 8)
        finite = jnp.all(jnp.isfinite(queries), axis=1) & jnp.isfinite(flux)
        clean = flush_subnormal(jnp.where(jnp.isfinite(queries), queries, 0.0))
        member = jax.vmap(reader.domain.contains)(clean)
        reflected = reader.grid.reflection.kind != "none"
        covered = jax.vmap(
            lambda q: coverage(q, reader.grid.source, reflected, reader.profile)
        )(clean)
        readable = finite & member & covered
        safe = jnp.where(readable[:, None], clean, reader._safe_query)
        stencil = jax.vmap(reader.stencil)(safe)
        node, weight = stencil.nodes, stencil.weights
        index_x = node[:, 0, None, :] - self._corner_xy[0]
        index_y = node[:, 1, :, None] - self._corner_xy[1]
        span_y, span_x = self.lookup.shape
        inside = (
            (index_x >= 0) & (index_x < span_x) & (index_y >= 0) & (index_y < span_y)
        )
        row = self.lookup[
            jnp.clip(index_y, 0, span_y - 1), jnp.clip(index_x, 0, span_x - 1)
        ]
        row = jnp.where(inside, row, rows)
        quadrant = self._quadrant(node[:, 0, None, :], node[:, 1, :, None])
        stored = jnp.all(row < rows, axis=(1, 2))
        contributing = flux != 0
        usable = readable & stored
        sampling_ok = jnp.all(usable | ~contributing)
        accepted = sampling_ok & (reader.policy.mode == "sampling")
        amount = jnp.where(usable & contributing, flux, 0.0)
        spread = amount[:, None, None] * weight[:, 1, :, None] * weight[:, 0, None, :]
        amplitudes = (
            jnp.zeros((groups, rows + 1))
            .at[quadrant.reshape(-1), row.reshape(-1)]
            .add(spread.reshape(-1))
        )
        return self._finish(
            amplitudes[:, :rows],
            accepted,
            sampling_ok,
            (
                (jnp.any(contributing & ~finite), Reason.NONFINITE_QUERY),
                (jnp.any(contributing & finite & ~member), Reason.OUTSIDE_DOMAIN),
                (
                    # ``stored`` is meaningful only for a readable source.
                    jnp.any(contributing & finite & (~covered | (readable & ~stored))),
                    Reason.INCOMPLETE_STENCIL,
                ),
            ),
        )

    def render(self, flux, grid):
        """Render the flux raster of a source grid, scaled by ``scale``.

        Returns the finite image and the mask of cells that are readable: finite
        coordinates, inside the domain, covered, with every tap stored. A lit
        cell outside the mask adds nothing. A bounded policy never reads a PSF:
        its image is exactly zero and its mask empty, so only an all-dark raster
        is valid.

        Args:
            flux: Source flux [H, W] at the pixel centers of ``grid``.
            grid: The source grid whose cell centers locate the sources.
        """
        x, y = grid.coordinates()
        flux = jnp.asarray(flux, jnp.float64)
        fold = self._fold_raster(flux, x[0], y[:, 0])
        image = self._image(fold.amplitudes) * self.scale
        return image, fold.readable & (self.reader.policy.mode == "sampling")

    def render_raster(self, scene: F64, x_lod: F64, y_lod: F64) -> ImageRead:
        """Render a raster scene [H, W] whose pixel centers are x_lod [W], y_lod [H].

        Every raster pixel is a point source at its center with the pixel's
        value as flux, so the image equals ``render_points`` on the flattened
        raster. The fold is separable: with spreading matrices W_y and W_x that
        hold each center's tap weights, the node amplitudes are W_y^T S W_x.

        Any nonzero pixel that is non-finite, outside the domain, not covered
        or outside the touched nodes refuses the whole image. A center within
        rounding of a cell's half point may take the neighboring stencil here
        and not in a point read, depending on how the backend compiles each
        path; the image then differs by the bump's edge value, the size of a
        stencil switch.
        """
        fold = self._fold_raster(scene, x_lod, y_lod)
        lit, finite, usable = fold.lit, fold.finite, fold.usable
        sampling_ok = jnp.all(usable | ~lit)
        accepted = sampling_ok & (self.reader.policy.mode == "sampling")
        return self._finish(
            fold.amplitudes,
            accepted,
            sampling_ok,
            (
                (jnp.any(lit & ~finite), Reason.NONFINITE_QUERY),
                (jnp.any(lit & finite & ~fold.member), Reason.OUTSIDE_DOMAIN),
                (
                    jnp.any(lit & finite & ~fold.covered)
                    | jnp.any(lit & finite & fold.covered & fold.member & ~fold.stored),
                    Reason.INCOMPLETE_STENCIL,
                ),
            ),
        )

    def _fold_raster(self, scene, x_lod, y_lod):
        """Fold a raster onto the node amplitudes and flag each cell's readability.

        A cell is readable when its coordinates are finite, inside the domain,
        covered and every tap lands on a stored node. Only readable, finite,
        lit cells in a sampling policy reach the amplitudes.
        """
        _require_precision()
        scene, x, y = jnp.asarray(scene), jnp.asarray(x_lod), jnp.asarray(y_lod)
        if scene.ndim != 2:
            raise ValueError("Raster scenes require two axes")
        height, width = scene.shape
        check_array(scene, (height, width), (jnp.float64,))
        check_array(x, (width,), (jnp.float64,))
        check_array(y, (height,), (jnp.float64,))
        reader, rows = self.reader, self.table.shape[0]
        span_y, span_x = self.lookup.shape
        self._scratch(
            16 * height * width
            + 16 * (height * (span_y + 1) + width * (span_x + 1))
            + 16 * (span_y + 1) * (span_x + 1)
            + 8 * self._groups * rows
        )
        clean_x, finite_x, covered_x = self._axis_ok(x, 0)
        clean_y, finite_y, covered_y = self._axis_ok(y, 1)
        member = jax.vmap(
            lambda v: jax.vmap(lambda u: reader.domain.contains(jnp.stack([u, v])))(
                clean_x
            )
        )(clean_y)
        finite = jnp.isfinite(scene) & finite_y[:, None] & finite_x[None, :]
        covered = covered_y[:, None] & covered_x[None, :]
        lit = scene != 0
        safe = reader._safe_query
        spreading, columns = [], []
        for axis, (clean, ok, span) in enumerate(
            (
                (clean_x, finite_x & covered_x, span_x),
                (clean_y, finite_y & covered_y, span_y),
            )
        ):
            values = jnp.where(ok, clean, safe[axis])
            stencil = jax.vmap(lambda v, a=axis: reader.stencil(safe.at[a].set(v)))(
                values
            )
            node, weight = stencil.nodes, stencil.weights
            index = node[:, axis, :] - self._corner_xy[axis]
            column = jnp.where((index >= 0) & (index < span), index, span)
            matrix = jnp.zeros((values.shape[0], span + 1))
            matrix = matrix.at[jnp.arange(values.shape[0])[:, None], column].add(
                weight[:, axis, :]
            )
            spreading.append(matrix)
            columns.append((column, weight[:, axis, :] != 0))
        # A cell is stored when every tap it weights is a stored node.
        present = jnp.pad(self.lookup < rows, ((0, 1), (0, 1)))
        (col_x, used_x), (col_y, used_y) = columns
        stored = jnp.ones((height, width), dtype=jnp.bool_)
        for ty in range(col_y.shape[1]):
            for tx in range(col_x.shape[1]):
                weighted = used_y[:, ty, None] & used_x[None, :, tx]
                node_ok = present[col_y[:, ty, None], col_x[None, :, tx]]
                stored = stored & (~weighted | node_ok)
        readable = finite_y[:, None] & finite_x[None, :] & member & covered & stored
        usable = finite & member & covered & stored
        sampling = reader.policy.mode == "sampling"
        amount = jnp.where(usable & lit & sampling, scene, 0.0)
        across, down = spreading
        folded = down.T @ amount @ across
        index_y, index_x = self.nodes[:, 0], self.nodes[:, 1]
        quadrant = self._quadrant(
            index_x + self._corner_xy[0], index_y + self._corner_xy[1]
        )
        amplitudes = (
            jnp.zeros((self._groups, rows))
            .at[quadrant, self.lookup[index_y, index_x]]
            .add(folded[index_y, index_x])
        )
        return _Fold(amplitudes, lit, finite, usable, readable, member, covered, stored)


class _Fold(NamedTuple):
    """Node amplitudes of a raster fold and the per-cell flags that gate it."""

    amplitudes: jax.Array
    lit: jax.Array
    finite: jax.Array
    usable: jax.Array
    readable: jax.Array
    member: jax.Array
    covered: jax.Array
    stored: jax.Array
