"""Extended-scene rendering from a fixed set of prefiltered coefficient images."""

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.precision import like, require_float64

from yippy.sinc._types import FLOAT_DTYPES, I32, RealArray, check_array, check_static
from yippy.sinc.domain import covered_start
from yippy.sinc.evidence import BoundKind, MemoryLimit, Reason, Status
from yippy.sinc.geometry import array_namespace, raster_axes
from yippy.sinc.reader import ImageRead, PrefilteredReader

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

    Attributes:
        reader: Prefiltered reader whose policy, domain and coverage apply.
        table: Coefficient images of the stored touched nodes, shape
            (rows, pixels), in the storage dtype and unflipped.
        lookup: Row of ``table`` for each signed touched node, shape
            (rows_y, columns_x); the row count marks an absent node.
        nodes: Touched signed lattice nodes relative to the touched rectangle,
            shape (n_nodes, 2) as (row, column).
        product_index: Index of the one product this renderer serves.
        margin_nodes: Lattice nodes added around the touched set.
        accumulate: Contraction arithmetic, "float64" or "storage".
        scale: Constant multiplied into every rendered image.
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
        accumulate=None,
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
                (accumulation in the table's storage dtype). None uses float64
                with a float64 reader and storage precision otherwise.
            scale: Constant that ``render`` multiplies into the image, for
                example a photon-unit conversion.
        """
        if not isinstance(reader, PrefilteredReader):
            raise TypeError("SceneRenderer needs a PrefilteredReader")
        count = reader.grid.intensity.shape[0]
        if type(product_index) is not int or not 0 <= product_index < count:
            raise ValueError("product_index must be an available Python int")
        if type(margin_nodes) is not int or margin_nodes < 0:
            raise ValueError("margin_nodes must be a nonnegative integer")
        if accumulate is None:
            accumulate = (
                "float64" if reader._safe_query.dtype == jnp.float64 else "storage"
            )
        if accumulate not in ("float64", "storage"):
            raise ValueError("accumulate must be 'float64' or 'storage'")
        if accumulate == "float64":
            require_float64("yippy.sinc scene float64 accumulation", reader._safe_query)
        self.accumulate = accumulate
        self.scale = float(scale)
        positions = reader._prepare(source_xy_lod)
        if positions.phase.ndim != 2 or positions.phase.shape[0] == 0:
            raise ValueError("Scene positions require a nonempty [source, 2] array")
        self.reader, self.product_index = reader, product_index
        self.margin_nodes = margin_nodes
        readable = positions.finite & positions.member & positions.covered
        if not bool(jnp.all(readable)):
            raise ValueError("Scene positions must be finite, permitted and covered")
        taps = jax.vmap(reader._stencil_at)(positions.phase, positions.start).nodes
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

        Args:
            reader: A PrefilteredReader; its policy, domain and coverage apply.
            product_index: Python int, the one product this renderer serves.
            x_lod: Raster pixel-center x coordinates, shape (W,), in lambda/D.
            y_lod: Raster pixel-center y coordinates, shape (H,), in lambda/D.
            scale: Constant that ``render`` multiplies into the image.

        Returns:
            A SceneRenderer covering the readable raster centers.
        """
        xp = array_namespace(x_lod, y_lod)
        x_lod, y_lod = xp.asarray(x_lod), xp.asarray(y_lod)
        if x_lod.ndim != 1 or y_lod.ndim != 1 or not x_lod.size or not y_lod.size:
            raise ValueError("Raster axes must be nonempty coordinate vectors")
        x, y = xp.meshgrid(x_lod, y_lod, indexing="xy")
        centers = xp.stack((x.ravel(), y.ravel()), axis=1)
        prepared = reader._prepare(centers)
        positions = centers[prepared.finite & prepared.covered & prepared.member]
        if not positions.shape[0]:
            positions = np.asarray(reader._safe_position)[None, :]
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
            return jnp.zeros(
                jnp.broadcast_shapes(node_x.shape, node_y.shape), jnp.int32
            )
        return (node_x < 0).astype(jnp.int32) + 2 * (node_y < 0).astype(jnp.int32)

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

    def _axis_stencil(self, values, axis):
        """Compute one stencil per coordinate along a raster axis."""
        reader = self.reader
        xp = array_namespace(values)
        other = xp.broadcast_to(reader._safe_position[1 - axis], values.shape)
        points = xp.stack((other, values) if axis else (values, other), axis=-1)
        prepared = reader._prepare(points)
        covered = covered_start(
            prepared.start[:, axis : axis + 1],
            (reader.grid.source.shape[1 - axis],),
            reader.grid.reflection.kind != "none",
            reader.profile,
        )
        ok = (prepared.finite & covered)[:, None]
        safe_phase, safe_start = reader._safe_lattice()
        stencil = jax.vmap(reader._stencil_at)(
            jnp.where(ok, prepared.phase, safe_phase),
            jnp.where(ok, prepared.start, safe_start),
        )
        return (
            stencil.nodes[:, axis],
            stencil.weights[:, axis],
            prepared.finite,
            covered,
        )

    def _contract(self, amplitudes):
        """Float64 partial images [groups, pixels] of amplitudes [groups, rows]."""
        table = self.table
        # Full precision keeps accelerator backends from a reduced-precision product.
        highest = jax.lax.Precision.HIGHEST
        if table.dtype == jnp.float64 or self.accumulate == "storage":
            product = jnp.matmul(
                amplitudes.astype(table.dtype), table, precision=highest
            )
            return like(product, amplitudes)
        pixels = table.shape[1]
        width = min(CONTRACT_COLUMNS, pixels)
        blocks = pixels // width

        def block(index, partial):
            columns = jax.lax.dynamic_slice_in_dim(table, index * width, width, 1)
            value = jnp.matmul(
                amplitudes, columns.astype(amplitudes.dtype), precision=highest
            )
            return jax.lax.dynamic_update_slice_in_dim(partial, value, index * width, 1)

        partial = jax.lax.fori_loop(
            0, blocks, block, jnp.zeros((amplitudes.shape[0], pixels))
        )
        if blocks * width < pixels:
            tail = table[:, blocks * width :].astype(amplitudes.dtype)
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
            like(jnp.inf, image),
            like(jnp.nan, image),
            ledger,
        )

    def render_points(self, source_xy_lod: RealArray, flux: RealArray) -> ImageRead:
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

        Args:
            source_xy_lod: Source coordinates, shape (Q, 2), in lambda/D.
            flux: Source fluxes, shape (Q,), as a multiplier on the delivered
                per-unit-flux read.

        Returns:
            ImageRead with the summed image in delivered units and the
            acceptance diagnostics.
        """
        reader, taps = self.reader, self.reader.profile.taps
        queries = reader._prepare(source_xy_lod)
        flux = like(flux, reader._safe_query)
        if queries.phase.ndim != 2:
            raise ValueError("Scene coordinates require [source, 2]")
        count = queries.phase.shape[0]
        check_array(flux, (count,), FLOAT_DTYPES)
        rows = self.table.shape[0]
        groups = self._groups
        self._scratch(count * taps * taps * 40 + groups * (rows + 1) * 8)
        finite = queries.finite & jnp.isfinite(flux)
        member, covered = queries.member, queries.covered
        readable = finite & member & covered
        safe_phase, safe_start = reader._safe_lattice()
        stencil = jax.vmap(reader._stencil_at)(
            jnp.where(readable[:, None], queries.phase, safe_phase),
            jnp.where(readable[:, None], queries.start, safe_start),
        )
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
        amount = jnp.where(usable, flux, 0.0)
        spread = amount[:, None, None] * weight[:, 1, :, None] * weight[:, 0, None, :]
        amplitudes = (
            jnp.zeros((groups, rows + 1), dtype=flux.dtype)
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

        Returns:
            Tuple of the image (detector shape, in delivered units times
            ``scale``) and the boolean readable-cell mask, shape (H, W).
        """
        x, y = raster_axes(grid)
        flux = like(flux, self.reader._safe_query)
        fold = self._fold_raster(flux, x, y)
        image = self._image(fold.amplitudes) * self.scale
        return image, fold.readable & (self.reader.policy.mode == "sampling")

    def render_raster(
        self, scene: RealArray, x_lod: RealArray, y_lod: RealArray
    ) -> ImageRead:
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

        Args:
            scene: Raster of per-pixel source fluxes, shape (H, W).
            x_lod: Pixel-center x coordinates, shape (W,), in lambda/D.
            y_lod: Pixel-center y coordinates, shape (H,), in lambda/D.

        Returns:
            ImageRead with the rendered image in delivered units and the
            acceptance diagnostics.
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
        covered and every tap lands on a stored node. Every readable, finite
        cell in a sampling policy contributes its linear flux response.
        """
        ref = self.reader._safe_query
        xp = array_namespace(x_lod, y_lod)
        scene, x, y = like(scene, ref), xp.asarray(x_lod), xp.asarray(y_lod)
        if scene.ndim != 2:
            raise ValueError("Raster scenes require two axes")
        height, width = scene.shape
        check_array(scene, (height, width), FLOAT_DTYPES)
        if x.shape != (width,) or y.shape != (height,):
            raise ValueError("Raster axes must match the scene width and height")
        reader, rows = self.reader, self.table.shape[0]
        span_y, span_x = self.lookup.shape
        self._scratch(
            16 * height * width
            + 16 * (height * (span_y + 1) + width * (span_x + 1))
            + 16 * (span_y + 1) * (span_x + 1)
            + 8 * self._groups * rows
        )
        centers = xp.stack(xp.meshgrid(x, y, indexing="xy"), axis=-1)
        member = reader.domain.contains(centers)
        stencil_x = self._axis_stencil(x, 0)
        stencil_y = self._axis_stencil(y, 1)
        finite_x, covered_x = stencil_x[2:]
        finite_y, covered_y = stencil_y[2:]
        finite = jnp.isfinite(scene) & finite_y[:, None] & finite_x[None, :]
        covered = covered_y[:, None] & covered_x[None, :]
        lit = scene != 0
        spreading, columns = [], []
        for axis, ((node, weight, _, _), span) in enumerate(
            ((stencil_x, span_x), (stencil_y, span_y))
        ):
            index = node - self._corner_xy[axis]
            column = jnp.where((index >= 0) & (index < span), index, span)
            matrix = jnp.zeros((node.shape[0], span + 1), dtype=weight.dtype)
            matrix = matrix.at[jnp.arange(node.shape[0])[:, None], column].add(weight)
            spreading.append(matrix)
            columns.append((column, weight != 0))
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
        amount = jnp.where(usable & sampling, scene, 0.0)
        across, down = spreading
        folded = down.T @ amount @ across
        index_y, index_x = self.nodes[:, 0], self.nodes[:, 1]
        quadrant = self._quadrant(
            index_x + self._corner_xy[0], index_y + self._corner_xy[1]
        )
        amplitudes = (
            jnp.zeros((self._groups, rows), dtype=scene.dtype)
            .at[quadrant, self.lookup[index_y, index_x]]
            .add(folded[index_y, index_x])
        )
        return _Fold(amplitudes, lit, finite, usable, readable, member, covered, stored)


class _Fold(NamedTuple):
    """Node amplitudes of a raster fold and the per-cell flags that gate it.

    Attributes:
        amplitudes: Folded node amplitudes, one row per sign group.
        lit: Cells with nonzero flux.
        finite: Cells with finite coordinates.
        usable: Cells that are finite, in the domain, covered and stored.
        readable: Cells with finite coordinates that are also in the domain,
            covered and stored.
        member: Cells inside the permission domain.
        covered: Cells whose stencil is covered.
        stored: Cells whose taps all land on stored nodes.
    """

    amplitudes: jax.Array
    lit: jax.Array
    finite: jax.Array
    usable: jax.Array
    readable: jax.Array
    member: jax.Array
    covered: jax.Array
    stored: jax.Array
