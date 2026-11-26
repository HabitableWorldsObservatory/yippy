"""Module for handling off-axis PSFs using JAX."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import HostGrid, is_traced
from hwoutils.precision import float_dtype, like
from jax import device_put, shard_map, vmap
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from optixstuff.grids import Image, Measure, PixelGrid
from scipy.spatial import ConvexHull, QhullError

from ._jax_funcs import (
    blend_pad,
    clamp_to_radius,
    synthesize_psf_idw,
    synthesize_psf_separable,
)
from ._offax import OffAx
from .sampling import fraction_image

# A source this close beyond the hull (relative to its radius) is on it: the
# radii of a single-precision table carry rounding of about 1e-7.
_DOMAIN_TOL = 1e-6


def _hull_halfplanes(offsets, x_symmetric, y_symmetric):
    """Half-planes ``normal . p <= distance`` of the hull of the node positions.

    A table stored for one quadrant is completed by its mirror images first.
    Returns ``None`` when the nodes do not span an area (a radially symmetric
    table tabulates one axis), where the domain is the disk of the outermost
    node instead.
    """
    points = np.asarray(offsets, float)
    if x_symmetric:
        points = np.vstack([points, points * [-1.0, 1.0]])
    if y_symmetric:
        points = np.vstack([points, points * [1.0, -1.0]])
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    # Each row of equations is (normal_x, normal_y, offset) with n . p + c <= 0.
    return hull.equations[:, :2], -hull.equations[:, 2]


def _host_source_domain(x, y, radius, hull):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    slack = _DOMAIN_TOL * radius
    limit = radius + slack
    # Discard nonfinite and far-out coordinates before taking projections.
    candidates = (np.abs(x) <= limit) & (np.abs(y) <= limit)
    x, y = np.where(candidates, x, 0), np.where(candidates, y, 0)
    inside = candidates & (np.hypot(x, y) <= limit)
    if hull is None:
        return inside
    normal_x, normal_y, distance = (axis.values for axis in hull)
    reach = normal_x * x[..., None] + normal_y * y[..., None]
    return inside & np.all(reach <= distance + slack, axis=-1)


class OffJAX(OffAx, eqx.Module):
    """Class for handling off-axis PSFs using JAX.

    This class inherits from OffAx and uses JAX for optimized computation.
    Memory-efficient: stores PSFs in flat array with index mapping.

    Attributes:
        grid: Native pixel-center geometry in the package frame.
        flat_psfs: Stored PSF images indexed by source node.
        flat_offsets: Source coordinates of the stored PSFs, in lambda/D.
        flat_x_offsets: JAX array of x coordinates of the stored PSFs.
        flat_y_offsets: JAX array of y coordinates of the stored PSFs.
        x_offsets: Sorted unique x coordinates of the stored PSFs.
        y_offsets: Sorted unique y coordinates of the stored PSFs.
        x_inds: X grid index for each stored PSF.
        y_inds: Y grid index for each stored PSF.
        x_range: Minimum and maximum x coordinates.
        y_range: Minimum and maximum y coordinates.
        offset_to_flat_idx: Map from grid indices (x, y) to stored PSF index.
        x_symmetric: Whether reads reflect across x = 0.
        y_symmetric: Whether reads reflect across y = 0.
        n_psfs: Number of stored PSFs.
        psf_shape: Native detector shape.
        type: Layout of the source offsets.
        measure: Measure of the native samples.
        source_radius_lod: Radius of the outermost source node in lambda/D.
        source_extent_lod: Maximum absolute source offset on each axis in lambda/D.
        kx: Padded real FFT frequencies along x.
        ky: Padded FFT frequencies along y.
    """

    grid: PixelGrid
    flat_psfs: jax.Array
    flat_offsets: jax.Array
    flat_x_offsets: jax.Array
    flat_y_offsets: jax.Array
    x_offsets: jax.Array
    y_offsets: jax.Array
    x_inds: jax.Array
    y_inds: jax.Array
    x_range: jax.Array
    y_range: jax.Array
    offset_to_flat_idx: jax.Array
    x_symmetric: bool = eqx.field(static=True)
    y_symmetric: bool = eqx.field(static=True)
    n_psfs: int = eqx.field(static=True)
    psf_shape: tuple[int, int] = eqx.field(static=True)
    type: str = eqx.field(static=True)
    measure: Measure = eqx.field(static=True)
    source_radius_lod: float = eqx.field(static=True)
    source_extent_lod: tuple[float, float] = eqx.field(static=True)
    kx: jax.Array
    ky: jax.Array
    _hull: tuple[HostGrid, HostGrid, HostGrid] | None = eqx.field(static=True)

    def __init__(
        self,
        psfs,
        offsets,
        grid: PixelGrid,
        x_symmetric: bool,
        y_symmetric: bool,
        downsample_shape: tuple[int, int] | None = None,
        *,
        sample_kind="point",
    ) -> None:
        """Prepare JAX interpolation arrays from the native samples.

        Args:
            psfs: Shipped off-axis PSF samples, one native image per source
                offset.
            offsets: Source offsets (x, y) in lambda/D, one row per PSF.
            grid: Pixel-center geometry of the native images in the package
                reference frame.
            x_symmetric: Whether the response is mirror symmetric about the x axis.
            y_symmetric: Whether the response is mirror symmetric about the y axis.
            downsample_shape: Optional (ny, nx) target shape for the stored PSFs.
            sample_kind: How stored pixel values sample the image: ``"point"``,
                ``"integral"`` or ``"mean"``.
        """
        super().__init__(
            psfs,
            offsets,
            grid,
            x_symmetric,
            y_symmetric,
            downsample_shape,
            sample_kind=sample_kind,
        )

        dtype = float_dtype()
        self.flat_psfs = device_put(jnp.asarray(self.flat_psfs, dtype))
        self.flat_x_offsets = device_put(jnp.asarray(self.flat_offsets[:, 0], dtype))
        self.flat_y_offsets = device_put(jnp.asarray(self.flat_offsets[:, 1], dtype))
        self.x_offsets = device_put(jnp.asarray(self.x_offsets, dtype))
        self.y_offsets = device_put(jnp.asarray(self.y_offsets, dtype))
        self.offset_to_flat_idx = device_put(jnp.asarray(self.offset_to_flat_idx))

        n_pixels = self.flat_psfs.shape[-1]
        n_fft = n_pixels + 2 * blend_pad(n_pixels)
        self.kx = device_put(jnp.fft.rfftfreq(n_fft, dtype=dtype))
        self.ky = device_put(jnp.fft.fftfreq(n_fft, dtype=dtype))

        # The source domain is the hull of the tabulated source positions: a
        # source beyond the outermost node has no data. The stamp is a smaller
        # square, and a node between the two holds its measured wing.
        offsets = np.asarray(self.flat_offsets, np.float64)
        self.source_radius_lod = float(np.max(np.hypot(offsets[:, 0], offsets[:, 1])))
        # Host floats, so the scene grid is sized without reading a device array.
        self.source_extent_lod = (
            float(np.max(np.abs(self.flat_offsets[:, 0]))),
            float(np.max(np.abs(self.flat_offsets[:, 1]))),
        )
        hull = (
            None
            if self.type == "1d"
            else _hull_halfplanes(offsets, self.x_symmetric, self.y_symmetric)
        )
        self._hull = (
            None
            if hull is None
            else tuple(HostGrid(axis) for axis in (*hull[0].T, hull[1]))
        )
        self.flat_offsets = device_put(jnp.asarray(self.flat_offsets, dtype))
        self.x_inds = device_put(jnp.asarray(self.x_inds, jnp.int32))
        self.y_inds = device_put(jnp.asarray(self.y_inds, jnp.int32))
        self.x_range = device_put(jnp.asarray(self.x_range, dtype))
        self.y_range = device_put(jnp.asarray(self.y_range, dtype))

    def _coordinate_dtype(self, x, y):
        return jnp.result_type(self.flat_psfs, *jax.tree.leaves((x, y)))

    @eqx.filter_jit
    def _create_psf(self, x, y):
        pitch = like(self.grid.step, x)
        n_pad = blend_pad(self.flat_psfs.shape[-1])
        if self.type == "1d":
            return synthesize_psf_separable(
                x,
                y,
                pitch,
                self.flat_psfs,
                self.x_offsets,
                self.y_offsets,
                self.offset_to_flat_idx,
                self.kx,
                self.ky,
                n_pad,
                self.x_symmetric,
                self.y_symmetric,
                self.type,
            )
        return synthesize_psf_idw(
            *clamp_to_radius(x, y, self.source_radius_lod),
            pitch,
            self.flat_psfs,
            self.flat_x_offsets,
            self.flat_y_offsets,
            self.kx,
            self.ky,
            n_pad,
            self.x_symmetric,
            self.y_symmetric,
            self.type,
        )

    @eqx.filter_jit
    def _create_psfs(self, x, y):
        return vmap(self._create_psf)(x, y)

    def in_source_domain(self, x, y):
        """Whether positions lie inside the hull of the tabulated source nodes.

        The hull is the polygon of the nodes (completed by their mirror images)
        for a two-dimensional table and the disk of the outermost node for a
        radially symmetric one. A position within a rounding tolerance of the
        edge is inside. Admission uses host float64 geometry, including for
        traced positions, and has no coordinate derivative.

        Args:
            x: x positions in lambda/D; any array shape.
            y: y positions in lambda/D, broadcastable with ``x``.

        Returns:
            Boolean array with the broadcast shape of ``x`` and ``y``; True
            where the position is inside the hull.
        """
        radius, hull = self.source_radius_lod, self._hull
        if not (is_traced(x) or is_traced(y)):
            return jnp.asarray(_host_source_domain(x, y, radius, hull))

        traced, concrete = eqx.partition((x, y), is_traced)

        def admit(*coordinates):
            x, y = eqx.combine(coordinates, concrete)
            return _host_source_domain(x, y, radius, hull)

        shape = np.broadcast_shapes(np.shape(x), np.shape(y))
        return jax.pure_callback(
            admit,
            jax.ShapeDtypeStruct(shape, np.bool_),
            *jax.tree.map(jax.lax.stop_gradient, traced),
            vmap_method="sequential",
        )

    def create_psf(self, x, y) -> Image:
        """Return one native source flux fraction image.

        Args:
            x: Source x position in lambda/D.
            y: Source y position in lambda/D.

        Returns:
            Native unocculted source flux fraction image at the position.
        """
        dtype = self._coordinate_dtype(x, y)
        x, y = jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)
        return fraction_image(
            self._create_psf(x, y),
            self.grid,
            (),
            self.measure,
        )

    def create_psfs(self, x, y) -> Image:
        """Return native source flux fractions with leading axis source.

        Args:
            x: Source x positions in lambda/D, shape (N,).
            y: Source y positions in lambda/D, shape (N,).

        Returns:
            Native source flux fraction images with leading axis ``source``
            of length N.
        """
        dtype = self._coordinate_dtype(x, y)
        x, y = jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)
        return fraction_image(
            self._create_psfs(x, y),
            self.grid,
            ("source",),
            self.measure,
        )

    def create_psfs_parallel(self, x_vals, y_vals):
        """Create off-axis PSFs at multiple positions in parallel using shard_map.

        Requires that ``hwoutils.set_host_device_count(N)`` was called at
        program startup to expose *N* CPU devices to JAX.

        Args:
            x_vals: Source x positions in lambda/D, shape (N,).
            y_vals: Source y positions in lambda/D, shape (N,).

        Returns:
            Native source flux fraction images with leading axis ``source``
            of length N, computed across the available devices.
        """
        dtype = self._coordinate_dtype(x_vals, y_vals)
        x_vals = jnp.asarray(x_vals, dtype=dtype)
        y_vals = jnp.asarray(y_vals, dtype=dtype)
        n_devices = jax.device_count()
        D = n_devices
        N = x_vals.shape[0]

        remainder = N % D
        padding_needed = (D - remainder) if remainder != 0 else 0

        if padding_needed > 0:
            x_vals_padded = jnp.pad(x_vals, (0, padding_needed), constant_values=0)
            y_vals_padded = jnp.pad(y_vals, (0, padding_needed), constant_values=0)
        else:
            x_vals_padded = x_vals
            y_vals_padded = y_vals

        mesh = Mesh(
            mesh_utils.create_device_mesh([D]),
            axis_names=("i",),
        )

        psfs_padded = shard_map(
            self._create_psfs,
            mesh=mesh,
            in_specs=(P("i"), P("i")),
            out_specs=P("i"),
            # The output is sharded, with no replication to check.
            check_vma=False,
        )(x_vals_padded, y_vals_padded)

        if padding_needed > 0:
            psfs = psfs_padded[:N]
        else:
            psfs = psfs_padded

        return fraction_image(
            psfs,
            self.grid,
            ("source",),
            self.measure,
        )
