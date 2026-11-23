"""Module for handling off-axis PSFs using JAX."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put, jit, shard_map, vmap
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from optixstuff.grids import Image, PixelGrid
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


class OffJAX(OffAx):
    """Class for handling off-axis PSFs using JAX.

    This class inherits from OffAx and uses JAX for optimized computation.
    Memory-efficient: stores PSFs in flat array with index mapping.

    Attributes:
        pixel_scale_arcsec (Quantity):
            Pixel scale of the PSF data in lambda/D.
        grid (PixelGrid):
            Pixel-center geometry in the package reference frame.
        flat_psfs (jnp.ndarray):
            Flat array of PSF data with shape (N_psfs, H, W), cast as JAX array.
        offset_to_flat_idx (jnp.ndarray):
            2D index mapping from (x_idx, y_idx) -> flat_psfs index.
    """

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
        """Prepare JAX interpolation arrays from the native samples."""
        super().__init__(
            psfs,
            offsets,
            grid,
            x_symmetric,
            y_symmetric,
            downsample_shape,
            sample_kind=sample_kind,
        )

        ##############
        # Convert the PSF data to JAX arrays
        # Note: We only store flat_psfs and the index mapping to minimize memory
        ##############
        self.flat_psfs = device_put(jnp.array(self.flat_psfs.astype(float)))
        self.flat_x_offsets = device_put(
            jnp.array(self.flat_offsets[:, 0].astype(float))
        )
        self.flat_y_offsets = device_put(
            jnp.array(self.flat_offsets[:, 1].astype(float))
        )

        # Convert offset arrays and index mapping to JAX arrays
        self.x_offsets = device_put(jnp.array(self.x_offsets.astype(float)))
        self.y_offsets = device_put(jnp.array(self.y_offsets.astype(float)))
        self.offset_to_flat_idx = device_put(
            jnp.array(self.offset_to_flat_idx.astype(int))
        )

        n_pixels = self.flat_psfs.shape[-1]
        n_pad = blend_pad(n_pixels)
        n_fft = n_pixels + 2 * n_pad  # Guaranteed even

        # Create 1D Frequency Vectors
        # kx: Real-to-Complex frequencies (0 to 0.5)
        self.kx = device_put(jnp.fft.rfftfreq(n_fft))
        # ky: Standard frequencies (0 to 0.5, -0.5 to -1/N)
        self.ky = device_put(jnp.fft.fftfreq(n_fft))

        # The source domain is the hull of the tabulated source positions: a
        # source beyond the outermost node has no data. The stamp is a smaller
        # square, and a node between the two holds its measured wing.
        self.source_radius_lod = float(
            np.max(np.hypot(self.flat_offsets[:, 0], self.flat_offsets[:, 1]))
        )
        # Host floats, so the scene grid is sized without reading a device array.
        self.source_extent_lod = (
            float(np.max(np.abs(self.flat_offsets[:, 0]))),
            float(np.max(np.abs(self.flat_offsets[:, 1]))),
        )
        radius = self.source_radius_lod
        self._hull = (
            None
            if self.type == "1d"
            else _hull_halfplanes(self.flat_offsets, self.x_symmetric, self.y_symmetric)
        )

        if self.type == "1d":
            # Optimized for 1D (radially symmetric) data using separable FFTs
            # Uses flat_psfs with offset_to_flat_idx mapping for memory efficiency

            def create_psf_kernel(x, y, psfs, x_off, y_off, idx_map, kx, ky):
                return synthesize_psf_separable(
                    x,
                    y,
                    pixel_scale_arcsec=self.pixel_scale_arcsec.value,
                    flat_psfs=psfs,
                    x_offsets=x_off,
                    y_offsets=y_off,
                    offset_to_flat_idx=idx_map,
                    kx=kx,
                    ky=ky,
                    n_pad=n_pad,
                    x_symmetric=self.x_symmetric,
                    y_symmetric=self.y_symmetric,
                    input_type=self.type,
                )

        else:
            # 2D, bilinear (radius, azimuth) blend over polar rings
            # Uses flat_psfs with flat_x_offsets/flat_y_offsets directly
            # Note: idx_map param is unused here but kept for a consistent signature
            def create_psf_kernel(x, y, psfs, x_off, y_off, idx_map, kx, ky):
                return synthesize_psf_idw(
                    *clamp_to_radius(x, y, radius),
                    pixel_scale_arcsec=self.pixel_scale_arcsec.value,
                    flat_psfs=psfs,
                    flat_x_offsets=x_off,
                    flat_y_offsets=y_off,
                    kx=kx,
                    ky=ky,
                    n_pad=n_pad,
                    x_symmetric=self.x_symmetric,
                    y_symmetric=self.y_symmetric,
                    input_type=self.type,
                )

        # ---------------------------------------------------------
        # BINDING
        # ---------------------------------------------------------
        self._create_psfs_kernel = vmap(
            create_psf_kernel, in_axes=(0, 0, None, None, None, None, None, None)
        )
        self._create_psfs_j = jit(self._create_psfs_kernel)
        self._create_psf_kernel_single = jit(create_psf_kernel)

        # ---------------------------------------------------------
        # WRAPPERS
        # ---------------------------------------------------------
        # Select appropriate handles based on interpolation type
        if self.type == "1d":
            # For separable interpolation: use unique offset arrays + index mapping
            x_handle = self.x_offsets
            y_handle = self.y_offsets
        else:
            # For the ring blend: use flat offset arrays
            x_handle = self.flat_x_offsets
            y_handle = self.flat_y_offsets

        def create_psfs_wrapper(x, y):
            return self._create_psfs_j(
                x,
                y,
                self.flat_psfs,
                x_handle,
                y_handle,
                self.offset_to_flat_idx,
                self.kx,
                self.ky,
            )

        self._create_psfs = create_psfs_wrapper

        def create_psf_wrapper(x, y):
            return self._create_psf_kernel_single(
                x,
                y,
                self.flat_psfs,
                x_handle,
                y_handle,
                self.offset_to_flat_idx,
                self.kx,
                self.ky,
            )

        self._create_psf = create_psf_wrapper

    def in_source_domain(self, x, y):
        """Whether positions lie inside the hull of the tabulated source nodes.

        The hull is the polygon of the nodes (completed by their mirror images)
        for a two-dimensional table and the disk of the outermost node for a
        radially symmetric one. A position within a rounding tolerance of the
        edge is inside.
        """
        x, y = jnp.asarray(x), jnp.asarray(y)
        radius = self.source_radius_lod
        slack = _DOMAIN_TOL * radius
        if self._hull is None:
            return jnp.hypot(x, y) <= radius + slack
        normals, distance = (jnp.asarray(a) for a in self._hull)
        reach = normals[:, 0] * x[..., None] + normals[:, 1] * y[..., None]
        return jnp.all(reach <= distance + slack, axis=-1) & (
            jnp.hypot(x, y) <= radius + slack
        )

    def create_psf(self, x, y) -> Image:
        """Return one native source flux fraction image."""
        return fraction_image(
            self._create_psf(x, y),
            self.grid,
            (),
            self.measure,
        )

    def create_psfs(self, x, y) -> Image:
        """Return native source flux fractions with leading axis source."""
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
        """
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

        # Select appropriate offset handles based on interpolation type
        if self.type == "1d":
            x_handle = self.x_offsets
            y_handle = self.y_offsets
        else:
            x_handle = self.flat_x_offsets
            y_handle = self.flat_y_offsets

        psfs_padded = shard_map(
            self._create_psfs_j,
            mesh=mesh,
            in_specs=(P("i"), P("i"), P(), P(), P(), P(), P(), P()),
            out_specs=P("i"),
            # check_rep=False,
        )(
            x_vals_padded,
            y_vals_padded,
            self.flat_psfs,
            x_handle,
            y_handle,
            self.offset_to_flat_idx,
            self.kx,
            self.ky,
        )

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
