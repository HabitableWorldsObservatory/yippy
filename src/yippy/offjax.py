"""Module for handling off-axis PSFs using JAX."""

from pathlib import Path

import jax
import jax.numpy as jnp
from astropy.units import Quantity
from hwoutils.grids import Image, PixelGrid
from jax import device_put, jit, shard_map, vmap
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from lod_unit import lod

from .jax_funcs import synthesize_psf_idw, synthesize_psf_separable
from .offax import OffAx


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
        yip_dir: Path,
        offax_data_file: str,
        offax_offsets_file: str,
        pixel_scale_arcsec: Quantity,
        x_symmetric: bool,
        y_symmetric: bool,
        downsample_shape: tuple[int, int] | None = None,
        *,
        grid: PixelGrid | None = None,
    ) -> None:
        """Initializes the OffJAX class by casting YIP data to JAX arrays.

        Args:
            yip_dir:
                Path to the directory containing PSF and offset data.
            offax_data_file:
                Name of the file containing the PSF data.
            offax_offsets_file:
                Name of the file containing the offsets data.
            pixel_scale_arcsec:
                Pixel scale of the PSF data in lambda/D.
            x_symmetric:
                Whether the PSFs are symmetric in x.
            y_symmetric:
                Whether the PSFs are symmetric in y.
            grid: Native package grid, built from its authoritative header.
            downsample_shape:
                Optional target shape (ny, nx) to downsample PSFs to.
                If provided, all PSFs will be resampled to this shape
                immediately after loading, conserving total flux.
        """
        super().__init__(
            yip_dir,
            offax_data_file,
            offax_offsets_file,
            pixel_scale_arcsec,
            x_symmetric,
            y_symmetric,
            downsample_shape=downsample_shape,
            grid=grid,
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
        n_pad = int(1.5 * n_pixels)
        n_fft = n_pixels + 2 * n_pad
        n_pad = int(1.5 * n_pixels)
        if (n_pixels + 2 * n_pad) % 2 != 0:
            n_pad += 1

        n_fft = n_pixels + 2 * n_pad  # Guaranteed even

        # Create 1D Frequency Vectors
        # kx: Real-to-Complex frequencies (0 to 0.5)
        self.kx = device_put(jnp.fft.rfftfreq(n_fft))
        # ky: Standard frequencies (0 to 0.5, -0.5 to -1/N)
        self.ky = device_put(jnp.fft.fftfreq(n_fft))

        # Calculate max_offset for bounds check
        max_offset = -1.0
        if self.type == "1d" and hasattr(self, "max_offset_in_image"):
            max_offset = self.max_offset_in_image.to(lod).value

        if self.type == "1d":
            # Optimized for 1D (radially symmetric) data using separable FFTs
            # Uses flat_psfs with offset_to_flat_idx mapping for memory efficiency
            max_offset = -1.0
            if hasattr(self, "max_offset_in_image"):
                max_offset = self.max_offset_in_image.to(lod).value

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
                    max_offset=max_offset,
                )

        else:
            # 2D, uses IDW interpolation for irregular grids
            # Uses flat_psfs with flat_x_offsets/flat_y_offsets directly
            # Note: idx_map param is unused for IDW but kept for consistent signature
            def create_psf_kernel(x, y, psfs, x_off, y_off, idx_map, kx, ky):
                return synthesize_psf_idw(
                    x,
                    y,
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
                    k_neighbors=4,
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
            # For IDW interpolation: use flat offset arrays
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

    def create_psf(self, x, y) -> Image:
        """Return one per-pixel source flux fraction image."""
        return Image(
            self._create_psf(x, y),
            self.grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
        )

    def create_psfs(self, x, y) -> Image:
        """Return per-pixel source flux fractions with leading axis source."""
        return Image(
            self._create_psfs(x, y),
            self.grid,
            ("source",),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
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

        return Image(
            psfs,
            self.grid,
            ("source",),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
        )
