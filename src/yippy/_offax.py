"""Package loading and coordinate handling for off-axis PSFs."""

from dataclasses import replace

import astropy.units as u
import jax
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from optixstuff.grids import (
    Frame,
    Image,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
)

from .logger import logger
from .sampling import fraction_image, sample_units


class OffAx:
    """Symmetry-extended off-axis samples used by the blend interpolator."""

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
        """Prepare interpolation geometry from shipped samples and their grid."""
        self.pixel_scale_arcsec = grid.step * lod / u.pix
        self.x_symmetric = x_symmetric
        self.y_symmetric = y_symmetric
        self.grid = grid
        psfs = psfs.astype(psfs.dtype.newbyteorder("="), copy=False)
        self.measure, factor = sample_units(sample_kind, grid)
        if sample_kind == "mean":
            psfs = psfs * factor
        offsets_x = np.unique(offsets[:, 0])
        offsets_y = np.unique(offsets[:, 1])

        if len(offsets_x) == 1:
            logger.info("Off-axis response is radially symmetric")
            self.type = "1d"
            # Instead of handling angles for 1dy, swap the x and y
            offsets_x, offsets_y = (offsets_y, offsets_x)
            offsets = np.vstack((offsets_y, offsets_x)).T
            raise NotImplementedError(
                "Verify that the PSFs are correct for this case!"
                " I don't have a test file for this case yet but I think they"
                " probably need to be rotated by 90 degrees."
            )
        elif len(offsets_y) == 1:
            logger.info("Off-axis response is radially symmetric")
            self.type = "1d"
        elif np.min(offsets) >= 0:
            logger.info("Off-axis response is quarterly symmetric")
            self.type = "2dq"
            # Check if 0 is included
            if 0 not in offsets_x:
                # Need to mirror the PSFs across the x axis for the interpolation
                min_x = np.min(offsets_x)
                # Add the mirrored offset to the original offsets
                offsets_x = np.insert(offsets_x, 0, -min_x)

                # Get all the PSFs that are at the minimum x value
                min_x_psfs = psfs[offsets[:, 0] == min_x]

                # Add the mirrored PSFs to the original PSFs
                psfs = np.insert(psfs, 0, np.flip(min_x_psfs, axis=2), axis=0)

                # Add the mirrored offset to the original offsets
                new_offsets = np.array(np.meshgrid(-min_x, offsets_y)).T.reshape(-1, 2)

                # Create an array of negative_min_x with the y offsets
                offsets = np.insert(offsets, 0, new_offsets, axis=0)
            if 0 not in offsets_y:
                # Need to mirror the PSFs across the y axis for the interpolation
                min_y = np.min(offsets_y)
                offsets_y = np.insert(offsets_y, 0, -min_y)

                # Get all the PSFs that are at the minimum y value
                min_y_psfs = psfs[offsets[:, 1] == min_y]

                # Add the mirrored PSFs to the original PSFs
                psfs = np.insert(psfs, 0, np.flip(min_y_psfs, axis=1), axis=0)

                # Add the mirrored offset to the original offsets
                new_offsets = np.array(np.meshgrid(offsets_x, -min_y)).T.reshape(-1, 2)

                # Create an array of negative_min_x with the y offsets
                offsets = np.insert(offsets, 0, new_offsets, axis=0)
        else:
            logger.info("Off-axis response is full 2D")
            self.type = "2df"

        # A lambda/D offset that represents the greatest separation where the PSF's
        # center is within the image. This applies to all coronagraph types.
        self.max_offset_in_image = psfs.shape[1] / 2 * u.pix * self.pixel_scale_arcsec

        # Downsample PSFs if requested
        if downsample_shape is not None:
            ny, nx = grid.shape
            ty, tx = downsample_shape
            if min(tx, ty) <= 0 or ty > ny or tx > nx or ny * tx != nx * ty:
                raise ValueError(
                    "downsample_shape must reduce both axes in the same ratio"
                )
            scale = nx / tx
            self.grid = replace(
                grid,
                shape=downsample_shape,
                center_x=(grid.center_x + 0.5) / scale - 0.5,
                center_y=(grid.center_y + 0.5) / scale - 0.5,
                basis=((grid.step * scale, 0.0), (0.0, grid.step * scale)),
            )
            source = fraction_image(psfs, grid, ("source",), self.measure)
            rebinned = integrate_footprints(source, footprint_overlap(grid, self.grid))
            psfs, self.measure = np.asarray(rebinned.data), rebinned.measure
            self.pixel_scale_arcsec = self.grid.step * lod / u.pix

            # Update max_offset_in_image after downsampling
            self.max_offset_in_image = (
                psfs.shape[1] / 2 * u.pix * self.pixel_scale_arcsec
            )

        self.flat_psfs = psfs
        self.flat_offsets = offsets
        self.n_psfs = len(offsets)

        # Create index mapping from (x_idx, y_idx) -> flat_psfs index
        # This avoids creating a sparse 4D reshaped_psfs array
        self.x_inds = np.searchsorted(offsets_x, offsets[:, 0])
        self.y_inds = np.searchsorted(offsets_y, offsets[:, 1])

        # Create a 2D index map: offset_to_flat_idx[x_idx, y_idx] = flat_psf_index
        # Use -1 to indicate no PSF at that (x_idx, y_idx) combination
        self.offset_to_flat_idx = np.full(
            (len(offsets_x), len(offsets_y)), -1, dtype=np.int32
        )
        for flat_idx in range(len(offsets)):
            self.offset_to_flat_idx[self.x_inds[flat_idx], self.y_inds[flat_idx]] = (
                flat_idx
            )

        self.x_offsets = offsets_x
        self.y_offsets = offsets_y
        # self.offsets = np.array(list(product(self.x_offsets, self.y_offsets)))
        self.x_range = np.array([self.x_offsets[0], self.x_offsets[-1]])
        self.y_range = np.array([self.y_offsets[0], self.y_offsets[-1]])

        # Store PSF shape for convenience
        self.psf_shape = self.flat_psfs.shape[1:]

    def get_psf_by_offset_idx(self, x_idx: int, y_idx: int) -> Image | None:
        """Get PSF at the given offset indices.

        Args:
            x_idx (int):
                Index into x_offsets array.
            y_idx (int):
                Index into y_offsets array.

        Returns:
            Image or None:
                Native unocculted source flux fraction, or None when absent.
        """
        flat_idx = self.offset_to_flat_idx[x_idx, y_idx]
        if flat_idx >= 0:
            return fraction_image(
                self.flat_psfs[flat_idx],
                self.grid,
                (),
                self.measure,
            )
        return None

    def __call__(self, x: Quantity, y: Quantity, lam=None, D=None, dist=None) -> Image:
        """Return the PSF at the given x/y position.

        This function has the following assumptions
        on the x/y values provided:

        - Pixel coordinates label centers: (0, 0) is the first pixel center.
        - If the x/y values are in lambda/D, angular, or length units, (0, 0)
          is the center of the image, where the star is located.

        Args:
            x (astropy.units.Quantity):
                x position. Can be either units of pixel, lod, an angular
                unit (e.g. arcsec), or a length unit (e.g. AU)
            y (astropy.units.Quantity):
                y position. Can be either units of pixel, lod, an angular
                unit (e.g. arcsec), or a length unit (e.g. AU)
            lam (astropy.units.Quantity):
                Wavelength of the observation
            D (astropy.units.Quantity):
                Diameter of the telescope
            dist (astropy.units.Quantity):
                Distance to the system

        Returns:
            Image:
                Native unocculted source flux fractions at the given position.
        """
        if (
            isinstance(x, Quantity)
            and isinstance(y, Quantity)
            and x.unit == y.unit == u.pix
        ):
            with jax.enable_x64():
                x, y = self.grid.pixel_to_unit(x.to_value(u.pix), y.to_value(u.pix))
            x, y = np.asarray(x), np.asarray(y)
        else:
            values = []
            for value, axis in ((x, "x"), (y, "y")):
                if isinstance(value, Quantity):
                    if value.unit == u.pix:
                        with jax.enable_x64():
                            point = self.grid.pixel_to_unit(
                                value.to_value(u.pix)
                                if axis == "x"
                                else self.grid.center_x,
                                value.to_value(u.pix)
                                if axis == "y"
                                else self.grid.center_y,
                            )
                        value = np.asarray(getattr(point, axis))
                    elif value.unit == lod:
                        value = value.to_value(lod)
                    else:
                        if value.unit.physical_type == "length":
                            value = (
                                np.arctan(value.to_value(u.m) / dist.to_value(u.m))
                                * u.rad
                            )
                        frame = (
                            self.grid.frame
                            if lam is None and D is None
                            else Frame(
                                lam.to_value(u.nm), D.to_value(u.m), "circumscribed"
                            )
                        )
                        wavelength = (
                            frame.reference_wavelength_nm
                            if lam is None
                            else lam.to_value(u.nm)
                        )
                        value = value.to_value(u.arcsec) / frame.lod_to_arcsec(
                            wavelength
                        )
                values.append(value)
            x, y = values

        if np.ndim(x) == np.ndim(y) == 0:
            return self.create_psf(x, y)
        else:
            return self.create_psfs(x, y)
