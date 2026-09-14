"""Base class for all offax_psfs.fits files.

This has been deprecated. Please use OffJAX instead.
"""

import warnings
from multiprocessing import Pool
from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from hwoutils.fft import fft_shift
from hwoutils.transforms import downsample_psfs
from lod_unit import lod
from numpy.typing import NDArray

from yippy.util import convert_to_lod, create_shift_mask

from .logger import logger


class OffAx:
    """Base class for off-axis PSF handling.

    Handles YIP data loading, offset parsing, and symmetry detection.
    ``OffJAX`` inherits from this class and overrides ``create_psf`` /
    ``create_psfs`` with JAX-accelerated implementations.

    The pure-Python ``create_psf`` and ``create_psfs`` methods on this class
    are retained as a reference implementation but are not used in production.
    All ``Coronagraph`` instances now use ``OffJAX`` exclusively.

    This class loads and processes PSF data from the yield input package (YIP).
    It currently supports oneD and quarter symmetric PSF YIPs. The primary use
    is to interpolate the PSF data to a given x/y position. This is done by
    calling the OffAx object with the x/y position as arguments, which itself
    calls the psf object after converting units.

    Attributes:
        pixel_scale_arcsec (Quantity):
            Pixel scale of the PSF data in lambda/D.
        center_x (Quantity):
            Central x position in the PSF data.
        center_y (Quantity):
            Central y position in the PSF data.
        psf:
            Instance of the appropriate PSF class (e.g., OneD, TwoD) based on input YIP.

    Args:
        yip_dir (Path):
            Path to the directory containing PSF and offset data.
        logger (Logger):
            Logger for logging events and information.
        offax_data_file (str):
            Name of the file containing the PSF data.
        offax_offsets_file (str):
            Name of the file containing the offsets data.
        pixel_scale_arcsec (Quantity):
            Pixel scale of the PSF data in lambda/D.
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
    ) -> None:
        """Initializes the OffAx class by loading PSF and offset data from YIP.

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
            downsample_shape:
                Optional target shape (ny, nx) to downsample PSFs to.
                If provided, all PSFs will be resampled to this shape
                immediately after loading, conserving total flux.
                The pixel_scale_arcsec will be updated accordingly.
        """
        # Pixel scale in lambda/D
        self.pixel_scale_arcsec = pixel_scale_arcsec

        # Load symmetry
        self.x_symmetric = x_symmetric
        self.y_symmetric = y_symmetric

        # Load off-axis PSF data (e.g. the planet) (unitless intensity maps)
        psfs = pyfits.getdata(Path(yip_dir, offax_data_file), 0)
        # FITS is big-endian; JAX downsampling requires native-endian arrays.
        psfs = psfs.astype(psfs.dtype.newbyteorder("="), copy=False)

        # Save the center of the pixel array, which is used for converting to
        # lambda/D when the x/y positions are in pixels. The stack is
        # (n_psfs, row, column), so x measures columns and y measures rows.
        self.center_x = psfs.shape[2] / 2 * u.pix
        self.center_y = psfs.shape[1] / 2 * u.pix

        # Load the offset list, which is in units of lambda/D
        offsets = pyfits.getdata(Path(yip_dir, offax_offsets_file), 0)

        # Check whether offsets are given as 1D or 2D
        one_d_offsets = len(offsets.shape) == 1
        if one_d_offsets:
            # Add a second dimension if the offsets are 1D
            offsets = np.vstack((offsets, np.zeros_like(offsets)))

        if len(offsets.shape) > 1:
            if (offsets.shape[1] != 2) and (offsets.shape[0] == 2):
                # This condition occurs when the offsets is transposed
                # from the expected format
                offsets = offsets.T
        assert len(offsets) == psfs.shape[0], (
            "Offsets and PSFs do not have the same number of elements"
        )

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################

        # Get the unique values of the offset list so that we can format the
        # data into
        offsets_x = np.unique(offsets[:, 0])
        offsets_y = np.unique(offsets[:, 1])

        if len(offsets_x) == 1:
            logger.info(f"{yip_dir.stem} is radially symmetric")
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
            logger.info(f"{yip_dir.stem} is radially symmetric")
            self.type = "1d"
        elif np.min(offsets) >= 0:
            logger.info(f"{yip_dir.stem} is quarterly symmetric")
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
            logger.info(f"{yip_dir.stem} response is full 2D")
            self.type = "2df"

        # A lambda/D offset that represents the greatest separation where the PSF's
        # center is within the image. This applies to all coronagraph types.
        self.max_offset_in_image = psfs.shape[1] / 2 * u.pix * self.pixel_scale_arcsec

        # Downsample PSFs if requested
        if downsample_shape is not None:
            original_shape = psfs.shape[1:]
            logger.info(
                f"Downsampling PSFs from {original_shape} to {downsample_shape}"
            )
            psfs, new_pixscale = downsample_psfs(
                psfs, self.pixel_scale_arcsec.value, downsample_shape
            )
            # Update pixel scale with the new value (preserving units)
            self.pixel_scale_arcsec = new_pixscale * self.pixel_scale_arcsec.unit
            logger.info(f"New pixel scale: {self.pixel_scale_arcsec}")

            # Transform pixel-center coordinates under the same bin-edge map.
            # For example, input index 4 becomes 1.75 after a 2x reduction.
            scale_y = original_shape[0] / psfs.shape[1]
            scale_x = original_shape[1] / psfs.shape[2]
            self.center_x = (self.center_x + 0.5 * u.pix) / scale_x - 0.5 * u.pix
            self.center_y = (self.center_y + 0.5 * u.pix) / scale_y - 0.5 * u.pix

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

    def get_psf_by_offset_idx(self, x_idx: int, y_idx: int):
        """Get PSF at the given offset indices.

        Args:
            x_idx (int):
                Index into x_offsets array.
            y_idx (int):
                Index into y_offsets array.

        Returns:
            np.ndarray:
                The PSF at the given offset indices, or None if no PSF exists
                at that combination.
        """
        flat_idx = self.offset_to_flat_idx[x_idx, y_idx]
        if flat_idx >= 0:
            return self.flat_psfs[flat_idx]
        return None

    def create_psf(self, x: float, y: float):
        """Create and return the PSF at the specified off-axis position.

        .. deprecated::
            The pure-Python implementation is deprecated.  Use
            ``Coronagraph`` (which uses ``OffJAX``) instead.

        Interpolates and returns the Point Spread Function (PSF) at the specified
        off-axis position (x, y). If the exact (x, y) position matches one of the
        PSFs in the YIP, that PSF is returned directly. Otherwise, the PSFs
        from the surrounding positions are combined using Gaussian weighting and
        Fourier interpolation to produce an interpolated PSF.

        Args:
            x (float):
                The x-coordinate of the off-axis position.
            y (float):
                The y-coordinate of the off-axis position.

        Returns:
            np.ndarray:
                The interpolated PSF corresponding to the input (x, y) position.

        Notes:
            - If `self.type` is "1d", the (x, y) position is converted to a
            radial separation and angle for interpolation.
            - If `self.type` is "2dq", the (x, y) position is mirrored to the
            first quadrant, and the PSF is flipped accordingly after
            interpolation.
            - Gaussian weighting is used to combine the nearest PSFs when the
            exact (x, y) position does not match any precomputed PSF. The
            weighting is based on the distance from the input position.
            - The PSFs are shifted to align with the input position before
            combining, and the final PSF is normalized by the cumulative weight
            for each pixel.

        """
        warnings.warn(
            "OffAx.create_psf is deprecated; use Coronagraph (OffJAX) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Set default values
        flip_lr, flip_ud = False, False

        # Check for exact matches
        if x in self.x_offsets and y in self.y_offsets:
            x_ind = np.searchsorted(self.x_offsets, x)
            y_ind = np.searchsorted(self.y_offsets, y)
            flat_idx = self.offset_to_flat_idx[x_ind, y_ind]
            if flat_idx >= 0:
                return self.flat_psfs[flat_idx]

        # Translate position based on type
        if self.type == "1d":
            flip_lr, flip_ud = x < 0, y < 0
            sep = np.sqrt(x**2 + y**2)
            _x, _y = sep, 0
        elif self.type == "2dq":
            flip_lr, flip_ud = x < 0, y < 0
            _x, _y = abs(x), abs(y)
        else:
            _x, _y = x, y

        # Get indices of nearest PSFs, in x and y directions
        x_match = _x in self.x_offsets
        _x_search = np.searchsorted(self.x_offsets, _x)
        if x_match:
            # If the x value is an exact match, we only need one index
            x_inds = _x_search.reshape(-1)
        else:
            x_inds = np.array([_x_search - 1, _x_search])

        y_match = _y in self.y_offsets
        _y_search = np.searchsorted(self.y_offsets, _y)
        if y_match:
            # If the y value is an exact match, we only need one index
            y_inds = _y_search.reshape(-1)
        else:
            y_inds = np.array([_y_search - 1, _y_search])

        x_vals, y_vals = self.x_offsets[x_inds], self.y_offsets[y_inds]
        # Get the indices of the nearest PSFs to the input (x, y)
        near_inds = np.array(np.meshgrid(x_inds, y_inds)).T.reshape(-1, 2)

        # Get the (x, y) offsets of the nearest PSFs to the input (x, y)
        near_offsets = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)

        # Get the PSFs at the nearest offsets using the index mapping
        flat_indices = self.offset_to_flat_idx[near_inds[:, 0], near_inds[:, 1]]
        near_psfs = self.flat_psfs[flat_indices]

        # Combine the PSFs
        if len(near_psfs) > 1:
            # Get the shift (in pixels) required to align with the input (x, y)
            near_shifts = (
                np.array([_x, _y]) - near_offsets
            ) / self.pixel_scale_arcsec.value

            # Calculate the distance of each PSF from the input (x, y)
            near_diffs = np.linalg.norm(near_shifts, axis=1)

            # Gaussian weighting
            sigma = 0.25
            weights = np.exp(-(near_diffs**2) / (2 * sigma**2))

            # Normalize the weights
            weights /= weights.sum()

            # Initialize the PSF array
            psf = np.zeros_like(near_psfs[0])

            # Initialize the weight array
            # This weight system is used because shifting a PSF right by one pixel
            # will leave a blank pixel on the left side of the image. The weight
            # array keeps track of which PSFs have contributions for each pixel.
            weight_array = np.zeros_like(psf)
            for i, near_psf in enumerate(near_psfs):
                shifted_psf = fft_shift(near_psf, *near_shifts[i])
                weight_mask = create_shift_mask(near_psf, *near_shifts[i], weights[i])
                # Add the weighted PSF to the total PSF
                psf += weight_mask * shifted_psf
                # Keep track of the weight for each pixel
                weight_array += weight_mask
            # Divide each pixel by its weight to get the final PSF
            psf /= weight_array
        else:
            psf = near_psfs[0]

        # Apply any necessary flips before shifting
        if self.x_symmetric and flip_lr:
            psf = np.fliplr(psf)
            remaining_x_shift = x + _x
        else:
            remaining_x_shift = x - _x
        if self.y_symmetric and flip_ud:
            psf = np.flipud(psf)
            remaining_y_shift = y + _y
        else:
            remaining_y_shift = y - _y

        if remaining_x_shift != 0 or remaining_y_shift != 0:
            psf = fft_shift(
                psf,
                remaining_x_shift / self.pixel_scale_arcsec.value,
                remaining_y_shift / self.pixel_scale_arcsec.value,
            )

        return psf

    def create_psfs(self, x: NDArray, y: NDArray) -> NDArray:
        """Create and return the PSFs at the specified off-axis positions.

        .. deprecated::
            The pure-Python implementation is deprecated.  Use
            ``Coronagraph`` (which uses ``OffJAX``) instead.
        """
        warnings.warn(
            "OffAx.create_psfs is deprecated; use Coronagraph (OffJAX) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        psfs = np.empty((len(x), *self.flat_psfs.shape[1:]))
        for i in range(len(x)):
            psfs[i] = self.create_psf(x[i], y[i])
        return psfs

    def create_psfs_parallel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam=None,
        D=None,
        dist=None,
        workers: int = 4,
    ) -> np.ndarray:
        """Compute PSFs for batches of (x, y) arrays using multiprocessing.

        .. deprecated::
            The pure-Python implementation is deprecated.  Use
            ``Coronagraph`` (which uses ``OffJAX``) instead.

        Args:
            x (np.ndarray):
                Array of x positions.
            y (np.ndarray):
                Array of y positions.
            lam (astropy.units.Quantity):
                Wavelength of the observation
            D (astropy.units.Quantity):
                Diameter of the telescope
            dist (astropy.units.Quantity):
                Distance to the system
            workers (int):
                Number of parallel processes to use.

        Returns:
            np.ndarray:
                A stacked numpy array of the computed PSFs with shape (N,
                height, width), where N = len(x).
        """
        if isinstance(x, Quantity):
            # Convert the x and y positions to lambda/D if they are in pixels
            if x.unit != lod:
                x = convert_to_lod(
                    x, self.center_x, self.pixel_scale_arcsec, lam, D, dist
                )
            else:
                x = x.value
        if isinstance(y, Quantity):
            if y.unit != lod:
                y = convert_to_lod(
                    y, self.center_y, self.pixel_scale_arcsec, lam, D, dist
                )
            else:
                y = y.value
        # For each x[i], we create a column of identical x-values and the full y-array.
        # This way, each process will handle (len(y)) points for that particular x[i].
        args = []
        for xi in x:
            # Create an array filled with xi, same shape as y
            x_col = np.full_like(y, xi)
            args.append((x_col, y))

        # Each process will handle a single (x_col, y) pair and run self.create_psfs
        # which returns a 3D stack: shape = (len(y), height, width).
        with Pool(processes=workers) as pool:
            psf_rows = pool.starmap(self.create_psfs, args)

        # psf_rows is now a list of arrays, each of shape (len(y), height, width)
        # Stack them along a new axis to form (len(x), len(y), height, width)
        psf_datacube = np.stack(psf_rows, axis=0)
        return psf_datacube

    def __call__(
        self, x: Quantity, y: Quantity, lam=None, D=None, dist=None
    ) -> NDArray:
        """Return the PSF at the given x/y position.

        This function (via util.convert_to_lod) has the following assumptions
        on the x/y values provided:
            - If units are pixels, they follow the 00LL convention. As in the
              (0,0) point is the lower left corner of the image.
            - If the x/y values are in lambda/D, angular, or length units the
              (0,0) point is the center of the image, where the star is
              (hopefully) located.

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
            NDArray:
                The PSF at the given x/y position
        """
        if isinstance(x, Quantity):
            # Convert the x and y positions to lambda/D if they are in pixels
            if x.unit != lod:
                x = convert_to_lod(
                    x, self.center_x, self.pixel_scale_arcsec, lam, D, dist
                )
            x = x.value
        if isinstance(y, Quantity):
            if y.unit != lod:
                y = convert_to_lod(
                    y, self.center_y, self.pixel_scale_arcsec, lam, D, dist
                )
            y = y.value

        if np.isscalar(x) and np.isscalar(y):
            return self.create_psf(x, y)
        else:
            return self.create_psfs(x, y)
